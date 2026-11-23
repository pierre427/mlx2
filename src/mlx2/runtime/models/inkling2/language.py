# SPDX-License-Identifier: Apache-2.0
# Adapted from Blaizzy/mlx-vlm@111ac668 mlx_vlm/models/inkling2/language.py,
# itself adapted for MLX from HuggingFace Transformers' Inkling2 implementation
# (Apache-2.0); see provenance/inkling2-mini.json.
"""Inkling 2 looped text decoder (``inkling2_loop``), native mlx2 model.

The 24 physical layers are visited 48 times: the middle block (layers
``core_start..core_end``) runs ``num_passes`` times with shared weights; every
visit has its own KV and four-shortconv cache.  Residual streams are
"attention residuals": each sublayer input is a learned softmax mix over the
completed 6-layer blocks plus the running partial block.  Later loop passes
blend a message read from the previous pass's block bank into their q/k/v/MLP
inputs.

Cache planes are flat alternating ``[kv, conv]`` per *visit*, the same
per-plane contract as Inkling v1 (``inkling-kv-four-shortconv-v1``), so APCv2
segmented COW and the v1 convolution rollback records apply unchanged.
"""

import math

import mlx.core as mx
from mlx import nn

from ..cache import (
    BatchKVCache,
    BatchRotatingKVCache,
    CacheList,
    KVCache,
    RotatingKVCache,
)
from ..inkling.cache import InklingConvolutionCache
from ..inkling.language import (
    _cache_padding_mask,
    _next_conv_state,
    banded_additive_mask,
)
from ..switch_layers import SwitchGLU
from . import act_quant as _act_quant
from . import fuse as _fuse
from . import kv_q8 as _kv_q8
from . import splitk
from .config import TextConfig

_MASK_BUDGET_BYTES = 64 << 20
MAX_LOOP_PASSES = 8  # varloop ceiling (trained count is 3)
_SINK_SDPA_MAX_ROWS = 8  # per-row SDPA calls for one-token decode
# ``inkling2_sink_slot``: when an append leaves the KVCache buffer exactly
# full, grow it by one cache step (zeros) so the sink row has its slot,
# instead of concatenating the zero row onto the whole K/V. Module switch for
# the A/B (False restores the 2026-11-21 full-buffer concatenation).
SINK_SLOT_SPARE = True


@mx.compile
def l2_clip(x, threshold, scale=None):
    dtype = x.dtype
    x = x.astype(mx.float32)
    if scale is not None:
        x = x * scale.astype(mx.float32)
    norm = mx.sqrt(mx.sum(mx.square(x), axis=-1, keepdims=True))
    return (x * mx.minimum(threshold / (norm + 1e-6), 1.0)).astype(dtype)


@mx.compile
def _softcapped_swiglu(up, gate, gate_cap, up_cap):
    dtype = gate.dtype
    gate, up = gate.astype(mx.float32), up.astype(mx.float32)
    return (
        mx.tanh(gate / gate_cap)
        * gate_cap
        * mx.sigmoid(gate)
        * mx.tanh(up / up_cap)
        * up_cap
    ).astype(dtype)


def _fp32(module, name):
    """FP32 copy of a small BF16 weight, cast once and reused (exact).

    Keyed on the parameter object: re-cast when it is replaced (load,
    update).  In-place writes to the array are not seen (nothing in mlx2
    writes these weights in place), and a replaced weight stays referenced
    until the module's next forward."""
    weight = module[name]
    if weight.dtype == mx.float32:
        return weight
    cached = module.__dict__.get("_fp32_" + name)
    if cached is None or cached[0] is not weight:
        cached = (weight, weight.astype(mx.float32))
        object.__setattr__(module, "_fp32_" + name, cached)
    return cached[1]


def _route(x, weight, bias, top_k, route_scale):
    # ``weight`` is already FP32 (``_fp32``); the cast is a no-op then.
    scores = mx.sigmoid(x.astype(mx.float32) @ weight.astype(mx.float32).T)
    indices = mx.argpartition(-(scores + bias), top_k - 1, axis=-1)[..., :top_k]
    indices = mx.stop_gradient(indices)
    weights = mx.take_along_axis(scores, indices, axis=-1)
    weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20) * route_scale
    return indices, weights


@mx.compile
def _combine_routes(out, weights):
    return (out.astype(mx.float32) * weights[..., None]).sum(axis=-2).astype(out.dtype)


@mx.compile
def _attention_residual(states, query, normalize_values):
    states = states.astype(mx.float32)
    keys = mx.fast.rms_norm(states, None, 1e-6)
    scores = (keys @ query.reshape(-1, states.shape[-1]).T) * states.shape[-1] ** -0.5
    weights = mx.softmax(scores, axis=-2).swapaxes(-1, -2)
    return weights @ (keys if normalize_values else states)


_HEAD_ROWS = 16  # activation rows per head-kernel launch
_HEAD_SOURCE = r"""
    // One simdgroup per vocabulary row n; lanes stride K by 4; FP32
    // accumulation of exact BF16 x BF16 products, one simd_sum per row.
    uint n = thread_position_in_grid.x / 32;
    uint lane = thread_index_in_simdgroup;
    if (n >= N) return;
    float acc[M];
    for (int m = 0; m < M; ++m) acc[m] = 0.0f;
    // One 64-bit load carries four BF16 values; a BF16 is the high half of
    // an FP32, so the conversion is a shift (exact).  Same products and
    // summation order as four scalar loads.
    const device uint2* row = (const device uint2*)(w + size_t(n) * K);
    for (uint k = lane; k < K / 4; k += 32) {
        uint2 wv = row[k];
        float w0 = as_type<float>(wv.x << 16), w1 = as_type<float>(wv.x & 0xFFFF0000u);
        float w2 = as_type<float>(wv.y << 16), w3 = as_type<float>(wv.y & 0xFFFF0000u);
        for (int m = 0; m < M; ++m) {
            uint2 xv = ((const device uint2*)(x + size_t(m) * K))[k];
            acc[m] += w0 * as_type<float>(xv.x << 16) + w1 * as_type<float>(xv.x & 0xFFFF0000u)
                    + w2 * as_type<float>(xv.y << 16) + w3 * as_type<float>(xv.y & 0xFFFF0000u);
        }
    }
    for (int m = 0; m < M; ++m) {
        float total = simd_sum(acc[m]);
        if (lane == 0) out[size_t(m) * N + n] = total;
    }
"""
_head_kernel = None
_COMPILED_HEAD_ROWS = 32
_SOFTCAP_SCALARS = {}


@mx.compile
def _softcap_compiled(logits, cap):
    return mx.tanh(logits / cap) * cap


def _softcap_scalar(cap):
    held = _SOFTCAP_SCALARS.get(cap)
    if held is None:
        held = _SOFTCAP_SCALARS[cap] = mx.array(cap, mx.float32)
        mx.eval(held)
    return held


def head_gemv(x, weight):
    """``x @ weight.T`` with FP32 output from a BF16 weight, no FP32 copy.

    ``x`` holds BF16-representable rows (the head input is BF16-rounded), so
    every product is exact in FP32; only the summation order differs from an
    FP32 GEMM.  Opt-in (``inkling2_head_kernel``); unqualified.
    """
    global _head_kernel
    if _head_kernel is None:
        _head_kernel = mx.fast.metal_kernel(
            name="inkling2_bf16_head_fp32",
            input_names=["x", "w"],
            output_names=["out"],
            source=_HEAD_SOURCE,
        )
    lead, k = x.shape[:-1], x.shape[-1]
    n = weight.shape[0]
    rows = mx.contiguous(x.reshape(-1, k).astype(mx.bfloat16))
    outputs = []
    for start in range(0, rows.shape[0], _HEAD_ROWS):
        chunk = rows[start : start + _HEAD_ROWS]
        outputs.append(
            _head_kernel(
                inputs=[chunk, weight],
                template=[("M", chunk.shape[0]), ("N", n), ("K", k)],
                grid=(n * 32, 1, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[(chunk.shape[0], n)],
                output_dtypes=[mx.float32],
            )[0]
        )
    out = outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=0)
    return out.reshape(*lead, n)


# ----- flash prefill with the relative bias in-kernel (opt-in) -----
#
# Original mlx2 kernel (2026-11-21). The design follows the other Inkling 2
# stacks (mlx-lm PR 2086 ``relative_attention``, as vendored by Rapid-MLX and
# mlx-serve; omlx ``flash_banded_attention``; ollama): one online-softmax pass
# over key tiles per query tile, the per-query bias row read in-kernel, and the
# sink logit as the softmax's starting term (its value is zero). Unlike those
# kernels, Q, K, V, scores and probabilities stay FP32 (no half staging).
#
# score(i, j) = scale * q_i . k_j + bias_i[d]  for d = OFF + i - j < E, else +0
# masked when d < 0, j < left_pad, or (WINDOW > 0 and d >= WINDOW).
_FLASH_BQ = 32  # query rows per threadgroup: 8 per simdgroup
_FLASH_TG = 128
_FLASH_SOURCE = r"""
    constexpr int BK = 16;
    constexpr int LDK = BK + 4;
    constexpr int LDV = D + 4;
    const int L = sizes[0];
    const int S = sizes[1];
    const int OFF = sizes[2];
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int b = threadgroup_position_in_grid.z;
    const int h = threadgroup_position_in_grid.y;
    const int hk = h / (HQ / HKV);
    const int q0 = threadgroup_position_in_grid.x * BQ;
    const int pad = left_pad[b];
    const float sc = scale[0];

    // 8x8 fragment layout: this lane holds row fm, columns fn and fn + 1.
    const short fm = ((lane / 4) & 4) + (lane / 2) % 4;
    const short fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
    const int row = q0 + int(sg) * 8 + fm;
    const bool row_ok = row < L;
    const int diag = OFF + row;

    threadgroup float Ks[D * LDK];
    threadgroup float Vs[BK * LDV];

    simdgroup_float8x8 qf[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        float a = 0.0f, c = 0.0f;
        if (row_ok) {
            const device T* qp = q + b * q_strides[0] + h * q_strides[1] + row * q_strides[2];
            a = float(qp[(db * 8 + fn) * q_strides[3]]);
            c = float(qp[(db * 8 + fn + 1) * q_strides[3]]);
        }
        qf[db].thread_elements()[0] = a;
        qf[db].thread_elements()[1] = c;
    }
    simdgroup_float8x8 acc[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        acc[db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    // The sink is a zero-value key with logit sink: start the softmax there.
    float m = row_ok ? float(sinks[(b * L + row) * HQ + h]) : 0.0f;
    float l = 1.0f;
    const device T* bias = rel + (size_t(b * L + (row_ok ? row : 0)) * HQ + h) * E;

    const int first_diag = OFF + q0;
    const int last_diag = OFF + min(q0 + BQ, L) - 1;
    int first_key = WINDOW > 0 ? max(0, first_diag - WINDOW + 1) : 0;
    first_key = max(first_key, pad);
    const device T* kh = k + b * k_strides[0] + hk * k_strides[1];
    const device T* vh = v + b * v_strides[0] + hk * v_strides[1];

    for (int k0 = first_key / BK * BK; k0 <= last_diag; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int e = tid; e < BK * D; e += TG) {
            const int r = e / D;
            const int col = e % D;
            const int key = k0 + r;
            float kv = 0.0f, vv = 0.0f;
            if (key < S) {
                kv = float(kh[key * k_strides[2] + col * k_strides[3]]);
                vv = float(vh[key * v_strides[2] + col * v_strides[3]]);
            }
            Ks[col * LDK + r] = kv;
            Vs[r * LDV + col] = vv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (q0 + int(sg) * 8 >= L) {
            continue;
        }
        simdgroup_float8x8 s8[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            s8[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int db = 0; db < D / 8; ++db) {
                simdgroup_float8x8 kf;
                simdgroup_load(kf, Ks + db * 8 * LDK + kb * 8, LDK);
                simdgroup_multiply_accumulate(s8[kb], qf[db], kf, s8[kb]);
            }
        }
        float tile_max = -INFINITY;
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const int key = k0 + kb * 8 + fn + e;
                const int d = diag - key;
                const bool vis = row_ok && key < S && d >= 0 && key >= pad
                    && (WINDOW == 0 || d < WINDOW);
                float x = -INFINITY;
                if (vis) {
                    x = s8[kb].thread_elements()[e] * sc;
                    if (d < E) {
                        x += float(bias[d]);
                    }
                }
                s8[kb].thread_elements()[e] = x;
                tile_max = max(tile_max, x);
            }
        }
        // The four lanes of a fragment row differ in lane bits 0 and 3.
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, 1));
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, 8));
        const float m_new = max(m, tile_max);
        const float factor = metal::exp(m - m_new);
        float tile_sum = 0.0f;
        simdgroup_float8x8 p8[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const float p = metal::exp(s8[kb].thread_elements()[e] - m_new);
                p8[kb].thread_elements()[e] = p;
                tile_sum += p;
            }
        }
        tile_sum += simd_shuffle_xor(tile_sum, 1);
        tile_sum += simd_shuffle_xor(tile_sum, 8);
        l = l * factor + tile_sum;
        m = m_new;
        for (int db = 0; db < D / 8; ++db) {
            acc[db].thread_elements()[0] *= factor;
            acc[db].thread_elements()[1] *= factor;
            for (int kb = 0; kb < BK / 8; ++kb) {
                simdgroup_float8x8 vf;
                simdgroup_load(vf, Vs + kb * 8 * LDV + db * 8, LDV);
                simdgroup_multiply_accumulate(acc[db], p8[kb], vf, acc[db]);
            }
        }
    }
    if (row_ok) {
        device T* op = out + ((size_t(b) * HQ + h) * L + row) * D;
        for (int db = 0; db < D / 8; ++db) {
            op[db * 8 + fn] = T(acc[db].thread_elements()[0] / l);
            op[db * 8 + fn + 1] = T(acc[db].thread_elements()[1] / l);
        }
    }
"""
_flash_kernel = None


def flash_relative_attention(q, k, v, rel, sinks, *, offset, left_padding, window, scale):
    """Prefill attention with Inkling's relative bias, window and sink.

    q: [B, H, L, D]; k, v: [B, Hk, S, D] (any strides); rel: [B, L, H, E], the
    per-query bias at distances below E; sinks: [B, L, H]; queries sit at key
    positions ``offset + i``. Returns [B, H, L, D] in q's dtype. GPU only;
    opt-in (``inkling2_flash_prefill``); unqualified.
    """
    global _flash_kernel
    if _flash_kernel is None:
        _flash_kernel = mx.fast.metal_kernel(
            name="inkling2_flash_relative_attention",
            input_names=["q", "k", "v", "rel", "sinks", "left_pad", "sizes", "scale"],
            output_names=["out"],
            source=_FLASH_SOURCE,
            ensure_row_contiguous=False,
        )
    b, hq, length, dim = q.shape
    hkv, size = k.shape[1], k.shape[2]
    pad = (mx.zeros((b,), mx.int32) if left_padding is None
           else mx.maximum(left_padding, 0).astype(mx.int32))
    return _flash_kernel(
        inputs=[q, k, v, mx.contiguous(rel), mx.contiguous(sinks), pad,
                mx.array([length, size, int(offset)], dtype=mx.int32),
                mx.array([float(scale)], dtype=mx.float32)],
        template=[("T", q.dtype), ("D", dim), ("HQ", hq), ("HKV", hkv),
                  ("E", rel.shape[-1]), ("WINDOW", int(window)),
                  ("BQ", _FLASH_BQ), ("TG", _FLASH_TG)],
        grid=((length + _FLASH_BQ - 1) // _FLASH_BQ * _FLASH_TG, hq, b),
        threadgroup=(_FLASH_TG, 1, 1),
        output_shapes=[(b, hq, length, dim)],
        output_dtypes=[q.dtype],
    )[0]


def _banded_mask_reference(_attn, *args, **kwargs):
    return banded_additive_mask(*args, **kwargs)


def _speculative_record(cache, index, state, inputs, mask):
    """Keep (pre-state, fp32 masked inputs, mask) for the v1 rollback record."""
    if cache is not None and getattr(cache, "speculating", False):
        cache._inkling_inputs[index] = (state, inputs, mask)


def _sconv_kernel_call(conv, x, cache, mask, tail, extra):
    from . import fuse_kernels

    return fuse_kernels.sconv(conv, x, cache, mask, tail, extra)


def _residual_kernel_call(module, blocks, partial):
    from . import fuse_kernels

    return fuse_kernels.residual(module, blocks, partial)


class ShortConvolution(nn.Module):
    """Depthwise causal conv, FP32 throughout: ``(conv(ctx) + x) / sqrt(K+1)``."""

    def __init__(self, channels: int, kernel_size: int, conv_idx: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_idx = conv_idx
        self.conv = nn.Conv1d(channels, channels, kernel_size, groups=channels, bias=False)

    def __call__(self, x, cache=None, mask=None, tail="plain", extra=()):
        """``tail`` names the fused epilogue (``inkling2_fuse_sconv`` only):
        ``plain`` returns the convolution; the decoder's residual forms
        (``attn_residual``, ``attn_boundary``, ``mlp_residual``) also do its
        FP32 round trip and residual add, with ``extra`` their operands."""
        fused = _fuse.on(self, "fuse_sconv")
        if tail != "plain" and not fused:
            raise ValueError("fused short-convolution tails need inkling2_fuse_sconv")
        if fused and _fuse.on(self, "fuse_sconv_kernel"):
            out = _sconv_kernel_call(self, x, cache, mask, tail, extra)
            if out is not None:
                return out
        k = self.kernel_size
        xf = x.astype(mx.float32)
        inputs = xf if mask is None else mx.where(mask[..., None], xf, 0)
        state = cache[self.conv_idx] if cache is not None else None
        if state is None:
            state = mx.zeros((x.shape[0], k - 1, x.shape[-1]), mx.float32)
        context = mx.concatenate([state, inputs], axis=1)
        if cache is not None:
            _speculative_record(cache, self.conv_idx, cache[self.conv_idx], inputs, mask)
            if fused and mask is None and k > 1:
                # ``_next_conv_state`` without a mask is concat(state,
                # inputs)[:, -(k-1):]: the same values as this view.
                cache[self.conv_idx] = context[:, -(k - 1):]
            else:
                cache[self.conv_idx] = _next_conv_state(state, inputs, mask)
        out = mx.conv1d(context, _fp32(self.conv, "weight"), groups=x.shape[-1])
        if fused:
            _fuse.count(self, "sconv_" + tail)
            return _fuse.sconv_tail(tail, out, xf, k, x, *extra)
        return ((out + xf) / math.sqrt(k + 1)).astype(x.dtype)


class DynamicShortConvolution(nn.Module):
    """Global-layer K/V conv whose per-head taps add an input-dependent term."""

    def __init__(self, config: TextConfig, channels: int, num_heads: int, conv_idx: int):
        super().__init__()
        self.kernel_size = config.sconv_kernel_size
        self.num_heads = num_heads
        self.conv_idx = conv_idx
        self.static_weight = mx.zeros((self.kernel_size, channels))
        self.weight_proj = nn.Linear(config.hidden_size, num_heads * self.kernel_size, bias=False)

    def __call__(self, x, dynamic_input, cache=None, mask=None):
        b, length, _ = x.shape
        k = self.kernel_size
        fused = _fuse.on(self, "fuse_sconv")
        kernel_out = None
        if fused and _fuse.on(self, "fuse_dconv_kernel"):
            from . import fuse_kernels

            dynamic = (self.weight_proj(dynamic_input).astype(mx.float32)
                       .reshape(b, length, self.num_heads, 1, k))
            kernel_out = fuse_kernels.dconv(self, x, dynamic, cache, mask)
            if kernel_out is not None:
                return kernel_out
        xf = x.astype(mx.float32)
        inputs = xf if mask is None else mx.where(mask[..., None], xf, 0)
        state = cache[self.conv_idx] if cache is not None else None
        if state is None:
            state = mx.zeros((b, k - 1, x.shape[-1]), mx.float32)
        context = mx.concatenate([state, inputs], axis=1)
        if cache is not None:
            _speculative_record(cache, self.conv_idx, cache[self.conv_idx], inputs, mask)
            if fused and mask is None and k > 1:
                cache[self.conv_idx] = context[:, -(k - 1):]
            else:
                cache[self.conv_idx] = _next_conv_state(state, inputs, mask)
        dynamic = (
            self.weight_proj(dynamic_input)
            .astype(mx.float32)
            .reshape(b, length, self.num_heads, 1, k)
        )
        if fused:
            _fuse.count(self, "dynamic_sconv")
            return _fuse.dynamic_conv(context, xf, _fp32(self, "static_weight").T, dynamic,
                                      k, length, self.num_heads, x)
        weights = _fp32(self, "static_weight").T.reshape(self.num_heads, -1, k) + dynamic
        out = mx.zeros_like(xf).reshape(b, length, self.num_heads, -1)
        for lag in range(k):
            values = context[:, k - 1 - lag : k - 1 - lag + length].reshape(out.shape)
            out = out + values * weights[..., lag]
        return ((out.reshape(x.shape) + xf) / math.sqrt(2 * k + 1)).astype(x.dtype)


class Attention(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.is_sliding = config.layer_is_sliding(layer_idx)
        self.head_dim = config.swa_head_dim if self.is_sliding else config.head_dim
        self.n_heads = (
            config.swa_num_attention_heads if self.is_sliding else config.num_attention_heads
        )
        self.n_kv = (
            config.swa_num_key_value_heads if self.is_sliding else config.num_key_value_heads
        )
        self.sliding = config.sliding_window_size if self.is_sliding else 0
        self.rel_extent = self.sliding or config.rel_extent
        self.scale = self.head_dim**-0.5
        self.log_floor = None if self.is_sliding else config.log_scaling_n_floor
        hidden = config.hidden_size
        self.q_proj = nn.Linear(hidden, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.n_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.n_kv * self.head_dim, bias=False)
        self.r_proj = nn.Linear(hidden, self.n_heads * config.d_rel, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, hidden, bias=False)
        self.gate_proj = nn.Linear(hidden, self.n_heads * self.head_dim, bias=False)
        self.sink_proj = nn.Linear(hidden, self.n_heads, bias=False)
        self.q_scale = mx.ones((self.n_heads, self.head_dim))
        self.rel_scale = mx.ones((self.n_heads, config.d_rel))
        self.rel_proj = mx.zeros((config.d_rel, self.rel_extent))
        channels = self.n_kv * self.head_dim
        for idx, name in enumerate(("k_sconv", "v_sconv")):
            self[name] = (
                ShortConvolution(channels, config.sconv_kernel_size, idx)
                if self.is_sliding
                else DynamicShortConvolution(config, channels, self.n_kv, idx)
            )

    # Rows of the fused input projection: the x-shared block first, then k, v.
    _FUSED = ("q_proj", "r_proj", "gate_proj", "sink_proj", "k_proj", "v_proj")

    def fuse_projections(self):
        """Row-stack the six input projections into one weight, in place.

        Each ``*_proj.weight`` becomes a contiguous row view of the stacked
        matrix, so memory is unchanged.  Visits whose q/k/v inputs coincide
        run one GEMV; others run the x-shared block plus k and v separately.
        Opt-in (``inkling2_fused_projections``); unqualified.
        """
        if getattr(self, "_fused_bounds", None) is not None:
            return
        if getattr(self, "_fused_weight", None) is not None:
            object.__setattr__(self, "_fused_bounds", self._fused_layout)
            return
        weights = [self[name].weight for name in self._FUSED]
        stacked = mx.concatenate(weights, axis=0)
        mx.eval(stacked)
        bounds, start = {}, 0
        for name, weight in zip(self._FUSED, weights):
            bounds[name] = (start, start + weight.shape[0])
            start += weight.shape[0]
        for name, (a, b) in bounds.items():
            self[name].weight = stacked[a:b]
        mx.eval([self[name].weight for name in self._FUSED])
        object.__setattr__(self, "_fused_weight", stacked)
        object.__setattr__(self, "_fused_bounds", bounds)
        object.__setattr__(self, "_fused_layout", bounds)
        object.__setattr__(self, "fused_calls", 0)

    def _project_inputs(self, x, key_input, value_input):
        """(q, r, gate, sink, k, v) projections, fused when configured."""
        bounds = getattr(self, "_fused_bounds", None)
        if bounds is None:
            return (self.q_proj(x), self.r_proj(x), self.gate_proj(x), self.sink_proj(x),
                    self.k_proj(key_input), self.v_proj(value_input))
        self.fused_calls += 1
        weight = self._fused_weight
        shared_end = bounds["sink_proj"][1]
        if key_input is x and value_input is x:
            y = x @ weight.T
            k = y[..., slice(*bounds["k_proj"])]
            v = y[..., slice(*bounds["v_proj"])]
        else:
            y = x @ weight[:shared_end].T
            k = key_input @ weight[slice(*bounds["k_proj"])].T
            v = value_input @ weight[slice(*bounds["v_proj"])].T
        q, r, g, s = (y[..., slice(*bounds[n])] for n in ("q_proj", "r_proj", "gate_proj", "sink_proj"))
        return q, r, g, s, k, v

    def _sink_slot(self, kv, b, size):
        """K/V plus the zero sink column without copying K/V
        (``inkling2_sink_slot``).

        The sink is a zero K/V row with its own logit. The reference
        concatenates that row onto the fetched K/V, which copies the whole
        K/V on every visit (about 268 MB per global visit at 64K). A plain
        ``KVCache`` buffer usually has capacity past ``offset`` (256-token
        growth), so the zero row is written into that slot and the attention
        reads the view ``[:offset + 1]``: the same values in the same order,
        so it is bit-identical. The slot lies past the committed fill level,
        which the next append overwrites and trim/rollback never read, so
        cache state is unchanged.

        An append of a multiple of 256 tokens (every 512-token prefill chunk)
        leaves the buffer exactly full. With ``SINK_SLOT_SPARE`` the buffer
        then grows by one cache step of zeros (``sink_slot_grows``): one copy,
        the same cost as the concatenation it replaces, and the spare step
        persists through the cache's own later growths (which append to the
        full buffer), so later visits of that cache find their slot. The
        attention still reads ``[:offset + 1]`` with a zero last row. Any other
        cache, a window-cropped fetch, or (switch off) a full buffer falls
        back to the concatenation, counted by reason."""
        reason = None
        if type(kv) is not KVCache:
            reason = "cache_type"
        elif kv.keys is None or kv.keys.shape[0] != b:
            reason = "rows"
        elif kv.offset != size:
            reason = "cropped"
        elif kv.keys.shape[2] <= kv.offset and not SINK_SLOT_SPARE:
            reason = "full"
        if reason is not None:
            d = self.__dict__
            d["sink_slot_fallbacks"] = d.get("sink_slot_fallbacks", 0) + 1
            reasons = d.setdefault("sink_slot_fallback_reasons", {})
            reasons[reason] = reasons.get(reason, 0) + 1
            return None
        off = kv.offset
        if kv.keys.shape[2] <= off:
            for name in ("keys", "values"):
                buf = getattr(kv, name)
                spare = mx.zeros((*buf.shape[:2], kv.step, buf.shape[3]), buf.dtype)
                setattr(kv, name, mx.concatenate([buf, spare], axis=2))
            self.__dict__["sink_slot_grows"] = self.__dict__.get("sink_slot_grows", 0) + 1
        for name in ("keys", "values"):
            buf = getattr(kv, name)
            buf[..., off : off + 1, :] = mx.zeros((*buf.shape[:2], 1, buf.shape[3]), buf.dtype)
        self.__dict__["sink_slot_calls"] = self.__dict__.get("sink_slot_calls", 0) + 1
        return kv.keys[..., : off + 1, :], kv.values[..., : off + 1, :]

    # Caches whose fetched K/V the split-K kernel reads as the composed path
    # does (chronological order, left padding, queries at the last L keys).
    _SPLITK_CACHES = (KVCache, RotatingKVCache, BatchKVCache, BatchRotatingKVCache)

    def _splitk_decline(self, kv, q, k, v):
        """None when ``inkling2_verify_splitk`` engages, else the reason."""
        if kv is not None and type(kv) not in self._SPLITK_CACHES:
            return "cache_type"
        return splitk.decline_reason(q, k, v, n_heads=self.n_heads, n_kv=self.n_kv,
                                     head_dim=self.head_dim)

    def _fetch_q8(self, kv, k, v):
        """``inkling2_kv_q8``: append packed int8 rows, fetch packed rows.

        Fails closed on a plane holding the other format (a BF16 plane in a
        Q8 model, or a ring): the APCv2 layout keeps the two apart, so one
        here is a bug, not a state to convert."""
        held = getattr(kv, "keys", None)
        if type(kv) in (RotatingKVCache, BatchRotatingKVCache) or (
                isinstance(held, mx.array) and held.dtype != mx.int8):
            raise ValueError(
                f"Inkling 2 kv_q8 global visit got a {type(kv).__name__} plane in another "
                "format; Q8 and BF16 caches do not mix")
        k, v = kv.update_and_fetch(_kv_q8.pack(k), _kv_q8.pack(v))
        if k.dtype != mx.int8 or k.shape[-1] != _kv_q8.packed_width(self.head_dim):
            raise ValueError("Inkling 2 kv_q8 plane returned rows in another format")
        return k, v

    def _fetch(self, kv, k, v):
        held = getattr(kv, "keys", None)
        if isinstance(held, mx.array) and held.dtype == mx.int8:
            raise ValueError(
                "Inkling 2 attention got an int8 (kv_q8) plane with inkling2_kv_q8 off")
        return self._chronological(kv, kv.update_and_fetch(k, v))

    @staticmethod
    def _chronological(kv, fetched):
        # Learned relative logits index backward distance, so ring buffers
        # are read in chronological order (RoPE attention does not care).
        if type(kv) is RotatingKVCache:
            return kv._temporal_order(kv.keys), kv._temporal_order(kv.values)
        if type(kv) is BatchRotatingKVCache:
            kv._temporal_order()
            return kv.keys[..., : kv._idx, :], kv.values[..., : kv._idx, :]
        return fetched

    def _ring_direct(self, kv, length):
        """True when this visit may read a ring cache in place through the
        split-K kernel (``splitk.RING_DIRECT``) instead of its chronological
        copy. The copy is made later only if the kernel declines."""
        return (
            splitk.RING_DIRECT
            and self.__dict__.get("_verify_splitk")
            and type(kv) in (RotatingKVCache, BatchRotatingKVCache)
            and 1 <= length < splitk.MAX_ROWS
        )

    @staticmethod
    def _ring_layout(kv):
        """``(rot, size)``: the chronological key j that ``_chronological``
        would return lives at physical row ``(j + rot) % width`` of the
        buffers, for j < size. None for a layout it does not map (a
        RotatingKVCache with ``keep`` rows)."""
        width = kv.keys.shape[2]
        if type(kv) is RotatingKVCache:
            if kv.keep:
                return None
            idx = kv._idx
            if idx == width:  # _temporal_order: the buffer as is
                return 0, width
            if idx < kv.offset:  # wrapped: [idx:] then [:idx]
                return idx, width
            return 0, idx  # not yet full: [:idx]
        if kv.rotated:  # BatchRotatingKVCache: roll by -_idx
            return kv._idx % width, width
        return 0, kv._idx

    def __call__(self, x, key_input=None, value_input=None, cache=None, conv_mask=None):
        b, length, _ = x.shape
        cfg = self.config
        kv, conv = cache if cache is not None else (None, None)
        key_input = x if key_input is None else key_input
        value_input = x if value_input is None else value_input
        q, r, gate, sink, k, v = self._project_inputs(x, key_input, value_input)
        if self.is_sliding:
            k = self.k_sconv(k, conv, conv_mask)
            v = self.v_sconv(v, conv, conv_mask)
        else:
            k = self.k_sconv(k, key_input, conv, conv_mask)
            v = self.v_sconv(v, value_input, conv, conv_mask)
        q = l2_clip(
            q.reshape(b, length, self.n_heads, self.head_dim),
            cfg.qk_l2_clip_threshold,
            self.q_scale,
        )
        k = l2_clip(k.reshape(b, length, self.n_kv, self.head_dim), cfg.qk_l2_clip_threshold)
        v = v.reshape(b, length, self.n_kv, self.head_dim)
        fa = _fuse.on(self, "fuse_attention")
        if cfg.use_value_norm_no_gamma and fa and v.dtype == x.dtype:
            # No gain: the norm accumulates in FP32 and rounds once to the
            # input dtype, so the FP32 round trip is the identity on its result.
            v = mx.fast.rms_norm(v, None, cfg.rms_norm_eps)
        elif cfg.use_value_norm_no_gamma:
            v = mx.fast.rms_norm(v.astype(mx.float32), None, cfg.rms_norm_eps).astype(x.dtype)
        r = l2_clip(
            r.reshape(b, length, self.n_heads, cfg.d_rel),
            cfg.rel_l2_clip_threshold,
            self.rel_scale,
        )
        q, k, v = (a.transpose(0, 2, 1, 3) for a in (q, k, v))
        ring, ring_start, ring_phys = None, 0, None
        # inkling2_kv_q8 (global visits only; the attribute is never set on
        # sliding ones): K/V are stored as packed int8 rows. Ring paths below
        # only apply to sliding (rotating) caches.
        q8 = self.__dict__.get("_kv_q8", False)
        if q8 and kv is None:
            # No cache: attend over the values a cache would have stored.
            k, v = _kv_q8.pack(k), _kv_q8.pack(v)
            counts = self.__dict__.setdefault("kv_q8_counts", {})
            counts["uncached"] = counts.get("uncached", 0) + 1
        elif q8:
            k, v = self._fetch_q8(kv, k, v)
        elif (
            fa
            and type(kv) is RotatingKVCache
            and not self.__dict__.get("_sink_sdpa")
            and not self.__dict__.get("_flash_prefill")
        ):
            # The chronological read and the zero sink row become one
            # concatenation below (the reference makes two full copies).
            kv.update_and_fetch(k, v)
            ring, width = kv, kv.keys.shape[2]
            size = width if (kv._idx == width or kv._idx < kv.offset) else kv._idx
            k = v = None
        elif kv is not None and self._ring_direct(kv, length):
            kv.update_and_fetch(k, v)
            ring_phys = self._ring_layout(kv)
            if ring_phys is None:
                _count_ring(self, "unmapped")
                k, v = self._chronological(kv, None)
            else:
                k = v = None
        elif kv is not None:
            k, v = self._fetch(kv, k, v)
        if ring is None:
            size = k.shape[2] if ring_phys is None else ring_phys[1]
        left_padding = getattr(kv, "left_padding", None)
        if self.sliding and size > length + self.sliding - 1:
            # No query reaches keys older than its window: drop them before
            # the dense relative mask and SDPA (they would be masked anyway).
            start = size - length - (self.sliding - 1)
            if ring is None and ring_phys is None:
                k, v = k[:, :, start:], v[:, :, start:]
            ring_start = start
            size = size - start
            if left_padding is not None:
                left_padding = mx.maximum(left_padding - start, 0)
        offset = size - length
        if (self.log_floor is not None and (fa or _fuse.on(self, "fuse_shared_tau"))
                and q.dtype == r.dtype == x.dtype):
            # (The fused chain returns q/r in their own dtype; the reference
            # casts to x.dtype, so it engages only when those agree.)
            q, r = _fuse.log_tau(self, q, r, length, offset, left_padding, x.dtype)
        elif self.log_floor is not None:
            # Global layers only (no slicing): absolute, padding-relative
            # positions, one tau row per batch row when padded.
            positions = (mx.arange(length) + offset + 1).astype(mx.float32)[None, :]
            if left_padding is not None:
                positions = positions - left_padding[:, None].astype(mx.float32)
            tau = 1 + cfg.log_scaling_alpha * mx.log(
                mx.maximum(positions / self.log_floor, 1)
            )
            tau = tau[:, :, None, None]
            q = (q.astype(mx.float32) * tau.transpose(0, 2, 1, 3)).astype(x.dtype)
            r = (r.astype(mx.float32) * tau).astype(x.dtype)
        rel_proj = self.rel_proj.astype(x.dtype)
        if q8:
            # The int8 K/V are read in place; nothing dequantizes the prefix.
            reason = _kv_q8.decline_reason(q[:, :, :1], k, v, n_heads=self.n_heads,
                                           n_kv=self.n_kv, head_dim=self.head_dim)
            counts = self.__dict__.setdefault("kv_q8_counts", {})
            kw = dict(offset=offset, left_padding=left_padding, window=0, scale=self.scale)
            if reason is None and length < _kv_q8.MAX_ROWS:
                key = "kernel_decode" if length == 1 else "kernel_rows"
                out = _kv_q8.q8_attention(q, k, v, r @ rel_proj, sink, **kw)
            elif reason is None:
                key = "kernel_prefill_tiled"
                out = _kv_q8.tiled_attention(q, k, v, r @ rel_proj, sink, counts=counts, **kw)
            else:
                key = "blockwise:" + reason
                out = _kv_q8.blockwise_attention(q, k, v, r @ rel_proj, sink, counts=counts,
                                                 **kw)
            counts[key] = counts.get(key, 0) + 1
            return self._finish(out, gate, b, length, x.dtype)
        if self.__dict__.get("_verify_splitk"):
            if ring_phys is not None:
                reason = self._splitk_decline(kv, q, kv.keys, kv.values)
            else:
                reason = self._splitk_decline(kv, q, k, v)
            counts = self.__dict__.setdefault("verify_splitk_counts", {})
            key = "calls" if reason is None else "declined:" + reason
            counts[key] = counts.get(key, 0) + 1
            if reason is None:
                ring_arg = None
                if ring_phys is not None:
                    # The ring's physical buffers with the chronological
                    # start (after the window crop): the kernel walks the
                    # same logical keys as on the reordered copy.
                    _count_ring(self, "calls")
                    k, v = kv.keys, kv.values
                    ring_arg = ((ring_phys[0] + ring_start) % k.shape[2], size)
                # Reads the fetched K/V in place: no sink column, no mask.
                out = splitk.gqa_splitk_attention(
                    q, k, v, r @ rel_proj, sink, offset=offset, left_padding=left_padding,
                    window=self.sliding, scale=self.scale, ring=ring_arg,
                )
                return self._finish(out, gate, b, length, x.dtype)
        if ring_phys is not None:
            # The kernel declined: the chronological copy the reference reads.
            _count_ring(self, "copies")
            k, v = self._chronological(kv, None)
            if ring_start:
                k, v = k[:, :, ring_start:], v[:, :, ring_start:]
            ring_phys = None
        if (
            length == 1
            and b <= _SINK_SDPA_MAX_ROWS
            and getattr(self, "_sink_sdpa", False)
        ):
            # One query per row: its sink logit is a per-head constant, so
            # MLX's ``sinks`` (shape [H]) takes it directly, one SDPA per
            # row; no K/V concatenation.  Same softmax; only the reduction
            # order differs (the sink term is added separately).
            mask = (_fuse.banded_mask if fa else _banded_mask_reference)(
                self, r, rel_proj, offset, size, self.sliding, self.rel_extent,
                left_padding=left_padding,
            )
            self.__dict__["sink_sdpa_calls"] = self.__dict__.get("sink_sdpa_calls", 0) + 1
            outs = [
                mx.fast.scaled_dot_product_attention(
                    q[i : i + 1], k[i : i + 1], v[i : i + 1], scale=self.scale,
                    mask=mask[i : i + 1], sinks=sink[i, 0].astype(mask.dtype),
                )
                for i in range(b)
            ]
            out = outs[0] if b == 1 else mx.concatenate(outs, axis=0)
            return self._finish(out, gate, b, length, x.dtype)
        if (
            length > 1
            and self.__dict__.get("_flash_prefill")
            and mx.default_device() == mx.gpu
            and self.head_dim % 8 == 0
            and self.head_dim <= 128  # threadgroup tiles sized for D <= 128 (18.7 KB)
            and (self.n_heads % self.n_kv) == 0
        ):
            self.__dict__["flash_prefill_calls"] = self.__dict__.get("flash_prefill_calls", 0) + 1
            out = flash_relative_attention(
                q, k, v, r @ rel_proj, sink, offset=offset, left_padding=left_padding,
                window=self.sliding, scale=self.scale,
            )
            return self._finish(out, gate, b, length, x.dtype)
        # A query-dependent sink is a zero K/V with its own logit.
        sinks = sink.transpose(0, 2, 1)[..., None]
        slot = (self._sink_slot(kv, b, size)
                if ring is None and not q8 and self.__dict__.get("_sink_slot_on") else None)
        if ring is not None:
            _fuse.count(self, "ring_sink_concat")
            zero = _fuse.zeros(self, (b, self.n_kv, 1, self.head_dim), ring.keys.dtype)
            k, v = _fuse.ring_with_sink(ring, ring_start, zero)
        elif slot is not None:
            k, v = slot
        else:
            zeros = (_fuse.zeros(self, (b, self.n_kv, 1, self.head_dim), k.dtype) if fa
                     else mx.zeros((b, self.n_kv, 1, self.head_dim), k.dtype))
            k = mx.concatenate([k, zeros], axis=2)
            v = mx.concatenate([v, zeros], axis=2)
        budget = self.__dict__.get("_mask_budget_bytes") or _MASK_BUDGET_BYTES
        # Budget counts 4 bytes per mask element; the transient peak of one
        # chunk (int32 gather index, gathered bias, where/concat copies) is a
        # few times that.
        rows = max(1, budget // max(1, b * self.n_heads * (size + 1) * 4))
        self.__dict__["mask_rows_last"] = rows
        outputs = []
        for begin in range(0, length, rows):
            end = min(length, begin + rows)
            mask = (_fuse.banded_mask if fa else _banded_mask_reference)(
                self,
                r[:, begin:end],
                rel_proj,
                offset + begin,
                size,
                self.sliding,
                self.rel_extent,
                left_padding=left_padding,
            )
            mask = mx.concatenate([mask, sinks[:, :, begin:end].astype(mask.dtype)], axis=-1)
            outputs.append(
                mx.fast.scaled_dot_product_attention(
                    q[:, :, begin:end], k, v, scale=self.scale, mask=mask
                )
            )
        out = outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=2)
        return self._finish(out, gate, b, length, x.dtype)

    def _finish(self, out, gate, b, length, dtype):
        out = out.transpose(0, 2, 1, 3).reshape(b, length, -1)
        if _fuse.on(self, "fuse_attention"):
            _fuse.count(self, "gate_finish")
            return self.o_proj(_fuse.gate_finish(out, gate, _fuse.dtype_ref(dtype)))
        out = (out.astype(mx.float32) * mx.sigmoid(gate.astype(mx.float32))).astype(dtype)
        return self.o_proj(out)


def _count_ring(attn, key):
    counts = attn.__dict__.setdefault("splitk_ring_counts", {})
    counts[key] = counts.get(key, 0) + 1


class SoftcappedSwiGLU(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.gate_cap = config.mlp_gate_softcap
        self.up_cap = config.mlp_up_softcap

    def __call__(self, up, gate):
        out = _softcapped_swiglu(up, gate, self.gate_cap, self.up_cap)
        fmt = self.__dict__.get("_act_quant")
        if fmt:
            # inkling2_moe_act_quant: the down projection's input (see act_quant).
            _act_quant.count(self.__dict__["_act_quant_counts"], f"reference:{fmt}:w2")
            out = _act_quant.apply(fmt, out)
        return out


class SparseMoE(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.route_scale = config.route_scale
        self.gate_weight = mx.zeros((config.n_routed_experts, config.hidden_size))
        self.e_score_correction_bias = mx.zeros((config.n_routed_experts,))
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.intermediate_size,
            config.n_routed_experts,
            activation=SoftcappedSwiGLU(config),
        )

    def set_fast_path(self, fused=False, weighted_sum=False):
        """Opt-in ``moe_fast`` routes (default off; see that module).

        The stacked gate/up table is built on the first enable and kept; the
        reference modules read row views of it, so turning the option off
        only stops the fused dispatch."""
        object.__setattr__(self, "_fast_fused", bool(fused))
        object.__setattr__(self, "_fast_weighted_sum", bool(fused and weighted_sum))
        if fused:
            self._fast_experts()

    def _compiled_moe(self, experts, weighted, x):
        """``inkling2_compile_decode_blocks``: the one-token fused MoE forward
        (router, gate/up gather, activation, down gather, combine) as one
        compiled graph per shape (``shapeless=False``; Rapid-MLX #5920,
        reimplemented). The weights are the compile's implicit inputs, so a
        replaced table retraces. Python-side counters inside the block run
        at trace time only; ``compiled_moe_calls`` counts calls."""
        from .. import gather_lhs
        from . import moe_fast

        key = (id(experts), bool(weighted), gather_lhs.enabled(experts.gate_up))
        held = self.__dict__.get("_compiled_moe_fn")
        if held is None or held[0] != key:
            # MLX substitutes tracers in these containers. parameters() builds
            # separate dictionaries; tracing the modules would still see the
            # original arrays, which MLX rejects as uncaptured inputs.
            fn = mx.compile(
                lambda row: moe_fast.fused_call(
                    self, experts, row, weighted_sum=weighted, compiled_decode=True
                ),
                inputs=[self, vars(experts)], shapeless=False,
            )
            held = (key, fn)
            object.__setattr__(self, "_compiled_moe_fn", held)
        result = held[1](x)
        counts = self.__dict__.setdefault("moe_fast_counts", {})
        counts["compiled_moe_calls"] = counts.get("compiled_moe_calls", 0) + 1
        counts["fused"] = counts.get("fused", 0) + 1
        return result

    def _fast_experts(self):
        """``(experts, refusal)``, rebuilt when any source array is replaced
        (a refused build is retried then too).  Turning the fused route off
        leaves the row views in place: the reference gathers read them in
        place, but the NAX kernels copy a strided view, so NAX with the
        fused route off after it was on costs a per-call copy."""
        from . import moe_fast

        key = moe_fast.source_key(self.switch_mlp)
        held = self.__dict__.get("_fast_built")
        if held is None or not moe_fast.same_source(self.__dict__.get("_fast_key", ()), key):
            held = moe_fast.build(self.switch_mlp)
            object.__setattr__(self, "_fast_built", held)
            # The build rewrote the gate/up arrays (row views); key on those.
            object.__setattr__(self, "_fast_key", moe_fast.source_key(self.switch_mlp))
        return held

    def __call__(self, x):
        if self.__dict__.get("_fast_fused"):
            from . import moe_fast

            experts, reason = self._fast_experts()
            reason = reason or moe_fast.decline_reason()
            if reason is None:
                weighted = self.__dict__.get("_fast_weighted_sum", False)
                if (self.__dict__.get("_compile_decode_blocks") and x.size == x.shape[-1]
                        and mx.default_device() == mx.gpu
                        and not self.__dict__.get("_act_quant")):
                    return self._compiled_moe(experts, weighted, x)
                return moe_fast.fused_call(self, experts, x, weighted_sum=weighted)
            counts = self.__dict__.setdefault("moe_fast_counts", {})
            counts["declined:" + reason] = counts.get("declined:" + reason, 0) + 1
        indices, weights = _route(
            x, _fp32(self, "gate_weight"), self.e_score_correction_bias, self.top_k, self.route_scale
        )
        fmt = self.__dict__.get("_act_quant")
        if fmt:
            # inkling2_moe_act_quant: the router keeps the unquantized input;
            # the expert gate/up gather reads the quantized one.
            _act_quant.count(self.__dict__["act_quant_counts"], f"reference:{fmt}:w13")
            x = _act_quant.apply(fmt, x)
        return _combine_routes(self.switch_mlp(x, indices), weights)

    def act_quant_format(self):
        """The experts' declared activation format; raises if they have none."""
        fmt = _act_quant.expert_format(self.switch_mlp)
        if fmt is None:
            kinds = sorted({type(self.switch_mlp[n]).__name__
                            for n in ("gate_proj", "up_proj", "down_proj")})
            raise ValueError(
                f"Inkling 2 inkling2_moe_act_quant needs NVFP4 or MXFP8 experts, not {kinds}")
        return fmt

    def set_act_quant(self, on):
        """``inkling2_moe_act_quant``: fake-quantize the expert inputs to the
        format the expert weights declare (``act_quant.expert_format``).
        Returns the format, or None when off; raises for experts with no
        declared activation format."""
        fmt = self.act_quant_format() if on else None
        counts = self.__dict__.setdefault("act_quant_counts", {})
        object.__setattr__(self, "_act_quant", fmt)
        object.__setattr__(self.switch_mlp.activation, "_act_quant", fmt)
        object.__setattr__(self.switch_mlp.activation, "_act_quant_counts", counts)
        return fmt


class AttentionResidual(nn.Module):
    def __init__(self, config: TextConfig, num_queries: int = 1):
        super().__init__()
        self.query = mx.zeros(
            (config.hidden_size,) if num_queries == 1 else (num_queries, config.hidden_size)
        )
        self.normalize_values = config.attn_res_normalize_values

    def __call__(self, blocks, partial):
        if _fuse.on(self, "fuse_residual_kernel"):
            out = _residual_kernel_call(self, blocks, partial)
            if out is not None:
                return out
        if _fuse.on(self, "fuse_residual"):
            # The reference stack promotes every BF16 block to FP32 on every
            # mix; the FP32 copies are made once per forward (same values).
            _fuse.count(self, "residual_mix")
            blocks = [_fuse.fp32_block(self, b) for b in blocks]
        return _attention_residual(
            mx.stack([*blocks, partial], axis=-2), self.query, self.normalize_values
        )


def _expert_switches(moe):
    """The routed-expert gather modules of one MoE: the reference projections
    and, when built, the fused stacked table and its down twin."""
    found = []
    switch = getattr(moe, "switch_mlp", None)
    if switch is not None:
        for name in ("gate_proj", "up_proj", "down_proj"):
            if name in switch:
                found.append(switch[name])
    held = moe.__dict__.get("_fast_built")
    if held is not None and held[0] is not None:
        found += [held[0].gate_up, held[0].down]
    return found


class MessageController(nn.Module):
    """Blend a previous pass's message into one sublayer input."""

    def __init__(self, config: TextConfig):
        super().__init__()
        rank, width = config.loop_config.gate_rank, config.hidden_size
        self.down_weight = mx.zeros((rank, 2 * width))
        self.forget_weight = mx.zeros((width, rank))
        self.gain_weight = mx.zeros((width, rank))
        self.eps = config.rms_norm_eps

    def __call__(self, x, message):
        pair = mx.concatenate(
            [mx.fast.rms_norm(v, None, self.eps) for v in (x, message)], axis=-1
        )
        hidden = (pair.astype(self.down_weight.dtype) @ self.down_weight.T).astype(mx.float32)
        forget = mx.tanh(hidden @ _fp32(self, "forget_weight").T)
        gain = mx.tanh(hidden @ _fp32(self, "gain_weight").T)
        return (1 - forget) * x + gain * message


def _record_conv_rollback(conv, records, length, conv_mask, batch):
    """Inkling v1's exact recurrent rollback record over the four histories."""
    previous = [record[0] for record in records]

    def restore_rows(counts):
        states = []
        for old, inputs, mask in records:
            state = (
                old
                if old is not None
                else mx.zeros((inputs.shape[0], 3, inputs.shape[-1]), mx.float32)
            )
            if mask is None:
                mask = mx.ones(inputs.shape[:2], dtype=mx.bool_)
            ordinal = mx.cumsum(mask.astype(mx.int32), axis=1)
            keep = mask & (ordinal <= mx.array(list(counts))[:, None])
            states.append(mx.contiguous(_next_conv_state(state, inputs, keep)))
        return states

    def restore(count):
        return restore_rows([count] * batch)

    spans = conv.rollback_spans(length, conv_mask)
    if spans is not None:
        conv.record_rollback(
            length, restore, previous, per_row_fn=restore_rows, depths=spans or None
        )


class DecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_block_boundary = layer_idx % config.n_layers_per_attn_res_block == 0
        self.is_sliding = config.layer_is_sliding(layer_idx)
        self.attention_input = (
            AttentionResidual(config, 1 if self.is_sliding else 3) if layer_idx else None
        )
        self.mlp_input = AttentionResidual(config)
        self.self_attn = Attention(config, layer_idx)
        self.mlp = SparseMoE(config)
        names = (
            ("input_layernorm",)
            if self.is_sliding
            else ("q_input_layernorm", "k_input_layernorm", "v_input_layernorm")
        )
        for name in (*names, "post_attention_layernorm"):
            self[name] = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_sconv = ShortConvolution(config.hidden_size, config.sconv_kernel_size, 2)
        self.mlp_sconv = ShortConvolution(config.hidden_size, config.sconv_kernel_size, 3)
        loop = config.loop_config
        if loop is not None and loop.core_start <= layer_idx < loop.core_end:
            self.controllers = {
                name: MessageController(config) for name in ("attn", "k", "v", "mlp")
            }

    def __call__(self, x, blocks, cache=None, messages=None):
        conv = cache[1] if cache is not None else None
        mask = _cache_padding_mask(conv, x.shape[1])
        speculating = conv is not None and conv.speculating
        if speculating:
            conv._inkling_inputs = [None] * 4
        mixed = (
            self.attention_input(blocks, x)
            if self.attention_input is not None
            else x[..., None, :]
        )
        fused_ctrl = _fuse.on(self, "fuse_controllers")
        batched = fused_ctrl and _fuse.batched_rows_ok(x)
        message_norms = None
        if self.is_sliding and messages is None:
            attn = self.self_attn(
                _fuse.rms(self, self.input_layernorm, mixed[..., 0, :]).astype(x.dtype),
                cache=cache,
                conv_mask=mask,
            )
        elif batched:
            # One-row decode: the three controllers and the three
            # input norms as batched chains (fuse.batched_rows_ok says when).
            eps = self.post_attention_layernorm.eps
            if messages is not None:
                message_norms = mx.fast.rms_norm(messages, None, eps)
                blended = _fuse.qkv_controllers(self, mixed, messages, message_norms)
            else:
                blended = mixed
            normed = _fuse.scale_cast(mx.fast.rms_norm(blended, None, eps),
                                      _fuse.norm_stack(self), x)
            _fuse.count(self, "input_norms_batched")
            attn = self.self_attn(*(normed[..., i, :] for i in range(3)),
                                  cache=cache, conv_mask=mask)
        else:
            parts = [mixed[..., min(i, mixed.shape[-2] - 1), :] for i in range(3)]
            if messages is not None and fused_ctrl:
                # Local layers feed one x to all three controllers: norm it once.
                shared = (mx.fast.rms_norm(parts[0], None, self.controllers["attn"].eps)
                          if self.is_sliding else None)
                parts = [
                    _fuse.one_controller(self.controllers[name], part, messages[..., i, :],
                                         x_norm=shared, owner=self)
                    for i, (name, part) in enumerate(zip(("attn", "k", "v"), parts))
                ]
                _fuse.count(self, "controllers_unbatched")
            elif messages is not None:
                parts = [
                    self.controllers[name](part, messages[..., i, :])
                    for i, (name, part) in enumerate(zip(("attn", "k", "v"), parts))
                ]
            norms = (
                (self.input_layernorm,) * 3
                if self.is_sliding
                else (self.q_input_layernorm, self.k_input_layernorm, self.v_input_layernorm)
            )
            parts = [_fuse.rms(self, norm, part).astype(x.dtype) for norm, part in zip(norms, parts)]
            attn = self.self_attn(*parts, cache=cache, conv_mask=mask)
        fused_sconv = _fuse.on(self, "fuse_sconv")
        if fused_sconv:
            if self.is_block_boundary:
                residual = self.attn_sconv(attn, conv, mask, tail="attn_boundary")
                blocks.append(x)
            else:
                residual = self.attn_sconv(attn, conv, mask, tail="attn_residual", extra=(x,))
        else:
            residual = self.attn_sconv(attn, conv, mask).astype(mx.float32)
            if self.is_block_boundary:
                blocks.append(x)
            else:
                residual = residual + x.astype(mx.float32)
        hidden = self.mlp_input(blocks, residual)[..., 0, :]
        if messages is not None and fused_ctrl:
            hidden = _fuse.one_controller(
                self.controllers["mlp"], hidden, messages[..., 3, :],
                message_norm=None if message_norms is None else message_norms[..., 3, :],
                owner=self)
        elif messages is not None:
            hidden = self.controllers["mlp"](hidden, messages[..., 3, :])
        hidden = self.mlp(_fuse.rms(self, self.post_attention_layernorm, hidden).astype(x.dtype))
        if fused_sconv:
            out = self.mlp_sconv(hidden, conv, mask, tail="mlp_residual", extra=(residual, x))
        else:
            out = (residual + self.mlp_sconv(hidden, conv, mask).astype(mx.float32)).astype(
                x.dtype)
        if conv is not None:
            if speculating:
                _record_conv_rollback(
                    conv, conv.__dict__.pop("_inkling_inputs"), x.shape[1], mask, x.shape[0]
                )
            conv.advance(x.shape[1])
        return out


class TextModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.layer_indices = config.layer_indices
        self.loop = config.loop_config
        self.num_passes = self.loop.num_passes if self.loop is not None else 1
        if self.loop is not None:
            self.message_queries = mx.zeros(
                (self.loop.core_end - self.loop.core_start, 4, config.hidden_size)
            )
        self.final_input = AttentionResidual(config)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed(self, inputs):
        return self.embed_norm(self.embed_tokens(inputs))

    def set_num_passes(self, passes):
        """Run the middle block ``passes`` times (varloop, opt-in).

        The checkpoint was trained at ``loop_config.num_passes`` with
        ``is_varloop: false``, so any other count is out of distribution.
        Caches are per visit, so a cache built for one count does not fit
        another (``make_cache`` and the APCv2 layout follow the count)."""
        loop = self.loop
        if loop is None:
            raise ValueError("Inkling 2 has no loop to vary")
        if type(passes) is not int or not 1 <= passes <= MAX_LOOP_PASSES:
            raise ValueError(f"Inkling 2 loop passes must be an integer 1..{MAX_LOOP_PASSES}")
        indices = tuple(range(self.config.num_hidden_layers))
        self.layer_indices = (
            indices[: loop.core_start]
            + indices[loop.core_start : loop.core_end] * passes
            + indices[loop.core_end :]
        )
        self.num_passes = passes

    def __call__(self, inputs, cache=None, input_embeddings=None, capture=None):
        if self.__dict__.get("_moe_nax_scope"):
            # inkling2_moe_nax_gather (opt-in): the NAX sorted expert gather is
            # prefill-only and the trunk decides the phase once (decode,
            # speculating caches and verify windows are not prefill).  The
            # route mode itself is the process global the adapter claims.
            from .. import moe_nax_gather

            with moe_nax_gather.forward_scope(
                inputs if inputs is not None else input_embeddings, cache
            ):
                return self._forward(inputs, cache, input_embeddings, capture)
        return self._forward(inputs, cache, input_embeddings, capture)

    def _forward(self, inputs, cache=None, input_embeddings=None, capture=None):
        holder = self.__dict__.get("_fuse_holder")
        if holder is None:
            return self._forward_body(inputs, cache, input_embeddings, capture)
        holder["memo"] = {}
        try:
            return self._forward_body(inputs, cache, input_embeddings, capture)
        finally:
            holder["memo"] = None

    def _forward_body(self, inputs, cache=None, input_embeddings=None, capture=None):
        x = self.embed(inputs) if input_embeddings is None else input_embeddings
        visits = len(self.layer_indices)
        if cache is None:
            caches = [None] * visits
        else:
            if len(cache) != 2 * visits:
                raise ValueError(
                    "Inkling 2 requires flat alternating KV/four-shortconv planes per layer visit"
                )
            caches = [CacheList(cache[2 * i], cache[2 * i + 1]) for i in range(visits)]
        blocks = []
        bank = None
        prelude = prelude_blocks = None
        kernel_messages = _fuse.on(self, "fuse_residual_kernel") and mx.default_device() == mx.gpu
        loop = self.loop
        for visit, index in enumerate(self.layer_indices):
            if loop is not None and index == loop.core_start:
                if visit == index:
                    prelude, prelude_blocks = x, list(blocks)
                else:
                    # The first block appended in the core is the prelude
                    # partial; completed core blocks and the last partial
                    # supply the next pass's messages.
                    bank_states = [*blocks[len(prelude_blocks) + 1 :], x]
                    if not (kernel_messages and x.shape[0] * x.shape[1] == 1):
                        states = mx.stack(bank_states, axis=-2)
                        bank = mx.fast.rms_norm(states.astype(mx.float32), None, 1e-6)
                    else:
                        bank = bank_states
                    x, blocks = prelude, list(prelude_blocks)
            messages = None
            if bank is not None and index < loop.core_end:
                query = self.message_queries[index - loop.core_start]
                if isinstance(bank, list):
                    # inkling2_fuse_residual_kernel: the same normed softmax
                    # mix as the residuals, one launch (near-exact).
                    from . import fuse_kernels

                    messages = fuse_kernels.softmax_mix(self, bank, query)
                    if messages is None:
                        states = mx.stack(bank, axis=-2)
                        bank = mx.fast.rms_norm(states.astype(mx.float32), None, 1e-6)
                    else:
                        _fuse.count(self, "message_kernel")
                if not isinstance(bank, list) and messages is None:
                    scores = (query @ bank.swapaxes(-1, -2)) * x.shape[-1] ** -0.5
                    messages = mx.softmax(scores, axis=-1) @ bank
            if capture is not None and visit in capture:
                capture[visit] = x
            x = self.layers[index](x, blocks, caches[visit], messages)
        return self.final_input(blocks, x)[..., 0, :]


class Model(nn.Module):
    BASE_LAYOUT = "inkling2-kv-four-shortconv-v1"
    supports_speculative_rollback = True

    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.args = config
        self.model_type = config.model_type
        self.completed_forwards = 0
        self.model = TextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._head_fp32 = None

    @property
    def layers(self):
        return self.model.layers

    @property
    def apc_v2_layout(self):
        # A different loop count has a different visit/plane topology.
        passes, trained = self.model.num_passes, self.config.loop_config.num_passes
        layout = self.BASE_LAYOUT if passes == trained else f"{self.BASE_LAYOUT}-passes{passes}"
        # inkling2_kv_q8 stores the global visits' K/V as int8 rows: a
        # different plane format, so a different layout (never restored
        # into a BF16 lane, nor the reverse).
        if getattr(self, "kernel_policy", {}).get("kv_q8"):
            layout += _kv_q8.LAYOUT_SUFFIX
        return layout

    def loop_receipt(self):
        trained = self.config.loop_config.num_passes
        return {"passes": self.model.num_passes, "trained_passes": trained,
                "visits": len(self.model.layer_indices),
                "out_of_distribution": self.model.num_passes != trained}

    def _project_logits(self, h):
        # hf_quant_config: BF16 unembedding with fwd_output_dtype float32.
        # The normed hidden is BF16-rounded, then the product is FP32.
        weight = self.lm_head.weight
        vocab = self.config.unpadded_vocab_size
        policy = getattr(self, "kernel_policy", {})
        if (
            policy.get("head_kernel")
            and weight.dtype == mx.bfloat16
            and weight.shape[1] % 128 == 0
            and mx.default_device() == mx.gpu
        ):
            self.__dict__["head_kernel_calls"] = self.__dict__.get("head_kernel_calls", 0) + 1
            # Each output row is its own reduction, so skipping the padded
            # rows (a leading-row view) leaves the kept logits unchanged.
            logits = head_gemv(h.astype(self.model.embed_norm.weight.dtype), weight[:vocab])
        else:
            h = h.astype(self.model.embed_norm.weight.dtype).astype(mx.float32)
            cached = self._head_fp32
            if cached is None or cached[0] is not weight:
                cached = self._head_fp32 = (weight, weight.astype(mx.float32))
            # Full-width GEMM and softcap, cropped last: cropping earlier
            # changes the kernels (GEMM tiling, strided tanh) and moves the
            # reference path off bit parity with mlx-vlm.
            logits = h @ cached[1].T
        cap = self.config.final_logit_softcapping
        if cap:
            rows = logits.size // logits.shape[-1]
            if (self.__dict__.get("_head_softcap_compiled") and rows <= _COMPILED_HEAD_ROWS
                    and mx.default_device() == mx.gpu):
                # inkling2_head_softcap_compiled: the divide, tanh and multiply
                # as one kernel (omlx#5538 compiles its logit tail). CPU tanh
                # differs under compile, so GPU only; bitwise is a Metal-test
                # question.
                self.__dict__["head_softcap_compiled_calls"] = (
                    self.__dict__.get("head_softcap_compiled_calls", 0) + 1)
                logits = _softcap_compiled(logits, _softcap_scalar(cap))
            else:
                logits = mx.tanh(logits / cap) * cap
        return logits[..., :vocab]

    def __call__(
        self,
        inputs=None,
        cache=None,
        input_embeddings=None,
        inputs_embeds=None,
        skip_logits=False,
        **kwargs,
    ):
        from ...prefill_checkpoint_rows import select_checkpoint_rows

        if inputs is None:
            inputs = kwargs.get("input_ids")
        if inputs_embeds is None:
            inputs_embeds = input_embeddings
        self.completed_forwards += 1
        pre_norm = self.model(inputs, cache, inputs_embeds)
        if skip_logits:
            return None
        return self._project_logits(
            _fuse.rms(self.model, self.model.norm, select_checkpoint_rows(pre_norm)))

    def configure_kernels(self, *, fused_projections=False, head_kernel=False, sink_sdpa=False,
                          loop_passes=None, prefill_mask_mb=None, flash_prefill=False,
                          moe_fused=False, moe_weighted_sum=False, moe_nax_gather=False,
                          sink_slot=False, fuse_controllers=False, fuse_residual=False,
                          fuse_sconv=False, fuse_attention=False, fuse_sconv_kernel=False,
                          fuse_residual_kernel=False, verify_splitk=False, moe_act_quant=False,
                          fuse_gather_lhs=False, fuse_norm_fp32=False, fuse_shared_tau=False,
                          fuse_dconv_kernel=False, fuse_message_kernel=False,
                          compile_decode_blocks=False, head_softcap_compiled=False,
                          kv_q8=False):
        """Adapter-owned opt-in execution switches (all default off).

        ``loop_passes`` None keeps the trained count.  ``moe_weighted_sum``
        needs ``moe_fused``.  ``moe_nax_gather`` only opens the prefill
        phase scope; the NAX route mode is a process global the adapter
        claims (``moe_nax_gather.set_mode``).  ``moe_act_quant`` fake-quantizes
        the expert inputs (``act_quant``; an accuracy experiment, slower) and
        refuses experts with no declared NVFP4/MXFP8 format."""
        if moe_weighted_sum and not moe_fused:
            raise ValueError("Inkling 2 moe_weighted_sum requires moe_fused")
        if fuse_sconv_kernel and not fuse_sconv:
            raise ValueError("Inkling 2 fuse_sconv_kernel requires fuse_sconv")
        if fuse_dconv_kernel and not fuse_sconv:
            raise ValueError("Inkling 2 fuse_dconv_kernel requires fuse_sconv")
        if compile_decode_blocks and not moe_fused:
            raise ValueError("Inkling 2 compile_decode_blocks requires moe_fused")
        if fuse_message_kernel and not fuse_controllers:
            raise ValueError("Inkling 2 fuse_message_kernel requires fuse_controllers")
        if moe_act_quant:
            # Every layer's activation format before anything changes, so a
            # refusal leaves the model as it was.
            for layer in self.model.layers:
                layer.mlp.act_quant_format()
        if kv_q8:
            for layer in self.model.layers:
                attn = layer.self_attn
                reason = None if attn.is_sliding else _kv_q8.geometry_reason(attn.head_dim)
                if reason is not None:
                    raise ValueError(
                        f"Inkling 2 kv_q8 has no int8 layout for head_dim {attn.head_dim} ({reason})")
        flags = {"fuse_controllers": fuse_controllers, "fuse_residual": fuse_residual,
                 "fuse_sconv": fuse_sconv, "fuse_attention": fuse_attention,
                 "fuse_sconv_kernel": fuse_sconv_kernel,
                 "fuse_residual_kernel": fuse_residual_kernel,
                 "fuse_gather_lhs": fuse_gather_lhs, "fuse_norm_fp32": fuse_norm_fp32,
                 "fuse_shared_tau": fuse_shared_tau, "fuse_dconv_kernel": fuse_dconv_kernel,
                 "fuse_message_kernel": fuse_message_kernel}
        enabled = frozenset(name for name, value in flags.items() if value)
        holder = self.model.__dict__.get("_fuse_holder") or {"memo": None}
        object.__setattr__(self.model, "_fuse_holder", holder if enabled else None)
        modules = [self.model, self.model.final_input]
        for layer in self.model.layers:
            modules += [layer, layer.self_attn, layer.attn_sconv, layer.mlp_sconv, layer.mlp_input,
                        layer.self_attn.k_sconv, layer.self_attn.v_sconv]
            if layer.attention_input is not None:
                modules.append(layer.attention_input)
        for module in modules:
            object.__setattr__(module, "_fuse", enabled)
            object.__setattr__(module, "_fuse_holder", holder if enabled else None)
        self.model.set_num_passes(
            self.config.loop_config.num_passes if loop_passes is None else loop_passes)
        object.__setattr__(self.model, "_moe_nax_scope", bool(moe_nax_gather))
        formats = [layer.mlp.set_act_quant(moe_act_quant) for layer in self.model.layers]
        object.__setattr__(self, "_act_quant_formats", formats)
        built = False
        for layer in self.model.layers:
            if moe_fused or layer.mlp.__dict__.get("_fast_fused"):
                built |= moe_fused and "_fast_built" not in layer.mlp.__dict__
                layer.mlp.set_fast_path(moe_fused, moe_weighted_sum)
            object.__setattr__(layer.self_attn, "_sink_sdpa", bool(sink_sdpa))
            object.__setattr__(layer.self_attn, "_sink_slot_on", bool(sink_slot))
            object.__setattr__(layer.self_attn, "_flash_prefill", bool(flash_prefill))
            object.__setattr__(layer.self_attn, "_verify_splitk", bool(verify_splitk))
            # Global visits only: the sliding-window rings stay BF16.
            object.__setattr__(layer.self_attn, "_kv_q8",
                               bool(kv_q8) and not layer.self_attn.is_sliding)
            object.__setattr__(layer.self_attn, "_mask_budget_bytes",
                               None if prefill_mask_mb is None else int(prefill_mask_mb) << 20)
            if fused_projections:
                layer.self_attn.fuse_projections()
            elif getattr(layer.self_attn, "_fused_bounds", None) is not None:
                # The row views stay; only the stacked dispatch is turned off.
                object.__setattr__(layer.self_attn, "_fused_bounds", None)
        from .. import gather_lhs as _gather_lhs

        for layer in self.model.layers:
            for module in _expert_switches(layer.mlp):
                _gather_lhs.set_enabled(module, fuse_gather_lhs)
        object.__setattr__(self, "_compile_decode_blocks", bool(compile_decode_blocks))
        for layer in self.model.layers:
            # The compiled block is kept across toggles (A/B scripts switch
            # per block); it is keyed on the fused build and the weights are
            # its implicit inputs.
            object.__setattr__(layer.mlp, "_compile_decode_blocks", bool(compile_decode_blocks))
        object.__setattr__(self, "_head_softcap_compiled", bool(head_softcap_compiled))
        if built:
            # The replaced gate/up buffers would otherwise stay in MLX's
            # buffer cache (about 10 GB on the Mini), too large for reuse.
            mx.clear_cache()
        object.__setattr__(
            self,
            "kernel_policy",
            {"fused_projections": bool(fused_projections), "head_kernel": bool(head_kernel),
             "sink_sdpa": bool(sink_sdpa), "loop_passes": loop_passes,
             "prefill_mask_mb": prefill_mask_mb, "flash_prefill": bool(flash_prefill),
             "moe_fused": bool(moe_fused), "moe_weighted_sum": bool(moe_weighted_sum),
             "moe_nax_gather": bool(moe_nax_gather), "sink_slot": bool(sink_slot),
             **{name: bool(value) for name, value in flags.items()},
             "verify_splitk": bool(verify_splitk), "moe_act_quant": bool(moe_act_quant),
             "compile_decode_blocks": bool(compile_decode_blocks),
             "head_softcap_compiled": bool(head_softcap_compiled), "kv_q8": bool(kv_q8)},
        )
        if head_kernel:
            # The FP32 head copy is only needed by the reference path.
            self._head_fp32 = None

    def kernel_receipt(self):
        calls = [getattr(layer.self_attn, "fused_calls", 0) for layer in self.model.layers]
        sinks = [layer.self_attn.__dict__.get("sink_sdpa_calls", 0) for layer in self.model.layers]
        flash = [layer.self_attn.__dict__.get("flash_prefill_calls", 0) for layer in self.model.layers]
        slot = [layer.self_attn.__dict__.get("sink_slot_calls", 0) for layer in self.model.layers]
        slot_fb = [layer.self_attn.__dict__.get("sink_slot_fallbacks", 0) for layer in self.model.layers]
        slot_grows = [layer.self_attn.__dict__.get("sink_slot_grows", 0) for layer in self.model.layers]
        slot_reasons = {}
        for layer in self.model.layers:
            for key, value in layer.self_attn.__dict__.get("sink_slot_fallback_reasons", {}).items():
                slot_reasons[key] = slot_reasons.get(key, 0) + value
        splitk_counts = {}
        for layer in self.model.layers:
            for key, value in layer.self_attn.__dict__.get("verify_splitk_counts", {}).items():
                splitk_counts[key] = splitk_counts.get(key, 0) + value
        ring_counts = {}
        for layer in self.model.layers:
            for key, value in layer.self_attn.__dict__.get("splitk_ring_counts", {}).items():
                ring_counts[key] = ring_counts.get(key, 0) + value
        moe = {}
        for layer in self.model.layers:
            for key, value in layer.mlp.__dict__.get("moe_fast_counts", {}).items():
                moe[key] = moe.get(key, 0) + value
        built = [layer.mlp.__dict__.get("_fast_built") for layer in self.model.layers]
        out = {"fused_projection_calls": sum(calls),
               "head_kernel_calls": self.__dict__.get("head_kernel_calls", 0),
               "sink_sdpa_calls": sum(sinks), "flash_prefill_calls": sum(flash),
               "sink_slot_calls": sum(slot), "sink_slot_fallbacks": sum(slot_fb),
               "sink_slot_grows": sum(slot_grows),
               "sink_slot_fallback_reasons": dict(sorted(slot_reasons.items())),
               "verify_splitk_calls": splitk_counts.pop("calls", 0),
               "verify_splitk_declines": splitk_counts,
               # Ring visits under split-K: read in place (calls), reordered
               # because the kernel declined (copies), or a layout the ring
               # read does not map (unmapped, also copied).
               "verify_splitk_ring": dict(sorted(ring_counts.items())),
               "moe_fused_calls": moe.pop("fused", 0),
               "moe_weighted_sum_calls": moe.pop("weighted_sum", 0),
               "compiled_moe_calls": moe.pop("compiled_moe_calls", 0),
               "moe_fast_declines": moe,
               "moe_fused_layers": sum(1 for b in built if b is not None and b[0] is not None),
               "moe_fused_refusals": sorted({b[1] for b in built if b is not None and b[1]})}
        fuse = {}
        for module in self.model.modules():
            for key, value in module.__dict__.get("fuse_counts", {}).items():
                fuse[key] = fuse.get(key, 0) + value
        out["fuse_counts"] = dict(sorted(fuse.items()))
        act = {}
        for layer in self.model.layers:
            for key, value in layer.mlp.__dict__.get("act_quant_counts", {}).items():
                act[key] = act.get(key, 0) + value
        formats = self.__dict__.get("_act_quant_formats") or []
        out["moe_act_quant"] = {
            "calls": dict(sorted(act.items())),
            "layers": {fmt: sum(1 for f in formats if f == fmt) for fmt in _act_quant.FORMATS},
        }
        q8 = {}
        for layer in self.model.layers:
            for key, value in layer.self_attn.__dict__.get("kv_q8_counts", {}).items():
                q8[key] = q8.get(key, 0) + value
        out["kv_q8"] = {
            "selected": bool(getattr(self, "kernel_policy", {}).get("kv_q8")),
            "fidelity": "approximate", "qualified": False, "format": _kv_q8.FORMAT,
            "global_layers": sum(1 for layer in self.model.layers
                                 if layer.self_attn.__dict__.get("_kv_q8")),
            "calls": {key: value for key, value in sorted(q8.items())
                      if key not in ("prefill_tiles", "block_dequant_calls")},
            "kv_q8_prefill_tiles": q8.get("prefill_tiles", 0),
            "kv_q8_block_dequant_calls": q8.get("block_dequant_calls", 0),
            # Process-wide whole-prefix dequantizes (the test oracle only);
            # stays 0 in serving.
            "kv_q8_full_prefix_dequant": _kv_q8.full_prefix_dequant_calls(),
        }
        if getattr(self, "kernel_policy", {}).get("moe_nax_gather"):
            from .. import moe_nax_gather

            status = moe_nax_gather.status()
            out["moe_nax_gather"] = {key: status[key] for key in (
                "mode", "nax_host", "device_name", "calls", "calls_by_format", "fallbacks",
                "not_candidates", "not_prefill")}
        return out

    # ----- external-draft feature taps (DSpark) -----

    def _check_taps(self, capture_layers):
        visits = len(self.model.layer_indices)
        if (
            not capture_layers
            or len(set(capture_layers)) != len(capture_layers)
            or any(type(i) is not int or not 0 <= i <= visits for i in capture_layers)
        ):
            raise ValueError("Inkling 2 taps are distinct visit indices in 0..48")
        return visits

    def tap_outputs(self, pre, taps, capture_layers, *, body_only=False):
        """Logits and features from one final pre-norm hidden.

        A tap ``i < 48`` is the partial block ENTERING visit ``i`` (the
        reference DSpark contract); tap ``48`` is the final normed hidden in
        the activation dtype.  Features are never row-narrowed.
        """
        from ...prefill_checkpoint_rows import select_checkpoint_rows

        visits = self._check_taps(capture_layers)
        normed = _fuse.rms(self.model, self.model.norm, pre)
        dtype = self.model.embed_norm.weight.dtype
        features = mx.concatenate(
            [normed.astype(dtype) if i == visits else taps[i] for i in capture_layers], axis=-1
        )
        if body_only:
            return None, features
        return self._project_logits(select_checkpoint_rows(normed)), features

    def forward_with_taps(self, inputs, cache, capture_layers, *, body_only=False):
        visits = self._check_taps(capture_layers)
        self.completed_forwards += 1
        taps = dict.fromkeys(i for i in capture_layers if i < visits)
        pre = self.model(inputs, cache, capture=taps)
        return self.tap_outputs(pre, taps, capture_layers, body_only=body_only)

    def prefill_body(self, inputs, cache, capture_layers):
        return self.forward_with_taps(inputs, cache, capture_layers, body_only=True)[1]

    def make_cache(self):
        caches = []
        for visit, index in enumerate(self.model.layer_indices):
            conv = InklingConvolutionCache()
            conv.apcv2_layer_index = visit
            caches.extend(
                (
                    RotatingKVCache(self.config.sliding_window_size)
                    if self.config.layer_is_sliding(index)
                    else KVCache(),
                    conv,
                )
            )
        return caches

    # ----- checkpoint loading -----

    _ATTN_NAMES = {
        "wq_du": "q_proj",
        "wk_dv": "k_proj",
        "wv_dv": "v_proj",
        "wr_du": "r_proj",
        "wo_ud": "o_proj",
        "gate_attn": "gate_proj",
        "attn_sink_proj": "sink_proj",
    }
    _NORMS = {
        "attn_norm": "input_layernorm",
        "mlp_norm": "post_attention_layernorm",
        "q_input_norm": "q_input_layernorm",
        "k_input_norm": "k_input_layernorm",
        "v_input_norm": "v_input_layernorm",
    }

    def sanitize(self, weights):
        """Checkpoint (``model.llm.*``) names to this module tree.

        Vision and MTP tensors must already be filtered out.  Routed experts
        come back as ``{layer: {checkpoint suffix: array}}`` under the
        ``"__experts__"`` key; ``expert_weights`` turns them into modules.
        """
        out, experts = {}, {}
        for key, value in weights.items():
            if not key.startswith("model.llm."):
                raise ValueError(f"unexpected Inkling 2 text tensor {key}")
            sub = key[len("model.llm.") :]
            if sub.startswith("layers."):
                layer, sub = sub[len("layers.") :].split(".", 1)
                base = f"model.layers.{layer}."
                if sub.startswith("mlp.experts."):
                    experts.setdefault(int(layer), {})[sub[len("mlp.experts.") :]] = value
                    continue
                if sub.startswith("attn."):
                    name, _, leaf = sub[len("attn.") :].partition(".")
                    sub = self._ATTN_NAMES.get(name, name) + ("." + leaf if leaf else "")
                    sub = {
                        "q_l2_clip.scale": "q_scale",
                        "rel_l2_clip.scale": "rel_scale",
                        "rel_logits_proj.proj": "rel_proj",
                    }.get(sub, sub)
                    for conv in ("k_sconv", "v_sconv"):
                        if sub == conv + ".weight":
                            sub, value = conv + ".conv.weight", value.transpose(0, 2, 1)
                        elif sub == conv + ".static_w":
                            sub = conv + ".static_weight"
                        elif sub == conv + ".weight_proj":
                            sub += ".weight"
                    sub = "self_attn." + sub
                elif sub in ("attn_sconv.weight", "mlp_sconv.weight"):
                    sub, value = sub.replace(".weight", ".conv.weight"), value.transpose(0, 2, 1)
                elif sub == "mlp.gate.weight":
                    sub = "mlp.gate_weight"
                elif sub == "mlp.gate.bias":
                    sub = "mlp.e_score_correction_bias"
                elif sub.startswith("controllers."):
                    pass
                else:
                    name, _, leaf = sub.partition(".")
                    sub = self._NORMS.get(name, name) + "." + leaf
                out[base + sub] = value
            elif sub.startswith("attn_res."):
                rest = sub[len("attn_res.") :]
                if rest == "output_query":
                    target = "final_input.query"
                else:
                    kind, index = rest.split(".")
                    layer = int(index) + (kind == "attn_queries")
                    target = (
                        f"layers.{layer}."
                        f"{'attention_input' if kind == 'attn_queries' else 'mlp_input'}.query"
                    )
                out["model." + target] = value
            elif sub == "unembed.weight":
                out["lm_head.weight"] = value
            elif sub == "embed.weight":
                out["model.embed_tokens.weight"] = value
            elif sub in ("middle_cycle_cross_pass_queries", "middle_cycle_cross_pass_kv_queries"):
                out["model." + sub] = value
            else:
                out["model." + sub] = value
        width = self.config.hidden_size
        cross = "model.middle_cycle_cross_pass_queries"
        if cross in out:
            attn_mlp = out.pop(cross).reshape(-1, 2, width)
            kv = out.pop("model.middle_cycle_cross_pass_kv_queries").reshape(-1, 2, width)
            out["model.message_queries"] = mx.concatenate(
                [attn_mlp[:, :1], kv, attn_mlp[:, 1:]], axis=1
            )
        return out, experts

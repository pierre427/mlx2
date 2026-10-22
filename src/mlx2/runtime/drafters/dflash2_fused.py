# SPDX-License-Identifier: MIT
"""Opt-in DFlash2 layer glue, design provenance: kapitsa-dflash2-fused-drafter.json.

Original MLX kernels; no upstream kernel text copied. Matmuls remain lane,
SDPA/cache append/sliding trim retain the ordinary attention contract.

Law v2 follows the unfused dtype promotion: prepare outputs result_type(hidden,
dynamic), because the unfused base is cast to hidden before arithmetic; finish
also promotes with residual. Hidden/residual reads keep separate input types.
All conv arithmetic remains fp32 with one final output cast, so intermediate
bf16 rounding (including the unfused base cast) is intentionally not reproduced.
The attention finish also emits post-attention RMSNorm from the UNROUNDED
fp32 residual sum, then independently casts residual and normalized stores;
the normalized dtype also promotes with the norm weight. There is no rounded
residual reload between those operations. MLP finish has no cross-layer side
channel and leaves the next layer's input norm alone. Q/K RMSNorm plus default
RoPE retain fp32 intermediates and one rotated-result cast, as in law v1.
This arithmetic is intentionally NOT bit-exact to the ordinary bf16 chain.

Tests use stage atol=0.0625, rtol=0.03 against the bf16 unfused chain and
whole-layer atol=0.125, rtol=0.05 (synthetic inputs; scales in tests); fp32
references use 2e-5. These are test tolerances, not acceptance/quality claims.
Coordinator round 1 at 1545549fb reported 59 Metal passes and two mixed-dtype
whole-layer failures; v2 GPU parity and acceptance A/B remain pending.

Kernel objects are cached lazily, only at GPU dispatch. Small metadata arrays
use a bounded cache keyed by device, dtype and immutable values. Import,
validation, installation and core-only references never build Metal kernels.

"""

import math
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn

STAGES = ("conv-prepare", "conv-finish-residual", "conv-finish-residual-rmsnorm", "qk-rmsnorm-rope")
STAGE_ATOL, STAGE_RTOL = 0.0625, 0.03
LAYER_ATOL, LAYER_RTOL = 0.125, 0.05
_KERNELS = {}
_DTYPES = (mx.float32, mx.float16, mx.bfloat16)

# Shared fp32 conv arithmetic. XT and RT keep hidden/residual reads in their
# respective activation types; OT is the promoted residual output type.
_CONV_ACC = r"""
    float acc = 0.0f;
    for (uint k = 0; k < K; ++k) {
        if (k <= t) {
            const size_t bi = k * base_strides[0] + c * base_strides[1];
            const size_t di = b * dynamic_strides[0] + t * dynamic_strides[1]
                            + k * dynamic_strides[2] + (c / GS) * dynamic_strides[3];
            const size_t hi = b * hidden_strides[0] + (t-k) * hidden_strides[1]
                            + c * hidden_strides[2];
            float coeff = float(base[bi]) + float(dynamic[di]);
            XT hv = hidden[hi];
            acc += coeff * float(hv);
        }
    }
    if (FINISH) {
        RT rv = residual[b * residual_strides[0] + t * residual_strides[1]
                       + c * residual_strides[2]];
        acc += float(rv);
    }
"""

_CONV = r"""
    uint i = thread_position_in_grid.x;
    const uint H = shape[0], L = shape[1], N = shape[2];
    if (i >= N) return;
    const uint c = i % H, row = i / H, t = row % L, b = row / L;
""" + _CONV_ACC + r"""
    output[i] = OT(acc);
"""

# One 256-thread group per BL row. Each thread retains its channels in
# registers across the reduction; shared storage is only eight fp32 sums.
# Normalization reads the UNROUNDED sum, independently of the residual store.
_FINISH_NORM = r"""
    const uint tid = thread_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint row = threadgroup_position_in_grid.x;
    const uint L = shape[1], t = row % L, b = row / L;
    float values[(H + 255) / 256];
    float ss = 0.0f;
    for (uint c = tid; c < H; c += 256) {
""" + _CONV_ACC + r"""
        values[c / 256] = acc;
        ss += acc * acc;
        output[row * H + c] = OT(acc);
    }
    threadgroup float partial[8];
    ss = simd_sum(ss);
    if (lane == 0) partial[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint group = 0; group < 8; ++group) total += partial[group];
    float inv = metal::precise::rsqrt(total / float(H) + eps[0]);
    for (uint c = tid; c < H; c += 256)
        normalized[row * H + c] = NT(values[c / 256] * inv * float(weight[c * weight_strides[0]]));
"""


@lru_cache(maxsize=256)
def _cached_metadata(device_key, dtype, values):
    # Called only on the keyed default device. CPU references do not use this
    # cache; CPU tests can inspect it without preparing a GPU object.
    return mx.array(values, dtype=dtype)


def _metadata(dtype, values):
    return _cached_metadata(str(mx.default_device()), dtype, tuple(values))


# One SIMD group per head-row: no threadgroup memory or cross-group barriers.
# All q/context-k/proposal-k are transformed in one launch. Inputs are BLHD;
# outputs BHLD; context and proposal outputs stay separate for cache ownership.
_QK = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint L = shape[0], S = shape[1], QH = shape[2], KH = shape[3];
    const uint per_batch = L * QH + (S + L) * KH;
    const uint task = threadgroup_position_in_grid.x;
    const uint b = task / per_batch, within = task % per_batch;
    const bool query = within < L * QH;
    const bool context = !query && within < L * QH + S * KH;
    const uint nh = query ? QH : KH;
    const uint nrows = context ? S : L;
    const uint at = query ? within : (context ? within - L * QH : within - L * QH - S * KH);
    const uint pos = at / nh, head = at % nh;
    const size_t src = query
        ? b * q_strides[0] + pos * q_strides[1] + head * q_strides[2]
        : (context ? b * ck_strides[0] + pos * ck_strides[1] + head * ck_strides[2]
                   : b * pk_strides[0] + pos * pk_strides[1] + head * pk_strides[2]);
    const size_t step = query ? q_strides[3] : (context ? ck_strides[3] : pk_strides[3]);
    const uint dst = ((b * nh + head) * nrows + pos) * D;
    float ss = 0.0f;
    for (uint d = lane; d < D; d += 32) {
        float v = query ? float(q[src+d*step]) : (context ? float(ck[src+d*step]) : float(pk[src+d*step]));
        ss += v * v;
    }
    ss = simd_sum(ss);
    float inv = metal::precise::rsqrt(ss / float(D) + params[query ? 0 : 1]);
    float position = float(offsets[b] + (context ? 0 : int(S)) + int(pos));
    for (uint f = lane; f < D/2; f += 32) {
        const uint a = TRAD ? 2*f : f, z = TRAD ? 2*f+1 : f+D/2;
        float va = query ? float(q[src+a*step]) : (context ? float(ck[src+a*step]) : float(pk[src+a*step]));
        float vz = query ? float(q[src+z*step]) : (context ? float(ck[src+z*step]) : float(pk[src+z*step]));
        va *= inv * (query ? float(qw[a*qw_strides[0]]) : float(kw[a*kw_strides[0]]));
        vz *= inv * (query ? float(qw[z*qw_strides[0]]) : float(kw[z*kw_strides[0]]));
        float theta = position * metal::precise::exp(-params[2] * float(f) / float(D/2));
        float co = metal::precise::cos(theta), si = metal::precise::sin(theta);
        T ra = T(va * co - vz * si), rz = T(vz * co + va * si);
        if (query) { oq[dst+a] = ra; oq[dst+z] = rz; }
        else if (context) { ock[dst+a] = ra; ock[dst+z] = rz; }
        else { opk[dst+a] = ra; opk[dst+z] = rz; }
    }
"""


def _kernel(name):
    if mx.default_device() != mx.gpu:
        raise ValueError("draft_kernel 'fused' dispatch requires the GPU; use the reference on CPU")
    # Read MLX view strides directly: projection groups and dynamic stage
    # slices must not acquire a separate contiguity-copy launch.
    if name not in _KERNELS:
        if name == "conv":
            inputs = ["hidden", "dynamic", "base", "residual", "shape"]
            outputs, source = ["output"], _CONV
        elif name == "finish_norm":
            inputs = ["hidden", "dynamic", "base", "residual", "weight", "shape", "eps"]
            outputs, source = ["output", "normalized"], _FINISH_NORM
        else:
            inputs = ["q", "ck", "pk", "qw", "kw", "offsets", "shape", "params"]
            outputs, source = ["oq", "ock", "opk"], _QK
        _KERNELS[name] = mx.fast.metal_kernel(
            name=f"dflash2_fused_{name}_v2", input_names=inputs,
            output_names=outputs, source=source, ensure_row_contiguous=False,
        )
    return _KERNELS[name]


def _conv_shape(hidden, dynamic, base, group_size, residual):
    if hidden.ndim != 3 or min(hidden.shape) <= 0 or hidden.dtype not in _DTYPES:
        raise ValueError("fused conv needs nonempty BLH float32/float16/bfloat16 inputs")
    b, length, h = hidden.shape
    if type(group_size) is not int or group_size <= 0 or h % group_size:
        raise ValueError("fused conv group_size must divide hidden size")
    if base.ndim != 2 or base.shape[1] != h or base.shape[0] <= 0:
        raise ValueError("fused conv base must be [positive kernel_size, hidden_size]")
    k = base.shape[0]
    if dynamic.shape != (b, length, k, h // group_size):
        raise ValueError("fused conv dynamic shape must be [B,L,K,H/group_size]")
    if base.dtype not in _DTYPES or dynamic.dtype not in _DTYPES:
        raise ValueError("fused conv weights must be floating point")
    if residual is not None and (residual.shape != hidden.shape or residual.dtype not in _DTYPES):
        raise ValueError("fused conv residual must match input shape and be floating point")
    return b, length, h, k


def _conv_float(hidden, dynamic, base, group_size, residual=None):
    """Validated, unrounded fp32 sum; tap k reads t-k within each batch row."""
    b, length, h, k = _conv_shape(hidden, dynamic, base, group_size, residual)
    blocks = hidden.astype(mx.float32).reshape(b, length, h // group_size, group_size)
    weights = dynamic.astype(mx.float32)[..., None]
    static = base.astype(mx.float32).reshape(k, h // group_size, group_size)
    out = mx.zeros_like(blocks)
    for tap in range(min(k, length)):
        values = blocks if tap == 0 else mx.concatenate(
            [mx.zeros_like(blocks[:, :tap]), blocks[:, :-tap]], axis=1
        )
        out = out + (static[tap] + weights[:, :, tap]) * values
    out = out.reshape(hidden.shape)
    if residual is not None:
        out = out + residual.astype(mx.float32)
    return out


def conv_output_dtype(hidden, dynamic, residual=None):
    """Unfused dtype law: base casts to hidden, then dynamic can promote it.

    MLX 0.32.4 calls the type-promotion API result_type (not promote_types).
    Adding the residual promotes again; neither input is cast prematurely.
    """
    return mx.result_type(hidden.dtype, dynamic.dtype, *(() if residual is None else (residual.dtype,)))


def conv_reference(hidden, dynamic, base, group_size, residual=None):
    """Core-only fp32 fused law with the unfused output dtype."""
    return _conv_float(hidden, dynamic, base, group_size, residual).astype(
        conv_output_dtype(hidden, dynamic, residual)
    )


def conv(hidden, dynamic, base, group_size, residual=None):
    """Single launch conv prepare, or conv finish + residual."""
    b, length, h, k = _conv_shape(hidden, dynamic, base, group_size, residual)
    return _kernel("conv")(
        inputs=[hidden, dynamic, base, hidden if residual is None else residual,
                _metadata(mx.uint32, (h, length, b * length * h))],
        template=[("XT", hidden.dtype), ("RT", hidden.dtype if residual is None else residual.dtype),
                  ("OT", conv_output_dtype(hidden, dynamic, residual)),
                  ("GS", group_size), ("K", k), ("FINISH", residual is not None)],
        grid=(b * length * h, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[hidden.shape], output_dtypes=[conv_output_dtype(hidden, dynamic, residual)],
    )[0]


def _norm_shape(weight, eps, h):
    if weight.shape != (h,) or weight.dtype not in _DTYPES or not math.isfinite(eps) or eps < 0:
        raise ValueError("fused finish norm requires hidden-width floating weights and finite nonnegative epsilon")


def conv_finish_norm_reference(hidden, dynamic, base, group_size, residual, weight, eps):
    """Both stores cast once; norm uses the unrounded fp32 residual sum."""
    if residual is None:
        raise ValueError("fused finish norm requires a residual")
    out = _conv_float(hidden, dynamic, base, group_size, residual)
    _norm_shape(weight, eps, hidden.shape[-1])
    dtype = conv_output_dtype(hidden, dynamic, residual)
    normalized = out * mx.rsqrt(mx.mean(out * out, axis=-1, keepdims=True) + eps)
    normalized = normalized * weight.astype(mx.float32)
    return out.astype(dtype), normalized.astype(mx.result_type(dtype, weight.dtype))


def conv_finish_norm(hidden, dynamic, base, group_size, residual, weight, eps):
    """Finish + residual + next RMSNorm in one row-reducing launch."""
    if residual is None:
        raise ValueError("fused finish norm requires a residual")
    b, length, h, k = _conv_shape(hidden, dynamic, base, group_size, residual)
    _norm_shape(weight, eps, h)
    dtype = conv_output_dtype(hidden, dynamic, residual)
    norm_dtype = mx.result_type(dtype, weight.dtype)
    return tuple(_kernel("finish_norm")(
        inputs=[hidden, dynamic, base, residual, weight,
                _metadata(mx.uint32, (h, length, b * length * h)), _metadata(mx.float32, (eps,))],
        template=[("XT", hidden.dtype), ("RT", residual.dtype), ("OT", dtype), ("NT", norm_dtype),
                  ("H", h), ("GS", group_size), ("K", k), ("FINISH", True)],
        grid=(b * length * 256, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[hidden.shape, hidden.shape], output_dtypes=[dtype, norm_dtype],
    ))


def _rope_law(rope, dim):
    if (type(rope) is not nn.RoPE or rope.dims != dim or rope.scale != 1.0
            or not isinstance(rope.base, (int, float))
            or not math.isfinite(rope.base) or rope.base <= 0):
        raise ValueError("fused attention supports full-head default nn.RoPE with scale=1 only")
    if type(dim) is not int or dim < 2 or dim > 256 or dim % 2:
        raise ValueError("fused attention head_dim must be even and in 2..256")


def _qk_shape(q, ck, pk, qw, kw, offsets, rope, qeps, keps):
    if q.ndim != 4 or ck.ndim != 4 or pk.ndim != 4:
        raise ValueError("fused qk needs BLHD tensors")
    b, length, qh, d = q.shape
    _rope_law(rope, d)
    if (min(b, length, qh, pk.shape[2]) <= 0 or ck.shape[0] != b
            or ck.shape[2:] != pk.shape[2:] or pk.shape[:2] != (b, length)
            or pk.shape[-1] != d or qh % pk.shape[2]):
        raise ValueError("fused qk incompatible batch, row, or head shapes")
    if (qw.shape != (d,) or kw.shape != (d,) or len(offsets) != b
            or any(type(v) is not int or v < 0 or v + ck.shape[1] + length >= 2**31 for v in offsets)):
        raise ValueError("fused qk needs head norm weights and one nonnegative int32 offset per row")
    if q.dtype not in _DTYPES or ck.dtype != q.dtype or pk.dtype != q.dtype:
        raise ValueError("fused qk inputs must share a floating activation dtype")
    if qw.dtype not in _DTYPES or kw.dtype not in _DTYPES or any(
        not math.isfinite(eps) or eps < 0 for eps in (qeps, keps)
    ):
        raise ValueError("fused qk needs floating weights and finite nonnegative norm epsilon")
    return b, length, ck.shape[1], qh, pk.shape[2], d


def qk_rope_reference(q, ck, pk, qw, kw, offsets, rope, qeps, keps):
    """Core-only norm/rotation with one final cast and per-batch offsets."""
    _, _, span, _, _, d = _qk_shape(q, ck, pk, qw, kw, offsets, rope, qeps, keps)
    out = []
    freq = mx.exp(-math.log(rope.base) * mx.arange(d // 2, dtype=mx.float32) / (d // 2))
    for x, w, eps, delta in ((q, qw, qeps, span), (ck, kw, keps, 0), (pk, kw, keps, span)):
        f = x.astype(mx.float32)
        f = f * mx.rsqrt(mx.mean(f * f, axis=-1, keepdims=True) + eps) * w.astype(mx.float32)
        pos = mx.array(offsets, dtype=mx.float32)[:, None] + delta + mx.arange(x.shape[1], dtype=mx.float32)[None]
        theta = pos[:, :, None, None] * freq
        co, si = mx.cos(theta), mx.sin(theta)
        a, z = (f[..., ::2], f[..., 1::2]) if rope.traditional else (f[..., :d//2], f[..., d//2:])
        ra, rz = a * co - z * si, z * co + a * si
        rotated = mx.stack([ra, rz], axis=-1).reshape(f.shape) if rope.traditional else mx.concatenate([ra, rz], axis=-1)
        out.append(rotated.transpose(0, 2, 1, 3).astype(x.dtype))
    return tuple(out)


def qk_rope(q, ck, pk, qw, kw, offsets, rope, qeps, keps):
    """One launch for q and both key blocks; output layouts are BHLD."""
    b, length, span, qh, kh, d = _qk_shape(q, ck, pk, qw, kw, offsets, rope, qeps, keps)
    return tuple(_kernel("qk")(
        inputs=[q, ck, pk, qw, kw, _metadata(mx.int32, offsets),
                _metadata(mx.uint32, (length, span, qh, kh)),
                _metadata(mx.float32, (qeps, keps, math.log(rope.base)))],
        template=[("T", q.dtype), ("D", d), ("TRAD", rope.traditional)],
        grid=(32 * b * (length * qh + (span + length) * kh), 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(b, qh, length, d), (b, kh, span, d), (b, kh, length, d)],
        output_dtypes=[q.dtype] * 3,
    ))


def _no_extra_attention(attn):
    if getattr(attn, "fold_policy", None) is not None:
        raise ValueError("draft_kernel 'fused' does not support FOLD")
    if getattr(attn, "attention_sink_bias", None) is not None:
        raise ValueError("draft_kernel 'fused' does not support attention sinks")


def validate_model(model):
    """Read-only, before lane installation; no device work or kernel creation."""
    if getattr(model, "fold_policy", None) is not None:
        raise ValueError("draft_kernel 'fused' does not support FOLD")
    if not model.layers:
        raise ValueError("draft_kernel 'fused' requires decoder layers")
    for layer in model.layers:
        attn = layer.self_attn
        _no_extra_attention(attn)
        _rope_law(model.rope, attn.head_dim)
        if attn.n_kv_heads <= 0 or attn.n_heads <= 0 or attn.n_heads % attn.n_kv_heads:
            raise ValueError("fused attention needs positive, divisible query/KV head counts")
        if attn.is_sliding and (attn.sliding_window is None or attn.sliding_window < 2):
            raise ValueError("fused attention sliding_window must be at least 2")
        for norm in (attn.q_norm, attn.k_norm):
            if (norm.weight.shape != (attn.head_dim,) or norm.weight.dtype not in _DTYPES
                    or not math.isfinite(norm.eps) or norm.eps < 0):
                raise ValueError("fused attention requires full-head RMSNorm weights/epsilon")
        _norm_shape(layer.post_attention_layernorm.weight, layer.post_attention_layernorm.eps,
                    model.config.hidden_size)
        for module in (layer.attention_conv, layer.mlp_conv):
            h = model.config.hidden_size
            if (module.group_size <= 0 or h % module.group_size or module.kernel_size <= 0
                    or module.base_kernel.shape != (2, module.kernel_size, h)
                    or module.base_kernel.dtype not in _DTYPES):
                raise ValueError("fused conv unsupported kernel/group/base shape")


def attention(attn, x, x_ctx, rope, cache):
    """Ordinary SDPA/cache law with fused q/k preparation, including B4 rows."""
    _no_extra_attention(attn)
    b, length, _ = x.shape
    span = x_ctx.shape[1]
    entries = list(cache.rows) if hasattr(cache, "rows") else [cache]
    if hasattr(cache, "rows") and len(entries) != b:
        raise ValueError("fused attention cache.rows must match batch size")
    offsets = [entry.offset for entry in entries] if hasattr(cache, "rows") else [cache.offset] * b
    q, ck, cv, pk, pv = attn._projections(x, x_ctx)
    q, ck, pk = qk_rope(
        q.reshape(b, length, attn.n_heads, attn.head_dim),
        ck.reshape(b, span, attn.n_kv_heads, attn.head_dim),
        pk.reshape(b, length, attn.n_kv_heads, attn.head_dim),
        attn.q_norm.weight, attn.k_norm.weight, offsets, rope,
        attn.q_norm.eps, attn.k_norm.eps,
    )
    cv = cv.reshape(b, span, attn.n_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
    pv = pv.reshape(b, length, attn.n_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
    o = attend_cached(attn, q, ck, cv, pk, pv, cache)
    return attn.o_proj(o.transpose(0, 2, 1, 3).reshape(b, length, -1))


def attend_cached(attn, q, ck, cv, pk, pv, cache):
    """Ordinary SDPA + cache append/trim, shared with stage instrumentation."""
    from .dflash_base import append_context_kv

    entries = list(cache.rows) if hasattr(cache, "rows") else [cache]
    outputs = []
    for row, entry in enumerate(entries):
        sl = slice(row, row + 1) if hasattr(cache, "rows") else slice(None)
        keys, values = append_context_kv(entry, ck[sl], cv[sl])
        if attn.is_sliding and keys.shape[2] > attn.sliding_window - 1:
            keys, values = keys[:, :, -(attn.sliding_window-1):], values[:, :, -(attn.sliding_window-1):]
            entry.keys, entry.values, entry._idx = keys, values, keys.shape[2]
        outputs.append(mx.fast.scaled_dot_product_attention(
            q[sl], mx.concatenate([keys, pk[sl]], axis=2),
            mx.concatenate([values, pv[sl]], axis=2), scale=attn.scale,
            mask=None, sinks=attn._sinks(q[sl]),
        ))
    return outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=0)


def layer_forward(layer, x, x_ctx, rope, cache):
    def prepare(module, normalized):
        groups = normalized.shape[-1] // module.group_size
        dynamic = module.kernel_projection(normalized).reshape(
            *normalized.shape[:-1], 2, module.kernel_size, groups
        )
        return conv(normalized, dynamic[..., 0, :, :], module.base_kernel[0], module.group_size), dynamic[..., 1, :, :]

    _no_extra_attention(layer.self_attn)
    residual = x
    x, dynamic = prepare(layer.attention_conv, layer.input_layernorm(x))
    norm = layer.post_attention_layernorm
    x, normalized = conv_finish_norm(layer.self_attn(x, x_ctx, rope, cache), dynamic,
                                    layer.attention_conv.base_kernel[1], layer.attention_conv.group_size,
                                    residual, norm.weight, norm.eps)
    residual = x
    x, dynamic = prepare(layer.mlp_conv, normalized)
    # Preserve the identical input object for gate_proj/up_proj in the MLP.
    return conv(layer.mlp(x), dynamic, layer.mlp_conv.base_kernel[1], layer.mlp_conv.group_size, residual)


def install(model):
    for layer in model.layers:
        object.__setattr__(layer, "_fused_layer", layer_forward)
        object.__setattr__(layer.self_attn, "_fused_forward", attention)

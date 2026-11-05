# SPDX-License-Identifier: MIT
# Adapted from Blaizzy/mlx-vlm; see provenance/inkling-small.json.
from functools import partial
from itertools import pairwise

import mlx.core as mx
from mlx import nn

from .. import invariant_prefill as _invariant
from .. import moe_nax_gather as _nax
from .. import switch_layers as _switch
from ..activations import swiglu
from ..base import scaled_dot_product_attention
from ..cache import (
    BatchRotatingKVCache,
    CacheList,
    KVCache,
    RotatingKVCache,
)


class SwiGLUMLP(nn.Module):
    def __init__(self, input_dims, hidden_dims):
        super().__init__()
        self.gate_proj = nn.Linear(input_dims, hidden_dims, bias=False)
        self.up_proj = nn.Linear(input_dims, hidden_dims, bias=False)
        self.down_proj = nn.Linear(hidden_dims, input_dims, bias=False)

    def __call__(self, x):
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


from ..switch_layers import SwitchGLU, _gather_sort, _scatter_unsort
from .cache import InklingConvolutionCache
from .config import TextConfig as ModelConfig

_MASK_SRC = r"""
    uint j  = thread_position_in_grid.x;   // key   position [0, S)
    uint i  = thread_position_in_grid.y;   // query position [0, LQ)
    uint bh = thread_position_in_grid.z;   // b * H + h
    if (i >= LQ || j >= S || bh >= B * H) return;
    uint b = bh / H, h = bh % H;
    int dist = (int(i) + int(Q_OFF)) - int(j);   // backward distance
    T val;
    if (dist < 0) {
        val = (T)(-1e30f);                                   // causal
    } else if (SLIDING > 0 && dist >= (int)SLIDING) {
        val = (T)(-1e30f);                                   // sliding-window cap
    } else if (dist < (int)REL_EXTENT) {
        float acc = 0.0f;
        uint rbase = ((b * LQ + i) * H + h) * D_REL;
        uint pcol = (uint)dist;
        for (uint d = 0; d < D_REL; ++d)
            acc += (float)rel[rbase + d] * (float)proj[d * REL_EXTENT + pcol];
        val = (T)acc;
    } else {
        val = (T)0;                                          // in-context, outside band
    }
    out[((b * H + h) * LQ + i) * S + j] = val;
"""
_mask_kernel = mx.fast.metal_kernel(
    name="inkling_banded_mask",
    input_names=["rel", "proj"],
    output_names=["out"],
    source=_MASK_SRC,
)


_MASK_V2_SRC = r"""
    // Shape-generic variant of the banded mask: every runtime dimension comes
    // from injected shapes (B/LQ/H from rel, S from the unread shape-carrier
    // input), and the query offset is S - LQ. Only the per-layer constants
    // (dtype, band geometry) are template args, so exactly one pipeline is
    // compiled per layer kind instead of one per (LQ, S) pair.
    const int B  = rel_shape[0];
    const int LQ = rel_shape[1];
    const int H  = rel_shape[2];
    const int S  = kshape_shape[2];
    uint j  = thread_position_in_grid.x;   // key   position [0, S)
    uint i  = thread_position_in_grid.y;   // query position [0, LQ)
    uint bh = thread_position_in_grid.z;   // b * H + h
    if ((int)i >= LQ || (int)j >= S || (int)bh >= B * H) return;
    uint b = bh / H, h = bh % H;
    int dist = ((int)i + (int)qoffset[0]) - (int)j;     // backward distance
    T val;
    if (dist < 0) {
        val = (T)(-1e30f);                                   // causal
    } else if (SLIDING > 0 && dist >= (int)SLIDING) {
        val = (T)(-1e30f);                                   // sliding-window cap
    } else if (dist < (int)REL_EXTENT) {
        float acc = 0.0f;
        const size_t rbase = (size_t)b * rel_strides[0]
            + (size_t)i * rel_strides[1] + (size_t)h * rel_strides[2];
        for (uint d = 0; d < D_REL; ++d)
            acc += (float)rel[rbase + (size_t)d * rel_strides[3]]
                 * (float)proj[(size_t)d * proj_strides[0]
                               + (size_t)dist * proj_strides[1]];
        val = (T)acc;
    } else {
        val = (T)0;                                          // in-context, outside band
    }
    out[(((size_t)b * H + h) * LQ + i) * S + j] = val;
"""
_mask_v2_kernel = mx.fast.metal_kernel(
    name="inkling_banded_mask_v2",
    input_names=["rel", "proj", "kshape", "qoffset"],
    output_names=["out"],
    source=_MASK_V2_SRC,
    ensure_row_contiguous=False,
)


def _rup(a, m):
    return ((a + m - 1) // m) * m


def banded_additive_mask(
    rel,
    proj,
    q_offset,
    S,
    sliding,
    rel_extent,
    left_padding=None,
    shape_ref=None,
    fused=False,
):
    """rel: [B, LQ, H, d_rel]; proj: [d_rel, rel_extent] -> additive mask [B, H, LQ, S]."""
    B, LQ, H, d_rel = rel.shape
    dtype = rel.dtype
    q_offset = int(q_offset)
    S = int(S)
    sliding = int(sliding)
    rel_extent = int(rel_extent)
    if (
        fused
        and shape_ref is not None
        and shape_ref.ndim >= 3
        and shape_ref.shape[2] == S
        and mx.default_device() == mx.gpu
    ):
        mask = _mask_v2_kernel(
            inputs=[rel, proj, shape_ref, mx.array([q_offset], dtype=mx.int32)],
            template=[
                ("T", dtype),
                ("D_REL", d_rel),
                ("REL_EXTENT", rel_extent),
                ("SLIDING", sliding),
            ],
            grid=(_rup(S, 8), _rup(LQ, 8), B * H),
            threadgroup=(8, 8, 1),
            output_shapes=[(B, H, LQ, S)],
            output_dtypes=[dtype],
        )[0]
    elif fused and mx.default_device() == mx.gpu:
        mask = _mask_kernel(
            inputs=[rel, proj],
            template=[
                ("T", dtype),
                ("B", B),
                ("H", H),
                ("LQ", LQ),
                ("S", S),
                ("Q_OFF", q_offset),
                ("D_REL", d_rel),
                ("REL_EXTENT", rel_extent),
                ("SLIDING", sliding),
            ],
            grid=(_rup(S, 8), _rup(LQ, 8), B * H),
            threadgroup=(8, 8, 1),
            output_shapes=[(B, H, LQ, S)],
            output_dtypes=[dtype],
        )[0]
    else:
        rl = (rel @ proj).transpose(0, 2, 1, 3)
        qp = mx.arange(LQ) + q_offset
        kp = mx.arange(S)
        dist = qp[:, None] - kp[None, :]
        gidx = mx.broadcast_to(
            mx.clip(dist, 0, rel_extent - 1)[None, None], (B, H, LQ, S)
        )
        pb = mx.take_along_axis(rl, gidx, axis=-1)
        pb = mx.where((dist >= rel_extent)[None, None], mx.array(0.0, dtype), pb)
        masked = dist < 0
        if sliding > 0:
            masked = masked | (dist >= sliding)
        mask = mx.where(masked[None, None], mx.array(-1e30, dtype), pb)

    if left_padding is not None:
        valid = mx.arange(S)[None, :] >= left_padding[:, None]
        mask = mx.where(valid[:, None, None, :], mask, mx.array(-1e30, dtype))
    return mask.astype(dtype)


def _cache_padding_mask(cache, length):
    if cache is None:
        return None
    # All rows valid (a merged equal-length batch keeps left_padding <= 0):
    # None, as ArraysCache.make_mask, so the fused shortconv decode and the
    # exact rollback spans still engage.  Host mirrors only, never a sync.
    all_valid = getattr(cache, "_host_all_valid", None)
    if callable(all_valid) and all_valid(length):
        return None
    positions = mx.arange(length)[None, :]
    mask = None
    left_padding = getattr(cache, "left_padding", None)
    if left_padding is not None:
        mask = positions >= left_padding[:, None]
    lengths = getattr(cache, "lengths", None)
    if lengths is not None:
        length_mask = positions < lengths[:, None]
        mask = length_mask if mask is None else mask & length_mask
    return mask


def _next_conv_state(state, inputs, mask):
    state_size = state.shape[1]
    if state_size == 0:
        return state
    if mask is None:
        return mx.concatenate([state, inputs], axis=1)[:, -state_size:]

    valid_count = mx.sum(mask, axis=1).astype(mx.int32)
    valid_start = mx.argmax(mask.astype(mx.int32), axis=1)
    logical = mx.arange(state_size)[None, :] + valid_count[:, None] - state_size
    state_indices = state_size + logical
    input_indices = state_size + valid_start[:, None] + mx.maximum(logical, 0)
    indices = mx.where(logical >= 0, input_indices, state_indices)
    combined = mx.concatenate([state, inputs], axis=1)
    indices = mx.broadcast_to(indices[..., None], (*indices.shape, inputs.shape[-1]))
    return mx.take_along_axis(combined, indices, axis=1)


_SCONV_SRC = r"""
    uint c = thread_position_in_grid.x;   // channel
    uint b = thread_position_in_grid.y;   // batch row
    if (c >= C || b >= B) return;
    float w0 = (float)w[c * K + 0];
    float w1 = (float)w[c * K + 1];
    float w2 = (float)w[c * K + 2];
    float w3 = (float)w[c * K + 3];
    // Virtual padded input xp = [state (K-1 rows, fp32); x (L rows, T)].
    const uint begin = PARALLEL ? thread_position_in_grid.z : 0;
    const uint end = PARALLEL ? min(begin + 1, uint(L)) : uint(L);
    for (uint i = begin; i < end; ++i) {
        float acc = 0.0f;
        for (uint k = 0; k < K; ++k) {
            int r = (int)(i + k) - (int)(K - 1);  // row into x; negative -> state
            float v = (r < 0)
                ? state[(b * (K - 1) + (uint)(r + (int)(K - 1))) * C + c]
                : (float)x[(b * L + (uint)r) * C + c];
            float wk = (k == 0) ? w0 : (k == 1) ? w1 : (k == 2) ? w2 : w3;
            acc += wk * v;
        }
        // Match the unfused path's rounding: the conv emits bf16 (rounded)
        // before the fp32 residual add; the layer residual is a second bf16
        // add on top (as the decoder layer's x + sconv(r) was).
        float conv_r = (float)((T)acc);
        T inner = (T)(conv_r + (float)x[(b * L + i) * C + c]);
        if (HAS_RES) {
            out[(b * L + i) * C + c] =
                (T)((float)inner + (float)res[(b * L + i) * C + c]);
        } else {
            out[(b * L + i) * C + c] = inner;
        }
    }
    if (!PARALLEL || begin == 0) for (uint s = 0; s < K - 1; ++s) {
        int r = (int)(L + s) - (int)(K - 1);
        nstate[(b * (K - 1) + s) * C + c] = (r < 0)
            ? state[(b * (K - 1) + (uint)(r + (int)(K - 1))) * C + c]
            : (float)x[(b * L + (uint)r) * C + c];
    }
"""
_sconv_kernel = mx.fast.metal_kernel(
    name="inkling_sconv_decode",
    input_names=["x", "state", "w", "res"],
    output_names=["out", "nstate"],
    source=_SCONV_SRC,
)


class InklingShortConvolution(nn.Module):
    """Depthwise causal 1-D conv over the previous ``kernel_size - 1`` states, plus a
    residual add. Kept in fp32 for stability (matches the reference). ``conv_idx`` selects
    this conv's slot in the layer's shared conv cache."""

    def __init__(self, channels: int, kernel_size: int, conv_idx: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_idx = conv_idx
        self.fused = False
        self.fused_calls = 0
        self.prefill_calls = 0
        self.tree_calls = 0
        self.tree_fused = False
        self.prefill_fused = False
        self.conv = nn.Conv1d(
            channels, channels, kernel_size, groups=channels, bias=False
        )

    def __call__(
        self,
        x: mx.array,
        cache=None,
        mask: mx.array | None = None,
        residual: mx.array | None = None,
    ):
        dt = x.dtype
        K = self.kernel_size
        if cache is not None and getattr(cache, "speculating", False):
            cache._inkling_inputs[self.conv_idx] = (
                cache[self.conv_idx],
                x.astype(mx.float32),
                mask,
            )
        if (
            self.fused
            and cache is not None
            and mask is None
            and K == 4
            and self.conv.weight.dtype == x.dtype
            and (x.shape[1] <= 8 or self.prefill_fused)
            and mx.default_device() == mx.gpu
        ):
            B, L, C = x.shape
            state = cache[self.conv_idx]
            if state is None:
                state = mx.zeros((B, K - 1, C), dtype=mx.float32)
            self.fused_calls += 1
            self.prefill_calls += int(x.shape[1] > 8)
            out, nstate = _sconv_kernel(
                inputs=[
                    mx.contiguous(x),
                    mx.contiguous(state),
                    mx.contiguous(self.conv.weight.reshape(-1)),
                    mx.contiguous(residual if residual is not None else x),
                ],
                template=[
                    ("T", dt),
                    ("B", B),
                    ("L", L),
                    ("C", C),
                    ("K", K),
                    ("HAS_RES", residual is not None),
                    ("PARALLEL", L > 8),
                ],
                grid=(_rup(C, 32), B, L if L > 8 else 1),
                threadgroup=(32, 1, 1),
                output_shapes=[(B, L, C), (B, K - 1, C)],
                output_dtypes=[dt, mx.float32],
            )
            cache[self.conv_idx] = nstate
            return out
        xf = x.astype(mx.float32)
        res = xf
        if mask is not None:
            xf = mx.where(mask[..., None], xf, 0)
        if cache is not None:
            state = cache[self.conv_idx]
            if state is None:
                state = mx.zeros((xf.shape[0], K - 1, xf.shape[-1]), dtype=xf.dtype)
            xp = mx.concatenate([state, xf], axis=1)
            cache[self.conv_idx] = _next_conv_state(state, xf, mask)
        else:
            xp = mx.pad(xf, [(0, 0), (K - 1, 0), (0, 0)])
        out = self.conv(xp.astype(self.conv.weight.dtype)).astype(mx.float32)
        out = (out + res).astype(dt)
        return out if residual is None else residual + out


# Escape hatch: skip the sliding-layer out-of-window K/V slicing.
_SLIDING_KV_SLICE = True
_MASK_BUDGET_BYTES = 64 << 20


class InklingAttention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.fused = False
        self.projection_calls = 0
        self.fused_projection_calls = 0
        self.mask_tiles = 0
        self.fused_mask_calls = 0
        self.attention_policy = "reference"
        self.fused_attention_calls = 0
        self.fused_attention_rows = 0
        self.fused_attention_prefill_calls = 0
        self.attention_reference_calls = 0
        self.fused_attention_nax_calls = 0
        self.mapped_attention_calls = 0
        self.is_sliding = config.layer_is_sliding(layer_idx)
        self.head_dim = config.swa_head_dim if self.is_sliding else config.head_dim
        self.n_heads = (
            config.swa_num_attention_heads
            if self.is_sliding
            else config.num_attention_heads
        )
        self.n_kv = (
            config.swa_num_key_value_heads
            if self.is_sliding
            else config.num_key_value_heads
        )
        self.sliding = config.sliding_window_size if self.is_sliding else 0
        self.rel_extent = (
            config.sliding_window_size if self.is_sliding else config.rel_extent
        )
        self.d_rel = config.d_rel
        self.scale = 1.0 / self.head_dim
        self.log_floor = None if self.is_sliding else config.log_scaling_n_floor
        self.log_alpha = config.log_scaling_alpha

        # q/k/v/r share the input row; their weights are stacked at load
        # (see fuse_qkvr) so decode does one matmul instead of four.
        self.qkvr_dims = (
            self.n_heads * self.head_dim,
            self.n_kv * self.head_dim,
            self.n_kv * self.head_dim,
            self.n_heads * self.d_rel,
        )
        self.qkvr_proj = nn.Linear(config.hidden_size, sum(self.qkvr_dims), bias=False)
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.k_sconv = InklingShortConvolution(
            self.n_kv * self.head_dim, config.sconv_kernel_size, conv_idx=0
        )
        self.v_sconv = InklingShortConvolution(
            self.n_kv * self.head_dim, config.sconv_kernel_size, conv_idx=1
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rel_proj = mx.zeros((self.d_rel, self.rel_extent))

    def project_qkvr(self, x):
        self.projection_calls += 1
        from ...lane.installer import installed

        if self.fused or installed(self.qkvr_proj):
            self.fused_projection_calls += 1
            qkvr = self.qkvr_proj(x)
        else:
            ends = [0]
            for width in self.qkvr_dims:
                ends.append(ends[-1] + width)
            qkvr = mx.concatenate(
                [x @ self.qkvr_proj.weight[a:b].T for a, b in pairwise(ends)], axis=-1
            )
        return qkvr

    def attend(self, q, k, v, r, offset, *, left_padding=None, query_position=None):
        B, _, L, _ = q.shape
        S = k.shape[2]
        from .attention import relative_attention, supported

        # The scalar all-row candidate was correct but slower at 16K on M5 Max
        # (68.7s cold TTFT vs41.3s composed, one directional run, not statistics).
        # NAX128-row prefill also lost the isolated M5 A/B: 11.37 vs6.41ms
        # at16K KV;20.91 vs19.14ms at32K. Eight-query SIMD lost3.14 vs1.89ms.
        # Keep both prefill/wider-verify candidates explicit/default-off.
        # Only global singleton decode at32K won this bounded operator A/B
        # (0.92-0.98 vs1.13ms); this is not generated-token throughput.
        selected = (
            self.attention_policy in ("all", "nax")
            or (
                self.attention_policy == "decode"
                and L == 1
                and S >= 16384
                and not self.sliding
            )
            or (
                self.attention_policy == "mapped"
                and B == 1
                and L == 1
                and left_padding is None
            )
        )
        if selected and supported(q, k, v, r, self.rel_proj.astype(q.dtype)):
            tau = None
            if self.log_floor is not None:
                positions = mx.arange(L)[None, :] + offset + 1
                if query_position is not None:
                    positions = mx.array([[query_position]], mx.float32)
                elif left_padding is not None:
                    positions = positions - left_padding[:, None]
                tau = (
                    1
                    + self.log_alpha
                    * mx.log(
                        mx.maximum(positions.astype(mx.float32) / self.log_floor, 1)
                    )
                ).astype(q.dtype)
                q = q * tau[:, None, :, None]
            self.fused_attention_calls += 1
            self.fused_attention_rows += B * L
            self.fused_attention_prefill_calls += int(L > 16)
            operation = relative_attention
            if self.attention_policy == "mapped":
                from .tree_attention import attend

                self.mapped_attention_calls += 1
                return attend(
                    q,
                    k,
                    v,
                    r,
                    self.rel_proj.astype(q.dtype),
                    scale=self.scale,
                    window=self.sliding,
                    tau=tau,
                )
            if self.attention_policy == "nax" and L > 16:
                from .attention_nax import relative_attention_nax

                if (
                    q.dtype in (mx.bfloat16, mx.float16)
                    and q.shape[1] // k.shape[1] == 4
                ):
                    operation = relative_attention_nax
                    self.fused_attention_nax_calls += 1
            return operation(
                q,
                k,
                v,
                r,
                self.rel_proj.astype(q.dtype),
                offset=offset,
                scale=self.scale,
                window=self.sliding,
                left_padding=left_padding,
                tau=tau,
            )
        self.attention_reference_calls += 1
        # Relative position bias is dense over visible keys. Tile query
        # rows to bound each temporary mask at long context; no full LxS
        # mask is staged for every query in a large prefill chunk.
        rows = max(1, _MASK_BUDGET_BYTES // max(1, B * self.n_heads * S * q.itemsize))
        outputs = []
        for begin in range(0, L, rows):
            end = min(L, begin + rows)
            qr = q[:, :, begin:end]
            mask = banded_additive_mask(
                r[:, begin:end],
                self.rel_proj.astype(q.dtype),
                offset + begin,
                S,
                self.sliding,
                self.rel_extent,
                left_padding=left_padding,
                shape_ref=k if self.fused else None,
                fused=self.fused,
            )
            self.mask_tiles += 1
            self.fused_mask_calls += int(self.fused and mx.default_device() == mx.gpu)
            if self.log_floor is not None:
                qpos = (mx.arange(begin, end) + offset + 1).astype(mx.float32)[None, :]
                if query_position is not None:
                    qpos = mx.array([[query_position]], mx.float32)
                elif left_padding is not None:
                    qpos = qpos - left_padding[:, None]
                tau = 1.0 + self.log_alpha * mx.log(
                    mx.maximum(qpos / self.log_floor, 1.0)
                )
                tau = tau.reshape(
                    B if left_padding is not None else 1, 1, end - begin, 1
                ).astype(q.dtype)
                qr = qr * tau
                mask = mx.where(mask > -1e29, mask * tau, mask)
            outputs.append(
                scaled_dot_product_attention(
                    qr, k, v, cache=None, scale=self.scale, mask=mask
                )
            )
        return mx.concatenate(outputs, axis=2)

    def __call__(self, x, cache=None, conv_mask=None):
        B, L, _ = x.shape
        kv = cache[0] if cache is not None else None
        conv = cache[1] if cache is not None else None

        qkvr = self.project_qkvr(x)
        dq, dk, dv, _ = self.qkvr_dims
        k = self.k_sconv(qkvr[..., dq : dq + dk], cache=conv, mask=conv_mask)
        v = self.v_sconv(qkvr[..., dq + dk : dq + dk + dv], cache=conv, mask=conv_mask)

        k = self.k_norm(k.reshape(B, L, self.n_kv, self.head_dim)).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv, self.head_dim).transpose(0, 2, 1, 3)

        if kv is not None:
            k, v = kv.update_and_fetch(k, v)
            # Learned relative bias needs chronological columns; unlike RoPE,
            # it cannot consume a physical ring with an unrotated mask.
            if type(kv) is RotatingKVCache:
                k, v = kv._temporal_order(kv.keys), kv._temporal_order(kv.values)
            elif type(kv) is BatchRotatingKVCache:
                kv._temporal_order()
                k, v = kv.keys[..., : kv._idx, :], kv.values[..., : kv._idx, :]
        S = k.shape[2]

        q = qkvr[..., :dq]
        r = qkvr[..., dq + dk + dv :].reshape(B, L, self.n_heads, self.d_rel)
        q = self.q_norm(q.reshape(B, L, self.n_heads, self.head_dim)).transpose(
            0, 2, 1, 3
        )
        left_padding = getattr(kv, "left_padding", None)
        if _SLIDING_KV_SLICE and self.sliding > 0 and S > L + self.sliding - 1:
            # No query row can reach keys older than its window; slice them
            # off before the mask and SDPA (they were -1e30 masked anyway).
            # The mask keeps the right positions: its offset is S_eff - L,
            # and any left padding shifts with the slice.
            j0 = S - L - (self.sliding - 1)
            k = k[:, :, j0:, :]
            v = v[:, :, j0:, :]
            S = k.shape[2]
            if left_padding is not None:
                left_padding = mx.maximum(left_padding - j0, 0)
        offset = S - L

        out = self.attend(q, k, v, r, offset, left_padding=left_padding)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class InklingDenseMLP(SwiGLUMLP):
    """Dense SwiGLU MLP (shared ``SwiGLUMLP``) with a learned output scale."""

    def __init__(self, config: ModelConfig):
        super().__init__(config.hidden_size, config.dense_intermediate_size)
        self.global_scale = mx.ones((1,))

    def __call__(self, x):
        return super().__call__(x) * self.global_scale


class InklingSwitchGLU(SwitchGLU):
    def __init__(self, input_dims, hidden_dims, num_experts, **kwargs):
        super().__init__(input_dims, hidden_dims, num_experts, **kwargs)
        self.gate_scale = mx.ones((num_experts,))  # s13 (fused gate/up scale2)
        self.out_scale = mx.ones((num_experts,))  # s13 * s2
        self._scales_trivial = None

    def _per_expert(self, scale, idx, like):
        s = scale[idx].astype(like.dtype)
        return s.reshape(s.shape + (1,) * (like.ndim - s.ndim))

    def scales_trivial(self):
        # Non-NVFP4 checkpoints carry all-ones expert scales; checked once,
        # after load.
        if self._scales_trivial is None:
            self._scales_trivial = bool(
                (mx.all(self.gate_scale == 1) & mx.all(self.out_scale == 1)).item()
            )
        return self._scales_trivial

    def __call__(self, x, indices) -> mx.array:
        # Split-table SwiGLU consumes the original q2/group128 tables:
        # no expanded or concatenated expert bank is retained.
        if self.scales_trivial():
            if (
                _nax.MODE == "fused"
                and not _invariant.active()
                and _switch._RHS_PAD_POLICY == "floor"
                and indices.size >= 20
                and hasattr(self.gate_proj, "bits")
                and _nax.admit_phase()
            ):
                expanded = mx.expand_dims(x, (-2, -3))
                ordered, idx, inverse = _gather_sort(expanded, indices)
                # Match the stock consumer's existing RHS floor before
                # offering a fused epilogue. At E=256, a 128-row prefill has
                # 768 assignments: its stock gathers pad to 1024 streaming
                # rows. Omitting that same pad would make fusion unreachable.
                rows = int(idx.size)
                pad = _switch._rhs_stream_pad(rows, self.gate_proj.num_experts)
                packed_x, packed_idx = (
                    _switch._pad_sorted_tail(ordered, idx, pad)
                    if pad
                    else (ordered, idx)
                )
                hidden = _nax.try_swiglu_split(
                    self.gate_proj, self.up_proj, packed_x, packed_idx
                )
                if hidden is not None:
                    if pad:
                        _switch._record_pad(pad)
                        hidden = hidden[:rows]
                    self._sort_counts["auto_sorted"] += 1
                    result = self.down_proj(hidden, idx, sorted_indices=True)
                    return _scatter_unsort(result, inverse, indices.shape).squeeze(-2)
            return super().__call__(x, indices)
        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
        x_gate = x_gate * self._per_expert(self.gate_scale, idx, x_gate)
        x = self.down_proj(self.activation(x_up, x_gate), idx, sorted_indices=do_sort)
        x = x * self._per_expert(self.out_scale, idx, x)
        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)


_ROUTE_SRC = r"""
    // One simdgroup (32 lanes) per token; each lane owns ceil(R/32) experts
    // (strided by 32 so logit reads coalesce). Top-K via K rounds of
    // simd_max; ties resolved to the lowest lane (deterministic).
    uint lane = thread_position_in_grid.x;   // 0..31, simd lane
    uint n = thread_position_in_grid.y;      // token
    if (n >= N) return;
    const device T* lg = logits + (size_t)n * (R + SH);
    float ws = wscale[0];
    constexpr int PER = (R + 31) / 32;
    float sc[PER];
    float tl_[PER];
    bool taken[PER];
    for (int t = 0; t < PER; ++t) {
        uint j = lane + (uint)t * 32u;
        taken[t] = false;
        if (j < R) {
            float l = (float)lg[j];
            tl_[t] = l;
            sc[t] = 1.0f / (1.0f + metal::exp(-l)) + (float)corr[j];
        } else {
            sc[t] = -INFINITY;
        }
    }
    uint bidx[K];
    float btl[K];
    for (uint kk = 0; kk < K; ++kk) {
        float lb = -INFINITY; int lt = -1;
        for (int t = 0; t < PER; ++t)
            if (!taken[t] && sc[t] > lb) { lb = sc[t]; lt = t; }
        float gb = simd_max(lb);
        ushort wl = (ushort)simd_min(lb == gb ? lane : 32u);
        uint wj = simd_shuffle(lt >= 0 ? lane + (uint)lt * 32u : 0u, wl);
        float wtl = simd_shuffle(lt >= 0 ? tl_[lt] : 0.0f, wl);
        if (lane == (uint)wl && lt >= 0 && sc[lt] == gb) taken[lt] = true;
        bidx[kk] = wj; btl[kk] = wtl;
    }
    // Routing weights: softmax over logsigmoid of the K routed + SH shared
    // logits, times route_scale * global_scale (folded into ws). Computed
    // redundantly on every lane (cheap; K + SH values).
    float lp[K + SH];
    float m = -INFINITY;
    for (uint t = 0; t < K + SH; ++t) {
        float tv = (t < K) ? btl[t] : (float)lg[R + (t - K)];
        float a = -tv;
        float lad = metal::max(a, 0.0f)
                  + metal::log(1.0f + metal::exp(-metal::fabs(a)));
        lp[t] = -lad;
        m = metal::max(m, lp[t]);
    }
    float se = 0.0f;
    for (uint t = 0; t < K + SH; ++t) se += metal::exp(lp[t] - m);
    float lse = m + metal::log(se);
    if (lane < K) {
        idx[(size_t)n * K + lane] = bidx[lane];
        wk[(size_t)n * K + lane] = (T)(metal::exp(lp[lane] - lse) * ws);
    }
    T gv[SH];
    for (uint s = 0; s < SH; ++s) gv[s] = (T)(metal::exp(lp[K + s] - lse) * ws);
    device T* gp = gamma + (size_t)n * SH * I;
    for (uint t = lane; t < SH * I; t += 32u) gp[t] = gv[t / I];
"""
_route_kernel = mx.fast.metal_kernel(
    name="inkling_moe_route",
    input_names=["logits", "corr", "wscale"],
    output_names=["idx", "wk", "gamma"],
    source=_ROUTE_SRC,
)


# Escape hatch: use gather_qmm for the routed down projection at decode.
_DOWN_COMBINE = True

_DOWN_COMBINE_SRC = r"""
    // Weighted routed-expert down-projection for decode: gather_qmm's
    // vector-per-expert mode runs at ~200 GB/s on the [2048 -> 4096] down
    // shape (vs ~750 broadcast), so this kernel dequantizes q2/g128 rows
    // directly: one threadgroup per output row, one simdgroup per selected
    // expert, one quant group per lane; the top-k weighted sum over experts
    // is folded in, so the [N, K, out] intermediate never exists.
    uint lane = thread_index_in_simdgroup;
    uint sg   = simdgroup_index_in_threadgroup;   // expert slot (8 sgs, K used)
    uint row  = threadgroup_position_in_grid.y;   // output row [0, OUT)
    uint n    = threadgroup_position_in_grid.z;   // token
    threadgroup float partial[8];
    if (sg < K) {
        uint e = idx[(size_t)n * K + sg];
        const device uint* wr = wq + ((size_t)e * OUT + row) * (IN / 16u);
        const device T* sr = sc + ((size_t)e * OUT + row) * GROUPS;
        const device T* br = bi + ((size_t)e * OUT + row) * GROUPS;
        const device T* xr = xin + ((size_t)n * K + sg) * IN;
        float s = lane < GROUPS ? (float)sr[lane] : 0.0f;
        float b = lane < GROUPS ? (float)br[lane] : 0.0f;
        float accq = 0.0f, accx = 0.0f;
        uint base = lane * 8u;      // 8 uint32 = 128 packed q2 values
        uint xbase = lane * 128u;
        for (uint u = 0; lane < GROUPS && u < 8u; ++u) {
            uint w8 = wr[base + u];
            for (uint t = 0; t < 16u; ++t) {
                float xv = (float)xr[xbase + u * 16u + t];
                accq += (float)((w8 >> (2u * t)) & 0x3u) * xv;
                accx += xv;
            }
        }
        float dot = simd_sum(s * accq + b * accx);
        if (lane == 0) {
            // match the unfused chain: down output rounds to T, then the
            // per-expert weight multiply rounds again before the sum
            float dv = (float)((T)dot);
            partial[sg] = (float)((T)(dv * (float)wk[(size_t)n * K + sg]));
        }
    } else if (lane == 0) {
        partial[sg] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane == 0) {
        float tot = 0.0f;
        for (uint t = 0; t < 8u; ++t) tot += partial[t];
        out[(size_t)n * OUT + row] = (T)tot;
    }
"""
_down_combine_kernel = mx.fast.metal_kernel(
    name="inkling_moe_down_combine",
    input_names=["xin", "wq", "sc", "bi", "idx", "wk"],
    output_names=["out"],
    source=_DOWN_COMBINE_SRC,
)


@partial(mx.compile, shapeless=True)
def _swiglu_scaled(gate, up, s):
    return nn.silu(gate) * up * s


class InklingSharedExpertsDense(nn.Module):
    """The ``n_shared`` always-on experts as one dense SwiGLU: expert weights are
    concatenated at load (see ``shared_experts_to_dense``) so the fixed-index
    gather_qmm path becomes three plain matmuls. Per-token expert weights arrive
    pre-broadcast over the expert-major intermediate (``gamma``) and are applied
    before down_proj, which distributes over the concatenated experts exactly."""

    def __init__(self, input_dims: int, hidden_dims: int, num_experts: int):
        super().__init__()
        self.gate_proj = nn.Linear(input_dims, num_experts * hidden_dims, bias=False)
        self.up_proj = nn.Linear(input_dims, num_experts * hidden_dims, bias=False)
        self.down_proj = nn.Linear(num_experts * hidden_dims, input_dims, bias=False)

    def __call__(self, x, gamma):
        return self.down_proj(_swiglu_scaled(self.gate_proj(x), self.up_proj(x), gamma))


def fuse_qkvr(weights):
    """Stack per-layer q/k/v/r projection tensors (rows, plus scales/biases for
    quantized checkpoints) into the single ``qkvr_proj``. Row-concat of
    quantized matrices is exact: each output row keeps its own groups."""
    out = dict(weights)
    prefixes = {
        k[: -len("q_proj.weight")]
        for k in weights
        if k.endswith(".self_attn.q_proj.weight")
    }
    for p in prefixes:
        for leaf in ("weight", "scales", "biases"):
            parts = [out.pop(f"{p}{n}_proj.{leaf}", None) for n in "qkvr"]
            if all(v is not None for v in parts):
                out[f"{p}qkvr_proj.{leaf}"] = mx.concatenate(parts, axis=0)
            elif any(v is not None for v in parts):
                raise ValueError(f"partial q/k/v/r {leaf} set under {p}")
    return out


def shared_experts_to_dense(weights):
    """Remap SwitchGLU-shaped shared-expert tensors ``[E, out, in]`` (bf16 or
    quantized triplets) to the dense concatenated layout of
    ``InklingSharedExpertsDense``. Expert-major on the intermediate axis, so
    gate/up stack experts along rows and down stacks along input columns."""
    out = {}
    for k, v in weights.items():
        if ".shared_experts." in k and isinstance(v, mx.array) and v.ndim == 3:
            if ".down_proj." in k:
                v = v.transpose(1, 0, 2).reshape(v.shape[1], -1)
            else:
                v = v.reshape(-1, v.shape[2])
        out[k] = v
    return out


class InklingSparseMoE(nn.Module):
    """Sigmoid-gated fine-grained MoE: top-k routed experts (+ correction-bias selection)
    plus always-on shared experts, weighted by a logsigmoid/logsumexp softmax."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_routed = config.n_routed_experts
        self.n_shared = config.n_shared_experts
        self.top_k = config.num_experts_per_tok
        self.route_scale = config.route_scale
        self.intermediate_size = config.intermediate_size
        # The exact lane stores four output channels per vector. Two dead
        # rows pad the released 258-way router to 260; crop before every
        # selection/weight consumer, so these rows are never expert logits.
        self.router_width = self.n_routed + self.n_shared
        self.router_padded_width = (self.router_width + 3) // 4 * 4
        self.gate = nn.Linear(config.hidden_size, self.router_padded_width, bias=False)
        self.e_score_correction_bias = mx.zeros((self.n_routed,))
        self.global_scale = mx.ones((1,))
        self.switch_mlp = InklingSwitchGLU(
            config.hidden_size, config.intermediate_size, self.n_routed
        )
        self.shared_experts = InklingSharedExpertsDense(
            config.hidden_size, config.intermediate_size, self.n_shared
        )
        self.fused = False
        self.down_combine = False
        # Private-row grouped q2 remains an explicit experiment: component
        # wins over per-node trees did not establish a model-throughput win.
        # The optimized serving profiles keep it off.
        self.grouped = False
        self.grouped_calls = 0
        self.route_calls = 0
        self.down_combine_calls = 0
        self._wscale = None

    def router_logits(self, x):
        return mx.contiguous(self.gate(x)[..., : self.router_width])

    def _route(self, logits):
        """Expert selection + routing weights. On GPU the whole post-matmul
        chain (sigmoid + bias + top-k + logsigmoid softmax + scaling) is one
        kernel; it also emits the shared-expert weights pre-broadcast over the
        expert-major dense intermediate."""
        N = logits.shape[0]
        if self._wscale is None:
            self._wscale = mx.array(
                [self.route_scale], dtype=mx.float32
            ) * self.global_scale.astype(mx.float32)
        if self.fused and mx.default_device() == mx.gpu:
            self.route_calls += 1
            return _route_kernel(
                inputs=[logits, self.e_score_correction_bias, self._wscale],
                template=[
                    ("T", logits.dtype),
                    ("N", N),
                    ("R", self.n_routed),
                    ("SH", self.n_shared),
                    ("K", self.top_k),
                    ("I", self.intermediate_size),
                ],
                grid=(32, N, 1),
                threadgroup=(32, 1, 1),
                output_shapes=[
                    (N, self.top_k),
                    (N, self.top_k),
                    (N, self.n_shared * self.intermediate_size),
                ],
                output_dtypes=[mx.uint32, logits.dtype, logits.dtype],
            )
        scores = mx.sigmoid(logits.astype(mx.float32))
        sfc = scores[:, : self.n_routed] + self.e_score_correction_bias
        idx = mx.argpartition(-sfc, self.top_k - 1, axis=-1)[:, : self.top_k]
        routed_logits = logits[:, : self.n_routed]
        shared_logits = logits[:, -self.n_shared :]
        tl = mx.concatenate(
            [mx.take_along_axis(routed_logits, idx, axis=-1), shared_logits], axis=-1
        ).astype(mx.float32)
        lp = -mx.logaddexp(mx.zeros_like(tl), -tl)
        w = (
            mx.exp(lp - mx.logsumexp(lp, axis=-1, keepdims=True))
            * self.route_scale
            * self.global_scale
        )
        topk_w = w[:, : self.top_k].astype(logits.dtype)
        gamma = mx.repeat(
            w[:, -self.n_shared :].astype(logits.dtype),
            self.intermediate_size,
            axis=-1,
        )
        return idx.astype(mx.uint32), topk_w, gamma

    def __call__(self, x):
        B, L, D = x.shape
        xf = x.reshape(-1, D)
        logits = self.router_logits(xf)
        idx, topk_w, gamma = self._route(logits)
        yr = self._routed(xf, idx, topk_w)
        ys = self.shared_experts(xf, gamma)
        return (yr + ys).reshape(B, L, D).astype(x.dtype)

    def _routed(self, xf, idx, topk_w):
        sm = self.switch_mlp
        if self.grouped:
            from .experts import routed, supported

            if supported(xf, idx, sm):
                self.grouped_calls += 1
                return (routed(xf, idx, sm) * topk_w[..., None]).sum(axis=-2)
        dp = sm.down_proj
        if xf.shape[0] <= 8 and self._can_down_combine(xf):
            # decode: gather_qmm's vector-per-expert mode is ~3.5x off peak on
            # the down shape; dequantize rows directly and fold the weighted
            # expert sum in.
            xe = mx.expand_dims(xf, (-2, -3))
            act = sm.activation(sm.up_proj(xe, idx), sm.gate_proj(xe, idx))
            self.down_combine_calls += 1
            yr = _down_combine_kernel(
                inputs=[act, dp.weight, dp.scales, dp.biases, idx, topk_w],
                template=[
                    ("T", xf.dtype),
                    ("OUT", dp.output_dims),
                    ("IN", dp.input_dims),
                    ("GROUPS", dp.input_dims // 128),
                    ("K", self.top_k),
                ],
                grid=(256, dp.output_dims, xf.shape[0]),
                threadgroup=(256, 1, 1),
                output_shapes=[(xf.shape[0], dp.output_dims)],
                output_dtypes=[xf.dtype],
            )[0]
        else:
            yr = (sm(xf, idx) * topk_w[..., None]).sum(axis=-2)
        return yr

    def _can_down_combine(self, xf):
        sm = self.switch_mlp
        dp = sm.down_proj
        return (
            self.down_combine
            and self.top_k <= 8
            and mx.default_device() == mx.gpu
            and getattr(dp, "bits", None) == 2
            and getattr(dp, "group_size", None) == 128
            and getattr(dp, "mode", "affine") == "affine"
            and getattr(dp, "biases", None) is not None
            and dp.input_dims == 2048  # kernel maps one 128-wide group per lane
            and dp.scales.dtype == xf.dtype
            and sm.scales_trivial()
        )

    def tree(self, x):
        from .experts import supported

        B, L, D = x.shape
        xf = x.reshape(-1, D)
        idx, weights, gamma = self._route(self.router_logits(xf))
        if self.grouped and supported(xf, idx, self.switch_mlp):
            routed = self._routed(xf, idx, weights)
            shared = self.shared_experts(xf, gamma)
            return (routed + shared).reshape(B, L, D).astype(x.dtype)
        # The q2/g128 gate/up gathers keep M=1 and unsorted expert indices;
        # down-combine assigns each token its own grid-z plane. Packing up to
        # eight rows therefore keeps the singleton reduction without invoking
        # SwitchGLU's row-count-dependent sorting. Actual-weight B2/4/8 probes
        # were bit-exact on all 40 routed layers. Preserve the per-row path for
        # other formats, dtypes and down-projection geometries.
        chunk = (
            8
            if supported(xf, idx, self.switch_mlp) and self._can_down_combine(xf)
            else 1
        )
        routed = mx.concatenate(
            [
                self._routed(
                    xf[i : i + chunk], idx[i : i + chunk], weights[i : i + chunk]
                )
                for i in range(0, xf.shape[0], chunk)
            ],
            axis=0,
        )
        shared = self.shared_experts(xf, gamma)
        return (routed + shared).reshape(B, L, D).astype(x.dtype)


class InklingDecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.self_attn = InklingAttention(config, layer_idx)
        self.mlp = (
            InklingDenseMLP(config)
            if config.layer_is_dense(layer_idx)
            else InklingSparseMoE(config)
        )
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attn_sconv = InklingShortConvolution(
            config.hidden_size, config.sconv_kernel_size, conv_idx=2
        )
        self.mlp_sconv = InklingShortConvolution(
            config.hidden_size, config.sconv_kernel_size, conv_idx=3
        )

    def __call__(self, x, cache=None, conv_mask=None):
        conv = cache[1] if cache is not None else None
        if conv_mask is None:
            conv_mask = _cache_padding_mask(conv, x.shape[1])
        if conv is not None and conv.speculating:
            conv._inkling_inputs = [None] * 4
        r = self.self_attn(self.input_layernorm(x), cache=cache, conv_mask=conv_mask)
        h = self.attn_sconv(r, cache=conv, mask=conv_mask, residual=x)
        r = self.mlp(self.post_attention_layernorm(h))
        out = self.mlp_sconv(r, cache=conv, mask=conv_mask, residual=h)
        if conv is not None:
            if conv.speculating:
                records = conv.__dict__.pop("_inkling_inputs")
                previous = [record[0] for record in records]

                def restore_rows(counts):
                    states = []
                    for old, inputs, mask in records:
                        state = (
                            old
                            if old is not None
                            else mx.zeros(
                                (inputs.shape[0], 3, inputs.shape[-1]), mx.float32
                            )
                        )
                        if mask is None:
                            mask = mx.ones(inputs.shape[:2], dtype=mx.bool_)
                        ordinal = mx.cumsum(mask.astype(mx.int32), axis=1)
                        keep = mask & (ordinal <= mx.array(list(counts))[:, None])
                        states.append(
                            mx.contiguous(_next_conv_state(state, inputs, keep))
                        )
                    return states

                def restore(count):
                    return restore_rows([count] * x.shape[0])

                spans = conv.rollback_spans(x.shape[1], conv_mask)
                if spans is not None:
                    conv.record_rollback(
                        x.shape[1],
                        restore,
                        previous,
                        per_row_fn=restore_rows,
                        depths=spans or None,
                    )
            conv.advance(x.shape[1])
        return out


class InklingModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_norm = (
            nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.use_embed_norm
            else None
        )
        self.layers = [
            InklingDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed(self, input_ids):
        h = self.embed_tokens(input_ids)
        if self.embed_norm is not None:
            h = self.embed_norm(h)
        return h

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings: mx.array | None = None,
        skip_final_norm: bool = False,
        capture=None,
    ):
        with _nax.forward_scope(
            inputs if inputs is not None else input_embeddings, cache
        ):
            return self._forward(
                inputs, cache, input_embeddings, skip_final_norm, capture
            )

    def _forward(
        self,
        inputs,
        cache=None,
        input_embeddings: mx.array | None = None,
        skip_final_norm: bool = False,
        capture=None,
    ):
        h = input_embeddings if input_embeddings is not None else self.embed(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        else:
            if len(cache) != 2 * len(self.layers):
                raise ValueError(
                    "Inkling requires flat alternating KV/four-shortconv cache planes"
                )
            cache = [
                CacheList(cache[2 * i], cache[2 * i + 1])
                for i in range(len(self.layers))
            ]
        for i, (layer, c) in enumerate(zip(self.layers, cache, strict=True)):
            h = layer(h, cache=c)
            if capture is not None and i in capture:
                capture[i] = h
        return h if skip_final_norm else self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.args = config
        self.completed_forwards = 0
        self.model_type = config.model_type
        self.model = InklingModel(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def _logits_from_norm(self, h):
        from ...prefill_checkpoint_rows import select_checkpoint_rows

        h = select_checkpoint_rows(h)
        h = h / self.config.logits_mup_width_multiplier
        if self.config.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(h)
        else:
            logits = self.lm_head(h)
        uv = self.config.unpadded_vocab_size
        if uv is not None and uv < logits.shape[-1]:
            logits = logits[..., :uv]
        return logits

    def __call__(
        self,
        inputs=None,
        cache=None,
        input_embeddings=None,
        inputs_embeds=None,
        return_hidden: bool = False,
        return_shared_kv: bool = False,
        skip_logits: bool = False,
        **kwargs,
    ):
        if inputs is None:
            inputs = kwargs.get("input_ids")
        if inputs_embeds is None:
            inputs_embeds = input_embeddings
        self.completed_forwards += 1
        pre_norm = self.model(inputs, cache, inputs_embeds, skip_final_norm=True)
        logits = (
            None if skip_logits else self._logits_from_norm(self.model.norm(pre_norm))
        )
        return logits

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for index in range(len(self.model.layers)):
            conv = InklingConvolutionCache()
            conv.apcv2_layer_index = index
            caches.extend(
                (
                    RotatingKVCache(self.config.sliding_window_size)
                    if self.config.layer_is_sliding(index)
                    else KVCache(),
                    conv,
                )
            )
        return caches


class Model(LanguageModel):
    apc_v2_layout = "inkling-kv-four-shortconv-v1"
    supports_speculative_rollback = True

    def forward_with_taps(self, inputs, cache, capture_layers, *, body_only=False):
        self.completed_forwards += 1
        if not capture_layers or any(
            not 0 <= i < len(self.layers) for i in capture_layers
        ):
            raise ValueError("Inkling target taps require valid layer indices")
        taps = dict.fromkeys(capture_layers)
        pre = self.model(inputs, cache, skip_final_norm=True, capture=taps)
        features = mx.concatenate([taps[i] for i in capture_layers], axis=-1)
        return (
            None if body_only else self._logits_from_norm(self.model.norm(pre))
        ), features

    def prefill_body(self, inputs, cache, capture_layers):
        return self.forward_with_taps(inputs, cache, capture_layers, body_only=True)[1]

    def sanitize(self, weights):
        prefix = "language_model."
        out = {
            key[len(prefix) :]: value
            for key, value in weights.items()
            if key.startswith(prefix)
        }
        for key in list(out):
            if key.endswith(".mlp.gate_weight"):
                weight = out.pop(key)
                pad = (-weight.shape[0]) % 4
                out[key[: -len("gate_weight")] + "gate.weight"] = (
                    mx.pad(weight, [(0, pad), (0, 0)]) if pad else weight
                )
            if key.endswith("switch_mlp.gate_proj.weight"):
                base = key[: -len("gate_proj.weight")]
                count = out[key].shape[0]
                out.setdefault(base + "gate_scale", mx.ones((count,)))
                out.setdefault(base + "out_scale", mx.ones((count,)))
        return fuse_qkvr(shared_experts_to_dense(out))

    def configure_kernels(
        self,
        *,
        fused=False,
        down_combine=False,
        attention="reference",
        grouped=False,
        tree_kernels=False,
        tree_cohort=False,
        shortconv_prefill=False,
    ):
        self.tree_cohort = tree_cohort
        for layer in self.layers:
            layer.self_attn.fused = fused
            layer.self_attn.attention_policy = attention
            for conv in (
                layer.self_attn.k_sconv,
                layer.self_attn.v_sconv,
                layer.attn_sconv,
                layer.mlp_sconv,
            ):
                conv.fused = fused
                conv.tree_fused = tree_kernels
                conv.prefill_fused = shortconv_prefill
            if isinstance(layer.mlp, InklingSparseMoE):
                layer.mlp.fused = fused
                layer.mlp.down_combine = down_combine
                layer.mlp.grouped = grouped

    def kernel_receipt(self):
        return {
            "mapped_attention_calls": sum(
                layer.self_attn.mapped_attention_calls for layer in self.layers
            ),
            "grouped_q2_calls": sum(
                getattr(layer.mlp, "grouped_calls", 0) for layer in self.layers
            ),
            "tree_shortconv_calls": sum(
                conv.tree_calls
                for layer in self.layers
                for conv in (
                    layer.self_attn.k_sconv,
                    layer.self_attn.v_sconv,
                    layer.attn_sconv,
                    layer.mlp_sconv,
                )
            ),
            "fused_attention_nax_calls": sum(
                layer.self_attn.fused_attention_nax_calls for layer in self.layers
            ),
            "fused_attention_calls": sum(
                layer.self_attn.fused_attention_calls for layer in self.layers
            ),
            "fused_attention_rows": sum(
                layer.self_attn.fused_attention_rows for layer in self.layers
            ),
            "fused_attention_prefill_calls": sum(
                layer.self_attn.fused_attention_prefill_calls for layer in self.layers
            ),
            "attention_reference_calls": sum(
                layer.self_attn.attention_reference_calls for layer in self.layers
            ),
            "qkvr_projection_calls": sum(
                layer.self_attn.projection_calls for layer in self.layers
            ),
            "fused_qkvr_projection_calls": sum(
                layer.self_attn.fused_projection_calls for layer in self.layers
            ),
            "relative_mask_tiles": sum(
                layer.self_attn.mask_tiles for layer in self.layers
            ),
            "fused_relative_mask_calls": sum(
                layer.self_attn.fused_mask_calls for layer in self.layers
            ),
            "fused_shortconv_calls": sum(
                conv.fused_calls
                for layer in self.layers
                for conv in (
                    layer.self_attn.k_sconv,
                    layer.self_attn.v_sconv,
                    layer.attn_sconv,
                    layer.mlp_sconv,
                )
            ),
            "fused_prefill_shortconv_calls": sum(
                conv.prefill_calls
                for layer in self.layers
                for conv in (
                    layer.self_attn.k_sconv,
                    layer.self_attn.v_sconv,
                    layer.attn_sconv,
                    layer.mlp_sconv,
                )
            ),
            "fused_router_calls": sum(
                getattr(layer.mlp, "route_calls", 0) for layer in self.layers
            ),
            "fused_q2_down_combine_calls": sum(
                getattr(layer.mlp, "down_combine_calls", 0) for layer in self.layers
            ),
            "expert_sort": [
                layer.mlp.switch_mlp.sort_status()
                for layer in self.layers
                if isinstance(layer.mlp, InklingSparseMoE)
            ],
        }

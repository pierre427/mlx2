"""Experimental fused Xing MLA over the exact, dense latent cache.

One simdgroup owns a (batch, head, query, sequence partition). It fuses the
latent and RoPE score, mask, online softmax and latent value accumulation.
A second kernel merges partition statistics. The caller applies the existing
unembed_out projection. This avoids a [B, H, L, S] positional score tensor.

Default off: numerical and model-path GPU qualification is required before
selection. See provenance/xing4-0-mla-metal.json. The implementation is
original; vLLM-Metal's MLA kernel was used as a design reference.
"""

from __future__ import annotations

import mlx.core as mx

_RANK = 512
_ROPE = 64
_THREADS = 32
_CHUNK = 512
_MAX_QUERY = 4
_MAX_CONTEXT = 131072
_MAX_SCRATCH_BYTES = 64 * 1024 * 1024
STATS = {"fused_calls": 0, "rejected_calls": 0}

_PART_SOURCE = r"""
    const uint row = threadgroup_position_in_grid.x;
    const uint part = threadgroup_position_in_grid.y;
    const uint lane = thread_index_in_simdgroup;
    const int S = kv_shape[2];
    const int b = int(row) / (H * L);
    const int h = (int(row) / L) % H;
    const int query = int(row) % L;
    const int begin = int(part) * CHUNK;
    const int end = metal::min(S, begin + CHUNK);
    constexpr int E = R / 32;
    constexpr int EP = RP / 32;
    float qv[E], qp[EP], accum[E];
    const size_t q_base = size_t(b) * q_strides[0] +
        size_t(h) * q_strides[1] + size_t(query) * q_strides[2];
    const size_t qp_base = size_t(b) * qpe_strides[0] +
        size_t(h) * qpe_strides[1] + size_t(query) * qpe_strides[2];
    const size_t kv_base = size_t(b) * kv_strides[0];
    const size_t kp_base = size_t(b) * kpe_strides[0];
    for (int i = 0; i < E; ++i) {
        qv[i] = float(q[q_base + size_t(lane * E + i) * q_strides[3]]);
        accum[i] = 0.0f;
    }
    for (int i = 0; i < EP; ++i)
        qp[i] = float(qpe[qp_base + size_t(lane * EP + i) * qpe_strides[3]]);
    float maximum = -INFINITY;
    float denominator = 0.0f;
    for (int t = begin; t < end; ++t) {
        if (HAS_MASK) {
            const int mh = MH == 1 ? 0 : h;
            const int mb = mask_shape[0] == 1 ? 0 : b;
            const size_t mi = size_t(mb) * mask_strides[0] +
                size_t(mh) * mask_strides[1] +
                size_t(query) * mask_strides[2] + size_t(t) * mask_strides[3];
            if (!bool(mask[mi])) continue;
        }
        const size_t ki = kv_base + size_t(t) * kv_strides[2];
        const size_t pi = kp_base + size_t(t) * kpe_strides[2];
        float values[E];
        float score = 0.0f;
        for (int i = 0; i < E; ++i) {
            values[i] = float(kv[ki + size_t(lane * E + i) * kv_strides[3]]);
            score += qv[i] * values[i];
        }
        for (int i = 0; i < EP; ++i)
            score += qp[i] * float(kpe[pi + size_t(lane * EP + i) * kpe_strides[3]]);
        score = simd_sum(score) * scale[0];
        const float next_max = metal::max(maximum, score);
        const float old_factor = denominator > 0.0f
            ? metal::exp(maximum - next_max) : 0.0f;
        const float weight = metal::exp(score - next_max);
        denominator = denominator * old_factor + weight;
        maximum = next_max;
        for (int i = 0; i < E; ++i)
            accum[i] = accum[i] * old_factor + weight * values[i];
    }
    const size_t state = size_t(row) * PARTS + part;
    if (lane == 0) {
        part_m[state] = maximum;
        part_l[state] = denominator;
    }
    for (int i = 0; i < E; ++i)
        part_o[state * R + lane * E + i] = accum[i];
"""

_MERGE_SOURCE = r"""
    const uint row = threadgroup_position_in_grid.x;
    const uint lane = thread_index_in_simdgroup;
    constexpr int E = R / 32;
    float maximum = -INFINITY;
    for (int p = 0; p < PARTS; ++p)
        if (part_l[size_t(row) * PARTS + p] > 0.0f)
            maximum = metal::max(maximum, part_m[size_t(row) * PARTS + p]);
    float denominator = 0.0f;
    float accum[E];
    for (int i = 0; i < E; ++i) accum[i] = 0.0f;
    if (maximum != -INFINITY) {
        for (int p = 0; p < PARTS; ++p) {
            const size_t state = size_t(row) * PARTS + p;
            if (part_l[state] <= 0.0f) continue;
            const float factor = metal::exp(part_m[state] - maximum);
            denominator += factor * part_l[state];
            for (int i = 0; i < E; ++i)
                accum[i] += factor * part_o[state * R + lane * E + i];
        }
    }
    const float inverse = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    for (int i = 0; i < E; ++i)
        output[size_t(row) * R + lane * E + i] =
            static_cast<OutT>(accum[i] * inverse);
"""

_PART_KERNEL = None
_MERGE_KERNEL = None


def _kernels():
    global _PART_KERNEL, _MERGE_KERNEL
    if _PART_KERNEL is None:
        _PART_KERNEL = mx.fast.metal_kernel(
            name="mlx2_xing_mla_partial",
            input_names=["q", "qpe", "kv", "kpe", "mask", "scale"],
            output_names=["part_o", "part_l", "part_m"],
            source=_PART_SOURCE,
            ensure_row_contiguous=False,
        )
    if _MERGE_KERNEL is None:
        _MERGE_KERNEL = mx.fast.metal_kernel(
            name="mlx2_xing_mla_merge",
            input_names=["part_o", "part_l", "part_m"],
            output_names=["output"],
            source=_MERGE_SOURCE,
        )
    return _PART_KERNEL, _MERGE_KERNEL


def reset_stats() -> None:
    for key in STATS:
        STATS[key] = 0


def snapshot_stats() -> dict[str, int]:
    return dict(STATS)


def supported(q, qpe, kv, kpe, mask, *, context_len=None) -> bool:
    """Bounded exact-cache shape gate; no host-side array evaluation."""
    if not hasattr(mx.fast, "metal_kernel") or mx.default_device() != mx.gpu:
        return False
    if not all(isinstance(x, mx.array) for x in (q, qpe, kv, kpe)):
        return False
    if any(x.dtype not in (mx.float16, mx.bfloat16) for x in (q, qpe, kv, kpe)):
        return False
    if len({x.dtype for x in (q, qpe, kv, kpe)}) != 1:
        return False
    if any(x.ndim != 4 for x in (q, qpe, kv, kpe)):
        return False
    b, h, length, rank = q.shape
    kv_length = kv.shape[2]
    context = kv_length if context_len is None else context_len
    if not (b >= 1 and h >= 1 and 1 <= length <= _MAX_QUERY
            and length <= context and kv_length <= context <= _MAX_CONTEXT
            and rank == _RANK):
        return False
    parts = (context + _CHUNK - 1) // _CHUNK
    scratch_bytes = b * h * length * parts * (_RANK + 2) * 4
    if scratch_bytes > _MAX_SCRATCH_BYTES:
        return False
    if qpe.shape != (b, h, length, _ROPE):
        return False
    if kv.shape != (b, 1, kv_length, _RANK) or kpe.shape != (b, 1, kv_length, _ROPE):
        return False
    if mask is None:
        return length == 1
    if not isinstance(mask, mx.array) or mask.dtype != mx.bool_:
        return False
    if mask.ndim == 2:
        # BaseModelArgs/create_causal_mask emits [L,S] for short verify.
        return mask.shape == (length, context)
    return (mask.ndim == 4 and mask.shape[0] in (1, b)
            and mask.shape[1] in (1, h)
            and mask.shape[2:] == (length, context))


def attend(q, qpe, kv, kpe, mask, *, scale: float):
    """Return latent attention output [B,H,L,512], or reject unsupported use."""
    if not supported(q, qpe, kv, kpe, mask):
        STATS["rejected_calls"] += 1
        raise ValueError(
            "fused Xing MLA requires GPU BF16/FP16 dense 512+64 cache, L<=4, "
            "a boolean mask and <=64 MiB partition scratch"
        )
    b, h, length, _ = q.shape
    context = kv.shape[2]
    parts = (context + _CHUNK - 1) // _CHUNK
    rows = b * h * length
    if mask is None:
        mask = mx.array([True], dtype=mx.bool_)
    elif mask.ndim == 2:
        mask = mask[None, None, :, :]
    partial, merge = _kernels()
    part_o, part_l, part_m = partial(
        inputs=[q, qpe, kv, kpe, mask, mx.array([scale], dtype=mx.float32)],
        template=[
            ("R", _RANK), ("RP", _ROPE), ("H", h), ("L", length),
            ("MH", mask.shape[1] if mask.ndim == 4 else 1),
            ("HAS_MASK", int(mask.ndim == 4)), ("CHUNK", _CHUNK),
            ("PARTS", parts),
        ],
        grid=(rows * _THREADS, parts, 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[(rows, parts, _RANK), (rows, parts), (rows, parts)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    (output,) = merge(
        inputs=[part_o, part_l, part_m],
        template=[("R", _RANK), ("PARTS", parts), ("OutT", q.dtype)],
        grid=(rows * _THREADS, 1, 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[(rows, _RANK)],
        output_dtypes=[q.dtype],
    )
    STATS["fused_calls"] += 1
    return output.reshape(b, h, length, _RANK)

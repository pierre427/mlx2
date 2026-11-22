# SPDX-License-Identifier: Apache-2.0
"""Near-exact Metal kernels for ``inkling2_fuse_sconv_kernel`` and
``inkling2_fuse_residual_kernel`` (original mlx2 code, 2026-11-22).

Both are GPU only, default off and decline (return ``None``, counted) on
anything outside their layout, so the caller runs the exact path.  Their
arithmetic follows the reference op for op in FP32, but the summation order
of the reductions is the kernels' own, so results are near-exact, not
bit-identical; ``tests/test_inkling2_fuse_metal.py`` measures the gap.

- ``sconv``: a static K=4 short convolution of up to 8 rows (decode and
  verify windows) with no padding mask: the FP32 depthwise conv over
  ``[state; x]``, ``(conv + x) / sqrt(K + 1)``, the BF16 rounding, the
  decoder's residual tail (``fuse.SCONV_TAILS``) and the next state, in one
  launch.  The reference is conv1d (an unfold plus a grouped GEMM on the
  GPU), two concatenations, casts and the elementwise tail.
- ``residual``: a one-row attention-residual mix, ``softmax(rms(S) q /
  sqrt(D)) rms(S)`` over the blocks plus the partial, in one launch instead
  of a stack, casts, an RMS norm, two GEMVs, the scale and a four-op
  softmax.  The loop messages (the same law over the previous pass's bank)
  use it too.
"""

from __future__ import annotations

import math

import mlx.core as mx

from . import fuse as _fuse

_SCONV_ROWS = 8
_TAIL_IDS = {"plain": 0, "attn_boundary": 1, "attn_residual": 2, "mlp_residual": 3}
_SCONV_SOURCE = r"""
    uint c = thread_position_in_grid.x;
    uint b = thread_position_in_grid.y;
    if (c >= C) return;
    const float inv = inv_arr[0];
    float w[4];
    for (int j = 0; j < 4; ++j) w[j] = wt[c * 4 + j];
    float ctx[3 + L];
    for (int j = 0; j < 3; ++j) ctx[j] = HAS_STATE ? state[(b * 3 + j) * C + c] : 0.0f;
    for (int i = 0; i < L; ++i) ctx[3 + i] = float(x[(b * L + i) * C + c]);
    for (int i = 0; i < L; ++i) {
        float acc = 0.0f;
        for (int j = 0; j < 4; ++j) acc += w[j] * ctx[i + j];
        float xf = ctx[3 + i];
        float y = float(T((acc + xf) / inv));
        size_t o = (b * L + i) * C + c;
        if (TAIL == 0) {
            out[o] = OutT(y);
        } else if (TAIL == 1) {
            out[o] = OutT(y);
        } else if (TAIL == 2) {
            out[o] = OutT(y + float(e0[o]));
        } else {
            out[o] = OutT(float(e0[o]) + y);
        }
    }
    for (int j = 0; j < 3; ++j) nstate[(b * 3 + j) * C + c] = ctx[L + j];
"""
_sconv_kernels = {}


def _decline(module, reason):
    _fuse.count(module, "kernel_declined:" + reason)
    return None


def sconv(conv, x, cache, mask, tail, extra):
    """One-launch static short convolution, or ``None`` to run the exact path."""
    from .language import _fp32, _speculative_record

    if mx.default_device() != mx.gpu:
        return _decline(conv, "not_gpu")
    if mask is not None:
        return _decline(conv, "mask")
    if conv.kernel_size != 4 or cache is None:
        return _decline(conv, "layout")
    b, length, channels = x.shape
    if length > _SCONV_ROWS:
        return _decline(conv, "rows")
    state = cache[conv.conv_idx]
    if state is not None and (state.shape != (b, 3, channels) or state.dtype != mx.float32):
        return _decline(conv, "state")
    if getattr(cache, "speculating", False):
        # The rollback record keeps the FP32 inputs (same values).
        _speculative_record(cache, conv.conv_idx, state, x.astype(mx.float32), mask)
    tail_id = _TAIL_IDS[tail]
    out_dtype = {0: x.dtype, 1: mx.float32, 2: mx.float32}.get(tail_id)
    if tail_id == 3:
        out_dtype = extra[1].dtype
        e0 = extra[0]
    elif tail_id == 2:
        e0 = extra[0]
    else:
        e0 = mx.zeros((1,), mx.float32)
    kernel = _sconv_kernels.get("k")
    if kernel is None:
        kernel = _sconv_kernels["k"] = mx.fast.metal_kernel(
            name="inkling2_fuse_sconv",
            input_names=["x", "state", "wt", "inv_arr", "e0"],
            output_names=["out", "nstate"],
            source=_SCONV_SOURCE,
        )
    weight = _fp32(conv.conv, "weight").reshape(channels, 4)
    out, nstate = kernel(
        inputs=[x, state if state is not None else mx.zeros((1,), mx.float32), weight,
                mx.array([math.sqrt(5)], mx.float32), e0],
        template=[("T", x.dtype), ("OutT", out_dtype), ("C", channels), ("L", length),
                  ("TAIL", tail_id), ("HAS_STATE", state is not None)],
        grid=(channels, b, 1),
        threadgroup=(min(256, channels), 1, 1),
        output_shapes=[(b, length, channels), (b, 3, channels)],
        output_dtypes=[out_dtype, mx.float32],
    )
    cache[conv.conv_idx] = nstate
    _fuse.count(conv, "sconv_kernel")
    return out


_RES_TG = 256
_res_kernels = {}


def _residual_source(n):
    loads = "\n".join(
        f"        v[{i}] = float(b{i}[d]);" for i in range(n - 1)
    ) + f"\n        v[{n - 1}] = float(p[d]);"
    return r"""
    constexpr int NS = TG / 32;
    const uint tid = thread_position_in_threadgroup.x;
    const uint qi = threadgroup_position_in_grid.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    threadgroup float red[N * NS];
    threadgroup float inv_s[N];
    threadgroup float wts[N];
    float v[N];
    float acc[N];
    for (int i = 0; i < N; ++i) acc[i] = 0.0f;
    for (int d = tid; d < D; d += TG) {
LOADS
        for (int i = 0; i < N; ++i) acc[i] += v[i] * v[i];
    }
    for (int i = 0; i < N; ++i) {
        float s = simd_sum(acc[i]);
        if (lane == 0) red[i * NS + sg] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < N) {
        float t = 0.0f;
        for (int g = 0; g < NS; ++g) t += red[tid * NS + g];
        inv_s[tid] = metal::precise::rsqrt(t / float(D) + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < N; ++i) acc[i] = 0.0f;
    for (int d = tid; d < D; d += TG) {
LOADS
        const float qd = float(q[qi * D + d]);
        for (int i = 0; i < N; ++i) acc[i] += (v[i] * inv_s[i]) * qd;
    }
    for (int i = 0; i < N; ++i) {
        float s = simd_sum(acc[i]);
        if (lane == 0) red[i * NS + sg] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float sc[N];
        float m = -INFINITY;
        for (int i = 0; i < N; ++i) {
            float t = 0.0f;
            for (int g = 0; g < NS; ++g) t += red[i * NS + g];
            sc[i] = t * scale[0];
            m = max(m, sc[i]);
        }
        float l = 0.0f;
        for (int i = 0; i < N; ++i) { sc[i] = metal::precise::exp(sc[i] - m); l += sc[i]; }
        for (int i = 0; i < N; ++i) wts[i] = sc[i] / l;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int d = tid; d < D; d += TG) {
LOADS
        float o = 0.0f;
        for (int i = 0; i < N; ++i) o += wts[i] * (v[i] * inv_s[i]);
        out[qi * D + d] = o;
    }
""".replace("LOADS", loads)


def softmax_mix(module, states, query):
    """``softmax(rms(S) q^T / sqrt(D), over S) rms(S)`` for one row.

    states: arrays of D elements (one row each), the last one any dtype;
    query: [nq, D] or [D].  Returns [1, 1, nq, D] FP32, or ``None``."""
    width = states[-1].shape[-1]
    n = len(states)
    if n > 16 or any(s.size != width for s in states):
        return _decline(module, "layout")
    kernel = _res_kernels.get(n)
    if kernel is None:
        kernel = _res_kernels[n] = mx.fast.metal_kernel(
            name=f"inkling2_fuse_residual_{n}",
            input_names=[f"b{i}" for i in range(n - 1)] + ["p", "q", "scale"],
            output_names=["out"],
            source=_residual_source(n),
        )
    q = query.reshape(-1, width)
    nq = q.shape[0]
    out = kernel(
        inputs=[*(s.reshape(width) for s in states[:-1]), states[-1].reshape(width), q,
                mx.array([width ** -0.5], mx.float32)],
        template=[("N", n), ("D", width), ("TG", _RES_TG)],
        grid=(nq * _RES_TG, 1, 1),
        threadgroup=(_RES_TG, 1, 1),
        output_shapes=[(1, 1, nq, width)],
        output_dtypes=[mx.float32],
    )[0]
    return out


def residual(module, blocks, partial):
    """One-launch attention-residual mix, or ``None`` to run the exact path."""
    if mx.default_device() != mx.gpu:
        return _decline(module, "not_gpu")
    if not module.normalize_values:
        return _decline(module, "values_not_normalized")
    if partial.shape[0] * partial.shape[1] != 1:
        return _decline(module, "rows")
    out = softmax_mix(module, [*blocks, partial], module.query)
    if out is not None:
        _fuse.count(module, "residual_kernel")
    return out

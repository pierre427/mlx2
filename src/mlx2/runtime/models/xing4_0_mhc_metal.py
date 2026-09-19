"""Fused Metal kernels for the Xing4.0 mHC (manifold-constrained hyper-connection).

``mhc_pre`` computes, per token, the unweighted RMS norm of the flattened
streams, the 24 mixing logits (``@ hc_fn.T``), the pre/post gates, the
clamped Sinkhorn-projected 4x4 combine matrix and the pre-weighted stream
collapse, in one threadgroup and one pass over the streams (plus a second,
cache-resident pass for the collapse).  ``mhc_update`` computes
``post_i * out + sum_j comb[i, j] * streams_j`` for every stream element in
one pass.  All arithmetic is fp32, as in the reference (``xing4_0``
``_mhc_coeffs_body`` / ``_mhc_update_body``); only summation order differs.

The kernels are an execution lever: ``xing4_0`` selects them when enabled
and a startup self-check against the reference math passes, and falls back
to the compiled path otherwise.
"""

from __future__ import annotations

import mlx.core as mx

_THREADS = 256
_MIX = 24
_HC = 4

_PRE_SOURCE = """
    const uint tok = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = tid & 31u;
    const uint sg = tid >> 5;
    constexpr uint K = HC * HID;
    constexpr uint NSG = THREADS / 32;
    const uint base = tok * K;

    float acc[MIX];
    for (uint r = 0; r < MIX; ++r) { acc[r] = 0.0f; }
    float ss = 0.0f;
    for (uint k = tid; k < K; k += THREADS) {
        const float v = float(streams[base + k]);
        ss += v * v;
        for (uint r = 0; r < MIX; ++r) {
            acc[r] += v * float(hc_fn[r * K + k]);
        }
    }
    threadgroup float red[NSG * (MIX + 1)];
    ss = simd_sum(ss);
    for (uint r = 0; r < MIX; ++r) { acc[r] = simd_sum(acc[r]); }
    if (lane == 0) {
        red[sg * (MIX + 1)] = ss;
        for (uint r = 0; r < MIX; ++r) { red[sg * (MIX + 1) + 1 + r] = acc[r]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float pre_s[HC];
    if (tid == 0) {
        float tot[MIX + 1];
        for (uint j = 0; j < MIX + 1; ++j) { tot[j] = 0.0f; }
        for (uint s = 0; s < NSG; ++s) {
            for (uint j = 0; j < MIX + 1; ++j) { tot[j] += red[s * (MIX + 1) + j]; }
        }
        // params: norm_eps, sinkhorn_eps, clamp_lo, clamp_hi, scale[3], base[MIX]
        const float inv = metal::rsqrt(tot[0] / float(K) + params[0]);
        const float eps = params[1];
        const float lo = params[2];
        const float hi = params[3];
        float c[HC * HC];
        for (uint i = 0; i < HC; ++i) {
            const float pre_logit = tot[1 + i] * inv * params[4] + params[7 + i];
            pre_s[i] = 1.0f / (1.0f + metal::exp(-pre_logit));
            const float post_logit = tot[1 + HC + i] * inv * params[5] + params[7 + HC + i];
            post[tok * HC + i] = 2.0f / (1.0f + metal::exp(-post_logit));
        }
        for (uint i = 0; i < HC; ++i) {
            float row_max = -INFINITY;
            for (uint j = 0; j < HC; ++j) {
                const uint m = 2 * HC + i * HC + j;
                float v = tot[1 + m] * inv * params[6] + params[7 + m];
                v = metal::clamp(v, lo, hi);
                c[i * HC + j] = v;
                row_max = metal::max(row_max, v);
            }
            for (uint j = 0; j < HC; ++j) {
                c[i * HC + j] = metal::exp(c[i * HC + j] - row_max);
            }
        }
        for (uint it = 0; it < ITERS; ++it) {
            for (uint i = 0; i < HC; ++i) {
                float s = 0.0f;
                for (uint j = 0; j < HC; ++j) { s += c[i * HC + j]; }
                s += eps;
                for (uint j = 0; j < HC; ++j) { c[i * HC + j] /= s; }
            }
            for (uint j = 0; j < HC; ++j) {
                float s = 0.0f;
                for (uint i = 0; i < HC; ++i) { s += c[i * HC + j]; }
                s += eps;
                for (uint i = 0; i < HC; ++i) { c[i * HC + j] /= s; }
            }
        }
        for (uint m = 0; m < HC * HC; ++m) { comb[tok * HC * HC + m] = c[m]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint h = tid; h < HID; h += THREADS) {
        float s = 0.0f;
        for (uint i = 0; i < HC; ++i) {
            s += pre_s[i] * float(streams[base + i * HID + h]);
        }
        collapsed[tok * HID + h] = static_cast<OutT>(s);
    }
"""

_UPDATE_SOURCE = """
    const uint gid = thread_position_in_grid.x;
    const uint tok = gid / HID;
    const uint h = gid - tok * HID;
    constexpr uint K = HC * HID;
    float s[HC];
    for (uint j = 0; j < HC; ++j) { s[j] = float(streams[tok * K + j * HID + h]); }
    const float o = float(out[tok * HID + h]);
    for (uint i = 0; i < HC; ++i) {
        float v = post[tok * HC + i] * o;
        for (uint j = 0; j < HC; ++j) { v += comb[tok * HC * HC + i * HC + j] * s[j]; }
        result[tok * K + i * HID + h] = static_cast<OutT>(v);
    }
"""

_KERNELS = {}


def _pre_kernel():
    kernel = _KERNELS.get("pre")
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name="xing4_0_mhc_pre",
            input_names=["streams", "hc_fn", "params"],
            output_names=["post", "comb", "collapsed"],
            source=_PRE_SOURCE,
        )
        _KERNELS["pre"] = kernel
    return kernel


def _update_kernel():
    kernel = _KERNELS.get("update")
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name="xing4_0_mhc_update",
            input_names=["streams", "out", "post", "comb"],
            output_names=["result"],
            source=_UPDATE_SOURCE,
        )
        _KERNELS["update"] = kernel
    return kernel


def mhc_pre(streams, hc_fn, params, *, iters):
    """streams [..., 4, H] -> (post [..., 4] f32, comb [..., 4, 4] f32, collapsed [..., H])."""
    *lead, hc, hidden = streams.shape
    if hc != _HC or hc_fn.shape != (_MIX, hc * hidden):
        raise ValueError("mHC kernel expects 4 streams and a [24, 4H] hc_fn")
    tokens = 1
    for size in lead:
        tokens *= size
    flat = mx.contiguous(streams.reshape(tokens, hc * hidden))
    post, comb, collapsed = _pre_kernel()(
        inputs=[flat, mx.contiguous(hc_fn), params],
        template=[
            ("HC", _HC),
            ("HID", hidden),
            ("MIX", _MIX),
            ("ITERS", int(iters)),
            ("THREADS", _THREADS),
            ("OutT", streams.dtype),
        ],
        grid=(tokens * _THREADS, 1, 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[(tokens, _HC), (tokens, _HC * _HC), (tokens, hidden)],
        output_dtypes=[mx.float32, mx.float32, streams.dtype],
    )
    return (
        post.reshape(*lead, _HC),
        comb.reshape(*lead, _HC, _HC),
        collapsed.reshape(*lead, hidden),
    )


def mhc_update(streams, out, post, comb):
    """``post[..., None] * out[..., None, :] + comb @ streams`` -> streams dtype."""
    *lead, hc, hidden = streams.shape
    tokens = 1
    for size in lead:
        tokens *= size
    (result,) = _update_kernel()(
        inputs=[
            mx.contiguous(streams.reshape(tokens, hc * hidden)),
            mx.contiguous(out.reshape(tokens, hidden)),
            mx.contiguous(post.reshape(tokens, hc).astype(mx.float32)),
            mx.contiguous(comb.reshape(tokens, hc * hc).astype(mx.float32)),
        ],
        template=[("HC", hc), ("HID", hidden), ("OutT", streams.dtype)],
        grid=(tokens * hidden, 1, 1),
        threadgroup=(min(256, tokens * hidden), 1, 1),
        output_shapes=[(tokens, hc * hidden)],
        output_dtypes=[streams.dtype],
    )
    return result.reshape(*lead, hc, hidden)


def pack_params(norm_eps, eps, lo, hi, hc_scale, hc_base):
    """The kernel's float parameter block for one HyperConnection."""
    head = mx.array([norm_eps, eps, lo, hi], dtype=mx.float32)
    return mx.concatenate(
        [head, hc_scale.astype(mx.float32).reshape(-1), hc_base.astype(mx.float32).reshape(-1)]
    )

"""Fused Metal kernels for Maxwell-V1-Flash (opt-in execution levers).

``mhc_pre`` computes, per token and in one threadgroup, the stream L2 norm,
the 24 mixing logits (``streams @ mapping_proj.T`` in fp32), the pre/post
gates, the softmax + Sinkhorn ``h_res`` and the pre-weighted stream collapse.
``mhc_post`` computes ``h_post * out + h_res^T @ streams`` for every stream
element in one pass.  The arithmetic is the reference's
(``maxwell_moe._hc_pre_body`` / ``_hc_post_body``), including its rounding of
``h_pre``, ``h_post`` and ``h_res`` to the activation dtype; only summation
order differs.

``head_gemv`` is the fp32 vocabulary head over bf16 weights: each bf16 weight
widens exactly to fp32 (a 16-bit shift), products and sums are fp32 and the
logits are fp32 -- the reference's ``linear(h.float(), W.float())`` without
materialising or reading a fp32 copy of the 200,704 x 2,048 table.

``maxwell_moe`` selects these when enabled; a kernel's first use at a shape is
evaluated inside a guard, and any build or dispatch failure disables it for
the process and falls back to the compiled graph.
"""

from __future__ import annotations

import mlx.core as mx

_THREADS = 256
_HC = 4
_MIX = _HC * _HC + 2 * _HC

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
        for (uint r = 0; r < MIX; ++r) { acc[r] += v * w[r * K + k]; }
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
        // params: eps, alpha_pre, alpha_post, alpha_res, bias[MIX]
        const float eps = params[0];
        const float r = 1.0f / (metal::sqrt(tot[0]) / metal::sqrt(float(K)) + eps);
        float h[MIX];
        for (uint m = 0; m < MIX; ++m) {
            const float alpha = m < HC ? params[1] : (m < 2 * HC ? params[2] : params[3]);
            h[m] = r * tot[1 + m] * alpha + params[4 + m];
        }
        for (uint i = 0; i < HC; ++i) {
            const float pre = 1.0f / (1.0f + metal::exp(-h[i])) + eps;
            pre_s[i] = float(static_cast<OutT>(pre));
            const float post = (1.0f / (1.0f + metal::exp(-h[HC + i]))) * 2.0f;
            h_post[tok * HC + i] = static_cast<OutT>(post);
        }
        float c[HC * HC];
        for (uint i = 0; i < HC; ++i) {
            float row_max = -INFINITY;
            for (uint j = 0; j < HC; ++j) { row_max = metal::max(row_max, h[2 * HC + i * HC + j]); }
            float s = 0.0f;
            for (uint j = 0; j < HC; ++j) {
                c[i * HC + j] = metal::exp(h[2 * HC + i * HC + j] - row_max);
                s += c[i * HC + j];
            }
            for (uint j = 0; j < HC; ++j) { c[i * HC + j] = c[i * HC + j] / s + eps; }
        }
        // column pass, then ITERS - 1 (row, column) passes
        for (uint j = 0; j < HC; ++j) {
            float s = 0.0f;
            for (uint i = 0; i < HC; ++i) { s += c[i * HC + j]; }
            s += eps;
            for (uint i = 0; i < HC; ++i) { c[i * HC + j] /= s; }
        }
        for (uint it = 1; it < ITERS; ++it) {
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
        for (uint m = 0; m < HC * HC; ++m) { h_res[tok * HC * HC + m] = static_cast<OutT>(c[m]); }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = tid; e < HID; e += THREADS) {
        float s = 0.0f;
        for (uint i = 0; i < HC; ++i) { s += float(streams[base + i * HID + e]) * pre_s[i]; }
        collapsed[tok * HID + e] = static_cast<OutT>(s);
    }
"""

_POST_SOURCE = """
    const uint gid = thread_position_in_grid.x;
    const uint tok = gid / HID;
    const uint e = gid - tok * HID;
    constexpr uint K = HC * HID;
    float s[HC];
    for (uint m = 0; m < HC; ++m) { s[m] = float(residual[tok * K + m * HID + e]); }
    const float o = float(out[tok * HID + e]);
    for (uint n = 0; n < HC; ++n) {
        float v = float(h_post[tok * HC + n]) * o;
        float mixed = 0.0f;
        for (uint m = 0; m < HC; ++m) { mixed += float(h_res[tok * HC * HC + m * HC + n]) * s[m]; }
        result[tok * K + n * HID + e] = static_cast<OutT>(v + mixed);
    }
"""

# One simdgroup per vocabulary row; each lane reads 8 bf16 weights (one
# uint4) per step and widens them exactly to fp32.
_HEAD_SOURCE = """
    const uint row = threadgroup_position_in_grid.x * (THREADS / 32) + (thread_position_in_threadgroup.x >> 5);
    const uint lane = thread_position_in_threadgroup.x & 31u;
    if (row >= V) { return; }
    const device uint4* wrow = reinterpret_cast<const device uint4*>(w) + row * (K / 8);
    float acc[T];
    for (uint t = 0; t < T; ++t) { acc[t] = 0.0f; }
    for (uint q = lane; q < K / 8; q += 32) {
        const uint4 p = wrow[q];
        float wv[8];
        wv[0] = as_type<float>(p.x << 16); wv[1] = as_type<float>(p.x & 0xffff0000u);
        wv[2] = as_type<float>(p.y << 16); wv[3] = as_type<float>(p.y & 0xffff0000u);
        wv[4] = as_type<float>(p.z << 16); wv[5] = as_type<float>(p.z & 0xffff0000u);
        wv[6] = as_type<float>(p.w << 16); wv[7] = as_type<float>(p.w & 0xffff0000u);
        for (uint t = 0; t < T; ++t) {
            const device float* hv = h + t * K + q * 8;
            float a = acc[t];
            for (uint j = 0; j < 8; ++j) { a += hv[j] * wv[j]; }
            acc[t] = a;
        }
    }
    for (uint t = 0; t < T; ++t) {
        const float v = simd_sum(acc[t]);
        if (lane == 0) { logits[t * V + row] = v; }
    }
"""

_KERNELS: dict = {}


def _kernel(name, inputs, outputs, source):
    kernel = _KERNELS.get(name)
    if kernel is None:
        kernel = _KERNELS[name] = mx.fast.metal_kernel(
            name=f"maxwell_{name}", input_names=inputs, output_names=outputs, source=source)
    return kernel


def pack_params(alpha_pre, alpha_post, alpha_res, bias, eps):
    head = mx.concatenate([mx.array([eps], dtype=mx.float32), alpha_pre.astype(mx.float32).reshape(-1),
                           alpha_post.astype(mx.float32).reshape(-1), alpha_res.astype(mx.float32).reshape(-1)])
    return mx.concatenate([head, bias.astype(mx.float32).reshape(-1)])


def mhc_pre(x, w, params, *, iters):
    """x [B, T, 4*H] -> (collapsed [B, T, H], h_post [B, T, 4], h_res [B, T, 4, 4]) in x.dtype."""
    b, t, k = x.shape
    hidden = k // _HC
    if w.shape != (_MIX, k) or w.dtype != mx.float32:
        raise ValueError("Maxwell mHC kernel expects a fp32 [24, 4H] mapping_proj")
    tokens = b * t
    collapsed, h_post, h_res = _kernel("mhc_pre", ["streams", "w", "params"],
                                       ["collapsed", "h_post", "h_res"], _PRE_SOURCE)(
        inputs=[mx.contiguous(x.reshape(tokens, k)), w, params],
        template=[("HC", _HC), ("HID", hidden), ("MIX", _MIX), ("ITERS", int(iters)),
                  ("THREADS", _THREADS), ("OutT", x.dtype)],
        grid=(tokens * _THREADS, 1, 1), threadgroup=(_THREADS, 1, 1),
        output_shapes=[(tokens, hidden), (tokens, _HC), (tokens, _HC * _HC)],
        output_dtypes=[x.dtype, x.dtype, x.dtype],
    )
    return (collapsed.reshape(b, t, hidden), h_post.reshape(b, t, _HC),
            h_res.reshape(b, t, _HC, _HC))


def mhc_post(out, residual, h_post, h_res):
    b, t, hidden = out.shape
    tokens = b * t
    (result,) = _kernel("mhc_post", ["out", "residual", "h_post", "h_res"], ["result"], _POST_SOURCE)(
        inputs=[mx.contiguous(out.reshape(tokens, hidden)),
                mx.contiguous(residual.reshape(tokens, _HC * hidden)),
                mx.contiguous(h_post.reshape(tokens, _HC)), mx.contiguous(h_res.reshape(tokens, _HC * _HC))],
        template=[("HC", _HC), ("HID", hidden), ("OutT", residual.dtype)],
        grid=(tokens * hidden, 1, 1), threadgroup=(min(256, tokens * hidden), 1, 1),
        output_shapes=[(tokens, _HC * hidden)], output_dtypes=[residual.dtype],
    )
    return result.reshape(b, t, _HC * hidden)


HEAD_MAX_ROWS = 8


def head_gemv(h, w):
    """fp32 logits ``h.float() @ w.float().T`` for bf16 ``w`` [V, K], ``h`` [..., K], <= 8 rows."""
    *lead, k = h.shape
    rows = 1
    for size in lead:
        rows *= size
    v = w.shape[0]
    if w.dtype != mx.bfloat16 or k % 8 or rows > HEAD_MAX_ROWS:
        raise ValueError("Maxwell head kernel expects bf16 weights, K % 8 == 0 and <= 8 rows")
    per = _THREADS // 32
    (logits,) = _kernel("head_gemv", ["h", "w"], ["logits"], _HEAD_SOURCE)(
        inputs=[mx.contiguous(h.reshape(rows, k).astype(mx.float32)), w],
        template=[("T", rows), ("K", k), ("V", v), ("THREADS", _THREADS)],
        grid=(((v + per - 1) // per) * _THREADS, 1, 1), threadgroup=(_THREADS, 1, 1),
        output_shapes=[(rows, v)], output_dtypes=[mx.float32],
    )
    return logits.reshape(*lead, v)

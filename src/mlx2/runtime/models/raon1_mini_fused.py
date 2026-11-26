# SPDX-License-Identifier: MIT
"""Fused Raon_mHC site kernels (original mlx2 code; provenance/raon1-mini.json).

One mHC site in the ops law is a bf16 selector GEMM, an fp32 RMS, the
coefficient math, the weighted aggregate and (after the branch) the stream
mix: about eight launches, fifty sites per decode step.  These two kernels
do a site in two launches (the branch-input RMSNorm stays ``mx.fast``):

* ``route_aggregate``: one threadgroup per row reads the n*C stream row once,
  accumulates the 17 selector dot products and the sum of squares in fp32,
  rounds each dot to the activation dtype (the ops law's bf16 GEMM output),
  computes h_pre/h_post/h_res with the Helmert chart and writes the
  aggregated branch input in the activation dtype;
* ``mix``: ``h_res @ streams + h_post * branch`` in fp32, rounded once.

Not bitwise to the ops law: the dot products and the 4-term mix sum in a
different order.  Opt-in (``mhc: "fused"``), exactness classified against
the ops law by scripts/raon1_mini_probe.py before any default.
"""

from __future__ import annotations

import math

import mlx.core as mx

STREAMS = 4
SELECTOR_ROWS = 2 * STREAMS + (STREAMS - 1) ** 2  # 17
THREADS = 1024

_ROUTE_SOURCE = """
    constexpr int N = 4;
    constexpr int K = 17;
    constexpr int T = 1024;
    const uint row = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int C = hidden_size;
    const int W = N * C;
    const device T_IN* s = stream + (size_t)row * W;

    float acc[K + 1];
    for (int k = 0; k <= K; ++k) acc[k] = 0.0f;
    for (int i = tid; i < W; i += T) {
        float x = float(s[i]);
        acc[K] += x * x;
        for (int k = 0; k < K; ++k) acc[k] += x * float(weight[(size_t)k * W + i]);
    }
    threadgroup float part[32][K + 1];
    for (int k = 0; k <= K; ++k) {
        float v = simd_sum(acc[k]);
        if (lane == 0) part[sg][k] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float total[K + 1];
    if (tid <= (uint)K) {
        float v = 0.0f;
        for (int g = 0; g < 32; ++g) v += part[g][tid];
        total[tid] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float coef[N + N * N];
    if (tid == 0) {
        float inv = metal::precise::rsqrt(total[K] / float(W) + params[22]);
        float z[K];
        for (int k = 0; k < K; ++k) z[k] = float(T_IN(total[k])) * inv;
        // params: 0 alpha_pre, 1-4 b_pre, 5 alpha_post, 6-9 b_post,
        //         10 alpha_z, 11-19 b_z, 20 gamma, 21 post_factor, 22 eps
        float post[N];
        for (int n = 0; n < N; ++n) {
            float tp = params[0] * z[n] + params[1 + n];
            coef[n] = 1.0f / (1.0f + metal::precise::exp(-tp));
            float tq = params[5] * z[N + n] + params[6 + n];
            post[n] = params[21] / (1.0f + metal::precise::exp(-tq));
            out_post[row * N + n] = post[n];
            out_pre[row * N + n] = coef[n];
        }
        float zz[3][3];
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                zz[a][b] = params[10] * z[2 * N + a * 3 + b] + params[11 + a * 3 + b];
        // Helmert basis Q [4, 3]: column k-1 = (1..1 (k), -k, 0..) / sqrt(k(k+1)).
        float q[N][3];
        for (int i = 0; i < N; ++i)
            for (int k = 1; k < N; ++k) {
                float norm = metal::precise::sqrt(float(k) * float(k + 1));
                q[i][k - 1] = (i < k ? 1.0f : (i == k ? -float(k) : 0.0f)) / norm;
            }
        float e[N][N];
        float rho = 0.0f;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                float v = 0.0f;
                for (int a = 0; a < 3; ++a) {
                    float qa = 0.0f;
                    for (int b = 0; b < 3; ++b) qa += zz[a][b] * q[j][b];
                    v += q[i][a] * qa;
                }
                e[i][j] = v;
                float chart = 0.5f * ((i == j ? 1.0f : 0.0f) + 0.25f);
                rho = metal::max(rho, -v / chart);
            }
        float gamma = params[20];
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                float chart = 0.5f * ((i == j ? 1.0f : 0.0f) + 0.25f);
                float phi = chart + e[i][j] / (1.0f + rho);
                float h = (1.0f - gamma) * (i == j ? 1.0f : 0.0f) + gamma * phi;
                coef[N + i * N + j] = h;
                out_res[(row * N + i) * N + j] = h;
            }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int c = tid; c < C; c += T) {
        float v = 0.0f;
        for (int n = 0; n < N; ++n) v += coef[n] * float(s[n * C + c]);
        out_agg[(size_t)row * C + c] = T_IN(v);
    }
"""

_MIX_SOURCE = """
    constexpr int N = 4;
    const uint gid = thread_position_in_grid.x;
    const int C = hidden_size;
    const uint row = gid / C;
    const uint c = gid % C;
    const device T_IN* s = stream + (size_t)row * N * C;
    float x[N];
    for (int n = 0; n < N; ++n) x[n] = float(s[n * C + c]);
    float b = float(branch[(size_t)row * C + c]);
    for (int i = 0; i < N; ++i) {
        float v = 0.0f;
        for (int j = 0; j < N; ++j) v += res[(row * N + i) * N + j] * x[j];
        v += post[row * N + i] * b;
        out[((size_t)row * N + i) * C + c] = T_IN(v);
    }
"""

_KERNELS: dict = {}


def _kernel(name):
    if name not in _KERNELS:
        if name == "route":
            _KERNELS[name] = mx.fast.metal_kernel(
                name="raon_mhc_route_aggregate", input_names=["stream", "weight", "params", "hidden_size"],
                output_names=["out_pre", "out_post", "out_res", "out_agg"], source=_ROUTE_SOURCE)
        else:
            _KERNELS[name] = mx.fast.metal_kernel(
                name="raon_mhc_mix", input_names=["stream", "branch", "post", "res", "hidden_size"],
                output_names=["out"], source=_MIX_SOURCE)
    return _KERNELS[name]


def site_params(site) -> mx.array:
    """The 23 fp32 site constants (gamma = sigmoid(eta) precomputed)."""
    cached = site.__dict__.get("_fused_params")
    tensors = (site.alpha_pre, site.b_pre, site.alpha_post, site.b_post,
               site.res.alpha_z, site.res.b_z, site.res.eta, site._consts[3])
    if (cached is None or cached[2] != site._eps
            or any(old is not current for old, current in zip(cached[0], tensors))):
        f = lambda a: a.astype(mx.float32).reshape(-1)  # noqa: E731
        params = mx.concatenate([
            f(site.alpha_pre), f(site.b_pre), f(site.alpha_post), f(site.b_post),
            f(site.res.alpha_z), f(site.res.b_z), mx.sigmoid(f(site.res.eta)),
            mx.array([float(site._consts[3]), site._eps], dtype=mx.float32)])
        mx.eval(params)
        cached = (tensors, params, site._eps)
        object.__setattr__(site, "_fused_params", cached)
    return cached[1]


def admits(stream) -> bool:
    return (mx.default_device() == mx.gpu and mx.metal.is_available()
            and stream.ndim == 4 and stream.shape[2] == STREAMS
            and stream.dtype in (mx.bfloat16, mx.float16))


def route_aggregate(site, stream):
    """``stream`` [B, T, 4, C] -> (h_pre, h_post, h_res fp32, aggregate [B, T, C])."""
    b, t, n, c = stream.shape
    rows = b * t
    weight = site.joined(stream.dtype)
    pre, post, res, agg = _kernel("route")(
        inputs=[stream.reshape(rows, n * c), weight, site_params(site), c],
        template=[("T_IN", stream.dtype)],
        grid=(rows * THREADS, 1, 1), threadgroup=(THREADS, 1, 1),
        output_shapes=[(rows, n), (rows, n), (rows, n, n), (rows, c)],
        output_dtypes=[mx.float32, mx.float32, mx.float32, stream.dtype])
    return (pre.reshape(b, t, n), post.reshape(b, t, n), res.reshape(b, t, n, n),
            agg.reshape(b, t, c))


def mix(stream, branch, h_post, h_res):
    b, t, n, c = stream.shape
    rows = b * t
    (out,) = _kernel("mix")(
        inputs=[stream.reshape(rows, n * c), branch.reshape(rows, c).astype(stream.dtype),
                h_post.reshape(rows, n), h_res.reshape(rows, n * n), c],
        template=[("T_IN", stream.dtype)],
        grid=(rows * c, 1, 1), threadgroup=(min(256, c), 1, 1),
        output_shapes=[(rows, n, c)], output_dtypes=[stream.dtype])
    return out.reshape(b, t, n, c)


__all__ = ["admits", "route_aggregate", "mix", "site_params", "math"]

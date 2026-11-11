# SPDX-License-Identifier: MIT
"""Opt-in Mamba-2 SSD prefill scan as 8x8 simdgroup matrix products.

Reimplemented from the algorithm described in nace-ai/kapitsa PR #157
("nh_ssd": chunks of 32 rows, C B^T once per group, one chunk walk per head
with the state held in registers, fp32 8x8 simdgroup products); no Kapitsa
code was read or copied.  See provenance/kapitsa-nemotron-prefill.json.

Same recurrence and the same inputs as :func:`ssm._ssd_seg` (``dtx`` already
rounded to the activation dtype, fp32 ``dtA``), so the masked-token law
(dtA = 0, dtx = 0) is inherited unchanged from :func:`ssm.ssd_prefill`.
Arithmetic differences from the default fused path:

* 32-row chunks instead of 64;
* every product on ``simdgroup_float8x8`` with fp32 accumulation;
* the per-chunk state contribution stays fp32 in registers (the default path
  rounds it to the activation dtype in its ``U`` buffer), so this path is
  closer to an fp32 reference, but not bit-identical to the default.

Two dispatches per segment:

``nemotron_ssd_mma_m`` (chunk x group): stages the chunk's B and C, forms
C B^T once for the group, then for each of the group's heads the inclusive
log-decay cumsum (``simd_prefix_inclusive_sum``) and the decayed lower
triangle ``M[i, j] = (C_i . B_j) exp(lcg_i - lcg_j)``.

``nemotron_ssd_mma_scan`` (head): Dh/8 simdgroups, simdgroup ``s`` owns state
rows ``8s..8s+7`` as Ds/8 register fragments.  Per chunk:
``y^T = exp(lcg) * (S C^T) + dtx^T M^T`` and
``S <- exp(lcg_last) S + dtx^T (w * B)`` with ``w_j = exp(lcg_last - lcg_j)``.

Switch: ``MLX2_NEMOTRON_SSD_MMA`` (off by default; any value but empty/0/false/off/no enables).
"""

from __future__ import annotations

import os
import threading

import mlx.core as mx

SSD_MMA_ENV = "MLX2_NEMOTRON_SSD_MMA"
CHUNK = 32
_THREADGROUP_BYTES = 32768
# Scan reads B and C fragments straight from device memory (L2-shared by the
# group's heads) instead of staging them in threadgroup memory.
DSPLIT = 1
DIRECT_BC = False  # measured 2026-11-10: direct 12.9 ms vs staged 6.8 ms at T=4096
_OFF = {"", "0", "false", "off", "no"}  # apc_numerics "flag" rule

_counts = {"mma": 0, "refused": 0}
_refusals: dict[str, int] = {}

_M_SRC = """
    constexpr int Q = 32;
    constexpr int HPG = H / G;
    constexpr int NT = 256;
    const int c = threadgroup_position_in_grid.x;
    const int g = threadgroup_position_in_grid.y;
    const int b = threadgroup_position_in_grid.z;
    const ushort sg = simdgroup_index_in_threadgroup;
    const ushort lane = thread_index_in_simdgroup;
    const int tid = thread_position_in_threadgroup.x;
    const int t0 = c * Q;
    const int tt = min(Q, T - t0);
    const int nC = (T + Q - 1) / Q;

    threadgroup InT Bs[Q][Ds];
    threadgroup InT Cs[Q][Ds];
    threadgroup float CB[Q][Q];
    threadgroup float lcg[HPG][Q];

    for (int f = tid; f < Q * Ds; f += NT) {
        const int i = f / Ds, s = f % Ds;
        const bool ok = i < tt;
        const size_t src = (((size_t)b * T + t0 + i) * G + g) * Ds + s;
        Bs[i][s] = ok ? Bm[src] : InT(0.0f);
        Cs[i][s] = ok ? Cm[src] : InT(0.0f);
    }
    for (int hh = sg; hh < HPG; hh += NT / 32) {
        const int h = g * HPG + hh;
        const float da = (lane < tt) ? dtA[((size_t)b * T + t0 + lane) * H + h] : 0.0f;
        const float v = simd_prefix_inclusive_sum(da);
        lcg[hh][lane] = v;
        LCG[(((size_t)b * H + h) * nC + c) * Q + lane] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    for (int t = sg; t < 16; t += NT / 32) {
        const int ti = t / 4, tj = t % 4;
        if (tj > ti) continue;
        simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (int k = 0; k < Ds / 8; ++k) {
            simdgroup_float8x8 a, bt;
            a.thread_elements()[0] = float(Cs[8 * ti + fm][8 * k + fn]);
            a.thread_elements()[1] = float(Cs[8 * ti + fm][8 * k + fn + 1]);
            bt.thread_elements()[0] = float(Bs[8 * tj + fn][8 * k + fm]);
            bt.thread_elements()[1] = float(Bs[8 * tj + fn + 1][8 * k + fm]);
            simdgroup_multiply_accumulate(acc, a, bt, acc);
        }
        simdgroup_store(acc, &CB[0][0], Q, ulong2(8 * tj, 8 * ti));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int f = tid; f < HPG * Q * Q; f += NT) {
        const int hh = f / (Q * Q), r = f % (Q * Q);
        const int i = r / Q, j = r % Q;
        float v = 0.0f;
        if (j <= i) v = CB[i][j] * exp(lcg[hh][i] - lcg[hh][j]);
        M[((((size_t)b * H + g * HPG + hh) * nC + c) * Q + i) * Q + j] = v;
    }
"""

_SCAN_SRC = """
    constexpr int Q = 32;
    constexpr int HPG = H / G;
    constexpr int NS = Ds / 8;
    constexpr int NT = Dh * 4 / DSPLIT;     // Dh/8/DSPLIT simdgroups
    const int h = threadgroup_position_in_grid.y;
    const int b = threadgroup_position_in_grid.z;
    const int g = h / HPG;
    const ushort sg = simdgroup_index_in_threadgroup;
    const ushort lane = thread_index_in_simdgroup;
    const int tid = thread_position_in_threadgroup.x;
    const int nC = (T + Q - 1) / Q;
    const int d0 = (int(threadgroup_position_in_grid.x) * (NT / 32) + sg) * 8;
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;

#if !SSD_DIRECT_BC
    threadgroup InT Bs[Q][Ds];
    threadgroup InT Cs[Q][Ds];
#endif
    threadgroup float Xs[Q][Dh];
    threadgroup float Ms[Q][Q];
    threadgroup float El[Q];      // exp(lcg_i)
    threadgroup float Wl[Q];      // exp(lcg_last - lcg_j)
    threadgroup float Gl[1];      // exp(lcg_last)

    const size_t sbase = ((size_t)b * H + h) * Dh * Ds;
    simdgroup_float8x8 S[NS];
    for (int st = 0; st < NS; ++st)
        simdgroup_load(S[st], state_in + sbase, Ds, ulong2(8 * st, d0));

    for (int c = 0; c < nC; ++c) {
        const int t0 = c * Q;
        const int tt = min(Q, T - t0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
#if !SSD_DIRECT_BC
        for (int f = tid; f < Q * Ds / 4; f += NT) {
            const int i = f / (Ds / 4), s = (f % (Ds / 4)) * 4;
            const bool ok = i < tt;
            const size_t src = (((size_t)b * T + t0 + i) * G + g) * Ds + s;
            *(threadgroup uint2*)&Bs[i][s] = ok ? *(const device uint2*)(Bm + src) : uint2(0);
            *(threadgroup uint2*)&Cs[i][s] = ok ? *(const device uint2*)(Cm + src) : uint2(0);
        }
#endif
        for (int f = tid; f < Q * Dh / 4; f += NT) {
            const int i = f / (Dh / 4), d = (f % (Dh / 4)) * 4;
            const device InT* xp = dtx + (((size_t)b * T + t0 + i) * H + h) * Dh + d;
            *(threadgroup float4*)&Xs[i][d] = (i < tt)
                ? float4(float(xp[0]), float(xp[1]), float(xp[2]), float(xp[3]))
                : float4(0.0f);
        }
        const size_t mbase = (((size_t)b * H + h) * nC + c) * Q * Q;
        for (int f = tid; f < Q * Q / 4; f += NT)
            ((threadgroup float4*)&Ms[0][0])[f] = ((const device float4*)(M + mbase))[f];
        if (tid < Q) {
            const size_t lb = (((size_t)b * H + h) * nC + c) * Q;
            const float li = LCG[lb + tid];
            const float ll = LCG[lb + Q - 1];
            El[tid] = exp(li);
            Wl[tid] = exp(ll - li);
            if (tid == 0) Gl[0] = exp(ll);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

#if SSD_DIRECT_BC
#define BCSRC(P, i, s) ((i) < tt ? float(P[(((size_t)b * T + t0 + (i)) * G + g) * Ds + (s)]) : 0.0f)
#define CROW(i, s) BCSRC(Cm, i, s)
#define BROW(i, s) BCSRC(Bm, i, s)
#else
#define CROW(i, s) float(Cs[i][s])
#define BROW(i, s) float(Bs[i][s])
#endif
        simdgroup_float8x8 xt[4];
        for (int jt = 0; jt < 4; ++jt)
            simdgroup_load(xt[jt], &Xs[0][0], Dh, ulong2(d0, 8 * jt), true);

        for (int it = 0; it < 4; ++it) {
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int st = 0; st < NS; ++st) {
                simdgroup_float8x8 ct;
                ct.thread_elements()[0] = CROW(8 * it + fn, 8 * st + fm);
                ct.thread_elements()[1] = CROW(8 * it + fn + 1, 8 * st + fm);
                simdgroup_multiply_accumulate(acc, S[st], ct, acc);
            }
            acc.thread_elements()[0] *= El[8 * it + fn];
            acc.thread_elements()[1] *= El[8 * it + fn + 1];
            for (int jt = 0; jt <= it; ++jt) {
                simdgroup_float8x8 mt;
                simdgroup_load(mt, &Ms[0][0], Q, ulong2(8 * jt, 8 * it), true);
                simdgroup_multiply_accumulate(acc, xt[jt], mt, acc);
            }
            for (short e = 0; e < 2; ++e) {
                const int i = 8 * it + fn + e;
                if (i < tt) {
                    y[(((size_t)b * T + t0 + i) * H + h) * Dh + d0 + fm] =
                        InT(acc.thread_elements()[e]);
                }
            }
        }

        const float gl = Gl[0];
        for (int st = 0; st < NS; ++st) {
            S[st].thread_elements()[0] *= gl;
            S[st].thread_elements()[1] *= gl;
            for (int jt = 0; jt < 4; ++jt) {
                simdgroup_float8x8 bw;
                const float w = Wl[8 * jt + fm];
                bw.thread_elements()[0] = BROW(8 * jt + fm, 8 * st + fn) * w;
                bw.thread_elements()[1] = BROW(8 * jt + fm, 8 * st + fn + 1) * w;
                simdgroup_multiply_accumulate(S[st], xt[jt], bw, S[st]);
            }
        }
    }
    for (int st = 0; st < NS; ++st)
        simdgroup_store(S[st], state_out + sbase, Ds, ulong2(8 * st, d0));
"""

_HEADER = """
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;
"""

_kernels: dict = {}
_LOCK = threading.Lock()  # kernel cache and counters


def _get_kernels(direct_bc: bool = DIRECT_BC, dsplit: int = DSPLIT):
    key = (direct_bc, dsplit)
    with _LOCK:
        if key not in _kernels:
            m = mx.fast.metal_kernel(
                name="nemotron_ssd_mma_m",
                input_names=["Bm", "Cm", "dtA", "T"],
                output_names=["M", "LCG"],
                header=_HEADER,
                source=_M_SRC,
            )
            scan = mx.fast.metal_kernel(
                name=f"nemotron_ssd_mma_scan_d{int(direct_bc)}_s{dsplit}",
                input_names=["dtx", "Bm", "Cm", "M", "LCG", "state_in", "T"],
                output_names=["y", "state_out"],
                header=_HEADER
                + f"#define SSD_DIRECT_BC {int(direct_bc)}\n#define DSPLIT {dsplit}\n",
                source=_SCAN_SRC,
            )
            _kernels[key] = (m, scan)
        return _kernels[key]


def enabled(environ=None) -> bool:
    env = os.environ if environ is None else environ
    return str(env.get(SSD_MMA_ENV, "")).strip().lower() not in _OFF


def refusal(dtx, Bm, Cm=None) -> str | None:
    """Why the MMA scan cannot take this segment, or ``None``."""
    if dtx.ndim != 4 or Bm.ndim != 4:
        return "rank"
    if Cm is not None and (Cm.dtype != dtx.dtype or Cm.shape != Bm.shape):
        return "C dtype/shape differs from B"
    _, T, h, dh = dtx.shape
    g, ds = Bm.shape[2:]
    if dtx.dtype not in (mx.bfloat16, mx.float16) or Bm.dtype != dtx.dtype:
        return "requires matching 16-bit dtx/B/C"
    if dh % 8 or not 8 <= dh <= 128:
        return f"head_dim {dh}"
    if ds % 8 or not 8 <= ds <= 256:
        return f"state_dim {ds}"
    if g <= 0 or h % g:
        return f"heads {h} / groups {g}"
    scan_bytes = 2 * CHUNK * ds * 2 + CHUNK * dh * 4 + CHUNK * CHUNK * 4 + 2 * CHUNK * 4 + 4
    m_bytes = 2 * CHUNK * ds * 2 + CHUNK * CHUNK * 4 + (h // g) * CHUNK * 4
    if max(scan_bytes, m_bytes) > _THREADGROUP_BYTES:
        return "threadgroup memory"
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return "requires Metal"
    return None


def ssd_seg_mma(
    dtx, Bm, Cm, dtA, state, *, direct_bc: bool = DIRECT_BC, dsplit: int = DSPLIT
):
    """Drop-in for ``ssm._ssd_seg``: ``(y_without_D, fp32 state)``."""
    b, T, h, dh = dtx.shape
    g, ds = Bm.shape[2:]
    n_chunks = (T + CHUNK - 1) // CHUNK
    if (dh // 8) % dsplit:
        dsplit = 1
    kern_m, kern_scan = _get_kernels(direct_bc, dsplit)
    tmpl = [("InT", dtx.dtype), ("H", h), ("G", g), ("Dh", dh), ("Ds", ds)]
    M, LCG = kern_m(
        inputs=[Bm, Cm, dtA, T],
        template=tmpl,
        grid=(256 * n_chunks, g, b),
        threadgroup=(256, 1, 1),
        output_shapes=[(b, h, n_chunks, CHUNK, CHUNK), (b, h, n_chunks, CHUNK)],
        output_dtypes=[mx.float32, mx.float32],
    )
    threads = dh * 4 // dsplit
    y, state_out = kern_scan(
        inputs=[dtx, Bm, Cm, M, LCG, state, T],
        template=tmpl,
        grid=(threads * dsplit, h, b),
        threadgroup=(threads, 1, 1),
        output_shapes=[(b, T, h, dh), state.shape],
        output_dtypes=[dtx.dtype, mx.float32],
    )
    return y, state_out


def select_seg(dtx, Bm, Cm=None):
    """The segment function ``ssd_prefill`` runs, counting the route taken."""
    if not enabled():
        return None
    reason = refusal(dtx, Bm, Cm)
    if reason is not None:
        with _LOCK:
            _counts["refused"] += 1
            _refusals[reason] = _refusals.get(reason, 0) + 1
        return None
    with _LOCK:
        _counts["mma"] += 1
    return ssd_seg_mma


def status() -> dict:
    return {
        "switch": SSD_MMA_ENV,
        "enabled": enabled(),
        "counts": dict(_counts),
        "refusal_reasons": dict(_refusals),
    }

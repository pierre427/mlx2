# SPDX-License-Identifier: Apache-2.0
"""Hy4-flash sparse prefill attention: one Metal kernel over the recalled keys.

The indexed layers' prefill (chunks of more than ``GATHER_MAX_QUERY`` rows)
otherwise attends every query to all ``T`` cached latent rows through a
``[B, L, T]`` hit mask (``hy_v4_flash.Attention._attend``), which is
``O(L * T)`` per chunk.  This kernel reads only each query's recalled rows
(``CompressedIndexer.select``, ``K = group_topk * ratio`` = 2048 slots, -1
unused), so a chunk costs ``O(L * K)`` whatever the context length.

Math (absorbed MLA, one latent KV head shared by every query head, the same
contract as ``mla_attend``): for query row ``l`` (position ``start + l``)
and head ``h``,

    s_j = scale * q[h, l, :] . kv[id_j, :]        (D = r + p lanes)
    out[h, l, :] = sum_j e^{s_j - m} kv[id_j, :r] / (e^{sink_h - m} + sum_j e^{s_j - m})

over the valid slots ``j`` (``0 <= id_j - first < T`` and ``id_j <= start +
l``: the causal mask the dense path applies).

Contract on ``recalled``: the valid ids of one row are distinct.  The kernel
sums one softmax term per slot, so a repeated id would count twice, while
the dense reference (a boolean hit mask) counts it once.
``CompressedIndexer.select`` cannot repeat one: it lists distinct groups
(``argpartition`` picks plus the query's own group, which ``valid`` excludes
from the picks) and expands each to its ``ratio`` distinct tokens.  Callers
with another id source must deduplicate first.

Rows: id ``i`` is read at ``kv[i - first]``.  The model's dense and gather
references index ``kv`` with the id itself, which agrees only when
``first == 0``, always the case on indexed layers (only sliding caches
advance ``start``); ``Attention._attend`` therefore takes the kernel only
when ``first == 0`` and counts any other forward as a ``kv_first_nonzero``
fallback.  The sink is the extra
zero-valued softmax column ``mx.fast.scaled_dot_product_attention`` applies;
the online softmax starts from it (``m = sink_h``, ``l = 1``, ``O = 0``), so
it is never NaN and a row whose every slot is empty returns 0 like the
reference.  Returns ``[B, H, L, r]`` in the query dtype.

Kernel layout: one threadgroup of 16 simdgroups (512 threads) per query row
``(b, l)``; the 32 heads are the MMA ``M`` dimension, so every gathered key
row is read from memory once per query and serves all heads.  Per tile of
16 recalled slots:

1. the slots' ids (validity, causality) and their 16 latent rows are staged
   in threadgroup memory straight from the cache (16-byte vector copies, zero
   rows for empty slots; no ``[L, K, D]`` gather is ever materialized);
2. scores: simdgroup ``(head block hb, dim quarter dq)`` keeps its query
   fragments ``q[8 heads, D / 4 dims]`` in registers for the whole row and
   writes a partial ``S[8 heads, 16 keys]`` (``simdgroup_float8x8``, fp32);
3. softmax: 16 lanes per head sum the partials in ascending quarter order,
   apply ``scale`` and the mask, and update the running max / sum with xor
   shuffles; ``P`` and the rescale ``alpha`` go to threadgroup memory (fp32);
4. values: simdgroup ``s`` owns output lanes ``[32 s, 32 s + 32)`` of all
   heads, rescales its fp32 ``O`` fragments by ``alpha`` and accumulates
   ``P[heads, 16] . V[16, 32]`` (V = the staged rows' first ``r`` lanes).

Inputs are bf16 or fp16; every product, the softmax and
``O`` are fp32; only the output is rounded to the query dtype.  The fragment
coordinate law (which ``(row, col)`` each lane holds in an 8x8 simdgroup
matrix) is the one MLX's steel ``BaseMMAFrag::get_coord`` uses (MIT; see
``provenance/hy-v4-flash-sparse-prefill.json``); the kernel itself is
written for this layer.  Same reduction contents as the dense masked path,
other order: numerically equivalent, not bit-identical.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

KEY_TILE = 16
THREADS = 512
SIMDGROUPS = THREADS // 32
# 16-bit operands only: a 16-row fp32 tile of 576 lanes would not fit 32 KiB.
SUPPORTED_DTYPES = (mx.bfloat16, mx.float16)
KERNEL_NAME = "mlx2_hy4_sparse_prefill_attention_v1"

_SOURCE = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int l = int(threadgroup_position_in_grid.y);
    const int b = int(threadgroup_position_in_grid.z);
    const int L = params[0];
    const int T = params[1];
    const int K = params[2];
    const int start = params[3];
    const int first = params[4];
    const float scale = scale_arr[0];
    const int qpos = start + l;

    // 8x8 fragment coordinates of this lane: (fm, fn) and (fm, fn + 1).
    const short qid = short(lane / 4);
    const short fm = (qid & 4) + short((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + short(lane % 2) * 2;

    constexpr int HB = H / 8;          // head blocks
    constexpr int DQ = 16 / HB;        // dim splits of the score MMA
    constexpr int DSTEPS = D / 8 / DQ; // 8-dim steps per split
    constexpr int OCB = R / 128;       // 8-lane output blocks per simdgroup
    constexpr int KT = 16;
    constexpr int ROW4 = D / 8;        // uint4 (8 x 16-bit) per latent row
    // Staged rows are padded by 16 bytes: a 1152-byte stride puts the 8 rows
    // of a fragment on one bank; 1168 spreads them over the banks.
    constexpr int DP = D + 8;
    constexpr int ROWP4 = DP / 8;
    constexpr int CH = (KT * ROW4 + 511) / 512;  // 16-byte chunks per thread per tile

    threadgroup uint4 kraw[KT * ROWP4];
    threadgroup const InT* ktile = (threadgroup const InT*)kraw;
    threadgroup float spart[DQ * H * KT];
    threadgroup float pbuf[H * KT];
    threadgroup float abuf[H];
    threadgroup float lbuf[H];
    threadgroup int kidx[KT];

    const int hb = int(sg) % HB;
    const int dq = int(sg) / HB;
    simdgroup_float8x8 qf[DSTEPS];
    {
        const device InT* qrow = q + (((size_t)b * H + hb * 8 + fm) * L + l) * D
            + dq * DSTEPS * 8 + fn;
        #pragma clang loop unroll(full)
        for (int s = 0; s < DSTEPS; ++s) {
            qf[s].thread_elements()[0] = float(qrow[s * 8]);
            qf[s].thread_elements()[1] = float(qrow[s * 8 + 1]);
        }
    }

    // Softmax lanes: 16 per head; every lane of a head keeps the same running state.
    const int sh = int(tid) / KT;
    const int sk = int(tid) % KT;
    float m_run = sh < H ? sinks[sh] : 0.0f;
    float l_run = 1.0f;

    simdgroup_float8x8 of[HB][OCB];
    #pragma clang loop unroll(full)
    for (int i = 0; i < HB; ++i)
        #pragma clang loop unroll(full)
        for (int j = 0; j < OCB; ++j)
            of[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const device int* ids = recalled + ((size_t)b * L + l) * K;
    const device InT* kvb = kv + (size_t)b * T * D;

    // The next tile's rows are loaded into registers while this tile computes.
    uint4 pre[CH];
    int nrow = -1;
#define MLX2_HY4_ROW_OF(k) ({ \
        int row_ = -1; \
        if ((k) < K) { \
            const int id_ = ids[(k)]; \
            const int r_ = id_ - first; \
            if (id_ >= 0 && id_ <= qpos && r_ >= 0 && r_ < T) row_ = r_; \
        } \
        row_; })
#define MLX2_HY4_FETCH(k0) { \
        for (int j = 0; j < CH; ++j) { \
            const int c = int(tid) + j * 512; \
            pre[j] = uint4(0); \
            if (c < KT * ROW4) { \
                const int row = MLX2_HY4_ROW_OF((k0) + c / ROW4); \
                if (row >= 0) pre[j] = ((const device uint4*)(kvb + (size_t)row * D))[c % ROW4]; \
            } \
        } \
        if (tid < KT) nrow = MLX2_HY4_ROW_OF((k0) + int(tid)); }

    MLX2_HY4_FETCH(0)
    for (int k0 = 0; k0 < K; k0 += KT) {
        #pragma clang loop unroll(full)
        for (int j = 0; j < CH; ++j) {
            const int c = int(tid) + j * 512;
            if (c < KT * ROW4) kraw[(c / ROW4) * ROWP4 + c % ROW4] = pre[j];
        }
        if (tid < KT) kidx[tid] = nrow;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k0 + KT < K) MLX2_HY4_FETCH(k0 + KT)

        // Scores: partial S[8 heads of hb, 16 keys] over this simdgroup's dims.
        #pragma clang loop unroll(full)
        for (int kb = 0; kb < KT / 8; ++kb) {
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            threadgroup const InT* kt = ktile + (kb * 8 + fn) * DP + dq * DSTEPS * 8 + fm;
            #pragma clang loop unroll(full)
            for (int s = 0; s < DSTEPS; ++s) {
                simdgroup_float8x8 kf;
                kf.thread_elements()[0] = float(kt[s * 8]);
                kf.thread_elements()[1] = float(kt[s * 8 + DP]);
                simdgroup_multiply_accumulate(acc, qf[s], kf, acc);
            }
            threadgroup float* dst = spart + (dq * H + hb * 8 + fm) * KT + kb * 8 + fn;
            dst[0] = acc.thread_elements()[0];
            dst[1] = acc.thread_elements()[1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sh < H) {
            float sc = 0.0f;
            #pragma clang loop unroll(full)
            for (int d = 0; d < DQ; ++d) sc += spart[(d * H + sh) * KT + sk];
            sc = kidx[sk] >= 0 ? sc * scale : -INFINITY;
            float mx = sc;
            for (ushort off = KT / 2; off > 0; off /= 2) mx = metal::max(mx, simd_shuffle_xor(mx, off));
            const float mn = metal::max(m_run, mx);
            const float a = fast::exp(m_run - mn);
            const float p = kidx[sk] >= 0 ? fast::exp(sc - mn) : 0.0f;
            float ps = p;
            for (ushort off = KT / 2; off > 0; off /= 2) ps += simd_shuffle_xor(ps, off);
            l_run = l_run * a + ps;
            m_run = mn;
            pbuf[sh * KT + sk] = p;
            if (sk == 0) abuf[sh] = a;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Values: O[heads, this simdgroup's 32 lanes] = alpha * O + P . V.
        #pragma clang loop unroll(full)
        for (int i = 0; i < HB; ++i) {
            const float a = abuf[i * 8 + fm];
            #pragma clang loop unroll(full)
            for (int j = 0; j < OCB; ++j) {
                of[i][j].thread_elements()[0] *= a;
                of[i][j].thread_elements()[1] *= a;
            }
        }
        #pragma clang loop unroll(full)
        for (int kk = 0; kk < KT / 8; ++kk) {
            simdgroup_float8x8 vf[OCB];
            threadgroup const InT* vt = ktile + (kk * 8 + fm) * DP + int(sg) * OCB * 8 + fn;
            #pragma clang loop unroll(full)
            for (int j = 0; j < OCB; ++j) {
                vf[j].thread_elements()[0] = float(vt[j * 8]);
                vf[j].thread_elements()[1] = float(vt[j * 8 + 1]);
            }
            #pragma clang loop unroll(full)
            for (int i = 0; i < HB; ++i) {
                simdgroup_float8x8 pf;
                threadgroup const float* pr = pbuf + (i * 8 + fm) * KT + kk * 8 + fn;
                pf.thread_elements()[0] = pr[0];
                pf.thread_elements()[1] = pr[1];
                #pragma clang loop unroll(full)
                for (int j = 0; j < OCB; ++j)
                    simdgroup_multiply_accumulate(of[i][j], pf, vf[j], of[i][j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (sh < H && sk == 0) lbuf[sh] = l_run;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (int i = 0; i < HB; ++i) {
        const int h = i * 8 + fm;
        const float inv = 1.0f / lbuf[h];
        device OutT* o = out + (((size_t)b * H + h) * L + l) * R + int(sg) * OCB * 8 + fn;
        #pragma clang loop unroll(full)
        for (int j = 0; j < OCB; ++j) {
            o[j * 8] = OutT(of[i][j].thread_elements()[0] * inv);
            o[j * 8 + 1] = OutT(of[i][j].thread_elements()[1] * inv);
        }
    }
#undef MLX2_HY4_FETCH
#undef MLX2_HY4_ROW_OF
"""


# Apple GPUs give a threadgroup at most 32 KiB.
THREADGROUP_MEMORY_LIMIT = 32 * 1024
# Conservative per-dimension threadgroup count of a dispatch (rows on y,
# lanes on z); prefill chunks are at most a few thousand rows.
MAX_GRID_ROWS = 65535


def threadgroup_bytes(num_heads: int, head_dim: int) -> int:
    """Threadgroup memory of one instantiation (the kernel's static arrays)."""
    dq = 16 // max(1, num_heads // 8)
    staged = KEY_TILE * (head_dim + 8) * 2           # kraw: 16 padded 16-bit rows
    scores = dq * num_heads * KEY_TILE * 4           # spart
    probs = num_heads * KEY_TILE * 4                 # pbuf
    return staged + scores + probs + 2 * num_heads * 4 + KEY_TILE * 4  # abuf, lbuf, kidx


@lru_cache(maxsize=None)
def _kernel():
    return mx.fast.metal_kernel(
        name=KERNEL_NAME,
        input_names=["q", "kv", "recalled", "sinks", "scale_arr", "params"],
        output_names=["out"],
        source=_SOURCE,
        ensure_row_contiguous=True,
    )


def unsupported_reason(num_heads: int, head_dim: int, rank: int, dtype, *, rows: int = 1,
                       lanes: int = 1, first: int = 0) -> str | None:
    """Why the kernel cannot run this forward (None: it can).

    Every reason is a counted fallback to the dense path in
    ``Attention._attend``, never an error: geometry, threadgroup memory,
    grid size, the row convention (``first``) and a one-time dispatch probe
    of this instantiation (``dispatch_probe``).
    """
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        return "metal_unavailable"
    if dtype not in SUPPORTED_DTYPES:
        return f"dtype_{str(dtype).rsplit('.', 1)[-1]}"
    if num_heads not in (16, 32):
        return f"heads_{num_heads}"  # head blocks x dim splits must fill 16 simdgroups
    hb = num_heads // 8
    if head_dim % 8 or (head_dim // 8) % (16 // hb):
        return f"head_dim_{head_dim}"
    if rank % 128 or rank > head_dim:
        return f"rank_{rank}"
    if threadgroup_bytes(num_heads, head_dim) > THREADGROUP_MEMORY_LIMIT:
        return f"threadgroup_memory_{threadgroup_bytes(num_heads, head_dim)}"
    if rows > MAX_GRID_ROWS or lanes > MAX_GRID_ROWS:
        return "grid_rows"
    if first != 0:
        return "kv_first_nonzero"
    return dispatch_probe(num_heads, head_dim, rank, dtype)


_PROBES: dict = {}


def dispatch_probe(num_heads: int, head_dim: int, rank: int, dtype) -> str | None:
    """Build and run this instantiation once on a one-row input (a few KB).

    Catches what only shows at pipeline build or dispatch (a compile error,
    a device whose pipeline allows fewer than 512 threads per threadgroup) so
    the forward can fall back to the dense path with a counted reason instead
    of failing the request.  The result is cached per instantiation.  Only
    this probe is guarded: an error while evaluating a real forward is not
    swallowed.  Inside a graph transformation, where MLX refuses ``eval``,
    the probe is deferred (that forward falls back, uncached).
    """
    key = (num_heads, head_dim, rank, str(dtype))
    if key in _PROBES:
        return _PROBES[key]
    try:
        q = mx.zeros((1, num_heads, 1, head_dim), dtype=dtype)
        kv = mx.zeros((1, KEY_TILE, head_dim), dtype=dtype)
        ids = mx.arange(KEY_TILE, dtype=mx.int32)[None, None]
        out = _launch(q, kv, ids, mx.zeros((num_heads,)), 1.0, rank, KEY_TILE - 1, 0)
        mx.eval(out)
        reason = None if bool(mx.all(out == 0).item()) else "dispatch_probe_wrong_result"
    except ValueError as err:  # MLX refuses eval inside vjp/grad: retry later
        if "transform" in str(err).lower():
            return "dispatch_probe_deferred"
        reason = f"dispatch_probe_{type(err).__name__}"
    except Exception as err:  # noqa: BLE001 - pipeline build / dispatch failures
        reason = f"dispatch_probe_{type(err).__name__}"
    _PROBES[key] = reason
    return reason


def _launch(q, kv, recalled, sinks, scale, rank, start, first):
    B, H, L, D = q.shape
    T = kv.shape[1]
    K = recalled.shape[-1]
    params = mx.array([L, T, K, int(start), int(first)], dtype=mx.int32)
    return _kernel()(
        inputs=[q, kv.astype(q.dtype), recalled.astype(mx.int32),
                sinks.astype(q.dtype).astype(mx.float32), mx.array([scale], dtype=mx.float32), params],
        template=[("InT", q.dtype), ("OutT", q.dtype), ("H", H), ("D", D), ("R", rank)],
        grid=(THREADS, L, B),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(B, H, L, rank)],
        output_dtypes=[q.dtype],
    )[0]


def sparse_prefill_attend(q, kv, recalled, sinks, scale: float, rank: int, *, start: int, first: int = 0):
    """``q [B, H, L, D]``, ``kv [B, T, D]`` latent rows, ``recalled [B, L, K]``
    int32 ids (-1 empty) -> ``[B, H, L, rank]`` in ``q.dtype``.

    ``start`` is the position of query row 0 and ``first`` that of
    ``kv[:, 0]``; an id is read at row ``id - first``.  ``sinks [H]`` are
    rounded to ``q.dtype`` first, as ``mla_attend`` does.  The valid ids of a
    row must be distinct (module docstring).  The model checks
    ``unsupported_reason`` first; this direct entry refuses an unsupported
    geometry (``first`` is allowed here: tests and other callers).
    """
    B, H, L, D = q.shape
    T = kv.shape[1]
    K = recalled.shape[-1]
    reason = unsupported_reason(H, D, rank, q.dtype, rows=L, lanes=B)
    if reason is not None:
        raise ValueError(f"Hy4-flash sparse prefill kernel cannot run this geometry: {reason}")
    if kv.shape != (B, T, D) or recalled.shape != (B, L, K):
        raise ValueError("Hy4-flash sparse prefill: q / kv / recalled shapes disagree")
    return _launch(q, kv, recalled, sinks, scale, rank, start, first)

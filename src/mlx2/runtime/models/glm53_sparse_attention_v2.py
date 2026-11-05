# SPDX-License-Identifier: MIT
"""GLM-5.3 fused gathered sparse attention v2 (design glm53-speed-round2.md 1.1, 2.1).

Absorbed MLA with one 512-wide latent head: the gathered rows of a query are
both its keys and its values and are shared by all 64 query heads, so each
gathered tile is staged once in threadgroup memory and serves every head
through ``simdgroup_float8x8`` MMAs (heads as ``M``).

Kernel ``mlx2_glm53_gathered_attention_v2``: one threadgroup per
``(key split s, query row z)``; ``SPLIT`` gathered ids per split.

* pass 1 (scores): tiles of ``8 * DSPLIT`` keys x ``512 / DSPLIT`` dims are
  staged ``DSPLIT`` times per key tile (16 KiB); simdgroup ``(g, kb)`` keeps
  the fragment ``S[8 heads of group g, 8 keys of block kb]`` in registers
  across the dim stages, so every score is one MMA chain over dims
  ``0..511`` in ascending 8-blocks whatever ``DSPLIT`` is. Scores land in the
  threadgroup ``S[64, SPLIT]`` scratch (16 KiB at ``SPLIT = 64``).
* statistics: 8 lanes per head over the split's keys, invalid keys (``-1``
  ids or slots past ``W``) at ``-inf``; ``m`` = max of ``SCALE * S``,
  ``P = exp(SCALE * S - m)`` written in place, ``l`` by a fixed xor tree.
  A split without a valid key writes ``m = -inf, l = 0`` and ``P = 0``; a
  NaN score makes ``m`` NaN (it is never dropped by ``max``).
* pass 2 (values): tiles of 8 keys x 512 dims are re-staged (rows from L2);
  simdgroup ``(g, slice)`` accumulates ``O[8 heads, 512 / DSPLIT dims] +=
  P[8, 8] . V[8, dims]``.

Every tile slot is written on every fill: rows of ``-1`` ids and slots past
the split's key count are zero, so no MMA reads stale threadgroup memory and
``P = 0`` meets ``V = 0`` (never NaN). Ids above the row's last committed
latent row are clipped to it, as ``gather_latent`` does.

Combine ``mlx2_glm53_gathered_attention_v2_combine``: per ``(head, row)``
the weights ``a_s = exp(m_s - m)`` (0 for empty splits) are computed once
per split, then ``out = sum_s a_s o_s / sum_s a_s l_s`` in ascending split
order; an empty selection gives ``0 / 0 = NaN`` (round-1 failure contract).

Rows live on grid z and share nothing, so any grouping of rows is bitwise
identical to one launch per row. The law is new (fp32 MMA order), so it is
accepted cross-policy under the near-tie contract, never as bitwise to
``gathered_v1``.

Sources ABI: ``base (T0, 512)`` committed rows shared by every row of a
launch, one concatenated ``tails (sum d_r, 512)`` buffer and ``row_meta
(L, 4) int32 = (T0_r, d_r, tail_offset_r, pos_r)``; a row's id ``< T0_r``
reads ``base``, otherwise ``tails[tail_offset_r + id - T0_r]``.

fp16-operand law ``gathered_f16_from_<n>`` (2026-11-02): the flash-style
kernel ``mlx2_glm53_gathered_attention_flash`` with ``T = half`` (``half=True``,
``kind="flash_f16"``). One threadgroup per (outer split, row) walks INNER
inner splits of SPLIT keys with an online softmax, so a launch writes
FLASH_OUTER partials per (row, head) instead of one per split: the per-split
kernel above spent ~40% of its time writing 4.3 MB of partials per row and
reading them back in the combine (kernel-diag-16k), and ~7% in its MMAs, so
operand precision alone bought nothing (kernel-bench-16k: an fp16 per-split
twin ran at the fp32 kernel's speed). ``q``, the staged tiles and ``P`` are
``T``; scores, statistics, the O fragments (rescaled by ``diag(a)`` through one MMA per
fragment when the running max moves; a NaN or +inf score is flagged per
head and never enters O) and the partials stay fp32.
Scores are never rounded: only ``P = exp(S * SCALE - m)`` is stored as ``T``
for the value MMA. The latent plane stays fp32 in the cache; the caller
hands the kernel a half copy of the rows a chunk can see. An fp16-operand
law (the class of the nax_fp16 prompt tiles), gated against float64 on the
half-rounded operands rather than at the fp32 floor. Rows still share
nothing. ``kind="flash_f32"`` is the same kernel on fp32 operands, for the
bench.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

HEADS, DIM, TOPK = 64, 512, 2048
SCALE = 256**-0.5
SPLITS_SUPPORTED = (32, 64)
DIM_SPLITS_SUPPORTED = (2, 4)
# The configuration the policy runs; the bench matrix decides it (R1).
DEFAULT_SPLIT, DEFAULT_DIM_SPLIT = 64, 4  # wp1-matrix-b0a7759-a1: fastest point in every cell
KINDS = ("partial", "flash_f16", "flash_f32")
# Outer partial slots of a flash launch (gathered_f16_from_<n>): the law's
# value; the bench matrix (kernel-bench-16k-outer*) decides it.
FLASH_OUTER = 4
_GATE: dict = {}
GATE_REPORT: dict = {}
_SCHEDULED = {"launches": 0, "rows": 0}

_HEADER = "constant constexpr float SCALE = 0.0625f;  // 256 ** -0.5, exact\n"

# Shared tile fill: KEYS keys x COLS dims starting at key k0 and dim c0 into
# tile (row stride COLS), as float4. Zero for -1 ids and slots past W.
_PARTIAL_SOURCE = r"""
const uint split = threadgroup_position_in_grid.x;
const uint row = threadgroup_position_in_grid.z;
const uint tid = thread_index_in_threadgroup;
const uint sg = simdgroup_index_in_threadgroup;
constexpr uint NT = 256 * DSPLIT;
constexpr uint KT1 = 8 * DSPLIT;          // pass-1 keys per tile
constexpr uint DW1 = 512 / DSPLIT;        // pass-1 dims per stage
constexpr uint DO = 512 / DSPLIT;         // pass-2 output dims per simdgroup
constexpr uint NF = DO / 8;
threadgroup float tile[4096];
threadgroup float S[64 * SPLIT];
const int T0 = row_meta[row * 4 + 0];
const int DR = row_meta[row * 4 + 1];
const int TOFF = row_meta[row * 4 + 2];
const int LAST = T0 + DR - 1;
const int first = int(split) * SPLIT;
const device int* rid = ids + size_t(row) * WIDTH;
const device float* qrow = q + size_t(row) * 64 * 512;
const uint g = sg / DSPLIT;
const uint sub = sg % DSPLIT;

// pass 1: scores
for (uint t = 0; t < SPLIT / KT1; t++) {
    const int k0 = first + int(t * KT1);
    simdgroup_float8x8 s = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint c = 0; c < DSPLIT; c++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < 1024; e += NT) {
            const uint kk = e / (DW1 / 4);
            const uint c4 = e % (DW1 / 4);
            const int k = k0 + int(kk);
            float4 v = float4(0.0f);
            if (k < WIDTH) {
                int id = rid[k];
                if (id >= 0) {
                    id = min(id, LAST);
                    const device float4* src = (const device float4*)(id < T0
                        ? base + size_t(id) * 512 : tails + size_t(TOFF + id - T0) * 512);
                    v = src[c * (DW1 / 4) + c4];
                }
            }
            ((threadgroup float4*)tile)[e] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const device float* qp = qrow + size_t(8 * g) * 512 + c * DW1;
        const threadgroup float* kp = tile + (8 * sub) * DW1;
        for (uint j = 0; j < DW1 / 8; j++) {
            simdgroup_float8x8 qf;
            simdgroup_float8x8 kf;
            simdgroup_load(qf, qp + 8 * j, 512);
            simdgroup_load(kf, kp + 8 * j, DW1, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(s, qf, kf, s);
        }
    }
    simdgroup_store(s, S + (8 * g) * SPLIT + t * KT1 + 8 * sub, SPLIT);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// statistics: 8 lanes per head, fixed xor tree
if (tid < 512) {
    const uint h = tid >> 3;
    const uint lane8 = tid & 7;
    threadgroup float* srow = S + h * SPLIT;
    float m = -INFINITY;
    uint bad = 0;
    for (uint i = 0; i < SPLIT / 8; i++) {
        const uint kk = lane8 + 8 * i;
        const int k = first + int(kk);
        const bool valid = k < WIDTH && rid[k] >= 0;
        const float x = valid ? srow[kk] * SCALE : -INFINITY;
        bad |= isnan(x) ? 1u : 0u;
        m = max(m, x);
    }
    for (ushort o = 1; o < 8; o <<= 1) {
        m = max(m, simd_shuffle_xor(m, o));
        bad |= simd_shuffle_xor(bad, o);
    }
    if (bad) m = NAN;
    const bool empty = m == -INFINITY;
    float l = 0.0f;
    for (uint i = 0; i < SPLIT / 8; i++) {
        const uint kk = lane8 + 8 * i;
        const int k = first + int(kk);
        const bool valid = k < WIDTH && rid[k] >= 0;
        const float p = (valid && !empty) ? metal::precise::exp(srow[kk] * SCALE - m) : 0.0f;
        srow[kk] = p;
        l += p;
    }
    for (ushort o = 1; o < 8; o <<= 1) l += simd_shuffle_xor(l, o);
    if (lane8 == 0) {
        const size_t slot = (size_t(row) * 64 + h) * SPLITS + split;
        partial_m[slot] = empty ? -INFINITY : m;
        partial_l[slot] = empty ? 0.0f : l;
    }
}

// pass 2: values
simdgroup_float8x8 o[NF];
for (uint j = 0; j < NF; j++) o[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
for (uint t = 0; t < SPLIT / 8; t++) {
    const int k0 = first + int(t * 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < 1024; e += NT) {
        const uint kk = e >> 7;
        const uint c4 = e & 127;
        const int k = k0 + int(kk);
        float4 v = float4(0.0f);
        if (k < WIDTH) {
            int id = rid[k];
            if (id >= 0) {
                id = min(id, LAST);
                const device float4* src = (const device float4*)(id < T0
                    ? base + size_t(id) * 512 : tails + size_t(TOFF + id - T0) * 512);
                v = src[c4];
            }
        }
        ((threadgroup float4*)tile)[e] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 pf;
    simdgroup_load(pf, S + (8 * g) * SPLIT + t * 8, SPLIT);
    const threadgroup float* vp = tile + sub * DO;
    for (uint j = 0; j < NF; j++) {
        simdgroup_float8x8 vf;
        simdgroup_load(vf, vp + 8 * j, 512);
        simdgroup_multiply_accumulate(o[j], pf, vf, o[j]);
    }
}
device float* op = partial_o + ((size_t(row) * 64 + 8 * g) * SPLITS + split) * 512 + sub * DO;
for (uint j = 0; j < NF; j++) simdgroup_store(o[j], op + 8 * j, SPLITS * 512);
"""

# Flash-style partial kernel (the fp16-operand law and its fp32 twin for the
# bench): one threadgroup per (outer split, row) walks INNER inner splits of
# SPLIT keys with an online softmax (fp32 running max m[h] and sum l[h] per
# head, the O fragments rescaled by diag(a) through one MMA per fragment)
# and writes ONE partial per (row, head, outer split). A NaN or +inf score
# never enters the statistics or O (its NaN would reach the other seven
# heads of its group through the rescale's 0 * NaN): it is dropped and the
# head flagged,
# and the flagged head's partial is written as (m = NaN, l = 1), which the
# combine turns into a NaN row for that head alone. Rescaling the fragments
# elementwise through thread_elements() instead was 2.9x slower
# (kernel-bench-16k-elementwise). With
# OUTER = 4 the partials are 8x smaller than the per-split kernel's (the
# diagnostic kernel-diag-16k: the partial_o write and the combine read were
# ~40% of that kernel's time, the MMAs ~7%). Operands are T (half for the
# law), staged tiles T, P stored as T over the fp32 score scratch once every
# statistics lane holds its P values in registers; scores, statistics,
# accumulators and partials stay fp32. Rows share nothing.
_PARTIAL_SOURCE_FLASH = r"""
const uint outer = threadgroup_position_in_grid.x;
const uint row = threadgroup_position_in_grid.z;
const uint tid = thread_index_in_threadgroup;
const uint sg = simdgroup_index_in_threadgroup;
constexpr uint NT = 256 * DSPLIT;
constexpr uint KT1 = 8 * DSPLIT;          // pass-1 keys per tile
constexpr uint DW1 = 512 / DSPLIT;        // pass-1 dims per stage
constexpr uint DO = 512 / DSPLIT;         // pass-2 output dims per simdgroup
constexpr uint NF = DO / 8;
threadgroup T tile[4096];
threadgroup float S[64 * SPLIT];
threadgroup T* P = (threadgroup T*)S;
threadgroup float D[8 * 64];              // per head group: diag(a) as an 8x8 fragment
threadgroup uint BAD[64];                 // per head: a NaN score was seen
threadgroup float M[64];
threadgroup float L[64];
const int T0 = row_meta[row * 4 + 0];
const int DR = row_meta[row * 4 + 1];
const int TOFF = row_meta[row * 4 + 2];
const int LAST = T0 + DR - 1;
const device int* rid = ids + size_t(row) * WIDTH;
const device T* qrow = q + size_t(row) * 64 * 512;
const uint g = sg / DSPLIT;
const uint sub = sg % DSPLIT;
const bool stats = tid < 512;
const uint h = tid >> 3;
const uint lane8 = tid & 7;
if (tid < 512) D[tid] = 0.0f;
if (tid < 64) { M[tid] = -INFINITY; L[tid] = 0.0f; BAD[tid] = 0u; }
simdgroup_float8x8 o[NF];
for (uint j = 0; j < NF; j++) o[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
float pv[SPLIT / 8];

for (uint is = 0; is < INNER; is++) {
    const int first = int((outer * INNER + is) * SPLIT);
    if (first >= WIDTH) break;

    // pass 1: scores (fp32 accumulation)
    for (uint t = 0; t < SPLIT / KT1; t++) {
        const int k0 = first + int(t * KT1);
        simdgroup_float8x8 s = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint c = 0; c < DSPLIT; c++) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = tid; e < 1024; e += NT) {
                const uint kk = e / (DW1 / 4);
                const uint c4 = e % (DW1 / 4);
                const int k = k0 + int(kk);
                vec<T, 4> v = vec<T, 4>(0);
                if (k < WIDTH) {
                    int id = rid[k];
                    if (id >= 0) {
                        id = min(id, LAST);
                        const device vec<T, 4>* src = (const device vec<T, 4>*)(id < T0
                            ? base + size_t(id) * 512 : tails + size_t(TOFF + id - T0) * 512);
                        v = src[c * (DW1 / 4) + c4];
                    }
                }
                ((threadgroup vec<T, 4>*)tile)[e] = v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const device T* qp = qrow + size_t(8 * g) * 512 + c * DW1;
            const threadgroup T* kp = tile + (8 * sub) * DW1;
            for (uint j = 0; j < DW1 / 8; j++) {
                simdgroup_matrix<T, 8, 8> qf;
                simdgroup_matrix<T, 8, 8> kf;
                simdgroup_load(qf, qp + 8 * j, 512);
                simdgroup_load(kf, kp + 8 * j, DW1, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s, qf, kf, s);
            }
        }
        simdgroup_store(s, S + (8 * g) * SPLIT + t * KT1 + 8 * sub, SPLIT);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // statistics: online softmax update per head; P kept in registers until
    // every lane has read its fp32 scores, then stored as T over the scratch.
    if (stats) {
        const threadgroup float* srow = S + h * SPLIT;
        float ms = -INFINITY;
        uint bad = 0;
        for (uint i = 0; i < SPLIT / 8; i++) {
            const uint kk = lane8 + 8 * i;
            const int k = first + int(kk);
            const bool valid = k < WIDTH && rid[k] >= 0;
            const float x = valid ? srow[kk] * SCALE : -INFINITY;
            // A NaN or +inf score (exp(inf - inf) is NaN too) is dropped here
            // and flagged; the head's row is written as NaN at the end
            // (never through O: see above).
            const bool poison = isnan(x) || x == INFINITY;
            bad |= poison ? 1u : 0u;
            ms = max(ms, poison ? -INFINITY : x);
        }
        for (ushort oo = 1; oo < 8; oo <<= 1) {
            ms = max(ms, simd_shuffle_xor(ms, oo));
            bad |= simd_shuffle_xor(bad, oo);
        }
        const float mo = M[h];
        const float mn = max(mo, ms);
        const bool empty = mn == -INFINITY;
        // Nothing accumulated yet (mo = -inf): a = 0 scales the zero O and L.
        const float a = mo == -INFINITY ? 0.0f : metal::precise::exp(mo - mn);
        float l = 0.0f;
        for (uint i = 0; i < SPLIT / 8; i++) {
            const uint kk = lane8 + 8 * i;
            const int k = first + int(kk);
            const bool valid = k < WIDTH && rid[k] >= 0;
            const float x = srow[kk] * SCALE;
            const bool poison = isnan(x) || x == INFINITY;
            const float p = (valid && !empty && !poison) ? metal::precise::exp(x - mn) : 0.0f;
            pv[i] = p;
            l += p;
        }
        for (ushort oo = 1; oo < 8; oo <<= 1) l += simd_shuffle_xor(l, oo);
        if (lane8 == 0) {
            M[h] = mn;
            L[h] = L[h] * a + l;
            D[(h >> 3) * 64 + (h & 7) * 9] = a;
            BAD[h] |= bad;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (stats) {
        for (uint i = 0; i < SPLIT / 8; i++) P[h * SPLIT + lane8 + 8 * i] = T(pv[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // rescale O by diag(a) of this simdgroup's head group (one MMA per
    // fragment; O holds no NaN, see the statistics)
    {
        simdgroup_float8x8 df;
        simdgroup_load(df, D + g * 64, 8);
        for (uint j = 0; j < NF; j++) {
            simdgroup_float8x8 t;
            simdgroup_multiply(t, df, o[j]);
            o[j] = t;
        }
    }

    // pass 2: values (T P x T V into fp32)
    for (uint t = 0; t < SPLIT / 8; t++) {
        const int k0 = first + int(t * 8);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < 1024; e += NT) {
            const uint kk = e >> 7;
            const uint c4 = e & 127;
            const int k = k0 + int(kk);
            vec<T, 4> v = vec<T, 4>(0);
            if (k < WIDTH) {
                int id = rid[k];
                if (id >= 0) {
                    id = min(id, LAST);
                    const device vec<T, 4>* src = (const device vec<T, 4>*)(id < T0
                        ? base + size_t(id) * 512 : tails + size_t(TOFF + id - T0) * 512);
                    v = src[c4];
                }
            }
            ((threadgroup vec<T, 4>*)tile)[e] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_matrix<T, 8, 8> pf;
        simdgroup_load(pf, P + (8 * g) * SPLIT + t * 8, SPLIT);
        const threadgroup T* vp = tile + sub * DO;
        for (uint j = 0; j < NF; j++) {
            simdgroup_matrix<T, 8, 8> vf;
            simdgroup_load(vf, vp + 8 * j, 512);
            simdgroup_multiply_accumulate(o[j], pf, vf, o[j]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// one partial per (row, head, outer): m, l and the un-normalized O. A head
// that saw a NaN score writes (NaN, 1): the combine's exp(NaN - m) makes
// that head's row NaN whatever its other slots hold.
if (tid < 64) {
    const size_t slot = (size_t(row) * 64 + tid) * SPLITS + outer;
    const bool empty = L[tid] == 0.0f && M[tid] == -INFINITY;
    partial_m[slot] = BAD[tid] ? NAN : (empty ? -INFINITY : M[tid]);
    partial_l[slot] = BAD[tid] ? 1.0f : (empty ? 0.0f : L[tid]);
}
device float* op = partial_o + ((size_t(row) * 64 + 8 * g) * SPLITS + outer) * 512 + sub * DO;
for (uint j = 0; j < NF; j++) simdgroup_store(o[j], op + 8 * j, SPLITS * 512);
"""

_COMBINE_SOURCE = r"""
const uint c = thread_position_in_grid.x;
const uint head = thread_position_in_grid.y;
const uint row = thread_position_in_grid.z;
const uint tid = thread_index_in_threadgroup;
threadgroup float a[SPLITS];
const size_t slots = (size_t(row) * 64 + head) * SPLITS;
float m = -INFINITY;
for (uint s = 0; s < SPLITS; s++)
    if (partial_l[slots + s] != 0.0f) m = max(m, partial_m[slots + s]);
if (tid < SPLITS) {
    const float ls = partial_l[slots + tid];
    a[tid] = ls != 0.0f ? metal::precise::exp(partial_m[slots + tid] - m) : 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float l = 0.0f;
float acc = 0.0f;
for (uint s = 0; s < SPLITS; s++) {
    l += a[s] * partial_l[slots + s];
    acc += a[s] * partial_o[(slots + s) * 512 + c];
}
out[(size_t(row) * 64 + head) * 512 + c] = acc / l;
"""


def kernel_spec():
    return {
        "partial": {"name": "mlx2_glm53_gathered_attention_v2",
                    "input_names": ["q", "base", "tails", "ids", "row_meta"],
                    "output_names": ["partial_m", "partial_l", "partial_o"],
                    "header": _HEADER, "source": _PARTIAL_SOURCE},
        # The flash-style kernel; its T template picks the law: half for
        # gathered_f16_from_<n>, float for the bench's fp32 twin.
        "partial_flash": {"name": "mlx2_glm53_gathered_attention_flash",
                          "input_names": ["q", "base", "tails", "ids", "row_meta"],
                          "output_names": ["partial_m", "partial_l", "partial_o"],
                          "header": _HEADER, "source": _PARTIAL_SOURCE_FLASH},
        "combine": {"name": "mlx2_glm53_gathered_attention_v2_combine",
                    "input_names": ["partial_m", "partial_l", "partial_o"], "output_names": ["out"],
                    "header": _HEADER, "source": _COMBINE_SOURCE},
    }


def _check_config(split, dim_split):
    if split not in SPLITS_SUPPORTED or dim_split not in DIM_SPLITS_SUPPORTED:
        raise ValueError(f"GLM-5.3 fused_v2 supports SPLIT {SPLITS_SUPPORTED} and DIM_SPLIT {DIM_SPLITS_SUPPORTED}")
    if split % (8 * dim_split):
        raise ValueError("GLM-5.3 fused_v2 SPLIT must be a multiple of the pass-1 tile (8 * DIM_SPLIT keys)")


def num_splits(width, split=DEFAULT_SPLIT):
    return -(-width // split)


def flash_plan(width, split, outer):
    """(outer partial slots, inner splits per threadgroup) of one flash launch."""
    if type(outer) is not int or outer < 1:
        raise ValueError("GLM-5.3 flash kernel needs at least one outer split")
    total = num_splits(width, split)
    outer = min(outer, total)
    return outer, -(-total // outer)


def launch_plan(rows, width, *, split=DEFAULT_SPLIT, dim_split=DEFAULT_DIM_SPLIT, kind="partial", outer=None):
    """Grids of one launch over ``rows`` query rows and ``width`` ids each.

    ``kind``: ``"partial"`` (the fp32 per-split fused_v2 kernel), ``"flash_f16"``
    (the fp16-operand flash kernel, the ``gathered_f16_from_<n>`` law) or
    ``"flash_f32"`` (the same kernel on fp32 operands; bench only). A flash
    launch has ``outer`` partial slots (default FLASH_OUTER) instead of one
    per split."""
    _check_config(split, dim_split)
    if type(rows) is not int or rows < 1 or type(width) is not int or width < 1:
        raise ValueError("GLM-5.3 fused_v2 launch needs at least one row and a positive width")
    if kind not in KINDS:
        raise ValueError(f"GLM-5.3 gathered kernel kind must be one of {KINDS}")
    threads = 256 * dim_split
    template = [("SPLIT", split), ("DSPLIT", dim_split), ("WIDTH", width)]
    if kind == "partial":
        splits = num_splits(width, split)
        kernel, dtype = "partial", mx.float32
        bytes_tg = 4 * 4096 + 4 * HEADS * split
        template.append(("SPLITS", splits))
    else:
        dtype = mx.float16 if kind == "flash_f16" else mx.float32
        splits, inner = flash_plan(width, split, FLASH_OUTER if outer is None else outer)
        kernel = "partial_flash"
        bytes_tg = dtype.size * 4096 + 4 * HEADS * split + 4 * (8 * 64 + 3 * 64)
        if bytes_tg > 32768:
            # The fp32 twin at SPLIT 64 (35 KiB): Metal's 32 KiB threadgroup limit.
            raise ValueError(f"GLM-5.3 flash kernel {kind} at SPLIT {split} needs {bytes_tg} threadgroup bytes "
                             "(32768 is the Metal limit); use SPLIT 32")
        template += [("SPLITS", splits), ("INNER", inner), ("T", dtype)]
    return {
        "splits": splits,
        "threads": threads,
        "kernel": kernel,
        "dtype": dtype,
        "threadgroup_bytes": bytes_tg,
        "partial": {"grid": (threads * splits, 1, rows), "threadgroup": (threads, 1, 1),
                    "output_shapes": [(rows, HEADS, splits)] * 2 + [(rows, HEADS, splits, DIM)]},
        "combine": {"grid": (DIM, HEADS, rows), "threadgroup": (DIM, 1, 1),
                    "output_shapes": [(rows, HEADS, DIM)]},
        "template": template,
    }


def workspace_bytes(rows, *, width=TOPK + 3, split=DEFAULT_SPLIT):
    """Partials of one launch: rows * splits * 64 * (512 + 2) fp32 words."""
    return rows * num_splits(width, split) * HEADS * (DIM + 2) * 4


@lru_cache(maxsize=3)
def _kernel(kind):
    return mx.fast.metal_kernel(**kernel_spec()[kind], ensure_row_contiguous=True)


def require_metal():
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        raise RuntimeError("GLM-5.3 fused_v2 requires Metal on the default GPU device; no fallback law")


def _launch(q_rows, base, tails, ids, row_meta, split, dim_split, kind="partial", outer=None):
    """The two kernels; q_rows (L, 64, 512) -> (L, 64, 512). No gate, no checks.

    ``kind`` picks the partial kernel (see ``launch_plan``); the caller casts
    q/base/tails to the kind's operand dtype. Partials, the combine and the
    output stay fp32."""
    rows, width = ids.shape
    plan = launch_plan(rows, width, split=split, dim_split=dim_split, kind=kind, outer=outer)
    m, l, o = _kernel(plan["kernel"])(
        inputs=[q_rows, base, tails, ids, row_meta], template=plan["template"],
        grid=plan["partial"]["grid"], threadgroup=plan["partial"]["threadgroup"],
        output_shapes=plan["partial"]["output_shapes"], output_dtypes=[mx.float32] * 3)
    (out,) = _kernel("combine")(
        inputs=[m, l, o], template=plan["template"],
        grid=plan["combine"]["grid"], threadgroup=plan["combine"]["threadgroup"],
        output_shapes=plan["combine"]["output_shapes"], output_dtypes=[mx.float32])
    _SCHEDULED["launches"] += 1
    _SCHEDULED["rows"] += rows
    return out


def launch(q_rows, base, tails, ids, row_meta, *, split=DEFAULT_SPLIT, dim_split=DEFAULT_DIM_SPLIT, half=False,
           kind=None, outer=None):
    """Validated ABI entry: q_rows (L, 64, 512), base (T0, 512), tails (>= 1, 512),
    ids (L, W) int32, row_meta (L, 4) int32 -> (L, 64, 512) fp32.

    ``half=True`` is the fp16-operand flash kernel (``kind="flash_f16"``); q,
    base and tails are cast to its operand dtype here (round to nearest even,
    as ``astype`` does) when not already. ``kind``/``outer`` name a kernel
    explicitly (the bench's fp32 flash twin, an outer-split sweep)."""
    _check_config(split, dim_split)
    if q_rows.ndim != 3 or q_rows.shape[1:] != (HEADS, DIM):
        raise ValueError("GLM-5.3 fused_v2 is built for 64 heads of 512")
    rows = q_rows.shape[0]
    if ids.ndim != 2 or ids.shape[0] != rows or row_meta.shape != (rows, 4):
        raise ValueError("GLM-5.3 fused_v2 needs ids (L, W) and row_meta (L, 4)")
    if base.ndim != 2 or base.shape[1] != DIM or tails.ndim != 2 or tails.shape[1] != DIM or tails.shape[0] < 1:
        raise ValueError("GLM-5.3 fused_v2 needs base (T0, 512) and a non-empty tails (n, 512) buffer")
    kind = kind or ("flash_f16" if half else "partial")
    dtype = launch_plan(rows, ids.shape[1], split=split, dim_split=dim_split, kind=kind, outer=outer)["dtype"]
    return _launch(q_rows.astype(dtype), base.astype(dtype), tails.astype(dtype),
                   ids.astype(mx.int32), row_meta.astype(mx.int32), split, dim_split, kind, outer)


def checked_launch(q_rows, base, tails, ids, row_meta, *, split=DEFAULT_SPLIT, dim_split=DEFAULT_DIM_SPLIT,
                   half=False):
    """``launch`` behind the Metal requirement and the first-use gate (fails closed)."""
    require_metal()
    self_check(split=split, dim_split=dim_split, half=half)
    return launch(q_rows, base, tails, ids, row_meta, split=split, dim_split=dim_split, half=half)


# ---------------------------------------------------------------------------
# float64 oracle and the first-use gate (design 1.1 test (a)).

def oracle(q_rows, base, tails, ids, row_meta):
    """float64 reference of the ABI: (L, 64, 512) numpy, NaN rows for empty selections."""
    import numpy as np

    q = np.asarray(q_rows, np.float64)
    base, tails = np.asarray(base, np.float64), np.asarray(tails, np.float64)
    ids, meta = np.asarray(ids), np.asarray(row_meta)
    out = np.full(q.shape, np.nan)
    for r in range(q.shape[0]):
        t0, d, off, _ = (int(v) for v in meta[r])
        row = ids[r][ids[r] >= 0]
        if not len(row):
            continue
        keys = np.concatenate([base[:t0], tails[off : off + d]])[np.minimum(row, t0 + d - 1)]
        s = q[r] @ keys.T * SCALE
        p = np.exp(s - s.max(-1, keepdims=True))
        out[r] = (p / p.sum(-1, keepdims=True)) @ keys
    return out


def oracle_errors(got, want):
    """(max abs, rel RMS) over the finite oracle entries; None when NaN placement differs."""
    import numpy as np

    got = np.asarray(got, np.float64)
    nan = np.isnan(want)
    if not np.array_equal(np.isnan(got), nan):
        return None
    if nan.all():
        return 0.0, 0.0
    diff = got[~nan] - want[~nan]
    return float(np.max(np.abs(diff))), float(np.sqrt(np.mean(diff**2)) / np.sqrt(np.mean(want[~nan] ** 2)))


def gate_cases(seed=5303):
    """Synthetic unit-scale cases: a decode row at W = 2051, half-filled and empty
    rows, a tree base + distinct tails, and dense causal rows at pos 0, 3, 2047."""
    import numpy as np

    rng = np.random.default_rng(seed)
    width = TOPK + 3
    cases = []
    total = 4096
    latent = rng.normal(size=(total, DIM)).astype(np.float32)
    ids = np.full((3, width), -1, np.int32)
    ids[0] = np.sort(rng.choice(total, size=width, replace=False))
    ids[1, : width // 2] = np.sort(rng.choice(total, size=width // 2, replace=False))
    meta = np.array([[total, 0, 0, total - 1]] * 3, np.int32)
    q = rng.normal(size=(3, HEADS, DIM)).astype(np.float32)
    cases.append(("plain", q, latent, np.zeros((1, DIM), np.float32), ids, meta))
    t0 = total - 7
    tails = rng.normal(size=(2 + 5, DIM)).astype(np.float32)
    ids = np.full((2, width), -1, np.int32)
    ids[0, :width] = np.sort(np.concatenate([rng.choice(t0, size=width - 2, replace=False), [t0, t0 + 1]]))
    ids[1, :width] = np.sort(np.concatenate([rng.choice(t0, size=width - 5, replace=False), t0 + np.arange(5)]))
    meta = np.array([[t0, 2, 0, t0 + 1], [t0, 5, 2, t0 + 4]], np.int32)
    q = rng.normal(size=(2, HEADS, DIM)).astype(np.float32)
    cases.append(("tails", q, latent[:t0], tails, ids, meta))
    positions = np.array([0, 3, TOPK - 1], np.int32)
    ids = np.asarray(dense_row_ids_np(positions, width))
    meta = np.array([[TOPK, 0, 0, int(p)] for p in positions], np.int32)
    q = rng.normal(size=(3, HEADS, DIM)).astype(np.float32)
    cases.append(("dense", q, latent[:TOPK], np.zeros((1, DIM), np.float32), ids, meta))
    return cases


def dense_row_ids_np(positions, width):
    import numpy as np

    arange = np.arange(width, dtype=np.int32)
    positions = np.asarray(positions, np.int32)
    return np.where(arange[None] <= positions[:, None], arange[None], -1).astype(np.int32)


def ops_reference(q_rows, base, tails, ids, row_meta):
    """The gathered_v1 ops law (gather + SDPA, one row per call) on the same ABI operands."""
    import numpy as np

    from . import glm53_sparse_attention as sa

    meta = np.asarray(row_meta)
    rows = []
    for r in range(meta.shape[0]):
        t0, d, off, _ = (int(x) for x in meta[r])
        source = (sa.TreeLatentView(mx.array(base[:t0]), mx.array(tails[off : off + d]), t0) if d
                  else mx.array(base[:t0])[None, None])
        q = mx.array(q_rows[r : r + 1]).transpose(1, 0, 2)[None]
        rows.append(sa.gathered_attention(q, *sa.gather_latent(source, mx.array(ids[r : r + 1])[None])))
    return np.array(mx.concatenate(rows, axis=2))[0].transpose(1, 0, 2)


# fp16-operand gate bounds against float64 on the half-rounded operands: what
# remains is the half rounding of P (2^-11 relative per weight) and the fp32
# MMA order, so a unit-scale weighted average of 2051 rows sits well under
# 1e-3 absolute and 2e-3 relative RMS (recorded per case in GATE_REPORT).
F16_ABS_BOUND, F16_REL_RMS_BOUND = 1e-3, 2e-3


def half_rounded(*arrays):
    """The operands as the fp16 kernel sees them: round to half, back to fp32 numpy."""
    import numpy as np

    return tuple(np.asarray(a, np.float32).astype(np.float16).astype(np.float32) for a in arrays)


def self_check(*, split=DEFAULT_SPLIT, dim_split=DEFAULT_DIM_SPLIT, abs_bound=1e-5, rel_rms_bound=1e-6, half=False):
    """First-use GPU gate against float64; sticky verdict per configuration.

    Bound per case: max abs <= 1e-5, and rel RMS <= max(1e-6, the gathered_v1
    ops law's rel RMS on the same case). The 1e-6 rel-RMS line sits at the fp32
    floor of a 2051-row weighted average (the ops law itself exceeds it on
    unit-scale cases, e.g. 5385578a2's attention receipt), so the kernel is held
    to "no worse than the reference law against float64" where that is above
    1e-6. Both arms' errors are kept in GATE_REPORT.

    ``half=True`` gates the fp16-operand kernel against float64 on the
    half-rounded operands (F16_ABS_BOUND, F16_REL_RMS_BOUND), NaN placement
    equal; the ops law's error is recorded for the report only.
    """
    key = (split, dim_split, half)
    verdict = _GATE.get(key)
    if verdict is True:
        return True
    if verdict is False:
        raise RuntimeError("GLM-5.3 fused_v2 kernel failed its first-use gate")
    require_metal()
    ok = True
    report = {}
    for name, q, base, tails, ids, meta in gate_cases():
        got = launch(mx.array(q), mx.array(base), mx.array(tails), mx.array(ids), mx.array(meta),
                     split=split, dim_split=dim_split, half=half)
        mx.eval(got)
        if half:
            q, base, tails = half_rounded(q, base, tails)
        want = oracle(q, base, tails, ids, meta)
        errors = oracle_errors(got, want)
        ops = oracle_errors(ops_reference(q, base, tails, ids, meta), want)
        if half:
            bound_abs, rel = F16_ABS_BOUND, F16_REL_RMS_BOUND
        else:
            bound_abs, rel = abs_bound, (max(rel_rms_bound, ops[1]) if ops is not None else rel_rms_bound)
        passed = errors is not None and errors[0] <= bound_abs and errors[1] <= rel
        report[name] = {"kernel": errors, "ops_reference": ops, "abs_bound": bound_abs, "rel_rms_bound": rel,
                        "passed": passed}
        ok &= passed
    GATE_REPORT[f"split{split}_dsplit{dim_split}" + ("_f16" if half else "")] = report
    _GATE[key] = ok
    if not ok:
        raise RuntimeError("GLM-5.3 fused_v2 kernel failed its first-use gate")
    return True


def route_status():
    return {
        "first_use_gate": {f"split{s}_dsplit{d}" + ("_f16" if h else ""): ("passed" if v else "failed")
                           for (s, d, h), v in _GATE.items()}
        or "not_run",
        "config": {"split": DEFAULT_SPLIT, "dim_split": DEFAULT_DIM_SPLIT},
        "gate_report": dict(GATE_REPORT),
        "scheduled": dict(_SCHEDULED),
    }

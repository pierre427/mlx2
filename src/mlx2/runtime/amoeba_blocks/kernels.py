"""Metal sources of the gated block-sparse MLP kernels, and their build/offline-compile helpers.

Four kernels per backend, plus one tile-gate kernel:

* ``gu``   gate + up + SiLU*mul, fused: one threadgroup per (packed chunk, work tile).  A chunk
           closed for every row of its work tile reads no weight: it writes zeros for its
           neurons (through ``perm``, back in original order) and returns.
* ``dn``   down projection over the original ``down`` layout: one threadgroup per (output
           column tile, work tile).  It first marks which original quantization groups have an
           open owner tile for any of its rows (``gsig``), then runs the K loop over the open
           groups only, in original group order.
* ``gate`` box probes -> box open bits -> tile open bits (OR over a precomputed closure bit
           matrix), one threadgroup per token, optional union over tokens (atomic OR).

Backends.  ``mpp`` is the M5 tensor-unit (NAX) path: MPP ``matmul2d`` over one weight group
(``GS`` inputs) per op, with the TensorFold-derived fragment mapping and in-kernel group sums of
the lane matmul (see provenance/amoeba-blocks.json).  ``simd`` is the M1-M4 path: each lane owns
8 consecutive inputs, 32 lanes cover 256 inputs per step, fp32 per-lane accumulation and one
``simd_sum`` (the shape of MLX's qmv, not its code).  The two are different numerical laws.

Law (both backends).  For weight group g with codes q, scale s and bias b, and the group's
inputs v: ``P = q . v`` (fp32), ``xs = sum(v)`` (fp32, fixed order), and the accumulator takes
``acc = fma(s, P, fma(b, xs, acc))`` (mpp) or ``acc += fma(s, P, b * xs)`` per lane (simd).
Groups are visited in index order; split-K slices (mpp) are fixed by the weight shape only and
added in slice order.  A skipped group is a group whose inputs are all zero, so it would add
``fma(s, 0, fma(b, 0, acc)) == acc``: skipping it leaves every output bit unchanged (an exact
zero's sign aside).  Gate/up rows are computed independently of their position, so the packed
layout gives the unpacked layout's bits.  The activation is
``a = XT(g * (1 / (1 + exp(-g))) * u)`` with ``g = float(XT(gate_acc))`` and
``u = float(XT(up_acc))`` (the stock projections' rounding to the activation dtype).

Nothing here is selected by any route.  Every kernel is opt-in and unqualified.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache

VARIANTS = ("dense", "rowmap")     # dense: work rows are x rows; rowmap: x row = RMAP[work row]
BACKENDS = ("mpp", "simd")
BITS = (4, 8)
GROUP_SIZES = (32, 64, 128)
NT = 32                            # mpp output columns per tile (one chunk = at most NT rows)
SIMD_DOWN_COLS = 8                 # simd down outputs per threadgroup (2 simdgroups x 4)
SIMD_GU_THREADS = 128              # simd gate/up: 4 simdgroups x 8 neurons
MAX_MR = 8                         # simd rows per work tile
MAX_TMR = 4                        # mpp 16-row fragments per work tile
TG_MEMORY_BUDGET = 28 * 1024       # bytes of threadgroup memory a kernel may declare
LAW = {"mpp": "amoeba-blocks-mpp-group-seq-v1", "simd": "amoeba-blocks-simd-qmv-shape-v1"}
GATE_LAW = "amoeba-blocks-gate-f32-lane-strided-simd-sum-v1"

_TYPES = {"bfloat16": "bfloat", "float16": "half"}
_NATIVE = {4: "uint4b_format", 8: "uint8_t"}

_MPP_HEADER = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
"""

_COMMON = r"""
inline float bs_silu_mul(float g, float u) { return (g * (1.0f / (1.0f + metal::exp(-g)))) * u; }
"""

# ---------------------------------------------------------------------------
# mpp (NAX) gate/up + SwiGLU
# ---------------------------------------------------------------------------

_PROLOGUE = r"""
  const uint tid = thread_index_in_threadgroup;
  const int c = int(threadgroup_position_in_grid.x);
  const int u = int(threadgroup_position_in_grid.y);
  const int row0 = WTL[u * 4 + 0], nrows = WTL[u * 4 + 1], e = WTL[u * 4 + 2];
  if (nrows <= 0) return;
  const int ci = CPTR[e] + c;
  if (ci >= CPTR[e + 1]) return;
  const int off = CK[ci * 3 + 0], len = CK[ci * 3 + 1], tile = CK[ci * 3 + 2];
  ulong open = 0;
  for (int m = 0; m < nrows; m++) {
    const int tok = @@TOK@@;
    if (TOPEN[(int64_t)(MS ? tok : 0) * NTILES + tile]) open |= (1ul << m);
  }
  const device ushort* perm = (const device ushort*)PERM + (int64_t)e * F + off;
  if (SKIP && open == 0) {
    // Closed for every row: no weight read; the neurons' activations are exact zeros.
    for (int i = int(tid); i < nrows * len; i += @@THREADS@@) {
      const int m = i / len, j = i - m * len;
      A[(int64_t)(row0 + m) * F + perm[j]] = static_cast<XT>(0.0f);
    }
    return;
  }
  if (COUNT && tid == 0) {
    atomic_fetch_add_explicit((device atomic_uint*)CTR + 0, 1u, memory_order_relaxed);
    atomic_fetch_add_explicit((device atomic_uint*)CTR + 1, uint(len), memory_order_relaxed);
  }
"""

_GU_MPP = r"""
  const ushort lane = thread_index_in_simdgroup;
  const ushort sg = simdgroup_index_in_threadgroup;
  const short qid = lane >> 2;
  const short fm = (qid & 4) | ((lane >> 1) & 3);
  const short fn = ((qid & 2) | (lane & 1)) * 4;
  constexpr int KG = K / GS;
  constexpr int RT = 16 * TMR;
  constexpr int64_t ROW_UNITS = K * BITS / 8;
@@PROLOGUE@@
  const int g_begin = (int(sg) * KG) / SK;
  const int g_end = ((int(sg) + 1) * KG) / SK;
  constexpr auto desc = matmul2d_descriptor(RT, 32, GS, false, true, false,
                                            matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  tensor<device WT, dextents<int32_t, 2>, tensor_inline> tG(
      (device WP*)WG + ((int64_t)e * F + off) * ROW_UNITS, dextents<int32_t, 2>(K, len));
  tensor<device WT, dextents<int32_t, 2>, tensor_inline> tU(
      (device WP*)WU + ((int64_t)e * F + off) * ROW_UNITS, dextents<int32_t, 2>(K, len));
@@A_DECL@@
  threadgroup float part[(SK > 1 ? SK - 1 : 1) * 2 * 16 * 32];
  float CG[TMR][16], CU[TMR][16];
  for (int t = 0; t < TMR; t++)
    for (int i = 0; i < 16; i++) { CG[t][i] = 0.0f; CU[t][i] = 0.0f; }
  for (int g = g_begin; g < g_end; g++) {
    float sgv[2][4], bgv[2][4], suv[2][4], buv[2][4];
    for (int f = 0; f < 2; f++)
      for (int j = 0; j < 4; j++) {
        const int n = f * 16 + fn + j;
        const bool ok = n < len;
        const int64_t q = ((int64_t)e * F + off + (ok ? n : 0)) * KG + g;
        sgv[f][j] = ok ? float(SG[q]) : 0.0f;
        bgv[f][j] = ok ? float(BG[q]) : 0.0f;
        suv[f][j] = ok ? float(SU[q]) : 0.0f;
        buv[f][j] = ok ? float(BU[q]) : 0.0f;
      }
@@A_GROUP@@
    auto bgt = tG.slice(g * GS, 0);
    auto but = tU.slice(g * GS, 0);
    auto PG = op.template get_destination_cooperative_tensor<decltype(a), decltype(bgt), float>();
    op.run(a, bgt, PG);
    auto PU = op.template get_destination_cooperative_tensor<decltype(a), decltype(but), float>();
    op.run(a, but, PU);
    const int quarter = (((lane >> 3) & 1) << 1) | (lane & 1);
    for (int t = 0; t < TMR; t++) {
      const int m0 = t * 16 + fm;
      float xs0 = 0.0f, xs1 = 0.0f;
      if (m0 < nrows) {
@@XS0@@
      }
      if (m0 + 8 < nrows) {
@@XS1@@
      }
      xs0 += simd_shuffle_xor(xs0, ushort(1));
      xs1 += simd_shuffle_xor(xs1, ushort(1));
      xs0 += simd_shuffle_xor(xs0, ushort(8));
      xs1 += simd_shuffle_xor(xs1, ushort(8));
      for (int f = 0; f < 2; f++)
        for (int r = 0; r < 2; r++)
          for (int j = 0; j < 4; j++) {
            const int i = f * 8 + r * 4 + j;
            const float xs = r ? xs1 : xs0;
            CG[t][i] = fma(sgv[f][j], PG[t * 16 + i], fma(bgv[f][j], xs, CG[t][i]));
            CU[t][i] = fma(suv[f][j], PU[t * 16 + i], fma(buv[f][j], xs, CU[t][i]));
          }
    }
@@A_AFTER@@
  }
  for (int t = 0; t < TMR; t++) {
    if (SK > 1) {
      if (sg > 0)
        for (int i = 0; i < 16; i++) {
          part[(((int(sg) - 1) * 2 + 0) * 16 + i) * 32 + lane] = CG[t][i];
          part[(((int(sg) - 1) * 2 + 1) * 16 + i) * 32 + lane] = CU[t][i];
        }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg == 0)
        for (int s2 = 1; s2 < SK; s2++)
          for (int i = 0; i < 16; i++) {
            CG[t][i] += part[(((s2 - 1) * 2 + 0) * 16 + i) * 32 + lane];
            CU[t][i] += part[(((s2 - 1) * 2 + 1) * 16 + i) * 32 + lane];
          }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sg == 0)
      for (int f = 0; f < 2; f++)
        for (int r = 0; r < 2; r++) {
          const int m = t * 16 + fm + 8 * r;
          if (m >= nrows) continue;
          const bool ro = ((open >> m) & 1ul) != 0;
          for (int j = 0; j < 4; j++) {
            const int n = f * 16 + fn + j;
            if (n >= len) continue;
            const int i = f * 8 + r * 4 + j;
            float v = 0.0f;
            if (ro) v = bs_silu_mul(float(static_cast<XT>(CG[t][i])), float(static_cast<XT>(CU[t][i])));
            A[(int64_t)(row0 + m) * F + perm[n]] = static_cast<XT>(v);
          }
        }
  }
"""

_MPP_A = {
    "dense": {
        "@@A_DECL@@": "  tensor<device XT, dextents<int32_t, 2>, tensor_inline> tA(\n"
                      "      (device XT*)X + (int64_t)row0 * K, dextents<int32_t, 2>(K, nrows));\n",
        "@@A_GROUP@@": "    auto a = tA.slice(g * GS, 0);\n",
        "@@XS0@@": "        const device XT* xp = (const device XT*)X + (int64_t)(row0 + m0) * K + g * GS + quarter * (GS / 4);\n"
                   "        for (int i = 0; i < GS / 4; i++) xs0 += float(xp[i]);\n",
        "@@XS1@@": "        const device XT* xp = (const device XT*)X + (int64_t)(row0 + m0 + 8) * K + g * GS + quarter * (GS / 4);\n"
                   "        for (int i = 0; i < GS / 4; i++) xs1 += float(xp[i]);\n",
        "@@A_AFTER@@": "",
    },
    # Routed rows: each simdgroup stages its group's x rows (through RMAP) in threadgroup memory,
    # so no gathered copy of x is ever written to device memory.
    "rowmap": {
        # The pad rows (nrows..RT) are zeroed once; each group restages only the live rows.
        "@@A_DECL@@": "  threadgroup XT xa[SK * RT * GS];\n"
                      "  threadgroup XT* mya = xa + int(sg) * RT * GS;\n"
                      "  const int nst = min(nrows, RT) * GS;\n"
                      "  for (int i = nst + int(lane); i < RT * GS; i += 32) mya[i] = static_cast<XT>(0.0f);\n",
        "@@A_GROUP@@": "    for (int i = int(lane); i < nst; i += 32) {\n"
                       "      const int m = i / GS, k = i - m * GS;\n"
                       "      mya[i] = ((const device XT*)X)[(int64_t)RMAP[row0 + m] * K + g * GS + k];\n"
                       "    }\n"
                       "    simdgroup_barrier(mem_flags::mem_threadgroup);\n"
                       "    tensor<threadgroup XT, dextents<int32_t, 2>, tensor_inline> a(mya, dextents<int32_t, 2>(GS, RT));\n",
        "@@XS0@@": "        threadgroup XT* xp = mya + m0 * GS + quarter * (GS / 4);\n"
                   "        for (int i = 0; i < GS / 4; i++) xs0 += float(xp[i]);\n",
        "@@XS1@@": "        threadgroup XT* xp = mya + (m0 + 8) * GS + quarter * (GS / 4);\n"
                   "        for (int i = 0; i < GS / 4; i++) xs1 += float(xp[i]);\n",
        "@@A_AFTER@@": "    simdgroup_barrier(mem_flags::mem_threadgroup);\n",
    },
}

# ---------------------------------------------------------------------------
# mpp (NAX) down
# ---------------------------------------------------------------------------

_GROUP_OPEN = r"""
  threadgroup uchar gopen[G];
  for (int g = int(tid); g < G; g += @@THREADS@@) {
    uchar o = SKIP ? 0 : 1;
    if (SKIP) {
      const int s0 = GPTR[e * G + g], s1 = GPTR[e * G + g + 1];
      for (int s = s0; s < s1 && !o; s++) {
        const int t = GSIG[s];
        for (int m = 0; m < nrows; m++) {
          const int tok = @@TOK@@;
          if (TOPEN[(int64_t)(MS ? tok : 0) * NTILES + t]) { o = 1; break; }
        }
      }
    }
    gopen[g] = o;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
"""

_DN_MPP = r"""
  const ushort lane = thread_index_in_simdgroup;
  const ushort sg = simdgroup_index_in_threadgroup;
  const uint tid = thread_index_in_threadgroup;
  const short qid = lane >> 2;
  const short fm = (qid & 4) | ((lane >> 1) & 3);
  const short fn = ((qid & 2) | (lane & 1)) * 4;
  constexpr int G = F / GS;
  constexpr int RT = 16 * TMR;
  constexpr int64_t ROW_UNITS = F * BITS / 8;
  const int n0 = int(threadgroup_position_in_grid.x) * 32;
  const int u = int(threadgroup_position_in_grid.y);
  const int row0 = WTL[u * 4 + 0], nrows = WTL[u * 4 + 1], e = WTL[u * 4 + 2];
  if (nrows <= 0) return;
@@GROUP_OPEN@@
  const int g_begin = (int(sg) * G) / SK;
  const int g_end = ((int(sg) + 1) * G) / SK;
  constexpr auto desc = matmul2d_descriptor(RT, 32, GS, false, true, false,
                                            matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  tensor<device XT, dextents<int32_t, 2>, tensor_inline> tA(
      (device XT*)A + (int64_t)row0 * F, dextents<int32_t, 2>(F, nrows));
  tensor<device WT, dextents<int32_t, 2>, tensor_inline> tB(
      (device WP*)W + ((int64_t)e * N + n0) * ROW_UNITS, dextents<int32_t, 2>(F, N - n0));
  threadgroup float part[(SK > 1 ? SK - 1 : 1) * 16 * 32];
  float C[TMR][16];
  for (int t = 0; t < TMR; t++)
    for (int i = 0; i < 16; i++) C[t][i] = 0.0f;
  const int quarter = (((lane >> 3) & 1) << 1) | (lane & 1);
  for (int g = g_begin; g < g_end; g++) {
    if (!gopen[g]) continue;               // every input of the group is zero: skipped, unread
    if (COUNT && lane == 0)
      atomic_fetch_add_explicit((device atomic_uint*)CTR + 2, 1u, memory_order_relaxed);
    float s[2][4], bb[2][4];
    for (int f = 0; f < 2; f++)
      for (int j = 0; j < 4; j++) {
        const int n = n0 + f * 16 + fn + j;
        const bool ok = n < N;
        const int64_t q = ((int64_t)e * N + (ok ? n : 0)) * G + g;
        s[f][j] = ok ? float(S[q]) : 0.0f;
        bb[f][j] = ok ? float(B[q]) : 0.0f;
      }
    auto a = tA.slice(g * GS, 0);
    auto b = tB.slice(g * GS, 0);
    auto P = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
    op.run(a, b, P);
    for (int t = 0; t < TMR; t++) {
      const int m0 = t * 16 + fm;
      float xs0 = 0.0f, xs1 = 0.0f;
      if (m0 < nrows) {
        const device XT* xp = (const device XT*)A + (int64_t)(row0 + m0) * F + g * GS + quarter * (GS / 4);
        for (int i = 0; i < GS / 4; i++) xs0 += float(xp[i]);
      }
      if (m0 + 8 < nrows) {
        const device XT* xp = (const device XT*)A + (int64_t)(row0 + m0 + 8) * F + g * GS + quarter * (GS / 4);
        for (int i = 0; i < GS / 4; i++) xs1 += float(xp[i]);
      }
      xs0 += simd_shuffle_xor(xs0, ushort(1));
      xs1 += simd_shuffle_xor(xs1, ushort(1));
      xs0 += simd_shuffle_xor(xs0, ushort(8));
      xs1 += simd_shuffle_xor(xs1, ushort(8));
      for (int f = 0; f < 2; f++)
        for (int r = 0; r < 2; r++)
          for (int j = 0; j < 4; j++) {
            const int i = f * 8 + r * 4 + j;
            C[t][i] = fma(s[f][j], P[t * 16 + i], fma(bb[f][j], r ? xs1 : xs0, C[t][i]));
          }
    }
  }
  for (int t = 0; t < TMR; t++) {
    if (SK > 1) {
      if (sg > 0)
        for (int i = 0; i < 16; i++) part[((int(sg) - 1) * 16 + i) * 32 + lane] = C[t][i];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg == 0)
        for (int s2 = 1; s2 < SK; s2++)
          for (int i = 0; i < 16; i++) C[t][i] += part[((s2 - 1) * 16 + i) * 32 + lane];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sg == 0)
      for (int f = 0; f < 2; f++)
        for (int r = 0; r < 2; r++) {
          const int m = t * 16 + fm + 8 * r;
          const int n = n0 + f * 16 + fn;
          if (m < nrows && n < N) {
            const int64_t dest = @@DEST@@;
            for (int j = 0; j < 4; j++) Y[dest * N + n + j] = static_cast<XT>(C[t][f * 8 + r * 4 + j]);
          }
        }
  }
"""

# ---------------------------------------------------------------------------
# simd (M1-M4) gate/up + SwiGLU and down
# ---------------------------------------------------------------------------

_UNPACK = {
    4: "        { const uint w = *(const device uint*)(@@ROW@@ + k0 / 2);\n"
       "          for (int v = 0; v < 8; v++) @@DST@@[v] = float((w >> (4 * v)) & 15u); }\n",
    8: "        { const uint2 w = *(const device uint2*)(@@ROW@@ + k0);\n"
       "          for (int v = 0; v < 4; v++) { @@DST@@[v] = float((w.x >> (8 * v)) & 255u);"
       " @@DST@@[4 + v] = float((w.y >> (8 * v)) & 255u); } }\n",
}


def _unpack(bits: int, row: str, dst: str) -> str:
    return _UNPACK[bits].replace("@@ROW@@", row).replace("@@DST@@", dst)


_GU_SIMD = r"""
  const ushort lane = thread_index_in_simdgroup;
  const ushort sg = simdgroup_index_in_threadgroup;
  constexpr int KG = K / GS;
  constexpr int64_t ROW_BYTES = K * BITS / 8;
@@PROLOGUE@@
  for (int q = 0; q < 2; q++) {
    const int nb = int(sg) * 8 + q * 4;
    if (nb >= len) break;
    const device uchar* wg[4];
    const device uchar* wu[4];
    int64_t srow[4];
    bool nok[4];
    for (int i = 0; i < 4; i++) {
      nok[i] = nb + i < len;
      srow[i] = (int64_t)e * F + off + (nok[i] ? nb + i : nb);
      wg[i] = (const device uchar*)WG + srow[i] * ROW_BYTES;
      wu[i] = (const device uchar*)WU + srow[i] * ROW_BYTES;
    }
    float ag[MR][4], au[MR][4];
    for (int m = 0; m < MR; m++)
      for (int i = 0; i < 4; i++) { ag[m][i] = 0.0f; au[m][i] = 0.0f; }
    for (int k0 = int(lane) * 8; k0 < K; k0 += 256) {
      const int g = k0 / GS;
      float qg[4][8], qu[4][8], sgv[4], bgv[4], suv[4], buv[4];
      for (int i = 0; i < 4; i++) {
@@UNPACK_G@@
@@UNPACK_U@@
        sgv[i] = float(SG[srow[i] * KG + g]); bgv[i] = float(BG[srow[i] * KG + g]);
        suv[i] = float(SU[srow[i] * KG + g]); buv[i] = float(BU[srow[i] * KG + g]);
      }
      for (int m = 0; m < MR; m++) {
        if (m >= nrows) break;
        if (SKIP && ((open >> m) & 1ul) == 0) continue;     // a closed row's output is zero
        const device XT* xp = (const device XT*)X + (int64_t)(@@TOKM@@) * K + k0;
        float xv[8], xsum = 0.0f;
        for (int v = 0; v < 8; v++) { xv[v] = float(xp[v]); xsum += xv[v]; }
        for (int i = 0; i < 4; i++) {
          float dg = 0.0f, du = 0.0f;
          for (int v = 0; v < 8; v++) { dg = fma(qg[i][v], xv[v], dg); du = fma(qu[i][v], xv[v], du); }
          ag[m][i] += fma(sgv[i], dg, bgv[i] * xsum);
          au[m][i] += fma(suv[i], du, buv[i] * xsum);
        }
      }
    }
    for (int m = 0; m < MR; m++) {
      if (m >= nrows) break;
      const bool ro = ((open >> m) & 1ul) != 0;
      for (int i = 0; i < 4; i++) {
        const float gsum = simd_sum(ag[m][i]);
        const float usum = simd_sum(au[m][i]);
        if (lane == 0 && nok[i]) {
          float v = 0.0f;
          if (ro) v = bs_silu_mul(float(static_cast<XT>(gsum)), float(static_cast<XT>(usum)));
          A[(int64_t)(row0 + m) * F + perm[nb + i]] = static_cast<XT>(v);
        }
      }
    }
  }
"""

_DN_SIMD = r"""
  const ushort lane = thread_index_in_simdgroup;
  const ushort sg = simdgroup_index_in_threadgroup;
  const uint tid = thread_index_in_threadgroup;
  constexpr int G = F / GS;
  constexpr int64_t ROW_BYTES = F * BITS / 8;
  const int u = int(threadgroup_position_in_grid.y);
  const int row0 = WTL[u * 4 + 0], nrows = WTL[u * 4 + 1], e = WTL[u * 4 + 2];
  if (nrows <= 0) return;
@@GROUP_OPEN@@
  if (COUNT && tid == 0) {
    uint opened = 0;
    for (int g = 0; g < G; g++) opened += gopen[g];
    atomic_fetch_add_explicit((device atomic_uint*)CTR + 2, opened, memory_order_relaxed);
  }
  const int nb = int(threadgroup_position_in_grid.x) * 8 + int(sg) * 4;
  const device uchar* wr[4];
  int64_t srow[4];
  bool nok[4];
  for (int i = 0; i < 4; i++) {
    nok[i] = nb + i < N;
    srow[i] = (int64_t)e * N + (nok[i] ? nb + i : 0);
    wr[i] = (const device uchar*)W + srow[i] * ROW_BYTES;
  }
  float acc[MR][4];
  for (int m = 0; m < MR; m++)
    for (int i = 0; i < 4; i++) acc[m][i] = 0.0f;
  for (int k0 = int(lane) * 8; k0 < F; k0 += 256) {
    const int g = k0 / GS;
    if (!gopen[g]) continue;               // every input of the group is zero: skipped, unread
    float qd[4][8], sv[4], bv[4];
    for (int i = 0; i < 4; i++) {
@@UNPACK_D@@
      sv[i] = float(S[srow[i] * G + g]);
      bv[i] = float(B[srow[i] * G + g]);
    }
    for (int m = 0; m < MR; m++) {
      if (m >= nrows) break;
      const device XT* ap = (const device XT*)A + (int64_t)(row0 + m) * F + k0;
      float av[8], asum = 0.0f;
      for (int v = 0; v < 8; v++) { av[v] = float(ap[v]); asum += av[v]; }
      for (int i = 0; i < 4; i++) {
        float d = 0.0f;
        for (int v = 0; v < 8; v++) d = fma(qd[i][v], av[v], d);
        acc[m][i] += fma(sv[i], d, bv[i] * asum);
      }
    }
  }
  for (int m = 0; m < MR; m++) {
    if (m >= nrows) break;
    const int64_t dest = @@DEST@@;
    for (int i = 0; i < 4; i++) {
      const float r = simd_sum(acc[m][i]);
      if (lane == 0 && nok[i]) Y[dest * N + nb + i] = static_cast<XT>(r);
    }
  }
"""

# ---------------------------------------------------------------------------
# Tile gates: box probes -> box bits -> tile bits (+ union)
# ---------------------------------------------------------------------------

_GATE = r"""
  const uint tid = thread_index_in_threadgroup;
  const ushort lane = thread_index_in_simdgroup;
  const ushort sg = simdgroup_index_in_threadgroup;
  const int t = int(threadgroup_position_in_grid.x);
  threadgroup atomic_uint bopen[NW];
  for (int w = int(tid); w < NW; w += 256) atomic_store_explicit(&bopen[w], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (MODE == 0) {
    // One simdgroup per box: lane-strided fp32 products, one simd_sum, plus the bias.
    for (int b = int(sg); b < NB; b += 8) {
      float acc = 0.0f;
      for (int i = int(lane); i < D; i += 32)
        acc = fma(float(H[(int64_t)t * D + i]), PW[(int64_t)b * D + i], acc);
      acc = simd_sum(acc) + PB[b];
      if (lane == 0 && acc > 0.0f)
        atomic_fetch_or_explicit(&bopen[b >> 5], 1u << (b & 31), memory_order_relaxed);
    }
  } else {
    for (int b = int(tid); b < NB; b += 256)
      if (BOX[(int64_t)t * NB + b])
        atomic_fetch_or_explicit(&bopen[b >> 5], 1u << (b & 31), memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int k = int(tid); k < NTILES; k += 256) {
    bool o = ALW[k] != 0;
    for (int w = 0; w < NW && !o; w++)
      o = (TBITS[k * NW + w] & atomic_load_explicit(&bopen[w], memory_order_relaxed)) != 0u;
    TOPEN[(int64_t)t * NTILES + k] = o ? 1 : 0;
    if (UNION && o) atomic_fetch_or_explicit((device atomic_uint*)UN + k, 1u, memory_order_relaxed);
  }
"""

# Buffer names (and their MLX dtypes, for the offline harness) per kernel.
GU_INPUTS = ("X", "WG", "SG", "BG", "WU", "SU", "BU", "PERM", "CPTR", "CK", "WTL", "RMAP", "TOPEN")
GU_OUTPUTS = ("A", "CTR")
DN_INPUTS = ("A", "W", "S", "B", "GPTR", "GSIG", "WTL", "RMAP", "OMAP", "TOPEN")
DN_OUTPUTS = ("Y", "CTR")
GATE_INPUTS = ("H", "PW", "PB", "BOX", "TBITS", "ALW")
GATE_OUTPUTS = ("TOPEN", "UN")


def _subst(src: str, parts: dict[str, str]) -> str:
    for marker, text in parts.items():
        if src.count(marker) < 1:
            raise AssertionError(f"kernel anchor {marker} missing")
        src = src.replace(marker, text)
    if "@@" in src:
        raise AssertionError("unsubstituted kernel anchor")
    return src


def source(kind: str, backend: str, variant: str, bits: int) -> str:
    """The body of one kernel (``kind`` gu/dn) for a backend, row variant and weight width."""
    if backend not in BACKENDS or variant not in VARIANTS or bits not in BITS:
        raise ValueError(f"unsupported kernel {kind}/{backend}/{variant}/q{bits}")
    tok = "RMAP[row0 + m]" if variant == "rowmap" else "row0 + m"
    if kind == "gu":
        threads = "32 * SK" if backend == "mpp" else str(SIMD_GU_THREADS)
        prologue = _subst(_PROLOGUE, {"@@TOK@@": tok, "@@THREADS@@": threads})
        if backend == "mpp":
            return _subst(_GU_MPP, {"@@PROLOGUE@@": prologue, **_MPP_A[variant]})
        return _subst(_GU_SIMD, {
            "@@PROLOGUE@@": prologue,
            "@@UNPACK_G@@": _unpack(bits, "wg[i]", "qg[i]"),
            "@@UNPACK_U@@": _unpack(bits, "wu[i]", "qu[i]"),
            "@@TOKM@@": "RMAP[row0 + m]" if variant == "rowmap" else "row0 + m",
        })
    if kind == "dn":
        threads = "32 * SK" if backend == "mpp" else "64"
        gopen = _subst(_GROUP_OPEN, {"@@TOK@@": tok, "@@THREADS@@": threads})
        dest = "OUTMAP ? (int64_t)OMAP[row0 + m] : (int64_t)(row0 + m)"
        if backend == "mpp":
            return _subst(_DN_MPP, {"@@GROUP_OPEN@@": gopen, "@@DEST@@": dest})
        return _subst(_DN_SIMD, {"@@GROUP_OPEN@@": gopen, "@@DEST@@": dest,
                                 "@@UNPACK_D@@": _unpack(bits, "wr[i]", "qd[i]")})
    if kind == "gate":
        return _GATE
    raise ValueError(f"unknown kernel kind {kind!r}")


def header(kind: str, backend: str, bits: int, xt: str, st: str) -> str:
    """Type aliases (and the MPP include for mpp matmuls) prepended to a kernel."""
    head = (_MPP_HEADER if backend == "mpp" and kind in ("gu", "dn") else "") + _COMMON
    head += f"typedef {xt} XT;\ntypedef {st} ST;\nconstexpr constant int BITS = {bits};\n"
    if backend == "mpp" and kind in ("gu", "dn"):
        head += f"typedef {_NATIVE[bits]} WT;\ntypedef uchar WP;\n"
    return head


def kernel_name(kind: str, backend: str, variant: str, bits: int, xt: str, st: str) -> str:
    body = header(kind, backend, bits, xt, st) + source(kind, backend, variant, bits)
    digest = hashlib.sha256(body.encode()).hexdigest()[:16]
    return f"mlx2_amoeba_blocks_{kind}_{backend}_{variant}_q{bits}_{xt}_{st}_{digest}"


@lru_cache(maxsize=None)
def build(kind: str, backend: str, variant: str, bits: int, xt: str, st: str):
    """The ``mx.fast.metal_kernel`` object (compiled by MLX on first launch, on the GPU)."""
    import mlx.core as mx

    inputs, outputs = {"gu": (GU_INPUTS, GU_OUTPUTS), "dn": (DN_INPUTS, DN_OUTPUTS),
                       "gate": (GATE_INPUTS, GATE_OUTPUTS)}[kind]
    return mx.fast.metal_kernel(
        name=kernel_name(kind, backend, variant, bits, xt, st),
        input_names=list(inputs), output_names=list(outputs),
        source=source(kind, backend, variant, bits), header=header(kind, backend, bits, xt, st),
        ensure_row_contiguous=True, compile_options={"math_mode": "safe"},
    )


# ---------------------------------------------------------------------------
# Offline compilation (xcrun metal; CPU only, never the GPU)
# ---------------------------------------------------------------------------

_MSL_TYPES = {"bf16": "bfloat16_t", "f16": "half", "f32": "float", "u32": "uint32_t", "u16": "uint16_t",
              "u8": "uint8_t", "i32": "int32_t"}


def buffer_types(kind: str, xt: str, st: str) -> dict[str, str]:
    x = "bf16" if xt == "bfloat" else "f16"
    s = "bf16" if st == "bfloat" else "f16"
    if kind == "gu":
        return {"X": x, "WG": "u32", "SG": s, "BG": s, "WU": "u32", "SU": s, "BU": s, "PERM": "u16",
                "CPTR": "i32", "CK": "i32", "WTL": "i32", "RMAP": "i32", "TOPEN": "u8", "A": x, "CTR": "u32"}
    if kind == "dn":
        return {"A": x, "W": "u32", "S": s, "B": s, "GPTR": "i32", "GSIG": "i32", "WTL": "i32",
                "RMAP": "i32", "OMAP": "i32", "TOPEN": "u8", "Y": x, "CTR": "u32"}
    return {"H": x, "PW": "f32", "PB": "f32", "BOX": "u8", "TBITS": "u32", "ALW": "u8",
            "TOPEN": "u8", "UN": "u32"}


def compilation_source(kind: str, backend: str, variant: str, bits: int, consts: dict[str, int],
                       *, xt: str = "bfloat", st: str = "bfloat") -> str:
    """A complete .metal translation unit equivalent to what MLX builds (template ints as constants)."""
    inputs, outputs = {"gu": (GU_INPUTS, GU_OUTPUTS), "dn": (DN_INPUTS, DN_OUTPUTS),
                       "gate": (GATE_INPUTS, GATE_OUTPUTS)}[kind]
    types = buffer_types(kind, xt, st)
    args = [f"const device {_MSL_TYPES[types[n]]}* {n} [[buffer({i})]]" for i, n in enumerate(inputs)]
    args += [f"device {_MSL_TYPES[types[n]]}* {n} [[buffer({len(inputs) + i})]]" for i, n in enumerate(outputs)]
    args += ["uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]",
             "uint thread_index_in_threadgroup [[thread_index_in_threadgroup]]",
             "uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]",
             "uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]"]
    const = "".join(f"constant constexpr int {k} = {int(v)};\n" for k, v in consts.items())
    return ('#include "mlx/backend/metal/kernels/utils.h"\n' + header(kind, backend, bits, xt, st) + const
            + f"[[kernel]] void amoeba_blocks_{kind}(\n  " + ",\n  ".join(args) + ") {\n"
            + source(kind, backend, variant, bits) + "\n}\n")


def threadgroup_bytes(kind: str, backend: str, variant: str, *, sk: int, tmr: int, gs: int,
                      groups: int = 0, xbytes: int = 2) -> int:
    """Threadgroup memory a kernel instantiation declares (the budget check)."""
    if kind == "gu" and backend == "mpp":
        stage = sk * 16 * tmr * gs * xbytes if variant == "rowmap" else 0
        return stage + max(sk - 1, 1) * 2 * 16 * 32 * 4
    if kind == "dn":
        return groups + (max(sk - 1, 1) * 16 * 32 * 4 if backend == "mpp" else 0)
    return 0

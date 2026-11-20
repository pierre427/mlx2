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
* ``gup`` / ``dnp``  the prefill law (rows >= ops.PREFILL_MIN_ROWS): 64x64 threadgroup tiles with
           the weight tile dequantized into threadgroup memory, fp32 multiply-accumulate (see
           "Prefill law" below).
* ``gbox`` / ``gtile``  the split tile gates: box words over many threadgroups, then tiles; the
           ``gate`` kernel's masks bit for bit.

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

# ---------------------------------------------------------------------------
# Prefill law (``gup`` / ``dnp``): dequantized tiles, 64-row x 64-column threadgroups
# ---------------------------------------------------------------------------
#
# The lane law above is a small-M law: every 16-row fragment pays a per-group epilogue and the
# group sums of its inputs, and a 32-column threadgroup re-reads the activations once per 32
# outputs.  At prefill that made the mpp kernels 3-10x slower than stock MLX's NAX qmm.  The
# prefill law is the stock shape: a threadgroup of 4 simdgroups owns a 64-row x 64-column tile;
# per K step it dequantizes its weight tile once into threadgroup memory as
# ``XT(fma(s, q, b))`` and accumulates ``x . w`` in fp32 (mpp: one MPP ``matmul2d``
# multiply-accumulate over the tile; simd: 8x8 ``simdgroup_matrix`` fp32 MMAs).
#
# gate/up: one threadgroup per (chunk, 64-row work tile); its 64 columns are the chunk's 32 gate
# rows and the same 32 up rows, so SwiGLU pairs them in the epilogue.  down: one threadgroup per
# (64 output columns, work tile); K steps are whole original groups, visited in index order, and
# a group with no open owner for any of the tile's rows is skipped (unread).
#
# Exactness within the law: a skipped down group has all-zero inputs, so its products are exact
# zeros and the accumulator is unchanged (an exact zero's sign aside); a gate/up output column
# depends only on its own weight row, whatever chunk or position it sits in; garbage in pad rows
# or pad columns of a threadgroup tile reaches only outputs that are never written.  So packed ==
# unpacked and skipped == masked bit for bit, as under the lane law.  The two laws differ from
# each other (and mpp from simd): a call picks one by its row count only (ops.PREFILL_MIN_ROWS).

PF_BM = 64                         # rows per prefill work tile
PF_BN = 64                         # columns per prefill threadgroup (gate/up: 32 gate + 32 up)
PF_THREADS = 128                   # 4 simdgroups
PF_BK = 64                         # mpp gate/up K step (needs K % 64 == 0)
PF_SIMD_BK = 32                    # simd K step (fp32 staging)
LAW_PREFILL = {"mpp": "amoeba-blocks-mpp-prefill-deq-v1", "simd": "amoeba-blocks-simd-prefill-deq-v1"}

# Dequantize CNT consecutive codes of one weight row (``SRC`` bytes, ``SROW`` its scale/bias row,
# starting at input K0) into ``DST`` (threadgroup, element type DT) as DT(XT(fma(s, q, b))).
_PF_DEQ = r"""
    for (int v0 = 0; v0 < @@CNT@@; v0 += 8) {
      const int k = @@K0@@ + v0;
      const int g = k / GS;
      const float s = @@S@@, b = @@B@@;
      float q[8];
      if (BITS == 4) {
        const uint w = *(const device uint*)(@@SRC@@ + k / 2);
        for (int v = 0; v < 8; v++) q[v] = float((w >> (4 * v)) & 15u);
      } else {
        const uint2 w = *(const device uint2*)(@@SRC@@ + k);
        for (int v = 0; v < 4; v++) { q[v] = float((w.x >> (8 * v)) & 255u); q[4 + v] = float((w.y >> (8 * v)) & 255u); }
      }
      @@STORE8@@
    }
"""


# One 8-value store per dequantized group of codes (16-byte aligned: rows are padded to 16-byte
# multiples).  XT: one vec<XT, 8>; float staging: two float4.
_STORE8 = {
    "XT": "{ vec<XT, 8> o; for (int v = 0; v < 8; v++) o[v] = static_cast<XT>(fma(s, q[v], b));\n"
          "        *(threadgroup vec<XT, 8>*)(@@DST@@ + v0) = o; }",
    "float": "{ float4 o0, o1; for (int v = 0; v < 4; v++) { o0[v] = float(static_cast<XT>(fma(s, q[v], b)));"
             " o1[v] = float(static_cast<XT>(fma(s, q[4 + v], b))); }\n"
             "        *(threadgroup float4*)(@@DST@@ + v0) = o0; *(threadgroup float4*)(@@DST@@ + v0 + 4) = o1; }",
}


def _pf_deq(cnt: str, k0: str, s: str, b: str, src: str, dst: str, dt: str) -> str:
    return _subst(_PF_DEQ, {"@@CNT@@": cnt, "@@K0@@": k0, "@@S@@": s, "@@B@@": b, "@@SRC@@": src,
                            "@@STORE8@@": _STORE8[dt].replace("@@DST@@", dst)})


# gate/up prologue values shared by both backends: the B row this thread loads (rows 0-31 gate,
# 32-63 up, of the chunk), and the chunk geometry (from _PROLOGUE).
_GUP_LOADER = r"""
  constexpr int KG = K / GS;
  constexpr int64_t ROW_BYTES = K * BITS / 8;
  const int brow = int(tid) >> 1, bh = int(tid) & 1;
  const int bn = brow & 31;
  const bool bup = brow >= 32;
  const bool bok = bn < len;
  const int64_t wrow = (int64_t)e * F + off + (bok ? bn : 0);
  const device uchar* wsrc = (const device uchar*)(bup ? WU : WG) + wrow * ROW_BYTES;
"""

# The SwiGLU epilogue from a [64 rows][64 cols] staging tile ST_ (cols 0-31 gate, 32-63 up).
_GUP_EPILOGUE = r"""
  for (int i = int(tid); i < nrows * len; i += PF_THREADS_) {
    const int m = i / len, n = i - m * len;
    float v = 0.0f;
    if (((open >> m) & 1ul) != 0)
      v = bs_silu_mul(float(static_cast<XT>(@@ST@@[m * 64 + n])), float(static_cast<XT>(@@ST@@[m * 64 + 32 + n])));
    A[(int64_t)(row0 + m) * F + perm[n]] = static_cast<XT>(v);
  }
"""

_GUP_MPP = r"""
  constexpr int PF_THREADS_ = 128;
@@PROLOGUE@@
@@LOADER@@
  constexpr int BK = 64;
  constexpr int BKP = BK + 8;          // padded row (144 B): spreads the tile's rows over the banks
  constexpr int NK = K / BK;
  alignas(16) threadgroup XT wb[64 * BKP];   // [n][k]: rows 0-31 gate, 32-63 up; then the [m][n] result
@@A_DECL@@
  tensor<threadgroup XT, dextents<int32_t, 2>, tensor_inline> tB(wb, dextents<int32_t, 2>(BK, 64),
                                                                 array<int32_t, 2>({1, BKP}));
  constexpr auto desc = matmul2d_descriptor(64, 64, BK, false, true, false,
                                            matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto C = op.template get_destination_cooperative_tensor<decltype(a0), decltype(tB), float>();
  for (uint16_t i = 0; i < C.get_capacity(); ++i) if (C.is_valid_element(i)) C[i] = 0.0f;
  for (int kb = 0; kb < NK; kb++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (bok) {
@@DEQ@@
    }
@@A_STAGE@@
    threadgroup_barrier(mem_flags::mem_threadgroup);
@@A_RUN@@
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint16_t i = 0; i < C.get_capacity(); ++i) {
    if (!C.is_valid_element(i)) continue;
    const auto idx = C.get_multidimensional_index(i);
    wb[int(idx[1]) * 64 + int(idx[0])] = static_cast<XT>(C[i]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
@@EPILOGUE@@
"""

_GUP_MPP_A = {
    "dense": {
        "@@A_DECL@@": "  tensor<device XT, dextents<int32_t, 2>, tensor_inline> tA(\n"
                      "      (device XT*)X + (int64_t)row0 * K, dextents<int32_t, 2>(K, nrows));\n"
                      "  auto a0 = tA.slice(0, 0);\n",
        "@@A_STAGE@@": "",
        "@@A_RUN@@": "    auto a = tA.slice(kb * BK, 0);\n    op.run(a, tB, C);\n",
    },
    # Routed rows are staged through RMAP in threadgroup memory (no gathered device copy); pad rows
    # (nrows..63) keep stale values, which reach only rows that are never written.
    "rowmap": {
        "@@A_DECL@@": "  alignas(16) threadgroup XT xa[64 * BKP];\n"
                      "  tensor<threadgroup XT, dextents<int32_t, 2>, tensor_inline> tA(xa, dextents<int32_t, 2>(BK, 64),\n"
                      "                                                                 array<int32_t, 2>({1, BKP}));\n"
                      "  auto a0 = tA;\n",
        "@@A_STAGE@@": "    if (brow < nrows) {\n"
                       "      const device XT* xp = (const device XT*)X + (int64_t)RMAP[row0 + brow] * K + kb * BK + bh * (BK / 2);\n"
                       "      for (int i = 0; i < BK / 2; i += 8)\n"
                       "        *(threadgroup vec<XT, 8>*)(xa + brow * BKP + bh * (BK / 2) + i) = *(const device vec<XT, 8>*)(xp + i);\n"
                       "    }\n",
        "@@A_RUN@@": "    op.run(tA, tB, C);\n",
    },
}

# simd: fp32 staging (A and B, 32-wide K steps), 2x2 simdgroups of 32x32 (4x4 8x8 fragments).
_PF_SIMD_MMA = r"""
    {
      const int sm = int(sg) >> 1, sn = int(sg) & 1;
      for (int kk = 0; kk < 32; kk += 8) {
        simdgroup_matrix<float, 8, 8> af[4], bf[4];
        for (int i = 0; i < 4; i++) simdgroup_load(af[i], xs + (sm * 32 + i * 8) * SP + kk, SP);
        for (int j = 0; j < 4; j++) simdgroup_load(bf[j], ws + (sn * 32 + j * 8) * SP + kk, SP, ulong2(0, 0), true);
        for (int i = 0; i < 4; i++)
          for (int j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
      }
    }
"""

_PF_SIMD_STORE = r"""
  threadgroup_barrier(mem_flags::mem_threadgroup);
  {
    const int sm = int(sg) >> 1, sn = int(sg) & 1;
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++) simdgroup_store(acc[i][j], stage + (sm * 32 + i * 8) * 64 + sn * 32 + j * 8, 64);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
"""

_PF_SIMD_DECL = r"""
  const ushort sg = simdgroup_index_in_threadgroup;
  constexpr int SP = 36;               // padded fp32 row of a 32-wide K step (144 B)
  alignas(16) threadgroup float stage[2 * 64 * SP];   // A [64][SP] and B [64][SP] per K step, then [64][64]
  threadgroup float* xs = stage;
  threadgroup float* ws = stage + 64 * SP;
  simdgroup_matrix<float, 8, 8> acc[4][4];
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
"""

_GUP_SIMD = r"""
  constexpr int PF_THREADS_ = 128;
@@PROLOGUE@@
@@LOADER@@
  constexpr int NK = K / 32;
@@DECL@@
  for (int kb = 0; kb < NK; kb++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (bok) {
@@DEQ@@
    }
    if (brow < nrows) {
      const device XT* xp = (const device XT*)X + (int64_t)(@@TOKM@@) * K + kb * 32 + bh * 16;
      for (int i = 0; i < 16; i += 4)
        *(threadgroup float4*)(xs + brow * SP + bh * 16 + i) = float4(float(xp[i]), float(xp[i + 1]), float(xp[i + 2]), float(xp[i + 3]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
@@MMA@@
  }
@@STORE@@
@@EPILOGUE@@
"""

# down, prefill law: shared head (work tile, group-open table, loader row).
_DNP_HEAD = r"""
  const uint tid = thread_index_in_threadgroup;
  constexpr int G = F / GS;
  constexpr int64_t ROW_BYTES = F * BITS / 8;
  const int n0 = int(threadgroup_position_in_grid.x) * 64;
  const int u = int(threadgroup_position_in_grid.y);
  const int row0 = WTL[u * 4 + 0], nrows = WTL[u * 4 + 1], e = WTL[u * 4 + 2];
  if (nrows <= 0) return;
@@GROUP_OPEN@@
  const int brow = int(tid) >> 1, bh = int(tid) & 1;
  const bool bok = n0 + brow < N;
  const int64_t wrow = (int64_t)e * N + (bok ? n0 + brow : 0);
  const device uchar* wsrc = (const device uchar*)W + wrow * ROW_BYTES;
  uint opened = 0;
"""

_DNP_COUNT = r"""
  if (COUNT && tid == 0)
    atomic_fetch_add_explicit((device atomic_uint*)CTR + 2, opened, memory_order_relaxed);
"""

_DNP_MPP = r"""
@@HEAD@@
  constexpr int GSP = GS + 8;          // padded row: spreads the tile's rows over the banks
  alignas(16) threadgroup XT wb[64 * GSP];   // [n][k] of one original group
  tensor<device XT, dextents<int32_t, 2>, tensor_inline> tA(
      (device XT*)A + (int64_t)row0 * F, dextents<int32_t, 2>(F, nrows));
  auto a0 = tA.slice(0, 0);
  tensor<threadgroup XT, dextents<int32_t, 2>, tensor_inline> tB(wb, dextents<int32_t, 2>(GS, 64),
                                                                 array<int32_t, 2>({1, GSP}));
  constexpr auto desc = matmul2d_descriptor(64, 64, GS, false, true, false,
                                            matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto C = op.template get_destination_cooperative_tensor<decltype(a0), decltype(tB), float>();
  for (uint16_t i = 0; i < C.get_capacity(); ++i) if (C.is_valid_element(i)) C[i] = 0.0f;
  for (int g = 0; g < G; g++) {
    if (!gopen[g]) continue;               // every input of the group is zero: skipped, unread
    opened++;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (bok) {
@@DEQ@@
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto a = tA.slice(g * GS, 0);
    op.run(a, tB, C);
  }
@@COUNT@@
  for (uint16_t i = 0; i < C.get_capacity(); ++i) {
    if (!C.is_valid_element(i)) continue;
    const auto idx = C.get_multidimensional_index(i);
    const int m = int(idx[1]), n = n0 + int(idx[0]);
    if (m < nrows && n < N) {
      const int64_t dest = @@DEST@@;
      Y[dest * N + n] = static_cast<XT>(C[i]);
    }
  }
"""

_DNP_SIMD = r"""
@@HEAD@@
@@DECL@@
  for (int g = 0; g < G; g++) {
    if (!gopen[g]) continue;               // every input of the group is zero: skipped, unread
    opened++;
    for (int kb = g * GS; kb < (g + 1) * GS; kb += 32) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (bok) {
@@DEQ@@
      }
      if (brow < nrows) {
        const device XT* ap = (const device XT*)A + (int64_t)(row0 + brow) * F + kb + bh * 16;
        for (int i = 0; i < 16; i += 4)
          *(threadgroup float4*)(xs + brow * SP + bh * 16 + i) = float4(float(ap[i]), float(ap[i + 1]), float(ap[i + 2]), float(ap[i + 3]));
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
@@MMA@@
    }
  }
@@COUNT@@
@@STORE@@
  for (int i = int(tid); i < nrows * 64; i += 128) {
    const int m = i / 64, n = n0 + (i - m * 64);
    if (n < N) {
      const int64_t dest = @@DEST@@;
      Y[dest * N + n] = static_cast<XT>(stage[i]);
    }
  }
"""


def _prefill_source(kind: str, backend: str, variant: str, bits: int) -> str:
    tok = "RMAP[row0 + m]" if variant == "rowmap" else "row0 + m"
    if kind == "gup":
        prologue = _subst(_PROLOGUE, {"@@TOK@@": tok, "@@THREADS@@": "128"})
        s = "(bup ? float(SU[wrow * KG + g]) : float(SG[wrow * KG + g]))"
        b = "(bup ? float(BU[wrow * KG + g]) : float(BG[wrow * KG + g]))"
        if backend == "mpp":
            deq = _pf_deq("BK / 2", "kb * BK + bh * (BK / 2)", s, b, "wsrc", "(wb + brow * BKP + bh * (BK / 2))", "XT")
            return _subst(_GUP_MPP, {"@@PROLOGUE@@": prologue, "@@LOADER@@": _GUP_LOADER, "@@DEQ@@": deq,
                                     **_GUP_MPP_A[variant],
                                     "@@EPILOGUE@@": _GUP_EPILOGUE.replace("@@ST@@", "wb")})
        deq = _pf_deq("16", "kb * 32 + bh * 16", s, b, "wsrc", "(ws + brow * SP + bh * 16)", "float")
        return _subst(_GUP_SIMD, {"@@PROLOGUE@@": prologue, "@@LOADER@@": _GUP_LOADER, "@@DECL@@": _PF_SIMD_DECL,
                                  "@@DEQ@@": deq, "@@MMA@@": _PF_SIMD_MMA, "@@STORE@@": _PF_SIMD_STORE,
                                  "@@TOKM@@": "RMAP[row0 + brow]" if variant == "rowmap" else "row0 + brow",
                                  "@@EPILOGUE@@": _GUP_EPILOGUE.replace("@@ST@@", "stage")})
    if kind == "dnp":
        gopen = _subst(_GROUP_OPEN, {"@@TOK@@": tok, "@@THREADS@@": "128"})
        head = _subst(_DNP_HEAD, {"@@GROUP_OPEN@@": gopen})
        dest = "OUTMAP ? (int64_t)OMAP[row0 + m] : (int64_t)(row0 + m)"
        s, b = "float(S[wrow * G + g])", "float(B[wrow * G + g])"
        if backend == "mpp":
            deq = _pf_deq("GS / 2", "g * GS + bh * (GS / 2)", s, b, "wsrc", "(wb + brow * GSP + bh * (GS / 2))", "XT")
            return _subst(_DNP_MPP, {"@@HEAD@@": head, "@@DEQ@@": deq, "@@COUNT@@": _DNP_COUNT, "@@DEST@@": dest})
        deq = _pf_deq("16", "kb + bh * 16", s, b, "wsrc", "(ws + brow * SP + bh * 16)", "float")
        return _subst(_DNP_SIMD, {"@@HEAD@@": head, "@@DECL@@": _PF_SIMD_DECL, "@@DEQ@@": deq, "@@MMA@@": _PF_SIMD_MMA,
                                  "@@COUNT@@": _DNP_COUNT, "@@STORE@@": _PF_SIMD_STORE, "@@DEST@@": dest})
    raise ValueError(f"unknown prefill kernel kind {kind!r}")


# ---------------------------------------------------------------------------
# Split tile gates: box bits over many threadgroups, then tiles (same masks as ``gate``)
# ---------------------------------------------------------------------------
#
# The fused ``gate`` kernel runs one threadgroup per token, so a decode step's probe matvec
# (boxes x d fp32) runs on one GPU core.  ``gbox`` gives each threadgroup one 32-box word for up
# to GATE_TT tokens: each simdgroup takes 4 boxes, each box keeps one fp32 accumulator per token
# with the fused kernel's exact sequence (lane-strided fma over d in index order, one simd_sum,
# plus the bias), so every logit -- and every bit -- is the fused kernel's.  The word is written
# by its one owner (no memset, no device atomics).  ``gtile`` then ORs each tile's closure bits
# against the token's words, as the fused kernel does.

GATE_TT = 8

_GBOX = r"""
  const uint tid = thread_index_in_threadgroup;
  const ushort lane = thread_index_in_simdgroup;
  const ushort sg = simdgroup_index_in_threadgroup;
  const int w = int(threadgroup_position_in_grid.x);
  const int T = MODE == 0 ? int(H_shape[0]) : int(BOX_shape[0]);
  const int t0 = int(threadgroup_position_in_grid.y) * TT;
  const int nt = min(TT, T - t0);
  threadgroup atomic_uint word[TT];
  if (tid < TT) atomic_store_explicit(&word[tid], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (MODE == 0) {
    for (int j = 0; j < 4; j++) {
      const int b = w * 32 + int(sg) * 4 + j;
      if (b >= NB) break;
      float acc[TT];
      for (int t = 0; t < TT; t++) acc[t] = 0.0f;
      for (int i = int(lane); i < D; i += 32) {
        const float pw = PW[(int64_t)b * D + i];
        for (int t = 0; t < TT; t++)
          if (t < nt) acc[t] = fma(float(H[(int64_t)(t0 + t) * D + i]), pw, acc[t]);
      }
      for (int t = 0; t < TT; t++) {
        if (t >= nt) break;
        const float v = simd_sum(acc[t]) + PB[b];
        if (lane == 0 && v > 0.0f)
          atomic_fetch_or_explicit(&word[t], 1u << (b & 31), memory_order_relaxed);
      }
    }
  } else {
    for (int i = int(tid); i < nt * 32; i += 256) {
      const int t = i >> 5, b = w * 32 + (i & 31);
      if (b < NB && BOX[(int64_t)(t0 + t) * NB + b])
        atomic_fetch_or_explicit(&word[t], 1u << (b & 31), memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (int(tid) < nt) BW[(int64_t)(t0 + int(tid)) * NW + w] = atomic_load_explicit(&word[tid], memory_order_relaxed);
"""

_GTILE = r"""
  const int i = int(thread_position_in_grid.x);       // the grid is exactly tokens x tiles threads
  const int t = i / NTILES, k = i - t * NTILES;
  bool o = ALW[k] != 0;
  for (int w = 0; w < NW && !o; w++) o = (TBITS[k * NW + w] & BW[(int64_t)t * NW + w]) != 0u;
  TOPEN[i] = o ? 1 : 0;
"""

GBOX_INPUTS = ("H", "PW", "PB", "BOX")
GBOX_OUTPUTS = ("BW",)
GTILE_INPUTS = ("BW", "TBITS", "ALW")
GTILE_OUTPUTS = ("TOPEN",)

# Buffer names (and their MLX dtypes, for the offline harness) per kernel.
GU_INPUTS = ("X", "WG", "SG", "BG", "WU", "SU", "BU", "PERM", "CPTR", "CK", "WTL", "RMAP", "TOPEN")
GU_OUTPUTS = ("A", "CTR")
DN_INPUTS = ("A", "W", "S", "B", "GPTR", "GSIG", "WTL", "RMAP", "OMAP", "TOPEN")
DN_OUTPUTS = ("Y", "CTR")
GATE_INPUTS = ("H", "PW", "PB", "BOX", "TBITS", "ALW")
GATE_OUTPUTS = ("TOPEN", "UN")


_MATMUL_KINDS = ("gu", "dn", "gup", "dnp")
KIND_BUFFERS = {"gu": (GU_INPUTS, GU_OUTPUTS), "dn": (DN_INPUTS, DN_OUTPUTS), "gup": (GU_INPUTS, GU_OUTPUTS),
                "dnp": (DN_INPUTS, DN_OUTPUTS), "gate": (GATE_INPUTS, GATE_OUTPUTS),
                "gbox": (GBOX_INPUTS, GBOX_OUTPUTS), "gtile": (GTILE_INPUTS, GTILE_OUTPUTS)}
# MLX passes ``<name>_shape`` (const constant int*) for an input the source names that way.
_SHAPE_ARGS = {"gbox": ("H", "BOX")}


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
    if kind in ("gup", "dnp"):
        return _prefill_source(kind, backend, variant, bits)
    if kind == "gbox":
        return _GBOX
    if kind == "gtile":
        return _GTILE
    raise ValueError(f"unknown kernel kind {kind!r}")


def header(kind: str, backend: str, bits: int, xt: str, st: str) -> str:
    """Type aliases (and the MPP include for mpp matmuls) prepended to a kernel."""
    head = (_MPP_HEADER if backend == "mpp" and kind in _MATMUL_KINDS else "") + _COMMON
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

    inputs, outputs = KIND_BUFFERS[kind]
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
    kind = {"gup": "gu", "dnp": "dn"}.get(kind, kind)
    if kind == "gbox":
        return {"H": x, "PW": "f32", "PB": "f32", "BOX": "u8", "BW": "u32"}
    if kind == "gtile":
        return {"BW": "u32", "TBITS": "u32", "ALW": "u8", "TOPEN": "u8"}
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
    inputs, outputs = KIND_BUFFERS[kind]
    types = buffer_types(kind, xt, st)
    args = [f"const device {_MSL_TYPES[types[n]]}* {n} [[buffer({i})]]" for i, n in enumerate(inputs)]
    args += [f"device {_MSL_TYPES[types[n]]}* {n} [[buffer({len(inputs) + i})]]" for i, n in enumerate(outputs)]
    nb = len(inputs) + len(outputs)
    args += [f"const constant int* {n}_shape [[buffer({nb + i})]]" for i, n in enumerate(_SHAPE_ARGS.get(kind, ()))]
    args += ["uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]",
             "uint3 thread_position_in_grid [[thread_position_in_grid]]",
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
    if kind == "gup":
        if backend == "mpp":
            return PF_BM * (PF_BK + 8) * xbytes * (2 if variant == "rowmap" else 1)
        return 2 * PF_BM * 36 * 4
    if kind == "dnp":
        return groups + (PF_BN * (gs + 8) * xbytes if backend == "mpp" else 2 * PF_BM * 36 * 4)
    return 0

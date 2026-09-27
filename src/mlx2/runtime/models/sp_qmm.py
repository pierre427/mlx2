# SPDX-License-Identifier: Apache-2.0
# Technique adapted from incoai/splash PR #88 (Apache-2.0, f786bed); see
# provenance/sp-qmm.json. Original kernel written for mlx2 on MLX's layout.
"""Small-M affine quantized matmul on simdgroup matrices (opt-in).

``y = x @ dequantize(w).T`` for M <= 32 rows, bf16 activations, affine 4- or
5-bit weights with group size 64, in MLX's packed layout (no repacking).

The mechanism is Splash's Apple9 Q4 decode tile: a nibble ``q`` becomes the
exact bfloat ``128 + q`` by OR-ing it into ``0x4300``, so the dequantized
operand never passes through a scale multiply. Eight weight rows by eight k
columns form the A operand of an 8x8x8 ``simdgroup_multiply_accumulate``
straight from registers; eight activation rows (zero padded) form B. The group
dot is accumulated in fp32 and the affine epilogue is applied per group:

    sum_k w x = s * (dot - 128 * sum_x) + b * sum_x

Each simdgroup loads one 64-wide quantization group of eight rows per step
(four lanes per row, 16 k each), so every weight is read once for all M
rows. The k order inside the fragment is permuted so that each lane's
activation operands are two 4-element runs it can load directly. A
threadgroup splits K across ``KS`` simdgroups and reduces the partials in a
fixed order through threadgroup memory, so the result is deterministic.

Not bitwise identical to ``mx.quantized_matmul``: the reduction order differs
and bf16 activations enter the product unrounded (stock qmv pre-scales them in
fp32). Selection is gated by ``MLX2_SP_QMM`` (default off).
"""

from __future__ import annotations

import os
from typing import Optional

import mlx.core as mx

GROUP_SIZE = 64
MAX_M = 32

_HEADER = "#include <metal_simdgroup_matrix>\n"

_SOURCE = r"""
  constexpr int G = K / 64;
  constexpr int WPR = K * BITS / 32;          // uint32 words per weight row
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const uint tgy = threadgroup_position_in_grid.y;
  const uint qid = lane >> 2;
  const uint fm = (qid & 4) | ((lane >> 1) & 3);
  const uint fn = ((qid & 2) << 1) | ((lane & 1) << 1);
  const uint c = fn >> 1;
  const uint n_base = tgy * 8 * NF;
  constexpr int GPER = (G + KS - 1) / KS;
  const int g0 = int(sg) * GPER;
  const int g1 = min(G, g0 + GPER);
  // Fragment column/row index r <-> k = 16 (r >> 1) + 4 (r & 1) + 8 (j >> 2) + (j & 3)
  // for MMA j in 0..7, so a lane's activations are runs of 4 at koff, koff + 8.
  // XL = 1: r <-> k = 16 (r >> 1) + 8 (r & 1) + j, so a lane's activations
  // are one contiguous run of 8 (one 16-byte load per row).
  const uint koff = XL ? 16 * (fm >> 1) + 8 * (fm & 1) : 16 * (fm >> 1) + 4 * (fm & 1);
  device const bfloat* xb = reinterpret_cast<device const bfloat*>(x);
  device const bfloat* sb = reinterpret_cast<device const bfloat*>(scales);
  device const bfloat* bb = reinterpret_cast<device const bfloat*>(biases);

  float2 acc[MT][NF];
  for (int mt = 0; mt < MT; ++mt)
    for (int f = 0; f < NF; ++f) acc[mt][f] = float2(0.0f);

  // Raw weight words and affine parameters for one group of every fragment:
  // 4-bit uses raw[f][0..1] (two words), 5-bit raw[f][0..4] (five halfwords).
  // The next group's loads are issued before the current group's MMAs.
  constexpr int RW = BITS == 4 ? 2 : 5;
  uint raw[NF][RW];
  float rsc[NF], rbi[NF];
  // (A statement macro rather than a lambda: lambdas need Metal 3.2.)
#define SPQ_LOAD_GROUP(GG)                                                      \
  for (int f = 0; f < NF; ++f) {                                                \
    const uint n = n_base + 8 * f + fm;                                         \
    device const uint* wr = w + ulong(n) * WPR + (GG) * (2 * BITS);             \
    if (DIAG == 2) {                                                            \
      for (int i = 0; i < RW; ++i) raw[f][i] = uint(GG) * 2654435761u + lane + i; \
    } else if (BITS == 4) {                                                     \
      const uint2 t = *reinterpret_cast<device const uint2*>(wr + 2 * c);       \
      raw[f][0] = t.x; raw[f][1] = t.y;                                         \
    } else {                                                                    \
      device const ushort* hp = reinterpret_cast<device const ushort*>(wr) + 5 * c; \
      for (int i = 0; i < RW; ++i) raw[f][i] = hp[i];                          \
    }                                                                           \
    rsc[f] = DIAG == 2 ? 1.0f : float(sb[ulong(n) * G + (GG)]);                 \
    rbi[f] = DIAG == 2 ? 0.0f : float(bb[ulong(n) * G + (GG)]);                 \
  }
  if (g0 < g1) { SPQ_LOAD_GROUP(g0) }
  for (int g = g0; g < g1; ++g) {
    uint cur[NF][RW];
    float csc[NF], cbi[NF];
    for (int f = 0; f < NF; ++f) {
      for (int i = 0; i < RW; ++i) cur[f][i] = raw[f][i];
      csc[f] = rsc[f]; cbi[f] = rbi[f];
    }
    if (PF && g + 1 < g1) { SPQ_LOAD_GROUP(g + 1) }
    // Activation operands and group sums for every M tile (shared by all
    // fragments), loaded once per group.
    vec<bfloat, 4> xa[MT][4];
    float2 sx[MT];
    const ulong kb = ulong(g) * 64 + koff;
    for (int mt = 0; mt < MT; ++mt) {
      const uint m0 = mt * 8 + fn;
      for (int r = 0; r < 2; ++r) {
        // Branchless: padded rows re-read the last valid row, then zero it.
        const uint m = m0 + r;
        const bool live = m < M;
        device const bfloat* xr = xb + ulong(live ? m : M - 1) * K + kb;
        vec<bfloat, 4> lo, hi;
        if (XL) {
          const vec<bfloat, 8> v = *reinterpret_cast<device const vec<bfloat, 8>*>(xr);
          lo = vec<bfloat, 4>(v[0], v[1], v[2], v[3]);
          hi = vec<bfloat, 4>(v[4], v[5], v[6], v[7]);
        } else {
          lo = *reinterpret_cast<device const vec<bfloat, 4>*>(xr);
          hi = *reinterpret_cast<device const vec<bfloat, 4>*>(xr + 8);
        }
        xa[mt][2 * r] = live ? lo : vec<bfloat, 4>(0);
        xa[mt][2 * r + 1] = live ? hi : vec<bfloat, 4>(0);
      }
      float s0 = 0.0f, s1 = 0.0f;
      for (int i = 0; i < 4; ++i) {
        s0 += float(xa[mt][0][i]) + float(xa[mt][1][i]);
        s1 += float(xa[mt][2][i]) + float(xa[mt][3][i]);
      }
      // Lanes sharing fn differ in lane bits 1, 2 and 4 (fm); sum the group.
      s0 += simd_shuffle_xor(s0, 2); s1 += simd_shuffle_xor(s1, 2);
      s0 += simd_shuffle_xor(s0, 4); s1 += simd_shuffle_xor(s1, 4);
      s0 += simd_shuffle_xor(s0, 16); s1 += simd_shuffle_xor(s1, 16);
      sx[mt] = float2(s0, s1);
    }
    for (int f = 0; f < NF; ++f) {
      // Eight bfloat pairs (128 + q) for MMA j = 0..7, decoded once and
      // reused for every M tile.
      uint pair[8];
      if (BITS == 4 && XL) {
        // Column fn <-> nibble j of word 0, fn + 1 <-> nibble j of word 1.
        const uint2 t = uint2(cur[f][0], cur[f][1]);
        for (int j = 0; j < 8; ++j)
          pair[j] = ((t.x >> (4 * j)) & 0xFu) | (((t.y >> (4 * j)) & 0xFu) << 16) | 0x43004300u;
      } else if (BITS == 4) {
        const uint2 t = uint2(cur[f][0], cur[f][1]);
        for (int j = 0; j < 4; ++j) {
          pair[j] = ((t.x >> (4 * j)) & 0x000F000Fu) | 0x43004300u;
          pair[j + 4] = ((t.y >> (4 * j)) & 0x000F000Fu) | 0x43004300u;
        }
      } else if (XL) {
        // 5-bit, column fn <-> value j of chunk 0, fn + 1 <-> value j of chunk 1.
        const uint h0 = cur[f][0], h1 = cur[f][1], h2 = cur[f][2], h3 = cur[f][3],
                   h4 = cur[f][4 % RW];
        const uint lo0 = h0 | (h1 << 16), hi0 = h2 & 0xFFu;
        const uint lo1 = (h2 >> 8) | (h3 << 8) | (h4 << 24), hi1 = h4 >> 8;
        for (int j = 0; j < 8; ++j) {
          const uint p = 5 * j;
          const uint a0 = p >= 32 ? (hi0 >> (p - 32)) & 31u
              : p + 5 > 32 ? ((lo0 >> p) | (hi0 << (32 - p))) & 31u : (lo0 >> p) & 31u;
          const uint a1 = p >= 32 ? (hi1 >> (p - 32)) & 31u
              : p + 5 > 32 ? ((lo1 >> p) | (hi1 << (32 - p))) & 31u : (lo1 >> p) & 31u;
          pair[j] = a0 | (a1 << 16) | 0x43004300u;
        }
      } else {
        // 5-bit: 80 bits = two 40-bit chunks of 8 values at halfword 5c.
        const uint h0 = cur[f][0], h1 = cur[f][1], h2 = cur[f][2], h3 = cur[f][3],
                   h4 = cur[f][4 % RW];
        const uint lo0 = h0 | (h1 << 16);            // chunk 0 bits 0..31
        const uint hi0 = h2 & 0xFFu;                 // chunk 0 bits 32..39
        const uint lo1 = (h2 >> 8) | (h3 << 8) | (h4 << 24);  // chunk 1 bits 0..31
        const uint hi1 = h4 >> 8;                    // chunk 1 bits 32..39
        // value v of a chunk sits at bit 5v; v = 6 straddles the 32-bit word.
        for (int i = 0; i < 4; ++i) {
          const uint pa = 5 * i, pb = 5 * i + 20;
          const uint a0 = (lo0 >> pa) & 31u;
          const uint a1 = (lo1 >> pa) & 31u;
          const uint b0 = pb >= 32 ? (hi0 >> (pb - 32)) & 31u
              : pb + 5 > 32 ? ((lo0 >> pb) | (hi0 << (32 - pb))) & 31u : (lo0 >> pb) & 31u;
          const uint b1 = pb >= 32 ? (hi1 >> (pb - 32)) & 31u
              : pb + 5 > 32 ? ((lo1 >> pb) | (hi1 << (32 - pb))) & 31u : (lo1 >> pb) & 31u;
          pair[i] = a0 | (b0 << 16) | 0x43004300u;
          pair[i + 4] = a1 | (b1 << 16) | 0x43004300u;
        }
      }
      const float sc = csc[f];
      const float bi = cbi[f];
      for (int mt = 0; mt < MT; ++mt) {
        simdgroup_matrix<bfloat, 8, 8> A, B;
        simdgroup_matrix<float, 8, 8> D;
        D.thread_elements()[0] = 0.0f;
        D.thread_elements()[1] = 0.0f;
        for (int j = 0; j < 8; ++j) {
          const bfloat2 ap = as_type<bfloat2>(pair[j]);
          A.thread_elements()[0] = ap[0];
          A.thread_elements()[1] = ap[1];
          const int h = j >> 2, jj = j & 3;
          B.thread_elements()[0] = xa[mt][h][jj];
          B.thread_elements()[1] = xa[mt][2 + h][jj];
          if (DIAG == 1) {
            // Diagnostic: keep the loads and decode live, skip the MMA.
            D.thread_elements()[0] += float(pair[j] & 0xFu) * float(B.thread_elements()[0]);
          } else {
            simdgroup_multiply_accumulate(D, A, B, D);
          }
        }
        const float2 dot = float2(D.thread_elements()[0], D.thread_elements()[1]);
        acc[mt][f] = fma(float2(sc), dot - 128.0f * sx[mt], acc[mt][f]);
        acc[mt][f] = fma(float2(bi), sx[mt], acc[mt][f]);
      }
    }
    if (!PF && g + 1 < g1) { SPQ_LOAD_GROUP(g + 1) }
  }

  threadgroup float2 red[KS > 1 ? KS * MT * NF * 32 : 1];
  if (KS > 1) {
    for (int mt = 0; mt < MT; ++mt)
      for (int f = 0; f < NF; ++f)
        red[((sg * MT + mt) * NF + f) * 32 + lane] = acc[mt][f];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Simdgroup s finalizes the (mt, f) tiles t with t % KS == s, summing the
    // K partials in fixed order.
    for (int t = int(sg); t < MT * NF; t += KS) {
      const int mt = t / NF, f = t % NF;
      float2 v = float2(0.0f);
      for (int s = 0; s < KS; ++s) v += red[((s * MT + mt) * NF + f) * 32 + lane];
      const uint n = n_base + 8 * f + fm;
      const uint m0 = mt * 8 + fn;
      if (m0 < M) out[ulong(m0) * N + n] = bfloat(v.x);
      if (m0 + 1 < M) out[ulong(m0 + 1) * N + n] = bfloat(v.y);
    }
  } else {
    for (int mt = 0; mt < MT; ++mt)
      for (int f = 0; f < NF; ++f) {
        const uint n = n_base + 8 * f + fm;
        const uint m0 = mt * 8 + fn;
        if (m0 < M) out[ulong(m0) * N + n] = bfloat(acc[mt][f].x);
        if (m0 + 1 < M) out[ulong(m0 + 1) * N + n] = bfloat(acc[mt][f].y);
      }
  }
"""

_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="mlx2_sp_qmm",
            input_names=["x", "w", "scales", "biases"],
            output_names=["out"],
            header=_HEADER,
            source=_SOURCE,
        )
    return _KERNEL


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def tiles(M: int, N: int, K: int):
    """(NF, KS): 8-row fragments per simdgroup and K-split simdgroups per threadgroup."""
    nf = _env_int("MLX2_SP_QMM_NF", 2)
    ks = _env_int("MLX2_SP_QMM_KS", 8)
    while nf > 1 and N % (8 * nf):
        nf //= 2
    groups = K // GROUP_SIZE
    ks = max(1, min(ks, groups))
    return nf, ks


def supports(M: int, N: int, K: int, bits: int, group_size: int,
             dtype=mx.bfloat16, mode: str = "affine") -> bool:
    return (mode == "affine" and bits in (4, 5) and group_size == GROUP_SIZE
            and dtype == mx.bfloat16 and 1 <= M <= MAX_M
            and K % GROUP_SIZE == 0 and N % 8 == 0)


def qmm(x: mx.array, w: mx.array, scales: mx.array, biases: mx.array,
        group_size: int = GROUP_SIZE, bits: int = 4,
        nf: Optional[int] = None, ks: Optional[int] = None,
        pf: Optional[int] = None, diag: int = 0,
        xl: Optional[int] = None) -> mx.array:
    """x [..., K] (bf16) @ dequantize(w, scales, biases).T -> [..., N] (bf16)."""
    K = x.shape[-1]
    N = w.shape[0]
    lead = x.shape[:-1]
    M = 1
    for d in lead:
        M *= d
    if not supports(M, N, K, bits, group_size, x.dtype):
        raise ValueError(f"sp_qmm does not support M={M} N={N} K={K} bits={bits} "
                         f"gs={group_size} dtype={x.dtype}")
    if scales.dtype != mx.bfloat16 or biases.dtype != mx.bfloat16:
        raise ValueError("sp_qmm reads bf16 scales and biases, got "
                         f"{scales.dtype} / {biases.dtype}")
    d_nf, d_ks = tiles(M, N, K)
    nf = nf or d_nf
    ks = ks or d_ks
    pf = _env_int("MLX2_SP_QMM_PF", 1) if pf is None else pf
    xl = _env_int("MLX2_SP_QMM_XL", 1) if xl is None else xl
    while nf > 1 and N % (8 * nf):
        nf //= 2
    ks = max(1, min(ks, K // GROUP_SIZE))
    mt = -(-M // 8)
    # The K-split reduction buffer (KS * MT * NF * 32 float2) must fit the
    # 32 KiB threadgroup memory.
    while ks > 1 and ks * mt * nf * 32 * 8 > 32768:
        ks //= 2
    x2 = x.reshape(M, K)
    (out,) = _kernel()(
        inputs=[x2, w, scales, biases],
        template=[("BITS", bits), ("NF", nf), ("KS", ks), ("MT", mt), ("PF", pf),
                  ("DIAG", diag), ("XL", xl),
                  ("K", K), ("N", N), ("M", M)],
        grid=(32 * ks, N // (8 * nf), 1),
        threadgroup=(32 * ks, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[mx.bfloat16],
    )
    return out.reshape(*lead, N)


# --------------------------------------------------------------------------
# Instance-scoped routing (no global monkeypatch)
# --------------------------------------------------------------------------

STATS = {"routed": 0, "stock": 0}


def prefer(M: int, N: int, K: int, bits: int) -> bool:
    """Measured routing policy (M5 Max, mlx 39400a0d4, weights from DRAM,
    ops chained so they cannot overlap as in a real forward; see
    qualification/runs/sp-qmm-20260925/serial1.jsonl).

    The simdgroup tile costs a flat ~1.4x of the stock one-row time for any
    M <= 8. Stock qmv / qmv_wide stay at or near the bandwidth roofline to
    M = 6, so the tile only wins where stock leaves it:
    - 4-bit M = 8..11, where MLX switches to its NAX small-M kernel (about
      2.3x the one-row time serially; its 12..16 tiles are faster than ours);
    - 5-bit M = 8..16 (no NAX path: qmv_wide in two tiles, then split-K);
    - vocabulary-sized outputs at M = 8..11 (same NAX switch).
    Tiny outputs (the 48-row GDN a/b projections) stay on stock.
    """
    if N < 256 or M < 8 or M > 16:
        return False
    if bits == 5 and N < 65536:
        return True
    return M <= 11


class _SpQuantizedLinear:
    """Mixin swapped onto selected ``nn.QuantizedLinear`` instances."""

    _sp_min_m = 2
    _sp_max_m = 16
    _sp_policy = True

    def __call__(self, x):
        K = x.shape[-1]
        M = x.size // K if K else 0
        N = self["weight"].shape[0]
        # The kernel reads scales and biases as bf16.  Another opt-in may
        # have re-typed them (fp32_head stores the vocab head's in float32),
        # so check at call time and leave any other layout to stock.
        if (self._sp_min_m <= M <= self._sp_max_m and getattr(self, "mode", "affine") == "affine"
                and self["scales"].dtype == mx.bfloat16
                and self["biases"].dtype == mx.bfloat16
                and supports(M, N, K, self.bits, self.group_size, x.dtype)
                and (not self._sp_policy or prefer(M, N, K, self.bits))):
            STATS["routed"] += 1
            y = qmm(x, self["weight"], self["scales"], self["biases"],
                    group_size=self.group_size, bits=self.bits)
            if "bias" in self:
                y = y + self["bias"]
            return y
        STATS["stock"] += 1
        return super().__call__(x)


_SUBCLASSES: dict = {}


def _subclass(cls):
    sub = _SUBCLASSES.get(cls)
    if sub is None:
        sub = type("Sp" + cls.__name__, (_SpQuantizedLinear, cls), {})
        _SUBCLASSES[cls] = sub
    return sub


def apply(model, *, min_m: int = 2, max_m: int = 16, policy: bool = True):
    """Route eligible ``QuantizedLinear`` instances of ``model`` through sp_qmm
    for ``min_m <= M <= max_m``. Returns a handle for :func:`remove`."""
    import mlx.nn as nn

    handle = []
    for _, module in model.named_modules():
        if type(module) is nn.QuantizedLinear and "biases" in module \
                and module.bits in (4, 5) and module.group_size == GROUP_SIZE:
            sub = _subclass(type(module))
            handle.append((module, type(module)))
            module.__class__ = sub
            module._sp_min_m = min_m
            module._sp_max_m = max_m
            module._sp_policy = policy
    return handle


def remove(handle) -> None:
    for module, cls in handle:
        module.__class__ = cls

# SPDX-License-Identifier: Apache-2.0
# Mined from jundot/omlx d6b2b92b (Apache-2.0), omlx/patches/m5_gather_qmm_nax.py
# as of #3995 + #4022 + #4029 + follow-up 195a69e5.  Motivated by
# ddalcu/mlx-serve#638 (MIT; nothing copied).  See docs/PROVENANCE.md,
# provenance/omlx-3995-4022-4029-nax-gather.json and its NOTICE.
"""Tensor-unit (NAX) sorted ``gather_qmm`` for M5 hosts, compiled at runtime.

mlx2 port of oMLX's ``m5_gather_qmm_nax``.  MLX (0.32.2.dev20260919+39400a0d4
in the mlx2 venv) sends every natively sorted quantized expert gather with
``rows >= 16`` and ``rows / experts >= 4`` to the ``*_gather_qmm_rhs_nax``
row-block kernel on M5: a threadgroup per 64 sorted rows that re-runs the K
loop for every expert present in the block with only that expert's rows
active.  This module runs the same product with segmented tile scheduling
instead (one expert per 64-128-row tile, a one-threadgroup pre-pass cuts
the tiles), as ``mx.fast.metal_kernel`` kernels built on the NAX tile
primitives of the installed MLX (``steel/gemm/nax.h`` read from the
package's ``include`` directory).

Every configuration dequantizes exactly like MLX and issues the same
16x32x16 tensor ops in the same K order, so each output element is
bit-identical to MLX's sorted kernel.  Each kernel instantiation proves it
once on a nonzero canary against the stock op (bitwise) before it is used,
and a failing or unbuildable instantiation keeps the stock path.

``sorted_gather_qmm_swiglu`` is the fused ``[gate; up]`` projection with
``silu(gate) * up`` in its epilogue (bit-identical to the split + MLX's
compiled SwiGLU; writes ``[M, n]`` instead of ``[M, 2n]``).  With
``row_map`` it reads the sorted rows from the token rows in place instead
of from the ``x[order // k]`` copy.

Changes vs the oMLX source: oMLX's env switches and the ``m5_gather_qmm``
reroute wrapper are not carried (the kernel keeps the GLM-5.3 clamped
SwiGLU epilogue template, but mlx2 wires only the plain SwiGLU);
mlx2 calls ``sorted_gather_qmm`` / ``sorted_gather_qmm_swiglu`` explicitly
from ``switch_layers`` / ``qwen3_next`` behind ``MLX2_MOE_NAX_GATHER``
(default off), and the host gate is mlx2's (Metal available, an M5 device).
``MLX2_MOE_NAX_GATHER_PLAN=sched,bm,bk,gx,pad`` pins a configuration
(testing only).

Supported: ``transpose=True``, rhs-indices only, ``x`` of shape
``[M, 1, K]`` with a flat sorted ``uint32`` index of length ``M`` (each
expert's rows one contiguous run), bf16/fp16 activations, N a multiple of
32, affine 2/3/4/5/6/8-bit with group 32/64/128 (scales and biases in the
activation dtype; mlx2: 2/3/5/6-bit unpack mlx's packed bit stream, 3/5-bit
only on loader splits of 32-value runs, see ``geometry_ok``) and MXFP4
(group 32); the route admits the affine widths in ``AFFINE_BITS``
(``MLX2_MOE_NAX_GATHER_BITS``, default 4/6/8); the epilogue additionally needs
``2 * n % 64 == 0``, the row map a uint32 ``[M]`` map and fewer than 2**32
token-row elements.  Anything else returns None and the caller keeps the
stock path.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial
from pathlib import Path
from typing import NamedTuple, Optional

import mlx.core as mx
import mlx.nn as nn

from .import_env import snapshot as _import_env_snapshot

_import_env_snapshot(__name__)

logger = logging.getLogger(__name__)

_ENV_PLAN = "MLX2_MOE_NAX_GATHER_PLAN"

# Output tile width and column simdgroups (fixed; the Metal source assumes
# them). Tile heights are multiples of 32 rows (one row simdgroup each).
_BN = 64
_WN = 2
_TILE_ROWS = (64, 96, 128)

# Largest expert count the one-threadgroup pre-pass handles (its run
# bounds live in threadgroup memory).
_MAX_EXPERTS = 2048

# Row tiles per grid-x group in the tile-on-x layout.
_GX = 32

_MLX_UTILS_HEADERS = (
    "mlx/backend/metal/kernels/utils.h",
    "mlx/backend/metal/kernels/bf16.h",
    "mlx/backend/metal/kernels/bf16_math.h",
    "mlx/backend/metal/kernels/complex.h",
    "mlx/backend/metal/kernels/defines.h",
    "mlx/backend/metal/kernels/logging.h",
)


# mlx headers the matmul kernels build on: the NAX tile primitives and the
# fp4/fp8 element types.
_MLX_MM_HEADERS = (
    "mlx/backend/metal/kernels/steel/gemm/nax.h",
    "mlx/backend/metal/kernels/fp4.h",
    "mlx/backend/metal/kernels/fp8.h",
)


def _read_mlx_headers(paths: tuple[str, ...]) -> Optional[str]:
    """Flatten mlx kernel headers from the installed package.

    ``mx.fast.metal_kernel`` already prepends mlx's ``utils.h`` preamble, so
    it (and what it includes) is skipped; quoted mlx includes are inlined
    once and ``#pragma once`` dropped, system includes are kept.
    """
    root = Path(mx.__file__).parent / "include"
    if not root.is_dir():
        return None
    seen = {root / p for p in _MLX_UTILS_HEADERS}

    def expand(rel: str) -> str:
        path = root / rel
        if path in seen:
            return ""
        seen.add(path)
        lines = []
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith('#include "mlx/') and stripped.endswith('"'):
                lines.append(expand(stripped[len('#include "') : -1]))
            elif stripped != "#pragma once":
                lines.append(line)
        return "\n".join(lines)

    try:
        return "\n".join(expand(p) for p in paths)
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Tile pre-pass
# ---------------------------------------------------------------------------

_SCAN_HEADER = """
using namespace metal;

// Cuts the sorted rows into (row_start, expert, rows, 0) tiles of at most
// BM rows of one expert, expert-major (the tile order of mlx's segmented
// gather_qmm). One threadgroup: the run bounds of every expert are found in
// parallel over the rows, then a threadgroup scan of the per-expert tile
// counts gives each expert's first tile. At most max_tiles tiles are
// written (a guard for unsorted input, which the contract excludes).
template <int BM>
METAL_FUNC void omlx_gqmm_tile_scan(
    const device uint32_t* idx,
    const constant int* params,
    device uint32_t* tiles,
    device uint32_t* tile_count,
    threadgroup uint32_t* run_start,
    threadgroup uint32_t* run_end,
    threadgroup uint32_t* simd_tot,
    const uint lid,
    const uint tg_size,
    const uint sg,
    const uint lane) {
  const int M = params[0];
  const int E = params[1];
  const uint32_t max_tiles = uint32_t(params[2]);
  for (int e = int(lid); e < E; e += int(tg_size)) {
    run_start[e] = 0;
    run_end[e] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Each thread walks 4 consecutive rows per step, with their neighbours
  // (0xffffffff past either end, never a valid expert).
  for (int g0 = 4 * int(lid); g0 < M; g0 += 4 * int(tg_size)) {
    const int cnt = min(4, M - g0);
    uint32_t v[6];
    v[0] = g0 > 0 ? idx[g0 - 1] : 0xffffffffu;
    for (int j = 0; j < 4; j++) {
      v[j + 1] = j < cnt ? idx[g0 + j] : 0xffffffffu;
    }
    v[5] = g0 + 4 < M ? idx[g0 + 4] : 0xffffffffu;
    for (int j = 0; j < cnt; j++) {
      const uint32_t e = v[j + 1];
      if (e < uint32_t(E)) {
        if (v[j] != e) {
          run_start[e] = uint32_t(g0 + j);
        }
        if (v[j + 2] != e) {
          run_end[e] = uint32_t(g0 + j + 1);
        }
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint n_simd = (tg_size + 31) / 32;
  uint32_t running = 0;
  for (int base = 0; base < E; base += int(tg_size)) {
    const int e = base + int(lid);
    uint32_t start = 0;
    uint32_t cnt = 0;
    if (e < E) {
      start = run_start[e];
      const uint32_t end = run_end[e];
      cnt = end > start ? end - start : 0;
    }
    const uint32_t nt = (cnt + BM - 1) / BM;
    const uint32_t local = simd_prefix_exclusive_sum(nt);
    const uint32_t stot = simd_sum(nt);
    if (lane == 0) {
      simd_tot[sg] = stot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint32_t prefix = 0;
    uint32_t total = 0;
    for (uint s = 0; s < n_simd; s++) {
      const uint32_t v = simd_tot[s];
      prefix += (s < sg) ? v : 0;
      total += v;
    }
    const uint32_t off = running + prefix + local;
    for (uint32_t j = 0; j < nt && off + j < max_tiles; j++) {
      const uint32_t r = start + j * BM;
      *((device uint4*)tiles + off + j) =
          uint4(r, uint32_t(e), min(uint32_t(BM), start + cnt - r), 0);
    }
    running += total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0) {
    tile_count[0] = min(running, max_tiles);
  }
}
"""

_SCAN_SOURCE = """
    threadgroup uint32_t run_start[MAXE];
    threadgroup uint32_t run_end[MAXE];
    threadgroup uint32_t simd_tot[32];
    omlx_gqmm_tile_scan<BM>(
        idx, params, tiles, tile_count, run_start, run_end, simd_tot,
        thread_index_in_threadgroup, threads_per_threadgroup.x,
        simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""

# ---------------------------------------------------------------------------
# Matmul
# ---------------------------------------------------------------------------

_MM_HEADER = """
using namespace metal;
using namespace mlx::steel;

namespace omlx_gqmm {

STEEL_CONST int kBN = 64;
STEEL_CONST int kWN = 2;
STEEL_CONST short kSM = 32;
STEEL_CONST short kSN = kBN / kWN;
STEEL_CONST short kSK = 32;
STEEL_CONST short kTM = kSM / 16;
STEEL_CONST short kTN = kSN / 16;
STEEL_CONST short kTK = kSK / 16;

// Tile geometry: BM rows in BM / 32 row simdgroups times kWN column
// simdgroups, K steps BK deep. kLT loader threads dequantize the kBN x BK
// weight tile, each kVPT consecutive values of one weight row: every
// thread when they split the tile evenly, else the largest power of two
// below the thread count (96-row tiles: 128 of 192).
template <int BM, int BK>
struct Geo {
  STEEL_CONST int kBM = BM;
  STEEL_CONST int kBK = BK;
  STEEL_CONST int kWM = BM / kSM;
  STEEL_CONST int kThreads = kWM * kWN * 32;
  STEEL_CONST int kLT = (kThreads & (kThreads - 1)) == 0
      ? kThreads
      : (kThreads > 256 ? 256 : (kThreads > 128 ? 128 : 64));
  STEEL_CONST int kVPT = kBN * BK / kLT;
  STEEL_CONST int kTPR = BK / kVPT;
  static_assert(BM % kSM == 0 && BK % kSK == 0, "tile geometry");
  static_assert(kTPR >= 1 && kTPR * kVPT == BK, "loader split");
};

// Affine: w = scale * q + bias computed in fp32 and rounded once to T, as
// mlx's dequantize() does (scale * q is exact in fp32).
template <typename T, int GS, int BITS>
struct AffineQ {
  using WT = T;
  STEEL_CONST int kBits = BITS;
  STEEL_CONST int kGroup = GS;
  const device T* scales;
  const device T* biases;

  struct P {
    float s;
    float b;
  };

  METAL_FUNC void advance(const size_t n) thread {
    scales += n;
    biases += n;
  }
  METAL_FUNC P params(const int g) const thread {
    return P{float(scales[g]), float(biases[g])};
  }
  METAL_FUNC static WT dq(thread const P& p, const uint32_t q) {
    return static_cast<WT>(p.s * float(q) + p.b);
  }
};

// MXFP4: e2m1 values times the e8m0 group scale, dequantized to bfloat like
// mlx's fp QuantizedBlockLoader (Wtype = bfloat).
template <int GS>
struct Mxfp4Q {
  using WT = bfloat;
  STEEL_CONST int kBits = 4;
  STEEL_CONST int kGroup = GS;
  const device uint8_t* scales;

  struct P {
    float s;
  };

  METAL_FUNC void advance(const size_t n) thread {
    scales += n;
  }
  METAL_FUNC P params(const int g) const thread {
    uint8_t sb = scales[g];
    return P{float(static_cast<bfloat>(*(thread fp8_e8m0*)(&sb)))};
  }
  METAL_FUNC static WT dq(thread const P& p, const uint32_t q) {
    uint8_t qb = uint8_t(q);
    return static_cast<WT>(p.s * float(*(thread fp4_e2m1*)(&qb)));
  }
};

// Gate/up pairing (activation epilogue): weight rows [gate; up] of one
// expert, half_n rows each. Row r of a paired kBN x BK weight tile loads
// weight row pair_row(r) past the tile's first gate row: 16-row blocks
// alternate the gate and the up rows of the same 16 output columns, so the
// two 16-column fragments of every simdgroup's 32-column block accumulate
// gate and up of the same output columns in the same lanes.
METAL_FUNC int pair_row(const int r, const int half_n) {
  return ((r >> 4) & 1) * half_n + ((r >> 5) << 4) + (r & 15);
}

// Activation epilogue of a paired simdgroup block (defined with the
// activation kernels only; see _ACT_HEADER).
template <typename T, int EPI, typename DTile>
METAL_FUNC void store_act(
    thread const DTile& D,
    device T* y,
    const int ld,
    const int rows,
    const T limit);

// Weight-tile loader: loader thread lid owns row lid / kTPR of the
// kBN x BK tile and the kVPT values from column (lid % kTPR) * kVPT, in
// kNG chunks that each lie in one quantization group. fetch() reads the
// packed words and group parameters of one K step, store() dequantizes
// them into threadgroup memory (row stride BKP). The *_tail variants
// cover a K tail of k_valid (a multiple of 32) columns and never touch a
// word or group at or past it. PAIR maps tile rows through pair_row().
//
// mlx packs every affine width as a little-endian bit stream along K
// (power-of-two widths 32 / bits values per uint32; 6-bit 4 values per 3
// bytes, 3/5-bit 8 values per 3/5 bytes: quantized.h get_pack_factor /
// get_bytes_per_pack), so value j of a row starts at bit j * bits. A thread
// whose kVPT values span whole words reads them as words: power-of-two
// widths unpack one word (kCV = 32 / bits values) at a time, the others a
// run of kCW words holding kCV values (6-bit: 16 values in 3 words; 3/5-bit:
// 32 values in 3/5 words), a value straddling two words taking its high
// bits from the next one.
template <typename Q, typename G, bool PAIR = false>
struct TileLoader {
  using WT = typename Q::WT;
  using P = typename Q::P;
  STEEL_CONST int kBits = Q::kBits;
  STEEL_CONST int kVPT = G::kVPT;
  STEEL_CONST int kWords = kVPT * kBits / 32;
  STEEL_CONST bool kPow2 = (kBits & (kBits - 1)) == 0;
  // Values and words per unpack chunk (see above).
  STEEL_CONST int kCV = kPow2 ? 32 / kBits : (kBits == 6 ? 16 : 32);
  STEEL_CONST int kCW = kCV * kBits / 32;
  STEEL_CONST int kNC = kVPT / kCV;
  STEEL_CONST uint32_t kMask = (1u << kBits) - 1u;
  STEEL_CONST int kGV = kVPT < Q::kGroup ? kVPT : Q::kGroup;
  STEEL_CONST int kNG = kVPT / kGV;
  STEEL_CONST int kWPG = kGV * kBits / 32;
  STEEL_CONST int kBKP = G::kBK + 16 / sizeof(WT);
  static_assert(kWords * 32 == kVPT * kBits, "whole words per thread");
  static_assert(kWPG >= 1 && kNG * kWPG == kWords, "group split");
  static_assert(kCW * 32 == kCV * kBits && kNC * kCV == kVPT, "chunk split");
  static_assert(kGV % kCV == 0, "chunks lie in one group");

  const device uint32_t* src;
  Q q;
  const short row;
  const short col;
  uint32_t raw[kWords];
  P p[kNG];

  // PAIR: w_tile / q_ hold the tile's first gate row and w_up / q_up its
  // first up row (mlx2: the same rows of a separate up table, or half_n
  // rows further in a concatenated [gate; up] table); 16-row blocks
  // alternate the two exactly as pair_row() maps them.
  METAL_FUNC TileLoader(
      const device uint8_t* w_tile,
      const device uint8_t* w_up,
      const int K,
      thread const Q& q_,
      thread const Q& q_up,
      const uint lid) thread
      : q((PAIR && (((lid / G::kTPR) >> 4) & 1)) ? q_up : q_),
        row(short(lid / G::kTPR)),
        col(short((lid % G::kTPR) * kVPT)) {
    const bool up = PAIR && ((row >> 4) & 1);
    const size_t w_off =
        PAIR ? size_t(((row >> 5) << 4) + (row & 15)) : size_t(row);
    src = (const device uint32_t*)((up ? w_up : w_tile) +
                                   w_off * (K * kBits / 8) + col * kBits / 8);
    q.advance(w_off * (K / Q::kGroup));
  }

  METAL_FUNC void fetch(const int kb) thread {
    const device uint32_t* ptr = src + kb * (G::kBK * kBits / 32);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kWords; i++) {
      raw[i] = ptr[i];
    }
    STEEL_PRAGMA_UNROLL
    for (short g = 0; g < kNG; g++) {
      p[g] = q.params((kb * G::kBK + col + g * kGV) / Q::kGroup);
    }
  }

  METAL_FUNC void fetch_tail(const int kb, const int k_valid) thread {
    const device uint32_t* ptr = src + kb * (G::kBK * kBits / 32);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kWords; i++) {
      // Word i starts at value (32 * i) / kBits; k_valid is a multiple of
      // 32 values, a word boundary for every width.
      if (col + (32 * i) / kBits < k_valid) {
        raw[i] = ptr[i];
      }
    }
    STEEL_PRAGMA_UNROLL
    for (short g = 0; g < kNG; g++) {
      if (col + g * kGV < k_valid) {
        p[g] = q.params((kb * G::kBK + col + g * kGV) / Q::kGroup);
      }
    }
  }

  METAL_FUNC void store_words(threadgroup WT* Ws, const int k_valid) const
      thread {
    threadgroup WT* dst = Ws + row * kBKP + col;
    if constexpr (kPow2) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kWords; i++) {
        if (col + i * kCV < k_valid) {
          vec<WT, kCV> v;
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < kCV; j++) {
            v[j] = Q::dq(p[i / kWPG], (raw[i] >> (kBits * j)) & kMask);
          }
          *(threadgroup vec<WT, kCV>*)(dst + i * kCV) = v;
        }
      }
    } else {
      // Chunks of kCV values (whole words, one group), stored 8 at a time.
      STEEL_PRAGMA_UNROLL
      for (short c = 0; c < kNC; c++) {
        if (col + c * kCV < k_valid) {
          const P pc = p[(c * kCV) / kGV];
          const thread uint32_t* rw = raw + c * kCW;
          STEEL_PRAGMA_UNROLL
          for (short j0 = 0; j0 < kCV; j0 += 8) {
            vec<WT, 8> v;
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < 8; jj++) {
              const short bit = (j0 + jj) * kBits;
              const short wi = bit / 32;
              const short sh = bit % 32;
              uint32_t qv = rw[wi] >> sh;
              if (sh + kBits > 32) {
                qv |= rw[wi + 1] << (32 - sh);
              }
              v[jj] = Q::dq(pc, qv & kMask);
            }
            *(threadgroup vec<WT, 8>*)(dst + c * kCV + j0) = v;
          }
        }
      }
    }
  }

  METAL_FUNC void store(threadgroup WT* Ws) const thread {
    store_words(Ws, G::kBK);
  }

  METAL_FUNC void zero(threadgroup WT* Ws) const thread {
    threadgroup WT* dst = Ws + row * kBKP + col;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kVPT; i++) {
      dst[i] = WT(0);
    }
  }
};

// One 32-deep sub-step of a simdgroup's 32 x 32 block: full row blocks run
// tile_matmad_nax; partial ones skip the 16-row fragments without rows
// (the tensor ops of the others are the ones tile_matmad_nax issues).
template <typename T, typename WT, int BKP, bool FULL>
METAL_FUNC void sub_step(
    thread NAXTile<float, kTM, kTN>& Dtile,
    const device T* xn,
    const threadgroup WT* ws,
    const int K,
    const short sgp_sm) {
  NAXTile<WT, kTN, kTK> Btile;
  if constexpr (FULL) {
    NAXTile<T, kTM, kTK> Atile;

    volatile int compiler_barrier;

    Atile.load(xn, K);
    Btile.template load<WT, BKP, 1>(ws);

    tile_matmad_nax(
        Dtile,
        Atile,
        metal::bool_constant<false>{},
        Btile,
        metal::bool_constant<true>{});

    (void)compiler_barrier;
  } else {
    Btile.template load<WT, BKP, 1>(ws);
    STEEL_PRAGMA_UNROLL
    for (short mm = 0; mm < kTM; mm++) {
      if (mm * 16 < sgp_sm) {
        NAXTile<T, 1, kTK> Arow;
        Arow.load_safe(xn + mm * 16 * K, K, short2(kSK, sgp_sm - mm * 16));
        STEEL_PRAGMA_UNROLL
        for (short nn = 0; nn < kTN; nn += 2) {
          STEEL_PRAGMA_UNROLL
          for (short kk = 0; kk < kTK; kk++) {
            BaseNAXFrag::mma(
                Dtile.frag_at(mm, nn),
                Dtile.frag_at(mm, nn + 1),
                Arow.frag_at(0, kk),
                metal::bool_constant<false>{},
                Btile.frag_at(nn, kk),
                Btile.frag_at(nn + 1, kk),
                metal::bool_constant<true>{});
          }
        }
      }
    }
  }
}

// Row-mapped activations (MAP): sorted row r of the product is token row
// rmap[r] of x, read in place instead of from a replicated copy. a_off[i][h]
// is this lane's element offset of activation row i * 16 + h * 8 + sc.y of
// its simdgroup block (rmap[row] * K + sc.x; rows past the block point at
// its last row), so lane values are exactly those NAXTile::load reads from
// the copy.
template <typename T, short R>
METAL_FUNC void load_a_map(
    thread NAXTile<T, R, kTK>& A,
    const device T* xk,
    const thread uint (&a_off)[kTM][2],
    const short i0) {
  STEEL_PRAGMA_UNROLL
  for (short r = 0; r < R; r++) {
    STEEL_PRAGMA_UNROLL
    for (short h = 0; h < 2; h++) {
      const device T* xp = xk + a_off[i0 + r][h];
      STEEL_PRAGMA_UNROLL
      for (short kk = 0; kk < kTK; kk++) {
        const vec<T, 4> v = *(const device vec<T, 4>*)(xp + kk * 16);
        STEEL_PRAGMA_UNROLL
        for (short c = 0; c < 4; c++) {
          A.frag_at(r, kk)[h * 4 + c] = v[c];
        }
      }
    }
  }
}

// sub_step with row-mapped activations (xk: the token rows advanced to this
// sub-step's K offset): the same fragment values (rows past sgp_sm of a
// partial block zero as load_safe makes them) and the same tensor ops in the
// same order.
template <typename T, typename WT, int BKP, bool FULL>
METAL_FUNC void sub_step_map(
    thread NAXTile<float, kTM, kTN>& Dtile,
    const device T* xk,
    const thread uint (&a_off)[kTM][2],
    const threadgroup WT* ws,
    const short sgp_sm) {
  NAXTile<WT, kTN, kTK> Btile;
  if constexpr (FULL) {
    NAXTile<T, kTM, kTK> Atile;

    volatile int compiler_barrier;

    load_a_map<T, kTM>(Atile, xk, a_off, 0);
    Btile.template load<WT, BKP, 1>(ws);

    tile_matmad_nax(
        Dtile,
        Atile,
        metal::bool_constant<false>{},
        Btile,
        metal::bool_constant<true>{});

    (void)compiler_barrier;
  } else {
    const short2 sc = BaseNAXFrag::get_coord();
    Btile.template load<WT, BKP, 1>(ws);
    STEEL_PRAGMA_UNROLL
    for (short mm = 0; mm < kTM; mm++) {
      if (mm * 16 < sgp_sm) {
        NAXTile<T, 1, kTK> Arow;
        load_a_map<T, 1>(Arow, xk, a_off, mm);
        STEEL_PRAGMA_UNROLL
        for (short h = 0; h < 2; h++) {
          if (mm * 16 + h * 8 + sc.y >= sgp_sm) {
            STEEL_PRAGMA_UNROLL
            for (short kk = 0; kk < kTK; kk++) {
              STEEL_PRAGMA_UNROLL
              for (short c = 0; c < 4; c++) {
                Arow.frag_at(0, kk)[h * 4 + c] = T(0);
              }
            }
          }
        }
        STEEL_PRAGMA_UNROLL
        for (short nn = 0; nn < kTN; nn += 2) {
          STEEL_PRAGMA_UNROLL
          for (short kk = 0; kk < kTK; kk++) {
            BaseNAXFrag::mma(
                Dtile.frag_at(mm, nn),
                Dtile.frag_at(mm, nn + 1),
                Arow.frag_at(0, kk),
                metal::bool_constant<false>{},
                Btile.frag_at(nn, kk),
                Btile.frag_at(nn + 1, kk),
                metal::bool_constant<true>{});
          }
        }
      }
    }
  }
}

// Element offsets of this lane's activation rows (see load_a_map) for the
// simdgroup block starting at tile row m0 of a tile of tile_rows rows at
// sorted row row_start.
METAL_FUNC void map_rows(
    thread uint (&a_off)[kTM][2],
    const device uint32_t* rmap,
    const int row_start,
    const int tile_rows,
    const int m0,
    const int K) {
  const short2 sc = BaseNAXFrag::get_coord();
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kTM; i++) {
    STEEL_PRAGMA_UNROLL
    for (short h = 0; h < 2; h++) {
      const int r = min(m0 + i * 16 + h * 8 + int(sc.y), tile_rows - 1);
      a_off[i][h] = rmap[row_start + r] * uint(K) + uint(sc.x);
    }
  }
}

// seg: mlx's segmented sorted gather kernel (affine_gather_qmm_rhs_seg_nax /
// fp_gather_qmm_rhs_seg_nax): one single-expert BM x kBN tile per
// threadgroup, the weight tile of each BK-deep K step dequantized into
// threadgroup memory between two barriers. K tail (K % BK, a multiple of
// 32): only its sub-steps run. N tail: weight rows past N are zero, stores
// are bounded.
//
// EPI > 0 (activation epilogue; N = 2 * half_n [gate; up] rows, aligned):
// the tile of fused columns [y_col, y_col + kBN) computes gate and up of
// output columns [y_col / 2, y_col / 2 + kBN / 2) through the paired row
// map and writes act(gate, up) to the [M, half_n] output.
template <
    typename T,
    typename Q,
    typename G,
    bool ALIGN_N,
    bool ALIGN_K,
    int EPI = 0,
    bool MAP = false,
    bool SPLIT = false>
METAL_FUNC void gather_seg(
    const device T* x,
    const device uint32_t* rmap,
    const device uint8_t* w,
    thread Q& q,
    const uint4 desc,
    const int y_col,
    device T* y,
    const int N,
    const int K,
    threadgroup typename Q::WT* Ws,
    const uint sgid,
    const uint lane,
    const T limit = T(0),
    const device uint8_t* w2 = nullptr,
    thread const Q* q2 = nullptr) {
  using WT = typename Q::WT;
  constexpr bool kPair = EPI != 0;
  static_assert(!kPair || ALIGN_N, "paired gate/up tiles are full");
  constexpr int BKP = G::kBK + 16 / sizeof(WT);
  const int row_start = int(desc.x);
  const uint32_t expert = desc.y;
  const int rows = int(desc.z);

  const int K_w = K * Q::kBits / 8;
  const int K_g = K / Q::kGroup;
  const int K_it = K / G::kBK;
  const short tgp_bn = ALIGN_N ? short(kBN) : short(min(kBN, N - y_col));
  const int k_remain = K - K_it * G::kBK;
  // First weight row of the tile past the expert's rows and the output
  // row stride (paired: the tile's first output column, half_n columns).
  const int half_n = N / 2;
  const int w_col = kPair ? y_col / 2 : y_col;
  const int ldy = kPair ? half_n : N;

  // SPLIT (mlx2): gate rows in w / q and up rows in w2 / q2, half_n rows
  // per expert each; otherwise one table of N rows per expert.
  const size_t w_row = size_t(expert) * (SPLIT ? half_n : N) + w_col;
  q.advance(w_row * K_g);
  Q q_up = q;
  const device uint8_t* w_up = w + w_row * K_w;
  if constexpr (SPLIT) {
    q_up = *q2;
    q_up.advance(w_row * K_g);
    w_up = w2 + w_row * K_w;
  } else if constexpr (kPair) {
    q_up.advance(size_t(half_n) * K_g);
    w_up += size_t(half_n) * K_w;
  }
  TileLoader<Q, G, kPair> loader(
      w + w_row * K_w, w_up, K, q, q_up, sgid * 32 + lane);
  const bool loads = G::kLT == G::kThreads || sgid * 32 + lane < uint(G::kLT);
  const bool row_live = ALIGN_N || loader.row < tgp_bn;

  if constexpr (!MAP) {
    x += size_t(row_start) * K;
  }
  y += size_t(row_start) * ldy + w_col;

  const short tm = kSM * short(sgid / kWN);
  const short tn = kSN * short(sgid % kWN);
  const short sgp_sm = short(min(int(kSM), max(0, rows - int(tm))));
  const short sgp_sn =
      ALIGN_N ? kSN : short(min(int(kSN), max(0, N - (y_col + tn))));
  const bool sg_active = sgp_sm > 0;
  uint a_off[kTM][2];
  if constexpr (MAP) {
    map_rows(a_off, rmap, row_start, rows, int(tm), K);
  }

  NAXTile<float, kTM, kTN> Dtile;
  Dtile.clear();
  // MAP: xn walks K over the token rows; a_off selects each lane's rows.
  const device T* xn = MAP ? x : x + tm * K;
  const threadgroup WT* ws = Ws + tn * BKP;

  dispatch_bool(sgp_sm == kSM, [&](auto kAlignedM) {
    for (int k = 0; k < K_it; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (loads) {
        if (row_live) {
          loader.fetch(k);
          loader.store(Ws);
        } else {
          loader.zero(Ws);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      STEEL_PRAGMA_NO_UNROLL
      for (int kk1 = 0; kk1 < G::kBK; kk1 += kSK) {
        if (sg_active) {
          if constexpr (MAP) {
            sub_step_map<T, WT, BKP, kAlignedM.value>(
                Dtile, xn + kk1, a_off, ws + kk1, sgp_sm);
          } else {
            sub_step<T, WT, BKP, kAlignedM.value>(
                Dtile, xn + kk1, ws + kk1, K, sgp_sm);
          }
        }
      }
      xn += G::kBK;
    }

    if (!ALIGN_K) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (loads) {
        if (row_live) {
          loader.fetch_tail(K_it, k_remain);
          loader.store_words(Ws, k_remain);
        } else {
          loader.zero(Ws);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      STEEL_PRAGMA_NO_UNROLL
      for (int kk1 = 0; kk1 < k_remain; kk1 += kSK) {
        if (sg_active) {
          if constexpr (MAP) {
            sub_step_map<T, WT, BKP, kAlignedM.value>(
                Dtile, xn + kk1, a_off, ws + kk1, sgp_sm);
          } else {
            sub_step<T, WT, BKP, kAlignedM.value>(
                Dtile, xn + kk1, ws + kk1, K, sgp_sm);
          }
        }
      }
    }

    if constexpr (kPair) {
      if (sg_active) {
        store_act<T, EPI>(
            Dtile, y + tm * ldy + tn / 2, ldy, int(sgp_sm), limit);
      }
    } else if (kAlignedM.value && sgp_sn == kSN) {
      Dtile.store(y + tm * N + tn, N);
    } else if (sg_active) {
      Dtile.store_safe(y + tm * N + tn, N, short2(sgp_sn, sgp_sm));
    }
  });
}

// db: the same tiles and arithmetic with double-buffered 64-deep weight
// tiles: the packed words of step k + 1 are fetched before the tensor ops
// of step k and dequantized into the other buffer after them, so each K
// step has a single barrier. Activation fragments are read straight from
// device memory (rows past the tile are clamped to its last row and never
// stored) and 16-row fragments without rows of the tile are skipped.
// Requires K % 64 == 0 and N % 64 == 0. EPI > 0: the activation epilogue
// of gather_seg.
template <
    typename T,
    typename Q,
    typename G,
    int EPI = 0,
    bool MAP = false,
    bool SPLIT = false>
METAL_FUNC void gather_db(
    const device T* x,
    const device uint32_t* rmap,
    const device uint8_t* w,
    thread Q& q,
    const uint4 desc,
    const int y_col,
    device T* y,
    const int N,
    const int K,
    threadgroup typename Q::WT* Ws,
    const uint sgid,
    const uint lane,
    const T limit = T(0),
    const device uint8_t* w2 = nullptr,
    thread const Q* q2 = nullptr) {
  using WT = typename Q::WT;
  static_assert(G::kBK == 64, "db runs 64-deep K steps");
  constexpr bool kPair = EPI != 0;
  constexpr int BKP = G::kBK + 16 / sizeof(WT);
  constexpr int kTile = kBN * BKP;
  const int row_start = int(desc.x);
  const uint32_t expert = desc.y;
  const int tile_rows = int(desc.z);

  const int K_w = K * Q::kBits / 8;
  const int K_g = K / Q::kGroup;
  const int K_it = K / G::kBK;
  const int half_n = N / 2;
  const int w_col = kPair ? y_col / 2 : y_col;

  // SPLIT (mlx2): gate rows in w / q and up rows in w2 / q2, half_n rows
  // per expert each; otherwise one table of N rows per expert.
  const size_t w_row = size_t(expert) * (SPLIT ? half_n : N) + w_col;
  q.advance(w_row * K_g);
  Q q_up = q;
  const device uint8_t* w_up = w + w_row * K_w;
  if constexpr (SPLIT) {
    q_up = *q2;
    q_up.advance(w_row * K_g);
    w_up = w2 + w_row * K_w;
  } else if constexpr (kPair) {
    q_up.advance(size_t(half_n) * K_g);
    w_up += size_t(half_n) * K_w;
  }
  TileLoader<Q, G, kPair> loader(
      w + w_row * K_w, w_up, K, q, q_up, sgid * 32 + lane);
  const bool loads = G::kLT == G::kThreads || sgid * 32 + lane < uint(G::kLT);

  const int m0 = kSM * int(sgid / kWN);
  const int rows = min(int(kSM), tile_rows - m0);
  // MAP: x holds the token rows; offsets address them through rmap.
  const device T* xs = MAP
      ? x
      : x + size_t(row_start + max(0, min(m0, tile_rows - 1))) * K;

  const short2 sc = BaseNAXFrag::get_coord();
  metal::conditional_t<MAP, uint, int> x_off[kTM][2];
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kTM; i++) {
    STEEL_PRAGMA_UNROLL
    for (short h = 0; h < 2; h++) {
      const int r = min(int(i * 16 + sc.y + h * 8), max(rows, 1) - 1);
      if constexpr (MAP) {
        x_off[i][h] = rmap[row_start + min(m0 + r, tile_rows - 1)] * uint(K) +
            uint(sc.x);
      } else {
        x_off[i][h] = r * K + sc.x;
      }
    }
  }
  const short m_frags = rows > 0 ? short((rows + 15) / 16) : short(0);
  const threadgroup WT* wsg = Ws + (sgid % kWN) * kSN * BKP;

  NAXTile<float, kTM, kTN> D;
  D.clear();

  if (loads) {
    loader.fetch(0);
    loader.store(Ws);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int kb = 0; kb < K_it; kb++) {
    const bool more = kb + 1 < K_it;
    if (more && loads) {
      loader.fetch(kb + 1);
    }
    const threadgroup WT* wb = wsg + (kb & 1) * kTile;
    STEEL_PRAGMA_UNROLL
    for (short kk1 = 0; kk1 < G::kBK; kk1 += kSK) {
      NAXTile<WT, kTN, 2> Btile;
      Btile.template load<WT, BKP, 1>(wb + kk1);
      const int k = kb * G::kBK + kk1;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kTM; i++) {
        if (i < m_frags) {
          NAXTile<T, 1, 2> Atile;
          STEEL_PRAGMA_UNROLL
          for (short h = 0; h < 2; h++) {
            const device T* xp = xs + x_off[i][h] + k;
            const vec<T, 4> a0 = *(const device vec<T, 4>*)(xp);
            const vec<T, 4> a1 = *(const device vec<T, 4>*)(xp + 16);
            STEEL_PRAGMA_UNROLL
            for (short c = 0; c < 4; c++) {
              Atile.frag_at(0, 0)[h * 4 + c] = a0[c];
              Atile.frag_at(0, 1)[h * 4 + c] = a1[c];
            }
          }
          STEEL_PRAGMA_UNROLL
          for (short kk = 0; kk < 2; kk++) {
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < kTN; j += 2) {
              BaseNAXFrag::mma(
                  D.frag_at(i, j),
                  D.frag_at(i, j + 1),
                  Atile.frag_at(0, kk),
                  metal::bool_constant<false>{},
                  Btile.frag_at(j, kk),
                  Btile.frag_at(j + 1, kk),
                  metal::bool_constant<true>{});
            }
          }
        }
      }
    }
    if (more && loads) {
      loader.store(Ws + ((kb + 1) & 1) * kTile);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if constexpr (kPair) {
    if (rows > 0) {
      store_act<T, EPI>(
          D,
          y + size_t(row_start + m0) * half_n + w_col + (kSN / 2) * (sgid % kWN),
          half_n,
          rows,
          limit);
    }
  } else {
    device T* yb = y + size_t(row_start + m0) * N + y_col + kSN * (sgid % kWN);
    if (rows >= kSM) {
      D.store(yb, N);
    } else if (rows > 0) {
      D.store_safe(yb, N, short2(kSN, short(rows)));
    }
  }
}

// The row tile and output column of this threadgroup. GX == 0: grid
// (columns, tiles) as mlx lays it out. GX > 0: tile t on grid x t % GX and
// (t / GX, column) on y, so all threadgroups of a row tile share one x
// coordinate and a tile's columns run GX threadgroups apart.
template <int GX>
METAL_FUNC bool tile_of(
    const device uint32_t* tiles,
    const uint tile_count,
    const uint3 tid,
    const int N,
    thread uint4& desc,
    thread int& y_col) {
  uint t;
  uint c;
  if constexpr (GX > 0) {
    const uint n_cols = uint((N + kBN - 1) / kBN);
    t = (tid.y / n_cols) * GX + tid.x;
    c = tid.y % n_cols;
  } else {
    t = tid.y;
    c = tid.x;
  }
  if (t >= tile_count) {
    return false;
  }
  desc = *((const device uint4*)tiles + t);
  y_col = int(c) * kBN;
  return true;
}

} // namespace omlx_gqmm
"""

_MM_SOURCE_TMPL = """
    {q_type}
    using G = omlx_gqmm::Geo<BM, BK>;
    using WT = typename Q::WT;
    constexpr int BKP = BK + 16 / sizeof(WT);
    threadgroup WT Ws[(SCHED == 1 ? 2 : 1) * omlx_gqmm::kBN * BKP +
                      PAD / sizeof(WT)];
    uint4 desc;
    int y_col;
    if (!omlx_gqmm::tile_of<GX>(
            tiles, tile_count[0], threadgroup_position_in_grid, params[0],
            desc, y_col)) {{
        return;
    }}
    {q_init}
    if constexpr (SCHED == 1) {{
        omlx_gqmm::gather_db<T, Q, G>(
            x, tiles, (const device uint8_t*)w, q, desc, y_col, y, params[0],
            params[1], Ws, simdgroup_index_in_threadgroup,
            thread_index_in_simdgroup);
    }} else {{
        omlx_gqmm::gather_seg<T, Q, G, ALIGN_N, ALIGN_K>(
            x, tiles, (const device uint8_t*)w, q, desc, y_col, y, params[0],
            params[1], Ws, simdgroup_index_in_threadgroup,
            thread_index_in_simdgroup);
    }}
"""

_AFFINE_SOURCE = _MM_SOURCE_TMPL.format(
    q_type="using Q = omlx_gqmm::AffineQ<T, GS, BITS>;",
    q_init="Q q{scales, biases};",
)

_FP_SOURCE = _MM_SOURCE_TMPL.format(
    q_type="using Q = omlx_gqmm::Mxfp4Q<GS>;",
    q_init="Q q{scales};",
)

# ---------------------------------------------------------------------------
# Gate/up activation epilogue
# ---------------------------------------------------------------------------

# MLX's elementwise functors (Sigmoid, Multiply, Minimum, Maximum): the ones
# its compiled-graph kernels call, from the installed package.
_MLX_OPS_HEADERS = (
    "mlx/backend/metal/kernels/unary_ops.h",
    "mlx/backend/metal/kernels/binary_ops.h",
)

_ACT_HEADER = """
namespace omlx_gqmm {

// The unfused path's activation on the two rounded projections, op for op
// and every intermediate in T like MLX's compiled kernel of
// nn.silu(gate) * up (Sigmoid, Multiply, Multiply); EPI == 2 first clips
// like GLM-5.3's clamped SwiGLU: gate = minimum(gate, limit),
// up = minimum(maximum(up, -limit), limit).
template <typename T, int EPI>
METAL_FUNC T act(T g, T u, const T limit) {
  if constexpr (EPI == 2) {
    g = Minimum()(g, limit);
    u = Minimum()(Maximum()(u, T(-limit)), limit);
  }
  return Multiply()(Multiply()(g, Sigmoid()(g)), u);
}

// D.frag_at(i, 0) holds gate and D.frag_at(i, 1) up of the same 16 output
// columns (pair_row), so each lane holds both projections of its (row,
// column) elements. Each is rounded to T as the plain store rounds it,
// then act() writes the [rows, 16] block of the [M, ld] output (y points
// at its first element).
template <typename T, int EPI, typename DTile>
METAL_FUNC void store_act(
    thread const DTile& D,
    device T* y,
    const int ld,
    const int rows,
    const T limit) {
  const short2 sc = BaseNAXFrag::get_coord();
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < DTile::kTileRows; i++) {
    STEEL_PRAGMA_UNROLL
    for (short h = 0; h < BaseNAXFrag::kElemRows; h++) {
      const int r = i * BaseNAXFrag::kFragRows +
          h * BaseNAXFrag::kElemRowsJump + sc.y;
      if (r < rows) {
        vec<T, BaseNAXFrag::kElemCols> v;
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < BaseNAXFrag::kElemCols; j++) {
          const short e = h * BaseNAXFrag::kElemCols + j;
          v[j] = act<T, EPI>(
              static_cast<T>(D.frag_at(i, 0)[e]),
              static_cast<T>(D.frag_at(i, 1)[e]),
              limit);
        }
        *(device vec<T, BaseNAXFrag::kElemCols>*)(y + size_t(r) * ld + sc.x) =
            v;
      }
    }
  }
}

} // namespace omlx_gqmm
"""

_ACT_SOURCE_TMPL = """
    {q_type}
    using G = omlx_gqmm::Geo<BM, BK>;
    using WT = typename Q::WT;
    constexpr int BKP = BK + 16 / sizeof(WT);
    threadgroup WT Ws[(SCHED == 1 ? 2 : 1) * omlx_gqmm::kBN * BKP +
                      PAD / sizeof(WT)];
    uint4 desc;
    int y_col;
    if (!omlx_gqmm::tile_of<GX>(
            tiles, tile_count[0], threadgroup_position_in_grid, params[0],
            desc, y_col)) {{
        return;
    }}
    {q_init}
    {q2_init}
    const T limit = lim[0];
    if constexpr (SCHED == 1) {{
        omlx_gqmm::gather_db<T, Q, G, EPI, {mapped}, {split}>(
            x, {rmap}, (const device uint8_t*)w, q, desc, y_col, y, params[0],
            params[1], Ws, simdgroup_index_in_threadgroup,
            thread_index_in_simdgroup, limit, {w2}, &q2);
    }} else {{
        omlx_gqmm::gather_seg<T, Q, G, true, ALIGN_K, EPI, {mapped}, {split}>(
            x, {rmap}, (const device uint8_t*)w, q, desc, y_col, y, params[0],
            params[1], Ws, simdgroup_index_in_threadgroup,
            thread_index_in_simdgroup, limit, {w2}, &q2);
    }}
"""


def _act_source(q_type: str, q_init: str, mapped: bool, split: bool = False,
                q2_init: str = "") -> str:
    # Mapped kernels read the sorted rows through the ``rmap`` input; the
    # others pass ``tiles`` as the (unread) row map.  Split kernels (mlx2)
    # read the up rows from the ``w2`` / ``scales2`` (/ ``biases2``) tables;
    # the others pass their own table and ``q`` (unread).
    return _ACT_SOURCE_TMPL.format(
        q_type=q_type,
        q_init=q_init,
        q2_init=q2_init if split else "Q q2 = q;",
        split="true" if split else "false",
        w2="(const device uint8_t*)w2" if split else "(const device uint8_t*)w",
        mapped="true" if mapped else "false",
        rmap="rmap" if mapped else "tiles",
    )


_AFFINE_Q = (
    "using Q = omlx_gqmm::AffineQ<T, GS, BITS>;",
    "Q q{scales, biases};",
)
_FP_Q = ("using Q = omlx_gqmm::Mxfp4Q<GS>;", "Q q{scales};")
_AFFINE_ACT_SOURCE = _act_source(*_AFFINE_Q, mapped=False)
_FP_ACT_SOURCE = _act_source(*_FP_Q, mapped=False)
_AFFINE_ACT_MAP_SOURCE = _act_source(*_AFFINE_Q, mapped=True)
_FP_ACT_MAP_SOURCE = _act_source(*_FP_Q, mapped=True)
_AFFINE_Q2 = "Q q2{scales2, biases2};"
_FP_Q2 = "Q q2{scales2};"
_SPLIT_SOURCES = {
    (affine, mapped): _act_source(
        *(_AFFINE_Q if affine else _FP_Q), mapped=mapped, split=True,
        q2_init=_AFFINE_Q2 if affine else _FP_Q2,
    )
    for affine in (True, False)
    for mapped in (False, True)
}

# Epilogue kinds (the kernel's EPI): silu(gate) * up, and the clamped form.
_EPI_SWIGLU = 1
_EPI_CLAMPED = 2

_SCHED_SEG = 0
_SCHED_DB = 1
_SCHED_NAMES = {_SCHED_SEG: "seg", _SCHED_DB: "db"}


class Plan(NamedTuple):
    """One kernel configuration: schedule, tile rows, K step, layout, pad.

    ``gx`` is the row tiles per grid-x group (0: mlx's (column, tile)
    grid); ``pad`` is extra threadgroup memory in bytes (fewer resident
    threadgroups).
    """

    sched: int
    bm: int
    bk: int
    gx: int
    pad: int

    def describe(self) -> str:
        s = f"{_SCHED_NAMES[self.sched]} {self.bm}x{_BN} bk{self.bk}"
        if self.gx:
            s += f" gx{self.gx}"
        if self.pad:
            s += f" pad{self.pad}"
        return s


_lock = threading.RLock()
_kernels: dict[str, object] = {}
_header_failed = False
_act_header_failed = False
# Self-test verdict per kernel instantiation:
# (dtype, mode, bits, group_size, plan, align_n, align_k) -> bool, and for
# the activation epilogue the same key + (epi, limit).
_verified: dict[tuple, bool] = {}


_nax_host: Optional[bool] = None
# The Metal device name the host gate was evaluated on ("" when unknown).
_device_name: Optional[str] = None


def nax_host() -> bool:
    """True on a Metal device of the M5 family (the NAX tensor units).

    Only a gate on where to try: every kernel instantiation still proves
    nonzero numerics bit-for-bit against the stock op before it is used
    (a compile probe alone can lie, see ``qwen4_qsa_nax``)."""
    global _nax_host, _device_name
    if _nax_host is None:
        try:
            metal = bool(mx.metal.is_available())
            _device_name = (
                str(mx.device_info().get("device_name", "")) if metal else ""
            )
            _nax_host = metal and "M5" in _device_name
        except Exception:  # noqa: BLE001
            _nax_host = False
            _device_name = _device_name or ""
    return _nax_host


def host_gate() -> dict:
    """The evaluated host gate: ``{"nax_host", "device_name"}``.

    Bound into a qualification receipt that waives this mechanism as
    host-gated, and re-evaluated by the loader on the serving host."""
    admitted = nax_host()
    return {"nax_host": bool(admitted), "device_name": _device_name or ""}


def enabled() -> bool:
    """True on an M5 host (the caller holds the policy switch)."""
    return nax_host()


def _get_act_kernel(kind: str):
    """Build (once) the ``affine_act`` or ``fp_act`` kernel object, or its
    row-mapped variant (``*_act_map``: sorted rows read through ``rmap``)."""
    global _act_header_failed
    kernel = _kernels.get(kind)
    if kernel is not None or _act_header_failed:
        return kernel
    with _lock:
        kernel = _kernels.get(kind)
        if kernel is not None:
            return kernel
        mlx_src = _read_mlx_headers(_MLX_MM_HEADERS + _MLX_OPS_HEADERS)
        if mlx_src is None:
            _act_header_failed = True
            logger.warning(
                "mlx kernel headers not found under %s; NAX gate/up "
                "activation epilogue disabled",
                Path(mx.__file__).parent / "include",
            )
            return None
        header = mlx_src + _MM_HEADER + _ACT_HEADER
        mapped = kind.endswith("_map")
        affine = kind.startswith("affine")
        split = "_split" in kind
        inputs = ["x", "w", "scales"] + (["biases"] if affine else [])
        if split:
            inputs += ["w2", "scales2"] + (["biases2"] if affine else [])
        inputs += ["tiles", "tile_count", "params", "lim"]
        if mapped:
            inputs.append("rmap")
        if split:
            source = _SPLIT_SOURCES[(affine, mapped)]
        else:
            source = {
                (True, False): _AFFINE_ACT_SOURCE,
                (False, False): _FP_ACT_SOURCE,
                (True, True): _AFFINE_ACT_MAP_SOURCE,
                (False, True): _FP_ACT_MAP_SOURCE,
            }[(affine, mapped)]
        kernel = mx.fast.metal_kernel(
            name=("mlx2_gqmm_affine_swiglu" if affine else "mlx2_gqmm_mxfp4_swiglu")
            + ("_split" if split else "")
            + ("_map" if mapped else ""),
            input_names=inputs,
            output_names=["y"],
            header=header,
            source=source,
        )
        _kernels[kind] = kernel
        return kernel


def _get_kernel(kind: str):
    """Build (once) the ``scan``, ``affine`` or ``fp`` kernel object (and
    the ``*_act`` epilogue variants)."""
    global _header_failed
    if "_act" in kind:
        return _get_act_kernel(kind)
    kernel = _kernels.get(kind)
    if kernel is not None or _header_failed:
        return kernel
    with _lock:
        kernel = _kernels.get(kind)
        if kernel is not None:
            return kernel
        if kind == "scan":
            kernel = mx.fast.metal_kernel(
                name="omlx_gqmm_tile_scan",
                input_names=["idx", "params"],
                output_names=["tiles", "tile_count"],
                header=_SCAN_HEADER,
                source=_SCAN_SOURCE,
            )
        else:
            mlx_src = _read_mlx_headers(_MLX_MM_HEADERS)
            if mlx_src is None:
                _header_failed = True
                logger.warning(
                    "mlx kernel headers not found under %s; NAX sorted "
                    "gather_qmm disabled",
                    Path(mx.__file__).parent / "include",
                )
                return None
            if kind == "affine":
                kernel = mx.fast.metal_kernel(
                    name="omlx_gqmm_affine_v2",
                    input_names=[
                        "x",
                        "w",
                        "scales",
                        "biases",
                        "tiles",
                        "tile_count",
                        "params",
                    ],
                    output_names=["y"],
                    header=mlx_src + _MM_HEADER,
                    source=_AFFINE_SOURCE,
                )
            else:
                kernel = mx.fast.metal_kernel(
                    name="omlx_gqmm_mxfp4_v2",
                    input_names=["x", "w", "scales", "tiles", "tile_count", "params"],
                    output_names=["y"],
                    header=mlx_src + _MM_HEADER,
                    source=_FP_SOURCE,
                )
        _kernels[kind] = kernel
        return kernel


def _parse_plan(text: str) -> Optional[Plan]:
    parts = [p.strip().lower() for p in text.split(",")]
    if len(parts) != 5 or parts[0] not in ("seg", "db"):
        return None
    try:
        bm, bk, gx, pad = (int(p) for p in parts[1:])
    except ValueError:
        return None
    sched = _SCHED_SEG if parts[0] == "seg" else _SCHED_DB
    if bm not in _TILE_ROWS or bk not in (64, 128) or gx < 0 or not 0 <= pad <= 8192:
        return None
    return Plan(sched, bm, bk, gx, pad)


def _plan(rows: int, experts: int, K: int, N: int) -> Plan:
    """Kernel configuration for a call (mean rows per expert and K).

    Measured on M5 Ultra (real routing profiles, Qwen3.8 / GLM-5.3 /
    MiMo-V2.6 expert shapes at 1024-8192-token chunks):

    - fewer than 36 rows per expert, or K < 1024 (a short down projection)
      below 120 rows: weight streaming dominates; 64-row db tiles in mlx's
      (column, tile) layout;
    - 36-47 rows: 64-row db tiles, tile-on-x layout (+10%);
    - 48-95 rows: 96-row db tiles, tile-on-x layout (+3-26%);
    - 96+ rows (K >= 1024): 128-row seg tiles with 128-deep K steps and 8 KB
      of extra threadgroup memory (fewer resident threadgroups), tile-on-x
      layout (+5-29%);
    - K < 1024 from 120 rows: 96-row seg tiles, 128-deep K steps (+5%).

    Ragged K or N keeps 64-row seg tiles in the plain layout.
    """
    forced = os.environ.get(_ENV_PLAN, "").strip()
    if forced:
        plan = _parse_plan(forced)
        if plan is not None:
            return plan
    if K % 64 or N % 64:
        return Plan(_SCHED_SEG, 64, 64, 0, 0)
    per_expert = rows / max(1, experts)
    if per_expert < 36 or (K < 1024 and per_expert < 120):
        return Plan(_SCHED_DB, 64, 64, 0, 0)
    if K < 1024:
        return Plan(_SCHED_SEG, 96, 128, _GX, 0)
    if per_expert < 48:
        return Plan(_SCHED_DB, 64, 64, _GX, 0)
    if per_expert < 96:
        return Plan(_SCHED_DB, 96, 64, _GX, 0)
    return Plan(_SCHED_SEG, 128, 128, _GX, 8192)


# Affine widths the kernels unpack (mlx's packed layouts, see TileLoader).
# Kernel capability only: the route admits the widths in ``affine_bits()``.
AFFINE_KERNEL_BITS = (2, 3, 4, 5, 6, 8)


def _loader_geometry(bm: int, bk: int) -> tuple[int, int]:
    """``(kLT, kVPT)`` of the Metal ``Geo<BM, BK>``: loader threads and the
    weight values each one dequantizes per K step."""
    threads = (bm // 32) * _WN * 32
    if threads & (threads - 1):
        lt = 256 if threads > 256 else (128 if threads > 128 else 64)
    else:
        lt = threads
    return lt, _BN * bk // lt


def _chunk_values(bits: int) -> int:
    """Values per unpack chunk (TileLoader ``kCV``)."""
    if bits & (bits - 1) == 0:
        return 32 // bits
    return 16 if bits == 6 else 32


def unpack_reference(words, bits: int, count: int) -> list:
    """The first ``count`` quantized values of a packed affine row (uint32
    ``words``), read the way TileLoader reads them: chunks of
    ``_chunk_values(bits)`` values in whole words, value j at bit j * bits
    of the little-endian stream, a straddling value taking its high bits
    from the next word.  CPU mirror of the Metal unpack (tests check it
    against ``mx.quantize``'s packing)."""
    cv = _chunk_values(bits)
    cw = cv * bits // 32
    mask = (1 << bits) - 1
    out = []
    for c in range(count // cv):
        for j in range(cv):
            bit = j * bits
            wi, sh = c * cw + bit // 32, bit % 32
            q = int(words[wi]) >> sh
            if sh + bits > 32:
                q |= int(words[wi + 1]) << (32 - sh)
            out.append(q & mask)
    return out


def geometry_ok(plan: "Plan", bits: int, group_size: int) -> bool:
    """Whether ``plan``'s loader split holds ``bits``-wide values: every
    thread's run (and group chunk) is whole uint32 words of whole unpack
    chunks (the TileLoader static_asserts).  3/5-bit need 32-value runs, so
    128-row tiles with 64-deep K steps (16 values a thread) do not fit."""
    _, vpt = _loader_geometry(plan.bm, plan.bk)
    gv = min(vpt, group_size)
    cv = _chunk_values(bits)
    return (
        (vpt * bits) % 32 == 0
        and (gv * bits) % 32 == 0
        and vpt % gv == 0
        and gv % cv == 0
    )


def _fit_plan(plan: "Plan", bits: int, group_size: int) -> "Plan":
    """``plan``, or 64-row 64-deep tiles (32 values a thread: every width)
    when its loader split does not hold ``bits``."""
    if geometry_ok(plan, bits, group_size):
        return plan
    return plan._replace(bm=64, bk=64, pad=0)


def supports(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: Optional[mx.array],
    indices: mx.array,
    group_size: int,
    bits: int,
    mode: str,
    row_map: Optional[mx.array] = None,
) -> bool:
    """True when ``sorted_gather_qmm`` handles this call (layout/dtypes).

    With ``row_map`` (uint32 ``[M]``, sorted row -> row of ``x``) the rows
    are read through the map and ``x`` may have any row count (32-bit
    element offsets: fewer than 2**32 elements)."""
    if x.dtype not in (mx.bfloat16, mx.float16):
        return False
    if x.ndim != 3 or x.shape[1] != 1 or indices.ndim != 1:
        return False
    M, K = int(indices.shape[0]), int(x.shape[2])
    if row_map is None:
        if int(x.shape[0]) != M:
            return False
    elif (
        row_map.ndim != 1
        or int(row_map.shape[0]) != M
        or row_map.dtype != mx.uint32
        or int(x.shape[0]) * K >= 2**32
    ):
        return False
    # Fewer than 8 indices would be bound as a constant buffer.
    if M < 8 or indices.dtype != mx.uint32:
        return False
    if w.ndim != 3 or w.dtype != mx.uint32:
        return False
    E, N = int(w.shape[0]), int(w.shape[1])
    # Ragged N is canaried at N % 64 == 32 only.
    if E == 0 or E > _MAX_EXPERTS or N == 0 or N % 32 or K % 32:
        return False
    if mode == "affine":
        if bits not in AFFINE_KERNEL_BITS or group_size not in (32, 64, 128):
            return False
        if biases is None or K % group_size:
            return False
        if scales.dtype != x.dtype or biases.dtype != x.dtype:
            return False
        if biases.shape != scales.shape:
            return False
    elif mode == "mxfp4":
        if bits != 4 or group_size != 32 or biases is not None:
            return False
        if scales.dtype != mx.uint8:
            return False
    else:
        return False
    if w.shape[2] * 32 != K * bits:
        return False
    return scales.shape == (E, N, K // group_size)


def _launch(
    x,
    w,
    scales,
    biases,
    indices,
    group_size,
    bits,
    mode,
    plan,
    stream,
    epi=0,
    limit=None,
    row_map=None,
    up=None,
):
    """Tile pre-pass + matmul. ``epi`` > 0 runs the activation epilogue on
    the ``[gate; up]`` rows of ``w`` (``N % 64 == 0``) and returns
    ``[M, 1, N / 2]``; with it, ``row_map`` reads the sorted rows as
    ``x[row_map]`` in place.  ``up = (w2, scales2, biases2)`` (mlx2, with
    ``epi``): ``w`` holds only the gate rows and ``w2`` the up rows."""
    scan = _get_kernel("scan")
    kind = "affine" if mode == "affine" else "fp"
    if (row_map is not None or up is not None) and not epi:
        return None
    mm = _get_kernel(
        (
            f"{kind}_act"
            + ("_split" if up is not None else "")
            + ("_map" if row_map is not None else "")
        )
        if epi
        else kind
    )
    if scan is None or mm is None:
        return None
    M, K = int(indices.shape[0]), int(x.shape[2])
    E, N = int(w.shape[0]), int(w.shape[1])
    if up is not None:
        N *= 2  # the logical [gate; up] width
    bm = plan.bm
    max_tiles = (M + bm - 1) // bm + min(E, M)
    kw = {} if stream is None else {"stream": stream}
    tiles, tile_count = scan(
        inputs=[indices, mx.array([M, E, max_tiles], dtype=mx.int32)],
        template=[("BM", bm), ("MAXE", _MAX_EXPERTS)],
        grid=(1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(max_tiles * 4,), (1,)],
        output_dtypes=[mx.uint32, mx.uint32],
        **kw,
    )
    inputs = [x, w, scales]
    template = [("T", x.dtype), ("GS", group_size)]
    if mode == "affine":
        inputs.append(biases)
        template.append(("BITS", bits))
    if up is not None:
        inputs += [up[0], up[1]] + ([up[2]] if mode == "affine" else [])
    inputs += [tiles, tile_count, mx.array([N, K], dtype=mx.int32)]
    template.append(("SCHED", int(plan.sched)))
    if epi:
        # The clip bound converts like the unfused path's Python-float
        # operand of mx.clip (float, then the activation dtype).
        bound = 0.0 if limit is None else limit
        inputs.append(mx.array(bound, dtype=x.dtype).reshape(1))
        template.append(("EPI", int(epi)))
        if row_map is not None:
            inputs.append(row_map)
    else:
        template.append(("ALIGN_N", N % _BN == 0))
    template += [
        ("ALIGN_K", K % plan.bk == 0),
        ("BM", bm),
        ("BK", plan.bk),
        ("GX", plan.gx),
        ("PAD", plan.pad),
    ]
    n_cols = (N + _BN - 1) // _BN
    if plan.gx:
        tg_grid = (plan.gx, ((max_tiles + plan.gx - 1) // plan.gx) * n_cols)
    else:
        tg_grid = (n_cols, max_tiles)
    return mm(
        inputs=inputs,
        template=template,
        grid=(tg_grid[0] * 32, tg_grid[1] * _WN, bm // 32),
        threadgroup=(32, _WN, bm // 32),
        output_shapes=[(M, 1, N // 2 if epi else N)],
        output_dtypes=[x.dtype],
        **kw,
    )[0]


def _stock_gather_qmm():
    """The raw MLX op."""
    return mx.gather_qmm


# Canary routing: an empty expert, runs spanning several tiles of every
# height, and partial tiles of every size class.
_CANARY_COUNTS = (70, 0, 5, 33, 64, 17, 140, 11)


def _canary_k(plan: Plan, align_k: bool) -> int:
    # Aligned: K % BK == 0. Unaligned: a 64-deep tail (bk 128, still
    # K % 64 == 0) or a 32-deep ragged one (bk 64).
    return 256 if align_k else (320 if plan.bk == 128 else 160)


def _canary_problem(dtype, mode, bits, group_size, N, K, x_scale=0.5):
    """Quantized [E, N, K] experts, their dequantized weights, and sorted
    canary rows ``x`` [M, 1, K] (``x_scale``: a scalar or per-row scale)."""
    E = len(_CANARY_COUNTS)
    k_w, k_x = mx.random.split(mx.random.key(0x2267), 2)
    wf = (mx.random.normal((E, N, K), key=k_w) * 0.05).astype(dtype)
    if mode == "affine":
        wq, scales, biases = mx.quantize(wf, group_size=group_size, bits=bits)
        wd = mx.dequantize(wq, scales, biases, group_size=group_size, bits=bits)
    else:
        wq, scales = mx.quantize(wf, group_size=group_size, bits=bits, mode=mode)
        biases = None
        wd = mx.dequantize(wq, scales, group_size=group_size, bits=bits, mode=mode)
    idx = mx.array(
        [e for e, n in enumerate(_CANARY_COUNTS) for _ in range(n)],
        dtype=mx.uint32,
    )
    M = int(idx.shape[0])
    x = (mx.random.normal((M, 1, K), key=k_x) * x_scale).astype(dtype)
    return wq, scales, biases, wd, x, idx


def _is_device_fault(exc) -> bool:
    """A Metal out-of-memory or GPU timeout raised while a canary ran.

    Not a verdict on the kernel: the canary's eval shares the device with
    the step, so the fault propagates to serving's device-fault recovery and
    nothing is cached; the next call runs the canary again."""
    from .served_exp import is_device_fault

    return is_device_fault(exc)


def _self_test(key: tuple) -> Optional[bool]:
    """Run one kernel instantiation on a small canary.

    K % 64 == 0 must be bit-identical to mlx's sorted kernel (correct
    there); ragged K must match an fp32 dequantized reference to bf16
    rounding. Returns None when the canary could not be evaluated here
    (e.g. while a function transformation is being traced); the caller then
    retries.
    """
    dtype, mode, bits, group_size, plan, align_n, align_k = key
    N = 128 if align_n else 96
    K = _canary_k(plan, align_k)
    try:
        wq, scales, biases, wd, x, idx = _canary_problem(
            dtype, mode, bits, group_size, N, K
        )
        out = _launch(x, wq, scales, biases, idx, group_size, bits, mode, plan, None)
        if out is None:
            return False
        if K % 64 == 0:
            ref = _stock_gather_qmm()(
                x,
                wq,
                scales,
                biases,
                rhs_indices=idx,
                transpose=True,
                group_size=group_size,
                bits=bits,
                mode=mode,
                sorted_indices=True,
            )
            ok = bool(mx.array_equal(out, ref).item())
            detail = "not bit-identical to mlx's sorted kernel"
        else:
            ref = (
                x.astype(mx.float32)
                @ wd[idx].swapaxes(-1, -2).astype(mx.float32)
            )
            err = mx.abs(out.astype(mx.float32) - ref).max().item()
            scale = mx.abs(ref).max().item()
            ok = err <= scale / 64
            detail = f"max err {err:.3g} vs fp32 reference (max {scale:.3g})"
    except Exception as e:  # noqa: BLE001
        if _is_device_fault(e):
            raise
        if "transformation" in str(e):
            return None
        logger.warning(
            "NAX sorted gather_qmm self-test raised for %s: %s", _describe(key), e
        )
        return False
    if ok:
        logger.debug("NAX sorted gather_qmm armed for %s", _describe(key))
    else:
        logger.warning(
            "NAX sorted gather_qmm disabled for %s: canary %s",
            _describe(key),
            detail,
        )
    return ok


def _describe(key: tuple) -> str:
    dtype, mode, bits, group_size, plan, align_n, align_k = key[:7]
    epi = ""
    if len(key) > 7:
        limit = key[8]
        epi = " + silu(gate) * up" if limit is None else f" + clamped SwiGLU {limit:g}"
        extras = key[9:]
        if "split" in extras:
            epi += ", split gate/up tables"
        if "map" in extras:
            epi += ", row map"
    return (
        f"{str(dtype).rsplit('.', 1)[-1]} {mode} {bits}-bit gs{group_size} "
        f"({plan.describe()}{'' if align_n else ', ragged N'}"
        f"{'' if align_k else ', K tail'}){epi}"
    )


@partial(mx.compile, shapeless=True)
def _ref_swiglu(x_gate: mx.array, x_up: mx.array) -> mx.array:
    # mlx-lm's / mlx-vlm's swiglu (SwiGLU of their SwitchGLU and of oMLX's
    # GLM DSA / DeepSeek V4 SwitchGLU).
    return nn.silu(x_gate) * x_up


@partial(mx.compile, shapeless=True)
def _ref_clamped_swiglu(x_up: mx.array, x_gate: mx.array, limit: float) -> mx.array:
    # GLM-5.3's Glm5NextClampedSwiGLU (glm5_next _clamped_swiglu).
    x_gate = mx.clip(x_gate, a_min=None, a_max=limit)
    x_up = mx.clip(x_up, a_min=-limit, a_max=limit)
    return nn.silu(x_gate) * x_up


def reference_activation(
    x_up: mx.array, x_gate: mx.array, limit: Optional[float] = None
) -> mx.array:
    """The unfused path's activation: ``silu(gate) * up`` as MLX's compiled
    kernel computes it, clamped first like GLM-5.3 when ``limit`` is set."""
    if limit is None:
        return _ref_swiglu(x_gate, x_up)
    return _ref_clamped_swiglu(x_up, x_gate, float(limit))


def _bits_equal(a: mx.array, b: mx.array) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    view = {2: mx.uint16, 4: mx.uint32}[a.dtype.size]
    return bool(mx.array_equal(a.view(view), b.view(view)).item())


def _self_test_act(key: tuple) -> Optional[bool]:
    """Run one activation-epilogue instantiation on a small canary against
    the unfused path: the plain kernel with the same configuration, split
    into gate and up, then ``reference_activation`` (bitwise, including the
    signs of zeros). The canary rows span 0.1x to 16x the plain canary's
    scale so the projections cover sigmoid's saturated tails and the clip
    bounds. None: could not be evaluated here (see ``_self_test``)."""
    dtype, mode, bits, group_size, plan, _, align_k, epi, limit = key
    K = _canary_k(plan, align_k)
    try:
        M = sum(_CANARY_COUNTS)
        row_scale = mx.power(10.0, mx.linspace(-1.0, 1.2, M)).reshape(M, 1, 1)
        wq, scales, biases, _, x, idx = _canary_problem(
            dtype, mode, bits, group_size, 2 * _BN, K, x_scale=row_scale
        )
        out = _launch(
            x, wq, scales, biases, idx, group_size, bits, mode, plan, None,
            epi=epi, limit=limit,
        )
        gate_up = _launch(
            x, wq, scales, biases, idx, group_size, bits, mode, plan, None
        )
        if out is None or gate_up is None:
            return False
        x_gate, x_up = mx.split(gate_up, 2, axis=-1)
        ok = _bits_equal(out, reference_activation(x_up, x_gate, limit))
    except Exception as e:  # noqa: BLE001
        if _is_device_fault(e):
            raise
        if "transformation" in str(e):
            return None
        logger.warning(
            "NAX gate/up activation self-test raised for %s: %s", _describe(key), e
        )
        return False
    if ok:
        logger.debug("NAX gate/up activation epilogue armed for %s", _describe(key))
    else:
        logger.warning(
            "NAX gate/up activation epilogue disabled for %s: canary not "
            "bit-identical to the unfused path",
            _describe(key),
        )
    return ok


def _self_test_act_map(key: tuple) -> Optional[bool]:
    """Run one row-mapped activation-epilogue instantiation on a canary: a
    scrambled, repeating row map over a smaller token array must reproduce
    the unmapped epilogue on the materialised rows ``x[row_map]`` bitwise
    (the same configuration's unmapped instantiation is itself checked
    against the unfused path). None: could not be evaluated here."""
    dtype, mode, bits, group_size, plan, _, align_k, epi, limit, _ = key
    K = _canary_k(plan, align_k)
    try:
        M = sum(_CANARY_COUNTS)
        T = M // 3
        tok_scale = mx.power(10.0, mx.linspace(-1.0, 1.2, T)).reshape(T, 1, 1)
        wq, scales, biases, _, _, idx = _canary_problem(
            dtype, mode, bits, group_size, 2 * _BN, K
        )
        x_tok = (
            mx.random.normal((T, 1, K), key=mx.random.key(0x70C)) * tok_scale
        ).astype(dtype)
        row_map = ((mx.arange(M, dtype=mx.uint32) * 7 + 3) % T).astype(mx.uint32)
        out = _launch(
            x_tok, wq, scales, biases, idx, group_size, bits, mode, plan, None,
            epi=epi, limit=limit, row_map=row_map,
        )
        ref = _launch(
            x_tok[row_map], wq, scales, biases, idx, group_size, bits, mode,
            plan, None, epi=epi, limit=limit,
        )
        if out is None or ref is None:
            return False
        ok = _bits_equal(out, ref)
    except Exception as e:  # noqa: BLE001
        if _is_device_fault(e):
            raise
        if "transformation" in str(e):
            return None
        logger.warning(
            "NAX gate/up row-map self-test raised for %s: %s", _describe(key), e
        )
        return False
    if ok:
        logger.debug("NAX gate/up activation epilogue armed for %s", _describe(key))
    else:
        logger.warning(
            "NAX gate/up row map disabled for %s: canary not bit-identical to "
            "the materialised rows",
            _describe(key),
        )
    return ok


def _split_canary(dtype, mode, bits, group_size, K, x_scale):
    """Gate and up tables [E, 64, K] (halves of one canary table) and rows."""
    wq, scales, biases, _, x, idx = _canary_problem(
        dtype, mode, bits, group_size, 2 * _BN, K, x_scale=x_scale
    )
    h = _BN

    def half(a, lo):
        return None if a is None else mx.contiguous(a[:, lo : lo + h])

    gate = (half(wq, 0), half(scales, 0), half(biases, 0))
    up = (half(wq, h), half(scales, h), half(biases, h))
    return gate, up, x, idx


def _self_test_split(key: tuple) -> Optional[bool]:
    """mlx2: the split-table epilogue (gate and up in separate tables, as
    the Flash-Next artifacts load them) against the path it replaces: two
    stock sorted ``mx.gather_qmm`` calls and ``reference_activation``
    (bitwise).  With ``"map"`` in the key, the row-mapped variant against
    the same kernel on the materialised rows.  None: not evaluable here."""
    dtype, mode, bits, group_size, plan, _, align_k, epi, limit = key[:9]
    mapped = "map" in key[9:]
    K = _canary_k(plan, align_k)
    try:
        M = sum(_CANARY_COUNTS)
        row_scale = mx.power(10.0, mx.linspace(-1.0, 1.2, M)).reshape(M, 1, 1)
        gate, up, x, idx = _split_canary(dtype, mode, bits, group_size, K, row_scale)
        kw = dict(epi=epi, limit=limit, up=up)
        if mapped:
            T = M // 3
            x_tok = (
                mx.random.normal((T, 1, K), key=mx.random.key(0x70C))
                * mx.power(10.0, mx.linspace(-1.0, 1.2, T)).reshape(T, 1, 1)
            ).astype(dtype)
            row_map = ((mx.arange(M, dtype=mx.uint32) * 7 + 3) % T).astype(mx.uint32)
            out = _launch(x_tok, *gate, idx, group_size, bits, mode, plan, None,
                          row_map=row_map, **kw)
            ref = _launch(x_tok[row_map], *gate, idx, group_size, bits, mode, plan,
                          None, **kw)
            if out is None or ref is None:
                return False
            ok = _bits_equal(out, ref)
        else:
            out = _launch(x, *gate, idx, group_size, bits, mode, plan, None, **kw)
            if out is None:
                return False
            stock = _stock_gather_qmm()

            def proj(t):
                return stock(
                    x, t[0], t[1], t[2], rhs_indices=idx, transpose=True,
                    group_size=group_size, bits=bits, mode=mode, sorted_indices=True,
                )

            ok = _bits_equal(out, reference_activation(proj(up), proj(gate), limit))
    except Exception as e:  # noqa: BLE001
        if _is_device_fault(e):
            raise
        if "transformation" in str(e):
            return None
        logger.warning("NAX split gate/up self-test raised for %s: %s", _describe(key), e)
        return False
    if not ok:
        logger.warning(
            "NAX split gate/up epilogue disabled for %s: canary not bit-identical",
            _describe(key),
        )
    return ok


def _checked(key: tuple, test) -> bool:
    """The cached self-test verdict for ``key`` (running ``test`` once).

    A device fault raised by ``test`` propagates and caches nothing."""
    ok = _verified.get(key)
    if ok is None:
        with _lock:
            ok = _verified.get(key)
            if ok is None:
                ok = test(key)
                if ok is not None:
                    _verified[key] = ok
    return bool(ok)


def sorted_gather_qmm(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: Optional[mx.array],
    indices: mx.array,
    *,
    group_size: int,
    bits: int,
    mode: str = "affine",
    stream=None,
    plan: Optional[Plan] = None,
    verify: bool = True,
) -> Optional[mx.array]:
    """``x @ w[indices].T`` for sorted rows on the tensor units.

    ``indices`` must group each expert's rows in one contiguous run (see
    the module docstring). ``plan`` pins a configuration (testing); by
    default ``_plan`` picks one. Returns None when the module is disabled,
    the call is not supported (see ``supports``), the kernels cannot be
    built or the instantiation failed its one-time self-test; the caller
    then keeps the stock path.
    """
    if not enabled() or not supports(
        x, w, scales, biases, indices, group_size, bits, mode
    ):
        return None
    M, K = int(x.shape[0]), int(x.shape[2])
    E, N = int(w.shape[0]), int(w.shape[1])
    if plan is None:
        plan = _plan(M, E, K, N)
    plan = _fit_plan(plan, bits, group_size)
    if plan.sched == _SCHED_DB and (K % 64 or N % 64 or plan.bk != 64):
        # db runs aligned 64-deep K steps only.
        plan = plan._replace(sched=_SCHED_SEG)
    if verify:
        key = (x.dtype, mode, bits, group_size, plan, N % _BN == 0, K % plan.bk == 0)
        if not _checked(key, _self_test):
            return None
    return _launch(x, w, scales, biases, indices, group_size, bits, mode, plan, stream)


def sorted_gather_qmm_swiglu(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: Optional[mx.array],
    indices: mx.array,
    *,
    group_size: int,
    bits: int,
    mode: str = "affine",
    limit: Optional[float] = None,
    stream=None,
    plan: Optional[Plan] = None,
    verify: bool = True,
    row_map: Optional[mx.array] = None,
) -> Optional[mx.array]:
    """The SwiGLU of a fused gate/up projection for sorted rows, in one kernel.

    ``w`` (with ``scales``/``biases``) holds each expert's gate rows followed
    by its up rows, ``[E, 2 * n, K]``. Returns ``[M, 1, n]``: for
    ``gate, up = split(sorted_gather_qmm(x, w, ...), 2, axis=-1)`` the
    activation ``silu(gate) * up``, or with ``limit`` GLM-5.3's clamped
    ``silu(minimum(gate, limit)) * clip(up, -limit, limit)``. The matmul
    computes gate and up of the same columns in each output tile (weight
    rows paired in the tile loader) and applies the activation to the
    rounded projections in its epilogue instead of writing ``[M, 2 * n]``
    for a separate elementwise pass. Bit-identical to the unfused path:
    each instantiation must match the plain kernel + split +
    ``reference_activation`` on a canary (and the plain kernel its own
    self-test) before it is used.

    With ``row_map`` (uint32 ``[M]``) the sorted rows are ``x[row_map]``
    (``x`` holds the token rows, ``[T, 1, K]``), read in place instead of
    from a replicated ``[M, 1, K]`` copy: the same tiles, values and tensor
    ops, so the output is bit-identical to passing ``x[row_map]`` (checked
    per instantiation on a scrambled canary map).

    Returns None when disabled (``OMLX_M5_GATHER_QMM_NAX=0``), unsupported
    (``supports``, or ``2 * n % 64 != 0``), or not verified; the caller then
    keeps the unfused path (materialising ``x[row_map]``).
    """
    if not enabled() or not supports(
        x, w, scales, biases, indices, group_size, bits, mode, row_map
    ):
        return None
    M, K = int(indices.shape[0]), int(x.shape[2])
    E, N = int(w.shape[0]), int(w.shape[1])
    if N % _BN:
        # Every 64-column tile pairs 32 gate with 32 up columns.
        return None
    if limit is not None:
        limit = float(limit)
        if not math.isfinite(limit):
            return None
    if plan is None:
        plan = _plan(M, E, K, N)
    plan = _fit_plan(plan, bits, group_size)
    if plan.sched == _SCHED_DB and (K % 64 or plan.bk != 64):
        plan = plan._replace(sched=_SCHED_SEG)
    epi = _EPI_SWIGLU if limit is None else _EPI_CLAMPED
    if verify:
        key = (x.dtype, mode, bits, group_size, plan, True, K % plan.bk == 0)
        # The plain instantiation (the unfused path's kernel) must hold too.
        if not _checked(key, _self_test) or not _checked(
            key + (epi, limit), _self_test_act
        ):
            return None
        if row_map is not None and not _checked(
            key + (epi, limit, "map"), _self_test_act_map
        ):
            return None
    return _launch(
        x, w, scales, biases, indices, group_size, bits, mode, plan, stream,
        epi=epi, limit=limit, row_map=row_map,
    )



def sorted_gather_qmm_swiglu_split(
    x: mx.array,
    gate: tuple,
    up: tuple,
    indices: mx.array,
    *,
    group_size: int,
    bits: int,
    mode: str = "affine",
    stream=None,
    plan: Optional[Plan] = None,
    verify: bool = True,
    row_map: Optional[mx.array] = None,
) -> Optional[mx.array]:
    """mlx2: ``silu(gate) * up`` of two separate expert tables in one kernel.

    ``gate`` and ``up`` are ``(weight, scales, biases)`` of ``[E, n, K]``
    tables (the split projections Flash-Next serves).  Bit-identical to
    ``swiglu(gather_qmm(x, *gate), gather_qmm(x, *up))`` with both gathers
    sorted (checked per instantiation on a canary), without concatenating
    the tables at load.  ``row_map`` as in ``sorted_gather_qmm_swiglu``.
    None: unsupported or unverified; the caller keeps the two gathers.
    """
    wg, sg, bg = gate
    wu, su, bu = up
    if not enabled():
        return None
    if wu.shape != wg.shape or wu.dtype != wg.dtype or su.shape != sg.shape:
        return None
    if su.dtype != sg.dtype or (bu is None) != (bg is None):
        return None
    if bg is not None and (bu.shape != bg.shape or bu.dtype != bg.dtype):
        return None
    for w_, s_, b_ in (gate, up):
        if not supports(x, w_, s_, b_, indices, group_size, bits, mode, row_map):
            return None
    M, K = int(indices.shape[0]), int(x.shape[2])
    E, N = int(wg.shape[0]), 2 * int(wg.shape[1])
    if N % _BN:
        return None
    if plan is None:
        plan = _plan(M, E, K, N)
    plan = _fit_plan(plan, bits, group_size)
    if plan.sched == _SCHED_DB and (K % 64 or plan.bk != 64):
        plan = plan._replace(sched=_SCHED_SEG)
    if verify:
        key = (x.dtype, mode, bits, group_size, plan, True, K % plan.bk == 0,
               _EPI_SWIGLU, None)
        if not _checked(key + ("split",), _self_test_split):
            return None
        if row_map is not None and not _checked(key + ("split", "map"), _self_test_split):
            return None
    return _launch(
        x, wg, sg, bg, indices, group_size, bits, mode, plan, stream,
        epi=_EPI_SWIGLU, row_map=row_map, up=(wu, su, bu),
    )

# ---------------------------------------------------------------------------
# mlx2 policy, admission and counters
# ---------------------------------------------------------------------------

ENV_MODE = "MLX2_MOE_NAX_GATHER"
# off: stock MLX (default).  gather: sorted quantized expert gathers run the
# segmented NAX kernel.  fused: gather, plus the fused [gate; up] projection
# of FusedGateUpSwitchGLU runs SwiGLU in the epilogue and reads its rows
# through the sorted row map (no x[order // k] copy).
MODES = ("off", "gather", "fused")


def mode_from_env(environ=None) -> str:
    environ = os.environ if environ is None else environ
    raw = (environ.get(ENV_MODE, "off") or "off").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return "off"
    if raw in {"1", "true", "yes", "on"}:
        return "fused"
    if raw not in MODES:
        raise ValueError(f"{ENV_MODE} must be one of {MODES}, got {raw!r}")
    return raw


MODE = mode_from_env()

ENV_BITS = "MLX2_MOE_NAX_GATHER_BITS"
# Affine widths the route admits (MXFP4 is always 4-bit).  The kernels
# unpack every width in AFFINE_KERNEL_BITS bit-exactly (canaried per
# instantiation); the route default is the widths measured faster than the
# stock kernel on M5: 6-bit 1.29-1.56x per call and 1.071x on the
# Qwen3.6-35B-A3B 8K prefill (qualification/runs/research-20261006/nax-6bit/).
# 2/3/5-bit are canary-exact but unmeasured, so not admitted by default.
DEFAULT_AFFINE_BITS = (4, 6, 8)


def affine_bits_from_env(environ=None) -> tuple:
    """The admitted affine widths: ``MLX2_MOE_NAX_GATHER_BITS`` as a comma
    list (e.g. ``4,6,8``), else ``DEFAULT_AFFINE_BITS``."""
    environ = os.environ if environ is None else environ
    raw = (environ.get(ENV_BITS, "") or "").strip()
    if not raw:
        return DEFAULT_AFFINE_BITS
    try:
        bits = tuple(sorted({int(b) for b in raw.split(",") if b.strip()}))
    except ValueError:
        bits = ()
    if not bits or any(b not in AFFINE_KERNEL_BITS for b in bits):
        raise ValueError(
            f"{ENV_BITS} must be a comma list of {AFFINE_KERNEL_BITS}, got {raw!r}"
        )
    return bits


AFFINE_BITS = affine_bits_from_env()


def set_affine_bits(bits) -> tuple:
    """Switch the admitted affine widths at run time (in-process A/B);
    returns the old tuple."""
    global AFFINE_BITS
    new = tuple(sorted({int(b) for b in bits}))
    if not new or any(b not in AFFINE_KERNEL_BITS for b in new):
        raise ValueError(f"affine bits must be a subset of {AFFINE_KERNEL_BITS}")
    old, AFFINE_BITS = AFFINE_BITS, new
    return old


def bits_admitted(mode: str, bits: int) -> bool:
    """The route's width admission (kernel capability is ``supports``)."""
    return mode != "affine" or int(bits) in AFFINE_BITS

# Observed-use counters (plain ints, no sync): engaged launches per route,
# declines of calls that were candidates (the reason is the key), and calls
# that are not candidates (below MLX's own rhs floor: they would not run the
# row-block kernel either, so nothing is lost).
calls = {"gather": 0, "swiglu": 0, "swiglu_map": 0, "swiglu_split": 0,
         "swiglu_split_map": 0}
# The same engaged launches split by weight format: "<route>/<mode>-<bits>".
calls_by_format: dict = {}
fallbacks: dict = {}
not_candidates = 0
# Sorted gathers of a non-prefill forward (decode, MTP drafting, verify):
# never offered to the kernel, whatever their (padded) row count.
not_prefill = 0
last_fallback: Optional[str] = None


# Execution phase (Codex port review 2026-10-02 item 7).  The port's contract
# is prefill only; admission by row count alone let a padded batched decode
# in (32 one-token lanes x top-10 of 512 experts = 320 rows, padded by the
# adaptive/always rhs pad to the 2048-row streaming floor).  A trunk forward
# opens ``forward_scope``: True for a prefill forward, False otherwise; the
# MoE call sites admit NAX only while it is True.  None = no trunk decided
# (a bare module call): not prefill.  Nested trunk calls keep the outer
# decision, as the invariant lane does.
_PREFILL: ContextVar[Optional[bool]] = ContextVar("mlx2_moe_nax_prefill", default=None)


def prefill_active() -> bool:
    """Whether the current forward is a prefill forward (NAX admissible)."""
    return _PREFILL.get() is True


@contextmanager
def prefill_scope(enabled: bool = True):
    """Mark the enclosed calls as one prefill forward (or not)."""
    token = _PREFILL.set(bool(enabled))
    try:
        yield
    finally:
        _PREFILL.reset(token)


def is_prefill_forward(inputs, cache) -> bool:
    """A trunk forward over more than one row per lane that is not a
    speculative verify window (verify scope, a speculating cache, or a
    prepared segmented verify block).  Decode at any batch width is not."""
    from .invariant_prefill import prefill_decline_reason

    if prefill_decline_reason(inputs, cache) is not None:
        return False
    if cache is not None and any(
        getattr(entry, "_step_lengths", None) is not None
        for entry in cache
        if entry is not None
    ):
        return False
    return True


@contextmanager
def forward_scope(inputs, cache):
    """Open the phase for one trunk forward (no-op while off or nested)."""
    if MODE == "off" or _PREFILL.get() is not None:
        yield
        return
    token = _PREFILL.set(is_prefill_forward(inputs, cache))
    try:
        yield
    finally:
        _PREFILL.reset(token)


def admit_phase() -> bool:
    """Call-site gate: True inside a prefill forward, else counted."""
    global not_prefill
    if _PREFILL.get() is True:
        return True
    not_prefill += 1
    return False


def set_mode(mode: str) -> str:
    """Switch the route at run time (in-process A/B); returns the old mode."""
    global MODE
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    old, MODE = MODE, mode
    return old


def status(*, reset: bool = False) -> dict:
    """Mode, host gate, engaged calls, fallbacks with reasons, armed kernels."""
    global not_candidates, not_prefill, last_fallback
    out = {
        "mode": MODE,
        "nax_host": nax_host() if MODE != "off" else _nax_host,
        "device_name": _device_name,
        "calls": dict(calls),
        "calls_by_format": dict(calls_by_format),
        "affine_bits": list(AFFINE_BITS),
        "fallbacks": dict(fallbacks),
        "not_candidates": not_candidates,
        "not_prefill": not_prefill,
        "last_fallback": last_fallback,
        "verified": {_describe(k): v for k, v in _verified.items()},
    }
    if reset:
        for k in calls:
            calls[k] = 0
        calls_by_format.clear()
        fallbacks.clear()
        not_candidates = 0
        not_prefill = 0
        last_fallback = None
    return out


def _fallback(reason: str) -> None:
    global last_fallback
    fallbacks[reason] = fallbacks.get(reason, 0) + 1
    last_fallback = reason


def _engaged(route: str, mode: str, bits: int) -> None:
    calls[route] += 1
    key = f"{route}/{mode}-{int(bits)}"
    calls_by_format[key] = calls_by_format.get(key, 0) + 1


def rows_ok(rows: int, experts: int) -> bool:
    """MLX's own choice of the sorted rhs kernel (GatherQMM::eval_gpu:
    B >= 16 and B / E >= 4).  Smaller calls run ``gather_qmv``, whose
    summation order they (MTP verify among them) depend on."""
    return rows >= 16 and experts > 0 and rows // experts >= 4


def _as_u32(indices):
    return indices if indices.dtype == mx.uint32 else indices.astype(mx.uint32)


def try_gather(x, layer, indices) -> Optional[mx.array]:
    """``gather_qmm(x, layer, indices, sorted_indices=True)`` on the NAX
    kernel, or None (counted) when the caller must run the stock op.

    ``layer`` is a resident ``QuantizedSwitchLinear``; the caller has
    already chosen the native sorted policy and applied any rhs pad."""
    global not_candidates
    w = layer.get("weight")
    rows = int(indices.size)
    if not rows_ok(rows, int(w.shape[0])):
        not_candidates += 1
        return None
    if not nax_host():
        _fallback("not_nax_host")
        return None
    if mx.default_device() != mx.gpu:
        _fallback("cpu_device")
        return None
    if not bits_admitted(layer.mode, layer.bits):
        _fallback("bits_not_admitted")
        return None
    idx = _as_u32(indices)
    if not supports(
        x, w, layer["scales"], layer.get("biases"), idx,
        layer.group_size, layer.bits, layer.mode,
    ):
        _fallback("unsupported_layout")
        return None
    out = sorted_gather_qmm(
        x, w, layer["scales"], layer.get("biases"), idx,
        group_size=layer.group_size, bits=layer.bits, mode=layer.mode,
    )
    if out is None:
        _fallback("kernel_unverified")
        return None
    _engaged("gather", layer.mode, layer.bits)
    return out


def try_swiglu(proj, x, indices, token_rows=None) -> Optional[mx.array]:
    """``silu(gate) * up`` of a fused ``[gate; up]`` projection in one
    kernel, or None (counted).  ``x`` / ``indices`` are the sorted rows;
    ``token_rows = (x_tok, row_map)`` with ``x == x_tok[row_map]`` lets the
    kernel read them in place so a lazy ``x`` is never computed."""
    global not_candidates
    w = proj.get("weight")
    rows = int(indices.size)
    if not rows_ok(rows, int(w.shape[0])):
        not_candidates += 1
        return None
    if not nax_host():
        _fallback("not_nax_host")
        return None
    if mx.default_device() != mx.gpu:
        _fallback("cpu_device")
        return None
    if not bits_admitted(proj.mode, proj.bits):
        _fallback("bits_not_admitted")
        return None
    idx = _as_u32(indices)
    kw = dict(group_size=int(proj.group_size), bits=int(proj.bits), mode=proj.mode)
    if token_rows is not None:
        x_tok, row_map = token_rows
        out = sorted_gather_qmm_swiglu(
            x_tok, w, proj["scales"], proj.get("biases"), idx,
            row_map=_as_u32(row_map), **kw
        )
        if out is not None:
            _engaged("swiglu_map", proj.mode, proj.bits)
            return out
        _fallback("row_map_declined")
    out = sorted_gather_qmm_swiglu(x, w, proj["scales"], proj.get("biases"), idx, **kw)
    if out is None:
        _fallback("swiglu_declined")
        return None
    _engaged("swiglu", proj.mode, proj.bits)
    return out


def _table(proj):
    return (proj["weight"], proj["scales"], proj.get("biases"))


def try_swiglu_split(gate_proj, up_proj, x, indices, token_rows=None) -> Optional[mx.array]:
    """``silu(gate) * up`` of separate gate and up tables in one kernel, or
    None (counted); see ``try_swiglu``."""
    global not_candidates
    w = gate_proj.get("weight")
    rows = int(indices.size)
    if not rows_ok(rows, int(w.shape[0])):
        not_candidates += 1
        return None
    if not nax_host():
        _fallback("not_nax_host")
        return None
    if mx.default_device() != mx.gpu:
        _fallback("cpu_device")
        return None
    if (gate_proj.group_size, gate_proj.bits, gate_proj.mode) != (
        up_proj.group_size, up_proj.bits, up_proj.mode
    ):
        _fallback("gate_up_formats_differ")
        return None
    if not bits_admitted(gate_proj.mode, gate_proj.bits):
        _fallback("bits_not_admitted")
        return None
    idx = _as_u32(indices)
    kw = dict(group_size=int(gate_proj.group_size), bits=int(gate_proj.bits),
              mode=gate_proj.mode)
    gate, up = _table(gate_proj), _table(up_proj)
    if token_rows is not None:
        x_tok, row_map = token_rows
        out = sorted_gather_qmm_swiglu_split(x_tok, gate, up, idx,
                                             row_map=_as_u32(row_map), **kw)
        if out is not None:
            _engaged("swiglu_split_map", gate_proj.mode, gate_proj.bits)
            return out
        _fallback("row_map_declined")
    out = sorted_gather_qmm_swiglu_split(x, gate, up, idx, **kw)
    if out is None:
        _fallback("swiglu_declined")
        return None
    _engaged("swiglu_split", gate_proj.mode, gate_proj.bits)
    return out

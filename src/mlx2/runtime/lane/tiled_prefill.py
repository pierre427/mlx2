# SPDX-License-Identifier: Apache-2.0
# Adapted from mlx-tree-spec, Copyright 2026 Aditya Gajbhiye.
# MLX NAX prefill algorithm/helpers: MIT, Copyright Apple Inc.
"""Prefill directly from existing tiled affine q4 weights.

See provenance/mlx-tree-spec-20261027.json. No alternative weight bank or
runtime dependency is introduced. Installed MLX headers supply their own
NAX types; this candidate is opt-in and unsupported formats retain the
existing projection path.
"""
from functools import lru_cache
from collections import Counter
import os
import mlx.core as mx

_MSL_TYPE = {mx.bfloat16: "bfloat", mx.float16: "half"}
ENV = "MLX2_TILED_PREFILL"
STATS = Counter()

def enabled():
    value = os.environ.get(ENV, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{ENV} must be 0 or 1")
    return value == "1"

def status():
    return {"implemented": True, "selected": enabled(), "qualified": False, **STATS}

def admit_model(model):
    """Adapter-owned admission; no other model inherits this environment flag."""
    selected = enabled()
    for _, module in model.named_modules():
        object.__setattr__(module, "_direct_prefill_candidate", selected)
    return selected


def project(x, module, nt):
    if not enabled():
        return None
    n, kw = module.weight.shape
    k = x.shape[-1]
    if (module.bits != 4 or module.group_size != 64 or nt not in (0, 32, 64)
            or n % 64 or k != kw * 8 or k % 64
            or x.dtype not in _MSL_TYPE or module.scales.dtype != x.dtype
            or module.biases.dtype != x.dtype or mx.default_device() != mx.gpu
            or not mx.metal.is_available() or "M5" not in mx.device_info().get("device_name", "")):
        STATS["declined"] += 1
        return None
    tile = (64, 64, 128 if k >= 16384 and k % 128 == 0 else 64, 2, 2)
    result = prefill_kernel(k, n, 64, x.dtype, tile, nt)(
        mx.contiguous(x), module.weight, mx.contiguous(module.scales), mx.contiguous(module.biases))
    STATS["calls"] += 1
    STATS["ordinary_layout_calls" if nt == 0 else "tiled_layout_calls"] += 1
    STATS["rows"] += x.size // k
    return result

_STEEL_HEADERS = (
    "steel/defines.h",
    "steel/utils/type_traits.h",
    "steel/utils/integral_constant.h",
    "steel/gemm/nax.h",
)

_PREFILL_IMPL = """
using namespace mlx::steel;

template <typename T, int K, int N, int GS, int BM, int BN, int BK, int WM, int WN, int NTCOL, bool aligned_m>
METAL_FUNC void qmm_prefill_packed(
    const device T* x, const device uint32_t* wq2, const device T* sp2, const device T* bp2,
    device T* y, threadgroup T* Ws, const int M, const int sgp_sm,
    const int y_row, const int y_col, const ushort sg, const ushort lane) {
  constexpr int BK_PAD = BK + 16 / sizeof(T);
  constexpr int KB16 = K / 16;
  constexpr int G = K / GS;
  constexpr int NT = WM * WN * 32;
  constexpr int ITEMS = BN * (BK / 16);       // (row, 16-wide K block) pairs per step
  static_assert(ITEMS % NT == 0, "loader items must split evenly");
  constexpr int LOADS = ITEMS / NT;
  constexpr int SM = BM / WM, SN = BN / WN, SK = 32;
  constexpr int TM = SM / 16, TN = SN / 16, TK = SK / 16;
  static_assert(GS % BK == 0 || BK % GS == 0, "group size vs BK");

  const int t = int(sg) * 32 + int(lane);
  int rows[LOADS];
  threadgroup T* wdst[LOADS];
  int kbo[LOADS];
  _Pragma("unroll")
  for (int i = 0; i < LOADS; ++i) {
    const int w = t + i * NT;
    const int r32 = w % 32, rest = w / 32;
    const int toff = rest % (BN / 32);
    kbo[i] = rest / (BN / 32);
    const int tile = y_col / 32 + toff;
    rows[i] = tile * 32 + r32;
    wdst[i] = Ws + (toff * 32 + r32) * BK_PAD + kbo[i] * 16;
  }

  const short tm = SM * (sg / WN);
  const short tn = SN * (sg % WN);
  x += size_t(y_row + tm) * K;

  NAXTile<float, TM, TN> Dtile;
  Dtile.clear();

  for (int k = 0; k < K; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    _Pragma("unroll")
    for (int i = 0; i < LOADS; ++i) {
      const int kk = k + kbo[i] * 16;
      const int g = kk / GS;
      const int row = rows[i];
      const size_t word = ((size_t(row / NTCOL) * G + g) * NTCOL + row % NTCOL) * (GS / 8) + (kk % GS) / 8;
      const uint2 p = *((const device uint2*)(wq2 + word));
      const float s = float(sp2[row * G + g]);
      const float b = float(bp2[row * G + g]);
      _Pragma("unroll")
      for (int ki = 0; ki < 8; ++ki) {
        wdst[i][ki] = T(float((p.x >> (ki * 4)) & 0xFu) * s + b);
        wdst[i][8 + ki] = T(float((p.y >> (ki * 4)) & 0xFu) * s + b);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_NO_UNROLL
    for (int kk1 = 0; kk1 < BK; kk1 += SK) {
      NAXTile<T, TM, TK> Atile;
      NAXTile<T, TN, TK> Btile;
      volatile int compiler_barrier;
      if (aligned_m) {
        Atile.load(x + kk1, K);
      } else {
        Atile.load_safe(x + kk1, K, short2(SK, sgp_sm));
      }
      Btile.template load<T, BK_PAD, 1>(Ws + tn * BK_PAD + kk1);
      tile_matmad_nax(Dtile, Atile, metal::bool_constant<false>{}, Btile, metal::bool_constant<true>{});
      (void)compiler_barrier;
    }
    x += BK;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  device T* yt = y + size_t(y_row + tm) * N + y_col + tn;
  if (aligned_m) {
    Dtile.store(yt, N);
  } else if (sgp_sm > 0) {
    Dtile.store_safe(yt, N, short2(SN, sgp_sm));
  }
}
"""

_PREFILL_SOURCE = """
threadgroup T Ws[BN * (BK + 16 / sizeof(T))];
const int M = int(M_size);
const int y_row = int(threadgroup_position_in_grid.y) * BM;
const int y_col = int(threadgroup_position_in_grid.x) * BN;
const ushort sg = simdgroup_index_in_threadgroup;
const ushort lane = thread_index_in_simdgroup;
const int sgp_sm = min(BM / WM, M - (y_row + (BM / WM) * int(sg / WN)));
if (sgp_sm == BM / WM) {
  qmm_prefill_packed<T, K, N, GS, BM, BN, BK, WM, WN, NTCOL, true>(x, wq2, sp2, bp2, y, Ws, M, sgp_sm, y_row, y_col, sg, lane);
} else {
  qmm_prefill_packed<T, K, N, GS, BM, BN, BK, WM, WN, NTCOL, false>(x, wq2, sp2, bp2, y, Ws, M, sgp_sm, y_row, y_col, sg, lane);
}
"""


@lru_cache(maxsize=None)
def _steel_header() -> str:
    import os
    import re

    root = os.path.join(os.path.dirname(mx.__file__), "include", "mlx", "backend", "metal", "kernels")
    parts = []
    for rel in _STEEL_HEADERS:
        with open(os.path.join(root, rel)) as f:
            src = f.read()
        parts.append(re.sub(r'^\s*(#pragma once|#include "mlx/.*")\s*$', "", src, flags=re.MULTILINE))
    return "\n".join(parts) + _PREFILL_IMPL


PREFILL_TILE = (64, 64, 32, 2, 2)  # (BM, BN, BK, WM, WN): MLX's affine_qmm_t_nax default


@lru_cache(maxsize=None)
def prefill_kernel(k: int, n: int, group_size: int, dtype, tile: tuple = PREFILL_TILE, ntcol: int = 32):
    """y = x @ W.T for any number of rows of x (..., K), W tiled by lane_qmm.tile_weight()."""
    bm, bn, bk, wm, wn = tile
    assert n % bn == 0 and k % bk == 0 and bn % 32 == 0 and bk % 16 == 0
    consts = (
        f"typedef {_MSL_TYPE[dtype]} T;\n"
        + "".join(f"constexpr int {a} = {v};\n" for a, v in
                  (("K", k), ("N", n), ("GS", group_size), ("BM", bm), ("BN", bn), ("BK", bk), ("WM", wm), ("WN", wn), ("NTCOL", ntcol)))
    )
    kern = mx.fast.metal_kernel(
        name=f"mlx2_tiled_prefill_k{k}_n{n}_g{group_size}_{bm}x{bn}x{bk}_{wm}x{wn}_{_MSL_TYPE[dtype]}_nt{ntcol}",
        input_names=["x", "wq2", "sp2", "bp2", "M_size"],
        output_names=["y"],
        source=consts + _PREFILL_SOURCE,
        header=(_steel_header() if ntcol else _steel_header().replace(
            "((size_t(row / NTCOL) * G + g) * NTCOL + row % NTCOL) * (GS / 8) + (kk % GS) / 8",
            "size_t(row) * (K / 8) + kk / 8")),
    )
    nt = 32 * wm * wn

    def run(x, wq2, sp2, bp2):
        m = x.size // k
        return kern(inputs=[x, wq2, sp2, bp2, m], grid=(nt * (n // bn), -(-m // bm), 1), threadgroup=(nt, 1, 1),
                    output_shapes=[(*x.shape[:-1], n)], output_dtypes=[dtype])[0]

    return run

# SPDX-License-Identifier: MIT
"""Multi-row dense ``x @ W.T`` with MLX's one-row gemv arithmetic per row.

MLX 0.32.4 runs a one-row ``x @ W.T`` (W ``[N, K]`` row-major, float/bf16)
through its ``gemv`` kernel (``kernels/gemv.h`` ``GEMVKernel::run``) with the
tile chosen in ``matmul.cpp::gemv_axbpy``: ``BM = N >= 4096 ? 8 : 4``,
``BN = 1``, ``SM = 1``, ``SN = 32``, ``TM = 4``, ``TN = 4`` (other branches for
``K <= 64`` or ``K >= 16 N``).  For two or more rows MLX switches to
``gemv_wide`` or a GEMM tile, whose reductions differ, so the exact verify
strip (``row_exact.ops.matvec_rows``) instead ran the one-row kernel once per
row through broadcast batching, re-reading ``W`` for every row.

This kernel is ``GEMVKernel::run`` for that tile with ``ROWS`` input vectors:
each K block's ``TM x TN`` weight tile is loaded once and applied to every
row, and each row's partial sums are the same ``result[tm] += inter[tn] *
v_coeff[tn]`` sequence, in the same block and lane order, finished by the
same ``simd_shuffle_down`` tree.  Opt-in (``MLX2_GEMV_ROWS_KERNEL=fused``);
the first real use of each (shape, dtype, rows) is checked bit for bit
against the stock per-row form, and a mismatch keeps the stock form.
Provenance: provenance/gemv-rows-exact.json.
"""

from __future__ import annotations

import os
import threading

import mlx.core as mx

MAX_ROWS = 16
_LOCK = threading.Lock()
_KERNEL = None
_VERDICTS: dict = {}

_SOURCE = r"""
  constexpr int TN = 4;
  constexpr int SN = 32;
  constexpr int blockM = BM * TM;   // threadsM = BM * SM, SM = 1
  constexpr int blockN = SN * TN;   // threadsN = BN * SN, BN = 1
  const int K = params[0];
  const int N = params[1];
  const uint simd_gid = simdgroup_index_in_threadgroup;
  const uint simd_lid = thread_index_in_simdgroup;

  thread float result[ROWS][TM];
  for (int r = 0; r < ROWS; ++r)
    for (int tm = 0; tm < TM; ++tm)
      result[r][tm] = 0;
  thread T inter[TN];
  thread float v_coeff[ROWS][TN];

  const int thrN = int(simd_lid);
  const int simdM = int(simd_gid);
  int bm = simdM * TM;
  int bn = thrN * TN;
  int out_row = int(threadgroup_position_in_grid.x) * blockM + bm;
  if (out_row >= N)
    return;
  out_row = out_row + TM <= N ? out_row : N - TM;
  const device T* m = mat + size_t(out_row) * K;

  const int n_iter = K / blockN;
  const int last_iter = blockN * n_iter;
  const int leftover = K - last_iter;

  for (int i = 0; i < n_iter; ++i) {
    for (int r = 0; r < ROWS; ++r) {
      MLX_MTL_PRAGMA_UNROLL
      for (int tn = 0; tn < TN; tn++)
        v_coeff[r][tn] = static_cast<float>(x[size_t(r) * K + bn + tn]);
    }
    int mat_offset = 0;
    MLX_MTL_PRAGMA_UNROLL
    for (int tm = 0; tm < TM; tm++) {
      MLX_MTL_PRAGMA_UNROLL
      for (int tn = 0; tn < TN; tn++)
        inter[tn] = m[mat_offset + bn + tn];
      for (int r = 0; r < ROWS; ++r) {
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < TN; tn++)
          result[r][tm] += inter[tn] * v_coeff[r][tn];
      }
      mat_offset += K;
    }
    bn += blockN;
  }

  if (leftover > 0) {
    for (int r = 0; r < ROWS; ++r) {
      if (bn + TN <= K) {
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < TN; tn++)
          v_coeff[r][tn] = static_cast<float>(x[size_t(r) * K + bn + tn]);
      } else {
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < TN; tn++)
          v_coeff[r][tn] = bn + tn < K
              ? static_cast<float>(x[size_t(r) * K + bn + tn]) : 0.0f;
      }
    }
    MLX_MTL_PRAGMA_UNROLL
    for (int tm = 0; tm < TM; tm++) {
      const device T* row = &m[tm * K];
      if (bn + TN <= K) {
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < TN; tn++)
          inter[tn] = row[bn + tn];
      } else {
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < TN; tn++)
          inter[tn] = bn + tn < K ? row[bn + tn] : T(0);
      }
      for (int r = 0; r < ROWS; ++r) {
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < TN; tn++)
          result[r][tm] += inter[tn] * v_coeff[r][tn];
      }
    }
  }

  for (int r = 0; r < ROWS; ++r) {
    MLX_MTL_PRAGMA_UNROLL
    for (int tm = 0; tm < TM; tm++) {
      MLX_MTL_PRAGMA_UNROLL
      for (ushort sn = (SN / 2); sn >= 1; sn >>= 1)
        result[r][tm] += simd_shuffle_down(result[r][tm], sn);
    }
  }
  if (thrN == 0) {
    for (int r = 0; r < ROWS; ++r) {
      MLX_MTL_PRAGMA_UNROLL
      for (int tm = 0; tm < TM; tm++)
        out[size_t(r) * N + out_row + tm] = static_cast<T>(result[r][tm]);
    }
  }
"""

_HEADER = """
#define MLX_MTL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
"""


def _kernel():
    global _KERNEL
    with _LOCK:
        if _KERNEL is None:
            _KERNEL = mx.fast.metal_kernel(
                name="mlx2_gemv_rows_exact",
                input_names=["mat", "x", "params"],
                output_names=["out"],
                header=_HEADER,
                source=_SOURCE,
            )
        return _KERNEL


def kernel_mode() -> str:
    """``MLX2_GEMV_ROWS_KERNEL``: ``loop`` (default, stock per row) or ``fused``."""
    mode = os.environ.get("MLX2_GEMV_ROWS_KERNEL", "loop")
    if mode not in ("loop", "fused"):
        raise ValueError("MLX2_GEMV_ROWS_KERNEL must be loop or fused")
    return mode


def tile(n: int, k: int) -> tuple[int, int] | None:
    """MLX 0.32.4's one-row gemv tile ``(BM, TM)``, or ``None`` when the
    one-row call takes a branch this kernel does not mirror."""
    if k <= 64 or k >= 16 * n:
        return None
    return (8 if n >= 4096 else 4), (1 if n < 4 else 4)


def decline_reason(weight, x) -> str | None:
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return "device"
    if weight.ndim != 2 or x.ndim != 3 or x.shape[0] != 1:
        return "shape"
    if weight.dtype not in (mx.bfloat16, mx.float16, mx.float32) or x.dtype != weight.dtype:
        return "dtype"
    rows = int(x.shape[1])
    if not 2 <= rows <= MAX_ROWS:
        return "rows"
    n, k = (int(v) for v in weight.shape)
    if int(x.shape[-1]) != k:
        return "shape"
    if tile(n, k) is None:
        return "tile"
    return None


def gemv_rows(weight, x):
    """``x @ weight.T`` for ``x`` ``[1, R, K]`` (2..16 rows), one weight read."""
    rows, k = int(x.shape[1]), int(x.shape[-1])
    n = int(weight.shape[0])
    bm, tm = tile(n, k)
    block_m = bm * tm
    groups = -(-n // block_m)
    (y,) = _kernel()(
        inputs=[weight, mx.contiguous(x.reshape(rows, k)),
                mx.array([k, n], dtype=mx.int32)],
        template=[("T", x.dtype), ("BM", bm), ("TM", tm), ("ROWS", rows)],
        grid=(groups * 32, 1, bm),
        threadgroup=(32, 1, bm),
        output_shapes=[(rows, n)],
        output_dtypes=[x.dtype],
    )
    return y.reshape(1, rows, n)


def checked_gemv_rows(weight, x, stock):
    """``gemv_rows`` after its first real use per (shape, dtype, rows) matched
    ``stock`` (the per-row broadcast form) bit for bit; else ``None``."""
    key = (tuple(weight.shape), str(weight.dtype), int(x.shape[1]))
    verdict = _VERDICTS.get(key)
    if verdict is False:
        return None
    y = gemv_rows(weight, x)
    if verdict is None:
        reference = stock()
        view = mx.uint32 if y.dtype == mx.float32 else mx.uint16
        same = mx.all(y.view(view) == reference.view(view))
        mx.eval(same)
        _VERDICTS[key] = bool(same)
        if not _VERDICTS[key]:
            return None
    return y


def status() -> dict:
    return {
        "mode": kernel_mode(),
        "verdicts": {f"{k[0]}:{k[1]}:R{k[2]}": v for k, v in _VERDICTS.items()},
    }


__all__ = ["MAX_ROWS", "checked_gemv_rows", "decline_reason", "gemv_rows", "kernel_mode",
           "status", "tile"]

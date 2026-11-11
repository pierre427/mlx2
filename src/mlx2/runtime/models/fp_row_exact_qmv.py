"""Multi-row MXFP8 / NVFP4 projection with one-row decode arithmetic per row.

Original mlx2 code; the arithmetic is MLX's own.  MLX 0.32's
``quantized_matmul`` runs its one-row ``fp_qmv_fast`` / ``fp_qmv`` kernel at
M = 1 but switches to ``fp_qmv_wide`` / ``qmm`` for M >= 2, whose reductions
differ, so a speculative verify row and the serial decode step of the same
token get different bits (measured: a random-input canary does not catch it,
qualification/runs/super-vl-nvfp4-20261110/unit3).  This kernel calls the
installed MLX's one-row implementations themselves (``fp_qmv_fast_impl`` /
``fp_qmv_impl``, extracted verbatim from ``fp_quantized.h`` with the two size
arguments passed by value) once per row inside one threadgroup, so row ``r``
equals ``quantized_matmul(x[r:r+1])`` by construction; rows after the first
re-read the threadgroup's weight tile, mostly from cache.

The one-row dispatch MLX uses is mirrored: ``fp_qmv_fast`` when
``N % 8 == 0`` and ``K`` is a multiple of its block (256 for 8-bit, 512 for
4-bit), else ``fp_qmv``.  Small-K quad dispatch (K <= 128) is declined.
Callers must still prove each (module, width) once on real activations
(``ScaledFP8Linear`` / ``ScaledNVFP4Linear`` do, against the stock one-row
call, bit for bit) before trusting it.  Provenance:
provenance/nemotron35-fp-row-exact-qmv.json.
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path

import mlx.core as mx

MAX_ROWS = 16
_KERNELS: dict = {}
_LOCK = threading.Lock()
_HEADER = None


def _block(bits: int) -> int:
    pack_factor = 32 // bits
    return pack_factor * 2 * 32  # values_per_thread * SIMD_SIZE (qmv_fast)


def fast_layout(k: int, n: int, bits: int) -> bool:
    return n % 8 == 0 and k % _block(bits) == 0


def _header():
    """MLX fp4/fp8 types plus fp_quantized.h up to the one-row qmv impls."""
    global _HEADER
    if _HEADER is not None:
        return _HEADER
    root = Path(mx.__file__).parent / "include" / "mlx/backend/metal/kernels"
    parts = []
    for name in ("fp4.h", "fp8.h"):
        parts.append("\n".join(
            line for line in (root / name).read_text().splitlines()
            if line.strip() != "#pragma once" and not line.strip().startswith('#include "mlx/')
        ))
    text = (root / "fp_quantized.h").read_text()
    start = text.index("#define MLX_MTL_CONST")
    end = text.index("// Quantized matrix-vector for a small batch")
    body = text[start:end]
    # By-value sizes (metal_kernel has no constant-address scalars); the
    # arithmetic is untouched.
    body = body.replace("const constant int& in_vec_size", "const int in_vec_size")
    body = body.replace("const constant int& out_vec_size", "const int out_vec_size")
    if "F8E4M3_MAX" not in "\n".join(parts) + body.split("fp_qmv_quad_impl")[0]:
        body = (
            "#ifndef F8E4M3_MAX\nconstant constexpr float F8E4M3_MAX = 448.0f;\n#endif\n"
            "#ifndef F4E2M1_MAX\nconstant constexpr float F4E2M1_MAX = 6.0f;\n#endif\n" + body
        )
    if re.search(r"function_constant", body):
        raise RuntimeError("unexpected function constants in the extracted qmv source")
    _HEADER = "\n".join(parts) + "\nusing namespace metal;\n" + body
    return _HEADER


# Single-pass variant of MLX's fp_qmv_fast_impl for ROWS rows: per K block
# every input row is loaded, then each output row's weights and scale are
# read once and applied to all input rows.  Per (row, output) the block sums
# are the same qdot calls added in the same K order and finished with the
# same simd_sum, so each row is the one-row arithmetic; the bits are still
# checked against the stock one-row call (profile + first real use).
_FUSED_FAST = """
template <typename T, int group_size, int bits, int ROWS>
METAL_FUNC void mlx2_fp_qmv_fast_rows(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device T* x,
    device T* y,
    const int in_vec_size,
    const int out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int packs_per_thread = 2;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<32, bits>();
  constexpr int bytes_per_pack = get_bytes_per_pack<32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;

  typedef float U;
  thread U x_thread[ROWS][values_per_thread];
  thread U result[ROWS][results_per_simdgroup];
  for (int r = 0; r < ROWS; r++) {
    for (int row = 0; row < results_per_simdgroup; row++) {
      result[r][row] = 0;
    }
  }

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += simd_lid * values_per_thread;
  y += out_row;

  for (int k = 0; k < in_vec_size; k += block_size) {
    for (int r = 0; r < ROWS; r++) {
      load_vector<T, U, values_per_thread>(x + r * in_vec_size, x_thread[r]);
    }

    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device auto* sl = scales + row * in_vec_size_g;

      U s = dequantize_scale<U, group_size>(sl[0]);
      for (int r = 0; r < ROWS; r++) {
        result[r][row] += qdot<U, values_per_thread, bits>(wl, x_thread[r], s);
      }
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    x += block_size;
  }

  for (int r = 0; r < ROWS; r++) {
    for (int row = 0; row < results_per_simdgroup; row++) {
      U total = simd_sum(result[r][row]);
      if (simd_lid == 0) {
        y[r * out_vec_size + row] = static_cast<T>(total);
      }
    }
  }
}
"""
# Single-pass variant of MLX's general fp_qmv_impl (layouts that miss
# qmv_fast, e.g. K = 5376) for ROWS rows, out_vec_size >= 8 branch only: per
# K block every input row is loaded once, each output row's weights are read
# once and applied to every row with the same qdot / qdot_safe calls in the
# same order, finished with the same simd_sum; the tail tile is moved back
# exactly as MLX does.  Checked against the stock one-row call like the rest.
_FUSED_GENERAL = """
template <typename T, int group_size, int bits, int ROWS>
METAL_FUNC void mlx2_fp_qmv_general_rows(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device T* x,
    device T* y,
    const int in_vec_size,
    const int out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int packs_per_thread = 1;
  constexpr int pack_factor = get_pack_factor<32, bits>();
  constexpr int bytes_per_pack = get_bytes_per_pack<32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;
  typedef float U;
  thread U x_thread[ROWS][values_per_thread];
  thread U result[ROWS][results_per_simdgroup];
  for (int r = 0; r < ROWS; r++)
    for (int row = 0; row < results_per_simdgroup; row++)
      result[r][row] = 0;

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  const int used_out_row = min(out_vec_size - results_per_simdgroup, out_row);
  if (out_row >= out_vec_size) {
    return;
  }
  ws += used_out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += simd_lid * values_per_thread;
  y += used_out_row;

  int k = 0;
  for (; k < in_vec_size - block_size; k += block_size) {
    for (int r = 0; r < ROWS; r++)
      load_vector<T, U, values_per_thread>(x + r * in_vec_size, x_thread[r]);
    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device auto* sl = scales + row * in_vec_size_g;
      U s = dequantize_scale<U, group_size>(sl[0]);
      for (int r = 0; r < ROWS; r++)
        result[r][row] += qdot<U, values_per_thread, bits>(wl, x_thread[r], s);
    }
    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    x += block_size;
  }
  const int remaining = clamp(
      static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
      0,
      values_per_thread);
  if (remaining > 0) {
    for (int r = 0; r < ROWS; r++)
      load_vector_safe<T, U, values_per_thread>(
          x + r * in_vec_size, x_thread[r], remaining);
    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device auto* sl = scales + row * in_vec_size_g;
      U s = dequantize_scale<U, group_size>(sl[0]);
      for (int r = 0; r < ROWS; r++)
        result[r][row] +=
            qdot_safe<U, values_per_thread, bits>(wl, x_thread[r], s, remaining);
    }
  }
  for (int r = 0; r < ROWS; r++) {
    for (int row = 0; row < results_per_simdgroup; row++) {
      U total = simd_sum(result[r][row]);
      if (simd_lid == 0) {
        y[r * out_vec_size + row] = static_cast<T>(total);
      }
    }
  }
}
"""
_FUSED_GENERAL_SOURCE = """
    const int K = params[0];
    const int N = params[1];
    uint3 tg = uint3(0, threadgroup_position_in_grid.y, 0);
    mlx2_fp_qmv_general_rows<T, GS, BITS, ROWS>(w, scales, x, y, K, N, tg,
        simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""
FUSED_GENERAL_MAX_ROWS = 6  # 8 rows spilled (7.9 vs loop 7.2 ms per 40 layers)
_FUSED_SOURCE = """
    const int K = params[0];
    const int N = params[1];
    uint3 tg = uint3(0, threadgroup_position_in_grid.y, 0);
    mlx2_fp_qmv_fast_rows<T, GS, BITS, ROWS>(w, scales, x, y, K, N, tg,
        simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""
FUSED_MAX_ROWS = 8
# Rows per fused launch by format: each thread holds ROWS x values_per_thread
# input values (8 for 8-bit, 16 for 4-bit) in registers.  NVFP4 at 6 rows
# spilled and ran 4x slower than the per-row loop (13.5 vs 3.3 ms per 40
# layers at 5376x4096, M5 Max, 2026-11-10); 4 rows is 1.45.  Wider strips
# run as balanced launches of at most this many rows: each row is still the
# one-row arithmetic, so the bits do not depend on the split.
FUSED_MAX_ROWS_BY_BITS = {8: 8, 4: 4}
_KERNEL_MODES = ("loop", "fused", "fused_chunked")

_SOURCE = """
    const int K = params[0];
    const int N = params[1];
    const int R = params[2];
    uint3 tg = uint3(0, threadgroup_position_in_grid.y, 0);
    for (int r = 0; r < R; r++) {
      FAST_OR_GENERAL
    }
"""
_FAST = ("fp_qmv_fast_impl<T, GS, BITS, false, false, 4>(w, scales, "
         "(const device float*)nullptr, x + r * K, y + r * N, K, N, tg, "
         "simdgroup_index_in_threadgroup, thread_index_in_simdgroup);")
_GENERAL = ("fp_qmv_impl<T, GS, BITS, false>(w, scales, (const device float*)nullptr, "
            "x + r * K, y + r * N, K, N, tg, simdgroup_index_in_threadgroup, "
            "thread_index_in_simdgroup);")


def _fused_kernel():
    with _LOCK:
        kernel = _KERNELS.get("fused")
        if kernel is None:
            kernel = mx.fast.metal_kernel(
                name="mlx2_fp_row_exact_qmv_fused",
                input_names=["x", "w", "scales", "params"],
                output_names=["y"],
                header=_header() + _FUSED_FAST,
                source=_FUSED_SOURCE,
            )
            _KERNELS["fused"] = kernel
        return kernel


def _fused_general_kernel():
    with _LOCK:
        kernel = _KERNELS.get("fused_general")
        if kernel is None:
            kernel = mx.fast.metal_kernel(
                name="mlx2_fp_row_exact_qmv_fused_general",
                input_names=["x", "w", "scales", "params"],
                output_names=["y"],
                header=_header() + _FUSED_GENERAL,
                source=_FUSED_GENERAL_SOURCE,
            )
            _KERNELS["fused_general"] = kernel
        return kernel


def kernel_mode() -> str:
    """``MLX2_FP_ROW_EXACT_KERNEL``: ``fused`` (default) or ``loop``.

    Fused is the default since 2026-11-10 (Pierre), after profile4 (bit-exact
    24/24 at the Super VL shapes) and unit5 (native MTP 42.7 -> 51.2 tok/s
    GSM8K, LiLiCorr 44.5 -> 47.1, greedy-equal, predictions identical);
    qualification/runs/super-vl-nvfp4-20261110.  ``loop`` reruns MLX's
    one-row impl per row.  ``fused_chunked`` (opt-in, 2026-11-10) is ``fused``
    plus 9..16-row strips as balanced fused launches of at most 8 rows
    (:func:`fused_chunks`): every row still runs the same one-row impl, so
    the bits are those of ``loop``; weights are read once per launch instead
    of once per row.
    """
    import os

    mode = os.environ.get("MLX2_FP_ROW_EXACT_KERNEL", "fused")
    if mode not in _KERNEL_MODES:
        raise ValueError("MLX2_FP_ROW_EXACT_KERNEL must be loop, fused or fused_chunked")
    return mode


def fused_chunks(rows: int, cap: int = FUSED_MAX_ROWS) -> tuple[int, ...]:
    """Balanced fused launches of at most ``cap`` rows covering a strip of
    ``cap + 1 .. 16`` rows."""
    if type(cap) is not int or not 2 <= cap <= FUSED_MAX_ROWS:
        raise ValueError("fused chunk cap must be 2..8 rows")
    # (callers pass the per-kernel cap; FUSED_GENERAL_MAX_ROWS is 6)
    if type(rows) is not int or not cap < rows <= MAX_ROWS:
        raise ValueError(f"fused chunks cover {cap + 1}..16 rows")
    count = -(-rows // cap)
    base, wider = divmod(rows, count)
    return tuple(base + int(index < wider) for index in range(count))


def _kernel(fast: bool):
    with _LOCK:
        kernel = _KERNELS.get(fast)
        if kernel is None:
            kernel = mx.fast.metal_kernel(
                name="mlx2_fp_row_exact_qmv_" + ("fast" if fast else "general"),
                input_names=["x", "w", "scales", "params"],
                output_names=["y"],
                header=_header(),
                source=_SOURCE.replace("FAST_OR_GENERAL", _FAST if fast else _GENERAL),
            )
            _KERNELS[fast] = kernel
        return kernel


def decline_reason(weight, scales, x, *, bits: int, group_size: int) -> str | None:
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return "device"
    if (bits, group_size) not in {(8, 32), (4, 16)}:
        return "mode"
    if x.dtype not in (mx.bfloat16, mx.float16):
        return "input_dtype"
    if weight.dtype != mx.uint32 or weight.ndim != 2 or scales.dtype != mx.uint8:
        return "weight_layout"
    n, k = int(weight.shape[0]), int(x.shape[-1])
    if int(weight.shape[1]) * 32 != k * bits or int(scales.shape[-1]) * group_size != k:
        return "weight_layout"
    if k <= 128:
        return "quad_k"
    if n < 8:
        return "small_n"
    return None


def rows_qmv(x, weight, scales, *, bits: int, group_size: int, mode: str | None = None):
    """``[1, R, K] -> [1, R, N]``, row r bit-equal to the one-row stock call.

    ``mode`` (default ``kernel_mode()``, fused): ``fused`` reads each weight
    block once per launch of at most ``FUSED_MAX_ROWS_BY_BITS[bits]`` rows
    (qmv_fast layouts, 2..8 rows; anything else takes ``loop``); ``loop``
    reruns the one-row impl per row.
    """
    rows, k = int(x.shape[1]), int(x.shape[-1])
    if not 1 <= rows <= MAX_ROWS:
        raise ValueError("fp row-exact qmv takes 1..16 rows")
    n = int(weight.shape[0])
    fast = fast_layout(k, n, bits)
    mode = kernel_mode() if mode is None else mode

    def fused(flat, count):
        (y,) = _fused_kernel()(
            inputs=[flat, weight, scales, mx.array([k, n, count], dtype=mx.int32)],
            template=[("T", x.dtype), ("GS", group_size), ("BITS", bits), ("ROWS", count)],
            grid=(32, 2 * (n // 8), 1),
            threadgroup=(32, 2, 1),
            output_shapes=[(count, n)],
            output_dtypes=[x.dtype],
        )
        return y

    cap = FUSED_MAX_ROWS_BY_BITS.get(int(bits), FUSED_MAX_ROWS)
    if mode in ("fused", "fused_chunked") and fast and 2 <= rows <= cap:
        return fused(mx.contiguous(x.reshape(rows, k)), rows).reshape(1, rows, n)
    if fast and (
        (mode == "fused" and cap < rows <= FUSED_MAX_ROWS)
        or (mode == "fused_chunked" and rows > cap)
    ):
        flat = x.reshape(rows, k)
        parts, first = [], 0
        for count in fused_chunks(rows, cap):
            parts.append(fused(mx.contiguous(flat[first:first + count]), count))
            first += count
        return mx.concatenate(parts, axis=0).reshape(1, rows, n)
    if not fast and mode in ("fused", "fused_chunked") and n >= 8 and os.environ.get(
        "MLX2_FP_ROW_EXACT_GENERAL", "loop"
    ) == "fused":
        flat = x.reshape(rows, k)
        counts = (
            [rows] if rows <= FUSED_GENERAL_MAX_ROWS
            else list(fused_chunks(rows, FUSED_GENERAL_MAX_ROWS))
        )
        parts, first = [], 0
        for count in counts:
            (y,) = _fused_general_kernel()(
                inputs=[mx.contiguous(flat[first:first + count]), weight, scales,
                        mx.array([k, n, count], dtype=mx.int32)],
                template=[("T", x.dtype), ("GS", group_size), ("BITS", bits),
                          ("ROWS", count)],
                grid=(32, 2 * ((n + 7) // 8), 1),
                threadgroup=(32, 2, 1),
                output_shapes=[(count, n)],
                output_dtypes=[x.dtype],
            )
            parts.append(y)
            first += count
        y = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)
        return y.reshape(1, rows, n)
    kernel = _kernel(fast)
    flat = mx.contiguous(x.reshape(rows, k))
    (y,) = kernel(
        inputs=[flat, weight, scales, mx.array([k, n, rows], dtype=mx.int32)],
        template=[("T", x.dtype), ("GS", group_size), ("BITS", bits)],
        grid=(32, 2 * ((n + 7) // 8), 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(rows, n)],
        output_dtypes=[x.dtype],
    )
    return y.reshape(1, rows, n)


__all__ = ["FUSED_MAX_ROWS", "FUSED_MAX_ROWS_BY_BITS", "MAX_ROWS", "decline_reason", "fast_layout", "fused_chunks", "kernel_mode", "rows_qmv"]

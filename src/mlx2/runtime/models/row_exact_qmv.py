# SPDX-License-Identifier: Apache-2.0
"""Multi-row quantized projection with one-row decode arithmetic per row.

Mined from jundot/omlx ``omlx/patches/row_exact_qmv.py`` (PR #4023, as
revised by #4050; Apache-2.0).  See provenance/omlx-row-exact-verify.json.

Stock ``quantized_matmul`` switches kernels with the row count (``qmv_fast``
or ``qmv`` at one row, ``qmv_wide``/NAX/tiled ``qmm`` for more), so a
speculative verify row and the serial decode step for the same token get
different bits from the same projection.  Here every row runs the one-row
``qmv_fast`` (N a multiple of 8, K a multiple of the kernel block: 512 for
4/5-bit, 256 for 6/8-bit) or ``qmv`` traversal, so row ``r`` of the output
equals ``quantized_matmul(x[r:r+1])`` bit for bit.

A threadgroup computes one ``2 * RPS``-column output tile for ``ROWS`` rows:
each weight block is decoded once and applied to all of them, and every
(row, column) accumulator still sums its K blocks in qmv order and finishes
with the same ``simd_sum``.

mlx2 adaptations: the dot-product helpers are the mlx2 ``QMV_HEADER`` (float
input sums, matching the venv MLX fork's qmv since 2d6705923; omlx's verbatim
0.32.2 header sums in T), ``MAX_ROWS`` covers the 17-row copy-draft verify
window, the one-row ``OneRowQmv``/``RowsQmv`` helpers are not carried, and a
layout the kernel does not cover runs one stock one-row call per row (still
the serial arithmetic) and is counted as such.
"""

from __future__ import annotations

from functools import cache

import mlx.core as mx
import mlx.nn as nn

from .qwen4_routed_decode import QMV_HEADER

MAX_ROWS = 32
BITS = (4, 5, 6, 8)
GROUP_SIZES = (32, 64, 128)
# Output columns per simdgroup (qmv's own tile: a threadgroup owns 8).
_RPS = 4
# Below this many tiles every row gets its own threadgroups; wider outputs
# (the vocabulary head) share each weight pass between rows, up to this many
# per-lane input values.
_SHARED_ROWS_MIN_TILES = 2048
_ROW_VALUES_BUDGET = 32

# Per row this is qmv_fast_impl (FAST) or qmv_impl; the row loop sits inside
# each K block so the block's weight bytes are fetched once.  Columns past N
# (qmv's partial last tile) are skipped: each column's arithmetic does not
# depend on which tile computes it.  (omlx row_exact_qmv.py _TILE, verbatim.)
_TILE = r"""
template <typename T, int K_SIZE, int N_OUT, int ROWS, int RPS, typename SP>
METAL_FUNC void row_exact_tile(
    const device T* x,
    const device uint32_t* w,
    SP scales,
    SP biases,
    device T* y,
    int tile,
    int row0,
    uint simd_gid,
    uint simd_lid) {
  constexpr bool PARTIAL = (N_OUT % (2 * RPS)) != 0;
  const int in_vec_size_w = K_SIZE * BYTES_PER_PACK / PACK_FACTOR;
  const int in_vec_size_g = K_SIZE / GS;
  const int out_row = tile * (2 * RPS) + int(simd_gid) * RPS;
  if (PARTIAL && out_row >= N_OUT) {
    return;
  }
  const int valid = PARTIAL ? min(RPS, N_OUT - out_row) : RPS;

  const device uint8_t* ws = (const device uint8_t*)w + out_row * in_vec_size_w +
      int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
  SP sc = scales + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
  SP bs = biases + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
  const device T* xp = x + row0 * K_SIZE + int(simd_lid) * VALUES_PER_THREAD;

  float result[ROWS][RPS];
  for (int i = 0; i < ROWS; i++) {
    for (int r = 0; r < RPS; r++) {
      result[i][r] = 0;
    }
  }

  int k = 0;
  const int full_limit = FAST ? K_SIZE : K_SIZE - BLOCK_SIZE;
  for (; k < full_limit; k += BLOCK_SIZE) {
    float xs[ROWS][VALUES_PER_THREAD];
    float sums[ROWS];
    for (int i = 0; i < ROWS; i++) {
      sums[i] = load_vector<T>(xp + i * K_SIZE, xs[i]);
    }
    for (int r = 0; r < RPS; r++) {
      if (PARTIAL && r >= valid) {
        break;
      }
      float wq[W_TERMS];
      decode_w(ws + r * in_vec_size_w, wq);
      float s = sc[r * in_vec_size_g];
      float b = bs[r * in_vec_size_g];
      for (int i = 0; i < ROWS; i++) {
        result[i][r] += s * wdot(wq, xs[i]) + sums[i] * b;
      }
    }
    ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
    sc += BLOCK_SIZE / GS;
    bs += BLOCK_SIZE / GS;
    xp += BLOCK_SIZE;
  }
  if (!FAST) {
    const int remaining = clamp(
        int(K_SIZE - k - int(simd_lid) * VALUES_PER_THREAD), 0, VALUES_PER_THREAD);
    if (remaining > 0) {
      for (int i = 0; i < ROWS; i++) {
        float x_thread[VALUES_PER_THREAD];
        float sum = load_vector_safe<T>(xp + i * K_SIZE, x_thread, remaining);
        for (int r = 0; r < RPS; r++) {
          if (PARTIAL && r >= valid) {
            break;
          }
          const device uint8_t* wl = ws + r * in_vec_size_w;
          float s = sc[r * in_vec_size_g];
          float b = bs[r * in_vec_size_g];
          result[i][r] += qdot_n(wl, x_thread, s, b, sum, remaining);
        }
      }
    }
  }

  for (int i = 0; i < ROWS; i++) {
    for (int r = 0; r < RPS; r++) {
      if (PARTIAL && r >= valid) {
        break;
      }
      float v = simd_sum(result[i][r]);
      if (simd_lid == 0) {
        y[(row0 + i) * N_OUT + out_row + r] = static_cast<T>(v);
      }
    }
  }
}
"""

_SOURCE = r"""
    row_exact_tile<T, K_SIZE, N_SIZE, ROWS, RPS>(
        x, w, scales, biases, y,
        int(threadgroup_position_in_grid.z),
        int(threadgroup_position_in_grid.y) * ROWS,
        simdgroup_index_in_threadgroup,
        thread_index_in_simdgroup);
"""


def _group_source(count: int) -> str:
    """Same-input projections in one launch: the tile axis walks each
    projection's tiles in turn, and every projection writes its own output.
    (omlx row_exact_qmv.py _group_source, verbatim.)"""
    lines = [
        "    const int tile = int(threadgroup_position_in_grid.z);",
        "    const int row0 = int(threadgroup_position_in_grid.y) * ROWS;",
        "    const uint sg = simdgroup_index_in_threadgroup;",
        "    const uint sl = thread_index_in_simdgroup;",
        "    int start = 0;",
    ]
    for i in range(count):
        lines += [
            f"    constexpr int TILES_{i} = (N_{i} + 2 * RPS - 1) / (2 * RPS);",
            f"    if (tile < start + TILES_{i}) {{",
            f"      row_exact_tile<T, K_SIZE, N_{i}, ROWS, RPS>(",
            f"          x, w{i}, scales{i}, biases{i}, y{i}, tile - start, row0, sg, sl);",
            "      return;",
            "    }",
            f"    start += TILES_{i};",
        ]
    return "\n".join(lines) + "\n"


# qmv's ``qdot`` split in two: the weight terms of a full pack run are decoded
# once per K block and output column, then every row accumulates them against
# its own inputs, each term and accumulation step in qdot's order.  (omlx
# row_exact_qmv.py _DECODED_DOT, verbatim.)
_DECODED_DOT = r"""
constant constexpr int W_TERMS = (BITS == 5) ? 12 * VALUES_PER_THREAD / 8
    : ((BITS == 6) ? 6 * VALUES_PER_THREAD / 4 : VALUES_PER_THREAD);

inline void decode_w(const device uint8_t* w, thread float* wq) {
  if (BITS == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < VALUES_PER_THREAD / 4; i++) {
      wq[4 * i] = ws[i] & 0x000f;
      wq[4 * i + 1] = ws[i] & 0x00f0;
      wq[4 * i + 2] = ws[i] & 0x0f00;
      wq[4 * i + 3] = ws[i] & 0xf000;
    }
  } else if (BITS == 5) {
    for (int i = 0; i < VALUES_PER_THREAD / 8; i++) {
      const device uint8_t* wb = w + 5 * i;
      thread float* wv = wq + 12 * i;
      wv[0] = wb[0] & 0x1f;
      wv[1] = wb[0] & 0xe0;
      wv[2] = wb[1] & 0x3;
      wv[3] = wb[1] & 0x7c;
      wv[4] = wb[1] & 0x80;
      wv[5] = wb[2] & 0xf;
      wv[6] = wb[2] & 0xf0;
      wv[7] = wb[3] & 0x1;
      wv[8] = wb[3] & 0x3e;
      wv[9] = wb[3] & 0xc0;
      wv[10] = wb[4] & 0x7;
      wv[11] = wb[4] & 0xf8;
    }
  } else if (BITS == 6) {
    for (int i = 0; i < VALUES_PER_THREAD / 4; i++) {
      const device uint8_t* wb = w + 3 * i;
      thread float* wv = wq + 6 * i;
      wv[0] = wb[0] & 0x3f;
      wv[1] = wb[0] & 0xc0;
      wv[2] = wb[1] & 0x0f;
      wv[3] = wb[1] & 0xf0;
      wv[4] = wb[2] & 0x03;
      wv[5] = wb[2] & 0xfc;
    }
  } else if (BITS == 8) {
    for (int i = 0; i < VALUES_PER_THREAD; i++) {
      wq[i] = w[i];
    }
  }
}

inline float wdot(const thread float* wq, const thread float* x) {
  float accum = 0;
  if (BITS == 4) {
    for (int i = 0; i < VALUES_PER_THREAD / 4; i++) {
      accum +=
          (x[4 * i] * wq[4 * i] + x[4 * i + 1] * wq[4 * i + 1] +
           x[4 * i + 2] * wq[4 * i + 2] + x[4 * i + 3] * wq[4 * i + 3]);
    }
  } else if (BITS == 5) {
    for (int i = 0; i < VALUES_PER_THREAD / 8; i++) {
      const thread float* xv = x + 8 * i;
      const thread float* wv = wq + 12 * i;
      accum += wv[0] * xv[0];
      accum += wv[1] * xv[1];
      accum += wv[2] * (xv[1] * 256.0f);
      accum += wv[3] * xv[2];
      accum += wv[4] * xv[3];
      accum += wv[5] * (xv[3] * 256.0f);
      accum += wv[6] * xv[4];
      accum += wv[7] * (xv[4] * 256.0f);
      accum += wv[8] * xv[5];
      accum += wv[9] * xv[6];
      accum += wv[10] * (xv[6] * 256.0f);
      accum += wv[11] * xv[7];
    }
  } else if (BITS == 6) {
    for (int i = 0; i < VALUES_PER_THREAD / 4; i++) {
      const thread float* xv = x + 4 * i;
      const thread float* wv = wq + 6 * i;
      accum += wv[0] * xv[0];
      accum += wv[1] * xv[1];
      accum += wv[2] * (xv[1] * 256.0f);
      accum += wv[3] * xv[2];
      accum += wv[4] * (xv[2] * 256.0f);
      accum += wv[5] * xv[3];
    }
  } else if (BITS == 8) {
    for (int i = 0; i < VALUES_PER_THREAD; i++) {
      accum += x[i] * wq[i];
    }
  }
  return accum;
}
"""


def qmv_fast_layout(k: int, n: int, bits: int) -> bool:
    """Whether a one-row affine product runs the ``qmv_fast`` K traversal
    (else ``qmv``).

    Stock MLX 0.32.2 needs N a multiple of 8 and K a multiple of the kernel
    block (``pack_factor * packs * 32``: 512 for 4/5-bit, 256 for 6/8-bit).
    The venv MLX fork also runs affine outputs with N % 8 != 0 on the fast
    traversal when K is aligned (``qmv_fast_rows``: the last simdgroup covers
    the remaining columns), so for affine weights only K decides.  omlx keys
    this on N as well; on this fork that picked the ``qmv`` traversal for the
    1- and 4-column projections and differed from the one-row call.
    """
    del n
    pack_factor = 8 if bits in (3, 5) else (4 if bits == 6 else 32 // bits)
    return k % (pack_factor * (1 if bits == 2 else 2) * 32) == 0


def header(bits: int, group_size: int, fast: bool) -> str:
    return (
        (QMV_HEADER + _DECODED_DOT + _TILE)
        .replace("__BITS__", str(bits))
        .replace("__GS__", str(group_size))
        .replace("__FAST__", "1" if fast else "0")
    )


@cache
def _kernel(bits: int, group_size: int, fast: bool):
    return mx.fast.metal_kernel(
        name=f"mlx2_row_exact_qmv_b{bits}_gs{group_size}_{int(fast)}",
        input_names=["x", "w", "scales", "biases"],
        output_names=["y"],
        header=header(bits, group_size, fast),
        source=_SOURCE,
    )


@cache
def _group_kernel(bits: int, group_size: int, fast: bool, count: int):
    inputs = ["x"]
    for i in range(count):
        inputs += [f"w{i}", f"scales{i}", f"biases{i}"]
    return mx.fast.metal_kernel(
        name=f"mlx2_row_exact_qmv_group{count}_b{bits}_gs{group_size}_{int(fast)}",
        input_names=inputs,
        output_names=[f"y{i}" for i in range(count)],
        header=header(bits, group_size, fast),
        source=_group_source(count),
    )


def _values_per_thread(bits: int, fast: bool) -> int:
    return (8 if bits == 5 else (4 if bits == 6 else 32 // bits)) * (2 if fast else 1)


def decline_reason(linear, x: mx.array) -> str | None:
    """Why the kernel does not cover ``linear`` at ``x`` (None when it does)."""
    if not isinstance(linear, nn.QuantizedLinear):
        return "not_quantized_linear"
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return "device"
    bits, group_size = linear.bits, linear.group_size
    if getattr(linear, "mode", "affine") != "affine":
        return "mode"
    if bits not in BITS:
        return "bits"
    if group_size not in GROUP_SIZES:
        return "group_size"
    if linear.get("biases") is None:
        return "no_affine_biases"
    if x.dtype not in (mx.bfloat16, mx.float16):
        return "input_dtype"
    if linear.scales.dtype != x.dtype or linear.biases.dtype != x.dtype:
        return "scale_dtype"
    if linear.weight.dtype != mx.uint32 or linear.weight.ndim != 2:
        return "weight_layout"
    n = int(linear.weight.shape[0])
    k = int(linear.scales.shape[-1]) * group_size
    if int(linear.weight.shape[1]) * 32 != k * bits:
        return "weight_layout"
    fast = qmv_fast_layout(k, n, bits)
    if x.shape[-1] != k or k % 8:
        return "k_shape"
    if group_size % _values_per_thread(bits, fast):
        return "group_tile"
    return None


def _launch_geometry(bits: int, fast: bool, n: int, rows: int) -> tuple[int, int]:
    """(rows per threadgroup, output columns per simdgroup) for one launch."""
    if (n + 2 * _RPS - 1) // (2 * _RPS) < _SHARED_ROWS_MIN_TILES:
        return 1, _RPS
    limit = _ROW_VALUES_BUDGET // _values_per_thread(bits, fast)
    rows_per_group = next(
        (d for d in range(min(rows, limit), 1, -1) if rows % d == 0), 1
    )
    return rows_per_group, _RPS


class _Plan:
    """Static launch parameters of one (linear, row count, dtype) shape."""

    __slots__ = ("kernel", "template", "grid", "output_shapes", "output_dtypes", "bias")

    def __init__(self, linear, x: mx.array, rows: int):
        k = int(x.shape[-1])
        n = int(linear.weight.shape[0])
        bits = int(linear.bits)
        fast = qmv_fast_layout(k, n, bits)
        rows_per_group, rps = _launch_geometry(bits, fast, n, rows)
        self.kernel = _kernel(bits, int(linear.group_size), fast)
        self.template = [
            ("T", x.dtype),
            ("K_SIZE", k),
            ("N_SIZE", n),
            ("ROWS", rows_per_group),
            ("RPS", rps),
        ]
        self.grid = (32, 2 * (rows // rows_per_group), (n + 2 * rps - 1) // (2 * rps))
        self.output_shapes = [(rows, n)]
        self.output_dtypes = [x.dtype]
        self.bias = "bias" in linear


def _plan(linear, x: mx.array, rows: int):
    # Plans live on the module, so they die with it; weights are read per call.
    plans = linear.__dict__.get("_mlx2_row_exact_plans")
    if plans is None:
        plans = {}
        object.__setattr__(linear, "_mlx2_row_exact_plans", plans)
    key = (rows, x.dtype, x.shape[-1])
    if key not in plans:
        supported = rows <= MAX_ROWS and decline_reason(linear, x) is None
        plans[key] = _Plan(linear, x, rows) if supported else None
    return plans[key]


def per_row(call, x: mx.array) -> mx.array:
    """``call`` on each row of ``x`` ([..., K]) as its own one-row input."""
    lead = x.shape[:-1]
    flat = x.reshape(-1, 1, x.shape[-1])
    y = mx.concatenate([call(flat[r : r + 1])[0] for r in range(flat.shape[0])], axis=0)
    return y.reshape(*lead, -1)


def quantized_linear(linear, x: mx.array, call=None):
    """``linear(x)`` for ``x`` of shape [..., K], each row with one-row arithmetic.

    Returns ``(y, route)`` where route is ``"kernel"`` or ``"per_row"`` (a
    layout the kernel does not cover runs one stock one-row call per row,
    which is the serial decode call itself).  ``call`` is the stock one-row
    call (default ``linear``).
    """
    lead = x.shape[:-1]
    rows = 1
    for size in lead:
        rows *= size
    stock = linear if call is None else call
    if rows <= 1:
        return stock(x), "one_row"
    plan = _plan(linear, x, rows)
    if plan is None:
        return per_row(stock, x), "per_row"
    flat = x.reshape(rows, x.shape[-1])
    y = plan.kernel(
        inputs=[flat, linear.weight, linear.scales, linear.biases],
        template=plan.template,
        grid=plan.grid,
        threadgroup=(32, 2, 1),
        output_shapes=plan.output_shapes,
        output_dtypes=plan.output_dtypes,
    )[0]
    if plan.bias:
        y = y + linear["bias"]
    return y.reshape(*lead, -1), "kernel"


class _GroupPlan:
    """Static launch parameters of one same-input projection group."""

    __slots__ = ("kernel", "template", "grid", "output_shapes", "output_dtypes")

    def __init__(self, linears, x: mx.array, rows: int):
        k = int(x.shape[-1])
        sizes = [int(linear.weight.shape[0]) for linear in linears]
        first = linears[0]
        bits = int(first.bits)
        fast = qmv_fast_layout(k, sizes[0], bits)
        rows_per_group, rps = _launch_geometry(bits, fast, max(sizes), rows)
        self.kernel = _group_kernel(bits, int(first.group_size), fast, len(linears))
        self.template = [("T", x.dtype), ("K_SIZE", k)]
        self.template += [(f"N_{i}", n) for i, n in enumerate(sizes)]
        self.template += [("ROWS", rows_per_group), ("RPS", rps)]
        tiles = sum((n + 2 * rps - 1) // (2 * rps) for n in sizes)
        self.grid = (32, 2 * (rows // rows_per_group), tiles)
        self.output_shapes = [(rows, n) for n in sizes]
        self.output_dtypes = [x.dtype] * len(sizes)


def _group_plan(linears, x: mx.array, rows: int):
    first = linears[0]
    plans = first.__dict__.get("_mlx2_row_exact_plans")
    if plans is None:
        plans = {}
        object.__setattr__(first, "_mlx2_row_exact_plans", plans)
    key = (tuple(id(linear) for linear in linears), rows, x.dtype, x.shape[-1])
    if key not in plans:
        grouped = (
            rows <= MAX_ROWS
            and all(
                isinstance(linear, nn.QuantizedLinear)
                and linear.bits == first.bits
                and linear.group_size == first.group_size
                and "bias" not in linear
                and decline_reason(linear, x) is None
                for linear in linears
            )
        )
        plans[key] = _GroupPlan(linears, x, rows) if grouped else None
    return plans[key]


def quantized_linears(linears, x: mx.array):
    """``(tuple(linear(x) for linear in linears), route)`` with one-row
    arithmetic per row, in one launch when the projections share their
    quantization (route ``"group_kernel"``); otherwise each projection goes
    through ``quantized_linear`` (route of the last one)."""
    lead = x.shape[:-1]
    rows = 1
    for size in lead:
        rows *= size
    plan = (
        _group_plan(linears, x, rows) if 1 < rows and 1 < len(linears) <= 4 else None
    )
    if plan is None:
        results = [quantized_linear(linear, x) for linear in linears]
        return tuple(y for y, _ in results), results[-1][1]
    inputs = [x.reshape(rows, x.shape[-1])]
    for linear in linears:
        inputs += [linear.weight, linear.scales, linear.biases]
    outputs = plan.kernel(
        inputs=inputs,
        template=plan.template,
        grid=plan.grid,
        threadgroup=(32, 2, 1),
        output_shapes=plan.output_shapes,
        output_dtypes=plan.output_dtypes,
    )
    return tuple(y.reshape(*lead, -1) for y in outputs), "group_kernel"


__all__ = [
    "BITS",
    "GROUP_SIZES",
    "MAX_ROWS",
    "decline_reason",
    "header",
    "per_row",
    "qmv_fast_layout",
    "quantized_linear",
    "quantized_linears",
]

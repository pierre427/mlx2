# SPDX-License-Identifier: MIT
"""Source M1 GU/DOWN dot laws, GPU-vectorized over AST occurrences.

No local-M wide or split-K dispatch is selected. The provider owns trusted
row maps and expert lifetimes. Nothing compiles or runs on import.
"""
from functools import lru_cache
from types import SimpleNamespace

MAX_ROWS = 16


def plan(rows, kind):
    if type(rows) is not int or not 1 <= rows <= MAX_ROWS or kind not in ("original", "q1"):
        raise ValueError("Source AST M1 GU requires original/Q1 occurrence rows1..16")
    return dict(grid=(72*256, rows, 1), threadgroup=(256, 1, 1),
                output_shape=(rows, 2304), dot_rows=1, occurrence_rows=rows)


def middle_spec(kind, *, exp_spelling="metal::exp", return_projections=False):
    from . import source_gate_up_fusion as gu
    return gu.kernel_spec(kind, exp_spelling=exp_spelling, selected_rows=1,
                          return_projections=return_projections)


@lru_cache(maxsize=8)
def _middle_kernel(kind, spelling, diagnostic):
    import mlx.core as mx
    return mx.fast.metal_kernel(name=f"mlx2_source_ast_m1_gu_{kind}_{int(diagnostic)}_{'precise' if 'precise' in spelling else 'fast'}_v1",
                               ensure_row_contiguous=True,
                               **middle_spec(kind, exp_spelling=spelling, return_projections=diagnostic))


def middle(x, rows, scores, expert, *, exp_spelling="metal::exp", return_projections=False):
    import mlx.core as mx
    from . import source_gate_up_fusion as gu
    if not 1 <= rows.size <= MAX_ROWS:
        raise ValueError("Source AST M1 GU requires1..16 occurrence rows")
    # Geometry for one row validates all packed tensors without selecting the
    # original M2..12 wide law or rejecting a valid M13..16 occurrence list.
    kind = gu._geometry(x, SimpleNamespace(shape=(1,), ndim=1, dtype=mx.uint32),
                        SimpleNamespace(shape=(1,), dtype=mx.float32), expert)
    if (rows.ndim != 1 or rows.dtype != mx.uint32 or scores.shape != rows.shape
            or scores.dtype != mx.float32 or type(return_projections) is not bool):
        raise ValueError("Source AST M1 GU occurrence-map shape/dtype")
    geometry = plan(rows.size, kind)
    if mx.default_device() != mx.gpu or not mx.metal.is_available() or mx.device_info().get("device_name") != "Apple M5 Max":
        raise ValueError("Source AST M1 GU requires an owned M5 Max GPU window")
    fields = (".qweight", ".scales", ".biases") if kind == "q1" else (".qweight", ".scales")
    inputs = [x, rows, scores] + [expert[p + f] for p in ("gate", "up") for f in fields]
    inputs.append(mx.array([1/448], mx.float32))
    count = 3 if return_projections else 1
    result = _middle_kernel(kind, exp_spelling, return_projections)(
        inputs=inputs, grid=geometry["grid"], threadgroup=geometry["threadgroup"],
        output_shapes=[geometry["output_shape"]]*count, output_dtypes=[mx.float32]*count)
    return tuple(result) if return_projections else result[0]

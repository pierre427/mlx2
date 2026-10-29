# SPDX-License-Identifier: MIT
"""Unchanged Source BF16/F32 M1 helper, independent row dimension in one grid.

No shared mutable accumulation or wide/split-K math. Explicit onchip reuse is
not claimed. Actual GPU raw parity remains a separate gate.
Provenance: deepseek-v41-source-ast-m1-batch-axis.json.
"""
from functools import lru_cache


def plan(kind, rows):
    from .source_dense import plan as stock
    if type(rows) is not int or not 1 <= rows <= 16:
        raise ValueError("Source AST M1 dense rows requires1..16 rows")
    base = stock(kind, 1, algorithm="gemv")
    shape = (1, rows, 8, 1024) if kind == "wo_a" else (1, rows, 129280)
    return dict(base, rows=rows, grid=(base["grid"][0], rows, base["grid"][2]),
                output_shape=shape, dot_rows=1, occurrence_rows=rows,
                onchip_weight_reuse_claim=False, exactness="pending actual GPU raw gate")


def kernel_spec(kind):
    from .source_dense import kernel_spec as stock
    # Source already addresses each row via threadgroup_position_in_grid.y.
    # Its M=1 constants and dot helper are retained byte-for-byte.
    return stock(kind, 1, algorithm="gemv")


@lru_cache(maxsize=2)
def _kernel(kind):
    import mlx.core as mx
    spec = kernel_spec(kind)
    return mx.fast.metal_kernel(name="mlx2_source_ast_m1_dense_rows_"+kind+"_v1",
        input_names=spec['input_names'], output_names=spec['output_names'],
        header=spec['header'], source=spec['source'], ensure_row_contiguous=True)


def project(x, weight, kind):
    import mlx.core as mx
    from .source_dense import _geometry
    # Full weight + one-row structural view validates the exact existing M1
    # coefficient geometry without slicing/evaluating the actual activation.
    from types import SimpleNamespace
    first_shape = (1, 1, 8, 4096) if kind == "wo_a" else (1, 1, 5120)
    _geometry(SimpleNamespace(shape=first_shape, ndim=len(first_shape), dtype=mx.float32),
              weight, kind, "gemv", (64, 64, 128, 2, 2))
    descriptor = plan(kind, x.shape[1])
    expected = (1, x.shape[1], 8, 4096) if kind == "wo_a" else (1, x.shape[1], 5120)
    if x.shape != expected or x.dtype != mx.float32:
        raise ValueError("Source AST M1 dense row activation geometry mismatch")
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise ValueError("Source AST M1 dense row kernel requires owned GPU")
    return _kernel(kind)(inputs=[x, weight], grid=descriptor['grid'],
        threadgroup=descriptor['threadgroup'], output_shapes=[descriptor['output_shape']],
        output_dtypes=[mx.float32])[0]

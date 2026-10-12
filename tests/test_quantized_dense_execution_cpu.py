"""Focused CPU tests for packed storage with bounded dense execution."""

import mlx.core as mx
import pytest
from mlx import nn

from mlx2.runtime.quantized_dense_execution import (
    QuantizedDenseExecutionPolicy,
    QuantizedDenseExecutionRefusal,
    dense_projection,
    geometry,
)


def _linear(*, bits=4, group_size=32, bias=False):
    dense = nn.Linear(64, 48, bias=bias)
    return nn.QuantizedLinear.from_linear(
        dense, group_size=group_size, bits=bits, mode="affine"
    )


@pytest.mark.parametrize("bits", [4, 6, 8])
def test_dense_projection_matches_the_explicit_storage_boundary(bits):
    module = _linear(bits=bits, bias=True)
    x = mx.random.normal((2, 5, 64)).astype(mx.bfloat16)
    policy = QuantizedDenseExecutionPolicy(
        minimum_rows=8,
        max_dense_weight_bytes=48 * 64 * 2,
        formats=frozenset({("affine", bits, 32)}),
    )
    actual = dense_projection(module, x, policy=policy)
    weight = mx.dequantize(
        module.weight,
        module.scales,
        module.biases,
        group_size=32,
        bits=bits,
        mode="affine",
    ).astype(x.dtype)
    expected = x @ weight.T + module.bias
    assert mx.array_equal(actual, expected).item()


def test_mxfp4_storage_can_use_the_same_dense_execution_primitive():
    module = nn.QuantizedLinear.from_linear(
        nn.Linear(64, 48, bias=False), group_size=32, bits=4, mode="mxfp4"
    )
    x = mx.random.normal((2, 5, 64)).astype(mx.bfloat16)
    policy = QuantizedDenseExecutionPolicy(
        minimum_rows=8,
        max_dense_weight_bytes=48 * 64 * 2,
        formats=frozenset({("mxfp4", 4, 32)}),
    )
    actual = dense_projection(module, x, policy=policy)
    weight = mx.dequantize(
        module.weight,
        module.scales,
        group_size=32,
        bits=4,
        mode="mxfp4",
    ).astype(x.dtype)
    assert mx.array_equal(actual, x @ weight.T).item()


def test_geometry_reports_storage_separately_from_execution_shape():
    module = _linear(bits=4, group_size=32)
    receipt = geometry(module, mx.zeros((3, 7, 64), dtype=mx.float16))
    assert receipt == {
        "mode": "affine",
        "bits": 4,
        "group_size": 32,
        "input_dims": 64,
        "output_dims": 48,
        "rows": 21,
        "dense_weight_bytes": 48 * 64 * 2,
    }


def test_selected_route_fails_closed_instead_of_changing_arithmetic():
    module = _linear(bits=4, group_size=32)
    x = mx.zeros((1, 4, 64), dtype=mx.bfloat16)
    policy = QuantizedDenseExecutionPolicy(
        minimum_rows=5,
        max_dense_weight_bytes=48 * 64 * 2,
        formats=frozenset({("affine", 4, 32)}),
    )
    with pytest.raises(QuantizedDenseExecutionRefusal, match="row count"):
        dense_projection(module, x, policy=policy)
    too_small = QuantizedDenseExecutionPolicy(
        minimum_rows=1,
        max_dense_weight_bytes=1,
        formats=frozenset({("affine", 4, 32)}),
    )
    with pytest.raises(QuantizedDenseExecutionRefusal, match="memory bound"):
        dense_projection(module, x, policy=too_small)
    wrong_format = QuantizedDenseExecutionPolicy(
        minimum_rows=1,
        max_dense_weight_bytes=48 * 64 * 2,
        formats=frozenset({("affine", 8, 32)}),
    )
    with pytest.raises(QuantizedDenseExecutionRefusal, match="not admitted"):
        dense_projection(module, x, policy=wrong_format)

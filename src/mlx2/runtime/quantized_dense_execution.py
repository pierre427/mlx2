"""Dense projection from quantized weight storage.

This module separates two decisions that are often conflated:

* how a checkpoint stores a linear weight; and
* which arithmetic kernel executes a particular projection.

The ordinary path remains ``nn.QuantizedLinear.__call__``.  Callers may opt in
to this primitive for a bounded large-row experiment, where the packed weight
is dequantized and consumed by a dense GEMM.  Admission is explicit and fails
closed; unsupported formats never fall back by silently changing arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
from mlx import nn


class QuantizedDenseExecutionRefusal(ValueError):
    """The packed weight or request does not satisfy the declared contract."""


@dataclass(frozen=True)
class QuantizedDenseExecutionPolicy:
    """Bounded opt-in policy for dense execution of packed linear weights."""

    minimum_rows: int
    max_dense_weight_bytes: int
    formats: frozenset[tuple[str, int, int]]

    def __post_init__(self):
        if type(self.minimum_rows) is not int or self.minimum_rows < 1:
            raise ValueError("minimum_rows must be a positive integer")
        if (
            type(self.max_dense_weight_bytes) is not int
            or self.max_dense_weight_bytes < 1
        ):
            raise ValueError("max_dense_weight_bytes must be a positive integer")
        if not self.formats or any(
            not isinstance(item, tuple)
            or len(item) != 3
            or item[0] not in {"affine", "mxfp4"}
            or type(item[1]) is not int
            or type(item[2]) is not int
            for item in self.formats
        ):
            raise ValueError("formats must contain (mode, bits, group_size) tuples")


def _shape_product(shape) -> int:
    result = 1
    for dimension in shape:
        result *= int(dimension)
    return result


def _dtype_bytes(dtype) -> int:
    if dtype in (mx.float16, mx.bfloat16):
        return 2
    if dtype == mx.float32:
        return 4
    raise QuantizedDenseExecutionRefusal(
        f"dense execution does not support activation dtype {dtype}"
    )


def geometry(module, x) -> dict:
    """Validate and describe one packed-linear/dense-input boundary."""

    if type(module) is not nn.QuantizedLinear:
        raise QuantizedDenseExecutionRefusal(
            "dense execution requires an ordinary nn.QuantizedLinear"
        )
    if not isinstance(x, mx.array) or x.ndim < 2:
        raise QuantizedDenseExecutionRefusal(
            "dense execution requires an MLX activation with a feature axis"
        )
    mode = getattr(module, "mode", None)
    bits = getattr(module, "bits", None)
    group_size = getattr(module, "group_size", None)
    if mode not in {"affine", "mxfp4"} or type(bits) is not int or type(group_size) is not int:
        raise QuantizedDenseExecutionRefusal(
            "dense execution requires declared affine or MXFP4 storage geometry"
        )
    if mode == "mxfp4" and bits != 4:
        raise QuantizedDenseExecutionRefusal("MXFP4 storage must use four bits")
    if mode == "affine" and bits not in {2, 3, 4, 5, 6, 8}:
        raise QuantizedDenseExecutionRefusal("unsupported affine storage width")
    if group_size < 1 or int(x.shape[-1]) % group_size:
        raise QuantizedDenseExecutionRefusal(
            "activation width is not integral in the quantization group size"
        )
    input_dims = int(x.shape[-1])
    output_dims = int(module.weight.shape[0])
    expected_packed = input_dims * bits // 32
    if input_dims * bits % 32 or int(module.weight.shape[-1]) != expected_packed:
        raise QuantizedDenseExecutionRefusal(
            "packed weight width disagrees with the activation and bit width"
        )
    groups = input_dims // group_size
    if tuple(module.scales.shape) != (output_dims, groups):
        raise QuantizedDenseExecutionRefusal("quantized scale geometry differs")
    biases = getattr(module, "biases", None)
    if mode == "affine" and (
        biases is None or tuple(biases.shape) != (output_dims, groups)
    ):
        raise QuantizedDenseExecutionRefusal("affine quantized biases differ")
    if mode == "mxfp4" and biases is not None:
        raise QuantizedDenseExecutionRefusal("MXFP4 storage must not carry affine biases")
    rows = _shape_product(x.shape[:-1])
    return {
        "mode": mode,
        "bits": bits,
        "group_size": group_size,
        "input_dims": input_dims,
        "output_dims": output_dims,
        "rows": rows,
        "dense_weight_bytes": input_dims * output_dims * _dtype_bytes(x.dtype),
    }


def dense_projection(module, x, *, policy: QuantizedDenseExecutionPolicy):
    """Execute an admitted packed weight as one dense projection.

    This function never chooses the route.  The caller owns selection and must
    retain its ordinary quantized reference path.  A selected call that no
    longer matches the admitted format or memory bound raises rather than
    falling back under the same route receipt.
    """

    receipt = geometry(module, x)
    signature = (receipt["mode"], receipt["bits"], receipt["group_size"])
    if signature not in policy.formats:
        raise QuantizedDenseExecutionRefusal(
            f"quantized storage format {signature!r} was not admitted"
        )
    if receipt["rows"] < policy.minimum_rows:
        raise QuantizedDenseExecutionRefusal(
            "activation row count is below the declared dense-execution frontier"
        )
    if receipt["dense_weight_bytes"] > policy.max_dense_weight_bytes:
        raise QuantizedDenseExecutionRefusal(
            "dequantized weight exceeds the declared transient-memory bound"
        )
    packed = [module.weight, module.scales]
    quantized_biases = getattr(module, "biases", None)
    if quantized_biases is not None:
        packed.append(quantized_biases)
    weight = mx.dequantize(
        *packed,
        group_size=module.group_size,
        bits=module.bits,
        mode=module.mode,
    ).astype(x.dtype)
    result = x @ weight.T
    if "bias" in module:
        result = result + module.bias
    return result

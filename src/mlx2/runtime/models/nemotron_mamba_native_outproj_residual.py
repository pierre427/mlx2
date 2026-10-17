"""Native-QMM Nemotron Mamba output projection plus residual boundary.

This candidate deliberately leaves MLX's quantized-matmul arithmetic intact.
The adapter-owned compiled graph only extends the boundary through the following
BF16 residual add, allowing MLX to schedule or fuse that epilogue when its
native backend supports it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx import nn

INPUT_WIDTH = 8192
OUTPUT_WIDTH = 4096
GROUP_SIZE = 64


@dataclass(frozen=True)
class NativeOutprojResidualAdmission:
    accepted: bool
    reason: str


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(getattr(value, "shape", ()))


def admit_native_outproj_residual_b1(
    *, y, residual, projection, training: bool
) -> NativeOutprojResidualAdmission:
    if training:
        return NativeOutprojResidualAdmission(False, "training")
    if _shape(y) != (1, 1, INPUT_WIDTH):
        return NativeOutprojResidualAdmission(False, f"input shape {_shape(y)}")
    if _shape(residual) != (1, 1, OUTPUT_WIDTH):
        return NativeOutprojResidualAdmission(
            False, f"residual shape {_shape(residual)}"
        )
    if not isinstance(projection, nn.QuantizedLinear):
        return NativeOutprojResidualAdmission(False, "requires QuantizedLinear")
    if int(projection.bits) != 8 or int(projection.group_size) != GROUP_SIZE:
        return NativeOutprojResidualAdmission(False, "requires affine q8 group64")
    if getattr(projection, "mode", "affine") != "affine":
        return NativeOutprojResidualAdmission(False, "requires affine q8 group64")
    if _shape(projection.weight) != (OUTPUT_WIDTH, INPUT_WIDTH // 4):
        return NativeOutprojResidualAdmission(
            False, f"weight shape {_shape(projection.weight)}"
        )
    groups = INPUT_WIDTH // GROUP_SIZE
    if _shape(projection.scales) != (OUTPUT_WIDTH, groups):
        return NativeOutprojResidualAdmission(
            False, f"scales shape {_shape(projection.scales)}"
        )
    if _shape(projection.biases) != (OUTPUT_WIDTH, groups):
        return NativeOutprojResidualAdmission(
            False, f"biases shape {_shape(projection.biases)}"
        )
    if getattr(projection, "bias", None) is not None:
        return NativeOutprojResidualAdmission(False, "linear bias")
    if any(
        value.dtype != mx.bfloat16
        for value in (y, residual, projection.scales, projection.biases)
    ):
        return NativeOutprojResidualAdmission(False, "requires bf16 operands")
    return NativeOutprojResidualAdmission(True, "eligible")


@mx.compile
def _native_qmm_residual(y, residual, weight, scales, biases):
    projected = mx.quantized_matmul(
        y,
        weight,
        scales=scales,
        biases=biases,
        transpose=True,
        group_size=GROUP_SIZE,
        bits=8,
        mode="affine",
    )
    return residual + projected


def native_outproj_residual_b1(y, residual, projection):
    """Use native MLX QMM arithmetic and own its following residual add."""
    return _native_qmm_residual(
        y,
        residual,
        projection.weight,
        projection.scales,
        projection.biases,
    )

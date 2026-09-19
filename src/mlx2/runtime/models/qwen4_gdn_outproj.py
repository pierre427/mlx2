# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; see docs/PROVENANCE.md and provenance/flashnext.json.
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import mlx.core as mx

INPUT_DIM = 6144
OUTPUT_DIM = 2560
GROUP_SIZE = 64
BITS = 4


@dataclass(frozen=True)
class GdnOutprojAdmission:
    accepted: bool
    reason: str


def admit_qwen4_gdn_outproj(module: Any, x: Any) -> GdnOutprojAdmission:
    if tuple(getattr(x, "shape", ())) != (1, 1, INPUT_DIM):
        return GdnOutprojAdmission(False, "only B1/M1/6144 input")
    if getattr(x, "dtype", None) != mx.bfloat16:
        return GdnOutprojAdmission(False, "input must be bfloat16")
    if not all((hasattr(module, name) for name in ("weight", "scales", "biases"))):
        return GdnOutprojAdmission(False, "output projection is not affine-quantized")
    if getattr(module, "group_size", None) != GROUP_SIZE:
        return GdnOutprojAdmission(False, "requires affine group-size 64")
    if getattr(module, "bits", None) != BITS:
        return GdnOutprojAdmission(False, "requires affine q4")
    if tuple(module.weight.shape) != (OUTPUT_DIM, INPUT_DIM // 8):
        return GdnOutprojAdmission(False, f"weight shape {tuple(module.weight.shape)}")
    expected_groups = INPUT_DIM // GROUP_SIZE
    for name in ("scales", "biases"):
        value = getattr(module, name)
        if tuple(value.shape) != (OUTPUT_DIM, expected_groups):
            return GdnOutprojAdmission(False, f"{name} shape {tuple(value.shape)}")
        if value.dtype != mx.bfloat16:
            return GdnOutprojAdmission(False, f"{name} must be bfloat16")
    if module.weight.dtype != mx.uint32:
        return GdnOutprojAdmission(False, "weight must be packed uint32")
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return GdnOutprojAdmission(False, "Metal GPU unavailable")
    return GdnOutprojAdmission(True, "eligible")

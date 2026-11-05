"""CPU projection comparisons on supplied arrays, never model qualification."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from mlx2.runtime.weight_space_metrics import set_weight_space_gauge, track_weight_space

from .programs import MAX_TOTAL_ELEMENTS, WeightProgram, _array, _json


def _hash(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def _difference(actual, expected):
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    denominator = float(np.linalg.norm(expected.astype(np.float64)))
    error = float(np.linalg.norm(delta))
    return {
        "exact_array_equal": bool(np.array_equal(actual, expected)),
        "max_absolute_error": float(np.max(np.abs(delta))),
        "rms_error": float(np.sqrt(np.mean(delta * delta))),
        "relative_frobenius_error": error / denominator if denominator else None,
        "absolute_frobenius_error": error,
    }


def round_bfloat16(value):
    """Simulate BF16 round-to-nearest-even storage in a float32 NumPy array."""
    value = np.ascontiguousarray(value, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError("BF16 simulation requires finite float32 input")
    bits = value.view(np.uint32)
    rounded = ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)).view(np.float32)
    if not np.isfinite(rounded).all():
        raise ValueError("BF16 rounding overflow")
    return rounded


@track_weight_space("projection_evaluate")
def evaluate_projections(program: WeightProgram, weights, inputs, *, simulate_bfloat16=False):
    """Compare supplied output×input weights with dynamic/fused CPU projections.

    The supplied weights' identity is recorded but cannot establish that they
    came from the declared model checkpoint. No model/functional oracle runs.
    """
    if type(simulate_bfloat16) is not bool:
        raise ValueError("simulate_bfloat16 must be boolean")
    targets = set(program.factor_map)
    if set(weights) != targets or set(inputs) != targets:
        raise ValueError("weights and inputs must cover exactly the program targets")
    validated, total = {}, 0
    for target, (a, b) in program.factor_map.items():
        w, x = _array(weights[target]), _array(inputs[target])
        if w.shape != (b.shape[1], a.shape[0]) or x.shape[1] != a.shape[0]:
            raise ValueError("projection capture dimensions do not match factors")
        total += w.size + x.size + x.shape[0] * b.shape[1]
        if total > MAX_TOTAL_ELEMENTS:
            raise ValueError("projection evaluation exceeds aggregate element bound")
        validated[target] = w, x
    rows = []
    for target, (a, b) in program.factor_map.items():
        w, x = validated[target]
        with np.errstate(over="ignore", invalid="ignore"):
            base = x @ w.T
            delta = (x @ a) @ b
            dynamic = base + delta
            fused = x @ (w + (a @ b).T).T
            zero = base + ((x @ np.zeros_like(a)) @ b)
        if not all(np.isfinite(v).all() for v in (base, delta, dynamic, fused, zero)):
            raise ValueError("projection arithmetic produced nonfinite values")
        result = {
            "target": target,
            "input_shape": list(x.shape),
            "weight_shape": list(w.shape),
            "inputs_sha256": _hash(x),
            "weights_sha256": _hash(w),
            "float32": {
                "dynamic_vs_fused": _difference(dynamic, fused),
                "dynamic_vs_base": _difference(dynamic, base),
                "zero_vs_base": _difference(zero, base),
                "delta_frobenius_norm": float(np.linalg.norm(delta.astype(np.float64))),
            },
        }
        if simulate_bfloat16:
            xb, wb = round_bfloat16(x), round_bfloat16(w)
            base32 = xb @ wb.T
            dynamic_bf = round_bfloat16(base32 + (xb @ a) @ b)
            fused_bf = round_bfloat16(xb @ round_bfloat16(wb + (a @ b).T).T)
            result["simulated_bfloat16"] = {
                "dynamic_vs_fused": _difference(dynamic_bf, fused_bf),
                "dynamic_vs_base": _difference(dynamic_bf, round_bfloat16(base32)),
                "arithmetic": "BF16-rounded inputs/base weights, CPU FP32 matmuls; fused weights and final outputs rounded to BF16",
                "device_arithmetic_reproduced": False,
            }
        rows.append(result)
    set_weight_space_gauge("projection_delta_norm", sum(row["float32"]["delta_frobenius_norm"] for row in rows))
    return {
        "schema": "mlx2.weight-program-projection-evaluation.v1",
        "checkpoint": program.checkpoint,
        "source_revision": program.source_revision,
        "program_id": program.program_id,
        "evidence_scope": "supplied_projection_arrays_cpu_only",
        "checkpoint_tensor_origin_verified": False,
        "model_executed": False,
        "qualified": False,
        "observed_used": False,
        "float32_arithmetic": "CPU NumPy float32; dynamic=(X@W.T)+((X@A)@B), fused=X@(W+(A@B).T).T",
        "targets": rows,
    }


def write_evaluation(report, output):
    """Publish a receipt in one atomic fresh-directory rename."""
    payload = _json(report) + "\n"
    output = Path(output).expanduser().absolute()
    if output.is_symlink() or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
        raise ValueError("evaluation output must be absent or empty")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        path = temporary / "report.json"
        path.write_text(payload)
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        descriptor = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output / "report.json"

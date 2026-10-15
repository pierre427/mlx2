#!/usr/bin/env python3
"""Bounded SIMD-q4 arithmetic diagnostic; not model qualification.

The caller must own the GPU campaign window before --execute-metal. Importing
this file only builds NumPy fixtures. Every arm uses a sparse dyadic product
whose independent float64 reference is exactly representable in BF16, so this
probe has no tolerance that can hide a lost tiny activation. The large but
finite group scale makes the final output ordinary-sized even when the input
is the smallest normal BF16 value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


def adversarial_case(group_size: int, rows: int, *, exponent: int):
    """All eight nibble positions, both signs, and two independent groups."""
    k, n = group_size * 2, 8
    x = np.zeros((rows, k), dtype=np.float32)
    codes = np.zeros((n, k), dtype=np.uint32)
    for row in range(rows):
        # Unequal rows catch wrong row strides and partial-tile replication.
        x[row, :8] = np.ldexp(np.float32((-1) ** row), exponent)
        x[row, group_size:group_size + 8] = np.ldexp(
            np.float32((-1) ** (row + 1)), exponent + 1
        )
    for output in range(n):
        codes[output, output] = 15
        codes[output, group_size + output] = 3
    packed = np.zeros((n, k // 8), dtype=np.uint32)
    for nibble in range(8):
        packed |= codes[:, nibble::8] << (4 * nibble)
    scales = np.full((n, k // group_size), np.ldexp(np.float32(1), 96), dtype=np.float32)
    biases = np.zeros_like(scales)
    # Decode from the packed artifact independently of the source codes.
    decoded = np.stack([(packed >> (4 * i)) & 15 for i in range(8)], axis=-1).reshape(n, k)
    weights = decoded.astype(np.float64) * np.repeat(scales.astype(np.float64), group_size, axis=1)
    reference = x.astype(np.float64) @ weights.T
    return x, packed, scales, biases, reference


def run_metal():
    import mlx.core as mx

    from mlx2.runtime.lane import simd
    from mlx2.runtime.lane.installer import LAW_IDS

    mx.set_default_device(mx.gpu)
    records = []
    for group in (32, 64):
        for exponent in (-126, -120, -80):
            for rows in (1, 2, 3, 4, 5, 7, 8, 9, 16):
                x, packed, scales, biases, reference = adversarial_case(group, rows, exponent=exponent)
                arrays = [mx.array(x, dtype=mx.bfloat16), mx.array(packed, dtype=mx.uint32),
                          mx.array(scales, dtype=mx.bfloat16), mx.array(biases, dtype=mx.bfloat16)]
                # Exercise the selected path and direct matrix path for every
                # width, plus the scalar twin's supported small widths.
                scalar_supported = simd._scalar_block(rows, simd.splits(8, group * 2), group)
                kinds = [None, "mma"] + (["scalar"] if scalar_supported else [])
                for kind in kinds:
                    result = simd.qmm(*arrays, group, 4, kind=kind).astype(mx.float32)
                    mx.eval(result)
                    actual = np.asarray(result).astype(np.float64)
                    records.append({"group_size": group, "rows": rows, "exponent": exponent,
                                    "kind": kind or "selected", "passed": bool(np.array_equal(actual, reference)),
                                    "max_abs_error": float(np.max(np.abs(actual - reference)))})
    source = Path(simd.__file__).read_bytes()
    info = mx.device_info()
    return {"schema": "mlx2.simd-q4-exponents.v1", "kind": "component-diagnostic",
            "qualified": False, "law_id": LAW_IDS["simd"],
            "source_sha256": hashlib.sha256(source).hexdigest(),
            "mlx_version": mx.__version__,
            "device": {key: str(info.get(key)) for key in ("device_name", "architecture")},
            "cases": records, "passed": all(item["passed"] for item in records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-metal", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpg-task", required=True)
    parser.add_argument("--cpg-owner-lock", type=Path, required=True)
    parser.add_argument("--cpg-generation", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to replace evidence: {args.output}")
    # Validate radio ownership, generation, both paired locks, and ancestry
    # before importing MLX or creating a device-backed object.
    from qualification_gpu_ownership import require_qualification_lease

    ownership = require_qualification_lease(
        task_id=args.cpg_task, cpg_owner_lock=args.cpg_owner_lock,
        generation=args.cpg_generation,
    )
    receipt = run_metal()
    receipt["ownership"] = ownership
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a complete receipt, without overwriting another run.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(receipt, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, args.output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(json.dumps({"passed": receipt["passed"], "cases": len(receipt["cases"])}))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Owned component probe for fused Super Mamba Q8 out-projection + residual."""
from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import mlx.core as mx
import nemotron35_probe_common as common
from mlx import nn

from mlx2.runtime.models.nemotron_mamba_outproj_residual import (
    admit_fused_mamba_outproj_residual_b1,
    fused_mamba_outproj_residual_b1,
)


def bits_sha(value) -> str:
    return hashlib.sha256(bytes(memoryview(mx.array(value).view(mx.uint16)))).hexdigest()


def elapsed_ms(call, repetitions: int) -> float:
    for _ in range(5):
        mx.eval(call())
    started = time.perf_counter()
    for _ in range(repetitions):
        mx.eval(call())
    return (time.perf_counter() - started) * 1000.0 / repetitions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cpg-owner-lock", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("Refusing to overwrite nonempty probe evidence")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    owner = common.require_owner(args.cpg_owner_lock)
    before = common.host_snapshot()
    errors = common.admission_errors(before, minimum_available_bytes=6 * common.GIB)
    if errors:
        raise RuntimeError("; ".join(errors))

    mx.random.seed(314159)
    projection = nn.QuantizedLinear(
        8192, 4096, bias=False, group_size=64, bits=8, mode="affine"
    )
    # Converted protected projections store both affine parameters in BF16.
    projection.scales = projection.scales.astype(mx.bfloat16)
    projection.biases = projection.biases.astype(mx.bfloat16)
    projection.eval()
    mx.eval(projection.parameters())
    cases = []
    for seed, scale in ((11, 0.25), (17, 1.0), (23, 4.0)):
        mx.random.seed(seed)
        y = (mx.random.normal((1, 1, 8192)) * scale).astype(mx.bfloat16)
        residual = (mx.random.normal((1, 1, 4096)) * scale).astype(mx.bfloat16)
        mx.eval(y, residual)
        admission = admit_fused_mamba_outproj_residual_b1(
            y=y, residual=residual, projection=projection, training=False
        )
        if not admission.accepted:
            raise RuntimeError(admission.reason)
        expected = residual + projection(y)
        actual = fused_mamba_outproj_residual_b1(y, residual, projection)
        mx.eval(expected, actual)
        exact = bool(mx.array_equal(expected, actual).item())
        mismatches = int(
            mx.sum(expected.view(mx.uint16) != actual.view(mx.uint16)).item()
        )
        delta = mx.abs(expected.astype(mx.float32) - actual.astype(mx.float32))
        stock_ms = elapsed_ms(
            lambda y=y, residual=residual: residual + projection(y), 50
        )
        fused_ms = elapsed_ms(
            lambda y=y, residual=residual: fused_mamba_outproj_residual_b1(
                y, residual, projection
            ),
            50,
        )
        cases.append(
            {
                "seed": seed,
                "scale": scale,
                "exact": exact,
                "mismatch_count": mismatches,
                "max_abs": float(mx.max(delta).item()),
                "expected_sha256": bits_sha(expected),
                "actual_sha256": bits_sha(actual),
                "stock_ms": stock_ms,
                "fused_ms": fused_ms,
                "relative_percent": (stock_ms / fused_ms - 1.0) * 100.0,
            }
        )
    after = common.host_snapshot()
    contamination = [
        reason
        for reason in common.contamination(before, after)
        if reason != "swap-outs increased"
    ]
    receipt = {
        "schema": "mlx2.nemotron35-super-mamba-outproj-residual-probe.v1",
        "owner": owner,
        "source": common.source_identity(),
        "dependencies": common.runtime_dependencies(),
        "before": before,
        "after": after,
        "swap_ignored_by_user": True,
        "geometry": {
            "batch": 1,
            "steps": 1,
            "input_width": 8192,
            "output_width": 4096,
            "bits": 8,
            "group_size": 64,
        },
        "cases": cases,
        "passed": all(case["exact"] for case in cases) and not contamination,
        "contamination": contamination,
    }
    common.atomic_json(args.out_dir / "result.json", receipt)
    if not receipt["passed"]:
        raise RuntimeError(
            "Mamba out-projection + residual exactness probe did not pass"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Owned fixed-HC projection row-law fixture; no model or qualification claim."""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from probe_deepseek_v4_hc_rowlaw import fixture, row_difference
from probe_deepseek_v4_packed import atomic, compare, source_receipt, verify_ownership


def run_case(outputs, kind, *, guard, repetitions):
    import mlx.core as mx

    from mlx2.runtime.models.deepseek_v4_hc_projection import project

    raw, half = fixture(outputs, kind)
    # Author one fixed FP32 input on CPU. Projection-only probe, no RMS fusion.
    norm = (
        raw.astype(np.float64)
        / np.sqrt(np.mean(raw.astype(np.float64) ** 2, axis=-1, keepdims=True) + 1e-20)
    ).astype(np.float32)
    weight = half.astype(np.float32)
    oracle = norm.astype(np.float64) @ weight.astype(np.float64).T
    x = mx.array(norm)
    w = mx.array(weight)
    mx.eval(x, w)
    singles = []
    for row in range(6):
        guard()
        a = project(x[row : row + 1], w)
        mx.eval(a)
        singles.append(np.array(a))
    expected = np.concatenate(singles)
    results = []
    for width in range(1, 7):
        guard()
        start = time.perf_counter()
        a = project(x[:width], w)
        mx.eval(a)
        first = time.perf_counter() - start
        host = np.array(a)
        warm = []
        rowlaw = row_difference(host, expected[:width])
        numerical = compare(host, oracle[:width], atol=5e-5, rtol=5e-5)
        # Only a candidate satisfying both laws gets observational timings.
        if rowlaw["bitwise_equal"] and numerical["passed"]:
            for _ in range(repetitions):
                guard()
                start = time.perf_counter()
                a = project(x[:width], w)
                mx.eval(a)
                warm.append(time.perf_counter() - start)
        results.append(
            {
                "outputs": outputs,
                "fixture": kind,
                "K": 16384,
                "rows": width,
                "rowlaw": rowlaw,
                "independent_float64_oracle": numerical,
                "normalized_input_sha256": hashlib.sha256(
                    norm[:width].tobytes()
                ).hexdigest(),
                "weight_fp32_sha256": hashlib.sha256(weight.tobytes()).hexdigest(),
                "first_evaluation_seconds": first,
                "warm_seconds": warm,
                "timing_scope": "observational synthetic diagnostic; compiler cache unproven; no controlled performance",
                "native_gemv_bitwise_parity_claim": False,
            }
        )
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run-gpu", action="store_true")
    p.add_argument("--ownership-window", type=Path)
    p.add_argument("--owner-path", type=Path, action="append")
    p.add_argument("--repetitions", type=int, default=3)
    args = p.parse_args(argv)
    if args.output.exists():
        p.error("refusing existing evidence directory")
    if not args.run_gpu:
        p.error(
            "GPU probe requires explicit --run-gpu and existing ownership; CPU tests separate"
        )
    if not 1 <= args.repetitions <= 5:
        p.error("bounded repetitions1..5")
    owners = args.owner_path or [
        Path("/Users/Shared/mlxuag/gpu.lock/owner.json"),
        Path("/tmp/gpu.lock/owner.json"),
    ]
    if args.ownership_window is None or len(owners) != 2:
        p.error("existing window and matching paired owners required")

    def guard():
        return verify_ownership(args.ownership_window, owners)

    ownership = guard()
    source = source_receipt()
    identity = hashlib.sha256(
        b"".join(
            a.tobytes() + w.tobytes()
            for o in (4, 24)
            for k in ("random", "cancellation")
            for a, w in [fixture(o, k)]
        )
    ).hexdigest()
    receipt = {
        "schema": "mlx2.deepseek-v4-fixed-hc-projection-fixtures.v1",
        "status": "running",
        "source": source,
        "checkpoint": "synthetic-HC16384-fixed-projection@" + identity,
        "ownership": ownership,
        "actual_artifact_loaded": False,
        "qualification": False,
        "performance_claim": False,
        "native_gemv_bitwise_parity_claim": False,
        "cases": [],
    }
    args.output.mkdir(parents=True)
    atomic(args.output / "results.json", receipt)
    import mlx.core as mx

    previous = mx.default_device()
    try:
        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(256 << 20)
        mx.set_cache_limit(32 << 20)
        receipt["device_info"] = mx.metal.device_info()
        for outputs in (24, 4):
            for kind in ("random", "cancellation"):
                receipt["cases"] += run_case(
                    outputs, kind, guard=guard, repetitions=args.repetitions
                )
                atomic(args.output / "results.json", receipt)
        receipt["passed"] = all(
            c["rowlaw"]["bitwise_equal"] and c["independent_float64_oracle"]["passed"]
            for c in receipt["cases"]
        )
        receipt["status"] = "completed"
        guard()
    except BaseException as error:
        receipt.update(
            status="failed", error={"type": type(error).__name__, "message": str(error)}
        )
        raise
    finally:
        atomic(args.output / "results.json", receipt)
        mx.set_default_device(previous)
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Owned, bounded exactness probe for the Super gate/group-RMS candidate."""
from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import mlx.core as mx
import nemotron35_probe_common as common

from mlx2.runtime.models.nemotron_h import MambaRMSNormGated
from mlx2.runtime.models.nemotron_mamba_norm import fused_mamba_norm_refusal


def bits_sha(value) -> str:
    return hashlib.sha256(bytes(memoryview(mx.array(value).view(mx.uint16)))).hexdigest()


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
    errors = common.admission_errors(before, minimum_available_bytes=4 * common.GIB)
    if errors:
        raise RuntimeError("; ".join(errors))

    refusal = fused_mamba_norm_refusal(mx.bfloat16)
    if refusal is not None:
        raise RuntimeError(refusal)
    cases = []
    for steps, seed, scale in (
        (1, 11, 0.25), (1, 17, 1.0), (4, 23, 3.0), (8, 29, 8.0)
    ):
        mx.random.seed(seed)
        y = (mx.random.normal((1, steps, 8192)) * scale).astype(mx.bfloat16)
        gate = (mx.random.normal((1, steps, 8192)) * scale).astype(mx.bfloat16)
        weight = (1 + 0.15 * mx.random.normal((8192,))).astype(mx.bfloat16)
        mx.eval(y, gate, weight)
        norm = MambaRMSNormGated(8192, 1e-5, 1024)
        norm.weight = weight
        norm.eval()
        expected = norm(y, gate)
        mx.eval(expected)
        norm.set_policy("fused_short")
        actual = norm(y, gate)
        mx.eval(actual)
        # Compile/warm before collecting a small component-only timing sample.
        for _ in range(10):
            mx.eval(norm(y, gate))
        started = time.perf_counter()
        for _ in range(100):
            mx.eval(norm(y, gate))
        fused_ms = (time.perf_counter() - started) * 10.0
        norm.set_policy("auto")
        for _ in range(10):
            mx.eval(norm(y, gate))
        started = time.perf_counter()
        for _ in range(100):
            mx.eval(norm(y, gate))
        ordinary_ms = (time.perf_counter() - started) * 10.0
        exact = bool(mx.array_equal(expected, actual).item())
        mismatch_count = int(mx.sum(expected.view(mx.uint16) != actual.view(mx.uint16)).item())
        cases.append({
            "seed": seed,
            "steps": steps,
            "scale": scale,
            "exact": exact,
            "mismatch_count": mismatch_count,
            "expected_sha256": bits_sha(expected),
            "actual_sha256": bits_sha(actual),
            "status": norm.status(),
            "ordinary_ms": ordinary_ms,
            "fused_ms": fused_ms,
            "relative_percent": (ordinary_ms / fused_ms - 1.0) * 100.0,
        })
    after = common.host_snapshot()
    contamination = common.contamination(before, after)
    # The user explicitly directed this campaign to ignore pre-existing swap;
    # a component probe still records it, but only process/service/thermal
    # contamination can invalidate this bounded exactness result.
    contamination = [reason for reason in contamination if reason != "swap-outs increased"]
    receipt = {
        "schema": "mlx2.nemotron35-super-mamba-gate-group-rms-probe.v1",
        "owner": owner,
        "source": common.source_identity(),
        "dependencies": common.runtime_dependencies(),
        "before": before,
        "after": after,
        "swap_ignored_by_user": True,
        "cases": cases,
        "passed": all(case["exact"] for case in cases) and not contamination,
        "contamination": contamination,
    }
    common.atomic_json(args.out_dir / "result.json", receipt)
    if not receipt["passed"]:
        raise RuntimeError("Mamba gate/group-RMS exactness probe failed")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Owned grouped packed-GGUF NAX fixtures; explicit FP16 operands, not qualification."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from probe_deepseek_v4_packed import (
    atomic,
    compare,
    independent_dequant,
    measure,
    source_receipt,
    synthetic,
    upload,
    verify_ownership,
)

FORMATS = {8: "Q8_0", 10: "Q2_K", 16: "IQ2_XXS"}


def working_bound(rows, k, n, groups, kind):
    block, size = {8: (32, 34), 10: (256, 84), 16: (256, 66)}[kind]
    packed = groups * n * k // block * size
    # Host+device packed, independent decoder for ONE group at a time,
    # FP32+half-rounded activations, two oracle/output pairs and compiler slack.
    return (
        256 * (1 << 20)
        + 2 * packed
        + 4 * n * k * 4
        + 4 * rows * groups * k * 4
        + 6 * rows * groups * n * 4
    )


def fixture(rows, k, n, kind, groups=8):
    if rows < 1 or k % 256 or n % 32 or groups != 8:
        raise ValueError("fixtures require positive rows,G8,Kmultiple256,Nmultiple32")
    rng = np.random.default_rng(20261024 + rows + kind)
    x = rng.normal(0, 0.2, (rows, groups, k)).astype(np.float32)
    x += np.arange(groups, dtype=np.float32)[None, :, None] * 0.025
    packed = synthetic((groups, n, k), kind, seed=17928 + kind)
    return x, packed


def independent_oracles(x, packed, kind, n):
    rows, groups, k = x.shape
    half = np.empty((rows, groups, n), np.float32)
    full = np.empty_like(half)
    # Each group has independent weights and activations. No flattened dense
    # surrogate or transposed group/time packing can accidentally pass.
    for group in range(groups):
        weight = independent_dequant(packed[group], kind, (n, k))
        full[:, group] = x[:, group] @ weight.T
        half[:, group] = (
            x[:, group].astype(np.float16).astype(np.float32)
            @ weight.astype(np.float16).astype(np.float32).T
        )
    return half, full


def run_case(rows, k, n, kind, *, gpu, repetitions, cap, guard):
    bound = working_bound(rows, k, n, 8, kind)
    if bound > cap:
        raise ValueError("grouped fixture allocation bound exceeds working cap")
    x, packed = fixture(rows, k, n, kind)
    half, full = independent_oracles(x, packed, kind, n)
    result = {
        "rows": rows,
        "groups": 8,
        "input": k,
        "output_per_group": n,
        "format": FORMATS[kind],
        "working_bound_bytes": bound,
        "qualified": False,
        "artifact_loaded": False,
        "route": "grouped_nax_fp16_operands_v1",
        "oracle": "independent gguf package block decode; per-group FP16rounded operands then FP32 matmul",
        "cpu_oracles_finite": bool(np.isfinite(half).all() and np.isfinite(full).all()),
    }
    if not gpu:
        return result
    import mlx.core as mx

    from mlx2.runtime.gguf.cooperative_metal import cooperative_grouped
    from mlx2.runtime.gguf.nax_metal import nax_grouped_prefill
    from mlx2.runtime.gguf.packed_metal import grouped_linear

    weight = upload(packed, kind, (8, n, k), "synthetic-grouped")
    xm = mx.array(x)
    guard()
    # Correctness must pass before warm repetitions or comparative timing.
    first_started = time.perf_counter()
    actual = nax_grouped_prefill(xm, weight)
    mx.eval(actual)
    result["candidate_first_evaluation_seconds"] = time.perf_counter() - first_started
    result["compiler_cache_state"] = (
        "unproven; first process evaluation may use persistent Metal cache"
    )
    if actual.shape != half.shape:
        raise RuntimeError("grouped output shape differs from independent oracle")
    numerical = compare(actual, half, atol=0.005, rtol=0.003)
    result["numerical_vs_half_operand_oracle"] = numerical
    if not numerical["passed"]:
        result["timing_skipped"] = "candidate numerical gate failed"
        return result
    _, result["nax_timing"] = measure(
        lambda: nax_grouped_prefill(xm, weight), repetitions
    )
    baseline = (
        (lambda: cooperative_grouped(xm, weight, rows=4))
        if kind in (8, 16)
        else (lambda: grouped_linear(xm, weight))
    )
    baseline_name = (
        "cooperative_grouped_rows4" if kind in (8, 16) else "ordinary_grouped_fp32"
    )
    baseline_actual = baseline()
    mx.eval(baseline_actual)
    baseline_check = compare(baseline_actual, full, atol=0.005, rtol=0.003)
    result["baseline"] = {
        "name": baseline_name,
        "different_arithmetic": True,
        "numerical_vs_fp32_oracle": baseline_check,
        "cooperative_grouped_available": kind in (8, 16),
    }
    if baseline_check["passed"]:
        _, timing = measure(baseline, repetitions)
        result["baseline"]["timing"] = timing
        result["observational_kernel_speed_ratio"] = (
            timing["warm_median_seconds"] / result["nax_timing"]["warm_median_seconds"]
        )
    else:
        result["baseline"]["timing_skipped"] = "baseline numerical gate failed"
    result["half_vs_full_operand_difference"] = compare(
        half, full, atol=0.005, rtol=0.003
    )
    result["allocator_peak_bytes"] = mx.get_peak_memory()
    guard()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-gpu", action="store_true")
    parser.add_argument("--ownership-window", type=Path)
    parser.add_argument("--owner-path", type=Path, action="append")
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 16, 128, 512])
    parser.add_argument("--formats", type=int, nargs="+", default=[8, 10, 16])
    parser.add_argument("--full-shapes", action="store_true")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--max-working-bytes", type=int, default=1 << 30)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("refusing existing evidence directory")
    if (
        any(type(x) is not int or not 1 <= x <= 512 for x in args.rows)
        or any(x not in FORMATS for x in args.formats)
        or not 1 <= args.repetitions <= 20
        or not 0 < args.max_working_bytes < 2 << 30
    ):
        parser.error(
            "rows1..512,format8/10/16,repetitions1..20 and workingcap<2GiB required"
        )
    owners = args.owner_path or [
        Path("/Users/Shared/mlxuag/gpu.lock/owner.json"),
        Path("/tmp/gpu.lock/owner.json"),
    ]
    if args.run_gpu and (args.ownership_window is None or len(owners) != 2):
        parser.error("GPU requires independently active CPG/paired ownership")

    def guard():
        return verify_ownership(args.ownership_window, owners) if args.run_gpu else None

    ownership = guard()
    source = source_receipt()
    from mlx2.runtime.gguf import cooperative_metal, nax_metal

    source["grouped_nax_module_sha256"] = hashlib.sha256(
        Path(nax_metal.__file__).read_bytes()
    ).hexdigest()
    source["cooperative_module_sha256"] = hashlib.sha256(
        Path(cooperative_metal.__file__).read_bytes()
    ).hexdigest()
    source["grouped_probe_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    k, n = (4096, 1024) if args.full_shapes else (256, 64)
    for rows in args.rows:
        for kind in args.formats:
            if working_bound(rows, k, n, 8, kind) > args.max_working_bytes:
                parser.error(
                    "conservative fixture memory bound exceeds requested cap before allocation"
                )
    args.output.mkdir(parents=True)
    fixture_hash = hashlib.sha256(
        json.dumps(
            [args.rows, args.formats, k, n, 20261024, "group-z-strided-v1"]
        ).encode()
    ).hexdigest()
    receipt = {
        "schema": "mlx2.deepseek-v4-grouped-nax-fixtures.v1",
        "status": "running",
        "source": source,
        "ownership": ownership,
        "checkpoint": "synthetic-grouped-packed@sha256:" + fixture_hash,
        "run_gpu": args.run_gpu,
        "qualification": False,
        "performance_claim": False,
        "artifact_loaded": False,
        "cases": [],
        "timing_scope": "observational component fixtures; no thermally controlled or end-to-end model speed claim",
        "first_evaluation": "numerical gate occurs before measure; measure first_seconds is a post-gate dispatch, not cold compilation",
    }
    atomic(args.output / "results.json", receipt)
    try:
        if args.run_gpu:
            import mlx.core as mx

            mx.set_default_device(mx.gpu)
            mx.set_memory_limit(args.max_working_bytes)
            mx.set_cache_limit(64 << 20)
            receipt["device_info"] = mx.metal.device_info()
        for rows in args.rows:
            for kind in args.formats:
                guard()
                case = run_case(
                    rows,
                    k,
                    n,
                    kind,
                    gpu=args.run_gpu,
                    repetitions=args.repetitions,
                    cap=args.max_working_bytes,
                    guard=guard,
                )
                receipt["cases"].append(case)
                atomic(args.output / "results.json", receipt)
                print(
                    json.dumps(
                        {
                            "rows": rows,
                            "format": kind,
                            "numerical": case.get("numerical_vs_half_operand_oracle"),
                            "timing": case.get("nax_timing"),
                            "ratio": case.get("observational_kernel_speed_ratio"),
                        }
                    ),
                    flush=True,
                )
                gc.collect()
                if args.run_gpu:
                    mx.clear_cache()
        receipt["all_numerical_checks_passed"] = all(
            c["cpu_oracles_finite"]
            and c.get("numerical_vs_half_operand_oracle", {}).get("passed", True)
            and c.get("baseline", {})
            .get("numerical_vs_fp32_oracle", {})
            .get("passed", True)
            for c in receipt["cases"]
        )
        receipt["status"] = "completed"
    except BaseException as error:
        receipt.update(
            status="failed", error={"type": type(error).__name__, "message": str(error)}
        )
        raise
    finally:
        atomic(args.output / "results.json", receipt)
    return 0 if receipt["all_numerical_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

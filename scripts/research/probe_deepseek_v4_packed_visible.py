#!/usr/bin/env python3
"""Owned direct-packed-pool attention fixtures; exact visible FP32 comparison."""

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
    measure,
    source_receipt,
    verify_ownership,
)

from mlx2.runtime.models.deepseek_v4_packed_kv import (
    PackedKV,
    pack_fp8_kv,
    unpack_fp8_kv,
)
from mlx2.runtime.models.deepseek_v4_visible_attention import (
    reference_visible_attention,
)

PLANS = (
    {"name": "local-only", "ratio": 0, "offset": 120, "length": 8},
    {"name": "ratio4-boundary", "ratio": 4, "offset": 125, "length": 5},
    {"name": "ratio128-boundary", "ratio": 128, "offset": 125, "length": 5},
    {"name": "ratio4-top512", "ratio": 4, "offset": 4093, "length": 8, "topk": 512},
    {"name": "ratio128-long-offset", "ratio": 128, "offset": 32765, "length": 8},
    {
        "name": "ratio4-long-offset-top512",
        "ratio": 4,
        "offset": 262141,
        "length": 8,
        "topk": 512,
    },
    {"name": "sink-dominates", "ratio": 4, "offset": 13, "length": 3, "sink": 1000.0},
)


def fixture(plan, heads=64):
    rng = np.random.default_rng(17928 + plan["offset"] + plan["ratio"])
    length = plan["length"]
    offset = plan["offset"]
    ratio = plan["ratio"]
    count = 0 if not ratio else (offset + length) // ratio
    q = rng.normal(0, 0.2, (1, heads, length, 512)).astype(np.float32)
    local = rng.normal(0, 0.3, (1, 1, min(offset, 127) + length, 512)).astype(
        np.float32
    )
    sinks = rng.normal(0, 0.3, (heads,)).astype(np.float32)
    if "sink" in plan:
        sinks[:] = plan["sink"]
    packed = unpacked = None
    if count:
        # Chunked CPU packing/decoding bounds oracle preparation: never create
        # all-history exponent/mantissa temporary arrays just to build fixture.
        packed_parts = []
        unpacked = np.empty((1, count, 512), np.float32)
        for start in range(0, count, 512):
            n = min(512, count - start)
            part = pack_fp8_kv(
                rng.normal(0, 0.3, (1, n, 512)).astype(np.float32), xp=np
            )
            unpacked[:, start : start + n] = unpack_fp8_kv(part, xp=np)
            packed_parts.append(part)
        packed = PackedKV(
            *(
                np.concatenate([getattr(part, name) for part in packed_parts], axis=1)
                for name in ("codes", "scales", "rope")
            )
        )
    indices = None
    if plan.get("topk"):
        indices = np.stack(
            [
                rng.choice((offset + row + 1) // ratio, plan["topk"], replace=False)
                for row in range(length)
            ]
        )[None].astype(np.uint32)
    return q, local, packed, unpacked, indices, sinks


def working_bound(plan):
    count = (
        0 if not plan["ratio"] else (plan["offset"] + plan["length"]) // plan["ratio"]
    )
    # Includes host oracle pool + complete GPU unpack graph for comparison.
    # Candidate itself requires only output/sortedIDs/params workspace.
    return (
        256 * (1 << 20)
        + count * (4 * 732 + 10 * 512 * 4)
        + 64 * plan["length"] * 512 * 16
    )


def run_case(plan, *, gpu, repetitions, guard):
    q, local, packed, unpacked, indices, sinks = fixture(plan)
    from mlx2.runtime.models.deepseek_v4_packed_visible_attention import workspace_bytes

    selected = [0, 17, 63]
    expected = reference_visible_attention(
        q[:, selected],
        local,
        unpacked,
        indices,
        sinks[selected],
        offset=plan["offset"],
        window=128,
        ratio=plan["ratio"],
    )
    result = {
        **plan,
        "physical_batch_width": 1,
        "head_dim": 512,
        "heads": 64,
        "pool_rows": 0 if packed is None else packed.codes.shape[1],
        "candidate_pool_fp32_materialization_bytes": 0,
        "candidate_workspace_bytes": workspace_bytes(
            heads=64, length=plan["length"], topk=plan.get("topk", 0)
        ),
        "baseline_full_pool_fp32_bytes": 0 if unpacked is None else unpacked.nbytes,
        "working_bound_bytes": working_bound(plan),
        "cpu_oracle_finite": bool(np.isfinite(expected).all()),
        "qualified": False,
        "artifact_loaded": False,
    }
    if not gpu:
        return result
    import mlx.core as mx

    from mlx2.runtime.models.deepseek_v4_packed_visible_attention import (
        packed_visible_attention,
    )
    from mlx2.runtime.models.deepseek_v4_visible_attention import visible_attention

    qm, lm, sm = mx.array(q), mx.array(local), mx.array(sinks)
    pm = (
        None
        if packed is None
        else PackedKV(
            *(mx.array(x) for x in (packed.codes, packed.scales, packed.rope))
        )
    )
    im = None if indices is None else mx.array(indices)
    kwargs = {"offset": plan["offset"], "window": 128, "ratio": plan["ratio"]}

    def direct():
        return packed_visible_attention(qm, lm, pm, im, sm, **kwargs)

    def baseline():
        full = None if pm is None else unpack_fp8_kv(pm, xp=mx)
        return visible_attention(qm, lm, full, im, sm, **kwargs)

    guard()
    start = time.perf_counter()
    actual = direct()
    mx.eval(actual)
    result["first_candidate_evaluation_seconds"] = time.perf_counter() - start
    # This graph deliberately includes full-pool unpack, matching old caller.
    start = time.perf_counter()
    ordinary = baseline()
    mx.eval(ordinary)
    result["first_baseline_evaluation_seconds"] = time.perf_counter() - start
    a, b = np.asarray(actual), np.asarray(ordinary)
    result["numerical_vs_unpackall_visible_fp32"] = compare(a, b, atol=0, rtol=0)
    result["bitwise_equal_unpackall_visible_fp32"] = bool(
        np.array_equal(a.view(np.uint32), b.view(np.uint32))
    )
    result["independent_float64_oracle"] = compare(
        a[:, selected], expected, atol=3e-6, rtol=3e-5
    )
    result["compiler_cache_state"] = (
        "unproven persistent Metal cache; first evaluation and warmed dispatch separate"
    )
    if (
        result["bitwise_equal_unpackall_visible_fp32"]
        and result["independent_float64_oracle"]["passed"]
    ):
        _, result["candidate_timing"] = measure(direct, repetitions)
        _, result["baseline_unpackall_timing"] = measure(baseline, repetitions)
        result["observational_speed_ratio"] = (
            result["baseline_unpackall_timing"]["warm_median_seconds"]
            / result["candidate_timing"]["warm_median_seconds"]
        )
    else:
        result["timing_skipped"] = "exact-value/order or independent oracle gate failed"
    result["allocator_peak_bytes"] = mx.get_peak_memory()
    guard()
    return result


def invalid_indices_probe(guard):
    import mlx.core as mx

    from mlx2.runtime.models.deepseek_v4_packed_visible_attention import (
        packed_visible_attention,
    )

    plan = {"ratio": 4, "offset": 13, "length": 3, "topk": 3}
    q, local, packed, _, indices, sinks = fixture(plan, heads=2)
    pm = PackedKV(*(mx.array(x) for x in (packed.codes, packed.scales, packed.rope)))
    result = {}
    for name in ("out_of_range", "negative", "duplicate"):
        bad = indices.astype(np.int32)
        bad[0, 0, 0] = (
            4
            if name == "out_of_range"
            else (-1 if name == "negative" else bad[0, 0, 1])
        )
        guard()
        out = packed_visible_attention(
            mx.array(q),
            mx.array(local),
            pm,
            mx.array(bad),
            mx.array(sinks),
            offset=13,
            window=128,
            ratio=4,
        )
        mx.eval(out)
        result[name] = {
            "first_query_all_nonfinite": bool(np.isnan(np.asarray(out)[:, :, 0]).all()),
            "other_queries_finite": bool(np.isfinite(np.asarray(out)[:, :, 1:]).all()),
            "packed_inputs_unchanged": all(
                np.array_equal(np.asarray(a).view(np.uint8), b.view(np.uint8))
                for a, b in zip(
                    (pm.codes, pm.scales, pm.rope),
                    (packed.codes, packed.scales, packed.rope),
                )
            ),
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-gpu", action="store_true")
    parser.add_argument("--ownership-window", type=Path)
    parser.add_argument("--owner-path", type=Path, action="append")
    parser.add_argument("--plans", nargs="+", default=[p["name"] for p in PLANS])
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--max-working-bytes", type=int, default=2 << 30)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("refusing existing evidence directory")
    plans = [p for p in PLANS if p["name"] in args.plans]
    if (
        len(plans) != len(set(args.plans))
        or not 1 <= args.repetitions <= 20
        or not 0 < args.max_working_bytes <= 2 << 30
        or any(working_bound(p) > args.max_working_bytes for p in plans)
    ):
        parser.error(
            "known unique plans,repetitions1..20,bounded workingcap<=2GiB required"
        )
    owners = args.owner_path or [
        Path("/Users/Shared/mlxuag/gpu.lock/owner.json"),
        Path("/tmp/gpu.lock/owner.json"),
    ]
    if args.run_gpu and (args.ownership_window is None or len(owners) != 2):
        parser.error("GPU requires independently active CPG/paired ownership")

    def guard():
        return verify_ownership(args.ownership_window, owners) if args.run_gpu else None

    source = source_receipt()
    ownership = guard()
    from mlx2.runtime.models import deepseek_v4_packed_visible_attention as candidate

    source["candidate_module_sha256"] = hashlib.sha256(
        Path(candidate.__file__).read_bytes()
    ).hexdigest()
    source["probe_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    fixture_hash = hashlib.sha256(
        json.dumps(plans, sort_keys=True).encode()
    ).hexdigest()
    args.output.mkdir(parents=True)
    receipt = {
        "schema": "mlx2.deepseek-v4-direct-packed-visible-fixtures.v1",
        "status": "running",
        "checkpoint": "synthetic-packed-visible@sha256:" + fixture_hash,
        "source": source,
        "ownership": ownership,
        "run_gpu": args.run_gpu,
        "qualification": False,
        "performance_claim": False,
        "artifact_loaded": False,
        "timing_scope": "observational component comparison; no model or thermally controlled speed claim",
        "cases": [],
    }
    atomic(args.output / "results.json", receipt)
    try:
        if args.run_gpu:
            import mlx.core as mx

            mx.set_default_device(mx.gpu)
            mx.set_memory_limit(args.max_working_bytes)
            mx.set_cache_limit(64 << 20)
        for plan in plans:
            guard()
            case = run_case(
                plan, gpu=args.run_gpu, repetitions=args.repetitions, guard=guard
            )
            receipt["cases"].append(case)
            atomic(args.output / "results.json", receipt)
            print(
                json.dumps(
                    {
                        "plan": plan["name"],
                        "bitwise": case.get("bitwise_equal_unpackall_visible_fp32"),
                        "ratio": case.get("observational_speed_ratio"),
                    }
                ),
                flush=True,
            )
            gc.collect()
            if args.run_gpu:
                mx.clear_cache()
        if args.run_gpu:
            receipt["invalid_indices"] = invalid_indices_probe(guard)
        receipt["passed"] = all(
            c["cpu_oracle_finite"]
            and c.get("bitwise_equal_unpackall_visible_fp32", True)
            and c.get("independent_float64_oracle", {}).get("passed", True)
            for c in receipt["cases"]
        ) and all(all(v.values()) for v in receipt.get("invalid_indices", {}).values())
        receipt["status"] = "completed"
    except BaseException as error:
        receipt.update(
            status="failed",
            passed=False,
            error={"type": type(error).__name__, "message": str(error)},
        )
        raise
    finally:
        atomic(args.output / "results.json", receipt)
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

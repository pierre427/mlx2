#!/usr/bin/env python3
"""GPU A/B for exact-shape segmented plain-KV attention coalescing."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlx2.runtime.models.cache import KVCache
from mlx2.runtime.segmented_plain_kv import SegmentedBatchKVCache


def require_locks() -> dict:
    owners = {}
    for path in (
        Path("/Users/Shared/mlxuag/gpu.lock/owner.json"),
        Path("/tmp/gpu.lock/owner.json"),
    ):
        if not path.is_file():
            raise SystemExit(f"missing owned GPU lock: {path}")
        owners[str(path)] = json.loads(path.read_text())
    lease_ids = {owner.get("lease_id") for owner in owners.values()}
    if len(lease_ids) != 1 or None in lease_ids:
        raise SystemExit(f"GPU lock identities disagree: {owners}")
    return owners


def measure(fn, *, warmups: int, repeats: int):
    for _ in range(warmups):
        mx.eval(fn())
    samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = fn()
        mx.eval(result)
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "median_ms": statistics.median(samples),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
        "samples_ms": samples,
    }, result


def make_case(*, batch: int, key_length: int, q_heads: int, kv_heads: int, head_dim: int):
    rows = []
    for _ in range(batch):
        row = KVCache()
        row.update_and_fetch(
            mx.random.normal((1, kv_heads, key_length, head_dim)).astype(mx.bfloat16),
            mx.random.normal((1, kv_heads, key_length, head_dim)).astype(mx.bfloat16),
        )
        rows.append(row)
    counters = {}

    def note(key, amount=1):
        counters[key] = counters.get(key, 0) + amount

    view = SegmentedBatchKVCache(rows, note=note)
    view.prepare(lengths=[1] * batch, right_padding=[0] * batch)
    mask = view.make_mask(1)
    view.update_and_fetch(
        mx.random.normal((batch, kv_heads, 1, head_dim)).astype(mx.bfloat16),
        mx.random.normal((batch, kv_heads, 1, head_dim)).astype(mx.bfloat16),
    )
    queries = mx.random.normal((batch, q_heads, 1, head_dim)).astype(mx.bfloat16)
    mx.eval(queries, mask, [array for row in rows for array in row.keys_and_values()])
    row_views = tuple(view.row_views(mask))
    scale = head_dim**-0.5

    def reference():
        return mx.concatenate(
            [view._reference_attention(queries, scale, None, row) for row in row_views],
            axis=0,
        )

    def coalesced():
        return view.bucketed_attention(queries, scale, mask)

    return view, counters, reference, coalesced


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch", default="2,4,8")
    parser.add_argument("--key-lengths", default="128,512,2048,8192")
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()

    locks = require_locks()
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise SystemExit("Metal GPU is unavailable")
    mx.random.seed(20260930)
    report = {
        "schema": "mlx2.segmented-plain-coalescing-ab.v1",
        "source_head": None,
        "source_sha256": {},
        "mlx_version": getattr(mx, "__version__", "unknown"),
        "platform": platform.platform(),
        "locks": locks,
        "qualified": False,
        "selected": False,
        "cases": [],
    }
    for path in (
        ROOT / "src/mlx2/runtime/segmented_plain_kv.py",
        Path(__file__).resolve(),
    ):
        report["source_sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if revision.returncode == 0:
        report["source_head"] = revision.stdout.strip()

    for batch in [int(value) for value in args.batch.split(",")]:
        for key_length in [int(value) for value in args.key_lengths.split(",")]:
            mx.clear_cache()
            view, counters, reference, coalesced = make_case(
                batch=batch,
                key_length=key_length,
                q_heads=args.q_heads,
                kv_heads=args.kv_heads,
                head_dim=args.head_dim,
            )
            reference_timing, expected = measure(
                reference, warmups=args.warmups, repeats=args.repeats
            )
            counters.clear()
            coalesced_timing, actual = measure(
                coalesced, warmups=args.warmups, repeats=args.repeats
            )
            mx.eval(expected, actual)
            delta = mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))
            mx.eval(delta)
            exact = bool(mx.array_equal(actual, expected).item())
            max_abs = float(mx.max(delta).item())
            report["cases"].append(
                {
                    "batch": batch,
                    "key_length": key_length,
                    "q_heads": args.q_heads,
                    "kv_heads": args.kv_heads,
                    "head_dim": args.head_dim,
                    "reference": reference_timing,
                    "coalesced": coalesced_timing,
                    "speedup": reference_timing["median_ms"]
                    / coalesced_timing["median_ms"],
                    "bit_exact": exact,
                    "max_abs": max_abs,
                    "engagement": dict(counters),
                }
            )
            view.finalize()
    report["passed"] = all(
        case["bit_exact"]
        and case["engagement"].get("coalesced_attention_dispatches", 0) > 0
        for case in report["cases"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True)
    args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()

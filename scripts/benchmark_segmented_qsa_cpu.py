#!/usr/bin/env python3
"""CPU microbenchmark for experimental shared-prefix segmented QSA math.

This is a workload-shape probe, not qualification.  CPU timings do not predict
Metal speed.  The prototype uses eager MLX operations rather than a fused
kernel; its main evidence is numerical agreement and the base-read/call proxy.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx

# This must happen before benchmark tensor creation.
mx.set_default_device(mx.cpu)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlx2.runtime.segmented_qsa_reference import (  # noqa: E402
    materialized_row_attention_reference,
    shared_prefix_segmented_attention,
)


def _case(*, batch, query, prefix, suffix_cap, seed):
    mx.random.seed(seed)
    query_heads, kv_heads, head_dim = 8, 2, 32
    suffix_lengths = tuple(
        0 if row == 0 else 1 + ((row * 13 + query) % suffix_cap)
        for row in range(batch)
    )
    queries = mx.random.normal(
        (batch, query_heads, query, head_dim), dtype=mx.float32
    )
    base_keys = mx.random.normal((1, kv_heads, prefix, head_dim), dtype=mx.float32)
    base_values = mx.random.normal(
        (1, kv_heads, prefix, head_dim), dtype=mx.float32
    )
    suffix_keys = [
        mx.random.normal((1, kv_heads, length, head_dim), dtype=mx.float32)
        for length in suffix_lengths
    ]
    suffix_values = [
        mx.random.normal((1, kv_heads, length, head_dim), dtype=mx.float32)
        for length in suffix_lengths
    ]
    return (
        queries,
        base_keys,
        base_values,
        suffix_keys,
        suffix_values,
        1.0 / math.sqrt(head_dim),
    )


def _invoke(function, values):
    queries, base_keys, base_values, suffix_keys, suffix_values, scale = values
    output, receipt = function(
        queries,
        base_keys,
        base_values,
        suffix_keys,
        suffix_values,
        scale=scale,
    )
    mx.eval(output)
    return output, receipt


def _measure(function, values, *, warmups, repeats):
    for _ in range(warmups):
        _invoke(function, values)
    samples = []
    receipt = None
    for _ in range(repeats):
        started = time.perf_counter_ns()
        _, receipt = _invoke(function, values)
        samples.append((time.perf_counter_ns() - started) / 1.0e6)
    samples.sort()
    return {
        "median_ms": statistics.median(samples),
        "minimum_ms": samples[0],
        "maximum_ms": samples[-1],
        "samples_ms": samples,
        "receipt": receipt.to_dict(),
    }


def _errors(actual, expected):
    delta = mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))
    relative = delta / mx.maximum(mx.abs(expected.astype(mx.float32)), 1.0e-6)
    maximum_absolute = mx.max(delta)
    maximum_relative = mx.max(relative)
    mx.eval(maximum_absolute, maximum_relative)
    return float(maximum_absolute.item()), float(maximum_relative.item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-b8", action="store_true")
    args = parser.parse_args()
    batches = [1, 2, 4] + ([8] if args.include_b8 else [])
    geometries = [
        {"name": "decode-short", "query": 1, "prefix": 256, "suffix_cap": 16},
        {"name": "decode-long", "query": 1, "prefix": 1024, "suffix_cap": 48},
        {"name": "verify", "query": 4, "prefix": 256, "suffix_cap": 32},
    ]
    results = []
    for geometry_index, geometry in enumerate(geometries):
        for batch in batches:
            values = _case(batch=batch, seed=911 + geometry_index * 17 + batch, **{
                key: value for key, value in geometry.items() if key != "name"
            })
            reference = _measure(
                materialized_row_attention_reference,
                values,
                warmups=args.warmups,
                repeats=args.repeats,
            )
            prototype = _measure(
                shared_prefix_segmented_attention,
                values,
                warmups=args.warmups,
                repeats=args.repeats,
            )
            expected, _ = _invoke(materialized_row_attention_reference, values)
            actual, _ = _invoke(shared_prefix_segmented_attention, values)
            max_abs, max_rel = _errors(actual, expected)
            results.append(
                {
                    "case": geometry["name"],
                    "batch": batch,
                    "query_length": geometry["query"],
                    "prefix_length": geometry["prefix"],
                    "suffix_lengths": prototype["receipt"]["suffix_lengths"],
                    "reference": reference,
                    "prototype": prototype,
                    "prototype_over_reference_median": (
                        prototype["median_ms"] / reference["median_ms"]
                    ),
                    "max_abs_error": max_abs,
                    "max_rel_error": max_rel,
                }
            )
    report = {
        "schema": "mlx2.segmented-qsa-cpu-microbench.v1",
        "device": "cpu",
        "prototype": True,
        "qualification": False,
        "warning": "CPU timing does not predict Metal speed; eager prototype is not a fused kernel.",
        "warmups": args.warmups,
        "repeats": args.repeats,
        "results": results,
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.write_text(payload + "\n")


if __name__ == "__main__":
    main()

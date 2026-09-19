#!/usr/bin/env python3
"""GPU microbenchmark for the isolated segmented shared-prefix QSA probe."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlx2.runtime.segmented_qsa_metal import segmented_shared_prefix_attention_metal


def _require_lock():
    locks = (Path("/tmp/gpu.lock/owner.json"), Path("/Users/Shared/mlxuag/gpu.lock/owner.json"))
    missing = [str(path) for path in locks if not path.exists()]
    if missing:
        raise SystemExit(f"GPU benchmark requires both lock receipts; missing {missing}")


def _case(batch, query, prefix, suffix, dim, query_heads, kv_heads, dtype, selected_tokens, seed):
    mx.random.seed(seed)
    hq, hkv = query_heads, kv_heads
    lengths = tuple(0 if row == 0 else 1 + ((row * 13 + query) % suffix) for row in range(batch))
    values = {
        "q": mx.random.normal((batch, hq, query, dim), dtype=dtype),
        "bk": mx.random.normal((1, hkv, prefix, dim), dtype=dtype),
        "bv": mx.random.normal((1, hkv, prefix, dim), dtype=dtype),
        "sk": mx.random.normal((batch, hkv, suffix, dim), dtype=dtype),
        "sv": mx.random.normal((batch, hkv, suffix, dim), dtype=dtype),
        "lengths": lengths,
        "length_array": mx.array(lengths, dtype=mx.uint32),
        "scale": 1.0 / math.sqrt(dim),
    }
    if selected_tokens:
        stride = max(1, prefix // selected_tokens)
        values["base_indices"] = mx.arange(0, stride * selected_tokens, stride, dtype=mx.uint32)
        values["selected_bk"] = mx.take(values["bk"], values["base_indices"], axis=2)
        values["selected_bv"] = mx.take(values["bv"], values["base_indices"], axis=2)
    else:
        values["base_indices"] = None
        values["selected_bk"] = values["bk"]
        values["selected_bv"] = values["bv"]
    values["full_k"] = [
        mx.concatenate((values["selected_bk"], values["sk"][row : row + 1, :, :length]), axis=2)
        for row, length in enumerate(lengths)
    ]
    values["full_v"] = [
        mx.concatenate((values["selected_bv"], values["sv"][row : row + 1, :, :length]), axis=2)
        for row, length in enumerate(lengths)
    ]
    mx.eval(*values["full_k"], *values["full_v"])
    return values


def _reference(values):
    rows = []
    for row, length in enumerate(values["lengths"]):
        if values["base_indices"] is None:
            base_k, base_v = values["bk"], values["bv"]
        else:
            base_k = mx.take(values["bk"], values["base_indices"], axis=2)
            base_v = mx.take(values["bv"], values["base_indices"], axis=2)
        keys = mx.concatenate((base_k, values["sk"][row : row + 1, :, :length]), axis=2)
        vals = mx.concatenate((base_v, values["sv"][row : row + 1, :, :length]), axis=2)
        rows.append(
            mx.fast.scaled_dot_product_attention(
                values["q"][row : row + 1], keys, vals, scale=values["scale"]
            )
        )
    return mx.concatenate(rows, axis=0)


def _reference_prepared(values):
    rows = [
        mx.fast.scaled_dot_product_attention(
            values["q"][row : row + 1],
            values["full_k"][row],
            values["full_v"][row],
            scale=values["scale"],
        )
        for row in range(len(values["lengths"]))
    ]
    return mx.concatenate(rows, axis=0)


def _candidate(values, splits):
    output, engaged, receipt = segmented_shared_prefix_attention_metal(
        values["q"], values["bk"], values["bv"], values["sk"], values["sv"],
        values["length_array"], scale=values["scale"], splits=splits,
        base_indices=values["base_indices"],
    )
    return output, engaged, receipt


def _measure(function, *, warmups, repeats):
    for _ in range(warmups):
        arrays = function()
        mx.eval(*arrays if isinstance(arrays, tuple) else (arrays,))
    samples = []
    mx.clear_cache()
    mx.reset_peak_memory()
    active_before = mx.get_active_memory()
    last = None
    for _ in range(repeats):
        started = time.perf_counter_ns()
        last = function()
        mx.eval(*last if isinstance(last, tuple) else (last,))
        samples.append((time.perf_counter_ns() - started) / 1.0e6)
    samples.sort()
    return {
        "median_ms": statistics.median(samples),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
        "samples_ms": samples,
        "peak_delta_bytes": max(0, mx.get_peak_memory() - active_before),
    }, last


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefixes", default="4096,16384,65536")
    parser.add_argument("--batches", default="1,2,4,8")
    parser.add_argument("--queries", default="1,4")
    parser.add_argument("--splits", default="8,16,32,64")
    parser.add_argument("--suffix", type=int, default=64)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--query-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--selected-tokens", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    _require_lock()
    mx.set_default_device(mx.gpu)
    dtype = mx.float16 if args.dtype == "float16" else mx.bfloat16
    results = []
    for prefix in map(int, args.prefixes.split(",")):
        for batch in map(int, args.batches.split(",")):
            for query in map(int, args.queries.split(",")):
                values = _case(
                    batch, query, prefix, args.suffix, args.dim,
                    args.query_heads, args.kv_heads, dtype, args.selected_tokens,
                    1009 + prefix + batch * 17 + query,
                )
                reference, reference_output = _measure(
                    lambda values=values: _reference(values),
                    warmups=args.warmups,
                    repeats=args.repeats,
                )
                prepared_reference, prepared_output = _measure(
                    lambda values=values: _reference_prepared(values),
                    warmups=args.warmups,
                    repeats=args.repeats,
                )
                split_results = []
                best = None
                for splits in map(int, args.splits.split(",")):
                    timing, candidate_result = _measure(
                        lambda splits=splits, values=values: _candidate(values, splits)[:2],
                        warmups=args.warmups,
                        repeats=args.repeats,
                    )
                    output, engaged = candidate_result
                    delta = mx.abs(output.astype(mx.float32) - reference_output.astype(mx.float32))
                    relative = delta / mx.maximum(mx.abs(reference_output.astype(mx.float32)), 1.0e-6)
                    maximum_absolute = mx.max(delta)
                    maximum_relative = mx.max(relative)
                    mx.eval(maximum_absolute, maximum_relative, engaged)
                    row = {
                        "splits": splits,
                        **timing,
                        "engaged": int(engaged.item()),
                        "max_abs_error": float(maximum_absolute.item()),
                        "max_rel_error": float(maximum_relative.item()),
                        "candidate_over_reference": timing["median_ms"] / reference["median_ms"],
                        "candidate_over_prepared_reference": (
                            timing["median_ms"] / prepared_reference["median_ms"]
                        ),
                    }
                    split_results.append(row)
                    if best is None or row["median_ms"] < best["median_ms"]:
                        best = row
                results.append({
                    "prefix": prefix,
                    "selected_tokens": args.selected_tokens or prefix,
                    "batch": batch,
                    "query": query,
                    "suffix_lengths": values["lengths"],
                    "reference": reference,
                    "prepared_reference": prepared_reference,
                    "prepared_reference_agrees": bool(
                        mx.allclose(reference_output, prepared_output, atol=0.0, rtol=0.0).item()
                    ),
                    "candidates": split_results,
                    "best": best,
                })
                del values
                mx.clear_cache()
    report = {
        "schema": "mlx2.segmented-qsa-metal-microbench.v1",
        "device": mx.device_info(),
        "dtype": args.dtype,
        "head_geometry": {
            "query_heads": args.query_heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.dim,
        },
        "qualification": False,
        "mechanism": "metal_shared_prefix_partition_v1",
        "results": results,
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")


if __name__ == "__main__":
    main()

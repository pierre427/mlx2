#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""CPU-only benchmark for the prompt-lookup host index.

This benchmark never loads MLX, a model, or a serving process. It compares the
optimized early-exit proposal path with the bounded newest-first scan present
on the integration base. The result is a host microbenchmark, not a serving-
throughput or qualification claim.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import time

from mlx2.runtime.prompt_lookup import IndexedPromptLookup


def _baseline_target_propose(oracle, max_span, lookback):
    """Integration-base newest-first scan for a matched timing control."""
    size_limit = min(oracle.ngram_max, len(oracle.tokens))
    for size in range(size_limit, oracle.ngram_min - 1, -1):
        key = tuple(oracle.tokens[-size:])
        chosen = None
        chosen_score = None
        for start in reversed(oracle.index[size].get(key, ())):
            if start >= len(oracle.tokens) - size:
                continue
            if len(oracle.tokens) - start > lookback:
                break
            continuation = tuple(oracle.tokens[start + size : start + size + max_span])
            if continuation:
                score = (len(continuation), start)
                if chosen_score is None or score > chosen_score:
                    chosen, chosen_score = continuation, score
        if chosen is not None:
            return list(chosen)
    return []


def _percentile(values, fraction):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _time_ns(call, *, warmup, rounds):
    for _ in range(warmup):
        call()
    values = []
    for _ in range(rounds):
        start = time.perf_counter_ns()
        call()
        values.append(time.perf_counter_ns() - start)
    return {
        "median_us": statistics.median(values) / 1_000,
        "p95_us": _percentile(values, 0.95) / 1_000,
        "min_us": min(values) / 1_000,
        "max_us": max(values) / 1_000,
    }


def _time_build(tokens, *, ngram_min, ngram_max, index_window, rounds):
    values = []
    receipt = None
    for _ in range(rounds):
        gc.collect()
        start = time.perf_counter_ns()
        oracle = IndexedPromptLookup(
            tokens,
            ngram_min=ngram_min,
            ngram_max=ngram_max,
            index_window=index_window,
        )
        values.append(time.perf_counter_ns() - start)
        receipt = oracle.index_receipt()
    return {
        "median_ms": statistics.median(values) / 1_000_000,
        "min_ms": min(values) / 1_000_000,
        "max_ms": max(values) / 1_000_000,
        "indexed_occurrences": receipt["target_occurrences_indexed"],
    }


def _case(label, tokens, args):
    complete = IndexedPromptLookup(
        tokens, ngram_min=args.ngram_min, ngram_max=args.ngram_max
    )
    bounded = IndexedPromptLookup(
        tokens,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        index_window=args.index_window,
    )
    expected = _baseline_target_propose(complete, args.num_draft, args.lookback)
    actual = bounded.propose(args.num_draft, lookback=args.lookback)
    if actual != expected:
        raise RuntimeError(f"proposal mismatch for {label}: {actual} != {expected}")
    baseline = _time_ns(
        lambda: _baseline_target_propose(complete, args.num_draft, args.lookback),
        warmup=args.warmup,
        rounds=args.rounds,
    )
    indexed = _time_ns(
        lambda: bounded.propose(args.num_draft, lookback=args.lookback),
        warmup=args.warmup,
        rounds=args.rounds,
    )
    return {
        "case": label,
        "tokens": len(tokens),
        "proposal_tokens": len(actual),
        "integration_base_scan": baseline,
        "indexed_window": indexed,
        "median_speedup": baseline["median_us"] / max(indexed["median_us"], 1e-9),
        "index_receipt": bounded.index_receipt(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--lookback", type=int, default=256)
    parser.add_argument("--index-window", type=int, default=16384)
    parser.add_argument("--num-draft", type=int, default=8)
    parser.add_argument("--ngram-min", type=int, default=3)
    parser.add_argument("--ngram-max", type=int, default=6)
    parser.add_argument("--periods", default="1,4,16,128")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=101)
    parser.add_argument("--build-rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if (
        min(
            args.tokens,
            args.lookback,
            args.index_window,
            args.num_draft,
            args.ngram_min,
            args.ngram_max,
            args.rounds,
            args.build_rounds,
        )
        < 1
    ):
        parser.error("token, lookup, n-gram, and round counts must be positive")
    if args.ngram_max < args.ngram_min:
        parser.error("--ngram-max must be at least --ngram-min")
    if args.index_window < args.lookback:
        parser.error("--index-window must be at least --lookback")

    periods = [int(value) for value in args.periods.split(",")]
    cases = []
    for period in periods:
        if period < 1:
            parser.error("--periods must contain positive integers")
        base = list(range(period))
        tokens = (base * (args.tokens // period + 1))[: args.tokens]
        cases.append(_case(f"period-{period}", tokens, args))
    rng = random.Random(args.seed)
    random_tokens = [rng.randrange(4096) for _ in range(args.tokens)]
    cases.append(_case("random-v4096", random_tokens, args))

    output = {
        "schema": "mlx2.prompt-lookup-host-benchmark.v1",
        "scope": "CPU-only synthetic host index; no model, MLX, Metal, or serving",
        "settings": vars(args),
        "build": {
            "complete": _time_build(
                random_tokens,
                ngram_min=args.ngram_min,
                ngram_max=args.ngram_max,
                index_window=None,
                rounds=args.build_rounds,
            ),
            "bounded": _time_build(
                random_tokens,
                ngram_min=args.ngram_min,
                ngram_max=args.ngram_max,
                index_window=args.index_window,
                rounds=args.build_rounds,
            ),
        },
        "cases": cases,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Measure host-only callable-contract overhead, never model throughput."""

import argparse
import inspect
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mlx2.runtime.call_contracts import KeywordSupportCache


class LegacyProvider:
    def draft_distributions(
        self,
        anchors,
        features,
        cache,
        count,
        rngs,
        temperatures,
        *,
        logits_processors=None,
        processor_histories=None,
    ):
        pass


def run(iterations, repetitions):
    method = LegacyProvider().draft_distributions
    # Every measured arm uses the same callable. Cache construction and the
    # first compatibility inspection are included in every cached repetition.
    times = {"inspect_each_group": [], "cached_legacy": [], "declared": []}
    counts = {}
    for repetition in range(repetitions):
        order = list(times)[:: 1 if repetition % 2 == 0 else -1]
        for arm in order:
            started = time.perf_counter_ns()
            cache = KeywordSupportCache("logits_processors")
            for _ in range(iterations):
                if arm == "inspect_each_group":
                    assert "logits_processors" in inspect.signature(method).parameters
                else:
                    assert cache.accepts(method, declared=arm == "declared")
            times[arm].append((time.perf_counter_ns() - started) / iterations)
            counts[arm] = (
                iterations if arm == "inspect_each_group" else cache.inspections
            )
    return {
        "schema": "mlx2.host-contract-microbenchmark.v1",
        "scope": "synthetic Python provider signature only; not serving/model performance",
        "iterations": iterations,
        "repetitions": repetitions,
        "ns_per_call": {
            key: {"median": statistics.median(values), "runs": values}
            for key, values in times.items()
        },
        "signature_inspections_per_repetition": counts,
        "framework_imported": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--repetitions", type=int, default=7)
    args = parser.parse_args()
    if args.iterations < 1 or args.repetitions < 1 or args.output.exists():
        parser.error("positive run sizes and a fresh output required")
    report = run(args.iterations, args.repetitions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()

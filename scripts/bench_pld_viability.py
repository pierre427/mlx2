#!/usr/bin/env python3
"""CPU-only benchmark for mlx2's current indexed PLD proposer.

Splash #300 reports a flat chained C++ index with no decode-time heap
allocation.  mlx2 already has an incremental, bounded, newest-first Python
index.  This harness therefore does not copy the C++ implementation: it
measures the current ``IndexedPromptLookup`` against the exhaustive indexed
oracle retained from before mlx2's lookup optimization, verifies identical
proposals, and emits timer-free lookup counters alongside host timings.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlx2.runtime.prompt_lookup import IndexedPromptLookup


class ExhaustiveIndexedPromptLookup:
    """Pre-optimization behavior: materialize and rank every eligible source."""

    def __init__(self, tokens=(), *, ngram_min=3, ngram_max=6):
        self.ngram_min, self.ngram_max = ngram_min, ngram_max
        self.tokens = []
        self.index = {size: defaultdict(list) for size in range(ngram_min, ngram_max + 1)}
        for token in tokens:
            self.observe(token)

    def observe(self, token):
        self.tokens.append(int(token))
        end = len(self.tokens)
        for size, buckets in self.index.items():
            start = end - size
            if start >= 0:
                buckets[tuple(self.tokens[start:end])].append(start)

    def propose(self, max_span, *, lookback=4096):
        if max_span <= 0 or lookback <= 0:
            return []
        for size in range(min(self.ngram_max, len(self.tokens)), self.ngram_min - 1, -1):
            key = tuple(self.tokens[-size:])
            candidates = []
            for start in self.index[size].get(key, ()):
                if start >= len(self.tokens) - size or start < len(self.tokens) - lookback:
                    continue
                continuation = tuple(self.tokens[start + size : start + size + max_span])
                if continuation:
                    candidates.append((len(continuation), start, continuation))
            if candidates:
                return list(max(candidates)[2])
        return []


class FastMissIndexedPromptLookup(IndexedPromptLookup):
    """Research arm: prove absence through the shortest indexed suffix first.

    Every larger n-gram match implies a match of its shortest suffix.  When no
    eligible target or hot-segment occurrence of that suffix exists, the stock
    search can only miss.  The early return advances the same public and
    timer-free counters the stock implementation would have advanced.  Hits,
    rejected sources, context matching and bounded-source scans delegate to the
    production implementation unchanged.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._nearest_shortest_start = None
        self._refresh_shortest_tail()

    def _refresh_shortest_tail(self):
        size = self.ngram_min
        self._nearest_shortest_start = None
        if len(self.tokens) < size:
            return
        key = tuple(self.tokens[-size:])
        final_start = len(self.tokens) - size
        for start in reversed(self.index[size].get(key, ())):
            if start < final_start:
                self._nearest_shortest_start = start
                return

    def observe(self, token):
        super().observe(token)
        self._refresh_shortest_tail()

    @staticmethod
    def _eligible(positions, *, first_start: int, final_start: int) -> bool:
        for start in reversed(positions):
            if start >= final_start:
                continue
            return start >= first_start
        return False

    def _shortest_suffix_absent(self, lookback: int) -> bool:
        size = self.ngram_min
        if len(self.tokens) < size:
            return True
        first_start = len(self.tokens) - lookback
        if (
            self._nearest_shortest_start is not None
            and self._nearest_shortest_start >= first_start
        ):
            return False
        if not self.hot:
            return True
        key = tuple(self.tokens[-size:])
        for segment in self.hot:
            hot_first_start = max(0, len(segment.tokens) - lookback - size)
            if self._eligible(
                segment.index[size].get(key, ()),
                first_start=hot_first_start,
                final_start=len(segment.tokens) - size,
            ):
                return False
        return True

    def _record_proved_miss(self, lookback: int) -> list[int]:
        self.clock += 1
        self.lookup_calls += 1
        self.index_stats.calls += 1
        self.last_source = None
        size_limit = min(self.ngram_max, len(self.tokens))
        for size in range(size_limit, self.ngram_min - 1, -1):
            key = tuple(self.tokens[-size:])
            positions = self.index[size].get(key, ())
            self.index_stats.target_key_lookups += 1
            self.index_stats.target_occurrences_skipped += len(positions)
            for segment in self.hot:
                self.index_stats.hot_key_lookups += 1
                first_start = max(0, len(segment.tokens) - lookback - size)
                final_start = len(segment.tokens) - size
                self.index_stats.hot_starts_avoided += max(0, final_start - first_start)
        self.index_stats.misses += 1
        return []

    def propose(self, max_span, *, lookback=4096, min_context_match=0, max_sources=0):
        if max_span > 0 and lookback > 0:
            if not self.hot:
                first_start = len(self.tokens) - lookback
                proved_absent = (
                    len(self.tokens) < self.ngram_min
                    or self._nearest_shortest_start is None
                    or self._nearest_shortest_start < first_start
                )
            else:
                proved_absent = self._shortest_suffix_absent(lookback)
            if proved_absent:
                return self._record_proved_miss(lookback)
        return super().propose(
            max_span,
            lookback=lookback,
            min_context_match=min_context_match,
            max_sources=max_sources,
        )


class AdaptiveFastMissPromptLookup(FastMissIndexedPromptLookup):
    """Keep the ordinary hit path until repeated misses justify the guard."""

    def __init__(self, *args, activation_misses: int = 4, **kwargs):
        if activation_misses < 1:
            raise ValueError("activation_misses must be positive")
        super().__init__(*args, **kwargs)
        self.activation_misses = int(activation_misses)
        self.consecutive_misses = 0
        self.fast_miss_active = False

    def propose(self, max_span, *, lookback=4096, min_context_match=0, max_sources=0):
        implementation = (
            FastMissIndexedPromptLookup.propose
            if self.fast_miss_active
            else IndexedPromptLookup.propose
        )
        result = implementation(
            self,
            max_span,
            lookback=lookback,
            min_context_match=min_context_match,
            max_sources=max_sources,
        )
        if result:
            self.consecutive_misses = 0
            self.fast_miss_active = False
        else:
            self.consecutive_misses += 1
            self.fast_miss_active = self.consecutive_misses >= self.activation_misses
        return result


def _histories(length: int, seed: int) -> dict[str, list[int]]:
    rng = random.Random(seed)
    periodic = ([11, 12, 13, 14, 15, 16, 17] * (length // 7 + 1))[:length]
    random_history = [rng.randrange(1 << 20) for _ in range(length)]
    # Many sites share the terminal trigram but differ afterward.  This is the
    # collision shape where scanning every source is expensive.
    collision = []
    while len(collision) + 7 <= length:
        collision.extend((1, 2, 3, rng.randrange(100, 1000), 1, 2, 3))
    collision.extend([1, 2, 3][: max(0, length - len(collision))])
    return {"periodic_hit": periodic, "random_miss": random_history, "collision_hit": collision}


def _measure(operation, repeats: int):
    samples = []
    for _ in range(5):
        start = time.perf_counter_ns()
        for _iteration in range(repeats):
            operation()
        samples.append((time.perf_counter_ns() - start) / repeats)
    return {
        "samples_ns_per_lookup": samples,
        "median_ns_per_lookup": statistics.median(samples),
    }


def run_benchmark(*, length: int = 16_384, repeats: int = 200, lookback: int = 4096, seed: int = 300) -> dict:
    rows = []
    for name, history in _histories(length, seed).items():
        current = IndexedPromptLookup(history, index_window=lookback)
        fast_miss = FastMissIndexedPromptLookup(history, index_window=lookback)
        adaptive = AdaptiveFastMissPromptLookup(history, index_window=lookback)
        reference = ExhaustiveIndexedPromptLookup(history)
        expected = reference.propose(8, lookback=lookback)
        actual = current.propose(8, lookback=lookback)
        candidate = fast_miss.propose(8, lookback=lookback)
        adaptive_result = adaptive.propose(8, lookback=lookback)
        if actual != expected or candidate != expected or adaptive_result != expected:
            raise RuntimeError(f"{name}: indexed proposer changed the proposal")
        if fast_miss.index_receipt() != current.index_receipt():
            raise RuntimeError(f"{name}: fast-miss arm changed timer-free counters")
        if adaptive.index_receipt() != current.index_receipt():
            raise RuntimeError(f"{name}: adaptive arm changed timer-free counters")
        current_timing = _measure(
            lambda current=current: current.propose(8, lookback=lookback), repeats
        )
        fast_miss_timing = _measure(
            lambda fast_miss=fast_miss: fast_miss.propose(8, lookback=lookback), repeats
        )
        adaptive_timing = _measure(
            lambda adaptive=adaptive: adaptive.propose(8, lookback=lookback), repeats
        )
        reference_timing = _measure(
            lambda reference=reference: reference.propose(8, lookback=lookback), repeats
        )
        rows.append(
            {
                "workload": name,
                "history_tokens": len(history),
                "lookback": lookback,
                "proposal": actual,
                "exact": True,
                "current": current_timing,
                "fast_miss_candidate": fast_miss_timing,
                "adaptive_fast_miss_candidate": adaptive_timing,
                "exhaustive_reference": reference_timing,
                "median_speedup": reference_timing["median_ns_per_lookup"]
                / max(current_timing["median_ns_per_lookup"], 1),
                "candidate_vs_current": current_timing["median_ns_per_lookup"]
                / max(fast_miss_timing["median_ns_per_lookup"], 1),
                "adaptive_candidate_vs_current": current_timing["median_ns_per_lookup"]
                / max(adaptive_timing["median_ns_per_lookup"], 1),
                "current_counters": current.index_receipt(),
                "candidate_counters": fast_miss.index_receipt(),
                "adaptive_candidate_counters": adaptive.index_receipt(),
                "adaptive_fast_miss_active": adaptive.fast_miss_active,
            }
        )
    return {
        "schema": "mlx2.pld-proposer-viability.v1",
        "gpu_used": False,
        "production_behavior_changed": False,
        "parameters": {
            "length": length,
            "repeats": repeats,
            "lookback": lookback,
            "seed": seed,
        },
        "candidate": "shortest-suffix proved-miss guard after repeated misses; research only",
        "source": {
            "repository": "incoai/splash",
            "pull_request": 300,
            "revision": "cde5e2372e015eef6060793a122b7e685b097425",
            "not_copied": "C++ flat chained hash table",
        },
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=16_384)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--lookback", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if min(args.length, args.repeats, args.lookback) < 1:
        parser.error("length, repeats, and lookback must be positive")
    report = run_benchmark(
        length=args.length, repeats=args.repeats, lookback=args.lookback, seed=args.seed
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(encoded + "\n")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

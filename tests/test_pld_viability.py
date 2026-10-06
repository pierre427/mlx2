from mlx2.runtime.prompt_lookup import IndexedPromptLookup
from scripts.bench_pld_viability import (
    AdaptiveFastMissPromptLookup,
    ExhaustiveIndexedPromptLookup,
    FastMissIndexedPromptLookup,
    run_benchmark,
)


def test_current_pld_index_matches_exhaustive_sources_and_incremental_append():
    history = [1, 2, 3, 9, 1, 2, 3]
    current = IndexedPromptLookup(history, ngram_min=3, ngram_max=3)
    reference = ExhaustiveIndexedPromptLookup(history, ngram_min=3, ngram_max=3)
    for token in (7, 1, 2, 3):
        assert current.propose(4, lookback=64) == reference.propose(4, lookback=64)
        current.observe(token)
        reference.observe(token)


def test_current_pld_index_prefers_newest_full_continuation_without_source_materialization():
    history = [1, 2, 3, 7, 8, 1, 2, 3, 9, 8, 1, 2, 3]
    current = IndexedPromptLookup(history, ngram_min=3, ngram_max=3)
    reference = ExhaustiveIndexedPromptLookup(history, ngram_min=3, ngram_max=3)
    assert current.propose(2, lookback=64) == reference.propose(2, lookback=64) == [9, 8]
    receipt = current.index_receipt()
    assert receipt["candidate_checks"] == 1
    assert receipt["target_occurrences_skipped"] >= 1


def test_pld_viability_benchmark_emits_exact_cpu_json_shape():
    report = run_benchmark(length=256, repeats=3, lookback=128, seed=7)
    assert report["gpu_used"] is False
    assert report["production_behavior_changed"] is False
    assert {row["workload"] for row in report["rows"]} == {
        "periodic_hit",
        "random_miss",
        "collision_hit",
    }
    assert all(row["exact"] for row in report["rows"])


def test_fast_miss_arm_preserves_proposal_and_timer_free_counters():
    for history in (
        list(range(128)),
        [1, 2, 3, 4] * 32,
        [1, 2, 3, 8, 1, 2, 3],
    ):
        current = IndexedPromptLookup(history, index_window=64)
        candidate = FastMissIndexedPromptLookup(history, index_window=64)
        current.add_hot_segment([9, 8, 7, 1, 2, 3, 6])
        candidate.add_hot_segment([9, 8, 7, 1, 2, 3, 6])
        for options in ({}, {"min_context_match": 4}, {"max_sources": 1}):
            expected = current.propose(4, lookback=64, **options)
            actual = candidate.propose(4, lookback=64, **options)
            assert actual == expected
            assert candidate.index_receipt() == current.index_receipt()
            current.feedback(len(expected), 0)
            candidate.feedback(len(actual), 0)
        assert candidate.index_receipt() == current.index_receipt()


def test_fast_miss_arm_preserves_randomized_incremental_semantics():
    import random

    rng = random.Random(19846)
    history = [rng.randrange(32) for _ in range(256)]
    current = IndexedPromptLookup(history, ngram_min=2, ngram_max=6, index_window=128)
    candidate = FastMissIndexedPromptLookup(history, ngram_min=2, ngram_max=6, index_window=128)
    for _round in range(200):
        options = {
            "lookback": rng.randrange(8, 129),
            "min_context_match": rng.randrange(0, 6),
            "max_sources": rng.randrange(0, 5),
        }
        span = rng.randrange(1, 9)
        expected = current.propose(span, **options)
        actual = candidate.propose(span, **options)
        assert actual == expected
        assert candidate.index_receipt() == current.index_receipt()
        token = rng.randrange(32)
        history.append(token)
        current.observe(token)
        candidate.observe(token)


def test_adaptive_fast_miss_arm_latches_only_after_repeated_misses():
    miss = AdaptiveFastMissPromptLookup(list(range(64)), activation_misses=3)
    for expected_active in (False, False, True):
        assert miss.propose(4, lookback=64) == []
        assert miss.fast_miss_active is expected_active

    hit = AdaptiveFastMissPromptLookup([1, 2, 3, 7, 1, 2, 3], activation_misses=1)
    assert hit.propose(1, lookback=64) == [7]
    assert hit.fast_miss_active is False

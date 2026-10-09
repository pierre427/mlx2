import random
from collections import defaultdict, deque

import pytest

from mlx2.runtime.prompt_lookup import (
    AdaptiveLookback,
    CostAwarePLDLatch,
    IndexedPromptLookup,
    RecentCommittedSegmentStore,
    plan_proposal_around_verify_cliff,
    verify_prompt_lookup,
)


class _ReferencePromptLookup:
    """The exhaustive pre-optimization lookup retained as a differential oracle."""

    def __init__(
        self, tokens=(), *, ngram_min=3, ngram_max=6, hot_segments=4, reject_ttl=8
    ):
        self.ngram_min, self.ngram_max = ngram_min, ngram_max
        self.tokens = []
        self.index = {
            size: defaultdict(list) for size in range(ngram_min, ngram_max + 1)
        }
        self.hot = deque(maxlen=hot_segments)
        self.reject_ttl = reject_ttl
        self.clock = 0
        self.rejected_until = {}
        self.last_source = None
        for token in tokens:
            self.observe(token)

    def observe(self, token):
        self.tokens.append(int(token))
        end = len(self.tokens)
        for size in self.index:
            start = end - size
            if start >= 0:
                self.index[size][tuple(self.tokens[start:end])].append(start)

    def add_hot_segment(self, tokens):
        self.hot.append(tuple(int(token) for token in tokens))

    def propose(self, max_span, *, lookback):
        self.clock += 1
        self.last_source = None
        size_limit = min(self.ngram_max, len(self.tokens))
        for size in range(size_limit, self.ngram_min - 1, -1):
            key = tuple(self.tokens[-size:])
            candidates = []
            for start in self.index[size].get(key, ()):
                if (
                    start >= len(self.tokens) - size
                    or len(self.tokens) - start > lookback
                ):
                    continue
                continuation = tuple(
                    self.tokens[start + size : start + size + max_span]
                )
                source = ("target", size, start, continuation)
                if continuation and self.rejected_until.get(source, -1) < self.clock:
                    candidates.append((start, continuation, source))
            for segment_id, segment in enumerate(self.hot):
                first = max(0, len(segment) - lookback - size)
                for start in range(first, len(segment) - size):
                    if segment[start : start + size] != key:
                        continue
                    continuation = segment[start + size : start + size + max_span]
                    source = ("hot", segment_id, size, start, continuation)
                    if (
                        continuation
                        and self.rejected_until.get(source, -1) < self.clock
                    ):
                        candidates.append((start, continuation, source))
            if candidates:
                chosen = max(candidates, key=lambda item: (len(item[1]), item[0]))
                self.last_source = chosen[2]
                return list(chosen[1])
        return []

    def feedback(self, proposed, accepted):
        if (
            self.last_source is not None
            and proposed
            and not accepted
            and self.reject_ttl
        ):
            self.rejected_until[self.last_source] = self.clock + self.reject_ttl


def test_indexed_oracle_hot_segment_and_rejection_ttl():
    oracle = IndexedPromptLookup([1, 2, 3, 9, 1, 2, 3], ngram_min=3, ngram_max=3)
    assert oracle.propose(4) == [9, 1, 2, 3]
    oracle.feedback(4, 0)
    assert oracle.propose(4) == []
    oracle.add_hot_segment([8, 1, 2, 3, 7, 6])
    assert oracle.propose(2) == [7, 6]


def test_indexed_oracle_matches_reference_across_randomized_state_changes():
    rng = random.Random(19)
    for _case in range(300):
        length = rng.randrange(3, 100)
        period = rng.randrange(1, min(24, length) + 1)
        base = [rng.randrange(32) for _ in range(period)]
        tokens = (base * (length // period + 1))[:length]
        ngram_min = rng.randrange(1, 4)
        ngram_max = rng.randrange(ngram_min, 7)
        index_window = rng.randrange(8, 65)
        hot_segments = rng.randrange(0, 4)
        reject_ttl = rng.randrange(0, 6)
        reference = _ReferencePromptLookup(
            tokens,
            ngram_min=ngram_min,
            ngram_max=ngram_max,
            hot_segments=hot_segments,
            reject_ttl=reject_ttl,
        )
        indexed = IndexedPromptLookup(
            tokens,
            ngram_min=ngram_min,
            ngram_max=ngram_max,
            hot_segments=hot_segments,
            reject_ttl=reject_ttl,
            index_window=index_window,
        )
        for _round in range(8):
            if hot_segments and rng.random() < 0.35:
                hot = [rng.randrange(32) for _ in range(rng.randrange(1, 80))]
                reference.add_hot_segment(hot)
                indexed.add_hot_segment(hot)
            max_span = rng.randrange(0, 12)
            lookback = rng.randrange(1, index_window + 1)
            expected = reference.propose(max_span, lookback=lookback)
            actual = indexed.propose(max_span, lookback=lookback)
            assert actual == expected
            assert indexed.last_source == reference.last_source
            accepted = len(actual) if rng.random() < 0.5 else 0
            reference.feedback(len(expected), accepted)
            indexed.feedback(len(actual), accepted)
            token = rng.randrange(32)
            reference.observe(token)
            indexed.observe(token)


def test_index_window_bounds_initial_memory_without_changing_lookup_result():
    tokens = [7] * 4096
    complete = IndexedPromptLookup(tokens, ngram_min=3, ngram_max=6)
    bounded = IndexedPromptLookup(tokens, ngram_min=3, ngram_max=6, index_window=256)
    assert bounded.propose(8, lookback=256) == complete.propose(8, lookback=256)
    assert (
        bounded.index_receipt()["target_occurrences_indexed"]
        < complete.index_receipt()["target_occurrences_indexed"] // 8
    )


def test_lookup_stops_after_the_first_optimal_candidate():
    oracle = IndexedPromptLookup([7] * 4096, ngram_min=3, ngram_max=6)
    assert oracle.propose(8, lookback=256) == [7] * 8
    receipt = oracle.index_receipt()
    assert receipt["candidate_checks"] == 1
    assert receipt["target_occurrences_visited"] <= 8
    assert receipt["target_occurrences_skipped"] > 3000


def test_hot_segments_use_the_index_and_preserve_target_tie_priority():
    oracle = IndexedPromptLookup(
        [1, 2, 3, 4, 1, 2, 3],
        ngram_min=3,
        ngram_max=3,
        index_window=64,
    )
    oracle.add_hot_segment([1, 2, 3, 4, 1])
    assert oracle.propose(2, lookback=64) == [4, 1]
    assert oracle.last_source[0] == "target"

    hot_only = IndexedPromptLookup([1, 2, 3], ngram_min=3, ngram_max=3, index_window=64)
    hot_only.add_hot_segment([0] * 128 + [1, 2, 3, 9, 8])
    assert hot_only.propose(2, lookback=64) == [9, 8]
    assert hot_only.last_source[0] == "hot"
    receipt = hot_only.index_receipt()
    assert receipt["hot_key_lookups"] == 1
    assert receipt["hot_starts_avoided"] > 50
    assert receipt["hot_segments_indexed"] == 1


def test_hot_segment_reports_cross_request_source_kind():
    lookup = IndexedPromptLookup([9, 1, 2], ngram_min=2, ngram_max=2)
    lookup.add_hot_segment(
        [7, 1, 2, 3, 4], source_kind="recent_committed"
    )
    assert lookup.propose(2) == [3, 4]
    assert lookup.last_source_kind == "recent_committed"


def test_recent_committed_store_is_scope_capacity_tail_and_age_bounded():
    now = [100.0]
    store = RecentCommittedSegmentStore(
        max_responses=2,
        max_tokens=5,
        max_response_tokens=3,
        max_age_seconds=10,
        clock=lambda: now[0],
    )
    first = store.publish("scope-a", [1, 2, 3, 4], segment_id="first")
    assert first.tokens == (2, 3, 4)
    store.publish("scope-b", [5, 6], segment_id="other")
    assert [entry.segment_id for entry in store.snapshot("scope-a")] == ["first"]
    scoped = store.status("scope-a")
    assert scoped["responses"] == 1
    assert scoped["tokens"] == 3
    assert "published" not in scoped
    store.publish("scope-a", [7, 8], segment_id="newest")
    assert [entry.segment_id for entry in store.snapshot("scope-a")] == ["newest"]
    assert store.status()["capacity_evictions"] == 1
    now[0] = 111.0
    assert store.snapshot("scope-a") == ()
    assert store.status()["responses"] == 0
    assert store.status()["age_evictions"] == 2


def test_recent_committed_store_clamps_one_response_to_total_budget():
    store = RecentCommittedSegmentStore(
        max_responses=4,
        max_tokens=3,
        max_response_tokens=8,
    )
    entry = store.publish("scope", [1, 2, 3, 4, 5], segment_id="wide")
    assert entry.tokens == (3, 4, 5)
    assert store.snapshot("scope") == (entry,)
    assert store.status()["tokens"] == 3
    assert store.status().get("capacity_evictions", 0) == 0


def test_recent_committed_store_refreshes_exact_scope_duplicates():
    now = [1.0]
    store = RecentCommittedSegmentStore(clock=lambda: now[0])
    first = store.publish("scope", [1, 2, 3], segment_id="first")
    indexed = store.indexed_segment(
        first.scope,
        first.tokens,
        source_kind="recent_committed",
        source_id=first.cache_id,
        ngram_min=2,
        ngram_max=2,
    )
    now[0] = 2.0
    newest = store.publish("scope", [1, 2, 3], segment_id="newest")
    assert newest != first
    assert store.snapshot("scope") == (newest,)
    status = store.status()
    assert status["responses"] == 1
    assert status["tokens"] == 3
    assert status["published"] == 2
    assert status["duplicate_refreshes"] == 1
    assert newest.cache_id == first.cache_id
    assert store.indexed_segment(
        newest.scope,
        newest.tokens,
        source_kind="recent_committed",
        source_id=newest.cache_id,
        ngram_min=2,
        ngram_max=2,
    ) is indexed


def test_recent_store_shared_indexes_are_immutable_scoped_and_reused():
    store = RecentCommittedSegmentStore(max_responses=2)
    first = store.indexed_segment(
        "scope-a",
        (1, 2, 3, 1, 2, 4),
        source_kind="recent_committed",
        source_id="one",
        ngram_min=2,
        ngram_max=3,
        index_window=16,
    )
    reused = store.indexed_segment(
        "scope-a",
        first.tokens,
        source_kind="recent_committed",
        source_id="one",
        ngram_min=2,
        ngram_max=3,
        index_window=16,
    )
    other_scope = store.indexed_segment(
        "scope-b",
        first.tokens,
        source_kind="recent_committed",
        source_id="one",
        ngram_min=2,
        ngram_max=3,
        index_window=16,
    )
    assert reused is first
    assert other_scope is not first
    assert isinstance(first.index[2][(1, 2)], tuple)
    with pytest.raises(TypeError):
        first.index[2][(1, 2)] = (99,)
    with pytest.raises(TypeError):
        first.index[2] = {}
    assert store.status("scope-a")["shared_index_entries"] == 1
    assert store.status("scope-a")["shared_index_hits"] == 1
    status = store.status()
    assert status["shared_index_entries"] == 2
    assert status["shared_index_hits"] == 1
    assert status["shared_index_builds"] == 2


def test_recent_store_expiry_removes_its_shared_index():
    now = [1.0]
    store = RecentCommittedSegmentStore(
        max_responses=2, max_age_seconds=2, clock=lambda: now[0]
    )
    entry = store.publish("scope", [1, 2, 3, 4], segment_id="response")
    store.indexed_segment(
        entry.scope,
        entry.tokens,
        source_kind="recent_committed",
        source_id=entry.cache_id,
        ngram_min=2,
        ngram_max=3,
        index_window=16,
    )
    assert store.status()["shared_index_entries"] == 1
    now[0] = 4.0
    assert store.snapshot("scope") == ()
    status = store.status()
    assert status["shared_index_entries"] == 0
    assert status["shared_index_lifecycle_evictions"] == 1


def test_recent_store_shared_index_obeys_total_token_budget():
    store = RecentCommittedSegmentStore(
        max_responses=4, max_tokens=5, max_response_tokens=5
    )
    first = store.indexed_segment(
        "scope",
        (1, 2, 3),
        source_kind="apcv2_transcript",
        source_id="first",
        ngram_min=2,
        ngram_max=2,
    )
    second = store.indexed_segment(
        "scope",
        (4, 5, 6),
        source_kind="apcv2_transcript",
        source_id="second",
        ngram_min=2,
        ngram_max=2,
    )
    assert first is not second
    status = store.status()
    assert status["shared_index_entries"] == 1
    assert status["shared_index_tokens"] == 3
    assert status["shared_index_capacity_evictions"] == 1


def test_recent_store_concurrent_shared_index_build_converges():
    from concurrent.futures import ThreadPoolExecutor

    store = RecentCommittedSegmentStore(max_responses=4)
    tokens = tuple(range(512))

    def build(_index):
        return store.indexed_segment(
            "scope",
            tokens,
            source_kind="apcv2_transcript",
            source_id="shared",
            ngram_min=3,
            ngram_max=6,
            index_window=1024,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = tuple(pool.map(build, range(16)))
    assert len({id(result) for result in results}) == 1
    assert store.status()["shared_index_entries"] == 1


def test_context_match_screens_ambiguous_recent_copy_source():
    # Both sites match the 3-token suffix; only the older one also matches
    # the preceding token. The new knobs are opt-in, including the scan cap.
    history = [7, 1, 2, 3, 9, 8, 1, 2, 3, 6, 7, 1, 2, 3]
    oracle = IndexedPromptLookup(history, ngram_min=3, ngram_max=3)
    assert oracle.propose(2) == [6, 7]
    assert oracle.propose(2, min_context_match=4) == [9, 8]
    assert oracle.propose(2, min_context_match=4, max_sources=1) == []
    assert oracle.propose(2, min_context_match=4, max_sources=2) == [9, 8]
    assert oracle.lookup_calls == 4
    assert oracle.context_mismatch_sites >= 2


def test_recent_lookup_stops_at_window_even_for_repeated_long_history():
    class CountedBucket(list):
        visited = 0

        def __reversed__(self):
            for item in super().__reversed__():
                self.visited += 1
                yield item

    oracle = IndexedPromptLookup([1, 2] * 5000, ngram_min=2, ngram_max=2)
    bucket = CountedBucket(oracle.index[2][(1, 2)])
    oracle.index[2][(1, 2)] = bucket
    assert oracle.propose(2, lookback=64) == [1, 2]
    assert bucket.visited <= 34  # 32 in-window sites and one boundary
    assert oracle.source_sites_scanned <= 32


def test_recent_prompt_segments_index_only_the_tail_and_evict_as_tip_moves():
    history = [1, 2, 3, 9] + list(range(20, 120)) + [1, 2, 3]
    oracle = IndexedPromptLookup(
        history, ngram_min=3, ngram_max=3,
        recent_prompt_segments=2, prompt_segment_tokens=4,
    )
    assert oracle.indexed_start == len(history) - 8
    assert oracle.propose(1, lookback=1000) == []  # old match is outside both segments
    assert oracle.index_entries <= 8

    oracle = IndexedPromptLookup(
        [1, 2, 3, 7, 1, 2, 3], ngram_min=3, ngram_max=3,
        recent_prompt_segments=2, prompt_segment_tokens=4,
    )
    assert oracle.propose(1, lookback=1000) == [7]
    oracle.observe(9)
    oracle.observe(8)
    assert oracle.indexed_start == 1
    assert tuple(oracle.index[3][(1, 2, 3)]) == (4,)
    assert oracle.index_entries <= 8


def test_recent_prompt_segment_index_matches_full_index_with_same_token_window():
    rng = random.Random(718)
    history = [rng.randrange(7) for _ in range(300)]
    full = IndexedPromptLookup(history, ngram_min=2, ngram_max=4)
    recent = IndexedPromptLookup(
        history, ngram_min=2, ngram_max=4,
        recent_prompt_segments=3, prompt_segment_tokens=16,
    )
    for _ in range(100):
        assert recent.propose(5, lookback=48) == full.propose(5, lookback=48)
        assert recent.propose(5, lookback=48, max_sources=8, min_context_match=5) == full.propose(
            5, lookback=48, max_sources=8, min_context_match=5,
        )
        assert recent.index_entries <= 48 * 3
        token = rng.randrange(7)
        recent.observe(token)
        full.observe(token)


def test_cost_latch_requires_multi_token_shadow_and_own_goodput():
    latch = CostAwarePLDLatch(
        shadow_span=4, probe_stride=4, shadow_window=2,
        plain_rounds=8, explore_rounds=2, park_rounds=4,
        reprobe_interval=16,
    )
    for i in range(8):
        if latch.should_probe(i):
            latch.start_shadow([1, 2, 3, 4])
        latch.observe_plain_token((1, 2, 3, 4)[i % 4])
        latch.observe_round(10, 1, 0, i + 1)
    assert latch.state == "explore"
    assert latch.receipt()["shadow_matches"] == 8
    latch.observe_round(40, 8, 7, 16)
    latch.observe_round(40, 8, 7, 24)
    assert latch.state == "active"
    for i in range(4):
        latch.observe_round(100, 1, 1, 25 + i)
    assert latch.state == "parked" and latch.reprobe_at >= 42
    assert not latch.should_probe(30)
    assert latch.receipt()["parks"] == 1


def test_cost_latch_stays_parked_on_one_token_hits_and_width_change():
    latch = CostAwarePLDLatch(shadow_span=3, probe_stride=1, shadow_window=2,
                              plain_rounds=2, reprobe_interval=8)
    for i in range(6):
        if latch.should_probe(i):
            latch.start_shadow([1, 2, 3])
        latch.observe_plain_token(1 if i % 2 == 0 else 9)
        latch.observe_round(10, 1, 0, i + 1)
    assert latch.state == "parked"
    assert latch.receipt()["shadow_matches"] < latch.receipt()["shadow_trials"] * 3
    latch.state = "active"
    latch.suspend_for_width(6)
    assert latch.state == "parked" and latch.receipt()["width_parks"] == 1


def test_verifier_accepts_prefix_and_returns_exact_boundary():
    full = verify_prompt_lookup([4, 5], [4, 5, 6])
    assert full.accepted == (4, 5) and full.next_token == 6
    assert full.receipt["bonus"]
    partial = verify_prompt_lookup([4, 8], [4, 5, 6])
    assert partial.accepted == (4,) and partial.next_token == 5
    assert partial.receipt["verified"]


def test_adaptive_lookback_widens_on_miss_and_narrows_on_reject():
    policy = AdaptiveLookback((8, 32), misses=2, rejects=1)
    policy.observe(0, 0)
    policy.observe(0, 0)
    assert policy.current == 32 and policy.widen_events == 1
    policy.observe(3, 0)
    assert policy.current == 8 and policy.narrow_events == 1


def test_verify_cliff_planner_extends_or_snaps_without_inventing_capacity():
    assert plan_proposal_around_verify_cliff(8, 15) == 15
    assert plan_proposal_around_verify_cliff(8, 10) == 7
    assert plan_proposal_around_verify_cliff(4, 20) == 4
    with pytest.raises(ValueError):
        plan_proposal_around_verify_cliff(-1, 2)


def test_prompt_lookup_policy_rejects_unknown_keys():
    import pytest

    from mlx2.runtime.pld import PromptLookupBatchGenerator

    assert PromptLookupBatchGenerator.validate_policy({"num_draft": 4}) == {"num_draft": 4}
    with pytest.raises(ValueError, match="unknown prompt_lookup policy keys: numdraft"):
        PromptLookupBatchGenerator.validate_policy({"numdraft": 4})
    with pytest.raises(ValueError, match="must be an object"):
        PromptLookupBatchGenerator.validate_policy([1])


def test_prompt_lookup_policy_rejects_coerced_or_nonpositive_num_draft():
    import pytest

    from mlx2.runtime.pld import PromptLookupBatchGenerator

    for invalid in (True, False, 0, -1, 1.9, "3", None):
        with pytest.raises(ValueError, match="positive integer"):
            PromptLookupBatchGenerator.validate_policy({"num_draft": invalid})

    assert PromptLookupBatchGenerator.validate_policy({"num_draft": 3}) == {
        "num_draft": 3
    }


def test_prompt_lookup_policy_strictly_validates_every_runtime_value():
    import math

    import pytest

    from mlx2.runtime.pld import PromptLookupBatchGenerator

    invalid = (
        {"ngram_min": True},
        {"ngram_max": 2.5},
        {"ngram_min": 4, "ngram_max": 3},
        {"hot_segments": -1},
        {"reject_ttl": "8"},
        {"min_context_match": True},
        {"min_context_match": 65},
        {"max_sources": -1},
        {"recent_prompt_segments": -1},
        {"prompt_segment_tokens": 0},
        {"recent_prompt_segments": 65, "prompt_segment_tokens": 1024},
        {"recent_prompt_segments": 2, "retrieval_segments": [[1, 2, 3]]},
        {"lookback_misses": 0},
        {"lookback_rejects": False},
        {"adaptive_warmup": -1},
        {"adaptive": "false"},
        {"adaptive_gate": "nan"},
        {"adaptive_gate": math.nan},
        {"adaptive_gate": math.inf},
        {"adaptive_gate": -0.1},
        {"adaptive_gate": 1.1},
        {"deferred_admission": 1},
        {"cliff_aware_span": "true"},
        {"admission_window": 0},
        {"admission_confirm_windows": False},
        {"admission_reprobe_interval": -1},
        {"admission_probe_stride": 0},
        {"cost_aware_admission": 1},
        {"cost_shadow_span": 1},
        {"cost_probe_stride": 0},
        {"cost_shadow_gate": 1.1},
        {"cost_margin": -0.1},
        {"cost_plain_rounds": 0},
        {"cost_aware_admission": True, "deferred_admission": True},
        {"cost_aware_admission": True, "num_draft": 2, "cost_shadow_span": 3},
        {"admission_gate": "0.5"},
        {"admission_gate": math.nan},
        {"admission_gate": math.inf},
        {"admission_gate": -0.1},
        {"admission_gate": 1.1},
        {"verify_cliff_start": 1, "verify_cliff_end": 1},
        {"verify_cliff_end": 2.5},
        {"verify_cliff_start": 10, "verify_cliff_end": 9},
        {"lookback_ladder": "8,32"},
        {"lookback_ladder": []},
        {"lookback_ladder": [8, "32"]},
        {"lookback_ladder": [8, 8]},
        {"retrieval_segments": "1,2"},
        {"retrieval_segments": [[]]},
        {"retrieval_segments": [[1, "2"]]},
        {"retrieval_segments": [[-1]]},
    )
    for policy in invalid:
        with pytest.raises(ValueError):
            PromptLookupBatchGenerator.validate_policy(policy)

    assert PromptLookupBatchGenerator.validate_policy(
        {
            "ngram_min": 2,
            "ngram_max": 4,
            "hot_segments": 0,
            "reject_ttl": 0,
            "lookback_ladder": (8, 32),
            "lookback_misses": 1,
            "lookback_rejects": 1,
            "retrieval_segments": ((1, 2),),
            "adaptive": False,
            "adaptive_warmup": 0,
            "adaptive_gate": 1,
            "deferred_admission": True,
            "admission_window": 8,
            "admission_gate": 0.5,
            "admission_confirm_windows": 2,
            "admission_reprobe_interval": 0,
            "cliff_aware_span": True,
            "verify_cliff_start": 9,
            "verify_cliff_end": 15,
        }
    ) == {
        "ngram_min": 2,
        "ngram_max": 4,
        "hot_segments": 0,
        "reject_ttl": 0,
        "lookback_ladder": [8, 32],
        "lookback_misses": 1,
        "lookback_rejects": 1,
        "retrieval_segments": [[1, 2]],
        "adaptive": False,
        "adaptive_warmup": 0,
        "adaptive_gate": 1.0,
        "deferred_admission": True,
        "admission_window": 8,
        "admission_gate": 0.5,
        "admission_confirm_windows": 2,
        "admission_reprobe_interval": 0,
        "cliff_aware_span": True,
        "verify_cliff_start": 9,
        "verify_cliff_end": 15,
    }


def test_prompt_lookup_lane_policy_overrides_use_the_same_strict_contract():
    import pytest

    from mlx2.runtime.models.cache import KVCache
    from mlx2.runtime.pld import PromptLookupBatchGenerator

    generator = PromptLookupBatchGenerator(object())
    with pytest.raises(ValueError, match="adaptive must be a boolean"):
        generator.insert(
            [[1]],
            caches=[[KVCache()]],
            prompt_lookup_configs=[{"adaptive": "false"}],
        )


def test_prompt_lookup_lane_bounds_default_index_to_the_configured_ladder():
    from mlx2.runtime.models.cache import KVCache
    from mlx2.runtime.pld import PromptLookupBatchGenerator

    generator = PromptLookupBatchGenerator(
        object(), prompt_lookup={"lookback_ladder": [8, 32]}
    )
    uid = generator.insert([[1, 2, 3]], caches=[[KVCache()]])[0]
    try:
        proposer = generator.lanes[uid].proposer
        assert proposer.index_window == 32
        assert proposer.index_receipt()["index_window"] == 32
    finally:
        generator.remove([uid])


def test_pld_recent_segments_share_apcv2_scope_and_transcript_segments():
    from mlx2.runtime.cache_planes import (
        TranscriptLedgerPlane,
        TranscriptLedgerSegment,
    )
    from mlx2.runtime.models.cache import KVCache
    from mlx2.runtime.pld import PromptLookupBatchGenerator

    scope = ("apcv2-key", "tenant-a")
    store = RecentCommittedSegmentStore()
    store.publish(scope, [8, 1, 2, 6], segment_id="recent")
    ledger = TranscriptLedgerPlane(
        tokenizer_identity="tok",
        tokenizer_version="v1",
        revision="rev",
        segments=(TranscriptLedgerSegment("prior", 0, 5, (9, 1, 2, 7, 5)),),
    )
    generator = PromptLookupBatchGenerator(
        object(),
        prompt_lookup={
            "recent_committed_segments": True,
            "ngram_min": 2,
            "ngram_max": 2,
        },
        recent_source_store=store,
    )
    uid = generator.insert(
        [[1, 2]],
        caches=[[KVCache()]],
        pld_source_scopes=[scope],
        apc_transcript_ledgers=[ledger],
    )[0]
    try:
        lane = generator.lanes[uid]
        assert lane.apcv2_source_segments == 1
        assert lane.recent_source_segments == 1
        assert lane.proposer.propose(2) == [7, 5]
        assert lane.proposer.last_source_kind == "apcv2_transcript"
    finally:
        generator.remove([uid])


def test_pld_lanes_share_engine_owned_recent_postings():
    from mlx2.runtime.models.cache import KVCache
    from mlx2.runtime.pld import PromptLookupBatchGenerator

    scope = ("apcv2-key", "tenant-a")
    store = RecentCommittedSegmentStore()
    store.publish(scope, [8, 1, 2, 6], segment_id="recent")
    generator = PromptLookupBatchGenerator(
        object(),
        completion_batch_size=2,
        prompt_lookup={
            "recent_committed_segments": True,
            "ngram_min": 2,
            "ngram_max": 2,
        },
        recent_source_store=store,
    )
    first, second = generator.insert(
        [[1, 2], [1, 2]],
        caches=[[KVCache()], [KVCache()]],
        pld_source_scopes=[scope, scope],
    )
    try:
        first_hot = generator.lanes[first].proposer.hot[0]
        second_hot = generator.lanes[second].proposer.hot[0]
        assert first_hot is second_hot
        assert first_hot.index is second_hot.index
        assert store.status()["shared_index_builds"] == 1
        assert store.status()["shared_index_hits"] == 1
    finally:
        generator.remove([first, second])


def test_pld_apcv2_transcript_sources_obey_response_and_total_bounds():
    from mlx2.runtime.cache_planes import (
        TranscriptLedgerPlane,
        TranscriptLedgerSegment,
    )
    from mlx2.runtime.models.cache import KVCache
    from mlx2.runtime.pld import PromptLookupBatchGenerator

    scope = ("apcv2-key", "tenant-a")
    ledger = TranscriptLedgerPlane(
        tokenizer_identity="tok",
        tokenizer_version="v1",
        revision="rev",
        segments=(
            TranscriptLedgerSegment("older", 0, 5, (1, 2, 3, 4, 5)),
            TranscriptLedgerSegment("newer", 5, 10, (6, 7, 8, 9, 10)),
        ),
    )
    generator = PromptLookupBatchGenerator(
        object(),
        prompt_lookup={
            "recent_committed_segments": True,
            "recent_committed_max_responses": 2,
            "recent_committed_max_tokens": 5,
            "recent_committed_response_tokens": 3,
        },
        recent_source_store=RecentCommittedSegmentStore(
            max_responses=2,
            max_tokens=5,
            max_response_tokens=3,
        ),
    )
    uid = generator.insert(
        [[8, 9]],
        caches=[[KVCache()]],
        pld_source_scopes=[scope],
        apc_transcript_ledgers=[ledger],
    )[0]
    try:
        hot = list(generator.lanes[uid].proposer.hot)
        assert [segment.tokens for segment in hot] == [(8, 9, 10), (4, 5)]
        assert sum(len(segment.tokens) for segment in hot) == 5
    finally:
        generator.remove([uid])


def test_pld_recent_segments_fail_closed_without_engine_store_or_scope():
    from mlx2.runtime.models.cache import KVCache
    from mlx2.runtime.pld import PromptLookupBatchGenerator

    with pytest.raises(ValueError, match="engine-owned source store"):
        PromptLookupBatchGenerator(
            object(), prompt_lookup={"recent_committed_segments": True}
        )
    generator = PromptLookupBatchGenerator(
        object(),
        prompt_lookup={"recent_committed_segments": True},
        recent_source_store=RecentCommittedSegmentStore(),
    )
    with pytest.raises(ValueError, match="APCv2-bound source scope"):
        generator.insert([[1]], caches=[[KVCache()]])


def test_prompt_lookup_policy_allowlist_covers_every_key_the_generator_reads():
    import re
    from pathlib import Path

    from mlx2.runtime.pld import PromptLookupBatchGenerator

    source = Path(PromptLookupBatchGenerator.__module__.replace(".", "/") + ".py")
    source = Path(__file__).resolve().parents[1] / "src" / source
    read_keys = set(re.findall(r'config(?:\.get\(|\[)"([a-z_]+)"', source.read_text()))
    assert read_keys, "expected the generator to read policy keys"
    assert read_keys <= PromptLookupBatchGenerator.POLICY_KEYS


def test_prompt_lookup_policy_null_is_still_a_misplaced_key(monkeypatch):
    """An explicit ``"prompt_lookup": null`` must not slip past the route check."""
    from types import SimpleNamespace as NS

    import pytest

    from mlx2 import serving

    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "fake"})

    class Adapter:
        max_context = 1000
        identity = {"fingerprint": "fake"}
        environment = {}
        layout = "fake"
        model = None
        tokenizer = NS(vocab_size=10, eos_token_ids=[])

        def __init__(self, _path, execution_policy=None):
            pass

        def profile_name(self, _mtp):
            return "fake"

        def execution_config(self, **_kw):
            return {"num_draft": 0}

        def diagnostics(self):
            return {}

        def close(self):
            pass

    engine = serving.ServingEngine(
        "fake", adapter_factory=Adapter, qualification_mode=True, mtp=False,
        execution_policy={"prompt_lookup": None},
    )
    try:
        engine.thread.join(5)
        assert engine.error and "prompt-lookup route is not selected" in engine.error
        with pytest.raises(RuntimeError):
            engine.submit({"prompt": "hi"})
    finally:
        engine.close()


@pytest.mark.parametrize(
    "source_kind", ["retrieval", "apcv2_transcript", "recent_committed"]
)
def test_hot_source_context_match_beyond_ngram_max(source_kind):
    # Hot segments hold their tokens as a tuple while local history is a
    # list, and a tuple never equals a list: every hot candidate failed a
    # min_context_match above the matched n-gram size.  The hot source agrees
    # with the live context for all 20 tokens, so a context requirement of 8
    # (> ngram_max=6) must admit it, exactly as it does from local history.
    context = list(range(100, 120))
    source = context + [700, 701, 702, 703]
    local = IndexedPromptLookup([1, 2, *source, 5, 6, *context], index_window=4096)
    assert local.propose(4, min_context_match=8, max_sources=8) == [700, 701, 702, 703]

    hot = IndexedPromptLookup(context, index_window=4096)
    if source_kind == "retrieval":
        hot.add_hot_segment(source)
    else:
        store = RecentCommittedSegmentStore()
        hot.add_indexed_hot_segment(
            store.indexed_segment(
                ("scope",),
                source,
                source_kind=source_kind,
                source_id="s",
                ngram_min=hot.ngram_min,
                ngram_max=hot.ngram_max,
                index_window=hot.index_window,
            )
        )
    assert hot.propose(4, min_context_match=8, max_sources=8) == [700, 701, 702, 703]
    assert hot.last_source_kind == source_kind
    assert hot.context_mismatch_sites == 0
    # A real disagreement before the n-gram is still screened out.
    screened = IndexedPromptLookup(context, index_window=4096)
    screened.add_hot_segment([999, *context[-7:], 700, 701])
    assert screened.propose(2, min_context_match=8, max_sources=8) == []
    assert screened.context_mismatch_sites > 0

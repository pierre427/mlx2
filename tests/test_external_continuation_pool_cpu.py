"""Actual complete-path execution, transactional feedback and sampled laws."""

import copy

import mlx.core as mx
import numpy as np
import pytest
from test_continuation_providers_cpu import wrapped
from test_parallel_draft_apcv2_batching_cpu import pair
from test_standard_xpress_serving_cpu import reference

from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
from mlx2.runtime.speculative_sampling import softmax
from mlx2.serving import proposal_session_scope_hash


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    from mlx2.runtime import proposal_pool

    monkeypatch.setattr(
        proposal_pool,
        "_SHARED_RANKING_REGISTRY",
        proposal_pool.ProposalRankingRegistry(),
    )
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def batch(kind="xpress", adaptive=None):
    model, base = pair(kind)
    draft = wrapped(model, base, {"sources": ["external"], "limit": 15})
    generator = ExternalDraftBatchGenerator(
        model,
        draft_model=draft,
        binding="test-pool",
        num_draft=2,
        prefill_step_size=64,
        adaptive_verification=adaptive,
        ready_drain="all",
    )
    return model, base, draft, generator


def prefill(generator, prompts, **kw):
    uids = generator.insert(prompts, max_tokens=[8] * len(prompts), **kw)
    for uid in uids:
        while generator.lanes[uid].anchor is None:
            generator._prefill(generator.lanes[uid])
    return [generator.lanes[uid] for uid in uids]


@pytest.mark.parametrize("kind", ["xpress", "lilicorr"])
def test_real_pool_executes_complete_paths_and_censors_feedback(kind, monkeypatch):
    model, _base, draft, generator = batch(kind)
    prompt = [1, 2, 3]
    lane = prefill(generator, [prompt])[0]
    expected = reference(model, prompt, 8)
    calls = []
    original = model.forward_with_taps

    def forward(tokens, *args, **kwargs):
        calls.append(tuple(tokens.shape))
        return original(tokens, *args, **kwargs)

    monkeypatch.setattr(model, "forward_with_taps", forward)
    while lane.generated < lane.maximum:
        generator._round([lane])
    emitted = [r.token for r in lane.ready]
    assert emitted == expected
    expanded = [shape for shape in calls if shape[1] > 1]
    assert expanded and all(
        1 <= shape[0] <= 15 and 2 <= shape[1] <= 3 for shape in expanded
    )
    if kind == "xpress":
        assert any(shape == (15, 3) for shape in expanded)
    final = lane.ready[-1]
    receipt = final.speculative_receipt["continuation_pool"]
    assert receipt["observed_used"] and not receipt["qualified"]
    assert receipt["target_rows"] == sum(b * s for b, s in expanded)
    assert draft.proposal_pool.feedback_revision == len(expanded)
    assert not draft.proposal_pool._pending
    counts = draft.proposal_pool.observation_counts("external")
    assert counts[0][0] + counts[0][1] >= len(expanded)
    final.cache_sidecar.validate(generator.binding, len(final.all_tokens))
    ordinary = model.make_cache()
    mx.eval(model(mx.array([prompt + emitted[:-1]]), cache=ordinary))
    for actual, expected_cache in zip(lane.cache, ordinary):
        assert actual.offset == expected_cache.offset
        for left, right in zip(actual.state, expected_cache.state):
            np.testing.assert_allclose(
                np.asarray(left[..., : actual.offset, :]),
                np.asarray(right[..., : expected_cache.offset, :]),
                atol=3e-6,
                rtol=3e-6,
            )


def test_mixed_temperature_pool_draws_once_from_each_actual_target_prefix(monkeypatch):
    model, _base, _draft, generator = batch()
    prompts = [[1, 2, 3], [3, 2, 1]]
    lanes = prefill(
        generator,
        prompts,
        sampling_configs=[{"sampling_temp": 0}, {"sampling_temp": 0.8}],
    )
    original = generator._target_law
    seen = []

    def law(lane, logits, history, *args, history_suffix=()):
        history = [*history, *history_suffix]
        actual = original(lane, logits, history, *args)
        ordinary = model(mx.array([history]), cache=model.make_cache())[0, -1]
        if lane.sampling.get("sampling_temp"):
            expected = softmax(np.asarray(ordinary.astype(mx.float32)), 0.8)
            np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-6)
        else:
            assert int(np.argmax(actual)) == int(mx.argmax(ordinary).item())
        seen.append((lane.uid, tuple(history)))
        return actual

    monkeypatch.setattr(generator, "_target_law", law)
    generator._round(lanes)
    for lane in lanes:
        assert lane.rng.draws == len(lane.ready)
        assert sum(uid == lane.uid for uid, _ in seen) == len(lane.ready)
    assert generator.scheduler_stats["external_continuation_rounds"] == 2


def test_later_request_failure_restores_all_state_and_releases_tickets(monkeypatch):
    model, _base, draft, generator = batch()
    lanes = prefill(generator, [[1, 2, 3], [3, 2, 1]])
    before = [
        (
            list(l.history),
            l.anchor,
            l.rng.snapshot(),
            copy.deepcopy(l.cache),
            copy.deepcopy(l.draft_cache),
        )
        for l in lanes
    ]
    original = model.forward_with_taps
    calls = 0

    def fail(tokens, *args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(tokens, *args, **kwargs)
        if calls == 2:
            raise RuntimeError("late target failure")
        return result

    monkeypatch.setattr(model, "forward_with_taps", fail)
    with pytest.raises(RuntimeError, match="late target"):
        generator._round(lanes)
    assert (
        draft.proposal_pool.feedback_revision == 0 and not draft.proposal_pool._pending
    )
    assert generator.scheduler_stats.get("external_continuation_rounds", 0) == 0
    for lane, (history, anchor, rng, cache, dc) in zip(lanes, before):
        assert (
            lane.history == history
            and lane.anchor == anchor
            and lane.rng.snapshot() == rng
        )
        assert not lane.ready and lane.generated == 0
        assert not getattr(lane, "continuation_coverage", {})
        for a, b in zip(lane.cache, cache):
            assert a.offset == b.offset
            for x, y in zip(a.state, b.state):
                np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
        assert [c.offset for c in lane.draft_cache] == [c.offset for c in dc]
    monkeypatch.setattr(model, "forward_with_taps", original)
    generator._round(lanes)
    assert (
        draft.proposal_pool.feedback_revision == 2 and not draft.proposal_pool._pending
    )


def test_explicit_session_hash_is_private_tenant_route_bound_and_anonymous_unique():
    key = proposal_session_scope_hash("tenant-a", "session", "route")
    assert key == proposal_session_scope_hash("tenant-a", "session", "route")
    assert key != proposal_session_scope_hash("tenant-b", "session", "route")
    assert key != proposal_session_scope_hash("tenant-a", "session", "other-route")
    assert proposal_session_scope_hash("tenant-a", None, "route") is None
    _, _, _, first = batch()
    _, _, _, second = batch()
    a = prefill(first, [[1, 2, 3]], session_keys=[key])[0]
    b = prefill(second, [[1, 2, 3]])[0]
    c = prefill(first, [[1, 2, 3]])[0]
    assert a.session_scope_hash == key and b.session_scope_hash != c.session_scope_hash
    assert b.proposal_request_id != c.proposal_request_id


def adaptive():
    # Synthetic CPU mechanism costs, never benchmark/performance evidence.
    return {
        "continuation_costs": {"1": [1, 1, 20], "15": [1, 15, 300]},
        "mode": "per_request",
        "min_observations": 2,
        "full_depth_interval": 32,
    }


def test_adaptive_pool_uses_separate_measured_shape_contract_and_physical_trimming(
    monkeypatch,
):
    model, _base, draft, generator = batch(adaptive=adaptive())
    lane = prefill(generator, [[1, 2, 3]])[0]
    calls = []
    original = model.forward_with_taps

    def forward(tokens, *args, **kwargs):
        calls.append(tuple(tokens.shape))
        return original(tokens, *args, **kwargs)

    monkeypatch.setattr(model, "forward_with_taps", forward)
    generator._round([lane])
    assert calls == [
        (15, 3)
    ]  # Cold evidence and periodic exploration use full width/depth.
    lane.ready.clear()
    # Additional prior target-committed observations calibrate each eligible top-W subset.
    for _ in range(4):
        generator._round([lane])
        lane.ready.clear()
        if lane.generated >= lane.maximum - 2:
            lane.maximum += 8
    # Actual prior observations have populated coverage; no current target results are peeked.
    calls.clear()
    generator._round([lane])
    assert calls == [(1, 2)]
    receipt = lane.ready[-1].speculative_receipt
    assert receipt["continuation_pool"]["physical_width"] == 1
    assert receipt["adaptive_verification"]["observed_used"]
    assert receipt["adaptive_verification"]["continuation_cost_path_widths"] == [1, 15]
    assert receipt["adaptive_verification"]["trimmed_target_rows"] > 0
    assert draft.proposal_pool.feedback_revision == 6


def test_pool_stop_grammar_processes_only_reached_prefix_and_censors_later_labels():
    from test_external_adaptive_hybrid_cpu import StopGrammar

    _model, _base, draft, generator = batch()
    grammar = StopGrammar()
    generator.stops = {0}
    lane = prefill(generator, [[1, 2, 3]], logits_processors=[[grammar]])[0]
    generator._round([lane])
    assert [r.token for r in lane.ready] == [0] and grammar.steps == 1
    assert lane.ready[0].finish_reason == "stop" and len(lane.history) == 3
    counts = draft.proposal_pool.observation_counts("external")
    assert counts[0][0] + counts[0][1] > 0
    assert counts[1] == (0, 0)


@pytest.mark.parametrize("kind", ["xpress", "lilicorr"])
def test_pool_actual_paired_apcv2_resume_and_request_scope(kind):
    from test_standard_xpress_serving_cpu import drain

    from mlx2.runtime.apc_v2 import APCKey, APCv2

    model, _base, _draft, first = batch(kind)
    prompt = [1, 2, 3]
    uid = first.insert([prompt], max_tokens=[5])[0]
    output, final = drain(first)
    end = final[uid]
    apc = APCv2(max_size=2, layout_name=first.binding)
    key = APCKey(kind, revision=first.binding, cache_layout_fingerprint=first.binding)
    apc.store(key, end.all_tokens, end.prompt_cache, sidecar=end.cache_sidecar)
    continuation = prompt + output[uid]
    hit = apc.lookup(key, continuation)
    assert (
        hit.hit
        and hit.cached_tokens == len(continuation) - 1
        and hit.sidecar is not None
    )
    second = ExternalDraftBatchGenerator(
        model,
        draft_model=first.draft,
        binding=first.binding,
        num_draft=2,
        prefill_step_size=64,
        ready_drain="all",
    )
    try:
        uid2 = second.insert(
            [hit.remaining_tokens],
            max_tokens=[4],
            caches=[hit.cache],
            all_tokens=[continuation[: hit.cached_tokens]],
            cache_states=[hit.sidecar],
        )[0]
        resumed, final2 = drain(second)
        assert resumed[uid2] == reference(model, continuation, 4)
        assert second.scheduler_stats["paired_cache_resumes"] == 1
        final2[uid2].cache_sidecar.validate(
            second.binding, len(final2[uid2].all_tokens)
        )
        assert final2[uid2].speculative_receipt["continuation_pool"]["observed_used"]
    finally:
        hit.cache.close()
        first.close()
        second.close()
        apc.clear(release_memory=False)
    assert apc.apc_stats["cow"]["active_leases"] == 0


def test_pool_feedback_outbox_submits_only_after_all_requests_commit(monkeypatch):
    model, base, _draft, generator = batch("lilicorr")

    class Manager:
        def __init__(self):
            self.submissions, self.settled, self.nonce = [], 0, 0

        def capture_lattice(self, candidates, *args):
            result = []
            for _ in range(candidates.shape[0]):
                self.nonce += 1
                result.append({"nonce": self.nonce})
            return result

        def submit_verified(self, payload, teacher_tokens, **kw):
            self.submissions.append((payload, list(teacher_tokens), kw))

        def settle_round(self):
            self.settled += 1

        def receipt(self):
            return {"committed": len(self.submissions)}

    manager = Manager()
    base.feedback_manager = manager
    lanes = prefill(generator, [[1, 2, 3], [3, 2, 1]])
    original = model.forward_with_taps
    calls = 0

    def fail(tokens, *args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(tokens, *args, **kwargs)
        if calls == 2:
            raise RuntimeError("late feedback fence")
        assert not manager.submissions
        return result

    monkeypatch.setattr(model, "forward_with_taps", fail)
    with pytest.raises(RuntimeError, match="late feedback"):
        generator._round(lanes)
    assert manager.submissions == [] and manager.settled == 0
    monkeypatch.setattr(model, "forward_with_taps", original)
    generator._round(lanes)
    assert len(manager.submissions) == 2 and manager.settled == 1
    assert {r[2]["request_id"] for r in manager.submissions} == {
        l.proposal_request_id for l in lanes
    }
    assert all(
        l.ready[-1].speculative_receipt["lilicorr_feedback"]["committed"] == 2
        for l in lanes
    )


@pytest.mark.parametrize("reason", ["missing_width", "exploration"])
def test_pool_missing_physical_cost_binding_and_exploration_keep_full_depth(
    reason, monkeypatch
):
    costs = adaptive()
    if reason == "missing_width":
        costs["continuation_costs"].pop("15")
    model, _base, _draft, generator = batch(adaptive=costs)
    lane = prefill(generator, [[1, 2, 3]])[0]
    lane.continuation_rounds = 32 if reason == "exploration" else 1
    lane.continuation_coverage = {(w, p): (10, 1) for w in (1, 15) for p in (0, 1)}
    calls = []
    original = model.forward_with_taps

    def forward(tokens, *args, **kwargs):
        calls.append(tuple(tokens.shape))
        return original(tokens, *args, **kwargs)

    monkeypatch.setattr(model, "forward_with_taps", forward)
    generator._round([lane])
    assert calls == [(15, 3)]
    assert not lane.ready[-1].speculative_receipt["adaptive_verification"][
        "observed_used"
    ]


def test_postcommit_ranking_receipt_failure_keeps_tokens_state_and_learning(
    monkeypatch,
):
    model, _base, draft, generator = batch()
    lane = prefill(generator, [[1, 2, 3]])[0]
    original = draft.proposal_pool.receipt
    monkeypatch.setattr(
        draft.proposal_pool,
        "receipt",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("receipt outage")),
    )
    generator._round([lane])
    actual = [r.token for r in lane.ready]
    assert actual == reference(model, [1, 2, 3], len(actual))
    assert lane.generated == len(actual) and lane.continuation_rounds == 1
    assert (
        draft.proposal_pool.feedback_revision == 1 and not draft.proposal_pool._pending
    )
    assert generator.scheduler_stats["external_feedback_failures"] == 1
    assert generator.last_feedback_error == "receipt outage"
    assert "ranking" not in lane.ready[-1].speculative_receipt["continuation_pool"]
    monkeypatch.setattr(draft.proposal_pool, "receipt", original)
    lane.ready.clear()
    generator._round([lane])
    assert draft.proposal_pool.feedback_revision == 2

"""CPU-only integration: actual target shapes, exact greedy outputs, recovery."""

import copy

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)
from test_external_dflash2_cpu import committed_state, drain, generator, tiny

from mlx2.runtime.external_speculative import HostDraftRow, RoundDecision
from mlx2.runtime.sample_utils import LaneRNG


def _policy(**extra):
    return {"verification_costs": [1.0, 2.0, 8.0], "min_observations": 1, **extra}


def _prime(b):
    fit = b.acceptance_estimator
    fit.observe([0, 0], 0, rejected=True)
    fit.intercepts.fill(-4)
    fit.rounds = 1


def _run(enabled, monkeypatch, *, cohort_size=1):
    m, d = tiny()
    b = generator(
        m,
        d,
        adaptive_verification=_policy(cohort_size=cohort_size) if enabled else None,
    )
    if enabled:
        _prime(b)
    calls = []
    original = m.forward_with_taps

    def forward(tokens, *args, **kwargs):
        calls.append(tuple(tokens.shape))
        return original(tokens, *args, **kwargs)

    monkeypatch.setattr(m, "forward_with_taps", forward)
    b.insert(
        [[1, 2, 3]] * cohort_size,
        max_tokens=[6] * cohort_size,
        sampling_configs=[{"sampling_temp": 0}] * cohort_size,
    )
    for lane in b.lanes.values():
        b._prefill(lane)
    calls.clear()
    output, final = drain(b)
    return b, calls, output, final


@pytest.mark.parametrize("cohort_size", [1, 2])
def test_reduces_actual_target_forward_rows_and_matches_exact_greedy(
    monkeypatch, cohort_size
):
    off, off_calls, off_output, _ = _run(False, monkeypatch, cohort_size=cohort_size)
    on, on_calls, on_output, final = _run(True, monkeypatch, cohort_size=cohort_size)
    assert on_output == off_output
    assert max(shape[1] for shape in off_calls) == 3
    assert max(shape[1] for shape in on_calls) == 2
    assert on.scheduler_stats["external_adaptive_trimmed_target_rows"] > 0
    assert (
        sum(b * t for b, t in on_calls)
        == on.scheduler_stats["external_adaptive_target_rows"]
    )
    receipt = final[0].speculative_receipt["adaptive_verification"]
    assert receipt["implemented"] and receipt["selected"] and receipt["observed_used"]
    assert not receipt["qualified"]
    assert not any("adaptive" in key for key in off.scheduler_stats)
    for response in final.values():
        response.cache_sidecar.validate("test", len(response.all_tokens))


def test_disabled_preserves_rng_cache_outputs_and_receipts(monkeypatch):
    def run(value):
        m, d = tiny()
        b = generator(m, d, adaptive_verification=value)
        b.insert(
            [[1, 2]],
            max_tokens=[5],
            lane_rngs=[LaneRNG(44)],
            sampling_configs=[{"sampling_temp": 0.8}],
        )
        output, final = drain(b)
        return output, final[0].speculative_receipt, final[0].cache_sidecar

    output, receipt, state = run(None)
    other, other_receipt, other_state = run(False)
    assert output == other and receipt == other_receipt
    assert "adaptive_verification" not in receipt
    np.testing.assert_array_equal(
        np.asarray(state.rng_key), np.asarray(other_state.rng_key)
    )
    assert state.rng_draws == other_state.rng_draws


def test_failure_restores_estimator_rng_caches_and_counters(monkeypatch):
    m, d = tiny()
    b = generator(m, d, adaptive_verification=_policy())
    _prime(b)
    b.insert([[1, 2]], max_tokens=[6])
    lane = b.lanes[0]
    b._prefill(lane)
    before = copy.deepcopy(lane.__dict__)
    estimator = copy.deepcopy(b.acceptance_estimator.__dict__)
    original = b._adaptive_observe

    def fail(*args):
        original(*args)
        raise RuntimeError("after estimator update")

    monkeypatch.setattr(b, "_adaptive_observe", fail)
    with pytest.raises(RuntimeError, match="estimator update"):
        b._round([lane])
    assert b.scheduler_stats["external_adaptive_rounds"] == 0
    assert not lane.ready and lane.history == before["history"]
    assert lane.rng.snapshot() == before["rng"].snapshot()
    for key, value in estimator.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(b.acceptance_estimator.__dict__[key], value)
        else:
            assert b.acceptance_estimator.__dict__[key] == value
    for cache, old in zip(lane.cache, before["cache"]):
        assert cache.offset == old.offset
        for now, previous in zip(committed_state(cache), committed_state(old)):
            np.testing.assert_array_equal(np.asarray(now), np.asarray(previous))
    monkeypatch.setattr(b, "_adaptive_observe", original)
    output, _ = drain(b)
    assert len(output[0]) == 6


def test_stop_censors_undelivered_accepts():
    m, d = tiny()
    b = generator(m, d, adaptive_verification=_policy())
    blocks = [HostDraftRow([1, 2], [np.array([0.5, 0.5])] * 2)]
    b._adaptive_observe(blocks, [RoundDecision(2, [1], None)])
    np.testing.assert_array_equal(b.acceptance_estimator.observed_counts, [1, 0])
    assert b.acceptance_estimator.grad[0, 1] > 0


def test_required_histories_are_supplied_without_processors(monkeypatch):
    m, d = tiny()
    d.requires_processor_histories = True
    b = generator(m, d)
    b.insert([[1, 2]], max_tokens=[3])
    lane = b.lanes[0]
    b._prefill(lane)
    history = list(lane.history)
    seen = []
    original = d.draft_distributions

    def draft(*args, **kwargs):
        seen.append(kwargs["processor_histories"])
        return original(*args, **kwargs)

    monkeypatch.setattr(d, "draft_distributions", draft)
    b._round([lane])
    assert seen and seen[0][0] == history


def test_no_gain_keeps_full_forward_width(monkeypatch):
    m, d = tiny()
    b = generator(m, d, adaptive_verification=_policy(verification_costs=[1, 1, 1]))
    _prime(b)
    shapes = []
    original = m.forward_with_taps

    def forward(tokens, *args, **kwargs):
        shapes.append(tuple(tokens.shape))
        return original(tokens, *args, **kwargs)

    monkeypatch.setattr(m, "forward_with_taps", forward)
    b.insert([[1, 2]], max_tokens=[4])
    b._prefill(b.lanes[0])
    shapes.clear()
    b._round([b.lanes[0]])
    assert shapes == [(1, 3)]
    assert b.scheduler_stats["external_adaptive_trimmed_rounds"] == 0


def test_adaptive_paired_resume_matches_full_prompt():
    m, d = tiny()
    first = generator(m, d, adaptive_verification=_policy())
    _prime(first)
    first.insert([[1, 2]], max_tokens=[4])
    _, final = drain(first)
    end = final[0]
    resumed = generator(m, d, adaptive_verification=_policy())
    _prime(resumed)
    resumed.insert(
        [[end.token]],
        max_tokens=[4],
        caches=[end.prompt_cache],
        all_tokens=[end.all_tokens],
        cache_states=[end.cache_sidecar],
    )
    actual, tail = drain(resumed)
    reference = generator(m, d)
    reference.insert([end.all_tokens + [end.token]], max_tokens=[4])
    expected, _ = drain(reference)
    assert actual == expected
    assert resumed.scheduler_stats["paired_cache_resumes"] == 1
    tail[0].cache_sidecar.validate("test", len(tail[0].all_tokens))


def test_compact_q_laws_are_scored_without_vocabulary_expansion():
    from mlx2.runtime.external_speculative import CompactDraftRow

    m, d = tiny()
    b = generator(m, d, adaptive_verification=_policy())
    block = CompactDraftRow(
        [4, 9], np.array([[4, 8], [9, 3]]), np.array([[0.7, 0.3], [0.8, 0.2]]), 1
    )
    b._adaptive_observe([block], [RoundDecision(0, [2], None)])
    np.testing.assert_array_equal(b.acceptance_estimator.observed_counts, [1, 0])
    np.testing.assert_allclose(
        b.acceptance_estimator.feature_means, [np.log(7 / 3), np.log(4)]
    )


def _per_request_engine(monkeypatch, *, sampled=False):
    from test_standard_xpress_serving_cpu import generator as xpress_generator
    from test_standard_xpress_serving_cpu import tiny as tiny_xpress

    model, draft = tiny_xpress()
    policy = {
        "verification_costs": [0.5, 1, 1.4],
        "verification_costs_by_cohort": {"2": [1, 2, 10]},
        "mode": "per_request",
        "min_observations": 1,
    }
    engine = xpress_generator(model, draft, adaptive_verification=policy)
    engine.acceptance_estimator.observe([0, 0], 0, rejected=True)
    engine.acceptance_estimator.rounds = 1
    original = draft.draft_distributions

    def propose(*args, **kwargs):
        result = original(*args, **kwargs)
        # Controlled confidence exercises admission independently of target
        # quality, while every proposal still runs the real deterministic head.
        draft.adaptive_confidence_features = [
            [40] * len(result[0][0]),
            [-40] * len(result[0][1]),
        ]
        return result

    monkeypatch.setattr(draft, "draft_distributions", propose)
    engine.insert(
        [[1, 2, 3], [1, 2, 3]],
        max_tokens=[8, 8],
        lane_rngs=[LaneRNG(42), LaneRNG(43)],
        sampling_configs=[{"sampling_temp": 0.8 if sampled else 0}] * 2,
    )
    for lane in engine.lanes.values():
        engine._prefill(lane)
    return engine


def test_per_request_current_confidence_executes_distinct_physical_shapes(monkeypatch):
    from test_standard_xpress_serving_cpu import reference

    engine = _per_request_engine(monkeypatch)
    shapes = []
    original = engine.model.forward_with_taps

    def forward(tokens, *args, **kwargs):
        shapes.append(tuple(tokens.shape))
        return original(tokens, *args, **kwargs)

    monkeypatch.setattr(engine.model, "forward_with_taps", forward)
    engine._round(list(engine.lanes.values()))
    assert shapes == [(1, 3), (1, 2)]
    assert engine.scheduler_stats["external_adaptive_target_rows"] == 5
    assert engine.scheduler_stats["external_adaptive_verification_groups"] == 2
    assert engine.lanes[0].ready[0].speculative_receipt["round_proposed"] == 2
    assert engine.lanes[1].ready[0].speculative_receipt["round_proposed"] == 1
    for uid in (0, 1):
        ready = list(engine.lanes[uid].ready)
        expected = reference(engine.model, [1, 2, 3], len(ready))
        assert [r.token for r in ready] == expected
        assert (
            ready[0].speculative_receipt["adaptive_verification"]["policy"]
            == "per_request_grouped_cost_model"
        )


def test_later_group_failure_restores_already_committed_first_group(monkeypatch):
    engine = _per_request_engine(monkeypatch, sampled=True)
    lanes = list(engine.lanes.values())
    before = copy.deepcopy([lane.__dict__ for lane in lanes])
    fit = copy.deepcopy(engine.acceptance_estimator.__dict__)
    original = engine.model.forward_with_taps
    calls = [0]

    def forward(*args, **kwargs):
        calls[0] += 1
        result = original(*args, **kwargs)
        if calls[0] == 2:
            raise RuntimeError("second physical group failed")
        return result

    monkeypatch.setattr(engine.model, "forward_with_taps", forward)
    with pytest.raises(RuntimeError, match="second physical"):
        engine._round(lanes)
    assert calls[0] == 2 and engine.scheduler_stats["external_adaptive_rounds"] == 0
    assert engine.scheduler_stats["external_adaptive_verification_groups"] == 0
    for lane, old in zip(lanes, before):
        assert (
            lane.history == old["history"]
            and not lane.ready
            and lane.generated == old["generated"]
        )
        assert lane.rng.snapshot() == old["rng"].snapshot()
        for plane in ("cache", "draft_cache"):
            for now, saved in zip(getattr(lane, plane), old[plane]):
                assert now.offset == saved.offset
                for a, b in zip(committed_state(now), committed_state(saved)):
                    if a is not None:
                        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    for key, value in fit.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(
                engine.acceptance_estimator.__dict__[key], value
            )
        else:
            assert engine.acceptance_estimator.__dict__[key] == value
    monkeypatch.setattr(engine.model, "forward_with_taps", original)
    engine._round(lanes)
    assert all(lane.ready for lane in lanes)


def test_current_confidence_distinct_depths_preserve_sampled_target_law(monkeypatch):
    import mlx2.runtime.external_speculative as external
    from mlx2.runtime.speculative_sampling import RequestRNG, verify_proposals

    engine = _per_request_engine(monkeypatch, sampled=True)
    laws = []

    def check(tokens, proposals, targets, rng):
        assert all(
            q[token] == 1 and np.count_nonzero(q) == 1
            for token, q in zip(tokens, proposals)
        )
        laws.append((tokens, proposals, targets))
        return verify_proposals(tokens, proposals, targets, rng)

    monkeypatch.setattr(external, "verify_proposals", check)
    engine._round(list(engine.lanes.values()))
    assert [len(tokens) for tokens, _, _ in laws] == [2, 1]
    rng = RequestRNG(2026)
    counts = np.zeros(9)
    for iteration in range(10000):
        tokens, proposals, targets = laws[iteration % 2]
        result = verify_proposals(tokens, proposals, targets, rng)
        counts[result.emitted[0]] += 1
    np.testing.assert_allclose(counts / counts.sum(), laws[0][2][0], atol=0.016)
    np.testing.assert_allclose(laws[0][2][0], laws[1][2][0], atol=1e-7)


def test_stochastic_per_request_admission_precedes_draws_and_ignores_current_hook(
    monkeypatch,
):
    m, d = tiny()
    engine = generator(
        m,
        d,
        adaptive_verification={
            "verification_costs": [0.5, 1, 1.4],
            "verification_costs_by_cohort": {"2": [1, 2, 10]},
            "mode": "per_request",
            "min_observations": 1,
        },
    )
    engine.acceptance_estimator.observe([0, 0], 0, rejected=True)
    engine.acceptance_estimator.rounds = 1
    engine.insert(
        [[1, 2, 3], [1, 2, 3]],
        max_tokens=[8, 8],
        lane_rngs=[LaneRNG(42), LaneRNG(43)],
        sampling_configs=[{"sampling_temp": 0.8}] * 2,
    )
    for row, lane in enumerate(engine.lanes.values()):
        engine._prefill(lane)
        lane.adaptive_feature_means = [40 if row == 0 else -40] * 2
        lane.adaptive_feature_counts = [1, 1]
    seen = []
    original = d.draft_distributions
    d.adaptive_confidence_features = [[np.nan], [np.nan]]

    def propose(anchors, hidden, cache, length, rngs, *args, **kwargs):
        seen.append((length, [rng.draws for rng in rngs]))
        return original(anchors, hidden, cache, length, rngs, *args, **kwargs)

    monkeypatch.setattr(d, "draft_distributions", propose)
    engine._round(list(engine.lanes.values()))
    assert seen == [(2, [0]), (1, [0])]
    assert engine.scheduler_stats["external_adaptive_current_confidence_rounds"] == 0
    for lane in engine.lanes.values():
        assert (
            lane.ready[0].speculative_receipt["adaptive_verification"][
                "current_feature_source"
            ]
            == "lagged_q"
        )


def test_cold_heterogeneous_budget_does_not_claim_adaptive_trimming(monkeypatch):
    from test_standard_xpress_serving_cpu import generator as xpress_generator
    from test_standard_xpress_serving_cpu import tiny as tiny_xpress

    model, draft = tiny_xpress()
    engine = xpress_generator(
        model,
        draft,
        adaptive_verification={
            "verification_costs": [0.5, 1, 1.4],
            "verification_costs_by_cohort": {"2": [1, 2, 10]},
            "mode": "per_request",
            "min_observations": 32,
        },
    )
    engine.insert([[1, 2], [1, 2]], max_tokens=[1, 8])
    for lane in engine.lanes.values():
        engine._prefill(lane)
    engine._round(list(engine.lanes.values()))
    assert engine.scheduler_stats["external_adaptive_trimmed_rounds"] == 0
    assert engine.scheduler_stats["external_adaptive_trimmed_target_rows"] == 0
    for lane in engine.lanes.values():
        receipt = lane.ready[0].speculative_receipt["adaptive_verification"]
        assert not receipt["observed_used"] and receipt["round_proposal_depth"] == 0

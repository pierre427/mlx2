"""Independent CPU contracts for mixed current and lagged deterministic sources.

The real tiny head supplies all tensor math. Until source arbitration is added,
these first tests isolate its per-row confidence and law admission contract.
"""

import copy

import mlx.core as mx
import numpy as np
import pytest
from test_standard_xpress_serving_cpu import reference, tiny

from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
from mlx2.runtime.sample_utils import LaneRNG


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def engine(monkeypatch, hook="mixed", bad_law=False):
    target, draft = tiny()
    original = draft.draft_distributions

    def propose(*args, **kwargs):
        tokens, laws = original(*args, **kwargs)
        draft.adaptive_confidence_features = (
            None
            if hook == "absent"
            else [None] * len(tokens)
            if hook == "none"
            else [
                [-40] * len(row) if index == 0 else None
                for index, row in enumerate(tokens)
            ]
        )
        if bad_law:
            for row, probs in zip(tokens, laws):
                if row:
                    probs[0] = np.full(9, 1 / 9)
        return tokens, laws

    monkeypatch.setattr(draft, "draft_distributions", propose)
    e = ExternalDraftBatchGenerator(
        target,
        draft_model=draft,
        binding="mixed-hook-contract",
        num_draft=2,
        prefill_step_size=8,
        adaptive_verification={
            "verification_costs": [0.5, 1, 1.4],
            "verification_costs_by_cohort": {"2": [1, 2, 10]},
            "mode": "per_request",
            "min_observations": 1,
        },
    )
    e.insert(
        [[1, 2, 3], [2, 3, 4]],
        max_tokens=[12, 12],
        lane_rngs=[LaneRNG(41), LaneRNG(42)],
    )
    for lane in e.lanes.values():
        while lane.remaining:
            e._prefill(lane)
    return target, draft, e


def state_equal(actual, expected, *, replay=False):
    for a, b in zip(actual, expected):
        assert a.offset == b.offset
        for x, y in zip(a.state, b.state):
            if x is None:
                assert y is None
            else:
                if replay:
                    np.testing.assert_allclose(
                        np.asarray(x[..., : a.offset, :]),
                        np.asarray(y[..., : b.offset, :]),
                        atol=2e-6,
                        rtol=2e-6,
                    )
                else:
                    np.testing.assert_array_equal(
                        np.asarray(x[..., : a.offset, :]),
                        np.asarray(y[..., : b.offset, :]),
                    )


def test_mixed_live_and_lagged_rows_physically_group_after_real_observations(
    monkeypatch,
):
    target, _, e = engine(monkeypatch)
    lanes = list(e.lanes.values())
    e._round(lanes)
    # The no-proxy source recorded its actual point-mass q peak, not a proxy.
    assert lanes[1].adaptive_feature_means == [40, 40]
    assert lanes[1].adaptive_feature_counts == [1, 1]
    shapes = []
    original = target.forward_with_taps

    def forward(tokens, *a, **kw):
        shapes.append(tuple(tokens.shape))
        return original(tokens, *a, **kw)

    monkeypatch.setattr(target, "forward_with_taps", forward)
    e._round(lanes)
    assert shapes == [(1, 2), (1, 3)]
    assert lanes[0].adaptive_feature_source == "deterministic_refined_logit_proxy"
    assert lanes[1].adaptive_feature_source == "lagged_q"
    for lane, prompt in zip(lanes, [[1, 2, 3], [2, 3, 4]]):
        ready = [r.token for r in lane.ready]
        assert ready == reference(target, prompt, len(ready))
        fresh = target.make_cache()
        mx.eval(target(mx.array([lane.history]), cache=fresh))
        state_equal(lane.cache, fresh, replay=True)
        e._sidecar(lane).validate("mixed-hook-contract", len(lane.history))


@pytest.mark.parametrize("hook", ["absent", "none", "mixed"])
def test_deterministic_law_contract_rejected_without_current_proxy_and_restored(
    monkeypatch, hook
):
    target, _, e = engine(monkeypatch, hook=hook, bad_law=True)
    lanes = list(e.lanes.values())
    before = copy.deepcopy([l.__dict__ for l in lanes])
    fit = copy.deepcopy(e.acceptance_estimator)
    forwards = []
    original = target.forward_with_taps

    def forward(*a, **kw):
        forwards.append(True)
        return original(*a, **kw)

    monkeypatch.setattr(target, "forward_with_taps", forward)
    with pytest.raises(ValueError, match="deterministic point-mass"):
        e._round(lanes)
    assert not forwards and e.scheduler_stats["external_adaptive_target_rows"] == 0
    assert e.acceptance_estimator.rounds == fit.rounds
    np.testing.assert_array_equal(
        e.acceptance_estimator.observed_counts, fit.observed_counts
    )
    for lane, saved in zip(lanes, before):
        assert lane.history == saved["history"] and not lane.ready
        assert lane.rng.snapshot() == saved["rng"].snapshot()
        state_equal(lane.cache, saved["cache"])
        state_equal(lane.draft_cache, saved["draft_cache"])


def composed_engine(monkeypatch, *, sampled=False, pairwise_selection="host"):
    from mlx2.runtime.proposal_composition import ComposedDraftModel

    target, backend = tiny()
    original = backend.draft_distributions

    def confidence(*args, **kwargs):
        tokens, laws = original(*args, **kwargs)
        backend.adaptive_confidence_features = [[-40] * len(row) for row in tokens]
        return tokens, laws

    monkeypatch.setattr(backend, "draft_distributions", confidence)
    wrapper = ComposedDraftModel(
        backend,
        {
            "ngram_min": 2,
            "ngram_max": 2,
            "lookback": 128,
            "min_context_match": 0,
        },
    )
    policy = {
        "verification_costs": [0.5, 1, 1.4],
        "verification_costs_by_cohort": {"2": [1, 2, 10]},
        "mode": "per_request",
        "min_observations": 1,
    }
    prompts = [[1, 2, 3, 1, 2], [4, 5, 6]]

    def make():
        e = ExternalDraftBatchGenerator(
            target,
            draft_model=wrapper,
            binding="composed-tiny",
            num_draft=2,
            prefill_step_size=8,
            adaptive_verification=policy,
            pairwise_selection=pairwise_selection,
        )
        e.insert(
            prompts,
            max_tokens=[8, 8],
            lane_rngs=[LaneRNG(51), LaneRNG(52)],
            sampling_configs=[{"sampling_temp": 0.8 if sampled else 0}] * 2,
        )
        for lane in e.lanes.values():
            while lane.remaining:
                e._prefill(lane)
        return e

    prior = make()
    prior._round(list(prior.lanes.values()))
    assert (
        prior.lanes[0].proposal_composition_counts["prompt_lookup"]["verified_rounds"]
        == 1
    )
    fit = copy.deepcopy(prior.acceptance_estimator)
    qmeans = prior.lanes[0].adaptive_feature_means.copy()
    qcounts = prior.lanes[0].adaptive_feature_counts.copy()
    prior.close()
    e = make()
    e.acceptance_estimator = fit
    e.lanes[0].adaptive_feature_means = qmeans
    e.lanes[0].adaptive_feature_counts = qcounts
    return target, wrapper, e, prompts


def test_batched_pairwise_setting_preserves_mixed_row_source_arbitration(monkeypatch):
    target, wrapper, e, prompts = composed_engine(
        monkeypatch, pairwise_selection="batched"
    )
    lanes = list(e.lanes.values())
    # Reproduce a fresh served warmup.  The old dispatch used the backend's
    # compact API and then rejected this empty request-bound source vector.
    wrapper.last_proposal_sources = ()
    assert not wrapper.last_proposal_sources

    # The compact backend path is deliberately not used: it cannot perform
    # the wrapper's request-bound PLD/external row arbitration.
    pairwise_calls = e.scheduler_stats["external_pairwise_selection_groups"]
    attempts = dict(wrapper.composition_stats)
    e._round(lanes)
    assert e.scheduler_stats["external_pairwise_selection_groups"] == pairwise_calls
    assert wrapper.last_proposal_sources == ("prompt_lookup", "external")
    assert wrapper.composition_stats == {
        "prompt_lookup": attempts["prompt_lookup"] + 1,
        "native_mtp": attempts["native_mtp"],
        "external": attempts["external"] + 1,
    }
    assert [lane.proposal_composition_current_source for lane in lanes] == [
        "prompt_lookup",
        "external",
    ]
    for lane, prompt in zip(lanes, prompts):
        ready = [result.token for result in lane.ready]
        assert ready == reference(target, prompt, len(ready))


@pytest.mark.parametrize("sampled", [False, True])
def test_actual_pld_head_arbitration_adaptive_groups_have_request_private_source_receipts(
    monkeypatch, sampled
):
    from mlx2.runtime.speculative_sampling import softmax

    target, wrapper, e, prompts = composed_engine(monkeypatch, sampled=sampled)
    lanes = list(e.lanes.values())
    shapes = []
    original = target.forward_with_taps

    def forward(tokens, *a, **kw):
        shapes.append(tuple(tokens.shape))
        return original(tokens, *a, **kw)

    monkeypatch.setattr(target, "forward_with_taps", forward)
    # A prior unrelated request's wrapper diagnostics cannot claim this use.
    assert wrapper.composition_stats["prompt_lookup"] > 0
    assert not e._proposal_composition_receipt(lanes[0])["proposal_composition"][
        "observed_used"
    ]
    e._round(lanes)
    assert shapes == [(1, 3), (1, 2)]
    for index, lane in enumerate(lanes):
        receipt = lane.ready[0].speculative_receipt["proposal_composition"]
        source = "prompt_lookup" if index == 0 else "external"
        assert receipt["current_source"] == source and receipt["observed_used"] is (
            index == 0
        )
        assert receipt["committed_verification"][source]["verified_rounds"] == 1
        assert receipt["committed_verification"][source]["proposed_tokens"] == (
            2 if index == 0 else 1
        )
        assert lane.ready[0].speculative_receipt["adaptive_verification"][
            "current_feature_source"
        ] == ("lagged_q" if index == 0 else "deterministic_refined_logit_proxy")
        if sampled:
            expected = softmax(
                np.asarray(
                    target(mx.array([prompts[index]]), cache=target.make_cache())[0, -1]
                ),
                0.8,
            )
            np.testing.assert_allclose(
                np.exp(np.asarray(lane.ready[0].logprobs)),
                expected,
                atol=1e-6,
                rtol=2e-5,
            )
        else:
            assert [r.token for r in lane.ready] == reference(
                target, prompts[index], len(lane.ready)
            )
        fresh = target.make_cache()
        mx.eval(target(mx.array([lane.history]), cache=fresh))
        state_equal(lane.cache, fresh, replay=True)
        e._sidecar(lane).validate("composed-tiny", len(lane.history))


def test_composed_late_group_failure_rolls_back_request_source_counts_and_q_calibration(
    monkeypatch,
):
    target, wrapper, e, _ = composed_engine(monkeypatch, sampled=True)
    lanes = list(e.lanes.values())
    before = copy.deepcopy([l.__dict__ for l in lanes])
    fit = copy.deepcopy(e.acceptance_estimator)
    score_counts = copy.deepcopy(wrapper._score_counts)
    original = target.forward_with_taps
    calls = [0]

    def fail(*a, **kw):
        result = original(*a, **kw)
        calls[0] += 1
        if calls[0] == 2:
            raise RuntimeError("composed second group")
        return result

    monkeypatch.setattr(target, "forward_with_taps", fail)
    with pytest.raises(RuntimeError, match="composed second"):
        e._round(lanes)
    assert calls[0] == 2 and not any(
        k.startswith("external_composed_") for k in e.scheduler_stats
    )
    for lane, saved in zip(lanes, before):
        assert not hasattr(lane, "proposal_composition_counts")
        assert not e._proposal_composition_receipt(lane)["proposal_composition"][
            "observed_used"
        ]
        assert (
            lane.history == saved["history"]
            and lane.rng.snapshot() == saved["rng"].snapshot()
        )
        state_equal(lane.cache, saved["cache"])
        state_equal(lane.draft_cache, saved["draft_cache"])
    np.testing.assert_array_equal(
        e.acceptance_estimator.observed_counts, fit.observed_counts
    )
    assert wrapper._score_counts == score_counts
    monkeypatch.setattr(target, "forward_with_taps", original)
    e._round(lanes)
    assert wrapper._score_counts != score_counts
    assert lanes[0].proposal_composition_counts["prompt_lookup"]["verified_rounds"] == 1


def test_invalid_composition_source_is_rejected_before_target_execution(monkeypatch):
    target, wrapper, e, _ = composed_engine(monkeypatch)
    original = wrapper.draft_distributions

    def invalid(*a, **kw):
        result = original(*a, **kw)
        wrapper.last_proposal_sources = ("unregistered",) * len(result[0])
        return result

    monkeypatch.setattr(wrapper, "draft_distributions", invalid)
    called = []
    forward = target.forward_with_taps

    def record(*a, **kw):
        called.append(True)
        return forward(*a, **kw)

    monkeypatch.setattr(target, "forward_with_taps", record)
    with pytest.raises(ValueError, match="composed proposal source"):
        e._round(list(e.lanes.values()))
    assert not called


def native_composed_engine(monkeypatch, kind, sampled):
    from test_proposal_composition_cpu import _native_qsa_pair

    from mlx2.adapters.proposal_sources import native_mtp_source
    from mlx2.runtime.proposal_composition import ComposedDraftModel

    target, backend = _native_qsa_pair(kind)
    original = backend.draft_distributions

    def confidence(*a, **kw):
        tokens, laws = original(*a, **kw)
        backend.adaptive_confidence_features = [[-40] * len(row) for row in tokens]
        return tokens, laws

    monkeypatch.setattr(backend, "draft_distributions", confidence)
    wrapper = ComposedDraftModel(
        backend,
        {"prompt_lookup": False, "native_mtp": True, "mtp_max_history": 3},
        native_mtp_source=native_mtp_source(target),
    )
    policy = {
        "verification_costs": [0.5, 1, 1.4],
        "verification_costs_by_cohort": {"2": [1, 2, 10]},
        "mode": "per_request",
        "min_observations": 1,
    }
    prompts = [[1, 2, 3], [4, 5, 6, 7, 8, 9]]

    def make():
        e = ExternalDraftBatchGenerator(
            target,
            draft_model=wrapper,
            binding="native-composed-tiny",
            num_draft=2,
            prefill_step_size=8,
            adaptive_verification=policy,
        )
        e.insert(
            prompts,
            max_tokens=[8, 8],
            lane_rngs=[LaneRNG(71), LaneRNG(72)],
            sampling_configs=[{"sampling_temp": 0.8 if sampled else 0}] * 2,
        )
        for lane in e.lanes.values():
            while lane.remaining:
                e._prefill(lane)
        return e

    prior = make()
    prior._round(list(prior.lanes.values()))
    assert (
        prior.lanes[0].proposal_composition_counts["native_mtp"]["verified_rounds"] == 1
    )
    fit = copy.deepcopy(prior.acceptance_estimator)
    qmeans = prior.lanes[0].adaptive_feature_means.copy()
    qcounts = prior.lanes[0].adaptive_feature_counts.copy()
    prior.close()
    e = make()
    e.acceptance_estimator = fit
    e.lanes[0].adaptive_feature_means = qmeans
    e.lanes[0].adaptive_feature_counts = qcounts
    return target, e, prompts


@pytest.mark.parametrize("kind", ["xpress", "lilicorr"])
@pytest.mark.parametrize("sampled", [False, True])
def test_actual_native_mtp_external_adaptive_groups_preserve_hybrid_target_law_and_source(
    monkeypatch, kind, sampled
):
    from test_external_adaptive_hybrid_cpu import assert_state

    from mlx2.runtime.speculative_sampling import softmax

    target, e, prompts = native_composed_engine(monkeypatch, kind, sampled)
    lanes = list(e.lanes.values())
    shapes = []
    original = target.forward_with_taps

    def forward(tokens, *a, **kw):
        shapes.append(tuple(tokens.shape))
        return original(tokens, *a, **kw)

    monkeypatch.setattr(target, "forward_with_taps", forward)
    e._round(lanes)
    assert shapes == [(1, 3), (1, 2)]
    for index, lane in enumerate(lanes):
        receipt = lane.ready[0].speculative_receipt["proposal_composition"]
        assert receipt["current_source"] == ("native_mtp" if index == 0 else "external")
        assert receipt["observed_used"] is (index == 0)
        if sampled:
            logits = target(mx.array([prompts[index]]), cache=target.make_cache())[
                0, -1
            ]
            np.testing.assert_allclose(
                np.exp(np.asarray(lane.ready[0].logprobs)),
                softmax(np.asarray(logits), 0.8),
                atol=2e-6,
                rtol=2e-5,
            )
        else:
            assert [r.token for r in lane.ready] == reference(
                target, prompts[index], len(lane.ready)
            )
        fresh = target.make_cache()
        mx.eval(target(mx.array([lane.history]), cache=fresh))
        assert_state(lane.cache, fresh)
        e._sidecar(lane).validate("native-composed-tiny", len(lane.history))


def test_composition_cannot_silently_delegate_selected_tree_route(monkeypatch):
    from mlx2.runtime.proposal_composition import ComposedDraftModel

    target, backend = tiny()
    wrapper = ComposedDraftModel(backend, {"prompt_lookup": True})
    monkeypatch.setenv("MLX2_DFLASH_TOPOLOGY", "tree15")
    with pytest.raises(ValueError, match="composition requires chain"):
        ExternalDraftBatchGenerator(
            target, draft_model=wrapper, binding="closed-tree", num_draft=2
        )

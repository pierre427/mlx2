"""Physical prefix sharing retains complete semantic candidates; CPU only."""

import copy

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_map
from test_continuation_providers_cpu import wrapped
from test_external_continuation_pool_cpu import prefill
from test_standard_xpress_serving_cpu import tiny

from mlx2.adapters.proposal_path_sources import SourceContinuation
from mlx2.runtime.acceptance_estimator import AdaptiveVerificationPolicy
from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
from mlx2.runtime.speculative_sampling import softmax


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    from mlx2.runtime import proposal_pool

    monkeypatch.setattr(
        proposal_pool,
        "_SHARED_RANKING_REGISTRY",
        proposal_pool.ProposalRankingRegistry(),
    )
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


def policy():
    # Synthetic CPU mechanism costs, never Metal performance evidence.
    return {
        "mode": "per_request",
        "min_observations": 1,
        "continuation_costs": {"1": [1, 100, 100], "2": [1, 1, 500], "4": [1, 50, 100]},
    }


def oracle(model, prompt, prefix=()):
    cache = model.make_cache()
    if len(prompt) > 1:
        mx.eval(model(mx.array([prompt[:-1]]), cache=cache))
    logits = model(mx.array([[prompt[-1]]]), cache=cache)
    for token in prefix:
        logits = model(mx.array([[token]]), cache=cache)
    return logits[0, -1]


def setup(
    monkeypatch, *, dtype=mx.bfloat16, selected=True, prompts=None, sampling=None
):
    model, base = tiny()
    model.update(tree_map(lambda value: value.astype(dtype), model.parameters()))
    base.update(tree_map(lambda value: value.astype(dtype), base.parameters()))
    model.configure_target_verify_row_exact(selected)
    prompts = prompts or [[1, 2, 3]]
    paths_by_context = {}
    for prompt in prompts:
        good = int(mx.argmax(oracle(model, prompt)).item())
        bad = (good + 1) % 9
        paths_by_context[tuple(prompt)] = [(bad, 0), (bad, 1), (good, 0), (good, 1)]
    draft = wrapped(model, base, {"sources": ["external", "prompt_lookup"], "limit": 4})

    def provider(context, limit):
        return [
            SourceContinuation(path, (None, None), float(4 - index), "synthetic-order")
            for index, path in enumerate(
                paths_by_context[(*context.history, context.anchor)]
            )
        ][:limit]

    draft.providers = {"external": provider, "prompt_lookup": provider}
    engine = ExternalDraftBatchGenerator(
        model,
        draft_model=draft,
        binding="prefix-dedup-cpu",
        num_draft=2,
        prefill_step_size=64,
        adaptive_verification=policy(),
        ready_drain="all",
    )
    lanes = (
        prefill(engine, prompts, sampling_configs=sampling)
        if sampling
        else prefill(engine, prompts)
    )
    for lane in lanes:
        lane.continuation_rounds = 1
        lane.continuation_coverage = {
            (w, p): (100, 0) for w in range(1, 5) for p in range(2)
        }
    calls = []
    original = model.forward_with_taps

    def capture(tokens, *args, **kwargs):
        calls.append(tuple(tokens.shape))
        return original(tokens, *args, **kwargs)

    monkeypatch.setattr(model, "forward_with_taps", capture)
    return model, base, draft, engine, lanes, calls, original


def test_stable_mapping_and_physical_cost_binding():
    paths = ((1,), (1,), (3,), (3,))
    assert ExternalDraftBatchGenerator._unique_continuation_paths(
        paths, enabled=True
    ) == (((1,), (3,)), (0, 2), (0, 0, 1, 1))
    assert ExternalDraftBatchGenerator._unique_continuation_paths(
        paths, enabled=False
    ) == (paths, (0, 1, 2, 3), (0, 1, 2, 3))
    p = AdaptiveVerificationPolicy.from_value(policy(), 2)
    coverage = {(w, i): (100, 0) for w in range(1, 5) for i in range(2)}
    mapping = {
        (w, d): (1 if w <= 2 else 2) if d == 1 else w
        for w in range(1, 5)
        for d in (1, 2)
    }
    assert p.choose_continuation_shape(4, 2, coverage, 1, physical_widths=mapping) == (
        4,
        1,
    )
    # A semantic-W table cannot stand in for an unmeasured physical width.
    mapping[4, 2] = 3
    assert p.choose_continuation_shape(4, 2, coverage, 1, physical_widths=mapping) == (
        4,
        2,
    )
    with pytest.raises(ValueError, match="mapping"):
        p.choose_continuation_shape(4, 2, coverage, 1, physical_widths={(4, 1): True})


def test_real_bf16_target_dedups_after_depth_choice_and_keeps_semantic_credit(
    monkeypatch,
):
    model, _base, draft, engine, lanes, calls, _original = setup(monkeypatch)
    lane = lanes[0]
    expected = int(mx.argmax(oracle(model, [1, 2, 3])).item())
    engine._round(lanes)
    assert calls == [(2, 2)]
    assert lane.ready[0].token == expected
    receipt = lane.ready[-1].speculative_receipt["continuation_pool"]
    assert (
        receipt["ranked_complete_sequences"]
        == receipt["admitted_complete_sequences"]
        == 4
    )
    assert receipt["physical_width"] == 2 and receipt["prefix_dedup_enabled"]
    assert receipt["physical_to_semantic"] == [0, 2]
    assert receipt["semantic_to_physical"] == [0, 0, 1, 1]
    assert receipt["selected_path"] == 2 and receipt["selected_physical_path"] == 1
    assert receipt["deduplicated_sequences"] == 2
    assert (
        draft.proposal_pool.feedback_revision == 1 and not draft.proposal_pool._pending
    )
    # Both source contributors survive sharing and receive reached-prefix labels.
    assert draft.proposal_pool.observation_counts(
        "external"
    ) == draft.proposal_pool.observation_counts("prompt_lookup")
    assert sum(draft.proposal_pool.observation_counts("external")[0]) > 0
    assert lane.rng.draws == len(lane.ready)
    fresh = model.make_cache()
    mx.eval(model(mx.array([[1, 2]]), cache=fresh))
    for token in [3, *[r.token for r in lane.ready][:-1]]:
        mx.eval(model(mx.array([[token]]), cache=fresh))
    for actual, expected_cache in zip(lane.cache, fresh):
        assert actual.offset == expected_cache.offset
        for a, b in zip(actual.keys_and_values(), expected_cache.keys_and_values()):
            np.testing.assert_array_equal(
                np.asarray(a.astype(mx.float32)), np.asarray(b.astype(mx.float32))
            )
    engine._sidecar(lane).validate(engine.binding, len(lane.history))
    ranking = draft.proposal_pool.receipt(lane.session_scope_hash)["ranking_registry"]
    assert (
        ranking["global_counts"]
        and len(ranking["model_counts"]) == len(ranking["session_counts"]) == 2
    )


def test_stop_on_lower_ranked_proposal_credits_that_semantic_path(monkeypatch):
    model, _base, _draft, engine, lanes, calls, _original = setup(monkeypatch)
    lane = lanes[0]
    expected = int(mx.argmax(oracle(model, [1, 2, 3])).item())
    # Only semantic paths 2/3 propose ``expected``; it is also a stop token,
    # so the walk ends on a draw only the lower-ranked physical path holds.
    engine.stops = {expected}
    engine._round(lanes)
    assert calls == [(2, 2)]
    assert [r.token for r in lane.ready] == [expected]
    assert lane.ready[-1].finish_reason == "stop"
    receipt = lane.ready[-1].speculative_receipt["continuation_pool"]
    assert receipt["physical_to_semantic"] == [0, 2]
    assert receipt["selected_path"] == 2 and receipt["selected_physical_path"] == 1


@pytest.mark.parametrize(
    "dtype,selected,capability",
    [
        (mx.bfloat16, True, True),
        (mx.float32, True, False),
        (mx.float16, True, False),
        (mx.bfloat16, False, False),
    ],
)
def test_capability_scope(dtype, selected, capability):
    model, _ = tiny()
    model.update(tree_map(lambda value: value.astype(dtype), model.parameters()))
    model.configure_target_verify_row_exact(selected)
    assert model.supports_contextual_prefix_equivalence is capability


@pytest.mark.parametrize("guard", ["off", "float32", "steering"])
def test_unproven_geometry_and_steering_keep_literal_rows(guard, monkeypatch):
    model, _base, _draft, engine, lanes, calls, _ = setup(
        monkeypatch,
        dtype=mx.float32 if guard == "float32" else mx.bfloat16,
        selected=guard != "off",
    )
    if guard == "steering":
        model.model.residual_taps = object()
    # Force only this test's admission to isolate the geometry capability gate.
    monkeypatch.setattr(
        AdaptiveVerificationPolicy, "choose_continuation_shape", lambda *a, **kw: (4, 1)
    )
    engine._round(lanes)
    assert calls == [(4, 2)]
    receipt = lanes[0].ready[-1].speculative_receipt["continuation_pool"]
    assert (
        not receipt["prefix_dedup_enabled"] and receipt["deduplicated_sequences"] == 0
    )
    assert receipt["physical_to_semantic"] == [0, 1, 2, 3]


def test_mixed_sampling_uses_one_draw_per_reached_prefix_and_actual_law(monkeypatch):
    prompts = [[1, 2, 3], [3, 2, 1]]
    model, _base, _draft, engine, lanes, calls, _ = setup(
        monkeypatch,
        prompts=prompts,
        sampling=[{"sampling_temp": 0}, {"sampling_temp": 0.8}],
    )
    original = engine._target_law
    seen = []

    def law(lane, logits, history, *args):
        value = original(lane, logits, history, *args)
        prompt = prompts[lanes.index(lane)]
        expected = oracle(model, prompt, history[len(prompt) :]).astype(mx.float32)
        if lane.sampling.get("sampling_temp"):
            np.testing.assert_allclose(
                value, softmax(np.asarray(expected), 0.8), atol=1e-7, rtol=1e-6
            )
        else:
            assert np.argmax(value) == int(mx.argmax(expected).item())
        seen.append(lane.uid)
        return value

    monkeypatch.setattr(engine, "_target_law", law)
    engine._round(lanes)
    assert calls == [(2, 2), (2, 2)]
    for lane in lanes:
        assert lane.rng.draws == len(lane.ready) == seen.count(lane.uid)


def test_later_cohort_failure_restores_maps_caches_rng_and_all_feedback(monkeypatch):
    model, _base, draft, engine, lanes, calls, original = setup(
        monkeypatch,
        prompts=[[1, 2, 3], [3, 2, 1]],
        sampling=[{"sampling_temp": 0.8}, {"sampling_temp": 0.8}],
    )
    snapshots = [
        (
            list(l.history),
            l.anchor,
            l.rng.snapshot(),
            copy.deepcopy(l.cache),
            copy.deepcopy(l.draft_cache),
            copy.deepcopy(l.continuation_coverage),
        )
        for l in lanes
    ]

    class Manager:
        def __init__(self):
            self.submissions, self.settled = [], 0

        def submit_verified(self, payload, tokens, **kwargs):
            self.submissions.append((payload, tuple(tokens), kwargs))

        def settle_round(self):
            self.settled += 1

        def receipt(self):
            return {"committed": len(self.submissions)}

    manager = Manager()
    _base.feedback_manager = manager
    base_forward = _base.draft_distributions

    def publish(*args, **kwargs):
        result = base_forward(*args, **kwargs)
        _base.draft_feedback_payloads = [{"row": i} for i in range(len(args[0]))]
        return result

    monkeypatch.setattr(_base, "draft_distributions", publish)
    count = 0

    def fail(tokens, *a, **kw):
        nonlocal count
        count += 1
        result = original(tokens, *a, **kw)
        if count == 2:
            raise RuntimeError("dedup late cohort failure")
        return result

    monkeypatch.setattr(model, "forward_with_taps", fail)
    with pytest.raises(RuntimeError, match="late cohort"):
        engine._round(lanes)
    assert (
        draft.proposal_pool.feedback_revision == 0 and not draft.proposal_pool._pending
    )
    assert not manager.submissions and manager.settled == 0
    assert not draft.proposal_pool.receipt(lanes[0].session_scope_hash)[
        "ranking_registry"
    ]["global_counts"]
    assert (
        engine.scheduler_stats.get("external_continuation_deduplicated_sequences", 0)
        == 0
    )
    for lane, (history, anchor, rng, cache, dc, coverage) in zip(lanes, snapshots):
        assert (
            lane.history == history
            and lane.anchor == anchor
            and lane.rng.snapshot() == rng
        )
        assert lane.continuation_coverage == coverage and not lane.ready
        assert not hasattr(lane, "continuation_physical_to_semantic")
        for actual, before in zip(lane.cache, cache):
            assert actual.offset == before.offset
            for a, b in zip(actual.state, before.state):
                np.testing.assert_array_equal(
                    np.asarray(a.astype(mx.float32)), np.asarray(b.astype(mx.float32))
                )
        assert [c.offset for c in lane.draft_cache] == [c.offset for c in dc]
    monkeypatch.setattr(
        model,
        "forward_with_taps",
        lambda t, *a, **kw: (calls.append(tuple(t.shape)), original(t, *a, **kw))[1],
    )
    engine._round(lanes)
    assert calls == [(2, 2), (2, 2)]
    assert (
        draft.proposal_pool.feedback_revision == 2 and not draft.proposal_pool._pending
    )
    assert len(manager.submissions) == 2 and manager.settled == 1

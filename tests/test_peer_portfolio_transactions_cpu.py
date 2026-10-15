"""Portfolio admission remains bounded and rolls back with target execution."""

import copy

import pytest
from test_continuation_providers_cpu import context, pair, wrapped
from test_external_continuation_pool_cpu import batch, prefill

from mlx2.adapters.proposal_path_sources import SourceContinuation
from mlx2.runtime.drafter_portfolio import DrafterPortfolio, PortfolioProvider
from mlx2.runtime.external_speculative import DraftUnavailable
from mlx2.runtime.proposal_providers import continuation_context_revision


@pytest.fixture(autouse=True)
def isolated_ranking_registry(monkeypatch):
    from mlx2.runtime import proposal_pool

    # These cases start with a neutral prior so the first provider consumes
    # the declared budget. Earlier suite feedback may correctly park it.
    # Restore the process registry after each test; never clear another owner.
    monkeypatch.setattr(
        proposal_pool, "_SHARED_RANKING_REGISTRY", proposal_pool.ProposalRankingRegistry()
    )


def invoke(model, draft, wrapper):
    ctx = context(model, draft)
    wrapper.draft_distributions(
        [ctx.anchor], ctx.pending_features, ctx.draft_cache, ctx.depth, [None], [0],
        processor_histories=[ctx.history],
        proposal_contexts=[{
            "request_id": "request", "round_id": 0,
            "context_revision": continuation_context_revision(
                wrapper.session.session_revision, ctx.history, ctx.anchor
            ),
        }],
    )
    return wrapper.last_continuation_admissions[0]


def complete_path(_context, _limit):
    return [SourceContinuation((1, 2), (None, None), 0.0, "test_private_proxy")]


@pytest.mark.parametrize("budget", ["lane_slots", "tile_rows"])
def test_runtime_reprobe_cannot_admit_an_over_budget_resident(budget):
    model, base = pair("xpress")
    portfolio = DrafterPortfolio([
        PortfolioProvider(
            "expensive", "xpress", "b" * 64, base, lane_cost=2, tile_rows=2
        )
    ])
    wrapper = wrapped(model, portfolio, {
        "sources": ["expensive"], "admission": {budget: 1},
    })
    called = []

    def provider(ctx, limit):
        called.append(True)
        return complete_path(ctx, limit)

    wrapper.providers["expensive"] = provider
    with pytest.raises(DraftUnavailable):
        invoke(model, portfolio, wrapper)
    assert not called
    assert not wrapper.proposal_pool._pending


@pytest.mark.parametrize("admission", [
    {"pld_fallback": False},
    {"max_active_providers": 1},
    {"lane_slots": 1},
    {"tile_rows": 1},
])
def test_runtime_pld_fallback_honors_policy_and_remaining_budget(admission):
    model, base = pair("xpress")
    wrapper = wrapped(model, base, {
        "sources": ["external", "prompt_lookup"], "admission": admission,
    })
    wrapper.providers["external"] = lambda _ctx, _limit: []
    called = []

    def provider(ctx, limit):
        called.append(True)
        return complete_path(ctx, limit)

    wrapper.providers["prompt_lookup"] = provider
    with pytest.raises(DraftUnavailable):
        invoke(model, base, wrapper)
    assert not called
    assert not wrapper.proposal_pool._pending


def test_permitted_runtime_fallback_charges_actual_attempted_work():
    model, base = pair("xpress")
    wrapper = wrapped(model, base, {"sources": ["external", "prompt_lookup"]})
    wrapper.providers["external"] = lambda _ctx, _limit: []
    wrapper.providers["prompt_lookup"] = complete_path
    receipt = invoke(model, base, wrapper)
    assert receipt["admitted"] == ("external", "prompt_lookup")
    assert receipt["runtime_fallback"] == "prompt_lookup"
    assert receipt["lane_slots_used"] == receipt["tile_rows_used"] == 2
    assert receipt["fallback_only"]
    wrapper.proposal_pool.discard(wrapper.last_continuation_selections[0])


@pytest.mark.parametrize("failure", ["provider", "later_target"])
def test_failed_executor_round_restores_parking_rounds_and_evicted_owners(
    monkeypatch, failure
):
    model, _base, draft, generator = batch()
    lanes = prefill(generator, [[1, 2, 3], [3, 2, 1]])
    planner = draft.admission_planner
    planner.max_requests = 2
    planner._rounds["existing"] = 7
    planner._parked[("existing", "external")] = 20
    before = copy.deepcopy((planner._rounds, planner._parked))
    lane_before = [(tuple(l.history), l.rng.snapshot()) for l in lanes]
    predict = draft.proposal_pool.source_prediction

    def weak_prediction(*args):
        return {**predict(*args), "observations": 100, "expected_accepted_tokens": 0.0}

    monkeypatch.setattr(draft.proposal_pool, "source_prediction", weak_prediction)
    with monkeypatch.context() as fault:
        if failure == "provider":
            def fail_provider(_context, _limit):
                raise RuntimeError("injected provider failure")

            fault.setitem(draft.providers, "external", fail_provider)
        else:
            forward = model.forward_with_taps
            calls = 0

            def fail_target(*args, **kwargs):
                nonlocal calls
                calls += 1
                result = forward(*args, **kwargs)
                if calls == 2:
                    raise RuntimeError("injected later target failure")
                return result

            fault.setattr(model, "forward_with_taps", fail_target)
        with pytest.raises(RuntimeError, match="injected"):
            generator._round(lanes)
    assert (planner._rounds, planner._parked) == before
    assert [(tuple(l.history), l.rng.snapshot()) for l in lanes] == lane_before
    assert all(l.generated == 0 and not l.ready for l in lanes)
    assert draft.proposal_pool.feedback_revision == 0
    assert not draft.proposal_pool._pending
    generator._round(lanes)
    assert draft.proposal_pool.feedback_revision == 2
    assert [row["round"] for row in draft.last_continuation_admissions] == [0, 0]
    assert all(l.generated > 0 for l in lanes)

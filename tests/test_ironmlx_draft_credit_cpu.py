"""CPU-only timing traces; checkpoint none, base mlx2 9fc9e848d.

No MLX import, tensors, model loading or accelerator invocation.
"""
import math

import pytest

from mlx2.runtime.cost_per_commit_handoff import (
    CostPerCommitHandoffSettings,
    CostPerCommitLaneState,
)


def state(**changes):
    settings = CostPerCommitHandoffSettings(
        enabled=True, reversible=True, ordinary_samples=1, speculative_samples=1,
        loss_decisions=1, window=2, settle_rounds=0, reprobe_tokens=2,
        reprobe_max_tokens=4, deferred_context_cap=10000,
        **{"reprobe_credit_share": .02, **changes},
    )
    return CostPerCommitLaneState(settings, lane_uid=7, owner_scope=("tenant", "target@digest"))


def observe(s, kind, elapsed=100, committed=1, width=1, context=None, **identity):
    return s.observe(
        kind=kind, elapsed_ns=elapsed, committed_tokens=committed,
        execution_width=width, context_key=context,
        lane_uid=identity.get("lane_uid", 7),
        owner_scope=identity.get("owner_scope", ("tenant", "target@digest")),
    )


def park(s):
    assert observe(s, "ordinary") is None
    assert observe(s, "speculative", 200)["event"] == "park"
    assert s.speculation_paused


def credit(s):
    return s.receipt()["reversible"]["exploration_credit"]


def test_credit_requires_explicit_reversible_selection_and_keeps_defaults_identical():
    ordinary = CostPerCommitHandoffSettings(enabled=True, reversible=True)
    assert "reprobe_credit_share" not in ordinary.as_dict()
    lane = CostPerCommitLaneState(ordinary, lane_uid=1, owner_scope="tenant")
    assert "exploration_credit" not in lane.receipt()["reversible"]
    assert CostPerCommitHandoffSettings.from_value(ordinary.as_dict()) == ordinary
    enabled = state().settings
    assert CostPerCommitHandoffSettings.from_value(enabled.as_dict()) == enabled
    with pytest.raises(ValueError, match="requires reversible"):
        CostPerCommitHandoffSettings(enabled=True, reprobe_credit_share=.02)
    with pytest.raises(ValueError, match="must cover ordinary_samples"):
        CostPerCommitHandoffSettings(
            enabled=True, reversible=True, reprobe_credit_share=.02,
            ordinary_samples=4, reprobe_credit_max_rounds=2,
        )


@pytest.mark.parametrize("share", [True, math.nan, math.inf, -1, 1.01, "0.02"])
def test_invalid_credit_settings_refuse(share):
    with pytest.raises(ValueError, match="reprobe_credit_share"):
        state(reprobe_credit_share=share)


def test_round_bound_forces_recovery_when_time_credit_is_insufficient():
    s = state(reprobe_credit_max_rounds=5)
    park(s)
    for _ in range(4):
        assert observe(s, "ordinary") is None
    event = observe(s, "ordinary")
    assert event["reason"] == "ordinary_round_bound"
    assert credit(s)["earned_ns"] == 10
    assert credit(s)["forced_probes"] == 1
    assert credit(s)["credit_probes"] == 0


def test_credit_funds_a_probe_and_catchup_overspend_remains_debt():
    s = state(reprobe_credit_share=1)
    park(s)
    # Reserve = two complete 200-ns rounds; each ordinary round earns 100.
    for _ in range(3):
        assert observe(s, "ordinary") is None
    assert observe(s, "ordinary")["reason"] == "time_credit"
    assert credit(s)["balance_ns"] == 400
    # 900 ns above ordinary is charged although this catch-up is excluded
    # from the steady-state decision. Cheap follow-up can still re-engage.
    assert observe(s, "speculative", 1000) is None
    assert credit(s)["balance_ns"] == -500
    assert observe(s, "speculative", 20)["event"] == "reengage"
    assert credit(s)["credit_probe_excess_ns"] == 900
    assert not s.speculation_paused


def test_loss_to_win_transition_recovers_and_sustained_loss_has_no_probe_burst():
    s = state(reprobe_credit_max_rounds=4)
    park(s)
    for cycle in range(3):
        for _ in range(3):
            assert observe(s, "ordinary") is None
        assert observe(s, "ordinary")["event"] == "reprobe"
        assert observe(s, "speculative", 500) is None  # catch-up
        outcome = observe(s, "speculative", 200 if cycle < 2 else 40)
        assert outcome["event"] == ("repark" if cycle < 2 else "reengage")
    assert credit(s)["forced_probes"] == 3
    assert credit(s)["credit_probes"] == 0
    assert credit(s)["balance_ns"] < 0


def test_mixed_cohort_progress_never_fabricates_credit_or_ordinary_rounds():
    s = state(reprobe_credit_max_rounds=2)
    park(s)
    for _ in range(20):
        s.observe_parked_progress(
            committed_tokens=1, lane_uid=7, owner_scope=("tenant", "target@digest")
        )
    assert credit(s)["earned_ns"] == 0
    assert credit(s)["ordinary_rounds"] == 0
    assert s.speculation_paused


def test_deferred_context_bound_forces_probe_even_without_clean_timing():
    s = state()
    park(s)
    event = s.observe_parked_progress(
        committed_tokens=10000, lane_uid=7, owner_scope=("tenant", "target@digest")
    )
    assert event["reason"] == "deferred_context_bound"
    assert credit(s)["earned_ns"] == 0
    assert credit(s)["forced_probes"] == 1


@pytest.mark.parametrize("width,context", [(4, None), (1, "32k")])
def test_width_and_explicit_context_regimes_cannot_reuse_credit_or_baselines(width, context):
    s = state()
    park(s)
    observe(s, "ordinary")
    assert credit(s)["balance_ns"] == 2
    observe(s, "ordinary", 300, width=width, context=context)
    assert credit(s)["regime_resets"] == 1
    assert credit(s)["balance_ns"] == 6
    assert s.receipt()["ordinary"]["0"]["samples"] == 1
    assert s.ordinary_baseline() == 300


def test_foreign_invalid_and_cancelled_observations_never_earn():
    s = state()
    park(s)
    with pytest.raises(ValueError, match="owner identity mismatch"):
        observe(s, "ordinary", owner_scope=("other-tenant", "target@digest"))
    assert observe(s, "ordinary", elapsed=0) is None
    assert credit(s)["earned_ns"] == 0
    s.cancel()
    assert observe(s, "ordinary") is None
    assert credit(s)["balance_ns"] == 0


def test_cohort_aligned_probe_costs_are_separate_from_credit_funded_costs():
    s = state()
    park(s)
    assert s.begin_cohort_reprobe()["reason"] == "cohort_aligned"
    observe(s, "speculative", 500)
    observe(s, "speculative", 200)
    assert credit(s)["credit_probes"] == 0
    assert credit(s)["forced_probe_excess_ns"] == 500


def test_credit_is_request_local_and_does_not_leak_through_a_session_prior():
    first, second = state(), state()
    park(first)
    observe(first, "ordinary")
    second.seed_reversible_park({
        "ordinary_ns_per_committed_token": 100,
        "speculative_ns_per_committed_token": 200,
        "speculative_to_ordinary_cost_ratio": 2,
    })
    assert credit(second)["balance_ns"] == 0
    assert credit(second)["probe_reserve_ns"] == 0
    assert credit(first)["balance_ns"] == 2


def test_unselected_legacy_token_backoff_still_reprobes_at_old_boundary():
    s = state(reprobe_credit_share=0)
    park(s)
    assert observe(s, "ordinary") is None
    event = observe(s, "ordinary")
    assert event["event"] == "reprobe" and "reason" not in event
    assert "exploration_credit" not in s.receipt()["reversible"]

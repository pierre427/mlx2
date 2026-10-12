from collections import deque
from types import SimpleNamespace

import pytest

from mlx2.runtime.cost_per_commit_handoff import (
    CostPerCommitHandoffSettings,
    CostPerCommitLaneState,
    cohort_needs_ordinary_probe,
)
from mlx2.runtime.adaptive_policy import MTPOrdinaryHandoffPolicy
from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
from mlx2.runtime.generate import MTPGenerationBatch
from mlx2.runtime.prompt_lookup import CostAwarePLDLatch


def _state(uid=7, scope=("tenant-a", "model-a"), **settings):
    values = {
        "enabled": True,
        "ordinary_samples": 2,
        "speculative_samples": 2,
        "loss_decisions": 2,
        **settings,
    }
    return CostPerCommitLaneState(
        CostPerCommitHandoffSettings(**values),
        lane_uid=uid,
        owner_scope=scope,
    )


def _observe(state, kind, elapsed, committed=1, width=1, uid=7, scope=("tenant-a", "model-a")):
    return state.observe(
        kind=kind,
        elapsed_ns=elapsed,
        committed_tokens=committed,
        execution_width=width,
        lane_uid=uid,
        owner_scope=scope,
    )


def test_width_one_greedy_lane_hands_off_once_after_sustained_loss():
    state = _state()
    assert state.needs_ordinary_probe(1)
    _observe(state, "ordinary", 100)
    _observe(state, "ordinary", 100)
    assert not state.needs_ordinary_probe(1)
    _observe(state, "speculative", 150)
    assert _observe(state, "speculative", 150) is None
    decision = _observe(state, "speculative", 150)
    assert decision["reason"] == "cost_per_committed_token"
    assert decision["execution_width"] == 1
    assert state.handed_off
    assert _observe(state, "speculative", 1000) is None
    assert state.receipt()["observed_used"] is True
    assert state.receipt()["handoff_triggered"] is True


def test_lane_and_tenant_evidence_cannot_cross_owners():
    first = _state()
    second = _state(uid=8, scope=("tenant-b", "model-a"))
    with pytest.raises(ValueError, match="owner identity mismatch"):
        _observe(first, "ordinary", 100, uid=8)
    with pytest.raises(ValueError, match="owner identity mismatch"):
        _observe(first, "ordinary", 100, scope=("tenant-b", "model-a"))
    _observe(first, "ordinary", 100)
    assert second.needs_ordinary_probe(1)


def test_widths_are_not_compared_across_physical_shapes():
    state = _state()
    _observe(state, "ordinary", 100, width=1)
    _observe(state, "ordinary", 100, width=1)
    for _ in range(5):
        assert _observe(state, "speculative", 1000, width=4) is None
    assert not state.handed_off
    assert state.needs_ordinary_probe(4)


def test_late_joining_lane_forces_shared_ordinary_probe_without_erasing_peer():
    established = _state()
    late = _state(uid=8, scope=("tenant-a", "model-a"))
    _observe(established, "ordinary", 100)
    _observe(established, "ordinary", 100)
    _observe(established, "speculative", 80)
    _observe(established, "speculative", 80)
    assert not established.needs_ordinary_probe(1)
    assert late.needs_ordinary_probe(1)
    assert cohort_needs_ordinary_probe((established, late), 1)
    # The executor records the cohort probe only for the lane that needs it;
    # established evidence remains untouched.
    established_before = established.receipt()
    late.observe(
        kind="ordinary",
        elapsed_ns=100,
        committed_tokens=1,
        execution_width=1,
        lane_uid=8,
        owner_scope=("tenant-a", "model-a"),
    )
    assert cohort_needs_ordinary_probe((established, late), 1)
    late.observe(
        kind="ordinary",
        elapsed_ns=100,
        committed_tokens=1,
        execution_width=1,
        lane_uid=8,
        owner_scope=("tenant-a", "model-a"),
    )
    assert not cohort_needs_ordinary_probe((established, late), 1)
    assert established.receipt()["ordinary"] == established_before["ordinary"]
    assert established.receipt()["speculative"]["1"]["samples"] == 2


def test_cancellation_discards_evidence_and_never_hands_off():
    state = _state()
    _observe(state, "ordinary", 100)
    state.cancel()
    for _ in range(8):
        assert _observe(state, "speculative", 1000) is None
    receipt = state.receipt()
    assert receipt["cancelled"] is True
    assert receipt["ordinary"] == receipt["speculative"] == {}


@pytest.mark.parametrize(
    ("elapsed", "committed", "width"),
    [(0, 1, 1), (-1, 1, 1), (1, 0, 1), (1, -1, 1), (1, 1, 0)],
)
def test_invalid_samples_fail_safe(elapsed, committed, width):
    state = _state()
    assert _observe(state, "ordinary", elapsed, committed, width) is None
    assert state.receipt()["invalid_samples"] == 1
    assert not state.handed_off


def test_settings_are_default_off_and_strict():
    assert CostPerCommitHandoffSettings.from_value(None) is None
    assert CostPerCommitHandoffSettings.from_value(False) is None
    with pytest.raises(ValueError, match="unknown"):
        CostPerCommitHandoffSettings.from_value({"enabled": True, "mystery": 1})


def test_receipt_separates_sampling_from_terminal_handoff():
    state = _state()
    receipt = state.receipt()
    assert receipt["selected"] is True
    assert receipt["defaulted"] is False
    assert receipt["observed_used"] is False
    assert receipt["handoff_triggered"] is False
    _observe(state, "ordinary", 100)
    receipt = state.receipt()
    assert receipt["observed_used"] is True
    assert receipt["observed_sampling"] is True
    assert receipt["handoff_triggered"] is False


def test_native_policy_is_default_off_strict_and_separate_from_park_memory():
    default = MTPOrdinaryHandoffPolicy.from_value(
        {"enabled": True, "max_mtp_width": 3}
    )
    assert default.cost_per_commit is None
    selected = MTPOrdinaryHandoffPolicy.from_value(
        {
            "enabled": True,
            "max_mtp_width": 3,
            "cost_per_commit": {"enabled": True},
        }
    )
    assert selected.cost_per_commit.enabled is True
    assert selected.as_dict()["cost_per_commit"]["enabled"] is True
    with pytest.raises(ValueError, match="mutually exclusive"):
        MTPOrdinaryHandoffPolicy.from_value(
            {
                "enabled": True,
                "max_mtp_width": 3,
                "adaptive_park": {"enabled": True},
                "cost_per_commit": {"enabled": True},
            }
        )


def test_native_handoff_detaches_only_losing_lane():
    first = SimpleNamespace(uid=1)
    second = SimpleNamespace(uid=2)
    first_package = SimpleNamespace(detached=SimpleNamespace(lane=first))
    batch = SimpleNamespace(
        state=SimpleNamespace(proposal_open=False, lanes=[first, second]),
        _plain_ready=deque(),
        _cost_handoff_states={
            1: SimpleNamespace(receipt=lambda: {"handoff_triggered": True}),
            2: object(),
        },
        scheduler_stats={},
    )
    detached_indices = []
    retired = []

    def detach(indices):
        detached_indices.extend(indices)
        return [first_package]

    batch._detach_packages = detach
    batch._mark_plain_handoff = lambda packages, **kwargs: setattr(
        packages[0], "handoff_receipt", kwargs["decision"]
    )
    batch._retire_plain_handoff_transactions = lambda packages: retired.extend(packages)
    MTPGenerationBatch._handoff_cost_selected_to_plain(
        batch,
        {1: {"reason": "cost_per_committed_token"}},
        projected_width=2,
    )
    assert detached_indices == [0]
    assert [package.detached.lane.uid for package in batch._plain_ready] == [1]
    assert set(batch._cost_handoff_states) == {2}
    assert retired == [first_package]
    assert first_package.handoff_receipt["cost_per_commit_handoff"] == {
        "handoff_triggered": True
    }


def test_pld_handoff_is_terminal_and_truthful():
    latch = CostAwarePLDLatch(
        plain_rounds=1,
        park_rounds=1,
        cost_margin=0.05,
        one_way_handoff=True,
        lane_uid=4,
        owner_scope=("tenant-a", "pld"),
    )
    latch.observe_round(100, 1, 0, 1)
    latch.observe_round(200, 1, 1, 2)
    assert latch.state == "handed_off"
    receipt = latch.receipt()["cost_per_commit_handoff"]
    assert receipt["observed_used"] is True
    assert receipt["handoff_triggered"] is True
    latch.suspend_for_width(3)
    assert latch.state == "handed_off"


def _external_lane(uid, state):
    return SimpleNamespace(
        uid=uid,
        cost_handoff_state=state,
        cost_handoff_receipt=None,
        session_scope_hash=f"scope-{uid}",
        generated=0,
        cancelled=False,
        ordinary=False,
        ready=deque(),
    )


def test_external_late_join_probe_records_only_missing_lane(monkeypatch):
    settings = CostPerCommitHandoffSettings(
        enabled=True,
        ordinary_samples=2,
        speculative_samples=2,
        loss_decisions=2,
    )
    established_state = CostPerCommitLaneState(
        settings, lane_uid=1, owner_scope=("test", "scope-1")
    )
    late_state = CostPerCommitLaneState(
        settings, lane_uid=2, owner_scope=("test", "scope-2")
    )
    for _ in range(2):
        established_state.observe(
            kind="ordinary",
            elapsed_ns=100,
            committed_tokens=1,
            execution_width=2,
            lane_uid=1,
            owner_scope=("test", "scope-1"),
        )
    established = _external_lane(1, established_state)
    late = _external_lane(2, late_state)
    batch = object.__new__(ExternalDraftBatchGenerator)
    batch.dynamic_singleton_tree = False
    batch.binding = "test"
    batch.scheduler_stats = {}
    forced_depths = []

    def round_stub(cohort, *, forced_depth=None):
        forced_depths.append(forced_depth)
        for lane in cohort:
            lane.generated += 1

    batch._round = round_stub
    ticks = iter((100, 200))
    monkeypatch.setattr(
        "mlx2.runtime.external_speculative.time.perf_counter_ns",
        lambda: next(ticks),
    )
    batch._round_at_batch_route([established, late], concurrent=False)
    assert forced_depths == [0]
    assert established_state.receipt()["ordinary"]["2"]["samples"] == 2
    assert late_state.receipt()["ordinary"]["2"]["samples"] == 1


def test_external_handoff_is_lane_local_and_updates_receipt(monkeypatch):
    settings = CostPerCommitHandoffSettings(
        enabled=True,
        ordinary_samples=1,
        speculative_samples=1,
        loss_decisions=1,
    )
    state = CostPerCommitLaneState(
        settings, lane_uid=1, owner_scope=("test", "scope-1")
    )
    state.observe(
        kind="ordinary",
        elapsed_ns=100,
        committed_tokens=1,
        execution_width=1,
        lane_uid=1,
        owner_scope=("test", "scope-1"),
    )
    response = SimpleNamespace(speculative_receipt={})
    lane = _external_lane(1, state)
    lane.generated = 1
    lane.ready.append(response)
    batch = object.__new__(ExternalDraftBatchGenerator)
    batch.binding = "test"
    batch.scheduler_stats = {}
    monkeypatch.setattr(
        "mlx2.runtime.external_speculative.time.perf_counter_ns", lambda: 300
    )
    batch._observe_cost_handoff_round(
        [lane], [0], 100, ordinary_probe=False
    )
    assert lane.ordinary is True
    assert lane.cost_handoff_receipt["handoff_triggered"] is True
    assert response.speculative_receipt["cost_per_commit_handoff"][
        "handoff_triggered"
    ] is True

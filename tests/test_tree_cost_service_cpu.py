"""Reversible controller prices actual ready-lane service, not isolated turns."""

from types import SimpleNamespace

import pytest

from mlx2.runtime import external_speculative as es
from mlx2.runtime.cost_per_commit_handoff import (
    CostPerCommitHandoffSettings,
    CostPerCommitLaneState,
)


def harness(monkeypatch):
    clock = {"now": 1}
    monkeypatch.setattr(es.time, "perf_counter_ns", lambda: clock["now"])
    batch = object.__new__(es.ExternalDraftBatchGenerator)
    batch.dynamic_singleton_tree = True
    batch.binding = "service-cost-test"
    batch.scheduler_stats = {}
    settings = CostPerCommitHandoffSettings(
        enabled=True,
        reversible=True,
        ordinary_samples=2,
        speculative_samples=2,
        loss_decisions=1,
        window=4,
        settle_rounds=0,
        reprobe_tokens=16,
        reprobe_max_tokens=64,
    )
    batch.lanes = {
        uid: SimpleNamespace(
            uid=uid,
            generated=0,
            anchor=1,
            ready=[],
            cancelled=False,
            ordinary=False,
            session_scope_hash=str(uid),
            cost_handoff_state=CostPerCommitLaneState(
                settings,
                lane_uid=uid,
                owner_scope=(batch.binding, str(uid)),
            ),
        )
        for uid in (1, 2)
    }
    batch._cost_handoff_receipt = lambda lane: lane.cost_handoff_state.receipt()
    plans = []

    def work(ordered):
        phase = getattr(batch, "_cost_service_phase", None)
        if phase is not None:
            phase["ready"] = [lane for lane in ordered if not lane.ready]
        for uids, kind, duration, committed in plans:
            cohort = [batch.lanes[uid] for uid in uids]
            before = [lane.generated for lane in cohort]
            start = clock["now"]
            clock["now"] += duration
            for lane in cohort:
                lane.generated += committed
            batch._observe_cost_handoff_round(
                cohort,
                before,
                start,
                ordinary_probe=kind == "ordinary",
                ordinary_probe_uids=set(uids) if kind == "ordinary" else (),
            )

    batch._decode_phase_work = work

    def run(*turns, account=True):
        plans[:] = turns
        (batch._decode_phase if account else work)(list(batch.lanes.values()))

    for _ in range(2):
        run(((1, 2), "ordinary", 45, 1))
    return batch, run


@pytest.mark.parametrize("account", [False, True])
def test_serialized_turns_cannot_beat_batched_ordinary(monkeypatch, account):
    batch, run = harness(monkeypatch)
    # One lane fits per poll: 44ns each means 88ns per ready session, not44.
    # The old isolated-turn price falsely beats the 45ns ordinary B2 step.
    for index in range(6):
        run(((1 + index % 2,), "speculative", 44, 1), account=account)
    states = [lane.cost_handoff_state for lane in batch.lanes.values()]
    assert all(state.speculation_paused == account for state in states)
    if account:
        assert batch.scheduler_stats["external_cost_ready_wait_ns"] > 0
        assert all(state.last_park_speculative_cost > 45 for state in states)


def test_peer_work_in_the_same_poll_is_in_both_prices(monkeypatch):
    batch, run = harness(monkeypatch)
    for _ in range(2):
        run(((1,), "speculative", 44, 1), ((2,), "speculative", 44, 1))
    for lane in batch.lanes.values():
        assert lane.cost_handoff_state.speculation_paused
        assert lane.cost_handoff_state.last_park_speculative_cost == 88


def test_acceptance_can_pay_for_serialized_turns(monkeypatch):
    batch, run = harness(monkeypatch)
    for index in range(12):
        run(((1 + index % 2,), "speculative", 44, 3))
    assert all(
        not lane.cost_handoff_state.speculation_paused for lane in batch.lanes.values()
    )


def test_draining_ready_tokens_does_not_accumulate_competing_wait(monkeypatch):
    batch, run = harness(monkeypatch)
    batch.lanes[2].ready = [object()]
    for _ in range(4):
        run(((1,), "speculative", 44, 2))
    assert batch._cost_service_wait_ns == {}
    assert not batch.lanes[1].cost_handoff_state.speculation_paused
    assert batch.scheduler_stats.get("external_cost_ready_wait_ns", 0) == 0


def test_cohort_change_discards_unconsumed_timing_debt(monkeypatch):
    batch, run = harness(monkeypatch)
    run(((1,), "speculative", 44, 2))
    assert batch._cost_service_wait_ns[2] == 44
    batch.lanes[1].cancelled = True
    run(((2,), "speculative", 44, 2))
    assert batch._cost_service_wait_ns == {}
    assert not batch.lanes[2].cost_handoff_state.speculation_paused


def test_failed_phase_never_publishes_partial_policy_observations(monkeypatch):
    batch, run = harness(monkeypatch)
    run(((1,), "speculative", 44, 2))
    before = batch.lanes[1].cost_handoff_state.receipt()["observations"]
    original = batch._decode_phase_work

    def fail(ordered):
        original(ordered)
        raise RuntimeError("after successful turn, before phase completes")

    batch._decode_phase_work = fail
    with pytest.raises(RuntimeError, match="before phase completes"):
        run(((1,), "speculative", 44, 1))
    assert batch.lanes[1].cost_handoff_state.receipt()["observations"] == before
    assert batch._cost_service_phase is None
    assert batch._cost_service_wait_ns == {}

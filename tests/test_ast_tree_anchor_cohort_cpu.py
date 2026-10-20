"""Admitted anchor-only dispatch preserves existing round and cache semantics."""

import copy

import mlx.core as mx
import pytest

from mlx2.runtime.cost_per_commit_handoff import (
    CostPerCommitHandoffSettings,
    CostPerCommitLaneState,
)
from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows
from test_ast_terminal_ordinary_cpu import _committed_digest
from test_ast_tree_cohort_flow_cpu import _cohort


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _prepare(kind):
    draft, batch, lanes = _cohort()
    if kind == "final":
        for lane in lanes:
            lane.maximum = lane.generated + 1
    else:
        settings = CostPerCommitHandoffSettings(
            enabled=True,
            reversible=True,
            ordinary_samples=1,
            speculative_samples=1,
            loss_decisions=1,
            settle_rounds=0,
        )
        batch.cost_per_commit_handoff = settings
        for lane in lanes:
            lane.cost_handoff_state = state = CostPerCommitLaneState(
                settings,
                lane_uid=lane.uid,
                owner_scope=(batch.binding, lane.session_scope_hash),
            )
            if kind == "parked":
                for mechanism, cost in (("ordinary", 100), ("speculative", 10000)):
                    state.observe(
                        kind=mechanism,
                        elapsed_ns=cost,
                        committed_tokens=1,
                        execution_width=4,
                        lane_uid=lane.uid,
                        owner_scope=(batch.binding, lane.session_scope_hash),
                    )
                assert state.speculation_paused
    return draft, batch, lanes


@pytest.mark.parametrize("kind", ["probe", "parked", "final"])
@pytest.mark.parametrize("fault", [False, True])
def test_anchor_cohort_real_round_state_rng_context_and_recovery(
    monkeypatch, kind, fault
):
    draft, batch, lanes = _prepare(kind)
    before = [
        (
            _committed_digest(l.cache),
            l.generated,
            list(l.history),
            l.rng.snapshot(),
            int(l.tail.shape[1]),
        )
        for l in lanes
    ]
    expected_rows = [copy.deepcopy(l.cache) for l in lanes]
    tx = HybridVerifyRows(expected_rows).begin([1] * 4)
    logits, features = batch.model.forward_with_taps(
        mx.array([[l.anchor] for l in lanes]), tx.caches, batch.layers
    )
    mx.eval(logits, features)
    expected_lanes = [copy.copy(l) for l in lanes]
    for lane in expected_lanes:
        lane.rng = copy.deepcopy(lane.rng)
    expected = batch._verify(expected_lanes, [None] * 4, logits)
    tx.commit([1] * 4)
    monkeypatch.setattr(
        draft,
        "draft_distributions",
        lambda *a, **kw: pytest.fail("anchor-only round drafted"),
    )
    commit = batch._commit
    if fault:

        def broken(*args, **kwargs):
            commit(*args, **kwargs)
            raise RuntimeError("injected anchor publication failure")

        monkeypatch.setattr(batch, "_commit", broken)
    try:
        if fault:
            with pytest.raises(RuntimeError, match="anchor publication"):
                batch._decode_phase(lanes)
            assert [
                (
                    _committed_digest(l.cache),
                    l.generated,
                    list(l.history),
                    l.rng.snapshot(),
                    int(l.tail.shape[1]),
                )
                for l in lanes
            ] == before
            assert all(not l.ready for l in lanes)
            assert not batch._open and batch._parked_inflight is None
            monkeypatch.setattr(batch, "_commit", commit)
        batch._decode_phase(lanes)
        assert batch.scheduler_stats["external_ast_anchor_target_launches"] == 1
        assert batch.scheduler_stats["external_ast_anchor_target_rows"] == 4
        assert batch.scheduler_stats["external_round_cohort_width_4"] >= 1
        assert batch._parked_inflight is None
        for lane, previous, reference, decision, rng in zip(
            lanes, before, expected_rows, expected, expected_lanes
        ):
            assert lane.generated == previous[1] + 1
            assert lane.ready[-1].token == decision.emitted[0]
            assert lane.rng.snapshot() == rng.rng.snapshot()
            assert _committed_digest(lane.cache) == _committed_digest(reference)
            assert lane.cohort_target_execution["ast_anchor:target_tree_packed"] == 1
            batch._sidecar(lane).validate(batch.binding, len(lane.history))
            if kind == "final":
                assert lane.ready[-1].finish_reason == "length"
                assert int(lane.tail.shape[1]) == 1
            else:
                assert int(lane.tail.shape[1]) == previous[4] + 1
                if kind == "probe":
                    # Reversible control intentionally owns a session bucket;
                    # physical width is independently recorded by dispatch.
                    assert set(lane.cost_handoff_state.receipt()["ordinary"]) == {"0"}
                else:
                    assert (
                        lane.cost_handoff_state.receipt()["reversible"]["parked_tokens"]
                        == 1
                    )
    finally:
        batch.close()


def test_fresh_probe_peers_do_not_split_a_ready_physical_cohort(monkeypatch):
    _draft, batch, lanes = _prepare("probe")
    for lane in lanes[:2]:
        lane.cost_handoff_state.observe(
            kind="ordinary",
            elapsed_ns=100,
            committed_tokens=1,
            execution_width=4,
            lane_uid=lane.uid,
            owner_scope=(batch.binding, lane.session_scope_hash),
        )
    observations = []
    observe = CostPerCommitLaneState.observe

    def recorded(state, **kwargs):
        observations.append(
            (kwargs["lane_uid"], kwargs["execution_width"], kwargs["kind"])
        )
        return observe(state, **kwargs)

    monkeypatch.setattr(CostPerCommitLaneState, "observe", recorded)
    try:
        batch._decode_phase(lanes)
        assert observations == [(lane.uid, 4, "ordinary") for lane in lanes[2:]]
        assert batch.scheduler_stats["external_ast_anchor_target_launches"] == 1
        assert all(
            not lane.cost_handoff_state.needs_ordinary_probe(4) for lane in lanes
        )
        for lane in lanes:
            lane.ready.clear()
        observations.clear()
        batch._decode_phase(lanes)
        assert observations == [(lane.uid, 4, "speculative") for lane in lanes]
        assert batch.scheduler_stats["external_ast_anchor_target_launches"] == 1
        assert all(lane.ast_tree_source_plan.target_sessions == 4 for lane in lanes)
    finally:
        batch.close()


def test_explicit_one_way_and_permanent_ordinary_keep_existing_dispatch(monkeypatch):
    _draft, batch, lanes = _prepare("probe")
    settings = CostPerCommitHandoffSettings(enabled=True, reversible=False)
    for lane in lanes:
        lane.cost_handoff_state = CostPerCommitLaneState(
            settings,
            lane_uid=lane.uid,
            owner_scope=(batch.binding, lane.session_scope_hash),
        )
    groups = []
    monkeypatch.setattr(
        batch,
        "_round_at_batch_route",
        lambda cohort, **kwargs: groups.append(tuple(l.uid for l in cohort)),
    )
    try:
        batch._decode_phase(lanes)
        assert groups == [(lane.uid,) for lane in lanes]
        assert all(lane.cost_handoff_state.needs_ordinary_probe(4) for lane in lanes)
        groups.clear()
        for lane in lanes:
            lane.ordinary = True
        batch._decode_phase(lanes)
        assert groups == [(lane.uid,) for lane in lanes]
        assert not batch._ast_anchor_cohort_eligible(lanes)
    finally:
        batch.close()

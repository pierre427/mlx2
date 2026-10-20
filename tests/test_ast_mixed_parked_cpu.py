"""Mixed parked/active AST cohorts preserve private state and timing scope."""

import copy
from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest
from test_qwen_hybrid_row_exact_cpu import _ast_batch
from test_qwen38_dflash2_cpu import PROMPTS, ordinary_greedy, tiny_target

from mlx2.runtime import external_speculative as es
from mlx2.runtime.ast_planner import ReadyRow

TEST_PROMPTS = [*PROMPTS, [11, 22, 33, 44]]


def _prepared(monkeypatch, parked_count=2):
    mx.set_default_device(mx.cpu)
    target = tiny_target()
    batch = _ast_batch(
        target,
        lanes=4,
        monkeypatch=monkeypatch,
        oracle=False,
        planner=True,
        auto=True,
        cost_handoff=None,
    )
    batch.draft_tail_equalize = True
    # A deterministic shared-launch cost makes the grouping assertion
    # independent of CPU timing. It does not alter tensor computation.
    profile = batch.ast_execution_planner.profile
    batch.ast_execution_planner.profile = replace(
        profile,
        row_costs=(1.0,) * len(profile.row_costs),
        valid_row_costs=(),
        launch_cost=10.0,
        context_row_cost=0.0,
        context_windows=(),
    )
    uids = batch.insert(
        TEST_PROMPTS,
        max_tokens=[20] * 4,
        sampling_configs=[{"sampling_temp": 0.0}] * 4,
    )
    lanes = [batch.lanes[uid] for uid in uids]
    previous = {}
    for lane in lanes:
        while lane.anchor is None:
            batch._prefill(lane)
        previous[lane.uid] = [response.token for response in lane.ready]
        lane.ready.clear()
        state = lane.cost_handoff_state
        scope = (batch.binding, lane.session_scope_hash)
        for _ in range(state.settings.ordinary_samples):
            state.observe(
                kind="ordinary",
                elapsed_ns=100,
                committed_tokens=1,
                execution_width=4,
                lane_uid=lane.uid,
                owner_scope=scope,
            )
    for lane in lanes[:parked_count]:
        state = lane.cost_handoff_state
        state.observe(
            kind="speculative",
            elapsed_ns=10000,
            committed_tokens=1,
            execution_width=4,
            lane_uid=lane.uid,
            owner_scope=(batch.binding, lane.session_scope_hash),
        )
        assert state.speculation_paused
    monkeypatch.setattr(es.time, "perf_counter_ns", lambda: 1000)
    return target, batch, lanes, previous


@pytest.mark.parametrize("parked_count", [1, 2, 3])
def test_mixed_parked_and_active_rows_share_exact_ast_round(monkeypatch, parked_count):
    target, batch, lanes, previous = _prepared(monkeypatch, parked_count)
    original = batch._progressive_verify_many
    cohorts = []
    appends = []
    append = batch._append_context

    def observe(cohort, blocks, **kwargs):
        cohorts.append([len(block.tokens) for block in blocks])
        return original(cohort, blocks, **kwargs)

    def record_append(lane, token):
        appends.append(lane.uid)
        return append(lane, token)

    monkeypatch.setattr(batch, "_progressive_verify_many", observe)
    monkeypatch.setattr(batch, "_append_context", record_append)
    tails = {lane.uid: int(lane.tail.shape[1]) for lane in lanes}
    costs = {
        lane.uid: copy.deepcopy(lane.cost_handoff_state.receipt()["ordinary"])
        for lane in lanes[:parked_count]
    }
    batch._decode_phase(lanes)
    assert cohorts and len(cohorts[0]) == 4
    assert sum(count == 0 for count in cohorts[0]) == parked_count
    assert batch.scheduler_stats["external_round_cohort_width_4"] == 1
    assert batch.scheduler_stats.get("external_decode_parked_split_phases", 0) == 0
    assert batch.scheduler_stats["external_ast_mixed_parked_lanes"] == parked_count
    for lane in lanes[:parked_count]:
        assert lane.uid not in appends
        assert int(lane.tail.shape[1]) == tails[lane.uid] + 1
        receipt = lane.cost_handoff_state.receipt()
        assert receipt["ordinary"] == costs[lane.uid]
        assert receipt["reversible"]["mixed_parked_rounds"] == 1
    for lane, prompt in zip(lanes, TEST_PROMPTS, strict=True):
        emitted = previous[lane.uid] + [response.token for response in lane.ready]
        target.configure_target_verify_row_exact(False)
        assert emitted == ordinary_greedy(target, prompt, len(emitted))
        target.configure_target_verify_row_exact(True)
        batch._sidecar(lane).validate(batch.binding, len(lane.history))


def test_zero_proposal_row_requires_a_terminal_anchor_law(monkeypatch):
    _target, batch, _lanes, _previous = _prepared(monkeypatch)
    ready = [ReadyRow(0, 0, 0, 20), ReadyRow(1, 1, 2, 20)]
    plan = batch.ast_execution_planner.plan_launch(
        ready,
        terminal=True,
        max_lanes=4,
    )
    assert plan.input_lengths == (1, 3)
    with pytest.raises(ValueError, match="geometry"):
        batch.ast_execution_planner.plan_launch(
            ready,
            terminal=False,
            max_lanes=4,
        )


def test_mixed_round_failure_restores_parked_tail_and_all_target_state(monkeypatch):
    from scripts.paired_direct_ab import state_digest

    _target, batch, lanes, _previous = _prepared(monkeypatch)
    before = [
        (
            list(lane.history),
            lane.anchor,
            np.asarray(lane.tail).copy(),
            lane.rng.snapshot(),
            state_digest(batch._freeze_cache(lane.cache)),
            state_digest(batch._sidecar(lane)),
        )
        for lane in lanes
    ]
    original = batch._commit

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected mixed commit failure")

    monkeypatch.setattr(batch, "_commit", fail)
    with pytest.raises(RuntimeError, match="injected mixed commit"):
        batch._decode_phase(lanes)
    for lane, (history, anchor, tail, rng, target, draft) in zip(
        lanes, before, strict=True
    ):
        assert lane.history == history and lane.anchor == anchor
        np.testing.assert_array_equal(np.asarray(lane.tail), tail)
        assert lane.rng.snapshot() == rng
        assert state_digest(batch._freeze_cache(lane.cache)) == target
        assert state_digest(batch._sidecar(lane)) == draft


def test_mixed_park_progress_reprobes_without_ordinary_timing_pollution(monkeypatch):
    _target, batch, lanes, _previous = _prepared(monkeypatch)
    lane = lanes[0]
    state = lane.cost_handoff_state
    before = copy.deepcopy(state.receipt())
    for _ in range(state.settings.reprobe_tokens):
        event = state.observe_parked_progress(
            committed_tokens=1,
            lane_uid=lane.uid,
            owner_scope=(batch.binding, lane.session_scope_hash),
        )
    assert event["event"] == "reprobe"
    assert not state.speculation_paused
    assert state.receipt()["ordinary"] == before["ordinary"]
    assert state.receipt()["valid_samples"] == before["valid_samples"]
    with pytest.raises(ValueError, match="owner identity"):
        state.observe_parked_progress(
            committed_tokens=1,
            lane_uid=lane.uid + 100,
            owner_scope=(batch.binding, lane.session_scope_hash),
        )


def test_permanent_ordinary_lane_is_not_joined_to_reversible_ast(monkeypatch):
    _target, batch, lanes, _previous = _prepared(monkeypatch)
    lanes[-1].ordinary = True
    groups = []
    monkeypatch.setattr(
        batch,
        "_round_at_batch_route",
        lambda cohort, **_kwargs: groups.append(list(cohort)),
    )
    batch._decode_phase(lanes)
    assert sorted(map(len, groups)) == [1, 3]
    assert any(group == [lanes[-1]] for group in groups)


def test_inflight_parked_round_drains_before_mixed_ast_transaction(monkeypatch):
    _target, batch, lanes, _previous = _prepared(monkeypatch, parked_count=4)
    lane = lanes[0]
    state = lane.cost_handoff_state
    state.observe_parked_progress(
        committed_tokens=state.settings.reprobe_tokens - 1,
        lane_uid=lane.uid,
        owner_scope=(batch.binding, lane.session_scope_hash),
    )
    batch._decode_phase(lanes)  # Dispatch the first parked round.
    assert batch._parked_inflight is not None
    batch._decode_phase(lanes)  # Book it; first lane enters reprobe.
    assert not state.speculation_paused
    assert batch._parked_inflight is not None
    for item in lanes:
        item.ready.clear()
    calls = []
    original = batch._progressive_verify_many

    def observe(cohort, blocks, **kwargs):
        calls.append(len(cohort))
        assert batch._parked_inflight is None
        return original(cohort, blocks, **kwargs)

    monkeypatch.setattr(batch, "_progressive_verify_many", observe)
    batch._decode_phase(lanes)
    assert batch._parked_inflight is None and not calls
    for item in lanes:
        item.ready.clear()
    batch._decode_phase(lanes)
    assert calls and calls[0] == 4


def test_planner_declined_reprobe_advances_clock_without_branch_learning(monkeypatch):
    _target, batch, lanes, _previous = _prepared(monkeypatch, parked_count=2)
    planner = batch.ast_execution_planner
    planner.profile = replace(planner.profile, max_lanes=4)
    before = {}
    for lane in lanes:
        latch = batch.progressive_topology_latch.session(lane.uid)
        latch.reprobe_interval = 1
        before[lane.uid] = copy.deepcopy(latch.snapshot())
    batch._decode_phase(lanes)
    assert (
        batch.scheduler_stats.get(
            "external_progressive_topology_observation_failures", 0
        )
        == 0
    )
    for index, lane in enumerate(lanes):
        latch = batch.progressive_topology_latch.session(lane.uid)
        snapshot = latch.snapshot()
        assert (
            snapshot["last_applied_token"] == before[lane.uid]["last_applied_token"] + 1
        )
        assert (
            snapshot["primary_only_samples"]
            == before[lane.uid]["primary_only_samples"] + 1
        )
        assert (
            snapshot["unavailable_samples"]
            == before[lane.uid]["unavailable_samples"] + 1
        )
        assert snapshot["window_samples"] == before[lane.uid]["window_samples"]
        assert latch.no_alternative_win == 0
        observation = lane.progressive_topology_latch_last_observation
        assert observation["executed_alternative_branches"] == 0
        assert observation["unavailability_reason"] == (
            "planner_zero_depth" if index < 2 else "planner_declined_alternatives"
        )
        assert (
            latch.choose(0, 3, 2).decision_token == snapshot["last_applied_token"] + 1
        )

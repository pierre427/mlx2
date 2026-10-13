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


def _drive_growing_context(speculative_ratio, *, limit=4000):
    """Drive one lane the way both executors do: probe whenever asked.

    Per-token target cost grows with context for both routes; speculation
    costs ``speculative_ratio`` times ordinary decode at the same position.
    """
    state = CostPerCommitLaneState(
        CostPerCommitHandoffSettings(enabled=True),
        lane_uid=1,
        owner_scope="t",
    )

    def ordinary_ns(position):
        return int(10_000_000 * (1 + position / 200))

    position = 0
    probes = 0
    while position < limit:
        if state.needs_ordinary_probe(1):
            probes += 1
            decision = state.observe(
                kind="ordinary",
                elapsed_ns=ordinary_ns(position),
                committed_tokens=1,
                execution_width=1,
                lane_uid=1,
                owner_scope="t",
            )
            position += 1
        else:
            decision = state.observe(
                kind="speculative",
                elapsed_ns=int(speculative_ratio * ordinary_ns(position) * 3),
                committed_tokens=3,
                execution_width=1,
                lane_uid=1,
                owner_scope="t",
            )
            position += 3
        if decision is not None:
            return state, decision, position, probes
    return state, None, position, probes


def test_stale_ordinary_baseline_does_not_hand_off_a_winning_speculator():
    # L3-1: the ordinary baseline was frozen at the first decode positions
    # while the speculative window slid, so growing context made a
    # speculator that stayed 1.25x faster look like a loser (handoff at ~94).
    state, decision, position, probes = _drive_growing_context(0.8)
    assert decision is None, (position, decision)
    receipt = state.receipt()
    assert receipt["handoff_triggered"] is False
    assert receipt["ordinary_refreshes"] >= 1
    # Re-probes are paid only when a stale comparison looked like a loss.
    assert probes < 0.05 * position


def test_fresh_baseline_still_hands_off_a_losing_speculator():
    for ratio in (0.95, 0.9):
        assert _drive_growing_context(ratio)[1] is None, ratio
    state, decision, position, _probes = _drive_growing_context(1.25)
    assert decision is not None
    assert decision["one_way"] is True
    assert decision["speculative_to_ordinary_cost_ratio"] > 1 / 0.97
    settings = state.settings
    assert decision["speculative_samples_since_ordinary"] <= (
        settings.speculative_samples + settings.loss_decisions - 1
    )
    # The baseline was fresh: no refresh was needed before a real loss.
    assert state.receipt()["ordinary_refreshes"] == 0
    assert position < 40
    # Deterministic: the same sample sequence yields the same decision.
    assert _drive_growing_context(1.25)[1] == decision


def test_stale_loss_refreshes_both_windows_before_terminal_decision():
    state = _state(ordinary_samples=2, speculative_samples=2, loss_decisions=1, window=4)
    _observe(state, "ordinary", 100)
    _observe(state, "ordinary", 100)
    for _ in range(4):
        assert _observe(state, "speculative", 90) is None
    # Fifth speculative sample since the baseline (a decision may use only the
    # first two): the baseline is stale, so a loss refreshes both windows.
    assert _observe(state, "speculative", 1000) is None
    assert not state.handed_off
    assert state.needs_ordinary_probe(1)
    receipt = state.receipt()
    assert receipt["ordinary_refreshes"] == 1
    assert receipt["ordinary"] == {}
    assert receipt["speculative"]["1"]["samples"] == 0
    _observe(state, "ordinary", 100)
    _observe(state, "ordinary", 100)
    assert _observe(state, "speculative", 1000) is None  # below sample minimum
    decision = _observe(state, "speculative", 1000)
    assert decision["reason"] == "cost_per_committed_token"
    assert decision["ordinary_ns_per_committed_token"] == 100
    assert decision["speculative_samples_since_ordinary"] == 2


class _ClosableTarget(list):
    def close(self):
        pass


class _ClosableTransaction:
    def close(self):
        pass


def _cost_package(uid):
    lane = SimpleNamespace(
        uid=uid,
        num_draft=3,
        ntoks=5,
        seed_h=None,
        stats=SimpleNamespace(total_emitted=5, draft_accepted=4, draft_proposed=6),
    )
    return SimpleNamespace(
        detached=SimpleNamespace(
            lane=lane,
            caches=SimpleNamespace(target=_ClosableTarget(), draft=[]),
            segment_transaction=_ClosableTransaction(),
        ),
        initial_output=None,
        num_tokens=5,
        handoff_receipt=None,
    )


def _cost_batch(uids=(5, 6)):
    batch = MTPGenerationBatch.__new__(MTPGenerationBatch)
    batch.state = SimpleNamespace(lanes=[], proposal_open=False)
    batch._paused, batch._memory_queued, batch._plain_ready = {}, set(), []
    batch._width_lock_deferrals, batch._observed_widths_by_uid = {}, {}
    batch._async_qsa_receipts_by_uid = {}
    batch.scheduler_stats = {}
    batch.adaptive_depth_policy = None
    batch.ordinary_handoff_policy = MTPOrdinaryHandoffPolicy.from_value(
        {"enabled": True, "max_mtp_width": 3, "cost_per_commit": {"enabled": True}}
    )
    batch._ordinary_handoff_latched = False
    batch._segmented_compute_width_locked = False
    batch.async_qsa_promotion = False
    batch.mtp_admission = None
    batch._atomic_handoff_pending = None
    batch._atomic_memory_wait = None
    batch.segmented_live_tip = False
    batch._async_qsa_ticket = None
    batch._async_qsa_pending = False
    batch._cost_handoff_settings = batch.ordinary_handoff_policy.cost_per_commit
    batch._cost_handoff_owner_scope = ("native_self_mtp", id(batch))
    batch._cost_handoff_states = {
        uid: CostPerCommitLaneState(
            batch._cost_handoff_settings,
            lane_uid=uid,
            owner_scope=batch._cost_handoff_owner_scope,
        )
        for uid in uids
    }
    for state in batch._cost_handoff_states.values():
        state.observe(
            kind="ordinary",
            elapsed_ns=100,
            committed_tokens=1,
            execution_width=1,
            lane_uid=state.lane_uid,
            owner_scope=state.owner_scope,
        )
    return batch


def test_cancelled_lane_releases_and_cancels_cost_state():
    # L1-F3: cost states outlived cancellation for the life of the batch.
    batch = _cost_batch()
    state = batch._cost_handoff_states[5]
    batch._paused[5] = _cost_package(5)
    batch._memory_queued.add(5)
    batch.remove_uids([5])
    assert 5 not in batch._cost_handoff_states
    assert state.receipt()["cancelled"] is True
    assert set(batch._cost_handoff_states) == {6}


def test_batch_close_releases_every_cost_state():
    batch = _cost_batch()
    states = list(batch._cost_handoff_states.values())
    batch.close()
    assert batch._cost_handoff_states == {}
    assert all(state.receipt()["cancelled"] for state in states)


def test_static_handoff_releases_cost_state_into_its_receipt():
    batch = _cost_batch()
    package = _cost_package(5)
    batch._paused[5] = package
    batch._handoff_all_to_plain(decision={"reason": "width"}, projected_width=4)
    assert 5 not in batch._cost_handoff_states
    evidence = package.handoff_receipt["cost_per_commit_handoff"]
    assert evidence["handoff_triggered"] is False
    assert evidence["ordinary"]["1"]["samples"] == 1


def test_fallback_releases_cost_state_into_its_receipt():
    batch = _cost_batch()
    package = _cost_package(6)
    batch._paused[6] = package
    batch._memory_queued.add(6)
    assert batch.demote_oldest_paused_to_plain() == 6
    assert package.handoff_receipt["transition"] == "fallback"
    assert 6 not in batch._cost_handoff_states
    evidence = package.handoff_receipt["cost_per_commit_handoff"]
    assert evidence["ordinary"]["1"]["samples"] == 1


class _RoundClock:
    """Monotonic fake perf_counter; MTP rounds advance it by a chosen cost."""

    def __init__(self, real):
        self._real = real
        self.now = 0.0

    def perf_counter(self):
        self.now += 1e-9
        return self.now

    def __getattr__(self, name):
        return getattr(self._real, name)


def _drive_native_cost_handoff(monkeypatch, speculative_round_seconds):
    """Run real MTPGenerationBatch rounds with cost_per_commit selected.

    Ordinary probe rounds cost one fake second; speculative rounds cost
    ``speculative_round_seconds``.  Everything else is the real executor.
    """
    import mlx.core as mx
    from test_batched_mtp import _tiny_qwen4_model

    from mlx2.runtime import generate, hybrid_speculative
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.sample_utils import LaneRNG

    clock = _RoundClock(generate.time)
    monkeypatch.setattr(generate, "time", clock)
    rounds = []

    def charge(state):
        kind = (
            "ordinary"
            if all(lane.num_draft == 0 for lane in state.lanes)
            else "speculative"
        )
        rounds.append((kind, tuple(lane.uid for lane in state.lanes)))
        clock.now += 1.0 if kind == "ordinary" else speculative_round_seconds

    real_propose = hybrid_speculative.propose_batched_self_mtp
    real_zero = hybrid_speculative.advance_batched_self_mtp_zero

    def propose(model, state, *args, **kwargs):
        result = real_propose(model, state, *args, **kwargs)
        charge(state)
        return result

    def zero(model, state, *args, **kwargs):
        result = real_zero(model, state, *args, **kwargs)
        charge(state)
        return result

    monkeypatch.setattr(hybrid_speculative, "propose_batched_self_mtp", propose)
    monkeypatch.setattr(hybrid_speculative, "advance_batched_self_mtp_zero", zero)

    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    generator = reference = None
    try:
        mx.random.seed(924)
        model = _tiny_qwen4_model()
        prompts = [[1, 7, 3, 9, 2], [4, 5, 6, 2, 8]]
        reference = BatchGenerator(
            model, completion_batch_size=2, prefill_batch_size=2, prefill_step_size=32
        )
        generator = BatchGenerator(
            model,
            completion_batch_size=2,
            prefill_batch_size=2,
            prefill_step_size=32,
            self_mtp={
                "num_draft": 2,
                "persistent": True,
                "segment_aware_live_tip": True,
                "segment_aware_cohort_size": 2,
            },
            mtp_ordinary_handoff=MTPOrdinaryHandoffPolicy.from_value(
                {
                    "enabled": True,
                    # Far above the cohort: only the per-lane cost law can fire.
                    "max_mtp_width": 8,
                    "cost_per_commit": {
                        "enabled": True,
                        "ordinary_samples": 2,
                        "speculative_samples": 2,
                        "loss_decisions": 2,
                        "window": 4,
                    },
                }
            ),
        )
        reference_uids = reference.insert(prompts, max_tokens=[16, 16])
        uids = generator.insert(
            prompts,
            max_tokens=[16, 16],
            lane_rngs=[LaneRNG(31), LaneRNG(32)],
            self_mtp_configs=[{"sampling_temp": 0.0}] * 2,
        )

        def finish(batch_generator, request_uids):
            output = {uid: [] for uid in request_uids}
            terminal = {}
            for _ in range(200):
                _, responses = batch_generator.next()
                for response in responses:
                    output[response.uid].append(response.token)
                    if response.finish_reason:
                        terminal[response.uid] = response
                if len(terminal) == len(request_uids):
                    return [output[uid] for uid in request_uids], terminal
            raise AssertionError("generator did not terminate")

        expected, _ = finish(reference, reference_uids)
        tokens, terminal = finish(generator, uids)
        assert tokens == expected
        return SimpleNamespace(
            generator=generator,
            terminal=[terminal[uid] for uid in uids],
            rounds=rounds,
            stats=dict(generator.scheduler_stats),
            leftover_states=dict(generator._generation_batch._cost_handoff_states),
        )
    finally:
        if reference is not None:
            reference.close()
        if generator is not None:
            generator.close()
        mx.set_default_device(previous_device)


def test_native_self_mtp_hands_off_a_slower_speculator(monkeypatch):
    # L8-4: the native observe loop and handoff call ran in no test.
    run = _drive_native_cost_handoff(monkeypatch, speculative_round_seconds=10.0)
    assert [kind for kind, _ in run.rounds[:2]] == ["ordinary", "ordinary"]
    assert run.stats["mtp_ordinary_handoff_cost_per_committed_token"] == 2
    assert run.stats["mtp_ordinary_handoff_lanes"] == 2
    assert "mtp_plain_fallback_lanes" not in run.stats
    for response in run.terminal:
        receipt = response.mtp_receipt
        assert receipt["route"] == "ordinary_after_mtp_handoff"
        handoff = receipt["mtp_ordinary_handoff"]
        assert handoff["reason"] == "cost_per_committed_token"
        evidence = handoff["cost_per_commit_handoff"]
        assert evidence["handoff_triggered"] is True
        assert evidence["ordinary"]["2"]["samples"] == 2
        assert evidence["speculative"]["2"]["samples"] == 3
        assert evidence["decision"]["speculative_to_ordinary_cost_ratio"] > 1
    # Speculation stopped at the decision: two probes, three losing rounds.
    assert [kind for kind, _ in run.rounds] == ["ordinary"] * 2 + ["speculative"] * 3
    assert run.leftover_states == {}


def test_native_self_mtp_keeps_a_faster_speculator(monkeypatch):
    run = _drive_native_cost_handoff(monkeypatch, speculative_round_seconds=0.1)
    assert "mtp_ordinary_handoff_cost_per_committed_token" not in run.stats
    assert "mtp_ordinary_handoff_lanes" not in run.stats
    for response in run.terminal:
        receipt = response.mtp_receipt
        assert receipt["route"] == "segmented_self_mtp"
        evidence = receipt["cost_per_commit_handoff"]
        assert evidence["handoff_triggered"] is False
        assert evidence["ordinary"]["2"]["samples"] == 2
        assert evidence["speculative"]["2"]["samples"] >= 2
    assert sum(kind == "speculative" for kind, _ in run.rounds) >= 2
    assert run.leftover_states == {}

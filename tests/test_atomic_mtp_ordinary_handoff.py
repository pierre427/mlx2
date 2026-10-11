"""Atomic ordinary handoff prices its selected route before unused MTP depth."""

import json

import mlx.core as mx
import pytest
from test_admission_progress import _SizedCache
from test_batched_mtp import _tiny_qwen4_model

from mlx2.runtime.adaptive_policy import MTPOrdinaryHandoffPolicy
from mlx2.runtime.generate import BatchGenerator
from mlx2.runtime.memory_policy import (
    SelfMTPLaneAdmissionController,
    _make_self_mtp_admission_callback,
)

POLICY = {"enabled": True, "max_mtp_width": 1}


def _batch(monkeypatch, *, selected=True, usable=2.0, maximum=8, segmented=True,
           prompts=None):
    monkeypatch.setenv("MLX_LM_SEGMENTED_SELF_MTP", "1")
    monkeypatch.setenv("MLX_LM_TRUE_BATCHED_SEGMENTED_MTP", "1")
    mx.random.seed(941)
    controller = SelfMTPLaneAdmissionController(
        host_memory_gib=16, advisory_gib=12,
        transient_gib_per_lane=1.0, cache_estimator=lambda _tokens: 0,
    )
    free = [controller.hard_reserve_gib + usable]
    admit = _make_self_mtp_admission_callback(controller, free_memory=lambda: free[0])
    batch = BatchGenerator(
        _tiny_qwen4_model(), completion_batch_size=4,
        self_mtp={
            "persistent": True, "num_draft": 2,
            "segment_aware_live_tip": segmented, "segment_aware_cohort_size": 4,
            "segment_aware_async_qsa_promotion": False,
        },
        mtp_admission=admit, mtp_ordinary_handoff=POLICY if selected else None,
    )
    cohort = {"tenant_id": "cpu", "id": "ordinary-admission", "size": 4}
    uids = batch.insert(
        prompts or [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
        max_tokens=[maximum] * 4,
        self_mtp_configs=[{"batch_cohort": dict(cohort)} for _ in range(4)],
    )
    return batch, free, controller, uids


def _select(batch):
    policy = MTPOrdinaryHandoffPolicy.from_value(POLICY)
    batch.mtp_ordinary_handoff = policy
    batch._generation_batch.ordinary_handoff_policy = policy


def _drain(batch, uids, output=None):
    output = output or {uid: [] for uid in uids}
    receipts = {}
    for _ in range(40):
        _, responses = batch.next()
        for response in responses:
            output[response.uid].append(int(response.token))
            if response.finish_reason:
                receipts[response.uid] = response.mtp_receipt
        assert not batch.take_atomic_cohort_failures()
        if len(receipts) == len(uids):
            return output, receipts
    pytest.fail("cohort did not finish")


@pytest.mark.parametrize("maximum", [1, 8])
def test_cold_atomic_cohort_uses_selected_ordinary_budget(monkeypatch, maximum):
    batch, _free, _controller, uids = _batch(monkeypatch, maximum=maximum)
    try:
        # The same joining rows cannot fit even K1; the selected ordinary
        # preparation fits, and the prepared merge is checked again.
        rows = [(uid, 3, 2, False, 0.0, 0.0) for uid in uids]
        assert set(batch.mtp_admission.atomic(rows).values()) == {"queue"}
        executor = batch._generation_batch
        pending_handoffs = []
        migrate = batch._migrate_plain_fallbacks

        def inspect_migration():
            if executor._atomic_handoff_pending is not None:
                assert [p.detached.lane.uid for p in executor._plain_ready] == uids
                pending_handoffs.append(executor._atomic_handoff_pending)
            return migrate()

        monkeypatch.setattr(batch, "_migrate_plain_fallbacks", inspect_migration)
        output, receipts = _drain(batch, uids)
        assert len(pending_handoffs) == 1
        pending = pending_handoffs[0]
        assert pending["uids"] == tuple(uids)
        assert pending["admission"]["max_draft"] == 0
        assert pending["admission"]["ordinary_merge_pending_gib"] > 0
        assert not executor._atomic_handoff_pending
        assert not executor.state.lanes and not executor._paused
        assert batch.scheduler_stats["mtp_ordinary_handoff_events"] == 1
        for receipt in receipts.values():
            assert receipt["route"] == "ordinary_after_mtp_handoff"
            handoff = receipt["mtp_ordinary_handoff"]
            assert handoff["one_way"] and handoff["engaged"]
            assert handoff["prepared_initial_token_pending"] is True
            assert handoff["committed_tokens_before_handoff"] == 0
            assert receipt["ordinary_compute_widths"] == ([] if maximum == 1 else [4])
    finally:
        batch.close()
    reference, _free, _controller, reference_uids = _batch(
        monkeypatch, usable=10.0, maximum=maximum,
    )
    try:
        expected, _ = _drain(reference, reference_uids)
        assert list(output.values()) == list(expected.values())
    finally:
        reference.close()


@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("segmented", [False, True])
def test_prepared_or_active_cohort_hands_off_without_unused_mtp_budget(
    monkeypatch, active, segmented,
):
    batch, free, controller, uids = _batch(
        monkeypatch, selected=False, usable=10.0, segmented=segmented,
    )
    try:
        output = {uid: [] for uid in uids}
        assert batch.next()[1] == []
        if active:
            for _ in range(2):
                for response in batch.next()[1]:
                    output[response.uid].append(int(response.token))
        executor = batch._generation_batch
        assert executor.segmented_live_tip is segmented
        if active:
            assert all(lane.ntoks > 1 for lane in executor.state.lanes)
        free[0] = controller.hard_reserve_gib + 2.0
        assert set(executor.mtp_admission(executor.mtp_cycle_state()).values()) == {"queue"}
        _select(batch)
        assert executor._apply_admission() is False
        assert [p.detached.lane.uid for p in executor._plain_ready] == uids
        output, receipts = _drain(batch, uids, output)
        assert all(len(tokens) == 8 for tokens in output.values())
        assert all(r["ordinary_compute_widths"] == [4] for r in receipts.values())
        assert all(
            r["mtp_ordinary_handoff"]["prepared_initial_token_pending"] is (not active)
            for r in receipts.values()
        )
    finally:
        batch.close()


def test_ordinary_refusal_keeps_owners_first_diagnostics_and_bounded_timeout(monkeypatch):
    batch, free, controller, uids = _batch(monkeypatch, selected=False, usable=10.0)
    try:
        batch.next()
        _select(batch)
        executor = batch._generation_batch
        owners = tuple(executor.state.lanes)
        pairs = tuple(executor.state.row_caches)
        free[0] = controller.hard_reserve_gib + 1.0
        assert executor._apply_admission() is False
        wait = batch.atomic_memory_waits()[0]
        first = json.dumps(wait["admission"], sort_keys=True)
        started = wait["started_at"]
        free[0] += 0.1
        for _ in range(4):
            assert executor._apply_admission() is False
            assert all(a is b for a, b in zip(owners, executor.state.lanes))
            assert all(a is b for a, b in zip(pairs, executor.state.row_caches))
            assert not executor._plain_ready
            assert not executor._ordinary_handoff_latched
            assert not executor.state.proposal_open
        assert json.dumps(wait["admission"], sort_keys=True) == first
        assert wait["started_at"] == started
        assert wait["latest_admission"]["final_free_gib"] == free[0]
        assert wait["admission"]["modes"] == ["queue"] * 4
        assert wait["admission"]["route"] == "ordinary_handoff"
        assert batch.expire_atomic_memory_waits((), now=started + 61, timeout=60) == set(uids)
        failure = batch.take_atomic_cohort_failures()[0]
        assert failure["uids"] == tuple(uids)
        assert failure["admission"]["memory_wait"]["timed_out"]
        assert not batch.mtp_cycle_state()
    finally:
        batch.close()


def test_ordinary_wait_recovers_without_leaving_memory_queued_packages(monkeypatch):
    batch, free, controller, uids = _batch(monkeypatch, selected=False, usable=10.0)
    try:
        batch.next()
        _select(batch)
        executor = batch._generation_batch
        free[0] = controller.hard_reserve_gib + 1.0
        executor._apply_admission()
        assert executor._memory_queued == set(uids)
        free[0] += 1.0
        executor._apply_admission()
        assert not batch.atomic_memory_waits()
        assert not executor._memory_queued and not executor._paused
        assert [p.detached.lane.uid for p in executor._plain_ready] == uids
        _drain(batch, uids)
    finally:
        batch.close()


@pytest.mark.parametrize("cancel", [False, True])
def test_pending_handoff_ownership_releases_once_on_migration_or_cancellation(monkeypatch, cancel):
    class Target(list):
        closes = 0

        def close(self):
            self.closes += 1

    batch, free, controller, uids = _batch(monkeypatch, selected=False, usable=10.0)
    try:
        batch.next()
        _select(batch)
        executor = batch._generation_batch
        free[0] = controller.hard_reserve_gib + 2.0
        executor._apply_admission()
        targets = []
        for package in executor._plain_ready:
            assert package.detached.segment_transaction is None
            package.detached.caches.target = Target(package.detached.caches.target)
            targets.append(package.detached.caches.target)
        if cancel:
            unrelated = batch.insert([[2, 3, 4]], max_tokens=[4])[0]
            queued = batch._unprocessed_sequences[0]
            batch.remove([uids[1]])
            batch.remove([uids[1]])
            failure = batch.take_atomic_cohort_failures()[0]
            assert failure["uids"] == (uids[0], uids[2], uids[3])
            assert batch._unprocessed_sequences[0] is queued
            assert queued[0] == unrelated
            assert not executor._plain_ready
            assert not executor._atomic_handoff_pending
        else:
            _drain(batch, uids)
        assert [target.closes for target in targets] == [1] * 4
        batch.close()
        assert [target.closes for target in targets] == [1] * 4
    finally:
        batch.close()


@pytest.mark.parametrize("action", ["queue", 1, False, None, "invalid"])
def test_mixed_ordinary_decisions_fail_closed_before_any_handoff(monkeypatch, action):
    batch, _free, _controller, uids = _batch(monkeypatch, selected=False, usable=10.0)
    try:
        batch.next()
        _select(batch)
        executor = batch._generation_batch
        executor.mtp_admission.ordinary_preparation = lambda rows: {
            uid: action if index == 0 else "plain"
            for index, (uid, *_rest) in enumerate(rows)
        }
        executor._apply_admission()
        failure = executor.take_atomic_cohort_failure()
        assert failure["uids"] == tuple(uids)
        assert not executor._plain_ready and not executor._ordinary_handoff_latched
        assert tuple(executor.uids) == tuple(uids)
    finally:
        batch.close()


def test_no_selected_handoff_keeps_native_atomic_admission(monkeypatch):
    batch, _free, _controller, uids = _batch(monkeypatch, selected=False)
    try:
        original = list(batch._unprocessed_sequences)
        assert batch.next() == ([], [])
        assert batch.atomic_memory_waits()[0]["uids"] == tuple(uids)
        assert all(a is b for a, b in zip(original, batch._unprocessed_sequences))
        assert not batch._generation_batch._plain_ready
        assert not batch.scheduler_stats.get("mtp_ordinary_handoff_events")
    finally:
        batch.close()


def test_cold_ordinary_refusal_waits_whole_and_recovers(monkeypatch):
    batch, free, controller, uids = _batch(monkeypatch, usable=1.0)
    try:
        original = list(batch._unprocessed_sequences)
        assert batch.next() == ([], [])
        wait = batch.atomic_memory_waits()[0]
        assert wait["uids"] == tuple(uids)
        assert wait["admission"]["max_draft"] == 0
        assert wait["admission"]["modes"] == ["queue"] * 4
        assert all(a is b for a, b in zip(original, batch._unprocessed_sequences))
        free[0] = controller.hard_reserve_gib + 2.0
        _, receipts = _drain(batch, uids)
        assert all(r["route"] == "ordinary_after_mtp_handoff" for r in receipts.values())
        assert not batch.atomic_memory_waits()
    finally:
        batch.close()


def test_ragged_ordinary_merge_prices_padding_not_only_row_sums(monkeypatch):
    batch, _free, _controller, _uids = _batch(
        monkeypatch, selected=False, usable=10.0,
        prompts=[[1, 2, 3], [4] * 300, [7, 8], [10] * 9],
    )
    try:
        batch.next()
        executor = batch._generation_batch
        sizes = [[cache.nbytes for cache in pair.target] for pair in executor.state.row_caches]
        expected = len(sizes) * sum(map(max, zip(*sizes))) / (1 << 30)
        assert expected > sum(map(sum, sizes)) / (1 << 30)
        _select(batch)
        executor._apply_admission()
        assert executor._atomic_handoff_pending["admission"]["ordinary_merge_pending_gib"] == expected
    finally:
        batch.close()


@pytest.mark.parametrize("actions", ["plain", "queue"])
def test_ordinary_selection_never_hides_a_prefill_bound_violation(monkeypatch, actions):
    batch, _free, _controller, uids = _batch(monkeypatch)
    try:
        sequence = batch._unprocessed_sequences[0]
        batch._unprocessed_sequences[0] = (
            *sequence[:3], [_SizedCache(8, 4)], *sequence[4:],
        )
        batch._mtp_prefill_resident = {uids[0]}
        batch._mtp_prefill_projection_bytes = {uids[0]: 1}
        batch.mtp_admission.atomic.ordinary_preparation = lambda rows: {
            row[0]: actions for row in rows
        }
        assert batch._admit_mtp_joining(4) == 0
        assert not batch.atomic_memory_waits()
        assert batch.take_atomic_cohort_failures()[0]["uids"] == tuple(uids)
        assert [sequence[0] for sequence in batch._unprocessed_sequences] == uids
        assert not batch._generation_batch._plain_ready
    finally:
        batch.close()


def test_prepared_cohort_waits_intact_when_headroom_drops_after_cold_admission(monkeypatch):
    batch, free, controller, uids = _batch(monkeypatch)
    try:
        prepare = batch.mtp_admission.atomic.ordinary_preparation

        def admit_then_lose_headroom(rows):
            result = prepare(rows)
            assert set(result.values()) == {"plain"}
            free[0] = controller.hard_reserve_gib + 1.0
            return result

        batch.mtp_admission.atomic.ordinary_preparation = admit_then_lose_headroom
        assert batch.next()[1] == []
        executor = batch._generation_batch
        packages = dict(executor._paused)
        assert list(packages) == uids
        assert all(package.initial_output is not None for package in packages.values())
        assert executor._memory_queued == set(uids)
        assert not executor.state.lanes and not executor._plain_ready
        wait = batch.atomic_memory_waits()[0]
        assert wait["admission"]["route"] == "ordinary_handoff"
        assert batch.next() == ([], [])
        assert all(executor._paused[uid] is packages[uid] for uid in uids)
        free[0] = controller.hard_reserve_gib + 2.0
        output, receipts = _drain(batch, uids)
        assert all(len(tokens) == 8 for tokens in output.values())
        assert all(r["ordinary_compute_widths"] == [4] for r in receipts.values())
        assert not batch.atomic_memory_waits()
    finally:
        batch.close()

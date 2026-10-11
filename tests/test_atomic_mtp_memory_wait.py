"""Whole-cohort retries at closed MTP boundaries; CPU lifecycle checks only."""

import json
from collections import deque

import mlx.core as mx
import pytest
from test_admission_progress import _atomic_joining_batch, _SizedCache
from test_batched_mtp import _tiny_qwen4_model

from mlx2.runtime.generate import BatchGenerator
from mlx2.runtime.memory_policy import _make_self_mtp_admission_callback


def test_joining_all_queue_preserves_identity_order_and_first_refusal():
    free = [20.0]
    batch = _atomic_joining_batch({})
    batch._unprocessed_sequences = deque(list(batch._unprocessed_sequences)[:4])
    for uid in range(4):
        batch._mtp_configs[uid]["batch_cohort"]["size"] = 4
    batch.mtp_admission = _make_self_mtp_admission_callback(free_memory=lambda: free[0])
    original = list(batch._unprocessed_sequences)
    assert batch._admit_mtp_joining(4) == 0
    wait = batch.atomic_memory_waits()[0]
    first = json.dumps(wait["admission"], sort_keys=True)
    started_at = wait["started_at"]
    free[0] = 20.5
    assert batch._admit_mtp_joining(4) == 0
    assert batch.atomic_memory_waits()[0] is wait
    assert wait["started_at"] == started_at
    assert wait["attempts"] == 2
    assert json.dumps(wait["admission"], sort_keys=True) == first
    assert wait["latest_admission"]["final_free_gib"] == 20.5
    assert batch._memory_queued_prefill == set(range(4))
    assert not batch.take_atomic_cohort_failures()
    assert all(left is right for left, right in zip(original, batch._unprocessed_sequences))

    free[0] = 100.0
    assert batch._admit_mtp_joining(4) == 4
    assert not batch.atomic_memory_waits()
    assert not batch._memory_queued_prefill
    assert all(left is right for left, right in zip(original, batch._unprocessed_sequences))


@pytest.mark.parametrize("refusal", ["plain", None, False, 0, -1, "invalid", 1, "queue"])
def test_mixed_or_invalid_joining_decisions_still_fail_closed(refusal):
    decisions = {uid: 2 for uid in range(20)}
    decisions[0] = refusal
    batch = _atomic_joining_batch(decisions)
    assert batch._admit_mtp_joining(20) == 0
    assert not batch.atomic_memory_waits()
    assert batch.take_atomic_cohort_failures()[0]["uids"] == tuple(range(20))


def test_all_queue_with_projection_bound_violation_still_fails_closed():
    batch = _atomic_joining_batch({uid: "queue" for uid in range(20)})
    sequence = batch._unprocessed_sequences[0]
    batch._unprocessed_sequences[0] = (*sequence[:3], [_SizedCache(8, 4)], *sequence[4:])
    batch._mtp_prefill_resident = {0}
    batch._mtp_prefill_projection_bytes = {0: 1}
    assert batch._admit_mtp_joining(20) == 0
    assert not batch.atomic_memory_waits()
    assert batch.take_atomic_cohort_failures()[0]["uids"] == tuple(range(20))


def _decode_wait(monkeypatch, *, paused=False, refuse=True):
    monkeypatch.setenv("MLX_LM_SEGMENTED_SELF_MTP", "1")
    monkeypatch.setenv("MLX_LM_TRUE_BATCHED_SEGMENTED_MTP", "1")
    mx.random.seed(941)
    free = [100.0]
    callback = _make_self_mtp_admission_callback(free_memory=lambda: free[0])
    batch = BatchGenerator(
        _tiny_qwen4_model(), completion_batch_size=4,
        self_mtp={"persistent": True, "num_draft": 2, "segment_aware_live_tip": True,
                  "segment_aware_cohort_size": 4, "segment_aware_async_qsa_promotion": False},
        mtp_admission=callback,
    )
    cohort = {"tenant_id": "cpu", "id": "whole-wait", "size": 4}
    uids = batch.insert(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], max_tokens=[12] * 4,
        self_mtp_configs=[{"batch_cohort": dict(cohort)} for _ in range(4)],
    )
    outputs = {uid: [] for uid in uids}
    _, generated = batch.next()
    for response in generated:
        outputs[response.uid].append(response.token)
    active = batch._generation_batch
    if not refuse:
        return batch, free, uids, outputs
    if paused:
        packages = active._detach_packages(range(len(active.state.lanes)))
        active._paused.update({package.detached.lane.uid: package for package in packages})
    free[0] = 20.0
    for _ in range(5):
        _, generated = batch.next()
        for response in generated:
            outputs[response.uid].append(response.token)
        if batch.atomic_memory_waits():
            break
    assert batch.atomic_memory_waits()
    return batch, free, uids, outputs


@pytest.mark.parametrize("paused", [False, True])
def test_decode_all_queue_keeps_owners_and_recovers_exact_width(monkeypatch, paused):
    batch, free, uids, outputs = _decode_wait(monkeypatch, paused=paused)
    try:
        active = batch._generation_batch
        owners = tuple(id(lane) for lane in active.state.lanes)
        packages = dict(active._paused)
        wait = batch.atomic_memory_waits()[0]
        first = json.dumps(wait["admission"], sort_keys=True)
        for _ in range(12):  # Beyond the eight-round ordinary-demotion threshold.
            assert batch.next() == ([], [])
            assert tuple(id(lane) for lane in active.state.lanes) == owners
            assert active._paused == packages
            assert {row[0] for row in batch.mtp_cycle_state()} == set(uids)
            assert not batch.scheduler_waiting_uids()
            assert not batch.take_atomic_cohort_failures()
            assert not active.state.proposal_open
            assert len(batch._plain_fallback_batch) == 0
        assert json.dumps(wait["admission"], sort_keys=True) == first
        assert batch.scheduler_stats.get("starved_mtp_plain_fallbacks", 0) == 0
        free[0] = 100.0
        _, responses = batch.next()
        for response in responses:
            outputs[response.uid].append(response.token)
        assert {lane.uid for lane in active.state.lanes} == set(uids)
        assert not active._paused
        assert not active._memory_queued
        finished = {}
        for _ in range(30):
            if not finished:
                assert len(active.state.lanes) == 4
            _, responses = batch.next()
            for response in responses:
                outputs[response.uid].append(response.token)
            finished.update({r.uid: r.mtp_receipt for r in responses if r.finish_reason})
            if len(finished) == 4:
                break
        assert set(finished) == set(uids)
        assert all(4 in receipt["observed_compute_widths"] for receipt in finished.values())
        assert not batch.atomic_memory_waits()
        assert not active._memory_queued
    finally:
        batch.close()
    reference, _free, reference_uids, expected = _decode_wait(monkeypatch, refuse=False)
    try:
        for _ in range(30):
            _, responses = reference.next()
            for response in responses:
                expected[response.uid].append(response.token)
            if all(len(tokens) == 12 for tokens in expected.values()):
                break
        assert [outputs[uid] for uid in uids] == [expected[uid] for uid in reference_uids]
    finally:
        reference.close()


@pytest.mark.parametrize("stalled_subset", [False, True])
def test_decode_timeout_retires_every_member_with_first_diagnostics(monkeypatch, stalled_subset):
    batch, free, uids, _outputs = _decode_wait(monkeypatch)
    try:
        wait = batch.atomic_memory_waits()[0]
        first = json.dumps(wait["admission"], sort_keys=True)
        free[0] = 20.5
        batch.next()
        # One stalled member, or the original deadline even with no stalled
        # members (an unrelated preemption may have refreshed progress clocks).
        expired = batch.expire_atomic_memory_waits(
            [uids[0]] if stalled_subset else [],
            now=wait["started_at"] + 61, timeout=60,
        )
        assert expired == set(uids)
        assert not batch.mtp_cycle_state()
        assert not batch.atomic_memory_waits()
        assert batch.next() == ([], [])
        failures = batch.take_atomic_cohort_failures()
        assert len(failures) == 1
        failure = failures[0]
        assert failure["uids"] == tuple(uids)
        diagnostic = failure["admission"]
        assert diagnostic["initial_free_gib"] == 20.0
        assert diagnostic["latest_admission"]["final_free_gib"] == 20.5
        assert diagnostic["memory_wait"]["timed_out"] is True
        assert diagnostic["memory_wait"]["elapsed_seconds"] == 61
        assert json.dumps(wait["admission"], sort_keys=True) == first
    finally:
        batch.close()


@pytest.mark.parametrize("native_remove", [False, True])
def test_cancelled_wait_member_cannot_leave_smaller_cohort(monkeypatch, native_remove):
    batch, _free, uids, _outputs = _decode_wait(monkeypatch)
    try:
        unrelated = batch.insert([[2, 3, 4]], max_tokens=[4])[0]
        queued = batch._unprocessed_sequences[0]
        if native_remove:
            batch._generation_batch.remove_uids([uids[0]])
            batch.next()  # Forward the executor's failure to the serving owner.
        else:
            batch.remove([uids[0]])
        assert not batch.mtp_cycle_state()
        assert not batch.atomic_memory_waits()
        assert list(batch._unprocessed_sequences) == [queued]
        assert queued[0] == unrelated
        failures = batch.take_atomic_cohort_failures()
        assert len(failures) == 1
        assert set(failures[0]["uids"]) == set(uids[1:])
    finally:
        batch.close()


def test_invalid_decision_after_wait_keeps_first_refusal():
    batch = _atomic_joining_batch({})
    batch.mtp_admission = _make_self_mtp_admission_callback(free_memory=lambda: 20.0)
    assert batch._admit_mtp_joining(20) == 0
    wait = batch.atomic_memory_waits()[0]
    first = json.dumps(wait["admission"], sort_keys=True)
    batch.mtp_admission = lambda rows: {row[0]: "plain" for row in rows}
    assert batch._admit_mtp_joining(20) == 0
    assert not batch.atomic_memory_waits()
    failure = batch.take_atomic_cohort_failures()[0]
    assert failure["admission"]["memory_wait"]["started_at"] == wait["started_at"]
    assert failure["admission"]["memory_wait"]["timed_out"] is False
    assert failure["admission"]["initial_free_gib"] == 20.0
    assert json.dumps(wait["admission"], sort_keys=True) == first

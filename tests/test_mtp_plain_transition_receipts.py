"""Focused CPU contracts for MTP-to-ordinary transition receipts."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx2.runtime.generate import BatchGenerator


class _OwnedTarget(list):
    def __init__(self):
        super().__init__()
        self.closes = 0

    def close(self):
        self.closes += 1


class _Transaction:
    def __init__(self):
        self.closes = 0

    def close(self):
        self.closes += 1


def _package(*, kind, reason, token=None, logprobs=None):
    target = _OwnedTarget()
    transaction = _Transaction()
    package = SimpleNamespace(
        detached=SimpleNamespace(
            lane=SimpleNamespace(uid=7),
            caches=SimpleNamespace(target=target, draft=[]),
            segment_transaction=transaction,
        ),
        initial_output=(
            None
            if token is None
            else SimpleNamespace(
                token=token,
                logprobs=logprobs,
                from_draft=False,
            )
        ),
        handoff_receipt={"transition": kind, "reason": reason},
    )
    return package, target, transaction


def _owner(package):
    return SimpleNamespace(
        self_mtp={},
        _generation_batch=SimpleNamespace(
            take_plain_fallbacks=lambda: [package]
        ),
    )


def test_unknown_plain_transition_kind_fails_closed_with_bounded_context():
    package, target, transaction = _package(
        kind="future_transition", reason="future-reason"
    )

    with pytest.raises(
        RuntimeError, match="unknown MTP-to-ordinary transition kind"
    ) as caught:
        BatchGenerator._migrate_plain_fallbacks(_owner(package))

    assert target.closes == transaction.closes == 1
    assert package.detached.segment_transaction is None
    notes = getattr(caught.value, "__notes__", ())
    assert len(notes) == 1
    assert '"kind": "future_transition"' in notes[0]
    assert '"reason": "future-reason"' in notes[0]
    assert '"uid": 7' in notes[0]


def test_invalid_fallback_output_reports_ordinary_transition_and_releases_owner():
    package, target, transaction = _package(
        kind="fallback",
        reason="width_lock_deferral_limit",
        token=9,
        logprobs=mx.array([0.0, -1.0]),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            r"ordinary transition \(kind=fallback, "
            r"reason=width_lock_deferral_limit\): .*outside vocabulary"
        ),
    ) as caught:
        BatchGenerator._migrate_plain_fallbacks(_owner(package))

    assert target.closes == transaction.closes == 1
    assert package.detached.segment_transaction is None
    notes = getattr(caught.value, "__notes__", ())
    assert len(notes) == 1
    assert '"kind": "fallback"' in notes[0]
    assert '"reason": "width_lock_deferral_limit"' in notes[0]


class _HandoffPolicy:
    def as_dict(self):
        return {"policy": "stub"}


def _paused_package(uid):
    lane = SimpleNamespace(
        uid=uid,
        num_draft=3,
        ntoks=5,
        seed_h=None,
        numerical_schedule_changed=True,
        numerical_initial_geometry=("self_mtp", 1, 4, "test"),
        numerical_last_geometry=("self_mtp", 2, 4, "test"),
        stats=SimpleNamespace(total_emitted=5, draft_accepted=4, draft_proposed=6),
    )
    return SimpleNamespace(
        detached=SimpleNamespace(
            lane=lane,
            caches=SimpleNamespace(target=_OwnedTarget(), draft=[]),
            segment_transaction=_Transaction(),
        ),
        initial_output=None,
        num_tokens=5,
        handoff_receipt=None,
    )


def _bare_mtp_batch():
    from mlx2.runtime.generate import MTPGenerationBatch

    batch = MTPGenerationBatch.__new__(MTPGenerationBatch)
    batch.state = SimpleNamespace(lanes=[], proposal_open=False)
    batch._paused = {}
    batch._memory_queued = set()
    batch._plain_ready = []
    batch._width_lock_deferrals = {}
    batch._observed_widths_by_uid = {11: {2}}
    batch._async_qsa_receipts_by_uid = {}
    batch.scheduler_stats = {}
    batch.adaptive_depth_policy = None
    batch.ordinary_handoff_policy = _HandoffPolicy()
    batch._ordinary_handoff_latched = False
    batch._segmented_compute_width_locked = True
    return batch


def test_starvation_demotion_keeps_a_policy_handoff_receipt():
    """L1-F1: a memory-queued lane handed off with its cohort that later
    starves must stay a policy handoff, counted once."""
    batch = _bare_mtp_batch()
    package = _paused_package(11)
    batch._paused[11] = package
    batch._memory_queued.add(11)
    batch._handoff_all_to_plain(decision={"reason": "width"}, projected_width=4)
    handoff = package.handoff_receipt
    assert handoff["mtp_observed_compute_widths_before_handoff"] == [2]
    assert handoff["numerical_schedule_changed"] is True
    assert 11 in batch._paused

    assert batch.demote_oldest_paused_to_plain() == 11

    assert package.handoff_receipt is handoff
    assert package.handoff_receipt.get("transition", "handoff") == "handoff"
    assert package.handoff_receipt["selected"] is True
    assert batch._plain_ready == [package]
    assert 11 not in batch._paused and 11 not in batch._memory_queued
    assert batch.scheduler_stats["mtp_ordinary_handoff_lanes"] == 1
    assert not any(
        key.startswith("mtp_plain_fallback_") for key in batch.scheduler_stats
    ), batch.scheduler_stats


def test_starvation_demotion_still_falls_back_an_unselected_lane():
    batch = _bare_mtp_batch()
    package = _paused_package(11)
    batch._paused[11] = package
    batch._memory_queued.add(11)

    assert batch.demote_oldest_paused_to_plain() == 11

    assert package.handoff_receipt["transition"] == "fallback"
    assert package.handoff_receipt["numerical_schedule_changed"] is True
    assert package.handoff_receipt["reason"] == "starvation_boundary_limit"
    assert batch.scheduler_stats["mtp_plain_fallback_lanes"] == 1
    assert batch.scheduler_stats["mtp_plain_fallback_starvation_boundary_limit"] == 1

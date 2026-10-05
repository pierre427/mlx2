"""CPU-only contract checks for the default-off atomic request owner."""

import threading

import pytest

from mlx2.runtime.paged_atomic_owner import AtomicOwnerError, PagedAtomicRequestOwner
from mlx2.runtime.paged_pack_scheduler import ReservedRows, choose_paged_pack
from mlx2.runtime.paged_request_transaction import (
    CandidateRequest, STATE_PLANES, execute_paged_request,
)


def owner(*, planes=STATE_PLANES, enabled=True):
    return PagedAtomicRequestOwner("r1", {p: ["base"] for p in planes},
                                   supported_planes=planes, enabled=enabled)


def begin_staged(subject, *, planes=STATE_PLANES):
    candidate = subject.begin(CandidateRequest(1, "r1", 3, planes))
    for plane in planes:
        candidate.stage(plane, [f"{plane}:{i}" for i in range(3)])
    return candidate


@pytest.mark.parametrize("accepted", [0, 1, 3])
def test_one_boundary_accepts_exact_prefix_in_every_plane(accepted):
    subject = owner()
    before = subject.snapshot()
    prepared = begin_staged(subject).prepare(accepted)
    assert subject.snapshot() == before
    prepared.publish()
    after = subject.snapshot()
    assert after.generation == 1 and after.revision == "r1"
    for plane in STATE_PLANES:
        assert after.rows(plane) == ("base",) + tuple(f"{plane}:{i}" for i in range(accepted))
    with pytest.raises(AtomicOwnerError, match="closed"):
        prepared.publish()


def test_default_off_and_capability_coverage_fail_closed():
    disabled = owner(enabled=False)
    assert disabled.atomic_publish is False
    with pytest.raises(AtomicOwnerError, match="disabled"):
        disabled.begin(CandidateRequest(1, "r1", 3, ("kv",)))
    subject = owner(planes=("kv", "gdn"))
    with pytest.raises(AtomicOwnerError, match="unsupported"):
        subject.begin(CandidateRequest(1, "r1", 3, ("kv", "qsa")))
    with pytest.raises(AtomicOwnerError, match="revision"):
        subject.begin(CandidateRequest(1, "r0", 3, ("kv",)))
    assert subject.snapshot().generation == 0


@pytest.mark.parametrize("stage", ["stage", "prepare", "publish"])
def test_failure_at_each_stage_keeps_public_state(stage):
    subject = owner()
    before = subject.snapshot()
    candidate = subject.begin(CandidateRequest(1, "r1", 3, STATE_PLANES))
    if stage == "stage":
        with pytest.raises(AtomicOwnerError, match="complete"):
            candidate.stage("kv", ["short"])
        candidate.rollback()
    elif stage == "prepare":
        candidate.stage("kv", [1, 2, 3])
        with pytest.raises(AtomicOwnerError, match="all requested"):
            candidate.prepare(2)
        candidate.rollback()
    else:
        candidate.rollback()
        candidate = begin_staged(subject)
        prepared = candidate.prepare(2)
        subject.replace_reference("r2", {p: ["new"] for p in STATE_PLANES})
        after_reference = subject.snapshot()
        with pytest.raises(AtomicOwnerError, match="drifted"):
            prepared.publish()
        prepared.rollback()
        assert subject.snapshot() == after_reference
        return
    assert subject.snapshot() == before


def test_cancelled_private_work_and_prepared_work_discard_without_publish():
    subject = owner()
    before = subject.snapshot()
    candidate = begin_staged(subject)
    candidate.rollback()
    with pytest.raises(AtomicOwnerError, match="closed"):
        candidate.prepare(1)
    begin_staged(subject).prepare(2).rollback()
    assert subject.snapshot() == before


def test_aba_generation_drift_and_failed_publish_leave_reference_state():
    subject = owner()
    prepared = begin_staged(subject).prepare(3)
    initial = subject.snapshot()
    subject.replace_reference("r2", {p: ["other"] for p in STATE_PLANES})
    subject.replace_reference("r1", {p: list(initial.rows(p)) for p in STATE_PLANES})
    before_failed_publish = subject.snapshot()
    with pytest.raises(AtomicOwnerError, match="drifted"):
        prepared.publish()
    prepared.rollback()
    assert subject.snapshot() == before_failed_publish


def test_observer_never_sees_partial_planes():
    subject = owner()
    prepared = begin_staged(subject).prepare(3)
    barrier = threading.Barrier(2)
    snapshots = []

    def observe():
        barrier.wait()
        for _ in range(1000):
            snapshots.append(subject.snapshot())

    thread = threading.Thread(target=observe)
    thread.start()
    barrier.wait()
    prepared.publish()
    thread.join()
    assert snapshots
    for snap in snapshots:
        lengths = {len(snap.rows(p)) for p in STATE_PLANES}
        assert lengths in ({1}, {4})


def test_staging_and_snapshots_do_not_alias_caller_mutable_rows():
    subject = owner()
    payload = {"x": [1]}
    candidate = subject.begin(CandidateRequest(1, "r1", 1, ("kv",)))
    candidate.stage("kv", [payload])
    payload["x"].append(2)
    candidate.prepare(1).publish()
    view = subject.snapshot()
    assert view.rows("kv")[-1] == {"x": [1]}
    view.rows("kv")[-1]["x"].append(9)
    assert subject.snapshot().rows("kv")[-1] == {"x": [1]}


def test_execution_protocol_publishes_complete_boundary_and_cancels_privately():
    class Price:
        profile_id = "cpu:atomic-owner"

        def estimate_ms(self, reserved, prefill):
            return 1.0

    decision = choose_paged_pack((ReservedRows(1, "verify", 3, 0, 100),), (),
                                 price=Price(), row_capacity=3, free_pages=0,
                                 permit_candidate=True)
    request = CandidateRequest(1, "r1", 3, STATE_PLANES)
    subject = owner()

    def run(candidate):
        for plane in STATE_PLANES:
            candidate.stage(plane, [f"{plane}:{i}" for i in range(3)])
        return 2

    before = subject.snapshot()
    receipt = execute_paged_request(decision, request, subject, run,
                                    permit_candidate=True, cancelled=lambda: True)
    assert receipt.reason == "cancelled" and subject.snapshot() == before
    receipt = execute_paged_request(decision, request, subject, run,
                                    permit_candidate=True)
    assert receipt.published and receipt.accepted_rows == 2
    assert all(len(subject.snapshot().rows(p)) == 3 for p in STATE_PLANES)

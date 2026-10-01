"""Scheduling evidence uses CPU events, never implies Apple GPU overlap."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event, Thread

import pytest

from mlx2.runtime.dpara import (
    DParaBinding,
    DParaPrepared,
    DParaVerification,
    launch_dpara,
    run_dpara_round,
)


def binding():
    return DParaBinding("request", "target-revision", "draft-revision", 1)


def ticket():
    return DParaPrepared(binding(), (3, 4, 5), (), "precomputed")


def outcome():
    return DParaVerification(binding(), 1, 7, ())


def test_actual_independent_task_overlap_and_barrier_selection():
    precompute_started, verification_started = Event(), Event()
    precompute_finished = Event()

    def precompute():
        precompute_started.set()
        assert verification_started.wait(3), (
            "precompute must not serialize before verify"
        )
        precompute_finished.set()
        return ticket()

    def verify():
        verification_started.set()
        assert precompute_started.wait(3), "verify must not serialize before precompute"
        return outcome()

    def resolve(prepared, verified):
        assert precompute_finished.is_set()
        return prepared.resolve(
            verified, lambda payload, v: (payload, v.accepted, v.bonus)
        )

    with ThreadPoolExecutor(max_workers=1) as executor:
        result = run_dpara_round(
            binding(), precompute, verify, resolve, executor=executor
        )
    assert result == ("precomputed", 1, 7)


def test_cancel_running_precompute_discards_late_completion():
    started, release, finished = Event(), Event(), Event()
    prepared = ticket()

    def precompute():
        started.set()
        assert release.wait(3)
        return prepared

    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = launch_dpara(binding(), precompute, executor=executor)
        assert started.wait(3)
        pending.cancel()
        release.set()
        pending._future.add_done_callback(lambda _: finished.set())
        assert finished.wait(3)
        with pytest.raises(ValueError, match="cancelled"):
            pending.finish(
                outcome(), lambda *args: pytest.fail("cancelled round reached resolve")
            )
    assert prepared.state == "discarded"


def test_verifier_failure_discards_tentative_state():
    prepared = ticket()

    def verify():
        raise RuntimeError("target verification failed")

    with (
        ThreadPoolExecutor(max_workers=1) as executor,
        pytest.raises(RuntimeError, match="target verification failed"),
    ):
        run_dpara_round(
            binding(),
            lambda: prepared,
            verify,
            lambda *args: pytest.fail("failure reached resolve"),
            executor=executor,
        )
    assert prepared.state == "discarded"


def test_precompute_failure_propagates_no_finalize():
    def precompute():
        raise RuntimeError("draft failed")

    with (
        ThreadPoolExecutor(max_workers=1) as executor,
        pytest.raises(RuntimeError, match="draft failed"),
    ):
        run_dpara_round(
            binding(),
            precompute,
            outcome,
            lambda *args: pytest.fail("failure reached resolve"),
            executor=executor,
        )


def test_cancellation_during_head_prevents_returned_continuation():
    started, release, complete = Event(), Event(), Event()
    prepared = ticket()
    errors = []

    def finalize(payload, verification):
        started.set()
        assert release.wait(3)
        return "must-not-publish"

    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = launch_dpara(binding(), lambda: prepared, executor=executor)

        def finish():
            try:
                pending.finish(outcome(), lambda t, v: t.resolve(v, finalize))
            except ValueError as error:
                errors.append(str(error))
            finally:
                complete.set()

        worker = Thread(target=finish)
        worker.start()
        assert started.wait(3)
        pending.cancel()  # Must not block on the head.
        release.set()
        assert complete.wait(3)
        worker.join()
    assert errors and "discarded during finalization" in errors[0]
    assert prepared.state == "discarded"


def test_wrong_generation_discards_and_round_is_one_shot():
    prepared = ticket()
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = launch_dpara(binding(), lambda: prepared, executor=executor)
        stale = DParaVerification(binding().next(), 1, 7, ())
        with pytest.raises(ValueError, match="mismatch"):
            pending.finish(
                stale, lambda *args: pytest.fail("stale outcome reached resolve")
            )
        with pytest.raises(ValueError, match="already consumed"):
            pending.finish(
                outcome(), lambda *args: pytest.fail("consumed round reached resolve")
            )
    assert prepared.state == "discarded"


@pytest.mark.parametrize("accepted", [-1, 3, True])
def test_invalid_branch_index_and_malformed_binding(accepted):
    prepared = ticket()
    with pytest.raises(ValueError, match="accepted length"):
        prepared.resolve(
            DParaVerification(binding(), accepted, 7, ()), lambda *args: None
        )
    with pytest.raises(ValueError, match="request, revisions"):
        DParaBinding("", "target", "draft", 1)


def test_ticket_identity_and_spine_are_read_only():
    prepared = ticket()
    with pytest.raises(AttributeError):
        prepared.binding = binding().next()
    with pytest.raises(AttributeError):
        prepared.spine = (1, 2, 3)

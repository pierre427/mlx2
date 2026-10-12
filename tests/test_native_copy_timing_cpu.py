"""Model-free copy timing contracts; these tests submit no GPU work."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx2.runtime.paged_kv_write import NativeWriteBackend

ROOT = Path(__file__).resolve().parents[1]


class _Native:
    def __init__(self) -> None:
        self.active = False
        self.capacity = 0
        self.events: list[dict[str, object]] = []
        self.pending = 0
        self.poisoned = False
        self.fault_stage: str | None = None
        self.fail_terminal_proof = False

    def begin_copy_timing(self, arena, capacity):
        assert arena == "arena"
        if self.active:
            raise RuntimeError("nested")
        self.active = True
        self.capacity = capacity

    def copy_page(self, arena, source, destination, count, epoch, stream):
        assert self.active and arena == "arena" and stream == "stream"
        self.pending += 1
        if self.fault_stage in {"reservation", "handler_registration"}:
            stage, self.fault_stage = self.fault_stage, None
            self.pending -= 1
            raise RuntimeError(f"injected {stage} failure")
        if self.fault_stage == "dispatch":
            self.fault_stage = None
            self.poisoned = True
            raise RuntimeError("injected dispatch failure")
        return (source, destination, count, epoch)

    def poll_copy_timings(self, arena):
        assert arena == "arena"
        result, self.events = self.events, []
        self.pending -= min(len(result), self.pending)
        return result

    def end_copy_timing(self, arena):
        assert arena == "arena"
        if self.poisoned:
            raise RuntimeError("copy timing session has an internal lifecycle fault")
        if self.events:
            raise RuntimeError("unread")
        if self.pending:
            raise RuntimeError("pending")
        self.active = False

    def _testing_set_copy_timing_fault_injection(self, arena, stage):
        assert arena == "arena" and self.active and self.fault_stage is None
        self.fault_stage = stage

    def copy_timing_poisoned(self, arena):
        assert arena == "arena"
        return self.active and self.poisoned

    def terminalize(self):
        if self.fail_terminal_proof:
            raise RuntimeError("terminal proof failure")
        if self.poisoned:
            assert self.pending == 1
            self.pending = 0

    def abort_copy_timing_after_terminal(self, arena, *, terminal_proved):
        assert arena == "arena" and terminal_proved
        assert self.active and self.poisoned and self.pending == 0
        discarded = len(self.events)
        self.events.clear()
        self.active = False
        self.poisoned = False
        return discarded


def _backend() -> tuple[NativeWriteBackend, _Native]:
    native = _Native()
    backend = object.__new__(NativeWriteBackend)
    backend._closed = False
    backend._native = native
    backend._arena = "arena"
    backend.stream = "stream"
    backend._mx = SimpleNamespace(
        async_eval=lambda dependency: None,
        synchronize=lambda stream: native.terminalize(),
    )
    backend._copy_timing_active = False
    backend._copy_timing_capacity = 0
    backend._copy_timing_enqueue_ns = {}
    backend._copy_timing_abort_receipt = None
    return backend, native


def _event(epoch: int, *, succeeded: bool = True) -> dict[str, object]:
    return {
        "epoch": epoch,
        "succeeded": succeeded,
        "completed_host_ns": 47,
        "command_buffer_gpu_start_seconds": 2.0,
        "command_buffer_gpu_end_seconds": 2.000_000_125,
    }


def test_copy_timing_is_default_off_and_declares_command_buffer_scope():
    backend, _ = _backend()
    assert backend._copy_timing_active is False
    capability = backend.copy_timing_capability()
    assert capability == {
        "schema": "mlx2.native-copy-command-buffer-timing.v1",
        "default": False,
        "max_events": 4096,
        "host_measurement": "copy_page_call_plus_async_eval_elapsed_ns",
        "gpu_measurement": "metal_command_buffer_start_to_end",
        "global_wait_inserted": False,
        "performance_claim": False,
    }


def test_copy_timing_separates_host_enqueue_and_terminal_gpu_interval():
    backend, native = _backend()
    backend.begin_copy_timing(capacity=2)
    assert backend.copy_page(0, 128, 64, 7) == (0, 128, 64, 7)
    native.events.append(_event(7))
    records = backend.poll_copy_timings()
    assert len(records) == 1
    assert records[0].epoch == 7
    assert records[0].enqueue_elapsed_ns >= 0
    assert records[0].completed_host_ns == 47
    assert records[0].command_buffer_gpu_ns == 125
    backend.end_copy_timing()
    assert backend._copy_timing_active is False


def test_copy_timing_refuses_nesting_capacity_and_premature_end_or_close():
    backend, native = _backend()
    with pytest.raises(ValueError, match=r"\[1, 4096\]"):
        backend.begin_copy_timing(capacity=0)
    backend.begin_copy_timing(capacity=1)
    with pytest.raises(RuntimeError, match="already active"):
        backend.begin_copy_timing(capacity=1)
    backend.copy_page(0, 128, 64, 1)
    with pytest.raises(RuntimeError, match="capacity exhausted"):
        backend.copy_page(256, 384, 64, 2)
    with pytest.raises(RuntimeError, match="undrained"):
        backend.end_copy_timing()
    with pytest.raises(RuntimeError, match="must be ended"):
        backend.close_after_terminal()
    native.events.append(_event(1, succeeded=False))
    record = backend.poll_copy_timings()[0]
    assert record.command_buffer_gpu_ns is None
    backend.end_copy_timing()


def test_copy_timing_rejects_unknown_duplicate_and_malformed_events():
    backend, native = _backend()
    backend.begin_copy_timing(capacity=2)
    backend.copy_page(0, 128, 64, 1)
    native.events.append(_event(2))
    with pytest.raises(RuntimeError, match="invalid native"):
        backend.poll_copy_timings()

    backend, native = _backend()
    backend.begin_copy_timing(capacity=2)
    backend.copy_page(0, 128, 64, 1)
    native.events.extend((_event(1), _event(1)))
    with pytest.raises(RuntimeError, match="invalid native"):
        backend.poll_copy_timings()


@pytest.mark.parametrize("stage", ("reservation", "handler_registration"))
def test_pre_handler_fault_rolls_back_reservation_without_poison(stage):
    backend, native = _backend()
    backend.begin_copy_timing(capacity=1)
    backend._testing_set_copy_timing_fault_injection(stage)
    with pytest.raises(RuntimeError, match="injected"):
        backend.copy_page(0, 128, 64, 1)
    assert native.pending == 0
    assert native.poisoned is False
    assert native.events == []
    assert backend.copy_timing_abort_receipt() is None
    backend.end_copy_timing()


def test_post_handler_dispatch_fault_aborts_only_after_terminal_proof():
    backend, native = _backend()
    backend.begin_copy_timing(capacity=1)
    backend._testing_set_copy_timing_fault_injection("dispatch")
    with pytest.raises(RuntimeError, match="dispatch"):
        backend.copy_page(0, 128, 64, 1)
    assert native.pending == 0
    assert native.active is False
    assert native.poisoned is False
    assert native.events == []
    assert backend._copy_timing_active is False
    assert backend.copy_timing_abort_receipt() == {
        "schema": "mlx2.native-copy-command-buffer-timing-abort.v1",
        "terminal_stream_proved": True,
        "poisoned_session_aborted": True,
        "native_events_discarded": 0,
        "host_submissions_discarded": 0,
        "timing_result_published": False,
        "performance_claim": False,
    }
    backend.close_after_terminal()
    assert backend._closed is True


def test_poisoned_session_stays_blocked_until_terminal_proof_then_recovers():
    backend, native = _backend()
    backend.begin_copy_timing(capacity=1)
    backend._testing_set_copy_timing_fault_injection("dispatch")
    native.fail_terminal_proof = True
    with pytest.raises(RuntimeError, match="terminal proof failure"):
        backend.copy_page(0, 128, 64, 1)
    assert native.pending == 1
    assert native.active is True
    assert native.poisoned is True
    assert backend._copy_timing_active is True
    with pytest.raises(RuntimeError, match="lifecycle fault|poisoned"):
        backend.end_copy_timing()

    native.fail_terminal_proof = False
    receipt = backend.abort_poisoned_copy_timing_after_terminal()
    assert receipt["terminal_stream_proved"] is True
    assert receipt["timing_result_published"] is False
    assert native.pending == 0
    backend.close_after_terminal()


def test_native_copy_timing_is_callback_only_and_bounded_statically():
    arena = (ROOT / "native/paged_kv/arena.cpp").read_text()
    header = (ROOT / "native/paged_kv/arena.h").read_text()
    copy_body = arena[arena.index("class CopyPrimitive"):arena.index("class DiagnosticReadPrimitive")]
    assert "GPUStartTime()" in copy_body
    assert "GPUEndTime()" in copy_body
    assert "addCompletedHandler" in copy_body
    assert ".wait" not in copy_body
    assert "copy timing capacity must be in [1, 4096]" in arena
    assert "std::deque<CopyTiming> copy_timings_" in header
    assert "class CopyTimingReservation final" in arena
    assert "hand_off_to_callback" in copy_body
    assert "poison_copy_timing_session" in copy_body
    assert "CopyTimingFaultInjection::Reservation" in arena
    assert "CopyTimingFaultInjection::HandlerRegistration" in copy_body
    assert "CopyTimingFaultInjection::Dispatch" in copy_body
    assert "abort_copy_timing_after_terminal" in arena

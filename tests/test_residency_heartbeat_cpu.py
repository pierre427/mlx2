"""CPU tests for the idle residency heartbeat (runtime/residency_heartbeat.py)."""

from __future__ import annotations

import threading
import time

import pytest

from mlx2.runtime import residency_heartbeat as rh
from mlx2.runtime.residency_heartbeat import ResidencyHeartbeat


class FakeClock:
    def __init__(self):
        self.now = 1000.0
        self._lock = threading.Lock()

    def __call__(self):
        with self._lock:
            return self.now

    def advance(self, seconds):
        with self._lock:
            self.now += seconds


def _until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _heartbeat(clock, pulses, *, keep=10.0, pressure=lambda: 0, **kwargs):
    def factory():
        def pulse():
            pulses.append(threading.current_thread().name)

        return pulse, 7

    return ResidencyHeartbeat(
        keep,
        interval_seconds=0.01,
        pulse_factory=factory,
        pressure=pressure,
        available=True,
        clock=clock,
        **kwargs,
    )


def test_validate_keep_seconds_bounds():
    assert rh.validate_keep_seconds(0) == 0.0
    assert rh.validate_keep_seconds("600") == 600.0
    for bad in (-1, 3601, True, "x", None, float("nan")):
        with pytest.raises(ValueError):
            rh.validate_keep_seconds(bad)


def test_start_pulses_on_its_own_thread_and_stops_at_shutdown():
    clock, pulses = FakeClock(), []
    hb = _heartbeat(clock, pulses, generation_stream_index=3)
    assert hb.start() is True
    clock.advance(1.0)  # idle past one interval
    assert _until(lambda: len(pulses) >= 3)
    assert set(pulses) == {"mlx2-residency-heartbeat"}
    status = hb.status()
    assert status["state"] == "beating" and status["running"]
    assert status["stream_index"] == 7 and status["generation_stream_index"] == 3
    hb.stop("shutdown")
    assert not hb.running
    count = len(pulses)
    time.sleep(0.05)
    assert len(pulses) == count
    assert hb.status()["state"] == "stopped"
    assert hb.status()["reason"] == "shutdown"


def test_no_pulse_while_serving_busy():
    clock, pulses = FakeClock(), []
    hb = _heartbeat(clock, pulses)
    hb.start()
    hb.mark(True)
    clock.advance(5.0)  # a long prefill evaluation: still busy
    time.sleep(0.1)
    assert pulses == []
    assert hb.status()["state"] == "serving"
    hb.mark(False)  # lanes drained: idle starts now
    time.sleep(0.05)
    assert pulses == []  # not yet one interval idle on the fake clock
    clock.advance(0.5)
    assert _until(lambda: len(pulses) >= 1)
    hb.stop()


def test_idle_expiry_then_activity_restarts():
    clock, pulses = FakeClock(), []
    hb = _heartbeat(clock, pulses, keep=2.0)
    hb.start()
    clock.advance(1.0)
    assert _until(lambda: len(pulses) >= 1)
    clock.advance(5.0)  # past the keep window
    assert _until(lambda: hb.status()["state"] == "idle_expired")
    assert hb.status()["idle_expiries"] == 1
    count = len(pulses)
    time.sleep(0.05)
    assert len(pulses) == count  # disarmed: no pulses after expiry
    hb.mark(True)  # a new request re-arms it
    hb.mark(False)
    assert hb.status()["restarts"] == 1
    clock.advance(1.0)
    assert _until(lambda: len(pulses) > count)
    hb.stop()


def test_memory_pressure_stops_until_new_activity():
    clock, pulses = FakeClock(), []
    level = {"value": 0}
    hb = _heartbeat(clock, pulses, pressure=lambda: level["value"])
    hb.start()
    clock.advance(1.0)
    assert _until(lambda: len(pulses) >= 1)
    level["value"] = 1  # WARN
    assert _until(lambda: hb.status()["state"] == "pressure")
    status = hb.status()
    assert status["pressure_stops"] == 1 and status["last_pressure_level"] == 1
    count = len(pulses)
    level["value"] = 0  # pressure gone: still off until a request
    time.sleep(0.05)
    assert len(pulses) == count
    hb.mark(True)
    hb.mark(False)
    clock.advance(1.0)
    assert _until(lambda: len(pulses) > count)
    hb.stop()


def test_pulse_error_turns_it_off():
    clock = FakeClock()

    def factory():
        def pulse():
            raise RuntimeError("boom")

        return pulse, 1

    hb = ResidencyHeartbeat(
        10, interval_seconds=0.01, pulse_factory=factory,
        pressure=lambda: 0, available=True, clock=clock,
    )
    hb.start()
    clock.advance(1.0)
    assert _until(lambda: hb.status()["state"] == "failed")
    assert "boom" in hb.status()["error"]
    assert _until(lambda: not hb.running)


def test_refuses_the_generation_stream():
    clock, pulses = FakeClock(), []
    hb = _heartbeat(clock, pulses, generation_stream_index=7)
    hb.start()
    assert _until(lambda: hb.status()["state"] == "failed")
    assert "generation stream" in hb.status()["error"]
    assert pulses == []


def test_off_and_not_wired_never_start():
    off = ResidencyHeartbeat(0, available=True)
    assert off.start() is False and off.status()["state"] == "off"
    unwired = ResidencyHeartbeat(600, available=True, wired=False)
    assert unwired.start() is False
    assert unwired.status()["reason"] == "weights were not wired"
    assert not unwired.running


def test_does_nothing_on_cpu_backend(monkeypatch):
    import mlx.core as mx

    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    built = []
    hb = ResidencyHeartbeat(600, pulse_factory=lambda: built.append(1))
    assert hb.start() is False
    assert hb.status()["state"] == "unavailable"
    assert hb.status()["reason"].startswith("no Metal device")
    assert built == [] and not hb.running
    hb.mark(True)  # serving-loop calls are no-ops
    hb.stop()


def test_metal_pulse_uses_its_own_thread_stream():
    """The real pulse, built on a worker thread, runs on that thread's stream."""
    import mlx.core as mx

    if not mx.metal.is_available():
        pytest.skip("pulse factory builds a GPU stream")
    main_index = rh.stream_index(mx.default_stream(mx.gpu))
    assert main_index is not None
    out = {}

    def worker():
        pulse, index = rh.metal_pulse_factory()
        pulse()
        out["index"] = index

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(10)
    assert out["index"] is not None and out["index"] != main_index


def test_engine_status_and_shutdown_without_a_model():
    """ServingEngine helpers: status before start, start on CPU, stop on close."""
    from mlx2.serving import ServingEngine

    engine = ServingEngine.__new__(ServingEngine)
    engine.residency_keep_seconds = 600.0
    status = engine._residency_heartbeat_status()
    assert status["state"] == "not_started" and status["enabled"] is True
    engine.weight_residency = {"wired": False, "reason": "no Metal working-set size"}
    engine.memory_pressure_level = lambda: 0
    hb = engine._start_residency_heartbeat()
    assert not hb.running
    assert engine._residency_heartbeat_status()["state"] == "unavailable"
    engine._stop_residency_heartbeat("shutdown")
    engine.residency_keep_seconds = 0.0
    engine.residency_heartbeat = None
    assert engine._residency_heartbeat_status()["state"] == "off"


def test_server_flag_parses_and_bounds():
    from mlx2 import server

    parser = server.build_parser()
    args = parser.parse_args(["--model", "/x", "--residency-keep-seconds", "120"])
    assert args.residency_keep_seconds == 120.0
    default = parser.parse_args(["--model", "/x"])
    assert default.residency_keep_seconds == rh.DEFAULT_KEEP_SECONDS
    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "/x", "--residency-keep-seconds", "7200"])

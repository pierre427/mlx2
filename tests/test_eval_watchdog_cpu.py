"""The eval watchdog reports a device evaluation that never returns.

apc-warm-hit-hang-20261021: a wedged GPU left the generation thread blocked
in ``mx.eval`` with ``/health`` answering ``ok`` for an hour.  The watchdog
cannot interrupt the evaluation; it must make the service report the stall
(unhealthy, with a receipt and the generation thread's stack in the log),
and must clear it if the evaluation does return.
"""

import logging
import threading
import time

from test_apc_hits_hybrid_gdn_self_mtp import host  # noqa: F401
from test_apcv2_restore_materialized_cpu import _engine, tiny_laguna

from mlx2.runtime import eval_watchdog as module
from mlx2.runtime import generate as generate_module
from mlx2.runtime.eval_watchdog import EvalWatchdog


def test_trip_and_recovery(caplog):
    watchdog = EvalWatchdog(0.05, model_label="tiny", interval=0.01)
    assert watchdog.enabled and not watchdog.stalled
    assert watchdog.check() is None, "nothing to watch before a phase"
    watchdog.enter("batch.next", lanes=3)
    assert watchdog.check(now=time.monotonic()) is None, "within the bound"
    with caplog.at_level(logging.ERROR, logger=module.__name__):
        stall = watchdog.check(now=time.monotonic() + 1.0)
    assert stall is not None and watchdog.stalled
    assert stall["phase"] == "batch.next" and stall["lanes"] == 3 and stall["model"] == "tiny"
    assert stall["elapsed_seconds"] > 0.05 and stall["bound_seconds"] == 0.05
    assert watchdog.trips == 1
    assert any("has not returned" in record.message for record in caplog.records)
    status = watchdog.status()
    assert status["stall"]["phase"] == "batch.next" and status["phase"]["phase"] == "batch.next"
    # A second pass does not trip again.
    assert watchdog.check(now=time.monotonic() + 2.0) == stall
    assert watchdog.trips == 1
    with caplog.at_level(logging.ERROR, logger=module.__name__):
        watchdog.leave()
    assert not watchdog.stalled and watchdog.recoveries == 1
    assert watchdog.status()["stall"] is None and watchdog.status()["phase"] is None
    assert any("healthy again" in record.message for record in caplog.records)


def test_disabled_watchdog_is_inert():
    watchdog = EvalWatchdog(0)
    assert not watchdog.enabled
    watchdog.enter("batch.next")
    assert watchdog.check(now=time.monotonic() + 1e6) is None
    watchdog.leave()
    watchdog.watch(threading.current_thread())
    assert watchdog.status()["enabled"] is False and not watchdog.stalled


def test_from_environment(monkeypatch):
    monkeypatch.delenv(module.SECONDS_ENV, raising=False)
    monkeypatch.delenv(module.ABORT_ENV, raising=False)
    assert EvalWatchdog.from_environment().seconds == module.DEFAULT_SECONDS
    monkeypatch.setenv(module.SECONDS_ENV, "12.5")
    monkeypatch.setenv(module.ABORT_ENV, "1")
    watchdog = EvalWatchdog.from_environment()
    assert watchdog.seconds == 12.5 and watchdog.abort
    monkeypatch.setenv(module.SECONDS_ENV, "nonsense")
    assert EvalWatchdog.from_environment().seconds == module.DEFAULT_SECONDS
    monkeypatch.setenv(module.SECONDS_ENV, "0")
    assert not EvalWatchdog.from_environment().enabled


def test_serving_engine_reports_a_stalled_evaluation_and_recovers(host, monkeypatch):  # noqa: F811
    monkeypatch.setenv(module.SECONDS_ENV, "0.2")
    monkeypatch.delenv(module.ABORT_ENV, raising=False)
    model, vocab = tiny_laguna()
    original = generate_module.BatchGenerator.next
    stalled_once = threading.Event()

    def slow_next(self):
        if not stalled_once.is_set():
            stalled_once.set()
            time.sleep(0.8)
        return original(self)

    monkeypatch.setattr(generate_module.BatchGenerator, "next", slow_next)
    tokens = [(7 * i + 3) % (vocab - 2) + 1 for i in range(20)]
    engine = _engine(model, vocab)
    try:
        assert engine.status()["healthy"]
        assert engine.eval_watchdog.enabled and engine.eval_watchdog.seconds == 0.2
        job = engine.submit({"tokens": tokens, "max_tokens": 2, "temperature": 0})
        seen_unhealthy = None
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = engine.status()
            if status["eval_watchdog"]["stall"] is not None:
                seen_unhealthy = status
                break
            time.sleep(0.02)
        assert seen_unhealthy is not None, "the watchdog never tripped on the 0.8 s stall"
        assert seen_unhealthy["healthy"] is False
        assert seen_unhealthy["eval_watchdog"]["stall"]["phase"] == "batch.next"
        assert seen_unhealthy["eval_watchdog"]["stall"]["lanes"] == 1
        while True:
            event = job.events.get(timeout=30)
            if "error" in event:
                raise AssertionError(event)
            if "finish_reason" in event:
                break
        status = engine.status()
        assert status["healthy"] is True
        assert status["eval_watchdog"]["stall"] is None
        assert status["eval_watchdog"]["trips"] == 1
        assert status["eval_watchdog"]["recoveries"] == 1
    finally:
        engine.close()

"""Resume must not reopen admission or hang after its worker exits.

Host-only lifecycle fixtures; no checkpoint or GPU execution.
"""

import threading

import pytest

from test_serving_quiesce import _engine


@pytest.mark.parametrize("state", ["serving", "draining", "suspended", "parked"])
def test_resume_refuses_a_stopped_worker(state):
    engine = _engine()
    engine._service_state = state
    engine._worker_stopped = True
    engine.error = "fixture worker failed"

    with pytest.raises(RuntimeError, match="fixture worker failed"):
        engine.resume()

    assert engine._service_state == state


@pytest.mark.parametrize("state", ["serving", "activating", "parked"])
def test_activate_refuses_a_stopped_worker(state):
    engine = _engine()
    engine._service_state = state
    engine._worker_stopped = True

    with pytest.raises(RuntimeError, match="server stopped"):
        engine.activate_model()

    assert engine._service_state == state


def test_resume_wait_detects_worker_exit_without_quiesce_completion():
    engine = _engine()
    engine.quiesce(suspend=False)
    engine._worker_quiesce_action()
    waiting = threading.Event()

    class ObservedEvent(threading.Event):
        def wait(self, timeout=None):
            waiting.set()
            return super().wait(timeout)

    engine._quiesce_complete = ObservedEvent()
    outcome = {}

    def resume():
        try:
            outcome["result"] = engine.resume()
        except RuntimeError as error:
            outcome["error"] = str(error)

    caller = threading.Thread(target=resume, daemon=True)
    caller.start()
    try:
        assert waiting.wait(1), "resume did not reach the owned quiesce boundary"
        with engine.lock:
            engine.error = "fixture worker failed during quiesce"
            engine._worker_stopped = True
        caller.join(1)
        assert not caller.is_alive(), "resume is stuck waiting on a dead worker"
        assert outcome == {"error": "fixture worker failed during quiesce"}
        assert engine._service_state != "serving"
    finally:
        # Also release the pre-fix implementation on a failing regression run.
        engine._quiesce_complete.set()
        caller.join(1)

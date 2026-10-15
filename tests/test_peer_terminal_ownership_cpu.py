"""A terminal may release HTTP capacity while failed resource owners persist."""

from types import SimpleNamespace as NS

import pytest
from test_serving_retention import _finishing_engine

from mlx2.serving import AdmissionClosed


def test_finish_is_idempotent_and_retained_owner_blocks_admission_until_retry():
    events = []
    attempts = []

    def close():
        attempts.append(True)
        if len(attempts) == 1:
            raise OSError("injected release failure")

    job = NS(id="j", cache_branch=NS(close=close), admission_hit=None, admission_tokens=None)
    engine = _finishing_engine(job, events)
    assert engine._finish(job, {"finish_reason": "stop"})
    assert not engine._finish(job, {"error": "late"})
    assert len([e for e in events if isinstance(e, tuple) and e[0] == "emit"]) == 1
    assert len(engine._terminal_cleanup_pending) == 1
    with pytest.raises(AdmissionClosed):
        engine._ensure_admission()
    assert engine._retry_terminal_cleanup()
    assert len(attempts) == 2
    assert not engine._terminal_cleanup_pending
    assert engine.error is None
    engine._ensure_admission()


def test_lora_release_waits_for_unbind_and_retries_in_order():
    events = []
    attempts = []

    def unbind(uid):
        attempts.append(uid)
        if len(attempts) == 1:
            raise OSError("still bound")

    job = NS(id="j", uid=3, lora_slot=2, cache_branch=None, admission_hit=None, admission_tokens=None)
    engine = _finishing_engine(job, events)
    released = []
    engine.multi_lora = NS(unbind_uid=unbind, release=released.append)
    engine._finish(job, {"finish_reason": "stop"})
    assert released == []
    assert len(engine._terminal_cleanup_pending) == 2
    assert engine._retry_terminal_cleanup()
    assert released == [2]
    assert attempts == [3, 3]


def test_closed_before_failed_pin_release_never_becomes_a_successful_noop():
    class BrokenLease:
        closed = False

        def close(self):
            if self.closed:
                return
            self.closed = True
            raise OSError("pin release uncertain")

    job = NS(id="j", cache_branch=BrokenLease(), admission_hit=None, admission_tokens=None)
    engine = _finishing_engine(job, [])
    engine._finish(job, {"finish_reason": "stop"})
    assert not engine._retry_terminal_cleanup()
    assert len(engine._terminal_cleanup_pending) == 1
    with pytest.raises(AdmissionClosed):
        engine._ensure_admission()


def test_drain_cannot_report_quiescence_with_failed_cleanup():
    job = NS(id="j")
    engine = _finishing_engine(job, [])
    engine._service_state = "draining"
    engine._terminal_cleanup_pending = [("j", "pin", lambda: None)]
    assert engine._worker_quiesce_action() is None


def test_observation_failures_cannot_strand_terminal_or_siblings():
    events = []
    leader = NS(id="leader", cache_branch=None, admission_hit=None,
                admission_tokens=None, fanout_role="prefill_leader", fanout_group="g")
    sibling = NS(id="sibling", cache_branch=None, admission_hit=None,
                 admission_tokens=None, fanout_role="apcv2_sibling", fanout_group="g")
    engine = _finishing_engine(leader, events)
    engine.jobs[sibling.id] = sibling
    engine.fanout_waiting = {"g": (sibling,)}

    def close():
        assert leader.id in engine.jobs
        assert not any(isinstance(e, tuple) and e[0] == "emit" for e in events)
        events.append("capsule_closed")

    def observe(*args):
        raise RuntimeError("broken telemetry")

    engine.fanout_capsules = {"g": NS(close=close)}
    engine.batch_metrics = NS(finishing=observe, terminal=observe)
    assert engine._finish(leader, {"finish_reason": "stop"})
    assert engine.jobs == {}
    assert events[0] == "capsule_closed"
    assert len([e for e in events if isinstance(e, tuple) and e[0] == "emit"]) == 2
    assert engine.counts["finish_observation_failures"] == 4


def test_failed_usage_release_retains_reservation_without_losing_terminal():
    events, attempts = [], []
    reservation = object()
    job = NS(id="j", cache_branch=None, admission_hit=None, admission_tokens=None,
             usage_reservation=reservation)
    engine = _finishing_engine(job, events)

    def record(*args, **kwargs):
        raise OSError("ledger write failed")

    def release(value):
        assert value is reservation
        attempts.append(value)
        if len(attempts) == 1:
            raise OSError("ledger release failed")

    engine.usage_ledger = NS(record=record, release=release)
    assert engine._finish(job, {"finish_reason": "stop"})
    assert events[-1] == ("emit", {"finish_reason": "stop"})
    assert len(engine._terminal_cleanup_pending) == 1
    assert engine._retry_terminal_cleanup()
    assert attempts == [reservation, reservation]

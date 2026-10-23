"""Synthetic clocks, workers and turns only. No model resumes or inference."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mlx2.desktop_wakeups import Wakeups


@pytest.fixture
def scheduler(tmp_path):
    now = [1000.0]
    worker = {"state": "Ready", "url": "http://127.0.0.1:12345", "checkpoint": "fixture@cpu", "resume_lineage_id": "original", "execution_kind": "autoregressive", "apc_persist_dir": str(tmp_path / "cache")}
    engine = SimpleNamespace(worker=worker, suspension=SimpleNamespace(record={"worker": dict(worker)}))
    controller = SimpleNamespace(home=tmp_path, engines={"default": engine}, user_profiles=SimpleNamespace(active="profile"), dispatch=Mock())
    value = Wakeups(controller, clock=lambda: now[0], scheduler=False)
    yield SimpleNamespace(wakeups=value, controller=controller, engine=engine, now=now)
    value.close()


def schedule(env, session="one", **options):
    return env.wakeups.schedule({"message": "scheduled synthetic turn", "every_seconds": 10, "suspend_after": False, **options}, session=session, target={"url": env.engine.worker["url"], "checkpoint": env.engine.worker["checkpoint"], "resume_lineage_id": "original"})["wakeup_id"]


def due(env):
    env.now[0] += 10
    env.wakeups.step()


def test_session_list_never_leaks_other_ready_conversations(scheduler):
    env = scheduler
    schedule(env, "one")
    schedule(env, "two", message="private sibling secret")
    due(env)
    view = env.wakeups.dispatch({"command": "wakeup_list"}, session="one")
    assert view["total"] == 1
    assert {row["session_id"] for row in view["deliveries"]} == {"one"}
    assert "private sibling secret" not in str(view)
    assert all("message" not in row for row in view["deliveries"])


def test_due_wakeup_only_resumes_retained_original_and_tracks_pending_lineage(scheduler):
    env = scheduler
    identity = schedule(env)
    env.engine.worker["state"] = "Suspended"
    due(env)
    env.controller.dispatch.assert_called_once_with({"command": "engine_resume", "profile_id": "default"})
    env.engine.worker.update(state="Resuming", resume_lineage_id="pending:new-process")
    env.wakeups.step()
    assert env.wakeups.snapshot()["schedules"][0]["state"] == "waiting_model"
    env.engine.worker.update(state="Ready", resume_lineage_id="original")
    env.wakeups.step()
    assert env.wakeups.dispatch({"command": "wakeup_claim", "wakeup_id": identity})["message"] == "scheduled synthetic turn"


def test_stopped_or_replaced_worker_never_triggers_fresh_load_or_delivery(scheduler):
    env = scheduler
    schedule(env)
    due(env)
    env.engine.worker["state"] = "Stopped"
    env.wakeups.step()
    assert env.wakeups.snapshot()["deliveries"] == []
    assert env.wakeups.snapshot()["schedules"][0]["state"] == "blocked"
    env.controller.dispatch.assert_not_called()


def test_claim_fence_blocks_duplicates_cancelled_and_stale_completion(scheduler):
    env = scheduler
    identity = schedule(env)
    due(env)
    claim = env.wakeups.dispatch({"command": "wakeup_claim", "wakeup_id": identity})
    with pytest.raises(ValueError, match="no longer ready"):
        env.wakeups.dispatch({"command": "wakeup_claim", "wakeup_id": identity})
    with pytest.raises(ValueError, match="Stale"):
        env.wakeups.dispatch({"command": "wakeup_complete", "wakeup_id": identity, "delivery_fence": "wrong", "success": True})
    env.wakeups.dispatch({"command": "wakeup_cancel", "wakeup_id": identity})
    with pytest.raises(ValueError, match="Stale"):
        env.wakeups.dispatch({"command": "wakeup_complete", "wakeup_id": identity, "delivery_fence": claim["delivery_fence"], "success": True})
    assert env.wakeups.snapshot()["schedules"][0]["state"] == "cancelled"


def test_repeat_coalesces_missed_intervals_and_suspend_warning_retains_state(scheduler):
    env = scheduler
    identity = schedule(env, suspend_after=True)
    due(env)
    claim = env.wakeups.dispatch({"command": "wakeup_claim", "wakeup_id": identity})
    env.now[0] = 1042
    env.controller.dispatch.side_effect = ValueError("ownership busy")
    result = env.wakeups.dispatch({"command": "wakeup_complete", "wakeup_id": identity, "delivery_fence": claim["delivery_fence"], "success": True})
    assert result["state"] == "scheduled" and result["warning"]
    row = env.wakeups.snapshot()["schedules"][0]
    assert row["next_due"] == 1050
    assert row["state"] == "scheduled"


def test_profile_departure_invalidates_delivery_without_cross_profile_replay(scheduler):
    env = scheduler
    identity = schedule(env)
    due(env)
    env.wakeups.dispatch({"command": "wakeup_claim", "wakeup_id": identity})
    env.wakeups.profile_departing("profile")
    assert env.wakeups.snapshot()["schedules"][0]["state"] == "interrupted"
    env.controller.user_profiles.active = "other"
    assert env.wakeups.snapshot()["total"] == 0
    env.wakeups.step()
    env.controller.dispatch.assert_not_called()


def test_paginated_arbitrary_count_and_bounded_snapshots_keep_claim_message_full(scheduler):
    env = scheduler
    for index in range(205):
        schedule(env, str(index), message="x" * 65536)
    due(env)
    first = env.wakeups.snapshot()
    second = env.wakeups.snapshot(offset=200)
    assert first["total"] == 205 and len(first["schedules"]) == 200 and len(second["schedules"]) == 5
    assert len(str(first).encode()) < 2 << 20
    assert first["schedules"][0]["message_truncated"] is True
    claim = env.wakeups.dispatch({"command": "wakeup_claim", "wakeup_id": first["deliveries"][0]["id"]})
    assert len(claim["message"]) == 65536


def test_exact_timezone_and_invalid_interval_validation(scheduler):
    env = scheduler
    target = {"url": env.engine.worker["url"], "checkpoint": "fixture@cpu"}
    receipt = env.wakeups.schedule({"message": "date", "at": "1970-01-01T01:20:00+01:00", "suspend_after": False}, session="one", target=target)
    assert receipt["next_due"] == 1200
    for request in ({"every_seconds": True}, {"every_seconds": float("inf")}, {"at": "2040-01-01T12:00:00"}, {"every_seconds": 10, "at": "2040-01-01T12:00:00Z"}):
        with pytest.raises(ValueError):
            env.wakeups.schedule({"message": "invalid", **request}, session="one", target=target)


def test_interrupted_delivery_is_not_replayed_after_restart(scheduler):
    env = scheduler
    identity = schedule(env)
    due(env)
    env.wakeups.dispatch({"command": "wakeup_claim", "wakeup_id": identity})
    env.wakeups.close()
    reopened = Wakeups(env.controller, clock=lambda: env.now[0], scheduler=False)
    try:
        reopened.step()
        assert reopened.snapshot()["schedules"][0]["state"] == "interrupted"
        assert reopened.snapshot()["deliveries"] == []
        env.controller.dispatch.assert_not_called()
    finally:
        reopened.close()


def test_native_operator_activity_can_defer_automatic_suspend(scheduler):
    env = scheduler
    identity = schedule(env, suspend_after=True)
    due(env)
    claim = env.wakeups.dispatch({"command": "wakeup_claim", "wakeup_id": identity})
    result = env.wakeups.dispatch({"command": "wakeup_complete", "wakeup_id": identity, "delivery_fence": claim["delivery_fence"], "success": True, "allow_suspend": False})
    assert result["state"] == "scheduled"
    env.controller.dispatch.assert_not_called()


"""Resume preparation does not block UI snapshots."""
import threading
import time


def test_step_releases_snapshot_lock_while_resuming(tmp_path):
    gate = threading.Event()
    entered = threading.Event()
    worker = {"url": "http://127.0.0.1:1", "resume_lineage_id": "lineage-1", "execution_kind": "autoregressive",
              "apc_persist_dir": "/x", "checkpoint": "c@f", "state": "Ready"}
    engine = SimpleNamespace(worker=worker)
    def dispatch(request):
        assert request["command"] == "engine_resume"
        entered.set(); gate.wait(5)   # begin_resume hashes the runner source tree + inspect_model here
        return {}
    control = SimpleNamespace(home=tmp_path, user_profiles=SimpleNamespace(active="alice"), engines={"e": engine}, dispatch=dispatch)
    now = [1000.0]
    wake = Wakeups(control, clock=lambda: now[0], scheduler=False)
    wake.schedule({"message": "hi", "every_seconds": 10, "at": None}, session="conv", target={"url": worker["url"], "checkpoint": "c@f"})
    worker["state"] = "Suspended"; now[0] += 11
    threading.Thread(target=wake.step, daemon=True).start()
    assert entered.wait(2)
    snap = {}
    reader = threading.Thread(target=lambda: snap.update(wake.snapshot()), daemon=True)
    reader.start(); reader.join(1.0)
    assert not reader.is_alive()
    assert snap["schedules"]
    gate.set(); reader.join(2); wake.close()


def test_unattempted_wakeup_keeps_recurring_schedule_ready_for_redelivery(scheduler):
    identity = schedule(scheduler)
    due(scheduler)
    claim = scheduler.wakeups.dispatch({'command': 'wakeup_claim', 'wakeup_id': identity})
    result = scheduler.wakeups.dispatch({'command': 'wakeup_complete', 'wakeup_id': identity,
        'delivery_fence': claim['delivery_fence'], 'success': False, 'attempted': False})
    assert result['state'] == 'ready'
    scheduler.now[0] += 2
    assert scheduler.wakeups.snapshot()['deliveries']
    claim = scheduler.wakeups.dispatch({'command': 'wakeup_claim', 'wakeup_id': identity})
    scheduler.wakeups.dispatch({'command': 'wakeup_complete', 'wakeup_id': identity,
        'delivery_fence': claim['delivery_fence'], 'success': True})
    assert scheduler.wakeups.snapshot()['schedules'][0]['state'] == 'scheduled'


def test_resume_refusal_backs_off_without_repeated_admission(scheduler):
    schedule(scheduler)
    scheduler.engine.worker['state'] = 'Suspended'
    scheduler.controller.dispatch.side_effect = ValueError('busy')
    due(scheduler)
    for _ in range(10):
        scheduler.now[0] += 0.5
        scheduler.wakeups.step()
    assert scheduler.controller.dispatch.call_count == 1

"""Durable planning fences, session isolation and claims versus verification."""
from types import SimpleNamespace

import pytest

from mlx2.desktop_planning import Planning


def setup(tmp_path):
    control = SimpleNamespace(home=tmp_path, preferences={"built_in_planning_enabled": True})
    planner = Planning(control)
    session = planner.dispatch({"command": "plan_create", "goal": "Synthetic CPU contract"})["session_id"]
    task = planner.dispatch({"command": "plan_add", "session_id": session, "goal": "One immutable goal", "dependencies": []})["task_id"]
    return planner, session, task


def test_explicit_sessions_and_stale_fence(tmp_path):
    planner, session, task = setup(tmp_path)
    with pytest.raises(ValueError, match="explicit"):
        planner.dispatch({"command": "plan_get"})
    other = planner.dispatch({"command": "plan_create", "goal": "other"})["session_id"]
    with pytest.raises(ValueError, match="belong"):
        planner.dispatch({"command": "plan_claim", "session_id": other, "task_id": task, "owner": "worker"})
    first = planner.dispatch({"command": "plan_claim", "session_id": session, "task_id": task, "owner": "old"})
    planner.db.execute("UPDATE tasks SET expires=0 WHERE id=?", (task,)); planner.db.commit()
    second = planner.dispatch({"command": "plan_claim", "session_id": session, "task_id": task, "owner": "new"})
    assert second["fence"] > first["fence"]
    with pytest.raises(ValueError, match="Stale"):
        planner.dispatch({"command": "plan_report", "session_id": session, "task_id": task, "owner": "old", "fence": first["fence"], "result": "claim"})
    planner.close()


def test_claim_is_not_verified_and_dependency_waits_for_human(tmp_path):
    planner, session, task = setup(tmp_path)
    child = planner.dispatch({"command": "plan_add", "session_id": session, "goal": "dependent", "dependencies": [task]})["task_id"]
    lease = planner.dispatch({"command": "plan_claim", "session_id": session, "task_id": task, "owner": "worker"})
    report = planner.dispatch({"command": "plan_report", "session_id": session, "task_id": task, "owner": "worker", "fence": lease["fence"], "result": {"claim": "all done"}})
    assert report["status"] == "claimed" and not report["verified"]
    with pytest.raises(ValueError, match="verified"):
        planner.dispatch({"command": "plan_claim", "session_id": session, "task_id": child, "owner": "worker"})
    with pytest.raises(ValueError, match="native user"):
        planner.dispatch({"command": "plan_verify", "session_id": session, "task_id": task, "evidence": "model says so"})
    planner.dispatch({"command": "plan_verify", "session_id": session, "task_id": task, "evidence": "human reviewed supplied receipt", "_user_verified": True})
    assert planner.dispatch({"command": "plan_claim", "session_id": session, "task_id": child, "owner": "worker"})["goal"] == "dependent"
    planner.close()


def test_cancellation_revokes_leases_and_persists_event_log(tmp_path):
    planner, session, task = setup(tmp_path)
    lease = planner.dispatch({"command": "plan_claim", "session_id": session, "task_id": task, "owner": "worker"})
    planner.controller.preferences["built_in_planning_enabled"] = False
    planner.dispatch({"command": "plan_cancel", "session_id": session})
    planner.close()
    planner = Planning(SimpleNamespace(home=tmp_path, preferences={}))
    graph = planner.dispatch({"command": "plan_get", "session_id": session})
    assert graph["session"]["status"] == "cancelled"
    assert graph["tasks"][0]["owner"] is None and graph["tasks"][0]["fence"] > lease["fence"]
    assert graph["events"][0]["action"] == "session_cancelled"
    with pytest.raises(ValueError, match="not active"):
        planner.dispatch({"command": "plan_report", "session_id": session, "task_id": task, "owner": "worker", "fence": lease["fence"], "result": "late"})
    planner.close()

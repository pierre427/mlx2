import copy
import signal
import subprocess
import sys

import pytest

from scripts import run_qualification_owned as owned


def records():
    owner = {
        "cpg_task": "task",
        "cpg_session": "session",
        "agent_id": "agent",
        "cpg_generation": 7,
    }
    result = {
        "nodes": [
            {
                "id": "task",
                "session_id": "session",
                "type": "TASK",
                "payload": {
                    "status": "in_progress",
                    "owner_agent_id": "agent",
                    "lease_generation": 7,
                    "lease_expires_at": 200,
                },
            }
        ],
        "missing": [],
    }
    return owner, result


@pytest.mark.parametrize(
    "field,value",
    [
        ("owner_agent_id", "other"),
        ("lease_generation", 8),
        ("lease_expires_at", 99),
        ("lease_expires_at", float("nan")),
        ("lease_expires_at", float("inf")),
        ("lease_expires_at", True),
    ],
)
def test_live_task_reassignment_or_expiry_stops_eligibility(field, value):
    owner, result = records()
    result["nodes"][0]["payload"][field] = value
    with pytest.raises(RuntimeError):
        owned.validate_live_lease(result, owner, now=100)


def test_live_lease_requires_exact_session_and_unambiguous_task():
    owner, result = records()
    assert owned.validate_live_lease(result, owner, now=100)["lease_generation"] == 7
    other = copy.deepcopy(result)
    other["nodes"][0]["session_id"] = "other"
    with pytest.raises(RuntimeError):
        owned.validate_live_lease(other, owner, now=100)
    result["nodes"] *= 2
    with pytest.raises(RuntimeError):
        owned.validate_live_lease(result, owner, now=100)


def test_watchdog_cleanup_stops_only_owned_process_group():
    command = [sys.executable, "-c", "import time; time.sleep(60)"]
    owned_child = subprocess.Popen(command, start_new_session=True)
    unrelated = subprocess.Popen(command, start_new_session=True)
    try:
        owned.stop_child(owned_child)
        assert owned_child.returncode == -signal.SIGTERM
        assert unrelated.poll() is None
    finally:
        owned.stop_child(owned_child)
        owned.stop_child(unrelated)


@pytest.mark.parametrize("now", [float("nan"), float("inf"), True, "100"])
def test_live_lease_refuses_invalid_clock(now):
    owner, result = records()
    with pytest.raises(RuntimeError, match="clock"):
        owned.validate_live_lease(result, owner, now=now)


@pytest.mark.parametrize("status", ["planned", "completed", "failed", None])
def test_live_lease_requires_task_still_in_progress(status):
    owner, result = records()
    result["nodes"][0]["payload"]["status"] = status
    with pytest.raises(RuntimeError):
        owned.validate_live_lease(result, owner, now=100)

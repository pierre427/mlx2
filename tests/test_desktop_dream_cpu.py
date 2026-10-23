"""Fake idle, completion, cancellation and enclave probes; no real inference."""
from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

from mlx2.desktop_dream import DEFAULTS, Dream, tool_definitions
from mlx2.desktop_memory import MemoryController


@pytest.fixture
def dream(tmp_path):
    clock = {"now": 1000.0}
    controller = SimpleNamespace(home=tmp_path, preferences={"memory_enabled": True}, event=lambda *args, **kwargs: None)
    controller.memory = MemoryController(controller)
    target = {"url": "http://127.0.0.1:18285", "pid": 123, "checkpoint": "synthetic@fingerprint", "runtime": {"source": "synthetic"}, "context_limit": 8192}
    admitted = {"allowed": True, "reason": "Synthetic admission", "calls": 0}
    def admission(selection, current):
        admitted["calls"] += 1
        return admitted["allowed"], admitted["reason"], {"target": target, "checkpoint": target["checkpoint"], "runtime": target["runtime"], "swapouts": 0}
    worker = Dream(controller, clock=lambda: clock["now"], admission=admission,
                   completion=lambda job, messages, tools: {"role": "assistant", "content": "Synthetic candidate note."}, scheduler=False)
    worker.test_clock = clock
    worker.test_admission = admitted
    yield worker
    worker.close()
    controller.memory.close()


def enable(dream, **changes):
    dream.dispatch({"command": "dream_configure", "changes": {"enabled": True, "target": {"url": "http://127.0.0.1:18285", "pid": 123}, **changes}})


def idle(dream, seconds=400):
    dream.dispatch({"command": "dream_activity", "idle_seconds": seconds, "busy": False})


def job(dream):
    return {"record": {"id": "synthetic"}, "policy": dict(DEFAULTS), "deadline": 2000,
            "cancel": threading.Event(), "sources": set()}


def test_default_off_and_activity_freshness_fail_closed(dream):
    assert not dream.snapshot()["policy"]["enabled"]
    dream.step()
    assert dream.test_admission["calls"] == 0
    enable(dream)
    dream.step()
    assert dream.test_admission["calls"] == 0
    idle(dream)
    dream.test_clock["now"] += 11
    dream.step()
    assert dream.test_admission["calls"] == 0
    assert dream.history == []


def test_idle_job_retains_candidate_note_and_budgets_then_backs_off(dream):
    enable(dream)
    idle(dream)
    dream.step()
    dream.worker.join(2)
    assert dream.history[-1]["state"] == "completed"
    assert dream.history[-1]["charged_tokens"] == 512
    assert dream.history[-1]["scope"] == "private:dream"
    path = dream.enclave / dream.history[-1]["artifact"]["path"]
    assert path.read_text() == "Synthetic candidate note."
    record = json.loads(next(dream.home.glob("job-*.json")).read_text())
    assert "Synthetic candidate note" not in json.dumps(record)
    dream.step()
    assert len(dream.history) == 1
    assert "Backing off" in dream.reason


def test_user_activity_cancels_active_socket_and_spends_inflight_budget(dream):
    started = threading.Event()
    shutdowns = []
    def completion(current, messages, tools):
        current["connection"] = SimpleNamespace(sock=SimpleNamespace(shutdown=lambda how: shutdowns.append(how)), close=lambda: None)
        started.set()
        assert current["cancel"].wait(2)
        raise InterruptedError("Synthetic cancellation")
    dream.completion = completion
    enable(dream)
    idle(dream)
    dream.step()
    assert started.wait(1)
    dream.dispatch({"command": "dream_activity", "idle_seconds": 0, "busy": True})
    dream.worker.join(2)
    assert shutdowns
    assert dream.history[-1]["state"] == "yielded"
    assert dream.history[-1]["charged_tokens"] == 512
    assert list((dream.enclave / "notes").glob("*")) == []


def test_foreground_pressure_and_daily_limits_do_not_start_work(dream):
    enable(dream)
    idle(dream)
    dream.test_admission.update(allowed=False, reason="Synthetic foreground work")
    dream.step()
    assert dream.history == []
    assert dream.reason == "Synthetic foreground work"
    dream.test_admission["allowed"] = True
    dream.next_run = 0
    dream.history.append({"started": dream.test_clock["now"], "charged_tokens": 16384, "charged_seconds": 0})
    dream.step()
    assert dream.current is None
    assert "Daily budget" in dream.reason


@pytest.mark.parametrize("changes", [{"enabled": "yes"}, {"idle_seconds": True}, {"max_job_seconds": 9999},
                                     {"target": {"url": "http://localhost", "pid": True}}, {"social_style": "deceptive"},
                                     {"allow_research": 1}, {"unknown": 1}])
def test_policy_is_typed_and_bounded(changes):
    with pytest.raises(ValueError):
        Dream.validate_policy(changes)


def test_enclave_refuses_escape_symlink_hardlink_and_low_confidence(dream, tmp_path):
    current = job(dream)
    dream.policy["enabled"] = True
    idle(dream)
    for path in ("../outside", "/absolute", ".archive/secret", "nested/../outside"):
        with pytest.raises(ValueError):
            dream.execute(current, "enclave_write", {"path": path, "text": "candidate", "confidence": 0.95})
    outside = tmp_path / "outside.txt"
    outside.write_text("private external data")
    (dream.enclave / "link").symlink_to(outside)
    with pytest.raises(ValueError, match="symbolic"):
        dream.execute(current, "enclave_read", {"path": "link"})
    (dream.enclave / "hardlink").hardlink_to(outside)
    with pytest.raises(ValueError, match="singly linked"):
        dream.execute(current, "enclave_read", {"path": "hardlink"})
    with pytest.raises(ValueError, match="self-assessed confidence"):
        dream.execute(current, "enclave_write", {"path": "notes/low.txt", "text": "candidate", "confidence": 0.89})
    assert outside.read_text() == "private external data"


def test_enclave_changes_retain_previous_bytes_and_archive_is_reversible(dream):
    dream.policy["enabled"] = True
    idle(dream)
    current = job(dream)
    args = {"path": "research/note.md", "text": "first", "confidence": 0.95}
    dream.execute(current, "enclave_write", args)
    dream.execute(current, "enclave_write", {**args, "text": "second"})
    assert next((dream.enclave / ".archive").iterdir()).read_text() == "first"
    result = dream.execute(current, "enclave_archive", {"path": args["path"]})
    assert result["recoverable"]
    assert not (dream.enclave / args["path"]).exists()
    assert {p.read_text() for p in (dream.enclave / ".archive").iterdir()} == {"first", "second"}


def test_memory_scope_is_fixed_private_candidate_no_promotion_tools(dream):
    dream.policy["enabled"] = True
    idle(dream)
    result = dream.execute(job(dream), "dream_memory_add", {"title": "Private candidate", "body": "Unreviewed synthetic observation", "sources": ["synthetic:fixture"]})
    assert result["status"] == "candidate"
    assert result["scope"] == "private:dream"
    with pytest.raises(ValueError):
        dream.execute(job(dream), "dream_memory_add", {"title": "Escape", "body": "x", "sources": [], "scope": "project:mlx2"})
    names = {row["function"]["name"] for row in tool_definitions(True)}
    assert not names & {"memory_review", "run_command", "model_start", "file_delete", "tool_call"}


def test_external_research_requires_current_run_admitted_source(dream):
    dream.policy["enabled"] = True
    idle(dream)
    calls = []
    dream.controller.research = SimpleNamespace(dispatch=lambda request, **kwargs: calls.append(request) or {"results": [{"id": "admitted"}]})
    current = job(dream)
    current["policy"]["allow_research"] = True
    with pytest.raises(PermissionError):
        dream.execute(current, "research_read", {"source_id": "foreign"})
    dream.execute(current, "research_search", {"query": "synthetic", "limit": 1})
    dream.execute(current, "research_read", {"source_id": "admitted"})
    assert len(calls) == 2


def test_real_admission_missing_thermal_waiters_foreign_and_own_busy_attribution(dream, tmp_path, monkeypatch):
    import os

    from mlx2 import desktop, inference_monitor
    host = {"cpu_busy_percent": [5, 10], "thermal_state": "unavailable", "swap": {"swapouts_bytes": 0, "swapouts_per_second": 0}}
    engine = SimpleNamespace(child=SimpleNamespace(pid=123, poll=lambda: None), worker={"url": "http://127.0.0.1:18285", "state": "Busy"}, preparing=lambda: False)
    dream.controller.engines = {"default": engine}
    dream.controller.owned_pids = lambda: {123}
    dream.controller.campaign = SimpleNamespace(running=False)
    dream.controller.primary = SimpleNamespace(host=SimpleNamespace(sample=lambda: host))
    dream.controller.dispatch = lambda request: {"url": engine.worker["url"], "checkpoint": "synthetic@fingerprint", "runtime": {"source": "synthetic"}}
    waiters = tmp_path / "waiters"
    waiters.mkdir()
    monkeypatch.setattr(desktop, "WAITERS", waiters)
    processes = []
    monkeypatch.setattr(desktop, "model_processes", lambda: processes)
    status = {"running": 0, "waiting": 0, "active_lanes": 0}
    monkeypatch.setattr(inference_monitor, "_get", lambda *args: json.dumps(status).encode())
    selection = {"url": engine.worker["url"], "pid": 123}
    allowed, reason, _ = dream._admission(selection, None)
    assert not allowed and "evidence unavailable" in reason
    host["thermal_state"] = "nominal"
    assert dream._admission(selection, None)[0]  # cached Busy is not live foreground evidence
    status.update(running=1, active_lanes=1)
    assert not dream._admission(selection, None)[0]
    current = {"connection": object(), "swapouts": 0}
    assert dream._admission(selection, current)[0]  # exactly this run's one request
    status["waiting"] = 1
    assert not dream._admission(selection, current)[0]
    status.update(running=0, active_lanes=0, waiting=0)
    host["swap"]["swapouts_bytes"] = 1
    assert not dream._admission(selection, current)[0]
    host["swap"]["swapouts_bytes"] = 0
    processes.append({"pid": 999})
    assert not dream._admission(selection, None)[0]
    processes.clear()
    waiter = waiters / f"session.label.{os.getpid()}.1000"
    waiter.touch()
    os.utime(waiter, (1000, 1000))
    assert not dream._admission(selection, None)[0]


def test_user_profiles_isolate_jobs_enclaves_memory_and_legacy(dream):
    # Existing unscoped candidate material stays operator-only after profiles exist.
    dream._write("legacy.txt", "legacy secret")
    dream.history.append({"id": "legacy", "started": 1000, "state": "completed"})
    dream.controller.user_profiles = SimpleNamespace(active="alice")
    enable(dream)
    idle(dream)
    dream.step()
    dream.worker.join(2)
    alice_record = dream.history[-1]
    assert alice_record["profile_id"] == "alice"
    alice_root = dream._profile_enclave()
    assert [row["id"] for row in dream.snapshot()["jobs"]] == [alice_record["id"]]
    assert not (alice_root / "legacy.txt").exists()
    current = job(dream)
    current["record"].update(profile_id="alice", scope=alice_record["scope"])
    note = dream.execute(current, "dream_memory_add", {"title": "Alice secret", "body": "private alice material", "sources": []})
    assert note["scope"] == alice_record["scope"] != "private:dream"
    dream.controller.user_profiles.active = "bob"
    bob_root = dream._profile_enclave()
    assert bob_root != alice_root
    assert dream.snapshot()["jobs"] == []
    assert dream.snapshot()["daily"]["jobs"] == 0
    assert list(bob_root.iterdir()) == []
    with pytest.raises(InterruptedError, match="profile changed"):
        dream.execute(current, "enclave_write", {"path": "escape.txt", "text": "stale", "confidence": 1})
    assert not (alice_root / "escape.txt").exists()
    assert not (bob_root / "escape.txt").exists()
    bob = job(dream)
    bob["record"].update(profile_id="bob", scope="private:dream:bob")
    found = dream.execute(bob, "dream_memory_search", {"query": "alice"})
    assert "Alice secret" not in json.dumps(found)


def test_profile_switch_during_completion_discards_output(dream):
    dream.controller.user_profiles = SimpleNamespace(active="alice")
    def completion(current, messages, tools):
        dream.controller.user_profiles.active = "bob"
        return {"role": "assistant", "content": "must not be written"}
    dream.completion = completion
    enable(dream)
    idle(dream)
    dream.step()
    dream.worker.join(2)
    assert dream.history[-1]["state"] == "yielded"
    assert not list(dream._profile_enclave("alice").rglob("*.md"))
    assert dream.snapshot()["jobs"] == []

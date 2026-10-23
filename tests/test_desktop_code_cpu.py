"""Synthetic code-workspace controls; no real provider, model or network calls."""
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx2 import desktop_code
from mlx2.desktop_code import (
    CodeWorkspace,
    checked_workspace,
    code_command,
    workspace_identity,
)
from mlx2.desktop_planning import Planning


@pytest.fixture
def code(tmp_path, monkeypatch):
    home = tmp_path / "private"
    home.mkdir()
    controller = SimpleNamespace(home=home, user_profiles=SimpleNamespace(active="alice"), preferences={"built_in_planning_enabled": True})
    controller.planning = Planning(controller)
    controller.dispatch = lambda request: {"url": "http://127.0.0.1:8285", "model": "synthetic", "checkpoint": "synthetic@fixture", "api_key": "never-persist-this"}
    value = CodeWorkspace(controller)
    monkeypatch.setattr(desktop_code, "executable", lambda provider: sys.executable)
    # A tiny local Python child exercises pipes/process ownership without launching a provider.
    monkeypatch.setattr(desktop_code, "code_command", lambda *_args: [sys.executable, "-c", 'import json; print(json.dumps({"type":"item.completed","item":{"type":"agent_message","text":"Synthetic result"}}))'])
    yield value
    value.close()
    controller.planning.close()


def create(code, tmp_path, **settings):
    workspace = tmp_path / ("work-" + str(len(code.sessions)))
    workspace.mkdir()
    return code.dispatch({"command": "code_session_create", "title": "Synthetic", "workspace": str(workspace), **settings})["session"]


def context(code, session, chat="code:root"):
    return code.dispatch({"command": "code_tools", "code_session_id": session["id"], "target": {"url": "http://127.0.0.1:8285", "pid": 123}, "chat_session_id": chat})


def wait(code, identity):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if code.jobs[identity]["status"] not in desktop_code.ACTIVE:
            code.jobs[identity]["thread"].join(timeout=1)
            return code.jobs[identity]
        time.sleep(0.01)
    raise AssertionError("Synthetic code child did not finish")


def test_session_defaults_persistence_profile_guard_and_recoverable_delete(code, tmp_path):
    session = create(code, tmp_path)
    assert (session["provider"], session["effort"], session["max_iterations"]) == ("codex", "medium", 3)
    code.dispatch({"command": "code_session_save", "code_session_id": session["id"], "profile_id": "alice", "document": {"conversations": {"code:root": []}}})
    restored = CodeWorkspace(code.controller)
    assert restored.dispatch({"command": "code_session_get", "code_session_id": session["id"]})["document"] == {"conversations": {"code:root": []}}
    restored.close()
    code.controller.user_profiles.active = "bob"
    assert code.snapshot()["sessions"] == []
    with pytest.raises(ValueError, match="profile"):
        code.dispatch({"command": "code_session_get", "code_session_id": session["id"]})
    code.controller.user_profiles.active = "alice"
    with pytest.raises(ValueError, match="profile"):
        code.dispatch({"command": "code_session_save", "code_session_id": session["id"], "profile_id": "bob", "document": {}})
    code.dispatch({"command": "code_session_delete", "code_session_id": session["id"]})
    assert code.snapshot()["sessions"] == []
    assert Path(session["workspace"]).is_dir()
    code.dispatch({"command": "code_session_restore", "code_session_id": session["id"]})
    assert code.snapshot()["sessions"][0]["id"] == session["id"]


def test_workspace_replacement_and_symlink_change_refused(tmp_path):
    workspace = tmp_path / "work"
    workspace.mkdir()
    identity = workspace_identity(str(workspace))
    workspace.rename(tmp_path / "original")
    workspace.symlink_to(tmp_path / "original", target_is_directory=True)
    with pytest.raises(ValueError, match="traversal"):
        checked_workspace(identity)
    workspace.unlink()
    workspace.mkdir()
    with pytest.raises(ValueError, match="identity"):
        checked_workspace(identity)
    with pytest.raises(ValueError, match="absolute"):
        workspace_identity("relative")


def test_provider_argv_preserves_sandbox_and_permissions():
    codex = code_command("codex", "/fixed/codex", Path("."), "medium")
    assert "workspace-write" in codex and "--json" in codex
    assert 'approval_policy="on-request"' in codex
    assert codex[codex.index("-C") + 1] == "."
    claude = code_command("claude", "/fixed/claude", Path("."), "high")
    assert "--restricted" in claude and "acceptEdits" in claude and "--permission-prompts" in claude
    assert not any("bypass" in flag or "danger" in flag for flag in codex + claude)
    with pytest.raises(ValueError):
        code_command("shell", "/bin/sh", Path("."), "medium")


def test_scoped_tools_require_approval_reject_override_and_other_session_job(code, tmp_path):
    first, second = create(code, tmp_path), create(code, tmp_path)
    tools = context(code, first)
    names = {item["function"]["name"] for item in tools["tools"]}
    assert names == {"mlx2__code_run", "mlx2__code_result", "mlx2__code_cancel"} | {"mlx2__" + name for name in desktop_code.PLAN_OPERATIONS}
    request = {"command": "code_tool_call", "session_id": tools["session_id"], "name": "mlx2__code_run", "arguments": {"prompt": "Synthetic change"}}
    assert code.dispatch(request)["requires_approval"]
    with pytest.raises(ValueError, match="arguments"):
        code.dispatch({**request, "arguments": {"prompt": "x", "workspace": "/"}, "approved": True})
    job = code.dispatch({**request, "approved": True})["result"]["job_id"]
    wait(code, job)
    other = context(code, second)
    with pytest.raises(ValueError, match="session/profile"):
        code.dispatch({"command": "code_tool_call", "session_id": other["session_id"], "name": "mlx2__code_result", "arguments": {"job_id": job}})
    assert "never-persist-this" not in "".join(path.read_text() for path in code.home.glob("*.json"))


def test_real_pipe_child_and_fenced_exact_parent_completion(code, tmp_path):
    session = create(code, tmp_path)
    tools = context(code, session)
    result = code.dispatch({"command": "code_tool_call", "session_id": tools["session_id"], "name": "mlx2__code_run", "arguments": {"prompt": "Synthetic"}, "approved": True})
    job = wait(code, result["result"]["job_id"])
    assert job["status"] == "completed"
    assert job["summary"] == "Synthetic result"
    assert code.dispatch({"command": "code_completion_claim", "code_session_id": session["id"], "chat_session_id": "other"})["delivery"] is None
    delivery = code.dispatch({"command": "code_completion_claim", "code_session_id": session["id"], "chat_session_id": "code:root"})["delivery"]
    assert delivery["target"]["pid"] == 123
    assert delivery["chat_session_id"] == "code:root"
    assert "Synthetic result" in delivery["content"]
    with pytest.raises(ValueError, match="Stale"):
        code.dispatch({"command": "code_completion_complete", "delivery_id": job["id"], "claim_id": "wrong", "success": True})
    code.dispatch({"command": "code_completion_complete", "delivery_id": job["id"], "claim_id": delivery["claim_id"], "success": True})
    assert code.dispatch({"command": "code_completion_claim", "code_session_id": session["id"]})["delivery"] is None


def test_secret_split_across_pipe_chunks_and_provider_semantic_failure(code, tmp_path, monkeypatch):
    monkeypatch.setattr(desktop_code, "code_command", lambda *_args: [sys.executable, "-c", 'import os,time; os.write(1,b"sk-abcdefghijkl"); time.sleep(.15); os.write(1,b"mnopqrstuvwxyz\\n"); print(\'{"type":"result","is_error":true,"result":"Denied"}\')'])
    session = create(code, tmp_path)
    identity = code.dispatch({"command": "code_start", "code_session_id": session["id"], "prompt": "Synthetic"})["job_id"]
    job = wait(code, identity)
    assert job["status"] == "failed"
    result = code.dispatch({"command": "code_status", "job_id": identity})
    assert "sk-abcdefghijkl" not in result["output"]
    assert "[redacted]" in result["output"]
    assert "permission" in job["error"]
    with pytest.raises(ValueError, match="already failed"):
        code.dispatch({"command": "code_start", "code_session_id": session["id"], "prompt": "Synthetic"})


def test_iteration_cap_and_old_context_cycle_fence(code, tmp_path):
    session = create(code, tmp_path, max_iterations=1)
    tools = context(code, session)
    identity = code.dispatch({"command": "code_start", "code_session_id": session["id"], "prompt": "First"})["job_id"]
    wait(code, identity)
    with pytest.raises(ValueError, match="iteration"):
        code.dispatch({"command": "code_start", "code_session_id": session["id"], "prompt": "Second"})
    code.dispatch({"command": "code_cycle_start", "code_session_id": session["id"]})
    with pytest.raises(ValueError, match="older goal"):
        code.dispatch({"command": "code_tool_call", "session_id": tools["session_id"], "name": "mlx2__code_run", "arguments": {"prompt": "Second"}, "approved": True})
    code.dispatch({"command": "code_stop_loop", "code_session_id": session["id"]})
    with pytest.raises(ValueError, match="stopped"):
        code.dispatch({"command": "code_start", "code_session_id": session["id"], "prompt": "Second"})


def test_callback_failures_are_bounded_and_user_retry_explicit(code, tmp_path):
    session = create(code, tmp_path)
    tools = context(code, session)
    job_id = code.dispatch({"command": "code_tool_call", "session_id": tools["session_id"], "name": "mlx2__code_run", "arguments": {"prompt": "First"}, "approved": True})["result"]["job_id"]
    job = wait(code, job_id)
    for index in range(3):
        job["delivery_retry_after"] = 0
        delivery = code.dispatch({"command": "code_completion_claim", "code_session_id": session["id"]})["delivery"]
        code.dispatch({"command": "code_completion_complete", "delivery_id": job_id, "claim_id": delivery["claim_id"], "success": False, "attempted": True})
    assert job["delivery_status"] == "blocked"
    assert code.dispatch({"command": "code_completion_claim", "code_session_id": session["id"]})["delivery"] is None
    code.dispatch({"command": "code_completion_retry", "delivery_id": job_id})
    assert job["delivery_status"] == "pending"
    assert job["delivery_failures"] == 0


def test_cancel_only_owned_process_group(code, tmp_path, monkeypatch):
    monkeypatch.setattr(desktop_code, "code_command", lambda *_args: [sys.executable, "-c", 'import time; print("started",flush=True); time.sleep(30)'])
    session = create(code, tmp_path)
    job_id = code.dispatch({"command": "code_start", "code_session_id": session["id"], "prompt": "Synthetic"})["job_id"]
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and code.jobs[job_id].get("child") is None:
        time.sleep(.01)
    child = code.jobs[job_id]["child"]
    code.dispatch({"command": "code_stop_loop", "code_session_id": session["id"]})
    assert wait(code, job_id)["status"] == "cancelled"
    assert child.poll() is not None


def test_pending_code_result_yields_until_automatic_callback(code, tmp_path, monkeypatch):
    release = threading.Event()
    def work(job, binary):
        release.wait(3)
        job['status'] = 'completed'
    monkeypatch.setattr(code, '_run', work)
    session = create(code, tmp_path)
    tools = context(code, session)
    request = {'command': 'code_tool_call', 'session_id': tools['session_id']}
    receipt = code.dispatch({**request, 'name': 'mlx2__code_run', 'arguments': {'prompt': 'Synthetic'}, 'approved': True})
    job_id = receipt['result']['job_id']
    read = {**request, 'name': 'mlx2__code_result', 'arguments': {'job_id': job_id}}
    try:
        assert code.dispatch(read)['native_action'] == 'wait_for_code'
    finally:
        release.set()
        code.jobs[job_id]['thread'].join(3)
    assert 'native_action' not in code.dispatch(read)


def test_exited_cli_does_not_wait_for_inherited_grandchild_stdout(code, tmp_path, monkeypatch):
    monkeypatch.setattr(desktop_code, "code_command", lambda *_args: [sys.executable, "-c",
        'import subprocess,sys; subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"]); print("done", flush=True)'])
    session = create(code, tmp_path)
    started = time.monotonic()
    job_id = code.dispatch({"command": "code_start", "code_session_id": session["id"], "prompt": "bounded"})["job_id"]
    job = wait(code, job_id)
    assert job["status"] == "completed" and time.monotonic() - started < 3

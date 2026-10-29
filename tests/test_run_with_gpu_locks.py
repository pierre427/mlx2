from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_with_gpu_locks.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_with_gpu_locks", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.observe_cpg = lambda *args: ({"cpg_task": "task", "cpg_session": "session", "cpg_generation": 1, "agent_id": "agent"}, {"status": "leased"})
    return module


def test_foreign_decision_server_blocks_gpu_admission(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module.os, "getpid", lambda: 10)
    monkeypatch.setattr(module.os, "getppid", lambda: 9)
    monkeypatch.setattr(
        module.subprocess, "check_output",
        lambda *args, **kwargs: (
            "11 /venv/bin/python -m mlx2.decisions.server --port 9000\n"
            "12 /venv/bin/python -m mlx2.top\n"
            "13 /venv/bin/python -m mlx2.server_helpers\n"
            "14 ssh gpu-host /venv/bin/python -m mlx2.server --port 9001\n"
            "15 /usr/bin/ssh gpu-host mlx_lm.server --port 9002\n"
            # A queued lock wrapper names its server command but owns no GPU.
            "16 /venv/bin/python /repo/scripts/run_with_gpu_locks.py --session s"
            " --label l --receipt r -- env /venv/bin/python -m mlx2.server --port 9003\n"
            # An agent CLI whose prompt quotes a server command line.
            "17 node /home/u/.local/bin/codex exec -s workspace-write 'run -m mlx2.server'\n"
        ),
    )
    assert module.foreign_model_processes() == [
        "11 /venv/bin/python -m mlx2.decisions.server --port 9000"
    ]


def test_agent_cli_prompt_that_quotes_a_server_does_not_block(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module.os, "getpid", lambda: 10)
    monkeypatch.setattr(module.os, "getppid", lambda: 9)
    monkeypatch.setattr(
        module.subprocess, "check_output",
        lambda *args, **kwargs: (
            "21 node /Users/x/.local/bin/codex exec -m gpt -s workspace-write "
            "Update docs: start python -m mlx2.server --model M\n"
            "22 /opt/homebrew/bin/node /usr/local/bin/claude -p 'run mlx_lm.server'\n"
            "23 /Users/x/.local/bin/codex exec 'python -m mlx2.server --port 1'\n"
            "24 /venv/bin/python -m mlx2.server --model M --port 8757\n"
        ),
    )
    assert module.foreign_model_processes() == [
        "24 /venv/bin/python -m mlx2.server --model M --port 8757"
    ]


def test_exclusive_receipt_refuses_to_overwrite(tmp_path: Path) -> None:
    module = _load_module()
    receipt = tmp_path / "attempt.json"

    module._write_json_exclusive(receipt, {"attempt_id": "first"})

    with pytest.raises(FileExistsError):
        module._write_json_exclusive(receipt, {"attempt_id": "second"})
    assert json.loads(receipt.read_text()) == {"attempt_id": "first"}


def test_terminal_receipt_is_written_before_legacy_receipt(monkeypatch) -> None:
    module = _load_module()
    calls = []
    monkeypatch.setattr(
        module,
        "_write_json_exclusive",
        lambda path, payload: calls.append(("immutable", path, payload.copy())),
    )
    monkeypatch.setattr(
        module,
        "_atomic_write_json",
        lambda path, payload: calls.append(("legacy", path, payload.copy())),
    )

    module._save_terminal(Path("latest.json"), Path("attempt.json"), {"status": "done"})

    assert [kind for kind, _, _ in calls] == ["immutable", "legacy"]


def test_retries_keep_immutable_attempts_and_update_legacy(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_module()
    waiters = tmp_path / "waiters"
    locks = (tmp_path / "shared.lock", tmp_path / "tmp.lock")
    for lock in locks:
        lock.touch()
    monkeypatch.setattr(module, "WAITERS", waiters)
    monkeypatch.setattr(module, "LOCKS", locks)
    monkeypatch.setattr(module, "foreign_model_processes", list)

    class FakeChild:
        pid = 4242
        returncode = 0

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: FakeChild())
    attempt_ids = iter(("a" * 32, "b" * 32))
    monkeypatch.setattr(
        module.uuid, "uuid4", lambda: SimpleNamespace(hex=next(attempt_ids))
    )
    legacy = tmp_path / "run.json"
    monkeypatch.setenv("CPG_OWNER_LOCK", str(tmp_path / "cpg.lock"))
    monkeypatch.setenv("CPG_TASK", "task")
    argv = [
        "--session",
        "session-1",
        "--label",
        "cell-1",
        "--receipt",
        str(legacy),
        "--",
        "ignored-command",
    ]

    assert module.main(argv) == 0
    first = json.loads(legacy.read_text())
    assert module.main(argv) == 0
    second = json.loads(legacy.read_text())

    assert first["logical_run_id"] == second["logical_run_id"] == "session-1:cell-1"
    assert first["attempt_id"] == "a" * 32
    assert second["attempt_id"] == "b" * 32
    first_immutable = Path(first["immutable_receipt"])
    second_immutable = Path(second["immutable_receipt"])
    assert first_immutable != second_immutable
    assert json.loads(first_immutable.read_text()) == first
    assert json.loads(second_immutable.read_text()) == second
    assert json.loads(legacy.read_text()) == second
    assert not list(tmp_path.glob(".*.tmp.*"))


@pytest.mark.parametrize("signum", [15, 1])
def test_signal_stops_owned_child_before_restoring_locks(tmp_path, signum):
    import os
    import subprocess
    import sys
    import time

    locks = (tmp_path / "shared.lock", tmp_path / "tmp.lock")
    for lock in locks:
        lock.touch()
    ready = tmp_path / "child-ready"
    receipt = tmp_path / "receipt.json"
    child_code = (
        "import os,time; from pathlib import Path; "
        f"Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    bootstrap = f"""
import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("wrapper", {str(SCRIPT)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.LOCKS = tuple(Path(p) for p in {tuple(map(str, locks))!r})
module.WAITERS = Path({str(tmp_path / 'waiters')!r})
module.foreign_model_processes = list
module.observe_cpg = lambda *args: (dict(cpg_task="task", cpg_session="session", cpg_generation=1, agent_id="agent"), dict(status="leased"))
raise SystemExit(module.main([
    "--cpg-owner-lock", "unused", "--cpg-task", "task",
    "--session", "cpu-signal-test", "--label", "cleanup", "--receipt", {str(receipt)!r},
    "--", sys.executable, "-c", {child_code!r},
]))
"""
    wrapper = subprocess.Popen([sys.executable, "-c", bootstrap])
    child_pid = None
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and wrapper.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), "CPU child did not start"
        child_pid = int(ready.read_text())
        os.kill(wrapper.pid, signum)
        assert wrapper.wait(timeout=15) == 130
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        assert all(lock.is_file() and lock.stat().st_size == 0 for lock in locks)
        assert not list((tmp_path / "waiters").iterdir())
        recorded = json.loads(receipt.read_text())
        assert recorded["status"] == "interrupted"
        assert recorded["child_terminated_by_wrapper"] is True
        assert recorded["command_returncode"] < 0
    finally:
        if wrapper.poll() is None:
            wrapper.kill()
            wrapper.wait(timeout=10)
        if child_pid is not None:
            try:
                os.kill(child_pid, 9)
            except ProcessLookupError:
                pass


def test_long_queued_live_waiter_keeps_its_place(tmp_path, monkeypatch):
    import os
    import time

    module = _load_module()
    waiters = tmp_path / "waiters"
    waiters.mkdir()
    monkeypatch.setattr(module, "WAITERS", waiters)
    now = int(time.time())
    # Queued two hours ago by a live process and touched recently.
    old = waiters / f"lane-a.label.{os.getpid()}.{now - 7200}"
    old.touch()
    mine = waiters / f"lane-b.label.{os.getpid()}.{now}"
    mine.touch()
    assert module.earlier_waiter(mine, now) == old.name
    # An abandoned file (stale mtime) is still skipped.
    os.utime(old, (now - 600, now - 600))
    assert module.earlier_waiter(mine, now) is None


def test_gpu_wrapper_requires_explicit_cpg_owner_and_task(tmp_path):
    module = _load_module()
    with pytest.raises(SystemExit):
        module.main(["--label", "test", "--receipt", str(tmp_path / "r.json"), "--", "unused"])
    assert not (tmp_path / "r.json").exists()


def test_desktop_lease_matches_ancestor_job_and_live_radio(tmp_path, monkeypatch):
    import os
    import time

    from mlx2 import desktop_campaign_worker
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    import qualification_gpu_ownership
    monkeypatch.setattr(qualification_gpu_ownership, '_ancestor_pids', lambda pid: {os.getpid()})
    expiry = time.time() + 180
    owner = {'agent':'mlx2_desktop_campaign', 'pid':os.getpid(), 'job_id':'job',
             'cpg_task':'task', 'cpg_session':'session', 'cpg_generation':1,
             'agent_id':'agent', 'expires':expiry,
             'claim':{'claimed':True, 'task_id':'task', 'lease_generation':1,
                      'owner_agent_id':'agent', 'lease_expires_at':expiry}}
    path = tmp_path / 'desktop-lease.json'; path.write_text(json.dumps(owner))
    payload = {'status':'in_progress', 'owner_agent_id':'agent', 'lease_generation':1,
               'lease_expires_at':expiry}
    class Radio:
        def __init__(self, url): pass
        def call(self, name, arguments):
            if name == 'join_session': return {'session_id':'session'}
            return {'nodes':[{'id':'task', 'session_id':'session', 'type':'TASK', 'payload':payload}]}
    monkeypatch.setattr(desktop_campaign_worker, 'Radio', Radio)
    module = _load_module()
    observed, live = module.observe_desktop_lease(path, 'session', 'job', 'unused')
    assert observed == owner and live['lease_generation'] == 1
    with pytest.raises(RuntimeError, match='ancestor/job'):
        module.observe_desktop_lease(path, 'session', 'foreign-job', 'unused')
    payload['owner_agent_id'] = 'reassigned'
    with pytest.raises(RuntimeError, match='owner'):
        module.observe_desktop_lease(path, 'session', 'job', 'unused')


def test_signal_stops_a_grandchild_in_its_own_session(tmp_path):
    # 2026-10-29: campaign.py died on the wrapper's SIGTERM while run_profile
    # and its server, each started with start_new_session, kept running
    # unlocked.  killpg on the child's group cannot reach them.
    import os
    import subprocess
    import sys
    import time

    locks = (tmp_path / "shared.lock", tmp_path / "tmp.lock")
    for lock in locks:
        lock.touch()
    ready = tmp_path / "grandchild-ready"
    receipt = tmp_path / "receipt.json"
    grandchild_code = (
        "import os,time; from pathlib import Path; "
        f"Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    child_code = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable, '-c', {grandchild_code!r}], start_new_session=True); "
        "time.sleep(60)"
    )
    bootstrap = f"""
import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("wrapper", {str(SCRIPT)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.LOCKS = tuple(Path(p) for p in {tuple(map(str, locks))!r})
module.WAITERS = Path({str(tmp_path / 'waiters')!r})
module.foreign_model_processes = list
module.observe_cpg = lambda *args: (dict(cpg_task="task", cpg_session="session", cpg_generation=1, agent_id="agent"), dict(status="leased"))
raise SystemExit(module.main([
    "--cpg-owner-lock", "unused", "--cpg-task", "task",
    "--session", "cpu-grandchild-test", "--label", "cleanup", "--receipt", {str(receipt)!r},
    "--", sys.executable, "-c", {child_code!r},
]))
"""
    wrapper = subprocess.Popen([sys.executable, "-c", bootstrap])
    grandchild_pid = None
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and wrapper.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), "CPU grandchild did not start"
        grandchild_pid = int(ready.read_text())
        os.kill(wrapper.pid, 15)
        assert wrapper.wait(timeout=60) == 130
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(grandchild_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        with pytest.raises(ProcessLookupError):
            os.kill(grandchild_pid, 0)
        recorded = json.loads(receipt.read_text())
        assert grandchild_pid in recorded["child_descendants_signalled"]
        assert "child_descendants_surviving" not in recorded
    finally:
        if wrapper.poll() is None:
            wrapper.kill()
            wrapper.wait(timeout=10)
        if grandchild_pid is not None:
            try:
                os.kill(grandchild_pid, 9)
            except ProcessLookupError:
                pass


def _watch_run(monkeypatch, tmp_path, observations, grace=None):
    """Run main in-process: the child waits three 10 s ticks, then exits 0.

    ``observations`` yields the result (or exception) of each ownership
    observation; the first is the pre-acquire check.
    """
    import time as real_time

    module = _load_module()
    if grace is not None:
        monkeypatch.setattr(module, "OBSERVE_TRANSIENT_GRACE_SECONDS", grace)
    locks = (tmp_path / "shared.lock", tmp_path / "tmp.lock")
    for lock in locks:
        lock.touch()
    monkeypatch.setattr(module, "WAITERS", tmp_path / "waiters")
    monkeypatch.setattr(module, "LOCKS", locks)
    monkeypatch.setattr(module, "foreign_model_processes", list)
    stopped = []
    monkeypatch.setattr(module, "stop_child", lambda child, receipt: stopped.append(child))

    class FakeChild:
        pid = 4242
        returncode = None
        ticks = 3

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            if self.ticks:
                self.ticks -= 1
                raise module.subprocess.TimeoutExpired("child", timeout)
            self.returncode = 0
            return 0

    child = FakeChild()
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: child)
    owner = dict(cpg_task="task", cpg_session="session", cpg_generation=1, agent_id="agent")
    lease = dict(status="leased", lease_expires_at=real_time.time() + 600)
    results = iter(observations)

    def observe(*args):
        result = next(results)
        if isinstance(result, BaseException):
            raise result
        return owner, lease

    monkeypatch.setattr(module, "observe_cpg", observe)
    receipt = tmp_path / "receipt.json"
    rc = module.main([
        "--cpg-owner-lock", "unused", "--cpg-task", "task",
        "--session", "watch-test", "--label", "cell", "--receipt", str(receipt),
        "--", "ignored-command",
    ])
    return module, rc, json.loads(receipt.read_text()), stopped


def test_transient_radio_timeout_does_not_tear_down_the_unit(monkeypatch, tmp_path):
    _, rc, recorded, _ = _watch_run(
        monkeypatch, tmp_path, [None, TimeoutError("radio"), None, None])
    assert rc == 0
    assert recorded["status"] == "command_completed"
    assert [e["error"] for e in recorded["transient_observe_errors"]] == ["TimeoutError: radio"]


def test_ownership_verdict_is_never_treated_as_transient(monkeypatch, tmp_path):
    _, rc, recorded, stopped = _watch_run(
        monkeypatch, tmp_path, [None, RuntimeError("live CPG lease is expired or malformed")])
    assert rc != 0
    assert recorded["status"] == "error"
    assert "RuntimeError" in recorded["error"] and stopped


def test_transport_failure_past_the_grace_window_stops_the_unit(monkeypatch, tmp_path):
    _, rc, recorded, stopped = _watch_run(
        monkeypatch, tmp_path, [None, TimeoutError("radio")], grace=-1)
    assert rc != 0
    assert recorded["error"] == "TimeoutError: radio" and stopped

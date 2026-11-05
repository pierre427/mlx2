"""Synthetic service/ownership failure tests; no MLX, launchctl, network or GPU."""
from __future__ import annotations

import copy
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import plistlib
import subprocess
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "scripts/research/kimi_lora_owned.py"
spec = importlib.util.spec_from_file_location("kimi_lora_owned_test", MODULE)
owned = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owned)


class FakeHost:
    def __init__(self, tmp_path):
        self.clock = 0
        self.pid = 100
        self.loaded = True
        self.actions = []
        self.busy = False
        self.foreign = False
        self.lifecycle = "serving"
        self.fail_quiesce = False
        self.fail_bootout = False
        self.policy = {"kda": "kernel", "mla": "auto"}
        self.guards = SimpleNamespace(LOCKS=(), stop_child=lambda *a: None)
        self.plist = tmp_path / "service.plist"
        self.model = tmp_path / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text("{}")
        self.args = ["python", "-m", "mlx2.server", "--model", str(self.model), "--host", "127.0.0.1", "--port", "8282"]
        self.plist.write_bytes(plistlib.dumps({"Label": owned.LABEL, "ProgramArguments": self.args, "WorkingDirectory": str(ROOT)}))
        self.spec = {"model": str(self.model), "checkpoint": "model@" + "a" * 64,
                     "source_revision": "b" * 40, "job_id": "test", "out": str(tmp_path / "out"),
                     "command": ["python", "training.py"], "timeout_seconds": 2,
                     "restore_timeout_seconds": 5, "mcp_url": "unused",
                     "service": {"label": owned.LABEL, "port": 8282, "plist": str(self.plist)}}

    def launchd(self, service):
        return {"state": "running", "pid": self.pid, "path": str(self.plist), "argv": self.args} if self.loaded else None

    def process(self, pid):
        if not self.loaded or pid != self.pid:
            return None
        return {"pid": pid, "argv": self.args, "uid": os.getuid(), "cwd": str(ROOT), "created": pid * 1.0}

    def listener_pids(self, port):
        return [999] if self.foreign else [self.pid] if self.loaded else []

    def request(self, service, method="GET", path="/v1/status", body=None):
        if method == "POST":
            self.actions.append((path, body))
            if path.endswith("quiesce"):
                self.lifecycle = "suspended"
                if self.fail_quiesce:
                    raise OSError("response lost after accepted quiesce")
            else:
                self.lifecycle = "serving"
            return {}
        return {"model": "model", "artifact": "a" * 64, "base_artifact": "a" * 64, "profile": "ordinary",
                "settings": {"max_lanes": 4}, "runtime": {"source_sha256": "source"},
                "execution": {"route": "ordinary", "policy": self.policy, "artifact": {"checkpoint": "repo@rev"}},
                "healthy": True, "state": "ready", "inflight": int(self.busy), "queue_depth": 0,
                "accepted_lifecycles": 0, "quiesce": {"state": self.lifecycle, "terminal_cleanup_pending": 0},
                "lora": {"enabled": False}, "multi_lora": {"enabled": False}}

    def idle(self, allowed_pid=None):
        if self.foreign:
            raise RuntimeError("foreign process")
        return {}

    @contextlib.contextmanager
    def transition_guard(self, allowed_pid=None):
        self.idle(allowed_pid)
        yield {"synthetic_flocks": True}

    def action(self, action, service):
        self.actions.append(action)
        if action == "bootout":
            if self.fail_bootout:
                raise OSError("bootout failed before action")
            self.loaded = False
        else:
            self.loaded = True
            self.pid = 101
            self.lifecycle = "serving"

    def sleep(self, seconds):
        self.clock += seconds

    def now(self):
        return self.clock

    def telemetry(self):
        return {"synthetic": True}


@pytest.fixture
def host(tmp_path):
    return FakeHost(tmp_path)


def test_exact_service_pauses_and_restores_without_rewriting_plist(host):
    before = host.plist.read_bytes()
    service = owned.ManagedService(host.spec, host)
    service.preflight()
    service.pause(lambda: None)
    assert not host.loaded
    service.restore()
    assert host.loaded and host.pid == 101
    assert host.plist.read_bytes() == before
    assert service.record["restored"] is True
    assert host.actions == [("/v1/admin/quiesce", {"drain_timeout_seconds": 120, "suspend": True}), "bootout", "bootstrap"]


@pytest.mark.parametrize("change", ["busy", "foreign", "policy", "checkpoint", "argv", "loaded"])
def test_unverifiable_service_never_mutates(host, change):
    if change in ("busy", "foreign"):
        setattr(host, change, True)
    elif change == "policy":
        host.spec["service"]["expected_policy"] = {"kda": "ops"}
    elif change == "checkpoint":
        host.spec["checkpoint"] = "model@" + "c" * 64
    elif change == "argv":
        host.args = ["foreign"]
    else:
        host.loaded = False
    with pytest.raises(RuntimeError):
        owned.ManagedService(host.spec, host).preflight()
    assert host.actions == []


def test_lost_quiesce_response_resumes_original_pid(host):
    service = owned.ManagedService(host.spec, host)
    service.preflight()
    host.fail_quiesce = True
    with pytest.raises(OSError):
        service.pause(lambda: None)
    service.restore()
    assert host.lifecycle == "serving" and host.pid == 100
    assert "bootout" not in host.actions
    assert service.record["restored"]


def test_bootout_failure_resumes_original_pid(host):
    service = owned.ManagedService(host.spec, host)
    service.preflight()
    host.fail_bootout = True
    with pytest.raises(OSError):
        service.pause(lambda: None)
    service.restore()
    assert host.lifecycle == "serving" and host.pid == 100
    assert "bootstrap" not in host.actions


def test_plist_change_refuses_restoration_and_never_overwrites(host):
    service = owned.ManagedService(host.spec, host)
    service.preflight()
    service.pause(lambda: None)
    changed = host.plist.read_bytes() + b"\n"
    host.plist.write_bytes(changed)
    with pytest.raises(RuntimeError, match="plist changed"):
        service.restore()
    assert host.plist.read_bytes() == changed
    assert "bootstrap" not in host.actions


def test_changed_process_is_not_resumed_or_booted_out(host):
    service = owned.ManagedService(host.spec, host)
    service.preflight()
    host.fail_quiesce = True
    with pytest.raises(OSError):
        service.pause(lambda: None)
    host.pid = 999
    with pytest.raises(RuntimeError, match="identity or policy changed"):
        service.restore()
    assert len(host.actions) == 1


def test_lease_loss_before_pause_leaves_service_untouched(host):
    service = owned.ManagedService(host.spec, host)
    service.preflight()
    def lost():
        raise RuntimeError("lease expired")
    with pytest.raises(RuntimeError, match="lease expired"):
        service.pause(lost)
    service.restore()
    assert host.actions == []


def test_live_waiter_blocks_even_empty_file_sentinels(tmp_path, monkeypatch):
    lock = tmp_path / "gpu.lock"
    lock.touch()
    guards = SimpleNamespace(LOCKS=(lock,), snapshot=lambda p: {"kind": "file", "size": 0}, foreign_model_processes=lambda: [])
    host = owned.Host(guards)
    monkeypatch.setattr(host, "live_waiters", lambda: ["earlier.job.1.1"])
    with pytest.raises(RuntimeError, match="live GPU waiters"):
        host.idle()


def test_flock_held_empty_sentinel_is_busy(tmp_path, monkeypatch):
    lock = tmp_path / "gpu.lock"
    lock.touch()
    guards = SimpleNamespace(LOCKS=(lock,), snapshot=lambda p: {"kind": "file", "size": 0}, foreign_model_processes=lambda: [])
    host = owned.Host(guards)
    monkeypatch.setattr(host, "live_waiters", list)
    with lock.open("r+") as held:
        owned.fcntl.flock(held, owned.fcntl.LOCK_EX | owned.fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="flock-held"):
            host.idle()


def test_transition_guard_holds_both_sentinels_until_action_finishes(tmp_path, monkeypatch):
    locks = (tmp_path / "shared.lock", tmp_path / "tmp.lock")
    for path in locks:
        path.touch()
    guards = SimpleNamespace(LOCKS=locks, snapshot=lambda p: {"kind": "file", "size": 0}, foreign_model_processes=lambda: [])
    host = owned.Host(guards)
    monkeypatch.setattr(host, "live_waiters", list)
    with host.transition_guard():
        assert host.idle()["transition_guard_held"]
        for path in locks:
            with path.open("r+") as rival:
                with pytest.raises(BlockingIOError):
                    owned.fcntl.flock(rival, owned.fcntl.LOCK_EX | owned.fcntl.LOCK_NB)
    with host.transition_guard():
        pass


def test_restore_allows_new_requests_after_admission_reopens(host):
    service = owned.ManagedService(host.spec, host)
    service.preflight()
    service.pause(lambda: None)
    host.busy = True
    service.restore()
    assert service.record["restored"]


class FakeLease:
    def __init__(self, *args, **kwargs):
        self.owner = {"cpg_session": "session"}
        self.success = False
        self.closed = False
        self.calls = 0
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.closed = True
    def check(self):
        self.calls += 1


class FakeChild:
    pid = 123
    def __init__(self, host, out, *, hanging=False, missing_receipt=False):
        self.host = host
        self.out = out
        self.returncode = None if hanging else 0
        self.signals = []
        self.missing_receipt = missing_receipt
        if not hanging:
            self.finish()
    def poll(self):
        return self.returncode
    def finish(self):
        if not self.missing_receipt:
            owned.atomic(self.out / "paired-locks.json", {"pid": self.pid, "finished_at": 1, "status": "command_completed"})
    def send_signal(self, sig):
        self.signals.append(sig)
        self.returncode = -sig
        self.finish()
    def wait(self, timeout=None):
        if self.returncode is None:
            self.host.clock += timeout
            raise subprocess.TimeoutExpired("fake", timeout)
        return self.returncode


@pytest.mark.parametrize("hanging,expected", [(False, 0), (True, 1)])
def test_round_success_and_timeout_both_restore_after_child_release(host, hanging, expected):
    lease = FakeLease()
    children = []
    def spawn(*args, **kwargs):
        child = FakeChild(host, Path(host.spec["out"]), hanging=hanging)
        children.append(child)
        return child
    code = owned.execute(host.spec, host=host, lease_factory=lambda *a, **k: lease, spawn=spawn)
    assert code == expected
    assert host.loaded and host.lifecycle == "serving"
    assert lease.closed
    assert lease.success is (not hanging)
    receipt = json.loads((Path(host.spec["out"]) / "ownership.json").read_text())
    assert receipt["service_restored"]
    assert receipt["paired_locks"]["pid"] == 123
    if hanging:
        assert children[0].signals == [owned.signal.SIGTERM]
        assert "TimeoutError" in receipt["error"]


def test_missing_terminal_lock_receipt_never_loads_second_model(host):
    def spawn(*args, **kwargs):
        return FakeChild(host, Path(host.spec["out"]), missing_receipt=True)
    assert owned.execute(host.spec, host=host, lease_factory=FakeLease, spawn=spawn) == 1
    assert not host.loaded
    record = json.loads((Path(host.spec["out"]) / "ownership.json").read_text())
    assert "restoration deferred" in record["restore_error"]
    assert "bootstrap" not in host.actions


def test_nonempty_output_refuses_before_service_actions(host):
    out = Path(host.spec["out"])
    out.mkdir()
    (out / "prior.json").write_text("{}")
    with pytest.raises(ValueError, match="nonempty"):
        owned.execute(host.spec, host=host)
    assert host.actions == []


def test_atomic_receipt_replaces_complete_json(tmp_path):
    path = tmp_path / "receipt.json"
    owned.atomic(path, {"first": True})
    owned.atomic(path, {"second": True})
    assert json.loads(path.read_text()) == {"second": True}
    assert list(tmp_path.iterdir()) == [path]

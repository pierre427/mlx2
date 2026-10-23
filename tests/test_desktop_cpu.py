"""Desktop control-plane tests using synthetic folders and CPU-only workers.

No model checkpoints are loaded and these are not model qualification results.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from mlx2 import desktop


def test_occupied_sticky_port_reports_actionable_error_without_disturbing_listener():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        with pytest.raises(ValueError, match=f"port {port}.*already in use"):
            desktop.serving_port({"port": port, "sticky_port": True})
        assert desktop.serving_port({"port": port, "sticky_port": False}) != port
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            connection, _ = listener.accept()
            connection.close()


def test_sticky_port_can_restart_after_recent_connection_closed():
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        # Reusable serving sockets must still reject another live listener.
        with pytest.raises(ValueError, match="already in use"):
            desktop.serving_port({"port": port})
        with socket.create_connection(("127.0.0.1", port), timeout=1) as client:
            peer, _ = listener.accept()
            peer.shutdown(socket.SHUT_WR)
            peer.close()
            assert client.recv(1) == b""
    assert desktop.serving_port({"port": port}) == port


@pytest.fixture
def artifact(tmp_path):
    root = tmp_path / "models with spaces"
    root.mkdir()
    model = root / "synthetic fixture"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "synthetic"}))
    (model / "fixture.bin").write_bytes(b"x" * 100)
    return root, model


def wait_scan(library):
    deadline = time.monotonic() + 5
    while library.snapshot()["scanning"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not library.snapshot()["scanning"]
    return library.snapshot()["rows"]


def test_inventory_preserves_timestamps_and_never_follows_foreign_links(
    artifact, tmp_path
):
    _, model = artifact
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "large.bin").write_bytes(b"z" * 10000)
    (model / "linked-dir").symlink_to(outside, target_is_directory=True)
    (model / "linked-weight").symlink_to(outside / "large.bin")
    os.link(model / "fixture.bin", model / "duplicate.bin")
    before = model.stat()
    row = desktop.artifact_row(model)
    assert row["logical_bytes"] == 100 + (model / "config.json").stat().st_size
    assert row["atime"] == before.st_atime
    assert row["mtime"] == before.st_mtime
    assert row["ctime"] == before.st_ctime


def test_draft_and_media_are_not_standalone_targets(artifact):
    _, model = artifact
    (model / "config.json").write_text(
        json.dumps({"model_type": "x", "dflash_config": {}})
    )
    assert desktop.artifact_row(model)["kind"] == "Draft"
    assert not desktop.artifact_row(model)["can_start"]
    (model / "config.json").unlink()
    (model / "model_index.json").write_text("{}")
    assert desktop.artifact_row(model)["kind"] == "Media"
    assert not desktop.artifact_row(model)["can_start"]


def test_library_rejects_stale_identity_symlink_and_path_escape(artifact, tmp_path):
    root, model = artifact
    library = desktop.Library(root)
    row = wait_scan(library)[0]
    assert library.checked_path(str(model), row["identity"]) == model
    (model / "new-file").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        library.checked_path(str(model), row["identity"])
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    link = root / "link"
    link.symlink_to(foreign, target_is_directory=True)
    with pytest.raises(ValueError):
        library.checked_path(str(link), desktop.artifact_row(link)["identity"])
    with pytest.raises(ValueError):
        library.checked_path(str(foreign), desktop.artifact_row(foreign)["identity"])


def test_library_replacement_cancels_old_scan(artifact, tmp_path):
    root, _ = artifact
    other = tmp_path / "empty"
    other.mkdir()
    library = desktop.Library(root)
    library.refresh(other)
    assert wait_scan(library) == []
    assert library.snapshot()["root"] == str(other)


def reservation(tmp_path):
    return desktop.GPUReservation(
        (tmp_path / "a.lock", tmp_path / "b.lock"), tmp_path / "waiters"
    )


def test_paired_reservation_restores_sentinels_and_refuses_existing_owner(tmp_path):
    lease = reservation(tmp_path)
    for path in lease.locks:
        path.touch()
    lease.acquire()
    assert [json.loads((p / "owner.json").read_text()) for p in lease.locks] == [
        lease.owner
    ] * 2
    with pytest.raises(RuntimeError, match="reserved"):
        reservation(tmp_path).acquire()
    lease.release()
    assert all(p.is_file() and p.stat().st_size == 0 for p in lease.locks)
    assert not list(tmp_path.glob("*.desktop-*"))


def test_partial_reservation_rolls_back_on_receipt_failure(tmp_path, monkeypatch):
    lease = reservation(tmp_path)
    for path in lease.locks:
        path.touch()
    original = desktop.atomic_json

    def fail_second(path, value):
        if path.parent == lease.locks[1]:
            raise OSError("synthetic write failure")
        original(path, value)

    monkeypatch.setattr(desktop, "atomic_json", fail_second)
    with pytest.raises(OSError, match="synthetic"):
        lease.acquire()
    assert all(p.is_file() and p.stat().st_size == 0 for p in lease.locks)


def test_queued_work_and_flock_owner_block_interactive_start(tmp_path):
    lease = reservation(tmp_path)
    lease.waiters.mkdir()
    waiter = lease.waiters / f"session.label.{os.getpid()}.{int(time.time())}"
    waiter.touch()
    with pytest.raises(RuntimeError, match="queued"):
        lease.acquire()
    assert not any(p.exists() for p in lease.locks)
    waiter.unlink()
    lease.locks[0].touch()
    import fcntl

    with lease.locks[0].open("r+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            lease.acquire()
    assert lease.locks[0].is_file()


def test_reservation_never_removes_foreign_receipt(tmp_path):
    lease = reservation(tmp_path)
    lease.acquire()
    foreign = {"lease_id": "someone-else"}
    (lease.locks[0] / "owner.json").write_text(json.dumps(foreign))
    with pytest.raises(RuntimeError, match="owner changed"):
        lease.release()
    assert json.loads((lease.locks[0] / "owner.json").read_text()) == foreign


def test_exit_observation_preserves_pid_until_descendants_are_stopped():
    code = "import subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)']); print(p.pid, flush=True)"
    child = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    descendant = int(child.stdout.readline())
    try:
        deadline = time.monotonic() + 5
        while not desktop.child_exited(child) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert desktop.child_exited(child)
        assert child.returncode is None  # no poll/wait: PID cannot be reused
        assert desktop.stop_group(child) == 0
        deadline = time.monotonic() + 2
        while psutil.pid_exists(descendant) and time.monotonic() < deadline:
            if psutil.Process(descendant).status() == psutil.STATUS_ZOMBIE:
                break
            time.sleep(0.01)
        assert (
            not psutil.pid_exists(descendant)
            or psutil.Process(descendant).status() == psutil.STATUS_ZOMBIE
        )
    finally:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        child.wait()


class FakeCollector:
    def __init__(self, **kwargs):
        self.closed = False

    def sample(self):
        return ()

    def close(self):
        self.closed = True


@pytest.fixture
def supervisor(artifact, tmp_path, monkeypatch):
    root, _ = artifact
    home = tmp_path / "app-home"
    home.mkdir()
    desktop.atomic_json(
        home / "desktop-settings.json",
        {
            "root": str(root),
            "port": 18285,
            "context": 8192,
            "lanes": 1,
            "route": "auto",
        },
    )
    monkeypatch.setattr(desktop, "HostCollector", FakeCollector)
    monkeypatch.setattr(desktop, "InferenceCollector", FakeCollector)
    monkeypatch.setattr(desktop, "model_processes", list)
    supervisor = desktop.Supervisor(home, tmp_path / "trash-helper")
    supervisor.runner_manager = SimpleNamespace(
        ensure=lambda *args, **kwargs: {
            "python": sys.executable,
            "source": str(Path.cwd()),
            "binding": {"test": "synthetic"},
        }
    )
    wait_scan(supervisor.library)
    yield supervisor
    supervisor.close()


def test_settings_reject_unknown_flags_and_non_loopback_profile(supervisor):
    with pytest.raises(ValueError):
        supervisor.configure({**supervisor.settings, "host": "0.0.0.0"})
    with pytest.raises(ValueError):
        supervisor.configure({**supervisor.settings, "port": 80})
    with pytest.raises(ValueError):
        supervisor.configure({**supervisor.settings, "lanes": True})


def test_duplicate_supervisor_cannot_take_ownership(supervisor):
    with pytest.raises(RuntimeError, match="already running"):
        desktop.Supervisor(supervisor.home, supervisor.trash_helper)


def test_trash_checks_identity_and_refuses_external_or_open_users(
    supervisor, monkeypatch
):
    row = supervisor.library.snapshot()["rows"][0]
    monkeypatch.setattr(
        desktop, "model_processes", lambda: [{"pid": 123, "model": row["path"]}]
    )
    with pytest.raises(ValueError, match="external"):
        supervisor.trash(row["path"], row["identity"])
    monkeypatch.setattr(desktop, "model_processes", list)
    calls = []

    def lsof(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout="open weights", stderr="")

    monkeypatch.setattr(desktop.subprocess, "run", lsof)
    with pytest.raises(ValueError, match="open files"):
        supervisor.trash(row["path"], row["identity"])
    assert len(calls) == 1 and calls[0][0] == "/usr/sbin/lsof"
    assert Path(row["path"]).exists()


def test_trash_uses_native_helper_without_shell(supervisor, monkeypatch):
    row = supervisor.library.snapshot()["rows"][0]
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(
            returncode=1 if len(calls) == 1 else 0, stdout="", stderr=""
        )

    monkeypatch.setattr(desktop.subprocess, "run", run)
    supervisor.trash(row["path"], row["identity"])
    assert calls[1] == [
        str(supervisor.trash_helper),
        "--trash",
        row["path"],
        row["identity"],
    ]


def test_nonfinite_telemetry_serializes_as_missing():
    assert (
        json.dumps(desktop.json_safe({"metric": float("nan"), "other": float("inf")}))
        == '{"metric": null, "other": null}'
    )


def test_argv_model_discovery_handles_spaces_and_ignores_prompts(monkeypatch):
    processes = [
        SimpleNamespace(
            info={
                "pid": 1,
                "cmdline": [
                    "/path with spaces/python",
                    "-m",
                    "mlx2.server",
                    "--model",
                    "/models/a b",
                ],
            }
        ),
        SimpleNamespace(
            info={"pid": 2, "cmdline": ["codex", "exec", "start -m mlx2.server"]}
        ),
    ]
    monkeypatch.setattr(psutil, "process_iter", lambda *args: processes)
    assert desktop.model_processes() == [{"pid": 1, "model": "/models/a b"}]


def synthetic_resolution(monkeypatch, execution="autoregressive"):
    from mlx2.adapters import registry

    monkeypatch.setattr(
        registry,
        "inspect_model",
        lambda path: SimpleNamespace(
            descriptor=SimpleNamespace(
                metadata={"execution_kind": execution}, capabilities=frozenset()
            ),
            artifact_fingerprint="synthetic-fixture-not-a-model",
        ),
    )


def test_start_refuses_nonserving_execution_and_foreign_gpu_users(
    supervisor, monkeypatch
):
    row = supervisor.library.snapshot()["rows"][0]
    synthetic_resolution(monkeypatch, "image_diffusion")
    with pytest.raises(ValueError, match="dedicated runtime"):
        supervisor.start(row["path"], row["identity"])
    synthetic_resolution(monkeypatch)
    monkeypatch.setattr(
        desktop, "model_processes", lambda: [{"pid": 123, "model": None}]
    )
    with pytest.raises(RuntimeError, match="external"):
        supervisor.start(row["path"], row["identity"])
    assert supervisor.child is None and supervisor.reservation is None


def test_managed_start_stop_auth_receipt_and_reservation_without_loading_a_model(
    supervisor, monkeypatch, tmp_path
):
    row = supervisor.library.snapshot()["rows"][0]
    synthetic_resolution(monkeypatch)
    lease = reservation(tmp_path)
    monkeypatch.setattr(desktop, "GPUReservation", lambda: lease)
    original_popen = subprocess.Popen
    calls = []

    def fake_worker(args, **kwargs):
        calls.append(args)
        return original_popen(
            [sys.executable, "-c", "import time; time.sleep(120)"], **kwargs
        )

    monkeypatch.setattr(
        desktop, "source_identity", lambda: {"source_revision": "synthetic"}
    )
    monkeypatch.setattr(desktop.subprocess, "Popen", fake_worker)
    supervisor.settings["route"] = "ordinary"
    worker = supervisor.start(row["path"], row["identity"])
    pid = supervisor.child.pid
    assert worker["qualification"] == "unqualified"
    assert "--ordinary" in calls[0]
    assert calls[0][calls[0].index("--model") + 1] == row["path"]
    assert worker["url"] == f"http://127.0.0.1:{calls[0][calls[0].index('--port') + 1]}"
    key = Path(worker["api_key_file"])
    assert key.stat().st_mode & 0o777 == 0o600
    assert len(key.read_text()) == 64
    receipt = json.loads((key.parent / "launch.json").read_text())
    assert receipt["source"]["source_revision"] == "synthetic"
    with pytest.raises(ValueError, match="current model"):
        supervisor.start(row["path"], row["identity"])
    with pytest.raises(ValueError, match="Stop"):
        supervisor.trash(row["path"], row["identity"])
    supervisor.stop()
    assert not psutil.pid_exists(pid)
    assert worker["state"] == "Stopped"
    assert not any(path.exists() for path in lease.locks)


def test_spawn_failure_returns_gpu_ownership(supervisor, monkeypatch, tmp_path):
    row = supervisor.library.snapshot()["rows"][0]
    synthetic_resolution(monkeypatch)
    lease = reservation(tmp_path)
    monkeypatch.setattr(desktop, "GPUReservation", lambda: lease)
    monkeypatch.setattr(desktop, "source_identity", dict)

    def fail(*args, **kwargs):
        raise OSError("synthetic spawn failure")

    monkeypatch.setattr(desktop.subprocess, "Popen", fail)
    with pytest.raises(OSError, match="spawn failure"):
        supervisor.start(row["path"], row["identity"])
    assert supervisor.child is None and supervisor.reservation is None
    assert not any(path.exists() for path in lease.locks)


def test_preparation_is_async_cancellable_and_does_not_reserve_gpu(
    supervisor, monkeypatch
):
    import threading

    row = supervisor.library.snapshot()["rows"][0]
    synthetic_resolution(monkeypatch)
    entered = threading.Event()

    def prepare(*args, cancel, **kwargs):
        entered.set()
        cancel.wait(5)
        if cancel.is_set():
            raise desktop.PreparationCancelled("cancelled")
        raise RuntimeError("test preparation exceeded its cancellation window")

    supervisor.runner_manager = SimpleNamespace(ensure=prepare)
    result = supervisor.dispatch(
        {"command": "start", "path": row["path"], "identity": row["identity"]}
    )
    assert result["state"] == "Preparing" and entered.wait(2)
    assert supervisor.child is None and supervisor.reservation is None
    with pytest.raises(ValueError, match="current model"):
        supervisor.begin_start(row["path"], row["identity"])
    supervisor.dispatch({"command": "stop", "path": row["path"]})
    assert not supervisor.preparing()
    assert supervisor.worker["state"] == "Stopped"
    assert supervisor.worker_error is None


def test_failed_preparation_surfaces_error_without_publishing_worker(
    supervisor, monkeypatch
):
    row = supervisor.library.snapshot()["rows"][0]
    synthetic_resolution(monkeypatch)

    def fail(*args, **kwargs):
        raise RuntimeError("dependency installation failed")

    supervisor.runner_manager = SimpleNamespace(ensure=fail)
    supervisor.begin_start(row["path"], row["identity"])
    supervisor.preparation_thread.join(timeout=2)
    assert supervisor.worker["state"] == "Error"
    assert supervisor.worker_error == "dependency installation failed"
    assert supervisor.child is None and supervisor.reservation is None


def test_context_stop_cannot_stop_a_different_model(supervisor):
    supervisor.worker = {"path": "/some/other/model", "state": "Stopped"}
    with pytest.raises(ValueError, match="no longer"):
        supervisor.dispatch({"command": "stop", "path": "/stale/context/model"})


def test_automatic_profile_keeps_runtime_shape_and_route_defaults(
    supervisor, monkeypatch, tmp_path
):
    row = supervisor.library.snapshot()["rows"][0]
    synthetic_resolution(monkeypatch)
    lease = reservation(tmp_path)
    monkeypatch.setattr(desktop, "GPUReservation", lambda: lease)
    monkeypatch.setattr(desktop, "source_identity", dict)
    original = subprocess.Popen
    calls = []

    def fake(args, **kwargs):
        calls.append(args)
        return original([sys.executable, "-c", "import time;time.sleep(120)"], **kwargs)

    monkeypatch.setattr(desktop.subprocess, "Popen", fake)
    supervisor.settings.update(context=0, lanes=0, route="auto")
    supervisor.start(row["path"], row["identity"])
    for option in (
        "--max-context",
        "--max-lanes",
        "--max-inflight",
        "--ordinary",
        "--no-auto-route",
    ):
        assert option not in calls[0]
    assert "--drafter-root" in calls[0]


def test_chat_target_credentials_bound_to_discovered_pid(
    supervisor, monkeypatch, tmp_path
):
    from mlx2 import inference_monitor

    url = "http://127.0.0.1:18285"
    endpoint = inference_monitor.ServerEndpoint(url=url, pid=123)
    monkeypatch.setattr(inference_monitor, "discover_servers", lambda: [endpoint])
    key = tmp_path / "private key"
    key.write_text("synthetic-secret\n")
    key.chmod(0o600)
    process = SimpleNamespace(
        cmdline=lambda: ["python", "-m", "mlx2.server", "--api-key-file", str(key)]
    )
    monkeypatch.setattr(psutil, "Process", lambda pid: process)
    result = supervisor.dispatch({"command": "chat_target", "url": url, "pid": 123})
    assert result == {"url": url, "api_key": "synthetic-secret"}
    with pytest.raises(ValueError, match="changed or stopped"):
        supervisor.chat_target(url, 124)
    key.chmod(0o644)
    with pytest.raises(ValueError, match="0600"):
        supervisor.chat_target(url, 123)


def test_chat_target_environment_and_no_auth(supervisor, monkeypatch):
    from mlx2 import inference_monitor

    url = "http://127.0.0.1:18285"
    monkeypatch.setattr(
        inference_monitor,
        "discover_servers",
        lambda: [inference_monitor.ServerEndpoint(url=url, pid=123)],
    )
    process = SimpleNamespace(
        cmdline=lambda: ["python", "-m", "mlx2.server", "--api-key-env=TEST_KEY"],
        environ=lambda: {"TEST_KEY": "synthetic-env-secret"},
    )
    monkeypatch.setattr(psutil, "Process", lambda pid: process)
    assert supervisor.chat_target(url, 123)["api_key"] == "synthetic-env-secret"
    process.cmdline = lambda: ["python", "-m", "mlx2.server"]
    assert supervisor.chat_target(url, 123)["api_key"] is None


@pytest.mark.parametrize(
    "url",
    [
        "http://example.org:8285",
        "https://127.0.0.1:8285",
        "http://user@127.0.0.1:8285",
        "http://127.0.0.1:8285/x",
        "http://127.0.0.1:8285?key=x",
    ],
)
def test_chat_target_rejects_arbitrary_destinations(supervisor, url):
    with pytest.raises(ValueError, match="discovered local"):
        supervisor.chat_target(url, 123)


def test_managed_snapshot_exposes_pid_for_chat(supervisor, monkeypatch):
    from mlx2.inference_monitor import ServerEndpoint, ServerReading

    reading = ServerReading(
        endpoint=ServerEndpoint(url="http://127.0.0.1:18285"), reachable=True
    )
    supervisor.collector = SimpleNamespace(sample=lambda: [reading], close=lambda: None)
    supervisor.child = SimpleNamespace(pid=123)
    supervisor.worker = {
        "url": reading.endpoint.url,
        "path": "/synthetic/model",
        "state": "Loading",
    }
    monkeypatch.setattr(supervisor, "reap", lambda: None)
    try:
        endpoint = supervisor.snapshot()["servers"][0]["endpoint"]
        assert endpoint["pid"] == 123
        assert endpoint["model_path"] == "/synthetic/model"
    finally:
        supervisor.child = None


def test_stop_during_ownership_acquisition_refuses_late_launch(supervisor, monkeypatch):
    """A blocking reservation can finish after Stop has already been requested."""
    row = supervisor.library.snapshot()["rows"][0]
    synthetic_resolution(monkeypatch)
    released = []
    def acquire():
        supervisor.preparation_cancel.set()
    lease = SimpleNamespace(acquire=acquire, release=lambda: released.append(True))
    monkeypatch.setattr(desktop, "GPUReservation", lambda: lease)
    monkeypatch.setattr(desktop.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Cancelled launch must not spawn"))
    with pytest.raises(desktop.PreparationCancelled, match="cancelled"):
        supervisor.start(row["path"], row["identity"])
    assert released == [True]
    assert supervisor.reservation is None
    assert supervisor.child is None


def test_suspended_worker_guards_fresh_start_and_settings_until_explicit_stop(supervisor):
    row = supervisor.library.snapshot()["rows"][0]
    supervisor.worker = {"path": row["path"], "state": "Suspended"}
    supervisor.suspension.record = {"id": "a" * 32, "state": "Suspended", "worker": dict(supervisor.worker), "settings": dict(supervisor.settings)}
    supervisor.suspension.save()
    with pytest.raises(ValueError, match="suspension"):
        supervisor.begin_start(row["path"], row["identity"])
    with pytest.raises(ValueError, match="suspended"):
        supervisor.configure(supervisor.settings)
    supervisor.stop(preserve_suspension=True)
    assert supervisor.worker["state"] == "Suspended"
    assert json.loads((supervisor.home / "desktop-suspension.json").read_text())["state"] == "Suspended"
    supervisor.stop()
    assert supervisor.worker["state"] == "Stopped"
    assert json.loads((supervisor.home / "desktop-suspension.json").read_text())["state"] == "Stopped"


def test_desktop_lora_settings_are_typed_confined_and_ordinary(supervisor, tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    settings = {**supervisor.settings, "route": "ordinary", "lora_root": str(adapter), "max_loras": 1, "max_lora_rank": 32}
    checked = desktop.Supervisor.valid_settings(settings)
    assert checked["lora_root"] == str(adapter.resolve())
    for changes in ({"route": "auto"}, {"max_loras": True}, {"max_loras": 65}, {"max_lora_rank": 0},
                    {"lora_root": None}, {"lora_root": str(adapter / "missing")}):
        with pytest.raises(ValueError):
            desktop.Supervisor.valid_settings({**settings, **changes})


def test_desktop_lora_flags_reach_only_explicit_ordinary_launch(supervisor, monkeypatch, tmp_path):
    row = supervisor.library.snapshot()["rows"][0]
    synthetic_resolution(monkeypatch)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    supervisor.settings = desktop.Supervisor.valid_settings({**supervisor.settings, "route": "ordinary",
        "lora_root": str(adapter), "max_loras": 1, "max_lora_rank": 32})
    lease = reservation(tmp_path)
    monkeypatch.setattr(desktop, "GPUReservation", lambda: lease)
    monkeypatch.setattr(desktop, "source_identity", dict)
    arguments = []
    def fail(args, **kwargs):
        arguments.extend(args)
        raise OSError("synthetic no-spawn")
    monkeypatch.setattr(desktop.subprocess, "Popen", fail)
    with pytest.raises(OSError, match="no-spawn"):
        supervisor.start(row["path"], row["identity"])
    assert "--ordinary" in arguments
    assert arguments[arguments.index("--lora-dir") + 1] == str(adapter.resolve())
    assert arguments[arguments.index("--max-loras") + 1] == "1"
    assert arguments[arguments.index("--max-lora-rank") + 1] == "32"
    assert supervisor.reservation is None


@pytest.mark.parametrize("state", ["ResumeError", "SuspendError", "SuspensionRecoveryRequired"])
def test_ready_telemetry_does_not_erase_suspension_failure(supervisor, monkeypatch, state):
    from mlx2.inference_monitor import ServerEndpoint, ServerReading
    reading = ServerReading(endpoint=ServerEndpoint(url="http://127.0.0.1:18285"), reachable=True, ready=True)
    supervisor.collector = SimpleNamespace(sample=lambda: [reading], close=lambda: None)
    supervisor.child = SimpleNamespace(pid=123)
    supervisor.worker = {"url": reading.endpoint.url, "path": "/synthetic/model", "state": state,
        "resume_lineage_id": "pending:untrusted", "cache_restore": {"confirmed": False}}
    monkeypatch.setattr(supervisor, "reap", lambda: None)
    try:
        supervisor.snapshot()
        assert supervisor.worker["state"] == state
        assert not supervisor.worker["cache_restore"]["confirmed"]
        assert not supervisor.recorded_load
    finally:
        supervisor.child = None


def test_ready_metrics_survive_optional_status_timeout(supervisor, monkeypatch):
    from mlx2.inference_monitor import ServerEndpoint, ServerReading
    reading = ServerReading(endpoint=ServerEndpoint(url="http://127.0.0.1:18285"), reachable=True,
                            ready=True, running=1, status_error="timed out")
    supervisor.collector = SimpleNamespace(sample=lambda: [reading], close=lambda: None)
    supervisor.child = SimpleNamespace(pid=123)
    supervisor.worker = {"url": reading.endpoint.url, "path": "/synthetic/model", "state": "Busy"}
    supervisor.recorded_load = True
    monkeypatch.setattr(supervisor, "reap", lambda: None)
    try:
        supervisor.snapshot()
        assert supervisor.worker["state"] == "Busy"
        assert "health_check" not in supervisor.worker
    finally:
        supervisor.child = None


def test_poll_timeout_requires_sustained_evidence_and_recovers(supervisor, monkeypatch):
    from mlx2.inference_monitor import ServerEndpoint, ServerReading
    endpoint = ServerEndpoint(url="http://127.0.0.1:18285")
    reading = ServerReading(endpoint=endpoint, reachable=False, error="timed out")
    supervisor.collector = SimpleNamespace(sample=lambda: [reading], close=lambda: None)
    supervisor.child = SimpleNamespace(pid=123)
    supervisor.worker = {"url": endpoint.url, "path": "/synthetic/model", "state": "Busy"}
    supervisor.recorded_load = True
    monkeypatch.setattr(supervisor, "reap", lambda: None)
    clock = [0.0]
    monkeypatch.setattr(desktop.time, "monotonic", lambda: clock[0])
    try:
        for now in [0.0, 4.0]:
            clock[0] = now
            supervisor.snapshot()
            assert supervisor.worker["state"] == "Busy"
        clock[0] = 11.0
        supervisor.snapshot()
        assert supervisor.worker["state"] == "Unreachable"
        reading = ServerReading(endpoint=endpoint, reachable=True, ready=True)
        supervisor.snapshot()
        assert supervisor.worker["state"] == "Ready"
        assert supervisor.health_misses == 0 and supervisor.health_miss_since is None
        assert "health_check" not in supervisor.worker
        reading = ServerReading(endpoint=endpoint, reachable=True, ready=False)
        supervisor.snapshot()
        assert supervisor.worker["state"] == "Unreachable"
    finally:
        supervisor.child = None

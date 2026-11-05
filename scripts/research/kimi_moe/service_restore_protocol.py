"""Private service capture and paired, revision-bound restoration for one unit.

Derived from the GPU-tested 2026-11-04 per-unit protocol, not a serving backend.
The project-local coordination IDs are explicit configuration, never authority
to stop another service. Only the captured idle Kimi LaunchAgent is suspended.
"""

import argparse
import hashlib
import json
import os
import plistlib
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts/research")]
import kimi_expert_budget_probe as budget
import run_with_gpu_locks as paired

from mlx2.desktop_campaign_lease import DesktopCPGLease
from mlx2.desktop_campaign_process import spawn
from mlx2.desktop_campaign_worker import Radio, atomic, check_ownership

MONITOR = os.environ.get("KIMI_MOE_MONITOR_SESSION", "436e8b3a1c9d4ee9b55e8d10310586c6")
FOREIGN_TASKS = os.environ.get("KIMI_MOE_FOREIGN_CPG_TASKS", "beff4e634ddb46978a84a1aed315f785").split(",")
MCP = "http://127.0.0.1:8766/mcp"


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def definition():
    return subprocess.check_output(
        ["launchctl", "print", f"gui/{os.getuid()}/{budget.LABEL}"],
        text=True, stderr=subprocess.DEVNULL,
    )


def service_pid(text):
    match = re.search(r"^\s*pid = (\d+)\s*$", text, re.MULTILINE)
    if match is None:
        raise RuntimeError("LaunchAgent lacks a live PID")
    return int(match.group(1))


def original_ready(snapshot=None, *, idle=False):
    try:
        raw = budget.PLIST.read_bytes()
        if snapshot is not None and hashlib.sha256(raw).hexdigest() != snapshot["plist_sha256"]:
            raise RuntimeError("Original definition changed; refuse overwrite")
        pid = service_pid(definition())
        if psutil.Process(pid).cmdline() != plistlib.loads(raw)["ProgramArguments"]:
            return None
        status = budget.http(8282, "/v1/status", timeout=3)
        if status.get("state") != "ready" or status.get("model") != budget.MODEL.name:
            return None
        if idle and (status.get("inflight") != 0 or status.get("queue_depth") != 0):
            return None
        return status
    except (OSError, TimeoutError, subprocess.CalledProcessError, psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def _cpg_free():
    radio = Radio(MCP)
    radio.call("join_session", {"session_id": MONITOR, "agent_id": "kimi-moe-service-protocol"})
    tasks = [x for x in FOREIGN_TASKS if x]
    if not tasks:
        raise RuntimeError("Configure the live project coordination task IDs")
    result = radio.call("get_nodes", {"node_ids": tasks, "session_scope": "project"})
    nodes = result.get("nodes", [])
    return len(nodes) == len(tasks) and all(
        n["payload"]["status"] in {"planned", "completed", "cancelled", "failed"} for n in nodes
    )


def cpg_free():
    try:
        return _cpg_free()
    except (OSError, RuntimeError, KeyError, TypeError):
        return False


def admit():
    if any(p.is_dir() for p in paired.LOCKS) or any(paired.WAITERS.glob("*")) or not cpg_free():
        raise RuntimeError("Another GPU owner/waiter needs the machine; yield without mutation")
    if original_ready(idle=True) is None:
        raise RuntimeError("Expected original service ready and idle")
    pid = service_pid(definition())
    foreign = paired.foreign_model_processes()
    if any(int(item.split(maxsplit=1)[0]) != pid for item in foreign):
        raise RuntimeError("Foreign model process; leave it running")


def capture(out, *, checkpoint, source_revision):
    admit()
    out = Path(out)
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name, data in [("original.plist", budget.PLIST.read_bytes()), ("launchctl.txt", definition().encode())]:
        fd = os.open(out / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
    pid = service_pid((out / "launchctl.txt").read_text())
    snapshot = {"private_directory": str(out.resolve()), "plist_sha256": sha(out / "original.plist"),
                "pid": pid, "created": psutil.Process(pid).create_time(),
                "checkpoint": checkpoint, "source_revision": source_revision,
                "service_runtime": "Captured original definition; distinct from research source_revision"}
    atomic(out / "snapshot.json", snapshot)
    os.chmod(out / "snapshot.json", 0o600)
    return snapshot


def suspend(snapshot):
    admit()
    if original_ready(snapshot, idle=True) is None:
        raise RuntimeError("Original service changed or became busy")
    pid = service_pid(definition())
    if pid != snapshot["pid"] or psutil.Process(pid).create_time() != snapshot["created"]:
        raise RuntimeError("Original service PID identity changed")
    result = subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}/{budget.LABEL}"], check=False)
    if result.returncode:
        # A failed invocation that already removed the service still requires restore.
        exists = subprocess.run(["launchctl", "print", f"gui/{os.getuid()}/{budget.LABEL}"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode == 0
        if exists:
            raise RuntimeError("Original bootout failed without suspension")


def restore_child(out):
    spec = load(out / "spec.json")
    snapshot = spec["service_snapshot"]
    check_ownership(spec)
    if sha(budget.PLIST) != snapshot["plist_sha256"]:
        raise RuntimeError("Original definition changed; refuse overwrite")
    started = False
    ready = None
    try:
        exists = subprocess.run(["launchctl", "print", f"gui/{os.getuid()}/{budget.LABEL}"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode == 0
        if not exists:
            check_ownership(spec)
            subprocess.run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(budget.PLIST)], check=True)
            started = True
        deadline = time.monotonic() + 300
        while ready is None:
            check_ownership(spec)
            ready = original_ready(snapshot)
            if ready is None and time.monotonic() > deadline:
                raise TimeoutError("Bounded original-service startup")
            if ready is None:
                time.sleep(1)
        atomic(out / "service-restored-ready.json", ready)
    finally:
        if ready is None and started:
            # Stop only the startup we initiated, while its ownership is still valid.
            check_ownership(spec)
            result = subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}/{budget.LABEL}"],
                                    capture_output=True, text=True, check=False)
            atomic(out / "incomplete-startup-cleanup.json", {"returncode": result.returncode})


def restore(out, *, snapshot):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    state = {"pid": os.getpid(), "created": psutil.Process().create_time(), "gpu_held_while_waiting": False}
    def save(status):
        state.update(status=status, updated_at=time.time())
        atomic(out / "state.json", state)
    def defer(number, frame):
        state["deferred_signal"] = number
        save("signal deferred until mandatory restoration completes")
    handlers = {s: signal.signal(s, defer) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        while True:
            if sha(budget.PLIST) != snapshot["plist_sha256"]:
                raise RuntimeError("Original service definition changed; refuse overwrite")
            ready = original_ready(snapshot)
            if ready is not None:
                atomic(out / "service-restored-ready.json", ready)
                save("verified original ready")
                return ready
            if any(p.is_dir() for p in paired.LOCKS) or any(paired.WAITERS.glob("*")) or paired.foreign_model_processes() or not cpg_free():
                save("waiting without GPU for ownership/FIFO/model activity to clear")
                time.sleep(3)
                continue
            attempt = out / f"startup-{time.time_ns()}"
            attempt.mkdir()
            spec = {"out": str(attempt), "job_id": "kimi-service-restore-" + str(time.time_ns()),
                    "mcp_url": MCP, "service_snapshot": snapshot}
            atomic(attempt / "spec.json", spec)
            with DesktopCPGLease(attempt, spec["job_id"], checkpoint=snapshot["checkpoint"], source_revision=snapshot["source_revision"]) as lease:
                command = [sys.executable, str(ROOT / "scripts/run_with_gpu_locks.py"), "--session", lease.owner["cpg_session"],
                           "--label", spec["job_id"], "--desktop-cpg-lease", str(lease.path), "--receipt", str(attempt / "paired-locks.json"),
                           "--", sys.executable, str(Path(__file__).resolve()), "--restore-child", str(attempt)]
                with (attempt / "startup.log").open("x") as log:
                    child = spawn(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    deadline = None
                    try:
                        while child.poll() is None:
                            lease.check()
                            try:
                                locked = load(attempt / "paired-locks.json").get("status") == "running"
                            except (OSError, json.JSONDecodeError):
                                locked = False
                            if locked and deadline is None:
                                deadline = time.monotonic() + 360
                            if deadline is not None and time.monotonic() > deadline:
                                raise TimeoutError("Bounded protected service startup")
                            time.sleep(1)
                        lease.success = child.returncode == 0
                        if not lease.success:
                            raise RuntimeError("Protected startup failed; retain receipts for recovery")
                    finally:
                        child.stop(timeout=25)
            save("startup completed; verify service identity after handoff")
    finally:
        for s, handler in handlers.items():
            signal.signal(s, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore-child", type=Path, required=True)
    args = parser.parse_args()
    restore_child(args.restore_child)

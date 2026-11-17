"""Child processes (one model per process) and the GPU-window guard."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

from . import paths


class ChildFailed(SystemExit):
    def __init__(self, what: str, rc: int, log: Path):
        self.rc = rc
        super().__init__(f"{what} failed (rc={rc}); see {log}")


class CleanStop(SystemExit):
    """A child stopped cleanly at --stop-at (exit 75): rerun `mlx2 amoeba resume`."""


def footprint_gb(pid: int) -> float | None:
    """Physical footprint of one process (the carve watchdog's measure), GB."""
    try:
        out = subprocess.run(["/usr/bin/footprint", "-p", str(pid)], capture_output=True, text=True,
                             timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        out = ""
    import re
    m = re.search(r"phys_footprint:\s*([\d.]+)\s*([KMG])B", out) or re.search(r"Footprint:\s*([\d.]+)\s*([KMG])B", out)
    if m:
        return float(m[1]) / {"K": 2 ** 20, "M": 2 ** 10, "G": 1}[m[2]]
    try:
        out = subprocess.run(["/usr/bin/top", "-l", "1", "-pid", str(pid), "-stats", "mem"], capture_output=True,
                             text=True, timeout=20).stdout.strip().splitlines()
    except (OSError, subprocess.TimeoutExpired):
        return None
    if not out:
        return None
    m = re.match(r"([\d.]+)([KMGB])", out[-1].strip())
    if not m:
        return None
    return float(m[1]) / {"B": 2 ** 30, "K": 2 ** 20, "M": 2 ** 10, "G": 1}[m[2]]


DEADLINE = {"t": None}     # --stop-at (epoch s): no child starts after it; a running carve checkpoints at it


def check_deadline(what: str) -> None:
    if DEADLINE["t"] is not None and time.time() >= DEADLINE["t"]:
        raise CleanStop(f"--stop-at reached before {what}; resume with `mlx2 amoeba resume`")


def run_child(cmd: list, *, stage: str, progress, log_path: Path, what: str = "", env_extra: dict | None = None,
              cwd: Path | None = None, sample_s: float = 15.0) -> int:
    """Run ``cmd`` in its own process group, tee its output to ``log_path``, feed every line to the progress
    parser, sample its footprint, and stop the whole group on interrupt. Returns 0, raises ChildFailed /
    CleanStop otherwise. No child starts after --stop-at; one that has started runs to its end (the carve
    itself checkpoints and stops at the deadline, exit 75)."""
    cmd = [str(c) for c in cmd]
    what = what or Path(cmd[1] if len(cmd) > 1 else cmd[0]).name
    check_deadline(what)
    check_agent_memory(progress, stage)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as lf:
        lf.write(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} $ {' '.join(cmd)}\n")
        lf.flush()
        p = subprocess.Popen(cmd, cwd=str(cwd or paths.repo_root()), env=paths.child_env(env_extra),
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                             start_new_session=True)
        progress.emit("child_start", stage=stage, what=what, pid=p.pid)
        done = threading.Event()

        def sampler():
            while not done.wait(sample_s):
                gb = footprint_gb(p.pid)
                if gb is not None:
                    progress.emit("footprint", stage=stage, pid=p.pid, gb=round(gb, 2))
        th = threading.Thread(target=sampler, daemon=True)
        th.start()
        try:
            for line in p.stdout:
                lf.write(line)
                lf.flush()
                progress.feed_line(line.rstrip("\n"), stage)
            rc = p.wait()
        except BaseException:
            stop_group(p)
            raise
        finally:
            done.set()
    progress.emit("child_end", stage=stage, what=what, rc=rc)
    if rc == 75:
        raise CleanStop(f"{what}: clean stop at --stop-at; resume with `mlx2 amoeba resume`")
    if rc:
        raise ChildFailed(what, rc, log_path)
    return 0


def stop_group(p: subprocess.Popen, grace_s: float = 120.0) -> None:
    """SIGTERM the group, then SIGKILL after ``grace_s`` (longer than a generation stage's own server stop)."""
    try:
        os.killpg(p.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        p.wait(grace_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


class Background:
    """A long-lived child (a generation server): output teed to a log and fed to the progress parser."""

    def __init__(self, cmd: list, *, stage: str, progress, log_path: Path, env_extra: dict | None = None):
        check_agent_memory(progress, stage)
        cmd = [str(c) for c in cmd]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.lf = log_path.open("a")
        self.lf.write(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} $ {' '.join(cmd)}\n")
        self.lf.flush()
        self.p = subprocess.Popen(cmd, cwd=str(paths.repo_root()), env=paths.child_env(env_extra),
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                  start_new_session=True)
        self.ready = threading.Event()
        progress.emit("child_start", stage=stage, what="background", pid=self.p.pid)

        def pump():
            for line in self.p.stdout:
                self.lf.write(line)
                self.lf.flush()
                if line.startswith("ready on "):
                    self.ready.set()
                progress.feed_line(line.rstrip("\n"), stage)
        self.th = threading.Thread(target=pump, daemon=True)
        self.th.start()

    def wait_ready(self, timeout_s: float = 900.0) -> None:
        t0 = time.time()
        while not self.ready.wait(1.0):
            if self.p.poll() is not None:
                raise SystemExit(f"background child exited rc={self.p.returncode} before it was ready")
            if time.time() - t0 > timeout_s:
                stop_group(self.p)
                raise SystemExit("background child not ready in time")

    def stop(self) -> None:
        stop_group(self.p)
        self.th.join(5)
        self.lf.close()


def free_port() -> int:
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# ------------------------------------------------------------------ GPU window guard

def _ancestors() -> set[int]:
    out, pid = set(), os.getpid()
    for _ in range(64):
        out.add(pid)
        try:
            ppid = int(subprocess.check_output(["ps", "-o", "ppid=", "-p", str(pid)], text=True).strip())
        except (OSError, ValueError, subprocess.CalledProcessError):
            break
        if ppid <= 1 or ppid == pid:
            break
        pid = ppid
    return out


def gpu_window(lock: Path = paths.GPU_LOCK) -> dict:
    """Who holds the GPU lock, and whether it is this process tree (section 8: take the GPU lock)."""
    owner = lock / "owner.json"
    if not lock.is_dir() or not owner.exists():
        return {"held": False, "ours": False, "lock": str(lock)}
    try:
        o = json.loads(owner.read_text())
    except (OSError, json.JSONDecodeError):
        return {"held": True, "ours": False, "lock": str(lock), "owner": "unreadable"}
    pids = {o.get("pid"), (o.get("cpg_owner") or {}).get("pid")} - {None}
    ours = bool(pids & _ancestors())
    return {"held": True, "ours": ours, "lock": str(lock), "label": o.get("label"), "pids": sorted(pids)}


def require_gpu_window(cpu: bool, override: str = "", lock: Path | None = None, *, ignore_queue: str = "") -> dict:
    """GPU stages run only inside an owned GPU window: this process descends from the lock owner, or the caller
    names the lock's current owner label (``--gpu-window-owner LABEL``: a window another job opened for this run)."""
    w = gpu_window(lock or paths.GPU_LOCK)
    if ignore_queue:
        return {**w, "device": "cpu" if cpu else "gpu", "required": False, "ignore_gpu_queue": ignore_queue}
    if cpu:
        return {**w, "device": "cpu", "required": False}
    if w["ours"]:
        return {**w, "device": "gpu", "required": True}
    if override and w["held"] and override == w.get("label"):
        return {**w, "device": "gpu", "required": True, "owner_stated": override}
    if override:
        raise SystemExit(f"--gpu-window-owner {override!r} does not match the GPU lock "
                         f"({'held by ' + str(w.get('label')) if w['held'] else 'not held'}): refusing")
    held = f"held by {w.get('label')} (pids {w.get('pids')})" if w["held"] else "not held"
    raise SystemExit(f"GPU stage outside a GPU window: the lock {w['lock']} is {held}. Run inside "
                     "scripts/run_with_gpu_locks.py (and scripts/run_with_service_paused.py when :8282 must pause), "
                     "use --cpu, or pass --gpu-window-owner WHO when another job already owns this window for you")


def check_agent_memory(progress, stage):
    """Auto's run-wide swap baseline propagates to every pipeline child, including background generation."""
    baseline = os.environ.get("MLX2_AMOEBA_AUTO_SWAP_BASELINE")
    if baseline is not None:
        from .agent import admit_memory
        facts = admit_memory(float(baseline))
        progress.emit("memory_admission", stage=stage, **facts)

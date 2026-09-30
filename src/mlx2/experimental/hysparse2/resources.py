"""Shared Apple GPU admission, without CPG or changes to other workloads."""

import fcntl
import json
import os
import platform
import subprocess
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path


def active_waiters(directory, *, before, now=None):
    now = time.time() if now is None else now
    result = []
    for path in Path(directory).glob("*"):
        try:
            pid, created = map(int, path.name.rsplit(".", 2)[-2:])
            if (
                created > before
                or now - created > 5400
                or now - path.stat().st_mtime > 300
            ):
                continue
            if created == before and pid > os.getpid():
                continue
            os.kill(pid, 0)
        except (ValueError, FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            pass
        result.append(path)
    return result


def serving_processes():
    rows = subprocess.check_output(
        ["ps", "-axo", "comm=,args="], text=True
    ).splitlines()
    found = []
    for row in rows:
        parts = row.strip().split(None, 1)
        if len(parts) != 2 or not Path(parts[0]).name.lower().startswith(
            ("python", "tensorfold")
        ):
            continue
        command = parts[1]
        if any(
            flag in command
            for flag in (
                "-m mlx2.server",
                "tensorfold serve",
                "run_perf.py",
                "run_stress.py",
            )
        ):
            found.append(command[:300])
    return found


@contextmanager
def gpu_guard(
    *,
    wait_seconds=0,
    locks=("/Users/Shared/mlxuag/gpu.lock", "/tmp/gpu.lock"),
    waiters="/Users/Shared/mlxuag/gpu.lock.waiters",
):
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("GPU training requires Apple Silicon")
    started = time.monotonic()
    timestamp = int(time.time())
    waiter = None
    if wait_seconds:
        directory = Path(waiters)
        directory.mkdir(parents=True, exist_ok=True)
        waiter = directory / f"hysparse2.{os.getpid()}.{timestamp}"
        waiter.touch(exist_ok=False)
    last_report = -60
    try:
        while True:
            if waiter is not None:
                waiter.touch()
            older = [
                p for p in active_waiters(waiters, before=timestamp) if p != waiter
            ]
            reason = "earlier GPU waiters" if older else None
            stack = ExitStack()
            try:
                if not reason:
                    try:
                        for path in locks:
                            if Path(path).is_dir():
                                raise BlockingIOError(
                                    "legacy directory GPU lease is held"
                                )
                            handle = stack.enter_context(open(path, "a"))  # noqa: SIM115
                            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        reason = "shared GPU locks held"
                    if not reason and serving_processes():
                        reason = "another serving/qualification process is running"
                if not reason:
                    if waiter is not None:
                        waiter.unlink(missing_ok=True)
                    yield
                    return
            finally:
                stack.close()
            elapsed = time.monotonic() - started
            if elapsed >= wait_seconds:
                raise RuntimeError(f"GPU unavailable: {reason}; no model was loaded")
            if elapsed - last_report >= 30:
                print(
                    json.dumps(
                        {
                            "event": "waiting_for_gpu",
                            "reason": reason,
                            "elapsed_seconds": round(elapsed),
                        }
                    ),
                    flush=True,
                )
                last_report = elapsed
            time.sleep(min(5, wait_seconds - elapsed))
    finally:
        if waiter is not None:
            waiter.unlink(missing_ok=True)

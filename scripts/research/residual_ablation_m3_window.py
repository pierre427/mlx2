#!/usr/bin/env python3
"""One M3 GPU window for a residual_ablation.py phase (run ON the M3).

Pierre (2026-11-05): run the Qwen3.5-4B ablation on the M3, stopping the Sage
service for each window and restoring it afterwards.  Order:

1. Record the Sage launchd job state; if running, ``launchctl kill SIGTERM``
   it (its serve.py stops the server and releases the GPU locks).
2. Wait for both lock paths to be free, no foreign model process and no
   earlier waiter; then take the paired locks (Sage's own lock helpers).
3. Run ``residual_ablation.py unit PHASE OUT ...`` in its own process group.
4. ``finally``: stop the child, release the locks, ``launchctl kickstart``
   Sage if it was running before, and wait for it to answer on :8439.

Usage: ``residual_ablation_m3_window.py PHASE OUT --model PATH [...]``.
Refuses a nonempty evidence directory.  M3 evidence, not M5.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTROL = Path.home() / "sage-m3/control"
SAGE = f"gui/{os.getuid()}/com.example.mlx2-sage-m3"
SAGE_URL = "http://127.0.0.1:8439/v1/models"
PYTHON = str(Path.home() / "mlx2-0324-venv/bin/python")
SESSION = "claude-residual-ablation-m3"


def sage_pid():
    out = subprocess.run(["launchctl", "print", SAGE], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.strip().startswith("pid ="):
            return int(line.split("=")[1])
    return None


def sage_healthy():
    try:
        with urllib.request.urlopen(SAGE_URL, timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def swapouts():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    line = next(l for l in out.splitlines() if l.startswith("Swapouts"))
    return int(line.split(":")[1].strip().rstrip("."))


def main():
    phase, out = sys.argv[1], Path(sys.argv[2]).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit("refusing nonempty evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(CONTROL))
    from run_with_gpu_locks import LOCKS, WAITERS, acquire, earlier_waiter, foreign_model_processes, release

    receipt = {"host": os.uname().nodename, "argv": sys.argv[1:], "started": time.time(),
               "sage_label": SAGE, "sage_pid_before": sage_pid(), "sage_healthy_before": sage_healthy()}
    path = out / "m3-window.json"

    def save():
        path.write_text(json.dumps(receipt, indent=1, sort_keys=True))

    save()
    backups, child = [], None
    created = int(time.time())
    WAITERS.mkdir(parents=True, exist_ok=True)
    waiter = WAITERS / f"{SESSION}.{phase}.{os.getpid()}.{created}"
    waiter.touch()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("SIGTERM")))
    try:
        if receipt["sage_pid_before"]:
            subprocess.run(["launchctl", "kill", "SIGTERM", SAGE], check=True)
        deadline = time.monotonic() + 240
        while True:
            waiter.touch()
            held = [str(p) for p in LOCKS if p.is_dir()]
            foreign = foreign_model_processes()
            if not held and not foreign and sage_pid() is None and earlier_waiter(waiter, created) is None:
                break
            if time.monotonic() > deadline:
                raise RuntimeError(f"M3 GPU not free: held={held} foreign={foreign[:3]}")
            time.sleep(2)
        owner = {"lease_id": f"{SESSION}-{phase}", "session": SESSION, "label": phase,
                 "pid": os.getpid(), "since": time.time(),
                 "purpose": "residual-ablation tooling run (Pierre-approved Sage stop/restore)"}
        receipt["locks_before"], backups = acquire(owner)
        waiter.unlink(missing_ok=True)
        receipt["swapouts_before"] = swapouts()
        receipt["acquired"] = time.time()
        save()
        env = dict(os.environ, PYTHONPATH=f"{ROOT / 'src'}:{ROOT / '.deps'}",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        command = [PYTHON, "-u", str(ROOT / "scripts/research/residual_ablation.py"), "unit", *sys.argv[1:]]
        with (out / "unit.stdout").open("w") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            receipt["returncode"] = child.wait()
    except KeyboardInterrupt as exc:
        receipt["interrupted"] = str(exc)
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=15)
        waiter.unlink(missing_ok=True)
        if backups:
            receipt["locks_after"] = release(backups)
            receipt["swapouts_after"] = swapouts()
        if receipt["sage_pid_before"]:
            subprocess.run(["launchctl", "kickstart", SAGE], check=False)
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline and not sage_healthy():
                time.sleep(3)
        receipt["sage_pid_after"] = sage_pid()
        receipt["sage_healthy_after"] = sage_healthy()
        receipt["finished"] = time.time()
        save()
    return receipt.get("returncode", 1)


if __name__ == "__main__":
    raise SystemExit(main())

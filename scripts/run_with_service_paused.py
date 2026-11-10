#!/usr/bin/env python3
"""Run one command with an idle pinned launchd model service paused.

The service is paused only when its status endpoint reports healthy with no
inflight requests and an empty queue; otherwise nothing runs. It is always
restored (bootstrap from its plist) and waited healthy in ``finally``, and
``service.json`` in the run directory records before/after status, the
plist hash, thermal and swap state. Same procedure as the 2026-11-04 Ripple
GPU windows (docs/research/ripple-20261104/nvfp4-serving/owned_ripple.py).

Usage: run_with_service_paused.py --run RUN_DIR [--label LABEL] -- COMMAND...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from run_with_gpu_locks import foreign_model_processes  # noqa: E402

DEFAULT_LABEL = "com.example.mlx2-kimi-linear-48b-q8"
DOMAIN = f"gui/{os.getuid()}"
# The wrapped command is usually run_with_gpu_locks.py, whose own stop_child
# takes up to 60 s to retire its workload and restore the GPU locks.
CHILD_STOP_GRACE_S = 120.0


def run_command(command, record, *, grace_s=CHILD_STOP_GRACE_S):
    """Run ``command``; on interruption forward SIGTERM and wait for its cleanup.

    ``subprocess.call`` answers an interrupt with SIGKILL after a 0.25 s
    wait, which kills a lock wrapper before it can stop its workload or
    release the GPU locks, so the service would come back beside an orphaned
    model process.  Forward the stop instead and kill only after ``grace_s``.
    """
    # Own session: a terminal Ctrl-C reaches only this wrapper, which forwards
    # exactly one SIGTERM, so the lock wrapper's cleanup is never re-interrupted.
    child = subprocess.Popen(command, start_new_session=True)
    try:
        return child.wait()
    except KeyboardInterrupt:
        record["command_interrupted"] = True
        previous = {signum: signal.signal(signum, signal.SIG_IGN)
                    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
        try:
            if child.poll() is None:
                child.send_signal(signal.SIGTERM)
                try:
                    child.wait(timeout=grace_s)
                except subprocess.TimeoutExpired:
                    record["command_killed"] = True
                    child.kill()
                    child.wait()
            record["command_returncode"] = child.returncode
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument("--status-url", default="http://127.0.0.1:8282/v1/status")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    plist = Path.home() / "Library/LaunchAgents" / f"{args.label}.plist"
    args.run.mkdir(parents=True, exist_ok=True)
    record = {"label": args.label, "command": command,
              "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def write():
        tmp = args.run / ".service.json.tmp"
        tmp.write_text(json.dumps(record, indent=2) + "\n")
        os.replace(tmp, args.run / "service.json")

    def loaded():
        return subprocess.run(["launchctl", "print", f"{DOMAIN}/{args.label}"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0

    def status():
        with urlopen(args.status_url, timeout=5) as response:
            return json.load(response)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupted)
    restore, rc = False, 1
    try:
        if loaded():
            before = status()
            if not before.get("healthy") or before.get("inflight") != 0 or before.get("queue_depth") != 0:
                raise RuntimeError("pinned service is busy or unhealthy; refusing to pause it")
            record.update(service_before=before, plist_sha256=hashlib.sha256(plist.read_bytes()).hexdigest(),
                          service_restoration_required=True)
            write()
            subprocess.run(["launchctl", "bootout", f"{DOMAIN}/{args.label}"], check=True)
            restore = True
            deadline = time.monotonic() + 60
            while foreign_model_processes():
                if time.monotonic() > deadline:
                    raise RuntimeError("pinned service process did not exit")
                time.sleep(0.5)
        if foreign_model_processes():
            raise RuntimeError("a foreign model process remains; not preempting it")
        record["thermal_before"] = subprocess.check_output(["pmset", "-g", "therm"], text=True)
        record["swap_before"] = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True)
        write()
        rc = run_command(command, record)
        record["command_returncode"] = rc
    finally:
        record["swap_after"] = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True)
        if restore:
            subprocess.run(["launchctl", "bootstrap", DOMAIN, str(plist)], check=False)
            deadline = time.monotonic() + 600
            after = None
            while time.monotonic() < deadline:
                try:
                    after = status()
                    if after.get("healthy"):
                        break
                except OSError:
                    pass
                time.sleep(5)
            record["service_after"] = after
            record["service_restored"] = bool(after and after.get("healthy"))
        write()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

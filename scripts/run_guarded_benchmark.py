"""Run a benchmark under its caller's lease and stop its own tree on cancellation."""
from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import time

from run_with_gpu_locks import stop_child


def owns_listener(pid: int, port: int) -> bool:
    """Only accept HTTP readiness from the server that owns the listening socket."""
    return subprocess.run(["lsof", "-nP", "-a", "-p", str(pid), f"-iTCP:{port}", "-sTCP:LISTEN"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5).returncode == 0


def _session_groups(sid: int) -> set[int]:
    # ps on macOS displays a kernel address in its "sess" column. getsid is
    # portable and keeps working after the benchmark's session leader dies.
    groups = set()
    rows = subprocess.check_output(["ps", "-axo", "pid=,stat="], text=True, timeout=10)
    for row in rows.splitlines():
        pid, state = row.split()
        if state.startswith("Z"):
            continue
        try:
            group = os.getpgid(int(pid))
            if os.getsid(int(pid)) == sid:
                groups.add(group)
        except ProcessLookupError:
            pass
    return groups


def _stop_session(sid: int) -> None:
    # Benchmark servers own their own process groups inside the benchmark
    # session. A crash can reparent them before the caller sees the exit.
    for sig, grace in ((signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        deadline = time.monotonic() + grace
        signalled = set()
        while groups := _session_groups(sid):
            for group in groups - signalled:
                try:
                    os.killpg(group, sig)
                    signalled.add(group)
                except ProcessLookupError:
                    pass
            if time.monotonic() >= deadline:
                break
            time.sleep(.1)
        else:
            return
    if _session_groups(sid):
        raise RuntimeError(f"could not stop benchmark session {sid}")


def swap_mb():
    output = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True)
    match = re.search(r"used\s*=\s*([\d.]+)M", output)
    if not match:
        raise RuntimeError("cannot read swap usage")
    return float(match[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--max-swap-mb", type=float)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("supply a command")
    interrupted = 0

    def cancel(signum, _frame):
        nonlocal interrupted
        interrupted = signum

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, cancel)
    deadline = None if args.timeout is None else time.monotonic() + args.timeout
    child = None
    try:
        while True:
            if interrupted:
                return 128 + interrupted
            if deadline is not None and time.monotonic() >= deadline:
                print("benchmark timed out", flush=True)
                return 124
            if args.max_swap_mb is not None and swap_mb() > args.max_swap_mb:
                print("benchmark exceeded swap limit", flush=True)
                return 21
            if child is None:
                child = subprocess.Popen(command, start_new_session=True)
            try:
                rc = child.wait(timeout=1)
                return rc if rc >= 0 else 128 - rc
            except subprocess.TimeoutExpired:
                pass
    finally:
        stop_child(child, {})
        if child is not None:
            _stop_session(child.pid)


if __name__ == "__main__":
    raise SystemExit(main())

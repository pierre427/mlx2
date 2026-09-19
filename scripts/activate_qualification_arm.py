#!/usr/bin/env python3
"""Safely replace one harness-owned candidate server and preserve its logs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time

from run_qualification_matrix import atomic_json


def process_command(pid: int) -> str | None:
    result = subprocess.run(["/bin/ps", "-p", str(pid), "-o", "stat=,command="], capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        return None
    stat, _, command = result.stdout.strip().partition(" ")
    return None if "Z" in stat else command.strip()


def exec_transition_command(argv: list[str], cwd: Path | None = None) -> str | None:
    """Return the one exact command an ``env`` launcher may exec into.

    Darwin's process table can expose the original ``/usr/bin/env`` argv for
    a short time and the exec'd program argv later.  Derive the latter only
    for the narrow env form used by the qualification harness: zero or more
    leading NAME=VALUE assignments followed by an absolute executable.
    """
    if not argv or argv[0] != "/usr/bin/env":
        return None
    index = 1
    while index < len(argv) and "=" in argv[index]:
        name, _, _ = argv[index].partition("=")
        if not name or not name.replace("_", "a").isalnum() or name[0].isdigit():
            return None
        index += 1
    if index >= len(argv):
        return None
    executable = Path(argv[index])
    if not executable.is_absolute():
        if cwd is None:
            return None
        resolved = (cwd / executable).resolve()
        if not resolved.is_file():
            return None
    return " ".join(argv[index:])


def owned_commands(state: dict) -> set[str]:
    """Commands that preserve the exact recorded launch identity."""
    commands = {str(state["command"])}
    transitioned = state.get("exec_command")
    if transitioned is None and isinstance(state.get("argv"), list):
        # Backward-compatible recovery for state written before exec_command
        # was explicit.  The target is still derived from the recorded argv.
        cwd = Path(state["cwd"]) if state.get("cwd") else None
        transitioned = exec_transition_command(
            [str(arg) for arg in state["argv"]], cwd=cwd
        )
    if transitioned:
        commands.add(str(transitioned))
    return commands


def process_group_members(pgid: int) -> list[dict[str, int]]:
    result = subprocess.run(["/bin/ps", "-axo", "pid=,pgid=,stat="],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"could not inspect candidate process group {pgid}")
    rows = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 3 and int(fields[1]) == pgid and "Z" not in fields[2]:
            pid = int(fields[0])
            try:
                sid = os.getsid(pid)
            except ProcessLookupError:
                continue
            rows.append({"pid": pid, "pgid": int(fields[1]), "sid": sid})
    return rows


def stop_owned(state_path: Path) -> None:
    if not state_path.exists():
        return
    state = json.loads(state_path.read_text())
    pid, expected = int(state["pid"]), owned_commands(state)
    # Older harness state predates explicit group fields, but this helper has
    # always launched with start_new_session=True, so the leader PID is both.
    pgid, sid = int(state.get("pgid", pid)), int(state.get("sid", pid))
    current = process_command(pid)
    members = process_group_members(pgid)
    if current is None and not members:
        state_path.unlink()
        return
    if current is not None and current not in expected:
        raise RuntimeError(f"refusing to stop reused/unowned PID {pid}: {current!r}")
    if pid != pgid or any(row["pgid"] != pgid or row["sid"] != sid for row in members):
        raise RuntimeError(f"refusing to stop process group with mismatched ownership: {members}")
    os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + 30
    remaining = members
    while time.monotonic() < deadline:
        remaining = process_group_members(pgid)
        if not remaining:
            break
        time.sleep(0.2)
    if remaining:
        os.killpg(pgid, signal.SIGKILL)
        kill_deadline = time.monotonic() + 5
        while time.monotonic() < kill_deadline:
            remaining = process_group_members(pgid)
            if not remaining:
                break
            time.sleep(0.1)
    if remaining:
        raise RuntimeError(f"candidate process group {pgid} did not stop after SIGKILL")
    state_path.unlink()


def clear_fresh_cache(cache_dir: Path, cwd: Path) -> None:
    """Clear only a qualification cache directory strictly below its cwd."""
    resolved, root = cache_dir.resolve(), cwd.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError(f"fresh cache directory must be strictly below cwd: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--fresh-cache-dir", type=Path)
    parser.add_argument("--bootout-label", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("candidate command is required after --")
    stop_owned(args.state)
    if args.fresh_cache_dir is not None:
        clear_fresh_cache(args.fresh_cache_dir, args.cwd)
    domain = f"gui/{os.getuid()}"
    for label in args.bootout_label:
        subprocess.run(["/bin/launchctl", "bootout", f"{domain}/{label}"], capture_output=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.state.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("ab", buffering=0) as log:
        process = subprocess.Popen(command, cwd=args.cwd, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    actual = process_command(process.pid)
    if actual is None:
        raise RuntimeError(f"candidate exited during launch; inspect {args.log}")
    pgid, sid = os.getpgid(process.pid), os.getsid(process.pid)
    if pgid != process.pid or sid != process.pid:
        process.terminate()
        raise RuntimeError("candidate did not start as an owned session/process-group leader")
    atomic_json(args.state, {"pid": process.pid, "pgid": pgid, "sid": sid,
                             "command": actual,
                             "exec_command": exec_transition_command(command, cwd=args.cwd),
                             "argv": command,
                             "cwd": str(args.cwd.resolve()), "log": str(args.log.resolve()),
                             "started_at": time.time()})
    print(process.pid)


if __name__ == "__main__":
    main()

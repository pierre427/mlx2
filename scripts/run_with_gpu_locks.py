#!/usr/bin/env python3
"""Run one command through the shared waiter ledger and both GPU locks.

The host currently uses empty regular files as idle sentinels.  During an
owned run they are atomically replaced by lock directories containing matching
owner.json receipts, then restored exactly to empty regular files.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import time
import traceback
from pathlib import Path

LOCKS = (Path("/Users/Shared/mlxuag/gpu.lock"), Path("/tmp/gpu.lock"))
WAITERS = Path("/Users/Shared/mlxuag/gpu.lock.waiters")
DEFAULT_SESSION = "codex-hils-declared-groups-20260930"
MODEL_PROCESS = re.compile(
    r"-m mlx2\.server|mlx_lm\.server|mlx_vlm\.server|rapid-mlx|llama-server|"
    r"qualify_hils_declared_groups\.py",
    re.IGNORECASE,
)


def snapshot(path: Path) -> dict:
    if not path.exists():
        return {"kind": "absent"}
    if path.is_file():
        return {"kind": "file", "size": path.stat().st_size}
    if path.is_dir():
        owner = path / "owner.json"
        return {
            "kind": "directory",
            "owner": owner.read_text(errors="replace") if owner.is_file() else None,
        }
    return {"kind": "other"}


def earlier_waiter(mine: Path, mine_time: int) -> str | None:
    now = time.time()
    for path in sorted(WAITERS.iterdir()):
        if path == mine or not path.is_file():
            continue
        fields = path.name.rsplit(".", 3)
        if len(fields) != 4:
            continue
        try:
            pid, created = int(fields[-2]), int(fields[-1])
            os.kill(pid, 0)
        except (ValueError, ProcessLookupError, PermissionError):
            continue
        if now - path.stat().st_mtime > 300 or now - created > 5400:
            continue
        if created < mine_time:
            return path.name
    return None


def foreign_model_processes() -> list[str]:
    rows = subprocess.check_output(["ps", "-axo", "pid=,command="], text=True)
    foreign = []
    for row in rows.splitlines():
        fields = row.strip().split(None, 1)
        if len(fields) != 2 or int(fields[0]) in {os.getpid(), os.getppid()}:
            continue
        executable = fields[1].split(None, 1)[0]
        if executable.endswith(("/zsh", "/bash", "/sh", "/git", "/git-remote-https")):
            continue
        if MODEL_PROCESS.search(fields[1]):
            foreign.append(row.strip()[:500])
    return foreign


def acquire(owner: dict) -> tuple[list[dict], list[tuple[Path, Path | None]]]:
    before = [{"path": str(path), **snapshot(path)} for path in LOCKS]
    held_files = []
    backups: list[tuple[Path, Path | None]] = []
    try:
        for path in LOCKS:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.is_file():
                if path.stat().st_size != 0:
                    raise RuntimeError(
                        f"nonempty GPU sentinel refuses conversion: {path}"
                    )
                descriptor = os.open(path, os.O_RDWR)
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    os.close(descriptor)
                    raise RuntimeError(f"GPU sentinel is flock-held: {path}")
                held_files.append(descriptor)
            elif path.exists():
                raise RuntimeError(
                    f"GPU lock is already held: {path}: {snapshot(path)}"
                )
        for path in LOCKS:
            backup = None
            if path.is_file():
                backup = path.with_name(f"{path.name}.sentinel.{os.getpid()}")
                if backup.exists():
                    raise RuntimeError(f"sentinel backup already exists: {backup}")
                path.rename(backup)
            try:
                path.mkdir()
            except Exception:
                if backup is not None and backup.exists() and not path.exists():
                    backup.rename(path)
                raise
            backups.append((path, backup))
        for path in LOCKS:
            (path / "owner.json").write_text(
                json.dumps(owner, indent=2, sort_keys=True) + "\n"
            )
        owners = [json.loads((path / "owner.json").read_text()) for path in LOCKS]
        if owners[0] != owners[1]:
            raise RuntimeError("GPU owner receipts differ after acquisition")
        return before, backups
    finally:
        for descriptor in held_files:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def release(backups: list[tuple[Path, Path | None]]) -> list[dict]:
    errors = []
    for path, backup in reversed(backups):
        try:
            for name in ("owner.json", "owner.txt"):
                item = path / name
                if item.exists():
                    item.unlink()
            path.rmdir()
            if backup is not None:
                backup.rename(path)
        except Exception as exc:  # noqa: BLE001 - attempt every lock restoration
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
    if errors:
        raise RuntimeError("; ".join(errors))
    return [{"path": str(path), **snapshot(path)} for path in LOCKS]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--label", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    WAITERS.mkdir(parents=True, exist_ok=True)
    created = int(time.time())
    waiter = WAITERS / f"{args.session}.{args.label}.{os.getpid()}.{created}"
    receipt = {
        "schema": "mlx2.gpu-lock-window.v1",
        "session": args.session,
        "label": args.label,
        "pid": os.getpid(),
        "created_at": time.time(),
        "waiter": str(waiter),
        "command": command,
        "status": "waiting",
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        receipt["updated_at"] = time.time()
        args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    backups: list[tuple[Path, Path | None]] = []
    child = None
    rc = 1
    waiter.touch()
    save()
    try:
        while predecessor := earlier_waiter(waiter, created):
            receipt["waiting_behind"] = predecessor
            waiter.touch()
            save()
            time.sleep(15)
        receipt.pop("waiting_behind", None)
        foreign = foreign_model_processes()
        if foreign:
            raise RuntimeError(f"foreign model process present: {foreign[:5]}")
        owner = {
            "lease_id": f"{args.session}-{args.label}",
            "session": args.session,
            "label": args.label,
            "pid": os.getpid(),
            "since": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        receipt["lock_state_before"], backups = acquire(owner)
        receipt["owner"] = owner
        receipt["status"] = "running"
        waiter.unlink(missing_ok=True)
        save()
        child = subprocess.Popen(command, start_new_session=True)
        receipt["child_pid"] = child.pid
        save()
        rc = child.wait()
        receipt["command_returncode"] = rc
        receipt["status"] = "command_completed" if rc == 0 else "command_failed"
    except BaseException as exc:  # noqa: BLE001 - also retire children on Ctrl-C
        receipt["status"] = "error"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        receipt["traceback"] = traceback.format_exc(limit=8)
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=30)
    finally:
        waiter.unlink(missing_ok=True)
        try:
            receipt["lock_state_after"] = (
                release(backups)
                if backups
                else [{"path": str(path), **snapshot(path)} for path in LOCKS]
            )
        except Exception as exc:  # noqa: BLE001 - persist sentinel restoration failure
            receipt["release_error"] = f"{type(exc).__name__}: {exc}"
            receipt["status"] = "release_failed"
            rc = 1
        receipt["finished_at"] = time.time()
        save()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

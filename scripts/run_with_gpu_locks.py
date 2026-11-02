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
import tempfile
import time
import traceback
import uuid
from pathlib import Path

LOCKS = (Path("/Users/Shared/mlxuag/gpu.lock"), Path("/tmp/gpu.lock"))
WAITERS = Path("/Users/Shared/mlxuag/gpu.lock.waiters")
DEFAULT_SESSION = "codex-hils-declared-groups-20260930"
# How long the head of the queue waits for a held lock before failing.
HEAD_WAIT_SECONDS = 4 * 3600
# A radio read that fails in transport (timeout, refused connection) while
# the child runs is retried for this long, bounded by the last confirmed
# lease expiry.  An ownership verdict (RuntimeError from validation) is never
# retried.  2026-10-29: one 15 s timeout right after a reboot tore down a
# Flash-Next ladder.
OBSERVE_TRANSIENT_GRACE_SECONDS = 180
MODEL_PROCESS = re.compile(
    r"-m\s+mlx2\.(?:server|decisions\.server)(?:\s|$)|mlx_lm\.server|mlx_vlm\.server|rapid-mlx|llama-server|"
    r"qualify_hils_declared_groups\.py",
    re.IGNORECASE,
)


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Replace a mutable JSON receipt without exposing a partial write."""
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.tmp."
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_exclusive(path: Path, payload: dict) -> None:
    """Create an immutable JSON receipt, refusing to replace any prior evidence."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _immutable_receipt_path(receipt_path: Path, attempt_id: str) -> Path:
    return receipt_path.with_name(
        f"{receipt_path.stem}.attempt-{attempt_id}{receipt_path.suffix}"
    )


def _save_terminal(
    receipt_path: Path, immutable_path: Path, receipt: dict
) -> None:
    """Seal terminal evidence before updating the mutable compatibility receipt."""
    _write_json_exclusive(immutable_path, receipt)
    _atomic_write_json(receipt_path, receipt)


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
        # A live wrapper touches its waiter every 15 s, so a dead pid or a
        # stale mtime already marks an abandoned file.  No age cap: one made
        # a waiter that had queued for 90 minutes invisible, and every newer
        # waiter then went ahead of it forever.
        try:
            modified = path.stat().st_mtime
        except FileNotFoundError:
            # A predecessor unlinks its waiter as soon as it acquires the
            # locks. Its normal departure must not fail another queued job.
            continue
        if now - modified > 300:
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
        arguments = fields[1].split()
        if len(arguments) > 1 and Path(arguments[1]).name == "run_with_gpu_locks.py":
            # Another lock wrapper names its server command in its argv, but
            # it owns no GPU state: while waiting it holds nothing, and once
            # running its lock directories and its child server say so.
            # Counting it made two queued sessions refuse each other forever.
            continue
        if Path(executable).name.lower() in {"ssh", "ssh.exe"}:
            # A local SSH client can mention mlx2.server in the command it
            # executes on another host; it does not own this host's GPU.
            continue
        if Path(executable).name.lower() in {"node", "codex", "claude"}:
            # Agent CLIs carry their prompt in argv; a prompt that quotes a
            # server command does not load a model (servers run as python).
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
            except BaseException:
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
    except BaseException:
        # The caller receives backups only after a complete acquisition.
        # Roll back partial ownership here, including signal interruption,
        # while the original sentinel file descriptors are still flock-held.
        if backups:
            release(backups)
        raise
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


def descendant_pids(root: int) -> list[int]:
    """Every live descendant of ``root``, whatever session or group it is in.

    A stage may start its own children with ``start_new_session``; those
    leave the wrapper's process group, so ``killpg`` alone cannot reach them.
    Snapshot the tree before signalling: once a parent dies its children are
    reparented to launchd and no longer trace back to ``root``.
    """
    try:
        rows = subprocess.run(["ps", "-axo", "pid=,ppid="], capture_output=True,
                              text=True, check=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    children: dict[int, list[int]] = {}
    for row in rows.splitlines():
        fields = row.split()
        if len(fields) == 2 and fields[0].isdigit() and fields[1].isdigit():
            children.setdefault(int(fields[1]), []).append(int(fields[0]))
    found, stack = [], [root]
    while stack:
        for pid in children.get(stack.pop(), []):
            if pid not in found:
                found.append(pid)
                stack.append(pid)
    return found


def _signal_tree(child: subprocess.Popen, tree: list[int], sig: int) -> list[int]:
    """Signal the child's process group and ``tree``; return the pids we may not signal.

    A descendant of another uid (a ``sudo powermetrics`` helper, or a pid
    another user's process reused after the snapshot) raises PermissionError;
    it is not ours to stop and must not abort the stop, which runs first in
    the ``finally`` that restores the GPU locks.
    """
    unsignallable = []
    try:
        os.killpg(child.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass
    for pid in tree:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass
        except PermissionError:
            unsignallable.append(pid)
    return unsignallable


def _alive(pids: list[int]) -> list[int]:
    """Pids in ``pids`` that still exist and that this uid may signal."""
    alive = []
    for pid in pids:
        try:
            os.kill(pid, 0)
            alive.append(pid)
        except ProcessLookupError:
            pass
        except PermissionError:
            # Exists, but belongs to another uid: nothing we send will stop
            # it, so waiting on it would only run out the deadline.
            pass
    return alive


def stop_child(child: subprocess.Popen | None, receipt: dict) -> None:
    """Stop the child and every descendant before releasing GPU ownership."""
    if child is None or child.poll() is not None:
        return
    receipt["child_terminated_by_wrapper"] = True
    tree = descendant_pids(child.pid)
    receipt["child_descendants_signalled"] = tree
    unsignallable = set(_signal_tree(child, tree, signal.SIGTERM))
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and (child.poll() is None or _alive(tree)):
        time.sleep(0.5)
    if child.poll() is None or _alive(tree):
        # Re-snapshot: a pid from the first snapshot may have exited and been
        # reused by an unrelated process of our own uid since then.
        tree = sorted(set(tree) & set(descendant_pids(child.pid)) | set(_alive(tree)))
        unsignallable |= set(_signal_tree(child, _alive(tree), signal.SIGKILL))
        receipt["child_killed_by_wrapper"] = True
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
    survivors = _alive(tree)
    if survivors:
        receipt["child_descendants_surviving"] = survivors
    if unsignallable:
        receipt["child_descendants_unsignallable"] = sorted(unsignallable)
    receipt["command_returncode"] = child.returncode


def observe_cpg(lock, task, url):
    try:
        from .run_qualification_owned import observe
    except ImportError:
        from run_qualification_owned import observe
    return observe(lock, task, url)


def observe_desktop_lease(path, session, label, url):
    import math

    from mlx2.desktop_campaign_worker import Radio, verify_claim
    try:
        from .qualification_gpu_ownership import _ancestor_pids
        from .run_qualification_owned import validate_live_lease
    except ImportError:
        from qualification_gpu_ownership import _ancestor_pids
        from run_qualification_owned import validate_live_lease
    owner = json.loads(Path(path).read_text())
    if (owner.get("agent") != "mlx2_desktop_campaign"
            or type(owner.get("pid")) is not int
            or owner["pid"] not in _ancestor_pids(os.getpid())
            or owner.get("job_id") != label or owner.get("cpg_session") != session
            or "release_task" in owner or "complete_worker" in owner or "finished_at" in owner):
        raise RuntimeError("desktop CPG lease does not match this live ancestor/job")
    expiry = owner.get("expires")
    if type(expiry) not in (int, float) or not math.isfinite(expiry) or expiry <= time.time() + 15:
        raise RuntimeError("desktop CPG lease is expired")
    verify_claim(owner.get("claim") or {}, task=owner.get("cpg_task"),
                 agent=owner.get("agent_id"), generation=owner.get("cpg_generation"), now=0)
    if "renewal" in owner:
        verify_claim(owner["renewal"], task=owner["cpg_task"],
                     generation=owner["cpg_generation"], renewed=True)
    radio = Radio(url)
    radio.call("join_session", {"session_id": owner["cpg_session"], "agent_id": owner["agent_id"]})
    live = radio.call("get_nodes", {"node_ids": [owner["cpg_task"]]})
    return owner, validate_live_lease(live, owner)


def observe_ownership(args):
    if args.desktop_cpg_lease is not None:
        return observe_desktop_lease(args.desktop_cpg_lease, args.session, args.label, args.mcp_url)
    return observe_cpg(args.cpg_owner_lock, args.cpg_task, args.mcp_url)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--label", required=True)
    parser.add_argument("--cpg-owner-lock", type=Path, default=os.environ.get("CPG_OWNER_LOCK"))
    parser.add_argument("--cpg-task", default=os.environ.get("CPG_TASK"))
    parser.add_argument("--desktop-cpg-lease", type=Path)
    parser.add_argument("--mcp-url", default=os.environ.get("CPG_MCP_URL", "http://127.0.0.1:8766/mcp"))
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.desktop_cpg_lease is None and (args.cpg_owner_lock is None or not args.cpg_task):
        parser.error("a CPG owner lock/task or desktop CPG lease is required")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    attempt_id = uuid.uuid4().hex
    logical_run_id = f"{args.session}:{args.label}"
    immutable_receipt = _immutable_receipt_path(args.receipt, attempt_id)
    WAITERS.mkdir(parents=True, exist_ok=True)
    created = int(time.time())
    waiter = WAITERS / f"{args.session}.{args.label}.{os.getpid()}.{created}"
    receipt = {
        "schema": "mlx2.gpu-lock-window.v1",
        "attempt_id": attempt_id,
        "logical_run_id": logical_run_id,
        "immutable_receipt": str(immutable_receipt),
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
        _atomic_write_json(args.receipt, receipt)

    backups: list[tuple[Path, Path | None]] = []
    child = None
    rc = 1
    waiter.touch()
    save()
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    previous_handlers = {
        signum: signal.signal(signum, interrupted)
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    }
    try:
        # At the head of the queue, wait for the current owner to release the
        # locks and its model processes to exit instead of failing and
        # re-queueing at the back (which starves a waiter while other
        # sessions run back-to-back windows).  The checks below still refuse
        # anything that persists past the wait.
        while True:
            predecessor = earlier_waiter(waiter, created)
            if predecessor is None:
                held = [str(path) for path in LOCKS if path.is_dir()]
                foreign = foreign_model_processes() if not held else []
                if not held and not foreign:
                    break
                predecessor = f"lock owner ({'held' if held else 'foreign process'})"
                if time.time() - created > HEAD_WAIT_SECONDS:
                    break
            receipt["waiting_behind"] = predecessor
            waiter.touch()
            save()
            time.sleep(15)
        receipt.pop("waiting_behind", None)
        foreign = foreign_model_processes()
        if foreign:
            raise RuntimeError(f"foreign model process present: {foreign[:5]}")
        cpg_owner, live_lease = observe_ownership(args)
        owner = {
            "lease_id": f"{args.session}-{args.label}",
            "session": args.session,
            "label": args.label,
            "pid": os.getpid(),
            "since": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "cpg_used": True,
            "cpg_owner": cpg_owner,
            "live_lease": live_lease,
            "cpg_task": cpg_owner["cpg_task"],
            "cpg_session": cpg_owner["cpg_session"],
            "cpg_generation": cpg_owner["cpg_generation"],
            "agent_id": cpg_owner["agent_id"],
        }
        receipt["lock_state_before"], backups = acquire(owner)
        receipt["owner"] = owner
        receipt["status"] = "running"
        waiter.unlink(missing_ok=True)
        save()
        child = subprocess.Popen(command, start_new_session=True)
        receipt["child_pid"] = child.pid
        save()
        last_confirmed = time.time()
        while True:
            try:
                rc = child.wait(timeout=10)
                break
            except subprocess.TimeoutExpired:
                try:
                    renewed_owner, live = observe_ownership(args)
                except RuntimeError:
                    raise  # an ownership verdict, never transient
                except Exception as exc:  # noqa: BLE001 - transport failure
                    now = time.time()
                    expiry = (receipt.get("live_lease") or live_lease or {}).get(
                        "lease_expires_at")
                    if (now - last_confirmed > OBSERVE_TRANSIENT_GRACE_SECONDS
                            or type(expiry) not in (int, float) or expiry <= now + 15):
                        raise
                    errors = receipt.setdefault("transient_observe_errors", [])
                    errors.append({"at": now, "error": f"{type(exc).__name__}: {exc}"[:300]})
                    del errors[:-20]
                    save()
                    continue
                last_confirmed = time.time()
                if any(renewed_owner.get(key) != cpg_owner.get(key)
                       for key in ("pid", "agent_id", "cpg_task", "cpg_session", "cpg_generation")):
                    raise RuntimeError("CPG ownership changed during GPU unit")
                receipt["live_lease"] = live
                save()
        receipt["command_returncode"] = rc
        receipt["status"] = "command_completed" if rc == 0 else "command_failed"
    except KeyboardInterrupt:
        rc = 130
        receipt["status"] = "interrupted"
        receipt["error"] = "KeyboardInterrupt"
        stop_child(child, receipt)
    except BaseException as exc:  # noqa: BLE001 - persist and retire every child
        receipt["status"] = "error"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        receipt["traceback"] = traceback.format_exc(limit=8)
        stop_child(child, receipt)
    finally:
        # Ownership must never be released while a child can resume GPU work.
        try:
            stop_child(child, receipt)
        except Exception as exc:  # noqa: BLE001 - the lock release below must run
            receipt["stop_child_error"] = f"{type(exc).__name__}: {exc}"
            receipt["status"] = "stop_child_failed"
            rc = 1
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
        receipt["updated_at"] = time.time()
        _save_terminal(args.receipt, immutable_receipt, receipt)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

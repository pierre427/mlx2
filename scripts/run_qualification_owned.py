#!/usr/bin/env python3
"""Run one bounded qualification job only while its live CPG lease is owned.

Launch beneath cpg_job with its dedicated owner lock. The child command normally
uses run_with_gpu_locks.py to own both host locks. Only an exact
@CPG_GENERATION@ argument is substituted, from the validated owner receipt.

This optional coordinator reuses the existing paired host-lock protocol; it
adds no dependency to the serving runtime. The MCP client is needed only when
this script is run against a configured coordinator endpoint.
"""

# Invalid remote protocol types are operational lease failures.
# ruff: noqa: TRY004
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path

if __package__:
    from .qualification_gpu_ownership import _ancestor_pids, validate_cpg_records
else:
    from qualification_gpu_ownership import _ancestor_pids, validate_cpg_records


def validate_live_lease(result, owner, *, now=None):
    now = time.time() if now is None else now
    if type(now) not in (int, float) or not math.isfinite(now):
        raise RuntimeError("live CPG validation clock is malformed")
    nodes = result.get("nodes") if isinstance(result, dict) else None
    if not isinstance(nodes, list) or len(nodes) != 1 or result.get("missing"):
        raise RuntimeError("live CPG task is absent or ambiguous")
    node = nodes[0]
    if not isinstance(node, dict):
        raise RuntimeError("live CPG task is malformed")
    payload = node.get("payload")
    if (
        node.get("id") != owner["cpg_task"]
        or node.get("session_id") != owner["cpg_session"]
        or node.get("type") != "TASK"
        or not isinstance(payload, dict)
        or payload.get("status") != "in_progress"
        or payload.get("owner_agent_id") != owner["agent_id"]
        or payload.get("lease_generation") != owner["cpg_generation"]
    ):
        raise RuntimeError("live CPG owner, task, session or generation changed")
    expiry = payload.get("lease_expires_at")
    if type(expiry) not in (int, float) or not math.isfinite(expiry) or expiry <= now:
        raise RuntimeError("live CPG lease is expired or malformed")
    return {
        key: payload.get(key)
        for key in ("status", "owner_agent_id", "lease_generation", "lease_expires_at")
    }


def _structured(result):
    if result.isError:
        raise RuntimeError("CPG read failed")
    if result.structuredContent is not None:
        value = result.structuredContent
    else:
        value = json.loads(
            "".join(item.text for item in result.content if getattr(item, "text", None))
        )
    if not isinstance(value, dict):
        raise RuntimeError("CPG response is malformed")
    return value


async def _read_live(url, owner):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with (
        streamablehttp_client(url) as (read, write, _),
        ClientSession(read, write) as client,
    ):
        await client.initialize()
        joined = _structured(
            await client.call_tool(
                "join_session",
                {"session_id": owner["cpg_session"], "agent_id": owner["agent_id"]},
            )
        )
        if joined.get("session_id") != owner["cpg_session"]:
            raise RuntimeError("CPG joined another session")
        return _structured(
            await client.call_tool(
                "get_nodes",
                {"node_ids": [owner["cpg_task"]]},
            )
        )


def observe(lock, task, url):
    owner = json.loads((lock / "owner.json").read_text())
    if not isinstance(owner, dict):
        raise RuntimeError("CPG owner receipt is malformed")
    log = owner.get("log")
    if not isinstance(log, str) or not log:
        raise RuntimeError("CPG owner log is absent")
    radio = json.loads(Path(log + ".radio.json").read_text())
    validate_cpg_records(
        task_id=task,
        generation=owner.get("cpg_generation"),
        cpg_owner=owner,
        radio=radio,
        ancestor_pids=_ancestor_pids(os.getpid()),
    )
    result = asyncio.run(asyncio.wait_for(_read_live(url, owner), timeout=15))
    return owner, validate_live_lease(result, owner)


def stop_child(child):
    if child is None or child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        child.wait(timeout=15)
        return
    try:
        child.wait(timeout=75)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=15)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpg-owner-lock", type=Path, required=True)
    parser.add_argument("--cpg-task", required=True)
    parser.add_argument("--mcp-url", default="http://127.0.0.1:8766/mcp")
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not math.isfinite(args.timeout) or not 1 <= args.timeout <= 28800:
        parser.error("a command and timeout in 1..28800 seconds are required")
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    # Refuse replacement before any GPU child can be launched.
    descriptor = os.open(args.receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    report = {
        "schema": "mlx2.owned-qualification-job.v1",
        "started_at": time.time(),
        "observations": [],
        "command": command,
    }
    child = None
    code = 1

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupted)
    try:
        owner, live = observe(args.cpg_owner_lock, args.cpg_task, args.mcp_url)
        report["owner"] = owner
        report["observations"].append({"at": time.time(), "live": live})
        command = [
            str(owner["cpg_generation"]) if value == "@CPG_GENERATION@" else value
            for value in command
        ]
        child = subprocess.Popen(command, start_new_session=True)
        report["child_pid"] = child.pid
        deadline = time.monotonic() + args.timeout
        while child.poll() is None:
            if time.monotonic() >= deadline:
                raise TimeoutError("bounded qualification job deadline reached")
            try:
                child.wait(timeout=min(10, max(0.01, deadline - time.monotonic())))
            except subprocess.TimeoutExpired:
                current, live = observe(
                    args.cpg_owner_lock,
                    args.cpg_task,
                    args.mcp_url,
                )
                if any(
                    current.get(key) != owner.get(key)
                    for key in (
                        "pid",
                        "agent_id",
                        "cpg_session",
                        "cpg_task",
                        "cpg_generation",
                    )
                ):
                    raise RuntimeError("CPG owner receipt changed during the job")
                report["observations"].append({"at": time.time(), "live": live})
        current, live = observe(args.cpg_owner_lock, args.cpg_task, args.mcp_url)
        if any(
            current.get(key) != owner.get(key)
            for key in ("pid", "agent_id", "cpg_session", "cpg_task", "cpg_generation")
        ):
            raise RuntimeError("CPG owner changed before job completion")
        report["observations"].append({"at": time.time(), "live": live, "final": True})
        code = child.returncode
        report["status"] = "completed" if code == 0 else "child_failed"
    except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 - stop any owned child
        report["status"] = "stopped"
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        stop_child(child)
        report["child_returncode"] = child.returncode if child else None
        report["finished_at"] = time.time()
        with os.fdopen(descriptor, "w") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

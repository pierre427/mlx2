#!/usr/bin/env python3
"""Run an aMoEba build one stage per GPU lease, at night, yielding between stages.

Pierre 2026-11-21: big carves run overnight (23:00-07:00 local) and pause between
stages so other GPU work can interleave.  Each remaining stage of ``--out`` runs as
``mlx2 amoeba resume --out OUT --stages STAGE`` under its own CPG lease and paired
GPU locks (cpg_job -> run_with_gpu_locks), so the lease is released at every stage
boundary.  Before each stage the driver waits until

* the local clock is inside the night window and before ``--latest-start``,
* no other live waiter is queued in the shared waiter ledger (the same liveness
  rule as run_with_gpu_locks: live pid and a waiter file touched within
  ``WAITER_FRESH_S``), and the GPU locks have no owner,
* port 8282 is not serving (production stays Pierre's call),

then sleeps ``--yield-s`` after each stage so other sessions' retry loops can take
the lease.  ``--then`` commands (removal check, serving bundle) run after the last
stage under the same gating, each with its own lease.  A failed stage or step stops
the driver; rerun it to continue (done stages are skipped).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

WAITERS = Path("/Users/Shared/mlxuag/gpu.lock.waiters")
LOCK_OWNERS = (Path("/tmp/gpu.lock/owner.json"), Path("/Users/Shared/mlxuag/gpu.lock/owner.json"))
WAITER_FRESH_S = 120  # run_with_gpu_locks touches a live waiter every 15 s
LEASE_REFUSED = 75    # cpg_job's exit code when another job holds the GPU lease
# The GPU-lease helper and its interpreter (set both to use the lease queue).
CPG_PY = os.environ.get("AMOEBA_CPG_PYTHON", sys.executable)
CPG_JOB = os.environ.get("AMOEBA_CPG_JOB", "cpg_job.py")


def parse_hhmm(s: str) -> dt.time:
    h, m = s.split(":")
    return dt.time(int(h), int(m))


def in_window(now: dt.time, start: dt.time, end: dt.time, latest: dt.time | None = None) -> bool:
    """Whether ``now`` is inside [start, end) (wrapping midnight) and not past ``latest``."""
    inside = (start <= now < end) if start <= end else (now >= start or now < end)
    if not inside or latest is None:
        return inside
    # latest is measured on the same night: after midnight, compare directly; before midnight it never binds
    if start > end and now >= start:
        return True if latest < start else now < latest
    return now < latest


def live_waiters(now: float | None = None, waiters: Path = WAITERS, me: int | None = None) -> list[str]:
    """Waiter files of live, fresh run_with_gpu_locks wrappers (names ``session.label.pid.created``)."""
    now = time.time() if now is None else now
    out = []
    if not waiters.is_dir():
        return out
    for path in sorted(waiters.iterdir()):
        fields = path.name.rsplit(".", 3)
        if len(fields) != 4 or not path.is_file():
            continue
        try:
            pid = int(fields[-2])
            if me is not None and pid == me:
                continue
            os.kill(pid, 0)
        except (ValueError, ProcessLookupError, PermissionError):
            continue
        if now - path.stat().st_mtime > WAITER_FRESH_S:
            continue  # a parked hold or an abandoned file, as run_with_gpu_locks reads it
        out.append(path.name)
    return out


def lock_owner() -> str | None:
    for p in LOCK_OWNERS:
        if p.exists():
            try:
                return json.loads(p.read_text()).get("agent_id") or str(p)
            except (OSError, ValueError):
                return str(p)
    return None


def port_serving(port: int = 8282) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def stage_status(py: str, out: str, env: dict) -> dict:
    r = subprocess.run([py, "-m", "mlx2.experimental.amoeba.cli", "status", "--out", out, "--json"],
                       capture_output=True, text=True, env=env, check=True)
    return {k: (v.get("status") if isinstance(v, dict) else v) for k, v in json.loads(r.stdout)["stages"].items()}


def log(msg: str) -> None:
    print(f"[staged {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def wait_turn(a) -> None:
    start, end = parse_hhmm(a.night_start), parse_hhmm(a.night_end)
    latest = parse_hhmm(a.latest_start) if a.latest_start else None
    said = None
    while True:
        now = dt.datetime.now().time()
        why = None
        if not in_window(now, start, end, latest):
            why = f"outside the night window {a.night_start}-{a.night_end} (latest start {a.latest_start})"
        elif (w := live_waiters(me=os.getpid())):
            why = f"yielding to queued GPU work {w}"
        elif (o := lock_owner()):
            why = f"GPU lock held by {o}"
        elif port_serving():
            why = ":8282 is serving"
        if why is None:
            return
        if why != said:
            log(f"waiting: {why}")
            said = why
        time.sleep(a.poll_s)


def run_leased(a, label: str, argv: list[str], env: dict) -> int:
    logp = Path(a.log_dir) / f"{label}.log"
    if logp.exists():
        logp = logp.with_name(f"{label}.{int(time.time())}.log")
    lock_dir = f"/tmp/{label}-cpg.lock"
    cmd = [CPG_PY, CPG_JOB, "run", "--label", label, "--out", str(logp), "--lease", "--lock",
           "--lock-dir", lock_dir, "--require-radio", "--heartbeat-s", "60", "--",
           CPG_PY, str(Path(a.tree) / "scripts" / "run_with_gpu_locks.py"), "--session", a.session,
           "--label", label, "--cpg-owner-lock", lock_dir, "--cpg-task", a.cpg_task,
           "--receipt", str(logp.with_suffix(".lock-receipt.json")), "--"] + argv
    log(f"start {label}: {shlex.join(argv)[:300]}")
    rc = subprocess.call(cmd, cwd=a.tree, env=env)
    log(f"done {label}: rc={rc} (log {logp})")
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="the aMoEba build directory")
    ap.add_argument("--create", default="", help="shell command that creates --out if it does not exist yet")
    ap.add_argument("--tree", required=True, help="mlx2 tree (snapshot) the stages run from")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--session", required=True)
    ap.add_argument("--label", required=True, help="label prefix; each stage gets <label>-<stage>")
    ap.add_argument("--cpg-task", required=True)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--night-start", default="23:00")
    ap.add_argument("--night-end", default="07:00")
    ap.add_argument("--latest-start", default="", help="no new stage after this local time (e.g. 05:30)")
    ap.add_argument("--yield-s", type=float, default=90.0)
    ap.add_argument("--poll-s", type=float, default=30.0)
    ap.add_argument("--then", action="append", default=[], help="shell command run after the last stage")
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE for every child")
    ap.add_argument("--resume-arg", action="append", default=[],
                    help="extra argument for every stage's `mlx2 amoeba resume` (e.g. --swap-growth-gb=6)")
    a = ap.parse_args(argv)
    Path(a.log_dir).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(PYTHONPATH=f"{a.tree}/src:{a.tree}", HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")
    for kv in a.env:
        k, _, v = kv.partition("=")
        env[k] = v
    out = Path(a.out)
    if not (out.is_dir() and any(out.iterdir())):
        if not a.create:
            raise SystemExit(f"{out} does not exist and no --create command was given")
        log(f"creating {out}")
        rc = subprocess.call(a.create, shell=True, cwd=a.tree, env=env)
        if rc != 0:
            log(f"create failed rc={rc}")
            return rc
    while True:
        status = stage_status(a.python, a.out, env)
        todo = [s for s, v in status.items() if v != "done"]
        if not todo:
            break
        stage = todo[0]
        wait_turn(a)
        rc = run_leased(a, f"{a.label}-{stage}",
                        [a.python, "-u", "-m", "mlx2.experimental.amoeba.cli", "resume", "--out", a.out,
                         "--stages", stage, *a.resume_arg], env)
        if rc == LEASE_REFUSED:
            # Another job took the lease between our check and our claim: wait for it, then retry.
            log(f"lease refused for {stage}; waiting for the holder")
            continue
        if rc != 0:
            return rc
        if stage_status(a.python, a.out, env).get(stage) != "done":
            log(f"stage {stage} exited rc=0 but is not done; stopping")
            return 3
        log(f"yielding {a.yield_s:.0f} s")
        time.sleep(a.yield_s)
    for i, cmd in enumerate(a.then):
        while True:
            wait_turn(a)
            rc = run_leased(a, f"{a.label}-then{i + 1}", ["/bin/zsh", "-c", cmd], env)
            if rc != LEASE_REFUSED:
                break
            log(f"lease refused for step {i + 1}; waiting for the holder")
        if rc != 0:
            return rc
        time.sleep(a.yield_s)
    log("all stages and steps done")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Validate externally owned CPG leases and paired GPU lock receipts."""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from pathlib import Path

HOST_LOCKS = (Path("/Users/Shared/mlxuag/gpu.lock"), Path("/tmp/gpu.lock"))


def _ancestor_pids(start_pid: int) -> set[int]:
    """Return a bounded live ancestor chain, including the starting process."""
    result: set[int] = set()
    pid = start_pid
    for _ in range(64):
        if pid <= 1 or pid in result:
            break
        result.add(pid)
        output = subprocess.check_output(
            ["ps", "-o", "ppid=", "-p", str(pid)], text=True
        ).strip()
        if not output:
            break
        parent = int(output)
        if parent == pid:
            break
        pid = parent
    return result


def validate_cpg_records(
    *,
    task_id: str,
    generation: int,
    cpg_owner: dict,
    radio: dict,
    ancestor_pids: set[int],
    now: float | None = None,
) -> dict:
    """Validate trusted CPG owner and live radio claim/renewal evidence."""
    now = time.time() if now is None else now
    if type(now) not in (int, float) or not math.isfinite(float(now)):
        raise RuntimeError("CPG validation clock must be finite")
    if (
        not isinstance(task_id, str)
        or not task_id.strip()
        or type(generation) is not int
        or generation < 1
    ):
        raise RuntimeError("a task id and positive CPG generation are required")
    if not isinstance(cpg_owner, dict) or cpg_owner.get("agent") != "cpg_job":
        raise RuntimeError("dedicated CPG lock is not owned by cpg_job")
    cpg_pid = cpg_owner.get("pid")
    if type(cpg_pid) is not int or cpg_pid not in ancestor_pids:
        raise RuntimeError("CPG owner is not a live process ancestor")
    if (
        cpg_owner.get("cpg_task") != task_id
        or cpg_owner.get("cpg_generation") != generation
    ):
        raise RuntimeError("CPG owner task or generation mismatch")
    for key in ("agent_id", "cpg_session", "cpg_worker", "log"):
        if not isinstance(cpg_owner.get(key), str) or not cpg_owner[key]:
            raise RuntimeError(f"CPG owner lacks {key}")
    if not isinstance(radio, dict) or radio.get("agent_id") != cpg_owner["agent_id"]:
        raise RuntimeError("CPG radio receipt has another owner")
    if "release_task" in radio or "complete_worker" in radio:
        raise RuntimeError("CPG radio receipt is already closed")
    claim = radio.get("claim_task")
    if not isinstance(claim, dict) or claim.get("claimed") is not True:
        raise RuntimeError("CPG radio receipt has no successful task claim")
    if (
        claim.get("task_id") != task_id
        or claim.get("lease_generation") != generation
        or claim.get("owner_agent_id") != cpg_owner["agent_id"]
    ):
        raise RuntimeError("CPG task claim does not match the owner")
    renewal = radio.get("renew_lease", claim)
    if (
        not isinstance(renewal, dict)
        or renewal.get("task_id") != task_id
        or renewal.get("lease_generation") != generation
        or (renewal is not claim and renewal.get("renewed") is not True)
        or type(renewal.get("lease_expires_at")) not in (int, float)
    ):
        raise RuntimeError("CPG lease renewal is mismatched or expired")
    expiry = renewal["lease_expires_at"]
    if not math.isfinite(float(expiry)) or expiry <= now:
        raise RuntimeError("CPG lease expiry is not finite")
    return {
        "cpg_owner": cpg_owner,
        "radio_claim": claim,
        "radio_renewal": radio.get("renew_lease"),
    }


def validate_ownership_records(
    *,
    task_id: str,
    generation: int,
    cpg_owner: dict,
    radio: dict,
    shared_owner: dict,
    temporary_owner: dict,
    ancestor_pids: set[int],
    now: float | None = None,
) -> dict:
    """Pure validation for CPU tests; no claims are accepted from env flags."""
    now = time.time() if now is None else now
    cpg = validate_cpg_records(
        task_id=task_id,
        generation=generation,
        cpg_owner=cpg_owner,
        radio=radio,
        ancestor_pids=ancestor_pids,
        now=now,
    )
    if not isinstance(shared_owner, dict) or shared_owner != temporary_owner:
        raise RuntimeError("paired GPU lock owner receipts are missing or differ")
    pair_pid = shared_owner.get("pid")
    if type(pair_pid) is not int or pair_pid not in ancestor_pids:
        raise RuntimeError("paired GPU lock owner is not a live process ancestor")
    for key in ("lease_id", "session", "label"):
        if not isinstance(shared_owner.get(key), str) or not shared_owner[key]:
            raise RuntimeError(f"paired GPU lock receipt lacks {key}")
    return {
        **cpg,
        "paired_gpu_owners": {"shared": shared_owner, "temporary": temporary_owner},
    }


def require_qualification_lease(
    *, task_id: str, cpg_owner_lock: Path, generation: int, now: float | None = None
) -> dict:
    cpg_owner_path = Path(cpg_owner_lock) / "owner.json"
    if not cpg_owner_path.is_file():
        raise RuntimeError(f"missing dedicated CPG owner receipt: {cpg_owner_path}")
    cpg_owner = json.loads(cpg_owner_path.read_text())
    log = cpg_owner.get("log") if isinstance(cpg_owner, dict) else None
    if not isinstance(log, str) or not log:
        raise RuntimeError("CPG owner receipt lacks its radio log path")
    radio_path = Path(log + ".radio.json")
    if not radio_path.is_file():
        raise RuntimeError(f"missing CPG radio lease receipt: {radio_path}")
    radio = json.loads(radio_path.read_text())
    owners = []
    for lock in HOST_LOCKS:
        owner_path = lock / "owner.json"
        if not owner_path.is_file():
            raise RuntimeError(f"paired GPU lock is not owned: {owner_path}")
        owners.append(json.loads(owner_path.read_text()))
    ancestors = _ancestor_pids(os.getpid())
    return validate_ownership_records(
        task_id=task_id,
        generation=generation,
        cpg_owner=cpg_owner,
        radio=radio,
        shared_owner=owners[0],
        temporary_owner=owners[1],
        ancestor_pids=ancestors,
        now=now,
    )

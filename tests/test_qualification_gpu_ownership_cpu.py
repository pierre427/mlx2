from __future__ import annotations

import pytest

from scripts.qualification_gpu_ownership import (
    validate_cpg_records,
    validate_ownership_records,
)


def records(now=1000.0):
    task = "cpg-task-42"
    generation = 7
    cpg = {
        "agent": "cpg_job",
        "pid": 10,
        "agent_id": "agent-a",
        "cpg_session": "session-a",
        "cpg_worker": "worker-a",
        "cpg_task": task,
        "cpg_generation": generation,
        "log": "/tmp/cpg.log",
    }
    claim = {
        "claimed": True,
        "task_id": task,
        "lease_generation": generation,
        "owner_agent_id": "agent-a",
        "lease_expires_at": now + 60,
    }
    radio = {
        "agent_id": "agent-a",
        "claim_task": claim,
        "renew_lease": {**claim, "renewed": True, "lease_expires_at": now + 60},
    }
    pair = {
        "pid": 11,
        "lease_id": "session-label",
        "session": "session",
        "label": "vlm-qualification",
    }
    return {
        "task_id": task,
        "generation": generation,
        "cpg_owner": cpg,
        "radio": radio,
        "shared_owner": pair,
        "temporary_owner": pair.copy(),
        "ancestor_pids": {10, 11, 12},
        "now": now,
    }


def test_valid_live_cpg_claim_renewal_and_paired_ancestor_locks():
    result = validate_ownership_records(**records())
    assert result["cpg_owner"]["cpg_task"] == "cpg-task-42"
    assert (
        result["paired_gpu_owners"]["shared"]
        == result["paired_gpu_owners"]["temporary"]
    )


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.update(generation=8),
        lambda r: r["cpg_owner"].update(pid=99),
        lambda r: r["cpg_owner"].update(cpg_task="other"),
        lambda r: r["radio"]["renew_lease"].update(lease_expires_at=999.0),
        lambda r: r["radio"]["renew_lease"].update(lease_expires_at=float("nan")),
        lambda r: r["radio"]["renew_lease"].update(lease_expires_at=float("inf")),
        lambda r: r["radio"].update(release_task={"released": True}),
        lambda r: r["temporary_owner"].update(pid=22),
        lambda r: r["shared_owner"].update(pid=99),
    ],
)
def test_invalid_or_expired_owner_evidence_fails_closed(change):
    evidence = records()
    change(evidence)
    with pytest.raises(RuntimeError):
        validate_ownership_records(**evidence)


def test_cpg_validation_rejects_nonfinite_validation_clock():
    values = records(now=float("nan"))
    values.pop("shared_owner")
    values.pop("temporary_owner")
    with pytest.raises(RuntimeError, match="clock"):
        validate_cpg_records(**values)

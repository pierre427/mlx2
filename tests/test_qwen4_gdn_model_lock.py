"""CPU-only checks for the model gate's external GPU ownership preflight."""

import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "qwen4_gdn_model_gate",
    Path(__file__).parents[1] / "scripts/qualify_qwen4_gdn_replay_model.py",
)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


@pytest.fixture
def lease(tmp_path, monkeypatch):
    monkeypatch.setattr(gate.os, "getppid", lambda: 123)
    lock = tmp_path / "gpu.lock" / "owner.json"
    lock.parent.mkdir()
    monkeypatch.setattr(gate, "HOST_LOCK", lock)
    log = tmp_path / "job.log"
    owner = dict(agent="cpg_job", agent_id="job-gdn-123", pid=123,
                 cpg_session="session", cpg_task="gpu-task", cpg_generation=7,
                 cpg_worker="worker", log=str(log))
    claim = dict(claimed=True, task_id="gpu-task", lease_generation=7,
                 owner_agent_id="job-gdn-123", lease_expires_at=2000)
    radio = dict(agent_id="job-gdn-123", claim_task=claim)
    lock.write_text(json.dumps(owner))
    radio_path = Path(str(log) + ".radio.json")
    radio_path.write_text(json.dumps(radio))
    return lock, radio_path, owner, radio


def test_matching_parent_lock_and_unexpired_cpg_receipt(lease):
    _, _, owner, radio = lease
    assert gate._require_lock("gpu-task", now=1000) == {
        "host_lock": owner, "radio_claim": radio["claim_task"], "radio_renewal": None,
    }


def test_current_renewal_receipt_does_not_require_claim_only_fields(lease):
    _, radio_path, _, radio = lease
    radio["renew_lease"] = dict(renewed=True, task_id="gpu-task",
                                lease_generation=7, lease_expires_at=2500)
    radio_path.write_text(json.dumps(radio))
    assert gate._require_lock("gpu-task", now=2100)["radio_renewal"] == radio["renew_lease"]


@pytest.mark.parametrize("change", [
    lambda owner, radio: owner.update(pid=999),
    lambda owner, radio: owner.update(cpg_task="another-task"),
    lambda owner, radio: owner.update(cpg_generation=8),
    lambda owner, radio: radio["claim_task"].update(lease_expires_at=1000),
    lambda owner, radio: radio.update(release_task={"released": True}),
    lambda owner, radio: radio.update(agent_id="foreign"),
])
def test_foreign_stale_or_closed_receipt_fails(lease, change):
    lock, radio_path, owner, radio = lease
    change(owner, radio)
    lock.write_text(json.dumps(owner))
    radio_path.write_text(json.dumps(radio))
    with pytest.raises(RuntimeError):
        gate._require_lock("gpu-task", now=1000)


def test_missing_host_lock_fails(lease):
    lock, _, _, _ = lease
    lock.unlink()
    with pytest.raises(RuntimeError, match="host lock"):
        gate._require_lock("gpu-task", now=1000)

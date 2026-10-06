from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "llada_denoising_ladder",
    ROOT / "scripts/run_llada_denoising_smoke_ladder.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _ownership_fixture(monkeypatch, tmp_path, *, now=100.0):
    locks = (tmp_path / "shared", tmp_path / "temporary")
    owner = {
        "session": "campaign",
        "label": "llada-q4-001",
        "lease_id": "campaign-llada-q4-001",
        "pid": 4242,
    }
    for lock in locks:
        lock.mkdir()
        (lock / "owner.json").write_text(json.dumps(owner))
    radio = tmp_path / "cpg.radio.json"
    radio.write_text(
        json.dumps(
            {
                "agent_id": "job-llada-q4-001-9988",
                "register_worker": {
                    "agent_id": "job-llada-q4-001-9988",
                    "worker_id": "worker",
                },
                "claim_task": {
                    "claimed": True,
                    "owner_agent_id": "job-llada-q4-001-9988",
                    "task_id": "task",
                    "lease_generation": 7,
                    "lease_expires_at": now + 60,
                },
            }
        )
    )
    monkeypatch.setattr(MODULE, "LOCKS", locks)
    monkeypatch.setattr(MODULE.os, "getppid", lambda: 4242)
    monkeypatch.setattr(MODULE.os, "kill", lambda pid, signal: None)
    return owner, radio


def test_ownership_binds_paired_locks_to_live_cpg(monkeypatch, tmp_path):
    owner, radio = _ownership_fixture(monkeypatch, tmp_path)
    receipt = MODULE.validate_ownership(
        "campaign", "llada-q4-001", radio, now=100.0
    )
    assert receipt["paired_locks"] == owner
    assert receipt["cpg_claim"]["lease_generation"] == 7


def test_ownership_rejects_closed_cpg_receipt(monkeypatch, tmp_path):
    _, radio = _ownership_fixture(monkeypatch, tmp_path)
    value = json.loads(radio.read_text())
    value["release_task"] = {"released": True}
    radio.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="already closed"):
        MODULE.validate_ownership("campaign", "llada-q4-001", radio, now=100.0)


def test_ownership_rejects_non_parent_lock_holder(monkeypatch, tmp_path):
    _, radio = _ownership_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(MODULE.os, "getppid", lambda: 9999)
    with pytest.raises(RuntimeError, match="direct parent"):
        MODULE.validate_ownership("campaign", "llada-q4-001", radio, now=100.0)


class LinearTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        # The fixed text has no instances of the exact filler token.
        units = messages[0]["content"].count(" record")
        return list(range(12 + units))


def test_calibration_requires_exact_total_canvas_prompt():
    row = MODULE.calibrate_prompt(LinearTokenizer(), 992, "nonce", "NEEDLE")
    assert row["prompt_tokens"] == 992
    assert row["filler_units"] > 0
    assert row["needle"] == "NEEDLE"


class EvenTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        units = messages[0]["content"].count(" record")
        return list(range(10 + 2 * units))


def test_calibration_fails_closed_when_exact_canvas_is_unreachable():
    with pytest.raises(ValueError, match="could not calibrate exact"):
        MODULE.calibrate_prompt(EvenTokenizer(), 991, "nonce", "NEEDLE")


def test_campaign_semantics_refuse_autoregressive_cache_claims():
    assert MODULE.CAMPAIGN_SEMANTICS["batch_width"] == 1
    assert MODULE.CAMPAIGN_SEMANTICS["total_canvas_tokens"] == [1024, 4096]
    assert MODULE.CAMPAIGN_SEMANTICS["generation"] == {
        "gen_length": 32,
        "block_length": 32,
        "steps": 32,
    }
    assert MODULE.CAMPAIGN_SEMANTICS["autoregressive_kv"] == "not_applicable"
    assert MODULE.CAMPAIGN_SEMANTICS["apcv2"] == "not_applicable"
    assert MODULE.CAMPAIGN_SEMANTICS["streaming"] == "not_applicable"


def _valid_result():
    return {
        "text": "LLADA-ABC123 is the code.",
        "canvas_token_ids": [1, 2, 3],
        "route": "denoising-exact",
        "artifact_fingerprint": "a" * 64,
        "stats": {
            "forwards": 32,
            "steps": 32,
            "prefix_snapshot": None,
            "prefix_snapshot_used": False,
            "active_block_postprocess": False,
            "postprocess_rows_per_forward": 1024,
            "active_block_head": False,
            "lm_head_rows_per_forward": 1024,
            "active_block_head_parity_checks": 0,
        },
    }


def test_result_gate_requires_exact_direct_route_and_no_cache_claim():
    checks = MODULE.validate_result(
        _valid_result(),
        expected_fingerprint="a" * 64,
        mask_token_id=126336,
        needle="LLADA-ABC123",
    )
    assert all(checks.values())
    assert checks["prefix_snapshot_unused"]


def test_result_gate_accepts_receipted_active_block_candidate():
    result = _valid_result()
    result["route"] = "denoising-exact-active-block"
    result["stats"]["active_block_postprocess"] = True
    result["stats"]["postprocess_rows_per_forward"] = 32
    checks = MODULE.validate_result(
        result,
        expected_fingerprint="a" * 64,
        mask_token_id=126336,
        needle="LLADA-ABC123",
        active_block_postprocess=True,
    )
    assert checks["active_block_receipt"]


def test_result_gate_accepts_verified_active_head_candidate():
    result = _valid_result()
    result["route"] = "denoising-exact-active-head"
    result["stats"].update(
        active_block_postprocess=True,
        postprocess_rows_per_forward=32,
        active_block_head=True,
        lm_head_rows_per_forward=32,
        active_block_head_parity_checks=32,
    )
    checks = MODULE.validate_result(
        result,
        expected_fingerprint="a" * 64,
        mask_token_id=126336,
        needle="LLADA-ABC123",
        active_block_postprocess=True,
        active_block_head=True,
        verify_active_block_head=True,
    )
    assert checks["active_head_receipt"]


def test_parity_only_receipt_records_but_does_not_require_semantic_needle():
    result = _valid_result()
    result["text"] = "visible but not the requested needle"
    checks = MODULE.validate_result(
        result,
        expected_fingerprint="a" * 64,
        mask_token_id=126336,
        needle="LLADA-ABC123",
        require_semantic_needle=False,
    )
    assert checks["semantic_needle"] is False


def test_parity_only_receipt_can_retain_empty_output_for_canvas_comparison():
    result = _valid_result()
    result["text"] = ""
    result["token_ids"] = []
    checks = MODULE.validate_result(
        result,
        expected_fingerprint="a" * 64,
        mask_token_id=126336,
        needle=None,
        require_visible_output=False,
    )
    assert checks["visible_output"] is False


@pytest.mark.parametrize(
    ("path", "value", "failure"),
    [
        (("route",), "other", "route_exact"),
        (("artifact_fingerprint",), "b" * 64, "fingerprint_exact"),
        (("stats", "forwards"), 31, "forwards_exact"),
        (("stats", "steps"), 31, "steps_exact"),
        (("stats", "prefix_snapshot"), {"layers": []}, "prefix_snapshot_absent"),
        (("stats", "prefix_snapshot_used"), True, "prefix_snapshot_unused"),
        (("canvas_token_ids",), [1, 126336], "no_mask_tokens"),
        (("text",), "  ", "visible_output"),
        (("text",), "wrong answer", "semantic_needle"),
    ],
)
def test_result_gate_fails_closed(path, value, failure):
    result = _valid_result()
    target = result
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(AssertionError, match=failure):
        MODULE.validate_result(
            result,
            expected_fingerprint="a" * 64,
            mask_token_id=126336,
            needle="LLADA-ABC123",
        )


def test_atomic_json_replaces_without_leaving_temporary_file(tmp_path):
    path = tmp_path / "receipt.json"
    MODULE.atomic_json(path, {"status": "running"})
    MODULE.atomic_json(path, {"status": "passed"})
    assert json.loads(path.read_text()) == {"status": "passed"}
    assert list(tmp_path.iterdir()) == [path]


def test_post_thermal_uses_full_policy_and_configured_interval(monkeypatch):
    policy = {
        "required_thermal_state": 0,
        "max_battery_temperature_c": 40.0,
        "max_virtual_temperature_c": 45.0,
        "require_temperatures": True,
        "sample_interval_seconds": 15,
        "post_sample_interval_seconds": 3,
    }
    samples = iter(
        [
            {
                "thermal_state": 0,
                "pmset_no_thermal_warning": True,
                "pmset_no_performance_warning": True,
                "pmset_no_cpu_power_warning": True,
                "battery_temperature_c": 40.5,
                "virtual_temperature_c": 44.0,
                "raw_evidence": {"probe": "first"},
            },
            {
                "thermal_state": 0,
                "pmset_no_thermal_warning": True,
                "pmset_no_performance_warning": True,
                "pmset_no_cpu_power_warning": True,
                "battery_temperature_c": 39.0,
                "virtual_temperature_c": 44.0,
                "raw_evidence": {"probe": "second"},
            },
        ]
    )

    def stable(sample, selected_policy):
        assert selected_policy is policy
        return sample["battery_temperature_c"] <= selected_policy[
            "max_battery_temperature_c"
        ]

    matrix = SimpleNamespace(
        sample_thermal=lambda command=None: next(samples),
        thermally_stable=stable,
    )
    sleeps = []
    monkeypatch.setattr(MODULE.time, "sleep", sleeps.append)
    result = MODULE.post_run_thermal(matrix, policy)

    assert sleeps == [3.0]
    assert result["stable"] == [False, True]
    assert result["breached"] is False
    assert all("raw_evidence" not in sample for sample in result["samples"])


def test_post_thermal_requires_two_consecutive_policy_failures(monkeypatch):
    policy = {"sample_interval_seconds": 0}
    rows = iter([{"stable": False}, {"stable": False}])
    matrix = SimpleNamespace(
        sample_thermal=lambda command=None: next(rows),
        thermally_stable=lambda sample, selected_policy: sample["stable"],
    )
    monkeypatch.setattr(MODULE.time, "sleep", lambda seconds: None)

    result = MODULE.post_run_thermal(matrix, policy)

    assert result["stable"] == [False, False]
    assert result["breached"] is True

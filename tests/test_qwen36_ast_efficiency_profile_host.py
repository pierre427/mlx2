from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts/profile_qwen36_ast_efficiency.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_counter_and_phase_deltas_are_round_local():
    module = load_module()
    before = {
        "external_rounds": 7,
        "external_progressive_verify_physical_launches": 5,
        "external_phase_draft_ns": 100,
        "reservation_bytes": 900,
    }
    after = {
        "external_rounds": 10,
        "external_progressive_verify_physical_launches": 8,
        "external_phase_draft_ns": 700,
        "reservation_bytes": 800,
    }
    delta = module.numeric_delta(before, after)
    assert delta["external_rounds"] == 3
    assert delta["external_progressive_verify_physical_launches"] == 3
    assert module.phase_nanoseconds(delta) == {"draft": 600}
    assert "reservation_bytes" not in module.selected_counters(delta)


def test_summary_exposes_launch_amortization_and_loss_location():
    module = load_module()
    ordinary = {
        "output_tokens": 40,
        "token_ids": list(range(40)),
        "decode_nanoseconds": 1_000,
        "target_forwards": [{}] * 40,
    }
    ast = {
        "output_tokens": 40,
        "token_ids": list(range(40)),
        "decode_nanoseconds": 2_000,
        "progressive_target_calls": [{}] * 10,
        "scheduler_delta": {
            "external_progressive_verify_physical_launches": 10,
            "external_progressive_verify_target_rows": 50,
            "proposed_tokens": 40,
            "accepted_proposals": 30,
        },
        "phase_nanoseconds": {"draft": 400, "progressive_verify": 1_200},
    }
    result = module.summarize(ordinary, ast)
    assert result["token_exact"] is True
    assert result["ast_tokens_per_physical_target_launch"] == 4
    assert result["ast_target_rows_per_physical_launch"] == 5
    assert result["ast_acceptance_rate"] == 0.75
    assert result["decode_wall_ratio_ast_over_ordinary"] == 2
    assert result["ast_synchronized_phase_share"]["progressive_verify"] == 0.6


def test_policy_and_gpu_owner_checks_fail_closed(tmp_path):
    module = load_module()
    policy = {
        "draft_model": "/draft", "draft_revision": "d", "target_revision": "t",
        "num_draft": 4, "target_verify_row_exact": True,
        "progressive_verification_tile": "auto",
    }
    module.validate_policy(policy, target_revision="t", draft_revision="d")
    with pytest.raises(ValueError, match="target_revision"):
        module.validate_policy(policy, target_revision="wrong", draft_revision="d")
    owners = []
    for index in range(2):
        path = tmp_path / f"owner-{index}.json"
        path.write_text(json.dumps({"session": "s", "lease_id": "l"}))
        owners.append(path)
    assert module.require_gpu_owners(
        {"GPUQ_SESSION": "s", "GPUQ_LEASE": "l"}, owners
    )["lease_id"] == "l"
    with pytest.raises(RuntimeError, match="does not match"):
        module.require_gpu_owners(
            {"GPUQ_SESSION": "s", "GPUQ_LEASE": "other"}, owners
        )


def test_dry_run_does_not_import_mlx(tmp_path, monkeypatch, capsys):
    module = load_module()
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({
        "draft_model": "/draft", "draft_revision": "d", "target_revision": "t",
        "num_draft": 4, "target_verify_row_exact": True,
        "progressive_verification_tile": "auto",
    }))
    monkeypatch.setitem(__import__("sys").modules, "mlx", None)
    assert module.main([
        "--model", str(tmp_path / "model"), "--policy", str(policy),
        "--out", str(tmp_path / "out.json"), "--expected-source", "source",
        "--expected-target-fingerprint", "artifact",
        "--expected-target-revision", "t", "--expected-draft-revision", "d",
    ]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["arms"] == ["ordinary", "ast"]
    assert plan["ast_device_synchronized_phase_timing"] is True
    assert not (tmp_path / "out.json").exists()


def test_wall_external_ordinary_plan_disables_instrumentation(tmp_path, capsys):
    module = load_module()
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({
        "draft_model": "/draft", "draft_revision": "d", "target_revision": "t",
        "num_draft": 4, "target_verify_row_exact": True,
        "progressive_verification_tile": "auto",
    }))
    assert module.main([
        "--model", str(tmp_path / "model"), "--policy", str(policy),
        "--out", str(tmp_path / "out.json"), "--expected-source", "source",
        "--expected-target-fingerprint", "artifact",
        "--expected-target-revision", "t", "--expected-draft-revision", "d",
        "--measurement-mode", "wall_external_ordinary",
    ]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["arms"] == ["ordinary", "external_ordinary"]
    assert plan["ast_device_synchronized_phase_timing"] is False
    assert plan["call_recorder"] is False
    assert plan["per_poll_device_sync"] is False
    assert plan["forced_after_prefill"] is True


def test_wall_ast_plan_keeps_policy_but_disables_timing(tmp_path, capsys):
    module = load_module()
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({
        "draft_model": "/draft", "draft_revision": "d", "target_revision": "t",
        "num_draft": 4, "target_verify_row_exact": True,
        "progressive_verification_tile": "auto",
    }))
    assert module.main([
        "--model", str(tmp_path / "model"), "--policy", str(policy),
        "--out", str(tmp_path / "out.json"), "--expected-source", "source",
        "--expected-target-fingerprint", "artifact",
        "--expected-target-revision", "t", "--expected-draft-revision", "d",
        "--measurement-mode", "wall_ast",
    ]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["arms"] == ["ordinary", "ast"]
    assert plan["call_recorder"] is False
    assert plan["external_round_timing"] is False
    assert plan["forced_after_prefill"] is False
    assert plan["cost_handoff_policy_exercised"] is True

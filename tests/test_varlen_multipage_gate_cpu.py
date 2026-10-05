"""The bounded diagnostic defaults to a GPU-free plan and marks fake evidence."""

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/research/varlen_multipage_gate.py"
spec = importlib.util.spec_from_file_location("varlen_multipage_gate", SCRIPT)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def test_default_plan_labels_native_gaps():
    result = gate.dry_run()
    assert result["gpu_executed"] is False
    assert result["planned_boundary_bytes"] == [63, 64, 65]
    assert "native command-buffer failure retirement" in result["not_proven"]
    assert "native COW byte copy" in result["not_proven"]


def test_cpu_fake_boundaries_churn_cow_and_failure_retirement():
    result = gate.cpu_fake()
    assert [case["pages"] for case in result["boundary_cases"]] == [1, 1, 2]
    assert [case["tokens"] for case in result["boundary_cases"]] == [63, 64, 65]
    assert result["unrelated_lane_churn"]
    assert result["shared_tail_cow_refused_before_write"]
    assert result["injected_failed_terminal_retired_without_publication"]
    assert not result["gpu_executed"] and not result["native_command_failure_tested"]

"""CPU-only preflight for the TensorFold cohort GPU A/B harness."""

import json
import subprocess
from pathlib import Path

import pytest

from scripts import qualify_qwen38_tensorfold_cohort as cohort

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/qualify_qwen38_tensorfold_cohort.py"


def _row(limit, *, prompt="code", rep=1, lane=0):
    return {
        "prompt": prompt,
        "sampling": "greedy",
        "rep": rep,
        "lane": lane,
        "warmup": False,
        "completion_tokens": 64,
        "makespan_seconds": 2.0,
        "output_sha256": "same",
        "speculation": {"tensorfold_target": {"cohort_limit": limit}},
    }


def _status(limit, *, rounds, width):
    return {
        "scheduler": {
            "external_tensorfold_cohort_limit": limit,
            "external_tensorfold_target_rounds": 3,
            "external_tensorfold_cohort_rounds": rounds,
            "external_tensorfold_cohort_max_width": width,
            "draft_fallbacks": 0,
            "recovery_checkpoint_restores": 0,
        }
    }


def test_dry_run_plans_only_singleton_and_width_four(tmp_path):
    command = [
        str(Path("~/Desktop/mlx2/.venv/bin/python")),
        str(SCRIPT),
        "--mlx2-root", str(ROOT),
        "--model", str(tmp_path / "model"),
        "--draft", str(tmp_path / "draft"),
        "--execution-policy", str(tmp_path / "policy.json"),
        "--tensorfold-root", str(tmp_path / "tensorfold"),
        "--output-dir", str(tmp_path / "output"),
        "--dry-run",
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    plan = json.loads(result.stdout)
    assert plan["limits"] == [1, 4]
    assert plan["concurrency"] == 4
    assert plan["controls"]["required_singleton_nonengagement"] is True
    assert plan["controls"]["required_width_four_receipt"] is True
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "limit, rounds, width",
    [(1, 0, 0), (4, 2, 4)],
)
def test_arm_validation_requires_truthful_engagement(limit, rounds, width):
    rows = [_row(limit, lane=lane) for lane in range(4)]
    summary = cohort.validate_arm(limit, rows, _status(limit, rounds=rounds, width=width))
    assert summary["code"]["samples"] == 4
    assert summary["code"]["median_batch_goodput_tps"] == 126.0


def test_width_four_arm_refuses_nonengagement():
    rows = [_row(4, lane=lane) for lane in range(4)]
    with pytest.raises(RuntimeError, match="did not observe width four"):
        cohort.validate_arm(4, rows, _status(4, rounds=1, width=2))

"""CPU-only CLI contract for the bounded Qwen3.8 physical B2-B4 probe."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/probe_qwen38_b2_crossover.py"
PLAN_ONLY = ("--dry-run", "--allow-unprovisioned-plan")


def test_default_remains_b2(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output-dir", str(tmp_path / "out"), *PLAN_ONLY],
        check=True, capture_output=True, text=True,
    )
    plan = json.loads(result.stdout)
    assert (plan["concurrency"], plan["cohort_limit"], plan["max_lanes"], plan["max_inflight"]) == (2, 2, 2, 2)
    assert isinstance(plan["input_readiness"]["ready"], bool)
    assert plan["input_readiness"]["ready"] or plan["input_readiness"]["failures"]


@pytest.mark.parametrize("width", (2, 3, 4))
def test_dry_run_bounds_server_and_client_to_requested_width(tmp_path, width):
    command = [sys.executable, str(SCRIPT), "--output-dir", str(tmp_path / "out"),
               *PLAN_ONLY, "--concurrency", str(width)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    plan = json.loads(result.stdout)
    assert plan["concurrency"] == plan["cohort_limit"] == width
    assert plan["max_lanes"] == plan["max_inflight"] == width
    assert plan["timeout_seconds"] == 480
    assert not (tmp_path / "out").exists()
    for arm in plan["arms"]:
        server, bench = arm["server"], arm["bench"]
        assert arm["environment"]["MLX2_TENSORFOLD_COHORT_LIMIT"] == str(width)
        assert server[server.index("--max-lanes") + 1] == str(width)
        assert server[server.index("--max-inflight") + 1] == str(width)
        assert bench[bench.index("--concurrency") + 1] == str(width)
        assert bench[bench.index("--label") + 1].endswith(f"-b{width}")


def test_explicit_server_capacity_can_exceed_b3_without_changing_cohort(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output-dir", str(tmp_path / "out"),
         *PLAN_ONLY, "--concurrency", "3", "--cohort-limit", "3",
         "--max-lanes", "4", "--max-inflight", "4"],
        check=True, capture_output=True, text=True,
    )
    plan = json.loads(result.stdout)
    assert (plan["concurrency"], plan["cohort_limit"], plan["max_lanes"], plan["max_inflight"]) == (3, 3, 4, 4)


@pytest.mark.parametrize("args", (
    ("--concurrency", "4", "--cohort-limit", "3"),
    ("--concurrency", "4", "--max-lanes", "3"),
    ("--concurrency", "4", "--max-inflight", "3"),
    ("--concurrency", "5"),
    ("--timeout-seconds", "481"),
))
def test_cli_refuses_unbounded_or_nonphysical_cell(tmp_path, args):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output-dir", str(tmp_path / "out"),
         *PLAN_ONLY, *args], check=False, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert not (tmp_path / "out").exists()

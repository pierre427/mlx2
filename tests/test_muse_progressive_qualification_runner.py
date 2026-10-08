from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_RUNNER = (
    Path(__file__).parents[1]
    / "qualification/runs/muse-progressive-qualification-m3-20261008"
    / "run_muse_progressive_m3.py"
)
# The public mirror does not carry qualification/ run directories; this
# runner test runs on the source tree only.
pytestmark = pytest.mark.skipif(
    not _RUNNER.is_file(), reason="qualification run directory not published"
)


def _runner():
    path = (
        Path(__file__).parents[1]
        / "qualification/runs/muse-progressive-qualification-m3-20261008"
        / "run_muse_progressive_m3.py"
    )
    spec = importlib.util.spec_from_file_location("muse_progressive_m3_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_runner_pins_the_bounded_b1_m3_profile():
    runner = _runner()
    command = runner.server_command("candidate")
    assert command[command.index("--max-context") + 1] == "4096"
    assert command[command.index("--cache-bytes") + 1] == "2147483648"
    assert command[command.index("--max-lanes") + 1] == "1"
    assert command[command.index("--max-inflight") + 1] == "1"
    assert runner.POLICY.name == "muse-dflash2-progressive-b1-m3.json"
    policy = json.loads(
        (
            Path(__file__).parents[1]
            / "qualification/policies/muse-dflash2-progressive-b1-m3.json"
        ).read_text()
    )
    assert policy["draft_model"].endswith("Muse-Glimmer-30B-DFlash2")


def test_qualified_restart_requires_run_local_tile_spanning_evidence():
    runner = _runner()
    counters = {
        "external_progressive_verify_rounds": 2,
        "external_progressive_verify_launches": 4,
        "external_progressive_verify_target_rows": 12,
        "external_progressive_verify_full_tiles": 2,
    }
    before = {
        "qualification": "qualified",
        "selected_capabilities": ["external_draft"],
        "qualified_capabilities": ["external_draft"],
        "route_receipt": "receipt",
        "scheduler": {key: 0 for key in counters},
    }
    after = {**before, "scheduler": counters}
    response = {
        "mlx2": {
            "qualification": "qualified",
            "route_receipt": "receipt",
            "speculation": {
                "progressive_verification": {
                    "selected": True,
                    "observed_used": True,
                    "verification_tile": 3,
                    "state_promotion": "request_private_then_atomic_lane_publish",
                    "rounds": 2,
                    "target_launches": 4,
                    "full_nonfinal_tiles": 2,
                }
            },
        },
    }
    failures, progressive, deltas = runner.validate_qualified(
        before, response, after
    )
    assert failures == []
    assert progressive["verification_tile"] == 3
    assert deltas == counters

    before["selected_capabilities"].append("continuous_batch")
    before["qualified_capabilities"].append("continuous_batch")
    failures, _, _ = runner.validate_qualified(before, response, after)
    assert "selected continuous_batch" in failures
    assert "qualified continuous_batch" in failures

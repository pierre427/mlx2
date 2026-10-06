"""CPU-only checks for the one-service tree15 transition gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/probe_qwen38_tree15_live_transition.py"
SPEC = importlib.util.spec_from_file_location("tree15_live_transition", SCRIPT)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def snapshot(tree, cohort, width, *, chain=0, fallbacks=0):
    return {"scheduler": {
        "external_auto_tree_rounds": tree,
        "external_tensorfold_target_rounds": tree,
        "external_tensorfold_cohort_rounds": cohort,
        "external_auto_tree_max_width": width,
        "external_tensorfold_cohort_max_width": width,
        "external_auto_chain_rounds": chain,
        "draft_fallbacks": fallbacks,
    }}


def test_stages_require_fresh_rounds_and_physical_width():
    b0, b1 = snapshot(0, 0, 0), snapshot(1, 1, 1)
    b2, b4 = snapshot(2, 2, 2), snapshot(3, 3, 4)
    gate.verify_stage(b0, b1, 1)
    gate.verify_stage(b1, b2, 2)
    gate.verify_stage(b2, b4, 4)
    gate.verify_stage(b4, snapshot(4, 4, 4), 1)
    with pytest.raises(ValueError, match="tree rounds"):
        gate.verify_stage(b2, b2, 4)
    wrong_physical = snapshot(3, 3, 4)
    wrong_physical["scheduler"]["external_tensorfold_cohort_max_width"] = 3
    with pytest.raises(ValueError, match="physical TensorFold"):
        gate.verify_stage(b2, wrong_physical, 4)


@pytest.mark.parametrize("change,reason", [
    ({"external_auto_chain_rounds": 1}, "chain/reference"),
    ({"external_auto_mode_switches": 1}, "chain/reference"),
    ({"draft_fallbacks": 1}, "fallback"),
    ({"recovery_checkpoint_restores": 1}, "fallback"),
])
def test_stage_fails_closed_on_reference_or_recovery(change, reason):
    before = snapshot(0, 0, 0)
    after = snapshot(1, 1, 1)
    after["scheduler"].update(change)
    with pytest.raises(RuntimeError, match=reason):
        gate.verify_stage(before, after, 1)

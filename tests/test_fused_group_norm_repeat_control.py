"""CPU-only checks for the fresh-cache repeat numerical control."""
from __future__ import annotations

import math

import numpy as np
import pytest

from scripts import eval_fused_group_norm_repeat_control as control


def test_phase_order_contains_fresh_repeats_and_both_orders():
    assert control.PHASES == (
        ("E1", False), ("E2", False), ("F1", True), ("F2", True),
        ("EF_E", False), ("EF_F", True), ("FE_F", True), ("FE_E", False),
    )
    assert ("E1", "E2") in control.PAIRS
    assert ("F1", "F2") in control.PAIRS
    assert ("EF_E", "EF_F") in control.PAIRS
    assert ("FE_F", "FE_E") in control.PAIRS


def test_full_matrix_comparison_detects_non_top1_change_and_ppl():
    a = np.log(np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], dtype=np.float32))
    b = a.copy()
    equal = control.compare_logprobs(a, b, [0, 1])
    assert equal["full_matrix_equal"]
    assert equal["compared_positions"] == 2
    assert equal["vocab_size"] == 3
    assert equal["kl_a_to_b_mean"] == 0
    assert equal["top1_agreement"] == 1
    assert math.isclose(equal["ppl_a"], equal["ppl_b"])

    b[0] = np.log(np.array([0.7, 0.15, 0.15], dtype=np.float32))
    changed = control.compare_logprobs(a, b, [0, 1])
    assert not changed["full_matrix_equal"]
    assert changed["top1_agreement"] == 1
    assert changed["kl_a_to_b_mean"] > 0
    assert changed["kl_b_to_a_mean"] > 0
    assert changed["ppl_relative_delta"] == pytest.approx(0, abs=1e-7)


def test_full_matrix_comparison_rejects_bad_alignment_or_nonfinite():
    a = np.array([[-0.1, -2.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="shape or target"):
        control.compare_logprobs(a, a, [0, 1])
    with pytest.raises(ValueError, match="outside vocabulary"):
        control.compare_logprobs(a, a, [2])
    a[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        control.compare_logprobs(a, a, [0])


def test_reference_validation_binds_source_and_both_heldout_parts():
    receipt = {
        "source_revision": "r", "model": "m", "model_config_sha256": "config",
        "source_sha256": {
            "scripts/eval_fused_group_norm_quality_v2.py": "harness",
            "src/mlx2/runtime/models/qwen4_fused_group_norm.py": "kernel"},
        "prompt_sha256": "prompt", "continuation_sha256": "continuation",
    }
    reference = {
        "schema": control.gate.SCHEMA,
        "source_revision": "r", "model": "m", "model_config_sha256": "config",
        "harness_sha256": "harness", "kernel_sha256": "kernel",
        "cases": [{"prompt_sha256": "prompt"}],
        "ppl_windows": [{"context_sha256": "context",
                         "continuation_sha256": "continuation"}],
    }
    control.validate_reference(reference, receipt, {"context_sha256": "context"})
    reference["ppl_windows"][0]["context_sha256"] = "different"
    with pytest.raises(ValueError, match="mismatch"):
        control.validate_reference(reference, receipt, {"context_sha256": "context"})

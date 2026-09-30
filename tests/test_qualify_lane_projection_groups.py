"""CPU contract tests for the declared-projection-group qualification harness."""

import mlx.core as mx
import pytest

from scripts import qualify_lane_projection_groups as qualify


def test_shipped_catalog_pins_every_declared_group_shape_and_format():
    by_family = {}
    for case in qualify.SHIPPED_CASES:
        by_family.setdefault(case.family, []).append(case)
        assert case.profile == "shipped"
        assert case.expected_k % 64 == 0
        assert sum(case.expected_widths) > 0
    assert {family: {case.weight_format for case in cases}
            for family, cases in by_family.items()} == {
        "muse": {"q4", "q8", "bf16"},
        "hils": {"q6", "bf16"},
        "xing": {"q6", "bf16"},
    }
    assert {case.family: (case.expected_k, case.expected_widths)
            for case in qualify.SHIPPED_CASES} == {
        "muse": (6656, (4096, 256, 256, 4096)),
        "hils": (4096, (4096, 4096, 4096, 256)),
        "xing": (3584, (768, 576)),
    }


def test_metal_guard_requires_both_external_lock_receipts(tmp_path):
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    with pytest.raises(RuntimeError, match="both GPU lock receipts"):
        qualify.require_gpu_lock_receipts((first, second))
    first.write_text("{}")
    with pytest.raises(RuntimeError, match=str(second)):
        qualify.require_gpu_lock_receipts((first, second))
    second.write_text("{}")
    qualify.require_gpu_lock_receipts((first, second))


def test_cpu_smoke_receipt_keeps_qualification_and_selection_false():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        receipt = qualify.run_suite(
            mode="cpu", cases=(qualify.TINY_CASES[0],), rows=(1, 2),
            timing_rows=2, warmups=0, repeats=1, batch=1,
        )
    finally:
        mx.set_default_device(previous)
    assert receipt["summary"]["cases"] == receipt["summary"]["passed"] == 1
    assert receipt["state"] == {
        "implemented": True,
        "tested_cases_passed": True,
        "tested_case_set_complete": False,
        "cpu_smoke_passed": False,
        "metal_qualification_candidate_passed": False,
        "qualified": False,
        "selected": False,
        "observed_used": False,
        "note": "A passing receipt does not modify route qualification or selection.",
    }
    case = receipt["cases"][0]
    assert case["parity"]["passed"] and case["row_invariance"]["passed"]
    assert case["parity"]["declared_vs_default_grouped"]["bitwise_equal"]
    assert case["parity"]["accuracy_vs_fp32"]["passed"]
    assert case["engagement"]["observed"] == {
        "launches": 1, "reuses": 3, "partial": 0,
    }
    assert case["timing"]["performance_evidence"] is False
    assert "not performance evidence" in case["timing"]["interpretation"]
    assert case["memory"]["scope"].startswith("declared regrouping")
    assert "+declared[muse-attn-qkv-gate@" in case["declared_law_id"]
    assert "declared" not in case["default_law_id"]

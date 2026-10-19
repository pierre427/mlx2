"""Host-only checks for sanitized AST geometry evidence aggregation."""

import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/summarize_ast_audit.py"
SPEC = importlib.util.spec_from_file_location("ast_audit_summary", PATH)
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def synthetic_unit(path, corruption=None):
    tile = {
        "kind": "target_tile", "shape": [2, 3],
        "mapping": [{"uid": 7, "input_rows": 3}, {"uid": 8, "input_rows": 1}],
        "layout": {"logical_rows": 4, "physical_rows": 6, "padding_rows": 2,
                   "packed_min_query_len": 2, "backend_mode": "padded",
                   "request_lane_uids": [7, 8]},
        "executor": {"row_exact": True, "adapter_detail": {}}, "windows": [],
    }
    if corruption == "foreign_session":
        tile["mapping"][1]["uid"] = 99
    elif corruption == "padding":
        tile["layout"]["padding_rows"] = 0
    elif corruption == "minimum_width":
        tile["layout"]["packed_min_query_len"] = 4
    elif corruption == "executor":
        tile["executor"]["row_exact"] = False
    elif corruption == "planner":
        tile["layout"]["ast_plan"] = {"input_lengths": [3, 2]}
    row = {"completed": True, "sessions": [{"uid": 7}, {"uid": 8}],
           "events": [tile], "target_execution": "reference", "tile_policy": None,
           "configured_tile": 3}
    meta = {"status": "complete", "checkpoint": "target@sha", "config": "synthetic",
            "source_revision": "source", "shape": {}, "policy": {"draft_model": "draft"},
            "inventory": {"auxiliaries": {"draft": {"checkpoint": "draft@sha"}}}}
    (path / "meta.json").write_text(json.dumps(meta))
    (path / "rounds.jsonl").write_text(json.dumps(row) + "\n")
    (path / "verdicts.json").write_text(json.dumps({"b1_equal_ordinary": {"test": [True]}}))


def test_geometry_summary_keeps_shape_and_drafter_not_session_identifiers(tmp_path):
    synthetic_unit(tmp_path)
    result = SUMMARY.geometry(tmp_path)
    assert result["drafter_checkpoint"] == "draft@sha"
    assert result["logical_valid_token_rows"] == 4
    assert result["physical_slab_token_positions"] == 6
    assert result["padding_token_positions"] == 2
    assert result["mixed_session_tiles"] == 1
    assert result["multi_lane_wider_launches"] == 1
    assert result["packed_minimum_attested_tiles"] == 1
    assert result["performance_evidence"] is False
    assert "lane_uids" not in json.dumps(result)
    assert "mapping" not in result["ragged_tile_example"]


@pytest.mark.parametrize("corruption", ["foreign_session", "padding", "minimum_width", "executor", "planner"])
def test_geometry_summary_refuses_inconsistent_evidence(tmp_path, corruption):
    synthetic_unit(tmp_path, corruption)
    with pytest.raises(AssertionError):
        SUMMARY.geometry(tmp_path)


def performance_unit(path, *, selected=True, corruption=None):
    policy = {"draft_model": "draft", "continuation_pool": {"ast_planner": selected}}
    meta = {"status": "completed", "checkpoint": "target@sha", "config": "synthetic",
            "hardware": "fixture", "source_heads": {"fixed": "a" * 40, "before": "b" * 40},
            "shape": {}, "policy": policy, "arms": {"fixed": {"policy": policy}},
            "inventory": {"auxiliaries": {"draft": {"checkpoint": "draft@sha"}}}}
    cell = {"fixed_vs_before_pct": 0, "fixed_vs_ordinary_ratio": 1}
    for arm in ("ordinary", "before", "fixed"):
        cell[arm] = {"decode_tps_median": 11, "wall_tps_median": 9,
                     "decode_tps": [10, 12], "exact_token_parity": [True, True],
                     "swapouts_delta": [0, 0], "thermal_start_c": [60, 61], "thermal_end_c": [62, 63]}
        spec = {"ast_execution_plan": {"schema": "mlx2.ast-execution-plan.v1", "cost_samples": 3,
                                      "backend": "padded_state_compact_exact_stages"},
                "external_rounds": 2, "ordinary_fallback": False,
                "progressive_verification": {"observed_used": True}}
        if arm != "fixed" or not selected:
            spec = {}
        if arm == "fixed" and corruption:
            if corruption == "missing_plan":
                spec.pop("ast_execution_plan")
            elif corruption == "no_samples":
                spec["ast_execution_plan"]["cost_samples"] = 0
            elif corruption == "no_rounds":
                spec["external_rounds"] = 0
            elif corruption == "fallback":
                spec["ordinary_fallback"] = True
            elif corruption == "not_executed":
                spec["progressive_verification"]["observed_used"] = False
        row = {"done": True, "error": None, "route": "external_draft", "speculation": spec,
               "wall_tokens_per_second": 9}
        if arm == "fixed" and corruption == "wrong_route":
            row["route"] = "ordinary"
        report = {"phases": {"b1_warmup": {"done": False}, "b1_r1_prompt": row, "b1_r2_prompt": row}}
        (path / f"{arm}.json").write_text(json.dumps(report))
    (path / "meta.json").write_text(json.dumps(meta))
    (path / "summary.json").write_text(json.dumps({"prompt": cell}))


def test_performance_retains_samples_and_requires_observed_planner(tmp_path):
    performance_unit(tmp_path)
    result = SUMMARY.performance(tmp_path)
    arm = result["arms"]["fixed"]
    assert arm["decode_tps_samples"] == {"prompt": [10, 12]}
    assert arm["wall_tps_samples"] == {"prompt": [9, 9]}
    assert arm["planner"]["observed_requests"] == arm["planner"]["measured_requests"] == 2
    assert arm["planner"]["minimum_cost_samples"] == 3
    assert len(arm["raw_report_sha256"]) == 64
    assert result["fixed_vs_ordinary_ratio"] == {"prompt": 1}


def test_performance_does_not_promote_legacy_policy(tmp_path):
    performance_unit(tmp_path, selected=False)
    arm = SUMMARY.performance(tmp_path)["arms"]["fixed"]
    assert not arm["planner"]["selected"]
    assert arm["planner"]["observed_requests"] == 0


@pytest.mark.parametrize("corruption", ["missing_plan", "no_samples", "no_rounds", "fallback", "not_executed", "wrong_route"])
def test_performance_refuses_selected_but_unobserved_planner(tmp_path, corruption):
    performance_unit(tmp_path, corruption=corruption)
    with pytest.raises(AssertionError, match="lacks actual execution evidence"):
        SUMMARY.performance(tmp_path)

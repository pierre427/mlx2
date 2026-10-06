import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "assess_apcv2_replay_equivalence.py"
SPEC = importlib.util.spec_from_file_location("assess_apcv2_replay_equivalence", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


SHA = "a" * 64


def _row(*, width=4, cold_margin=0.125, warm_margin=0.0, functional=True):
    return {
        "prompt_sha256": "b" * 64,
        "physical_width": width,
        "cold_token_ids": [5, 7, 11],
        "warm_token_ids": [5, 7, 13],
        "first_divergence": {
            "index": 2,
            "shared_prefix_exact": True,
            "cold": {
                "selected_token_id": 11,
                "top_two": [
                    {"token_id": 11, "logprob": -0.7},
                    {"token_id": 13, "logprob": -0.7 - cold_margin},
                ],
            },
            "warm": {
                "selected_token_id": 13,
                "top_two": [
                    {"token_id": 13, "logprob": -0.8},
                    {"token_id": 11, "logprob": -0.8 - warm_margin},
                ],
            },
        },
        "functional_oracles": {"cold": functional, "warm": functional},
    }


def _evidence(rows=None, *, control_hits=1, control_n=8):
    return {
        "schema": MODULE.INPUT_SCHEMA,
        "identity": {
            "model": "model",
            "artifact_sha256": SHA,
            "runtime_source_sha256": SHA,
            "profile": "ordinary",
            "cache_layout": "layout-v1",
            "serving_shape_sha256": SHA,
        },
        "state_checks": {name: True for name in MODULE.REQUIRED_STATE_CHECKS},
        "rows": rows or [_row()],
        "ordinary_control": {
            "serving_shape_sha256": SHA,
            "mismatches": control_hits,
            "comparisons": control_n,
        },
    }


def test_classifies_proven_multi_lane_near_tie():
    result = MODULE.verdict(_evidence())
    assert result["passed"] is True
    assert result["qualification_claim"] is False
    assert result["rows"][0]["classification"] == "near_tie_equivalent"
    assert result["rows"][0]["margins_nats"] == {"cold": 0.125, "warm": 0.0}


def test_width_one_remains_exact():
    result = MODULE.verdict(_evidence([_row(width=1)]))
    assert result["passed"] is False
    assert any("width-one" in failure for failure in result["failures"])


def test_exact_multi_lane_row_needs_no_control():
    row = _row()
    row["warm_token_ids"] = list(row["cold_token_ids"])
    row.pop("first_divergence")
    result = MODULE.verdict(_evidence([row], control_n=0))
    assert result["passed"] is True
    assert result["rows"][0]["classification"] == "exact_token_parity"
    assert result["differential"]["control_required_for_equivalence"] is False


def test_rejects_high_margin_flip():
    result = MODULE.verdict(_evidence([_row(cold_margin=0.625)]))
    assert result["passed"] is False
    assert any("exceeds 0.5" in failure for failure in result["failures"])


def test_rejects_different_top_two_set():
    row = _row()
    row["first_divergence"]["warm"]["top_two"][1]["token_id"] = 17
    result = MODULE.verdict(_evidence([row]))
    assert result["passed"] is False
    assert any("top-two token sets differ" in failure for failure in result["failures"])


def test_rejects_failed_functional_oracle():
    result = MODULE.verdict(_evidence([_row(functional=False)]))
    assert result["passed"] is False
    assert any("functional oracle" in failure for failure in result["failures"])


def test_control_is_optional_diagnostic_evidence():
    evidence = _evidence()
    del evidence["ordinary_control"]
    missing = MODULE.verdict(evidence)
    assert missing["passed"] is True
    assert missing["differential"]["control_present"] is False
    evidence = _evidence()
    evidence["ordinary_control"]["serving_shape_sha256"] = "c" * 64
    mismatched = MODULE.verdict(evidence)
    assert mismatched["passed"] is True
    assert mismatched["differential"]["control_present"] is True
    assert mismatched["differential"]["comparable"] is False


def test_reports_statistically_material_excess_without_failing_equivalence():
    rows = [_row() for _ in range(8)]
    result = MODULE.verdict(_evidence(rows, control_hits=0, control_n=8))
    assert result["passed"] is True
    assert result["differential"]["p_value"] < 0.05
    assert result["differential"]["material_excess_observed"] is True


def test_state_checks_fail_closed():
    evidence = _evidence()
    evidence["state_checks"]["warm_hit"] = False
    result = MODULE.verdict(evidence)
    assert result["passed"] is False
    assert any("warm_hit" in failure for failure in result["failures"])

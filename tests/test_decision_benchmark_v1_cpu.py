from __future__ import annotations

import copy

import pytest

from benchmarks.decisions.v1.compare import recommend
from benchmarks.decisions.v1.run import (
    DEFAULT_CASES,
    brier,
    load_cases,
    metric_value,
    percentile,
    run_quality,
    shuffled_question,
    summarize_quality,
    swap_used_mib,
)
from mlx2.decisions.schema import normalize_request


def test_frozen_corpus_has_exact_shape_and_valid_rotations():
    cases = load_cases(DEFAULT_CASES)
    assert len(cases) == 100
    assert {case["category"] for case in cases} == {
        "tool_selection",
        "agent_next_action",
        "support_operations",
        "policy_safety",
    }
    for case in cases:
        shuffled = shuffled_question(case)
        normalize_request(
            {
                "model": "test",
                "state": case["state"],
                "questions": {"decision": case["question"]},
                "truncate": False,
            },
            default_model="test",
        )
        if case["question"]["type"] == "choice":
            assert list(shuffled["criteria"]) != list(case["question"]["criteria"])
            assert shuffled["criteria"] == case["question"]["criteria"]


def test_brier_and_percentile_are_small_and_explicit():
    assert brier(
        {
            "type": "choice",
            "probabilities": {"yes": 0.75, "no": 0.25},
        },
        "yes",
    ) == pytest.approx(0.125)
    assert brier({"type": "noul", "probability": 0.8}, True) == pytest.approx(0.04)
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([1, 2, 3, 4], 0.95) == pytest.approx(3.85)
    assert metric_value("metric{family=\"x\"} 42\n", "metric") == 42
    assert swap_used_mib({"swap": "vm.swapusage: used = 3054.00M free"}) == 3054


def test_quality_runner_scores_fake_decision_service():
    cases = load_cases(DEFAULT_CASES)[:2]
    expected = {case["id"]: case["expected"] for case in cases}

    def fake(payload):
        state = payload["state"]
        case_id = next(case["id"] for case in cases if case["state"] == state)
        value = expected[case_id]
        criteria = payload["questions"]["decision"]["criteria"]
        probabilities = {
            label: 0.9 if label == value else 0.1 / (len(criteria) - 1)
            for label in criteria
        }
        return (
            {
                "answers": {
                    "decision": {
                        "type": "choice",
                        "value": value,
                        "confidence": 0.9,
                        "probabilities": probabilities,
                    }
                },
                "mlx2": {"observed_used": True},
                "usage": {"input_tokens": 20, "output_tokens": 0},
            },
            0.01,
        )

    rows = run_quality(cases, "fake", fake)
    summary = summarize_quality(rows)
    assert len(rows) == 4
    assert summary["accuracy"] == 1.0
    assert summary["order_stability"] == 1.0
    assert summary["coverage_at_threshold"] == 1.0


def _summary(model, *, accuracy, brier_score, stable=1.0, qualified=True, p95=10):
    return {
        "model": model,
        "states": {"qualified": qualified},
        "quality": {
            "accuracy": accuracy,
            "mean_brier": brier_score,
            "order_stability": stable,
        },
        "performance": {"short": {"p95_latency_ms": p95}},
        "peak_resident_memory_bytes": 1 << 30,
    }


def test_recommendation_uses_accuracy_band_then_calibration():
    accurate = _summary("accurate", accuracy=0.91, brier_score=0.20)
    calibrated = _summary("calibrated", accuracy=0.90, brier_score=0.08)
    unstable = _summary("unstable", accuracy=0.99, brier_score=0.01, stable=0.97)
    unqualified = _summary(
        "unqualified", accuracy=1.0, brier_score=0.0, qualified=False
    )
    result = recommend([accurate, calibrated, unstable, unqualified])
    assert result["selected"] == "calibrated"
    assert result["eligible"] == ["calibrated", "accurate"]


def test_recommendation_does_not_mutate_input():
    summaries = [_summary("one", accuracy=0.8, brier_score=0.2)]
    before = copy.deepcopy(summaries)
    recommend(summaries)
    assert summaries == before

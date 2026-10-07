import copy
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ligo_flow_policy_replay",
    ROOT / "scripts/research/ligo_flow_policy_replay.py",
)
REPLAY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPLAY
SPEC.loader.exec_module(REPLAY)


@pytest.fixture(scope="module")
def receipt():
    return REPLAY.build_receipt()


def test_trace_covers_declared_stressors_without_exposing_outcomes(receipt):
    features = receipt["trace"]["features"]
    assert features == {
        "heavy_tail": True,
        "cancellations": 1,
        "warm_requests": 4,
        "atomic_cohorts": 1,
        "cost_epochs": ["interference", "normal"],
    }
    view = REPLAY.policy_request_view(REPLAY.default_trace()[0])
    assert set(receipt["policy_contract"]["future_outcomes_hidden"]).isdisjoint(view)
    assert "max_output_tokens" in view


def test_every_arm_preserves_hard_authorities_and_conservation(receipt):
    assert receipt["status"] == "passed"
    for arm in receipt["arms"].values():
        assert arm["conservation"] == {
            "passed": True,
            "residuals": {
                "eligible_minus_terminal": 0,
                "created_minus_released_minus_live": 0,
                "committed_minus_emitted_minus_buffered_minus_discarded": 0,
            },
            "violations": [],
        }
        metrics = arm["metrics"]
        assert metrics["hard_memory_violations"] == 0
        assert metrics["peak_projected_memory_units"] <= receipt["config"][
            "memory_budget_units"
        ]
        assert metrics["refused_requests"] == 0
        assert metrics["completed_requests"] + metrics["cancelled_requests"] == 14


def test_atomic_cohort_is_admitted_at_one_boundary(receipt):
    for arm in receipt["arms"].values():
        admitted = {
            event["logical_request_id"]: event["time_ms"]
            for event in arm["events"]
            if event["kind"] == "admitted"
        }
        assert admitted[8] == admitted[9]


def test_every_pack_decision_is_legal_and_explicitly_proxy_labeled(receipt):
    for arm in receipt["arms"].values():
        for decision in arm["decisions"]:
            assert decision["reason"] in {"mixed_pack", "decode_only"}
            assert decision["price_profile"] == "offline-proxy:not-measured:not-qualified:v1"
            assert decision["proxy"] is True
            assert decision["charged_rows"] <= receipt["config"]["row_capacity"]
            if decision["mandatory_decode_uids"]:
                assert decision["estimated_ms"] <= receipt["config"][
                    "decode_deadline_ms"
                ]


def test_residual_bound_is_helpful_but_wait_pack_fails_harm_gate(receipt):
    residual = receipt["comparisons"]["residual_bound"]
    wait_pack = receipt["comparisons"]["wait_pack"]
    assert residual["deadline_miss_relative_reduction"] >= 0.20
    assert residual["helpful_on_this_proxy_trace"] is True
    assert residual["kill_reasons"] == []
    assert wait_pack["deadline_miss_relative_reduction"] >= 0.20
    assert wait_pack["helpful_on_this_proxy_trace"] is False
    assert "throughput_regression_gt_5pct" in wait_pack["kill_reasons"]


def test_wait_like_policy_exposes_fairness_harm_without_hidden_refusals(receipt):
    baseline = receipt["arms"]["baseline"]["metrics"]
    wait_pack = receipt["arms"]["wait_pack"]["metrics"]
    assert wait_pack["max_queue_wait_ms"] > baseline["max_queue_wait_ms"]
    assert wait_pack["refused_requests"] == baseline["refused_requests"]
    assert wait_pack["completed_requests"] == baseline["completed_requests"]
    comparison = receipt["comparisons"]["wait_pack"]
    assert comparison["hard_gates"]["queue_wait_bounded"] is False
    assert "queue_wait_bounded" in comparison["kill_reasons"]


def test_conservation_auditor_detects_lost_emission(receipt):
    events = copy.deepcopy(receipt["arms"]["baseline"]["events"])
    index = next(i for i, event in enumerate(events) if event["kind"] == "tokens_emitted")
    events.pop(index)
    for sequence, event in enumerate(events):
        event["sequence"] = sequence
    audit = REPLAY.audit_events(events)
    assert audit["passed"] is False
    assert audit["residuals"][
        "committed_minus_emitted_minus_buffered_minus_discarded"
    ] > 0


def test_incomplete_atomic_cohort_is_refused_before_replay():
    member = REPLAY.TraceRequest(
        90,
        0,
        64,
        8,
        4,
        cohort_id="incomplete",
        cohort_size=2,
    )
    with pytest.raises(ValueError, match="incomplete"):
        REPLAY.validate_trace((member,))


def test_impossible_request_is_refused_by_hard_memory_authority():
    trace = (REPLAY.TraceRequest(1, 0, 4096, 64, 4),)
    config = REPLAY.ReplayConfig(memory_budget_units=512)
    receipt = REPLAY.build_receipt(trace, config, logical_run_id="oversized")
    for arm in receipt["arms"].values():
        assert arm["metrics"]["refused_requests"] == 1
        refusal = next(
            event for event in arm["events"] if event["kind"] == "terminal_refused"
        )
        assert refusal["reason"] == "hard_memory_authority"


def test_cli_receipt_uses_exclusive_creation(tmp_path, monkeypatch, capsys):
    output = tmp_path / "attempt.json"
    argv = ["ligo_flow_policy_replay.py", "--output", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    REPLAY.main()
    capsys.readouterr()
    assert output.exists()
    with pytest.raises(FileExistsError):
        REPLAY.main()

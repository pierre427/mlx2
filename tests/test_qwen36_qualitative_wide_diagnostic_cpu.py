"""CPU/static contracts for the one-off Qwen3.6 qualitative diagnostic."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "qualification"
    / "runs"
    / "speculative-composition-qualification-20261012"
    / "diagnose_qwen36_qualitative_wide.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("qwen36_qualitative_wide", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_is_diagnostic_only_and_describes_counterfactual_state_publish():
    module = _module()
    plan = module._plan(
        SimpleNamespace(execute=False, max_new_tokens=64, output_dir=None)
    )
    assert plan["schema"] == "mlx2.qwen36-wide-state-qualitative-diagnostic.v1"
    assert plan["diagnostic_only"] is True
    assert plan["qualification_claim"] is False
    assert plan["performance_claim"] is False
    assert plan["perplexity_measured"] is False
    assert plan["preflight_run"] is False
    assert plan["feature_arm"] == {
        "name": "diagnostic_progressive_external_draft_near_tie_bounded",
        "verification_tile": 2,
        "num_draft": 3,
        "proposal_source": "pinned Qwen3.6 DFlash2 artifact",
        "accept_reject": "runtime verify_proposal_tile",
        "state_promotion": "request-private then accepted-prefix commit",
        "production_route": False,
        "gate_change": False,
        "purpose": (
            "measure narrow answer correctness under repeated Qwen3.6 "
            "progressive wide-row verification"
        ),
    }
    assert {task["id"] for task in plan["tasks"]} == {
        "arithmetic",
        "structured_extraction",
        "transitive_logic",
        "set_reasoning",
    }


def test_marker_evaluator_is_case_insensitive_but_requires_a_complete_line():
    module = _module()
    check = {"kind": "marker", "marker": "ANSWER", "value": "91"}
    assert module.evaluate_output("answer=91\n", check)["passed"]
    assert module.evaluate_output("Reasoning\nANSWER = 91.\n", check)["passed"]
    assert not module.evaluate_output("The answer is 91", check)["passed"]
    assert not module.evaluate_output("ANSWER=19", check)["passed"]


def test_json_evaluator_requires_exact_keys_and_typed_values():
    module = _module()
    expected = {
        "id": "K7",
        "quantity": 12,
        "item": "blue valves",
        "city": "Oslo",
        "day": "Tuesday",
    }
    check = {"kind": "json_object", "value": expected}
    assert module.evaluate_output("```json\n" + __import__("json").dumps(expected) + "\n```", check)[
        "passed"
    ]
    wrong_type = {**expected, "quantity": "12"}
    assert not module.evaluate_output(__import__("json").dumps(wrong_type), check)[
        "passed"
    ]
    extra_key = {**expected, "country": "Norway"}
    assert not module.evaluate_output(__import__("json").dumps(extra_key), check)[
        "passed"
    ]


def test_comparison_distinguishes_feature_regression_from_shared_task_failure():
    module = _module()
    check = {"kind": "marker", "marker": "YOUNGEST", "value": "Chen"}
    regression = module.compare_task_outputs(
        "YOUNGEST=Chen", "YOUNGEST=Bob", check
    )
    assert regression["feature_regressed_correct_reference"]
    assert not regression["feature_preserved_correctness"]

    shared_failure = module.compare_task_outputs(
        "I cannot tell", "I also cannot tell", check
    )
    assert shared_failure["both_incorrect"]
    assert not shared_failure["feature_regressed_correct_reference"]


def test_summary_counts_each_outcome_without_promoting_qualification():
    module = _module()
    rows = [
        {
            "correctness": {
                "reference": {"passed": True},
                "feature": {"passed": True},
                "feature_preserved_correctness": True,
                "feature_regressed_correct_reference": False,
                "both_incorrect": False,
            }
        },
        {
            "correctness": {
                "reference": {"passed": True},
                "feature": {"passed": False},
                "feature_preserved_correctness": False,
                "feature_regressed_correct_reference": True,
                "both_incorrect": False,
            }
        },
        {
            "correctness": {
                "reference": {"passed": False},
                "feature": {"passed": False},
                "feature_preserved_correctness": False,
                "feature_regressed_correct_reference": False,
                "both_incorrect": True,
            }
        },
    ]
    summary = module.summarize_results(rows)
    assert summary == {
        "tasks": 3,
        "reference_passes": 2,
        "feature_passes": 1,
        "feature_preserved_correctness": 1,
        "feature_regressed_correct_reference": 1,
        "both_incorrect": 1,
        "all_feature_answers_correct": False,
        "qualitative_only": True,
        "qualification_claim": False,
    }


def test_cycle_trace_records_real_accept_reject_commit_and_divergence():
    module = _module()
    decision = SimpleNamespace(
        accepted=1,
        emitted=[10, 99],
        commit_count=2,
        committed_inputs=[7, 10],
    )
    stages = [
        {
            "input_tokens": [7, 10],
            "input_shape": [1, 2],
            "target_argmax": [10, 99],
        }
    ]
    row = module.cycle_trace(
        [10, 11, 12],
        stages,
        decision,
        {"rejected": 1, "launches": 1, "target_rows": 2, "full_tiles": 0},
    )
    assert row["proposed_tokens"] == [10, 11, 12]
    assert row["target_wide_argmax"] == [10, 99]
    assert row["accepted_length"] == 1
    assert row["committed_rows"] == 2
    assert row["committed_inputs"] == [7, 10]
    assert row["first_divergence"] == {
        "proposal_index": 1,
        "proposed_token": 11,
        "target_correction_token": 99,
    }
    assert row["rejected"] is True


def test_cycle_trace_has_no_divergence_when_all_proposals_are_accepted():
    module = _module()
    decision = SimpleNamespace(
        accepted=3,
        emitted=[10, 11, 12, 13],
        commit_count=4,
        committed_inputs=[7, 10, 11, 12],
    )
    row = module.cycle_trace(
        [10, 11, 12],
        [{"input_tokens": [7, 10], "target_argmax": [10, 11]}],
        decision,
        {"rejected": 0, "launches": 2, "target_rows": 4, "full_tiles": 1},
    )
    assert row["accepted_length"] == 3
    assert row["committed_rows"] == 4
    assert row["first_divergence"] is None
    assert row["rejected"] is False


def test_trace_wrapper_exercises_real_progressive_accept_reject_runtime():
    module = _module()
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
    from mlx2.runtime.progressive_external_verify import (
        PROGRESSIVE_APPROXIMATE_TARGET_PROTOCOL_VERSION,
        PROGRESSIVE_FIDELITY_NEAR_TIE_BOUNDED,
        progressive_target_protocol_binding,
    )
    from mlx2.runtime.sample_utils import LaneRNG
    from test_qwen38_dflash2_cpu import tiny_pair

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    target, draft = tiny_pair()
    binding = progressive_target_protocol_binding(
        target, PROGRESSIVE_FIDELITY_NEAR_TIE_BOUNDED
    )
    traced = module.traced_progressive_batch_class(ExternalDraftBatchGenerator)
    batch = traced(
        target,
        draft_model=draft,
        binding="qualitative-trace-cpu",
        num_draft=3,
        completion_batch_size=1,
        prefill_step_size=8,
        progressive_verification_tile=2,
        progressive_verification_fidelity=PROGRESSIVE_FIDELITY_NEAR_TIE_BOUNDED,
        progressive_target_protocol=PROGRESSIVE_APPROXIMATE_TARGET_PROTOCOL_VERSION,
        progressive_protocol_binding=binding,
    )
    try:
        uid = batch.insert(
            [[1, 2, 3, 4]],
            max_tokens=[8],
            lane_rngs=[LaneRNG(0)],
            sampling_configs=[{"sampling_temp": 0.0}],
        )[0]
        finished = False
        for _ in range(32):
            _active, responses = batch.next()
            if any(row.uid == uid and row.finish_reason for row in responses):
                finished = True
                break
        assert finished
        assert batch.qualitative_cycle_traces
        for row in batch.qualitative_cycle_traces:
            assert row["proposed_tokens"]
            assert row["target_stages"]
            assert row["target_wide_argmax"]
            assert 0 <= row["accepted_length"] <= len(row["proposed_tokens"])
            assert 1 <= row["committed_rows"] <= len(row["proposed_tokens"]) + 1
            assert row["launches"] == len(row["target_stages"])
    finally:
        batch.close()
        mx.set_default_device(previous)

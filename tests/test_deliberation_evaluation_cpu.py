"""Parent-publication and matched-resource evaluation contracts (CPU only)."""

import copy

import pytest

from mlx2.runtime.deliberation_evaluation import (
    assess_parent_synthesis,
    evaluate_matched_arms,
)

RESOURCE = {
    "prompt_tokens": 128,
    "parent_completion_tokens": 16,
    "child_completion_tokens": 32,
    "forward_calls": 6,
    "physical_width": 2,
}


def arm(name, treatment, output, *, publishable=True, resource=None):
    return {
        "arm_id": name,
        "task_id": "gate-code",
        "replicate": 0,
        "treatment": treatment,
        "expected": "beta",
        "output": output,
        "parent_publishable": publishable,
        "resource": RESOURCE if resource is None else resource,
    }


def test_parent_synthesis_allows_one_bounded_correction_without_content_leak():
    bad = assess_parent_synthesis(
        attempt=0,
        content='<tool_call>{"name":"deliberation_finish"}</tool_call>',
        remaining_output_tokens=8,
    )
    failed = assess_parent_synthesis(
        attempt=1,
        content="",
        remaining_output_tokens=8,
    )
    accepted = assess_parent_synthesis(
        attempt=1,
        content="FINAL: beta",
        remaining_output_tokens=0,
    )

    assert (bad.trigger, bad.disposition) == (
        "tool_markup_content", "correct_once",
    )
    assert (failed.trigger, failed.disposition) == ("empty_output", "fail_closed")
    assert (accepted.trigger, accepted.disposition) == (None, "accept")
    assert "beta" not in repr(accepted.receipt())


def test_parent_synthesis_rejects_calls_reasoning_markup_and_unbounded_inputs():
    call = assess_parent_synthesis(
        attempt=0, content="beta", tool_calls=[{"name": "other"}],
        remaining_output_tokens=0,
    )
    reasoning = assess_parent_synthesis(
        attempt=0, content="beta", reasoning="<function=semantic_deliberate>",
        remaining_output_tokens=1,
    )
    assert (call.trigger, call.disposition) == (
        "prohibited_tool_call", "fail_closed",
    )
    assert reasoning.trigger == "tool_markup_reasoning"
    with pytest.raises(ValueError):
        assess_parent_synthesis(
            attempt=2, content="beta", remaining_output_tokens=1,
        )
    with pytest.raises(ValueError):
        assess_parent_synthesis(
            attempt=0, content="beta", tool_calls=[{}] * 9,
            remaining_output_tokens=1,
        )


def test_equal_observed_resources_score_conditioning_without_capability_claim():
    document = {
        "schema": "mlx2.semantic-matched-arms.v1",
        "arms": [
            arm("ordinary", "ordinary", "alpha"),
            arm("strategy", "strategy", "FINAL: <answer>beta</answer>"),
            arm("placebo", "placebo", "alpha"),
        ],
    }
    report = evaluate_matched_arms(document)

    assert report["all_blocks_resource_matched"] is True
    assert report["matched_quality_contrast_reportable"] is True
    assert report["general_quality_gain_claim_allowed"] is False
    outcomes = report["comparisons"][0]["outcomes"]
    assert outcomes["strategy"]["intention_to_treat_correct"] is True
    assert outcomes["ordinary"]["intention_to_treat_correct"] is False
    assert "FLOP" in report["resource_claim"]
    assert all("output" not in row for row in report["arms"])


def test_resource_mismatch_and_invalid_parent_fail_closed_in_scoring():
    changed = copy.deepcopy(RESOURCE)
    changed["child_completion_tokens"] += 1
    document = {
        "schema": "mlx2.semantic-matched-arms.v1",
        "arms": [
            arm("ordinary", "ordinary", "beta"),
            arm("dialogue", "dialogue", "beta", resource=changed),
            arm("bad-parent", "corrected", "beta", publishable=False),
        ],
    }
    report = evaluate_matched_arms(document)

    assert report["all_blocks_resource_matched"] is False
    assert report["matched_quality_contrast_reportable"] is False
    bad = report["comparisons"][0]["outcomes"]["corrected"]
    assert bad == {
        "valid": False,
        "conditional_correct": None,
        "intention_to_treat_correct": False,
    }


def test_matched_input_rejects_duplicate_ids_treatments_and_extra_fields():
    base = arm("ordinary", "ordinary", "beta")
    with pytest.raises(ValueError, match="identifiers"):
        evaluate_matched_arms(
            {"schema": "mlx2.semantic-matched-arms.v1", "arms": [base, base]}
        )
    duplicate_treatment = arm("second", "ordinary", "beta")
    with pytest.raises(ValueError, match="treatment"):
        evaluate_matched_arms(
            {
                "schema": "mlx2.semantic-matched-arms.v1",
                "arms": [base, duplicate_treatment],
            }
        )
    malformed = copy.deepcopy(base)
    malformed["raw_prompt"] = "secret"
    with pytest.raises(ValueError, match="fields"):
        evaluate_matched_arms(
            {
                "schema": "mlx2.semantic-matched-arms.v1",
                "arms": [malformed, arm("second", "strategy", "beta")],
            }
        )


def test_matched_blocks_require_one_answer_key_and_one_treatment_panel():
    foreign_answer = arm("strategy", "strategy", "beta")
    foreign_answer["expected"] = "gamma"
    with pytest.raises(ValueError, match="one exact expected answer"):
        evaluate_matched_arms(
            {
                "schema": "mlx2.semantic-matched-arms.v1",
                "arms": [arm("ordinary", "ordinary", "beta"), foreign_answer],
            }
        )

    second_ordinary = arm("ordinary-2", "ordinary", "beta")
    second_ordinary["task_id"] = "second"
    second_ordinary["replicate"] = 1
    second_strategy = arm("strategy-2", "different", "beta")
    second_strategy["task_id"] = "second"
    second_strategy["replicate"] = 1
    report = evaluate_matched_arms(
        {
            "schema": "mlx2.semantic-matched-arms.v1",
            "arms": [
                arm("ordinary", "ordinary", "beta"),
                arm("strategy", "strategy", "beta"),
                second_ordinary,
                second_strategy,
            ],
        }
    )
    assert report["all_blocks_treatment_matched"] is False
    assert report["matched_quality_contrast_reportable"] is False


def test_evaluation_and_synthesis_reject_oversized_text():
    with pytest.raises(ValueError, match="byte limit"):
        assess_parent_synthesis(
            attempt=0,
            content="x" * ((1 << 20) + 1),
            remaining_output_tokens=1,
        )
    oversized = arm("ordinary", "ordinary", "beta")
    oversized["output"] = "x" * ((64 << 10) + 1)
    with pytest.raises(ValueError, match="byte limit"):
        evaluate_matched_arms(
            {
                "schema": "mlx2.semantic-matched-arms.v1",
                "arms": [oversized, arm("strategy", "strategy", "beta")],
            }
        )


def test_empty_normalized_answers_are_not_scoreable():
    empty_output = arm("ordinary", "ordinary", "<answer></answer>")
    report = evaluate_matched_arms(
        {
            "schema": "mlx2.semantic-matched-arms.v1",
            "arms": [empty_output, arm("strategy", "strategy", "beta")],
        }
    )
    assert report["comparisons"][0]["outcomes"]["ordinary"] == {
        "valid": False,
        "conditional_correct": None,
        "intention_to_treat_correct": False,
    }
    bad_key = arm("bad-key", "ordinary", "beta")
    bad_key["expected"] = "<answer></answer>"
    with pytest.raises(ValueError, match="expected answer must be nonempty"):
        evaluate_matched_arms(
            {
                "schema": "mlx2.semantic-matched-arms.v1",
                "arms": [bad_key, arm("strategy", "strategy", "beta")],
            }
        )

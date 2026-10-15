"""Fail-closed scoring for bounded, matched-resource deliberation studies.

This module does not run a model or implement a deliberation route.  It keeps
the two easy-to-confuse questions separate: whether a parent continuation is
publishable, and whether completed experimental arms consumed the same
observable work before their answer quality is compared.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

PARENT_SYNTHESIS_SCHEMA = "mlx2.semantic-parent-synthesis-assessment.v1"
MATCHED_EVALUATION_SCHEMA = "mlx2.semantic-matched-evaluation.v1"

_TOOL_MARKUP = re.compile(r"<\s*(?:function\s*=|tool_call\b)", re.IGNORECASE)
_ANSWER_WRAPPER = re.compile(r"^\s*<answer>(.*?)</answer>\s*$", re.DOTALL)
_MAX_PARENT_TEXT_BYTES = 1 << 20
_MAX_TOOL_CALL_BYTES = 64 << 10
_MAX_ARM_TEXT_BYTES = 64 << 10
_MAX_EVALUATION_TEXT_BYTES = 16 << 20


def _digest(value: object) -> str:
    raw = _canonical_json(value)
    return hashlib.sha256(raw).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _text_digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _bounded_text(name: str, value: object, *, maximum: int) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text")
    if len(value.encode()) > maximum:
        raise ValueError(f"{name} exceeds the {maximum}-byte limit")
    return value


def _plain_answer(value: str) -> str:
    value = value.strip()
    if value.casefold().startswith("final:"):
        value = value[6:].strip()
    match = _ANSWER_WRAPPER.fullmatch(value)
    return (match.group(1) if match else value).strip().casefold()


@dataclass(frozen=True, slots=True)
class ParentSynthesisAssessment:
    schema: str
    attempt: int
    trigger: str | None
    disposition: str
    content_sha256: str
    content_bytes: int
    reasoning_sha256: str
    reasoning_bytes: int
    tool_calls_sha256: str
    tool_call_count: int
    remaining_output_tokens: int

    def receipt(self) -> dict:
        return asdict(self)


def assess_parent_synthesis(
    *,
    attempt: int,
    content: str,
    reasoning: str = "",
    tool_calls: Sequence[Mapping] = (),
    remaining_output_tokens: int,
) -> ParentSynthesisAssessment:
    """Accept, request one correction, or reject a parent continuation.

    Content is represented only by hashes and byte counts in the receipt.  A
    second invalid attempt fails closed, as does an invalid first attempt with
    no token left for a correction.
    """
    if type(attempt) is not int or attempt not in {0, 1}:
        raise ValueError("parent synthesis attempt must be zero or one")
    content = _bounded_text(
        "parent synthesis content", content, maximum=_MAX_PARENT_TEXT_BYTES
    )
    reasoning = _bounded_text(
        "parent synthesis reasoning", reasoning, maximum=_MAX_PARENT_TEXT_BYTES
    )
    if type(remaining_output_tokens) is not int or remaining_output_tokens < 0:
        raise ValueError("remaining output tokens must be a nonnegative integer")
    if not isinstance(tool_calls, (tuple, list)) or len(tool_calls) > 8:
        raise ValueError("parent synthesis accepts at most eight tool-call records")
    calls = []
    for call in tool_calls:
        if not isinstance(call, Mapping):
            raise TypeError("parent synthesis tool calls must be mappings")
        calls.append(dict(call))
    call_payload = json.loads(json.dumps(calls, allow_nan=False))
    if len(_canonical_json(call_payload)) > _MAX_TOOL_CALL_BYTES:
        raise ValueError("parent synthesis tool calls exceed the byte limit")
    if calls:
        trigger = "prohibited_tool_call"
    elif _TOOL_MARKUP.search(content):
        trigger = "tool_markup_content"
    elif _TOOL_MARKUP.search(reasoning):
        trigger = "tool_markup_reasoning"
    elif not content.strip():
        trigger = "empty_output"
    else:
        trigger = None
    if trigger is None:
        disposition = "accept"
    elif attempt == 0 and remaining_output_tokens > 0:
        disposition = "correct_once"
    else:
        disposition = "fail_closed"
    return ParentSynthesisAssessment(
        schema=PARENT_SYNTHESIS_SCHEMA,
        attempt=attempt,
        trigger=trigger,
        disposition=disposition,
        content_sha256=_text_digest(content),
        content_bytes=len(content.encode()),
        reasoning_sha256=_text_digest(reasoning),
        reasoning_bytes=len(reasoning.encode()),
        tool_calls_sha256=_digest(call_payload),
        tool_call_count=len(calls),
        remaining_output_tokens=remaining_output_tokens,
    )


_RESOURCE_FIELDS = (
    "prompt_tokens",
    "parent_completion_tokens",
    "child_completion_tokens",
    "forward_calls",
    "physical_width",
)


def _resource_signature(resource: Mapping) -> dict[str, int]:
    if not isinstance(resource, Mapping) or set(resource) != set(_RESOURCE_FIELDS):
        raise ValueError(f"resource must contain exactly {_RESOURCE_FIELDS}")
    result = {}
    for field in _RESOURCE_FIELDS:
        value = resource[field]
        minimum = 1 if field in {"prompt_tokens", "forward_calls", "physical_width"} else 0
        if type(value) is not int or value < minimum:
            raise ValueError(f"resource {field} must be an integer >= {minimum}")
        result[field] = value
    return result


def evaluate_matched_arms(document: Mapping) -> dict:
    """Score paired arms only when their observable resource signatures match.

    This establishes equality of the recorded token/forward counters, not equal
    FLOPs or wall time.  An invalid parent synthesis is an intention-to-treat
    miss and is excluded from the conditional-on-valid score.
    """
    if not isinstance(document, Mapping) or document.get("schema") != (
        "mlx2.semantic-matched-arms.v1"
    ):
        raise ValueError("unsupported matched-arm input schema")
    arms = document.get("arms")
    if not isinstance(arms, list) or not 2 <= len(arms) <= 1024:
        raise ValueError("matched evaluation requires 2..1024 arms")
    normalized = []
    seen = set()
    total_text_bytes = 0
    for arm in arms:
        if not isinstance(arm, Mapping):
            raise TypeError("matched arms must be mappings")
        required = {
            "arm_id", "task_id", "replicate", "treatment", "expected",
            "output", "parent_publishable", "resource",
        }
        if set(arm) != required:
            raise ValueError("matched arm fields differ from the v1 contract")
        arm_id, task_id, treatment = arm["arm_id"], arm["task_id"], arm["treatment"]
        if not all(isinstance(value, str) and value for value in (arm_id, task_id, treatment)):
            raise ValueError("arm, task and treatment identifiers must be nonempty text")
        for name, value in (
            ("arm_id", arm_id), ("task_id", task_id), ("treatment", treatment)
        ):
            _bounded_text(name, value, maximum=256)
        if arm_id in seen:
            raise ValueError("matched arm identifiers must be unique")
        seen.add(arm_id)
        replicate = arm["replicate"]
        if type(replicate) is not int or replicate < 0:
            raise ValueError("replicate must be a nonnegative integer")
        expected = _bounded_text(
            "expected answer", arm["expected"], maximum=_MAX_ARM_TEXT_BYTES
        )
        output = _bounded_text(
            "arm output", arm["output"], maximum=_MAX_ARM_TEXT_BYTES
        )
        if not _plain_answer(expected):
            raise ValueError("expected answer must be nonempty after normalization")
        total_text_bytes += sum(
            len(value.encode())
            for value in (arm_id, task_id, treatment, expected, output)
        )
        if total_text_bytes > _MAX_EVALUATION_TEXT_BYTES:
            raise ValueError("matched evaluation text exceeds the aggregate byte limit")
        if type(arm["parent_publishable"]) is not bool:
            raise TypeError("parent_publishable must be boolean")
        resource = _resource_signature(arm["resource"])
        valid = arm["parent_publishable"] and bool(_plain_answer(output))
        correct = valid and _plain_answer(output) == _plain_answer(expected)
        normalized.append(
            {
                "arm_id": arm_id,
                "task_id": task_id,
                "replicate": replicate,
                "treatment": treatment,
                "resource": resource,
                "resource_digest": _digest(resource),
                "expected_sha256": _text_digest(expected),
                "parent_publishable": arm["parent_publishable"],
                "valid": valid,
                "correct": correct,
                "output_sha256": _text_digest(output),
            }
        )
    groups = defaultdict(list)
    for arm in normalized:
        groups[(arm["task_id"], arm["replicate"])].append(arm)
    comparisons = []
    for (task_id, replicate), group in sorted(groups.items()):
        treatments = [arm["treatment"] for arm in group]
        if len(set(treatments)) != len(treatments):
            raise ValueError("a matched block cannot repeat a treatment")
        expected_digests = {arm["expected_sha256"] for arm in group}
        if len(expected_digests) != 1:
            raise ValueError("a matched block must use one exact expected answer")
        digests = {arm["resource_digest"] for arm in group}
        matched = len(digests) == 1
        comparisons.append(
            {
                "task_id": task_id,
                "replicate": replicate,
                "resource_matched": matched,
                "resource_digest": next(iter(digests)) if matched else None,
                "expected_sha256": next(iter(expected_digests)),
                "outcomes": {
                    arm["treatment"]: {
                        "valid": arm["valid"],
                        "conditional_correct": arm["correct"] if arm["valid"] else None,
                        "intention_to_treat_correct": arm["correct"],
                    }
                    for arm in sorted(group, key=lambda item: item["treatment"])
                },
            }
        )
    matched = [item for item in comparisons if item["resource_matched"]]
    treatment_panels = {
        tuple(sorted(item["outcomes"])) for item in comparisons
    }
    report = {
        "schema": MATCHED_EVALUATION_SCHEMA,
        "input_digest": _digest(document),
        "arm_count": len(normalized),
        "block_count": len(comparisons),
        "resource_matched_blocks": len(matched),
        "resource_mismatched_blocks": len(comparisons) - len(matched),
        "all_blocks_resource_matched": len(matched) == len(comparisons),
        "all_blocks_treatment_matched": len(treatment_panels) == 1,
        "resource_claim": (
            "exact equality of recorded prompt, parent completion, child completion, "
            "forward-call and physical-width counters; no FLOP or performance claim"
        ),
        "arms": normalized,
        "comparisons": comparisons,
    }
    report["matched_quality_contrast_reportable"] = bool(comparisons) and report[
        "all_blocks_resource_matched"
    ] and report["all_blocks_treatment_matched"] and all(
        len(item["outcomes"]) >= 2 for item in comparisons
    )
    report["general_quality_gain_claim_allowed"] = False
    return report


__all__ = [
    "MATCHED_EVALUATION_SCHEMA",
    "PARENT_SYNTHESIS_SCHEMA",
    "ParentSynthesisAssessment",
    "assess_parent_synthesis",
    "evaluate_matched_arms",
]

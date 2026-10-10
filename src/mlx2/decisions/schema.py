"""Backend-neutral request validation for typed decision models."""

from __future__ import annotations

import functools
import json
import math
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

QUESTION_TYPES = frozenset({"noul", "choice", "score"})
MEDIA_PART_TYPES = frozenset(
    {"image", "image_url", "input_image", "video", "video_url", "input_video"}
)
MAX_JSON_DEPTH = 64
MAX_QUESTIONS = 64
MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 10
MAX_RENDERED_STATE_BYTES = 1 << 20
_RESERVED_TOKEN = re.compile(r"<\|[^|\r\n]{1,64}\|>")


@functools.lru_cache(maxsize=8)
def _reserved_matcher(reserved_tokens: tuple[str, ...]) -> re.Pattern[str]:
    """The ``<|...|>`` shape plus every string the bound tokenizer folds."""
    if not reserved_tokens:
        return _RESERVED_TOKEN
    longest_first = sorted(reserved_tokens, key=len, reverse=True)
    return re.compile(
        "|".join([f"(?:{_RESERVED_TOKEN.pattern})", *map(re.escape, longest_first)])
    )


class DecisionRequestError(ValueError):
    """A request that cannot enter a decision-model backend."""

    def __init__(
        self, message: str, *, status: int = 400, code: str = "invalid_request"
    ):
        super().__init__(message)
        self.status = status
        self.code = code


class DecisionInputTooLong(DecisionRequestError):
    def __init__(self, message: str):
        super().__init__(message, status=413, code="input_too_long")


class DecisionExecutionFailure(RuntimeError):
    """An internal failure annotated with whether model execution started."""

    def __init__(self, *, observed_used: bool):
        super().__init__("decision execution failed")
        self.observed_used = bool(observed_used)


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    model: str
    state: Any
    questions: Mapping[str, Mapping[str, Any]]
    truncate: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "questions", MappingProxyType(dict(self.questions)))


def _validate_json_value(
    value: Any,
    *,
    field: str,
    reject_media: bool = False,
    reserved: re.Pattern[str] = _RESERVED_TOKEN,
):
    """Reject values that cannot be rendered safely and deterministically."""
    stack = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        if depth > MAX_JSON_DEPTH:
            raise DecisionRequestError(
                f"{field} exceeds the maximum nesting depth of {MAX_JSON_DEPTH}"
            )
        if current is None or isinstance(current, (bool, int)):
            continue
        if isinstance(current, str):
            try:
                current.encode("utf-8")
            except UnicodeEncodeError as error:
                raise DecisionRequestError(
                    f"{field} must contain valid Unicode text"
                ) from error
            if reserved.search(current):
                raise DecisionRequestError(
                    f"{field} must not contain reserved model control tokens"
                )
            continue
        if isinstance(current, float):
            if not math.isfinite(current):
                raise DecisionRequestError(f"{field} must not contain NaN or Infinity")
            continue
        if isinstance(current, Mapping):
            kind = current.get("type")
            if reject_media and isinstance(kind, str) and kind in MEDIA_PART_TYPES:
                raise DecisionRequestError(
                    "media inside state is not implemented in this text-only service",
                    code="unsupported_capability",
                )
            for key, child in current.items():
                if not isinstance(key, str):
                    raise DecisionRequestError(f"{field} object keys must be strings")
                try:
                    key.encode("utf-8")
                except UnicodeEncodeError as error:
                    raise DecisionRequestError(
                        f"{field} object keys must contain valid Unicode text"
                    ) from error
                # Keys are rendered verbatim like values (JSON state, option
                # ids, instruction objects), so they get the same guard.
                if reserved.search(key):
                    raise DecisionRequestError(
                        f"{field} object keys must not contain reserved model control tokens"
                    )
                stack.append((child, depth + 1))
            continue
        if isinstance(current, (list, tuple)):
            stack.extend((child, depth + 1) for child in current)
            continue
        raise DecisionRequestError(f"{field} must contain only JSON values")


def _check_label(value: Any, *, field: str, reserved: re.Pattern[str]) -> None:
    """Question names and choice labels become prompt structure verbatim."""
    if not isinstance(value, str) or not value:
        raise DecisionRequestError(f"{field} must be nonempty strings")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise DecisionRequestError(
            f"{field} must contain valid Unicode text"
        ) from error
    if any(
        unicodedata.category(character) in {"Cc", "Cs", "Zl", "Zp"}
        for character in value
    ):
        raise DecisionRequestError(f"{field} must not contain control characters")
    if reserved.search(value):
        raise DecisionRequestError(
            f"{field} must not contain reserved model control tokens"
        )


def _criteria(
    question_name: str,
    question: Mapping[str, Any],
    kind: str,
    *,
    reserved: re.Pattern[str],
):
    criteria = question.get("criteria")
    if kind == "noul":
        if criteria is None:
            return None
        if not isinstance(criteria, Mapping):
            raise DecisionRequestError(
                f"question {question_name!r}: noul criteria must be an object"
            )
        unknown = set(criteria) - {"true", "false"}
        if unknown:
            raise DecisionRequestError(
                f"question {question_name!r}: unknown noul criteria {sorted(unknown)}"
            )
        return dict(criteria)
    if kind == "score":
        if not isinstance(criteria, (list, tuple)) or len(criteria) < 2:
            raise DecisionRequestError(
                f"question {question_name!r}: score needs at least two ordered criteria"
            )
        if len(criteria) > MAX_SCORE_LEVELS:
            raise DecisionRequestError(
                f"question {question_name!r}: score supports at most "
                f"{MAX_SCORE_LEVELS} levels"
            )
        return list(criteria)
    if isinstance(criteria, (list, tuple)):
        if len(criteria) < 2:
            raise DecisionRequestError(
                f"question {question_name!r}: choice needs at least two criteria"
            )
        if len(criteria) > MAX_CHOICE_OPTIONS:
            raise DecisionRequestError(
                f"question {question_name!r}: choice supports at most "
                f"{MAX_CHOICE_OPTIONS} options"
            )
        for label in criteria:
            _check_label(
                label,
                field=f"question {question_name!r} choice labels",
                reserved=reserved,
            )
        if len(set(criteria)) != len(criteria):
            raise DecisionRequestError(
                f"question {question_name!r}: choice labels must be unique"
            )
        return dict.fromkeys(criteria)
    if not isinstance(criteria, Mapping) or len(criteria) < 2:
        raise DecisionRequestError(
            f"question {question_name!r}: choice needs at least two criteria"
        )
    if len(criteria) > MAX_CHOICE_OPTIONS:
        raise DecisionRequestError(
            f"question {question_name!r}: choice supports at most "
            f"{MAX_CHOICE_OPTIONS} options"
        )
    for label in criteria:
        _check_label(
            label,
            field=f"question {question_name!r} choice labels",
            reserved=reserved,
        )
    return dict(criteria)


def normalize_request(
    payload: Any, *, default_model: str, reserved_tokens: Iterable[str] = ()
) -> DecisionRequest:
    """Validate one System One request without importing a model runtime.

    ``reserved_tokens`` is the bound tokenizer's added-token list (see
    ``decisions.tokenizer.reserved_token_strings``); the ``<|...|>`` shape is
    always refused.
    """
    reserved = _reserved_matcher(tuple(sorted(set(reserved_tokens))))
    if not isinstance(payload, Mapping):
        raise DecisionRequestError("request body must be a JSON object")
    unknown = set(payload) - {
        "model",
        "state",
        "questions",
        "truncate",
        "images",
        "videos",
        "temperature",
    }
    if unknown:
        raise DecisionRequestError(f"unknown request fields: {sorted(unknown)}")
    if payload.get("images"):
        raise DecisionRequestError(
            "media input is not implemented in this text-only decision service",
            code="unsupported_capability",
        )
    if payload.get("videos"):
        raise DecisionRequestError(
            "video input is not implemented in this text-only decision service",
            code="unsupported_capability",
        )
    temperature = payload.get("temperature", 1)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise DecisionRequestError("temperature must be numeric")
    if temperature != 1:
        raise DecisionRequestError(
            "decision models use trained calibration and require temperature=1",
            code="unsupported_capability",
        )
    if "state" not in payload or payload["state"] is None:
        raise DecisionRequestError("state is required and must not be null")
    _validate_json_value(
        payload["state"], field="state", reject_media=True, reserved=reserved
    )
    try:
        rendered_state = json.dumps(
            payload["state"],
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as error:
        raise DecisionRequestError(
            "state must have a bounded canonical JSON representation"
        ) from error
    if len(rendered_state) > MAX_RENDERED_STATE_BYTES:
        raise DecisionRequestError(
            f"state exceeds the {MAX_RENDERED_STATE_BYTES}-byte rendered limit",
            status=413,
            code="input_too_long",
        )
    questions = payload.get("questions")
    if not isinstance(questions, Mapping) or not questions:
        raise DecisionRequestError("at least one named question is required")
    if len(questions) > MAX_QUESTIONS:
        raise DecisionRequestError(
            f"at most {MAX_QUESTIONS} questions are allowed per request"
        )
    normalized: dict[str, Mapping[str, Any]] = {}
    for name, raw in questions.items():
        _check_label(name, field="question names", reserved=reserved)
        if not isinstance(raw, Mapping):
            raise DecisionRequestError(f"question {name!r} must be an object")
        unknown_question = set(raw) - {"type", "instructions", "criteria"}
        if unknown_question:
            raise DecisionRequestError(
                f"question {name!r}: unknown fields {sorted(unknown_question)}"
            )
        kind = raw.get("type")
        if kind == "bool":
            kind = "noul"
        if not isinstance(kind, str) or kind not in QUESTION_TYPES:
            raise DecisionRequestError(
                f"question {name!r}: type must be noul, choice, or score"
            )
        instructions = raw.get("instructions")
        if instructions is not None and not isinstance(
            instructions, (str, list, tuple, Mapping)
        ):
            raise DecisionRequestError(
                f"question {name!r}: instructions must be text, an object, or an array"
            )
        if instructions is not None:
            _validate_json_value(
                instructions,
                field=f"question {name!r} instructions",
                reject_media=True,
                reserved=reserved,
            )
        criteria = _criteria(name, raw, kind, reserved=reserved)
        if criteria is not None:
            _validate_json_value(
                criteria,
                field=f"question {name!r} criteria",
                reject_media=True,
                reserved=reserved,
            )
        normalized[name] = MappingProxyType(
            {
                "type": kind,
                "instructions": instructions,
                "criteria": criteria,
            }
        )
    model = payload.get("model", default_model)
    if not isinstance(model, str) or not model:
        raise DecisionRequestError("model must be a nonempty string")
    if model != default_model:
        raise DecisionRequestError(
            f"unknown model {model!r}; this server exposes {default_model!r}",
            status=404,
            code="model_not_found",
        )
    truncate = payload.get("truncate", True)
    if type(truncate) is not bool:
        raise DecisionRequestError("truncate must be boolean")
    return DecisionRequest(
        model=model,
        state=payload["state"],
        questions=normalized,
        truncate=truncate,
    )

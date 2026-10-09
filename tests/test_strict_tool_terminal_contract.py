"""Strict tool schemas admitted at the HTTP boundary are checked after generation.

``validate_request`` admits strict function schemas through the structured-
output compiler, whose subset includes string ``minLength``/``maxLength``,
``anyOf`` (Pydantic's ``Optional[...]``) and object schemas that carry
``properties`` without ``type``.  With the default ``constrained_tool_grammar``
off, the terminal ``enforce_tool_contract`` is the only check, so it must
enforce every keyword the subset admits: reject an out-of-bounds string, and
accept a call that matches an admitted union or typeless object instead of
failing every call with a 502.
"""

import json

import pytest

from mlx2.openai_compat import ToolContractError, enforce_tool_contract
from mlx2.server import validate_request

_BOUNDED = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "minLength": 3, "maxLength": 5},
        "note": {"type": "string", "maxLength": 2},
    },
    "required": ["code"],
    "additionalProperties": False,
}
_NULLABLE = {
    # OpenAI SDK / Pydantic strict output for ``unit: Optional[str]``.
    "type": "object",
    "additionalProperties": False,
    "required": ["city", "unit"],
    "properties": {
        "city": {"type": "string"},
        "unit": {"anyOf": [{"type": "string"}, {"type": "null"}], "title": "Unit"},
    },
}
_NULLABLE_REF = {
    # Pydantic ``detail: Optional[Detail]``: anyOf [$ref, null].
    "type": "object",
    "additionalProperties": False,
    "required": ["p"],
    "properties": {"p": {"anyOf": [{"$ref": "#/$defs/P"}, {"type": "null"}]}},
    "$defs": {
        "P": {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
            "additionalProperties": False,
        }
    },
}
_TYPELESS = {
    "properties": {"x": {"type": "string"}},
    "required": ["x"],
    "additionalProperties": False,
}


def _admit(parameters, choice):
    return validate_request({
        "messages": [{"role": "user", "content": "call"}],
        "tools": [{
            "type": "function",
            "function": {"name": "f", "strict": True, "parameters": parameters},
        }],
        "tool_choice": choice,
    })


def _call(arguments):
    return [{
        "id": "call_1",
        "type": "function",
        "function": {"name": "f", "arguments": json.dumps(arguments)},
    }]


@pytest.mark.parametrize("choice", ["auto", "required"])
def test_strict_tool_string_length_bounds_fail_closed(choice):
    body = _admit(_BOUNDED, choice)
    for good in (
        {"code": "abc"},
        {"code": "abcde"},
        {"code": "abcd", "note": ""},
        {"code": "abcd", "note": "hi"},
    ):
        enforce_tool_contract(body, _call(good), finish_reason="tool_calls")
    for bad in (
        {"code": "x"},
        {"code": ""},
        {"code": "abcdefghij"},
        {"code": "abcd", "note": "too long"},
    ):
        with pytest.raises(ToolContractError, match="length"):
            enforce_tool_contract(body, _call(bad), finish_reason="tool_calls")


@pytest.mark.parametrize("choice", ["auto", "required"])
@pytest.mark.parametrize(
    "parameters,arguments",
    [
        (_NULLABLE, {"city": "Paris", "unit": None}),
        (_NULLABLE, {"city": "Paris", "unit": "C"}),
        (_NULLABLE_REF, {"p": {"x": "a"}}),
        (_NULLABLE_REF, {"p": None}),
        (_TYPELESS, {"x": "a"}),
    ],
)
def test_admitted_strict_schema_accepts_schema_valid_call(parameters, arguments, choice):
    body = _admit(parameters, choice)
    enforce_tool_contract(body, _call(arguments), finish_reason="tool_calls")


@pytest.mark.parametrize(
    "parameters,arguments",
    [
        (_NULLABLE, {"city": "Paris", "unit": 3}),
        (_NULLABLE, {"city": "Paris"}),
        (_NULLABLE_REF, {"p": {"y": "a"}}),
        (_NULLABLE_REF, {"p": "a"}),
        (_TYPELESS, {"x": 1}),
        (_TYPELESS, {"x": "a", "extra": 1}),
        (_TYPELESS, []),
    ],
)
def test_admitted_strict_union_still_fails_closed_on_invalid_call(parameters, arguments):
    body = _admit(parameters, "auto")
    with pytest.raises(ToolContractError):
        enforce_tool_contract(body, _call(arguments), finish_reason="tool_calls")


def _single(prop):
    return {
        "type": "object",
        "properties": {"v": prop},
        "required": ["v"],
        "additionalProperties": False,
    }


@pytest.mark.parametrize(
    "prop,value",
    [
        # Python's True == 1 (also inside containers) must not let a JSON
        # boolean satisfy a numeric enum/const, or the reverse.
        ({"enum": [1]}, True),
        ({"enum": [0, "x"]}, False),
        ({"const": 1}, True),
        ({"const": 0}, False),
        ({"enum": [True]}, 1),
        ({"const": False}, 0),
        ({"enum": [[1]]}, [True]),
        ({"const": [1, 2]}, [True, 2]),
        ({"const": {"a": 1}}, {"a": True}),
        ({"enum": [{"a": [0]}]}, {"a": [False]}),
        ({"const": {"a": True}}, {"a": 1}),
    ],
)
def test_strict_enum_and_const_distinguish_booleans_from_numbers(prop, value):
    body = _admit(_single(prop), "auto")
    with pytest.raises(ToolContractError, match="enum|const"):
        enforce_tool_contract(body, _call({"v": value}), finish_reason="tool_calls")


@pytest.mark.parametrize(
    "prop,value",
    [
        ({"enum": [1]}, 1),
        ({"enum": [1]}, 1.0),
        ({"const": 1.0}, 1),
        ({"enum": [True]}, True),
        ({"const": False}, False),
        ({"enum": [[1]]}, [1.0]),
        ({"const": {"a": 1}}, {"a": 1.0}),
        ({"enum": [{"a": [True]}]}, {"a": [True]}),
    ],
)
def test_strict_enum_and_const_keep_json_value_equality(prop, value):
    body = _admit(_single(prop), "auto")
    enforce_tool_contract(body, _call({"v": value}), finish_reason="tool_calls")

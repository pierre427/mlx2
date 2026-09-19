# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; original notices in provenance/NOTICE.

"""
Modified from:
https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct/blob/main/qwen3coder_tool_parser.py
"""

import ast
import json
import logging
import math
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

import regex as re

from ._schema import infer_type_from_json_schema

# Match each <function=...>...</function> block individually (no trailing `$`
# anchor, which would otherwise merge several blocks into one greedy match and
# drop calls 2..n).
_function_regex = re.compile(r"<function=(.*?)</function>", re.DOTALL)
_parameter_regex = re.compile(r"<parameter=(.*?)</parameter>", re.DOTALL)
_name_regex = re.compile(r"\s*([^\s<>]+)>?")

_string_types = {"string", "str", "text", "varchar", "char", "enum"}
_bool_types = {"boolean", "bool", "binary"}
_obj_types = {"object", "array", "arr"}


def _get_arguments_config(func_name: str, tools: Optional[Any]) -> dict:
    """Extract argument configuration for a function."""
    if tools is None:
        return {}
    for tool in tools:
        if not (function := tool.get("function", False)):
            continue
        if function["name"] == func_name:
            if not (params := function.get("parameters", False)):
                return {}
            return params.get("properties", {})
    return {}


def _convert_param_value(param_value: str, param_name: str, param_config: dict) -> Any:
    """Convert parameter value based on its type in the schema."""
    if not (param := param_config.get(param_name, False)):
        return None if param_value.lower() == "null" else param_value

    # Resolve anyOf/oneOf/list-form unions to a concrete non-null type; an
    # unresolved schema is treated as a string (values returned verbatim).
    inferred = infer_type_from_json_schema(param)
    param_type = inferred.strip().lower() if inferred else "string"
    if param_value.lower() == "null":
        if _declares_null(param):
            return None
        if param_type not in _string_types:
            raise ValueError(f"Null is not allowed for {param_name}")
    if param_type in _string_types:
        return param_value
    elif (
        param_type.startswith("int")
        or param_type.startswith("uint")
        or param_type.startswith("long")
        or param_type.startswith("short")
        or param_type.startswith("unsigned")
    ):
        try:
            value = Decimal(param_value)
        except InvalidOperation as exc:
            raise ValueError(f"Invalid integer literal {param_value!r}") from exc
        if not value.is_finite() or value != value.to_integral_value():
            raise ValueError(f"Invalid integer literal {param_value!r}")
        # Bound exponent expansion just as Python bounds decimal int strings.
        if value and value.adjusted() >= 4300:
            raise ValueError("Integer literal exceeds 4300 digits")
        return int(value)
    elif param_type.startswith("num") or param_type.startswith("float"):
        float_param_value = float(param_value)
        if not math.isfinite(float_param_value):
            raise ValueError(f"Invalid number literal {param_value!r}")
        # Preserve integral JSON numbers exactly here too.
        exact = Decimal(param_value)
        return int(exact) if exact == exact.to_integral_value() else float_param_value
    elif param_type in _bool_types:
        if param_value.lower() not in ("true", "false"):
            raise ValueError(f"Invalid boolean literal {param_value!r}")
        return param_value.lower() == "true"
    else:
        if (
            param_type in _obj_types
            or param_type.startswith("dict")
            or param_type.startswith("list")
        ):
            try:
                return json.loads(param_value, strict=False)
            except json.JSONDecodeError:
                return _safe_literal_eval(param_value)

        # Unknown / unresolved type: try a literal, but never let a malformed
        # value raise (e.g. SyntaxError) — fall back to the raw string.
        return _safe_literal_eval(param_value)


def _safe_literal_eval(param_value: str) -> Any:
    """ast.literal_eval that returns the raw string instead of raising."""
    try:
        return ast.literal_eval(param_value)
    except (ValueError, SyntaxError):
        return param_value


def _declares_null(schema):
    if not isinstance(schema, dict):
        return False
    declared = schema.get("type")
    if declared == "null" or isinstance(declared, list) and "null" in declared:
        return True
    return any(
        _declares_null(branch)
        for key in ("anyOf", "oneOf")
        for branch in schema.get(key, [])
    )


def _parse_xml_function_call(function_call_str: str, tools: Optional[Any]):
    name_match = _name_regex.match(function_call_str)
    if name_match is None:
        raise ValueError("Malformed function name")
    function_name = name_match.group(1)
    if not function_name.strip() or "<" in function_name:
        raise ValueError("Malformed function name")
    param_config = _get_arguments_config(function_name, tools)
    parameters = function_call_str[name_match.end() :]
    param_dict = {}
    cursor = 0
    for match in _parameter_regex.finditer(parameters):
        if parameters[cursor : match.start()].strip():
            raise ValueError("Malformed parameter markup")
        cursor = match.end()
        match_text = match.group(1)
        param_match = _name_regex.match(match_text)
        if param_match is None:
            raise ValueError("Malformed parameter name")
        param_name = param_match.group(1)
        if not param_name.strip() or "<" in param_name or param_name in param_dict:
            raise ValueError("Malformed or duplicate parameter name")
        param_value = str(match_text[param_match.end() :])
        if param_value.startswith("\n"):
            param_value = param_value[1:]
        if param_value.endswith("\n"):
            param_value = param_value[:-1]

        param_dict[param_name] = _convert_param_value(
            param_value, param_name, param_config
        )
    if parameters[cursor:].strip():
        raise ValueError("Incomplete or malformed parameter markup")
    return dict(name=function_name, arguments=param_dict)


tool_call_start = "<tool_call>"

tool_call_end = "</tool_call>"


def parse_tool_call(
    model_output: str,
    tools: Optional[Any] = None,
):
    matches = _function_regex.findall(model_output)
    if not matches:
        raise ValueError("No function provided.")
    calls = []
    for match in matches:
        try:
            calls.append(_parse_xml_function_call(match, tools))
        except ValueError as exc:
            logging.warning("Dropping malformed Qwen function: %s", exc)
    if not calls:
        raise ValueError("No valid function provided.")
    return calls[0] if len(calls) == 1 else calls

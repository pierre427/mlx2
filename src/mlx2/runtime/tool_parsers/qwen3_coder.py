# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; original notices in provenance/NOTICE.

"""
Modified from:
https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct/blob/main/qwen3coder_tool_parser.py
"""

import ast
import json
import math
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

import regex as re

from ._schema import (
    executable_schema,
    infer_type_from_json_schema,
    json_native,
    raw_string_pattern,
    required_parameter_names,
    resolve_local_refs,
    schema_value_matches,
)

_name_regex = re.compile(r"\s*([^\s<>]+)>?")

_FUNCTION_OPEN = "<function="
_FUNCTION_CLOSE = "</function>"
_PARAMETER_OPEN = "<parameter="
_PARAMETER_CLOSE = "</parameter>"
_TOOL_CALL_CLOSE = "</tool_call>"
_CLOSER = re.compile(r"</parameter>|</function>")
_JSON_STRING = re.compile(r'"[^"\\]*(?:\\.[^"\\]*)*"', re.DOTALL)
# Markup a raw value can never contain.  The parser ends a raw value at the
# first parameter or function closer (a function closer also closes an open
# last parameter, vllm #57707), reads an opener as an unclosed parameter, and
# the output parser ends the call at the first tool-call closer.  The chat
# template writes a string argument raw, so a model could spell any of them
# inside one; no reader of this wire can tell it from markup, and the
# reference parser truncates the value there too.
_RAW_DELIMITERS = (_PARAMETER_CLOSE, _PARAMETER_OPEN, _FUNCTION_CLOSE, _TOOL_CALL_CLOSE)
_RAW_VALUE_CHAR = r"(?:(?!</parameter>|<parameter=|</function>|</tool_call>)[\s\S])"


class UnclosedJSONString(ValueError):
    """The text ends inside a string of a strict parameter's JSON value.

    Qwen writes JSON values with ``tojson``, which leaves ``<`` raw, and the
    strict grammar admits a ``</tool_call>`` inside a JSON string.  The output
    parser cut the call at that closer, so the call runs on to a later one.
    """

    tool_call_close_quoted = True


_string_types = {"string", "str", "text", "varchar", "char", "enum"}
_bool_types = {"boolean", "bool", "binary"}
_obj_types = {"object", "array", "arr"}


def _get_arguments_config(func_name: str, tools: Optional[Any]) -> tuple[dict, bool]:
    """Extract argument configuration for a function."""
    if tools is None:
        return {}, False
    for tool in tools:
        if not (function := tool.get("function", False)):
            continue
        if function["name"] == func_name:
            if not (params := function.get("parameters", False)):
                return {}, bool(function.get("strict", False))
            try:
                # Strict values are checked against the executable schema,
                # which carries no annotation keywords; non-strict parsing
                # reads only types and keeps the declared shape.
                params = (
                    executable_schema(params)
                    if function.get("strict", False)
                    else resolve_local_refs(params)
                )
            except ValueError:
                if function.get("strict", False):
                    raise
                # Request validation deliberately keeps unsupported references
                # untouched for non-strict tools. Their legacy parser remains
                # best-effort too rather than turning an accepted request into
                # a terminal parse error.
                pass
            return params.get("properties", {}), bool(function.get("strict", False))
    return {}, False


def _convert_param_value(
    param_value: str, param_name: str, param_config: dict, *, strict: bool = False
) -> Any:
    """Convert parameter value based on its type in the schema."""
    param = param_config.get(param_name)
    # An empty schema ``{}`` is falsy but present: it admits any JSON value,
    # so a non-strict value takes the untyped path below and an object or
    # array the model wrote is decoded, and a strict value decodes as JSON.
    # Only an undeclared parameter is unknown.
    if param is None:
        return None if param_value.lower() == "null" else param_value

    if strict:
        if _raw_parameter_pattern(param) is not None:
            value = (
                None
                if param_value.lower() == "null" and _declares_null(param)
                else param_value
            )
        else:
            try:
                value = json.loads(param_value)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Strict parameter {param_name} must contain valid JSON"
                ) from exc
        if not schema_value_matches(value, param):
            raise ValueError(f"Strict parameter {param_name} violates its schema")
        return value

    # Resolve anyOf/oneOf/list-form unions to a concrete non-null type.
    inferred = infer_type_from_json_schema(param)
    if not inferred:
        # A schema without a resolvable type admits any JSON value.  Decode
        # objects and arrays, which the model can only mean structurally;
        # keep scalars verbatim so text such as "123" is not coerced to a
        # number (mlx-lm#1910).
        if param_value.lower() == "null" and _declares_null(param):
            return None
        try:
            value = json.loads(param_value, strict=False)
        except ValueError:
            # Not only a JSONDecodeError: an integer past the interpreter's
            # digit limit raises a plain ValueError.
            return param_value
        if isinstance(value, (dict, list)):
            return _servable(value, param_value)
        return param_value
    param_type = inferred.strip().lower()
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
                return _servable(json.loads(param_value, strict=False), param_value)
            except ValueError:  # a JSONDecodeError, or an integer too long
                return _safe_literal_eval(param_value)

        # Unknown / unresolved type: try a literal, but never let a malformed
        # value raise (e.g. SyntaxError) — fall back to the raw string.
        return _safe_literal_eval(param_value)


def _safe_literal_eval(param_value: str) -> Any:
    """ast.literal_eval that returns the raw string instead of raising.

    Building a literal can raise more than ``ValueError`` and ``SyntaxError``:
    an unhashable dict key or set member (``{[1]: 2}``) raises ``TypeError``.
    """
    try:
        value = ast.literal_eval(param_value)
    except Exception:  # noqa: BLE001 - any failure means "not a literal"
        return param_value
    return _servable(value, param_value)


def _servable(value: Any, param_value: str) -> Any:
    """``value``, or the raw text when a tool call's arguments cannot carry it.

    Best-effort decoding of free text can yield a non-finite float (``NaN``,
    ``1e400``) or a Python-only literal (a set, bytes).  The arguments are
    serialized with ``allow_nan=False``, so such a value would turn a call the
    grammar admitted into an error; a tuple or a non-string key would be
    served as the array or string key JSON makes of it, which the model did
    not write.  Like other undecodable text it stays the string the model
    wrote.
    """
    return value if json_native(value) else param_value


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


def _raw_parameter_pattern(schema):
    """Exact raw Qwen string language, bounded only by the request token cap."""
    if not isinstance(schema, dict):
        return None
    declared = schema.get("type")
    nullable_string = (
        isinstance(declared, list)
        and "string" in declared
        and set(declared) <= {"string", "null"}
        and set(schema) == {"type"}
    )
    if (declared == "string" and set(schema) == {"type"}) or nullable_string:
        return _RAW_VALUE_CHAR + "*"
    if declared == "string" and set(schema) <= {"type", "minLength", "maxLength"}:
        minimum = schema.get("minLength", 0)
        maximum = schema.get("maxLength")
        if type(minimum) is not int or minimum < 0:
            raise ValueError("string minLength must be a non-negative integer")
        if maximum is not None and (
            type(maximum) is not int or maximum < minimum
        ):
            raise ValueError("string maxLength must be an ordered non-negative integer")
        quantifier = (
            f"{{{minimum},}}"
            if maximum is None
            else f"{{{minimum},{maximum}}}"
        )
        return _RAW_VALUE_CHAR + quantifier
    if set(schema) <= {"type", "enum"} and isinstance(schema.get("enum"), list):
        values = schema["enum"]
        if values and all(isinstance(value, str) for value in values):
            if any(
                delimiter in value
                for value in values
                for delimiter in _RAW_DELIMITERS
            ):
                raise ValueError("string enum contains a tool-wire delimiter")
            return "(?:" + "|".join(re.escape(value) for value in values) + ")"
    if set(schema) <= {"type", "const"} and isinstance(schema.get("const"), str):
        value = schema["const"]
        if any(delimiter in value for delimiter in _RAW_DELIMITERS):
            raise ValueError("string const contains a tool-wire delimiter")
        return re.escape(value)
    if declared == "string":
        # Preserve the existing fail-closed error for unsupported string
        # keywords such as pattern.
        return raw_string_pattern(schema)
    return None


def _decodes_as_json(param, strict):
    """Whether ``_convert_param_value`` decodes this parameter's text as JSON."""
    if not strict or param is None:
        return False
    try:
        return _raw_parameter_pattern(param) is None
    except ValueError:
        return False  # the conversion raises the same error


def _json_value_end(text, start, name):
    """Where the closer after a strict JSON value that starts at ``start`` is.

    Qwen writes JSON values with ``tojson``, which leaves ``<`` raw, and the
    strict grammar admits any closing tag inside a JSON string.  Outside a
    string ``<`` is not JSON, so the value ends at the first parameter or
    function closer outside every string.  Returns -1 when none follows.
    """
    position = start
    while True:
        closer = _CLOSER.search(text, position)
        close = closer.start() if closer is not None else -1
        quote = text.find('"', position, close if close >= 0 else len(text))
        if quote < 0:
            return close
        string = _JSON_STRING.match(text, quote)
        if string is None:
            raise UnclosedJSONString(
                f"Malformed Qwen function: parameter {name} ends inside a JSON string"
            )
        position = string.end()


def _parse_function(text: str, start: int, tools: Optional[Any]):
    """``(call, end)`` for the ``<function=`` block whose body starts at ``start``.

    ``end`` is the index of the ``</function>`` that closes the block, found by
    walking the parameters in order: a raw value ends at the first closer,
    while a strict JSON value ends at the first closer outside its strings.
    ``None`` means no closer follows, and the block is ignored like any
    unclosed markup, unless the text ends inside a JSON string.
    """
    try:
        return _walk_function(text, start, tools)
    except UnclosedJSONString:
        raise
    except ValueError:
        if text.find(_FUNCTION_CLOSE, start) >= 0:
            raise
        return None  # unclosed markup, malformed or not


def _walk_function(text: str, start: int, tools: Optional[Any]):
    name_match = _name_regex.match(text, start)
    if name_match is None:
        raise ValueError("Malformed function name")
    function_name = name_match.group(1)
    if not function_name.strip() or "<" in function_name:
        raise ValueError("Malformed function name")
    param_config, strict = _get_arguments_config(function_name, tools)
    param_dict = {}

    def add(match_text, *, implicit_close=False, json_value=False):
        param_match = _name_regex.match(match_text)
        if param_match is None or (
            implicit_close and not param_match.group(0).endswith(">")
        ):
            raise ValueError("Malformed parameter name")
        param_name = param_match.group(1)
        if not param_name.strip() or "<" in param_name or param_name in param_dict:
            raise ValueError("Malformed or duplicate parameter name")
        param_value = str(match_text[param_match.end() :])
        if not json_value and _PARAMETER_OPEN in param_value:
            # The previous parameter was never closed and swallowed the next
            # one, whose argument would silently disappear.  The reference
            # parser never lets a value contain an opener either.  In a JSON
            # value an opener outside a string fails the decode instead.
            raise ValueError("Unclosed parameter before the next parameter")
        if param_value.startswith("\n"):
            param_value = param_value[1:]
        if param_value.endswith("\n"):
            param_value = param_value[:-1]

        param_dict[param_name] = _convert_param_value(
            param_value, param_name, param_config, strict=strict
        )

    position = name_match.end()
    while True:
        opener = text.find(_PARAMETER_OPEN, position)
        close = text.find(_FUNCTION_CLOSE, position)
        if opener < 0 or 0 <= close < opener:
            if close < 0:
                return None
            if text[position:close].strip():
                raise ValueError("Incomplete or malformed parameter markup")
            return dict(name=function_name, arguments=param_dict), close
        if text[position:opener].strip():
            raise ValueError("Malformed parameter markup")
        body = opener + len(_PARAMETER_OPEN)
        param_match = _name_regex.match(text, body)
        json_value = param_match is not None and _decodes_as_json(
            param_config.get(param_match.group(1)), strict
        )
        if json_value:
            end = _json_value_end(text, param_match.end(), param_match.group(1))
        else:
            closer = _CLOSER.search(text, body)
            end = closer.start() if closer is not None else -1
        if end < 0:
            return None
        # vllm #57707: a model can close </function> without closing its last
        # parameter.  The function end closes that one parameter implicitly;
        # a later opener inside the value is still malformed.
        implicit_close = text.startswith(_FUNCTION_CLOSE, end)
        add(text[body:end], implicit_close=implicit_close, json_value=json_value)
        if implicit_close:
            return dict(name=function_name, arguments=param_dict), end
        position = end + len(_PARAMETER_CLOSE)


tool_call_start = "<tool_call>"

tool_call_end = "</tool_call>"


def parse_tool_call(
    model_output: str,
    tools: Optional[Any] = None,
):
    """Every closed ``<function=`` block of one tool-call block, in order.

    Raises ``UnclosedJSONString`` when the text ends inside a strict JSON
    string: the ``</tool_call>`` that ended it was quoted inside the value.
    """
    calls, cursor = [], 0
    while (start := model_output.find(_FUNCTION_OPEN, cursor)) >= 0:
        # One malformed block makes the whole block malformed.  Dropping it
        # and returning its siblings would bypass OutputParser's policy: a
        # required tool choice would pass with a call missing, and a tolerant
        # request would never count the fallback.
        try:
            parsed = _parse_function(
                model_output, start + len(_FUNCTION_OPEN), tools
            )
        except UnclosedJSONString:
            raise
        except ValueError as exc:
            raise ValueError(f"Malformed Qwen function: {exc}") from exc
        if parsed is None:
            break
        call, end = parsed
        calls.append(call)
        cursor = end + len(_FUNCTION_CLOSE)
    if not calls:
        raise ValueError("No function provided.")
    return calls[0] if len(calls) == 1 else calls


def _selected_functions(tools, tool_choice):
    definitions = [tool["function"] for tool in tools]
    if isinstance(tool_choice, dict):
        name = tool_choice["function"]["name"]
        return [function for function in definitions if function["name"] == name]
    return definitions


def _strict_parameter_body(function):
    """Canonical Qwen XML parameters constrained by a strict JSON schema."""
    from ...structured_output import _schema_pattern

    schema = executable_schema(function.get("parameters", {}))
    def require_postcheck_schema(node):
        if not isinstance(node, dict):
            return
        if "anyOf" in node or "oneOf" in node:
            raise ValueError(
                "strict tool schema unions are unsupported by the post-generation validator"
            )
        for child in (node.get("properties") or {}).values():
            require_postcheck_schema(child)
        if isinstance(node.get("items"), dict):
            require_postcheck_schema(node["items"])

    require_postcheck_schema(schema)
    if schema.get("type") not in (None, "object") and "properties" not in schema:
        raise ValueError("strict function parameters must use an object schema")
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    if not isinstance(properties, dict) or not isinstance(required, list):
        raise ValueError("strict function parameters require object properties")
    if schema.get("additionalProperties", False) not in (False, None):
        raise ValueError("strict function parameters require additionalProperties:false")
    ordered = [name for name in properties if name in required] + [
        name for name in properties if name not in required
    ]
    blocks = []
    for name in ordered:
        if not isinstance(name, str) or not re.fullmatch(r"[^\s<>]+", name):
            raise ValueError("tool parameter names are not representable in Qwen XML")
        subschema = properties[name]
        value = _raw_parameter_pattern(subschema)
        if value is None:
            value = _schema_pattern(subschema, finite_numbers=True)
        block = rf"\n<parameter={re.escape(name)}>\n{value}\n</parameter>"
        blocks.append((block, name in required))
    return "".join(block if needed else f"(?:{block})?" for block, needed in blocks)


def _non_strict_value(schema):
    """Value language the non-strict parser converts without raising.

    Scalar-typed values are converted (``_convert_param_value``), so free text
    for them is admitted by the grammar and then rejected by the parser; every
    other value is kept or decoded best-effort and stays free text.
    """
    from ...structured_output import _FINITE_NUMBER, _INTEGER

    kind = infer_type_from_json_schema(schema)
    kind = kind.strip().lower() if isinstance(kind, str) else None
    pattern = {
        "integer": _INTEGER,
        "number": _FINITE_NUMBER,
        "boolean": "(?:true|false)",
    }.get(kind)
    if pattern is None:
        return _RAW_VALUE_CHAR + "*"
    return f"(?:{pattern}|null)" if _declares_null(schema) else pattern


def _non_strict_parameter_body(function):
    """Qwen XML parameters of a non-strict tool, as its parser accepts them.

    Every required argument appears, before any optional one (sglang #40051:
    an all-optional body lets greedy decoding emit a call with no arguments).
    Optional arguments are the declared ones, each at most once in declared
    order, since the parser rejects a repeated name.  Only a schema that
    declares no properties keeps free argument names.
    """
    schema = function.get("parameters")
    if not isinstance(schema, dict):
        schema = {}
    try:
        schema = resolve_local_refs(schema)
    except ValueError:
        pass  # best-effort, as for non-strict parsing
    properties = schema.get("properties")
    properties = properties if isinstance(properties, dict) else {}
    required = required_parameter_names(function)
    for parameter in required:
        if not isinstance(parameter, str) or not re.fullmatch(r"[^\s<>]+", parameter):
            raise ValueError("tool parameter names are not representable in Qwen XML")

    def block(name):
        value = _non_strict_value(properties.get(name))
        return rf"\n<parameter={re.escape(name)}>\n{value}\n</parameter>"

    body = "".join(block(name) for name in required)
    if properties:
        # An optional name the wire cannot spell is simply never emitted.
        body += "".join(
            f"(?:{block(name)})?"
            for name in properties
            if name not in required
            and isinstance(name, str)
            and re.fullmatch(r"[^\s<>]+", name)
        )
    elif schema.get("additionalProperties") is not False:
        body += (
            r"(?:\n<parameter=[A-Za-z_][A-Za-z0-9_.-]{0,127}>\n"
            rf"{_RAW_VALUE_CHAR}*\n</parameter>){{0,{max(0, 64 - len(required))}}}"
        )
    return body


def constrained_tool_grammar(tools, tool_choice, *, parallel_tool_calls=True):
    """Regex for Qwen3-Coder's XML tool-call wire format.

    Design references: vLLM structural tags and vllm#57455. Strict functions
    reuse mlx2's bounded JSON-schema compiler for non-string parameter values;
    XML string values use bounded tag-free text.
    """
    functions = _selected_functions(tools, tool_choice)
    if not functions:
        raise ValueError("constrained tool_choice selects no declared function")
    calls = []
    for function in functions:
        name = function["name"]
        if not re.fullmatch(r"[^\s<>]+", name):
            raise ValueError("tool name is not representable in Qwen XML")
        if function.get("strict", False):
            body = _strict_parameter_body(function)
        else:
            body = _non_strict_parameter_body(function)
        calls.append(
            rf"<tool_call>\n<function={re.escape(name)}>{body}\n</function>\n</tool_call>"
        )
    call = "(?:" + "|".join(calls) + ")"
    return call if parallel_tool_calls is False else call + rf"(?:\n{call})*"

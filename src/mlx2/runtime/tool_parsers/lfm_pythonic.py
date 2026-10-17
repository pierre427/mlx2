"""Parser for LFM pythonic tool-call lists.

LiquidAI LFM2/LFM2.5 chat templates render an assistant tool call as a
Python-style list of keyword calls between two single added tokens::

    <|tool_call_start|>[get_weather(city='Paris', units="c"), n(q=[1, 2])]<|tool_call_end|>

The template (``format_arg_value``) writes a string argument single-quoted
with ``\\``, ``'``, newline and carriage return backslash-escaped, a mapping or
list argument with ``tojson`` (so ``true``/``false``/``null`` may appear inside
it), and any other scalar with Jinja's ``string`` filter (``True``, ``False``,
``None``, ``3``, ``1.5``).  A function or argument name is written raw.

This parser reads that language and nothing more general: every argument must
be ``name=<literal>``.  Values are decoded by a whitelist walk over the
``ast`` of the value alone (never ``eval``): strings, numbers, booleans,
``None``, lists, tuples (served as lists) and string-keyed dicts, with the
JSON names ``true``/``false``/``null`` admitted as well.  A list or dict value
is decoded as JSON first, because the template wrote it with ``tojson`` (the
two languages disagree on ``"\\/"``).  Positional or starred arguments, calls,
operators, f-strings, bytes, complex numbers, sets and values JSON cannot carry
(non-finite floats, non-string keys) are rejected with ``ValueError``.

Names are read by delimiters rather than as Python identifiers: a function
name is the text before ``(`` and an argument name the text before ``=``, so
tool names such as ``db.lookup`` or ``get-weather`` and parameters such as
``from`` round-trip as the template renders them.  Neither may contain
whitespace or any of ``()[]{},'"=``; :func:`representable_name` says so.

Shapes follow the other parsers of this package: :func:`parse_tool_call`
returns one ``{"name", "arguments"}`` dict, or a list of them for a parallel
call, and carries :func:`partial_function_names` for the output parser's
truncation records.  When the body ends inside a string literal the closer it
was cut at was quoted inside an argument (the template leaves ``<`` raw), so
the error carries ``tool_call_close_quoted`` and the output parser runs the
call on to a later closer.

Design reference only (no code taken): the regex-based ``pythonic.py`` tool
parser of mlx-lm-unified 1104ced1 (MIT); see
``provenance/lfm25-pythonic-tools.json``.
"""

from __future__ import annotations

import ast
import json
import re
import warnings
from typing import Any

from ._schema import json_native

tool_call_start = "<|tool_call_start|>"
tool_call_end = "<|tool_call_end|>"

# What a function or argument name may not contain: the delimiters the
# scanner reads names by, and whitespace.
_NAME = re.compile(r"[^\s()\[\]{},'\"=]+")
_WHITESPACE = re.compile(r"\s*")
_PAIRS = {"(": ")", "[": "]", "{": "}"}
_JSON_NAMES = {"true": True, "false": False, "null": None}
_MAX_DEPTH = 64


class UnclosedPythonString(ValueError):
    """The text ends inside a string literal of an argument value.

    The template leaves ``<`` raw inside strings, so a string may quote
    ``<|tool_call_end|>``.  The output parser cut the call at that closer, so
    the call runs on to a later one.
    """

    tool_call_close_quoted = True


def representable_name(name: Any) -> bool:
    """Whether ``name`` survives the template's raw rendering and this parser."""
    return isinstance(name, str) and _NAME.fullmatch(name) is not None


def _skip_ws(text: str, position: int) -> int:
    return _WHITESPACE.match(text, position).end()


def _string_end(text: str, start: int) -> int:
    """Index just past the string literal whose quote is at ``start``.

    A backslash always protects the next character (also in a raw string, as
    in Python's tokenizer).  A raw newline ends nothing in a one-quote string
    and is a malformed value; text ending inside a literal raises
    :class:`UnclosedPythonString`.
    """
    quote = text[start]
    delimiter = quote * 3 if text.startswith(quote * 3, start) else quote
    position = start + len(delimiter)
    while position < len(text):
        char = text[position]
        if char == "\\":
            position += 2
            continue
        if text.startswith(delimiter, position):
            return position + len(delimiter)
        if char == "\n" and len(delimiter) == 1:
            raise ValueError("LFM tool-call string literal contains a raw newline")
        position += 1
    raise UnclosedPythonString("LFM tool call ends inside a string literal")


def _value_end(text: str, position: int) -> int:
    """Index of the ``,`` or bracket that ends the value starting at ``position``.

    Brackets nest and string literals are skipped.  Returns -1 when the text
    ends first (raises :class:`UnclosedPythonString` inside a literal).
    """
    stack = []
    while position < len(text):
        char = text[position]
        if char in "'\"":
            position = _string_end(text, position)
            continue
        if char in _PAIRS:
            stack.append(_PAIRS[char])
        elif char in ")]}":
            if not stack:
                return position
            if stack.pop() != char:
                raise ValueError("LFM tool-call argument has mismatched brackets")
        elif char == "," and not stack:
            return position
        position += 1
    return -1


def _convert(node: ast.AST, depth: int = 0) -> Any:
    if depth > _MAX_DEPTH:
        raise ValueError("LFM tool-call argument nests too deeply")
    if isinstance(node, ast.Constant):
        value = node.value
        if value is None or type(value) in (str, int, float, bool):
            return value
        raise ValueError(f"LFM tool-call argument has an unsupported {type(value).__name__} literal")
    if isinstance(node, ast.Name):
        if node.id in _JSON_NAMES:
            return _JSON_NAMES[node.id]
        raise ValueError(f"LFM tool-call argument names a variable ({node.id!r}), not a literal")
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.USub, ast.UAdd))
        and isinstance(node.operand, ast.Constant)
        and type(node.operand.value) in (int, float)
    ):
        value = node.operand.value
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_convert(item, depth + 1) for item in node.elts]
    if isinstance(node, ast.Dict):
        result = {}
        for key, value in zip(node.keys, node.values):
            if key is None:
                raise ValueError("LFM tool-call argument unpacks a mapping")
            name = _convert(key, depth + 1)
            if not isinstance(name, str):
                raise ValueError("LFM tool-call argument mapping keys must be strings")  # noqa: TRY004
            result[name] = _convert(value, depth + 1)
        return result
    raise ValueError(
        f"LFM tool-call argument is not a literal ({type(node).__name__})"
    )


def literal_value(source: str) -> Any:
    """Decode one argument value as the template wrote it."""
    text = source.strip()
    if not text:
        raise ValueError("LFM tool-call argument has no value")
    if text[0] in "[{":
        try:
            value = json.loads(text)
        except (ValueError, RecursionError):
            pass
        else:
            if json_native(value):
                return _refuse_nul(value)
    try:
        with warnings.catch_warnings():
            # An unknown escape (``'\\p'``) stays in the string, as Python
            # reads it; the SyntaxWarning it raises is not a parse failure.
            warnings.simplefilter("ignore")
            node = ast.parse(text, mode="eval").body
    except SyntaxError as exc:
        raise ValueError(f"LFM tool-call argument is not a literal: {exc.msg}") from None
    except (ValueError, RecursionError, MemoryError):
        raise ValueError("LFM tool-call argument is not a literal") from None
    value = _convert(node)
    if not json_native(value):
        raise ValueError("LFM tool-call argument cannot be carried as JSON")
    return _refuse_nul(value)


def _refuse_nul(value):
    # The wire grammar never writes NUL; an escape (Python ``\\x00``/``\\0``
    # or JSON ``\\u0000``) must not smuggle one past it.
    if _has_nul(value):
        raise ValueError("LFM tool-call argument contains a NUL character")
    return value


def _has_nul(value) -> bool:
    if isinstance(value, str):
        return "\x00" in value
    if isinstance(value, dict):
        return any(_has_nul(k) or _has_nul(v) for k, v in value.items())
    if isinstance(value, list):
        return any(_has_nul(item) for item in value)
    return False


def _parse_arguments(
    text: str, position: int, *, repeats_keep_last: bool = False
) -> tuple[dict, int]:
    """Arguments from just after ``(``; returns them and the index after ``)``."""
    arguments = {}
    while True:
        position = _skip_ws(text, position)
        if position == len(text):
            raise ValueError("LFM tool call has an unclosed argument list")
        if text[position] == ")":
            return arguments, position + 1
        key = _NAME.match(text, position)
        after = _skip_ws(text, key.end()) if key else position
        if key is None or not text.startswith("=", after):
            raise ValueError(
                "LFM tool calls take keyword arguments only (name=value)"
            )
        start = after + 1
        end = _value_end(text, start)
        if end < 0:
            raise ValueError("LFM tool call has an unclosed argument value")
        if text[end] not in ",)":
            raise ValueError("LFM tool-call argument has mismatched brackets")
        name = key.group(0)
        if name in arguments and not repeats_keep_last:
            raise ValueError(f"LFM tool call repeats argument {name!r}")
        arguments[name] = literal_value(text[start:end])
        position = end + 1 if text[end] == "," else end


def parse_tool_call(text: str, tools: Any | None = None, *, repeats_keep_last=False):
    """Every call of one tool-call block body, in order.

    ``text`` is what lies between the markers: a bracketed list of calls (the
    template's form) or, leniently, the calls without the brackets.  Names
    are not checked against ``tools`` here; the output parser rejects an
    undeclared tool, as for every other wire.  A repeated argument name is
    malformed, except under ``repeats_keep_last``: the forced-call grammar's
    free-argument tail cannot express uniqueness, so a grammar-forced call
    keeps the last value as a JSON object would.
    """
    del tools  # names are validated by the output parser
    position = _skip_ws(text, 0)
    bracketed = text.startswith("[", position)
    if bracketed:
        position += 1
    calls = []
    while True:
        position = _skip_ws(text, position)
        if bracketed and text.startswith("]", position):
            position += 1
            break
        if position == len(text):
            if bracketed:
                raise ValueError("LFM tool-call list is not closed")
            break
        name = _NAME.match(text, position)
        if name is None:
            raise ValueError("LFM tool call names no function")
        position = _skip_ws(text, name.end())
        if not text.startswith("(", position):
            raise ValueError("LFM tool-call name must be followed by '('")
        arguments, position = _parse_arguments(
            text, position + 1, repeats_keep_last=repeats_keep_last
        )
        calls.append({"name": name.group(0), "arguments": arguments})
        position = _skip_ws(text, position)
        if text.startswith(",", position):
            position += 1
            continue
        if bracketed and text.startswith("]", position):
            position += 1
            break
        if not bracketed and position == len(text):
            break
        raise ValueError("LFM tool calls must be separated by ','")
    if text[position:].strip():
        raise ValueError("text after the LFM tool-call list")
    if not calls:
        raise ValueError("No function provided.")
    return calls[0] if len(calls) == 1 else calls


def partial_function_names(partial: str, tools: Any | None = None) -> list[str]:
    """The functions a cut-short tool-call body had committed to, in order.

    The reader behind the output parser's truncation records (see
    :func:`mlx2.output.stop_cut_record`), by :func:`parse_tool_call`'s rules:
    a name is committed once the ``(`` after it has arrived, and the walk
    goes on through its arguments (string literals and brackets skipped) to
    the next call.  A cut inside a name, inside a value, or anywhere the
    parser itself would reject past a committed name ends the walk with the
    names so far.  Raises ``ValueError`` when a call position holds no
    readable name (``[(``, ``[get weather(``), which the parser rejects.
    """
    del tools
    names = []
    position = _skip_ws(partial, 0)
    bracketed = partial.startswith("[", position)
    if bracketed:
        position += 1
    while True:
        position = _skip_ws(partial, position)
        if position == len(partial) or (bracketed and partial[position] == "]"):
            return names
        name = _NAME.match(partial, position)
        if name is None:
            raise ValueError("LFM tool call names no function")
        position = _skip_ws(partial, name.end())
        if position == len(partial):
            return names  # the name may still continue
        if partial[position] != "(":
            raise ValueError("LFM tool-call name is not followed by '('")
        names.append(name.group(0))
        position += 1
        # Walk the arguments to the ``)`` closing this call.
        while True:
            position = _skip_ws(partial, position)
            if position == len(partial):
                return names
            if partial[position] == ")":
                position += 1
                break
            try:
                end = _value_end(partial, position)
            except ValueError:
                return names  # cut inside a literal, or malformed past the name
            if end < 0 or partial[end] not in ",)":
                return names
            position = end + 1 if partial[end] == "," else end
        position = _skip_ws(partial, position)
        if position == len(partial) or not partial.startswith(",", position):
            return names
        position += 1


parse_tool_call.partial_function_names = partial_function_names


def parse_forced_tool_call(text: str, tools: Any | None = None):
    """:func:`parse_tool_call` for a grammar-forced call (repeats keep last)."""
    return parse_tool_call(text, tools, repeats_keep_last=True)


parse_forced_tool_call.partial_function_names = partial_function_names


# ---------------------------------------------------------------------------
# Constrained decoding: the forced (``required`` / named) tool-call block.

# A string argument as the template writes it: single-quoted, with the
# template's escapes (and the other one-character escapes Python reads the
# same way).  No raw newline (a syntax error in a one-quote literal) and no
# NUL (``ast`` refuses source holding one).
_PY_STRING = r"'(?:[^'\\\x00\n\r]|\\[\\'\"nrt])*'"
_FREE_KEY = r"[A-Za-z_][A-Za-z0-9_]{0,63}"
_SEPARATOR = ", "
_MAX_FREE_ARGUMENTS = 64


def _python_string(value: str) -> str:
    if "\x00" in value:
        raise ValueError("a NUL character is not representable in an LFM tool call")
    return "'" + (
        value.replace("\\", "\\\\").replace("'", "\\'")
        .replace("\n", "\\n").replace("\r", "\\r")
    ) + "'"


def _literal_pattern(value: Any) -> str:
    """Regex for the template's spelling of one enum or const value."""
    import regex

    from ...structured_output import json_literal_pattern

    if isinstance(value, str):
        return regex.escape(_python_string(value))
    if value is True or value is False or value is None:
        return str(value)
    if isinstance(value, (int, float)):
        if not json_native(value):
            raise ValueError("tool schema literal is not a finite JSON value")
        return regex.escape(str(value))
    return json_literal_pattern(value)  # rendered with ``tojson``


def _generic_value() -> str:
    from ...structured_output import _FINITE_NUMBER

    return (
        rf"(?:{_PY_STRING}|{_FINITE_NUMBER}|True|False|None|(?&object)|(?&array))"
    )


def _declares_null(schema: Any) -> bool:
    if not isinstance(schema, dict):
        return False
    kind = schema.get("type")
    if kind == "null" or isinstance(kind, list) and "null" in kind:
        return True
    return any(
        isinstance(branch, dict) and _declares_null(branch)
        for key in ("anyOf", "oneOf")
        for branch in (schema.get(key) or ())
    )


def _loose_value(schema: Any) -> str:
    """A non-strict argument: narrowed by its declared type, else free.

    The parser converts nothing by schema, so a non-strict value is any
    literal; the grammar narrows it to the declared type (and ``None`` when
    the schema declares null) so forced decoding writes what the schema asks
    for, as the template would render it.
    """
    from ...structured_output import _FINITE_NUMBER, _INTEGER
    from ._schema import infer_type_from_json_schema

    declared = infer_type_from_json_schema(schema)
    pattern = {
        "string": _PY_STRING,
        "integer": _INTEGER,
        "number": _FINITE_NUMBER,
        "boolean": "(?:True|False)",
        "array": "(?&array)",
        "object": "(?&object)",
    }.get(declared.strip().lower() if isinstance(declared, str) else None)
    if pattern is None:
        return _generic_value()
    return f"(?:{pattern}|None)" if _declares_null(schema) else pattern


def _strict_value(schema: Any) -> str:
    """A strict argument's language: the schema, in the template's spellings."""
    from ...structured_output import (
        _FINITE_NUMBER,
        _INTEGER,
        _require_schema_keys,
        _schema_pattern,
    )
    from ._schema import string_length_bounds

    if not isinstance(schema, dict):
        raise ValueError("strict tool parameter schemas must be objects")  # noqa: TRY004
    if not schema:
        return _generic_value()
    if {"anyOf", "oneOf", "allOf"} & set(schema):
        raise ValueError(
            "strict tool schema unions are unsupported by the post-generation validator"
        )
    if "enum" in schema:
        _require_schema_keys(schema, {"enum", "type"})
        values = schema["enum"]
        if not isinstance(values, list) or not 1 <= len(values) <= 256:
            raise ValueError("JSON schema enum must contain 1 to 256 values")
        return "(?:" + "|".join(_literal_pattern(value) for value in values) + ")"
    if "const" in schema:
        _require_schema_keys(schema, {"const", "type"})
        return _literal_pattern(schema["const"])
    kind = schema.get("type")
    if isinstance(kind, list):
        _require_schema_keys(schema, {"type"})
        if not kind or len(kind) > 8:
            raise ValueError("JSON schema type union is invalid")
        return "(?:" + "|".join(_strict_value({"type": item}) for item in kind) + ")"
    if kind == "string":
        _require_schema_keys(schema, {"type", "minLength", "maxLength"})
        if set(schema) == {"type"}:
            return _PY_STRING
        minimum, maximum = string_length_bounds(schema)
        # Unescaped characters only, so the written length is the length.
        return rf"'[^'\\\x00-\x1f]{{{minimum},{maximum}}}'"
    if kind in ("integer", "number", "boolean", "null"):
        _require_schema_keys(schema, {"type"})
        return {
            "integer": _INTEGER,
            "number": _FINITE_NUMBER,
            "boolean": "(?:True|False)",
            "null": "None",
        }[kind]
    if kind in ("array", "object") or "properties" in schema:
        # The template renders containers with ``tojson``.
        return _schema_pattern(schema, finite_numbers=True)
    raise ValueError(f"unsupported JSON schema type {kind!r}")


def _argument_list(parts: list[tuple[str, bool]]) -> str:
    """``name=value`` parts joined by ``, ``: required ones always, in order,
    then each optional one at most once, in declared order."""
    required = [part for part, needed in parts if needed]
    optional = [part for part, needed in parts if not needed]
    if required:
        return _SEPARATOR.join(required) + "".join(
            f"(?:{_SEPARATOR}{part})?" for part in optional
        )
    chains = [
        part + "".join(f"(?:{_SEPARATOR}{later})?" for later in optional[index + 1:])
        for index, part in enumerate(optional)
    ]
    return "(?:" + "|".join(chains) + ")?" if chains else ""


def _arguments_body(function: dict) -> str:
    import regex

    from ._schema import executable_schema, required_parameter_names, resolve_local_refs

    strict = bool(function.get("strict", False))
    parameters = function.get("parameters")
    if strict:
        schema = executable_schema(parameters if isinstance(parameters, dict) else {})
        if schema.get("type") not in (None, "object"):
            raise ValueError("strict function parameters must use an object schema")
        if schema.get("additionalProperties", False) not in (False, None):
            raise ValueError("strict function parameters require additionalProperties:false")
        required = schema.get("required", [])
        if not isinstance(required, list):
            raise ValueError("strict function parameters require a required list")
    else:
        schema = parameters if isinstance(parameters, dict) else {}
        try:
            schema = resolve_local_refs(schema)
        except ValueError:
            pass  # best-effort, as for non-strict parsing
        required = required_parameter_names(function)
    properties = schema.get("properties")
    properties = properties if isinstance(properties, dict) else {}
    if strict and set(required) - set(properties):
        raise ValueError("strict required parameters must be declared properties")
    for name in required:
        if not representable_name(name):
            raise ValueError("tool parameter names are not representable in LFM calls")
    ordered = [name for name in properties if name in required] + [
        name for name in required if name not in properties
    ]
    optional = []
    for name in properties:
        if name in required:
            continue
        if representable_name(name):
            optional.append(name)
        elif strict:
            raise ValueError("tool parameter names are not representable in LFM calls")
        # A non-strict optional name the wire cannot spell is never emitted.
    value = _strict_value if strict else _loose_value
    parts = [
        (f"{regex.escape(name)}={value(properties.get(name, {}))}", True)
        for name in ordered
    ] + [
        (f"{regex.escape(name)}={value(properties[name])}", False)
        for name in optional
    ]
    body = _argument_list(parts)
    if not strict and not properties and schema.get("additionalProperties") is not False:
        free = f"{_FREE_KEY}={_generic_value()}"
        room = max(0, _MAX_FREE_ARGUMENTS - len(required))
        if required:
            body += f"(?:{_SEPARATOR}{free}){{0,{room}}}"
        elif room:
            body = f"(?:{free}(?:{_SEPARATOR}{free}){{0,{room - 1}}})?"
    return body


def constrained_tool_grammar(tools, tool_choice, *, parallel_tool_calls=True):
    """Regex for one forced LFM tool-call block (``required`` or named).

    The block is ``<|tool_call_start|>[call, ...]<|tool_call_end|>`` with
    exactly one call when ``parallel_tool_calls`` is false.  Each call is a
    declared function name and its keyword arguments, required ones first,
    then optional ones at most once, all in declared order, spelled as the
    template renders them (``, `` separators, single-quoted strings,
    ``True``/``False``/``None``, ``tojson`` containers).  Strict functions
    constrain every value by its schema (unions refused, as the strict
    post-generation validator cannot check them); non-strict values are
    narrowed by declared type and otherwise free literals.  Raises
    ``ValueError`` when a schema or name is not representable; the planner
    then reports ``skipped_grammar_unrepresentable`` and the request is
    served without the grammar (or refused, for a forced call).
    """
    import regex

    from ...structured_output import (
        _MAX_SCHEMA_PATTERN_CHARS,
        recursive_json_object_pattern,
    )

    functions = [tool["function"] for tool in tools]
    if isinstance(tool_choice, dict):
        selected = tool_choice["function"]["name"]
        functions = [function for function in functions if function["name"] == selected]
    if not functions:
        raise ValueError("constrained tool_choice selects no declared function")
    calls = []
    for function in functions:
        name = function["name"]
        if not representable_name(name):
            raise ValueError("tool name is not representable in LFM calls")
        calls.append(rf"{regex.escape(name)}\({_arguments_body(function)}\)")
    call = "(?:" + "|".join(calls) + ")"
    listing = call if parallel_tool_calls is False else rf"{call}(?:{_SEPARATOR}{call})*"
    pattern = (
        regex.escape(tool_call_start) + rf"\[{listing}\]" + regex.escape(tool_call_end)
    )
    if len(pattern) > _MAX_SCHEMA_PATTERN_CHARS:
        raise ValueError("LFM tool grammar is too large")
    if "(?&" in pattern:
        pattern += recursive_json_object_pattern(finite_numbers=True)[1]
    return pattern

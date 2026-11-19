# SPDX-License-Identifier: Apache-2.0
# Original mlx2 code; behaviour follows vLLM's hy_v4 tool parser (PR 68296,
# read not copied) and its fixes vllm#65851 / vllm#55550.  See
# provenance/hy-v4-flash.json.
"""Tencent Hy4 tool calls.

Wire format (``chat_template.jinja`` of tencent/Hy4-flash; every structural
token carries the checkpoint suffix, ``:opensource`` there)::

    <tool_calls:S><tool_call:S>NAME<arg_key:S>KEY</arg_key:S><arg_value:S>VALUE</arg_value:S>...</tool_call:S></tool_calls:S>

The template writes a string argument raw and every other value with
``tojson``.  Values are typed by the request's JSON Schema in vLLM's order
(null, boolean, integer, number, JSON for other declared types, string, then
JSON / Python literal / raw text).  The function name is stripped; argument
names keep their whitespace (vllm#65851).  A call that names no function is
malformed (vLLM drops it in streaming, vllm#55550; mlx2's output channel
fails it closed or, with tolerant markers, serves it as text).
"""

from __future__ import annotations

import ast
import copy
import json
import math
import re
from urllib.parse import unquote

from ._schema import json_native

DEFAULT_SUFFIX = ":opensource"
_STRUCTURAL = re.compile(r"<(?:think|tool_calls|tool_call|arg_key|arg_value)(:[^\s>]+)?>")


_TEMPLATE_SUFFIX = re.compile(r"""set\s+HYTK\s*=\s*(?:'([^']*)'|"([^"]*)")""")


def detect_suffix(vocab, chat_template: str | None = None) -> str:
    """The structural-token suffix: from the tokenizer vocab (vLLM
    ``detect_token_suffix``), else the chat template's ``HYTK``, else none."""
    for token in vocab:
        match = _STRUCTURAL.fullmatch(token)
        if match:
            return match.group(1) or ""
    if chat_template:
        match = _TEMPLATE_SUFFIX.search(chat_template)
        if match:
            return match.group(1) if match.group(1) is not None else match.group(2)
    return ""


def markers(suffix: str = DEFAULT_SUFFIX) -> dict:
    return {
        "calls_open": f"<tool_calls{suffix}>", "calls_close": f"</tool_calls{suffix}>",
        "call_open": f"<tool_call{suffix}>", "call_close": f"</tool_call{suffix}>",
        "key_open": f"<arg_key{suffix}>", "key_close": f"</arg_key{suffix}>",
        "value_open": f"<arg_value{suffix}>", "value_close": f"</arg_value{suffix}>",
    }


_TYPE_ALIASES = {"str": "string", "text": "string", "varchar": "string", "char": "string",
                 "enum": "string", "bool": "boolean", "binary": "boolean", "int": "integer",
                 "float": "number", "double": "number", "list": "array", "dict": "object",
                 "map": "object"}


_SCHEMA_TYPE_KEYS = ("type", "anyOf", "oneOf")
_MAX_REF_HOPS = 3
_MAX_ANCHOR_DEPTH = 8


def _normalize_type(kind) -> str:
    """vLLM ``_normalize_type``: exact alias, then int/num prefixes, else as given."""
    kind = str(kind)
    exact = _TYPE_ALIASES.get(kind)
    if exact is not None:
        return exact
    lower = kind.lower()
    if lower.startswith(("int", "uint", "long", "short", "unsigned")):
        return "integer"
    if lower.startswith(("num", "float")):
        return "number"
    return kind


def _is_ref_type(schema) -> bool:
    return (isinstance(schema, dict) and isinstance(schema.get("$ref"), str)
            and not any(schema.get(key) for key in _SCHEMA_TYPE_KEYS))


def _json_pointer(root, pointer):
    if pointer == "":
        return root
    if not pointer.startswith("/"):
        return None
    target = root
    for raw in pointer[1:].split("/"):
        token = unquote(raw).replace("~1", "/").replace("~0", "~")
        if isinstance(target, dict):
            if token not in target:
                return None
            target = target[token]
        elif isinstance(target, list):
            if not token.isdigit() or int(token) >= len(target):
                return None
            target = target[int(token)]
        else:
            return None
    return target


def _anchor(root, anchor):
    stack, visited = [(root, 0)], set()
    while stack:
        node, depth = stack.pop()
        if depth > _MAX_ANCHOR_DEPTH or id(node) in visited:
            continue
        visited.add(id(node))
        if isinstance(node, dict):
            if node.get("$anchor") == anchor:
                return node
            children = list(node.values())
        elif isinstance(node, list):
            children = node
        else:
            continue
        stack.extend((child, depth + 1) for child in children)
    return None


def _local_ref(root, ref):
    """Same-document pointer, root reference or anchor; never fetches."""
    base, separator, fragment = ref.partition("#")
    if not separator or base:
        return None
    target = (_json_pointer(root, fragment) if fragment == "" or fragment.startswith("/")
              else _anchor(root, unquote(fragment)))
    return target if isinstance(target, dict) else None


def _resolve_ref(schema, root, seen=frozenset(), hops=0):
    """vLLM ``_resolve_ref_type``: bounded (3 hops), cycle-safe, local only."""
    ref = schema.get("$ref")
    if not isinstance(ref, str) or ref in seen or hops >= _MAX_REF_HOPS:
        return {}
    target = _local_ref(root, ref)
    if target is None:
        return {}
    typed = {key: copy.deepcopy(target[key]) for key in _SCHEMA_TYPE_KEYS if target.get(key)}
    if typed:
        return _resolve_unions(typed, root, seen | {ref}, hops + 1) or typed
    if _is_ref_type(target):
        return _resolve_ref(target, root, seen | {ref}, hops + 1)
    return {}


def _resolve_unions(schema, root, seen=frozenset(), hops=0):
    resolved = None
    for key in ("anyOf", "oneOf"):
        branches = schema.get(key)
        if not isinstance(branches, list) or not any(_is_ref_type(b) for b in branches):
            continue
        if resolved is None:
            resolved = copy.deepcopy(schema)
        for index, branch in enumerate(resolved[key]):
            if _is_ref_type(branch):
                value = _resolve_ref(branch, root, seen, hops)
                if value:
                    resolved[key][index] = value
    return resolved


def _arg_schema(tool_name, arg, tools):
    for tool in tools or ():
        function = tool.get("function", tool) if isinstance(tool, dict) else {}
        if not isinstance(function, dict) or function.get("name") != tool_name:
            continue
        params = function.get("parameters") or {}
        props = params.get("properties") if isinstance(params, dict) else None
        schema = props.get(arg) if isinstance(props, dict) else None
        if not isinstance(schema, dict):
            return {}
        try:
            if _is_ref_type(schema):
                return _resolve_ref(schema, params) or schema
            return _resolve_unions(schema, params) or schema
        except (RecursionError, TypeError, ValueError):
            return {}
    return {}


def _options(schema):
    """vLLM ``_get_schema_options``: ``type`` first, then recursive unions,
    else string."""
    if "type" in schema:
        kind = schema["type"]
        return [{"type": k} for k in kind] if isinstance(kind, list) else [schema]
    for key in ("anyOf", "oneOf"):
        if isinstance(schema.get(key), list):
            return [o for branch in schema[key] if isinstance(branch, dict) for o in _options(branch)]
    return [{"type": "string"}]


def _types(schema):
    return {_normalize_type(s.get("type", "string")) for s in _options(schema)} - {"null"}


def _nullable(schema):
    return any(_normalize_type(s.get("type", "string")) == "null" for s in _options(schema))


def _deserialize(value):
    for decode in (json.loads, ast.literal_eval):
        try:
            decoded = decode(value)
        except Exception:  # noqa: BLE001 - any failure (incl. RecursionError) means "not this literal"
            continue
        return decoded if json_native(decoded) else value
    return value


def parse_value(value: str, tool_name: str, arg: str, tools):
    """vLLM ``_parse_value``: schema-typed argument value."""
    schema = _arg_schema(tool_name, arg, tools)
    types = _types(schema)
    if value == "null" and _nullable(schema):
        return None
    if "boolean" in types and value.lower() in ("true", "false"):
        return value.lower() == "true"
    if "integer" in types:
        try:
            return int(value)
        except ValueError:
            pass
    if "number" in types:
        try:
            number = float(value) if any(c in value for c in ".eE") else int(value)
            if math.isfinite(number):
                return number
        except (ValueError, OverflowError):
            pass
    if types - {"string", "boolean", "integer", "number"}:
        try:
            decoded = json.loads(value)
            if json_native(decoded):
                return decoded
        except (ValueError, RecursionError):
            pass
    if "string" in types:
        return value
    return _deserialize(value)


def _scan_arguments(args: str, m: dict):
    """``[(key, value)]`` of a ``<arg_key>K</arg_key><arg_value>V</arg_value>``
    run, in one linear pass (a regex rescans the body from every key on
    degenerate output).  Anything else between pairs is malformed."""
    pairs, position, n = [], 0, len(args)
    ko, kc, vo, vc = m["key_open"], m["key_close"], m["value_open"], m["value_close"]
    while True:
        while position < n and args[position].isspace():
            position += 1
        if position == n:
            return pairs
        if not args.startswith(ko, position):
            raise ValueError("Unparsed Hy4 tool-call argument payload")
        key_start = position + len(ko)
        key_end = args.find(kc, key_start)
        if key_end < 0:
            raise ValueError("Unclosed Hy4 tool-call argument name")
        key = args[key_start:key_end]
        if ko in key or vo in key:
            raise ValueError("Malformed Hy4 tool-call argument name")
        position = key_end + len(kc)
        while position < n and args[position].isspace():
            position += 1
        if not args.startswith(vo, position):
            raise ValueError("Hy4 tool-call argument has no value")
        value_start = position + len(vo)
        value_end = args.find(vc, value_start)
        if value_end < 0:
            raise ValueError("Unclosed Hy4 tool-call argument value")
        value = args[value_start:value_end]
        if ko in value or vo in value:
            raise ValueError("Malformed Hy4 tool-call argument value")
        pairs.append((key, value))
        position = value_end + len(vc)


def make_parser(suffix: str = DEFAULT_SUFFIX):
    """``parse_tool_call(body, tools)`` for one ``<tool_call:S>`` body."""
    m = markers(suffix)

    def _name(text):
        if not text:
            raise ValueError("Hy4 tool call names no function")
        if any(c in text for c in "<>"):
            raise ValueError("Hy4 tool call name holds markup")
        return text

    def parse_tool_call(text, tools=None):
        body = text
        if m["call_open"] in body or m["call_close"] in body:
            raise ValueError("Hy4 tool call nests another call marker")
        start = body.find(m["key_open"])
        name = _name((body if start < 0 else body[:start]).strip())
        arguments = {}
        if start >= 0:
            for key, value in _scan_arguments(body[start:], m):
                # Argument names keep their whitespace (vllm#65851).
                arguments[key] = parse_value(value, name, key, tools)
        return {"name": name, "arguments": arguments}

    def partial_function_names(partial, tools=None):
        """The function a cut-short body had committed to (name ends at the
        nearer of ``<arg_key>`` and the end of the call, vllm#55550); the same
        name rules as the full parser."""
        body = partial.lstrip().removeprefix(m["call_open"])
        position = body.find(m["key_open"])
        if position < 0:
            return []
        return [_name(body[:position].strip())]

    parse_tool_call.partial_function_names = partial_function_names
    parse_tool_call.markers = m
    return parse_tool_call


parse_tool_call = make_parser(DEFAULT_SUFFIX)


# --- Forced-call grammar (tool_choice "required" / a named function) -------
#
# Original mlx2 code.  Shape follows vLLM's ``get_hy_v4_structural_tag``
# (PR 68296, read not copied): ``<tool_calls:S>`` then back-to-back
# ``<tool_call:S>NAME`` blocks, required argument keys first, then optional
# ones, then ``</tool_calls:S>``.  mlx2 narrows two things vLLM leaves free:
# scalar-typed values (integer, number, boolean, null, enums) are spelled the
# way the template's ``tojson`` writes them and ``parse_value`` reads them
# back, and every optional key appears at most once, in declared order.
# Strings and JSON-typed values are raw text holding no structural marker
# (the parser cuts a value at the first ``</arg_value:S>``, and the output
# channel cuts a call at the first ``</tool_call:S>``).  A ``strict`` function
# whose schema needs more than that (nested JSON, string bounds) is refused,
# not served loosely.

_STRICT_UNSUPPORTED = frozenset({
    "minLength", "maxLength", "pattern", "format", "minimum", "maximum",
    "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "const", "allOf", "not",
})


def _marker_free(m):
    """Raw text that spells no structural marker.

    Every suffixed marker ends with ``S>`` (``:opensource>``), so excluding
    that one literal excludes them all with an automaton a few states wide;
    the exclusion of each marker spelled out overran the token automaton's
    state cap at a dozen arguments.  It also keeps ``S>`` itself out of a
    value, which ordinary text does not contain.  An unsuffixed wire excludes
    the six markers that break a key, a value or a call.
    """
    from ...tool_grammar import text_excluding

    tail = m["call_open"][len("<tool_call"):]
    if tail != ">":
        return text_excluding(tail)
    return text_excluding(m["key_open"], m["key_close"], m["value_open"], m["value_close"],
                          m["call_open"], m["call_close"])


def _holds_marker(text, m):
    return any(marker in text for marker in m.values())


def _enum_pattern(schema, types, nullable, m):
    """The enum's literals as the template writes them, or None when the
    parser would not read them back as the same values."""
    values = schema.get("enum")
    if not isinstance(values, list) or not values:
        return None
    if None in values:
        if not nullable:
            return None
        values = [v for v in values if v is not None]
        if not values:
            return "null"
    if types == {"string"} and all(isinstance(v, str) and not _holds_marker(v, m) for v in values):
        spelled = [re.escape(v) for v in values]
    elif types == {"boolean"} and all(isinstance(v, bool) for v in values):
        spelled = [json.dumps(v) for v in values]
    elif types == {"integer"} and all(type(v) is int for v in values):
        spelled = [re.escape(json.dumps(v)) for v in values]
    elif "number" in types and types <= {"integer", "number"} and all(
            type(v) in (int, float) and math.isfinite(v) for v in values):
        spelled = [re.escape(json.dumps(v)) for v in values]
    else:
        return None
    if nullable:
        spelled.append("null")
    return "(?:" + "|".join(dict.fromkeys(spelled)) + ")"


def _strict_nodes(schema, root, hops=0, seen=frozenset()):
    """Every subschema of a strict property: union branches and local
    ``$ref`` targets included, since the parser's resolution keeps only their
    types and drops the constraints a strict grammar would have to enforce."""
    if not isinstance(schema, dict):
        return
    yield schema
    ref = schema.get("$ref")
    if isinstance(ref, str):
        target = _local_ref(root, ref)
        if target is None or ref in seen or hops >= _MAX_REF_HOPS:
            raise ValueError("strict Hy4 tool $ref is not resolvable for the call grammar")
        yield from _strict_nodes(target, root, hops + 1, seen | {ref})
    for key in ("anyOf", "oneOf"):
        for branch in schema.get(key) or ():
            yield from _strict_nodes(branch, root, hops, seen)


def check_strict_property(schema, root):
    """Refuse a strict property whose constraints the grammar cannot enforce
    anywhere in it (a nested ``enum``, a bound inside a union branch)."""
    for depth, node in enumerate(_strict_nodes(schema, root)):
        if _STRICT_UNSUPPORTED & set(node) or (depth and "enum" in node):
            raise ValueError("strict Hy4 tool schema keyword is not representable in the call grammar")


def value_pattern(schema, *, strict=False, suffix: str = DEFAULT_SUFFIX):
    """Regex of one ``<arg_value:S>`` body that ``parse_value`` reads back as
    a value of ``schema`` (already resolved as the parser resolves it)."""
    from ...structured_output import _FINITE_NUMBER, _INTEGER

    m = markers(suffix)
    schema = schema if isinstance(schema, dict) else {}
    types = {t for t in _types(schema)}
    nullable = _nullable(schema)
    if strict and _STRICT_UNSUPPORTED & set(schema):
        raise ValueError("strict Hy4 tool schema keyword is not representable in the call grammar")
    if not types:
        return "null" if nullable else _marker_free(m)
    enum = _enum_pattern(schema, types, nullable, m)
    if enum is not None:
        return enum
    if "enum" in schema and strict:
        raise ValueError("strict Hy4 tool enum is not representable in the call grammar")
    if types == {"string"}:
        return _marker_free(m)
    scalar = None
    if types == {"integer"}:
        scalar = _INTEGER
    elif types <= {"integer", "number"}:
        scalar = _FINITE_NUMBER
    elif types == {"boolean"}:
        scalar = "(?:true|false)"
    if scalar is not None:
        return f"(?:{scalar}|null)" if nullable else scalar
    if strict:
        raise ValueError("strict Hy4 tool value type needs a JSON grammar the call wire cannot bound")
    # Mixed or JSON-typed: the parser decodes JSON and keeps any other text.
    return _marker_free(m)


def _selected_functions(tools, tool_choice):
    functions = [tool["function"] for tool in tools or ()
                 if isinstance(tool, dict) and isinstance(tool.get("function"), dict)]
    if isinstance(tool_choice, dict):
        name = (tool_choice.get("function") or {}).get("name")
        functions = [f for f in functions if f.get("name") == name]
    elif tool_choice != "required":
        raise ValueError("Hy4 forced-call grammar needs tool_choice 'required' or a named function")
    if not functions:
        raise ValueError("constrained tool_choice selects no declared function")
    return functions


def _call_pattern(function, tools, m, suffix):
    from ._schema import required_parameter_names

    name = function.get("name")
    if not isinstance(name, str) or not name or name != name.strip() or any(c in name for c in "<>"):
        raise ValueError("tool name is not representable on the Hy4 call wire")
    strict = function.get("strict", False) is True
    params = function.get("parameters")
    params = params if isinstance(params, dict) else {}
    properties = params.get("properties")
    properties = properties if isinstance(properties, dict) else {}
    required = required_parameter_names(function)

    def pair(key):
        if not isinstance(key, str) or _holds_marker(key, m):
            raise ValueError("tool parameter name is not representable on the Hy4 call wire")
        if strict:
            check_strict_property(properties.get(key), params)
        value = value_pattern(_arg_schema(name, key, tools), strict=strict, suffix=suffix)
        return (re.escape(m["key_open"]) + re.escape(key) + re.escape(m["key_close"])
                + re.escape(m["value_open"]) + value + re.escape(m["value_close"]))

    body = "".join(pair(key) for key in required)
    optional = [key for key in properties if key not in required]
    body += "".join(f"(?:{pair(key)})?" for key in optional)
    if not properties and not strict and params.get("additionalProperties") is not False:
        free = _marker_free(m)
        body += (f"(?:{re.escape(m['key_open'])}{free}{re.escape(m['key_close'])}"
                 f"{re.escape(m['value_open'])}{free}{re.escape(m['value_close'])})"
                 "*")
    return re.escape(m["call_open"]) + re.escape(name) + body + re.escape(m["call_close"])


def constrained_tool_grammar(tools, tool_choice, *, parallel_tool_calls=True,
                             suffix: str = DEFAULT_SUFFIX):
    """Regex of a whole forced Hy4 call block.

    ``required`` admits any declared function, a named choice that one; one
    call when ``parallel_tool_calls`` is false, one or more otherwise.  The
    block is the whole answer: the template writes ``</tool_calls:S>`` then
    the end-of-turn token, with nothing between calls.
    """
    m = markers(suffix)
    calls = [_call_pattern(f, tools, m, suffix) for f in _selected_functions(tools, tool_choice)]
    call = "(?:" + "|".join(calls) + ")"
    sequence = call if parallel_tool_calls is False else f"{call}+"
    return re.escape(m["calls_open"]) + sequence + re.escape(m["calls_close"])

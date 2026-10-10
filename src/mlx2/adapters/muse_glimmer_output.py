"""Incremental Muse recipient channels and ATEM tool calls (CPU only)."""

import json
import re
import uuid

from ..output import (
    StopSequenceMatcher,
    _safe_prefix,
    truncated_tool_record,
    within_parallel_bound,
)
from ..runtime.tool_parsers._schema import (
    executable_schema,
    raw_string_pattern,
    required_parameter_names,
    schema_value_matches,
)

_MESSAGE = "<|message|>"
_START = "<|start|>assistant"
_TOOL_OPEN = "<atem:function_calls>"
_TOOL_CLOSE = "</atem:function_calls>"
_HEADER = re.compile(r"(?:<\|start\|>assistant)?(?:\s*to=([\w.-]+))?\s*", re.ASCII)
_INVOKE = re.compile(r'<atem:invoke name="([\w.-]+)">')
_INVOKE_CLOSE = "</atem:invoke>"
_PARAMETER = re.compile(r'<atem:parameter name="([\w.-]+)">')
_PARAMETER_CLOSE = "</atem:parameter>"
# What a raw ATEM value cannot contain: ``parse_atem`` ends it at the first
# parameter closer and rejects an invoke closer inside it, and the output
# parser ends the whole block at the first tool closer outside JSON strings
# (a raw value cannot quote one).  Any other text, ``<`` included (code,
# markup, shell redirects), is the value's own.
_RAW_CLOSERS = (_PARAMETER_CLOSE, _INVOKE_CLOSE, _TOOL_CLOSE)
# The raw-value language, defined once per grammar (``_RAW_TEXT_DEFINITION``)
# and referenced per parameter: spelled out it is about a kilobyte.
_RAW_TEXT = "(?&atem_text)"
_JSON_STRING = re.compile(r'"[^"\\]*(?:\\.[^"\\]*)*"', re.DOTALL)


class _UnclosedJSONString(ValueError):
    """The text ends inside a string of an ATEM value decoded as JSON."""


def _json_value_end(text, start):
    """Where the ``</atem:parameter>`` closing a JSON value at ``start`` is.

    Muse writes JSON values with raw ``<`` (the template's ``tojson`` does not
    escape it), and a JSON string may even spell a closing tag.  Outside a
    string ``<`` is not JSON, so the value ends at the first closer outside
    every string.  Returns -1 when no closer follows the value.
    """
    position = start
    while True:
        close = text.find(_PARAMETER_CLOSE, position)
        quote = text.find('"', position, close if close >= 0 else len(text))
        if quote < 0:
            return close
        string = _JSON_STRING.match(text, quote)
        if string is None:
            raise _UnclosedJSONString("ATEM value ends inside a JSON string")
        position = string.end()


def _partial_header(text):
    text = text.lstrip()
    if _START.startswith(text):
        return True
    if text.startswith(_START):
        text = text[len(_START) :].lstrip()
    if _MESSAGE.startswith(text) or "to=".startswith(text):
        return True
    if text.startswith("to="):
        recipient, separator, marker = text[3:].partition("<")
        return bool(re.fullmatch(r"[\w.-]*", recipient, re.ASCII)) and (
            not separator or _MESSAGE.startswith(separator + marker)
        )
    return False


def parse_atem(text: str, tools: list[dict]) -> list[dict]:
    """ATEM values are raw text, with JSON for non-string schema types.

    A raw value ends at the first ``</atem:parameter>`` and may not contain
    ``</atem:invoke>``; a JSON value ends at the first closer outside its
    strings, so a quoted tag does not cut it short.
    """
    definitions = {tool["function"]["name"]: tool["function"] for tool in tools}
    calls, cursor = [], 0
    while (match := _INVOKE.search(text, cursor)) is not None:
        if text[cursor : match.start()].strip():
            raise ValueError("Malformed ATEM tool call")
        name = match.group(1)
        if name not in definitions:
            raise ValueError("Model called an undeclared tool")
        function = definitions[name]
        schema = function.get("parameters", {})
        if function.get("strict", False):
            schema = executable_schema(schema)
        properties, arguments, end = schema.get("properties", {}), {}, match.end()
        if not isinstance(properties, dict):
            properties = {}
        while (parameter := _PARAMETER.search(text, end)) is not None:
            closing = text.find(_INVOKE_CLOSE, end)
            if 0 <= closing < parameter.start():
                break
            if text[end : parameter.start()].strip():
                raise ValueError("Malformed ATEM parameter")
            key = parameter.group(1)
            if key in arguments:
                raise ValueError("Duplicate ATEM parameter")
            if key not in properties and schema.get("additionalProperties") is False:
                raise ValueError("Undeclared ATEM parameter")
            parameter_schema = properties.get(key, {})
            # Boolean subschemas are valid JSON Schema: ``true`` admits any
            # value (an untyped parameter), ``false`` admits none.
            if parameter_schema is False:
                raise ValueError("ATEM parameter is forbidden by its schema")
            if not isinstance(parameter_schema, dict):
                parameter_schema = {}
            expected = parameter_schema.get("type")
            if function.get("strict", False):
                decode = raw_string_pattern(parameter_schema, forbidden=_RAW_CLOSERS) is None
            else:
                # An untyped value is decoded best-effort, so it stays raw
                # text up to the first closer.
                decode = expected not in ("string", None)
            if decode:
                value_end = _json_value_end(text, parameter.end())
            else:
                value_end = text.find(_PARAMETER_CLOSE, parameter.end())
            if value_end < 0:
                raise ValueError("Unclosed ATEM parameter")
            value = text[parameter.end() : value_end]
            if not decode and (_INVOKE_CLOSE in value or _TOOL_CLOSE in value):
                raise ValueError("Unclosed ATEM parameter")
            if function.get("strict", False):
                if decode:
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        raise ValueError(
                            "ATEM parameter must contain valid JSON"
                        ) from None
                if not schema_value_matches(value, parameter_schema):
                    raise ValueError("ATEM parameter violates its strict schema")
            elif expected is None:
                # Best effort: like any other text that is not JSON, text
                # that decodes only to a non-finite number (``NaN``,
                # ``1e400``, ``[Infinity]``) stays the raw string, which the
                # arguments can carry.
                try:
                    decoded = json.loads(value)
                    json.dumps(decoded, allow_nan=False)
                except ValueError:
                    pass
                else:
                    value = decoded
            elif expected != "string":
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    raise ValueError("ATEM parameter must contain valid JSON") from None
            types = {
                "integer": lambda x: type(x) is int,
                "number": lambda x: type(x) in {int, float},
                "boolean": lambda x: type(x) is bool,
                "object": lambda x: isinstance(x, dict),
                "array": lambda x: isinstance(x, list),
                "null": lambda x: x is None,
            }
            if (
                isinstance(expected, str)
                and expected in types
                and not types[expected](value)
            ):
                raise ValueError("ATEM parameter type mismatch")
            arguments[key], end = value, value_end + len(_PARAMETER_CLOSE)
        closing = text.find(_INVOKE_CLOSE, end)
        if (
            closing < 0
            or text[end:closing].strip()
            or any(key not in arguments for key in schema.get("required", []))
        ):
            raise ValueError("Malformed or missing ATEM parameters")
        # Reject non-finite JSON numbers before an API response can contain them.
        json.dumps(arguments, allow_nan=False)
        calls.append({"name": name, "arguments": arguments})
        cursor = closing + len(_INVOKE_CLOSE)
    if text[cursor:].strip() or not calls:
        raise ValueError("Malformed or empty ATEM block")
    return calls


_INVOKE_NAME_OPEN = '<atem:invoke name="'
_INVOKE_NAME = re.compile(r"[\w.-]+")


def partial_function_names(partial, tools=None):
    """The functions a cut-short ATEM block had committed to, in order.

    The reader behind the stop-cut record (see
    :func:`mlx2.output.truncated_tool_record`), by :func:`parse_atem`'s own rule
    (``_INVOKE``): an invoke names its function between ``name="`` and
    ``">``, and only ``[\\w.-]+`` is a name.  The name is committed at its
    closing quote; a cut before it has committed to nothing.  Raises
    ``ValueError`` for a closed name the rule does not admit (``db:lookup``,
    an empty name, a quote not followed by ``>``): the parser would find no
    invoke there and reject the block.

    Past an invoke's name its parameter values are skipped as
    :func:`parse_atem` reads them (a raw value runs to the first
    ``</atem:parameter>``, a JSON-typed one to the first closer outside its
    strings), so an invoke tag quoted inside a value is value text, not a
    sibling invoke; a cut inside a value ends the walk.
    """
    definitions = {
        tool["function"]["name"]: tool["function"] for tool in tools or ()
    }
    names, cursor = [], 0
    while (start := partial.find(_INVOKE_NAME_OPEN, cursor)) >= 0:
        body = start + len(_INVOKE_NAME_OPEN)
        close = partial.find('"', body)
        if close < 0:
            break
        name = partial[body:close]
        if _INVOKE_NAME.fullmatch(name) is None or partial[close + 1 : close + 2] not in (
            "",
            ">",
        ):
            raise ValueError("Malformed ATEM invoke name")
        names.append(name)
        cursor = close + 1
        end = _skip_parameter_values(partial, cursor, definitions.get(name))
        if end < 0:
            break
        cursor = end
    return names


def _skip_parameter_values(text, start, function):
    """Where the parameter walk of the invoke open at ``start`` leaves off.

    Mirrors ``parse_atem``'s value rules for the committed-name reader:
    each ``<atem:parameter name="…">`` value is skipped to its closer, raw
    or JSON by the function's schema, so a quoted invoke inside it is never
    read as a sibling.  Returns -1 when the text ends inside a value; the
    position of the next invoke closer, invoke opener, or tag the parser
    would reject otherwise (the caller looks for the next invoke from there).
    """
    schema = function.get("parameters", {}) if function else {}
    strict = bool(function and function.get("strict", False))
    if strict:
        try:
            schema = executable_schema(schema)
        except ValueError:
            schema = {}  # the parser fails the block
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    if not isinstance(properties, dict):
        properties = {}
    end = start
    while (parameter := _PARAMETER.search(text, end)) is not None:
        for tag in (_INVOKE_CLOSE, _INVOKE_NAME_OPEN):
            position = text.find(tag, end)
            if 0 <= position < parameter.start():
                return position
        parameter_schema = properties.get(parameter.group(1), {})
        if not isinstance(parameter_schema, dict):
            parameter_schema = {}
        if strict:
            try:
                decode = raw_string_pattern(parameter_schema, forbidden=_RAW_CLOSERS) is None
            except ValueError:
                decode = False
        else:
            decode = parameter_schema.get("type") not in ("string", None)
        if decode:
            try:
                value_end = _json_value_end(text, parameter.end())
            except _UnclosedJSONString:
                return -1  # cut inside a JSON string: value text
        else:
            # A raw value cannot contain any closer: the parser rejects an
            # invoke or tool closer inside it, so the walk stops there.
            closers = [
                position
                for position in (text.find(tag, parameter.end()) for tag in _RAW_CLOSERS)
                if position >= 0
            ]
            value_end = min(closers) if closers else -1
            if value_end >= 0 and not text.startswith(_PARAMETER_CLOSE, value_end):
                return value_end
        if value_end < 0:
            return -1  # cut inside the value: the rest is value text
        end = value_end + len(_PARAMETER_CLOSE)
    return end


def _non_strict_value(schema):
    """Value language ``parse_atem`` accepts for a non-strict parameter.

    It reads the schema's own ``type``. A ``"string"`` value is kept as raw
    text and an untyped one is decoded best-effort, so both stay free text,
    unbounded like Qwen's: a ``{0,4096}`` run costs the exact automaton
    thousands of states per parameter (tens of seconds to compile one call),
    and the parser has no length bound to agree with. Raw text excludes
    only the closers (``_RAW_CLOSERS``), not ``<``. Every other type,
    a list of types included (``["integer", "null"]``), must decode as JSON,
    so the value is the union of the declared alternatives' JSON: free text
    there is admitted by the grammar and then rejected. Numbers are finite
    (a non-finite one fails serialization), and a string alternative keeps
    a raw ``<`` as the template writes JSON: the parser ends a JSON value
    outside its strings. Objects, arrays and types the parser does not check
    use the shared recursive JSON rules, which ``constrained_tool_grammar``
    then defines with the same finite numbers.
    """
    from ..structured_output import _FINITE_NUMBER, _INTEGER, _STRING

    expected = schema.get("type") if isinstance(schema, dict) else None
    if expected is None or expected == "string":
        return _RAW_TEXT
    patterns = {
        "integer": _INTEGER,
        "number": _FINITE_NUMBER,
        "boolean": "(?:true|false)",
        "null": "null",
        "string": _STRING,
        "object": "(?&object)",
        "array": "(?&array)",
    }
    kinds = expected if isinstance(expected, list) else [expected]
    alternatives = []
    for kind in kinds:
        pattern = patterns.get(kind) if isinstance(kind, str) else None
        if pattern is None:
            # The parser decodes any JSON for a type it does not check.
            return "(?&value)"
        if pattern not in alternatives:
            alternatives.append(pattern)
    if not alternatives:
        return "(?&value)"
    return "(?:" + "|".join(alternatives) + ")"


def _non_strict_parameter_body(function):
    """ATEM parameters of a non-strict tool, as ``parse_atem`` accepts them.

    Required arguments must appear (sglang #40051).  Optional arguments are
    the declared ones, each at most once in declared order, since the parser
    rejects a repeated name.  Only a schema that declares no properties keeps
    free argument names.
    """
    schema = function.get("parameters", {})
    schema = schema if isinstance(schema, dict) else {}
    properties = schema.get("properties", {})
    properties = properties if isinstance(properties, dict) else {}
    required = required_parameter_names(function)
    for key in required:
        if not isinstance(key, str) or not re.fullmatch(r"[\w.-]+", key, re.ASCII):
            raise ValueError("tool parameter name is not representable in ATEM")

    def block(key):
        value = _non_strict_value(properties.get(key))
        return rf'<atem:parameter name="{re.escape(key)}">{value}</atem:parameter>'

    body = "".join(block(key) for key in required)
    if properties:
        # An optional name the wire cannot spell is simply never emitted.
        body += "".join(
            f"(?:{block(key)})?"
            for key in properties
            if key not in required
            and isinstance(key, str)
            and re.fullmatch(r"[\w.-]+", key, re.ASCII)
        )
    elif schema.get("additionalProperties") is not False:
        body += (
            r'(?:<atem:parameter name="[\w.-]{1,128}">'
            rf"{_non_strict_value(None)}</atem:parameter>){{0,{max(0, 64 - len(required))}}}"
        )
    return body


def constrained_tool_grammar(tools, tool_choice, *, parallel_tool_calls=True):
    """Regex for Muse's ATEM tool-call wire format."""
    from ..structured_output import _schema_pattern, recursive_json_object_pattern

    functions = [tool["function"] for tool in tools]
    if isinstance(tool_choice, dict):
        selected = tool_choice["function"]["name"]
        functions = [function for function in functions if function["name"] == selected]
    if not functions:
        raise ValueError("constrained tool_choice selects no declared function")
    invocations = []
    for function in functions:
        name = function["name"]
        if not re.fullmatch(r"[\w.-]+", name, re.ASCII):
            raise ValueError("tool name is not representable in ATEM")
        if function.get("strict", False):
            schema = executable_schema(function.get("parameters", {}))
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            ordered = [key for key in properties if key in required] + [
                key for key in properties if key not in required
            ]
            blocks = []
            for key in ordered:
                if not re.fullmatch(r"[\w.-]+", key, re.ASCII):
                    raise ValueError("tool parameter name is not representable in ATEM")
                subschema = properties[key]
                value = raw_string_pattern(
                    subschema, forbidden=_RAW_CLOSERS, free_text=_RAW_TEXT
                )
                if value is None:
                    # ``parse_atem`` decodes the value and rejects infinity.
                    value = _schema_pattern(subschema, finite_numbers=True)
                block = (
                    rf'<atem:parameter name="{re.escape(key)}">'
                    rf"{value}</atem:parameter>"
                )
                blocks.append(block if key in required else f"(?:{block})?")
            body = "".join(blocks)
        else:
            body = _non_strict_parameter_body(function)
        invocations.append(
            rf'<atem:invoke name="{re.escape(name)}">{body}</atem:invoke>'
        )
    invoke = "(?:" + "|".join(invocations) + ")"
    body = invoke if parallel_tool_calls is False else invoke + f"(?:{invoke})*"
    pattern = re.escape(_TOOL_OPEN) + body + re.escape(_TOOL_CLOSE)
    if _RAW_TEXT in body:
        pattern += _raw_text_definition()
    # Only a non-strict value lowered to JSON rules calls them; names are
    # escaped and cannot spell a call.  The JSON block stays last: the
    # whole-output composer strips and merges it with an answer's.
    if re.search(r"\(\?&(?:value|object|array)\)", body):
        pattern += recursive_json_object_pattern(finite_numbers=True)[1]
    return pattern


def _raw_text_definition():
    """The ``atem_text`` rule: any text containing none of the closers."""
    from ..tool_grammar import text_excluding

    return "(?(DEFINE)(?P<atem_text>" + text_excluding(*_RAW_CLOSERS) + "))"


class MuseOutputParser:
    def __init__(
        self, *, chat=False, tools=None, stops=(), parallel_tool_calls=True
    ):
        self.chat, self.tools = chat, tools or []
        self.stop_matcher = StopSequenceMatcher(stops)
        self.buffer = ""
        self.state = "header" if chat else "body"
        self.channel = "content"
        self.parallel_tool_calls = bool(parallel_tool_calls)
        self.stopped, self.tool_count = False, 0
        self.tool_call_constraint_truncations = 0
        self.stop_truncated_tool_call = None  # legacy stop-only evidence
        self.truncated_tool_call = None
        # Set only for the duration of :meth:`finish` on a ``length`` stop.
        self._length_finish = False

    def _record_cut_call(self, partial, *, stop_hit):
        self.truncated_tool_call = truncated_tool_record(
            partial_function_names, partial, self.tools,
            cause="stop_sequence" if stop_hit else "max_tokens",
        )
        if stop_hit:
            self.stop_truncated_tool_call = {
                key: value for key, value in self.truncated_tool_call.items()
                if key != "cause"
            }

    @property
    def stop_sequence(self):
        return self.stop_matcher.stop_sequence

    def finish(self, text, finish_reason):
        """Push the last text of a generation that ended with ``finish_reason``.

        Same contract as :meth:`mlx2.output.OutputParser.finish`: a ``length``
        finish (``max_tokens``) inside an unclosed ATEM block, or inside a
        channel header, drops the unfinished markup instead of failing the
        request; EOS still fails closed.
        """
        self._length_finish = finish_reason == "length"
        try:
            return self.push(text, final=True)
        finally:
            self._length_finish = False

    def push(self, text, *, final=False):
        if self.stopped:
            return []
        text, stop_hit = self.stop_matcher.push(text, final=final)
        if stop_hit:
            self.stopped, final = True, True
        # A requested stop (max_tokens or a client stop string) that cuts
        # markup short is not malformed model output.
        truncated = final and (stop_hit or self._length_finish)
        self.buffer += text
        events = []
        while self.buffer:
            if not self.chat:
                events.append({"content": self.buffer})
                self.buffer = ""
                break
            if self.state == "header":
                marker = self.buffer.find(_MESSAGE)
                if marker >= 0:
                    header = _HEADER.fullmatch(self.buffer[:marker])
                    if header:
                        recipient = header.group(1)
                        self.channel = (
                            "reasoning_content" if recipient == "self" else "content"
                        )
                        self.buffer = self.buffer[marker + len(_MESSAGE) :]
                        self.state = "body"
                        continue
                potential = _partial_header(self.buffer)
                if potential and not final and len(self.buffer) < 512:
                    break
                if potential and final and truncated:
                    self.buffer = ""
                    break
                if potential and final:
                    raise ValueError("Model produced an incomplete Muse channel header")
                if self.buffer.lstrip().startswith("<|start|>"):
                    raise ValueError("Model produced an invalid Muse channel header")
                self.state = "body"
                continue
            if self.state == "tool":
                calls, end = None, self.buffer.find(_TOOL_CLOSE)
                while end >= 0:
                    try:
                        calls = parse_atem(self.buffer[:end], self.tools)
                        break
                    except _UnclosedJSONString:
                        # That closer is quoted inside a JSON value.
                        end = self.buffer.find(_TOOL_CLOSE, end + 1)
                if calls is None:
                    if truncated:
                        # Drop the partial call: it is neither a call nor
                        # answer text.  The finish reason stays as it was.
                        self._record_cut_call(self.buffer, stop_hit=stop_hit)
                        self.buffer = ""
                        self.state, self.channel = "body", "content"
                        break
                    if final:
                        raise ValueError("Model produced an incomplete ATEM tool call")
                    break
                calls = within_parallel_bound(self, calls)
                for call in calls:
                    events.append(
                        {
                            "tool_calls": [
                                {
                                    "index": self.tool_count,
                                    "id": "call_" + uuid.uuid4().hex[:24],
                                    "type": "function",
                                    "function": {
                                        "name": call["name"],
                                        "arguments": json.dumps(
                                            call["arguments"], allow_nan=False
                                        ),
                                    },
                                }
                            ]
                        }
                    )
                    self.tool_count += 1
                self.buffer = self.buffer[end + len(_TOOL_CLOSE) :]
                self.state, self.channel = "body", "content"
                continue
            markers = {
                "<|eom|>": "header",
                "<|eot|>": "stop",
                "<|end_of_text|>": "stop",
                "<|start|>": "restart",
                _TOOL_OPEN: "tool",
            }
            matches = [
                (self.buffer.find(marker), marker)
                for marker in markers
                if marker in self.buffer
            ]
            if matches:
                end, marker = min(matches)
                if end:
                    events.append({self.channel: self.buffer[:end]})
                self.buffer = self.buffer[end + len(marker) :]
                action = markers[marker]
                if action == "stop":
                    self.stopped, self.buffer = True, ""
                elif action == "restart":
                    self.buffer, self.state = marker + self.buffer, "header"
                else:
                    self.state = action
            else:
                end = len(self.buffer) if final else _safe_prefix(self.buffer, markers)
                if end:
                    events.append({self.channel: self.buffer[:end]})
                self.buffer = self.buffer[end:]
                break
        if final and self.state == "tool":
            if truncated:
                # Nothing buffered after the opener: the loop never saw the
                # call, but one was in progress when the stop cut it.
                self._record_cut_call(self.buffer, stop_hit=stop_hit)
                self.state, self.channel = "body", "content"
            else:
                raise ValueError("Model produced an incomplete ATEM tool call")
        return events

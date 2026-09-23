# SPDX-License-Identifier: MIT
"""Incremental Xing4.0 reasoning, content, and tool-call channels.

Input is the streaming decode of the generated ids with special tokens kept
(``XingStreamingDetokenizer``), so markers arrive as text and may be split
across chunks.  Output events are the mlx2 delta dicts used by
``mlx2.output.OutputParser`` and the North/Muse parsers:
``{"reasoning_content": str}``, ``{"content": str}`` and
``{"tool_calls": [{"index", "id", "type": "function", "function": {"name",
"arguments": <JSON string>}}]}``.

Channel rules (vLLM ``xing4_0`` reasoning parser semantics):

* thinking on: the prompt ends with ``<_bot><think>\\n`` so the output starts
  inside reasoning; ``</think>`` (id 10) ends it.  The one ``\\n`` the chat
  template places before ``</think>`` is template structure and is dropped.
  A stray ``<think>`` inside reasoning is dropped.
* thinking off: the prompt ends with ``<_bot></think>``; everything is content.
  A stray ``</think>`` in content is dropped, a ``<think>`` opens reasoning.
* Tool calls are recognized in content only (as vLLM runs the tool parser on
  content after the reasoning parser), and only when the request declared
  tools.  Several calls may follow each other back to back.
* ``<_end>`` and any other turn marker (``<_user>``, ``<_bot>``, ``<_system>``,
  ``<_start>``, ``<_pad>``, ``<tool_response>``, ``</tool_response>``) ends the
  turn.
* Output that ends inside reasoning (no ``</think>``) stays reasoning.  SGLang's
  ``xing4_0`` detector instead sets ``force_nonempty_content=True`` and moves
  unterminated reasoning to content at the end; that is not reproduced: the
  reasoning was already streamed as reasoning deltas and cannot be retracted,
  and a truncated thought is not an answer.  This matches vLLM and HF.
* Whitespace that directly precedes a tool call, or trails the content after
  the last tool call, is not content (the template renders calls back to back
  after stripped content).
* Client stop strings apply to visible content only (as in the North parser).

Tool block grammar (chat template, vLLM/SGLang parsers)::

    <tool_call>NAME<param_key>K</param_key><param_value>V</param_value>...</tool_call>

``V`` is raw text for string-typed parameters and JSON for anything else (the
template renders non-strings with ``tojson``).  Following both reference
parsers, keys and values are stripped, a value is kept raw when the declared
schema admits a string and is otherwise decoded with ``json.loads`` then
``ast.literal_eval``, falling back to the raw text.  Two refinements: union
schemas (``anyOf``/``oneOf``/list ``type``) that admit a string keep the raw
text (vLLM only inspects a scalar ``type``), except that ``null`` decodes to
``None`` when the schema also admits null; and decoded values that JSON
cannot carry as written (a non-finite float, a set, or a tuple or
non-string key it would rewrite) stay raw text.  ``tojson`` leaves ``<``
raw, so a JSON value (a declared type without a string) ends at the first
closer outside its strings; a raw or untyped value ends at the first
closer.  A JSON body is accepted too:
``<tool_call>{"name": ..., "arguments": {...}}</tool_call>`` (SGLang) and
``<tool_call>NAME{...}</tool_call>`` (vLLM).  As in the other mlx2 parsers,
an undeclared tool, a missing required or undeclared parameter
(``additionalProperties: false``), duplicate or malformed parameters, and an
incomplete block raise ``ValueError`` instead of leaking markup into content.
"""

from __future__ import annotations

import ast
import json
import re
import uuid

from ..output import StopSequenceMatcher, _safe_prefix, within_parallel_bound
from ..runtime.tool_parsers._schema import json_native

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"
PARAM_KEY_OPEN = "<param_key>"
PARAM_VALUE_CLOSE = "</param_value>"
TURN_END_MARKERS = (
    "<_end>",
    "<_user>",
    "<_bot>",
    "<_system>",
    "<_start>",
    "<_pad>",
    "<tool_response>",
    "</tool_response>",
)

# A parameter up to its value.  The key never runs past a block closer, so a
# block the model cut short inside a key still ends at the first one.
_PARAM_HEAD = re.compile(
    r"\s*<param_key>((?:(?!</tool_call>)[\s\S])*?)</param_key>\s*<param_value>"
)
_JSON_OPEN = re.compile(r'\s*[\[{"]')
_JSON_CLOSER = re.compile(r"</param_value>|</tool_call>")
_JSON_STRING = re.compile(r'"[^"\\]*(?:\\.[^"\\]*)*"', re.DOTALL)


def _schema_types(schema) -> set:
    """All JSON types a schema fragment admits (scalar/list ``type``, unions)."""
    if not isinstance(schema, dict):
        return set()
    declared = schema.get("type")
    types = set()
    if isinstance(declared, str):
        types.add(declared)
    elif isinstance(declared, (list, tuple)):
        types.update(t for t in declared if isinstance(t, str))
    for key in ("anyOf", "oneOf"):
        for branch in schema.get(key) or ():
            types |= _schema_types(branch)
    return types


def _deserialize(text: str):
    """vLLM/SGLang ``_deserialize``: JSON, then a Python literal, else raw text.

    A decoded value the arguments cannot carry as written (a tuple, a
    non-string key, a non-finite float, a set) stays the raw text too.
    """
    try:
        value = json.loads(text)
        if json_native(value):
            return value
        return text
    except ValueError:
        pass
    try:
        value = ast.literal_eval(text)
    except Exception:  # noqa: BLE001 - any failure means "not a literal"
        return text
    return value if json_native(value) else text


def _parameter_value(raw: str, schema):
    types = _schema_types(schema)
    if "string" in types:
        if raw == "null" and "null" in types:
            return None
        return raw
    return _deserialize(raw)


def _json_value_end(text: str, start: int):
    """Where the closer after the JSON value that starts at ``start`` is.

    The template writes non-string values with ``tojson``, which leaves ``<``
    raw, so a JSON string may quote ``</param_value>`` or ``</tool_call>``.
    Outside a string ``<`` is not JSON, so the value ends at the first closer
    outside every string.  Returns -1 while none has arrived, including while
    the text ends inside a string.
    """
    position = start
    while True:
        closer = _JSON_CLOSER.search(text, position)
        close = closer.start() if closer is not None else -1
        quote = text.find('"', position, close if close >= 0 else len(text))
        if quote < 0:
            return close
        string = _JSON_STRING.match(text, quote)
        if string is None:
            return -1
        position = string.end()


def _walk_parameters(text: str, position: int, properties: dict):
    """``(pairs, position, open_json)`` for the parameters from ``position``.

    ``pairs`` holds ``(key, raw value, quoted)`` for every closed parameter
    and ``position`` is where the walk stopped.  A raw value (string-typed or
    untyped) ends at the first closer, as before: a closer the string itself
    contains cannot be told from markup.  A value whose schema admits no
    string and whose text opens a JSON object, array or string ends at the
    first closer outside its strings; ``quoted`` says it contains one.
    ``open_json`` says the text ends inside such a value before any closer
    outside its strings, so every closer so far is quoted.
    """
    pairs = []
    while True:
        head = _PARAM_HEAD.match(text, position)
        if head is None:
            return pairs, position, False
        start = head.end()
        first = text.find(PARAM_VALUE_CLOSE, start)
        types = _schema_types(properties.get(head.group(1).strip()))
        if types and "string" not in types and _JSON_OPEN.match(text, start):
            end = _json_value_end(text, start)
            if end < 0:
                return pairs, position, True
        else:
            block = text.find(TOOL_CLOSE, start)
            end = -1 if 0 <= block < first else first
        if end < 0 or not text.startswith(PARAM_VALUE_CLOSE, end):
            return pairs, position, False
        quoted = _JSON_CLOSER.search(text, start, end) is not None
        pairs.append((head.group(1), text[start:end], quoted))
        position = end + len(PARAM_VALUE_CLOSE)


def _tool_close(text: str, tools: list[dict]) -> int:
    """Where the ``</tool_call>`` closing the tool-call block in ``text`` is.

    A JSON value may quote the closer (see ``_walk_parameters``), so the block
    ends at the first one after its parameters.  Returns -1 while none has
    arrived, including while every closer so far sits in a JSON value.  The
    walk runs only once a closer has arrived.
    """
    close = text.find(TOOL_CLOSE)
    position = text.find(PARAM_KEY_OPEN)
    if close < 0 or position < 0 or close < position:
        return close
    definitions = {tool["function"]["name"]: tool["function"] for tool in tools or ()}
    function = definitions.get(text[:position].strip()) or {}
    properties = (function.get("parameters") or {}).get("properties") or {}
    _, position, open_json = _walk_parameters(text, position, properties)
    return -1 if open_json else text.find(TOOL_CLOSE, position)


def _tag_arguments(text: str, properties: dict) -> dict:
    arguments = {}
    pairs, position, _ = _walk_parameters(text, 0, properties)
    if text[position:].strip():
        raise ValueError("Malformed Xing tool-call parameters")
    for key, raw, quoted in pairs:
        key, raw = key.strip(), raw.strip()
        if not key:
            raise ValueError("Xing tool-call parameter has an empty name")
        if key in arguments:
            raise ValueError("Duplicate Xing tool-call parameter")
        if quoted:
            # The value quotes a closer.  Only JSON puts one there.
            try:
                json.loads(raw)
            except ValueError:
                raise ValueError("Malformed Xing JSON parameter") from None
        arguments[key] = _parameter_value(raw, properties.get(key))
    return arguments


def _json_arguments(value) -> dict:
    if isinstance(value, str):
        try:
            value = json.loads(value) if value.strip() else {}
        except ValueError:
            raise ValueError("Xing tool-call arguments are not a JSON object") from None
    if not isinstance(value, dict):
        raise ValueError("Xing tool-call arguments are not a JSON object")
    return value


def _complete_tool_payload(payload: str) -> bool:
    """A tool-call body the turn end may close: nothing is left half-open."""
    body = payload.strip()
    if not body:
        return False
    if PARAM_KEY_OPEN in body:
        opened = body.count(PARAM_KEY_OPEN)
        return (
            body.endswith("</param_value>")
            and body.count("</param_key>") == opened
            and body.count("<param_value>") == opened
            and body.count("</param_value>") == opened
        )
    if body.startswith("{") or "{" in body:
        start = body.find("{")
        try:
            json.loads(body[start:])
        except ValueError:
            return False
        return True
    return False


def parse_tool_block(payload: str, tools: list[dict]) -> dict:
    """Parse the text between ``<tool_call>`` and ``</tool_call>``.

    Returns ``{"name": str, "arguments": dict}``; raises ``ValueError`` on
    malformed or undeclared calls.
    """
    definitions = {tool["function"]["name"]: tool["function"] for tool in tools or ()}
    body = payload.strip()
    if not body:
        raise ValueError("Empty Xing tool call")
    position = body.find(PARAM_KEY_OPEN)
    if position >= 0:
        name = body[:position].strip()
        parameters = (definitions.get(name) or {}).get("parameters") or {}
        arguments = _tag_arguments(body[position:], parameters.get("properties") or {})
    elif body.startswith("{"):
        try:
            value = json.loads(body)
        except ValueError:
            raise ValueError("Malformed Xing JSON tool call") from None
        if not isinstance(value, dict) or not isinstance(value.get("name"), str):
            raise ValueError("Xing JSON tool call needs a string name")
        name = value["name"]
        arguments = _json_arguments(value.get("arguments", value.get("parameters", {})))
    else:
        name, arguments = body, {}
        for candidate in sorted(definitions, key=len, reverse=True):
            if body.startswith(candidate):
                rest = body[len(candidate) :].strip()
                if not rest:
                    name = candidate
                    break
                if rest.startswith("{"):
                    try:
                        value = json.loads(rest)
                    except ValueError:
                        raise ValueError("Malformed Xing JSON tool arguments") from None
                    value = value if isinstance(value, dict) else None
                    if value is None:
                        raise ValueError("Xing tool-call arguments are not a JSON object")
                    name = candidate
                    arguments = _json_arguments(value.get("arguments", value.get("parameters", value)))
                    break
    if name not in definitions:
        raise ValueError("Model called an undeclared tool")
    schema = definitions[name].get("parameters") or {}
    if any(key not in arguments for key in schema.get("required", [])):
        raise ValueError("Xing tool call is missing a required parameter")
    if schema.get("additionalProperties") is False and any(
        key not in (schema.get("properties") or {}) for key in arguments
    ):
        raise ValueError("Xing tool call contains an undeclared parameter")
    json.dumps(arguments, allow_nan=False)
    return {"name": name, "arguments": arguments}


class XingOutputParser:
    """Streaming Xing4.0 channel parser; ``push`` returns mlx2 delta events."""

    def __init__(
        self,
        *,
        chat=False,
        thinking=True,
        tools=None,
        stops=(),
        parallel_tool_calls=True,
    ):
        self.chat = chat
        self.tools = list(tools or [])
        self.parallel_tool_calls = bool(parallel_tool_calls)
        self.tool_call_constraint_truncations = 0
        self.channel = "reasoning_content" if chat and thinking else "content"
        self.stop_matcher = StopSequenceMatcher(stops or ())
        self.buffer = self.held = ""
        self.stopped = False
        self.tool_count = 0
        self.turn_closed_tool_calls = 0

    @property
    def stop_sequence(self):
        return self.stop_matcher.stop_sequence

    # -- visible text -----------------------------------------------------

    def _visible(self, text: str, *, final: bool = False) -> list[dict]:
        """Apply client stop strings to content; reasoning passes through."""
        if self.channel != "content":
            return [{self.channel: text}] if text else []
        visible, stop_hit = self.stop_matcher.push(text, final=final)
        if stop_hit:
            self.stopped = True
            return [{"content": visible}] if visible else []
        return [{"content": visible}] if visible else []

    def _text(self, text: str) -> list[dict]:
        """Channel text with template-structure whitespace held back."""
        if not text:
            return []
        combined = self.held + text
        if self.channel == "reasoning_content":
            # Hold one trailing newline: it belongs to "\n</think>" if the
            # close marker follows.
            if combined.endswith("\n"):
                self.held = "\n"
                combined = combined[:-1]
            else:
                self.held = ""
            return self._visible(combined)
        # Trailing whitespace is held until more text arrives, so whitespace
        # directly before a tool call (or after the last one) is dropped no
        # matter how the stream was chunked.
        body = combined.rstrip()
        self.held = combined[len(body) :]
        return self._visible(body) if body else []

    def _release_held(self, *, drop: bool) -> list[dict]:
        held, self.held = self.held, ""
        return [] if drop or not held else self._visible(held)

    def _finish_visible(self) -> list[dict]:
        events = self._release_held(drop=self.channel == "content" and self.tool_count > 0)
        if self.channel == "content" and not self.stopped:
            events.extend(self._visible("", final=True))
        return events

    # -- tool calls -------------------------------------------------------

    def _tool_events(self, call: dict) -> list[dict]:
        events = []
        for bounded in within_parallel_bound(self, [call]):
            events.append(
                {
                    "tool_calls": [
                        {
                            "index": self.tool_count,
                            "id": "call_" + uuid.uuid4().hex[:24],
                            "type": "function",
                            "function": {
                                "name": bounded["name"],
                                "arguments": json.dumps(
                                    bounded["arguments"], allow_nan=False
                                ),
                            },
                        }
                    ]
                }
            )
            self.tool_count += 1
        return events

    def _markers(self) -> dict:
        if self.channel == "reasoning_content":
            markers = {THINK_CLOSE: "content", THINK_OPEN: "drop"}
        else:
            markers = {THINK_OPEN: "reasoning_content", THINK_CLOSE: "drop"}
            if self.tools:
                markers[TOOL_OPEN] = "tool"
        for marker in TURN_END_MARKERS:
            markers[marker] = "stop"
        return markers

    # -- driver -----------------------------------------------------------

    def push(self, text: str, *, final: bool = False) -> list[dict]:
        if self.stopped:
            return []
        self.buffer += text
        events: list[dict] = []
        if not self.chat:
            events.extend(self._visible(self.buffer, final=final))
            self.buffer = ""
            return events
        while not self.stopped:
            if self.channel == "tool":
                end = _tool_close(self.buffer, self.tools)
                turn = [self.buffer.find(m) for m in TURN_END_MARKERS if m in self.buffer]
                if turn and (end < 0 or min(turn) < end):
                    # The turn ended inside the block.  A payload that is
                    # structurally complete (every parameter pair closed, or
                    # a whole JSON object) is closed by the turn end; the
                    # reference parsers would drop it as plain text, and a
                    # truncated call still fails closed.
                    payload = self.buffer[: min(turn)]
                    if not _complete_tool_payload(payload):
                        raise ValueError("Model produced an incomplete Xing tool call")
                    events.extend(self._tool_events(parse_tool_block(payload, self.tools)))
                    self.turn_closed_tool_calls += 1
                    events.extend(self._finish_visible())
                    self.stopped, self.buffer = True, ""
                    break
                if end < 0:
                    if final:
                        # EOS ends generation without reaching the parser as
                        # text: the same completeness rule as a turn marker.
                        if not _complete_tool_payload(self.buffer):
                            raise ValueError("Model produced an incomplete Xing tool call")
                        events.extend(
                            self._tool_events(parse_tool_block(self.buffer, self.tools))
                        )
                        self.turn_closed_tool_calls += 1
                        events.extend(self._finish_visible())
                        self.stopped, self.buffer = True, ""
                    return events
                events.extend(
                    self._tool_events(parse_tool_block(self.buffer[:end], self.tools))
                )
                self.buffer = self.buffer[end + len(TOOL_CLOSE) :]
                self.channel = "content"
                continue
            markers = self._markers()
            hits = [(self.buffer.find(m), m) for m in markers if m in self.buffer]
            if not hits:
                end = len(self.buffer) if final else _safe_prefix(self.buffer, markers)
                events.extend(self._text(self.buffer[:end]))
                self.buffer = self.buffer[end:]
                break
            index, marker = min(hits)
            events.extend(self._text(self.buffer[:index]))
            if self.stopped:
                break
            self.buffer = self.buffer[index + len(marker) :]
            action = markers[marker]
            if action == "drop":
                continue
            if action == "stop":
                events.extend(self._finish_visible())
                self.stopped, self.buffer = True, ""
                break
            if action == "tool":
                self._release_held(drop=True)
            elif self.channel == "reasoning_content" and action == "content":
                self._release_held(drop=True)  # the "\n" before </think>
            else:
                events.extend(self._release_held(drop=False))
            self.channel = action
        if final and not self.stopped:
            events.extend(self._finish_visible())
        return events

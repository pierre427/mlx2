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
``None`` when the schema also admits null; and decoded values that are not
finite JSON stay raw text.  A JSON body is accepted too:
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

from ..output import _safe_prefix

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"
PARAM_KEY_OPEN = "<param_key>"
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

_PARAM = re.compile(
    r"<param_key>(.*?)</param_key>\s*<param_value>(.*?)</param_value>", re.DOTALL
)


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


def _finite_json(value) -> bool:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return False
    return True


def _deserialize(text: str):
    """vLLM/SGLang ``_deserialize``: JSON, then a Python literal, else raw text."""
    try:
        value = json.loads(text)
        if _finite_json(value):
            return value
        return text
    except ValueError:
        pass
    try:
        value = ast.literal_eval(text)
    except Exception:  # noqa: BLE001 - any failure means "not a literal"
        return text
    return value if _finite_json(value) else text


def _parameter_value(raw: str, schema):
    types = _schema_types(schema)
    if "string" in types:
        if raw == "null" and "null" in types:
            return None
        return raw
    return _deserialize(raw)


def _tag_arguments(text: str, properties: dict) -> dict:
    arguments, cursor = {}, 0
    for match in _PARAM.finditer(text):
        if text[cursor : match.start()].strip():
            raise ValueError("Malformed Xing tool-call parameters")
        key, raw = match.group(1).strip(), match.group(2).strip()
        if not key:
            raise ValueError("Xing tool-call parameter has an empty name")
        if key in arguments:
            raise ValueError("Duplicate Xing tool-call parameter")
        arguments[key] = _parameter_value(raw, properties.get(key))
        cursor = match.end()
    if text[cursor:].strip():
        raise ValueError("Malformed Xing tool-call parameters")
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

    def __init__(self, *, chat=False, thinking=True, tools=None, stops=()):
        self.chat = chat
        self.tools = list(tools or [])
        self.channel = "reasoning_content" if chat and thinking else "content"
        self.stops = (stops,) if isinstance(stops, str) else tuple(stops or ())
        self.buffer = self.stop_buffer = self.held = ""
        self.stopped = False
        self.tool_count = 0
        self.turn_closed_tool_calls = 0

    # -- visible text -----------------------------------------------------

    def _visible(self, text: str, *, final: bool = False) -> list[dict]:
        """Apply client stop strings to content; reasoning passes through."""
        if self.channel != "content":
            return [{self.channel: text}] if text else []
        self.stop_buffer += text
        hits = [self.stop_buffer.find(stop) for stop in self.stops if stop in self.stop_buffer]
        if hits:
            visible, self.stop_buffer = self.stop_buffer[: min(hits)], ""
            self.stopped = True
            return [{"content": visible}] if visible else []
        end = len(self.stop_buffer) if final else _safe_prefix(self.stop_buffer, self.stops)
        visible, self.stop_buffer = self.stop_buffer[:end], self.stop_buffer[end:]
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

    def _tool_event(self, call: dict) -> dict:
        event = {
            "tool_calls": [
                {
                    "index": self.tool_count,
                    "id": "call_" + uuid.uuid4().hex[:24],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(call["arguments"], allow_nan=False),
                    },
                }
            ]
        }
        self.tool_count += 1
        return event

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
                end = self.buffer.find(TOOL_CLOSE)
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
                    events.append(self._tool_event(parse_tool_block(payload, self.tools)))
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
                        events.append(self._tool_event(parse_tool_block(self.buffer, self.tools)))
                        self.turn_closed_tool_calls += 1
                        events.extend(self._finish_visible())
                        self.stopped, self.buffer = True, ""
                    return events
                events.append(self._tool_event(parse_tool_block(self.buffer[:end], self.tools)))
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

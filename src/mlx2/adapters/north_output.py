"""Incremental Cohere North reasoning, text, and action channels."""

from __future__ import annotations

import json
import re
import uuid

from ..output import (
    StopSequenceMatcher,
    _safe_prefix,
    truncated_tool_record,
    within_parallel_bound,
)
from ..runtime.tool_parsers._partial_json import partial_json_name

THINK_OPEN = "<|START_THINKING|>"
THINK_CLOSE = "<|END_THINKING|>"
TEXT_OPEN = "<|START_TEXT|>"
TEXT_CLOSE = "<|END_TEXT|>"
ACTION_OPEN = "<|START_ACTION|>"
ACTION_CLOSE = "<|END_ACTION|>"
TURN_END = "<|END_OF_TURN_TOKEN|>"
_JSON_STRING = re.compile(r'"[^"\\]*(?:\\.[^"\\]*)*"', re.DOTALL)
# A model-written ``tool_call_id`` worth keeping: a short printable token.
_CALL_ID = re.compile(r"[A-Za-z0-9_.-]{1,64}", re.ASCII)


def _call_id(candidate, seen):
    """``(id, replaced)``: the model's id when well formed and new, else ours.

    mlx2 never renders ids to North, so the value is the model's invention;
    Chat, Responses and Anthropic clients key tool results by id and require
    them unique within a message, so a duplicate, an id with whitespace or
    control characters, or an overlong one is replaced.
    """
    kept = (
        isinstance(candidate, str)
        and _CALL_ID.fullmatch(candidate) is not None
        and candidate not in seen
    )
    call_id = candidate if kept else "call_" + uuid.uuid4().hex[:24]
    seen.add(call_id)
    return call_id, not kept


def _action_end(text: str) -> int:
    """Where the ``<|END_ACTION|>`` closing the action JSON in ``text`` is.

    North writes arguments with ``tojson``, which leaves ``<`` raw, and the
    action grammar admits ``<|END_ACTION|>`` (an ordinary added token) inside
    a JSON string.  Outside a string ``<`` is not JSON, so the block ends at
    the first closer outside every string.  Returns -1 while none has arrived,
    including while the text ends inside a string.
    """
    position = 0
    while True:
        close = text.find(ACTION_CLOSE, position)
        if close < 0:
            return -1
        quote = text.find('"', position, close)
        if quote < 0:
            return close
        string = _JSON_STRING.match(text, quote)
        if string is None:
            return -1
        position = string.end()


def parse_actions(text: str, tools: list[dict], *, seen_ids=None) -> list[dict]:
    """Decode one action block; ``seen_ids`` keeps ids unique across blocks."""
    definitions = {tool["function"]["name"]: tool["function"] for tool in tools}
    seen = set() if seen_ids is None else seen_ids
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Malformed North action JSON") from exc
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list) or not value:
        raise ValueError("North action block must be a nonempty JSON array")
    calls = []
    for item in value:
        if not isinstance(item, dict):
            raise TypeError("North action entries must be objects")
        name = item.get("tool_name")
        arguments = item.get("parameters", {})
        if name not in definitions:
            raise ValueError("Model called an undeclared tool")
        if not isinstance(arguments, dict):
            raise TypeError("North tool parameters must be an object")
        schema = definitions[name].get("parameters", {})
        if any(key not in arguments for key in schema.get("required", [])):
            raise ValueError("North tool call is missing a required parameter")
        if schema.get("additionalProperties") is False and any(
            key not in schema.get("properties", {}) for key in arguments
        ):
            raise ValueError("North tool call contains an undeclared parameter")
        json.dumps(arguments, allow_nan=False)
        call_id, replaced = _call_id(item.get("tool_call_id"), seen)
        calls.append(
            {"id": call_id, "name": name, "arguments": arguments, "id_replaced": replaced}
        )
    return calls


def partial_function_names(partial, tools=None):
    """Read only North's top-level action names, including array siblings."""
    body = partial.strip()
    if not body:
        return []
    if body.startswith("{"):
        name = partial_json_name(body, field="tool_name", context="North action")
        return [] if name is None else [name]
    if not body.startswith("["):
        raise ValueError("North action block must be a JSON array or object")
    names, position = [], 1
    decoder = json.JSONDecoder()
    while True:
        while position < len(body) and body[position] in " \t\n\r":
            position += 1
        if position == len(body):
            return names
        if body[position] == "]":
            if body[position + 1:].strip():
                raise ValueError("text after North action array")
            return names
        if body[position] != "{":
            raise ValueError("North action entries must be objects")
        try:
            item, end = decoder.raw_decode(body, position)
        except json.JSONDecodeError:
            name = partial_json_name(
                body[position:], field="tool_name", context="North action"
            )
            return names + ([] if name is None else [name])
        name = item.get("tool_name")
        if not isinstance(name, str) or not name:
            raise ValueError("North action needs a string tool_name")
        names.append(name)
        position = end
        while position < len(body) and body[position] in " \t\n\r":
            position += 1
        if position == len(body):
            return names
        if body[position] == "]":
            if body[position + 1:].strip():
                raise ValueError("text after North action array")
            return names
        if body[position] != ",":
            raise ValueError("North action entries need a comma")
        position += 1


def constrained_tool_grammar(tools, tool_choice, *, parallel_tool_calls=True):
    """Regex for North's JSON-in-action-tags tool-call wire format."""
    import regex

    from ..runtime.tool_parsers._schema import (
        executable_schema,
        required_parameter_names,
    )
    from ..structured_output import (
        _STRING,
        _WS,
        _schema_pattern,
        json_literal_pattern,
        recursive_json_object_pattern,
    )

    functions = [tool["function"] for tool in tools]
    if isinstance(tool_choice, dict):
        selected = tool_choice["function"]["name"]
        functions = [function for function in functions if function["name"] == selected]
    if not functions:
        raise ValueError("constrained tool_choice selects no declared function")
    # ``parse_actions`` decodes the arguments and rejects infinity.
    generic_root, definitions = recursive_json_object_pattern(finite_numbers=True)
    needs_definitions = False
    calls = []
    for function in functions:
        if function.get("strict", False):
            parameters = _schema_pattern(
                executable_schema(function.get("parameters", {})),
                finite_numbers=True,
            )
        else:
            # Free values, but required keys must appear (sglang #40051).
            required = required_parameter_names(function)
            if required:
                members = [
                    rf"{_WS}{json_literal_pattern(key)}{_WS}:(?&value)"
                    for key in required
                ]
                parameters = (
                    r"\{" + ",".join(members)
                    + rf"(?:,{_WS}{_STRING}{_WS}:(?&value))*{_WS}\}}"
                )
            else:
                parameters = generic_root
            needs_definitions = True
        # Names reach the model through Jinja's ``tojson`` (escaped) and
        # come back as its own JSON: either spelling decodes to the name.
        name = json_literal_pattern(function["name"])
        calls.append(
            rf'\{{"tool_name":{name},"parameters":{parameters}\}}'
        )
    call = "(?:" + "|".join(calls) + ")"
    array = rf"\[{call}\]" if parallel_tool_calls is False else rf"\[{call}(?:,{call})*\]"
    pattern = regex.escape(ACTION_OPEN) + array + regex.escape(ACTION_CLOSE)
    return pattern + definitions if needs_definitions else pattern


class NorthOutputParser:
    def __init__(
        self, *, chat=False, thinking=True, tools=None, stops=(),
        parallel_tool_calls=True,
    ):
        self.chat = chat
        self.tools = tools or []
        self.channel = "reasoning_content" if chat and thinking else "content"
        self.buffer = ""
        self.stop_matcher = StopSequenceMatcher(stops)
        self.stopped = False
        self.tool_count = 0
        self.tool_call_constraint_truncations = 0
        self.tool_call_id_replacements = 0
        self.truncated_tool_call = None
        self._call_ids = set()  # ids emitted this turn, model-written or ours
        self.parallel_tool_calls = bool(parallel_tool_calls)

    @property
    def stop_sequence(self):
        return self.stop_matcher.stop_sequence

    def _emit(self, text: str, *, final: bool = False) -> list[dict]:
        """Apply user stop strings only to visible content."""
        if self.channel != "content":
            return [{self.channel: text}] if text else []
        visible, stop_hit = self.stop_matcher.push(text, final=final)
        if stop_hit:
            self.stopped = True
            return [{"content": visible}] if visible else []
        return [{"content": visible}] if visible else []

    def push(self, text: str, *, final=False):
        return self._push(text, final=final, allow_incomplete_action=False)

    def finish(self, text: str, finish_reason: str):
        """Finalize, preserving max-token truncation as a length finish."""
        return self._push(
            text,
            final=True,
            allow_incomplete_action=finish_reason == "length",
        )

    def _push(
        self,
        text: str,
        *,
        final: bool,
        allow_incomplete_action: bool,
    ):
        if self.stopped:
            return []
        self.buffer += text
        events = []
        while self.buffer:
            if not self.chat:
                events.extend(self._emit(self.buffer, final=final))
                self.buffer = ""
                break
            if self.channel == "tool":
                end = _action_end(self.buffer)
                if end < 0:
                    if final:
                        if allow_incomplete_action:
                            self.truncated_tool_call = truncated_tool_record(
                                partial_function_names, self.buffer, self.tools,
                                cause="max_tokens",
                            )
                            self.buffer = ""
                            self.channel = "content"
                            break
                        raise ValueError("Model produced an incomplete North action block")
                    break
                calls = parse_actions(
                    self.buffer[:end], self.tools, seen_ids=self._call_ids
                )
                retained = within_parallel_bound(self, calls)
                # Only emitted calls carry an id: a dropped call's does not
                # count as a replacement nor stay reserved for the turn.
                for call in calls[len(retained):]:
                    self._call_ids.discard(call["id"])
                calls = retained
                self.tool_call_id_replacements += sum(
                    1 for call in calls if call["id_replaced"]
                )
                for call in calls:
                    events.append(
                        {
                            "tool_calls": [
                                {
                                    "index": self.tool_count,
                                    "id": call["id"],
                                    "type": "function",
                                    "function": {
                                        "name": call["name"],
                                        "arguments": json.dumps(call["arguments"], allow_nan=False),
                                    },
                                }
                            ]
                        }
                    )
                    self.tool_count += 1
                self.buffer = self.buffer[end + len(ACTION_CLOSE) :]
                self.channel = "content"
                continue
            markers = {
                THINK_OPEN: "reasoning_content",
                THINK_CLOSE: "content",
                TEXT_OPEN: "content",
                TEXT_CLOSE: "content",
                TURN_END: "stop",
            }
            if self.tools:
                markers[ACTION_OPEN] = "tool"
            hits = [(self.buffer.find(marker), marker) for marker in markers if marker in self.buffer]
            if hits:
                end, marker = min(hits)
                # Flush even when the marker starts the buffer: a stop-string
                # prefix held back from an earlier push must not be completed
                # by text on the far side of a channel marker.  Otherwise the
                # outcome depends on how the text was chunked (per-token pushes
                # versus several tokens per decode step).
                events.extend(self._emit(self.buffer[:end], final=True))
                if self.stopped:
                    self.buffer = ""
                    break
                self.buffer = self.buffer[end + len(marker) :]
                action = markers[marker]
                if action == "stop":
                    events.extend(self._emit("", final=True))
                    self.stopped, self.buffer = True, ""
                else:
                    self.channel = action
            else:
                end = len(self.buffer) if final else _safe_prefix(self.buffer, markers)
                if end:
                    events.extend(self._emit(self.buffer[:end], final=final))
                self.buffer = self.buffer[end:]
                break
        if final and self.channel == "tool":
            if not allow_incomplete_action:
                raise ValueError("Model produced an incomplete North action block")
            self.truncated_tool_call = truncated_tool_record(
                partial_function_names, self.buffer, self.tools, cause="max_tokens",
            )
            self.buffer = ""
            self.channel = "content"
        if final and not self.buffer and not self.stopped:
            events.extend(self._emit("", final=True))
        return events

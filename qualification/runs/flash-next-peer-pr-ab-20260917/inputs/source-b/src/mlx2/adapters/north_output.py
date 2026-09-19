"""Incremental Cohere North reasoning, text, and action channels."""

from __future__ import annotations

import json
import uuid

from ..output import _safe_prefix

THINK_OPEN = "<|START_THINKING|>"
THINK_CLOSE = "<|END_THINKING|>"
TEXT_OPEN = "<|START_TEXT|>"
TEXT_CLOSE = "<|END_TEXT|>"
ACTION_OPEN = "<|START_ACTION|>"
ACTION_CLOSE = "<|END_ACTION|>"
TURN_END = "<|END_OF_TURN_TOKEN|>"


def parse_actions(text: str, tools: list[dict]) -> list[dict]:
    definitions = {tool["function"]["name"]: tool["function"] for tool in tools}
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
        calls.append(
            {
                "id": str(item.get("tool_call_id") or ("call_" + uuid.uuid4().hex[:24])),
                "name": name,
                "arguments": arguments,
            }
        )
    return calls


class NorthOutputParser:
    def __init__(self, *, chat=False, thinking=True, tools=None, stops=()):
        self.chat = chat
        self.tools = tools or []
        self.channel = "reasoning_content" if chat and thinking else "content"
        self.buffer = self.stop_buffer = ""
        self.stops = (stops,) if isinstance(stops, str) else tuple(stops)
        self.stopped = False
        self.tool_count = 0

    def _emit(self, text: str, *, final: bool = False) -> list[dict]:
        """Apply user stop strings only to visible content."""
        if self.channel != "content":
            return [{self.channel: text}] if text else []
        self.stop_buffer += text
        hits = [
            self.stop_buffer.find(stop)
            for stop in self.stops
            if stop in self.stop_buffer
        ]
        if hits:
            end = min(hits)
            visible, self.stop_buffer = self.stop_buffer[:end], ""
            self.stopped = True
            return [{"content": visible}] if visible else []
        end = (
            len(self.stop_buffer)
            if final
            else _safe_prefix(self.stop_buffer, self.stops)
        )
        visible, self.stop_buffer = self.stop_buffer[:end], self.stop_buffer[end:]
        return [{"content": visible}] if visible else []

    def push(self, text: str, *, final=False):
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
                end = self.buffer.find(ACTION_CLOSE)
                if end < 0:
                    if final:
                        raise ValueError("Model produced an incomplete North action block")
                    break
                for call in parse_actions(self.buffer[:end], self.tools):
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
                if end:
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
        if final and not self.buffer and not self.stopped:
            events.extend(self._emit("", final=True))
        return events

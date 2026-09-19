"""Incremental Muse recipient channels and ATEM tool calls (CPU only)."""

import json
import re
import uuid

from ..output import _safe_prefix

_MESSAGE = "<|message|>"
_START = "<|start|>assistant"
_TOOL_OPEN = "<atem:function_calls>"
_TOOL_CLOSE = "</atem:function_calls>"
_HEADER = re.compile(r"(?:<\|start\|>assistant)?(?:\s*to=([\w.-]+))?\s*", re.ASCII)
_INVOKE = re.compile(r'<atem:invoke name="([\w.-]+)">(.*?)</atem:invoke>', re.DOTALL)
_PARAMETER = re.compile(
    r'<atem:parameter name="([\w.-]+)">(.*?)</atem:parameter>', re.DOTALL
)


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
    """ATEM values are raw text, with JSON for non-string schema types."""
    definitions = {tool["function"]["name"]: tool["function"] for tool in tools}
    calls, cursor = [], 0
    for match in _INVOKE.finditer(text):
        if text[cursor : match.start()].strip():
            raise ValueError("Malformed ATEM tool call")
        name, body = match.groups()
        if name not in definitions:
            raise ValueError("Model called an undeclared tool")
        schema = definitions[name].get("parameters", {})
        properties, arguments, end = schema.get("properties", {}), {}, 0
        for parameter in _PARAMETER.finditer(body):
            if body[end : parameter.start()].strip():
                raise ValueError("Malformed ATEM parameter")
            key, value = parameter.groups()
            if key in arguments:
                raise ValueError("Duplicate ATEM parameter")
            if key not in properties and schema.get("additionalProperties") is False:
                raise ValueError("Undeclared ATEM parameter")
            expected = properties.get(key, {}).get("type")
            if expected != "string":
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    if expected is not None:
                        raise ValueError(
                            "ATEM parameter must contain valid JSON"
                        ) from None
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
            arguments[key], end = value, parameter.end()
        if body[end:].strip() or any(
            key not in arguments for key in schema.get("required", [])
        ):
            raise ValueError("Malformed or missing ATEM parameters")
        # Reject non-finite JSON numbers before an API response can contain them.
        json.dumps(arguments, allow_nan=False)
        calls.append({"name": name, "arguments": arguments})
        cursor = match.end()
    if text[cursor:].strip() or not calls:
        raise ValueError("Malformed or empty ATEM block")
    return calls


class MuseOutputParser:
    def __init__(self, *, chat=False, tools=None, stops=()):
        self.chat, self.tools = chat, tools or []
        self.stops = (stops,) if isinstance(stops, str) else tuple(stops)
        self.buffer = self.stop_buffer = ""
        self.state = "header" if chat else "body"
        self.channel = "content"
        self.stopped, self.tool_count = False, 0

    def push(self, text, *, final=False):
        if self.stopped:
            return []
        self.stop_buffer += text
        matches = [
            self.stop_buffer.find(s) for s in self.stops if s in self.stop_buffer
        ]
        if matches:
            text, self.stop_buffer = self.stop_buffer[: min(matches)], ""
            self.stopped, final = True, True
        else:
            end = (
                len(self.stop_buffer)
                if final
                else _safe_prefix(self.stop_buffer, self.stops)
            )
            text, self.stop_buffer = self.stop_buffer[:end], self.stop_buffer[end:]
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
                if potential and final:
                    raise ValueError("Model produced an incomplete Muse channel header")
                if self.buffer.lstrip().startswith("<|start|>"):
                    raise ValueError("Model produced an invalid Muse channel header")
                self.state = "body"
                continue
            if self.state == "tool":
                end = self.buffer.find(_TOOL_CLOSE)
                if end < 0:
                    if final:
                        raise ValueError("Model produced an incomplete ATEM tool call")
                    break
                for call in parse_atem(self.buffer[:end], self.tools):
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
        return events

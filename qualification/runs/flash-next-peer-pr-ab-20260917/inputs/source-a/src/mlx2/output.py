"""Incremental text channels with marker and stop strings spanning chunks."""

import json
import uuid


def _safe_prefix(text, markers):
    hold = max(
        (
            n
            for marker in markers
            for n in range(1, len(marker))
            if text.endswith(marker[:n])
        ),
        default=0,
    )
    return len(text) - hold


class OutputParser:
    def __init__(
        self, *, chat=False, thinking=False, tools=None, parse_tool=None, stops=()
    ):
        self.chat, self.tools, self.parse_tool = chat, tools or [], parse_tool
        self.channel = "reasoning_content" if chat and thinking else "content"
        self.buffer = self.stop_buffer = ""
        self.stops = (stops,) if isinstance(stops, str) else tuple(stops)
        self.stopped = False
        self.tool_count = 0

    def push(self, text, *, final=False):
        if self.stopped:
            return []
        self.stop_buffer += text
        matches = [
            (self.stop_buffer.find(s), s) for s in self.stops if s in self.stop_buffer
        ]
        if matches:
            end, _ = min(matches)
            text, self.stop_buffer = self.stop_buffer[:end], ""
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
            if self.channel == "tool":
                end = self.buffer.find("</tool_call>")
                if end < 0:
                    if final:
                        raise ValueError("model produced an incomplete tool call")
                    break
                calls = self.parse_tool(self.buffer[:end], self.tools)
                calls = calls if isinstance(calls, list) else [calls]
                names = {tool["function"]["name"] for tool in self.tools}
                for call in calls:
                    if call["name"] not in names:
                        raise ValueError("model called an undeclared tool")
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
                self.buffer = self.buffer[end + len("</tool_call>") :]
                self.channel = "content"
                continue
            markers = {"<think>": "reasoning_content", "</think>": "content"}
            if self.tools:
                markers["<tool_call>"] = "tool"
            matches = [(self.buffer.find(m), m) for m in markers if m in self.buffer]
            if matches:
                end, marker = min(matches)
                if end:
                    events.append({self.channel: self.buffer[:end]})
                self.buffer = self.buffer[end + len(marker) :]
                self.channel = markers[marker]
            else:
                end = len(self.buffer) if final else _safe_prefix(self.buffer, markers)
                if end:
                    events.append({self.channel: self.buffer[:end]})
                self.buffer = self.buffer[end:]
                break
        return events

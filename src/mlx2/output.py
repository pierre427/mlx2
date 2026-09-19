"""Incremental text channels with marker and stop strings spanning chunks."""

import json
import uuid

# Reasoning-channel markers of the generic parser.  Adapters that use this
# parser derive their thinking-close token ids from the same constant, so the
# structured-output deferral point and the content channel switch coincide.
THINKING_OPEN_MARKER = "<think>"
THINKING_CLOSE_MARKER = "</think>"


def constrained_tool_choice(request):
    """Whether malformed tool output must fail closed for this request."""
    choice = request.get("tool_choice", "auto")
    return choice == "required" or isinstance(choice, dict)


class ToolCallConstraintError(ValueError):
    """A parsed tool-call sequence violates an explicit request bound."""

    tool_call_constraint_error = True


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
    """Incremental channel parser with opt-in tolerant tool markers.

    Design references: vllm#56661, vllm#57553, ollama#18467, ollama#18476.
    """

    def __init__(
        self, *, chat=False, thinking=False, tools=None, parse_tool=None, stops=(),
        constrained_tools=False, parallel_tool_calls=True,
        tolerant_tool_markers=False,
    ):
        self.chat, self.tools, self.parse_tool = chat, tools or [], parse_tool
        self.constrained_tools = bool(constrained_tools)
        self.parallel_tool_calls = bool(parallel_tool_calls)
        self.tolerant_tool_markers = bool(tolerant_tool_markers)
        self.channel = "reasoning_content" if chat and thinking else "content"
        self.buffer = self.stop_buffer = ""
        self.stops = (stops,) if isinstance(stops, str) else tuple(stops)
        self.stopped = False
        self.stop_sequence = None
        self.tool_count = 0
        self.tool_call_parse_fallbacks = 0

    def push(self, text, *, final=False):
        if self.stopped:
            return []
        self.stop_buffer += text
        matches = [
            (self.stop_buffer.find(s), s) for s in self.stops if s in self.stop_buffer
        ]
        stop_hit = bool(matches)
        if stop_hit:
            end, matched = min(matches)
            text, self.stop_buffer = self.stop_buffer[:end], ""
            self.stopped, final = True, True
            self.stop_sequence = matched
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
                    if final and stop_hit:
                        # The client's stop string cut the call short.  That is
                        # a requested stop, not a malformed model output: drop
                        # the partial call rather than fail the request.
                        self.buffer = ""
                        break
                    if final:
                        if self.constrained_tools or not self.tolerant_tool_markers:
                            raise ValueError("model produced an incomplete tool call")
                        events.append({"content": "<tool_call>" + self.buffer})
                        self.tool_call_parse_fallbacks += 1
                        self.buffer = ""
                        self.channel = "content"
                    break
                raw = "<tool_call>" + self.buffer[:end] + "</tool_call>"
                try:
                    calls = self.parse_tool(self.buffer[:end], self.tools)
                    calls = calls if isinstance(calls, list) else [calls]
                    if not calls:
                        raise ValueError("model produced an empty tool call")
                    names = {
                        tool["function"]["name"]
                        for tool in self.tools
                        if isinstance(tool, dict)
                        and isinstance(tool.get("function"), dict)
                        and isinstance(tool["function"].get("name"), str)
                    }
                    if any(call["name"] not in names for call in calls):
                        raise ValueError("model called an undeclared tool")
                    if (
                        not self.parallel_tool_calls
                        and self.tool_count + len(calls) > 1
                    ):
                        raise ToolCallConstraintError(
                            "parallel_tool_calls:false permits at most one tool call"
                        )
                    parsed_events = []
                    for index, call in enumerate(calls):
                        parsed_events.append(
                            {
                                "tool_calls": [
                                    {
                                        "index": self.tool_count + index,
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
                except (KeyError, TypeError, ValueError) as exc:
                    if getattr(exc, "tool_call_constraint_error", False):
                        raise
                    if self.constrained_tools or not self.tolerant_tool_markers:
                        raise ValueError(str(exc)) from exc
                    events.append({"content": raw})
                    self.tool_call_parse_fallbacks += 1
                    self.buffer = self.buffer[end + len("</tool_call>") :]
                    self.channel = "content"
                    continue
                events.extend(parsed_events)
                self.tool_count += len(parsed_events)
                self.buffer = self.buffer[end + len("</tool_call>") :]
                self.channel = "content"
                continue
            markers = {
                THINKING_OPEN_MARKER: "reasoning_content",
                THINKING_CLOSE_MARKER: "content",
            }
            if self.tools and (
                not self.tolerant_tool_markers
                or self.channel != "reasoning_content"
            ):
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

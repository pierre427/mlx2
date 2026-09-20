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
    """A parsed tool-call sequence violates an explicit request bound.

    Retained so a parser outside this module can still fail closed, and so the
    engine keeps its escape hatch and its ``tool_call_constraint_failures``
    counter.  The parsers in this tree no longer raise it: over-calling is a
    model behaviour the request bound already answers (see
    :func:`within_parallel_bound`), not a server fault.
    """

    tool_call_constraint_error = True


def within_parallel_bound(parser, calls):
    """Drop the calls ``parallel_tool_calls:false`` leaves no room for.

    A model that emits a second call under ``parallel_tool_calls:false`` has
    produced output the request cannot carry.  Failing the request there means
    a 5xx for a model behaviour, and on a stream the first call's deltas have
    already been written, so the failure can only arrive as a truncated stream
    carrying an error event.  Honouring the bound instead keeps every surface's
    response well formed and keeps the parser's output inside the contract that
    ``openai_compat.enforce_tool_contract`` validates terminally, so the two can
    never disagree.  The drop is counted, never silent.
    """
    if parser.parallel_tool_calls:
        return calls
    room = max(0, 1 - parser.tool_count)
    if len(calls) <= room:
        return calls
    parser.tool_call_constraint_truncations += len(calls) - room
    return calls[:room]


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


class StopSequenceMatcher:
    """Incrementally trim client stop strings and retain the exact match."""

    def __init__(self, stops=()):
        self.stops = (stops,) if isinstance(stops, str) else tuple(stops)
        self.buffer = ""
        self.stop_sequence = None

    def push(self, text, *, final=False):
        self.buffer += text
        matches = [
            (self.buffer.find(stop), stop)
            for stop in self.stops
            if stop in self.buffer
        ]
        if matches:
            end, self.stop_sequence = min(matches)
            visible, self.buffer = self.buffer[:end], ""
            return visible, True
        end = len(self.buffer) if final else _safe_prefix(self.buffer, self.stops)
        visible, self.buffer = self.buffer[:end], self.buffer[end:]
        return visible, False


class _CodeSpans:
    """Markdown code state of the content channel, fed incrementally.

    Tool-call markup inside a fenced block or an inline code span is a quoted
    example, not a call (vllm #57553, sglang #38624).  Fences follow
    CommonMark (3+ backticks or tildes at line start, at most 3 spaces of
    indent, closed by a run of the same character at least as long).  An
    inline span also ends at a newline, so a stray backtick in prose cannot
    swallow a later call.
    """

    def __init__(self):
        self.fence = None  # (character, run length) of the open fenced block
        self.inline = 0  # backtick-run length of the open inline span
        self.line_start, self.indent = True, 0
        self.run_char, self.run, self.run_at_line_start = "", 0, False

    def feed(self, text):
        for char in text:
            if char == self.run_char:
                self.run += 1
                continue
            self._end_run()
            if char in "`~":
                self.run_char, self.run = char, 1
                self.run_at_line_start, self.line_start = self.line_start, False
            elif char == "\n":
                self.inline, self.line_start, self.indent = 0, True, 0
            elif char == " " and self.line_start and self.indent < 3:
                self.indent += 1
            else:
                self.line_start = False

    def _end_run(self):
        char, length, at_line_start = self.run_char, self.run, self.run_at_line_start
        self.run_char, self.run = "", 0
        if not length:
            return
        if self.fence is not None:
            if at_line_start and char == self.fence[0] and length >= self.fence[1]:
                self.fence = None
        elif at_line_start and length >= 3 and not self.inline:
            self.fence = (char, length)
        elif char == "`":
            self.inline = 0 if self.inline == length else self.inline or length

    def in_code(self):
        """Whether the next character (a marker, never a backtick) is code."""
        self._end_run()
        return self.fence is not None or bool(self.inline)


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
        self.buffer = ""
        self.stop_matcher = StopSequenceMatcher(stops)
        self.stopped = False
        self.tool_count = 0
        self.tool_call_parse_fallbacks = 0
        self.tool_call_constraint_truncations = 0
        self._code = _CodeSpans()

    def _emit(self, events, text):
        if self.channel == "content":
            self._content(events, text)
        else:
            events.append({self.channel: text})

    def _content(self, events, text):
        events.append({"content": text})
        self._code.feed(text)

    @property
    def stop_sequence(self):
        return self.stop_matcher.stop_sequence

    def push(self, text, *, final=False):
        if self.stopped:
            return []
        text, stop_hit = self.stop_matcher.push(text, final=final)
        if stop_hit:
            self.stopped, final = True, True
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
                        self._content(events, "<tool_call>" + self.buffer)
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
                    calls = within_parallel_bound(self, calls)
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
                    self._content(events, raw)
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
                    self._emit(events, self.buffer[:end])
                if (
                    marker == "<tool_call>"
                    and self.channel == "content"
                    and self._code.in_code()
                ):
                    self._content(events, marker)  # a quoted example
                else:
                    self.channel = markers[marker]
                self.buffer = self.buffer[end + len(marker) :]
            else:
                end = len(self.buffer) if final else _safe_prefix(self.buffer, markers)
                if end:
                    self._emit(events, self.buffer[:end])
                self.buffer = self.buffer[end:]
                break
        return events

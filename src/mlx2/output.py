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


def callable_tools(request):
    """The request's tools, or None when ``tool_choice`` "none" asks for no call.

    The engine admits tools with "none" onto a route that declares no TOOLS;
    an adapter without a tool route serves that request as plain text.
    """
    if request.get("tool_choice", "auto") == "none":
        return None
    return request.get("tools") or None


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
    """Length of ``text`` that can be released without splitting a marker.

    The held tail is the longest suffix of ``text`` that is a proper prefix of
    some marker.  Only a suffix starting with the marker's first character can
    be one, so the scan visits those positions, longest suffix first, instead
    of slicing and comparing every prefix length: with 16 client stop strings
    of 256 characters that was about 200 us of host time per token.
    """
    hold = 0
    for marker in markers:
        width = min(len(marker) - 1, len(text))
        if width <= hold:
            continue
        tail = text[len(text) - width :]
        start = tail.find(marker[0])
        while start != -1 and width - start > hold:
            if marker.startswith(tail[start:]):
                hold = width - start
                break
            start = tail.find(marker[0], start + 1)
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


def stop_cut_record(partial_names, partial, tools):
    """The ``stop_truncated_tool_call`` record for a call a client stop cut.

    ``partial_names`` is the grammar's own reader of a cut-short tool-call
    body (``parse_tool.partial_function_names``): it returns the function
    names the body had committed to, in order, by the same rules its parser
    binds a completed call's name with, and raises ``ValueError`` when a name
    delimiter has been seen but the name is not one the grammar can read.
    Tool admission accepts any nonempty name up to 128 characters, so no
    generic character class can stand in for those rules (``db:lookup``,
    ``a/b``, an escaped JSON string).

    The record is ``{"function": name}`` once the name is committed (with
    ``"functions": [...]`` when the body committed to several, as a Qwen
    block with two ``<function=`` openers does), ``{"function": None}`` for
    a cut before the delimiter (the api group's case: Qwen ``<tool_call>\\n``
    cut by ``stop: ["\\n"]``), and ``{"function": None, "name_unreadable":
    True}`` for a committed name the grammar cannot read.  The terminal tool
    contract exempts the second, judges the first, and fails the third: a
    name the receipt cannot state is never ``None`` by default.  A grammar
    that publishes no reader is read the same way: nothing after the opener
    is a cut before the name, anything else is unreadable.
    """
    try:
        if partial_names is None:
            if partial.strip():
                raise ValueError("the grammar publishes no partial-name reader")
            names = []
        else:
            names = list(partial_names(partial, tools))
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("a committed function name is empty")
    except ValueError:
        return {"function": None, "name_unreadable": True}
    record = {"function": names[0] if names else None}
    if len(names) > 1:
        record["functions"] = names
    return record


def truncated_tool_record(partial_names, partial, tools, *, cause):
    """Grammar-owned evidence for an unfinished, never executable tool call."""
    if cause not in ("max_tokens", "stop_sequence"):
        raise ValueError("unsupported tool truncation cause")
    return {**stop_cut_record(partial_names, partial, tools), "cause": cause}


class OutputParser:
    """Incremental channel parser with opt-in tolerant tool markers.

    Design references: vllm#56661, vllm#57553, ollama#18467, ollama#18476.
    """

    def __init__(
        self, *, chat=False, thinking=False, tools=None, parse_tool=None, stops=(),
        constrained_tools=False, parallel_tool_calls=True,
        tolerant_tool_markers=False, think_close_separator="", partial_names=None,
    ):
        self.chat, self.tools, self.parse_tool = chat, tools or [], parse_tool
        # The grammar's reader of a stop-cut call's committed name (see
        # :func:`stop_cut_record`); a tool parser publishes its own as
        # ``partial_function_names``.
        self.partial_names = (
            partial_names
            if partial_names is not None
            else getattr(parse_tool, "partial_function_names", None)
        )
        self.constrained_tools = bool(constrained_tools)
        self.parallel_tool_calls = bool(parallel_tool_calls)
        self.tolerant_tool_markers = bool(tolerant_tool_markers)
        self.channel = "reasoning_content" if chat and thinking else "content"
        # Reasoning markers follow the vLLM Qwen3 reasoning parser: only a
        # ``<think>`` at the very start of the output opens reasoning (with
        # thinking on the template has already opened it, so a leading marker
        # is just consumed), and only the first ``</think>`` closes it.  After
        # that both are literal text: Qwen tokenizers encode them as single
        # ordinary added tokens, so a JSON grammar admits them inside strings
        # and an answer may quote them.
        self._leading = bool(chat)
        self._reasoning_open = self.channel == "reasoning_content"
        # Characters the chat template places between ``</think>`` and the
        # answer (Nemotron 3 Super: ``\n``).  The run of them directly after
        # the close marker is template structure, not content; whitespace
        # anywhere else is kept.  Empty (the default) drops nothing.  With
        # thinking off the template's generation prompt already ends with the
        # close marker, so the run at the start of the output is dropped.
        self.think_close_separator = think_close_separator
        self._drop_separator = bool(
            think_close_separator and chat and self.channel == "content"
        )
        self.buffer = ""
        self.stop_matcher = StopSequenceMatcher(stops)
        self.stopped = False
        self.tool_count = 0
        self.tool_call_parse_fallbacks = 0
        self.tool_call_constraint_truncations = 0
        # A tool call in progress that the client's stop string cut short
        # (dropped below): the :func:`stop_cut_record` of what the partial
        # call had committed to.  The receipt carries it so the terminal tool
        # contract can tell a stop that ended a call the model was making
        # (and which one) from a turn the model ended without the call it
        # owed.
        self.stop_truncated_tool_call = None
        self.truncated_tool_call = None
        self._code = _CodeSpans()
        # Whitespace-only content after a tool call is held back: the newline
        # between two parallel calls is template structure, not an answer.  It
        # is dropped at the end of the output or when the next call parses;
        # otherwise it goes out with the next text.
        self._after_call = False
        self._pending_space = ""
        self._shown_text = False
        # Set only for the duration of :meth:`finish` on a ``length`` stop.
        self._length_finish = False

    def _emit(self, events, text):
        if self.channel == "content":
            self._content(events, text)
        else:
            events.append({self.channel: text})

    def _content(self, events, text):
        if self._after_call:
            if text.isspace():
                self._pending_space += text
                return
            text = self._pending_space + text
            self._pending_space = ""
            self._after_call = False
        if not text.isspace():
            self._shown_text = True
        events.append({"content": text})
        self._code.feed(text)

    @property
    def stop_sequence(self):
        return self.stop_matcher.stop_sequence

    def _record_cut_call(self, partial, *, stop_hit):
        self.truncated_tool_call = truncated_tool_record(
            self.partial_names, partial, self.tools,
            cause="stop_sequence" if stop_hit else "max_tokens",
        )
        if stop_hit:
            self.stop_truncated_tool_call = {
                key: value for key, value in self.truncated_tool_call.items()
                if key != "cause"
            }

    def _stop_gated(self):
        """Whether buffered text is (or may still become) reasoning.

        Client stop strings apply to the answer, never to reasoning: OpenAI
        and Anthropic define them over the returned text, and the dedicated
        Xing, North and Muse parsers already match only content.  Text is held
        back from the stop matcher while reasoning is open, or while a leading
        ``<think>`` may still open it, or while the template's separator after
        ``</think>`` is still being dropped, and handed to the matcher at the
        point the answer starts.
        """
        return self.chat and (
            self.channel == "reasoning_content"
            or self._leading
            or self._drop_separator
        )

    def _match_stops(self, text, final):
        """Pass unmatched answer text through the client stop matcher."""
        visible, stop_hit = self.stop_matcher.push(text, final=final)
        if stop_hit:
            self.stopped = True
        return visible, stop_hit

    def finish(self, text, finish_reason):
        """Push the last text of a generation that ended with ``finish_reason``.

        The serving loop calls this instead of ``push(..., final=True)``.  A
        ``length`` finish (``max_tokens``) that lands inside a tool call is a
        requested stop, not malformed output: the partial call is dropped, as
        for a client stop string and in the North parser, and the finish
        reason stays ``length``.  A half-formed call is never emitted.
        """
        self._length_finish = finish_reason == "length"
        try:
            return self.push(text, final=True)
        finally:
            self._length_finish = False

    def push(self, text, *, final=False):
        if self.stopped:
            return []
        gated = self._stop_gated()
        stop_hit = False
        if not gated:
            text, stop_hit = self._match_stops(text, final)
        self.buffer += text
        final = final or stop_hit
        events = []
        while self.buffer:
            if self._drop_separator:
                self.buffer = self.buffer.lstrip(self.think_close_separator)
                if not self.buffer:
                    break  # the answer may still start with a separator
                self._drop_separator = False
            if gated and not self._stop_gated():
                # The answer starts here: everything still buffered is answer
                # text the stop matcher has not seen yet.
                gated = False
                self.buffer, stop_hit = self._match_stops(self.buffer, final)
                final = final or stop_hit
                if not self.buffer:
                    break
            if not self.chat:
                events.append({"content": self.buffer})
                self.buffer = ""
                break
            if self._leading:
                if (
                    not final
                    and len(self.buffer) < len(THINKING_OPEN_MARKER)
                    and THINKING_OPEN_MARKER.startswith(self.buffer)
                ):
                    break  # may still become the leading open marker
                self._leading = False
                if self.buffer.startswith(THINKING_OPEN_MARKER):
                    self.buffer = self.buffer[len(THINKING_OPEN_MARKER) :]
                    self.channel = "reasoning_content"
                    self._reasoning_open = True
                continue  # the answer may start here: recheck the stop gate
            if self.channel == "tool":
                first = end = self.buffer.find("</tool_call>")
                outcome = None
                while end >= 0:
                    try:
                        outcome = self.parse_tool(self.buffer[:end], self.tools)
                        break
                    except (KeyError, TypeError, ValueError) as exc:
                        if end == first:
                            outcome = exc
                        if not getattr(exc, "tool_call_close_quoted", False):
                            end = first
                            break
                    # A parser that decodes JSON values says when the closer
                    # it was cut at sits inside one of their strings (Qwen
                    # writes JSON arguments with a raw ``<``): the call runs
                    # on to a later closer, and fails at the first one unless
                    # a later one parses.
                    end = self.buffer.find("</tool_call>", end + 1)
                truncated = final and (stop_hit or self._length_finish)
                if end < 0 and first >= 0 and final and not truncated:
                    # No later closer: the call is malformed.  (Under a stop
                    # the only closer seen was quoted: the call is unfinished.)
                    end = first
                if end < 0:
                    if truncated:
                        # The client's stop string or max_tokens cut the call
                        # short.  That is a requested stop, not a malformed
                        # model output: drop the partial call (it is neither
                        # a call nor answer text) rather than fail the request.
                        self._record_cut_call(self.buffer, stop_hit=stop_hit)
                        self.buffer = ""
                        self.channel = "content"
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
                    if isinstance(outcome, BaseException):
                        raise outcome
                    calls = outcome
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
                # A parsed call confirms that pending whitespace separated two
                # calls, even if answer text appeared before the first one.
                # Keep it for the tolerant raw-content fallback above.
                self._pending_space = ""
                events.extend(parsed_events)
                self.tool_count += len(parsed_events)
                self.buffer = self.buffer[end + len("</tool_call>") :]
                self.channel = "content"
                self._after_call = True
                continue
            markers = {}
            if self._reasoning_open:
                markers[THINKING_CLOSE_MARKER] = "content"
            # Tool markup inside reasoning is the model thinking about a call,
            # never a call (vLLM parses tools from content only), under either
            # marker policy.  A call after ``</think>`` still parses.
            if self.tools and self.channel != "reasoning_content":
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
                    if marker == "<tool_call>" and not self._shown_text:
                        self._pending_space = ""
                    if marker == THINKING_CLOSE_MARKER:
                        self._reasoning_open = False
                        self._drop_separator = bool(self.think_close_separator)
                self.buffer = self.buffer[end + len(marker) :]
            else:
                end = len(self.buffer) if final else _safe_prefix(self.buffer, markers)
                if end:
                    self._emit(events, self.buffer[:end])
                self.buffer = self.buffer[end:]
                break
        if final and (stop_hit or self._length_finish) and self.channel == "tool":
            # The stop landed right after the opener: a call was in progress
            # with nothing buffered yet, so the tool branch never ran.
            self._record_cut_call(self.buffer, stop_hit=stop_hit)
        if final:
            self._pending_space = ""
        return events

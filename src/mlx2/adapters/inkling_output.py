"""Streaming Inkling message/channel framing from the checkpoint chat template."""

from __future__ import annotations

from ..output import StopSequenceMatcher, _safe_prefix

MESSAGE = "<|message_model|>"
TEXT = "<|content_text|>"
THINK = "<|content_thinking|>"
END = "<|end_message|>"
TURN_END = "<|content_model_end_sampling|>"
TOOL = "<|content_invoke_tool_json|>"
MARKERS = (MESSAGE, TEXT, THINK, END, TURN_END, TOOL)
# Inkling 2's response_template also opens content with these, and allows a
# ``[^<]*`` name after ``<|message_model|>``.
INKLING2_TEXT_MARKERS = ("<|content_tool_error|>", "<|content_invoke_tool_text|>")

# Framing the lenient (Inkling 2) parser also recognises.  None is an answer:
XML = "<|content_xml|>"  # payload dropped (tool declarations)
IMAGE = "<|content_image|>"  # payload dropped (an image slot)
PAYLOAD_DROPS = (XML, IMAGE)
BEGIN = "<|begin_of_text|>"  # skipped
SKIPPED = (BEGIN,)
# Another participant's header: the model is writing a turn that is not its
# own, so its turn ends there.
FOREIGN_HEADERS = ("<|message_user|>", "<|message_system|>", "<|message_tool|>")
TURN_ENDS = (TURN_END, "<|endoftext|>")
LENIENT_MARKERS = PAYLOAD_DROPS + SKIPPED + FOREIGN_HEADERS + ("<|endoftext|>",)


class InklingOutputParser:
    """Inkling channel framing to content / reasoning_content events.

    ``lenient`` (Inkling 2): model output never raises mid-stream.  Framing
    outside the template's response grammar is repaired the way other engines
    do (SGLang #51301, TensorRT-LLM #22626, llama.cpp #32268, ollama #19799;
    ideas only) and counted in ``framing_receipt``:

    - ``unframed_text``: text with no header (at the start, after an
      ``<|end_message|>``) is served as content;
    - ``headerless_text``: text after ``<|message_model|>`` that no content
      marker follows (so it is not a name) is served as content;
    - ``dropped_payloads``: ``<|content_xml|>`` and ``<|content_image|>``
      payloads are dropped;
    - ``skipped_markers``: ``<|begin_of_text|>`` is skipped;
    - ``foreign_header_stops``: a generated user/system/tool header, or
      ``<|endoftext|>``, ends the turn as a stop.

    A generated tool call (this route has no tool route) is served as text:
    ``<|content_invoke_tool_json|>`` and its JSON on their own line, every
    consecutive call, a terminator inside a JSON string kept as payload
    (ollama #19806 idea); the turn ends at the first thing after the calls
    that is not another call.  The default (``lenient=False``, Inkling v1)
    keeps failing closed on an unsupported header.
    """

    def __init__(self, *, chat=False, stops=(), text_markers=(), header_names=False, lenient=False):
        self.chat = chat
        self.lenient = bool(lenient)
        self.text_markers = tuple(text_markers)
        self.markers = MARKERS + self.text_markers + (LENIENT_MARKERS if self.lenient else ())
        self.header_names = header_names
        self.after_message = False  # a name may follow <|message_model|> only
        self.buffer = ""
        self.channel = "header" if chat else "content"
        self.stop_matcher = StopSequenceMatcher(stops)
        self.stopped = False
        self.tool_count = 0
        self.tool_call_constraint_truncations = 0
        self.truncated_tool_call = None
        # Tool markers this route served as text (it has no tool route).
        self.tool_markers_as_text = 0
        self.in_tool_text = False
        # Lenient state.
        self.pending = ""  # header-position text not yet classified
        self.after_tool = False  # a served tool call closed; only another may follow
        self.json_in_string = False
        self.json_escape = False
        self.served_any = False
        self.served_newline = False
        self.framing_receipt = {
            "unframed_text": 0,
            "headerless_text": 0,
            "dropped_payloads": 0,
            "skipped_markers": 0,
            "foreign_header_stops": 0,
        }

    @property
    def stop_sequence(self):
        return self.stop_matcher.stop_sequence

    @property
    def framing_repairs(self):
        return sum(self.framing_receipt.values())

    def finish(self, text, finish_reason):
        return self.push(text, final=True)

    # ----- content emission -----

    def _serve(self, text, events, *, final=False):
        """Push content through the stop matcher; True when a stop hit."""
        if text:
            self.served_any = True
            self.served_newline = text.endswith("\n")
        visible, stopped = self.stop_matcher.push(text, final=final)
        if visible:
            events.append({"content": visible})
        if stopped:
            self.stopped = True
        return stopped

    def _end_turn(self, events):
        visible, _ = self.stop_matcher.push("", final=True)
        if visible:
            events.append({"content": visible})
        self.stopped = True

    def _track_json(self, text):
        for character in text:
            if self.json_escape:
                self.json_escape = False
            elif character == "\\" and self.json_in_string:
                self.json_escape = True
            elif character == '"':
                self.json_in_string = not self.json_in_string

    def push(self, text, *, final=False):
        if self.stopped:
            return []
        if self.lenient:
            return self._push_lenient(text, final=final)
        self.buffer += text
        events = []
        while self.buffer:
            matches = [(p, m) for m in self.markers if (p := self.buffer.find(m)) >= 0]
            if matches:
                end, marker = min(matches)
            else:
                end = len(self.buffer) if final else _safe_prefix(self.buffer, self.markers)
                marker = None
            body, self.buffer = self.buffer[:end], self.buffer[end:]
            if self.channel == "header":
                if body.strip() and not (
                    self.header_names and self.after_message and "<" not in body
                ):
                    raise ValueError(
                        "unsupported Inkling generated message header (tool/media routing unavailable)"
                    )
            elif body:
                if self.channel == "content":
                    visible, stopped = self.stop_matcher.push(
                        body, final=final and marker is None
                    )
                    if visible:
                        events.append({"content": visible})
                    if stopped:
                        self.stopped = True
                        return events
                else:
                    events.append({"reasoning_content": body})
            if marker is None:
                break
            self.buffer = self.buffer[len(marker) :]
            self.after_message = marker == MESSAGE
            if self.in_tool_text:
                # The served tool message is closed: the model now hands the
                # turn to a tool result this route cannot supply, and what it
                # writes next (a tool-role header, a second call) is not an
                # answer.  End the turn here as a stop.
                visible, _ = self.stop_matcher.push("", final=True)
                if visible:
                    events.append({"content": visible})
                self.stopped = True
                break
            if marker == TURN_END:
                visible, _ = self.stop_matcher.push("", final=True)
                if visible:
                    events.append({"content": visible})
                self.stopped = True
                break
            if marker == TOOL:
                # The adapters refuse tools unless tool_choice is "none", so a
                # tool marker here was generated without a callable tool.  Per
                # output.callable_tools, a route without a tool route serves
                # that request as plain text: the JSON payload is content, and
                # the count reaches the engine counters (it used to raise
                # mid-stream and end the response with a 502).  The turn ends
                # when that tool message closes (above).
                self.tool_markers_as_text += 1
                self.in_tool_text = True
            if marker in (MESSAGE, END):
                self.channel = "header"
            else:
                self.channel = (
                    "content"
                    if marker in (TEXT, TOOL) or marker in self.text_markers or not self.chat
                    else "reasoning_content"
                )
        if final and not self.stopped:
            visible, stopped = self.stop_matcher.push("", final=True)
            if visible:
                events.append({"content": visible})
            self.stopped = stopped
        return events

    # ----- lenient (Inkling 2) -----

    def _opens_content(self, marker):
        return marker in (TEXT, THINK, TOOL) or marker in self.text_markers or marker in PAYLOAD_DROPS

    def _resolve_pending(self, marker, events, final):
        """Classify held header-position text when ``marker`` (or the end) arrives.

        Returns True when the turn ended."""
        pending, self.pending = self.pending, ""
        if not pending.strip():
            return False
        if (self.after_message and self.header_names and "<" not in pending
                and marker is not None and self._opens_content(marker)):
            return False  # a header name (the template's ``[^<]*``)
        if self.after_tool:
            self._end_turn(events)  # text after a served call: not an answer
            return True
        self.framing_receipt["headerless_text" if self.after_message else "unframed_text"] += 1
        return self._serve(pending, events, final=final and marker is None)

    def _push_lenient(self, text, *, final):
        self.buffer += text
        events = []
        while self.buffer:
            matches = [(p, m) for m in self.markers if (p := self.buffer.find(m)) >= 0]
            if matches:
                end, marker = min(matches)
            else:
                end = len(self.buffer) if final else _safe_prefix(self.buffer, self.markers)
                marker = None
            body, self.buffer = self.buffer[:end], self.buffer[end:]
            if self.channel == "header":
                self.pending += body
                if not self.after_message and not self.after_tool and self.pending.strip():
                    # Text with no header: content from here on.
                    pending, self.pending = self.pending, ""
                    self.framing_receipt["unframed_text"] += 1
                    self.channel = "content"
                    if self._serve(pending, events, final=final and marker is None):
                        return events
            elif self.channel == "content" and body:
                if self.in_tool_text:
                    self._track_json(body)
                if self._serve(body, events, final=final and marker is None):
                    return events
            elif self.channel == "reasoning_content" and body:
                events.append({"reasoning_content": body})
            # channel "drop": the payload is discarded
            if marker is None:
                break
            self.buffer = self.buffer[len(marker):]
            if self.in_tool_text and self.json_in_string and marker not in TURN_ENDS:
                # A terminator inside a JSON string is payload, not framing.
                if self._serve(marker, events):
                    return events
                continue
            if self.channel == "header" and self._resolve_pending(marker, events, final):
                return events
            if marker in SKIPPED:
                self.framing_receipt["skipped_markers"] += 1
                continue
            self.after_message = marker == MESSAGE
            if marker in TURN_ENDS or marker in FOREIGN_HEADERS:
                if marker != TURN_END and not self.after_tool:
                    # (after a served call, the tool header is expected)
                    self.framing_receipt["foreign_header_stops"] += 1
                self._end_turn(events)
                return events
            if self.in_tool_text and marker == END:
                # The served call closed; only another call may follow.
                self.in_tool_text = False
                self.after_tool = True
                self.channel = "header"
                continue
            if self.after_tool and marker not in (MESSAGE, TOOL):
                # What follows the calls is the tool turn (or anything else
                # this route cannot answer): end the turn as a stop.
                self._end_turn(events)
                return events
            if marker == TOOL:
                # No tool route here (tool_choice "none"): served as text, per
                # output.callable_tools, on its own line, every call.
                self.tool_markers_as_text += 1
                self.in_tool_text = True
                self.json_in_string = self.json_escape = False
                prefix = "\n" if self.served_any and not self.served_newline else ""
                if self._serve(prefix + TOOL, events):
                    return events
                self.channel = "content"
                continue
            if marker in (MESSAGE, END):
                self.channel = "header"
            elif marker in PAYLOAD_DROPS:
                self.framing_receipt["dropped_payloads"] += 1
                self.channel = "drop"
            elif marker == THINK and self.chat:
                self.channel = "reasoning_content"
            else:
                self.channel = "content"
        if final and not self.stopped:
            if self.channel == "header" and self._resolve_pending(None, events, True):
                return events
            if not self.stopped:
                visible, stopped = self.stop_matcher.push("", final=True)
                if visible:
                    events.append({"content": visible})
                self.stopped = stopped
        return events

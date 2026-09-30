"""Incremental harmony channel parser for GPT-OSS chat output (CPU only).

GPT-OSS writes every assistant message as ``<|start|>assistant<|channel|>NAME
<|message|>BODY<|end|>``; the answer is the ``final`` channel and ends with
``<|return|>`` (an EOS id, so it never reaches the text).  The chat prompt
already ends with ``<|start|>assistant``, so the output opens inside a header.
``analysis`` (and ``commentary``, a tool preamble) is reasoning: it goes to
``reasoning_content`` when the request shows reasoning and is dropped
otherwise.  Client stop strings apply to the final channel only.
"""

import re

from ..output import StopSequenceMatcher, _safe_prefix

_MESSAGE = "<|message|>"
_START = "<|start|>"
_TERMINATORS = ("<|end|>", "<|return|>", "<|call|>", _START)
_CHANNEL = re.compile(r"<\|channel\|>\s*([A-Za-z_]+)")
# A header is ``assistant<|channel|>final`` plus optional ``to=...`` and
# ``<|constrain|>...`` annotations; anything much longer is not a header.
_MAX_HEADER = 256


class HarmonyOutputParser:
    """Route harmony channels to ``reasoning_content`` and ``content``."""

    tool_count = 0
    tool_call_parse_fallbacks = 0
    tool_call_constraint_truncations = 0

    def __init__(self, *, chat=True, show_reasoning=True, start_in_final=False, stops=()):
        self.chat = bool(chat)
        self.show_reasoning = bool(show_reasoning)
        self.stop_matcher = StopSequenceMatcher(stops)
        self.stopped = False
        self.buffer = ""
        if not self.chat or start_in_final:
            self._state, self._body = "body", "final"
        else:
            self._state, self._body = "header", None
        self.headers = []

    @property
    def stop_sequence(self):
        return self.stop_matcher.stop_sequence

    @property
    def channel(self):
        """``content`` inside the final channel, else ``reasoning_content``."""
        return "content" if self._state == "body" and self._body == "final" else "reasoning_content"

    def finish(self, text, finish_reason):
        return self.push(text, final=True)

    def push(self, text, *, final=False):
        if self.stopped:
            return []
        self.buffer += text
        events = []
        if not self.chat:
            self._final_text(events, self.buffer, final)
            self.buffer = ""
            return events
        while self.buffer and not self.stopped:
            if self._state == "header":
                end = self.buffer.find(_MESSAGE)
                if end < 0:
                    if final or len(self.buffer) > _MAX_HEADER:
                        # Not harmony framing: keep the text as the answer.
                        self._state, self._body = "body", "final"
                        continue
                    break
                header = self.buffer[:end]
                match = _CHANNEL.search(header)
                self._body = match.group(1).lower() if match else "final"
                self.headers.append(self._body)
                self.buffer = self.buffer[end + len(_MESSAGE):]
                self._state = "body"
                continue
            if self._state == "between":
                start = self.buffer.find(_START)
                if start < 0:
                    if not final and _START.startswith(self.buffer.lstrip()):
                        break
                    if self.buffer.strip():
                        # Text after a message without a new header.
                        self._state, self._body = "body", "final"
                        continue
                    self.buffer = ""
                    break
                self.buffer = self.buffer[start + len(_START):]
                self._state = "header"
                continue
            ends = [(self.buffer.find(m), m) for m in _TERMINATORS if m in self.buffer]
            if ends:
                end, marker = min(ends)
                self._body_text(events, self.buffer[:end], True)
                self.buffer = self.buffer[end + (0 if marker == _START else len(marker)):]
                self._state = "between"
                continue
            keep = len(self.buffer) if final else _safe_prefix(self.buffer, _TERMINATORS)
            self._body_text(events, self.buffer[:keep], final)
            self.buffer = self.buffer[keep:]
            break
        if final and not self.stopped and self._body == "final":
            self._final_text(events, "", True)
        return events

    def _body_text(self, events, text, closes):
        if self._body == "final":
            self._final_text(events, text, closes)
        elif text and self.show_reasoning:
            events.append({"reasoning_content": text})

    def _final_text(self, events, text, final):
        visible, stop_hit = self.stop_matcher.push(text, final=final)
        if visible:
            events.append({"content": visible})
        if stop_hit:
            self.stopped = True

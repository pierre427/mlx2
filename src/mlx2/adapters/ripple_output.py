# SPDX-License-Identifier: Apache-2.0
"""Ripple 65B Preview output channels: optional reasoning, Hermes JSON calls.

Original mlx2 code over the shared :class:`mlx2.output.OutputParser`.  The
source chat template (provenance/ripple-65b.json) never pre-opens a think
block and writes an assistant turn as::

    <think>\\n{reasoning}\\n</think>\\n{content}\\n<tool_call>\\n{json}\\n</tool_call>...

The newlines inside that framing belong to the markers, so they reach
neither reasoning nor content:

* the newline right after a leading ``<think>`` and the one right before
  ``</think>`` are dropped from reasoning;
* the newline run right after ``</think>`` is dropped: the template writes
  one, and the model writes two before an answer (live smoke,
  qualification/runs/ripple-tools-20261105), so dropping exactly one would
  start every answer with a stray newline;
* one newline directly before a ``<tool_call>`` is dropped from content;
  whitespace between and after calls is dropped by the shared parser;
* the newlines inside a call are JSON whitespace around the body.

Each held newline is released unchanged when what follows shows it was
text, so the joined channels do not depend on how the output was chunked:
streaming and non-streaming deliver the same reasoning and content.  Output
that starts without ``<think>`` is all content.  Behaviour reference only
(no code): vLLM PR vllm-project/vllm#65745.
"""

from ..output import OutputParser


class RippleOutputParser(OutputParser):
    def __init__(self, **kwargs):
        kwargs.setdefault("think_close_separator", "\n")
        super().__init__(**kwargs)
        # The base drops a separator at the very start of a thinking-off
        # output, whose generation prompt already closed the think block.
        # Ripple's prompt closes nothing, so a leading newline is content.
        self._drop_separator = False
        self._reasoning_started = False
        self._reasoning_held = ""
        self._content_held = False
        self._content_held_calls = 0

    def _emit(self, events, text):
        if self.channel != "reasoning_content":
            # A newline still held from reasoning preceded ``</think>``.
            self._reasoning_held = ""
            super()._emit(events, text)
            return
        if not self._reasoning_started:
            if not text:
                return
            self._reasoning_started = True
            if text.startswith("\n"):
                text = text[1:]
        text = self._reasoning_held + text
        self._reasoning_held = ""
        if text.endswith("\n"):
            text, self._reasoning_held = text[:-1], "\n"
        if text:
            events.append({"reasoning_content": text})

    def _calls_seen(self):
        # A call ``parallel_tool_calls: false`` dropped still owned its
        # framing newline.
        return self.tool_count + self.tool_call_constraint_truncations

    def _release_content_newline(self, events):
        """Settle a held newline: text unless a tool call opened after it."""
        if not self._content_held:
            return
        self._content_held = False
        # A call parsed since, or one a stop/max_tokens cut, owned it.  A
        # tolerant raw-text fallback of a malformed call gets it back.
        if self._calls_seen() == self._content_held_calls and self.truncated_tool_call is None:
            # Already fed to the code-span tracker when it was held.
            events.append({"content": "\n"})

    def _content(self, events, text):
        if not self.tools:
            super()._content(events, text)
            return
        self._release_content_newline(events)
        # Whitespace right after a call is the shared parser's to hold.
        if text.endswith("\n") and not (self._after_call and text.isspace()):
            text = text[:-1]
            if text:
                super()._content(events, text)
            # The newline closes a markdown fence line; the tool-marker
            # quoting check must see it even while it is held.
            self._code.feed("\n")
            self._content_held = True
            self._content_held_calls = self._calls_seen()
            return
        if text:
            super()._content(events, text)

    def push(self, text, *, final=False):
        was_stopped = self.stopped
        events = super().push(text, final=final)
        if (final or self.stopped) and not was_stopped:
            if self.channel == "reasoning_content" and self._reasoning_held:
                events.append({"reasoning_content": self._reasoning_held})
                self._reasoning_held = ""
            self._release_content_newline(events)
        return events

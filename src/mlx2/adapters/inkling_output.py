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


class InklingOutputParser:
    def __init__(self, *, chat=False, stops=()):
        self.chat = chat
        self.buffer = ""
        self.channel = "header" if chat else "content"
        self.stop_matcher = StopSequenceMatcher(stops)
        self.stopped = False
        self.tool_count = 0
        self.tool_call_constraint_truncations = 0
        self.truncated_tool_call = None

    @property
    def stop_sequence(self):
        return self.stop_matcher.stop_sequence

    def finish(self, text, finish_reason):
        return self.push(text, final=True)

    def push(self, text, *, final=False):
        if self.stopped:
            return []
        self.buffer += text
        events = []
        while self.buffer:
            matches = [(p, m) for m in MARKERS if (p := self.buffer.find(m)) >= 0]
            if matches:
                end, marker = min(matches)
            else:
                end = len(self.buffer) if final else _safe_prefix(self.buffer, MARKERS)
                marker = None
            body, self.buffer = self.buffer[:end], self.buffer[end:]
            if self.channel == "header":
                if body.strip():
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
            if marker == TURN_END:
                visible, _ = self.stop_matcher.push("", final=True)
                if visible:
                    events.append({"content": visible})
                self.stopped = True
                break
            if marker == TOOL:
                raise ValueError("Inkling tool routing is not implemented")
            if marker in (MESSAGE, END):
                self.channel = "header"
            else:
                self.channel = (
                    "content"
                    if marker == TEXT or not self.chat
                    else "reasoning_content"
                )
        if final and not self.stopped:
            visible, stopped = self.stop_matcher.push("", final=True)
            if visible:
                events.append({"content": visible})
            self.stopped = stopped
        return events

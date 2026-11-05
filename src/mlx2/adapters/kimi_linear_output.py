"""Kimi Linear chat output: content plus Kimi K2 tool-call blocks.

Calls arrive between ``<|tool_call_begin|>`` and ``<|tool_call_end|>``
inside a ``<|tool_calls_section_*|>`` wrapper (see
:mod:`mlx2.runtime.tool_parsers.kimi`).  ``KimiOutputParser`` is
:class:`mlx2.output.OutputParser` with those call markers; the wrapper is
template structure and never reaches content, even when a stream splits a
marker across pushes; a call body (its JSON string arguments) keeps it.  Kimi Linear Instruct has no reasoning channel.
"""

from __future__ import annotations

import copy
import re

from ..output import OutputParser, _safe_prefix, callable_tools, constrained_tool_choice
from ..runtime.tool_parsers.kimi import (
    ARGUMENT_BEGIN, CALL_BEGIN, CALL_END, SECTION_MARKERS, parse_tool_call,
)

# Wire structure the client stop matcher never sees (it still reaches the
# parser verbatim, and a call body keeps a quoted one).
WIRE_MARKERS = (*SECTION_MARKERS, CALL_BEGIN, ARGUMENT_BEGIN, CALL_END)

# Kimi's template writes each past call's *id* as its header
# (``<|tool_call_begin|>{id}<|tool_call_argument_begin|>``), so ids must be
# ``functions.<name>:<n>``: an OpenAI-style ``call_<hex>`` in history teaches
# the model to head new calls with an id instead of a function name.
KIMI_CALL_ID = re.compile(r"^functions\.[^:\s]+:\d+$")


def _call_name(call):
    function = call.get("function", call) if isinstance(call, dict) else {}
    return function.get("name") if isinstance(function, dict) else None


def history_call_count(messages):
    return sum(len(m.get("tool_calls") or ()) for m in messages or () if isinstance(m, dict)
               and m.get("role") == "assistant")


def normalize_call_ids(messages):
    """Deep copy with every assistant call id in Kimi's ``functions.<name>:<n>``
    form (n counts calls across the conversation) and each tool message's
    ``tool_call_id`` rewritten to match. Already-Kimi ids are kept."""
    messages = copy.deepcopy(messages)
    mapping, count = {}, 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or ():
                if not isinstance(call, dict):
                    continue
                old, name = call.get("id"), _call_name(call)
                if not (isinstance(old, str) and KIMI_CALL_ID.match(old)) and isinstance(name, str) and name:
                    new = f"functions.{name}:{count}"
                    if isinstance(old, str):
                        mapping[old] = new
                    call["id"] = new
                count += 1
        elif message.get("role") == "tool" and message.get("tool_call_id") in mapping:
            message["tool_call_id"] = mapping[message["tool_call_id"]]
    return messages


class KimiOutputParser(OutputParser):
    tool_open = CALL_BEGIN
    tool_close = CALL_END
    # Dropped from answer text only: a JSON string argument inside a call
    # body may quote a wrapper marker and keeps it.
    content_structure_markers = SECTION_MARKERS

    def __init__(self, *, call_id_offset=0, **kwargs):
        super().__init__(**kwargs)
        self._held = ""
        self._stop_raw = ""
        self.section_markers_stripped = 0
        self.call_id_offset = call_id_offset

    def _tool_call_id(self, call, index):
        # Continues the conversation's call numbering, as the template and
        # the model's own headers do.
        return f"functions.{call['name']}:{self.call_id_offset + index}"

    def _drop_structure_marker(self, marker):
        self.section_markers_stripped += 1

    def _match_stops(self, text, final):
        """Match client stop strings over the text with the wire markers removed.

        The released text keeps the markers in place.  A trailing proper
        prefix of a marker is held until the next push decides it, as is
        the stop matcher's own partial-match tail (with any markers inside).
        A stop string spelled across a marker matches as if it were absent.
        """
        stops = self.stop_matcher.stops
        if not stops:
            return text, False
        raw = self._stop_raw + text
        plain, where, index, tail = [], [], 0, len(raw)
        while index < len(raw):
            marker = next((m for m in WIRE_MARKERS if raw.startswith(m, index)), None)
            if marker is not None:
                index += len(marker)
                continue
            if not final and raw[index] == "<" and any(m.startswith(raw[index:]) for m in WIRE_MARKERS):
                tail = index
                break
            plain.append(raw[index])
            where.append(index)
            index += 1
        plain = "".join(plain)
        where.append(tail)
        matches = [(plain.find(stop), stop) for stop in stops if stop in plain]
        if matches:
            end, self.stop_matcher.stop_sequence = min(matches)
            self._stop_raw = ""
            self.stopped = True
            return raw[:where[end]], True
        cut = where[len(plain) if final else _safe_prefix(plain, stops)]
        self._stop_raw = raw[cut:]
        return raw[:cut], False

    def push(self, text, *, final=False):
        if self.chat:
            return super().push(text, final=final)
        # A raw completion has no tool channel: strip the wrapper as text,
        # holding a trailing proper prefix of a marker for the next push.
        text = self._held + text
        self._held = ""
        for marker in SECTION_MARKERS:
            self.section_markers_stripped += text.count(marker)
            text = text.replace(marker, "")
        if not final:
            hold = 0
            for marker in SECTION_MARKERS:
                for size in range(min(len(marker) - 1, len(text)), hold, -1):
                    if text.endswith(marker[:size]):
                        hold = size
                        break
            if hold:
                text, self._held = text[:-hold], text[-hold:]
        return super().push(text, final=final)


def make_output_parser(request):
    if request.get("enable_thinking"):
        raise ValueError("Kimi Linear Instruct declares no reasoning channel")
    return KimiOutputParser(
        call_id_offset=history_call_count(request.get("messages")),
        chat="messages" in request, thinking=False, tools=callable_tools(request),
        parse_tool=parse_tool_call, stops=request.get("stop", ()),
        parallel_tool_calls=request.get("parallel_tool_calls", True),
        tolerant_tool_markers=request.get("_tolerant_tool_markers", False),
    )


def tool_constraint(request):
    # No forced-call grammar is staged for the Kimi wire: refuse rather than
    # silently serving "auto" for "required" or a named function.
    if callable_tools(request) and constrained_tool_choice(request):
        raise ValueError("Kimi Linear serves tool_choice auto/none only; forced tool choice is not staged")
    return None

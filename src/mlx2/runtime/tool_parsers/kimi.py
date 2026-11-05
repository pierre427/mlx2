# SPDX-License-Identifier: Apache-2.0
# Wire format and name rule adapted from Rapid-MLX vllm_mlx/tool_parsers/
# kimi_tool_parser.py (Apache-2.0, raullenchai/Rapid-MLX, read at be5d8bd65 of a local fork);
# see provenance/kimi-linear-tools.json. Rewritten for mlx2's OutputParser.
"""Kimi K2 / Kimi Linear tool calls.

The chat template writes each call as::

    <|tool_calls_section_begin|>
    <|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"city": "Paris"}<|tool_call_end|>
    <|tool_calls_section_end|>

``OutputParser`` cuts the text between ``<|tool_call_begin|>`` and
``<|tool_call_end|>``; :func:`parse_tool_call` reads one such body.  The id
before the argument marker is ``functions.<name>:<index>`` (the template
echoes the client's call id there for history, so a bare name or another
prefix is read the same way: after ``functions.``, the longest dotted suffix
before ``:<digits>`` that names a declared tool, else the last segment).
Arguments are one JSON object.
"""

from __future__ import annotations

import json
import re

ARGUMENT_BEGIN = "<|tool_call_argument_begin|>"
CALL_BEGIN = "<|tool_call_begin|>"
CALL_END = "<|tool_call_end|>"
SECTION_BEGIN = "<|tool_calls_section_begin|>"
SECTION_END = "<|tool_calls_section_end|>"
# Section markers some Kimi releases spell in the singular.
SECTION_MARKERS = (SECTION_BEGIN, SECTION_END,
                   "<|tool_call_section_begin|>", "<|tool_call_section_end|>")
_INDEX = re.compile(r":\d+$")
# The K2 vocabulary's ids for the markers above, as the 48B q8 artifact's
# tokenizer_config.json names them (mlx-community 553a7a56).
K2_TOOL_MARKER_IDS = {163595: SECTION_BEGIN, 163596: SECTION_END, 163597: CALL_BEGIN,
                      163598: ARGUMENT_BEGIN, 163599: CALL_END}


class UnclosedJSONString(ValueError):
    """The arguments end inside a JSON string: the cut closer was quoted."""

    tool_call_close_quoted = True


def _declared(tools):
    return {tool["function"]["name"] for tool in tools or ()
            if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
            and isinstance(tool["function"].get("name"), str)}


def _name(call_id, tools=None):
    name = _INDEX.sub("", call_id.strip()).strip()
    name = name.removeprefix("functions.")
    # A declared name may itself be dotted (``web.search``): the longest
    # dotted suffix that names a declared tool wins, then the last segment.
    declared = _declared(tools)
    parts = name.split(".")
    for start in range(len(parts)):
        candidate = ".".join(parts[start:]).strip()
        if candidate in declared:
            return candidate
    name = parts[-1].strip()
    if not name:
        raise ValueError("Kimi tool call has no function name")
    return name


def _unclosed_string(text):
    quoted, escaped = False, False
    for char in text:
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = quoted
        elif char == '"':
            quoted = not quoted
    return quoted


def parse_tool_call(text, tools=None):
    """Parse the body of one ``<|tool_call_begin|>`` block."""
    head, marker, arguments = text.partition(ARGUMENT_BEGIN)
    if not marker:
        raise ValueError("Kimi tool call has no argument marker")
    name = _name(head, tools)
    arguments = arguments.strip()
    if not arguments:
        return {"name": name, "arguments": {}}
    try:
        value = json.loads(arguments)
    except ValueError as exc:
        if _unclosed_string(arguments):
            raise UnclosedJSONString("Kimi tool arguments end inside a JSON string") from exc
        raise ValueError("Kimi tool arguments are not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Kimi tool arguments must be a JSON object")
    return {"name": name, "arguments": value}


def partial_function_names(partial, tools=None):
    """The function a cut-short call body had committed to: named once the
    argument marker has been written, nothing before it."""
    head, marker, _ = partial.partition(ARGUMENT_BEGIN)
    return [_name(head, tools)] if marker else []


parse_tool_call.partial_function_names = partial_function_names

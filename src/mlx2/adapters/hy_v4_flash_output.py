"""Hy4-flash chat output: reasoning, content and ``<tool_call:S>`` blocks.

Original mlx2 code.  ``HYV4OutputParser`` is :class:`mlx2.output.OutputParser`
with Hy4's suffixed call and reasoning markers; the ``<tool_calls:S>`` wrapper
is template structure dropped from answer text (as Kimi's section wrapper).
The base parser buffers markers across pushes, so a delta carrying several
tags (batched or speculative decode, vllm#55550) parses the same as one tag
per delta.

Reasoning follows vLLM's ``HYV4ReasoningParser`` (PR 68296, read not copied):
with ``reasoning_effort`` ``high`` the generation prompt ends with
``<think:S>`` open, so the output is ``reasoning</think:S>answer`` and
everything before the first ``</think:S>`` is ``reasoning_content``.  Unlike
vLLM, whose default follows the template (``high``), mlx2 renders
``no_think`` unless the request opts in: in ``high`` mode Hy4 can spend its
whole token budget reasoning (vllm#66454: 67.9% invalid GSM8K answers).
"""

from __future__ import annotations

import threading

from ..output import OutputParser, callable_tools, constrained_tool_choice
from ..runtime.tool_parsers.hy_v4 import (
    DEFAULT_SUFFIX,
    constrained_tool_grammar,
    make_parser,
    markers,
)

# The template's levels.  ``no_think`` is mlx2's default.  Upstream dropped
# ``max`` in tencent/Hy4-flash@b2498eba (2026-11-19); that template raises on
# it, so mlx2 refuses it before rendering rather than mapping it to ``high``.
TEMPLATE_EFFORTS = ("no_think", "high")


def reasoning_effort(request) -> str:
    """The template ``reasoning_effort`` a chat request renders with.

    ``no_think`` unless the request opts in: ``enable_thinking: true`` alone
    selects ``high``; ``reasoning_effort`` ``high`` selects it.  An explicit ``enable_thinking: false`` wins over an effort level.
    Other effort names have no template level and are refused, as is
    ``enable_thinking: true`` with effort ``none``.
    """
    thinking = request.get("enable_thinking")
    effort = request.get("reasoning_effort")
    if thinking is False:
        return "no_think"
    if effort is None:
        return "high" if thinking else "no_think"
    if effort in ("none", "no_think"):
        if thinking:
            raise ValueError("Hy4-flash enable_thinking=true disagrees with reasoning_effort none")
        return "no_think"
    if effort == "high":
        return effort
    if effort == "max":
        raise ValueError("Hy4-flash reasoning_effort max was removed upstream "
                         "(tencent/Hy4-flash@b2498eba); use high")
    raise ValueError("Hy4-flash reasoning_effort must be none (no_think, the default) or high")


class FeatureCounters:
    """Adapter diagnostics counters; serving calls the hooks from HTTP and
    generation threads."""

    NAMES = ("reasoning_requests", "reasoning_emitted", "tool_grammar_hook_calls")

    def __init__(self):
        self._lock = threading.Lock()
        self._counts = dict.fromkeys(self.NAMES, 0)

    def add(self, name):
        with self._lock:
            self._counts[name] += 1

    def snapshot(self):
        with self._lock:
            return dict(self._counts)


def thinking_enabled(request) -> bool:
    if "messages" not in request:
        return False
    try:
        return reasoning_effort(request) != "no_think"
    except ValueError:
        return False  # refused where the prompt is rendered


class HYV4OutputParser(OutputParser):
    def __init__(self, *, suffix: str = DEFAULT_SUFFIX, stats=None, **kwargs):
        m = markers(suffix)
        self.tool_open = m["call_open"]
        self.tool_close = m["call_close"]
        self.content_structure_markers = (m["calls_open"], m["calls_close"])
        self.thinking_open_marker = f"<think{suffix}>"
        self.thinking_close_marker = f"</think{suffix}>"
        self._stats = stats
        self._reasoning_seen = False
        kwargs.setdefault("parse_tool", make_parser(suffix))
        super().__init__(**kwargs)
        if self.channel == "content":
            # no_think (vLLM HYV4ReasoningExtractor thinking=False): the
            # prompt already closed the think block, so the whole output is
            # content and a leading ``<think:S>`` is text, not a channel.
            self._leading = False

    def _emit(self, events, text):
        if (self.channel == "reasoning_content" and text and not self._reasoning_seen
                and self._stats is not None):
            self._reasoning_seen = True
            self._stats.add("reasoning_emitted")
        super()._emit(events, text)


def output_parser(request, *, suffix=DEFAULT_SUFFIX, stats=None):
    chat = "messages" in request
    thinking = chat and reasoning_effort(request) != "no_think"
    if thinking and stats is not None:
        stats.add("reasoning_requests")
    return HYV4OutputParser(
        suffix=suffix,
        stats=stats,
        chat=chat,
        thinking=thinking,
        tools=callable_tools(request),
        stops=request.get("stop", ()),
        constrained_tools=constrained_tool_choice(request),
        parallel_tool_calls=request.get("parallel_tool_calls", True),
        tolerant_tool_markers=request.get("_tolerant_tool_markers", False),
    )


def tool_constraint(request, *, suffix=DEFAULT_SUFFIX, stats=None):
    """The forced-call grammar for ``required`` / a named function, else None.

    Unrepresentable tools (a name or key holding markup, a strict schema the
    wire cannot bound) raise ``ValueError``: serving refuses the forced
    request rather than decoding it unconstrained.
    """
    tools = callable_tools(request)
    if not tools or not constrained_tool_choice(request):
        return None
    grammar = constrained_tool_grammar(
        tools, request["tool_choice"],
        parallel_tool_calls=request.get("parallel_tool_calls", True), suffix=suffix)
    if stats is not None:
        # Hook calls, not requests: serving asks once to precompile and once
        # at admission, and the auto-shape planner asks for auto requests too.
        stats.add("tool_grammar_hook_calls")
    return grammar

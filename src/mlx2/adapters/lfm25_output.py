"""LFM2.5 tool calling: output channels, tool grammar and template inputs.

LFM2.5 chat output is ``reasoning</think>answer`` (the generation prompt ends
with ``<think>``), optionally followed by pythonic tool-call blocks::

    <|tool_call_start|>[get_weather(city='Paris'), search(q='x', n=3)]<|tool_call_end|>

The two markers and ``</think>`` are single, non-special added tokens (ids
124905, 124906, 124902), so they reach the parser as text.  The wire format is
parsed by :mod:`mlx2.runtime.tool_parsers.lfm_pythonic`.

Public API for the adapter (``adapters/lfm25_moe.py``)::

    TOOL_CALL_START, TOOL_CALL_END        # marker text
    TOOL_CALL_START_ID, TOOL_CALL_END_ID  # their token ids in the 8-bit artifact

    validate_tool_request(request) -> None
        Raise ValueError for a declared tool or required parameter name the
        wire cannot carry (call from the adapter's request validation).
    normalize_messages_for_template(messages) -> list[dict]
        Deep copy with assistant ``tool_calls[*].function.arguments`` decoded
        from OpenAI JSON strings to dicts (the template refuses strings).
    render_chat(tokenizer, request, *, tokenize) -> list[int] | str
        ``apply_chat_template`` with the normalized messages, the callable
        tools (``tools=``; none under ``tool_choice: "none"``) and the
        generation prompt.
    make_output_parser(request, *, thinking=None) -> LfmOutputParser
        The ``output_parser`` hook.
    make_tool_constraint(request) -> str | None
        The ``tool_constraint`` hook: the forced-call block grammar for
        ``required`` / named ``tool_choice``, else None.

``LfmOutputParser`` is :class:`mlx2.output.OutputParser` with LFM markers: the
same channel, stop-string, code-span, parallel-bound, fallback and truncation
behaviour, so every receipt field the serving loop reads
(``tool_count``, ``tool_call_parse_fallbacks``,
``tool_call_constraint_truncations``, ``truncated_tool_call``,
``stop_truncated_tool_call``, ``stop_sequence``) means the same thing.  It
overrides only the ``tool_open``/``tool_close`` markers;
``tests/test_lfm25_tools_cpu.py`` checks the two agree event for event.
"""

from __future__ import annotations

import copy
import json

from ..output import (
    THINKING_CLOSE_MARKER,
    OutputParser,
    callable_tools,
    constrained_tool_choice,
)
from ..runtime.tool_parsers.lfm_pythonic import (
    constrained_tool_grammar,
    parse_tool_call,
    representable_name,
)
from ..runtime.tool_parsers.lfm_pythonic import (
    tool_call_end as TOOL_CALL_END,
)
from ..runtime.tool_parsers.lfm_pythonic import (
    tool_call_start as TOOL_CALL_START,
)

TOOL_CALL_START_ID = 124905
TOOL_CALL_END_ID = 124906
THINK_CLOSE_ID = 124902

__all__ = [
    "THINK_CLOSE_ID",
    "TOOL_CALL_END",
    "TOOL_CALL_END_ID",
    "TOOL_CALL_START",
    "TOOL_CALL_START_ID",
    "LfmOutputParser",
    "make_output_parser",
    "make_tool_constraint",
    "normalize_messages_for_template",
    "render_chat",
    "validate_tool_request",
]


def validate_tool_request(request: dict) -> None:
    """Fail closed on tools whose calls the LFM wire cannot carry.

    The template writes function and argument names raw, so a name holding
    whitespace or one of ``()[]{},'"=`` would come back unparseable.  Every
    declared parameter is checked: the model sees optional ones in the tool
    list too, and a grammar that silently omits one advertises a contract
    the route cannot carry.
    """

    for tool in request.get("tools") or ():
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict):
            continue  # admission validates the tool shape
        if not representable_name(function.get("name")):
            raise ValueError(
                f"tool name {function.get('name')!r} is not representable in "
                "LFM2.5 pythonic tool calls"
            )
        parameters = function.get("parameters")
        properties = (
            parameters.get("properties") if isinstance(parameters, dict) else None
        )
        for name in properties if isinstance(properties, dict) else ():
            if not representable_name(name):
                raise ValueError(
                    f"tool parameter name {name!r} is not representable in "
                    "LFM2.5 pythonic tool calls"
                )


def normalize_messages_for_template(messages: list) -> list:
    """Messages the LFM template can render, without mutating the request.

    OpenAI clients send prior assistant calls with ``arguments`` as a JSON
    string; the template renders each argument by type and raises on a
    string.  Arguments are decoded here and must be a JSON object; an empty
    string or ``null`` is a call without arguments.
    """
    messages = copy.deepcopy(messages)
    for message in messages:
        if not isinstance(message, dict):
            continue
        calls = message.get("tool_calls")
        if not calls:
            continue
        if not isinstance(calls, list):
            raise ValueError("assistant tool_calls must be a list")  # noqa: TRY004
        for call in calls:
            if not isinstance(call, dict):
                raise ValueError("assistant tool_calls entries must be objects")  # noqa: TRY004
            function = call.get("function", call)
            if not isinstance(function, dict):
                raise ValueError("assistant tool call function must be an object")  # noqa: TRY004
            arguments = function.get("arguments")
            if arguments is None or (
                isinstance(arguments, str) and not arguments.strip()
            ):
                arguments = {}
            elif isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError as exc:
                    raise ValueError(
                        "assistant tool call arguments are not valid JSON"
                    ) from exc
            if not isinstance(arguments, dict):
                raise ValueError("assistant tool call arguments must be a JSON object")  # noqa: TRY004
            function["arguments"] = arguments
    return messages


def render_chat(tokenizer, request: dict, *, tokenize: bool):
    """Render a chat request through the LFM2.5 template (ids or text)."""
    return tokenizer.apply_chat_template(
        normalize_messages_for_template(request["messages"]),
        tools=callable_tools(request),
        add_generation_prompt=True,
        tokenize=tokenize,
    )


def make_output_parser(request: dict, *, thinking=None):
    """The adapter's ``output_parser`` hook.

    ``thinking`` defaults to on for chat: the LFM2.5 generation prompt always
    ends with ``<think>``.
    """
    chat = "messages" in request
    return LfmOutputParser(
        chat=chat,
        thinking=chat if thinking is None else bool(thinking),
        tools=callable_tools(request),
        parse_tool=parse_tool_call,
        stops=request.get("stop", ()),
        constrained_tools=constrained_tool_choice(request),
        parallel_tool_calls=request.get("parallel_tool_calls", True),
        tolerant_tool_markers=request.get("_tolerant_tool_markers", False),
        # The model writes ``</think>\n\n`` before its answer; the template
        # trims content when it renders history, so those newlines are
        # structure, not the answer (as Nemotron 3 Super declares).
        think_close_separator="\r\n",
        hold_late_reasoning=chat and thinking is False,
    )


def make_tool_constraint(request: dict):
    """The adapter's ``tool_constraint`` hook (forced calls only)."""
    if not constrained_tool_choice(request) or not request.get("tools"):
        return None
    return constrained_tool_grammar(
        request["tools"],
        request["tool_choice"],
        parallel_tool_calls=request.get("parallel_tool_calls", True),
    )


class LfmOutputParser(OutputParser):
    """:class:`OutputParser` whose tool blocks use the LFM markers.

    With reasoning off the prompt already closes the think channel, yet the
    always-reasoning model sometimes reasons anyway and closes it again with
    its own ``</think>`` (2026-10-14 ladder: "We need to read the shared
    material ... </think>NELL-7D13C8-3").  Text before that marker cannot be
    told from a direct answer until the marker arrives or the output ends, so
    ``hold_late_reasoning`` holds the output back until one of the two
    (Pierre, 2026-10-15): a marker turns the held text into reasoning and the
    rest streams as the answer; the end of output releases it as the answer.
    """

    tool_open = TOOL_CALL_START
    tool_close = TOOL_CALL_END

    def __init__(self, *args, hold_late_reasoning=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._holding = bool(hold_late_reasoning) and self.channel == "content"
        self._held = ""
        self.late_reasoning_chars = 0

    def push(self, text, *, final=False):
        if not self._holding:
            return super().push(text, final=final)
        self._held += text
        index = self._held.find(THINKING_CLOSE_MARKER)
        if index < 0 and not final:
            return []
        self._holding = False
        held, self._held = self._held, ""
        if index < 0:
            return super().push(held, final=final)
        reasoning = held[:index]
        self.late_reasoning_chars = len(reasoning)
        events = [{"reasoning_content": reasoning}] if reasoning else []
        # The answer after the model's own close starts like any answer after
        # a close: its separator run is template structure.
        self._drop_separator = bool(self.think_close_separator)
        return events + super().push(held[index + len(THINKING_CLOSE_MARKER):], final=final)

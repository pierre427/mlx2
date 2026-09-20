"""Whole-output tool grammars composed from adapter tool-call blocks (item 12).

An adapter's ``tool_constraint`` hook returns the grammar of one tool-call
*block* in its wire format (Qwen ``<tool_call>`` elements, North's
``<|START_ACTION|>`` array, Muse's ``<atem:function_calls>`` element), already
carrying the call-count quantifier: one call when ``parallel_tool_calls`` is
false, one or more otherwise.  This module wraps that block, model-agnostically,
into the language of the whole answer:

* forced (``required`` / named): the block itself -- ``+`` or exactly one;
* ``auto`` with a strict tool: free text, optionally followed by one block and
  a text tail -- ``*`` calls when parallel, ``?`` when single;
* any mode with a JSON ``response_format``: the block *or* the JSON answer.

Free text is "any text that never contains the adapter's tool-call opener",
written without lookaround so the exact token automaton can compile it.  The
thinking channel is not part of the grammar: the processor still defers the
whole language past the adapter's thinking-close marker (``defer_until``), so
the thinking guard and budget processors compose unchanged.

Design reference: Splash ``server/tool_schema.py`` ``tool_grammar`` (Apache-2.0,
rev f58d36dd), which expresses the same shapes in llguidance Lark.
"""

from __future__ import annotations

import regex

# Receipt shapes.
SHAPE_CALLS = "calls"
SHAPE_TEXT_OR_CALLS = "text_or_calls"
SHAPE_CALLS_OR_ANSWER = "calls_or_answer"


def _char(character):
    point = ord(character)
    return rf"\u{point:04x}" if point <= 0xFFFF else rf"\U{point:08x}"


def text_excluding(literal):
    """Regex for any text (including empty) that never contains ``literal``.

    When the opener's first character does not recur inside it (true of every
    shipped tool-call opener) the language is written as blocks: a character
    other than the first, or a partial opener that breaks off.  A partial
    opener broken off by the first character again restarts a new block, so
    ``(?:D*B)`` covers runs such as ``<to<tool``.  Other literals fall back to
    a lookahead, which only the scanner engine enforces.
    """
    if not isinstance(literal, str) or not literal:
        raise ValueError("tool-call opener must be a nonempty string")
    first, rest = literal[0], literal[1:]
    if first in rest:
        return rf"(?:(?!{regex.escape(literal)})[\s\S])*"
    if not rest:
        return rf"[^{_char(first)}]*"
    partials = [_char(first) + "".join(map(_char, rest[:k])) for k in range(len(rest))]
    dangling = "(?:" + "|".join(partials) + ")"
    broken = "(?:" + "|".join(
        rf"{partial}[^{_char(rest[k])}{_char(first)}]"
        for k, partial in enumerate(partials)
    ) + ")"
    return rf"(?:[^{_char(first)}]|{dangling}*{broken})*{dangling}*"


def _split_definitions(pattern):
    """``(body, definitions)``: strip the shared recursive-JSON DEFINE block."""
    from .structured_output import recursive_json_object_pattern

    definitions = recursive_json_object_pattern()[1]
    if pattern.endswith(definitions):
        return pattern[: -len(definitions)], definitions
    if "(?(DEFINE)" in pattern:
        raise ValueError("tool grammar carries definitions that cannot be merged")
    return pattern, ""


def compose_tool_grammar(block, *, shape, open_marker=None, answer=None, lead=""):
    """The whole-output pattern for ``shape`` around one adapter ``block``."""
    body, definitions = _split_definitions(block)
    if shape == SHAPE_CALLS:
        return block
    if shape == SHAPE_TEXT_OR_CALLS:
        text = text_excluding(open_marker)
        pattern = rf"{text}(?:{body}{text})?"
    elif shape == SHAPE_CALLS_OR_ANSWER:
        answer_body, answer_definitions = _split_definitions(answer)
        definitions = definitions or answer_definitions
        pattern = rf"(?:{lead}(?:{body})|{answer_body})"
    else:
        raise ValueError(f"unknown tool grammar shape {shape!r}")
    return pattern + definitions


def call_quantifier(forced, parallel):
    """How many calls the engaged grammar admits, for the receipt."""
    if forced:
        return "+" if parallel else "1"
    return "*" if parallel else "?"


def plan_tool_grammar(request, accessor, *, open_marker=None, leading_whitespace=False):
    """Decide the extended (``constrained_tool_grammar_auto``) tool grammar.

    Returns ``(pattern, status, receipt)``.  ``pattern`` is None unless the
    status is ``engaged``; ``receipt`` then describes the engaged language.
    ``status`` is ``disabled`` when the request carries no tool contract the
    grammar could enforce (tools absent or ``none``, or plain non-strict
    ``auto`` without a JSON answer), which leaves the request untouched.
    """
    from .structured_output import _WS, compile_constraint

    tools = request.get("tools") or ()
    choice = request.get("tool_choice", "auto")
    if not tools or choice == "none":
        return None, "disabled", None
    forced = choice == "required" or isinstance(choice, dict)
    answer_format = request.get("response_format")
    if answer_format == {"type": "text"}:
        answer_format = None
    strict = any(tool["function"].get("strict", False) for tool in tools)
    if not forced and not strict and answer_format is None:
        return None, "disabled", None
    if "grammar" in request or request.get("min_tokens", 0):
        return None, "skipped_request_combination", None
    if not callable(accessor):
        return None, "skipped_adapter_unsupported", None
    if forced:
        shape = SHAPE_CALLS
    elif answer_format is not None:
        shape = SHAPE_CALLS_OR_ANSWER
    else:
        shape = SHAPE_TEXT_OR_CALLS
        if not open_marker:
            return None, "skipped_adapter_unsupported", None
    try:
        # The adapter hook builds forced blocks only; ``auto`` asks it for the
        # block over every declared tool, which is what a forced ``required``
        # call would admit.
        block = accessor(request if forced else {**request, "tool_choice": "required"})
    except ValueError:
        return None, "skipped_grammar_unrepresentable", None
    if not block:
        return None, "skipped_adapter_unsupported", None
    answer = None
    if shape == SHAPE_CALLS_OR_ANSWER:
        # The answer is exactly the response_format language (validated the
        # same way); both alternatives get the same deferred-whitespace lead.
        answer = compile_constraint(
            answer_format, leading_whitespace=leading_whitespace
        ).pattern.pattern
    pattern = compose_tool_grammar(
        block, shape=shape, open_marker=open_marker, answer=answer,
        lead=_WS if leading_whitespace else "",
    )
    parallel = request.get("parallel_tool_calls", True) is not False
    return pattern, "engaged", {
        "shape": shape,
        "calls": call_quantifier(forced, parallel),
    }

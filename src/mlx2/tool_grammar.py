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
    """One character of a literal, spelled for inside and outside a class.

    Printable ASCII is escaped as itself (an exclusion is spelled once per
    partial literal, so the ``\\uXXXX`` form would be six characters each);
    everything else is a code point escape.
    """
    point = ord(character)
    if 0x21 <= point < 0x7F:
        return regex.escape(character)
    return rf"\u{point:04x}" if point <= 0xFFFF else rf"\U{point:08x}"


def text_excluding(*literals):
    """Regex for any text (including empty) that never contains a ``literal``.

    Several literals exclude text containing any of them (an adapter's two
    closing tags, say).  When they share a first character that recurs in
    none of them (true of every shipped tool-call opener and closer) the
    language is written as blocks: a character other than the first, or a
    partial literal that breaks off.  A partial literal broken off by the
    first character again restarts a new block, so ``(?:D*B)`` covers runs
    such as ``<to<tool``; with several literals the partials form a trie, and
    a partial breaks off at a character that continues none of them.  Other
    literals fall back to a lookahead, which only the scanner engine enforces.
    """
    if not literals or any(
        not isinstance(literal, str) or not literal for literal in literals
    ):
        raise ValueError("tool-call opener must be a nonempty string")
    kept = []  # a literal another one prefixes is excluded by that prefix
    for literal in sorted(set(literals), key=lambda item: (len(item), item)):
        if not any(literal.startswith(shorter) for shorter in kept):
            kept.append(literal)
    first = kept[0][0]
    if any(literal[0] != first or first in literal[1:] for literal in kept):
        excluded = "|".join(regex.escape(literal) for literal in kept)
        return rf"(?:(?!{excluded})[\s\S])*"
    if len(kept[0]) == 1:
        return rf"[^{_char(first)}]*"
    partials = {}  # proper prefix -> the characters that continue it
    for literal in kept:
        for k in range(1, len(literal)):
            following = partials.setdefault(literal[:k], [])
            if literal[k] not in following:
                following.append(literal[k])
    dangling = "(?:" + "|".join("".join(map(_char, partial)) for partial in partials) + ")"
    broken = "(?:" + "|".join(
        "".join(map(_char, partial))
        + "[^"
        + "".join(map(_char, [*following, first]))
        + "]"
        for partial, following in partials.items()
    ) + ")"
    return rf"(?:[^{_char(first)}]|{dangling}*{broken})*{dangling}*"


def _split_definitions(pattern):
    """``(body, definitions)``: strip the shared recursive-JSON DEFINE block."""
    from .structured_output import recursive_json_object_pattern

    # Tool blocks carry the finite-number rules; a JSON answer the plain ones.
    # The merged pattern keeps one block, so a tool block's finite rules also
    # bound the numbers of a JSON answer composed with it.
    for finite_numbers in (True, False):
        definitions = recursive_json_object_pattern(finite_numbers=finite_numbers)[1]
        if pattern.endswith(definitions):
            return pattern[: -len(definitions)], definitions
    if any(f"(?P<{name}>" in pattern for name in ("value", "object", "array")):
        # A JSON DEFINE block anywhere but the end cannot be merged with the
        # answer's; an adapter's own rules (other names) ride in the body.
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

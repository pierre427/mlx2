# Copyright © 2026 Pierre Lamy. mlx2 original.
"""Placement of exact APCv2 interior hybrid checkpoints.

A hybrid (GDN + attention) prompt cache cannot be trimmed, so a later request
reuses prefill only at a position where the recurrent state was captured.
Every request publishes its ``P-1`` prompt boundary and its finished lane.
Those serve the next turn only when the chat template re-renders the finished
turn starting with the whole generation prompt. Many templates do not: the
Qwen3.6 template renders a past assistant turn without the
``<think>\n\n</think>\n\n`` its generation prompt ends with, so both entries
diverge a few tokens before their end and the next turn reuses nothing.
Serving therefore plans one exact boundary just before the generation-prompt
suffix whenever :func:`detect_generation_prompt_suffixes` finds a template
that drops it (see ``generation_prompt_boundary``). Beyond that, interior
checkpoints pay off where a divergence falls *inside* one request's prompt:
a long system prompt/tool preamble shared across sessions, or a shared
document (RAG).

Placements (all content-independent except the adapter/tokenizer turn marker):

``pow2``   legacy lattice ``min_stride * 2^k`` (deepest ``count`` points).
``turns``  chat-template turn starts (state covers every token before the
           turn-start marker), deepest first.
``tail``   nested absolute tail lattice ``floor((P-2)/s)*s`` for
           ``s = min_stride * 4^k``: dense near the prompt end, identical for
           requests sharing a prefix (omlx#3456 nesting argument).
``auto``   preamble boundary, penultimate turn boundary (branch point), then
           tail points spaced at least ``min_stride`` apart, then remaining
           turn boundaries deepest-first.

Design references: omlx#3456 (budgeted nested boundary retention),
vllm#45238 (K checkpoints spread across the prompt).
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

PLACEMENTS = ("pow2", "turns", "tail", "auto")


def _check_int(name: str, value, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"interior checkpoint {name} must be an integer >= {minimum}")
    return value


def pow2_positions(prompt_tokens: int, *, count: int, min_stride: int) -> Tuple[int, ...]:
    values = []
    position = min_stride
    while position < prompt_tokens - 1:
        values.append(position)
        position *= 2
    return tuple(values[-count:]) if count else ()


def tail_positions(prompt_tokens: int, *, min_stride: int) -> Tuple[int, ...]:
    """Nested tail lattice, deepest (finest stride) first."""
    out = []
    stride = min_stride
    limit = prompt_tokens - 2
    while stride <= limit:
        point = (limit // stride) * stride
        if point > 0 and point not in out:
            out.append(point)
        stride *= 4
    return tuple(out)


def turn_boundary_positions(
    tokens: Sequence[int], marker_ids: Optional[Iterable[int]]
) -> Tuple[int, ...]:
    """Ascending positions ``i`` (0 < i < P-1) where a turn-start marker sits."""
    markers = frozenset(int(value) for value in (marker_ids or ()))
    if not markers:
        return ()
    last = len(tokens) - 1
    return tuple(
        index
        for index, token in enumerate(tokens)
        if 0 < index < last and int(token) in markers
    )


def plan_interior_positions(
    tokens: Sequence[int],
    *,
    count: int,
    min_stride: int,
    placement: str = "pow2",
    marker_ids: Optional[Iterable[int]] = None,
    cached_tokens: int = 0,
    floor_tokens: int = 0,
) -> Tuple[Tuple[int, ...], dict]:
    """Return sorted checkpoint positions and per-source planned counts.

    Positions satisfy ``max(cached_tokens, floor_tokens) < p < P-1``.
    ``floor_tokens`` excludes positions inside a media span.
    """
    count = _check_int("count", count, 0)
    min_stride = _check_int("min_stride", min_stride, 1)
    if placement not in PLACEMENTS:
        raise ValueError(f"unknown interior checkpoint placement: {placement!r}")
    total = len(tokens)
    low = max(int(cached_tokens), int(floor_tokens))
    sources = {"turn": 0, "tail": 0, "lattice": 0}
    if count == 0 or total < 3:
        return (), sources

    def usable(position):
        return low < position < total - 1

    picked: dict[int, str] = {}

    def take(position, source, *, spacing=0):
        if len(picked) >= count or position in picked or not usable(position):
            return
        if spacing and any(abs(position - other) < spacing for other in picked):
            return
        picked[position] = source

    if placement == "pow2":
        for position in pow2_positions(total, count=count, min_stride=min_stride):
            take(position, "lattice")
        # pow2 keeps legacy semantics: the deepest ``count`` lattice points,
        # filtered afterwards (never refilled from shallower points).
    elif placement == "tail":
        for position in tail_positions(total, min_stride=min_stride):
            take(position, "tail")
    else:
        turns = turn_boundary_positions(tokens, marker_ids)
        if placement == "turns":
            for position in reversed(turns):
                take(position, "turn")
        else:  # auto
            if turns:
                take(turns[0], "turn")
            if len(turns) >= 2:
                take(turns[-2], "turn")
            for position in tail_positions(total, min_stride=min_stride):
                take(position, "tail", spacing=min_stride)
            for position in reversed(turns):
                take(position, "turn")
    positions = tuple(sorted(picked))
    for position in positions:
        sources[picked[position]] += 1
    return positions, sources


def _ids(value) -> list:
    if hasattr(value, "keys") and "input_ids" in value:
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [int(token) for token in value]


def detect_turn_marker_ids(tokenizer) -> Tuple[int, ...]:
    """Detect the chat template's turn-start token without model-name rules.

    Renders one user message with and without ``add_generation_prompt``; the
    first token of the generation-prompt suffix is the marker when it is a
    special/added token that also opens the rendered user turn.  Returns
    ``()`` when the template does not expose such a token.
    """
    apply = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply):
        return ()
    probe = [{"role": "user", "content": "x"}]
    try:
        plain = _ids(apply(probe, add_generation_prompt=False, tokenize=True))
        prompt = _ids(apply(probe, add_generation_prompt=True, tokenize=True))
    except Exception:  # noqa: BLE001 - templates vary; no marker is safe
        return ()
    if len(prompt) <= len(plain) or prompt[: len(plain)] != plain:
        return ()
    marker = prompt[len(plain)]
    if marker not in plain:
        return ()
    special = set()
    for attribute in ("all_special_ids", "added_tokens_decoder"):
        value = getattr(tokenizer, attribute, None)
        if value is None:
            inner = getattr(tokenizer, "_tokenizer", None)
            value = getattr(inner, attribute, None) if inner is not None else None
        if isinstance(value, dict):
            special.update(int(key) for key in value)
        elif value is not None:
            special.update(int(key) for key in value)
    if special and marker not in special:
        return ()
    return (int(marker),)


# Chat-template flags probed for the generation prompt. Templates ignore
# variables they do not read, so probing a flag a template lacks is harmless;
# probing both values covers adapters that pass either one per request.
_GENERATION_PROMPT_PROBES = ({}, {"enable_thinking": False}, {"enable_thinking": True})


def detect_generation_prompt_suffixes(tokenizer) -> Tuple[Tuple[int, ...], ...]:
    """Generation-prompt suffixes a finished turn does not re-render.

    For each probed template mode this renders one user message with and
    without ``add_generation_prompt`` (the suffix is the difference, as in
    :func:`detect_turn_marker_ids`) and the next turn's conversation (that
    message, an assistant reply, a new user message). When the next prompt
    extends the whole first prompt, its ``P-1`` boundary already serves the
    next turn and nothing is returned for that mode. When it diverges inside
    the suffix, the suffix is returned: a boundary at ``P - len(suffix)`` is
    the deepest position the next turn shares. Longest first, no model-name
    rules; ``()`` when no mode needs one or the template cannot be probed.
    """
    apply = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply):
        return ()
    first = [{"role": "user", "content": "x"}]
    turn = first + [
        {"role": "assistant", "content": "y"},
        {"role": "user", "content": "z"},
    ]
    suffixes = set()
    for flags in _GENERATION_PROMPT_PROBES:
        try:
            plain = _ids(apply(first, add_generation_prompt=False, tokenize=True, **flags))
            prompt = _ids(apply(first, add_generation_prompt=True, tokenize=True, **flags))
            following = _ids(apply(turn, add_generation_prompt=True, tokenize=True, **flags))
        except Exception:  # noqa: BLE001 - templates vary; no boundary is safe
            continue
        if len(prompt) <= len(plain) or prompt[: len(plain)] != plain:
            continue
        shared = 0
        for left, right in zip(prompt, following):
            if left != right:
                break
            shared += 1
        if len(plain) <= shared < len(prompt) - 1:
            suffixes.add(tuple(prompt[len(plain) :]))
    return tuple(sorted(suffixes, key=lambda suffix: (-len(suffix), suffix)))


def generation_prompt_boundary(
    tokens: Sequence[int], suffixes: Iterable[Sequence[int]]
) -> Optional[int]:
    """Position just before the generation-prompt suffix the prompt ends with.

    ``None`` when the prompt ends with none of ``suffixes`` (a raw completion
    or a template mode that was not detected) or the boundary would not be an
    interior position ``0 < p < P-1``.
    """
    total = len(tokens)
    for suffix in suffixes:
        width = len(suffix)
        if 1 < width < total and [int(t) for t in tokens[total - width :]] == [
            int(t) for t in suffix
        ]:
            return total - width
    return None

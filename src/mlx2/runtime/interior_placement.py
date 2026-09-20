# Copyright © 2026 Pierre Lamy. mlx2 original.
"""Placement of exact APCv2 interior hybrid checkpoints.

A hybrid (GDN + attention) prompt cache cannot be trimmed, so a later request
reuses prefill only at a position where the recurrent state was captured.
Every request already publishes its ``P-1`` prompt boundary and its finished
lane, and both end on chat-message boundaries, so linear multi-turn chat,
edits and regenerations already resume from a previous turn's own entry.
Interior checkpoints pay off only where a divergence falls *inside* one
request's prompt at a position no earlier request ended on: a long system
prompt/tool preamble shared across sessions, or a shared document (RAG).

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

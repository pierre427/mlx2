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
           tail points at least ``min_stride`` above any shallower picked
           point; a picked point less than ``min_stride`` above a tail point
           suppresses it unless dropping it would lose at least
           ``4 * min_stride`` tokens of reuse; then remaining turn boundaries
           deepest-first.

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

    def take(position, source, *, spacing=0, fallback=None):
        if len(picked) >= count or position in picked or not usable(position):
            return
        if spacing:
            # A hit resumes from the deepest checkpoint at or below the
            # divergence.  A picked point at most ``spacing`` *below* the
            # candidate makes it redundant: falling back costs < spacing
            # tokens.
            if any(0 <= position - other < spacing for other in picked):
                return
            # A picked point just *above* it (the request's own
            # generation-prompt marker at P-k) does not make it redundant: a
            # request that diverges between them (same document, new
            # question) falls back to the next shallower point.  Rescue the
            # candidate only when that fallback would lose at least one
            # coarse stride (4 * spacing).  The finest lattice point always
            # sits in the last ``spacing`` tokens, so rescuing it would add
            # a snapshot plus a tiny final prefill slice to every cold chat
            # request for < 4 * spacing tokens of reuse (27B MTP batch:4:
            # -8..10% aggregate, 4 extra stores spilling at the 8 GiB cap);
            # rescuing a coarse point that coincides with it keeps the warm
            # hit at 7168 instead of 4096 (27B warm:8000, P=7405).
            if any(0 < other - position < spacing for other in picked):
                below = [other for other in picked if other < position]
                if fallback is not None:
                    below.append(fallback)
                if position - max(below, default=low) < 4 * spacing:
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
            tails = tail_positions(total, min_stride=min_stride)
            for index, position in enumerate(tails):
                shallower = [point for point in tails[index + 1:] if usable(point)]
                take(
                    position,
                    "tail",
                    spacing=min_stride,
                    fallback=max(shallower, default=None),
                )
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


def trimmed_tail_position(
    prompt_tokens: int,
    *,
    tokens_before: int,
    floor_tokens: int = 0,
    shallower: Iterable[int] = (),
    min_gain_tokens: int = 0,
) -> Optional[int]:
    """Exact boundary ``tokens_before`` tokens below the replay boundary.

    Agent clients often rewrite the tail of the previous turn's last message
    (a trimmed status block, tens of tokens) before they send the next turn,
    so that turn diverges just below the previous prompt's ``P-1`` boundary.
    A hybrid cache cannot cut back, so without a state near there the turn
    restores from the last rolling checkpoint, up to an interval back.  This
    places one boundary at ``P - 1 - tokens_before`` (relative to the prompt
    end, like the generation-prompt boundary).  ``None`` unless it lies above
    ``floor_tokens`` (cached prefix, media end) and at least
    ``min_gain_tokens`` above every ``shallower`` planned boundary below it,
    so a prompt that already has a state nearby pays for no extra snapshot.

    Design references: incoai/splash#784 (tail checkpoint 256 tokens before
    the replay boundary), jundot/omlx#5251 (minimum re-prefill gain).  Read,
    not copied.
    """
    tokens_before = _check_int("tail tokens_before", tokens_before, 1)
    min_gain_tokens = _check_int("tail min_gain_tokens", min_gain_tokens, 0)
    position = int(prompt_tokens) - 1 - tokens_before
    if position <= max(0, int(floor_tokens)):
        return None
    below = [int(point) for point in shallower if int(point) < position]
    base = max([int(floor_tokens), *below])
    if position - base < min_gain_tokens:
        return None
    return position


def cold_prefill_cuts(
    tokens: Sequence[int],
    *,
    policy,
    marker_ids: Optional[Iterable[int]] = None,
    generation_suffixes: Iterable[Sequence[int]] = (),
    media_floor: int = 0,
    media_position: Optional[int] = None,
    tail_tokens: int = 0,
) -> Tuple[int, ...]:
    """Planned slice ends of a cold (uncached) prefill of ``tokens``.

    The serving plan with ``cached_tokens=0``: interior positions (re-planned
    above ``media_floor`` when any falls inside the media span), the
    generation-prompt boundary, the adapter's media boundary and, when the
    opt-in trimmed-tail policy is on, its ``P - 1 - tail_tokens`` boundary.  No
    continuation skip and no headroom budget: those decide which positions
    are *captured*, never where a cold prefill is cut.  Junction boundaries
    depend on the lookup and rolling ones are absolute multiples, so neither
    is part of this list.  A warm request cut at the positions above its
    cached offset ``c`` slices exactly as the cold prefill whenever ``c`` is
    itself one of these cuts (and the server is idle: adaptive and fairness
    slicing are load dependent).
    """
    cuts = set()
    count = int(policy.get("count", 0) or 0)
    if count > 0:
        options = dict(
            count=count,
            min_stride=int(policy["min_stride"]),
            placement=policy.get("placement", "pow2"),
            marker_ids=marker_ids,
        )
        positions, _ = plan_interior_positions(tokens, **options)
        if media_floor and any(position < media_floor for position in positions):
            positions, _ = plan_interior_positions(
                tokens, floor_tokens=int(media_floor), **options
            )
        cuts.update(positions)
    boundary = generation_prompt_boundary(tokens, generation_suffixes)
    if boundary is not None and boundary > int(media_floor):
        cuts.add(boundary)
    if media_position is not None:
        cuts.add(int(media_position))
    if tail_tokens:
        # Gain is a capture decision, never a cut decision (as the budget):
        # cut at the tail whenever it is an interior position.
        tail = trimmed_tail_position(
            len(tokens), tokens_before=int(tail_tokens), floor_tokens=int(media_floor)
        )
        if tail is not None:
            cuts.add(tail)
    return tuple(sorted(cuts))


# --------------------------------------------------------------------------
# Paragraph checkpoint (opt-in ``apc_paragraph_checkpoints``)
# --------------------------------------------------------------------------

# Content tokens of the final user message that must precede the break: a
# shorter preamble is not a document worth a snapshot of its own.
PARAGRAPH_MIN_CONTENT_TOKENS = 256
# Prompt tokens that must follow the break: closer to the end, the committed
# P-1 boundary already covers nearly as much.
PARAGRAPH_MIN_TAIL_TOKENS = 16
# A "question" longer than this is not an appended question (omlx#5430 bounds
# its search to the last 4,096 tokens before the generation prompt).
PARAGRAPH_MAX_QUESTION_TOKENS = 4096
# Template tokens allowed between the end of the final user content and the
# prompt end (end-of-turn marker plus generation prompt).
PARAGRAPH_MAX_TEMPLATE_TAIL = 64
# Characters of document text encoded to prove the content floor; 32 chars a
# token is a conservative bound, a document denser than that plans nothing.
_PARAGRAPH_CHARS_PER_TOKEN_BOUND = 32

_PARAGRAPH_BREAK = None


def _paragraph_break_pattern():
    global _PARAGRAPH_BREAK
    if _PARAGRAPH_BREAK is None:
        import re

        _PARAGRAPH_BREAK = re.compile(r"\n{2,}")
    return _PARAGRAPH_BREAK


def _encode_plain(tokenizer, text):
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        return None
    try:
        ids = encode(text, add_special_tokens=False)
    except TypeError:
        ids = encode(text)
    return _ids(ids)


def _final_user_text(request) -> Optional[str]:
    messages = request.get("messages") if isinstance(request, dict) else None
    if not isinstance(messages, list) or not messages:
        return None
    final = messages[-1]
    if not isinstance(final, dict) or final.get("role") != "user":
        return None
    content = final.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content:
        # Only the trailing text part: the break must lie in the text that
        # directly precedes the template's end-of-turn tokens.
        part = content[-1]
        if isinstance(part, dict) and part.get("type") == "text":
            text = part.get("text")
            return text if isinstance(text, str) else None
    return None


def paragraph_checkpoint_position(
    tokens: Sequence[int],
    request,
    tokenizer,
    *,
    floor_tokens: int = 0,
    min_content_tokens: int = PARAGRAPH_MIN_CONTENT_TOKENS,
    min_tail_tokens: int = PARAGRAPH_MIN_TAIL_TOKENS,
    max_question_tokens: int = PARAGRAPH_MAX_QUESTION_TOKENS,
    max_template_tail: int = PARAGRAPH_MAX_TEMPLATE_TAIL,
) -> Tuple[Optional[int], str]:
    """Boundary just after the last paragraph break before the final question.

    Returns ``(position, "planned")`` or ``(None, reason)``.  ``position`` is
    the number of prompt tokens the snapshot covers: every token through the
    newline run, none of the question.  A different question appended to the
    same document after the same break shares exactly that prefix, so the
    first new question already resumes there (a junction needs a prior
    divergence to learn it).

    Selection, without model-name rules:

    1. The final message must be a ``user`` message whose content (or last
       content part) is text.  The last run of two or more ``"\\n"`` that is
       followed by non-whitespace text splits it into document and question.
    2. The question is encoded alone with the request's tokenizer and located
       in the prompt, ending at most ``max_template_tail`` tokens before the
       prompt end (exact match first; then a match that tolerates only the
       first question token, for SentencePiece dummy prefixes).
    3. The tokens just before the match must decode to text ending in
       ``"\\n\\n"``.  This accepts a single ``"\\n\\n"`` token, two ``"\\n"``
       tokens, and a break merged into the preceding text (``".\\n\\n"``);
       it rejects a newline absorbed into the question's first token.
    4. At least ``min_content_tokens`` document tokens precede the break,
       at least ``min_tail_tokens`` prompt tokens follow it, and it lies above
       ``floor_tokens`` (the end of any media span).

    Any mismatch plans nothing; exactness never depends on this choice, since
    APCv2 resumes only on an exact token prefix.
    """
    text = _final_user_text(request)
    if text is None:
        return None, "no_user_text"
    # Ignore trailing whitespace once. Copying and stripping the entire
    # remaining document at every break is quadratic on paragraph-heavy input.
    content_end = len(text.rstrip())
    last_break = None
    for match in _paragraph_break_pattern().finditer(text, 0, content_end):
        last_break = match
    if last_break is None:
        return None, "no_break"
    document_end, question_start = last_break.span()
    if not text[:document_end].strip():
        return None, "no_break"
    question = text[question_start:]
    try:
        question_ids = _encode_plain(tokenizer, question)
    except Exception:  # noqa: BLE001 - tokenizers vary; no boundary is safe
        return None, "unmatched"
    if not question_ids or len(question_ids) > int(max_question_tokens):
        return None, "unmatched"
    total = len(tokens)
    width = len(question_ids)
    last = total - width
    first = max(0, last - int(max_template_tail))
    window = [int(token) for token in tokens[first:total]]
    position = None
    for tolerant in (False, True):
        if tolerant and width < 2:
            break
        skip = 1 if tolerant else 0
        target = question_ids[skip:]
        for start in range(last, first - 1, -1):
            offset = start - first + skip
            if window[offset : offset + width - skip] == target:
                position = start
                break
        if position is not None:
            break
    if position is None or position < 1:
        return None, "unmatched"
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        return None, "unmatched"
    try:
        before = decode([int(token) for token in tokens[max(0, position - 3) : position]])
    except Exception:  # noqa: BLE001
        return None, "unmatched"
    if not isinstance(before, str) or not before.endswith("\n\n"):
        return None, "unmatched"
    if position > total - int(min_tail_tokens) or position >= total - 1:
        return None, "near_end"
    if position <= int(floor_tokens):
        return None, "inside_media"
    document = text[:document_end]
    bound = int(min_content_tokens) * _PARAGRAPH_CHARS_PER_TOKEN_BOUND
    try:
        document_ids = _encode_plain(tokenizer, document[-bound:])
    except Exception:  # noqa: BLE001
        return None, "unmatched"
    # One token of slack when truncated: the cut edge may split a token.
    slack = 1 if len(document) > bound else 0
    if document_ids is None or len(document_ids) - slack < int(min_content_tokens):
        return None, "short_document"
    if position <= int(min_content_tokens):
        return None, "short_document"
    return position, "planned"

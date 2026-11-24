"""Ling-3.2-air (BailingMoeV3 VL) 3D M-RoPE: the vendor position and frequency laws.

Read from inclusionAI/Ling-3.2-air @1613b1a4 (MIT) ``modeling_bailing_moe_v3_vl.py``
(``BailingMoeV3VLModel.get_rope_index``, ``BailingMoeV3VLRotaryEmbedding.apply_video_rope``,
``apply_rotary_pos_emb_interleave``); see provenance/ling32-air-vl.json.

Positions (VideoRoPE, body-diagonal): text tokens get one position on all three planes.
An image of merged grid ``(t, h, w)`` that starts where the next text position would be,
``p``, gives token ``(ti, hi, wi)`` the positions

    T = p + 2 * ti,  H = T + hi - (h - 1) // 2,  W = T + wi - (w - 1) // 2

and the text after it resumes at ``max(T) + 1 = p + 2 * (t - 1) + 1`` (the vendor takes
the maximum of the T plane only).  So after the last image every position is
``slot + delta`` for one request-constant ``delta``.

Frequencies: of the ``d / 2`` rotary frequencies (``d = qk_rope_head_dim``), the first
``H + W`` sections alternate H, W, H, W, ... and the remaining ``T`` section reads T
(``apply_video_rope`` with ``mrope_section = [T, H, W]`` and the pair-interleaved
rotation that reads one value per pair from the first half).  With all three planes
equal this is exactly 1D RoPE, which the text path keeps using unchanged.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import mlx.core as mx
import numpy as np

Span = Tuple[int, int, int, int, int]  # (slot, start_position, t, h, w), merged grid


def frequency_planes(half: int, section: Sequence[int]) -> np.ndarray:
    """Plane (0 T, 1 H, 2 W) read by each of the ``half`` rotary frequencies."""
    if len(section) != 3 or any(type(int(v)) is not int or int(v) < 0 for v in section):
        raise ValueError("Ling M-RoPE section must be [T, H, W]")
    t, h, w = (int(v) for v in section)
    if t + h + w != half or h != w:
        raise ValueError("Ling M-RoPE section does not cover the rotary width")
    plane = np.zeros(half, dtype=np.int64)
    plane[: h + w] = np.where(np.arange(h + w) % 2 == 0, 1, 2)
    return plane


def span_tokens(span: Span) -> int:
    return span[2] * span[3] * span[4]


def span_end(span: Span) -> int:
    return span[0] + span_tokens(span)


def span_next_position(span: Span) -> int:
    return span[1] + 2 * (span[2] - 1) + 1


def text_delta(spans: Sequence[Span], slot: int) -> int:
    """``position - slot`` for a text slot (no span may cover it)."""
    delta = 0
    for span in spans:
        if span_end(span) <= slot:
            delta = span_next_position(span) - span_end(span)
        elif span[0] <= slot:
            raise ValueError("Ling M-RoPE text slot lies inside an image span")
    return delta


def find_spans(tokens: Sequence[int], *, image_token_id: int, start_token_id: int,
               grids: Sequence[Sequence[int]], merge: int, start: int = 0,
               prior: Sequence[Span] = ()) -> Tuple[Span, ...]:
    """Spans of the image runs in ``tokens`` (absolute slots from ``start``).

    Every run of image tokens must follow a vision-start token and hold exactly its
    image's merged token count; runs and grids pair in prompt order.
    """
    tokens = np.asarray(list(tokens), dtype=np.int64)
    hits = np.flatnonzero(tokens == image_token_id)
    runs = []
    if hits.size:
        breaks = np.flatnonzero(np.diff(hits) != 1)
        starts = np.concatenate([[hits[0]], hits[breaks + 1]])
        ends = np.concatenate([hits[breaks], [hits[-1]]]) + 1
        runs = list(zip(starts.tolist(), ends.tolist()))
    if len(runs) != len(grids):
        raise ValueError(f"Ling prompt has {len(runs)} image runs for {len(grids)} images")
    spans = list(prior)
    new = []
    for (a, b), grid in zip(runs, grids):
        t, gh, gw = (int(v) for v in grid)
        if t < 1 or gh < merge or gw < merge or gh % merge or gw % merge:
            raise ValueError("Ling image grid is not a multiple of the merge size")
        h, w = gh // merge, gw // merge
        if b - a != t * h * w:
            raise ValueError("Ling image run length disagrees with its grid")
        if a == 0 or int(tokens[a - 1]) != start_token_id:
            raise ValueError("Ling image run must follow its vision-start token")
        slot = start + a
        span = (slot, slot + text_delta(spans, slot), t, h, w)
        spans.append(span)
        new.append(span)
    return tuple(new)


def slot_positions(spans: Sequence[Span], slots: np.ndarray) -> np.ndarray:
    """``[3, n]`` int64 positions (T, H, W) of absolute ``slots``."""
    slots = np.asarray(slots, dtype=np.int64)
    out = np.empty((3, slots.size), dtype=np.int64)
    covered = np.zeros(slots.size, dtype=bool)
    for span in spans:
        slot, p, t, h, w = span
        inside = (slots >= slot) & (slots < span_end(span))
        if not inside.any():
            continue
        k = slots[inside] - slot
        ti, rest = k // (h * w), k % (h * w)
        hi, wi = rest // w, rest % w
        tt = p + 2 * ti
        out[0, inside] = tt
        out[1, inside] = tt + hi - (h - 1) // 2
        out[2, inside] = tt + wi - (w - 1) // 2
        covered |= inside
    for index in np.flatnonzero(~covered):
        value = int(slots[index]) + text_delta(spans, int(slots[index]))
        out[:, index] = value
    return out


def chunk_rope_positions(spans: Sequence[Span], start: int, length: int):
    """RoPE positions of slots ``[start, start + length)``.

    An ``int`` (``start + delta``) when no span intersects the chunk, so a text chunk
    keeps the scalar ``fast.rope`` path; otherwise ``[3, length]`` int64.
    """
    stop = start + length
    if not any(s[0] < stop and span_end(s) > start for s in spans):
        return start + text_delta(spans, start)
    return slot_positions(spans, np.arange(start, stop))


def inv_frequencies(base: float, dims: int) -> mx.array:
    return 1.0 / (base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))


def mrope_rotate(x: mx.array, positions: np.ndarray, *, base: float, section) -> mx.array:
    """Pair-interleaved 3D RoPE over the last axis of ``x [..., L, d]``.

    ``positions`` is ``[3, L]``; frequency ``i`` rotates pair ``(2i, 2i + 1)`` by the
    position of ``frequency_planes(d // 2, section)[i]`` (fp32, then cast back).
    """
    d = x.shape[-1]
    plane = frequency_planes(d // 2, section)
    pos = np.asarray(positions, dtype=np.int64)
    if pos.ndim != 2 or pos.shape[0] != 3 or pos.shape[1] != x.shape[-2]:
        raise ValueError("Ling M-RoPE positions must be [3, L] for the rotated rows")
    freqs = mx.array(pos[plane].T.astype(np.float32)) * inv_frequencies(base, d)  # [L, d/2]
    cos, sin = mx.cos(freqs), mx.sin(freqs)
    dtype = x.dtype
    pairs = x.astype(mx.float32).reshape(*x.shape[:-1], d // 2, 2)
    a, b = pairs[..., 0], pairs[..., 1]
    out = mx.stack([a * cos - b * sin, b * cos + a * sin], axis=-1)
    return out.reshape(x.shape).astype(dtype)

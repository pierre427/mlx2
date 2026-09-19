"""Revision-bound Spomin surgery for standard attention caches.

Covers the two stock cache shapes used by dense sliding/full attention models:
an append-only ``KVCache`` on full-attention layers and a single-row
``RotatingKVCache`` ring on sliding-window layers.  Rotary parameters are read
from each layer's own ``nn.RoPE`` module so the common lifecycle never learns a
model name; a layer with no rotary module is treated as position-free.

The result is approximate: retained rows keep the hidden-state influence of the
removed history.  Callers must not publish it as exact prefix state.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace

import mlx.core as mx
import mlx.nn as nn

from .models.cache import KVCache, RotatingKVCache
from .spomin_layer import (
    SpominBackendCapabilities,
    SpominCapabilityError,
    SpominPlan,
    SpominTargetState,
)


def shift_rope(x, delta, *, dims, base, traditional, scale=1.0):
    """Rotate already-rotated rows by ``delta`` positions (``delta``: [..., T])."""
    if dims == 0:
        return x
    freqs = mx.exp(-math.log(base) * mx.arange(0, dims, 2) / dims)
    angles = (delta[..., None].astype(mx.float32) * scale) * freqs
    cos, sin = mx.cos(angles), mx.sin(angles)
    rope, tail = x[..., :dims], x[..., dims:]
    source = rope.astype(mx.float32)
    if traditional:
        left, right = source[..., 0::2], source[..., 1::2]
        rotated = mx.stack(
            [left * cos - right * sin, right * cos + left * sin], axis=-1
        ).reshape(source.shape)
    else:
        half = dims // 2
        left, right = source[..., :half], source[..., half:]
        rotated = mx.concatenate(
            [left * cos - right * sin, right * cos + left * sin], axis=-1
        )
    return mx.concatenate([rotated.astype(x.dtype), tail], axis=-1)


def _text_layers(model):
    candidate = getattr(model, "language_model", model)
    candidate = getattr(candidate, "model", candidate)
    layers = getattr(candidate, "layers", None)
    if not layers:
        raise SpominCapabilityError("backend requires a decoder layer stack")
    return list(layers)


def _rotary(layer, index):
    attention = getattr(layer, "self_attn", None)
    if attention is None:
        raise SpominCapabilityError(f"layer {index} has no attention module")
    rope = getattr(attention, "rope", None)
    if rope is None or getattr(attention, "use_rope", True) is False:
        return None
    if type(rope) is not nn.RoPE:
        raise SpominCapabilityError(
            f"layer {index} uses {type(rope).__name__}; only default RoPE can be re-phased"
        )
    dims = int(rope.dims)
    if dims <= 0 or dims % 2:
        raise SpominCapabilityError("rotary dimensions must be positive and even")
    return {
        "dims": dims,
        "base": float(rope.base),
        "traditional": bool(rope.traditional),
        "scale": float(rope.scale),
    }


def _kept_positions(state, removed):
    by_id = {segment.segment_id: segment for segment in state.transcript.segments}
    cursor = 0
    keep = []
    for segment_id in state.visible_segment_ids:
        stop = cursor + len(by_id[segment_id].token_ids)
        if segment_id not in removed:
            keep.extend(range(cursor, stop))
        cursor = stop
    return cursor, tuple(keep)


class StandardAttentionSpominBackend:
    capabilities = SpominBackendCapabilities(
        attention_kv_edit=True,
        recurrent_state_repair=False,
        noncontiguous_edit=True,
    )

    def __init__(self, model, prompt_cache: Sequence[object]):
        self.model = model
        self.prompt_cache = prompt_cache

    def _preflight(self, state: SpominTargetState, plan: SpominPlan):
        if plan.replacement_token_ids:
            raise SpominCapabilityError(
                "KV surgery cannot materialize replacement summary tokens"
            )
        if state.has_mtp_state:
            raise SpominCapabilityError("live MTP state cannot be surgically repaired")
        layers = _text_layers(self.model)
        if len(layers) != len(self.prompt_cache):
            raise SpominCapabilityError("prompt-cache layer count does not match the model")
        removed = set(plan.selection.segment_ids)
        visible, keep = _kept_positions(state, removed)
        if visible != state.target_tokens:
            raise SpominCapabilityError(
                "target token count must exactly match the visible transcript"
            )
        if len(keep) != plan.projected_target_tokens:
            raise SpominCapabilityError("planned token count disagrees with row selection")
        dropped = sorted(set(range(visible)) - set(keep))
        edits = []
        for index, (layer, cache) in enumerate(zip(layers, self.prompt_cache)):
            rotary = _rotary(layer, index)
            if cache.keys is None or cache.values is None:
                raise SpominCapabilityError(f"layer {index} cache is empty")
            if cache.keys.shape[0] != 1:
                raise SpominCapabilityError(f"layer {index} cache is batched; surgery requires B=1")
            if int(cache.offset) != visible:
                raise SpominCapabilityError(
                    f"layer {index} cache offset {cache.offset} does not match the visible transcript ({visible})"
                )
            if type(cache) is KVCache:
                edits.append(("full", cache, rotary, None))
            elif type(cache) is RotatingKVCache:
                if cache.keep:
                    raise SpominCapabilityError(f"layer {index} ring pins sink rows")
                if cache.speculating:
                    raise SpominCapabilityError(f"layer {index} ring has an armed rollback")
                # A prefill chunk leaves more than ``max_size`` rows behind; the
                # next decode step drops all but the newest ``max_size``, so
                # only those are live state.
                held = min(
                    int(cache._temporal_order(cache.keys).shape[2]),
                    int(cache.max_size),
                )
                # The ring can only stay a legal ring if every removed token has
                # already left it; then the edit is a uniform phase shift.
                if dropped and dropped[-1] >= visible - held:
                    raise SpominCapabilityError(
                        f"layer {index} sliding window still holds tokens selected for removal"
                    )
                edits.append(("ring", cache, rotary, held))
            else:
                raise SpominCapabilityError(
                    f"layer {index} cache {type(cache).__name__} is not surgically editable"
                )
        return edits, keep, len(dropped)

    def apply(self, state, plan):
        edits, keep, dropped = self._preflight(state, plan)
        indices = mx.array(keep, dtype=mx.int32)
        delta = (mx.arange(len(keep), dtype=mx.float32) - indices.astype(mx.float32))[
            None, None, :
        ]
        staged = []
        for kind, cache, rotary, _held in edits:
            if kind == "full":
                keys = mx.take(cache.keys[..., : cache.offset, :], indices, axis=2)
                values = mx.take(cache.values[..., : cache.offset, :], indices, axis=2)
                if rotary is not None:
                    keys = shift_rope(keys, delta, **rotary)
            else:
                keys = cache._temporal_order(cache.keys)[..., -_held:, :]
                values = cache._temporal_order(cache.values)[..., -_held:, :]
                if rotary is not None:
                    uniform = mx.full((1, 1, keys.shape[2]), -float(dropped), dtype=mx.float32)
                    keys = shift_rope(keys, uniform, **rotary)
            staged.append((mx.contiguous(keys), mx.contiguous(values)))
        mx.eval(*(array for pair in staged for array in pair))
        for (kind, cache, _rotary, _held), (keys, values) in zip(edits, staged):
            cache.keys = keys
            cache.values = values
            cache.offset = len(keep)
            if kind == "ring":
                cache._idx = keys.shape[2]
                cache._rollbacks.clear()
                cache._checkpoints = []
        removed = set(plan.selection.segment_ids)
        return replace(
            state,
            revision=f"{state.revision}:spomin-kv",
            target_tokens=plan.projected_target_tokens,
            visible_segment_ids=tuple(
                segment_id
                for segment_id in state.visible_segment_ids
                if segment_id not in removed
            ),
        )

"""Revision-bound Spomin surgery for Qwen4-Exp attention caches.

Only unquantized, single-sequence QSA caches are supported. Recurrent and PLE
state is preserved as the model's fixed-size summary of the complete history;
active MTP, packed/ragged caches, and non-default RoPE are refused pre-mutation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import mlx.core as mx

from .models.qwen4_exp import QSAKVCache, _apply_rope_positions
from .spomin_layer import (
    SpominBackendCapabilities,
    SpominCapabilityError,
    SpominPlan,
    SpominTargetState,
)


def _text_model(model):
    candidate = getattr(model, "language_model", model)
    candidate = getattr(candidate, "model", candidate)
    if not hasattr(candidate, "layers") or not hasattr(candidate, "args"):
        raise SpominCapabilityError("backend requires a Qwen4-Exp text model")
    return candidate


def _visible_token_positions(state, removed):
    by_id = {segment.segment_id: segment for segment in state.transcript.segments}
    cursor = 0
    keep = []
    for segment_id in state.visible_segment_ids:
        segment = by_id[segment_id]
        stop = cursor + len(segment.token_ids)
        if segment_id not in removed:
            keep.extend(range(cursor, stop))
        cursor = stop
    return cursor, tuple(keep)


class Qwen4SpominSurgeryBackend:
    capabilities = SpominBackendCapabilities(
        attention_kv_edit=True,
        recurrent_state_repair=True,
        noncontiguous_edit=True,
    )

    def __init__(self, model, prompt_cache: Sequence[object]):
        self.model = model
        self.prompt_cache = prompt_cache

    def _preflight(self, state: SpominTargetState, plan: SpominPlan):
        if plan.replacement_token_ids:
            raise SpominCapabilityError(
                "KV surgery cannot materialize replacement summary tokens; use an exact rebuild backend"
            )
        if state.has_mtp_state:
            raise SpominCapabilityError("live MTP state cannot be surgically repaired")
        text = _text_model(self.model)
        layers = list(text.layers)
        if len(layers) != len(self.prompt_cache):
            raise SpominCapabilityError("prompt-cache layer count does not match the Qwen4 model")
        scaling = getattr(text.args, "rope_scaling", None)
        if scaling is None:
            scaling = getattr(text.args, "rope_parameters", None)
        rope_type = "default" if not scaling else scaling.get("type") or scaling.get("rope_type", "default")
        if rope_type != "default":
            raise SpominCapabilityError(f"Qwen4 KV surgery does not support {rope_type!r} RoPE scaling")
        dims = int(text.args.head_dim * text.args.partial_rotary_factor)
        base = float(getattr(text.args, "rope_theta", None) or scaling.get("rope_theta", 1_000_000.0))
        if dims <= 0 or dims % 2:
            raise SpominCapabilityError("Qwen4 rotary dimensions must be positive and even")
        removed = set(plan.selection.segment_ids)
        visible_tokens, keep = _visible_token_positions(state, removed)
        if visible_tokens != state.target_tokens:
            raise SpominCapabilityError("target token count must exactly match the visible transcript for surgery")
        if len(keep) != plan.projected_target_tokens:
            raise SpominCapabilityError("planned token count does not match the transcript row selection")
        attention = []
        for index, (layer, cache) in enumerate(zip(layers, self.prompt_cache)):
            if getattr(layer, "is_linear", False):
                continue
            if type(cache) is not QSAKVCache:
                raise SpominCapabilityError(f"layer {index} requires an unquantized B=1 QSAKVCache")
            if cache.offset != visible_tokens:
                raise SpominCapabilityError(
                    f"layer {index} cache offset {cache.offset} does not match the visible transcript ({visible_tokens})"
                )
            if cache.keys is None or cache.values is None or cache.index_keys is None:
                raise SpominCapabilityError(f"layer {index} QSA cache is missing K/V or its index ledger")
            if cache.keys.shape[0] != 1 or cache.values.shape[0] != 1:
                raise SpominCapabilityError(f"layer {index} cache is batched; surgery currently requires B=1")
            if cache.index_keys.shape[0] != 1 or cache.index_keys.shape[1] != visible_tokens:
                raise SpominCapabilityError(f"layer {index} QSA index ledger is not exactly cursor-aligned")
            if cache._mtp_share_topk or cache._mtp_shared_topk is not None:
                raise SpominCapabilityError(f"layer {index} has an armed MTP/QSA selection cycle")
            attention.append((index, cache))
        if not attention:
            raise SpominCapabilityError("model has no surgically editable attention cache")
        return attention, keep, dims, base

    def apply(self, state, plan):
        attention, keep, dims, base = self._preflight(state, plan)
        indices = mx.array(keep, dtype=mx.int32)
        old_positions = mx.array(keep, dtype=mx.float32)
        new_positions = mx.arange(len(keep), dtype=mx.float32)
        delta = (new_positions - old_positions)[None, None, :]
        replacements = []
        for _, cache in attention:
            live_keys = cache.keys[..., : cache.offset, :]
            live_values = cache.values[..., : cache.offset, :]
            keys = mx.contiguous(_apply_rope_positions(mx.take(live_keys, indices, axis=2), delta, dims=dims, base=base))
            values = mx.contiguous(mx.take(live_values, indices, axis=2))
            ledger = mx.contiguous(mx.take(cache.index_keys, indices, axis=1))
            replacements.append((keys, values, ledger))
        mx.eval(*(array for replacement in replacements for array in replacement))
        for (_, cache), (keys, values, ledger) in zip(attention, replacements):
            cache.keys = keys
            cache.values = values
            cache.index_keys = ledger
            cache.offset = len(keep)
            cache._mtp_share_topk = False
            cache._mtp_shared_topk = None
            cache._mtp_shared_topk_n_blocks = None
            cache._qsa_pooled_keys = None
            cache._qsa_pooled_ratio = None
            cache._qsa_summary_identity = None
            cache._qsa_summary_restored = False
            cache._qsa_pending_pooled = None
        removed = set(plan.selection.segment_ids)
        return replace(
            state,
            revision=f"{state.revision}:spomin-kv",
            target_tokens=plan.projected_target_tokens,
            visible_segment_ids=tuple(segment_id for segment_id in state.visible_segment_ids if segment_id not in removed),
        )

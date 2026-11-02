"""Gemma 3n prompt-chunk row pruning before the KV-shared layers (default on).

Gemma 3n's upper ``num_kv_shared_layers`` layers write no cache: they read
the K/V of the last full and last sliding layer below them.  Their outputs at
a prompt row feed only later shared layers of that row and the vocabulary
head.  In a prompt chunk only one row per lane reaches the head (the
checkpoint guard's row, ``prefill_checkpoint_rows``), so every other row's
shared-layer work is dead.  This forward drops those rows -- hidden states,
per-layer inputs and mask rows -- right before the first shared layer.
mechanism: ggml-org/llama.cpp#31356 (open at head df6b4794, MIT); see
provenance/gemma3n-kv-shared-prune.json.

What it changes and what it does not:

* Layers below the first shared layer run unchanged on every row, so every
  cache byte is the ordinary path's.  The prompt's final token runs as its
  own one-row decode-shaped step, which this forward never touches, so the
  first generated token and everything after it are the ordinary path's.
* The chunk's checkpoint row is computed by shared layers at one row instead
  of ``width``; matmuls at another row count may round differently.  That
  row only feeds the finiteness guard.
* The kept row keeps its query position: the pinned mlx-vlm shared attention
  rotates queries from ``cache.offset`` *after* the source layer appended the
  chunk, so row ``width - 1`` is rotated at that offset plus ``width - 1``.
  ``_ShiftedOffset`` reproduces it for the one remaining row.

It applies only inside an open, unapplied checkpoint request whose rows are
all ``width - 1`` (no ragged lanes), with ``width > 1``, a cache, and no
caller mask.  Anything else runs the ordinary mlx-vlm forward.  A consumer of
full-width hidden states (an MTP drafter, prompt logprobs) would have to
decline it; mlx2 serves neither for Gemma 3n.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect

import mlx.core as mx

from ..runtime.prefill_checkpoint_rows import claim_final_row_checkpoint

# sha256 of inspect.getsource() of the pinned mlx-vlm 67599f2e methods this
# forward mirrors.  A different source refuses the option instead of
# silently pruning against code it was not written for.
MIRRORED_SOURCES = {
    "Gemma3Model.__call__": "f8f3e94e0bd408ad2fc4b1b4709d4fb1fb10b64cb39c9091f9de5ae9677d89f9",
    "Gemma3nAttention.__call__": "a3d7aecdc22bf49379dad1110f1f088cd6398d6db7b13b92ccec060d8d9702a7",
    "LanguageModel.__call__": "ee4b0099e99cd1dd9a3e7e61f93acb0323d71cb8877739b6d05a51a0640c3c23",
}


def check_mirrored_source(language_model) -> None:
    module = importlib.import_module(type(language_model).__module__)
    for name, expected in MIRRORED_SOURCES.items():
        owner, method = name.split(".")
        source = inspect.getsource(getattr(getattr(module, owner), method))
        digest = hashlib.sha256(source.encode()).hexdigest()
        if digest != expected:
            raise ValueError(
                f"Gemma 3n KV-shared row pruning mirrors mlx-vlm 67599f2e; {name} "
                f"source differs ({digest[:12]}), refusing the option"
            )


class PruneCounters:
    __slots__ = ("pruned_forwards", "rows_in", "rows_kept", "ordinary_forwards")

    def __init__(self):
        self.pruned_forwards = 0
        self.rows_in = 0
        self.rows_kept = 0
        self.ordinary_forwards = 0

    def as_dict(self) -> dict:
        return {name: getattr(self, name) for name in self.__slots__}


class _ShiftedOffset:
    """A shared layer's cache view that reports ``offset + shift``."""

    def __init__(self, cache, shift: int):
        self._cache = cache
        self._shift = shift

    def __getattr__(self, name):
        return getattr(self._cache, name)

    @property
    def offset(self):
        return self._cache.offset + self._shift


def _last_rows(mask):
    return mask[..., -1:, :] if isinstance(mask, mx.array) else mask


def eligible(inputs_embeds, cache, mask) -> bool:
    return (
        cache is not None
        and mask is None
        and inputs_embeds is not None
        and inputs_embeds.ndim == 3
        and inputs_embeds.shape[1] > 1
    )


def pruned_language_forward(language_model, inputs, *, inputs_embeds, cache, counters, **kwargs):
    """``LanguageModel.__call__`` logits with dead shared-layer rows dropped.

    Returns ``[batch, 1, vocab]`` logits after claiming the checkpoint
    request, or None (nothing claimed) when the call is not eligible; the
    caller then runs the ordinary forward.
    """
    if not eligible(inputs_embeds, cache, kwargs.get("mask")):
        return None
    batch, width = int(inputs_embeds.shape[0]), int(inputs_embeds.shape[1])
    if not claim_final_row_checkpoint(batch, width):
        return None
    kwargs.pop("mask", None)
    module = importlib.import_module(type(language_model).__module__)
    model = language_model.model
    config = model.config

    # From here to the layer loop: Gemma3Model.__call__ (mlx-vlm 67599f2e).
    per_layer_inputs = kwargs.pop("per_layer_inputs", None)
    kwargs.pop("n_to_process", None)
    h = inputs_embeds
    if per_layer_inputs is None and inputs is not None:
        per_layer_inputs = model.get_per_layer_inputs(inputs)
    elif per_layer_inputs is not None:
        target_len = h.shape[1]
        c0 = next((c for c in cache if c is not None and hasattr(c, "_idx")), None)
        if c0 is not None:
            cache_offset = int(c0._idx)
        else:
            raw_offset = next(
                (c.offset for c in cache if c is not None and hasattr(c, "offset")), 0
            )
            if isinstance(raw_offset, mx.array):
                cache_offset = (
                    int(raw_offset.max().item()) if raw_offset.size > 1 else int(raw_offset.item())
                )
            else:
                cache_offset = int(raw_offset)
        max_start = max(per_layer_inputs.shape[1] - target_len, 0)
        start = min(cache_offset, max_start)
        per_layer_inputs = per_layer_inputs[:, start : start + target_len]

    per_layer_inputs = model.project_per_layer_inputs(h, per_layer_inputs)

    full_mask = module.create_attention_mask(h, cache[model.first_full_idx])
    sliding_window_mask = module.create_attention_mask(
        h, cache[model.first_sliding_idx], window_size=config.sliding_window
    )
    h0 = h
    target_magnitude = mx.mean(h0**2, axis=-1, keepdims=True) ** 0.5
    h_list = [h0]
    h_list.extend([proj(h0) for proj in model.altup_projections])
    h = mx.stack(h_list, axis=0)
    mags = mx.mean(h[1:] ** 2, axis=-1, keepdims=True) ** 0.5
    h[1:] = h[1:] * (target_magnitude / mx.maximum(mags, mx.finfo(h0.dtype).min))

    first_shared = model.first_kv_shared_layer_idx
    for i, layer in enumerate(model.layers):
        layer_cache = cache[model.layer_idx_to_cache_idx[i]]
        if i == first_shared:
            # Every row but the last is dead from here on.
            h = h[:, :, -1:, :]
            per_layer_inputs = per_layer_inputs[:, -1:]
            full_mask = _last_rows(full_mask)
            sliding_window_mask = _last_rows(sliding_window_mask)
        if i >= first_shared and layer_cache is not None:
            layer_cache = _ShiftedOffset(layer_cache, width - 1)
        is_global = config.layer_types[i] == "full_attention"
        local_mask = full_mask if is_global else sliding_window_mask
        h = layer(h, local_mask, layer_cache, per_layer_inputs[:, :, i, :])

    target_magnitude = mx.mean(h[0] ** 2, axis=-1, keepdims=True) ** 0.5
    for i, proj in enumerate(model.altup_unembed_projections):
        h[i + 1] = proj(h[i + 1])
    mags = mx.mean(h[1:] ** 2, axis=-1, keepdims=True) ** 0.5
    h[1:] = h[1:] * (target_magnitude / mx.maximum(mags, mx.finfo(h0.dtype).min))
    h = model.norm(mx.mean(h, axis=0))

    # LanguageModel.__call__ head.
    out = model.embed_tokens.as_linear(h)
    if language_model.final_logit_softcapping is not None:
        out = module.logit_softcap(language_model.final_logit_softcapping, out)
    counters.pruned_forwards += 1
    counters.rows_in += batch * width
    counters.rows_kept += batch
    return out

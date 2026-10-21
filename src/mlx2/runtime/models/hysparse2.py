"""Serving form of the HySparse2 research model over mlx2's standard caches.

Original mlx2 code.  The tensor math is the research model's own
(``mlx2.experimental.hysparse2``): this module reuses its modules, weights,
n-gram PLE, cross full-attention oracle selection and sparse attention, and
only changes where per-request state lives so that the generic serving
machinery (continuous batching, APCv2, speculative rollback) can own it.

Cache planes, in order (``make_cache``):

* one ``KVCache`` holding the committed token IDs for the n-gram PLE history
  (encoded as two exact base-256 digits so any floating dtype is exact);
* one cache per self-decoder layer: ``RotatingKVCache(local_window)`` for SWA
  (exact speculative rollback inside the verify window, as for Granite SWA)
  and ``KVCache`` for the self full-attention layer;
* one ``KVCache`` per cross full-attention anchor, holding its own K/V
  projections of the self full-attention input (the KV-sharing bridge).
  Sparse layers reuse their anchor's selected K/V and own no cache.

Self-decoder attention runs on the fused SDPA kernel with the cache's mask
(window, causal, ragged left/right padding) and learned sinks.  Cross anchors
and sparse layers call the research ``Attention`` per row, with each row's
absolute positions recovered from the cache geometry, because oracle top-k
selection is position-dependent and has no padded-batch kernel.  Sparse
layers never allocate KV.

During prompt processing only each lane's checkpoint row runs the cross
decoder (``select_checkpoint_rows``), mirroring the research prefill, which
builds caches through the self decoder and runs the cross decoder only for
the last prompt position.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ...experimental.hysparse2.config import Config
from ...experimental.hysparse2.model import Model as ResearchModel
from ..prefill_checkpoint_rows import select_checkpoint_rows_with_indices
from .base import create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache

# Token IDs are stored as (id // 256, id % 256): both digits are integers
# below 256, exact in bfloat16/float16/float32 even if a cache is cast.
_DIGIT = 256


def _encode_tokens(tokens):
    tokens = tokens.astype(mx.int32)
    digits = mx.stack((tokens // _DIGIT, tokens % _DIGIT), axis=-1)
    return digits.astype(mx.float32)[:, None]  # B, 1, T, 2


def _decode_tokens(digits):
    value = digits[..., 0] * _DIGIT + digits[..., 1]
    return mx.round(value.astype(mx.float32)).astype(mx.int32)


def _append_rows(cache, keys, values):
    """Append a ``B, H, W, D`` block; return each row's own history.

    Each entry is ``(row_keys, row_values, first, count)``: the row's keys
    from its absolute position 0, the absolute position of its first new
    token and how many of the ``W`` new positions are real (``None`` for a
    row without tokens).  Plain and batched caches expose rows through their
    left padding (pending right padding is causally invisible to real
    queries); a segmented verify view (batched prompt-lookup verification)
    exposes each lane's request-private history through ``row_views``.
    """
    batch, width = keys.shape[0], keys.shape[2]
    if callable(getattr(cache, "row_views", None)):
        mask = cache.make_mask(width)
        cache.update_and_fetch(keys, values)
        rows = [None] * batch
        for index, count, row_keys, row_values, _mask in cache.row_views(mask):
            if count:
                rows[index] = (row_keys, row_values, row_keys.shape[2] - count, count)
        return rows
    stored_keys, stored_values = cache.update_and_fetch(keys, values)
    padding = getattr(cache, "left_padding", None)
    padding = [0] * batch if padding is None else [int(v) for v in padding.tolist()]
    if len(padding) != batch:
        raise ValueError("HySparse2 cache row count differs from the batch")
    stored = stored_keys.shape[2]
    rows = []
    for row, begin in enumerate(padding):
        if not 0 <= begin <= stored - width:
            raise ValueError("HySparse2 cache row padding exceeds its history")
        rows.append(
            (
                stored_keys[row : row + 1, :, begin:],
                stored_values[row : row + 1, :, begin:],
                stored - width - begin,
                width,
            )
        )
    return rows


class Model(nn.Module):
    # SWA planes are stock RotatingKVCache rings, which record exact rollback
    # within the verify window; every other plane is an append-only KVCache.
    supports_speculative_rollback = True

    def __init__(self, config: Config):
        super().__init__()
        if not isinstance(config, Config):
            raise TypeError("HySparse2 serving needs a research Config")
        self.config = config
        self.model_type = config.model_type
        self.core = ResearchModel(config)
        self.ple_history = bool(config.semantic_ple_rows) and config.semantic_ngram > 1
        self._self_kinds = [layer.kind for layer in self.core.self_decoder]
        self._cross_anchor_layers = [
            i for i, layer in enumerate(self.core.cross_decoder) if layer.kind == "cross"
        ]

    # -- cache contract ------------------------------------------------------
    def make_cache(self):
        caches = [KVCache()] if self.ple_history else []
        for kind in self._self_kinds:
            caches.append(
                RotatingKVCache(max_size=self.config.local_window)
                if kind == "swa"
                else KVCache()
            )
        caches.extend(KVCache() for _ in self._cross_anchor_layers)
        return caches

    def _split_cache(self, cache):
        expected = (
            int(self.ple_history) + len(self._self_kinds) + len(self._cross_anchor_layers)
        )
        if not isinstance(cache, (list, tuple)) or len(cache) != expected:
            raise ValueError("HySparse2 needs its complete cache plane list")
        start = int(self.ple_history)
        history = cache[0] if self.ple_history else None
        self_caches = cache[start : start + len(self._self_kinds)]
        cross = cache[start + len(self._self_kinds) :]
        for entry in cache:
            if hasattr(entry, "bits"):
                raise ValueError("HySparse2 serving does not support quantized KV caches")
        return history, self_caches, cross

    @property
    def layers(self):
        return list(self.core.self_decoder) + list(self.core.cross_decoder)

    # -- forward -------------------------------------------------------------
    def _history(self, cache, tokens):
        """Append ``tokens`` and return each row's previous n-1 token IDs."""
        order = self.config.semantic_ngram - 1
        encoded = _encode_tokens(tokens)
        histories = []
        for row in _append_rows(cache, encoded, encoded):
            if row is None:
                histories.append(mx.zeros((1, order), dtype=mx.int32))
                continue
            stored, _, first, _ = row
            previous = _decode_tokens(stored[:, 0, max(0, first - order) : first])
            missing = order - previous.shape[1]
            if missing:
                previous = mx.concatenate(
                    (mx.zeros((1, missing), dtype=mx.int32), previous), axis=1
                )
            histories.append(previous)
        return histories[0] if len(histories) == 1 else mx.concatenate(histories, axis=0)

    def _self_attention(self, attention, x, cache, kind, mask):
        c = self.config
        batch, width, _ = x.shape
        q = attention.q(x).reshape(batch, width, c.num_heads, c.head_dim).transpose(0, 2, 1, 3)
        k = attention.k(x)[:, None]
        v = attention.v(x)[:, None]
        if kind == "swa" and c.rope_dims:
            offset = cache.offset
            q = mx.fast.rope(
                q, c.rope_dims, traditional=False, base=c.rope_base, scale=1.0, offset=offset
            )
            k = mx.fast.rope(
                k, c.rope_dims, traditional=False, base=c.rope_base, scale=1.0, offset=offset
            )
        keys, values = cache.update_and_fetch(k, v)
        out = scaled_dot_product_attention(
            q,
            keys,
            values,
            cache=cache,
            scale=c.head_dim**-0.5,
            mask=mask,
            sinks=attention.sinks.astype(q.dtype) if kind == "swa" else None,
        ).transpose(0, 2, 1, 3)
        if kind == "swa":
            out = out * mx.sigmoid(attention.gate(x))[..., None]
        return attention.out(out.reshape(batch, width, -1))

    def _cross_rows(self, x, cross_caches, source, rows):
        """Run the cross decoder per row; ``x`` holds only the needed rows."""
        core = self.core
        batch, width = x.shape[:2]
        anchors = []
        for cache, index in zip(cross_caches, self._cross_anchor_layers):
            k, v, _ = core.cross_decoder[index].attention.project_kv(source, 0)
            anchors.append(_append_rows(cache, k, v))
        outputs = []
        for row in range(batch):
            geometry = {(None if a[row] is None else a[row][2:]) for a in anchors}
            if len(geometry) != 1:
                raise ValueError("HySparse2 cross anchors disagree on cache geometry")
            (geometry,) = geometry
            if geometry is None:
                outputs.append(mx.zeros_like(x[row : row + 1]))
                continue
            first, count = geometry
            if rows is not None:
                # Prompt processing: only this lane's checkpoint row runs.
                first, count = first + rows[row], width
            h = x[row : row + 1, :count]
            selected = None
            anchor = 0
            for layer in core.cross_decoder:
                blocks = None
                if layer.kind == "cross":
                    keys, values = anchors[anchor][row][:2]
                    anchor += 1
                    blocks = [(keys, values, 0)]
                h, selected, _, _ = layer(h, blocks, first, selected)
            if count < width:
                h = mx.pad(h, [(0, 0), (0, width - count), (0, 0), (0, 0)])
            outputs.append(h)
        return outputs[0] if batch == 1 else mx.concatenate(outputs, axis=0)

    def __call__(self, inputs, cache=None, **kwargs):
        if kwargs:
            raise ValueError(f"HySparse2 serving got unsupported inputs {sorted(kwargs)}")
        if inputs.ndim != 2 or inputs.shape[1] < 1:
            raise ValueError("HySparse2 serving consumes B,T token IDs")
        core, c = self.core, self.config
        if cache is None:
            cache = self.make_cache()
        history_cache, self_caches, cross_caches = self._split_cache(cache)
        batch, width = inputs.shape
        x = core.embedding(inputs)
        if c.semantic_ple_rows:
            history = self._history(history_cache, inputs) if self.ple_history else None
            x = x + core.semantic_ple(inputs, x, history)
        x = mx.broadcast_to(x[..., None, :], (batch, width, c.residual_streams, c.hidden_size))
        # One mask per attention kind, built before any append (as Granite
        # SWA does): every SWA ring shares one geometry, as do full planes.
        masks = {}
        for entry, kind in zip(self_caches, self._self_kinds):
            if kind not in masks:
                masks[kind] = create_attention_mask(
                    inputs, entry, window_size=c.local_window if kind == "swa" else None
                )
        source = None
        for layer, entry, kind in zip(core.self_decoder, self_caches, self._self_kinds):
            raw, post = layer.attention_hc.read(x)
            out = self._self_attention(
                layer.attention, layer.attention_norm(raw), entry, kind, masks[kind]
            )
            x = layer.attention_hc.write(x, out, post)
            y, post = layer.ffn_hc.read(x)
            y, _ = layer.moe(layer.ffn_norm(y))
            x = layer.ffn_hc.write(x, y, post)
            if kind == "full":
                source = raw
        # Narrow to each lane's checkpoint row during prompt processing; the
        # identity everywhere else (decode, verification, drafting).
        flat = x.reshape(batch, width, -1)
        narrowed, rows = select_checkpoint_rows_with_indices(flat)
        if rows is not None:
            x = narrowed.reshape(batch, narrowed.shape[1], c.residual_streams, c.hidden_size)
        x = self._cross_rows(x, cross_caches, source, rows)
        return core.embedding.as_linear(core.norm(mx.mean(x, axis=-2)))

    # -- loading ---------------------------------------------------------------
    def load_research_checkpoint(self, path):
        """Strictly load a research checkpoint through the research loader."""
        from ...experimental.hysparse2.train import initialize_from_checkpoint

        return initialize_from_checkpoint(path, self.core)

"""Per-layer token selection from a quantized key copy, exact attention on the rows kept.

Each full-attention layer keeps its ordinary fp16 KV cache plus a quantized
copy of K (8 or 4 bits). At decode, the layer scores every position with its
own query against the cheap copy, keeps a forced recent window plus its top
``budget`` positions, and runs exact fp16 attention over just those rows.

This follows the 2026-09-23 cross-layer reuse experiment
(``docs/experiments/FA-TOPK-REUSE-2026-09-23.md``). There, borrowing another
layer's selection failed because layers attend to different tokens, while
each layer's *own* top-k captured 0.93 of its attention mass at a
4096-token budget. Scoring with quantized keys makes own-layer selection
cheap enough to be worth trying. Quantized-key indexers are prior art
(Quest, Double Sparsity, SparQ, and DeepSeek's FP8 lightning indexer); no
code from them is used.

Experimental: not declared as a capability, not qualified, not reachable
from serving. Plan and gates: ``docs/experiments/KV-INDEX-2026-09-23.md``.
"""

from __future__ import annotations

from collections import Counter
from typing import Literal

import mlx.core as mx

from .models.cache import KVCache

Sharing = Literal["shared", "kv_head"]


class QuantizedKeyIndexCache(KVCache):
    """A plain ``KVCache`` whose decode attention reads only selected rows.

    Selection runs only while ``armed`` and only for single-row decode calls
    with more than ``window + budget`` positions; everything else (prefill,
    verify rows, short contexts) takes the stock dense path. The quantized
    key copy is built lazily on the first selected call and then extended
    one row per step. ``exact_scores`` scores with the fp16 keys instead (an
    upper bound on selection quality that saves nothing).
    """

    step = 256

    def __init__(
        self,
        *,
        bits: int = 8,
        group_size: int = 64,
        budget: int = 4096,
        window: int = 128,
        sharing: Sharing = "shared",
        exact_scores: bool = False,
    ):
        super().__init__()
        if budget <= 0 or window < 0:
            raise ValueError("need budget > 0 and window >= 0")
        if sharing not in ("shared", "kv_head"):
            raise ValueError(f"unknown sharing {sharing!r}")
        # Not `bits`: models.base routes any cache with a `bits` attribute to
        # the quantized-SDPA path, and attention here stays fp16.
        self.index_bits = bits
        self.group_size = group_size
        self.budget = budget
        self.window = window
        self.sharing = sharing
        self.exact_scores = exact_scores
        self.armed = False
        self.counts: Counter = Counter()
        self._qkeys = None  # (packed, scales, biases), grown in `step` rows
        self._q_offset = 0

    # -- quantized key copy ------------------------------------------------

    def _sync_index(self) -> None:
        start, stop = self._q_offset, self.offset
        if stop <= start:
            return
        rows = mx.quantize(
            self.keys[..., start:stop, :], group_size=self.group_size, bits=self.index_bits
        )
        capacity = 0 if self._qkeys is None else self._qkeys[0].shape[-2]
        if stop > capacity:
            new_capacity = -(-stop // self.step) * self.step
            grown = []
            for i, part in enumerate(rows):
                shape = part.shape[:-2] + (new_capacity, part.shape[-1])
                fresh = mx.zeros(shape, dtype=part.dtype)
                if self._qkeys is not None:
                    fresh[..., :start, :] = self._qkeys[i][..., :start, :]
                grown.append(fresh)
            self._qkeys = tuple(grown)
        for i, part in enumerate(rows):
            self._qkeys[i][..., start:stop, :] = part
        self._q_offset = stop

    def trim(self, n):
        n = super().trim(n)
        self._q_offset = min(self._q_offset, self.offset)
        return n

    def index_nbytes(self) -> int:
        return 0 if self._qkeys is None else sum(part.nbytes for part in self._qkeys)

    # -- attention ------------------------------------------------------------

    def select(self, queries: mx.array, keys: mx.array, scale: float) -> mx.array:
        """``(B, G, budget + window)`` int32 row indices (unsorted, no repeats) for one decode row."""
        B, Hq, _, D = queries.shape
        Hkv, T = keys.shape[1], keys.shape[2]
        grouped = (queries * scale).reshape(B, Hkv, Hq // Hkv, D)
        if self.exact_scores:
            logits = grouped @ keys.swapaxes(-1, -2)
        else:
            self._sync_index()
            packed = tuple(part[..., :T, :] for part in self._qkeys)
            logits = mx.quantized_matmul(
                grouped, *packed, transpose=True,
                group_size=self.group_size, bits=self.index_bits,
            )
        # Mean probability over each KV head's query group (and over KV heads
        # when shared): every query head counts equally however peaked it is.
        scores = mx.softmax(logits.astype(mx.float32), axis=-1).mean(axis=2)
        if self.sharing == "shared":
            scores = scores.mean(axis=1, keepdims=True)
        head = T - self.window
        top = mx.argpartition(-scores[..., :head], kth=self.budget - 1, axis=-1)
        recent = mx.broadcast_to(
            mx.arange(head, T, dtype=mx.int32), scores.shape[:-1] + (self.window,)
        )
        return mx.concatenate([top[..., : self.budget].astype(mx.int32), recent], axis=-1)

    def bucketed_attention(self, queries, scale, mask, sinks=None):
        T = self.offset
        if (not self.armed or queries.shape[2] != 1 or sinks is not None
                or T <= self.window + self.budget):
            self.counts["dense"] += 1
            return None
        keys, values = self.keys_and_values()
        indices = self.select(queries, keys, scale)
        B, Hkv = keys.shape[:2]
        rows = mx.broadcast_to(indices, (B, Hkv, indices.shape[-1]))[..., None]
        kg = mx.take_along_axis(keys, rows, axis=2)
        vg = mx.take_along_axis(values, rows, axis=2)
        self.counts["indexed"] += 1
        return mx.fast.scaled_dot_product_attention(queries, kg, vg, scale=scale)


def install_index(cache: list, **kwargs) -> tuple[list, list]:
    """Swap each fresh plain ``KVCache`` (full attention) for an index cache.

    Returns the new cache list and the swapped caches, which the caller arms
    after prefill. Sliding-window and recurrent caches are left alone. Must
    run before prefill: the swapped caches start empty.
    """
    positions = [i for i, c in enumerate(cache) if type(c) is KVCache]
    if not positions:
        raise ValueError("no plain KVCache (full-attention) layers to index")
    if any(cache[i].offset for i in positions):
        raise ValueError("install_index needs empty caches (install before prefill)")
    swapped = list(cache)
    indexed = []
    for i in positions:
        swapped[i] = QuantizedKeyIndexCache(**kwargs)
        indexed.append(swapped[i])
    return swapped, indexed


def quantize_full_attention(cache: list, *, key_bits: int, value_bits: int,
                            group_size: int = 64) -> tuple[list, int]:
    """Convert every plain ``KVCache`` to the serving ``QuantizedKVCache`` form.

    The same conversion ``kv_q8`` / ``kv_k8v4`` use (``KVCache.to_quantized``
    with no rotation). Returns the new list and the number converted.
    """
    out, converted = [], 0
    for c in cache:
        if type(c) is KVCache:
            out.append(c.to_quantized(group_size=group_size, key_bits=key_bits,
                                      value_bits=value_bits))
            converted += 1
        else:
            out.append(c)
    return out, converted

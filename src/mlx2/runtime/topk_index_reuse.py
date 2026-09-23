"""Cross-layer top-k index reuse for decode attention: reference math and probe.

A *source* full-attention layer computes exact attention at decode. Its
probabilities pick a token set: a forced window of the most recent positions
plus the ``budget`` highest-scoring positions outside it. Later full-attention
layers then attend only to that set, each in its own KV cache.

The selection signal comes from HySparse2's KV Reuse (arXiv 2609.26368, sec.
3.3), where it is trained end to end. Here it is applied post hoc to models
never trained with it, so every sparse output of this module is approximate.
This module is an experiment: nothing in it is declared as a capability,
qualified, or reachable from a serving route. The pre-registered plan and its
gates are in ``docs/experiments/FA-TOPK-REUSE-2026-09-23.md``.

Shapes follow the runtime's attention convention: queries ``(B, Hq, L, D)``,
keys and values ``(B, Hkv, T, D)``. A selection is a boolean mask ``(B, G, T)``
where ``G`` is 1 (one set shared by every head, as with HySparse2's MQA) or
``Hkv`` (one set per KV head). Masks, not index lists, are the reference form:
they cannot hold duplicates and need no equal-width rows. The gathered form
(:func:`indices_from_mask`, :func:`gathered_attention`) is what a kernel
would consume, and is tested equal to the masked form.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import mlx.core as mx

from .models.cache import KVCache

HeadSharing = Literal["shared", "kv_head"]
Granularity = Literal["token", "block"]


# --- Selection -------------------------------------------------------------


def attention_probs(
    q: mx.array, k: mx.array, scale: float, chunk: int = 8192
) -> mx.array:
    """Exact float32 attention probabilities ``(B, Hq, L, T)`` for the last L rows.

    Query heads are grouped onto their KV head by reshaping, so K is never
    repeated per query head. K is upcast ``chunk`` rows at a time: a whole
    float32 copy is ~1 GB per layer at 128K, and because the context grows
    every step, each copy has a new size that MLX's buffer cache keeps but
    rarely reuses (it swapped the host on 2026-09-23).
    """
    B, Hq, L, D = q.shape
    Hkv, T = k.shape[1], k.shape[2]
    groups = Hq // Hkv
    qg = q.astype(mx.float32).reshape(B, Hkv, groups * L, D) * scale
    parts = [
        qg @ k[:, :, start : start + chunk].astype(mx.float32).swapaxes(-1, -2)
        for start in range(0, T, chunk)
    ]
    logits = (parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=-1)).reshape(
        B, Hq, L, T
    )
    if L > 1:
        rows = mx.arange(T - L, T)[:, None]
        logits = mx.where(mx.arange(T)[None, :] <= rows, logits, -mx.inf)
    return mx.softmax(logits, axis=-1)


def selection_scores(
    probs: mx.array, n_kv_heads: int, sharing: HeadSharing = "shared"
) -> mx.array:
    """Reduce ``(B, Hq, L, T)`` probabilities to ``(B, G, T)`` selection scores.

    The score is mean attention probability, so every query head and row
    carries equal weight however peaked it is. For the shared token variant,
    the top-k of this score is exactly the set that maximizes the mean
    per-head attention mass captured at that size.
    """
    B, Hq, L, T = probs.shape
    if sharing == "shared":
        return probs.mean(axis=(1, 2))[:, None, :]
    if sharing == "kv_head":
        return probs.reshape(B, n_kv_heads, Hq // n_kv_heads, L, T).mean(axis=(2, 3))
    raise ValueError(f"unknown head sharing {sharing!r}")


def select_mask(
    scores: mx.array,
    budget: int,
    window: int,
    granularity: Granularity = "token",
    block_size: int = 64,
) -> mx.array:
    """Forced recent window plus the top ``budget`` positions outside it.

    Returns a ``(B, G, T)`` boolean mask. When the context fits in
    ``window + budget`` every position is kept, so masked attention is dense
    attention. Token granularity keeps exactly ``window + budget`` positions.
    Block granularity spends the budget on ``budget // block_size`` blocks
    (at least one), ranked by summed score and aligned to absolute positions
    from 0. A block that reaches into the window adds only its positions
    before the window, so a block set can hold fewer extra tokens than the
    budget. Ties among equal scores resolve in an unspecified order.
    """
    if budget < 0 or window < 0 or budget + window == 0:
        raise ValueError("need budget >= 0, window >= 0 and budget + window > 0")
    B, G, T = scores.shape
    if T <= window + budget:
        return mx.ones((B, G, T), dtype=mx.bool_)
    head = T - window  # positions [0, head) compete for the budget
    recent = mx.ones((B, G, window), dtype=mx.bool_)
    if budget == 0:
        return mx.concatenate([mx.zeros((B, G, head), dtype=mx.bool_), recent], -1)
    ranked = scores[..., :head].astype(mx.float32)

    if granularity == "token":
        top = mx.argpartition(-ranked, kth=budget - 1, axis=-1)[..., :budget]
        picked = mx.put_along_axis(
            mx.zeros((B, G, head), dtype=mx.bool_), top, mx.array(True), axis=-1
        )
    elif granularity == "block":
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        n_blocks = -(-head // block_size)
        pad = n_blocks * block_size - head
        padded = mx.pad(ranked, [(0, 0), (0, 0), (0, pad)])
        mass = padded.reshape(B, G, n_blocks, block_size).sum(axis=-1)
        take = min(max(budget // block_size, 1), n_blocks)
        top = mx.argpartition(-mass, kth=take - 1, axis=-1)[..., :take]
        blocks = mx.put_along_axis(
            mx.zeros((B, G, n_blocks), dtype=mx.bool_), top, mx.array(True), axis=-1
        )
        picked = mx.repeat(blocks, block_size, axis=-1)[..., :head]
    else:
        raise ValueError(f"unknown granularity {granularity!r}")
    return mx.concatenate([picked, recent], axis=-1)


def window_mask(shape: tuple[int, int, int], window: int) -> mx.array:
    """``(B, G, T)`` mask of only the most recent ``window`` positions."""
    B, G, T = shape
    return mx.broadcast_to(mx.arange(T) >= T - window, (B, G, T))


def _head_mask(mask: mx.array, n_q_heads: int) -> mx.array:
    """Lift a ``(B, G, T)`` selection to ``(B, Hq, 1, T)`` for attention."""
    G = mask.shape[1]
    lifted = mask if G == 1 else mx.repeat(mask, n_q_heads // G, axis=1)
    return lifted[:, :, None, :]


# --- Attention over a selection --------------------------------------------


def masked_attention(
    q: mx.array, k: mx.array, v: mx.array, mask: mx.array, scale: float
) -> mx.array:
    """SDPA over the positions in ``mask``; reference for the sparse arm.

    It still reads every KV row, so it is a fidelity reference, not a speedup.
    For ``L > 1`` the causal constraint is combined with the selection.
    """
    L, T = q.shape[2], k.shape[2]
    visible = _head_mask(mask, q.shape[1])
    if L > 1:
        rows = mx.arange(T - L, T)[:, None]
        visible = mx.logical_and(visible, mx.arange(T)[None, :] <= rows)
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=visible)


def indices_from_mask(mask: mx.array) -> mx.array:
    """Sorted ``(B, G, S)`` int32 positions of a mask with equal row counts.

    Raises when rows differ in count (block and per-head selections can), as
    a gathered kernel needs a fixed width. Synchronizes to read the counts.
    """
    counts = mask.sum(axis=-1)
    width = int(counts.max().item())
    if int(counts.min().item()) != width:
        raise ValueError("selection rows differ in size; gather needs equal widths")
    T = mask.shape[-1]
    positions = mx.where(mask, mx.arange(T, dtype=mx.int32), T)
    return mx.sort(positions, axis=-1)[..., :width]


def gathered_attention(
    q: mx.array, k: mx.array, v: mx.array, indices: mx.array, scale: float
) -> mx.array:
    """Decode (``L == 1``) SDPA over K/V rows gathered at ``(B, G, S)`` indices.

    This is the kernel-shaped form: it reads only the selected rows.
    """
    if q.shape[2] != 1:
        raise ValueError("gathered_attention is decode-only (L == 1)")
    B, Hkv = k.shape[:2]
    idx = mx.broadcast_to(indices, (B, Hkv, indices.shape[-1]))[..., None]
    kg = mx.take_along_axis(k, idx, axis=2)
    vg = mx.take_along_axis(v, idx, axis=2)
    return mx.fast.scaled_dot_product_attention(q, kg, vg, scale=scale)


# --- Metrics ---------------------------------------------------------------


def mass_recall(probs: mx.array, mask: mx.array) -> mx.array:
    """Fraction of each query head's attention mass inside ``mask``.

    ``probs`` is ``(B, Hq, L, T)`` from the *target* layer's exact attention.
    Returns ``(B, Hq, L)``; 1.0 means masked attention is exact for that head.
    """
    return (probs * _head_mask(mask, probs.shape[1])).sum(axis=-1)


def relative_error(reference: mx.array, candidate: mx.array) -> mx.array:
    """Per ``(B, Hq, L)`` relative L2 error of attention outputs."""
    ref = reference.astype(mx.float32)
    diff = candidate.astype(mx.float32) - ref
    num = mx.sqrt((diff * diff).sum(axis=-1))
    den = mx.maximum(mx.sqrt((ref * ref).sum(axis=-1)), 1e-12)
    return num / den


# --- Probe: record or substitute decode attention in a live model ----------


@dataclass(frozen=True)
class Variant:
    granularity: Granularity = "token"
    sharing: HeadSharing = "shared"

    @property
    def name(self) -> str:
        return f"{self.granularity}/{self.sharing}"


class ReuseProbe:
    """Coordinates selection and measurement across one model's FA layers.

    Full-attention layers are numbered by *ordinal* (0 = first FA layer).
    ``sources`` must include 0; every other layer reuses the selection of the
    nearest source at or before it. The probe acts only while armed and only
    on single-row decode calls; prefill and anything with sinks pass through
    to the stock attention path untouched.

    Modes:

    * ``shadow``: every layer computes exact probabilities and its own
      selections; target layers record the mass recall of their source's
      selection (per variant), the recall of their own top-k (the oracle upper
      bound) and of the window alone (the lower bound), the relative error of
      the sparse output, and, with ``record_matrix``, the recall of every
      earlier layer's selection. Attention output stays the stock dense path,
      so the model's trajectory is unchanged.
    * ``substitute``: sources run dense and select with the primary variant
      (``variants[0]``); targets return masked attention over it. Errors
      compound through decode, which is what the end-to-end gate measures.

    ``counts`` records what actually happened, so a harness can refuse to
    report when the mechanism did not fire.
    """

    def __init__(
        self,
        *,
        mode: Literal["shadow", "substitute"],
        n_layers: int,
        sources: Sequence[int] = (0,),
        budget: int = 1024,
        window: int = 128,
        variants: Sequence[Variant] = (Variant(),),
        block_size: int = 64,
        record_matrix: bool = True,
    ):
        if mode not in ("shadow", "substitute"):
            raise ValueError(f"unknown probe mode {mode!r}")
        sources = sorted(set(sources))
        if not sources or sources[0] != 0 or sources[-1] >= n_layers:
            raise ValueError("sources must include ordinal 0 and lie within the FA layers")
        if not variants:
            raise ValueError("need at least one variant")
        self.mode = mode
        self.n_layers = n_layers
        self.sources = tuple(sources)
        self.source_of = {t: max(s for s in sources if s <= t) for t in range(n_layers)}
        self.budget = budget
        self.window = window
        self.variants = tuple(variants)
        self.block_size = block_size
        self.record_matrix = record_matrix
        self.armed = False
        self.step = -1
        self.counts: Counter = Counter()
        self.rows: list[dict] = []
        self._selections: dict[int, dict[Variant, mx.array]] = {}
        self._pending: list[tuple[dict, dict[str, mx.array]]] = []

    @property
    def targets(self) -> tuple[int, ...]:
        return tuple(t for t in range(self.n_layers) if t not in self.sources)

    def _select(self, probs: mx.array, n_kv_heads: int, variant: Variant) -> mx.array:
        scores = selection_scores(probs, n_kv_heads, variant.sharing)
        return select_mask(
            scores, self.budget, self.window, variant.granularity, self.block_size
        )

    def attend(self, ordinal, q, k, v, scale, mask=None, sinks=None):
        """Attention override for one FA layer; ``None`` means use the stock path."""
        if not self.armed or q.shape[2] != 1 or sinks is not None:
            self.counts["passthrough"] += 1
            return None
        if ordinal == 0:
            self.step += 1
            self._selections = {}
        primary = self.variants[0]
        if ordinal not in self.source_of:
            raise ValueError(f"FA ordinal {ordinal} outside the probe's {self.n_layers}")
        is_source = ordinal in self.sources

        if self.mode == "substitute":
            if is_source:
                probs = attention_probs(q, k, scale)
                self._selections[ordinal] = {primary: self._select(probs, k.shape[1], primary)}
                self.counts["source"] += 1
                return None
            selection = self._selections[self.source_of[ordinal]][primary]
            self.counts["substituted"] += 1
            return masked_attention(q, k, v, selection, scale).astype(q.dtype)

        probs = attention_probs(q, k, scale)
        own = {var: self._select(probs, k.shape[1], var) for var in self.variants}
        metrics = {
            "oracle_recall": mass_recall(probs, own[primary]),
            "window_recall": mass_recall(probs, window_mask(own[primary].shape, self.window)),
        }
        if not is_source:
            source = self._selections[self.source_of[ordinal]]
            for var in self.variants:
                metrics[f"recall/{var.name}"] = mass_recall(probs, source[var])
            dense = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
            sparse = masked_attention(q, k, v, source[primary], scale)
            metrics["relerr"] = relative_error(dense, sparse)
        if self.record_matrix:
            for earlier in range(ordinal):
                if earlier in self._selections:
                    metrics[f"from/{earlier}"] = mass_recall(
                        probs, self._selections[earlier][primary]
                    )
        self._selections[ordinal] = own
        row = {"step": self.step, "layer": ordinal, "context": k.shape[2],
               "role": "source" if is_source else "target"}
        self._pending.append((row, metrics))
        self.counts["shadowed"] += 1
        return None

    def flush(self) -> None:
        """Materialize pending metrics into ``rows`` (mean and min over heads)."""
        if not self._pending:
            return
        mx.eval([m for _, metrics in self._pending for m in metrics.values()])
        for row, metrics in self._pending:
            for name, value in metrics.items():
                flat = value.reshape(-1)
                row[name] = float(flat.mean().item())
                row[f"{name}:min"] = float(flat.min().item())
            self.rows.append(row)
        self._pending = []


class ReuseProbeKVCache(KVCache):
    """A plain ``KVCache`` whose attention call is routed through a probe.

    It uses the ``bucketed_attention`` seam in
    ``models.base.scaled_dot_product_attention``, which runs after
    ``update_and_fetch``, so the model code is unchanged and the cache still
    holds every row.
    """

    def __init__(self, probe: ReuseProbe, ordinal: int):
        super().__init__()
        self.probe = probe
        self.ordinal = ordinal

    def bucketed_attention(self, queries, scale, mask, sinks=None):
        keys, values = self.keys_and_values()
        return self.probe.attend(
            self.ordinal, queries, keys, values, scale, mask=mask, sinks=sinks
        )


def install_probe(cache: list, probe_factory) -> tuple[list, "ReuseProbe"]:
    """Swap each fresh plain ``KVCache`` (the full-attention layers) for a probe cache.

    ``probe_factory(n_layers)`` builds the probe once the FA count is known.
    Rotating (sliding-window) and recurrent caches are left alone, so the
    same call works for any model whose full-attention layers use plain
    ``KVCache`` and route SDPA through ``models.base``. Must run before
    prefill: the swapped caches start empty.
    """
    ordinals = [i for i, c in enumerate(cache) if type(c) is KVCache]
    if any(cache[i].offset for i in ordinals):
        raise ValueError("install_probe needs empty caches (install before prefill)")
    if not ordinals:
        raise ValueError("no plain KVCache (full-attention) layers to probe")
    probe = probe_factory(len(ordinals))
    swapped = list(cache)
    for ordinal, index in enumerate(ordinals):
        swapped[index] = ReuseProbeKVCache(probe, ordinal)
    return swapped, probe


# --- Decode traffic ceiling ------------------------------------------------


@dataclass(frozen=True)
class DecodeTraffic:
    """Bytes moved per decoded token by a batch-1 decode step.

    ``weight_bytes`` is one token's weight traffic (active experts only).
    ``kv_row_bytes`` gives K+V bytes per context position for each
    full-attention layer, in order. Sliding-window and recurrent state are
    bounded, so they go in ``fixed_bytes``. This is a bandwidth ceiling: it
    assumes decode is purely memory-bound and that a gathered row costs what a
    contiguous one does, so it bounds what a real kernel could achieve.
    """

    weight_bytes: float
    kv_row_bytes: Sequence[float]
    fixed_bytes: float = 0.0

    def dense(self, context: int) -> float:
        return self.weight_bytes + self.fixed_bytes + context * sum(self.kv_row_bytes)

    def reused(
        self,
        context: int,
        *,
        sources: Sequence[int] = (0,),
        budget: int = 1024,
        window: int = 128,
        score_pass: float = 0.5,
    ) -> float:
        """Traffic when ``sources`` stay dense and every other FA layer is sparse.

        A source also needs explicit scores, which fused SDPA does not return;
        ``score_pass`` is the extra fraction of its rows read to get them
        (0.5 = reading K a second time).
        """
        kept = min(context, budget + window)
        total = self.weight_bytes + self.fixed_bytes
        source_set = set(sources)
        for layer, row in enumerate(self.kv_row_bytes):
            if layer in source_set:
                total += context * row * (1.0 + score_pass)
            else:
                total += kept * row
        return total

    def ceiling(self, context: int, **kwargs) -> float:
        """Upper bound on decode speedup at ``context`` (dense / reused)."""
        return self.dense(context) / self.reused(context, **kwargs)

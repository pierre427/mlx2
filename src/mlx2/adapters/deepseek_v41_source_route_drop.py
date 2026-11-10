# SPDX-License-Identifier: MIT
"""APPROXIMATE soft-reap of the lowest-weight routed expert on chosen layers, opt-in.

Pierre 2026-11-08: test soft-reaping the 6th expert on middle layers only.
For B1 single-row forwards on layers ``first..last`` the native router runs
unchanged; then the selected expert with the smallest combine weight gets
weight zero and the other five are rescaled so the row's total combine weight
(``routed_scaling_factor`` under ``norm_topk_prob``) is preserved, which is
what renormalising over the kept five gives. Multi-row prefill chunks, image
rows and layers outside the range keep native routing. The router sees only
the row shape, so a prefill chunk of exactly one row is also dropped (and the
forward stamped); verify routes are refused by policy.

Dropped experts are never read (Pierre 2026-11-08): each dropped slot's index
is replaced by the row's top-weight expert, which is always kept, with weight
zero. The six-slot route geometry is unchanged, the expert stream acquires
only the distinct kept experts, and the duplicate slot contributes an exact
zero.

Composition with route bias (2026-11-09): the drop wrapper's inner router is
the bias router when one is installed (resolved at call time), so the chain
is native -> bias re-selection -> drop on the biased selection -> redirect. Cost: a row with a duplicate takes the per-expert path instead of the
fused down-combine, and the duplicate acquire is counted as a cache hit. Isolation and APCv2 refusal: see
:mod:`deepseek_v41_source_approx`. Offline sizing (router weights by rank,
drop-if-miss curves): docs/research/ds41-source-expert-cache-design-2026-11-08.md.
"""
from collections import deque

import numpy as np

from .deepseek_v41_source_approx import DROP_KEY, DROP_MAX_WEIGHT, DROP_PROTECT, drop_counts, stamp_cache

MODES = ('drop_min_v1', 'drop_miss_v1')


def reference_drop(weights, count=1):
    """NumPy oracle for one row: zero the ``count`` smallest weights, keep the total."""
    w = np.asarray(weights, np.float64).copy()
    total = w.sum()
    for k in np.argsort(w, kind='stable')[:count]:
        w[k] = 0.0
    kept = w.sum()
    return w * (total / kept) if kept > 0 else w


def redirect_dropped(indices, weights):
    """Point zero-weight slots at the row's top-weight expert so they are never read."""
    import mlx.core as mx
    top = mx.take_along_axis(indices, mx.argmax(weights, axis=-1, keepdims=True), axis=-1)
    return mx.where(weights == 0, top, indices).astype(indices.dtype)


def reference_redirect(indices, weights):
    i = np.asarray(indices).copy(); w = np.asarray(weights)
    i[w == 0] = i[int(np.argmax(w))]
    return i


def drop_missing(weights, resident, *, count, protect, max_weight):
    """Zero up to ``count`` lowest-weight NON-resident selections outside the top ``protect`` ranks
    whose share of the row is below ``max_weight``; keep the row total. ``resident`` is bool like weights."""
    import mlx.core as mx
    w = weights.astype(mx.float32)
    total = mx.sum(w, axis=-1, keepdims=True)
    order = mx.argsort(mx.argsort(-w, axis=-1), axis=-1)               # 0 = top weight
    cand = mx.logical_and(mx.logical_and(order >= protect, mx.logical_not(resident)),
                          w < max_weight * total)
    mask = mx.zeros(w.shape, mx.bool_)
    for _ in range(count):
        free = mx.logical_and(cand, mx.logical_not(mask))
        masked = mx.where(free, w, mx.inf)
        k = mx.argmin(masked, axis=-1, keepdims=True)
        hit = mx.take_along_axis(free, k, axis=-1)
        mask = mx.logical_or(mask, mx.logical_and(mx.arange(w.shape[-1]) == k, hit))
    dropped = mx.sum(mx.where(mask, w, 0.0), axis=-1, keepdims=True)
    kept = mx.where(mask, 0.0, w)
    return kept * (total / mx.maximum(total - dropped, 1e-20)), dropped / mx.maximum(total, 1e-20)


def resident_or_pooled(store, layer, experts):
    """Original-cache/leased residency plus reduced-tier pool membership (review 2026-11-09):
    an expert held in a Q1/Q2 pool costs no read, so it is not 'missing'."""
    from .deepseek_v41_source_route_bias import resident_mask
    mask = resident_mask(store, layer, experts)
    rule = getattr(store, 'q1_on_miss', None)
    pools = getattr(rule, 'pools', None) or {}
    if pools:
        with store._lock:
            for pool in pools.values():
                held = pool.layers[layer] if layer < len(pool.layers) else {}
                for e in held:
                    if 0 <= int(e) < experts:
                        mask[int(e)] = True
    return mask


def drop_min(weights, count=1):
    import mlx.core as mx
    w = weights.astype(mx.float32)
    total = mx.sum(w, axis=-1, keepdims=True)
    mask = mx.zeros(w.shape, mx.bool_)
    for _ in range(count):              # first index wins ties, like the oracle's stable sort
        k = mx.argmin(mx.where(mask, mx.inf, w), axis=-1, keepdims=True)
        mask = mx.logical_or(mask, mx.arange(w.shape[-1]) == k)
    dropped = mx.sum(mx.where(mask, w, 0.0), axis=-1, keepdims=True)
    kept = mx.where(mask, 0.0, w)
    return kept * (total / mx.maximum(total - dropped, 1e-20)), dropped / mx.maximum(total, 1e-20)


class RankDrop:
    """Model-attached; reads no expert state."""

    def __init__(self, *, mode, layers=None, count=1, counts=None, store=None, protect=2, max_weight=0.10):
        if mode not in MODES:
            raise ValueError('unknown route drop mode')
        if mode == 'drop_miss_v1' and store is None:
            raise ValueError('drop_miss_v1 requires the expert store for residency')
        self.store, self.protect, self.max_weight = store, int(protect), float(max_weight)
        if counts is None:
            counts = {layer: int(count) for layer in range(int(layers[0]), int(layers[1]) + 1)}
        self.mode = mode
        self.counts = {int(k): int(v) for k, v in counts.items() if v}
        self.stats = dict(dropped_layer_rows=0, native_rows=0, decode_forwards=0)
        self._fractions = []          # this forward's lazy MLX values
        self._history = deque(maxlen=256)   # (lazy per-forward sum, rows); read only in receipt()
        self._touched = False

    @classmethod
    def from_policy(cls, policy, store=None):
        return cls(mode=policy[DROP_KEY], counts=drop_counts(policy), store=store,
                   protect=policy.get(DROP_PROTECT, 2), max_weight=policy.get(DROP_MAX_WEIGHT, 0.10))

    def applies(self, layer):
        return self.counts.get(layer, 0) > 0

    def count_for(self, layer):
        return self.counts.get(layer, 0)

    def after_forward(self, model, cache):
        if self._fractions:
            import mlx.core as mx
            self._history.append((mx.sum(mx.concatenate(self._fractions)), len(self._fractions)))
            self._fractions = []
        if self._touched:
            self.stats['decode_forwards'] += 1
            self._touched = False
            if cache is not None:
                stamp_cache(cache, model.source_approx_private_profile)

    def receipt(self):
        recent_rows = sum(n for _, n in self._history)
        weight = (sum(float(v.item()) for v, _ in self._history) / recent_rows) if recent_rows else None
        extra = ({'protect_top_ranks': self.protect, 'max_weight': self.max_weight,
                  'rule': 'drop only non-resident (original cache, lease or reduced pool) selections'}
                 if self.mode == 'drop_miss_v1' else {})
        return dict(mode=self.mode, counts_by_layer={str(k): v for k, v in sorted(self.counts.items())}, **extra, **self.stats,
                    recent_forwards=len(self._history), recent_mean_dropped_weight_fraction=weight, io_saved=True,
                    scope='B1 single-row forwards on the layer range (decode, and any 1-row prefill chunk); '
                          'dropped slots redirected to the kept top expert with weight zero (never read)')


def _native_router(block):
    from ..runtime.models.deepseek_v41 import Block
    return Block.router.__get__(block)


def install_router_wrappers(model):
    """Idempotently wrap each Block's router (blocks are rebuilt on restore).

    Always the outermost wrapper: it calls the bias router when one is installed.
    """
    for block in model.layers:
        if getattr(block, '_mlx2_route_drop_wrapped', False):
            continue
        block.router = _make_router(block)
        block._mlx2_route_drop_wrapped = True


def _make_router(block):
    native_router = _native_router(block)

    def router(x, image_mask=None):
        drop = getattr(block.model, 'source_route_drop', None)
        inner = getattr(block, '_mlx2_route_bias_router', None) or native_router
        native = inner(x) if image_mask is None else inner(x, image_mask=image_mask)
        if (drop is None or image_mask is not None or not drop.applies(block.layer)
                or getattr(x, 'ndim', 0) != 3 or x.shape[0] != 1 or x.shape[1] != 1):
            if drop is not None and getattr(x, 'ndim', 0) == 3:
                drop.stats['native_rows'] += int(x.shape[1])
            return native
        indices, weights = native
        if drop.mode == 'drop_miss_v1':
            import mlx.core as mx
            resident = mx.array(resident_or_pooled(drop.store, block.layer, block.c.n_routed_experts))[indices]
            new_weights, fraction = drop_missing(weights, resident, count=drop.count_for(block.layer),
                                                 protect=drop.protect, max_weight=drop.max_weight)
        else:
            new_weights, fraction = drop_min(weights, drop.count_for(block.layer))
        indices = redirect_dropped(indices, new_weights)
        drop.stats['dropped_layer_rows'] += 1
        drop._fractions.append(fraction.reshape(-1))
        drop._touched = True
        return indices, new_weights.astype(weights.dtype)

    return router

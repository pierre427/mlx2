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

This is a QUALITY probe: the zero-weight expert is still selected, so it may
still be read and computed. It saves no I/O or compute until the expert stream
learns to skip zero-weight slots. Isolation and APCv2 refusal: see
:mod:`deepseek_v41_source_approx`. Offline sizing (router weights by rank,
drop-if-miss curves): docs/research/ds41-source-expert-cache-design-2026-11-08.md.
"""
from collections import deque

import numpy as np

from .deepseek_v41_source_approx import DROP_COUNT, DROP_KEY, DROP_LAYERS, stamp_cache

MODES = ('drop_min_v1',)


def reference_drop(weights, count=1):
    """NumPy oracle for one row: zero the ``count`` smallest weights, keep the total."""
    w = np.asarray(weights, np.float64).copy()
    total = w.sum()
    for k in np.argsort(w, kind='stable')[:count]:
        w[k] = 0.0
    kept = w.sum()
    return w * (total / kept) if kept > 0 else w


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

    def __init__(self, *, mode, layers, count=1):
        if mode not in MODES:
            raise ValueError('unknown route drop mode')
        self.mode, self.first, self.last, self.count = mode, int(layers[0]), int(layers[1]), int(count)
        self.stats = dict(dropped_layer_rows=0, native_rows=0, decode_forwards=0)
        self._fractions = []          # this forward's lazy MLX values
        self._history = deque(maxlen=256)   # (lazy per-forward sum, rows); read only in receipt()
        self._touched = False

    @classmethod
    def from_policy(cls, policy):
        return cls(mode=policy[DROP_KEY], layers=policy[DROP_LAYERS], count=policy.get(DROP_COUNT, 1))

    def applies(self, layer):
        return self.first <= layer <= self.last

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
        return dict(mode=self.mode, layers=[self.first, self.last], count=self.count, **self.stats,
                    recent_forwards=len(self._history), recent_mean_dropped_weight_fraction=weight, io_saved=False,
                    scope='B1 single-row forwards on the layer range (decode, and any 1-row prefill chunk); '
                          'weight zeroed, expert still selected')


def _native_router(block):
    from ..runtime.models.deepseek_v41 import Block
    return Block.router.__get__(block)


def install_router_wrappers(model):
    """Idempotently wrap each Block's router (blocks are rebuilt on restore)."""
    for block in model.layers:
        if getattr(block, '_mlx2_route_drop_wrapped', False):
            continue
        block.router = _make_router(block)
        block._mlx2_route_drop_wrapped = True


def _make_router(block):
    native_router = _native_router(block)

    def router(x, image_mask=None):
        drop = getattr(block.model, 'source_route_drop', None)
        native = native_router(x) if image_mask is None else native_router(x, image_mask=image_mask)
        if (drop is None or image_mask is not None or not drop.applies(block.layer)
                or getattr(x, 'ndim', 0) != 3 or x.shape[0] != 1 or x.shape[1] != 1):
            if drop is not None and getattr(x, 'ndim', 0) == 3:
                drop.stats['native_rows'] += int(x.shape[1])
            return native
        indices, weights = native
        new_weights, fraction = drop_min(weights, drop.count)
        drop.stats['dropped_layer_rows'] += 1
        drop._fractions.append(fraction.reshape(-1))
        drop._touched = True
        return indices, new_weights.astype(weights.dtype)

    return router

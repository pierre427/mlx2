"""Cached ``lhs_indices`` for unsorted ``mx.gather_qmm`` calls.

Original mlx2 code (2026-11-22); the idea is from omlx#5538 (jundot/omlx,
Apache-2.0), reimplemented. Left to ``gather_qmm``, the default lhs indices
(an ``arange`` over the batch of ``x``, broadcast against the rhs indices)
cost one small kernel per call. Built once per shape from host data they are
a constant, and the gather reads the same indices, so the result is
identical. Unsorted calls are decode and verify rows, so the cache stays
small; sorted calls (prefill) never use it.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

_CACHE: dict = {}
_MAX_ENTRIES = 256


def lhs_indices(x_batch_shape, rhs_shape):
    key = (tuple(int(d) for d in x_batch_shape), tuple(int(d) for d in rhs_shape))
    lhs = _CACHE.get(key)
    if lhs is None:
        if len(_CACHE) >= _MAX_ENTRIES:
            _CACHE.clear()
        rows = np.arange(math.prod(key[0]), dtype=np.uint32).reshape(key[0])
        shape = np.broadcast_shapes(key[0], key[1])
        # From host data: a constant, safe to build inside a compiled trace.
        lhs = mx.array(np.ascontiguousarray(np.broadcast_to(rows, shape)))
        _CACHE[key] = lhs
    return lhs


def enabled(module) -> bool:
    return bool(module.__dict__.get("_cached_lhs_indices", False))


def set_enabled(module, on: bool) -> None:
    object.__setattr__(module, "_cached_lhs_indices", bool(on))


def maybe(module, x, indices, sorted_indices):
    """The cached lhs for this call, or None (the gather's own default)."""
    if sorted_indices or not enabled(module):
        return None
    count = module.__dict__.setdefault("_lhs_cache_counts", {"hits": 0})
    count["hits"] += 1
    return lhs_indices(x.shape[:-2], indices.shape)

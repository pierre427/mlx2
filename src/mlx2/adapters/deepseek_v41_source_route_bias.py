# SPDX-License-Identifier: MIT
"""APPROXIMATE cache-resident-biased expert selection for B1 decode, opt-in.

DS4.1 routing (``runtime/models/deepseek_v4.py: route_experts``) computes
``s = sqrt(softplus(logits))``, ranks experts by ``s + exp_probs_b`` and
combines the top-6 with ``s / sum(s) * 1.5``. This operation changes the
*selection* only: a penalty ``delta_e`` is subtracted from the rank of every
non-resident expert, the top-6 is re-selected, and the combine weights are
recomputed from the ORIGINAL ``s`` of the selected experts (renormalized as
the reference law does). Logits, bias, scale and every other operator are
unchanged.

Modes (``deepseek_source_route_bias``):

* ``resident_v1``: ``delta_e = lambda`` for every non-resident expert.
* ``resident_guarded_v1``: let ``A`` be the best rank of a resident expert
  outside the native top-6 and ``width`` the native top-1 minus top-6 rank
  spread. A non-resident expert gets ``lambda * max(0, 1 - max(rank_e-A, 0)
  / width)``: a missing expert clearly ahead of the best resident replacement
  (by at least the local spread) keeps its place. No change when every native
  expert is resident or no resident replacement exists.

``deepseek_source_route_bias_adaptive=margin_v1`` replaces the fixed lambda
by ``minimum + (maximum-minimum) * clip((g-hard)/(easy-hard), 0, 1)`` where
``g`` is the previous forward's final-row top-1 minus top-2 logit gap (a
confident previous token tolerates more bias). Non-finite previous logits or
no previous forward use ``minimum``.

Residency is read, never mutated: membership of ``(layer, expert)`` in the
expert store's resident/leased maps under its lock, at router graph
construction (per layer, so it reflects experts that earlier layers of the
same token just acquired). No LFU touch, ordering, statistic or prefetch.

Scope: B1 single-row decode forwards only. Prefill, verification and image
rows keep native routing. When no native expert would be changed the native
(indices, weights) are returned unchanged.

Expected benefit: at the measured 78 GB automatic expert maximum a warm B1
decode already reads ~0 expert bytes, so this operation has nothing to save
there. Its expected benefit is right after long prefill (when the decode
working set was evicted), under live memory pressure (adaptive maximum
shrunk), or on the 36 GB M3 where the hot set cannot fit.

Reimplemented from the idea in streaming-llama-stack
(engines/ds4-win/ds4_cache_affinity.h, ds4_cache_affinity_adaptive.h,
MIT); no code copied. Isolation and APCv2 refusal: see
:mod:`deepseek_v41_source_approx`.
"""
import math
from collections import deque
from contextlib import contextmanager

import numpy as np

from .deepseek_v41_source_approx import (
    ROUTE_ADAPTIVE,
    ROUTE_KEY,
    ROUTE_LAMBDA,
    ROUTE_MARGIN,
    stamp_cache,
)

MODES = ('resident_v1', 'resident_guarded_v1')
WIDTH_FLOOR = 1e-8


# ---------------------------------------------------------------- reference

def reference_selection(scores, bias, resident, lam, *, mode, top_k, scale, normalize=True, return_selection=False):
    """NumPy oracle for one row: returns (sorted selected IDs, weights by ID order).

    ``scores`` is ``sqrt(softplus(logits))``; ``bias`` the router correction.
    """
    scores = np.asarray(scores, np.float64)
    rank = scores + np.asarray(bias, np.float64)
    resident = np.asarray(resident, bool)
    native = np.argsort(-rank, kind='stable')[:top_k]
    missing = ~resident
    selection = rank
    if lam <= 0 or not missing[native].any():
        chosen = native
    else:
        if mode == 'resident_v1':
            delta = np.where(missing, lam, 0.0)
        elif mode == 'resident_guarded_v1':
            native_mask = np.zeros(rank.shape, bool)
            native_mask[native] = True
            alternatives = resident & ~native_mask
            if not alternatives.any():
                delta = np.zeros_like(rank)
            else:
                best = rank[alternatives].max()
                width = max(rank[native].max() - rank[native].min(), WIDTH_FLOOR)
                gap = np.maximum(rank - best, 0.0)
                delta = np.where(missing, lam * np.maximum(0.0, 1.0 - gap / width), 0.0)
        else:
            raise ValueError('unknown route bias mode')
        selection = rank - delta
        chosen = np.argsort(-selection, kind='stable')[:top_k]
    chosen = np.sort(chosen)
    weights = scores[chosen]
    if normalize:
        weights = weights / (weights.sum() + 1e-20)
    if return_selection:
        return chosen, weights * scale, selection
    return chosen, weights * scale


def margin_lambda(logits, margin):
    """Host oracle for ``margin_v1``; mirrors :func:`_margin_lambda_mx`."""
    low, high = margin['minimum'], margin['maximum']
    if low == high:
        return low
    v = np.asarray(logits, np.float64).reshape(-1)
    if np.isnan(v).any() or (v == np.inf).any():
        return low
    finite = v[v > -np.inf]
    if finite.size < 2:
        return low
    top = np.sort(finite)[-2:]
    gap = top[1] - top[0]
    frac = min(1.0, max(0.0, (gap - margin['hard_gap']) / (margin['easy_gap'] - margin['hard_gap'])))
    return low + (high - low) * frac


# ---------------------------------------------------------------- residency

def resident_mask(store, layer, experts):
    """Read-only residency of one layer's experts from the Source expert store."""
    lock = getattr(store, '_lock', None)
    tables = getattr(store, 'tables', None)
    if lock is None or not isinstance(tables, dict) or not hasattr(store, '_cache') or not hasattr(store, '_leased'):
        raise ValueError('route bias requires a Source expert store with readable residency')
    with lock:
        keys = tables.get(f'blk.{layer}')
        if keys is None or len(keys) != experts:
            raise ValueError('route bias expert table geometry mismatch')
        cached, leased = store._cache, store._leased
        return np.fromiter(((key in cached or key in leased) for key in keys), bool, count=experts)


# ---------------------------------------------------------------- MLX math

def biased_selection(logits, correction, resident, lam, *, mode, top_k, scale, normalize,
                     native_indices, native_weights):
    """Selection-only bias; returns (indices uint32, weights f32, changed bool[...,1])."""
    import mlx.core as mx
    from mlx import nn
    if mode not in MODES:
        raise ValueError('unknown route bias mode')
    scores = mx.sqrt(nn.softplus(logits.astype(mx.float32)))
    rank = scores + correction.astype(mx.float32)
    experts = rank.shape[-1]
    resident = mx.array(np.asarray(resident, bool)) if not isinstance(resident, mx.array) else resident
    missing = mx.logical_not(resident)
    native = native_indices.astype(mx.int32)
    native_mask = mx.any(native[..., :, None] == mx.arange(experts, dtype=mx.int32), axis=-2)
    native_miss = mx.any(mx.logical_and(native_mask, missing), axis=-1, keepdims=True)
    lam = mx.array(lam, mx.float32) if not isinstance(lam, mx.array) else lam.astype(mx.float32)
    if mode == 'resident_v1':
        delta = mx.where(missing, lam, 0.0)
        apply = native_miss
    else:
        alternatives = mx.logical_and(resident, mx.logical_not(native_mask))
        found = mx.any(alternatives, axis=-1, keepdims=True)
        neg = mx.array(-math.inf, mx.float32)
        best = mx.max(mx.where(alternatives, rank, neg), axis=-1, keepdims=True)
        first = mx.max(mx.where(native_mask, rank, neg), axis=-1, keepdims=True)
        last = mx.min(mx.where(native_mask, rank, -neg), axis=-1, keepdims=True)
        width = mx.maximum(first - last, WIDTH_FLOOR)
        gap = mx.maximum(rank - mx.where(found, best, 0.0), 0.0)
        delta = mx.where(missing, lam * mx.maximum(0.0, 1.0 - gap / width), 0.0)
        apply = mx.logical_and(found, native_miss)
    apply = mx.logical_and(apply, lam > 0)
    selection = rank - mx.where(apply, delta, 0.0)
    indices = mx.argpartition(-selection, kth=top_k - 1, axis=-1)[..., :top_k]
    weights = mx.take_along_axis(scores, indices, axis=-1)
    if normalize:
        weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
    weights = weights * scale
    same = mx.all(mx.take_along_axis(native_mask, indices, axis=-1), axis=-1, keepdims=True)
    changed = mx.logical_and(apply, mx.logical_not(same))
    out_indices = mx.where(changed, indices.astype(mx.uint32), native_indices.astype(mx.uint32))
    out_weights = mx.where(changed, weights, native_weights.astype(mx.float32))
    return out_indices, out_weights, changed


def _margin_lambda_mx(row, margin):
    import mlx.core as mx
    low, high = margin['minimum'], margin['maximum']
    if low == high:
        return mx.array(low, mx.float32)
    v = row.astype(mx.float32).reshape(-1)
    bad = mx.any(mx.logical_or(mx.isnan(v), v == math.inf))
    finite = v > -math.inf
    count = mx.sum(finite)
    neg = mx.array(-math.inf, mx.float32)
    top = mx.max(mx.where(finite, v, neg))
    arg = mx.argmax(mx.where(finite, v, neg))
    second = mx.max(mx.where(mx.logical_or(mx.arange(v.shape[0]) == arg, mx.logical_not(finite)), neg, v))
    frac = mx.clip((top - second - margin['hard_gap']) / (margin['easy_gap'] - margin['hard_gap']), 0.0, 1.0)
    valid = mx.logical_and(mx.logical_not(bad), count >= 2)
    return mx.where(valid, low + (high - low) * frac, low).astype(mx.float32)


# ---------------------------------------------------------------- binding

class ResidentRouteBias:
    """Model-attached selection policy; reads residency, owns no expert state."""

    def __init__(self, *, mode, store, args, lam=None, margin=None):
        if mode not in MODES:
            raise ValueError('unknown route bias mode')
        if (lam is None) == (margin is None):
            raise ValueError('route bias needs exactly one of a fixed lambda or a margin policy')
        self.mode, self.store, self.args = mode, store, args
        self.lam, self.margin = lam, margin
        self.previous_lambda = None
        self.suspended = False
        self.stats = dict(router_calls=0, decode_forwards=0, native_rows=0)
        self._changed = []
        self._history = deque(maxlen=256)

    @classmethod
    def from_policy(cls, policy, store, args):
        if store is None:
            raise ValueError('route bias requires an expert store')
        return cls(mode=policy[ROUTE_KEY], store=store, args=args,
                   lam=policy.get(ROUTE_LAMBDA) if policy[ROUTE_ADAPTIVE] == 'off' else None,
                   margin=policy.get(ROUTE_MARGIN) if policy[ROUTE_ADAPTIVE] == 'margin_v1' else None)

    def current_lambda(self):
        if self.margin is None:
            return self.lam
        return self.margin['minimum'] if self.previous_lambda is None else self.previous_lambda

    @contextmanager
    def suspend(self):
        """Diagnostic control arm (GPU script): native routing, same model."""
        previous, self.suspended = self.suspended, True
        try:
            yield self
        finally:
            self.suspended = previous

    def select(self, block, logits, correction, native):
        c = self.args
        resident = resident_mask(self.store, block.layer, c.n_routed_experts)
        indices, weights, changed = biased_selection(
            logits, correction, resident, self.current_lambda(), mode=self.mode,
            top_k=c.num_experts_per_tok, scale=c.routed_scaling_factor, normalize=c.norm_topk_prob,
            native_indices=native[0], native_weights=native[1])
        self.stats['router_calls'] += 1
        self._changed.append(changed.reshape(-1))
        return indices, weights

    def after_forward(self, model, inputs, cache, result, options):
        import mlx.core as mx
        if self._changed:
            changed, self._changed = self._changed, []
            self._history.append(mx.sum(mx.concatenate(changed).astype(mx.int32)))
            self.stats['decode_forwards'] += 1
            if cache is not None:
                stamp_cache(cache, model.source_approx_private_profile)
        if self.margin is None or options.get('head_hidden_only') or options.get('body_only'):
            return
        logits = result[0] if isinstance(result, tuple) else result
        if isinstance(logits, mx.array) and logits.ndim == 3 and logits.shape[-1] == model.args.vocab_size:
            self.previous_lambda = _margin_lambda_mx(logits[0, -1], self.margin)

    def receipt(self):
        changed = [int(value.item()) for value in self._history]
        return dict(mode=self.mode, adaptive='margin_v1' if self.margin is not None else 'off',
                    **self.stats, recent_forwards=len(changed), recent_changed_layer_rows=sum(changed),
                    suspended=self.suspended, residency_read='read-only store membership under lock')


def _native_router(block):
    from ..runtime.models.deepseek_v41 import Block
    return Block.router.__get__(block)


def install_router_wrappers(model):
    """Idempotently wrap each Block's router (blocks are rebuilt on restore)."""
    for block in model.layers:
        if getattr(block, '_mlx2_route_bias_wrapped', False):
            continue
        block.router = _make_router(block)
        block._mlx2_route_bias_wrapped = True


def _make_router(block):
    native_router = _native_router(block)

    def router(x, image_mask=None):
        bias = getattr(block.model, 'source_route_bias', None)
        if (image_mask is not None or bias is None or bias.suspended
                or getattr(x, 'ndim', 0) != 3 or x.shape[0] != 1 or x.shape[1] != 1):
            if bias is not None and getattr(x, 'ndim', 0) == 3:
                bias.stats['native_rows'] += int(x.shape[1])
            return native_router(x) if image_mask is None else native_router(x, image_mask=image_mask)
        # Same projection and native law as Block.router, computed once.
        m, c = block.model, block.c
        logits = m.linear(x, block.prefix + 'ffn_gate_inp.weight', rounded=False)
        from ..runtime.models.deepseek_v4 import route_experts
        route = route_experts
        if getattr(m, 'pointwise_policy', 'reference') == 'fused':
            from ..runtime.models.deepseek_v41_fused import router as fused_router
            from ..runtime.models.deepseek_v41_fused import self_check
            self_check()
            route = fused_router
        native = route(logits, top_k=c.num_experts_per_tok, scale=c.routed_scaling_factor,
                       correction=block.correction, normalize=c.norm_topk_prob)
        return bias.select(block, logits, block.correction, native)

    return router

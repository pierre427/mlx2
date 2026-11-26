"""Bit-exact parity: technologies.sparse_selection vs each model's old copy.

Every ``_legacy_*`` function below is the pre-migration body, verbatim (only
the enclosing ``self`` became an argument where it was a method). Each test
asserts ``mx.array_equal`` (exact) between the legacy copy and the migrated
model code on small random inputs, in every dtype the model uses, with exact
ties (integer-valued inputs give many exact-zero ReLU scores), masked rows
and offsets.
"""
from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest


DTYPES = (mx.float32, mx.float16, mx.bfloat16)


def _tied(shape, dtype, seed, *, low=-2, high=3):
    """Integer-valued (exactly representable) values: many exact ties."""
    mx.random.seed(seed)
    return mx.random.randint(low, high, shape).astype(dtype)


def _normal(shape, dtype, seed):
    mx.random.seed(seed)
    return mx.random.normal(shape).astype(dtype)


def _eq(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype, (a.shape, b.shape, a.dtype, b.dtype)
    assert mx.array_equal(a, b, equal_nan=True).item()


# --------------------------------------------------------------------------
# Kimi K3.1 (kimi_k31.py dsa_keep)


def _legacy_kimi_k31_dsa_keep(index: dict, r0: int, r1: int, topk: int, offset: int = 0) -> mx.array:
    n = offset + r1
    iq, ik, iw = index["q"][r0:r1], index["k"][:n], index["w"][r0:r1]
    logits = mx.maximum(mx.einsum("thd,sd->ths", iq, ik), 0)
    score = mx.sum(logits * iw[..., None], axis=1)                     # [rows, n]
    causal = mx.arange(n)[None, :] <= mx.arange(offset + r0, offset + r1)[:, None]
    score = mx.where(causal, score, -mx.inf)
    # Rows with fewer causal keys than top-k pick up -inf columns, which the
    # causal mask drops again: they keep every causal key.
    order = mx.argsort(-score, axis=-1)[:, :topk]
    keep = mx.put_along_axis(mx.zeros(score.shape, mx.bool_), order,
                             mx.ones(order.shape, mx.bool_), axis=-1)
    return keep & causal


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("tied", (True, False))
def test_kimi_k31_dsa_keep(dtype, tied):
    from mlx2.runtime.models import kimi_k31

    make = _tied if tied else _normal
    t, h, d, offset = 8, 4, 16, 5
    index = {
        "q": make((t, h, d), dtype, 60),
        "k": make((offset + t, d), dtype, 61),
        "w": make((t, h), dtype, 62),
    }
    for r0, r1, topk in ((0, 8, 3), (2, 6, 20), (0, 1, 1)):
        _eq(kimi_k31.dsa_keep(index, r0, r1, topk, offset),
            _legacy_kimi_k31_dsa_keep(index, r0, r1, topk, offset))


# --------------------------------------------------------------------------
# HY-V4 (hy_v4_flash.py CompressedIndexer.select, non-fused branch)


def _legacy_hy_v4_select(self, hidden, q_resid, start: int, group_keys, where=None):
    from mlx2.runtime.models.hy_v4_flash import BOUNDED_SCORE_BLOCKS, SCORE_BUDGET

    B, L, _ = hidden.shape
    G = group_keys.shape[1]
    q = self.query_projection(q_resid).reshape(B, L, self.num_heads, self.head_dim)
    # Query phases follow positions; groups (own/causal) follow slots.
    q = self._rope_tail(q.swapaxes(1, 2), start if where is None else where).astype(mx.float32)
    w = self.head_weight_projection(hidden).astype(mx.float32) * (
        self.num_heads ** -0.5 * self.head_dim ** -0.5)
    own_all = (mx.arange(start, start + L) // self.ratio).astype(mx.int32)  # [L]
    k = min(self.group_topk - 1, G)
    fused = False
    rows = max(1, SCORE_BUDGET // max(1, B * (1 if fused else self.num_heads) * max(G, 1)))
    picks = []
    for a in range(0, L, rows):
        b = min(L, a + rows)
        own = own_all[a:b]
        own_col = mx.broadcast_to(own[None, :, None], (B, b - a, 1))
        scores = nn.relu(q[:, :, a:b] @ group_keys[:, None].swapaxes(-1, -2))
        logits = (w[:, a:b].swapaxes(1, 2)[..., None] * scores).sum(axis=1)  # [B, l, G]
        valid = mx.arange(G)[None, None, :] < own[None, :, None]
        logits = mx.where(valid, logits, -mx.inf)
        if k > 0:
            picked = mx.argpartition(-logits, kth=k - 1, axis=-1)[..., :k].astype(mx.int32)
            picked = mx.where(mx.take_along_axis(valid, picked, axis=-1), picked, -1)
            picks.append(mx.concatenate([picked, own_col], axis=-1))
        else:
            picks.append(own_col)
        if BOUNDED_SCORE_BLOCKS and rows < L:
            mx.eval(picks[-1])  # one score block live at a time (see mla_attend)
    groups = picks[0] if len(picks) == 1 else mx.concatenate(picks, axis=1)
    offsets = mx.arange(self.ratio, dtype=mx.int32)
    tokens = groups[..., None] * self.ratio + offsets
    tokens = mx.where(groups[..., None] >= 0, tokens, -1)
    return mx.stop_gradient(tokens.reshape(B, L, -1).astype(mx.int32))


def _hy_fake_indexer(heads, head_dim, ratio, group_topk):
    return SimpleNamespace(
        num_heads=heads,
        head_dim=head_dim,
        ratio=ratio,
        group_topk=group_topk,
        fused_scores=False,
        query_projection=lambda x: x,
        head_weight_projection=lambda x: x[..., :heads],
        _rope_tail=lambda x, offset, scale=1.0: x,
    )


@pytest.mark.parametrize("key_dtype", DTYPES)
@pytest.mark.parametrize("tied", (True, False))
@pytest.mark.parametrize("group_topk", (1, 3, 8))
def test_hy_v4_group_select(key_dtype, tied, group_topk):
    from mlx2.runtime.models import hy_v4_flash

    make = _tied if tied else _normal
    heads, head_dim, ratio, B, L, G = 4, 8, 4, 1, 9, 6
    fake = _hy_fake_indexer(heads, head_dim, ratio, group_topk)
    hidden = make((B, L, heads * head_dim), mx.float32, 70)
    q_resid = make((B, L, heads * head_dim), mx.bfloat16, 71)
    group_keys = make((B, G, head_dim), key_dtype, 72)
    for start in (0, 13):
        new = hy_v4_flash.CompressedIndexer.select(fake, hidden, q_resid, start, group_keys)
        old = _legacy_hy_v4_select(fake, hidden, q_resid, start, group_keys)
        _eq(new, old)




@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
def test_hy_group_selection_accepts_any_leading_batch_shape(batch_shape):
    from mlx2.runtime.technologies.sparse_selection import SelectionSpec, select_groups

    scores = mx.broadcast_to(mx.array([[1., 2.], [2., 1.]]), batch_shape + (2, 2))
    own = mx.array([0, 1], mx.int32)
    spec = SelectionSpec(topk=2, tie_break="argpartition_low")
    got = select_groups(spec, scores, own)
    expected = mx.broadcast_to(mx.array([[-1, 0], [0, 1]]), batch_shape + (2, 2))
    assert mx.array_equal(got, expected).item()

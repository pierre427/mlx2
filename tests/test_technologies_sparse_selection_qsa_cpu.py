"""Bit-exact parity: technologies.sparse_selection vs each model's old copy.

Every ``_legacy_*`` function below is the pre-migration body, verbatim (only
the enclosing ``self`` became an argument where it was a method). Each test
asserts ``mx.array_equal`` (exact) between the legacy copy and the migrated
model code on small random inputs, in every dtype the model uses, with exact
ties (integer-valued inputs give many exact-zero ReLU scores), masked rows
and offsets.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx2.runtime.technologies import sparse_selection as ss

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
# Qwen4 QSA (qwen4_exp.py three sites, qwen4_qsa_stage1.py, qwen4_qsa_scores.py)


def _legacy_qwen4_qsa_score(q, keys, head_dim):
    values = mx.einsum(
        "blhd,bnd->blnh", q.astype(mx.float32), keys.astype(mx.float32)
    )
    return mx.sum(mx.maximum(values, 0), axis=-1) / math.sqrt(head_dim)


def _legacy_qwen4_qsa_stock_scores(q, pooled, valid_blocks, head_dim):
    scores = mx.einsum(
        "blhd,bnd->blnh", q.astype(mx.float32), pooled.astype(mx.float32)
    )
    scores = mx.sum(mx.maximum(scores, 0), axis=-1) / math.sqrt(head_dim)
    return mx.where(valid_blocks, scores, -mx.inf)


def _legacy_qwen4_qsa_exact_band(q, candidate_keys):
    exact = mx.einsum(
        "lhd,lcd->lch", q[0].astype(mx.float32), candidate_keys.astype(mx.float32)
    )
    return mx.sum(mx.maximum(exact, 0), axis=-1) / math.sqrt(q.shape[-1])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("tied", (True, False))
def test_qwen4_qsa_scores_and_topk(dtype, tied):
    from mlx2.runtime.models import qwen4_exp, qwen4_qsa_scores

    make = _tied if tied else _normal
    b, length, h, d, n = 2, 4, 3, 16, 10
    q = make((b, length, h, d), dtype, 80)
    pooled = make((b, n, d), dtype, 81)
    spec = qwen4_exp._qsa_selection_spec(d)
    _eq(ss.score(spec, q, pooled), _legacy_qwen4_qsa_score(q, pooled, d))
    valid = (mx.arange(n)[None, None, :] < (mx.arange(length)[None, :, None] + 3))
    _eq(qwen4_qsa_scores.stock_scores(q, pooled, valid, d),
        _legacy_qwen4_qsa_stock_scores(q, pooled, valid, d))
    scores = mx.where(valid, _legacy_qwen4_qsa_score(q, pooled, d), -mx.inf)
    for k in (1, 4, n):
        _eq(ss.topk(spec, scores, k),
            mx.argpartition(scores, kth=n - k, axis=-1)[..., -k:])
    cand = make((length, 5, d), dtype, 82)
    stage1_spec = ss.SelectionSpec(score_layout="lhd_rows_einsum", score_weighting="uniform",
                                   score_scale_site="divide", score_cast="float32")
    _eq(ss.score(stage1_spec, q[:1][0], cand), _legacy_qwen4_qsa_exact_band(q[:1], cand))


# --------------------------------------------------------------------------
# QSA sidecar + trainer (qsa_sidecar.py LayerIndexer.scores, qsa_indexer_train.py)


def _legacy_qsa_sidecar_scores(self, q: mx.array, q_chunk: int = 512) -> mx.array:
    """[H, L, D] queries -> [L, nb] block scores (invalid blocks not masked)."""
    kb = self.blocks
    out = []
    for start in range(0, q.shape[1], q_chunk):
        part = q[:, start:start + q_chunk]
        acc = None
        for h in range(self.heads):
            s = mx.maximum(part[h] @ kb.T, 0.0)
            acc = s if acc is None else acc + s
        out.append(acc)
    return out[0] if len(out) == 1 else mx.concatenate(out, axis=0)


@pytest.mark.parametrize("tied", (True, False))
def test_qsa_sidecar_and_trainer_scores(tied):
    from mlx2.runtime.models import qsa_indexer_train, qsa_sidecar

    make = _tied if tied else _normal
    heads, length, d, nb = 4, 11, 8, 6
    fake = SimpleNamespace(blocks=make((nb, d), mx.float32, 90), heads=heads)
    q = make((heads, length, d), mx.float32, 91)
    for chunk in (512, 4):
        _eq(qsa_sidecar.LayerIndexer.scores(fake, q, chunk), _legacy_qsa_sidecar_scores(fake, q, chunk))
    _eq(ss.score(qsa_indexer_train.SCORE_SPEC, q, fake.blocks),
        mx.maximum(q @ fake.blocks.T, 0.0).sum(axis=0))



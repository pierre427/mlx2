"""Bit-exact parity: technologies.sparse_selection vs each model's old copy.

Every ``_legacy_*`` function below is the pre-migration body, verbatim (only
the enclosing ``self`` became an argument where it was a method). Each test
asserts ``mx.array_equal`` (exact) between the legacy copy and the migrated
model code on small random inputs, in every dtype the model uses, with exact
ties (integer-valued inputs give many exact-zero ReLU scores), masked rows
and offsets.
"""
from __future__ import annotations


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
# GLM-5.3 (glm53_gguf_indexer.py)

_GLM_DIM = 128


def _legacy_glm53_reference_scores(q, pooled, score_weights, hd=128):
    scores = mx.maximum(q @ pooled[:, None].swapaxes(-1, -2) * hd**-0.5, 0)
    return (scores * score_weights[..., None]).sum(axis=2)


def _legacy_glm53_pool_scores_reference(q, score_weights, pooled):
    scores = mx.maximum(q @ pooled[:, None].swapaxes(-1, -2) * _GLM_DIM**-0.5, 0)
    return (scores * score_weights[..., None]).sum(axis=2)


def _legacy_glm53_stable_topk(scores, count):
    """Exact top ``count`` columns: strictly greater, then ties by ascending index.

    The law of deepseek_v4.stable_pool_topk, re-implemented here so GLM's
    numerics identity binds GLM files only. Result sorted ascending.
    """
    columns = scores.shape[-1]
    if type(count) is not int or not 1 <= count <= columns:
        raise ValueError("pool top-k must fit the available rows")
    candidates = mx.argpartition(-scores, kth=count - 1, axis=-1)[..., :count]
    cutoff = mx.min(mx.take_along_axis(scores, candidates, axis=-1), axis=-1, keepdims=True)
    better = scores > cutoff
    tied = scores == cutoff
    remaining = count - mx.sum(better, axis=-1, keepdims=True)
    keep = better | (tied & (mx.cumsum(tied.astype(mx.int32), axis=-1) <= remaining))
    positions = mx.arange(columns, dtype=mx.int32)
    ordered = mx.where(keep, positions, columns)
    selected = mx.argpartition(ordered, kth=count - 1, axis=-1)[..., :count]
    return mx.sort(selected, axis=-1)


def _legacy_glm53_masked(scores, visible):
    return mx.where(visible & mx.isfinite(scores), scores, -mx.inf)


def _legacy_glm53_select_pools(scores, visible, count, *, law):
    """(pool_ids, pool_valid), both (B, L, count); pool_valid = selected score > -inf."""
    masked = _legacy_glm53_masked(scores, visible)
    if law == "argsort":
        pool_ids = mx.argsort(-masked, axis=-1)[..., :count].astype(mx.int32)
    elif law in ("topk_v1", "topk_v1_kernel"):
        pool_ids = _legacy_glm53_stable_topk(masked, count).astype(mx.int32)
    else:
        raise ValueError(f"unknown GLM-5.3 selection law {law!r}")
    return pool_ids, mx.take_along_axis(masked, pool_ids, axis=-1) > -mx.inf


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("tied", (True, False))
def test_glm53_scores_and_select(dtype, tied):
    from mlx2.runtime.models import glm53_gguf_indexer as glm

    make = _tied if tied else _normal
    b, length, h, pools = 1, 3, 4, 9
    q = make((b, length, h, _GLM_DIM), dtype, 50)
    pooled = make((b, pools, _GLM_DIM), dtype, 51)
    w = make((b, length, h), dtype, 52) * h**-0.5
    _eq(glm.pool_scores(q, w, pooled), _legacy_glm53_pool_scores_reference(q, w, pooled))
    _eq(ss.score(glm.SELECTION_SPEC, q, pooled, w), _legacy_glm53_reference_scores(q, pooled, w))
    scores = _legacy_glm53_reference_scores(q, pooled, w).astype(mx.float32)
    positions = mx.arange(5, 5 + length)
    vis = glm.visible_pools(positions, pools)[None]
    for law in ("argsort", "topk_v1"):
        for count in (1, 4, 9):
            new = glm.select_pools(scores, vis, count, law=law)
            old = _legacy_glm53_select_pools(scores, vis, count, law=law)
            _eq(new[0], old[0])
            _eq(new[1], old[1])
    for count in (1, 4, 9):
        _eq(glm.stable_topk(scores, count), _legacy_glm53_stable_topk(scores, count))
    # reference law's inline argsort over -1e30-masked scores
    masked = mx.where(vis, scores, -1e30)
    _eq(ss.topk(glm.SELECTION_SPEC, masked, 2), mx.argsort(-masked, axis=-1)[..., :2])


def test_glm53_selection_identity_is_stable_and_spec_bound():
    from mlx2.runtime.models import glm53_gguf_indexer as glm

    ident = glm.selection_identity()
    assert ident == glm.selection_identity()
    # Another model's law (HY group selection) is not part of GLM's identity.
    import inspect

    assert ident != ss.implementation_identity(glm.SELECTION_SPEC, laws=("select_groups",))
    assert "select_groups" not in "".join(
        inspect.getsource(f) for f in ss._LAW_FUNCTIONS["stable_chronological"]
    )




def test_glm53_source_lists_bind_sparse_selection_by_path():
    """The moved laws are bound by a src/mlx2/... path (ast_physical_target regex)."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    pattern = re.compile(r"src/mlx2/[A-Za-z0-9_./-]+\Z")
    path = "src/mlx2/runtime/technologies/sparse_selection.py"
    assert pattern.fullmatch(path)
    from mlx2.adapters import glm53_ast_target

    assert path in glm53_ast_target.SOURCE_PATHS
    adapter = (root / "src/mlx2/adapters/glm53_gguf.py").read_text()
    assert f'"{path}",' in adapter and "technologies.sparse_selection:glm53" not in adapter

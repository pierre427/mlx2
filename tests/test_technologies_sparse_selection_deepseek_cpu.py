"""Bit-exact parity: technologies.sparse_selection vs each model's old copy.

Every ``_legacy_*`` function below is the pre-migration body, verbatim (only
the enclosing ``self`` became an argument where it was a method). Each test
asserts ``mx.array_equal`` (exact) between the legacy copy and the migrated
model code on small random inputs, in every dtype the model uses, with exact
ties (integer-valued inputs give many exact-zero ReLU scores), masked rows
and offsets.
"""
from __future__ import annotations

from dataclasses import replace
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
# DeepSeek V4 (deepseek_v4.py): attend, index_select, stable_pool_topk


def _legacy_deepseek_v4_attend(
    q, local, local_mask, *, pooled=None, pool_mask=None, indices=None, sinks=None
):
    """Ordinary stable attention; sink contributes denominator mass, never value."""
    # Bound score workspace independently of context length. Tiling changes no
    # attended set, masks, sinks or reductions across the key dimension.
    count = local.shape[2] + (
        0
        if pooled is None
        else (indices.shape[-1] if indices is not None else pooled.shape[1])
    )
    tile = max(
        1, min(128, (16 * 1024 * 1024) // max(1, q.shape[0] * q.shape[1] * count * 4))
    )
    if q.shape[2] > tile:
        pieces = []
        for start in range(0, q.shape[2], tile):
            end = start + tile
            pieces.append(
                _legacy_deepseek_v4_attend(
                    q[:, :, start:end],
                    local,
                    local_mask[..., start:end, :],
                    pooled=pooled,
                    pool_mask=None
                    if pool_mask is None
                    else pool_mask[..., start:end, :],
                    indices=None if indices is None else indices[:, start:end],
                    sinks=sinks,
                )
            )
        return mx.concatenate(pieces, axis=2)
    scale = q.shape[-1] ** -0.5
    if indices is None:
        kv = (
            local
            if pooled is None
            else mx.concatenate((local, pooled[:, None]), axis=2)
        )
        mask = (
            local_mask
            if pooled is None
            else mx.concatenate((local_mask, pool_mask), axis=-1)
        )
        scores = (q.astype(mx.float32) * scale) @ mx.swapaxes(
            kv.astype(mx.float32), -1, -2
        )
        scores = mx.where(mask, scores, -mx.inf)
        normalizer = mx.logsumexp(scores, axis=-1, keepdims=True)
        if sinks is not None:
            normalizer = mx.logaddexp(
                normalizer, sinks[None, :, None, None].astype(mx.float32)
            )
        return (mx.exp(scores - normalizer) @ kv.astype(mx.float32)).astype(q.dtype)
    # Each query has its own indexer-selected compressed vectors.
    b, _heads, length, dim = q.shape
    gathered = mx.take_along_axis(
        mx.broadcast_to(pooled[:, None], (b, length, pooled.shape[1], dim)),
        mx.broadcast_to(indices[..., None], (*indices.shape, dim)),
        axis=2,
    )
    query = q.transpose(0, 2, 1, 3).astype(mx.float32) * scale
    ls = (q.astype(mx.float32) * scale) @ mx.swapaxes(local.astype(mx.float32), -1, -2)
    ls = mx.where(local_mask, ls, -mx.inf)
    ps = (query @ mx.swapaxes(gathered.astype(mx.float32), -1, -2)).transpose(
        0, 2, 1, 3
    )
    selected_mask = mx.take_along_axis(pool_mask[:, 0], indices, axis=-1)[:, None]
    ps = mx.where(selected_mask, ps, -mx.inf)
    norm = mx.logaddexp(
        mx.logsumexp(ls, -1, keepdims=True), mx.logsumexp(ps, -1, keepdims=True)
    )
    if sinks is not None:
        norm = mx.logaddexp(norm, sinks[None, :, None, None].astype(mx.float32))
    result = mx.exp(ls - norm) @ local.astype(mx.float32)
    result += (
        mx.exp(ps - norm).transpose(0, 2, 1, 3) @ gathered.astype(mx.float32)
    ).transpose(0, 2, 1, 3)
    return result.astype(q.dtype)


def _legacy_deepseek_v4_index_select(
    query,
    pooled,
    weights,
    mask,
    top_k,
    *,
    workspace_bytes=16 * 1024 * 1024,
    offset=None,
    ratio=4,
):
    """Exact flat indexer with bounded query-row workspace, no coarse pruning."""
    heads, length, dim = query.shape[1:]
    tile = max(
        1,
        min(
            128, workspace_bytes // max(1, query.shape[0] * heads * pooled.shape[1] * 4)
        ),
    )
    selected = []
    for start in range(0, length, tile):
        end = start + tile
        score = query[:, :, start:end].astype(mx.float32) @ mx.swapaxes(
            pooled[:, None].astype(mx.float32), -1, -2
        )
        score = mx.maximum(score, 0)
        weight = weights[:, start:end].transpose(0, 2, 1)[..., None]
        score = mx.sum(score * weight * dim**-0.5, axis=1)
        if mask is None:
            if type(offset) is not int or offset < 0 or ratio != 4:
                raise ValueError(
                    "flat indexer requires explicit ratio4 causal geometry"
                )
            visible = (mx.arange(start, min(end, length)) + offset + 1) // ratio
            valid = mx.arange(pooled.shape[1])[None, :] < visible[:, None]
        else:
            valid = mask[:, 0, start:end]
        score = mx.where(valid, score, -mx.inf)
        selected.append(_legacy_deepseek_v4_stable_pool_topk(score, top_k))
    return mx.concatenate(selected, axis=1)


def _legacy_deepseek_v4_stable_pool_topk(scores, top_k):
    """Select exact scores with a chronological cutoff tie-break and order.

    ReLU indexer scores have exact zero ties. Partition's unspecified ordering
    must not change the attended set or summation order across replay shapes.
    See provenance/deepseek-v4-audit-64k.json for the peer mechanism review.
    """
    count = scores.shape[-1]
    if type(top_k) is not int or not 1 <= top_k <= count:
        raise ValueError("pool top-k must fit the available rows")
    candidates = mx.argpartition(-scores, kth=top_k - 1, axis=-1)[..., :top_k]
    cutoff = mx.min(
        mx.take_along_axis(scores, candidates, axis=-1), axis=-1, keepdims=True
    )
    better = scores > cutoff
    tied = scores == cutoff
    remaining = top_k - mx.sum(better, axis=-1, keepdims=True)
    keep = better | (tied & (mx.cumsum(tied.astype(mx.int32), axis=-1) <= remaining))
    positions = mx.arange(count, dtype=mx.int32)
    ordered = mx.where(keep, positions, count)
    selected = mx.argpartition(ordered, kth=top_k - 1, axis=-1)[..., :top_k]
    return mx.sort(selected, axis=-1)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("tied", (True, False))
def test_deepseek_v4_stable_pool_topk(dtype, tied):
    from mlx2.runtime.models import deepseek_v4

    make = _tied if tied else _normal
    scores = make((2, 5, 37), dtype, 1).astype(mx.float32)
    scores = mx.maximum(scores, 0)  # exact-zero ties
    scores = mx.where(mx.arange(37) < 30, scores, -mx.inf)  # masked columns
    for k in (1, 7, 30, 37):
        _eq(deepseek_v4.stable_pool_topk(scores, k), _legacy_deepseek_v4_stable_pool_topk(scores, k))
    with pytest.raises(ValueError):
        deepseek_v4.stable_pool_topk(scores, 38)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("tied", (True, False))
@pytest.mark.parametrize("offset", (0, 13))
def test_deepseek_v4_index_select(dtype, tied, offset):
    from mlx2.runtime.models import deepseek_v4

    make = _tied if tied else _normal
    b, h, length, d, pools = 1, 4, 9, 16, 12
    q = make((b, h, length, d), dtype, 2)
    pooled = make((b, pools, d), dtype, 3)
    w = make((b, length, h), mx.float32, 4) * h**-0.5
    for k in (3, 12):
        new = deepseek_v4.index_select(q, pooled, w, None, k, offset=offset, ratio=4)
        old = _legacy_deepseek_v4_index_select(q, pooled, w, None, k, offset=offset, ratio=4)
        _eq(new, old)
        # explicit mask, with a small workspace to force row tiling
        mask = (mx.arange(pools)[None, :] < (mx.arange(length)[:, None] % 5 + 1))[None, None]
        new = deepseek_v4.index_select(q, pooled, w, mask, k, workspace_bytes=600)
        old = _legacy_deepseek_v4_index_select(q, pooled, w, mask, k, workspace_bytes=600)
        _eq(new, old)
    with pytest.raises(ValueError):
        deepseek_v4.index_select(q, pooled, w, None, 3, offset=None)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("with_sinks", (True, False))
@pytest.mark.parametrize("selected", (True, False))
def test_deepseek_v4_attend(dtype, with_sinks, selected):
    from mlx2.runtime.models import deepseek_v4

    b, h, length, d, window, pools, k = 1, 4, 6, 8, 5, 7, 3
    q = _normal((b, h, length, d), dtype, 5)
    local = _normal((b, 1, window, d), dtype, 6)
    local_mask = (mx.arange(window)[None, :] <= mx.arange(length)[:, None] - 1)[None, None]
    local_mask = local_mask | (mx.arange(window)[None, :] == 0)[None, None]
    pooled = _normal((b, pools, d), dtype, 7)
    pool_mask = (mx.arange(pools)[None, :] < (mx.arange(length)[:, None] // 2))[None, None]
    sinks = _normal((h,), mx.float32, 8) if with_sinks else None
    indices = None
    if selected:
        mx.random.seed(9)
        indices = mx.random.randint(0, pools, (b, length, k)).astype(mx.uint32)
    kwargs = dict(pooled=pooled, pool_mask=pool_mask, indices=indices, sinks=sinks)
    _eq(deepseek_v4.attend(q, local, local_mask, **kwargs),
        _legacy_deepseek_v4_attend(q, local, local_mask, **kwargs))
    # local only
    _eq(deepseek_v4.attend(q, local, local_mask, sinks=sinks),
        _legacy_deepseek_v4_attend(q, local, local_mask, sinks=sinks))


def test_deepseek_v4_attend_tiled():
    """The recursive row tiling still lands on the same bits."""
    from mlx2.runtime.models import deepseek_v4

    b, h, length, d, window, pools, k = 1, 64, 140, 8, 64, 300, 16
    q = _normal((b, h, length, d), mx.bfloat16, 10)
    local = _normal((b, 1, window, d), mx.bfloat16, 11)
    local_mask = mx.ones((1, 1, length, window), mx.bool_)
    pooled = _normal((b, pools, d), mx.bfloat16, 12)
    pool_mask = (mx.arange(pools)[None, :] < (mx.arange(length)[:, None] + 1))[None, None]
    mx.random.seed(13)
    indices = mx.random.randint(0, pools, (b, length, k)).astype(mx.uint32)
    sinks = _normal((h,), mx.float32, 14)
    kwargs = dict(pooled=pooled, pool_mask=pool_mask, indices=indices, sinks=sinks)
    _eq(deepseek_v4.attend(q, local, local_mask, **kwargs),
        _legacy_deepseek_v4_attend(q, local, local_mask, **kwargs))


# --------------------------------------------------------------------------
# DeepSeek V4.1 (deepseek_v41.py, deepseek_v41_index_select.py)


def _legacy_deepseek_v41_selected_attention(
    q, local, local_mask, pooled, indices, valid, sinks, *, workspace_bytes=16 << 20,
    gathered=None, trusted_broadcast=False, packed_pooled=None, counters=None,
):
    """HySparse2-style bounded selected-row gather, no full-context score/mask.

    Attention visits the same local then chronological compressed rows as the
    ordinary V4 reference. Tile bounds include gathered vectors and scores.
    """
    from mlx2.runtime.models.deepseek_v41 import _selected_rows

    attend = _legacy_deepseek_v4_attend
    if packed_pooled is not None and (pooled is not None or trusted_broadcast):
        raise ValueError("packed selected attention requires one native source and reference math")
    if pooled is None and packed_pooled is None:
        if gathered is not None:
            raise ValueError("V4.1 reused gather requires a pooled source")
        return attend(q, local, local_mask, sinks=sinks)
    heads, length, dim = q.shape[1:]
    count = indices.shape[-1]
    if gathered is not None and (
        length != 1 or gathered.shape != (1, 1, count, dim)
        or gathered.dtype != (mx.float32 if packed_pooled is not None else pooled.dtype)
    ):
        raise ValueError("V4.1 reused gather must match width-one attention")
    per_row = 4 * (heads * (local.shape[2] + count) + count * dim + 2 * heads * dim)
    tile = max(1, min(128, workspace_bytes // max(1, per_row)))
    output = []
    for first in range(0, length, tile):
        last = min(length, first + tile)
        query = q[:, :, first:last].astype(mx.float32) * dim**-0.5
        if gathered is not None:
            rows = gathered
        elif packed_pooled is not None:
            from mlx2.runtime.models.deepseek_v41_packed_qat import gather_decode

            rows = gather_decode(packed_pooled, indices[:, first:last])
            if counters is not None:
                staged = (last-first)*count
                counters["packed_selected_decode_rows"] = counters.get("packed_selected_decode_rows", 0) + staged
                counters["packed_selected_staging_max_rows"] = max(counters.get("packed_selected_staging_max_rows", 0), staged)
        else:
            rows = (mx.broadcast_to(pooled[:, None], (1, last-first, count, dim))
                    if trusted_broadcast and length > 1 and count == pooled.shape[1] <= 512
                    else _selected_rows(pooled, indices[:, first:last]))
        ls = query @ mx.swapaxes(local.astype(mx.float32), -1, -2)
        ls = mx.where(local_mask[..., first:last, :], ls, -mx.inf)
        ps = mx.einsum("bhld,blkd->bhlk", query, rows)
        ps = mx.where(valid[:, None, first:last], ps, -mx.inf)
        norm = mx.logaddexp(
            mx.logsumexp(ls, -1, keepdims=True), mx.logsumexp(ps, -1, keepdims=True)
        )
        norm = mx.logaddexp(norm, sinks[None, :, None, None].astype(mx.float32))
        result = mx.exp(ls - norm) @ local.astype(mx.float32)
        result += mx.einsum("bhlk,blkd->bhld", mx.exp(ps - norm), rows)
        output.append(result.astype(q.dtype))
    return mx.concatenate(output, 2)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("pool_dtype", (mx.float32, mx.bfloat16))
@pytest.mark.parametrize("trusted", (False, True))
def test_deepseek_v41_selected_attention(dtype, pool_dtype, trusted):
    from mlx2.runtime.models import deepseek_v41

    b, h, length, d, window, pools = 1, 4, 7, 8, 5, 6
    q = _normal((b, h, length, d), dtype, 20)
    local = _normal((b, 1, window, d), dtype, 21)
    local_mask = (mx.arange(window)[None, :] <= mx.arange(length)[:, None])[None, None]
    pooled = _normal((b, pools, d), pool_dtype, 22)
    mx.random.seed(23)
    k = pools if trusted else 4
    indices = (mx.broadcast_to(mx.arange(pools, dtype=mx.uint32), (b, length, pools)) if trusted
               else mx.random.randint(0, pools, (b, length, k)).astype(mx.uint32))
    valid = mx.random.uniform(shape=(b, length, k)) < 0.7
    sinks = _normal((h,), mx.float32, 24)
    for ws in (16 << 20, 500):
        args = (q, local, local_mask, pooled, indices, valid, sinks)
        kw = dict(workspace_bytes=ws, trusted_broadcast=trusted)
        _eq(deepseek_v41.selected_attention(*args, **kw),
            _legacy_deepseek_v41_selected_attention(*args, **kw))
    # width-one reused gather
    q1 = q[:, :, :1]
    gathered = deepseek_v41._selected_rows(pooled, indices[:, :1])
    args = (q1, local, local_mask[..., :1, :], pooled, indices[:, :1], valid[:, :1], sinks)
    _eq(deepseek_v41.selected_attention(*args, gathered=gathered),
        _legacy_deepseek_v41_selected_attention(*args, gathered=gathered))


def _legacy_deepseek_v41_index_score(query, keys, weight):
    score = mx.einsum("bhsd,btd->bsht", query, keys)
    return mx.sum(mx.maximum(score, 0) * weight[:, :, :, None], 2)


def _legacy_deepseek_v41_candidate_score(query, keys, weight):
    small = mx.einsum("bhsd,bstd->bsht", query, keys)
    return mx.sum(mx.maximum(small, 0) * weight[:, :, :, None], 2)


def _legacy_deepseek_v41_index_visible(start, first, last, ratio, count):
    lens = ((mx.arange(start + first, start + last) + 1) // ratio)[None, :, None]
    return mx.arange(count)[None, None] < lens


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("tied", (True, False))
def test_deepseek_v41_index_scores(dtype, tied):
    from mlx2.runtime.models import deepseek_v41

    make = _tied if tied else _normal
    h, rows, d, count = 4, 5, 16, 11
    query = make((1, h, rows, d), dtype, 30)
    keys = make((1, count, d), dtype, 31)
    weight = make((1, rows, h), dtype, 32) * (d**-0.5 * h**-0.5)
    _eq(ss.score(deepseek_v41.INDEX_SPEC, query, keys, weight),
        _legacy_deepseek_v41_index_score(query, keys, weight))
    row_keys = make((1, rows, 6, d), dtype, 33)
    _eq(ss.score(deepseek_v41.INDEX_ROWS_SPEC, query, row_keys, weight),
        _legacy_deepseek_v41_candidate_score(query, row_keys, weight))
    for start, first, last, ratio in ((0, 0, 5, 2), (37, 3, 9, 4), (5, 0, 1, 1)):
        spec = replace(deepseek_v41.INDEX_SPEC, ratio=ratio)
        _eq(ss.visible(spec, mx.arange(start + first, start + last), count)[None],
            _legacy_deepseek_v41_index_visible(start, first, last, ratio, count))


def _legacy_deepseek_v41_stable_keep_mask(scores, top_k):
    """The kept-column mask of ``deepseek_v4.stable_pool_topk`` (same law)."""
    count = scores.shape[-1]
    if type(top_k) is not int or not 1 <= top_k <= count:
        raise ValueError("pool top-k must fit the available rows")
    candidates = mx.argpartition(-scores, kth=top_k - 1, axis=-1)[..., :top_k]
    cutoff = mx.min(mx.take_along_axis(scores, candidates, axis=-1), axis=-1, keepdims=True)
    better = scores > cutoff
    tied = scores == cutoff
    remaining = top_k - mx.sum(better, axis=-1, keepdims=True)
    return better | (tied & (mx.cumsum(tied.astype(mx.int32), axis=-1) <= remaining))


def _legacy_deepseek_v41_reduced_scores_chunked(query, weight, keys_for, count, chunk):
    parts = []
    for first in range(0, count, chunk):
        last = min(count, first + chunk)
        part = mx.einsum("bhsd,btd->bsht", query, keys_for(first, last))
        parts.append(mx.sum(mx.maximum(part, 0) * weight[:, :, :, None], 2))
    return parts[0] if len(parts) == 1 else mx.concatenate(parts, -1)


@pytest.mark.parametrize("tied", (True, False))
def test_deepseek_v41_index_select_rewrites(tied):
    from mlx2.runtime.models import deepseek_v41_index_select as dis

    make = _tied if tied else _normal
    scores = mx.maximum(make((1, 6, 29), mx.float32, 40), 0)
    for k in (1, 5, 29):
        _eq(dis.stable_keep_mask(scores, k), _legacy_deepseek_v41_stable_keep_mask(scores, k))
        _eq(dis.stable_topk_compact(scores, k), ss.stable_topk(scores, k).astype(mx.uint32))
    query = make((1, 4, 3, 8), mx.float32, 41)
    weight = make((1, 3, 4), mx.float32, 42)
    keys = make((1, 23, 8), mx.float32, 43)

    def keys_for(a, b):
        return keys[:, a:b]

    for chunk in (23, 7):
        _eq(dis.reduced_scores_chunked(query, weight, keys_for, 23, chunk),
            _legacy_deepseek_v41_reduced_scores_chunked(query, weight, keys_for, 23, chunk))
    ident = dis.implementation_identity()
    assert len(ident) == 64 and ident == dis.implementation_identity()


# --------------------------------------------------------------------------
# DeepSeek V4.1 Source AST batched attention (adapters/deepseek_v41_ast_batched_attention.py)


def _legacy_deepseek_v41_ast_attention_core(q,segments,sinks,selected=None,valid=None):
    """One grouped16-query graph; local table is prefix+private delta once."""
    query=q.astype(mx.float32)*q.shape[-1]**-.5
    local=segments.local_table.astype(mx.float32)
    ls=query@mx.swapaxes(local,-1,-2)
    ls=mx.where(segments.local_visibility[None,None],ls,-mx.inf)
    norm=mx.logsumexp(ls,-1,keepdims=True)
    if selected is not None:
        ps=mx.einsum('bhld,blkd->bhlk',query,selected)
        ps=mx.where(valid[:,None],ps,-mx.inf)
        norm=mx.logaddexp(norm,mx.logsumexp(ps,-1,keepdims=True))
    norm=mx.logaddexp(norm,sinks[None,:,None,None].astype(mx.float32))
    out=mx.exp(ls-norm)@local
    if selected is not None:out+=mx.einsum('bhlk,blkd->bhld',mx.exp(ps-norm),selected)
    return out.astype(q.dtype)


def _legacy_deepseek_v41_ast_score(query, index_table, weight):
    score=mx.einsum('bhld,btd->blht',query,index_table)
    return mx.sum(mx.maximum(score,0)*weight[...,None],2)[0]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("with_selected", (True, False))
@pytest.mark.parametrize("selected_dtype", (mx.float32, mx.bfloat16))
def test_deepseek_v41_ast_attention_core(dtype, with_selected, selected_dtype):
    from mlx2.adapters import deepseek_v41_ast_batched_attention as ast

    h, rows, d, local_rows, k = 4, 16, 8, 10, 5
    q = _normal((1, h, rows, d), dtype, 110)
    segments = SimpleNamespace(
        local_table=_normal((1, 1, local_rows, d), dtype, 111),
        local_visibility=mx.random.uniform(shape=(rows, local_rows)) < 0.8,
    )
    sinks = _normal((h,), mx.float32, 112)
    selected = valid = None
    if with_selected:
        selected = _normal((1, rows, k, d), selected_dtype, 113)
        valid = mx.random.uniform(shape=(1, rows, k)) < 0.6
    _eq(ast.attention_core(q, segments, sinks, selected, valid),
        _legacy_deepseek_v41_ast_attention_core(q, segments, sinks, selected, valid))
    query = _tied((1, h, rows, d), dtype, 114)
    table = _tied((1, 21, d), dtype, 115)
    weight = _tied((1, rows, h), dtype, 116)
    from mlx2.runtime.models.deepseek_v41 import INDEX_SPEC

    _eq(ss.score(INDEX_SPEC, query, table, weight)[0],
        _legacy_deepseek_v41_ast_score(query, table, weight))
    assert len(ast.implementation_identity()) == 64


def test_v41_selected_rows_reach_einsum_uncast(monkeypatch):
    """Legacy V4.1 / Source AST einsums saw bf16 rows as-is (no fp32 pre-cast)."""
    from mlx2.adapters import deepseek_v41_ast_batched_attention as ast
    from mlx2.runtime.models import deepseek_v41

    seen = []
    real = mx.einsum

    def spy(spec, *ops):
        seen.append((spec, tuple(o.dtype for o in ops)))
        return real(spec, *ops)

    monkeypatch.setattr(mx, "einsum", spy)
    h, rows, d = 2, 3, 8
    q = _normal((1, h, rows, d), mx.bfloat16, 120)
    local = _normal((1, 1, 4, d), mx.bfloat16, 121)
    mask = mx.ones((1, 1, rows, 4), mx.bool_)
    pooled = _normal((1, 5, d), mx.bfloat16, 122)
    indices = mx.zeros((1, rows, 2), mx.uint32)
    valid = mx.ones((1, rows, 2), mx.bool_)
    sinks = _normal((h,), mx.float32, 123)
    deepseek_v41.selected_attention(q, local, mask, pooled, indices, valid, sinks)
    segments = SimpleNamespace(local_table=local, local_visibility=mx.ones((rows, 4), mx.bool_))
    ast.attention_core(q, segments, sinks, _normal((1, rows, 2, d), mx.bfloat16, 124), valid)
    row_ops = [dtypes[1] for spec, dtypes in seen if spec in ("bhld,blkd->bhlk", "bhlk,blkd->bhld")]
    assert len(row_ops) == 4 and all(dt == mx.bfloat16 for dt in row_ops)
    # DeepSeek V4 (cast="float32", the default) still casts.
    assert ss.Source(local).cast == "float32"
    with pytest.raises(ValueError):
        ss.Source(local, cast="bf16")

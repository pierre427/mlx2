# SPDX-License-Identifier: MIT
"""Opt-in exact rewrites of the V4.1 CSA2 indexer selection loop.

Two independent laws, both default off and implemented_untested_unqualified
(CPU equality only; no GPU or model evidence). See
provenance/ds41-index-select-tiling.json.

``key_chunked_v1`` (policy ``deepseek_source_index_tiling``)
    The reference ``Attention.select`` sizes its query-row tile by the
    head-score scratch ``index_n_heads x count x 4`` bytes, so at a long
    history the tile collapses (64K pool rows -> 2 rows) and every tile
    re-reads the whole key history and runs its own top-k. This law sizes the
    row tile by the head-reduced score row ``count x 4`` bytes instead and
    bounds the head scratch by scoring key chunks, then concatenates the
    reduced rows and runs the unchanged selection on whole rows. Mined from
    the GLM-5.3 indexer's fixed 128-row query chunks with key blocks
    (glm53_gguf_indexer.blocked_select / QUERY_CHUNK). Each score element is
    the same 128-term dot product, ReLU, head-weight product and head sum as
    the reference; only the key slice and row-tile extents change. On CPU the
    values are bitwise equal (tests); on Metal, GEMM kernel choice may differ
    with M/N, so GPU bitwise equality is unverified.

``compact_v1`` (policy ``deepseek_source_index_select``)
    ``stable_pool_topk`` runs two arg-partitions over the row (MLX sorts on
    the GPU) plus a final sort of the kept ids. The kept mask is already
    exactly determined after the first partition, and ascending ids are a
    prefix-sum compaction of that mask. This replaces the second partition
    and the final sort by one cumsum and one scatter, and replaces the
    candidate-only consumer's full ``argsort`` + ``sort`` by the same
    compaction. Index sets, order and dtype (uint32) are identical for
    scores without NaN (a NaN row is partition-order dependent in the
    reference too).
"""

import mlx.core as mx

TILING_POLICIES = ("reference", "key_chunked_v1")
SELECT_POLICIES = ("reference", "compact_v1")
# Reduced (head-summed) score row budget per query-row tile. Same value as
# the reference head-scratch bound; the head scratch itself stays bounded by
# INDEX_SCORE_SCRATCH_BYTES through key chunks.
INDEX_REDUCED_SCORE_BYTES = 16 << 20
MAX_ROW_TILE = 128


def plan_key_chunked(length, count, heads, *, head_scratch_bytes, reduced_bytes=None):
    """Return ``(row_tile, key_chunk)`` for the key-chunked law."""
    reduced = INDEX_REDUCED_SCORE_BYTES if reduced_bytes is None else reduced_bytes
    if min(length, count, heads) < 1:
        raise ValueError("index tiling requires positive rows, keys and heads")
    rows = max(1, min(MAX_ROW_TILE, length, reduced // max(1, count * 4)))
    chunk = max(1, min(count, head_scratch_bytes // max(1, rows * heads * 4)))
    return rows, chunk


def compact_ascending(keep, width):
    """Ascending column ids of a mask with exactly ``width`` True per row.

    One prefix sum ranks the kept columns; one scatter writes each kept column
    id to its rank. Unkept columns write to a discarded extra slot.
    """
    columns = keep.shape[-1]
    if type(width) is not int or not 0 <= width <= columns:
        raise ValueError("compaction width must fit the row")
    rank = mx.cumsum(keep.astype(mx.int32), axis=-1) - 1
    slot = mx.where(keep, rank, width)
    positions = mx.broadcast_to(mx.arange(columns, dtype=mx.uint32), keep.shape)
    out = mx.zeros((*keep.shape[:-1], width + 1), mx.uint32)
    out = mx.put_along_axis(out, slot.astype(mx.int32), positions, axis=-1)
    return out[..., :width]


def stable_keep_mask(scores, top_k):
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


def stable_topk_compact(scores, top_k):
    """``stable_pool_topk`` ids (uint32, ascending) without a second partition/sort."""
    return compact_ascending(stable_keep_mask(scores, top_k), top_k)


def candidate_blocks_compact(scores, lens, count, block):
    """``deepseek_v41.candidate_blocks`` (non-shortcut branch) via the keep mask.

    The reference scatters ``picked = selected block score > -inf`` back into
    a zero mask; that equals ``keep & (block score > -inf)`` directly.
    """
    width = scores.shape[-1]
    pad = (-width) % block
    padded = mx.concatenate((scores, mx.full((*scores.shape[:-1], pad), -mx.inf)), -1) if pad else scores
    blocks = mx.max(padded.reshape(*scores.shape[:-1], -1, block), -1)
    newest = (lens - 1) // block
    blocks = mx.where(mx.arange(blocks.shape[-1])[None, None] == newest, mx.inf, blocks)
    keep = stable_keep_mask(blocks, min(count, blocks.shape[-1])) & (blocks > -mx.inf)
    return mx.repeat(keep, block, axis=-1)[..., :width]


def candidate_ids_compact(prior, capacity):
    """Candidate-only gather ids, equal to the reference argsort law.

    Reference: ``sort(argsort(where(prior, p, count + p))[..., :capacity])``:
    the first ``capacity`` candidates in position order, padded by the
    earliest non-candidates, returned ascending.
    """
    count = prior.shape[-1]
    if type(capacity) is not int or not 0 <= capacity <= count:
        raise ValueError("candidate capacity must fit the row")
    as_int = prior.astype(mx.int32)
    rank_c = mx.cumsum(as_int, axis=-1) - 1
    rank_n = mx.cumsum(1 - as_int, axis=-1) - 1
    padding = capacity - mx.minimum(mx.sum(as_int, axis=-1, keepdims=True), capacity)
    take = mx.where(prior, rank_c < capacity, rank_n < padding)
    return compact_ascending(take, capacity)


def reduced_scores_chunked(query, weight, keys_for, count, chunk):
    """Head-reduced index scores for one row tile, scored ``chunk`` keys at a time.

    ``query`` (1, H, R, D), ``weight`` (1, R, H), ``keys_for(first, last)`` ->
    (1, last-first, D) float32. Returns (1, R, count) float32 with the
    reference expression per element.
    """
    parts = []
    for first in range(0, count, chunk):
        last = min(count, first + chunk)
        part = mx.einsum("bhsd,btd->bsht", query, keys_for(first, last))
        parts.append(mx.sum(mx.maximum(part, 0) * weight[:, :, :, None], 2))
    return parts[0] if len(parts) == 1 else mx.concatenate(parts, -1)


def implementation_identity():
    """Bind numerics identity to this file's bytes until GPU bitwise evidence exists."""
    import hashlib
    from pathlib import Path

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

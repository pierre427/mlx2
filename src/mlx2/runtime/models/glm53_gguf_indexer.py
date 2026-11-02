# SPDX-License-Identifier: MIT
"""GLM-5.3 pooled indexer over exact unpadded B1 cache state.

``pooled_selection`` with its default keywords runs the reference law,
``pooled_selection_reference``: today's body verbatim, pools derived afresh
from committed raw indexer state every call, so APCv2 restore cannot retain a
stale pool. See provenance/glm53-gguf-forward.json.

The opt-in laws of docs/design/glm53-sparse-dsa.md (sections 4.1, 5.2, 6) live
beside it and are reached only through explicit keywords:

``fixed_order``  the shape-invariant pooling law (6.2), computed in full
                 (``pool_fixed_order``) or incrementally (``Glm53PoolCache``);
                 both give the same bits on one device.
``topk_v1``      the chronological selection law (``stable_topk``): strictly
                 greater pools, then tied pools by ascending index, ids in
                 ascending pool order. ``blocked_select`` is the same law over
                 pool blocks; ``topk_v1_kernel`` is the same law in one Metal
                 launch (``mlx2_glm53_select_expand_v1``), first-use gated
                 bitwise against the ops path. See provenance/glm53-sparse-dsa.json.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache

import mlx.core as mx

# The indexer key norm is nn.LayerNorm(head_dim, eps=1e-6) in the official
# modeling_glm5_next.py (GLM-5.3 DSA indexer); every other norm in the graph
# uses the GGUF's rms_norm_eps (1e-5).  This GGUF carries no
# attention.layer_norm_epsilon key, so the constant comes from the reference.
INDEXER_K_NORM_EPS = 1e-6

HEADS, DIM, KPOOL, TOPK = 32, 128, 4, 2048
TAIL = KPOOL - 1
WIDTH = TOPK + KPOOL - 1
POOL_STEP = 64  # pools per allocation step = 256 tokens = KVCache.step
QUERY_CHUNK = 128
SELECT_LAWS = ("argsort", "topk_v1", "topk_v1_kernel")
POOL_LAWS = ("reference", "fixed_order")
SELECT_KERNEL = "mlx2_glm53_select_expand_v1"
SELECT_THREADS = 1024

_SCHEDULED = Counter()
_GATE = {"select_kernel_error": None}


def pooled_selection_reference(x, qr, cache, weights, prefix, *, topk=2048, pool=4, projected=None):
    """Return token IDs [B,L,topk+pool-1], or None when all keys fit.

    Query chunks bound score memory. Every query selects complete visible
    pools plus its own incomplete tail; no first-query selection broadcast.
    """
    b, length, _ = x.shape
    if b != 1 or pool != 4:
        raise ValueError("GLM-5.3 packed indexer currently requires unpadded B1/kpool4")
    hd, heads = 128, 32
    # projected: outputs of the same projections from a grouped launch (bit-identical).
    projected = projected or {}
    key = projected["key"] if "key" in projected else weights.linear(prefix + ".indexer.attn_k.weight")(x)
    mean = key.mean(-1, keepdims=True)
    key = (key - mean) * mx.rsqrt(
        ((key - mean) ** 2).mean(-1, keepdims=True) + INDEXER_K_NORM_EPS
    )
    key = key * weights.array(prefix + ".indexer.k_norm.weight") + weights.array(
        prefix + ".indexer.k_norm.bias"
    )
    gates = weights.linear(prefix + ".indexer.kpool_gate")(x)
    packed = mx.concatenate((key, gates), axis=-1)[:, None]
    full, _ = cache.update_and_fetch(packed, mx.zeros((b, 1, length, 0), dtype=x.dtype))
    full = full[:, 0]
    total = full.shape[1]
    if total <= topk:
        return None
    q = (projected["q"] if "q" in projected
         else weights.linear(prefix + ".indexer.attn_q_b.weight")(qr)).reshape(b, length, heads, hd)
    score_weights = (projected["proj"] if "proj" in projected
                     else weights.linear(prefix + ".indexer.proj.weight")(x)) * heads**-0.5
    key, gates = mx.split(full, 2, axis=-1)
    pools = (total + pool - 1) // pool
    padding = pools * pool - total
    if padding:
        key = mx.concatenate((key, mx.zeros((b, padding, hd), key.dtype)), axis=1)
        gates = mx.concatenate((gates, mx.zeros((b, padding, hd), gates.dtype)), axis=1)
    key = key.reshape(b, pools, pool, hd)
    gates = gates.reshape(b, pools, pool, hd)
    positions = mx.arange(pools * pool).reshape(pools, pool)
    valid = positions < total
    logits = gates + weights.array(prefix + ".indexer.kpool_ape")[None, None]
    probs = mx.softmax(mx.where(valid[None, ..., None], logits, -1e30), axis=2)
    pooled = (probs * key).sum(axis=2)
    ends = mx.arange(pools) * pool + pool - 1
    count = min(topk // pool, pools)
    results = []
    for start in range(0, length, 128):
        stop = min(length, start + 128)
        pos = total - length + mx.arange(start, stop)
        scores = mx.maximum(
            q[:, start:stop] @ pooled[:, None].swapaxes(-1, -2) * hd**-0.5, 0
        )
        scores = (scores * score_weights[:, start:stop, :, None]).sum(axis=2)
        visible = ends[None, None, :] <= pos[None, :, None]
        scores = mx.where(visible, scores, -1e30)
        selected = mx.argsort(-scores, axis=-1)[..., :count]
        selected_valid = mx.take_along_axis(
            mx.broadcast_to(visible, (b, stop - start, pools)), selected, axis=-1
        )
        ids = selected[..., None] * pool + mx.arange(pool)
        ids = mx.where(selected_valid[..., None], ids, -1).reshape(
            b, stop - start, count * pool
        )
        visible_count = pos + 1
        tail_count = visible_count % pool
        tail_start = visible_count - tail_count
        tail = tail_start[:, None] + mx.arange(pool - 1)
        tail = mx.where(mx.arange(pool - 1)[None, :] < tail_count[:, None], tail, -1)
        ids = mx.concatenate(
            (ids, mx.broadcast_to(tail[None], (b, stop - start, pool - 1))), axis=-1
        )
        if ids.shape[-1] < topk + pool - 1:
            ids = mx.concatenate(
                (
                    ids,
                    mx.full(
                        (b, stop - start, topk + pool - 1 - ids.shape[-1]),
                        -1,
                        dtype=ids.dtype,
                    ),
                ),
                axis=-1,
            )
        results.append(ids.astype(mx.int32))
    return mx.concatenate(results, axis=1)


def pack_indexer_rows(x, weights, prefix, *, projected=None):
    """x (B, L, 4096) -> raw indexer rows (B, L, 256): normed key ++ kpool gate.

    The reference body's expressions (pooled_selection_reference), unchanged.
    """
    projected = projected or {}
    key = projected["key"] if "key" in projected else weights.linear(prefix + ".indexer.attn_k.weight")(x)
    mean = key.mean(-1, keepdims=True)
    key = (key - mean) * mx.rsqrt(
        ((key - mean) ** 2).mean(-1, keepdims=True) + INDEXER_K_NORM_EPS
    )
    key = key * weights.array(prefix + ".indexer.k_norm.weight") + weights.array(
        prefix + ".indexer.k_norm.bias"
    )
    gates = weights.linear(prefix + ".indexer.kpool_gate")(x)
    return mx.concatenate((key, gates), axis=-1)


def pool_reference(raw, ape):
    """raw (B, 4P, 256) complete pools -> pooled (B, P, 128): the reference law.

    The reference body's softmax/sum expression; the -1e30 validity mask is
    the identity on complete pools and is dropped.
    """
    b, rows, _ = raw.shape
    pools = rows // KPOOL
    key, gates = mx.split(raw, 2, axis=-1)
    key = key.reshape(b, pools, KPOOL, DIM)
    gates = gates.reshape(b, pools, KPOOL, DIM)
    probs = mx.softmax(gates + ape[None, None], axis=2)
    return (probs * key).sum(axis=2)


def pool_fixed_order(raw, ape):
    """raw (..., 4N, 256) -> pooled (..., N, 128): the fixed-order law of 6.2.

    Elementwise on (N, 128) slices with an explicit member order, so a pool's
    value does not depend on how many pools share the call.
    """
    if raw.shape[-2] % KPOOL or raw.shape[-1] != 2 * DIM:
        raise ValueError("GLM-5.3 fixed-order pooling needs complete pools of 256-wide rows")
    rows = raw.reshape(*raw.shape[:-2], raw.shape[-2] // KPOOL, KPOOL, 2 * DIM)
    k = [rows[..., j, :DIM] for j in range(KPOOL)]
    lg = [rows[..., j, DIM:] + ape[j] for j in range(KPOOL)]
    m = mx.maximum(mx.maximum(lg[0], lg[1]), mx.maximum(lg[2], lg[3]))
    e = [mx.exp(v - m) for v in lg]
    s = ((e[0] + e[1]) + e[2]) + e[3]
    p = [v / s for v in e]
    return ((p[0] * k[0] + p[1] * k[1]) + p[2] * k[2]) + p[3] * k[3]


class Glm53PoolCache:
    """Derived pooled-key plane of one MLA layer; never committed state.

    Pools ``[0, complete)`` are valid; ``extend`` computes the missing suffix
    with ``pool_fixed_order``. A plain object (``copy.copy`` clones it).
    """

    def __init__(self):
        self.pooled = None  # (1, capacity, 128) fp32, capacity a multiple of POOL_STEP
        self.complete = 0
        self.last_computed = 0  # pools computed by the last extend
        self.last_from = 0  # first pool computed by the last extend

    def cut(self, tokens):
        # Python ints; a scalar array offset is converted (one sync, rewinds only).
        self.complete = min(self.complete, int(tokens) // KPOOL)

    def extend(self, raw_rows_fn, total, ape, *, pool_fn=None):
        """Pools of the first ``total`` committed rows: (1, total // 4, 128).

        ``pool_fn(raw, ape)`` replaces ``pool_fixed_order`` for the missing
        suffix only when it computes the same bits (glm53_mla_fused's one-pool
        kernel, gated bitwise against it).
        """
        total = int(total)
        self.cut(total)
        target = total // KPOOL
        start = self.complete
        self.last_from, self.last_computed = start, target - start
        if target > start:
            new = (pool_fn or pool_fixed_order)(raw_rows_fn(KPOOL * start, KPOOL * target), ape)
            capacity = 0 if self.pooled is None else self.pooled.shape[1]
            if capacity < target:
                grown = mx.zeros((1, -(-target // POOL_STEP) * POOL_STEP, DIM), dtype=new.dtype)
                if start:
                    grown[:, :start] = self.pooled[:, :start]
                self.pooled = grown
            self.pooled[:, start:target] = new[None] if new.ndim == 2 else new
            self.complete = target
        if self.pooled is None:
            return mx.zeros((1, 0, DIM), dtype=mx.float32)
        return self.pooled[:, : self.complete]

    def clone(self):
        """A private successor: new array descriptor, same ``complete``."""
        other = Glm53PoolCache()
        other.pooled = None if self.pooled is None else mx.stop_gradient(self.pooled)
        other.complete = self.complete
        return other

    @property
    def nbytes(self):
        return 0 if self.pooled is None else self.pooled.nbytes


def audit_pools(pooled, raw_rows_fn, total, ape):
    """Raise unless ``pooled`` equals a full fixed-order recompute bit for bit."""
    import numpy as np

    expected = pool_fixed_order(raw_rows_fn(0, KPOOL * (int(total) // KPOOL)), ape)
    actual = pooled.reshape(-1, DIM)
    mx.eval(actual, expected)
    a, e = np.array(actual).view(np.uint32), np.array(expected).view(np.uint32)
    if a.shape != e.shape or not np.array_equal(a, e):
        words = int(np.count_nonzero(a != e)) if a.shape == e.shape else -1
        raise ValueError(f"GLM-5.3 incremental pools differ from recompute ({words} words)")
    _SCHEDULED["pool_audit"] += 1
    return True


def pool_scores(q, score_weights, pooled):
    """q (B,L,32,128), weights (B,L,32) pre-scaled, pooled (B,P,128) -> (B,L,P)."""
    scores = mx.maximum(q @ pooled[:, None].swapaxes(-1, -2) * DIM**-0.5, 0)
    return (scores * score_weights[..., None]).sum(axis=2)


def visible_pools(positions, pools, start=0):
    """positions (L,) -> (L, pools) bool for pools [start, start+pools): 4p + 3 <= pos."""
    ends = (mx.arange(start, start + pools) * KPOOL + KPOOL - 1).astype(mx.int32)
    return ends[None, :] <= positions.astype(mx.int32)[:, None]


def stable_topk(scores, count):
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


def _masked(scores, visible):
    # Visibility first, then finiteness: a nonfinite or invisible pool is -inf.
    return mx.where(visible & mx.isfinite(scores), scores, -mx.inf)


def select_pools(scores, visible, count, *, law):
    """(pool_ids, pool_valid), both (B, L, count); pool_valid = selected score > -inf."""
    masked = _masked(scores, visible)
    if law == "argsort":
        pool_ids = mx.argsort(-masked, axis=-1)[..., :count].astype(mx.int32)
    elif law in ("topk_v1", "topk_v1_kernel"):
        pool_ids = stable_topk(masked, count).astype(mx.int32)
    else:
        raise ValueError(f"unknown GLM-5.3 selection law {law!r}")
    return pool_ids, mx.take_along_axis(masked, pool_ids, axis=-1) > -mx.inf


def expand_selection(pool_ids, pool_valid, positions):
    """Token ids (B, L, WIDTH) int32: selected pools, own tail, -1 padding."""
    b, length, count = pool_ids.shape
    ids = pool_ids.astype(mx.int32)[..., None] * KPOOL + mx.arange(KPOOL, dtype=mx.int32)
    ids = mx.where(pool_valid[..., None], ids, -1).reshape(b, length, count * KPOOL)
    visible_count = positions.astype(mx.int32) + 1
    tail_count = visible_count % KPOOL
    tail_start = visible_count - tail_count
    lanes = mx.arange(TAIL, dtype=mx.int32)
    tail = tail_start[:, None] + lanes
    tail = mx.where(lanes[None, :] < tail_count[:, None], tail, -1)
    parts = [ids, mx.broadcast_to(tail[None], (b, length, TAIL))]
    if count * KPOOL + TAIL < WIDTH:
        parts.append(mx.full((b, length, WIDTH - count * KPOOL - TAIL), -1, dtype=mx.int32))
    return mx.concatenate(parts, axis=-1).astype(mx.int32)


def blocked_select(q, score_weights, pooled, positions, count, *, block=4096):
    """Running ``topk_v1`` over pool blocks: equals the global select exactly.

    Scores one block of ``block`` pools at a time, so the transient is bounded
    by the block, not by P. The running set stays in ascending pool order and
    every new block's ids are larger, so the (score desc, index asc) total
    order of the concatenation is the global one.
    """
    b, length = q.shape[:2]
    pools = pooled.shape[1]
    best_scores = mx.zeros((b, length, 0), dtype=mx.float32)
    best_ids = mx.zeros((b, length, 0), dtype=mx.int32)
    for start in range(0, pools, block):
        stop = min(pools, start + block)
        keys = pooled[:, start:stop]
        if start:
            # One live score block: block b+1's scores wait on block b's reduction.
            keys = mx.depends(keys, best_scores)
        scores = pool_scores(q, score_weights, keys)
        masked = _masked(scores, visible_pools(positions, stop - start, start)[None])
        scores = mx.concatenate((best_scores, masked.astype(mx.float32)), axis=-1)
        ids = mx.concatenate(
            (best_ids, mx.broadcast_to(mx.arange(start, stop, dtype=mx.int32), (b, length, stop - start))),
            axis=-1,
        )
        keep = stable_topk(scores, min(count, scores.shape[-1]))
        best_scores = mx.take_along_axis(scores, keep, axis=-1)
        best_ids = mx.take_along_axis(ids, keep, axis=-1)
    valid = best_scores > -mx.inf
    if best_ids.shape[-1] < count:
        pad = count - best_ids.shape[-1]
        best_ids = mx.concatenate((best_ids, mx.full((b, length, pad), -1, dtype=mx.int32)), axis=-1)
        valid = mx.concatenate((valid, mx.zeros((b, length, pad), dtype=mx.bool_)), axis=-1)
    _SCHEDULED["select_blocked"] += 1
    return best_ids, valid


def select_from_rows(x, qr, weights, prefix, *, total, raw_rows_fn, pools, pool_law, select, block,
                     projected=None):
    """Selection over rows the caller has already appended; None when total <= TOPK.

    ``total`` is T after the append (Python int); ``raw_rows_fn(start, stop)``
    returns committed raw rows (stop - start, 256). Returns ids (1, L, WIDTH)
    for the last L = x.shape[1] positions.
    """
    b, length, _ = x.shape
    if b != 1:
        raise ValueError("GLM-5.3 packed indexer currently requires unpadded B1/kpool4")
    if pool_law != "fixed_order":
        raise ValueError("select_from_rows runs the fixed_order pooling law only")
    _check_select(select, block)
    total = int(total)
    if total <= TOPK:
        return None
    projected = projected or {}
    q = (projected["q"] if "q" in projected
         else weights.linear(prefix + ".indexer.attn_q_b.weight")(qr)).reshape(b, length, HEADS, DIM)
    score_weights = (projected["proj"] if "proj" in projected
                     else weights.linear(prefix + ".indexer.proj.weight")(x)) * HEADS**-0.5
    ape = weights.array(prefix + ".indexer.kpool_ape")
    if pools is not None:
        pooled = pools.extend(raw_rows_fn, total, ape)
    else:
        pooled = pool_fixed_order(raw_rows_fn(0, KPOOL * (total // KPOOL)), ape)[None]
    count_pools = pooled.shape[1]
    count = min(TOPK // KPOOL, count_pools)
    # Prefill over many pools: blocked (bounded score transient, same law, so
    # the kernel is not needed there). Otherwise the kernel when it may serve.
    blocked = block is not None and length > 1 and count_pools > block
    kernel = not blocked and select == "topk_v1_kernel" and select_kernel_ready()
    if select == "topk_v1_kernel" and not blocked and not kernel:
        _SCHEDULED["select_kernel_declined"] += 1
    if not blocked and not kernel:
        _SCHEDULED["select_argsort" if select == "argsort" else "select_topk"] += 1
    results = []
    for start in range(0, length, QUERY_CHUNK):
        stop = min(length, start + QUERY_CHUNK)
        pos = (total - length + mx.arange(start, stop)).astype(mx.int32)
        qc, wc = q[:, start:stop], score_weights[:, start:stop]
        if results:
            # One live chunk: chunk c+1's scores wait on chunk c's ids.
            qc = mx.depends(qc, results[-1])
        if blocked:
            results.append(expand_selection(*blocked_select(qc, wc, pooled, pos, count, block=block), pos))
            continue
        scores = pool_scores(qc, wc, pooled)
        if kernel:
            results.append(select_expand(scores[0], pos, count)[None])
            continue
        pool_ids, pool_valid = select_pools(scores, visible_pools(pos, count_pools)[None], count, law=select)
        results.append(expand_selection(pool_ids, pool_valid, pos))
    return results[0] if len(results) == 1 else mx.concatenate(results, axis=1)


def _check_select(select, block):
    if select not in SELECT_LAWS:
        raise ValueError(f"unknown GLM-5.3 selection law {select!r}")
    if block is not None and select == "argsort":
        raise ValueError("GLM-5.3 blocked selection runs the topk_v1 law; argsort selects globally")
    if block is not None and (type(block) is not int or block < 1):
        raise ValueError("GLM-5.3 selection block must be a positive int")


def pooled_selection(x, qr, cache, weights, prefix, *, topk=2048, pool=4, projected=None,
                     pools=None, pool_law="reference", select="argsort", block=None):
    """Return token IDs [B,L,topk+pool-1], or None when all keys fit.

    Default keywords call ``pooled_selection_reference`` (today's law, byte for
    byte). Otherwise the rows are packed, appended to ``cache`` once, and
    selected by ``select_from_rows`` under the fixed-order pooling law.
    """
    if pool_law not in POOL_LAWS:
        raise ValueError(f"unknown GLM-5.3 pooling law {pool_law!r}")
    if pool_law == "reference":
        if pools is not None or select != "argsort" or block is not None:
            raise ValueError(
                "GLM-5.3 reference pooling runs only inside pooled_selection_reference "
                "(no pool cache, argsort selection, global scores)"
            )
        return pooled_selection_reference(x, qr, cache, weights, prefix, topk=topk, pool=pool,
                                          projected=projected)
    b, length, _ = x.shape
    if b != 1 or pool != KPOOL or topk != TOPK:
        raise ValueError("GLM-5.3 packed indexer currently requires unpadded B1/kpool4/topk2048")
    _check_select(select, block)
    packed = pack_indexer_rows(x, weights, prefix, projected=projected)[:, None]
    full, _ = cache.update_and_fetch(packed, mx.zeros((b, 1, length, 0), dtype=x.dtype))
    return select_from_rows(
        x, qr, weights, prefix, total=full.shape[2], raw_rows_fn=lambda a, z: full[0, 0, a:z],
        pools=pools, pool_law=pool_law, select=select, block=block, projected=projected,
    )


# ---------------------------------------------------------------------------
# mlx2_glm53_select_expand_v1: topk_v1 select + expand in one launch per call.
#
# One threadgroup of 1024 threads per query row. Each pool's fp32 score maps to
# a 32-bit ordered key (invisible or nonfinite -> the -inf key; -0.0 -> +0.0;
# sign bit flipped for positives, all bits flipped for negatives). Four 8-bit
# radix passes resolve the count-th largest key completely and how many pools
# tied with it are kept; one ordered sweep over pool index then emits pools
# above the cutoff and the first tied ones, in ascending pool order, as
# 4p..4p+3 when visible and finite and as -1 otherwise, followed by the
# query's tail and -1 padding. Integer, bit-level throughout (no float
# compares), so fast-math cannot change it; gated bitwise against the ops path.

_SELECT_HEADER = """
constant uint GLM53_NEG_INF_KEY = 0x007FFFFFu;
inline uint glm53_pool_key(float score, int pool, int pos) {
    if (4 * pool + 3 > pos) return GLM53_NEG_INF_KEY;
    uint bits = as_type<uint>(score);
    if ((bits & 0x7F800000u) == 0x7F800000u) return GLM53_NEG_INF_KEY;
    if (bits == 0x80000000u) bits = 0u;
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}
inline uint glm53_tg_exclusive(uint v, threadgroup uint *sums, uint lane, uint sg, thread uint &total) {
    const uint x = simd_prefix_exclusive_sum(v);
    if (lane == 31) sums[sg] = x + v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const uint s = sums[lane];
        const uint e = simd_prefix_exclusive_sum(s);
        sums[lane] = e;
        if (lane == 31) sums[32] = e + s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint result = x + sums[sg];
    total = sums[32];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}
"""

_SELECT_SOURCE = """
const uint row = threadgroup_position_in_grid.y;
const uint lid = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
const int P = meta[0], count = meta[1];
const int pos = positions[row];
const device float *s = scores + size_t(row) * size_t(P);
device int *out = ids + size_t(row) * size_t(WIDTH);
threadgroup atomic_uint hist[256];
threadgroup uint state[2];
threadgroup uint sums[33];
uint prefix = 0u, mask = 0u, want = uint(count);
for (int shift = 24; shift >= 0; shift -= 8) {
    if (lid < 256) atomic_store_explicit(&hist[lid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int p = int(lid); p < P; p += TG) {
        const uint key = glm53_pool_key(s[p], p, pos);
        if ((key & mask) == prefix)
            atomic_fetch_add_explicit(&hist[(key >> uint(shift)) & 255u], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        uint above = 0u, digit = 0u;
        for (int bin = 255; bin >= 0; --bin) {
            const uint h = atomic_load_explicit(&hist[bin], memory_order_relaxed);
            if (above + h >= want) { digit = uint(bin); break; }
            above += h;
        }
        state[0] = digit;
        state[1] = want - above;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    prefix |= state[0] << uint(shift);
    mask |= 255u << uint(shift);
    want = state[1];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
uint tied_seen = 0u, slots = 0u;
for (int base = 0; base < P; base += TG) {
    const int p = base + int(lid);
    const bool live = p < P;
    const uint key = live ? glm53_pool_key(s[p], p, pos) : 0u;
    const uint tied = (live && key == prefix) ? 1u : 0u;
    uint tied_total = 0u, take_total = 0u;
    const uint rank = tied_seen + glm53_tg_exclusive(tied, sums, lane, sg, tied_total);
    const uint take = (live && (key > prefix || (tied != 0u && rank < want))) ? 1u : 0u;
    const uint slot = slots + glm53_tg_exclusive(take, sums, lane, sg, take_total);
    if (take != 0u) {
        const bool valid = key != GLM53_NEG_INF_KEY;
        for (int i = 0; i < 4; ++i) out[slot * 4u + uint(i)] = valid ? 4 * p + i : -1;
    }
    tied_seen += tied_total;
    slots += take_total;
}
const int tail_count = (pos + 1) % 4;
const int tail_start = pos + 1 - tail_count;
for (int j = 4 * count + int(lid); j < WIDTH; j += TG) {
    const int i = j - 4 * count;
    out[j] = (i < 3 && i < tail_count) ? tail_start + i : -1;
}
"""


def select_kernel_spec():
    """The kernel's construction arguments (inspectable without Metal)."""
    return {
        "name": SELECT_KERNEL,
        "input_names": ["scores", "positions", "meta"],
        "output_names": ["ids"],
        "header": _SELECT_HEADER,
        "source": _SELECT_SOURCE,
        "ensure_row_contiguous": True,
        "template": [("WIDTH", WIDTH), ("TG", SELECT_THREADS)],
        "threadgroup": (SELECT_THREADS, 1, 1),
    }


@lru_cache(maxsize=1)
def _select_kernel():
    spec = select_kernel_spec()
    return mx.fast.metal_kernel(
        name=spec["name"], input_names=spec["input_names"], output_names=spec["output_names"],
        header=spec["header"], source=spec["source"], ensure_row_contiguous=spec["ensure_row_contiguous"],
    )


def _on_gpu():
    return mx.default_device() == mx.gpu and mx.metal.is_available()


def select_expand(scores, positions, count):
    """Kernel: scores (L, P) fp32 raw pool scores, positions (L,) -> ids (L, WIDTH).

    Equals ``expand_selection(*select_pools(scores, visible, count, law="topk_v1"))``.
    """
    if not _on_gpu():
        raise ValueError(f"{SELECT_KERNEL} requires the Metal GPU")
    length, pools = scores.shape
    if type(count) is not int or not 1 <= count <= min(pools, TOPK // KPOOL):
        raise ValueError("GLM-5.3 select kernel count must fit the pools")
    spec = select_kernel_spec()
    (ids,) = _select_kernel()(
        inputs=[scores.astype(mx.float32), positions.astype(mx.int32), mx.array([pools, count], dtype=mx.int32)],
        template=spec["template"], grid=(SELECT_THREADS, length, 1), threadgroup=spec["threadgroup"],
        output_shapes=[(length, WIDTH)], output_dtypes=[mx.int32],
    )
    _SCHEDULED["select_kernel"] += 1
    return ids


def select_expand_ops(scores, positions, count):
    """The ops law the kernel is gated against: (L, P) -> (L, WIDTH)."""
    pool_ids, pool_valid = select_pools(
        scores[None], visible_pools(positions, scores.shape[-1])[None], count, law="topk_v1"
    )
    return expand_selection(pool_ids, pool_valid, positions)[0]


def gate_cases(seed=5302):
    """Synthetic (scores, positions, count) cases for the kernel gates.

    Tie-free, heavy ties (ReLU zeros, repeated values), scores sharing their
    top 16 bits, signed zeros, nonfinite and very negative scores, prefill
    positions with invisible pools, P not a multiple of 1024 and P < 512.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    cases = []

    def add(scores, positions, count):
        cases.append((np.asarray(scores, np.float32), np.asarray(positions, np.int32), count))

    p = 1500
    add(rng.normal(size=(3, p)), [4 * p + 1, 4 * p + 2, 4 * p + 3], 512)
    tied = rng.integers(0, 5, size=(3, p)).astype(np.float32) * 0.5
    tied[:, ::7] = 0.0
    add(tied, [4 * p, 4 * p + 1, 4 * p + 3], 512)
    base = np.float32(1.5).view(np.uint32) & np.uint32(0xFFFF0000)
    shared = (base | rng.integers(0, 1 << 16, size=(2, p), dtype=np.uint32)).view(np.float32)
    add(shared, [4 * p + 2, 4 * p + 2], 512)
    signed = rng.choice(np.array([-0.0, 0.0, 1.0, -1.0], np.float32), size=(2, p))
    add(signed, [4 * p + 1, 4 * p + 2], 512)
    odd = rng.normal(size=(2, p)).astype(np.float32) * 1e3
    odd[0, :40] = np.nan
    odd[0, 40:80] = np.inf
    odd[1, 80:120] = -np.inf
    odd[1, 120:160] = -3e38
    add(odd, [4 * p + 3, 4 * p + 1], 512)
    prefill = rng.normal(size=(4, 2100)).astype(np.float32)
    add(prefill, [2048, 2050, 4100, 8399], 512)
    add(rng.normal(size=(2, 300)).astype(np.float32), [1203, 1201], 300)
    allbad = np.full((1, 600), np.nan, np.float32)
    add(allbad, [2399], 512)
    return cases


@lru_cache(maxsize=1)
def select_kernel_self_check():
    """First-use bitwise GPU gate: kernel ids == ops ids on ``gate_cases``."""
    import numpy as np

    for index, (scores, positions, count) in enumerate(gate_cases()):
        s, pos = mx.array(scores), mx.array(positions)
        actual, expected = select_expand(s, pos, count), select_expand_ops(s, pos, count)
        mx.eval(actual, expected)
        a, e = np.array(actual), np.array(expected)
        if a.shape != e.shape or not np.array_equal(a, e):
            diff = int(np.count_nonzero(a != e)) if a.shape == e.shape else -1
            raise ValueError(f"{SELECT_KERNEL} bit gate failed (case {index}, {diff} ids)")
    return True


def select_kernel_ready():
    """True when the kernel may serve: Metal GPU and the first-use gate passed."""
    if not _on_gpu() or _GATE["select_kernel_error"] is not None:
        return False
    try:
        return select_kernel_self_check()
    except (RuntimeError, ValueError) as error:  # declined, counted, reported by route_status
        _GATE["select_kernel_error"] = f"{type(error).__name__}: {error}"
        return False


def route_status():
    return {
        "policies": {"pools": ["reference", "fixed_order"], "select": list(SELECT_LAWS)},
        "kernels": {SELECT_KERNEL: {"implemented": True, "qualified": False, "selected_by_default": True,
                                    "gate_error": _GATE["select_kernel_error"]}},
        "scheduled_calls": dict(_SCHEDULED),
    }

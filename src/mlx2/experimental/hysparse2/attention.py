"""Bounded attention and exact oracle selection, expressed in differentiable MLX.

No full-context score matrix. Selection uses mean normalized attention across
query heads (an explicit research choice where HySparse2 does not specify the
head reduction). The global and forced-local sets have no overlap.
"""

import mlx.core as mx


def _logits(q, k, qp, kp, window):
    scores = (q.astype(mx.float32) @ k.astype(mx.float32).swapaxes(-1, -2)) * q.shape[
        -1
    ] ** -0.5
    valid = kp[None, :] <= qp[:, None]
    if window is not None:
        valid = valid & (kp[None, :] > qp[:, None] - window)
    return mx.where(valid[None, None], scores, -1e30), valid


def _tiles(blocks, key_tile, *, minimum=None, maximum=None):
    for k, v, start in blocks:
        lo = 0 if minimum is None else max(0, minimum - start)
        hi = k.shape[2] if maximum is None else min(k.shape[2], maximum + 1 - start)
        for offset in range(lo, hi, key_tile):
            end = min(offset + key_tile, hi)
            yield (
                k[:, :, offset:end],
                v[:, :, offset:end],
                mx.arange(start + offset, start + end),
            )


def _candidate_tiles(blocks, block_size, *, include_values=True):
    """Yield absolute, fixed-size cache blocks without joining full history."""
    pending_k, pending_v, pending_positions, pending_start = [], [], [], None
    pending_length = 0
    previous_end = None
    for k, v, start in blocks:
        if (
            type(start) is not int
            or start < 0
            or (previous_end is not None and start < previous_end)
        ):
            raise ValueError("candidate segments must be ordered and nonoverlapping")
        previous_end = start + k.shape[2]
        cursor = 0
        while cursor < k.shape[2]:
            absolute = start + cursor
            block_start = absolute - absolute % block_size
            room = block_size - (absolute - block_start)
            take = min(room, k.shape[2] - cursor)
            if pending_start is not None and block_start != pending_start:
                yield (
                    mx.concatenate(pending_k, axis=2),
                    mx.concatenate(pending_v, axis=2) if include_values else None,
                    mx.concatenate(pending_positions),
                    pending_start // block_size,
                )
                pending_k, pending_v, pending_positions, pending_length = [], [], [], 0
            pending_start = block_start
            pending_k.append(k[:, :, cursor : cursor + take])
            if include_values:
                pending_v.append(v[:, :, cursor : cursor + take])
            pending_positions.append(mx.arange(absolute, absolute + take))
            pending_length += take
            cursor += take
            if absolute + take == block_start + block_size:
                yield (
                    mx.concatenate(pending_k, axis=2),
                    mx.concatenate(pending_v, axis=2) if include_values else None,
                    mx.concatenate(pending_positions),
                    block_start // block_size,
                )
                (
                    pending_k,
                    pending_v,
                    pending_positions,
                    pending_start,
                    pending_length,
                ) = [], [], [], None, 0
    if pending_k:
        yield (
            mx.concatenate(pending_k, axis=2),
            mx.concatenate(pending_v, axis=2) if include_values else None,
            mx.concatenate(pending_positions),
            pending_start // block_size,
        )


def _candidate_groups(blocks, block_size, key_tile, *, maximum=None):
    """Bounded batches of absolute blocks for one coarse scoring operation.

    Partial blocks use causally invalid position padding, so their score is
    still the sum over their real tokens. No full-context score matrix is kept.
    """
    group_count = max(1, key_tile // block_size)
    keys, positions, ids = [], [], []
    def visible_blocks():
        for k, v, start in blocks:
            if maximum is not None:
                if start > maximum:
                    break
                size = min(k.shape[2], maximum - start + 1)
                k = k[:, :, :size]
            # Coarse selection uses keys only: do not slice or assemble values.
            yield k, None, start

    visible = list(visible_blocks())
    # Contiguous history beginning at an absolute block boundary can be
    # assembled directly at coarse tile width, including one-token decode
    # segments. Keep the general block path for gaps or unaligned beginnings.
    aligned = bool(visible) and all(
        type(start) is int and start >= 0
        and (i != 0 or start % block_size == 0)
        and k.shape[2] > 0
        and k.shape[:2] == visible[0][0].shape[:2]
        and k.shape[3:] == visible[0][0].shape[3:]
        and k.dtype == visible[0][0].dtype
        and (i == 0 or start == visible[i - 1][2] + visible[i - 1][0].shape[2])
        for i, (k, _, start) in enumerate(visible)
    )
    if aligned:
        parts, length, beginning = [], 0, visible[0][2]
        capacity = group_count * block_size

        def emit():
            key = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=2)
            padded = ((length + block_size - 1) // block_size) * block_size
            positions = mx.arange(beginning, beginning + length)
            if padded != length:
                key = mx.pad(key, [(0, 0), (0, 0), (0, padded - length), (0, 0)])
                positions = mx.pad(positions, [(0, padded - length)], constant_values=2147483647)
            return key, positions, mx.arange(beginning // block_size, beginning // block_size + padded // block_size)

        for k, _, start in visible:
            cursor = 0
            while cursor < k.shape[2]:
                if not parts:
                    beginning = start + cursor
                take = min(capacity - length, k.shape[2] - cursor)
                parts.append(k[:, :, cursor:cursor + take])
                length += take
                cursor += take
                if length == capacity:
                    yield emit()
                    parts, length = [], 0
        if parts:
            yield emit()
        return

    for k, _, kp, block_id in _candidate_tiles(visible, block_size, include_values=False):
        padding = block_size - k.shape[2]
        keys.append(mx.pad(k, [(0, 0), (0, 0), (0, padding), (0, 0)]))
        positions.append(mx.pad(kp, [(0, padding)], constant_values=2147483647))
        ids.append(block_id)
        if len(ids) == group_count:
            yield mx.concatenate(keys, axis=2), mx.concatenate(positions), mx.array(ids)
            keys, positions, ids = [], [], []
    if ids:
        yield mx.concatenate(keys, axis=2), mx.concatenate(positions), mx.array(ids)


def _gather_groups(blocks, *, max_bytes=16 << 20, max_tokens=16384):
    """Coalesce adjacent KV segments within a bounded temporary-copy budget."""
    keys, values, beginning, length, geometry = [], [], None, 0, None

    def emit():
        return (
            keys[0] if len(keys) == 1 else mx.concatenate(keys, axis=2),
            values[0] if len(values) == 1 else mx.concatenate(values, axis=2),
            beginning,
        )

    for k, v, start in blocks:
        if k.shape[2] == 0:
            continue
        current = (k.shape[:2], k.shape[3:], k.dtype, v.dtype)
        bytes_per_token = (k.nbytes + v.nbytes) // k.shape[2]
        limit = max(1, min(max_tokens, max_bytes // bytes_per_token))
        for offset in range(0, k.shape[2], limit):
            size = min(limit, k.shape[2] - offset)
            position = start + offset
            if keys and (
                position != beginning + length
                or length + size > limit
                or current != geometry
            ):
                yield emit()
                keys, values, length = [], [], 0
            if not keys:
                beginning, geometry = position, current
            keys.append(k[:, :, offset : offset + size])
            values.append(v[:, :, offset : offset + size])
            length += size
    if keys:
        yield emit()


def _rank_candidate_tokens(
    query, blocks, qp, candidate_ids, block_size, local, global_count, maximum, denom
):
    """Fine-score candidate tokens and forced local positions, not all history."""
    b, _, t, d = query.shape
    candidates = (
        candidate_ids[..., None] * block_size + mx.arange(block_size)
    ).reshape(b, t, -1)
    recent = mx.broadcast_to(
        qp[None, :, None] - local + 1 + mx.arange(local), (b, t, local)
    )
    positions = mx.concatenate((candidates, recent), axis=-1)
    k = mx.zeros((*positions.shape, d), dtype=query.dtype)
    v = mx.zeros_like(k)
    found = mx.zeros(positions.shape, dtype=mx.bool_)
    # Bounded coalescing avoids candidate-sized copies for every tiny segment.
    for keys, values, start in _gather_groups(blocks):
        present = (positions >= start) & (positions < start + keys.shape[2])
        indices = mx.clip(positions - start, 0, keys.shape[2] - 1)[..., None]
        gathered_k = mx.take_along_axis(keys[:, 0, None], indices, axis=2)
        gathered_v = mx.take_along_axis(values[:, 0, None], indices, axis=2)
        k = k + mx.where(present[..., None], gathered_k, 0)
        v = v + mx.where(present[..., None], gathered_v, 0)
        found = found | present
    scores = (
        mx.einsum(
            "bhtd,btkd->bhtk",
            mx.stop_gradient(query).astype(mx.float32),
            mx.stop_gradient(k).astype(mx.float32),
        )
        * d**-0.5
    )
    valid = found & (positions <= qp[None, :, None])
    scores = mx.where(valid[:, None], scores, -1e30)
    probabilities = mx.exp(scores - mx.stop_gradient(maximum)) / mx.maximum(
        mx.stop_gradient(denom), 1e-30
    )
    rank = mx.mean(probabilities, axis=1)
    candidate_width = candidates.shape[-1]
    # Candidate-local overlap appears only once: the forced-local copy wins.
    candidate_valid = valid[..., :candidate_width] & (
        candidates <= qp[None, :, None] - local
    )
    rank = mx.concatenate(
        (
            mx.where(candidate_valid, rank[..., :candidate_width], -1e30),
            mx.where(valid[..., candidate_width:], 2.0, -1e30),
        ),
        axis=-1,
    )
    take = min(local + global_count, sum(keys.shape[2] for keys, _, _ in blocks))
    by_position = mx.argsort(positions, axis=-1)
    by_rank = mx.argsort(-mx.take_along_axis(rank, by_position, axis=-1), axis=-1)[
        ..., :take
    ]
    indices = mx.stop_gradient(mx.take_along_axis(by_position, by_rank, axis=-1))
    best = mx.take_along_axis(rank, indices, axis=-1)
    chosen_positions = mx.take_along_axis(positions, indices, axis=-1)
    selected = (
        mx.take_along_axis(k, indices[..., None], axis=2),
        mx.take_along_axis(v, indices[..., None], axis=2),
        mx.where(best > -1e29, chosen_positions, 2147483647),
    )
    padding = take - selected[2].shape[-1]
    if padding:
        selected = (
            mx.pad(selected[0], [(0, 0), (0, 0), (0, padding), (0, 0)]),
            mx.pad(selected[1], [(0, 0), (0, 0), (0, padding), (0, 0)]),
            mx.pad(
                selected[2], [(0, 0), (0, 0), (0, padding)], constant_values=2147483647
            ),
        )
    return selected


def nlsa_cluster(keys, present, *, pool, buckets, bucket_max):
    """Cluster each completed block into at most ``buckets`` buckets.

    keys: N,B,D (B = cluster block); present: N,B bool. Follows LSA's
    clustering (arXiv 2610.42147, Algorithm 1): keys are L2-normalised and
    summed in consecutive pools, then pairs of buckets that are each other's
    cheapest merge are merged in parallel, at cost a_i + a_j - a_(i+j) with a
    the norm of a bucket's key sum, until ``buckets`` remain. A merge that
    would exceed ``bucket_max`` keys is skipped. When more mutual pairs exist
    than merges remain, the cheapest go first (ties: lower index).

    Returns centroids N,P,D (unit norm sums), active N,P and the bucket slot
    of every token N,B. Clustering depends only on one block's own keys, so
    it is identical however the history was chunked.
    """
    n, block, d = keys.shape
    pools = block // pool
    unit = keys.astype(mx.float32)
    unit = unit / mx.maximum(mx.linalg.norm(unit, axis=-1, keepdims=True), 1e-12)
    unit = mx.where(present[..., None], unit, 0.0)
    sums = unit.reshape(n, pools, pool, d).sum(axis=2)
    size = present.reshape(n, pools, pool).sum(axis=-1).astype(mx.int32)
    active = (size > 0) & (mx.linalg.norm(sums, axis=-1) > 0)
    owner = mx.broadcast_to(mx.arange(pools), (n, pools))
    index = mx.arange(pools)
    eye = index[:, None] == index[None, :]
    for _ in range(pools):
        need = mx.sum(active, axis=-1) - buckets
        gram = sums @ sums.swapaxes(-1, -2)
        norm2 = mx.maximum(mx.diagonal(gram, axis1=-2, axis2=-1), 0.0)
        norm = mx.sqrt(norm2)
        merged = mx.sqrt(
            mx.maximum(norm2[:, :, None] + norm2[:, None, :] + 2 * gram, 0.0)
        )
        cost = norm[:, :, None] + norm[:, None, :] - merged
        valid = (
            active[:, :, None]
            & active[:, None, :]
            & ~eye[None]
            & (size[:, :, None] + size[:, None, :] <= bucket_max)
        )
        cost = mx.where(valid, cost, mx.inf)
        nearest = mx.argmin(cost, axis=-1).astype(mx.int32)
        best = mx.min(cost, axis=-1)
        mutual = (
            mx.any(valid, axis=-1)
            & (mx.take_along_axis(nearest, nearest, axis=-1) == index[None])
            & (index[None] < nearest)
        )
        # Rank mutual pairs by (cost, index); keep only as many as needed.
        pair_cost = mx.where(mutual, best, mx.inf)
        order = mx.argsort(pair_cost, axis=-1)
        rank = mx.argsort(order, axis=-1)
        merge = mutual & (rank < need[:, None])
        mx.eval(merge)
        if not bool(mx.any(merge).item()):
            break
        # absorbed[j]: j is merged into the lower-index partner target[j].
        into = merge[:, :, None] & (nearest[:, :, None] == index[None, None, :])
        absorbed = mx.any(into, axis=1)
        target = mx.argmax(into.astype(mx.int32), axis=1).astype(mx.int32)
        partner_sums = mx.take_along_axis(sums, nearest[..., None], axis=1)
        partner_size = mx.take_along_axis(size, nearest, axis=-1)
        sums = mx.where(merge[..., None], sums + partner_sums, sums)
        sums = mx.where(absorbed[..., None], 0.0, sums)
        size = mx.where(merge, size + partner_size, mx.where(absorbed, 0, size))
        active = active & ~absorbed
        moved = mx.take_along_axis(absorbed, owner, axis=-1)
        owner = mx.where(moved, mx.take_along_axis(target, owner, axis=-1), owner)
    centroids = sums / mx.maximum(mx.linalg.norm(sums, axis=-1, keepdims=True), 1e-12)
    labels = mx.repeat(owner, pool, axis=-1)
    return centroids, active, labels


def _nlsa_index(blocks, block, maximum, nlsa):
    """Cluster every cluster block that ends at or before ``maximum``.

    Returns (centroids B,NB,P,D, active B,NB,P, labels B,NB,block,
    clustered NB) indexed by absolute block id, or None if no block closed.
    """
    pool, buckets, bucket_max = nlsa["pool"], nlsa["buckets"], nlsa["bucket_max"]
    closed = (maximum + 1) // block
    if closed == 0:
        return None
    rows = {}
    for k, _, kp, block_id in _candidate_tiles(
        [(k, None, start) for k, _, start in blocks], block, include_values=False
    ):
        if block_id >= closed:
            break
        rows[block_id] = (k[:, 0], kp - block_id * block)
    if not rows:
        return None
    batch, d = next(iter(rows.values()))[0].shape[0], blocks[0][0].shape[-1]
    ids = sorted(rows)
    keys, present = [], []
    for block_id in ids:
        k, local = rows[block_id]
        full = mx.zeros((batch, block, d), dtype=mx.float32)
        mask = mx.zeros((block,), dtype=mx.bool_)
        if k.shape[1] == block:
            full, mask = k.astype(mx.float32), mx.ones((block,), dtype=mx.bool_)
        else:
            hit = local[None, :] == mx.arange(block)[:, None]
            full = mx.einsum("ib,nbd->nid", hit.astype(mx.float32), k.astype(mx.float32))
            mask = mx.any(hit, axis=-1)
        keys.append(full)
        present.append(mx.broadcast_to(mask, (batch, block)))
    keys = mx.stop_gradient(mx.stack(keys, axis=1)).reshape(-1, block, d)
    present = mx.stack(present, axis=1).reshape(-1, block)
    centroids, active, labels = nlsa_cluster(
        keys, present, pool=pool, buckets=buckets, bucket_max=bucket_max
    )
    count = len(ids)
    pools = block // pool
    # Scatter into absolute block ids; gaps stay unclustered.
    slots = mx.array(ids)
    dense = closed
    out_c = mx.zeros((batch, dense, pools, d), dtype=mx.float32)
    out_a = mx.zeros((batch, dense, pools), dtype=mx.bool_)
    out_l = mx.zeros((batch, dense, block), dtype=mx.int32)
    out_c[:, slots] = centroids.reshape(batch, count, pools, d)
    out_a[:, slots] = active.reshape(batch, count, pools)
    out_l[:, slots] = labels.reshape(batch, count, block)
    clustered = mx.zeros((dense,), dtype=mx.bool_)
    clustered[slots] = True
    return out_c, out_a, out_l, clustered


def _nlsa_scores(index, query):
    """Bucket scores B,T,NB*P for one query tile, or None without an index."""
    if index is None:
        return None
    centroids, active, _, _ = index
    b = centroids.shape[0]
    qhat = mx.stop_gradient(query).astype(mx.float32)
    qhat = qhat / mx.maximum(mx.linalg.norm(qhat, axis=-1, keepdims=True), 1e-12)
    scores = mx.max(mx.einsum("bhtd,bnpd->bhtnp", qhat, centroids), axis=1)
    scores = mx.where(active[:, None], scores, -1e30)
    return scores.reshape(b, query.shape[2], -1)


def _nlsa_rank(index, scores, qp, kp, local, block, sink_tokens, batch):
    """Per-token selection rank for the N-LSA arm.

    Clustered tokens rank by their bucket's score, the maximum over query
    heads of normalised query . centroid (LSA eq. 3). Forced keys outrank
    every score: the local window (3), the rest of the window, which is the
    query's own not-yet-searchable cluster block (2.5), then sinks (2).
    Equal scores in one bucket expand in position order downstream.
    """
    visible = kp[None, :] <= qp[:, None]
    rank = mx.broadcast_to(mx.full(visible.shape, 2.5)[None], (batch, *visible.shape))
    if index is not None:
        _, _, labels, clustered = index
        b, nb = labels.shape[:2]
        pools = scores.shape[-1] // nb
        block_id = kp // block
        inside = block_id < nb
        safe = mx.where(inside, block_id, 0)
        # Keys past the last closed block must not index past its labels.
        within = mx.where(inside, kp - safe * block, 0)
        slot = labels[:, safe, within]  # B,K
        flat = safe[None] * pools + slot
        token = mx.take_along_axis(
            scores, mx.broadcast_to(flat[:, None, :], (b, qp.shape[0], kp.shape[0])), axis=-1
        )
        searchable = (
            inside[None] & clustered[safe][None]
            & ((safe + 1) * block - 1 <= qp[:, None])
        )
        rank = mx.where(searchable[None], token, rank)
    rank = mx.where((kp < sink_tokens)[None, None], mx.maximum(rank, 2.0), rank)
    recent = visible & (kp[None, :] > qp[:, None] - local)
    rank = mx.where(recent[None], 3.0, rank)
    return mx.where(visible[None], rank, -1e30)


def attention(
    q,
    blocks,
    *,
    offset,
    query_tile,
    key_tile,
    window=None,
    sinks=None,
    select=None,
    block_select=None,
    nlsa=None,
):
    """Return output and optional (selected K, V, absolute positions).

    q: B,H,T,D; blocks contain B,1,K,D MQA arrays and absolute offsets.
    Oracle selection needs a second bounded scan because head softmax
    denominators differ. Selected K/V retain gradients; token indices do not.
    """
    outputs, selections = [], []
    nlsa_index = None
    if nlsa is not None and select is not None:
        if block_select is not None:
            raise ValueError("nlsa selection replaces candidate-block selection")
        nlsa_index = _nlsa_index(
            blocks, nlsa["block"], offset + q.shape[2] - 1, nlsa
        )
    for begin in range(0, q.shape[2], query_tile):
        query = q[:, :, begin : begin + query_tile]
        b, h, t, d = query.shape
        qp = mx.arange(offset + begin, offset + begin + t)
        maximum = mx.full((b, h, t, 1), -1e30)
        denom = mx.zeros((b, h, t, 1))
        accum = mx.zeros((b, h, t, d))
        if sinks is not None:
            maximum = mx.broadcast_to(
                sinks.astype(mx.float32)[None, :, None, None], maximum.shape
            )
            denom = mx.ones_like(denom)
        minimum = None if window is None else offset + begin - window + 1
        maximum_position = offset + begin + t - 1
        for k, v, kp in _tiles(
            blocks, key_tile, minimum=minimum, maximum=maximum_position
        ):
            scores, valid = _logits(query, k, qp, kp, window)
            new_max = mx.maximum(maximum, mx.max(scores, axis=-1, keepdims=True))
            factor = mx.exp(maximum - new_max)
            weights = mx.exp(scores - new_max) * valid[None, None]
            accum = accum * factor + weights @ v.astype(mx.float32)
            denom = denom * factor + mx.sum(weights, axis=-1, keepdims=True)
            maximum = new_max
        outputs.append((accum / mx.maximum(denom, 1e-30)).astype(q.dtype))
        if select is None:
            continue
        local, global_count = select
        candidate_ids = None
        if block_select is not None:
            block_size, candidate_count = block_select
            candidate_scores = mx.zeros((b, t, 0), dtype=mx.float32)
            candidate_ids = mx.zeros((b, t, 0), dtype=mx.int32)
            for k, kp, block_ids in _candidate_groups(blocks, block_size, key_tile, maximum=maximum_position):
                scores, valid = _logits(
                    mx.stop_gradient(query), mx.stop_gradient(k), qp, kp, None
                )
                probabilities = mx.exp(scores - mx.stop_gradient(maximum)) / mx.maximum(
                    mx.stop_gradient(denom), 1e-30
                )
                token_scores = mx.mean(probabilities, axis=1)
                valid_blocks = valid.reshape(t, -1, block_size)
                score = mx.where(
                    mx.any(valid_blocks, axis=-1)[None],
                    mx.sum(
                        mx.where(valid[None], token_scores, 0.0).reshape(
                            b, t, -1, block_size
                        ),
                        axis=-1,
                    ),
                    -1e30,
                )
                block_ids = mx.broadcast_to(block_ids, score.shape)
                merged_scores = mx.concatenate((candidate_scores, score), axis=-1)
                merged_ids = mx.concatenate((candidate_ids, block_ids), axis=-1)
                take = min(candidate_count, merged_scores.shape[-1])
                order = mx.argsort(-merged_scores, axis=-1)[..., :take]
                candidate_scores = mx.take_along_axis(merged_scores, order, axis=-1)
                candidate_ids = mx.take_along_axis(merged_ids, order, axis=-1)
            if (
                sum(k.shape[2] for k, _, _ in blocks)
                > candidate_ids.shape[-1] * block_size + local
            ):
                selections.append(
                    _rank_candidate_tokens(
                        query,
                        blocks,
                        qp,
                        candidate_ids,
                        block_size,
                        local,
                        global_count,
                        maximum,
                        denom,
                    )
                )
                continue
        budget = local + global_count
        bucket_scores = (
            _nlsa_scores(nlsa_index, query) if nlsa is not None else None
        )
        best = mx.zeros((b, t, 0), dtype=mx.float32)
        sk = mx.zeros((b, t, 0, d), dtype=q.dtype)
        sv = mx.zeros_like(sk)
        positions = mx.zeros((b, t, 0), dtype=mx.int32)
        # Eligibility is per absolute token position. Fine ranking can use
        # larger bounded tiles even when coarse candidates use small blocks.
        # The N-LSA arm reads no key past the tile's last query. The oracle
        # loop keeps its historical full scan (future keys are masked).
        for k, v, kp in _tiles(
            blocks, key_tile, maximum=maximum_position if nlsa is not None else None
        ):
            if nlsa is not None:
                rank = _nlsa_rank(
                    nlsa_index,
                    bucket_scores,
                    qp,
                    kp,
                    local,
                    nlsa["block"],
                    nlsa["sinks"],
                    b,
                )
            else:
                scores, valid = _logits(
                    mx.stop_gradient(query), mx.stop_gradient(k), qp, kp, None
                )
                probabilities = mx.exp(
                    scores - mx.stop_gradient(maximum)
                ) / mx.maximum(mx.stop_gradient(denom), 1e-30)
                rank = mx.mean(probabilities, axis=1)
                recent = valid & (kp[None, :] > qp[:, None] - local)
                eligible = valid[None]
                if candidate_ids is not None:
                    block_ids = kp // block_select[0]
                    eligible = eligible & mx.any(
                        block_ids[None, None, :, None]
                        == candidate_ids[:, :, None, :],
                        axis=-1,
                    )
                rank = mx.where(recent[None], 2.0, mx.where(eligible, rank, -1e30))
            merged = mx.concatenate((best, rank), axis=-1)
            take = min(budget, merged.shape[-1])
            ck = mx.concatenate(
                (sk, mx.broadcast_to(k[:, 0, None], (b, t, k.shape[2], d))), axis=2
            )
            cv = mx.concatenate(
                (sv, mx.broadcast_to(v[:, 0, None], (b, t, v.shape[2], d))), axis=2
            )
            cp = mx.concatenate(
                (positions, mx.broadcast_to(kp, (b, t, k.shape[2]))), axis=-1
            )
            # Stable lexicographic ordering: score descending, position ascending.
            # Ties must not change the sparse support with prefill/key tile shape.
            by_position = mx.argsort(cp, axis=-1)
            by_rank = mx.argsort(
                -mx.take_along_axis(merged, by_position, axis=-1), axis=-1
            )[..., :take]
            indices = mx.stop_gradient(
                mx.take_along_axis(by_position, by_rank, axis=-1)
            )
            best = mx.take_along_axis(merged, indices, axis=-1)
            sk = mx.take_along_axis(ck, indices[..., None], axis=2)
            sv = mx.take_along_axis(cv, indices[..., None], axis=2)
            positions = mx.take_along_axis(cp, indices, axis=-1)
        # A restricted block set can contain fewer eligible tokens than the
        # requested support budget. Keep padding slots causally invalid; their
        # stored K/V must never re-enter sparse attention just because the
        # discarded source position happened to be in the past.
        positions = mx.where(best > -1e29, positions, 2147483647)
        width = min(budget, sum(k.shape[2] for k, _, _ in blocks))
        if positions.shape[-1] < width:
            # The N-LSA scan stops at the tile's last query, so early tiles
            # can see fewer keys than the oracle; pad to the same support
            # width with causally invalid slots.
            padding = width - positions.shape[-1]
            sk = mx.pad(sk, [(0, 0), (0, 0), (0, padding), (0, 0)])
            sv = mx.pad(sv, [(0, 0), (0, 0), (0, padding), (0, 0)])
            positions = mx.pad(
                positions, [(0, 0), (0, 0), (0, padding)], constant_values=2147483647
            )
        selections.append((sk, sv, positions))
    output = mx.concatenate(outputs, axis=2)
    selected = (
        None
        if not selections
        else tuple(mx.concatenate([s[i] for s in selections], axis=1) for i in range(3))
    )
    return output, selected


def sparse_attention(q, selected, *, offset, sinks):
    k, v, positions = selected
    qp = mx.arange(offset, offset + q.shape[2])
    valid = positions <= qp[None, :, None]
    scores = (
        mx.einsum("bhtd,btkd->bhtk", q.astype(mx.float32), k.astype(mx.float32))
        * q.shape[-1] ** -0.5
    )
    scores = mx.where(valid[:, None], scores, -1e30)
    sink = mx.broadcast_to(
        sinks.astype(mx.float32)[None, :, None, None], (*scores.shape[:-1], 1)
    )
    weights = mx.softmax(mx.concatenate((scores, sink), axis=-1), axis=-1)[..., :-1]
    return mx.einsum("bhtk,btkd->bhtd", weights, v.astype(mx.float32)).astype(q.dtype)

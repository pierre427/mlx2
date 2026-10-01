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


def _candidate_tiles(blocks, block_size):
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
                    mx.concatenate(pending_v, axis=2),
                    mx.concatenate(pending_positions),
                    pending_start // block_size,
                )
                pending_k, pending_v, pending_positions, pending_length = [], [], [], 0
            pending_start = block_start
            pending_k.append(k[:, :, cursor : cursor + take])
            pending_v.append(v[:, :, cursor : cursor + take])
            pending_positions.append(mx.arange(absolute, absolute + take))
            pending_length += take
            cursor += take
            if absolute + take == block_start + block_size:
                yield (
                    mx.concatenate(pending_k, axis=2),
                    mx.concatenate(pending_v, axis=2),
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
            mx.concatenate(pending_v, axis=2),
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
                k, v = k[:, :, :size], v[:, :, :size]
            yield k, v, start

    for k, _, kp, block_id in _candidate_tiles(visible_blocks(), block_size):
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
):
    """Return output and optional (selected K, V, absolute positions).

    q: B,H,T,D; blocks contain B,1,K,D MQA arrays and absolute offsets.
    Oracle selection needs a second bounded scan because head softmax
    denominators differ. Selected K/V retain gradients; token indices do not.
    """
    outputs, selections = [], []
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
        best = mx.zeros((b, t, 0), dtype=mx.float32)
        sk = mx.zeros((b, t, 0, d), dtype=q.dtype)
        sv = mx.zeros_like(sk)
        positions = mx.zeros((b, t, 0), dtype=mx.int32)
        # Eligibility is per absolute token position. Fine ranking can use
        # larger bounded tiles even when coarse candidates use small blocks.
        for k, v, kp in _tiles(blocks, key_tile):
            scores, valid = _logits(
                mx.stop_gradient(query), mx.stop_gradient(k), qp, kp, None
            )
            probabilities = mx.exp(scores - mx.stop_gradient(maximum)) / mx.maximum(
                mx.stop_gradient(denom), 1e-30
            )
            rank = mx.mean(probabilities, axis=1)
            recent = valid & (kp[None, :] > qp[:, None] - local)
            eligible = valid[None]
            if candidate_ids is not None:
                block_ids = kp // block_select[0]
                eligible = eligible & mx.any(
                    block_ids[None, None, :, None] == candidate_ids[:, :, None, :],
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

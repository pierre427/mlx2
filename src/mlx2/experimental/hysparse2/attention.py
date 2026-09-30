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


def _tiles(blocks, key_tile):
    for k, v, start in blocks:
        for offset in range(0, k.shape[2], key_tile):
            end = min(offset + key_tile, k.shape[2])
            yield (
                k[:, :, offset:end],
                v[:, :, offset:end],
                mx.arange(start + offset, start + end),
            )


def attention(
    q, blocks, *, offset, query_tile, key_tile, window=None, sinks=None, select=None
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
        for k, v, kp in _tiles(blocks, key_tile):
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
        budget = local + global_count
        best = mx.zeros((b, t, 0), dtype=mx.float32)
        sk = mx.zeros((b, t, 0, d), dtype=q.dtype)
        sv = mx.zeros_like(sk)
        positions = mx.zeros((b, t, 0), dtype=mx.int32)
        for k, v, kp in _tiles(blocks, key_tile):
            scores, valid = _logits(
                mx.stop_gradient(query), mx.stop_gradient(k), qp, kp, None
            )
            probabilities = mx.exp(scores - mx.stop_gradient(maximum)) / mx.maximum(
                mx.stop_gradient(denom), 1e-30
            )
            rank = mx.mean(probabilities, axis=1)
            recent = valid & (kp[None, :] > qp[:, None] - local)
            rank = mx.where(recent[None], 2.0, mx.where(valid[None], rank, -1e30))
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

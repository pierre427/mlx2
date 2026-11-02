"""GLM-5.3 pooled indexer over exact unpadded B1 cache state.

Pooled values are derived afresh from committed raw indexer state, so APCv2
restore cannot retain a stale pool. See provenance/glm53-gguf-forward.json.
"""

from __future__ import annotations


import mlx.core as mx

# The indexer key norm is nn.LayerNorm(head_dim, eps=1e-6) in the official
# modeling_glm5_next.py (GLM-5.3 DSA indexer); every other norm in the graph
# uses the GGUF's rms_norm_eps (1e-5).  This GGUF carries no
# attention.layer_norm_epsilon key, so the constant comes from the reference.
INDEXER_K_NORM_EPS = 1e-6


def pooled_selection(x, qr, cache, weights, prefix, *, topk=2048, pool=4):
    """Return token IDs [B,L,topk+pool-1], or None when all keys fit.

    Query chunks bound score memory. Every query selects complete visible
    pools plus its own incomplete tail; no first-query selection broadcast.
    """
    b, length, _ = x.shape
    if b != 1 or pool != 4:
        raise ValueError("GLM-5.3 packed indexer currently requires unpadded B1/kpool4")
    hd, heads = 128, 32
    key = weights.linear(prefix + ".indexer.attn_k.weight")(x)
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
    q = weights.linear(prefix + ".indexer.attn_q_b.weight")(qr).reshape(
        b, length, heads, hd
    )
    score_weights = weights.linear(prefix + ".indexer.proj.weight")(x) * heads**-0.5
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

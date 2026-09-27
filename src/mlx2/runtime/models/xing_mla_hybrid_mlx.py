"""Opt-in MLX experiment for exact shared-prefix Xing MLA attention.

This is a model-path research callable, not a serving route. It does not own
APCv2 cache state or select itself. Shared prefix K/V are expanded one tile at
a time and reused by all lanes; private suffix tiles use absorbed MLA. A Metal
kernel would be needed before this Python/MLX graph should be considered for
production performance. See ``xing_mla_research.py`` for the independent CPU
oracle and source references.
"""

from __future__ import annotations


def hybrid_shared_prefix_attention_mlx(
    *,
    query_nope,
    query_rope,
    shared_latent,
    shared_rope,
    suffix_latent,
    suffix_rope,
    embed_weight,
    unembed_weight,
    query_positions,
    shared_positions,
    suffix_positions,
    scale: float,
    tile_size: int = 1024,
):
    """Return ``[B,H,L,Dv]``; shapes/positions match the NumPy CPU oracle.

    Inputs are MLX arrays. Explicit key positions enforce the causal boundary.
    This experiment requires at least one query and one cached key; empty
    prefix or suffix is allowed. It does not inspect or mutate a cache.
    """
    import mlx.core as mx

    if isinstance(tile_size, bool) or not isinstance(tile_size, int) or tile_size <= 0:
        raise ValueError("tile_size must be a positive integer")
    b, h, l, dn = query_nope.shape
    rank = shared_latent.shape[-1]
    rope = query_rope.shape[-1]
    prefix = shared_latent.shape[0]
    suffix = suffix_latent.shape[1]
    if (
        l == 0 or prefix + suffix == 0
        or query_rope.shape != (b, h, l, rope)
        or shared_rope.shape != (prefix, rope)
        or suffix_latent.shape != (b, suffix, rank)
        or suffix_rope.shape != (b, suffix, rope)
        or embed_weight.shape != (h, rank, dn)
        or unembed_weight.shape[0] != h
        or unembed_weight.shape[-1] != rank
        or query_positions.shape != (b, l)
        or shared_positions.shape != (prefix,)
        or suffix_positions.shape != (b, suffix)
        or scale <= 0
    ):
        raise ValueError("inconsistent hybrid MLA geometry")

    # The model weights and products retain their normal dtype. Accumulate
    # softmax and values in fp32 so a long succession of tiles stays stable.
    absorbed_queries = mx.einsum("bhld,hrd->bhlr", query_nope, embed_weight)
    maximum = mx.full((b, h, l), -1e30, dtype=mx.float32)
    denominator = mx.zeros((b, h, l), dtype=mx.float32)
    numerator = mx.zeros((b, h, l, unembed_weight.shape[1]), dtype=mx.float32)

    def merge(scores, values, visible, *, latent):
        nonlocal maximum, denominator, numerator
        scores = mx.where(visible[:, None, :, :], scores.astype(mx.float32), -1e30)
        tile_max = mx.max(scores, axis=-1)
        next_max = mx.maximum(maximum, tile_max)
        previous_factor = mx.exp(maximum - next_max)
        current_factor = mx.exp(tile_max - next_max)
        probabilities = mx.where(
            visible[:, None, :, :], mx.exp(scores - tile_max[..., None]), 0.0
        )
        partial = mx.einsum("bhlt,btr->bhlr", probabilities, values.astype(mx.float32)) if latent else mx.einsum(
            "bhlt,htv->bhlv", probabilities, values.astype(mx.float32)
        )
        if latent:
            partial = mx.einsum("bhlr,hvr->bhlv", partial, unembed_weight.astype(mx.float32))
        numerator = numerator * previous_factor[..., None] + partial * current_factor[..., None]
        denominator = denominator * previous_factor + mx.sum(probabilities, axis=-1) * current_factor
        maximum = next_max

    for start in range(0, prefix, tile_size):
        end = min(start + tile_size, prefix)
        # Expanded prefix lives only for one tile, and no batch copy is made.
        tile = shared_latent[start:end]
        keys = mx.einsum("tr,hrd->htd", tile, embed_weight)
        values = mx.einsum("tr,hvr->htv", tile, unembed_weight)
        scores = mx.einsum("bhld,htd->bhlt", query_nope, keys)
        scores += mx.einsum("bhld,td->bhlt", query_rope, shared_rope[start:end])
        visible = shared_positions[None, None, start:end] <= query_positions[:, :, None]
        merge(scores * scale, values, visible, latent=False)

    for start in range(0, suffix, tile_size):
        end = min(start + tile_size, suffix)
        tile = suffix_latent[:, start:end]
        scores = mx.einsum("bhlr,btr->bhlt", absorbed_queries, tile)
        scores += mx.einsum("bhld,btd->bhlt", query_rope, suffix_rope[:, start:end])
        visible = suffix_positions[:, None, start:end] <= query_positions[:, :, None]
        merge(scores * scale, tile, visible, latent=True)

    output = mx.where(denominator[..., None] > 0, numerator / mx.maximum(denominator[..., None], 1e-30), 0.0)
    return output.astype(query_nope.dtype)

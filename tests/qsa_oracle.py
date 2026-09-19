# Adapted from unified qwen4_qsa_indexed.py at 1e2bc604, Apache-2.0.
import mlx.core as mx
import math
import numpy as np
from mlx2.runtime.models.qwen4_qsa_indexed import (
    compact_token_validity,
    mlx_sequential_merge,
    _SDPA_BLOCKS,
)


def _validate_no_duplicate_blocks(ids, counts) -> None:
    """Reject a malformed compact producer before it can double-count keys."""

    ids_np = np.asarray(ids)
    counts_np = np.asarray(counts)
    for index in np.ndindex(counts_np.shape):
        count = int(counts_np[index])
        row = ids_np[index][:count].tolist()
        if len(row) != len(set(row)):
            raise ValueError("compact QSA block ids must be unique per row")


def _reference_partials(q, k, v, compact, *, scale: float, splits: int):
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or k.shape != v.shape:
        raise ValueError("indexed QSA wants matching rank-4 q/k/v tensors")
    batch, nqh, length, dim = map(int, q.shape)
    if k.shape[0] != batch or int(k.shape[2]) != int(compact.physical_width):
        raise ValueError("indexed QSA tensors do not match compact selection")
    nkh = int(k.shape[1])
    if nkh < 1 or nqh % nkh:
        raise ValueError("indexed QSA requires integral GQA")

    ids, counts, _, u_width, _, _, _, physical, valid = compact_token_validity(compact)
    _validate_no_duplicate_blocks(ids, counts)
    if splits < 1 or splits > u_width:
        raise ValueError("splits must be in [1, u_width]")

    token_width = u_width * int(compact.block_size)
    steps = math.ceil(token_width / _SDPA_BLOCKS)
    padded_width = steps * _SDPA_BLOCKS
    padding = padded_width - token_width
    physical = physical.reshape(batch, length, token_width)
    valid = valid.reshape(batch, length, token_width)
    if padding:
        physical = mx.pad(physical, [(0, 0), (0, 0), (0, padding)])
        valid = mx.pad(
            valid,
            [(0, 0), (0, 0), (0, padding)],
            constant_values=False,
        )
    physical = physical.reshape(batch, length, steps, _SDPA_BLOCKS).transpose(
        0, 1, 3, 2
    )
    valid = valid.reshape(batch, length, steps, _SDPA_BLOCKS).transpose(0, 1, 3, 2)

    q_rows = q.transpose(0, 2, 1, 3).astype(mx.float32) * float(scale)
    k_by_token = k.transpose(0, 2, 1, 3)
    v_by_token = v.transpose(0, 2, 1, 3)
    gqa = nqh // nkh
    head_map = mx.arange(nqh, dtype=mx.int32) // gqa
    gather_index = physical[..., None, None]
    gathered_k = mx.take_along_axis(k_by_token[:, None, None], gather_index, axis=3)
    gathered_v = mx.take_along_axis(v_by_token[:, None, None], gather_index, axis=3)
    gathered_k = mx.take(gathered_k, head_map, axis=4).transpose(0, 1, 4, 2, 3, 5)
    gathered_v = mx.take(gathered_v, head_map, axis=4).transpose(0, 1, 4, 2, 3, 5)
    scores = mx.sum(q_rows[..., None, None, :] * gathered_k.astype(mx.float32), axis=-1)
    head_valid = valid[:, :, None]
    scores = mx.where(head_valid, scores, -mx.inf)
    part_m = mx.max(scores, axis=-1)
    live = mx.isfinite(part_m)
    safe_m = mx.where(live, part_m, mx.zeros_like(part_m))
    probabilities = mx.where(
        head_valid,
        mx.exp(scores - safe_m[..., None]),
        mx.zeros_like(scores),
    )
    part_l = mx.sum(probabilities, axis=-1)
    part_o = mx.sum(
        probabilities[..., None] * gathered_v.astype(mx.float32), axis=-2
    ).astype(q.dtype)
    return (
        part_m.transpose(0, 2, 1, 3),
        part_l.transpose(0, 2, 1, 3),
        part_o.transpose(0, 2, 1, 3, 4),
    )


def _combine_reference_sdpa_partials(m, l, o, *, output_dtype):
    return mlx_sequential_merge(m, l, o, output_dtype=output_dtype)


def qwen4_qsa_indexed_reference(q, k, v, compact, *, scale: float, splits: int):
    """MLX-ops mirror of fixed-chunk two-pass indexed attention."""

    m, l, o = _reference_partials(q, k, v, compact, scale=scale, splits=int(splits))
    return _combine_reference_sdpa_partials(m, l, o, output_dtype=q.dtype)


def private_delta_reference(
    q,
    base_k,
    base_v,
    delta_k,
    delta_v,
    compact,
    *,
    scale: float,
    splits: int,
):
    """Materialized oracle for the shared-base/private-delta kernel."""

    batch = int(q.shape[0])
    keys = mx.concatenate(
        [mx.broadcast_to(base_k, (batch, *base_k.shape[1:])), delta_k],
        axis=2,
    )
    values = mx.concatenate(
        [mx.broadcast_to(base_v, (batch, *base_v.shape[1:])), delta_v],
        axis=2,
    )
    return qwen4_qsa_indexed_reference(
        q, keys, values, compact, scale=scale, splits=splits
    )


from mlx2.runtime.models.qwen4_exp import TextModelArgs
def tiny_args(**overrides):
    values = dict(
        hidden_size=16,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=64,
        max_position_embeddings=64,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        hc_count=4,
        hc_lowrank=4,
        ple_layer_ids=[],
        ple_embed_dim=16,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=4,
        split_ngram_parts=4,
        eos_token_id=63,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=8,
        indexer_compress_ratio=4,
        rope_parameters={
            "type": "default",
            "rope_theta": 10000,
            "partial_rotary_factor": 0.5,
        },
    )
    values.update(overrides)
    return TextModelArgs(**values)

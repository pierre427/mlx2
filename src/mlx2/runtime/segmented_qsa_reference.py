# SPDX-License-Identifier: Apache-2.0
"""Experimental reference math for attention over shared and private segments.

This module is intentionally disconnected from production routing.  It exists
to qualify the numerical decomposition needed by a future segment-aware QSA
kernel: one immutable prefix is reduced for the whole batch, each row's
private suffix is reduced independently, and the two softmax states are
merged without concatenating or copying the shared prefix per row.

All accumulation is float32.  That preserves the mathematical attention
result, but is not expected to be bit-identical to a fused SDPA kernel with a
different reduction order.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import mlx.core as mx


@dataclass(frozen=True)
class SegmentedQSAReceipt:
    """Bounded, host-readable evidence that the experimental path ran."""

    mechanism: str
    batch_size: int
    query_length: int
    base_length: int
    suffix_lengths: tuple[int, ...]
    query_heads: int
    kv_heads: int
    head_dim: int
    shared_base_calls: int
    private_segment_calls: int
    materialized_row_calls: int
    logical_score_elements: int
    base_kv_row_read_proxy: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class _SoftmaxState:
    maximum: mx.array
    denominator: mx.array
    numerator: mx.array


def _validate(
    queries: mx.array,
    base_keys: mx.array,
    base_values: mx.array,
    suffix_keys: Sequence[mx.array],
    suffix_values: Sequence[mx.array],
) -> tuple[int, int, int, int, int]:
    if queries.ndim != 4:
        raise ValueError("queries must have shape [B, Hq, Q, D]")
    if base_keys.ndim != 4 or base_values.ndim != 4:
        raise ValueError("shared K/V must have shape [1, Hkv, K, D]")
    if int(base_keys.shape[0]) != 1 or int(base_values.shape[0]) != 1:
        raise ValueError("the immutable shared K/V base must have batch one")
    if base_keys.shape != base_values.shape:
        raise ValueError("shared K/V shapes differ")
    batch, query_heads, query_length, head_dim = map(int, queries.shape)
    _, kv_heads, base_length, key_dim = map(int, base_keys.shape)
    if min(batch, query_heads, query_length, head_dim, kv_heads, base_length) <= 0:
        raise ValueError("shared-prefix attention geometry must be positive")
    if key_dim != head_dim:
        raise ValueError("query and shared K/V head dimensions differ")
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if len(suffix_keys) != batch or len(suffix_values) != batch:
        raise ValueError("private K/V sequences must cover every query row")
    for row, (keys, values) in enumerate(zip(suffix_keys, suffix_values)):
        if keys.ndim != 4 or values.ndim != 4:
            raise ValueError(f"private K/V row {row} must be rank four")
        if keys.shape != values.shape:
            raise ValueError(f"private K/V row {row} shapes differ")
        if tuple(map(int, keys.shape[:2])) != (1, kv_heads):
            raise ValueError(f"private K/V row {row} must have B=1, Hkv={kv_heads}")
        if int(keys.shape[3]) != head_dim:
            raise ValueError(f"private K/V row {row} head dimension differs")
    return batch, query_heads, query_length, head_dim, kv_heads


def _group_queries(queries: mx.array, kv_heads: int) -> mx.array:
    batch, query_heads, query_length, head_dim = map(int, queries.shape)
    groups = query_heads // kv_heads
    return queries.astype(mx.float32).reshape(
        batch, kv_heads, groups, query_length, head_dim
    )


def _group_mask(
    mask: mx.array | None,
    *,
    batch: int,
    query_heads: int,
    kv_heads: int,
    query_length: int,
    key_length: int,
) -> mx.array | None:
    """Normalize an additive mask to [B, Hkv, G, Q, K]."""
    if mask is None:
        return None
    if mask.ndim != 4:
        raise ValueError("attention masks must have shape [B|1, Hq|1, Q, K]")
    mask_batch, mask_heads, mask_queries, mask_keys = map(int, mask.shape)
    if mask_batch not in (1, batch) or mask_heads not in (1, query_heads):
        raise ValueError("attention mask batch/head geometry is not broadcastable")
    if mask_queries != query_length or mask_keys != key_length:
        raise ValueError("attention mask query/key geometry differs")
    groups = query_heads // kv_heads
    if mask_heads == 1:
        mask = mx.broadcast_to(mask, (mask_batch, query_heads, query_length, key_length))
    if mask_batch == 1 and batch != 1:
        mask = mx.broadcast_to(mask, (batch, query_heads, query_length, key_length))
    return _additive_mask(mask).reshape(
        batch, kv_heads, groups, query_length, key_length
    )


def _additive_mask(mask: mx.array | None) -> mx.array | None:
    """Return ``mask`` as a float32 additive mask (boolean True keeps a key)."""
    if mask is None:
        return None
    if mask.dtype == mx.bool_:
        mask = mx.where(mask, mx.array(0.0), mx.array(-mx.inf))
    return mask.astype(mx.float32)


def _segment_state(
    grouped_queries: mx.array,
    keys: mx.array,
    values: mx.array,
    *,
    scale: float,
    mask: mx.array | None,
) -> _SoftmaxState:
    """Reduce one K/V segment into mergeable online-softmax state."""
    batch, kv_heads, groups, query_length, head_dim = map(
        int, grouped_queries.shape
    )
    key_length = int(keys.shape[2])
    if key_length == 0:
        shape = (batch, kv_heads, groups, query_length, 1)
        return _SoftmaxState(
            maximum=mx.full(shape, -mx.inf, dtype=mx.float32),
            denominator=mx.zeros(shape, dtype=mx.float32),
            numerator=mx.zeros(
                (batch, kv_heads, groups, query_length, head_dim),
                dtype=mx.float32,
            ),
        )
    grouped_keys = keys.astype(mx.float32)[:, :, None, :, :]
    grouped_values = values.astype(mx.float32)[:, :, None, :, :]
    scores = mx.matmul(grouped_queries, mx.swapaxes(grouped_keys, -1, -2))
    scores = scores * float(scale)
    if mask is not None:
        scores = scores + mask
    maximum = mx.max(scores, axis=-1, keepdims=True)
    finite = mx.isfinite(maximum)
    shifted = mx.where(finite, scores - maximum, -mx.inf)
    weights = mx.exp(shifted)
    denominator = mx.sum(weights, axis=-1, keepdims=True)
    numerator = mx.matmul(weights, grouped_values)
    return _SoftmaxState(maximum, denominator, numerator)


def _merge(left: _SoftmaxState, right: _SoftmaxState) -> _SoftmaxState:
    maximum = mx.maximum(left.maximum, right.maximum)
    maximum_finite = mx.isfinite(maximum)
    left_scale = mx.where(
        mx.logical_and(maximum_finite, mx.isfinite(left.maximum)),
        mx.exp(left.maximum - maximum),
        mx.zeros_like(maximum),
    )
    right_scale = mx.where(
        mx.logical_and(maximum_finite, mx.isfinite(right.maximum)),
        mx.exp(right.maximum - maximum),
        mx.zeros_like(maximum),
    )
    return _SoftmaxState(
        maximum=maximum,
        denominator=left.denominator * left_scale
        + right.denominator * right_scale,
        numerator=left.numerator * left_scale
        + right.numerator * right_scale,
    )


def _finish(state: _SoftmaxState, query_heads: int) -> mx.array:
    safe_denominator = mx.maximum(state.denominator, mx.array(1.0e-30))
    output = mx.where(
        state.denominator > 0,
        state.numerator / safe_denominator,
        mx.zeros_like(state.numerator),
    )
    batch, _, _, query_length, head_dim = map(int, output.shape)
    return output.reshape(batch, query_heads, query_length, head_dim)


def _receipt(
    *,
    mechanism: str,
    queries: mx.array,
    base_keys: mx.array,
    suffix_keys: Sequence[mx.array],
    kv_heads: int,
    shared_base_calls: int,
    private_segment_calls: int,
    materialized_row_calls: int,
    base_kv_row_read_proxy: int,
) -> SegmentedQSAReceipt:
    batch, query_heads, query_length, head_dim = map(int, queries.shape)
    base_length = int(base_keys.shape[2])
    suffix_lengths = tuple(int(value.shape[2]) for value in suffix_keys)
    return SegmentedQSAReceipt(
        mechanism=mechanism,
        batch_size=batch,
        query_length=query_length,
        base_length=base_length,
        suffix_lengths=suffix_lengths,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        shared_base_calls=shared_base_calls,
        private_segment_calls=private_segment_calls,
        materialized_row_calls=materialized_row_calls,
        logical_score_elements=query_heads
        * query_length
        * (batch * base_length + sum(suffix_lengths)),
        base_kv_row_read_proxy=base_kv_row_read_proxy,
    )


def shared_prefix_segmented_attention(
    queries: mx.array,
    base_keys: mx.array,
    base_values: mx.array,
    suffix_keys: Sequence[mx.array],
    suffix_values: Sequence[mx.array],
    *,
    scale: float,
    base_mask: mx.array | None = None,
    suffix_masks: Sequence[mx.array | None] | None = None,
) -> tuple[mx.array, SegmentedQSAReceipt]:
    """Batch the immutable-base reduction, then merge row-private reductions.

    The shared K/V arrays retain batch one and are passed to exactly one
    batched reduction.  No ``[base, suffix]`` array is formed by this path.
    """
    batch, query_heads, query_length, _, kv_heads = _validate(
        queries, base_keys, base_values, suffix_keys, suffix_values
    )
    if suffix_masks is None:
        suffix_masks = [None] * batch
    if len(suffix_masks) != batch:
        raise ValueError("private masks must cover every query row")
    grouped_queries = _group_queries(queries, kv_heads)
    grouped_base_mask = _group_mask(
        base_mask,
        batch=batch,
        query_heads=query_heads,
        kv_heads=kv_heads,
        query_length=query_length,
        key_length=int(base_keys.shape[2]),
    )
    base_state = _segment_state(
        grouped_queries,
        base_keys,
        base_values,
        scale=scale,
        mask=grouped_base_mask,
    )
    rows = []
    private_calls = 0
    for row, (keys, values, mask) in enumerate(
        zip(suffix_keys, suffix_values, suffix_masks)
    ):
        if int(keys.shape[2]) == 0:
            state = _SoftmaxState(
                base_state.maximum[row : row + 1],
                base_state.denominator[row : row + 1],
                base_state.numerator[row : row + 1],
            )
        else:
            private_calls += 1
            private_mask = _group_mask(
                mask,
                batch=1,
                query_heads=query_heads,
                kv_heads=kv_heads,
                query_length=query_length,
                key_length=int(keys.shape[2]),
            )
            private_state = _segment_state(
                grouped_queries[row : row + 1],
                keys,
                values,
                scale=scale,
                mask=private_mask,
            )
            state = _merge(
                _SoftmaxState(
                    base_state.maximum[row : row + 1],
                    base_state.denominator[row : row + 1],
                    base_state.numerator[row : row + 1],
                ),
                private_state,
            )
        rows.append(_finish(state, query_heads))
    output = mx.concatenate(rows, axis=0)
    return output, _receipt(
        mechanism="shared_prefix_online_softmax_prototype",
        queries=queries,
        base_keys=base_keys,
        suffix_keys=suffix_keys,
        kv_heads=kv_heads,
        shared_base_calls=1,
        private_segment_calls=private_calls,
        materialized_row_calls=0,
        base_kv_row_read_proxy=1,
    )


def materialized_row_attention_reference(
    queries: mx.array,
    base_keys: mx.array,
    base_values: mx.array,
    suffix_keys: Sequence[mx.array],
    suffix_values: Sequence[mx.array],
    *,
    scale: float,
    base_mask: mx.array | None = None,
    suffix_masks: Sequence[mx.array | None] | None = None,
) -> tuple[mx.array, SegmentedQSAReceipt]:
    """Independent per-row oracle that explicitly forms ``[base, suffix]``."""
    batch, query_heads, query_length, _, kv_heads = _validate(
        queries, base_keys, base_values, suffix_keys, suffix_values
    )
    if suffix_masks is None:
        suffix_masks = [None] * batch
    if len(suffix_masks) != batch:
        raise ValueError("private masks must cover every query row")
    rows = []
    for row, (keys, values, suffix_mask) in enumerate(
        zip(suffix_keys, suffix_values, suffix_masks)
    ):
        full_keys = mx.concatenate((base_keys, keys), axis=2)
        full_values = mx.concatenate((base_values, values), axis=2)
        row_base_mask = (
            None
            if base_mask is None
            else base_mask[0:1]
            if int(base_mask.shape[0]) == 1
            else base_mask[row : row + 1]
        )
        if row_base_mask is None and suffix_mask is None:
            full_mask = None
        else:
            # Put both sides in additive form before joining them. Concatenating
            # a boolean side with a float side (the zero fill below, or an
            # additive caller mask) promotes True/False to 1.0/0.0, which the
            # additive path then adds to the scores, so masked tokens are kept.
            row_base_mask = _additive_mask(row_base_mask)
            suffix_mask = _additive_mask(suffix_mask)
            if row_base_mask is None:
                row_base_mask = mx.zeros(
                    (1, 1, query_length, int(base_keys.shape[2])),
                    dtype=mx.float32,
                )
            if suffix_mask is None:
                suffix_mask = mx.zeros(
                    (1, 1, query_length, int(keys.shape[2])), dtype=mx.float32
                )
            full_mask = mx.concatenate((row_base_mask, suffix_mask), axis=-1)
        grouped_mask = _group_mask(
            full_mask,
            batch=1,
            query_heads=query_heads,
            kv_heads=kv_heads,
            query_length=query_length,
            key_length=int(full_keys.shape[2]),
        )
        state = _segment_state(
            _group_queries(queries[row : row + 1], kv_heads),
            full_keys,
            full_values,
            scale=scale,
            mask=grouped_mask,
        )
        rows.append(_finish(state, query_heads))
    output = mx.concatenate(rows, axis=0)
    return output, _receipt(
        mechanism="materialized_per_row_reference",
        queries=queries,
        base_keys=base_keys,
        suffix_keys=suffix_keys,
        kv_heads=kv_heads,
        shared_base_calls=0,
        private_segment_calls=0,
        materialized_row_calls=batch,
        base_kv_row_read_proxy=batch,
    )

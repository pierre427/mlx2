# SPDX-License-Identifier: Apache-2.0
"""Isolated Metal probe for exact attention over shared and private KV segments.

This module is deliberately not imported by serving.  It tests whether one
threadgroup can amortize immutable-prefix K/V loads across multiple independent
query rows while preserving a row-private suffix and exact online softmax.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache

import mlx.core as mx


_SUPPORTED_BATCHES = (1, 2, 4, 8)
_SUPPORTED_DTYPES = (mx.float16, mx.bfloat16, mx.float32)


@dataclass(frozen=True)
class SegmentedQSAMetalReceipt:
    mechanism: str
    batch_size: int
    query_length: int
    base_length: int
    suffix_width: int
    query_heads: int
    kv_heads: int
    head_dim: int
    splits: int
    shared_base_row_read_proxy: int
    private_suffix_rows: int

    def to_dict(self) -> dict:
        return asdict(self)


_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""


_PARTITION_SOURCE = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint query = threadgroup_position_in_grid.y;
    const uint unit = threadgroup_position_in_grid.z;
    const uint split = unit % SPLITS;
    const uint qh = unit / SPLITS;
    const uint kvh = qh / GQA;
    const uint elements = D / 32;
    const uint base_length = uint(dims[0]);
    const uint suffix_width = uint(dims[1]);
    const uint query_length = uint(dims[2]);
    const uint base_storage_length = uint(dims[3]);

    float q_values[BATCH][D / 32];
    float maximum[BATCH];
    float denominator[BATCH];
    float numerator[BATCH][D / 32];
    for (uint b = 0; b < BATCH; ++b) {
        maximum[b] = -3.402823466e+38F;
        denominator[b] = 0.0f;
        for (uint part = 0; part < elements; ++part) {
            const uint d = lane * elements + part;
            const size_t index =
                (((size_t)b * QUERY_HEADS + qh) * query_length + query) * D + d;
            q_values[b][part] = float(scale[0]) * float(q[index]);
            numerator[b][part] = 0.0f;
        }
    }

    const uint base_begin = (base_length * split) / SPLITS;
    const uint base_end = (base_length * (split + 1)) / SPLITS;
    const size_t base_head = (size_t)kvh * base_storage_length * D;
    for (uint token = base_begin; token < base_end; ++token) {
        const uint physical_token = HAS_INDEX ? uint(base_indices[token]) : token;
        float scores[BATCH];
        for (uint b = 0; b < BATCH; ++b) scores[b] = 0.0f;
        for (uint part = 0; part < elements; ++part) {
            const uint d = lane * elements + part;
            const float key_value = float(base_k[base_head + (size_t)physical_token * D + d]);
            for (uint b = 0; b < BATCH; ++b)
                scores[b] += q_values[b][part] * key_value;
        }
        float factors[BATCH];
        float probabilities[BATCH];
        for (uint b = 0; b < BATCH; ++b) {
            scores[b] = simd_sum(scores[b]);
            const float next_maximum = metal::max(maximum[b], scores[b]);
            factors[b] = fast::exp(maximum[b] - next_maximum);
            probabilities[b] = fast::exp(scores[b] - next_maximum);
            maximum[b] = next_maximum;
            denominator[b] = denominator[b] * factors[b] + probabilities[b];
        }
        for (uint part = 0; part < elements; ++part) {
            const uint d = lane * elements + part;
            const float value = float(base_v[base_head + (size_t)physical_token * D + d]);
            for (uint b = 0; b < BATCH; ++b)
                numerator[b][part] = numerator[b][part] * factors[b]
                    + probabilities[b] * value;
        }
    }

    for (uint b = 0; b < BATCH; ++b) {
        const uint length = metal::min(uint(suffix_lengths[b]), suffix_width);
        const uint suffix_begin = (length * split) / SPLITS;
        const uint suffix_end = (length * (split + 1)) / SPLITS;
        const size_t suffix_head =
            ((size_t)b * KV_HEADS + kvh) * suffix_width * D;
        for (uint token = suffix_begin; token < suffix_end; ++token) {
            float score = 0.0f;
            for (uint part = 0; part < elements; ++part) {
                const uint d = lane * elements + part;
                score += q_values[b][part] *
                    float(suffix_k[suffix_head + (size_t)token * D + d]);
            }
            score = simd_sum(score);
            const float next_maximum = metal::max(maximum[b], score);
            const float factor = fast::exp(maximum[b] - next_maximum);
            const float probability = fast::exp(score - next_maximum);
            maximum[b] = next_maximum;
            denominator[b] = denominator[b] * factor + probability;
            for (uint part = 0; part < elements; ++part) {
                const uint d = lane * elements + part;
                const float value =
                    float(suffix_v[suffix_head + (size_t)token * D + d]);
                numerator[b][part] = numerator[b][part] * factor
                    + probability * value;
            }
        }
    }

    for (uint b = 0; b < BATCH; ++b) {
        const size_t state =
            (((size_t)b * QUERY_HEADS + qh) * query_length + query) * SPLITS + split;
        if (lane == 0) {
            part_m[state] = maximum[b];
            part_l[state] = denominator[b];
        }
        for (uint part = 0; part < elements; ++part) {
            const uint d = lane * elements + part;
            part_o[state * D + d] = numerator[b][part];
        }
    }
    if (unit == 0 && query == 0 && lane == 0) engaged[0] = 1;
"""


_COMBINE_SOURCE = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint query = threadgroup_position_in_grid.y;
    const uint bh = threadgroup_position_in_grid.z;
    const uint query_length = uint(dims[0]);
    const uint elements = D / 32;
    const size_t state = ((size_t)bh * query_length + query) * SPLITS;

    float maximum = -3.402823466e+38F;
    for (uint split = 0; split < SPLITS; ++split)
        maximum = metal::max(maximum, part_m[state + split]);
    float denominator = 0.0f;
    for (uint split = 0; split < SPLITS; ++split)
        denominator += fast::exp(part_m[state + split] - maximum)
            * part_l[state + split];

    for (uint part = 0; part < elements; ++part) {
        const uint d = lane * elements + part;
        float value = 0.0f;
        for (uint split = 0; split < SPLITS; ++split) {
            const float factor = fast::exp(part_m[state + split] - maximum);
            value += factor * part_o[(state + split) * D + d];
        }
        out[((size_t)bh * query_length + query) * D + d] =
            T(denominator > 0.0f ? value / denominator : 0.0f);
    }
"""


@lru_cache(maxsize=1)
def _partition_kernel():
    return mx.fast.metal_kernel(
        name="mlx2_segmented_shared_prefix_partition_v1",
        input_names=[
            "q", "base_k", "base_v", "base_indices", "suffix_k", "suffix_v",
            "suffix_lengths", "scale", "dims",
        ],
        output_names=["part_m", "part_l", "part_o", "engaged"],
        header=_HEADER,
        source=_PARTITION_SOURCE,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=1)
def _combine_kernel():
    return mx.fast.metal_kernel(
        name="mlx2_segmented_shared_prefix_combine_v1",
        input_names=["part_m", "part_l", "part_o", "dims"],
        output_names=["out"],
        header=_HEADER,
        source=_COMBINE_SOURCE,
        ensure_row_contiguous=True,
    )


def segmented_shared_prefix_attention_metal(
    queries: mx.array,
    base_keys: mx.array,
    base_values: mx.array,
    suffix_keys: mx.array,
    suffix_values: mx.array,
    suffix_lengths: mx.array,
    *,
    scale: float,
    splits: int = 32,
    base_indices: mx.array | None = None,
) -> tuple[mx.array, mx.array, SegmentedQSAMetalReceipt]:
    """Run the isolated shared-prefix Metal probe.

    Shapes are ``q=[B,Hq,Q,D]``, ``base=[1,Hkv,K,D]`` and
    ``suffix=[B,Hkv,S,D]``. Masks are intentionally unsupported in this first
    probe; callers must provide only visible suffix tokens in ``suffix_lengths``.
    """
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        raise RuntimeError("segmented QSA Metal probe requires the GPU device")
    arrays = (queries, base_keys, base_values, suffix_keys, suffix_values)
    if any(value.ndim != 4 for value in arrays):
        raise ValueError("segmented QSA Metal arrays must be rank four")
    if base_keys.shape != base_values.shape or suffix_keys.shape != suffix_values.shape:
        raise ValueError("segmented QSA Metal K/V geometries differ")
    batch, query_heads, query_length, head_dim = map(int, queries.shape)
    if batch not in _SUPPORTED_BATCHES:
        raise ValueError(f"segmented QSA Metal batch must be one of {_SUPPORTED_BATCHES}")
    if int(base_keys.shape[0]) != 1 or int(suffix_keys.shape[0]) != batch:
        raise ValueError("segmented QSA Metal requires one base and B suffix rows")
    kv_heads = int(base_keys.shape[1])
    if tuple(map(int, suffix_keys.shape[1::2])) != (kv_heads, head_dim):
        raise ValueError("segmented QSA Metal suffix layout differs from the base")
    if int(base_keys.shape[3]) != head_dim or query_heads % kv_heads:
        raise ValueError("segmented QSA Metal head geometry is unsupported")
    if head_dim % 32 or head_dim < 32:
        raise ValueError("segmented QSA Metal head dimension must be divisible by 32")
    if queries.dtype not in _SUPPORTED_DTYPES or any(
        value.dtype != queries.dtype for value in arrays[1:]
    ):
        raise ValueError("segmented QSA Metal requires one supported dtype")
    if suffix_lengths.ndim != 1 or int(suffix_lengths.shape[0]) != batch:
        raise ValueError("segmented QSA Metal suffix lengths must have shape [B]")
    if suffix_lengths.dtype not in (mx.int32, mx.uint32):
        suffix_lengths = suffix_lengths.astype(mx.uint32)
    splits = int(splits)
    if splits < 1 or splits > 128 or (splits & (splits - 1)):
        raise ValueError("segmented QSA Metal splits must be a power of two through 128")

    base_storage_length = int(base_keys.shape[2])
    if base_indices is None:
        base_length = base_storage_length
        base_indices = mx.zeros((1,), dtype=mx.uint32)
        has_index = 0
    else:
        if base_indices.ndim != 1 or int(base_indices.shape[0]) <= 0:
            raise ValueError("segmented QSA Metal base indices must be a nonempty vector")
        if base_indices.dtype not in (mx.int32, mx.uint32):
            base_indices = base_indices.astype(mx.uint32)
        base_length = int(base_indices.shape[0])
        has_index = 1
    suffix_width = int(suffix_keys.shape[2])
    templates = [
        ("T", queries.dtype), ("BATCH", batch), ("QUERY_HEADS", query_heads),
        ("KV_HEADS", kv_heads), ("GQA", query_heads // kv_heads),
        ("D", head_dim), ("SPLITS", splits), ("HAS_INDEX", has_index),
    ]
    part_m, part_l, part_o, engaged = _partition_kernel()(
        inputs=[
            mx.contiguous(queries), mx.contiguous(base_keys), mx.contiguous(base_values),
            mx.contiguous(base_indices),
            mx.contiguous(suffix_keys), mx.contiguous(suffix_values),
            mx.contiguous(suffix_lengths.astype(mx.uint32)),
            mx.array([scale], dtype=mx.float32),
            mx.array(
                [base_length, suffix_width, query_length, base_storage_length],
                dtype=mx.int32,
            ),
        ],
        template=templates,
        grid=(32, query_length, query_heads * splits),
        threadgroup=(32, 1, 1),
        output_shapes=[
            (batch, query_heads, query_length, splits),
            (batch, query_heads, query_length, splits),
            (batch, query_heads, query_length, splits, head_dim),
            (1,),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32, mx.uint32],
    )
    (output,) = _combine_kernel()(
        inputs=[part_m, part_l, part_o, mx.array([query_length], dtype=mx.int32)],
        template=[("T", queries.dtype), ("D", head_dim), ("SPLITS", splits)],
        grid=(32, query_length, batch * query_heads),
        threadgroup=(32, 1, 1),
        output_shapes=[(batch, query_heads, query_length, head_dim)],
        output_dtypes=[queries.dtype],
    )
    receipt = SegmentedQSAMetalReceipt(
        mechanism="metal_shared_prefix_partition_v1",
        batch_size=batch,
        query_length=query_length,
        base_length=base_length,
        suffix_width=suffix_width,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        splits=splits,
        shared_base_row_read_proxy=1,
        private_suffix_rows=batch,
    )
    return output, engaged, receipt

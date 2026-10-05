"""Default-off immutable paged attention read candidate.

This is a deliberately simple, one-SIMD-group-per-query-head read kernel. It
does not append KV, acquire a page lease, or participate in serving. The arena
owner must pin every handle through GPU completion and validate generations
again immediately before submission. No native writable pool is implied.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .paged_attention_plan import PAGE_SIZE, PagedAttentionPlan


@dataclass(frozen=True)
class PagedReadMetadata:
    """Immutable integer snapshot used by both the CPU and Metal read paths."""

    row_span: tuple[int, ...]
    row_begin: tuple[int, ...]
    query_start: tuple[int, ...]
    retained_start: tuple[int, ...]
    first_block: tuple[int, ...]
    table_begin: tuple[int, ...]
    window: tuple[int, ...]
    page_ids: tuple[int, ...]


def build_paged_read_metadata(
    plan: PagedAttentionPlan, *, omit_long_row_span: bool = False,
) -> PagedReadMetadata:
    """Snapshot only validated host coordinates; physical page order is opaque."""
    if type(plan) is not PagedAttentionPlan:
        raise TypeError("a validated PagedAttentionPlan is required")
    if type(omit_long_row_span) is not bool or (omit_long_row_span and
            plan.profile not in ("prefill_long_nax_v1", "prefill_long_n20_v1")):
        raise ValueError("row-span omission requires an exact long native profile")
    # Native long NAX resolves spans from grid.z and ordered row_begin;
    # the generic immutable Metal reader still needs the full row map.
    return PagedReadMetadata(
        row_span=() if omit_long_row_span else tuple(
            i for i, span in enumerate(plan.spans) for _ in range(span.row_count)),
        row_begin=tuple(span.row_begin for span in plan.spans),
        query_start=tuple(span.query_start for span in plan.spans),
        retained_start=tuple(span.retained_start for span in plan.spans),
        first_block=tuple(span.first_block for span in plan.spans),
        table_begin=tuple(span.table_begin for span in plan.spans),
        window=tuple(span.window or 0 for span in plan.spans),
        page_ids=tuple(handle.page_id for handle in plan.page_table),
    )


def paged_attention_cpu_reference(
    plan: PagedAttentionPlan,
    q: np.ndarray,
    k_arena: np.ndarray,
    v_arena: np.ndarray,
    *,
    scale: float | None = None,
) -> np.ndarray:
    """Readable CPU reference for immutable pages, independent of Metal math.

    Float64 scores and softmax make this an accuracy oracle, not an exact-bit
    predictor for the fp32 Metal reduction. The caller owns the arena bytes.
    """
    expected_q = (plan.total_rows, plan.query_heads, plan.head_dim)
    expected_kv = (plan.pool_capacity, plan.kv_heads, PAGE_SIZE, plan.head_dim)
    if q.shape != expected_q or k_arena.shape != expected_kv or v_arena.shape != expected_kv:
        raise ValueError("query or arena shape disagrees with the paged plan")
    if str(q.dtype) != plan.dtype or str(k_arena.dtype) != plan.dtype or str(v_arena.dtype) != plan.dtype:
        # NumPy has no native bf16 in many installs; MLX is used for bf16 native gates.
        raise ValueError("query and arena dtype must match the paged profile")
    factor = float(scale) if scale is not None else plan.head_dim ** -0.5
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("attention scale must be finite and positive")
    out = np.empty(expected_q, dtype=q.dtype)
    group = plan.query_heads // plan.kv_heads
    for sequence_index, span in enumerate(plan.spans):
        for local_row in range(span.row_count):
            row = span.row_begin + local_row
            lower, upper = span.visible_bounds(local_row)
            for query_head in range(plan.query_heads):
                kv_head = query_head // group
                keys = np.empty((upper - lower, plan.head_dim), dtype=np.float64)
                values = np.empty_like(keys)
                for i, token in enumerate(range(lower, upper)):
                    handle, position = plan.page_for_token(sequence_index, token)
                    keys[i] = k_arena[handle.page_id, kv_head, position]
                    values[i] = v_arena[handle.page_id, kv_head, position]
                scores = keys @ q[row, query_head].astype(np.float64) * factor
                probabilities = np.exp(scores - scores.max())
                probabilities /= probabilities.sum()
                out[row, query_head] = probabilities @ values
    return out


_HEADER = "#include <metal_stdlib>\n#include <metal_simdgroup>\nusing namespace metal;\n"

# The same logical traversal is used for every row regardless of other lanes.
# Chunk boundaries are absolute, and physical IDs affect only addresses.
_SOURCE = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint row = threadgroup_position_in_grid.y;
    const uint qh = threadgroup_position_in_grid.z;
    const uint span = row_span[row];
    const uint local = row - row_begin[span];
    const uint upper = query_start[span] + local + 1;
    const uint lower = window[span] == 0 ? retained_start[span] :
        metal::max(retained_start[span], upper - metal::min(upper, window[span]));
    const uint kvh = qh / (NQH / NKVH);
    float query[D / 32];
    float numerator[D / 32] = {0};
    for (uint part = 0; part < D / 32; ++part)
        query[part] = float(q[((size_t)row * NQH + qh) * D + lane * (D / 32) + part]);
    float maximum = -INFINITY;
    float denominator = 0.0f;
    const uint first_chunk = lower / 512;
    const uint last_chunk = (upper - 1) / 512;
    for (uint chunk = first_chunk; chunk <= last_chunk; ++chunk) {
        const uint begin = metal::max(lower, chunk * 512);
        const uint end = metal::min(upper, (chunk + 1) * 512);
        for (uint token = begin; token < end; ++token) {
            const uint block = token / 64;
            const uint page = page_ids[table_begin[span] + block - first_block[span]];
            const size_t base = (((size_t)page * NKVH + kvh) * 64 + (token & 63)) * D;
            float score = 0.0f;
            for (uint part = 0; part < D / 32; ++part) {
                const uint d = lane * (D / 32) + part;
                score += query[part] * float(k_arena[base + d]);
            }
            score = simd_sum(score) * scale[0];
            const float next_max = metal::max(maximum, score);
            const float previous = maximum == -INFINITY ? 0.0f : metal::exp(maximum - next_max);
            const float current = metal::exp(score - next_max);
            maximum = next_max;
            denominator = denominator * previous + current;
            for (uint part = 0; part < D / 32; ++part) {
                const uint d = lane * (D / 32) + part;
                numerator[part] = numerator[part] * previous +
                    current * float(v_arena[base + d]);
            }
        }
    }
    for (uint part = 0; part < D / 32; ++part) {
        const uint d = lane * (D / 32) + part;
        out[((size_t)row * NQH + qh) * D + d] = numerator[part] / denominator;
    }
"""


@lru_cache(maxsize=1)
def _read_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="mlx2_paged_immutable_read_vector_candidate_v1",
        input_names=["q", "k_arena", "v_arena", "row_span", "row_begin", "query_start",
                     "retained_start", "first_block", "table_begin", "window", "page_ids", "scale"],
        output_names=["out"],
        header=_HEADER,
        source=_SOURCE,
    )


def paged_attention_metal_candidate(
    plan: PagedAttentionPlan,
    q,
    k_arena,
    v_arena,
    *,
    live_generations: Mapping[int, int],
    owner_pins_through_completion: bool = False,
    permit_candidate: bool = False,
    scale: float | None = None,
):
    """Explicit experimental dispatch. Never selected by serving defaults.

    A native owner must supply a real GPU completion lease. The boolean is an
    assertion by that owner, not a lease implementation. This API cannot prove
    write ordering or lifetime and must not be connected to serving yet.
    """
    if not (permit_candidate and owner_pins_through_completion):
        raise RuntimeError("paged Metal read requires explicit candidate and owner completion lease")
    if any(live_generations.get(h.page_id) != h.generation for h in plan.page_table):
        raise ValueError("page generation changed before paged read submission")
    import mlx.core as mx

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise RuntimeError("paged Metal read requires an available GPU")
    expected_q = (plan.total_rows, plan.query_heads, plan.head_dim)
    expected_kv = (plan.pool_capacity, plan.kv_heads, PAGE_SIZE, plan.head_dim)
    dtype = mx.float16 if plan.dtype == "float16" else mx.bfloat16
    if (tuple(q.shape) != expected_q or tuple(k_arena.shape) != expected_kv or
            tuple(v_arena.shape) != expected_kv or any(x.dtype != dtype for x in (q, k_arena, v_arena))):
        raise ValueError("Metal input geometry or dtype disagrees with the paged plan")
    if plan.total_rows == 0:
        return mx.empty(expected_q, dtype=dtype)
    factor = float(scale) if scale is not None else plan.head_dim ** -0.5
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("attention scale must be finite and positive")
    meta = build_paged_read_metadata(plan)
    ints = [mx.array(values, dtype=mx.uint32) for values in (
        meta.row_span, meta.row_begin, meta.query_start, meta.retained_start,
        meta.first_block, meta.table_begin, meta.window, meta.page_ids,
    )]
    return _read_kernel()(
        inputs=[mx.contiguous(q), mx.contiguous(k_arena), mx.contiguous(v_arena),
                *ints, mx.array([factor], dtype=mx.float32)],
        template=[("D", plan.head_dim), ("NQH", plan.query_heads), ("NKVH", plan.kv_heads)],
        grid=(32, plan.total_rows, plan.query_heads),
        threadgroup=(32, 1, 1),
        output_shapes=[expected_q],
        output_dtypes=[dtype],
    )[0]

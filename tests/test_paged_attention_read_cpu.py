"""Independent dense oracle for the default-off immutable page reader."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mlx2.runtime.paged_attention_metal import (
    build_paged_read_metadata,
    paged_attention_cpu_reference,
    paged_attention_metal_candidate,
)
from mlx2.runtime.paged_attention_plan import (
    PagedAttentionPlan,
    PageHandle,
    SequenceSpan,
)


def fixture(dim: int, sliding: bool, physical: tuple[int, ...] = (4, 1, 5, 2)):
    rng = np.random.default_rng(91822 + dim)
    mask = "sliding" if sliding else "causal"
    window = 3 if sliding else None
    spans = (
        SequenceSpan(0, 3, 63, 66, 61, 0, 0, 2, 11, mask, window),
        SequenceSpan(3, 1, 65, 66, 62, 0, 2, 2, 19, mask, window),
    )
    handles = tuple(PageHandle(page, i + 1) for i, page in enumerate(physical))
    plan = PagedAttentionPlan(
        spans=spans, page_table=handles, total_rows=4, query_heads=4,
        kv_heads=2, head_dim=dim, dtype="float16", pool_capacity=6,
        live_generations={h.page_id: h.generation for h in handles},
    )
    q = rng.normal(0, 0.3, (4, 4, dim)).astype(np.float16)
    logical_k = rng.normal(0, 0.3, (2, 66, 2, dim)).astype(np.float16)
    logical_v = rng.normal(0, 0.3, (2, 66, 2, dim)).astype(np.float16)
    # Poison every unowned slot and each partial-page region. Only the four
    # mapped pages receive live logical tokens.
    k = np.full((6, 2, 64, dim), np.nan, dtype=np.float16)
    v = np.full_like(k, np.nan)
    for sequence_index, span in enumerate(spans):
        for token in range(span.retained_start, span.kv_end):
            h, position = plan.page_for_token(sequence_index, token)
            k[h.page_id, :, position] = logical_k[sequence_index, token]
            v[h.page_id, :, position] = logical_v[sequence_index, token]
    return plan, q, k, v, logical_k, logical_v


def independent_dense_reference(plan, q, logical_k, logical_v):
    result = np.empty_like(q)
    for s, span in enumerate(plan.spans):
        for local in range(span.row_count):
            row = span.row_begin + local
            upper = span.query_start + local + 1
            lower = span.retained_start
            if span.mask_kind == "sliding":
                lower = max(lower, upper - span.window)
            for head in range(plan.query_heads):
                kv_head = head // (plan.query_heads // plan.kv_heads)
                keys = logical_k[s, lower:upper, kv_head].astype(np.float64)
                values = logical_v[s, lower:upper, kv_head].astype(np.float64)
                scores = keys @ q[row, head].astype(np.float64) / np.sqrt(plan.head_dim)
                probs = np.exp(scores - np.max(scores))
                result[row, head] = (probs / np.sum(probs)) @ values
    return result


@pytest.mark.parametrize("dim", [128, 256])
@pytest.mark.parametrize("sliding", [False, True])
def test_page_boundary_causal_sliding_gqa_against_dense_oracle(dim, sliding):
    plan, q, k, v, logical_k, logical_v = fixture(dim, sliding)
    metadata = build_paged_read_metadata(plan)
    assert metadata.row_span == (0, 0, 0, 1)
    assert metadata.page_ids == (4, 1, 5, 2)
    expected = independent_dense_reference(plan, q, logical_k, logical_v)
    actual = paged_attention_cpu_reference(plan, q, k, v)
    np.testing.assert_array_equal(actual, expected)
    assert np.isfinite(actual).all()


@pytest.mark.parametrize("dim", [128, 256])
def test_physical_shuffle_and_unrelated_lane_are_bit_invariant(dim):
    plan, q, k, v, _, _ = fixture(dim, False)
    baseline = paged_attention_cpu_reference(plan, q, k, v)
    # A different physical page assignment preserves every logical KV byte.
    shuffled = (2, 5, 1, 4)
    new_handles = tuple(PageHandle(page, h.generation) for page, h in zip(shuffled, plan.page_table))
    new_k, new_v = np.full_like(k, np.nan), np.full_like(v, np.nan)
    for old, new in zip(plan.page_table, new_handles):
        new_k[new.page_id] = k[old.page_id]
        new_v[new.page_id] = v[old.page_id]
    moved = replace(plan, page_table=new_handles,
                    live_generations={h.page_id: h.generation for h in new_handles})
    np.testing.assert_array_equal(paged_attention_cpu_reference(moved, q, new_k, new_v), baseline)
    # Companion activity changes no query, page table, or bytes of lane zero.
    changed_q, changed_k, changed_v = q.copy(), k.copy(), v.copy()
    changed_q[3] *= -2
    for h in plan.page_table[2:]:
        changed_k[h.page_id] *= -3
        changed_v[h.page_id] *= 4
    np.testing.assert_array_equal(
        paged_attention_cpu_reference(plan, changed_q, changed_k, changed_v)[:3], baseline[:3]
    )


def test_candidate_refuses_by_default_before_metal_construction():
    plan, q, k, v, _, _ = fixture(128, False)
    with pytest.raises(RuntimeError, match="explicit candidate"):
        paged_attention_metal_candidate(plan, q, k, v, live_generations=plan.live_generations)
    with pytest.raises(RuntimeError, match="explicit candidate"):
        paged_attention_metal_candidate(
            plan, q, k, v, live_generations=plan.live_generations, permit_candidate=True
        )
    with pytest.raises(ValueError, match="generation changed"):
        paged_attention_metal_candidate(
            plan, q, k, v, live_generations={}, permit_candidate=True,
            owner_pins_through_completion=True,
        )


def test_native_query_addressing_uses_evaluated_strides_for_head_major_q():
    """The pinned MLX projection may evaluate to head-major query storage."""
    import mlx.core as mx

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        rows, heads, dim = 64, 16, 128
        physical = (np.arange(rows * heads * dim, dtype=np.int32) % 8192).astype(
            np.float16).reshape(heads, rows, dim)
        query = mx.array(physical).transpose(1, 0, 2)
        row_major = mx.contiguous(query)
        mx.eval(query, row_major)
        assert np.asarray(query).strides == (dim * 2, rows * dim * 2, 2)
        assert np.asarray(row_major).strides == (heads * dim * 2, dim * 2, 2)
        for candidate, storage, strides in (
            (query, physical.reshape(-1), (dim, rows * dim, 1)),
            (row_major, np.asarray(row_major).reshape(-1),
             (heads * dim, dim, 1)),
        ):
            values = np.asarray(candidate)
            for row, head, channel in ((0, 0, 0), (1, 2, 3),
                                       (32, 15, 127), (63, 1, 64)):
                element = (row * strides[0] + head * strides[1] +
                           channel * strides[2])
                assert storage[element] == values[row, head, channel]
        # Keep the native contract bound to evaluated strides. A flat
        # row-major query index caused the prior full-prefill divergence.
        source = (Path(__file__).resolve().parents[1] /
                  "native/paged_kv/arena.cpp").read_text()
        assert "const auto& strides = query.strides();" in source
        assert "(size_t)row * query_row_stride" in source
        assert "encoder.set_bytes(query_head_stride, 18);" in source
    finally:
        mx.set_default_device(previous)

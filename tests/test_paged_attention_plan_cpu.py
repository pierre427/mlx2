"""CPU-only bounds for the default-off paged attention descriptor."""

from dataclasses import replace

import pytest

from mlx2.runtime.paged_attention_plan import (
    PAGE_SIZE,
    U32_MAX,
    PagedAttentionPlan,
    PageHandle,
    SequenceSpan,
)


def plan(*, spans=None, pages=None, generations=None, **overrides):
    spans = spans if spans is not None else (
        SequenceSpan(0, 2, 63, 65, 61, 0, 0, 2, 1),
    )
    pages = pages if pages is not None else (PageHandle(2, 1), PageHandle(3, 4))
    args = {
        "spans": spans,
        "page_table": pages,
        "total_rows": 2,
        "query_heads": 8,
        "kv_heads": 2,
        "head_dim": 128,
        "dtype": "float16",
        "pool_capacity": 4,
        "live_generations": generations if generations is not None else {2: 1, 3: 4},
    }
    args.update(overrides)
    return PagedAttentionPlan(**args)


def test_partial_pages_and_offset_causality():
    p = plan()
    assert p.spans[0].visible_bounds(0) == (61, 64)
    assert p.spans[0].visible_bounds(1) == (61, 65)
    assert p.page_for_token(0, 61) == (PageHandle(2, 1), 61)
    assert p.page_for_token(0, 64) == (PageHandle(3, 4), 0)
    for token in (60, 65):
        with pytest.raises(ValueError, match="retained"):
            p.page_for_token(0, token)


def test_sliding_window_and_shared_page():
    spans = (
        SequenceSpan(0, 2, 63, 65, 61, 0, 0, 2, 1, "sliding", 2),
        SequenceSpan(2, 1, 63, 64, 62, 0, 2, 1, 5),
    )
    p = plan(spans=spans, pages=(PageHandle(2, 1), PageHandle(3, 4), PageHandle(2, 1)), total_rows=3)
    assert p.spans[0].visible_bounds(0) == (62, 64)
    assert p.spans[0].visible_bounds(1) == (63, 65)
    assert p.page_for_token(1, 63) == (PageHandle(2, 1), 63)


@pytest.mark.parametrize("change, message", [
    ({"row_begin": 1}, "row offsets"),
    ({"row_count": 0}, "needs rows"),
    ({"row_count": 3}, "row interval"),
    ({"query_start": 60}, "retained or causal"),
    ({"query_start": 62}, "new KV suffix"),
    ({"kv_end": U32_MAX + 1}, "integer"),
    ({"retained_start": 64}, "retained or causal"),
    ({"first_block": 1}, "first block"),
    ({"table_begin": 1}, "table offsets"),
    ({"table_count": 1}, "exactly cover"),
    ({"owner_generation": 0}, "needs rows"),
    ({"mask_kind": "sink"}, "unsupported mask"),
    ({"mask_kind": "sliding", "window": 0}, "positive"),
    ({"window": 2}, "cannot specify"),
])
def test_reject_bad_span(change, message):
    with pytest.raises(ValueError, match=message):
        plan(spans=(replace(plan().spans[0], **change),))


@pytest.mark.parametrize("change, message", [
    ({"query_heads": 3}, "divisible"),
    ({"kv_heads": 0}, "positive"),
    ({"head_dim": 96}, "unsupported dtype"),
    ({"dtype": "float32"}, "unsupported dtype"),
    ({"profile": "matrix"}, "unsupported paged"),
    ({"page_size": 32}, "unsupported paged"),
    ({"pool_capacity": 3}, "outside the pool"),
    ({"total_rows": 1}, "exceeds packed"),
    ({"max_work_items": 1}, "capacity exceeded"),
    ({"max_scratch_bytes": 1}, "capacity exceeded"),
])
def test_reject_unsupported_or_unbounded_plan(change, message):
    with pytest.raises(ValueError, match=message):
        plan(**change)


def test_stale_handle_and_immutable_generation_snapshot():
    with pytest.raises(ValueError, match="stale"):
        plan(generations={2: 1, 3: 5})
    generations = {2: 1, 3: 4}
    p = plan(generations=generations)
    generations[2] = 7
    assert p.live_generations[2] == 1
    with pytest.raises(TypeError):
        p.live_generations[2] = 7


def test_no_unused_table_or_rows_and_zero_work():
    assert plan(spans=(), pages=(), total_rows=0).spans == ()
    with pytest.raises(ValueError, match="zero-work"):
        plan(spans=(), total_rows=0)
    with pytest.raises(ValueError, match="unused"):
        plan(pages=(PageHandle(2, 1), PageHandle(3, 4), PageHandle(2, 1)))


def test_uint64_address_and_uint32_row_overflow():
    with pytest.raises(ValueError, match="uint64"):
        plan(pool_capacity=U32_MAX, kv_heads=U32_MAX, query_heads=U32_MAX)
    with pytest.raises(ValueError, match="row interval"):
        plan(
            spans=(
                SequenceSpan(0, 1, 63, 64, 63, 0, 0, 1, 1),
                SequenceSpan(1, U32_MAX, 0, 1, 0, 0, 1, 1, 1),
            ),
            pages=(PageHandle(2, 1), PageHandle(2, 1)),
            total_rows=U32_MAX,
        )


def test_page_size_contract_is_fixed_at_64():
    assert PAGE_SIZE == 64


def test_high_absolute_offset_maps_one_partial_final_page():
    span = SequenceSpan(
        row_begin=0,
        row_count=1,
        query_start=U32_MAX - 1,
        kv_end=U32_MAX,
        retained_start=U32_MAX - 1,
        first_block=(U32_MAX - 1) // PAGE_SIZE,
        table_begin=0,
        table_count=1,
        owner_generation=9,
    )
    p = plan(spans=(span,), pages=(PageHandle(2, 1),), total_rows=1)
    assert p.page_for_token(0, U32_MAX - 1) == (PageHandle(2, 1), 62)
    with pytest.raises(ValueError, match="retained"):
        p.page_for_token(0, U32_MAX)


def test_bool_is_not_accepted_as_an_integer_field():
    with pytest.raises(ValueError, match="integer"):
        plan(spans=(replace(plan().spans[0], row_count=True),))

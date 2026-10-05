"""CPU-only preflight and receipt contract; no Metal or serving route."""

import pytest

from mlx2.contracts import PagedCandidateReceipt
from mlx2.runtime.batch_admission import decide_private_paged_use
from mlx2.runtime.memory_policy import estimate_paged_memory
from mlx2.runtime.paged_attention_plan import (
    PAGE_SIZE, PageHandle, PagedAttentionPlan, SequenceSpan,
)
from mlx2.runtime.paged_kv_cache import PagedKVPrivateCache
from mlx2.runtime.paged_kv_pool import PagedKVPool
from mlx2.runtime.paged_kv_write import PagedKVWriteOwner

TOKEN_BYTES = 2 * 128 * 2
PAGE_BYTES = PAGE_SIZE * TOKEN_BYTES


def plan(*, tokens=65, rows=1, window=None, capacity=4):
    pages = (tokens + PAGE_SIZE - 1) // PAGE_SIZE
    handles = tuple(PageHandle(i, 1) for i in range(pages))
    return PagedAttentionPlan(
        spans=(SequenceSpan(0, rows, tokens - rows, tokens, 0, 0, 0,
                            pages, 1, "sliding" if window else "causal", window),),
        page_table=handles, total_rows=rows, query_heads=8,
        kv_heads=2, head_dim=128, dtype="float16", pool_capacity=capacity,
        live_generations={i: 1 for i in range(pages)},
    )


class FakeBackend:
    def __init__(self, capacity):
        self.plane_bytes = capacity * PAGE_BYTES
        self.events = []

    def write(self, key, value, offset, byte_count, epoch):
        return object()

    def poll_completions(self):
        result, self.events = self.events, []
        return result


def accepted_cache(tokens=65):
    pool = PagedKVPool(4)
    backend = FakeBackend(4)
    writer = PagedKVWriteOwner(pool, backend, page_bytes=PAGE_BYTES,
                               permit_candidate=True)
    cache = PagedKVPrivateCache(writer, kv_heads=2, head_dim=128,
                                dtype="float16", permit_candidate=True)
    payload = bytes(tokens * TOKEN_BYTES)
    tickets = cache.append(payload, payload)
    backend.events.extend((ticket.epoch, True) for ticket in tickets)
    assert cache.poll_completions()
    return cache


def test_arena_charged_once_with_per_use_table_scratch_and_staging():
    use = plan()
    cold = estimate_paged_memory(use, arena_allocated=False,
                                 cow_pages=1, staging_tokens=3)
    warm = estimate_paged_memory(use, arena_allocated=True,
                                 cow_pages=1, staging_tokens=3)
    assert cold.arena_bytes == 2 * 4 * PAGE_BYTES
    assert warm.arena_bytes == 0
    assert cold.total_bytes - warm.total_bytes == cold.arena_bytes
    assert warm.table_bytes == 4 * (1 + 6 + 3 * 2)
    assert warm.scratch_bytes == 4 * 8 * (128 + 2)
    assert warm.cow_staging_bytes == 2 * PAGE_BYTES
    assert warm.payload_staging_bytes == 2 * 3 * TOKEN_BYTES


def test_default_off_capacity_feature_and_headroom_refusals():
    use = plan()
    kwargs = dict(available_bytes=10**9, arena_allocated=False,
                  free_pages=2, new_pages=2)
    assert decide_private_paged_use(use, **kwargs).reason == "paged_candidate_disabled"
    assert decide_private_paged_use(
        use, permit_candidate=True, requested_features=("q8", "sinks"), **kwargs
    ).reason == "unsupported_paged_features:q8,sinks"
    assert decide_private_paged_use(
        use, permit_candidate=True, cow_pages=1, **kwargs
    ).reason == "insufficient_retired_pages"
    allowed = decide_private_paged_use(use, permit_candidate=True, **kwargs)
    assert allowed.accepted and allowed.arena_bytes == 2 * 4 * PAGE_BYTES
    assert decide_private_paged_use(
        use, permit_candidate=True, **{**kwargs, "available_bytes": allowed.estimated_bytes - 1}
    ).reason == "insufficient_memory_headroom"
    warm = decide_private_paged_use(
        use, permit_candidate=True, **{**kwargs, "arena_allocated": True,
                                       "available_bytes": allowed.temporary_bytes}
    )
    assert warm.accepted and warm.arena_bytes == 0
    with pytest.raises(ValueError, match="available_bytes"):
        decide_private_paged_use(use, available_bytes=True, arena_allocated=True,
                                 free_pages=2)


def test_receipt_counts_only_successfully_completed_use_once():
    cache = accepted_cache()
    receipt = PagedCandidateReceipt()
    use = cache.attention(row_count=1, query_heads=8)
    with pytest.raises(ValueError, match="successful terminal proof"):
        receipt.record_completed_attention(use, kernel_executed=True)
    use.mark_submitted()
    with pytest.raises(ValueError, match="successful terminal proof"):
        receipt.record_completed_attention(use, kernel_executed=True)
    use.complete_after_proof()
    with pytest.raises(ValueError, match="executed read evidence"):
        receipt.record_completed_attention(use, kernel_executed=False)
    receipt.record_completed_attention(use, kernel_executed=True)
    assert receipt.as_dict()["actual_paged_rows"] == 1
    assert receipt.as_dict()["actual_paged_pages"] == 2
    assert receipt.as_dict()["observed_used"]
    assert not receipt.as_dict()["selected"] and not receipt.as_dict()["qualified"]
    with pytest.raises(ValueError, match="already recorded"):
        receipt.record_completed_attention(use, kernel_executed=True)

    failed = cache.attention(row_count=1, query_heads=8)
    failed.mark_submitted()
    failed.complete_after_proof(succeeded=False)
    with pytest.raises(ValueError, match="successful terminal proof"):
        receipt.record_completed_attention(failed, kernel_executed=True)
    aborted = cache.attention(row_count=1, query_heads=8)
    aborted.abort_before_submit()
    with pytest.raises(ValueError, match="successful terminal proof"):
        receipt.record_completed_attention(aborted, kernel_executed=True)
    receipt.record_admission(decide_private_paged_use(
        plan(), available_bytes=0, arena_allocated=False, free_pages=0,
    ))
    assert receipt.as_dict()["refusals"] == ["paged_candidate_disabled"]
    receipt.record_refusal("insufficient_retired_pages")
    assert receipt.as_dict()["refusals"] == ["paged_candidate_disabled", "insufficient_retired_pages"]


def test_sliding_receipt_counts_touched_page_not_unused_retained_table():
    cache = accepted_cache(tokens=129)
    use = cache.attention(row_count=1, query_heads=8,
                          mask_kind="sliding", window=1)
    use.mark_submitted()
    use.complete_after_proof()
    receipt = PagedCandidateReceipt()
    receipt.record_completed_attention(use, kernel_executed=True)
    assert len(use.plan.page_table) == 3
    assert receipt.actual_paged_pages == 1

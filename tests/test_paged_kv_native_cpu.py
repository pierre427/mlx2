"""CPU checks for the host completion contract; no GPU completion claim."""

import pytest

from mlx2.runtime.paged_kv_native import CompletionLease, PagedKVCompletionLedger
from mlx2.runtime.paged_kv_pool import PagedKVPool


def test_cancelled_submitted_use_pins_after_owner_release():
    pool = PagedKVPool(1)
    ledger = PagedKVCompletionLedger(pool)
    handle = pool.reserve()[0]
    lease = ledger.prepare((handle,))
    ledger.submit(lease)
    pool.release((handle,), after_epoch=lease.epoch)
    ledger.cancel(lease)
    assert pool.references(handle) == 1
    assert pool.free_count == 0
    with pytest.raises(ValueError, match="completion proof"):
        ledger.abort_before_submit(lease)
    ledger.complete(lease)
    ledger.complete(lease)  # Duplicate callback is harmless.
    assert ledger.pending_count == 0
    assert pool.free_count == 1
    next_handle = pool.reserve()[0]
    assert next_handle.generation == handle.generation + 1


def test_out_of_order_completion_holds_retirement_watermark():
    pool = PagedKVPool(2)
    ledger = PagedKVCompletionLedger(pool)
    first, second = pool.reserve(2)
    a = ledger.prepare((first,))
    b = ledger.prepare((second,))
    ledger.submit(a)
    ledger.submit(b)
    pool.release((first,), after_epoch=a.epoch)
    pool.release((second,), after_epoch=b.epoch)
    ledger.complete(b)
    assert ledger.completed_epoch == 0
    assert pool.completed_epoch == 0
    assert pool.free_count == 0
    assert ledger.pending_count == 1
    ledger.complete(a)
    assert ledger.completed_epoch == 2
    assert pool.free_count == 2


def test_pre_submission_abort_and_token_identity():
    pool = PagedKVPool(1)
    ledger = PagedKVCompletionLedger(pool)
    handle = pool.reserve()[0]
    lease = ledger.prepare((handle,))
    with pytest.raises(ValueError, match="unsubmitted"):
        ledger.complete(lease)
    with pytest.raises(ValueError, match="unknown"):
        ledger.submit(CompletionLease(lease.epoch))
    pool.release((handle,), after_epoch=lease.epoch)
    ledger.abort_before_submit(lease)
    assert pool.free_count == 1
    with pytest.raises(ValueError, match="not prepared|unknown"):
        ledger.abort_before_submit(lease)
    with pytest.raises(ValueError, match="not prepared|unknown"):
        ledger.submit(lease)


def test_prepare_failure_is_atomic_and_rejects_stale_generation():
    pool = PagedKVPool(2)
    ledger = PagedKVCompletionLedger(pool)
    first, second = pool.reserve(2)
    pool.release((second,), after_epoch=0)
    before = pool.references(first)
    with pytest.raises(ValueError, match="stale|unreferenced"):
        ledger.prepare((first, second))
    assert pool.references(first) == before
    with pytest.raises(ValueError, match="unique"):
        ledger.prepare((first, first))
    assert ledger.pending_count == 0

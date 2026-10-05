"""CPU proof that native public layers retire only after reader and event drain."""

import mlx.core as mx
import pytest

from mlx2.runtime.paged_attention_pack import prepare_packed_token_read
from mlx2.runtime.paged_attention_native import complete_packed_read_after_event
from mlx2.runtime.paged_native_atomic_owner import NativeAtomicError
from mlx2.runtime.paged_request_transaction import STATE_PLANES, CandidateRequest
from test_paged_native_atomic_owner_cpu import (
    append_complete, prove_all, setup, stage_all,
)


def _publish_cow_successor(owner, backend):
    """Make the old 63-token page unique to a retired public generation."""
    first = owner.begin(CandidateRequest(1, "r1", 63, STATE_PLANES))
    append_complete(first, backend, 63)
    stage_all(first, 63)
    prove_all(first, 63)
    first.prepare(63).publish()
    assert owner.reap_retired() == 1  # Empty generation zero.
    old = owner.snapshot()
    old_handle = old.layer_tables[0][0]

    branch = owner.begin(CandidateRequest(2, "r1", 3, STATE_PLANES))
    layer = branch.layers[0]
    spans = layer.planned_spans(3)
    chunks = tuple(mx.array([1] * span.byte_count, dtype=mx.uint8)
                   for span in spans)
    copy, = layer.append(chunks, chunks, token_count=3)
    backend.events.append((copy.epoch, True))
    assert not layer.poll_completions()
    backend.events.extend((ticket.epoch, True) for ticket in layer._pending)
    assert layer.poll_completions()
    stage_all(branch, 3)
    prove_all(branch, 3)
    branch.prepare(1).publish()
    with owner.snapshot() as current:
        assert current.offset == 64
        assert current.layer_tables[0][0] != old_handle
    return old


def test_explicit_reader_lease_blocks_retirement_until_close():
    owner, backend, writer = setup(layers=1)
    old = _publish_cow_successor(owner, backend)
    before = writer.pool.free_count
    assert owner.reap_retired() == 0
    assert writer.pool.free_count == before
    assert old.layer_owners[0].offset == 63
    old.close()
    old.close()  # Release is idempotent.
    with pytest.raises(NativeAtomicError, match="closed"):
        _ = old.layer_tables
    with pytest.raises(NativeAtomicError, match="closed"):
        _ = old.layer_owners
    assert owner.reap_retired() == 1
    assert writer.pool.free_count == before + 1
    assert owner.reap_retired() == 0


def test_closed_reader_still_waits_for_terminal_native_read():
    owner, backend, writer = setup(layers=1)
    old = _publish_cow_successor(owner, backend)
    use = prepare_packed_token_read((old.layer_owners[0],), (1,), query_heads=4,
                                    permit_candidate=True)
    use.mark_submitted()
    before = writer.pool.free_count
    old.close()
    assert owner.reap_retired() == 0
    assert writer.pool.free_count == before
    assert complete_packed_read_after_event(use, (use.lease.epoch, True))
    assert owner.reap_retired() == 1
    assert writer.pool.free_count == before + 1


def test_poisoned_writer_refuses_retirement_even_without_readers():
    owner, backend, writer = setup(layers=1)
    old = _publish_cow_successor(owner, backend)
    old.close()
    writer.poisoned = True
    before = writer.pool.free_count
    assert owner.reap_retired() == 0
    assert writer.pool.free_count == before


def test_owner_close_retains_final_public_pages_until_reader_release():
    owner, backend, writer = setup(layers=1)
    branch = owner.begin(CandidateRequest(1, "r1", 3, STATE_PLANES))
    append_complete(branch, backend, 3)
    stage_all(branch, 3)
    prove_all(branch, 3)
    branch.prepare(3).publish()
    assert owner.reap_retired() == 1  # Empty initial public generation.
    reader = owner.snapshot()
    assert not owner.fully_retired
    before = writer.pool.free_count
    owner.close()
    assert not owner.fully_retired
    owner.close()
    with pytest.raises(NativeAtomicError, match="closed"):
        owner.snapshot()
    with pytest.raises(NativeAtomicError, match="closed"):
        owner.begin(CandidateRequest(2, "r1", 1, STATE_PLANES))
    assert owner.reap_retired() == 0
    assert writer.pool.free_count == before
    reader.close()
    assert not owner.fully_retired
    assert owner.reap_retired() == 1
    assert owner.fully_retired
    assert writer.pool.free_count == before + 1


def test_fully_retired_waits_for_native_epoch_even_after_close():
    owner, _, writer = setup(layers=1)
    handle = writer.pool.reserve()[0]
    lease = writer.ledger.prepare((handle,))
    writer.ledger.submit(lease)
    owner.close()
    assert owner.reap_retired() == 0
    assert not owner.fully_retired
    writer.ledger.complete(lease)
    writer.pool.release((handle,), after_epoch=lease.epoch)
    assert not owner.fully_retired
    assert owner.reap_retired() == 1
    assert owner.fully_retired


def test_repeated_snapshots_balance_reader_counts_exactly():
    owner, _, _ = setup(layers=1)
    views = [owner.snapshot() for _ in range(8)]
    key = id(owner._public)
    assert owner._readers[key] == 8
    for view in views[:4]:
        view.close()
        view.close()
    assert owner._readers[key] == 4
    owner.close()
    assert owner.reap_retired() == 0
    for view in views[4:]:
        view.close()
    assert key not in owner._readers
    assert owner.reap_retired() == 1

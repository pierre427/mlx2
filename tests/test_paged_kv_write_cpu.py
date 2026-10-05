"""Host-only write submission checks; the backend never constructs Metal work."""

import pytest

from mlx2.runtime.paged_kv_pool import PagedKVPool
from mlx2.runtime.paged_kv_write import PagedKVWriteOwner, WriteSubmissionError


class FakeBackend:
    def __init__(self, plane_bytes):
        self.plane_bytes = plane_bytes
        self.calls = []
        self.events = []
        self.raise_after_record = False
        self.copies = []
        self.closed = 0

    def write(self, key, value, offset, byte_count, epoch):
        self.calls.append((key, value, offset, byte_count, epoch))
        if self.raise_after_record:
            raise RuntimeError("ambiguous native enqueue")
        return object()

    def poll_completions(self):
        events, self.events = self.events, []
        return events

    def copy_page(self, source_offset, destination_offset, byte_count, epoch):
        self.copies.append((source_offset, destination_offset, byte_count, epoch))
        if self.raise_after_record:
            raise RuntimeError("ambiguous native copy enqueue")
        return object()

    def close_after_terminal(self):
        self.closed += 1


def test_default_off_and_geometry_refusal_before_mutation():
    pool = PagedKVPool(2)
    backend = FakeBackend(128)
    with pytest.raises(RuntimeError, match="explicit candidate"):
        PagedKVWriteOwner(pool, backend, page_bytes=64)
    with pytest.raises(ValueError, match="plane size"):
        PagedKVWriteOwner(pool, FakeBackend(64), page_bytes=64, permit_candidate=True)
    owner = PagedKVWriteOwner(pool, backend, page_bytes=64, permit_candidate=True)
    handle = pool.reserve()[0]
    with pytest.raises(ValueError, match="crosses"):
        owner.submit_write(handle, within_page_offset=63, byte_count=2, key_bytes=b"k", value_bytes=b"v")
    assert pool.references(handle) == 1
    assert owner.pending_epochs == ()
    assert backend.calls == []


def test_submission_cancel_terminal_completion_and_reuse():
    pool = PagedKVPool(1)
    backend = FakeBackend(64)
    owner = PagedKVWriteOwner(pool, backend, page_bytes=64, permit_candidate=True)
    handle = pool.reserve()[0]
    ticket = owner.submit_write(handle, within_page_offset=4, byte_count=8,
                                key_bytes=b"key", value_bytes=b"value")
    assert backend.calls == [(b"key", b"value", 4, 8, 1)]
    assert ticket.dependency is not None
    pool.release((handle,), after_epoch=ticket.epoch)
    owner.cancel(ticket)
    assert pool.free_count == 0
    backend.events.append((ticket.epoch, True))
    result = owner.poll_completions()
    assert result[0].ticket is ticket and result[0].succeeded
    assert pool.free_count == 1
    backend.events.append((ticket.epoch, True))
    assert owner.poll_completions() == ()
    assert pool.reserve()[0].generation == handle.generation + 1


def test_out_of_order_failure_poisons_and_retains_contiguous_watermark():
    pool = PagedKVPool(2)
    backend = FakeBackend(128)
    owner = PagedKVWriteOwner(pool, backend, page_bytes=64, permit_candidate=True)
    a, b = pool.reserve(2)
    first = owner.submit_write(a, within_page_offset=0, byte_count=64, key_bytes=1, value_bytes=2)
    second = owner.submit_write(b, within_page_offset=0, byte_count=64, key_bytes=3, value_bytes=4)
    pool.release((a, b), after_epoch=second.epoch)
    backend.events.append((second.epoch, False))
    assert owner.poll_completions()[0].succeeded is False
    assert owner.poisoned
    assert owner.ledger.completed_epoch == 0
    assert pool.free_count == 0
    with pytest.raises(RuntimeError, match="poisoned"):
        owner.submit_write(a, within_page_offset=0, byte_count=1, key_bytes=1, value_bytes=2)
    backend.events.append((first.epoch, True))
    owner.poll_completions()
    assert owner.ledger.completed_epoch == 2
    assert pool.free_count == 1
    assert pool.quarantined_count == 1
    assert b.page_id not in pool.live_generations()


def test_ambiguous_submission_keeps_pin_until_terminal_event():
    pool = PagedKVPool(1)
    backend = FakeBackend(64)
    backend.raise_after_record = True
    owner = PagedKVWriteOwner(pool, backend, page_bytes=64, permit_candidate=True)
    handle = pool.reserve()[0]
    with pytest.raises(WriteSubmissionError) as error:
        owner.submit_write(handle, within_page_offset=0, byte_count=1, key_bytes=1, value_bytes=2)
    epoch = error.value.epoch
    pool.release((handle,), after_epoch=epoch)
    assert pool.references(handle) == 1
    assert owner.pending_epochs == (epoch,)
    backend.events.append((epoch, False))
    assert not owner.poll_completions()[0].succeeded
    assert pool.free_count == 0
    assert pool.quarantined_count == 1
    with pytest.raises(MemoryError, match="retired pages"):
        pool.reserve()


def test_unknown_completion_event_fails_closed():
    pool = PagedKVPool(1)
    backend = FakeBackend(64)
    owner = PagedKVWriteOwner(pool, backend, page_bytes=64, permit_candidate=True)
    handle = pool.reserve()[0]
    ticket = owner.submit_write(handle, within_page_offset=0, byte_count=1, key_bytes=1, value_bytes=2)
    backend.events.append((ticket.epoch + 1, True))
    with pytest.raises(RuntimeError, match="unknown"):
        owner.poll_completions()
    assert owner.poisoned
    assert pool.references(handle) == 2


def test_copy_pins_both_generations_through_cancel_and_terminal_success():
    pool = PagedKVPool(2)
    backend = FakeBackend(128)
    owner = PagedKVWriteOwner(pool, backend, page_bytes=64, permit_candidate=True)
    source, destination = pool.reserve(2)
    pool.retain((source,))  # A second logical branch owns the source.
    ticket = owner.submit_copy(source, destination, byte_count=17)
    assert backend.copies == [(0, 64, 17, ticket.epoch)]
    assert pool.references(source) == 3 and pool.references(destination) == 2
    pool.release((destination,), after_epoch=ticket.epoch)
    owner.cancel(ticket)
    assert pool.free_count == 0
    backend.events.append((ticket.epoch, True))
    result = owner.poll_completions()
    assert result[0].ticket is ticket and result[0].succeeded
    assert pool.free_count == 1
    assert pool.reserve()[0].generation == destination.generation + 1
    assert pool.references(source) == 2


def test_failed_copy_does_not_release_until_terminal_and_poisons_owner():
    pool = PagedKVPool(2)
    backend = FakeBackend(128)
    owner = PagedKVWriteOwner(pool, backend, page_bytes=64, permit_candidate=True)
    source, destination = pool.reserve(2)
    ticket = owner.submit_copy(source, destination, byte_count=64)
    pool.release((destination,), after_epoch=ticket.epoch)
    assert pool.free_count == 0
    backend.events.append((ticket.epoch, False))
    assert not owner.poll_completions()[0].succeeded
    assert owner.poisoned and pool.free_count == 0
    assert pool.quarantined_count == 2
    with pytest.raises(RuntimeError, match="poisoned"):
        owner.submit_copy(source, destination, byte_count=1)


def test_copy_rejects_stale_or_oversize_before_submission():
    pool = PagedKVPool(2)
    backend = FakeBackend(128)
    owner = PagedKVWriteOwner(pool, backend, page_bytes=64, permit_candidate=True)
    source, destination = pool.reserve(2)
    with pytest.raises(ValueError, match="distinct"):
        owner.submit_copy(source, source, byte_count=1)
    with pytest.raises(ValueError, match="crosses"):
        owner.submit_copy(source, destination, byte_count=65)
    pool.release((destination,), after_epoch=0)
    pool.retire(0)
    with pytest.raises(ValueError, match="stale"):
        owner.submit_copy(source, destination, byte_count=1)
    assert owner.pending_epochs == () and backend.copies == []


def test_ambiguous_copy_enqueue_remains_pinned_until_terminal_event():
    pool = PagedKVPool(2)
    backend = FakeBackend(128)
    backend.raise_after_record = True
    owner = PagedKVWriteOwner(pool, backend, page_bytes=64, permit_candidate=True)
    source, destination = pool.reserve(2)
    with pytest.raises(WriteSubmissionError) as error:
        owner.submit_copy(source, destination, byte_count=5)
    epoch = error.value.epoch
    pool.release((destination,), after_epoch=epoch)
    assert pool.references(destination) == 1 and pool.free_count == 0
    backend.events.append((epoch, False))
    assert not owner.poll_completions()[0].succeeded
    assert pool.free_count == 0
    assert pool.quarantined_count == 2


def test_invalid_batch_does_not_retire_earlier_valid_event():
    pool = PagedKVPool(1)
    backend = FakeBackend(64)
    owner = PagedKVWriteOwner(pool, backend, page_bytes=64, permit_candidate=True)
    handle = pool.reserve()[0]
    ticket = owner.submit_write(handle, within_page_offset=0, byte_count=1,
                                key_bytes=1, value_bytes=2)
    pool.release((handle,), after_epoch=ticket.epoch)
    backend.events.extend(((ticket.epoch, True), (ticket.epoch + 1, False)))
    with pytest.raises(RuntimeError, match="unknown"):
        owner.poll_completions()
    assert owner.poisoned and owner.pending_epochs == (ticket.epoch,)
    assert pool.free_count == 0


def test_conflicting_duplicate_terminal_status_keeps_pin():
    pool = PagedKVPool(1)
    backend = FakeBackend(64)
    owner = PagedKVWriteOwner(pool, backend, page_bytes=64, permit_candidate=True)
    handle = pool.reserve()[0]
    ticket = owner.submit_write(handle, within_page_offset=0, byte_count=1,
                                key_bytes=1, value_bytes=2)
    backend.events.extend(((ticket.epoch, True), (ticket.epoch, False)))
    with pytest.raises(RuntimeError, match="conflicting"):
        owner.poll_completions()
    assert owner.poisoned and owner.pending_epochs == (ticket.epoch,)
    assert pool.references(handle) == 2


def test_failed_arena_teardown_waits_for_terminal_and_is_one_way():
    pool = PagedKVPool(1)
    backend = FakeBackend(64)
    owner = PagedKVWriteOwner(pool, backend, page_bytes=64, permit_candidate=True)
    handle = pool.reserve()[0]
    ticket = owner.submit_write(handle, within_page_offset=0, byte_count=1,
                                key_bytes=1, value_bytes=2)
    owner.poisoned = True  # A later ambiguous native submission poisoned it.
    with pytest.raises(RuntimeError, match="unproven in-flight"):
        owner.teardown_failed_arena()
    assert backend.closed == 0 and pool.references(handle) == 2
    backend.events.append((ticket.epoch, False))
    assert not owner.poll_completions()[0].succeeded
    owner.teardown_failed_arena()
    owner.teardown_failed_arena()
    assert owner.failed_arena_torn_down and backend.closed == 1
    pool.release((handle,), after_epoch=ticket.epoch)
    pool.retire(ticket.epoch)
    assert pool.quarantined_count == 1 and pool.free_count == 0

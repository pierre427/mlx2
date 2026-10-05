"""Token-page ownership tests use CPU MLX arrays and a fake write backend."""

import mlx.core as mx
import pytest

from mlx2.runtime.paged_kv_pool import PagedKVPool
from mlx2.runtime.paged_kv_token import PagedKVTokenOwner, TokenKVProfile
from mlx2.runtime.paged_kv_write import PagedKVWriteOwner


@pytest.fixture(autouse=True)
def cpu_default():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


PROFILE = TokenKVProfile(2, 128, "float16")


class FakeBackend:
    def __init__(self, capacity):
        self.plane_bytes = capacity * PROFILE.page_bytes
        self.writes = []
        self.events = []
        self.copies = []
        self.raise_copy = False

    def validate_sources(self, keys, values):
        return (type(keys) is mx.array and type(values) is mx.array and
                keys.dtype == values.dtype == mx.uint8 and
                keys.ndim == values.ndim == 1)

    def write(self, keys, values, offset, byte_count, epoch):
        self.writes.append((keys, values, offset, byte_count, epoch))
        return object()

    def poll_completions(self):
        events, self.events = self.events, []
        return events

    def copy_page(self, source_offset, destination_offset, byte_count, epoch):
        self.copies.append((source_offset, destination_offset, byte_count, epoch))
        if self.raise_copy:
            raise RuntimeError("ambiguous copy enqueue")
        return object()


def setup(capacity=3):
    pool = PagedKVPool(capacity)
    backend = FakeBackend(capacity)
    writer = PagedKVWriteOwner(pool, backend, page_bytes=PROFILE.page_bytes,
                               permit_candidate=True)
    return PagedKVTokenOwner(writer, PROFILE, permit_candidate=True), backend


def chunks(owner, tokens):
    spans = owner.planned_spans(tokens)
    return (
        tuple(mx.array([3] * span.byte_count, dtype=mx.uint8) for span in spans),
        tuple(mx.array([7] * span.byte_count, dtype=mx.uint8) for span in spans),
    )


def submit(owner, tokens):
    keys, values = chunks(owner, tokens)
    return owner.append(keys, values, token_count=tokens)


def finish(owner, backend, tickets, *, success=True):
    backend.events.extend((ticket.epoch, success) for ticket in tickets)
    return owner.poll_completions()


def test_profile_refusal_and_preflight_do_not_mutate_pool():
    pool = PagedKVPool(1)
    backend = FakeBackend(1)
    writer = PagedKVWriteOwner(pool, backend, page_bytes=PROFILE.page_bytes,
                               permit_candidate=True)
    with pytest.raises(RuntimeError, match="explicit candidate"):
        PagedKVTokenOwner(writer, PROFILE)
    with pytest.raises(ValueError, match="geometry"):
        PagedKVTokenOwner(writer, TokenKVProfile(1, 128, "float16"),
                          permit_candidate=True)
    owner = PagedKVTokenOwner(writer, PROFILE, permit_candidate=True)
    with pytest.raises(ValueError, match="chunk pair"):
        owner.append((), (), token_count=1)
    bad = (mx.array([1], dtype=mx.uint8),) * 2
    with pytest.raises(ValueError, match="exact span bytes"):
        owner.append(bad, bad, token_count=1)
    with pytest.raises(MemoryError, match="insufficient"):
        owner.planned_spans(65)
    assert owner.offset == owner.sequence.kv_end == 0
    assert pool.free_count == 1 and backend.writes == []


def test_reader_compatible_head_major_page_addresses_and_completion_gate():
    owner, backend = setup()
    first = submit(owner, 63)
    assert owner.offset == 0 and len(first) == 2
    assert [(w[2], w[3]) for w in backend.writes] == [
        (0, 63 * PROFILE.head_token_bytes),
        (64 * PROFILE.head_token_bytes, 63 * PROFILE.head_token_bytes),
    ]
    with pytest.raises(RuntimeError, match="pending"):
        owner.accepted_handles()
    assert finish(owner, backend, first)
    spans = owner.planned_spans(2)
    assert [(s.source_token_offset, s.token_count, s.kv_head) for s in spans] == [
        (0, 1, 0), (0, 1, 1), (1, 1, 0), (1, 1, 1),
    ]
    second = submit(owner, 2)
    assert len(second) == 4
    assert [(w[2], w[3]) for w in backend.writes[-4:]] == [
        (63 * PROFILE.head_token_bytes, PROFILE.head_token_bytes),
        (127 * PROFILE.head_token_bytes, PROFILE.head_token_bytes),
        (PROFILE.page_bytes, PROFILE.head_token_bytes),
        (PROFILE.page_bytes + 64 * PROFILE.head_token_bytes, PROFILE.head_token_bytes),
    ]
    backend.events.extend((ticket.epoch, True) for ticket in second[1:])
    assert not owner.poll_completions() and owner.offset == 63
    backend.events.append((second[0].epoch, True))
    assert owner.poll_completions() and owner.offset == 65
    assert len(owner.accepted_handles()) == 2


def test_failure_never_publishes_and_close_waits_for_retirement():
    owner, backend = setup(1)
    tickets = submit(owner, 1)
    handle = owner.sequence.handles[0]
    owner.close()
    assert owner.writer.pool.free_count == 0
    assert not finish(owner, backend, tickets, success=False)
    assert owner.offset == 0
    assert owner.writer.pool.free_count == 0
    assert owner.writer.pool.quarantined_count == 1
    with pytest.raises(MemoryError, match="retired pages"):
        owner.writer.pool.reserve()


def test_shared_tail_copy_and_suffix_publish_only_after_terminal_success():
    owner, backend = setup(3)
    assert finish(owner, backend, submit(owner, 63))
    sibling = owner.sequence.fork()
    before = owner.sequence.handles
    keys, values = chunks(owner, 2)
    copy, = owner.append(keys, values, token_count=2)
    assert owner.sequence.handles == sibling.handles == before
    assert owner.sequence.kv_end == owner.offset == 63
    assert backend.copies == [(before[-1].page_id * PROFILE.page_bytes,
                               owner._stage.destination.page_id * PROFILE.page_bytes,
                               PROFILE.page_bytes, copy.epoch)]
    assert len(backend.writes) == 2
    assert not finish(owner, backend, (copy,))
    assert owner.sequence.handles == before and owner.offset == 63
    assert len(owner._pending) == 4  # two heads on each side of page 64
    assert owner._stage.handles[-2] != before[-1]
    assert not finish(owner, backend, owner._pending[:-1])
    assert owner.sequence.handles == before
    assert finish(owner, backend, owner._pending)
    assert owner.offset == owner.sequence.kv_end == 65
    assert owner.sequence.handles[0] != sibling.handles[0]
    assert len(owner.sequence.handles) == 2
    assert sibling.handles == before
    sibling.abort(after_epoch=owner.writer.ledger.completed_epoch)
    owner.close()


@pytest.mark.parametrize("success,cancel", [(False, False), (True, True)])
def test_shared_tail_failure_or_cancel_keeps_old_table_and_pins(success, cancel):
    owner, backend = setup(2)
    assert finish(owner, backend, submit(owner, 1))
    sibling = owner.sequence.fork()
    old = owner.sequence.handles
    copy, = submit(owner, 1)
    destination = owner._stage.destination
    if cancel:
        owner.cancel_pending()
    assert owner.writer.pool.references(old[0]) == 3
    assert owner.writer.pool.references(destination) == 2
    assert owner.writer.pool.free_count == 0
    assert not finish(owner, backend, (copy,), success=success)
    assert owner.sequence.handles == old and owner.sequence.kv_end == owner.offset == 1
    assert sibling.handles == old
    assert owner.writer.pool.references(old[0]) == 2
    assert owner.writer.pool.free_count == int(success)
    if success:
        reused = owner.writer.pool.reserve()[0]
        assert reused.page_id == destination.page_id
        assert reused.generation == destination.generation + 1
        with pytest.raises(ValueError, match="stale"):
            owner.writer.pool.references(destination)
        owner.writer.pool.release((reused,), after_epoch=owner.writer.ledger.completed_epoch)
    else:
        assert owner.writer.pool.quarantined_count == 2
        with pytest.raises(MemoryError, match="retired pages"):
            owner.writer.pool.reserve()
    sibling.abort(after_epoch=owner.writer.ledger.completed_epoch)
    assert len(backend.writes) == 2
    owner.close()


def test_ambiguous_copy_enqueue_preserves_source_and_pins_destination():
    owner, backend = setup(2)
    assert finish(owner, backend, submit(owner, 1))
    sibling = owner.sequence.fork()
    old = owner.sequence.handles
    backend.raise_copy = True
    with pytest.raises(RuntimeError, match="drain required"):
        submit(owner, 1)
    assert owner.sequence.handles == old and owner.sequence.kv_end == 1
    assert owner.writer.pool.free_count == 0
    epoch = owner.writer.pending_epochs[0]
    backend.events.append((epoch, False))
    assert not owner.poll_completions()
    assert owner.writer.pool.free_count == 0
    assert owner.writer.pool.quarantined_count == 2
    sibling.abort(after_epoch=owner.writer.ledger.completed_epoch)
    owner.close()


def test_suffix_failure_after_successful_copy_does_not_publish_append():
    owner, backend = setup(2)
    assert finish(owner, backend, submit(owner, 1))
    sibling = owner.sequence.fork()
    old = owner.sequence.handles
    copy, = submit(owner, 1)
    assert not finish(owner, backend, (copy,))
    writes = owner._pending
    backend.events.extend([(writes[0].epoch, False), (writes[1].epoch, True)])
    assert not owner.poll_completions()
    assert owner.sequence.handles == old and owner.sequence.kv_end == owner.offset == 1
    assert owner.writer.pool.free_count == 0
    assert owner.writer.pool.quarantined_count >= 1
    sibling.abort(after_epoch=owner.writer.ledger.completed_epoch)
    owner.close()


def test_terminal_prefix_truncation_retires_whole_pages_and_rewrites_tail():
    owner, backend = setup(4)
    assert finish(owner, backend, submit(owner, 63))
    sibling = owner.sequence.fork()
    original = sibling.handles[0]
    copy, = submit(owner, 3)
    assert not finish(owner, backend, (copy,))
    assert finish(owner, backend, owner._pending)
    private, rejected_page = owner.accepted_handles()
    assert private != original and owner.offset == 66
    owner.truncate_accepted(64)
    assert owner.offset == owner.sequence.kv_end == 64
    assert owner.accepted_handles() == (private,)
    assert sibling.handles == (original,) and sibling.kv_end == 63
    assert owner.writer.pool.free_count == 2
    assert finish(owner, backend, submit(owner, 1))
    replacement = owner.accepted_handles()[-1]
    assert replacement.page_id == rejected_page.page_id
    assert replacement.generation == rejected_page.generation + 1
    sibling.abort(after_epoch=owner.writer.ledger.completed_epoch)
    owner.close()


def test_prefix_truncation_refuses_live_reader_lease():
    owner, backend = setup(2)
    assert finish(owner, backend, submit(owner, 65))
    original = owner.accepted_handles()
    lease = owner.writer.ledger.prepare(original)
    owner.writer.ledger.submit(lease)
    with pytest.raises(RuntimeError, match="not terminal"):
        owner.truncate_accepted(1)
    assert owner.offset == 65 and owner.accepted_handles() == original
    owner.writer.ledger.complete(lease)
    owner.truncate_accepted(1)
    assert owner.offset == 1 and len(owner.accepted_handles()) == 1
    owner.close()

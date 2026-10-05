"""CPU contract for private native page-table publication."""

import mlx.core as mx
import pytest

from mlx2.runtime.paged_attention_pack import (prepare_packed_token_read,
                                               prepare_staged_token_read)
from mlx2.runtime.paged_kv_pool import PagedKVPool
from mlx2.runtime.paged_kv_token import PagedKVTokenOwner, TokenKVProfile
from mlx2.runtime.paged_kv_write import PagedKVWriteOwner, WriteSubmissionError
from mlx2.runtime.paged_native_atomic_owner import (
    NativeAtomicError,
    NativeAtomicRequestOwner,
    complete_staged_read_after_event,
)
from mlx2.runtime.paged_request_transaction import STATE_PLANES, CandidateRequest

PROFILE = TokenKVProfile(2, 128, "float16")


class FakeBackend:
    def __init__(self, pages):
        self.plane_bytes = pages * PROFILE.page_bytes
        self.events = []
        self.closed = 0
        self.grouped_calls = []

    def validate_sources(self, key, value):
        return all(type(x) is mx.array and x.dtype == mx.uint8 and x.ndim == 1
                   for x in (key, value))

    def write(self, key, value, offset, byte_count, epoch):
        return object()

    def copy_page(self, source, destination, byte_count, epoch):
        return object()

    def grouped_q1_write(self, keys, values, pages, slots, kv_heads, dim, epoch):
        self.grouped_calls.append((pages, slots, epoch))
        return object()

    def depend_source(self, source, _dependency):
        return source

    def poll_completions(self):
        events, self.events = self.events, []
        return events

    def close_after_terminal(self):
        self.closed += 1


@pytest.fixture(autouse=True)
def cpu_default():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def setup(*, enabled=True, layers=2, prefix=0, reuse_private_tail=False):
    pool = PagedKVPool(16)
    backend = FakeBackend(16)
    writer = PagedKVWriteOwner(pool, backend, page_bytes=PROFILE.page_bytes,
                               permit_candidate=True)
    public = tuple(PagedKVTokenOwner(writer, PROFILE, permit_candidate=True)
                   for _ in range(layers))
    for layer in public:
        if prefix:
            spans = layer.planned_spans(prefix)
            chunks = tuple(mx.array([1] * span.byte_count, dtype=mx.uint8) for span in spans)
            tickets = layer.append(chunks, chunks, token_count=prefix)
            backend.events.extend((ticket.epoch, True) for ticket in tickets)
            assert layer.poll_completions()
    owner = NativeAtomicRequestOwner("r1", public,
                                     {p: ["base"] for p in STATE_PLANES if p != "kv"},
                                     enabled=enabled,
                                     reuse_private_tail=reuse_private_tail)
    return owner, backend, writer


def append_complete(branch, backend, rows=3):
    for layer in branch.layers:
        spans = layer.planned_spans(rows)
        chunks = tuple(mx.array([1] * span.byte_count, dtype=mx.uint8) for span in spans)
        tickets = layer.append(chunks, chunks, token_count=rows)
        backend.events.extend((ticket.epoch, True) for ticket in tickets)
        assert layer.poll_completions()


def prove_all(branch, rows=3):
    for index, layer in enumerate(branch.layers):
        use = prepare_packed_token_read((layer,), (rows,), query_heads=4,
                                        permit_candidate=True)
        use.mark_submitted()
        branch.prove_layer_read(index, use, (use.lease.epoch, True))


def stage_all(branch, rows=3):
    for plane in STATE_PLANES[1:]:
        branch.stage(plane, [f"{plane}:{i}" for i in range(rows)])


@pytest.mark.parametrize("prefix", [63, 65])
def test_exclusive_request_tail_uses_one_grouped_ticket_without_cow(prefix):
    owner, backend, writer = setup(prefix=prefix, reuse_private_tail=True)
    prior = owner.snapshot()
    prior.close()
    branch = owner.begin(CandidateRequest(1, "r1", 1, STATE_PLANES))
    assert all(layer._reuse_shared_tail_refs == 2 for layer in branch.layers)
    with pytest.raises(NativeAtomicError, match="excludes public readers"):
        owner.snapshot()
    destinations = tuple(layer.stage_grouped_q1(object(), object())
                         for layer in branch.layers)
    assert all(layer._stage.source is None for layer in branch.layers)
    ticket = writer.submit_grouped_q1_write(
        tuple(row[0] for row in destinations), tuple(row[1] for row in destinations),
        keys=object(), values=object(), kv_heads=PROFILE.kv_heads, dim=PROFILE.head_dim)
    for layer in branch.layers:
        layer.adopt_grouped_q1(ticket)
    assert len(backend.grouped_calls) == 1
    backend.events.append((ticket.epoch, True))
    for layer in branch.layers:
        layer.poll_completions()
    stage_all(branch, 1)
    prove_all(branch, 1)
    branch.prepare(1).publish()
    with owner.snapshot() as published:
        assert published.offset == prefix + 1
        assert all(table[-1] == destination[0] for table, destination in
                   zip(published.layer_tables, destinations))


def test_external_tail_reference_and_public_reader_keep_cow():
    owner, _, _ = setup(prefix=63, reuse_private_tail=True)
    external = owner._public.layers[0].sequence.fork()  # APCv2-style retained page.
    branch = owner.begin(CandidateRequest(1, "r1", 1, STATE_PLANES))
    assert branch.layers[0]._reuse_shared_tail_refs == 1
    with pytest.raises(ValueError, match="shared tail"):
        branch.layers[0].stage_grouped_q1(object(), object())
    branch.rollback()
    external.abort(after_epoch=0)
    owner2, _, _ = setup(prefix=63, reuse_private_tail=True)
    reader = owner2.snapshot()
    branch2 = owner2.begin(CandidateRequest(1, "r1", 1, STATE_PLANES))
    assert all(layer._reuse_shared_tail_refs == 1 for layer in branch2.layers)
    with pytest.raises(ValueError, match="shared tail"):
        branch2.layers[0].stage_grouped_q1(object(), object())
    branch2.rollback()
    reader.close()


def test_cancelled_shared_tail_excludes_readers_until_terminal_and_reap():
    owner, backend, writer = setup(prefix=63, reuse_private_tail=True)
    branch = owner.begin(CandidateRequest(1, "r1", 1, STATE_PLANES))
    destinations = tuple(layer.stage_grouped_q1(object(), object())
                         for layer in branch.layers)
    ticket = writer.submit_grouped_q1_write(
        tuple(row[0] for row in destinations), tuple(row[1] for row in destinations),
        keys=object(), values=object(), kv_heads=PROFILE.kv_heads, dim=PROFILE.head_dim)
    for layer in branch.layers:
        layer.adopt_grouped_q1(ticket)
    branch.rollback()
    with pytest.raises(NativeAtomicError, match="excludes public readers"):
        owner.snapshot()
    assert owner.reap_quarantine() == 0
    backend.events.append((ticket.epoch, True))
    for layer in branch.layers:
        layer.poll_completions()
    assert owner.reap_quarantine() == 1
    with owner.snapshot() as published:
        assert published.offset == 63 and published.generation == 0


def test_successful_private_write_rollback_allows_exact_next_continuation():
    owner, backend, writer = setup(prefix=65, reuse_private_tail=True)
    parent_tables = tuple(layer.accepted_handles() for layer in owner._public.layers)
    for publish in (False, True):
        branch = owner.begin(CandidateRequest(1, "r1", 1, STATE_PLANES))
        destinations = tuple(layer.stage_grouped_q1(object(), object())
                             for layer in branch.layers)
        assert tuple(row[0] for row in destinations) == tuple(
            table[-1] for table in parent_tables)
        ticket = writer.submit_grouped_q1_write(
            tuple(row[0] for row in destinations), tuple(row[1] for row in destinations),
            keys=object(), values=object(), kv_heads=PROFILE.kv_heads, dim=PROFILE.head_dim)
        for layer in branch.layers:
            layer.adopt_grouped_q1(ticket)
        backend.events.append((ticket.epoch, True))
        for layer in branch.layers:
            assert layer.poll_completions()
        if publish:
            stage_all(branch, 1)
            prove_all(branch, 1)
            branch.prepare(1).publish()
        else:
            branch.rollback()
            with owner.snapshot() as public:
                assert public.offset == 65 and public.layer_tables == parent_tables
    with owner.snapshot() as public:
        assert public.offset == 66 and public.generation == 1


def test_failed_shared_tail_requires_one_way_arena_teardown():
    owner, backend, writer = setup(prefix=65, reuse_private_tail=True)
    branch = owner.begin(CandidateRequest(1, "r1", 1, STATE_PLANES))
    destinations = tuple(layer.stage_grouped_q1(object(), object())
                         for layer in branch.layers)
    ticket = writer.submit_grouped_q1_write(
        tuple(row[0] for row in destinations), tuple(row[1] for row in destinations),
        keys=object(), values=object(), kv_heads=PROFILE.kv_heads, dim=PROFILE.head_dim)
    for layer in branch.layers:
        layer.adopt_grouped_q1(ticket)
    backend.events.append((ticket.epoch, False))
    for layer in branch.layers:
        assert not layer.poll_completions()
    branch.rollback()
    with pytest.raises(NativeAtomicError, match="excludes public readers"):
        owner.snapshot()
    assert owner.reap_quarantine() == 0
    owner.close()
    writer.teardown_failed_arena()
    assert owner.reap_failed_after_teardown() == 2
    assert owner._tail_guard is None
    assert owner.fully_retired


def test_default_off_and_forks_preserve_public_tables():
    disabled, _, _ = setup(enabled=False)
    request = CandidateRequest(1, "r1", 3, STATE_PLANES)
    with pytest.raises(NativeAtomicError, match="disabled"):
        disabled.begin(request)
    owner, backend, writer = setup()
    before = owner.snapshot()
    branch = owner.begin(request)
    assert all(child is not parent for child, parent in zip(branch.layers, owner._public.layers))
    append_complete(branch, backend)
    assert owner.snapshot() == before
    branch.rollback()
    assert owner.snapshot() == before
    assert not writer.pending_epochs


def test_unpublished_write_cannot_be_used_as_a_packed_read_plan():
    owner, backend, writer = setup(layers=1)
    branch = owner.begin(CandidateRequest(1, "r1", 3, STATE_PLANES))
    layer = branch.layers[0]
    spans = layer.planned_spans(3)
    chunks = tuple(mx.array([1] * span.byte_count, dtype=mx.uint8) for span in spans)
    tickets = layer.append(chunks, chunks, token_count=3)
    public = owner.snapshot()
    try:
        assert public.offset == 0 and public.layer_tables == ((),)
    finally:
        public.close()
    with pytest.raises(ValueError, match="completed KV"):
        prepare_packed_token_read((layer,), (3,), query_heads=4,
                                  permit_candidate=True)
    assert layer.offset == 0 and writer.pending_epochs
    backend.events.extend((ticket.epoch, True) for ticket in tickets)
    assert layer.poll_completions()
    use = prepare_packed_token_read((layer,), (3,), query_heads=4,
                                    permit_candidate=True)
    use.abort_before_submit()
    branch.rollback()
    assert not writer.pending_epochs and writer.ledger.pending_count == 0


def test_staged_private_read_needs_every_write_and_read_terminal_before_publish():
    owner, backend, writer = setup(layers=1)
    branch = owner.begin(CandidateRequest(1, "r1", 3, STATE_PLANES))
    layer = branch.layers[0]
    spans = layer.planned_spans(3)
    chunks = tuple(mx.array([1] * span.byte_count, dtype=mx.uint8) for span in spans)
    tickets = layer.append_staged(chunks, chunks, token_count=3)
    use = prepare_staged_token_read((layer,), (3,), query_heads=4,
                                    permit_candidate=True)
    assert use.plan.spans[0].kv_end == 3 and layer.offset == 0
    with owner.snapshot() as public:
        assert public.offset == 0
    use.mark_submitted()
    proof = complete_staged_read_after_event(use, (use.lease.epoch, True))
    with pytest.raises(NativeAtomicError, match="private layer"):
        branch.prove_staged_layer_read(0, 0, proof)
    backend.events.extend((ticket.epoch, True) for ticket in tickets)
    assert layer.poll_completions()
    branch.prove_staged_layer_read(0, 0, proof)
    stage_all(branch)
    branch.prepare(3).publish()
    with owner.snapshot() as published:
        assert published.offset == 3 and len(published.layer_tables[0]) == 1
    assert not writer.pending_epochs and writer.ledger.pending_count == 0


def test_ambiguous_staged_write_retains_arena_and_private_pages_until_teardown():
    owner, backend, writer = setup(layers=1)
    branch = owner.begin(CandidateRequest(1, "r1", 3, STATE_PLANES))
    layer = branch.layers[0]
    spans = layer.planned_spans(3)
    chunks = tuple(mx.array([1] * span.byte_count, dtype=mx.uint8) for span in spans)

    def ambiguous_write(_key, _value, _offset, _count, epoch):
        backend.events.append((epoch, True))
        raise RuntimeError("submission outcome unknown")

    backend.write = ambiguous_write
    with pytest.raises(WriteSubmissionError):
        layer.append_staged(chunks, chunks, token_count=3)
    assert owner._public.generation == 0 and owner._public.layers[0].offset == 0
    assert writer.poisoned and writer.pending_epochs and layer._stage is not None
    assert layer._sources is not None
    branch.rollback()
    owner.close()
    assert owner._quarantine and not owner.fully_retired
    with pytest.raises(RuntimeError, match="in-flight"):
        writer.teardown_failed_arena()
    writer.poll_completions()
    writer.teardown_failed_arena()
    assert backend.closed == 1
    assert owner.reap_failed_after_teardown() == 2
    assert owner.fully_retired


def test_staged_shared_tail_copy_keeps_public_page_until_all_terminals():
    owner, backend, writer = setup(layers=1)
    first = owner.begin(CandidateRequest(1, "r1", 63, STATE_PLANES))
    append_complete(first, backend, 63)
    stage_all(first, 63)
    prove_all(first, 63)
    first.prepare(63).publish()
    with owner.snapshot() as public:
        original = public.layer_tables[0][0]
    branch = owner.begin(CandidateRequest(2, "r1", 1, STATE_PLANES))
    layer = branch.layers[0]
    spans = layer.planned_spans(1)
    chunks = tuple(mx.array([1] * span.byte_count, dtype=mx.uint8) for span in spans)
    tickets = layer.append_staged(chunks, chunks, token_count=1)
    assert len(tickets) == 1 + len(spans)  # COW plus every head-local write.
    use = prepare_staged_token_read((layer,), (1,), query_heads=4,
                                    permit_candidate=True)
    assert use.plan.page_table[0] != original
    with owner.snapshot() as public:
        assert public.offset == 63 and public.layer_tables[0][0] == original
    use.mark_submitted()
    backend.events.extend((ticket.epoch, True) for ticket in tickets)
    assert layer.poll_completions()
    proof = complete_staged_read_after_event(use, (use.lease.epoch, True))
    branch.prove_staged_layer_read(0, 0, proof)
    stage_all(branch, 1)
    branch.prepare(1).publish()
    with owner.snapshot() as public:
        assert public.offset == 64 and public.layer_tables[0][0] != original
    assert not writer.pending_epochs and writer.ledger.pending_count == 0


def test_full_boundary_publishes_layers_and_companions_together():
    owner, backend, _ = setup()
    request = CandidateRequest(1, "r1", 3, STATE_PLANES)
    branch = owner.begin(request)
    append_complete(branch, backend)
    stage_all(branch)
    prove_all(branch)
    prior = owner.snapshot()
    prepared = branch.prepare(3)
    assert owner.snapshot() == prior
    prepared.publish()
    published = owner.snapshot()
    assert published.generation == 1 and published.offset == 3
    assert all(len(table) == 1 for table in published.layer_tables)
    assert all(len(rows) == 4 for _, rows in published.companions)
    assert len(owner._retired) == 1


@pytest.mark.parametrize("accepted", [0, 1, 2])
def test_partial_or_zero_acceptance_publishes_exact_prefix(accepted):
    owner, backend, writer = setup()
    request = CandidateRequest(1, "r1", 3, STATE_PLANES)
    before = owner.snapshot()
    branch = owner.begin(request)
    append_complete(branch, backend)
    stage_all(branch)
    prove_all(branch)
    prepared = branch.prepare(accepted)
    assert owner.snapshot() == before
    prepared.publish()
    after = owner.snapshot()
    assert after.offset == accepted and after.generation == 1
    assert all(layer.offset == accepted and layer.sequence.kv_end == accepted
               for layer in owner._public.layers)
    assert all(len(rows) == 1 + accepted for _, rows in after.companions)
    assert all(len(table) == (1 if accepted else 0) for table in after.layer_tables)
    assert not writer.pending_epochs and writer.ledger.pending_count == 0


def test_partial_acceptance_at_page_boundary_reclaims_rejected_page():
    owner, backend, writer = setup(layers=1)
    first = owner.begin(CandidateRequest(1, "r1", 63, STATE_PLANES))
    append_complete(first, backend, 63)
    stage_all(first, 63)
    prove_all(first, 63)
    first.prepare(63).publish()
    origin = owner.snapshot()
    branch = owner.begin(CandidateRequest(2, "r1", 3, STATE_PLANES))
    layer = branch.layers[0]
    spans = layer.planned_spans(3)
    chunks = tuple(mx.array([1] * span.byte_count, dtype=mx.uint8) for span in spans)
    copy, = layer.append(chunks, chunks, token_count=3)
    backend.events.append((copy.epoch, True))
    assert not layer.poll_completions()
    backend.events.extend((ticket.epoch, True) for ticket in layer._pending)
    assert layer.poll_completions()
    rejected = layer.accepted_handles()[-1]
    stage_all(branch, 3)
    prove_all(branch, 3)
    branch.prepare(1).publish()
    published = owner.snapshot()
    assert origin.offset == 63 and published.offset == 64
    assert len(origin.layer_tables[0]) == len(published.layer_tables[0]) == 1
    assert origin.layer_tables[0][0] != published.layer_tables[0][0]
    assert rejected not in published.layer_tables[0]
    assert writer.pool.free_count >= 1


def test_closed_read_without_matched_success_proof_refuses():
    owner, backend, _ = setup()
    branch = owner.begin(CandidateRequest(1, "r1", 3, STATE_PLANES))
    append_complete(branch, backend)
    stage_all(branch)
    use = prepare_packed_token_read((branch.layers[0],), (3,), query_heads=4,
                                    permit_candidate=True)
    use.mark_submitted()
    use.complete_after_proof()  # Closed alone carries no success bit.
    with pytest.raises(NativeAtomicError, match="does not cover"):
        branch.prove_layer_read(0, use, (use.lease.epoch, True))
    with pytest.raises(NativeAtomicError, match="every layer"):
        branch.prepare(3)
    branch.rollback()


def test_failed_terminal_read_refuses_and_rollback_preserves_public():
    owner, backend, _ = setup()
    request = CandidateRequest(1, "r1", 3, STATE_PLANES)
    before = owner.snapshot()
    branch = owner.begin(request)
    append_complete(branch, backend)
    stage_all(branch)
    use = prepare_packed_token_read((branch.layers[0],), (3,), query_heads=4,
                                    permit_candidate=True)
    use.mark_submitted()
    with pytest.raises(NativeAtomicError, match="failed"):
        branch.prove_layer_read(0, use, (use.lease.epoch, False))
    branch.rollback()
    with pytest.raises(NativeAtomicError, match="poisoned"):
        owner.snapshot()
    assert owner._public.generation == before.generation
    assert owner._quarantine == [branch]
    assert owner.reap_quarantine() == 0
    assert owner._public.layers[0].writer.pool.quarantined_count >= 1
    owner.close()
    assert not owner.fully_retired
    with pytest.raises(NativeAtomicError, match="reader"):
        owner.reap_failed_after_teardown()
    before.close()
    with pytest.raises(NativeAtomicError, match="teardown"):
        owner.reap_failed_after_teardown()
    owner._public.layers[0].writer.teardown_failed_arena()
    assert not owner.fully_retired
    assert owner.reap_failed_after_teardown() == 2
    assert owner.fully_retired
    assert owner.reap_failed_after_teardown() == 0
    assert backend.closed == 1


def test_cancelled_pending_branch_is_quarantined_until_terminal_event():
    owner, backend, writer = setup(layers=1)
    request = CandidateRequest(1, "r1", 3, STATE_PLANES)
    before = owner.snapshot()
    branch = owner.begin(request)
    layer = branch.layers[0]
    spans = layer.planned_spans(3)
    chunks = tuple(mx.array([1] * span.byte_count, dtype=mx.uint8) for span in spans)
    tickets = layer.append(chunks, chunks, token_count=3)
    branch.rollback()
    assert owner.reap_quarantine() == 0 and owner.snapshot() == before
    backend.events.extend((ticket.epoch, True) for ticket in tickets)
    layer.poll_completions()
    assert owner.reap_quarantine() == 1
    assert not writer.pending_epochs and owner.snapshot() == before


def test_competing_prepared_branch_cannot_publish_after_public_swap():
    owner, backend, _ = setup()
    request = CandidateRequest(1, "r1", 3, STATE_PLANES)
    first, second = owner.begin(request), owner.begin(request)
    for branch in (first, second):
        append_complete(branch, backend)
        stage_all(branch)
        prove_all(branch)
    prepared_first, prepared_second = first.prepare(3), second.prepare(3)
    prepared_first.publish()
    winner = owner.snapshot()
    with pytest.raises(NativeAtomicError, match="drifted"):
        prepared_second.publish()
    prepared_second.rollback()
    assert owner.snapshot() == winner

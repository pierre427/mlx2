"""CPU ownership tests for bounded revision-bound GDN/KV publication."""

from dataclasses import dataclass

import mlx.core as mx
import pytest

from mlx2.runtime.paged_attention_pack import (prepare_packed_token_read,
                                               prepare_staged_token_read)
from mlx2.runtime.paged_gdn_checkpoint import GDNBoundaryCheckpoint
from mlx2.runtime.paged_kv_pool import PagedKVPool
from mlx2.runtime.paged_kv_token import PagedKVTokenOwner, TokenKVProfile
from mlx2.runtime.paged_kv_write import PagedKVWriteOwner
from mlx2.runtime.paged_native_atomic_owner import (
    NativeAtomicError, NativeAtomicRequestOwner,
    complete_staged_read_after_event, publish_native_cohort)
from mlx2.runtime.paged_request_transaction import CandidateRequest


PROFILE = TokenKVProfile(2, 128, "float16")


@dataclass
class Box:
    leaf: object


class Backend:
    def __init__(self):
        self.plane_bytes = 128 * PROFILE.page_bytes
        self.events = []
        self.closed = 0
        self.grouped_calls = []

    def validate_sources(self, key, value):
        return all(type(x) is mx.array and x.dtype == mx.uint8 and x.ndim == 1
                   for x in (key, value))

    def write(self, *_args):
        return object()

    def copy_page(self, *_args):
        return object()

    def grouped_q1_write(self, keys, values, pages, slots, kv_heads, dim, epoch):
        self.grouped_calls.append((pages, slots, epoch))
        return object()

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


def setup(*, prefix=0, reuse_private_tail=False,
          accepted_prefix_checkpoints=False):
    pool = PagedKVPool(128)
    backend = Backend()
    writer = PagedKVWriteOwner(pool, backend, page_bytes=PROFILE.page_bytes,
                               permit_candidate=True)
    layers = tuple(PagedKVTokenOwner(writer, PROFILE, permit_candidate=True)
                   for _ in range(2))
    if prefix:
        for layer in layers:
            spans = layer.planned_spans(prefix)
            chunks = tuple(mx.array([1] * span.byte_count, dtype=mx.uint8)
                           for span in spans)
            tickets = layer.append(chunks, chunks, token_count=prefix)
            backend.events.extend((ticket.epoch, True) for ticket in tickets)
            assert layer.poll_completions()
    leaves = (object(), object(), object())
    initial = tuple(Box(leaf) for leaf in leaves)
    checkpoint = GDNBoundaryCheckpoint("revision", 17, prefix, 0, initial)
    clone = lambda caches: tuple(Box(cache.leaf) for cache in caches)
    owner = NativeAtomicRequestOwner(
        "revision", layers, {"gdn": (checkpoint,)},
        supported_planes=("kv", "gdn"), enabled=True,
        checkpoint_planes=("gdn",),
        accepted_prefix_checkpoints=accepted_prefix_checkpoints,
        recurrent_clone=clone, lane_id=17,
        reuse_private_tail=reuse_private_tail)
    return owner, backend, leaves, initial


def request():
    return CandidateRequest(17, "revision", 1, ("kv", "gdn"))


def append_and_prove(branch, backend):
    for index, layer in enumerate(branch.layers):
        spans = layer.planned_spans(1)
        chunks = tuple(mx.array([1] * span.byte_count, dtype=mx.uint8)
                       for span in spans)
        tickets = layer.append(chunks, chunks, token_count=1)
        backend.events.extend((ticket.epoch, True) for ticket in tickets)
        for _ in range(3):
            if layer.poll_completions():
                break
            backend.events.extend((epoch, True) for epoch in layer.writer.pending_epochs)
        else:
            raise AssertionError("native KV copy/write did not reach terminal success")
        use = prepare_packed_token_read((layer,), (1,), query_heads=4,
                                        permit_candidate=True)
        use.mark_submitted()
        branch.prove_layer_read(index, use, (use.lease.epoch, True))


def checkpoint_from(view):
    assert len(view.companions) == 1
    plane, rows = view.companions[0]
    assert plane == "gdn" and len(rows) == 1
    return rows[0]


def test_checkpoint_replaces_one_generation_without_history_growth():
    owner, backend, leaves, initial = setup()
    with owner.snapshot() as view:
        assert checkpoint_from(view).caches == initial
    for generation in range(1, 7):
        branch = owner.begin(request())
        assert len(branch.recurrent_caches) == len(initial)
        assert all(private is not public and private.leaf is leaf
                   for private, public, leaf in zip(branch.recurrent_caches,
                                                    initial if generation == 1 else prior,
                                                    leaves))
        append_and_prove(branch, backend)
        branch.stage_recurrent_boundary(branch.recurrent_caches, offset=generation)
        branch.prepare(1).publish()
        assert owner.reap_retired() == 1
        with owner.snapshot() as view:
            checkpoint = checkpoint_from(view)
            assert (view.generation, view.offset) == (generation, generation)
            assert (checkpoint.generation, checkpoint.offset) == (generation, generation)
            assert len(view.companions) == 1 and len(checkpoint.caches) == 3
            prior = checkpoint.caches
        assert not owner._retired


def test_bootstrap_cache_alias_cannot_mutate_active_checkpoint():
    owner, _, leaves, bootstrap = setup()
    with owner.snapshot() as view:
        active = checkpoint_from(view)
        assert all(owned is not source and owned.leaf is leaf
                   for owned, source, leaf in zip(active.caches, bootstrap, leaves))
    bootstrap[0].leaf = object()
    with owner.snapshot() as view:
        assert checkpoint_from(view).caches[0].leaf is leaves[0]
    branch = owner.begin(request())
    assert branch.recurrent_caches[0].leaf is leaves[0]
    branch.rollback()


def test_reader_retains_old_kv_and_checkpoint_until_lease_closes():
    owner, backend, _, initial = setup()
    old = owner.snapshot()
    branch = owner.begin(request())
    append_and_prove(branch, backend)
    branch.stage_recurrent_boundary(branch.recurrent_caches, offset=1)
    branch.prepare(1).publish()
    assert owner.reap_retired() == 0
    assert old.offset == 0 and checkpoint_from(old).caches == initial
    with owner.snapshot() as new:
        assert new.offset == 1 and checkpoint_from(new).caches is not initial
    old.close()
    with pytest.raises(NativeAtomicError, match="closed"):
        _ = old.companions
    assert owner.reap_retired() == 1


def test_stale_prepared_branch_cannot_publish_partial_hybrid_state():
    owner, backend, _, _ = setup()
    first = owner.begin(request())
    second = owner.begin(request())
    for branch in (first, second):
        append_and_prove(branch, backend)
        branch.stage_recurrent_boundary(branch.recurrent_caches, offset=1)
    a, b = first.prepare(1), second.prepare(1)
    a.publish()
    with owner.snapshot() as published:
        public_table = published.layer_tables
        public_checkpoint = checkpoint_from(published)
    with pytest.raises(NativeAtomicError, match="drifted"):
        b.publish()
    b.rollback()
    with owner.snapshot() as published:
        assert published.offset == 1 and published.generation == 1
        assert published.layer_tables == public_table
        assert checkpoint_from(published) is public_checkpoint
    assert owner.reap_quarantine() == 0
    assert owner.reap_retired() == 1


def test_bad_lane_revision_offset_and_clone_fail_before_publication():
    owner, _, _, _ = setup()
    for bad in (CandidateRequest(18, "revision", 1, ("kv", "gdn")),
                CandidateRequest(17, "wrong", 1, ("kv", "gdn")),
                CandidateRequest(17, "revision", 2, ("kv", "gdn"))):
        with pytest.raises(NativeAtomicError):
            owner.begin(bad)
    with owner.snapshot() as view:
        assert view.offset == 0 and view.generation == 0
    for bad_clone in (lambda caches: caches,
                      lambda caches: (Box(caches[0].leaf),) * len(caches)):
        layers = owner._public.layers
        with pytest.raises(ValueError, match="private"):
            NativeAtomicRequestOwner(
                "revision", layers, {"gdn": owner._public.companions[0][1]},
                supported_planes=("kv", "gdn"), enabled=True,
                checkpoint_planes=("gdn",), recurrent_clone=bad_clone,
                lane_id=17).begin(request())


def test_unstaged_or_wrong_boundary_and_unproved_read_refuse_prepare():
    owner, backend, _, _ = setup()
    branch = owner.begin(request())
    with pytest.raises(NativeAtomicError, match="offset"):
        branch.stage_recurrent_boundary(branch.recurrent_caches, offset=2)
    with pytest.raises(NativeAtomicError, match="private"):
        branch.stage_recurrent_boundary(tuple(Box(1) for _ in branch.recurrent_caches),
                                        offset=1)
    with pytest.raises(NativeAtomicError, match="boundary"):
        branch.stage("gdn", [branch.recurrent_caches])
    append_and_prove(branch, backend)
    with pytest.raises(NativeAtomicError, match="not staged"):
        branch.prepare(1)
    branch.stage_recurrent_boundary(branch.recurrent_caches, offset=1)
    with pytest.raises(NativeAtomicError, match="positive accepted prefix"):
        branch.prepare(0)
    branch.rollback()
    with owner.snapshot() as view:
        assert view.offset == 0 and view.generation == 0
    branch = owner.begin(request())
    branch.stage_recurrent_boundary(branch.recurrent_caches, offset=1)
    with pytest.raises(NativeAtomicError, match="terminal read proof"):
        branch.prepare(1)
    branch.rollback()


def test_online_prefix_checkpoint_publishes_short_terminal_proposal():
    owner, backend, _, _ = setup(accepted_prefix_checkpoints=True)
    branch = owner.begin(CandidateRequest(17, "revision", 3, ("kv", "gdn")))
    for _step in range(2):
        for index, layer in enumerate(branch.layers):
            spans = layer.planned_spans(1)
            chunks = tuple(mx.array([1] * span.byte_count, dtype=mx.uint8)
                           for span in spans)
            tickets = layer.append_staged(chunks, chunks, token_count=1)
            use = prepare_staged_token_read((layer,), (1,), query_heads=4,
                                            permit_candidate=True)
            use.mark_submitted()
            backend.events.extend((ticket.epoch, True) for ticket in tickets)
            assert layer.poll_completions()
            proof = complete_staged_read_after_event(
                use, (use.lease.epoch, True))
            branch.prove_staged_layer_read(index, 0, proof)
    branch.stage_recurrent_prefix(
        branch.recurrent_caches, accepted_rows=2, offset=2)
    branch.seal_executed_rows(2)
    branch.prepare(2).publish()
    with owner.snapshot() as view:
        assert (view.offset, view.generation) == (2, 1)
        checkpoint = checkpoint_from(view)
        assert (checkpoint.offset, checkpoint.generation) == (2, 1)
    assert owner.reap_retired() == 1


def test_cohort_publish_preflights_every_owner_before_first_pointer_swap():
    first, first_backend, _, _ = setup()
    second, second_backend, _, _ = setup()
    prepared=[]
    for owner,backend in ((first,first_backend),(second,second_backend)):
        branch=owner.begin(request())
        append_and_prove(branch,backend)
        branch.stage_recurrent_boundary(branch.recurrent_caches,offset=1)
        prepared.append(branch.prepare(1))
    second._public.layers[0].writer.poisoned=True
    with pytest.raises(NativeAtomicError,match='drifted'):
        publish_native_cohort(tuple(prepared))
    assert first._public.generation==0 and second._public.generation==0
    second._public.layers[0].writer.poisoned=False
    for state in prepared:state.rollback()


def test_cohort_publish_swaps_all_validated_owners():
    first, first_backend, _, _ = setup()
    second, second_backend, _, _ = setup()
    prepared=[]
    for owner,backend in ((first,first_backend),(second,second_backend)):
        branch=owner.begin(request())
        append_and_prove(branch,backend)
        branch.stage_recurrent_boundary(branch.recurrent_caches,offset=1)
        prepared.append(branch.prepare(1))
    publish_native_cohort(tuple(prepared))
    assert (first._public.generation,second._public.generation)==(1,1)


def test_pending_or_ambiguous_write_quarantines_private_checkpoint():
    owner, backend, _, initial = setup()
    branch = owner.begin(request())
    layer = branch.layers[0]
    spans = layer.planned_spans(1)
    chunks = tuple(mx.array([1] * span.byte_count, dtype=mx.uint8) for span in spans)
    layer.append(chunks, chunks, token_count=1)
    private = branch.recurrent_caches
    branch.rollback()
    assert owner.reap_quarantine() == 0
    assert owner._quarantine[0].recurrent_caches is private
    with owner.snapshot() as view:
        assert view.offset == 0 and checkpoint_from(view).caches == initial
    # A later successful write terminal permits exact private retirement.
    backend.events.extend((epoch, True) for epoch in tuple(layer.writer.pending_epochs))
    layer.poll_completions()
    assert owner.reap_quarantine() == 1
    assert branch.recurrent_caches == ()
    with owner.snapshot() as view:
        assert view.offset == 0 and checkpoint_from(view).caches == initial


def test_failed_terminal_keeps_private_checkpoint_until_one_way_teardown():
    owner, backend, _, initial = setup()
    branch = owner.begin(request())
    layer = branch.layers[0]
    spans = layer.planned_spans(1)
    chunks = tuple(mx.array([1] * span.byte_count, dtype=mx.uint8) for span in spans)
    tickets = layer.append(chunks, chunks, token_count=1)
    private = branch.recurrent_caches
    backend.events.extend((ticket.epoch, False) for ticket in tickets)
    assert not layer.poll_completions()
    assert layer.writer.poisoned
    branch.rollback()
    assert owner.reap_quarantine() == 0
    assert owner._quarantine[0].recurrent_caches is private
    assert owner._public.layers[0].offset == 0
    assert owner._public.companions[0][1][0].caches == initial
    owner.close()
    layer.writer.teardown_failed_arena()
    assert backend.closed == 1
    assert owner.reap_failed_after_teardown() == 2
    assert branch.recurrent_caches == () and owner.fully_retired


def test_partial_tail_grouped_write_reuses_only_without_public_reader():
    owner, backend, _, initial = setup(prefix=63, reuse_private_tail=True)
    branch = owner.begin(request())
    assert all(layer._reuse_shared_tail_refs == 2 for layer in branch.layers)
    assert all(private is not public for private, public in
               zip(branch.recurrent_caches, initial))
    destinations = tuple(layer.stage_grouped_q1(object(), object())
                         for layer in branch.layers)
    writer = branch.layers[0].writer
    ticket = writer.submit_grouped_q1_write(
        tuple(row[0] for row in destinations),
        tuple(row[1] for row in destinations),
        keys=object(), values=object(), kv_heads=PROFILE.kv_heads,
        dim=PROFILE.head_dim)
    for layer in branch.layers:
        layer.adopt_grouped_q1(ticket)
    assert len(backend.grouped_calls) == 1
    backend.events.append((ticket.epoch, True))
    for index, layer in enumerate(branch.layers):
        layer.poll_completions()
        use = prepare_packed_token_read((layer,), (1,), query_heads=4,
                                        permit_candidate=True)
        use.mark_submitted()
        branch.prove_layer_read(index, use, (use.lease.epoch, True))
    branch.stage_recurrent_boundary(branch.recurrent_caches, offset=64)
    branch.prepare(1).publish()
    with owner.snapshot() as view:
        assert (view.offset, view.generation) == (64, 1)
        assert checkpoint_from(view).offset == 64
    assert owner.reap_retired() == 1

    blocked, _, _, _ = setup(prefix=63, reuse_private_tail=True)
    old_reader = blocked.snapshot()
    private = blocked.begin(request())
    assert all(layer._reuse_shared_tail_refs == 1 for layer in private.layers)
    with pytest.raises(ValueError, match="shared tail"):
        private.layers[0].stage_grouped_q1(object(), object())
    private.rollback()
    assert old_reader.offset == 63 and checkpoint_from(old_reader).offset == 63
    old_reader.close()
    with blocked.snapshot() as view:
        assert (view.offset, view.generation) == (63, 0)

"""Private paged leaf contract with a fake backend; no Metal construction."""

import pytest

from mlx2.runtime.paged_kv_cache import PagedKVPrivateCache
from mlx2.runtime.paged_kv_pool import PagedKVPool
from mlx2.runtime.paged_kv_write import NativeWriteBackend, PagedKVWriteOwner

TOKEN_BYTES = 2 * 128 * 2
PAGE_BYTES = 64 * TOKEN_BYTES


class FakeBackend:
    def __init__(self, capacity):
        self.plane_bytes = capacity * PAGE_BYTES
        self.writes = []
        self.events = []

    def write(self, key, value, offset, byte_count, epoch):
        self.writes.append((key, value, offset, byte_count, epoch))
        return object()

    def copy_page(self, source, destination, byte_count, epoch):
        self.writes.append(("copy", source, destination, byte_count, epoch))
        return object()

    def poll_completions(self):
        result, self.events = self.events, []
        return result


def cache(capacity=3):
    pool = PagedKVPool(capacity)
    backend = FakeBackend(capacity)
    writer = PagedKVWriteOwner(pool, backend, page_bytes=PAGE_BYTES,
                               permit_candidate=True)
    return PagedKVPrivateCache(writer, kv_heads=2, head_dim=128,
                               dtype="float16", permit_candidate=True), backend


def payload(tokens):
    return b"k" * (tokens * TOKEN_BYTES), b"v" * (tokens * TOKEN_BYTES)


def complete(owner, backend, tickets, *, succeeded=True):
    backend.events.extend((ticket.epoch, succeeded) for ticket in tickets)
    return owner.poll_completions()


def test_default_off_and_preflight_refusal_before_mutation():
    pool = PagedKVPool(1)
    backend = FakeBackend(1)
    writer = PagedKVWriteOwner(pool, backend, page_bytes=PAGE_BYTES,
                               permit_candidate=True)
    with pytest.raises(RuntimeError, match="explicit candidate"):
        PagedKVPrivateCache(writer, kv_heads=2, head_dim=128, dtype="float16")
    owner = PagedKVPrivateCache(writer, kv_heads=2, head_dim=128,
                                dtype="float16", permit_candidate=True)
    with pytest.raises(ValueError, match="whole tokens"):
        owner.append(b"x", b"x")
    with pytest.raises(MemoryError, match="insufficient"):
        owner.append(*payload(65))
    assert owner.offset == owner.sequence.kv_end == 0
    assert pool.free_count == 1 and backend.writes == []


def test_token_major_host_cache_refuses_native_head_major_arena_before_mutation():
    pool = PagedKVPool(1)
    backend = object.__new__(NativeWriteBackend)  # No MLX or Metal construction.
    backend.plane_bytes = PAGE_BYTES
    writer = PagedKVWriteOwner(pool, backend, page_bytes=PAGE_BYTES,
                               permit_candidate=True)
    owner = PagedKVPrivateCache(writer, kv_heads=2, head_dim=128,
                                dtype="float16", permit_candidate=True)
    with pytest.raises(ValueError, match="token-major private cache staging"):
        owner.append(*payload(1))
    assert owner.sequence.handles == () and owner.offset == 0
    assert pool.free_count == 1 and writer.pending_epochs == ()


def test_boundary_append_accepts_only_after_completion_and_exports_exact_bytes():
    owner, backend = cache()
    first = owner.append(*payload(63))
    assert owner.offset == 0 and len(first) == 1
    with pytest.raises(RuntimeError, match="pending"):
        owner.export_exact()
    assert complete(owner, backend, first)
    second_payload = payload(2)
    second = owner.append(*second_payload)
    assert len(second) == 2
    assert backend.writes[-2][2:4] == (63 * TOKEN_BYTES, TOKEN_BYTES)
    assert backend.writes[-1][2:4] == (PAGE_BYTES, TOKEN_BYTES)
    assert complete(owner, backend, second)
    exported = owner.export_exact()
    assert exported.tokens == 65
    assert exported.key_bytes == payload(63)[0] + second_payload[0]
    assert exported.value_bytes == payload(63)[1] + second_payload[1]


def test_attention_lease_pins_generations_until_terminal_proof():
    owner, backend = cache(1)
    assert complete(owner, backend, owner.append(*payload(1)))
    use = owner.attention(row_count=1, query_heads=8)
    handle = owner.sequence.handles[0]
    assert use.plan.page_for_token(0, 0)[0] == handle
    assert owner.writer.pool.references(handle) == 2
    with pytest.raises(ValueError, match="unsupported mask"):
        owner.attention(row_count=1, query_heads=8, mask_kind="sink")
    assert owner.writer.pool.references(handle) == 2
    use.mark_submitted()
    owner.close()
    assert owner.writer.pool.free_count == 0
    with pytest.raises(ValueError, match="terminal proof"):
        use.abort_before_submit()
    use.complete_after_proof()
    assert owner.writer.pool.free_count == 1
    assert owner.writer.pool.reserve()[0].generation == handle.generation + 1


def test_failed_write_never_publishes_and_poisons_cache():
    owner, backend = cache(1)
    tickets = owner.append(*payload(1))
    assert not complete(owner, backend, tickets, succeeded=False)
    assert owner.offset == 0
    with pytest.raises(RuntimeError, match="failed"):
        owner.export_exact()
    owner.close()
    assert owner.writer.pool.free_count == 0
    assert owner.writer.pool.quarantined_count == 1


def test_frozen_partial_tail_branch_refuses_append_without_mutation():
    owner, backend = cache(4)
    assert complete(owner, backend, owner.append(*payload(3)))
    frozen = owner.freeze_exact()
    old = frozen.sequence.handles[0]
    branch = frozen.branch(permit_candidate=True, staging_headroom_pages=1)
    assert branch.sequence.handles[0] == old
    assert owner.writer.pool.references(old) == 3
    assert owner.writer.pool.free_count == 2  # One actual staging reservation.
    writes = len(backend.writes)
    with pytest.raises(ValueError, match="staged native COW"):
        branch.append(*payload(1))
    assert len(backend.writes) == writes
    assert branch.sequence.handles[0] == old
    assert branch.offset == 3
    assert owner.export_exact().tokens == frozen.export.tokens == 3
    branch.close()
    assert owner.writer.pool.references(old) == 2
    frozen.close()
    assert owner.writer.pool.references(old) == 1
    owner.close()
    assert owner.writer.pool.free_count == 4


def test_frozen_branch_headroom_refusal_is_atomic():
    owner, backend = cache(1)
    assert complete(owner, backend, owner.append(*payload(1)))
    frozen = owner.freeze_exact()
    with pytest.raises(MemoryError, match="headroom"):
        frozen.branch(permit_candidate=True, staging_headroom_pages=1)
    assert owner.writer.pool.references(owner.sequence.handles[0]) == 2
    frozen.close()
    owner.close()


def test_full_page_frozen_branch_appends_independently():
    owner, backend = cache(2)
    assert complete(owner, backend, owner.append(*payload(64)))
    frozen = owner.freeze_exact()
    branch = frozen.branch(permit_candidate=True)
    tickets = branch.append(*payload(1))
    assert branch.offset == 64
    assert complete(branch, backend, tickets)
    assert branch.offset == 65
    assert owner.offset == frozen.export.tokens == 64
    branch.close()
    assert owner.writer.pool.free_count == 1
    frozen.close()
    owner.close()
    assert owner.writer.pool.free_count == 2


def test_segmented_group_refuses_paged_row_before_building_any_adapter():
    from mlx2.runtime.models.cache import KVCache
    from mlx2.runtime.segmented_batch_cache import (
        SegmentedBatchUnsupported,
        build_segmented_batch_cache_group,
    )

    owner, _ = cache()
    with pytest.raises(SegmentedBatchUnsupported, match="explicit segmented"):
        build_segmented_batch_cache_group([[KVCache()], [owner]])

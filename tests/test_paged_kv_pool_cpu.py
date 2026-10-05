"""CPU-only page ownership checks; no GPU lifetime or byte-copy claim."""

import random
from collections import Counter

import pytest

from mlx2.runtime.paged_kv_pool import PagedKVPool, PagedKVSequence


def test_reserve_retain_release_retire_and_aba():
    pool = PagedKVPool(1)
    first = pool.reserve()[0]
    pool.retain((first,))
    pool.release((first,), after_epoch=8)
    assert pool.references(first) == 1
    pool.release((first,), after_epoch=8)
    assert pool.pending_count == 1
    assert pool.live_generations() == {}
    pool.retire(7)
    with pytest.raises(MemoryError):
        pool.reserve()
    pool.retire(8)
    second = pool.reserve()[0]
    assert second.page_id == first.page_id
    assert second.generation == first.generation + 1
    with pytest.raises(ValueError, match="stale"):
        pool.retain((first,))
    with pytest.raises(ValueError, match="stale"):
        pool.release((first,), after_epoch=8)
    assert pool.references(second) == 1


def test_atomic_reservation_and_double_release():
    pool = PagedKVPool(2)
    with pytest.raises(MemoryError):
        pool.reserve(3)
    assert pool.free_count == 2
    handles = pool.reserve(2)
    with pytest.raises(ValueError, match="unique"):
        pool.release((handles[0], handles[0]), after_epoch=1)
    assert [pool.references(h) for h in handles] == [1, 1]
    pool.release(handles, after_epoch=1)
    with pytest.raises(ValueError, match="unreferenced"):
        pool.release((handles[0],), after_epoch=1)
    pool.retire(1)
    assert pool.free_count == 2


def test_terminal_failure_quarantines_only_touched_generation_for_arena_lifetime():
    pool = PagedKVPool(2)
    failed, clean = pool.reserve(2)
    pool.quarantine((failed,))
    assert pool.live_generations() == {clean.page_id: clean.generation}
    with pytest.raises(ValueError, match="quarantined"):
        pool.retain((failed,))
    pool.release((failed, clean), after_epoch=3)
    pool.retire(3)
    assert pool.free_count == 1 and pool.quarantined_count == 1
    assert pool.allocated_count == 0
    assert pool.reserve()[0].page_id == clean.page_id
    with pytest.raises(MemoryError, match="retired pages"):
        pool.reserve()


def test_branch_cows_partial_tail_but_shares_full_pages():
    pool = PagedKVPool(5)
    parent = PagedKVSequence(pool)
    assert parent.append(65, after_epoch=0) == ()
    original = parent.handles
    child = parent.fork()
    assert child.handles == original
    copied = child.append(1, after_epoch=3)
    assert copied == ((original[-1], child.handles[-1]),)
    assert child.handles[0] == original[0]
    assert child.handles[-1] != original[-1]
    assert parent.handles == original
    child.abort(after_epoch=4)
    parent.abort(after_epoch=4)
    assert pool.pending_count == 3
    pool.retire(3)
    assert pool.free_count == 2  # only never-allocated slots are free
    pool.retire(4)
    assert pool.free_count == 5


def test_trim_partial_and_append_after_empty_trim():
    pool = PagedKVPool(4)
    owner = PagedKVSequence(pool)
    owner.append(130, after_epoch=0)
    first = owner.handles[0]
    owner.trim(65, after_epoch=5)
    assert owner.first_block == 1
    assert len(owner.handles) == 2
    with pytest.raises(ValueError, match="unreferenced"):
        pool.references(first)
    owner.trim(130, after_epoch=6)
    assert owner.handles == ()
    assert owner.first_block == 2
    owner.append(1, after_epoch=6)
    assert len(owner.handles) == 1
    assert owner.kv_end == 131
    owner.abort(after_epoch=7)
    pool.retire(7)
    assert pool.free_count == 4


def test_append_capacity_failure_leaves_branch_unchanged():
    pool = PagedKVPool(1)
    parent = PagedKVSequence(pool)
    parent.append(1, after_epoch=0)
    child = parent.fork()
    before = (child.kv_end, child.handles, pool.references(child.handles[0]))
    with pytest.raises(MemoryError):
        child.append(1, after_epoch=2)
    assert (child.kv_end, child.handles, pool.references(child.handles[0])) == before
    child.abort(after_epoch=2)
    parent.abort(after_epoch=2)


def test_randomized_branch_append_trim_abort_accounting():
    rng = random.Random(0xC0FFEE)
    pool = PagedKVPool(512)
    owners = [PagedKVSequence(pool)]
    for epoch in range(1, 801):
        owner = rng.choice(owners)
        action = rng.choice(("append", "append", "fork", "trim", "abort"))
        if action == "append":
            owner.append(rng.randint(1, 81), after_epoch=epoch)
        elif action == "fork" and len(owners) < 12:
            owners.append(owner.fork())
        elif action == "trim" and owner.kv_end > owner.retained_start:
            owner.trim(rng.randint(owner.retained_start, owner.kv_end), after_epoch=epoch)
        elif action == "abort" and len(owners) > 1:
            owner.abort(after_epoch=epoch)
            owners.remove(owner)
        if epoch % 7 == 0:
            pool.retire(epoch - 2)
        references = Counter(handle for live in owners for handle in live.handles)
        assert pool.allocated_count == len(references)
        assert pool.free_count + pool.pending_count + pool.allocated_count == pool.capacity
        assert set(pool.live_generations().items()) == {
            (handle.page_id, handle.generation) for handle in references
        }
        assert all(pool.references(handle) == count for handle, count in references.items())
        for live in owners:
            assert live.retained_start <= live.kv_end
            assert len(live.handles) == (
                (live.kv_end - 1) // 64 - live.first_block + 1
                if live.retained_start < live.kv_end else 0
            )
    for owner in owners:
        owner.abort(after_epoch=801)
    pool.retire(801)
    assert pool.free_count == pool.capacity

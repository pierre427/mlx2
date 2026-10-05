"""CPU protocol checks for the opt-in native Qwen3 backend."""

import asyncio
import time
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters.qwen3_paged_candidate import PackedLane, Qwen3PackedCandidate
from mlx2.runtime import qwen3_paged_native_backend as module
from mlx2.runtime.paged_kv_pool import PagedKVPool
from mlx2.runtime.paged_kv_token import PagedKVTokenOwner, TokenKVProfile
from mlx2.runtime.paged_kv_write import PagedKVWriteOwner
from mlx2.runtime.paged_native_atomic_owner import (
    NativeAtomicError,
    NativeAtomicRequestOwner,
)
from mlx2.runtime.paged_request_transaction import STATE_PLANES, CandidateRequest

PROFILE = TokenKVProfile(2, 128, "float16")


class FakeNativeArena:
    def __init__(self, pages):
        self.stream = mx.cpu
        self.plane_bytes = pages * PROFILE.page_bytes
        self.keys = bytearray(self.plane_bytes)
        self.values = bytearray(self.plane_bytes)
        self.events = []
        self.read_events = []
        self.dependencies = []
        self.succeed = True
        self.poll_calls = 0
        self.wait_calls = 0
        self.group_count = 0
        self.group_raise_after = False
        self._native = self
        self.storage_dtype = "float16"
        self.multirow_count = 0
        self.multirow_rows = 0
        self.multirow_raise_after = False

    def grouped_multirow_write_count(self):
        return self.multirow_count

    def grouped_multirow_row_count(self):
        return self.multirow_rows

    def grouped_multirow_write(self, keys, values, counts, starts, first_blocks,
                               table_begins, page_ids, kv_heads, dim, epoch):
        self.multirow_count += 1
        self.multirow_rows += sum(counts)
        for lane in range(2):
            for local in range(counts[lane]):
                row = local + (counts[0] if lane else 0)
                logical = starts[lane] + local
                page = page_ids[table_begins[lane] + logical // 64 - first_blocks[lane]]
                for head in range(kv_heads):
                    target = page * PROFILE.page_bytes + (
                        head * 64 + logical % 64) * PROFILE.head_token_bytes
                    self.keys[target:target + dim * 2] = np.array(keys)[row, head].tobytes()
                    self.values[target:target + dim * 2] = np.array(values)[row, head].tobytes()
        dependency = mx.array([1], dtype=mx.uint8)
        self.dependencies.append(dependency)
        self.events.append((epoch, self.succeed))
        if self.multirow_raise_after:
            raise RuntimeError("ambiguous packed multirow enqueue")
        return dependency

    def q1_tile_dispatch_count(self):
        return 0

    def q1_split_partial_dispatch_count(self):
        return 0

    def q1_split_reduce_dispatch_count(self):
        return 0

    def grouped_q1_write_count(self):
        return self.group_count

    def write_dispatch_count(self):
        return 0

    def grouped_q1_write(self, keys, values, pages, slots, kv_heads, dim, epoch):
        self.group_count += 1
        for row in range(2):
            for head in range(kv_heads):
                target = pages[row] * PROFILE.page_bytes + (
                    head * 64 + slots[row]) * PROFILE.head_token_bytes
                self.keys[target:target + dim * 2] = np.array(keys)[row, head].tobytes()
                self.values[target:target + dim * 2] = np.array(values)[row, head].tobytes()
        dependency = mx.array([1], dtype=mx.uint8)
        self.dependencies.append(dependency)
        self.events.append((epoch, self.succeed))
        if self.group_raise_after:
            raise RuntimeError("ambiguous grouped enqueue")
        return dependency

    def depend_source(self, source, dependency):
        assert dependency is not None
        return source

    def validate_sources(self, key, value):
        return all(type(x) is mx.array and x.dtype == mx.uint8 and x.ndim == 1
                   for x in (key, value))

    def write(self, key, value, offset, byte_count, epoch):
        self.keys[offset:offset + byte_count] = np.array(key).tobytes()
        self.values[offset:offset + byte_count] = np.array(value).tobytes()
        dependency = mx.array([1], dtype=mx.uint8)
        self.dependencies.append(dependency)
        self.events.append((epoch, self.succeed))
        return dependency

    def poll_completions(self):
        self.poll_calls += 1
        result, self.events = self.events, []
        return result

    def wait_completions(self, timeout_seconds):
        self.wait_calls += 1
        assert 0 <= timeout_seconds <= 120
        result, self.events = self.events, []
        return result

    def copy_page(self, source_offset, destination_offset, byte_count, epoch):
        self.keys[destination_offset:destination_offset + byte_count] = self.keys[
            source_offset:source_offset + byte_count]
        self.values[destination_offset:destination_offset + byte_count] = self.values[
            source_offset:source_offset + byte_count]
        dependency = mx.array([1], dtype=mx.uint8)
        self.dependencies.append(dependency)
        self.events.append((epoch, self.succeed))
        return dependency


def setup(monkeypatch, pages=12, profile_host=False):
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(module, "NativeWriteBackend", FakeNativeArena)
    pool = PagedKVPool(pages)
    arena = FakeNativeArena(pages)
    writer = PagedKVWriteOwner(pool, arena, page_bytes=PROFILE.page_bytes,
                               permit_candidate=True)
    backend = module.NativeQwen3PagedBackend(writer, permit_candidate=True,
                                              timeout_s=0.05,
                                              profile_host=profile_host)
    return pool, arena, writer, backend


def projected(rows, bias=0):
    values = np.arange(rows * 2 * 128, dtype=np.float16).reshape(rows, 2, 128)
    return mx.array(values + bias)


def test_head_major_bytes_ragged_lanes_and_all_terminal_writes(monkeypatch):
    pool, arena, writer, backend = setup(monkeypatch)
    owners = tuple(PagedKVTokenOwner(writer, PROFILE, permit_candidate=True)
                   for _ in range(2))
    keys, values = projected(128), projected(128, 7)
    backend.append_completed(owners, keys, values, (65, 63))
    assert tuple(owner.offset for owner in owners) == (65, 63)
    assert not writer.pending_epochs and writer.ledger.pending_count == 0
    assert backend._dependency is arena.dependencies[-1]
    assert arena.wait_calls > 0 and arena.poll_calls == 0
    expected = np.array(keys)
    for lane, owner in enumerate(owners):
        start, count = ((0, 65), (65, 63))[lane]
        for token in range(count):
            handle = owner.sequence.handles[token // 64]
            for head in range(2):
                offset = handle.page_id * PROFILE.page_bytes + (
                    head * 64 + token % 64) * PROFILE.head_token_bytes
                assert arena.keys[offset:offset + 256] == expected[start + token, head].tobytes()
    assert pool.free_count == 9


def test_packing_uses_the_writer_stream(monkeypatch):
    _, arena, writer, backend = setup(monkeypatch)
    seen = []
    original = mx.contiguous

    def recording(values, *args, **kwargs):
        seen.append(kwargs.get("stream"))
        return original(values, *args, **kwargs)

    monkeypatch.setattr(mx, "contiguous", recording)
    owner = PagedKVTokenOwner(writer, PROFILE, permit_candidate=True)
    backend.append_completed((owner,), projected(1), projected(1), (1,))
    assert seen and all(stream is arena.stream for stream in seen)


def test_optional_host_profile_counts_real_pinned_plan(monkeypatch):
    from mlx2.runtime.paged_attention_pack import prepare_packed_token_read

    _, _, writer, backend = setup(monkeypatch, profile_host=True)
    owner = PagedKVTokenOwner(writer, PROFILE, permit_candidate=True)
    backend.append_completed((owner,), projected(1), projected(1), (1,))
    use = prepare_packed_token_read((owner,), (1,), query_heads=4,
                                    permit_candidate=True, profile_host=True)
    backend._profile_read(use)
    receipt = backend.profile_snapshot()
    assert receipt["reads"][0]["physical_threadgroups"] == 4
    assert receipt["reads"][0]["serial_kv_steps"] == 4
    assert receipt["host_ns"]["plan_metadata"] >= 0
    use.abort_before_submit()
    owner.close()


def test_shared_partial_tail_cow_waits_for_new_write_dependency(monkeypatch):
    _, arena, writer, backend = setup(monkeypatch)
    owner = PagedKVTokenOwner(writer, PROFILE, permit_candidate=True)
    backend.append_completed((owner,), projected(1), projected(1), (1,))
    sibling = owner.sequence.fork()
    prior = owner.sequence.handles
    backend.append_completed((owner,), projected(1, 3), projected(1, 4), (1,))
    assert owner.offset == 2 and owner.sequence.handles != prior
    assert sibling.handles == prior
    assert backend._dependency is arena.dependencies[-1]
    assert not writer.pending_epochs and writer.ledger.pending_count == 0
    sibling.abort(after_epoch=writer.ledger.completed_epoch)


def test_grouped_q1_one_ticket_pins_both_owners_until_terminal(monkeypatch):
    monkeypatch.setenv("MLX2_PAGED_GROUPED_Q1_WRITE", "1")
    pool, arena, writer, backend = setup(monkeypatch)
    owners = tuple(PagedKVTokenOwner(writer, PROFILE, permit_candidate=True)
                   for _ in range(2))
    keys, values = projected(2), projected(2, 3)
    tickets = backend.append_staged(owners, keys, values, (1, 1))
    assert len(tickets) == 1 and arena.group_count == 1
    foreign = PagedKVTokenOwner(writer, PROFILE, permit_candidate=True)
    foreign.stage_grouped_q1(keys, values)
    with pytest.raises(RuntimeError, match="ticket does not match"):
        foreign.adopt_grouped_q1(tickets[0])
    foreign.abort_prepared_grouped_q1()
    assert len(writer.pending_epochs) == writer.ledger.pending_count == 1
    assert all(owner.offset == 0 and owner.staged_handles(1) for owner in owners)
    assert pool.free_count == 10
    writer.poll_completions()
    assert tuple(owner.poll_completions() for owner in owners) == (True, True)
    assert tuple(owner.offset for owner in owners) == (1, 1)
    assert not writer.pending_epochs and writer.ledger.pending_count == 0
    for row, owner in enumerate(owners):
        page = owner.accepted_handles()[0].page_id
        for head in range(2):
            target = page * PROFILE.page_bytes + head * 64 * PROFILE.head_token_bytes
            assert arena.keys[target:target + 256] == np.array(keys)[row, head].tobytes()


def test_packed_multirow_one_terminal_exact_head_major_pages(monkeypatch):
    monkeypatch.setenv("MLX2_PAGED_GROUPED_MULTIROW_WRITE", "1")
    pool, arena, writer, backend = setup(monkeypatch, pages=8)
    owners = tuple(PagedKVTokenOwner(writer, PROFILE, permit_candidate=True)
                   for _ in range(2))
    counts = (63, 65)
    keys, values = projected(sum(counts)), projected(sum(counts), 5)
    ticket, = backend.append_packed_multirow(owners, keys, values, counts,
                                             permit_candidate=True)
    assert ticket.packed_multirow and not ticket.grouped_q1
    assert arena.multirow_count == 1 and arena.multirow_rows == 128
    assert writer.ledger.pending_count == 1
    assert tuple(owner.offset for owner in owners) == (0, 0)
    assert tuple(len(owner.staged_handles(n)) for owner, n in zip(owners, counts)) == (1, 2)
    for lane, owner in enumerate(owners):
        for local in (0, counts[lane] - 1):
            row = local + (counts[0] if lane else 0)
            page = owner.staged_handles(counts[lane])[local // 64].page_id
            for head in range(PROFILE.kv_heads):
                target = page * PROFILE.page_bytes + (
                    head * 64 + local % 64) * PROFILE.head_token_bytes
                assert arena.keys[target:target + 256] == np.array(keys)[row, head].tobytes()
                assert arena.values[target:target + 256] == np.array(values)[row, head].tobytes()
    writer.poll_completions()
    assert tuple(owner.poll_completions() for owner in owners) == (True, True)
    assert tuple(owner.offset for owner in owners) == counts
    assert writer.ledger.pending_count == 0
    assert pool.free_count == 5


def test_packed_multirow_failure_paths_keep_or_release_both_reservations(monkeypatch):
    monkeypatch.setenv("MLX2_PAGED_GROUPED_MULTIROW_WRITE", "1")
    pool, arena, writer, backend = setup(monkeypatch, pages=1)
    owners = tuple(PagedKVTokenOwner(writer, PROFILE, permit_candidate=True)
                   for _ in range(2))
    with pytest.raises(MemoryError):
        backend.append_packed_multirow(owners, projected(2), projected(2), (1, 1),
                                       permit_candidate=True)
    assert pool.free_count == 1 and writer.ledger.pending_count == 0
    assert all(owner._stage is None for owner in owners)
    pool, arena, writer, backend = setup(monkeypatch)
    owners = tuple(PagedKVTokenOwner(writer, PROFILE, permit_candidate=True)
                   for _ in range(2))
    arena.multirow_raise_after = True
    with pytest.raises(RuntimeError, match="epoch"):
        backend.append_packed_multirow(owners, projected(2), projected(2), (1, 1),
                                       permit_candidate=True)
    assert writer.poisoned and writer.ledger.pending_count == 1
    assert all(owner._stage is not None and len(owner._pending) == 1 for owner in owners)
    writer.poll_completions()
    assert tuple(owner.poll_completions() for owner in owners) == (False, False)
    assert tuple(owner.offset for owner in owners) == (0, 0)


def test_packed_multirow_terminal_failure_and_late_adoption(monkeypatch):
    monkeypatch.setenv("MLX2_PAGED_GROUPED_MULTIROW_WRITE", "1")
    _, arena, writer, backend = setup(monkeypatch)
    arena.succeed = False
    owners = tuple(PagedKVTokenOwner(writer, PROFILE, permit_candidate=True)
                   for _ in range(2))
    backend.append_packed_multirow(owners, projected(2), projected(2), (1, 1),
                                   permit_candidate=True)
    writer.poll_completions()
    assert writer.poisoned
    assert tuple(owner.poll_completions() for owner in owners) == (False, False)
    assert tuple(owner.offset for owner in owners) == (0, 0)
    _, arena, writer, backend = setup(monkeypatch)
    owners = tuple(PagedKVTokenOwner(writer, PROFILE, permit_candidate=True)
                   for _ in range(2))

    def late_failure(_ticket):
        raise RuntimeError("second adoption rejected")

    monkeypatch.setattr(owners[1], "adopt_packed_multirow", late_failure)
    with pytest.raises(RuntimeError, match="second adoption rejected"):
        backend.append_packed_multirow(owners, projected(2), projected(2), (1, 1),
                                       permit_candidate=True)
    assert writer.poisoned and writer.ledger.pending_count == 1
    assert all(owner._failed and owner._pending for owner in owners)
    writer.poll_completions()
    assert tuple(owner.poll_completions() for owner in owners) == (False, False)


def test_packed_multirow_shared_tail_refuses_before_dispatch(monkeypatch):
    monkeypatch.setenv("MLX2_PAGED_GROUPED_MULTIROW_WRITE", "1")
    _, arena, writer, backend = setup(monkeypatch)
    owners = tuple(PagedKVTokenOwner(writer, PROFILE, permit_candidate=True)
                   for _ in range(2))
    backend.append_packed_multirow(owners, projected(2), projected(2), (1, 1),
                                   permit_candidate=True)
    writer.poll_completions()
    assert all(owner.poll_completions() for owner in owners)
    sibling = owners[0].sequence.fork()
    with pytest.raises(ValueError, match="shared-tail"):
        backend.append_packed_multirow(owners, projected(2), projected(2), (1, 1),
                                       permit_candidate=True)
    assert arena.multirow_count == 1 and writer.ledger.pending_count == 0
    assert tuple(owner.offset for owner in owners) == (1, 1)
    sibling.abort(after_epoch=writer.ledger.completed_epoch)


def test_packed_multirow_selector_and_source_preflight_allocate_nothing(monkeypatch):
    pool, arena, writer, backend = setup(monkeypatch)
    owners = tuple(PagedKVTokenOwner(writer, PROFILE, permit_candidate=True)
                   for _ in range(2))
    with pytest.raises(ValueError, match="capability"):
        backend.append_packed_multirow(owners, projected(2), projected(2), (1, 1),
                                       permit_candidate=True)
    monkeypatch.setenv("MLX2_PAGED_GROUPED_MULTIROW_WRITE", "1")
    with pytest.raises(ValueError, match="shape/dtype"):
        backend.append_packed_multirow(owners, projected(3), projected(2), (1, 1),
                                       permit_candidate=True)
    with pytest.raises(ValueError, match="capability"):
        backend.append_packed_multirow(owners, projected(2), projected(2), (0, 2),
                                       permit_candidate=True)
    assert pool.free_count == 12 and arena.multirow_count == 0
    assert writer.ledger.pending_count == 0
    assert all(owner._stage is None for owner in owners)


def test_grouped_q1_terminal_failure_never_publishes(monkeypatch):
    monkeypatch.setenv("MLX2_PAGED_GROUPED_Q1_WRITE", "1")
    _, arena, writer, backend = setup(monkeypatch)
    arena.succeed = False
    owners = tuple(PagedKVTokenOwner(writer, PROFILE, permit_candidate=True)
                   for _ in range(2))
    backend.append_staged(owners, projected(2), projected(2), (1, 1))
    writer.poll_completions()
    assert writer.poisoned
    assert tuple(owner.poll_completions() for owner in owners) == (False, False)
    assert tuple(owner.offset for owner in owners) == (0, 0)


def test_ambiguous_grouped_q1_enqueue_retains_both_reservations(monkeypatch):
    monkeypatch.setenv("MLX2_PAGED_GROUPED_Q1_WRITE", "1")
    _, arena, writer, backend = setup(monkeypatch)
    arena.group_raise_after = True
    owners = tuple(PagedKVTokenOwner(writer, PROFILE, permit_candidate=True)
                   for _ in range(2))
    with pytest.raises(RuntimeError, match="epoch"):
        backend.append_staged(owners, projected(2), projected(2), (1, 1))
    assert writer.poisoned and writer.ledger.pending_count == 1
    assert all(owner._stage is not None and len(owner._pending) == 1 for owner in owners)
    writer.poll_completions()
    assert tuple(owner.poll_completions() for owner in owners) == (False, False)
    assert tuple(owner.offset for owner in owners) == (0, 0)


def test_second_owner_adoption_failure_poisons_shared_group(monkeypatch):
    monkeypatch.setenv("MLX2_PAGED_GROUPED_Q1_WRITE", "1")
    _, _, writer, backend = setup(monkeypatch)
    owners = tuple(PagedKVTokenOwner(writer, PROFILE, permit_candidate=True)
                   for _ in range(2))

    def fail_adoption(_ticket):
        raise RuntimeError("late adoption failure")

    monkeypatch.setattr(owners[1], "adopt_grouped_q1", fail_adoption)
    with pytest.raises(RuntimeError, match="late adoption failure"):
        backend.append_staged(owners, projected(2), projected(2), (1, 1))
    assert writer.poisoned and writer.ledger.pending_count == 1
    assert all(owner._failed and len(owner._pending) == 1 for owner in owners)
    writer.poll_completions()
    assert tuple(owner.poll_completions() for owner in owners) == (False, False)
    assert tuple(owner.offset for owner in owners) == (0, 0)


def test_grouped_q1_second_reservation_failure_rolls_back_first(monkeypatch):
    monkeypatch.setenv("MLX2_PAGED_GROUPED_Q1_WRITE", "1")
    pool, arena, writer, backend = setup(monkeypatch, pages=1)
    owners = tuple(PagedKVTokenOwner(writer, PROFILE, permit_candidate=True)
                   for _ in range(2))
    with pytest.raises(MemoryError):
        backend.append_staged(owners, projected(2), projected(2), (1, 1))
    assert not writer.poisoned and not writer.pending_epochs
    assert pool.free_count == 1 and arena.group_count == 0
    assert all(owner._stage is None and owner.offset == 0 for owner in owners)


def test_grouped_q1_shared_tail_falls_back_before_submission(monkeypatch):
    monkeypatch.setenv("MLX2_PAGED_GROUPED_Q1_WRITE", "1")
    _, arena, writer, backend = setup(monkeypatch)
    owners = tuple(PagedKVTokenOwner(writer, PROFILE, permit_candidate=True)
                   for _ in range(2))
    backend.append_staged(owners, projected(2), projected(2), (1, 1))
    writer.poll_completions()
    for owner in owners:
        owner.poll_completions()
    sibling = owners[0].sequence.fork()
    tickets = backend.append_staged(owners, projected(2), projected(2), (1, 1))
    assert arena.group_count == 1 and len(tickets) > 1
    sibling.abort(after_epoch=writer.ledger.completed_epoch)


class FakeModel:
    args = SimpleNamespace(model_type="qwen3", num_experts=0, rope_scaling=None,
                           head_dim=128, num_attention_heads=4, num_key_value_heads=2)
    layers = (object(), object())

    def paged_embed(self, tokens):
        return mx.array(tokens, dtype=mx.float16)

    def paged_project(self, layer, hidden, counts, offsets):
        rows = len(hidden)
        return (mx.zeros((rows, 4, 128), dtype=mx.float16),
                projected(rows, layer), projected(rows, layer + 4))

    def paged_finish_layer(self, layer, hidden, attended):
        return hidden

    def paged_logits(self, hidden):
        return hidden


def test_candidate_two_layers_waits_for_native_read_callback(monkeypatch):
    pool, arena, writer, backend = setup(monkeypatch)
    seen = []

    def submit(use, native, query, dependency, *, scale, permit_candidate):
        assert native is arena and dependency is arena.dependencies[-1]
        assert permit_candidate and scale == 128 ** -0.5
        use.mark_submitted()
        seen.append((use.lease.epoch, tuple(s.row_count for s in use.plan.spans)))
        arena.read_events.append((use.lease.epoch, True))
        return mx.zeros(query.shape, dtype=mx.float16)

    def poll(native):
        result, arena.read_events = arena.read_events, []
        return tuple(result)

    monkeypatch.setattr(module, "native_paged_attention_read_fp16", submit)
    monkeypatch.setattr(module, "poll_native_paged_read_events", poll)
    lanes = tuple(PackedLane(tokens, tuple(
        PagedKVTokenOwner(writer, PROFILE, permit_candidate=True) for _ in range(2)))
        for tokens in ((1, 2, 3), (4,)))
    logits, receipt = Qwen3PackedCandidate(FakeModel(), backend).forward(
        lanes, permit_candidate=True)
    assert logits.shape == (4,)
    assert [counts for _, counts in seen] == [(3, 1), (3, 1)]
    assert [[owner.offset for owner in lane.layers] for lane in lanes] == [[3, 3], [1, 1]]
    assert writer.ledger.pending_count == 0 and pool.free_count == 8
    assert receipt["selected"] and not receipt["qualified"]
    assert backend.read_submissions == backend.terminal_successes == 2


def test_failed_write_prevents_publication_and_poisoned_backend(monkeypatch):
    _, arena, writer, backend = setup(monkeypatch)
    arena.succeed = False
    owner = PagedKVTokenOwner(writer, PROFILE, permit_candidate=True)
    with pytest.raises(RuntimeError, match="write failed"):
        backend.append_completed((owner,), projected(1), projected(1), (1,))
    assert owner.offset == 0 and writer.poisoned
    with pytest.raises(RuntimeError, match="failed"):
        backend.append_completed((owner,), projected(1), projected(1), (1,))


def test_unsupported_bfloat16_refuses_before_write(monkeypatch):
    _, arena, writer, backend = setup(monkeypatch)
    owner = PagedKVTokenOwner(writer, TokenKVProfile(2, 128, "bfloat16"),
                              permit_candidate=True)
    with pytest.raises(ValueError, match="fp16"):
        backend.append_completed((owner,), projected(1), projected(1), (1,))
    assert not arena.dependencies and owner.offset == 0


def test_missing_read_callback_keeps_lease_pinned(monkeypatch):
    from mlx2.runtime.paged_attention_pack import prepare_packed_token_read

    _, _, writer, backend = setup(monkeypatch)
    owner = PagedKVTokenOwner(writer, PROFILE, permit_candidate=True)
    backend.append_completed((owner,), projected(1), projected(1), (1,))
    use = prepare_packed_token_read((owner,), (1,), query_heads=4,
                                    permit_candidate=True)

    def submit(use, *args, **kwargs):
        use.mark_submitted()
        return mx.zeros((1, 4, 128), dtype=mx.float16)

    monkeypatch.setattr(module, "native_paged_attention_read_fp16", submit)
    monkeypatch.setattr(module, "poll_native_paged_read_events", lambda _: ())
    waits = []

    def wait_without_callback(_, timeout_seconds):
        waits.append(timeout_seconds)
        time.sleep(timeout_seconds)
        return ()

    monkeypatch.setattr(module, "wait_native_paged_read_events", wait_without_callback)
    with pytest.raises(TimeoutError, match="read timed out"):
        backend.read_completed(use, mx.zeros((1, 4, 128), dtype=mx.float16),
                               scale=128 ** -0.5)
    assert use.state == "submitted" and writer.ledger.pending_count == 1
    assert len(waits) == 1 and waits[0] > 0


@pytest.mark.parametrize("success", [True, False])
def test_read_wait_wakes_once_and_proves_exact_terminal(monkeypatch, success):
    from mlx2.runtime.paged_attention_pack import prepare_packed_token_read

    _, _, writer, backend = setup(monkeypatch)
    owner = PagedKVTokenOwner(writer, PROFILE, permit_candidate=True)
    backend.append_completed((owner,), projected(1), projected(1), (1,))
    use = prepare_packed_token_read((owner,), (1,), query_heads=4,
                                    permit_candidate=True)
    calls = []

    def submit(read, *args, **kwargs):
        read.mark_submitted()
        return mx.zeros((1, 4, 128), dtype=mx.float16)

    def poll(_):
        calls.append("poll")
        return ()

    def wait(_, timeout_seconds):
        calls.append("wait")
        assert timeout_seconds > 0
        return ((use.lease.epoch, success),)

    monkeypatch.setattr(module, "native_paged_attention_read_fp16", submit)
    monkeypatch.setattr(module, "poll_native_paged_read_events", poll)
    monkeypatch.setattr(module, "wait_native_paged_read_events", wait)
    if success:
        backend.read_completed(use, mx.zeros((1, 4, 128), dtype=mx.float16),
                               scale=128 ** -0.5)
        assert backend.terminal_successes == 1 and not backend._failed
    else:
        with pytest.raises(RuntimeError, match="read failed"):
            backend.read_completed(use, mx.zeros((1, 4, 128), dtype=mx.float16),
                                   scale=128 ** -0.5)
        assert backend.terminal_successes == 0 and backend._failed
    assert calls == ["poll", "wait"]
    assert use.state == "closed" and writer.ledger.pending_count == 0


@pytest.mark.parametrize("success", [True, False])
def test_timed_out_read_retains_lease_until_late_terminal(monkeypatch, success):
    from mlx2.runtime.paged_attention_pack import prepare_packed_token_read
    from mlx2.runtime.paged_native_retirement import reap_native_request_owner

    _, arena, writer, backend = setup(monkeypatch)
    owner = PagedKVTokenOwner(writer, PROFILE, permit_candidate=True)
    backend.append_completed((owner,), projected(1), projected(1), (1,))
    request = NativeAtomicRequestOwner("revision", (owner,), {},
                                       supported_planes=("kv",), enabled=True)
    use = prepare_packed_token_read((owner,), (1,), query_heads=4,
                                    permit_candidate=True)

    def submit(read, *args, **kwargs):
        read.mark_submitted()
        return mx.zeros((1, 4, 128), dtype=mx.float16)

    monkeypatch.setattr(module, "native_paged_attention_read_fp16", submit)
    monkeypatch.setattr(module, "poll_native_paged_read_events",
                        lambda _: tuple(arena.read_events.pop(0))
                        if arena.read_events else ())
    monkeypatch.setattr(module, "wait_native_paged_read_events", lambda *_: ())
    with pytest.raises(TimeoutError, match="read timed out"):
        backend.read_completed(use, mx.zeros((1, 4, 128), dtype=mx.float16),
                               scale=128 ** -0.5)
    assert backend._orphaned_reads[use.lease.epoch] is use
    assert use.state == "submitted" and writer.ledger.pending_count == 1
    request.close()
    poll = module.poll_native_paged_read_events

    def failed_poll(_backend):
        raise RuntimeError("callback queue unavailable")

    monkeypatch.setattr(module, "poll_native_paged_read_events", failed_poll)
    with pytest.raises(RuntimeError, match="callback queue unavailable"):
        reap_native_request_owner(request, writer, backend)
    assert not request.fully_retired and writer.ledger.pending_count == 1
    monkeypatch.setattr(module, "poll_native_paged_read_events", poll)
    reap_native_request_owner(request, writer, backend)
    assert not request.fully_retired and writer.ledger.pending_count == 1
    arena.close_after_terminal = lambda: setattr(arena, "closed", True)
    arena.read_events.append(((use.lease.epoch, success),))
    reap_native_request_owner(request, writer, backend)
    assert not backend._orphaned_reads
    assert use.state == "closed" and writer.ledger.pending_count == 0
    assert writer.poisoned is not success
    assert request.fully_retired
    assert bool(getattr(arena, "closed", False)) is not success


def test_cancellation_after_terminal_proof_releases_pin_and_quarantines(monkeypatch):
    from mlx2.runtime.paged_attention_pack import prepare_packed_token_read

    _, arena, writer, backend = setup(monkeypatch)
    owner = PagedKVTokenOwner(writer, PROFILE, permit_candidate=True)
    backend.append_completed((owner,), projected(1), projected(1), (1,))
    use = prepare_packed_token_read((owner,), (1,), query_heads=4,
                                    permit_candidate=True)
    install_read_callbacks(monkeypatch, arena)

    def cancel_after_callback(_read, _event):
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        backend.read_completed(use, mx.zeros((1, 4, 128), dtype=mx.float16),
                               scale=128 ** -0.5, terminal_proof=cancel_after_callback)
    assert use.state == "closed" and writer.ledger.pending_count == 0
    assert backend._failed and backend.terminal_successes == 0


def test_waited_wrong_epoch_keeps_physical_read_pin(monkeypatch):
    from mlx2.runtime.paged_attention_pack import prepare_packed_token_read

    _, _, writer, backend = setup(monkeypatch)
    owner = PagedKVTokenOwner(writer, PROFILE, permit_candidate=True)
    backend.append_completed((owner,), projected(1), projected(1), (1,))
    use = prepare_packed_token_read((owner,), (1,), query_heads=4,
                                    permit_candidate=True)

    def submit(read, *args, **kwargs):
        read.mark_submitted()
        return mx.zeros((1, 4, 128), dtype=mx.float16)

    monkeypatch.setattr(module, "native_paged_attention_read_fp16", submit)
    monkeypatch.setattr(module, "poll_native_paged_read_events", lambda _: ())
    monkeypatch.setattr(module, "wait_native_paged_read_events",
                        lambda _, timeout: ((use.lease.epoch + 1, True),))
    with pytest.raises(RuntimeError, match="unexpected or duplicate"):
        backend.read_completed(use, mx.zeros((1, 4, 128), dtype=mx.float16),
                               scale=128 ** -0.5)
    assert use.state == "submitted" and writer.ledger.pending_count == 1
    assert backend._failed


def atomic_fixture(monkeypatch):
    _, arena, writer, backend = setup(monkeypatch)
    public = tuple(PagedKVTokenOwner(writer, PROFILE, permit_candidate=True)
                   for _ in range(2))
    owner = NativeAtomicRequestOwner(
        "revision", public, {plane: () for plane in STATE_PLANES if plane != "kv"},
        enabled=True)
    branch = owner.begin(CandidateRequest(1, "revision", 3, STATE_PLANES))
    lane = PackedLane((1, 2, 3), branch.layers)
    return arena, writer, backend, owner, branch, lane


def install_read_callbacks(monkeypatch, arena, *, success=True, close_early=False):
    reads = []

    def submit(use, native, query, dependency, *, scale, permit_candidate):
        use.mark_submitted()
        reads.append(use)
        if close_early:
            use.complete_after_proof()
        arena.read_events.append((use.lease.epoch, success))
        return mx.zeros(query.shape, dtype=mx.float16)

    def poll(native):
        result, arena.read_events = arena.read_events, []
        return tuple(result)

    monkeypatch.setattr(module, "native_paged_attention_read_fp16", submit)
    monkeypatch.setattr(module, "poll_native_paged_read_events", poll)
    return reads


def test_atomic_candidate_hands_each_terminal_event_to_matching_layer(monkeypatch):
    arena, writer, backend, owner, branch, lane = atomic_fixture(monkeypatch)
    reads = install_read_callbacks(monkeypatch, arena)
    Qwen3PackedCandidate(FakeModel(), backend).forward(
        (lane,), permit_candidate=True, atomic_branch=branch)
    assert len(reads) == 2 and all(use.state == "closed" for use in reads)
    assert backend.read_submissions == backend.terminal_successes == 2
    assert branch._proved == {0, 1} and writer.ledger.pending_count == 0
    for plane in STATE_PLANES[1:]:
        branch.stage(plane, (plane,) * 3)
    branch.prepare(3).publish()
    assert owner.snapshot().offset == 3


def test_atomic_candidate_refuses_wrong_layer_before_writes(monkeypatch):
    arena, writer, backend, _, branch, lane = atomic_fixture(monkeypatch)
    swapped = PackedLane(lane.token_ids, tuple(reversed(lane.layers)))
    with pytest.raises(ValueError, match="matching native branch"):
        Qwen3PackedCandidate(FakeModel(), backend).forward(
            (swapped,), permit_candidate=True, atomic_branch=branch)
    assert not arena.dependencies and writer.ledger.pending_count == 0


@pytest.mark.parametrize("success,close_early", [(False, False), (True, True)])
def test_atomic_candidate_needs_matched_success_not_closed_state(
        monkeypatch, success, close_early):
    arena, _, backend, _, branch, lane = atomic_fixture(monkeypatch)
    reads = install_read_callbacks(monkeypatch, arena, success=success,
                                   close_early=close_early)
    with pytest.raises(NativeAtomicError, match="failed|does not cover"):
        Qwen3PackedCandidate(FakeModel(), backend).forward(
            (lane,), permit_candidate=True, atomic_branch=branch)
    assert branch._proved == set() and backend._failed
    assert backend.read_submissions == 1 and backend.terminal_successes == 0
    assert len(reads) == 1 and reads[0].state == "closed"


@pytest.mark.parametrize("events", ["wrong_epoch", "duplicate"])
def test_atomic_candidate_rejects_unmatched_or_duplicate_callback(monkeypatch, events):
    arena, writer, backend, _, branch, lane = atomic_fixture(monkeypatch)
    reads = install_read_callbacks(monkeypatch, arena)
    original_poll = module.poll_native_paged_read_events

    def invalid_poll(native):
        callbacks = original_poll(native)
        if not callbacks:
            return callbacks
        if events == "wrong_epoch":
            return ((callbacks[0][0] + 1, True),)
        return callbacks + callbacks

    monkeypatch.setattr(module, "poll_native_paged_read_events", invalid_poll)
    with pytest.raises(RuntimeError, match="unexpected or duplicate"):
        Qwen3PackedCandidate(FakeModel(), backend).forward(
            (lane,), permit_candidate=True, atomic_branch=branch)
    assert branch._proved == set() and backend._failed
    assert backend.read_submissions == 1 and backend.terminal_successes == 0
    assert len(reads) == 1 and reads[0].state == "submitted"
    assert writer.ledger.pending_count == 1


def test_rejected_atomic_layer_proof_releases_matching_terminal_read(monkeypatch):
    from mlx2.runtime.paged_attention_pack import prepare_packed_token_read

    arena, writer, backend, _, branch, _ = atomic_fixture(monkeypatch)
    backend.append_completed((branch.layers[0],), projected(3), projected(3), (3,))
    reads = install_read_callbacks(monkeypatch, arena)
    use = prepare_packed_token_read((branch.layers[0],), (3,), query_heads=4,
                                    permit_candidate=True)
    with pytest.raises(NativeAtomicError, match="does not cover"):
        backend.read_completed(
            use, mx.zeros((3, 4, 128), dtype=mx.float16), scale=128 ** -0.5,
            terminal_proof=lambda read, event: branch.prove_layer_read(1, read, event))
    assert reads == [use] and use.state == "closed"
    assert branch._proved == set() and backend._failed
    assert writer.ledger.pending_count == 0

"""CPU checks for planned prompt growth of append-only K/V buffers.

A long chunked prefill used to grow ``KVCache`` / ``BatchKVCache`` (and the
Hy4-flash append-only planes) by one 256-row quantum past each chunk, so every
chunk copied the whole buffer.  Inside a prompt-chunk scope the growth now
covers the lane's remaining planned prompt.  These tests pin that:

* the number of reallocations over a long prefill is constant per planned
  span (here 1, or one per capture checkpoint) instead of one per chunk;
* every read, the state, trim/rollback, batch merge/extract/filter/extend,
  left/right padding and APCv2 freeze/restore are bitwise identical to the
  stepped policy (the same appends run outside any scope);
* the capacity at the prompt boundary equals the stepped policy's for an
  aligned schedule and otherwise stays within ``fill + step - 1`` (the
  envelope every KV projection charges), so projections, APCv2 snapshots and
  persisted state do not grow.
"""

from collections import deque

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.cow_cache import freeze_prompt_cache
from mlx2.runtime.generate import BatchGenerator, PromptProcessingBatch
from mlx2.runtime.models import hy_v4_flash as H
from mlx2.runtime.models.cache import BatchKVCache, KVCache
from mlx2.runtime.prefill_lookahead import (
    allocation_rows_after,
    planned_prompt_rows,
    planned_rows_for,
    prefill_lookahead,
)

HEADS, DIM = 2, 8


def _rows(batch, rows, seed):
    rng = np.random.default_rng(seed)
    keys = mx.array(rng.standard_normal((batch, HEADS, rows, DIM)).astype(np.float32))
    values = mx.array(rng.standard_normal((batch, HEADS, rows, DIM)).astype(np.float32))
    return keys, values


def _equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(a, b).item())


class _Allocations:
    """Count fresh K buffers (growth rebinds ``keys``; in-place writes do not)."""

    def __init__(self, cache, name="keys"):
        self.cache, self.name, self.count, self._last = cache, name, 0, None

    def observe(self):
        value = getattr(self.cache, self.name)
        if value is not self._last:
            self.count += 1
            self._last = value


def _append(cache, keys, values, *, width=None, rows_after=None, planned=None):
    if width is None:
        return cache.update_and_fetch(keys, values)
    with prefill_lookahead(rows_after, step=width, width=width, planned_after=planned):
        return cache.update_and_fetch(keys, values)


def _attend(keys, values, seed):
    q = _rows(keys.shape[0], 3, seed)[0]
    return mx.fast.scaled_dot_product_attention(q, keys, values, scale=0.5, mask=None)


def _sliced_prefill(new, old, total, chunk, *, batch=1, base_seed=0, call_rows=None):
    """One chunk per scheduler call (as serving slices), planned remainder."""
    new_alloc, old_alloc = _Allocations(new), _Allocations(old)
    done = 0
    while done < total:
        n = min(chunk, total - done)
        k, v = _rows(batch, n, base_seed + done)
        remaining = total - done - n
        got = _append(new, k, v, width=n, rows_after=[0] * batch,
                      planned=[remaining if call_rows is None else min(remaining, call_rows(done + n))] * batch)
        want = _append(old, k, v)
        new_alloc.observe(), old_alloc.observe()
        assert _equal(got[0], want[0]) and _equal(got[1], want[1])
        assert _equal(_attend(*got, done), _attend(*want, done))
        done += n
    return new_alloc.count, old_alloc.count


# -- the scope ---------------------------------------------------------------


def test_allocation_hint_needs_matching_width_and_lanes():
    assert allocation_rows_after(512, 1) == 0
    with prefill_lookahead([100], step=512, width=512, planned_after=[1000]):
        assert allocation_rows_after(512, 1) == 1100
        assert allocation_rows_after(511, 1) == 0  # another cache's append shape
        assert allocation_rows_after(512, 2) == 0  # another lane count
    with prefill_lookahead([100, 7], step=512, width=512, planned_after=[1000, 0]):
        # A batch shares one capacity: multi-lane scopes plan nothing.
        assert allocation_rows_after(512, 2) == 0
    with prefill_lookahead([100], step=512):  # a scope without a width plans nothing
        assert allocation_rows_after(512, 1) == 0
    with planned_prompt_rows([5, 6]):
        assert planned_rows_for(2) == (5, 6)
        assert planned_rows_for(1) is None
    assert planned_rows_for(2) is None


# -- KVCache -------------------------------------------------------------------


@pytest.mark.parametrize("total,chunk", [(32768, 512), (20000, 2048), (5000, 256), (300, 512)])
def test_kv_cache_sliced_prefill_allocates_once_and_is_bitwise(total, chunk):
    new, old = KVCache(), KVCache()
    new_count, old_count = _sliced_prefill(new, old, total, chunk)
    assert new_count == 1
    assert old_count == -(-total // chunk)
    # The prompt boundary keeps the stepped policy's capacity: projections,
    # APCv2 snapshots and persisted state are unchanged.
    assert new.keys.shape == old.keys.shape and new.offset == old.offset == total
    assert new.nbytes == old.nbytes
    # Decode keeps 256-row growth and stays bitwise.
    for step in range(600):
        k, v = _rows(1, 1, 10_000 + step)
        got, want = new.update_and_fetch(k, v), old.update_and_fetch(k, v)
        assert _equal(got[0], want[0]) and _equal(got[1], want[1])
    assert new.keys.shape == old.keys.shape


def test_reallocations_are_constant_per_planned_span():
    counts = []
    for total in (4096, 16384, 65536):
        new, old = KVCache(), KVCache()
        counts.append(_sliced_prefill(new, old, total, 1024)[0])
    assert counts == [1, 1, 1]
    assert max(counts) <= np.log2(65536)


def test_within_call_lookahead_alone_also_allocates_once():
    new, old = KVCache(), KVCache()
    alloc = _Allocations(new)
    total, chunk = 10_000, 1024
    for start in range(0, total, chunk):
        n = min(chunk, total - start)
        k, v = _rows(1, n, start)
        got = _append(new, k, v, width=n, rows_after=[total - start - n])
        want = _append(old, k, v)
        alloc.observe()
        assert _equal(got[0], want[0]) and _equal(got[1], want[1])
    assert alloc.count == 1 and new.keys.shape == old.keys.shape


@pytest.mark.parametrize("chunk", [1000, 1024, 777])
def test_unaligned_resume_stays_inside_the_projected_envelope(chunk):
    new, old = KVCache(), KVCache()
    k, v = _rows(1, 1000, 1)
    new.update_and_fetch(k, v), old.update_and_fetch(k, v)
    new.trim(13), old.trim(13)  # an unaligned fill below capacity
    _sliced_prefill(new, old, 7000, chunk, base_seed=50)
    assert new.offset == old.offset == 7987
    # Projections charge ``units + step - 1`` (or ``(ceil(n/256) + 1) * 256``).
    assert new.offset <= new.keys.shape[2] <= new.offset + KVCache.step - 1
    assert new.keys.shape[2] <= old.keys.shape[2] + KVCache.step - 1
    assert _equal(new.keys_and_values()[0], old.keys_and_values()[0])


def test_reservation_is_still_honoured():
    new, old = KVCache().reserve(8192), KVCache().reserve(8192)
    _sliced_prefill(new, old, 3000, 512)
    assert new.keys.shape == old.keys.shape == (1, HEADS, 8192, DIM)


def test_trim_rollback_and_state_round_trip_are_bitwise():
    new, old = KVCache(), KVCache()
    _sliced_prefill(new, old, 4000, 1000)
    assert new.trim(321) == old.trim(321) == 321
    for step in range(5):
        k, v = _rows(1, 7, 900 + step)
        got, want = new.update_and_fetch(k, v), old.update_and_fetch(k, v)
        assert _equal(got[0], want[0]) and _equal(got[1], want[1])
    restored = [KVCache.from_state(c.state, c.meta_state) for c in (new, old)]
    assert restored[0].offset == restored[1].offset
    for a, b in zip(restored[0].keys_and_values(), restored[1].keys_and_values()):
        assert _equal(a, b)
    clones = [c.committed_descriptor_clone() for c in (new, old)]
    assert clones[0].offset == clones[1].offset
    assert _equal(clones[0].keys_and_values()[0], clones[1].keys_and_values()[0])


def test_apcv2_freeze_restore_round_trip_is_bitwise_and_not_larger():
    new, old = KVCache(), KVCache()
    _sliced_prefill(new, old, 6000, 2048)
    frozen = [
        freeze_prompt_cache([c], key=f"k{i}", tokens=range(c.offset), cache_type="kv")[0]
        for i, c in enumerate((new, old))
    ]
    branches = [f.branch() for f in frozen]
    assert branches[0][0].nbytes == branches[1][0].nbytes  # stored snapshot does not grow
    for a, b in zip(branches[0][0].keys_and_values(), branches[1][0].keys_and_values()):
        assert _equal(a, b)
    # Appending to the restored branch stays exact and leaves the source intact.
    k, v = _rows(1, 9, 77)
    got = branches[0][0].update_and_fetch(k, v)
    want = branches[1][0].update_and_fetch(k, v)
    assert _equal(got[0], want[0])
    assert _equal(new.keys_and_values()[0], old.keys_and_values()[0])
    for b in branches:
        b.close()


def test_subclasses_keep_their_own_growth():
    class Ledgered(KVCache):
        pass

    cache = Ledgered()
    k, v = _rows(1, 512, 3)
    _append(cache, k, v, width=512, rows_after=[0], planned=[50_000])
    assert cache.keys.shape[2] == 512


# -- BatchKVCache ----------------------------------------------------------------


def _batch_pair(left_padding):
    return BatchKVCache(list(left_padding)), BatchKVCache(list(left_padding))


def test_batch_cache_single_lane_with_left_padding_allocates_once():
    new, old = _batch_pair([37])
    new_count, old_count = _sliced_prefill(new, old, 9000, 1024)
    assert new_count == 1 and old_count == 9
    assert new.keys.shape == old.keys.shape
    assert _equal(new.extract(0).keys, old.extract(0).keys)


def test_batch_cache_multi_lane_keeps_stepped_growth_and_is_bitwise():
    new, old = _batch_pair([0, 37, 300])
    new_count, old_count = _sliced_prefill(new, old, 9000, 1024, batch=3)
    assert new_count == old_count == 9
    assert new.keys.shape == old.keys.shape
    for idx in range(3):
        a, b = new.extract(idx), old.extract(idx)
        assert a.offset == b.offset and _equal(a.keys, b.keys) and _equal(a.values, b.values)
    new.filter([2, 0]), old.filter([2, 0])
    assert _equal(new.keys_and_values()[0], old.keys_and_values()[0])
    for kind in ("trim", "decode"):
        if kind == "trim":
            assert new.trim(100) == old.trim(100)
        else:
            k, v = _rows(2, 1, 5)
            got, want = new.update_and_fetch(k, v), old.update_and_fetch(k, v)
            assert _equal(got[0], want[0]) and _equal(got[1], want[1])
    assert _equal(new.offset, old.offset) and _equal(new.left_padding, old.left_padding)


def test_batch_cache_merge_extend_and_right_padding_are_bitwise():
    rows = []
    for seed, length in ((1, 700), (2, 1500)):
        k, v = _rows(1, length, seed)
        rows.append((k, v))
    pair = []
    for _ in range(2):
        caches = []
        for k, v in rows:
            c = KVCache()
            c.update_and_fetch(k, v)
            caches.append(c)
        pair.append(BatchKVCache.merge(caches))
    new, old = pair
    # A ragged slice: lane lengths 1800 and 1100 right-padded to 1800.
    lengths, padding = [1800, 1100], [0, 700]
    for c in (new, old):
        c.prepare(lengths=lengths, right_padding=padding)
    k, v = _rows(2, 1800, 9)
    got = _append(new, k, v, width=1800, rows_after=[0, 0], planned=[4000, 4000])
    want = _append(old, k, v)
    assert _equal(got[0], want[0])
    new.finalize(), old.finalize()
    assert _equal(new.keys_and_values()[0], old.keys_and_values()[0])
    assert _equal(new.left_padding, old.left_padding) and _equal(new.offset, old.offset)
    for idx in range(2):
        a, b = new.extract(idx), old.extract(idx)
        assert _equal(a.keys, b.keys) and _equal(a.values, b.values)
    others = []
    for _ in range(2):
        c = KVCache()
        c.update_and_fetch(*_rows(1, 222, 11))
        others.append(BatchKVCache.merge([c]))
    new.extend(others[0]), old.extend(others[1])
    k, v = _rows(3, 1, 12)
    got, want = new.update_and_fetch(k, v), old.update_and_fetch(k, v)
    assert _equal(got[0], want[0]) and _equal(got[1], want[1])
    for idx in range(3):
        assert _equal(new.extract(idx).keys, old.extract(idx).keys)


def test_lane_split_off_a_planned_prompt_batch_does_not_carry_its_plan():
    """A joiner padded to a planned lane's capacity leaves with its own envelope."""
    lanes = []
    for _ in range(2):
        planned = BatchKVCache([0])
        for start in range(0, 4096, 1024):
            k, v = _rows(1, 1024, start)
            _append(planned, k, v, width=1024, rows_after=[0], planned=[60_000 - start - 1024])
        lanes.append(planned)
    new, old = lanes[0], BatchKVCache([0])
    for start in range(0, 4096, 1024):
        old.update_and_fetch(*_rows(1, 1024, start))
    assert new.keys.shape[2] == 60_160 and old.keys.shape[2] == 4096
    joiners = []
    for _ in range(2):
        c = KVCache()
        c.update_and_fetch(*_rows(1, 300, 99))
        joiners.append(BatchKVCache.merge([c]))
    new.extend(joiners[0]), old.extend(joiners[1])
    # While the two lanes share the batch, the joiner is not padded to the
    # planned lane's capacity (integration review 2026-11-22).
    assert new.keys.shape[2] <= -(-new._idx // 256) * 256 + 256
    assert _equal(new.keys_and_values()[0], old.keys_and_values()[0])
    new.filter([1]), old.filter([1])
    assert new.keys.shape[2] <= -(-new._idx // 256) * 256 + 256
    assert _equal(new.keys_and_values()[0], old.keys_and_values()[0])
    k, v = _rows(1, 1, 5)
    got, want = new.update_and_fetch(k, v), old.update_and_fetch(k, v)
    assert _equal(got[0], want[0]) and _equal(got[1], want[1])
    assert _equal(new.extract(0).keys, old.extract(0).keys)


# -- the prompt loop and the scheduler plan ---------------------------------------


class _AppendModel:
    """Appends one deterministic K/V row per token to every cache."""

    def __call__(self, tokens, *, cache):
        b, n = tokens.shape
        base = tokens.astype(mx.float32)[:, None, :, None]
        keys = mx.broadcast_to(base, (b, HEADS, n, DIM)) * 0.5
        for c in cache:
            c.update_and_fetch(keys, keys + 1)
        return mx.zeros((b, n, 4), mx.float32)


def _prompt_batch(cache):
    batch = object.__new__(PromptProcessingBatch)
    batch.uids = [5]
    batch.tokens = [[]]
    batch.prompt_cache = [cache]
    batch.prefill_step_size = 512
    batch.model = _AppendModel()
    batch.decode_active = False
    batch.prompt_trim_rollback_tokens = 0
    return batch


def test_prompt_loop_scopes_width_and_scheduler_plan():
    prompt = list(range(1, 6001))
    new, old = BatchKVCache([0]), BatchKVCache([0])
    seen = _Allocations(new)
    for start in range(0, len(prompt), 1000):  # one scheduler slice per call
        piece = prompt[start:start + 1000]
        with planned_prompt_rows([len(prompt) - start - len(piece)]):
            _prompt_batch(new).prompt([piece])
        _prompt_batch(old).prompt([piece])
        seen.observe()
    assert seen.count == 1
    assert new._idx == old._idx == 6000
    assert new.keys.shape[2] <= new._idx + BatchKVCache.step - 1  # projected envelope
    assert _equal(new.keys_and_values()[0], old.keys_and_values()[0])
    assert _equal(new.keys_and_values()[1], old.keys_and_values()[1])


def _generator(remaining, covered, captures):
    gen = object.__new__(BatchGenerator)
    gen._currently_processing = [[[list(range(remaining))], covered, covered + remaining, None, 0]]
    gen._prompt_batch = type("PB", (), {"uids": [9]})()
    gen._interior_checkpoint_positions = {9: deque(captures)}
    return gen


def test_scheduler_plan_stops_at_capture_checkpoints():
    # 5000 prompt rows left (one is the decode token), no capture: plan them all.
    assert _generator(5001, 2048, [])._planned_prompt_rows_after() == [5000]
    # A capture at 4096 bounds the plan so the captured boundary pins no more
    # than the stepped growth would have.
    assert _generator(5001, 2048, [4096])._planned_prompt_rows_after() == [2048]
    # A capture at the end of this slice plans nothing past it.
    assert _generator(5001, 2048, [2048, 6000])._planned_prompt_rows_after() == [0]


def test_capture_bounded_plan_reallocates_once_per_span():
    new, old = KVCache(), KVCache()
    captures = (8192, 16384, 24576)

    def call_rows(done):
        nxt = next((c for c in captures if c >= done), None)
        return 10**9 if nxt is None else nxt - done

    new_count, old_count = _sliced_prefill(new, old, 32768, 1024, call_rows=call_rows)
    assert new_count == len(captures) + 1 and old_count == 32
    assert new.keys.shape == old.keys.shape


# -- Hy4-flash append-only planes ---------------------------------------------------


class _Indexer:
    def pool(self, keys, have, positions=None):
        b, n, d = keys.shape
        return keys.reshape(b, n // 4, 4, d).astype(mx.float32).mean(axis=2)


def _hy4_rows(n, seed, width):
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal((1, n, width)).astype(np.float32))


def _hy4_step(cache, n, seed, scope):
    keys, token_keys = _hy4_rows(n, seed, 12), _hy4_rows(n, seed + 1, 6)
    if scope is None:
        kv, first = cache.append(keys)
        gk = cache.append_index_keys(token_keys, _Indexer())
    else:
        with prefill_lookahead([0], step=n, width=n, planned_after=[scope]):
            kv, first = cache.append(keys)
            gk = cache.append_index_keys(token_keys, _Indexer())
    return kv, first, gk


@pytest.mark.parametrize("total,chunk", [(20000, 2048), (4097, 512)])
def test_hy4_planes_allocate_once_and_are_bitwise(total, chunk):
    new, old = H.HY4FlashLayerCache(H.INDEXED, ratio=4), H.HY4FlashLayerCache(H.INDEXED, ratio=4)
    counters = {p: (_Allocations(new, p + "_data"), _Allocations(old, p + "_data")) for p in ("kv", "tk", "gk")}
    done = 0
    while done < total:
        n = min(chunk, total - done)
        got = _hy4_step(new, n, done, total - done - n)
        want = _hy4_step(old, n, done, None)
        for c in counters.values():
            c[0].observe(), c[1].observe()
        assert got[1] == want[1]
        assert _equal(got[0], want[0]) and _equal(got[2], want[2])
        done += n
    for plane, (a, b) in counters.items():
        assert a.count == 1, plane
        assert b.count == -(-total // chunk) or plane == "gk"
        assert getattr(new, plane + "_data").shape == getattr(old, plane + "_data").shape, plane
    assert new.nbytes == old.nbytes
    for a, b in zip(new.state, old.state):
        assert _equal(a, b)
    assert new.trim(37) == old.trim(37)
    got, want = _hy4_step(new, 1, 5, None), _hy4_step(old, 1, 5, None)
    assert _equal(got[0], want[0]) and _equal(got[2], want[2])
    restored = [H.HY4FlashLayerCache.from_state(c.state, c.meta_state) for c in (new, old)]
    for a, b in zip(restored[0].state, restored[1].state):
        assert _equal(a, b)


def test_hy4_boundary_extracted_mid_prompt_pins_no_planned_capacity():
    new = H.HY4FlashBatchLayerCache([H.HY4FlashLayerCache(H.INDEXED, ratio=4)])
    old = H.HY4FlashBatchLayerCache([H.HY4FlashLayerCache(H.INDEXED, ratio=4)])
    total, chunk = 32768, 2048
    for done in range(0, total, chunk):
        _hy4_step(new.rows[0], chunk, done, total - done - chunk)
        _hy4_step(old.rows[0], chunk, done, None)
        if done + chunk == 4096:  # a boundary added in flight
            snaps = new.extract(0), old.extract(0)
            for plane in ("kv", "tk", "gk"):
                a, b = getattr(snaps[0], plane + "_data"), getattr(snaps[1], plane + "_data")
                assert a.shape[1] <= b.shape[1] + H._STEP, plane
                assert _equal(H._buf_view(snaps[0], plane), H._buf_view(snaps[1], plane)), plane
            assert new.rows[0].kv_data.shape[1] == total  # live keeps its plan
    for plane in ("kv", "tk", "gk"):
        assert _equal(H._buf_view(new.rows[0], plane), H._buf_view(old.rows[0], plane))
        # The snapshot was copied, not aliased: later appends left it intact.
        assert _equal(H._buf_view(snaps[0], plane), H._buf_view(snaps[1], plane))
    # At the prompt boundary capacity is stepped, so extract aliases again.
    assert new.extract(0).kv_data.shape == old.extract(0).kv_data.shape


def test_hy4_sliding_ring_and_batches_keep_stepped_growth():
    ring = H.HY4FlashLayerCache(H.SLIDING, window=64)
    with prefill_lookahead([0], step=512, width=512, planned_after=[50_000]):
        ring.append(_hy4_rows(512, 1, 12))
    assert ring.ring_data.shape[1] == 512
    assert H._prompt_reserve(_hy4_rows(512, 1, 12)) == 0
    with prefill_lookahead([0, 0], step=512, width=512, planned_after=[9, 9]):
        assert H._prompt_reserve(_hy4_rows(512, 1, 12)) == 0  # B>1 rows are private

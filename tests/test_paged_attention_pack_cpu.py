"""Packed token-page reader contracts use no MLX import or Metal device."""

import numpy as np
import pytest

from mlx2.runtime.paged_attention_metal import paged_attention_cpu_reference
from mlx2.runtime.paged_attention_native import _validate_pinned_read_handles
from mlx2.runtime.paged_attention_pack import prepare_packed_token_read
from mlx2.runtime.paged_kv_pool import PagedKVPool
from mlx2.runtime.paged_kv_token import PagedKVTokenOwner, TokenKVProfile
from mlx2.runtime.paged_kv_write import PagedKVWriteOwner
from mlx2.runtime.paged_pack_diagnostics import describe_packed_read

PROFILE = TokenKVProfile(2, 128, "float16")


class FakeByteArena:
    def __init__(self, capacity):
        self.plane_bytes = capacity * PROFILE.page_bytes
        self.keys = bytearray(self.plane_bytes)
        self.values = bytearray(self.plane_bytes)
        self.events = []

    def validate_sources(self, key, value):
        return (type(key) is np.ndarray and type(value) is np.ndarray and
                key.dtype == value.dtype == np.uint8 and key.ndim == value.ndim == 1)

    def write(self, key, value, offset, byte_count, epoch):
        self.keys[offset:offset + byte_count] = key.tobytes()
        self.values[offset:offset + byte_count] = value.tobytes()
        self.events.append((epoch, True))
        return object()

    def poll_completions(self):
        events, self.events = self.events, []
        return events


def fixture():
    rng = np.random.default_rng(3307)
    pool = PagedKVPool(5)
    backend = FakeByteArena(pool.capacity)
    writer = PagedKVWriteOwner(pool, backend, page_bytes=PROFILE.page_bytes,
                               permit_candidate=True)
    owners, logical = [], []
    for tokens in (65, 63):
        owner = PagedKVTokenOwner(writer, PROFILE, permit_candidate=True)
        k = rng.normal(0, 0.25, (tokens, PROFILE.kv_heads, PROFILE.head_dim)).astype(np.float16)
        v = rng.normal(0, 0.25, k.shape).astype(np.float16)
        spans = owner.planned_spans(tokens)
        chunks = tuple(tuple(np.frombuffer(
            array[span.source_token_offset:span.source_token_offset + span.token_count,
                  span.kv_head].tobytes(), dtype=np.uint8).copy() for span in spans)
            for array in (k, v))
        owner.append(*chunks, token_count=tokens)
        assert owner.offset == 0
        assert owner.poll_completions() and owner.offset == tokens
        owners.append(owner)
        logical.append((k, v))
    return tuple(owners), tuple(logical), backend, pool


def dense_reference(q, logical, row_counts, masks):
    output = np.empty_like(q)
    row_begin = 0
    for (keys, values), rows, mask in zip(logical, row_counts, masks):
        for local in range(rows):
            row = row_begin + local
            upper = len(keys) - rows + local + 1
            lower = 0 if mask is None else max(0, upper - mask)
            for head in range(q.shape[1]):
                kv_head = head // (q.shape[1] // PROFILE.kv_heads)
                scores = keys[lower:upper, kv_head].astype(np.float64) @ q[row, head].astype(np.float64)
                scores *= PROFILE.head_dim ** -0.5
                scores -= scores.max()
                weights = np.exp(scores)
                output[row, head] = (weights / weights.sum()) @ values[lower:upper, kv_head].astype(np.float64)
        row_begin += rows
    return output


def test_packed_unequal_lanes_match_independent_dense_oracle_and_pin():
    owners, logical, backend, pool = fixture()
    with pytest.raises(RuntimeError, match="explicit candidate"):
        prepare_packed_token_read(owners, (3, 1), query_heads=4)
    use = prepare_packed_token_read(owners, (3, 1), query_heads=4,
                                    masks=(("causal", None), ("sliding", 17)),
                                    permit_candidate=True)
    assert [span.row_count for span in use.plan.spans] == [3, 1]
    assert use.metadata.row_span == (0, 0, 0, 1)
    assert len(use.plan.page_table) == 3
    q = np.random.default_rng(4).normal(0, 0.25, (4, 4, 128)).astype(np.float16)
    k = np.frombuffer(backend.keys, dtype=np.float16).reshape(5, 2, 64, 128)
    v = np.frombuffer(backend.values, dtype=np.float16).reshape(5, 2, 64, 128)
    expected = dense_reference(q, logical, (3, 1), (None, 17))
    np.testing.assert_array_equal(paged_attention_cpu_reference(use.plan, q, k, v), expected)
    pinned = use.plan.page_table[0]
    owners[0].close()
    assert pool.references(pinned) == 1 and pool.free_count == 2
    use.mark_submitted()
    with pytest.raises(ValueError, match="terminal proof"):
        use.abort_before_submit()
    use.complete_after_proof()
    assert pool.free_count == 4
    owners[1].close()
    assert pool.free_count == 5


def test_invalid_pack_releases_prepared_lease_and_does_not_change_owner():
    owners, _, _, pool = fixture()
    before = [pool.references(h) for owner in owners for h in owner.sequence.handles]
    with pytest.raises(ValueError, match="unsupported mask"):
        prepare_packed_token_read(owners, (1, 1), query_heads=4,
                                  masks=(("causal", None), ("sink", None)),
                                  permit_candidate=True)
    assert [pool.references(h) for owner in owners for h in owner.sequence.handles] == before
    assert owners[0].offset == 65 and owners[1].offset == 63
    with pytest.raises(ValueError, match="unique"):
        prepare_packed_token_read((owners[0], owners[0]), (1, 1), query_heads=4,
                                  permit_candidate=True)
    with pytest.raises(ValueError, match="completed KV"):
        prepare_packed_token_read(owners, (66, 1), query_heads=4,
                                  permit_candidate=True)
    with pytest.raises(ValueError, match="share the native arena"):
        other, _, _, _ = fixture()
        prepare_packed_token_read((owners[0], other[0]), (1, 1), query_heads=4,
                                  permit_candidate=True)
    use = prepare_packed_token_read(owners, (1, 1), query_heads=4,
                                    permit_candidate=True)
    use.abort_before_submit()
    assert pool.free_count == 2


def test_pinned_generation_snapshot_and_row_work_counters(monkeypatch):
    owners, _, _, pool = fixture()
    # The plan must use the generations already validated by its lease,
    # without an O(arena capacity) scan for every model layer.
    def forbid_scan():
        raise AssertionError("full arena scan")

    monkeypatch.setattr(pool, "live_generations", forbid_scan)
    use = prepare_packed_token_read(owners, (3, 1), query_heads=4,
                                    permit_candidate=True, profile_host=True)
    assert use.metadata_build_ns is not None and use.metadata_build_ns >= 0
    work = describe_packed_read(use.plan)
    assert work.spans == 2 and work.rows == 4
    assert work.physical_threadgroups == 16
    # First lane sees 63, 64, 65 KV; second sees 63.
    assert work.serial_kv_steps == (63 + 64 + 65 + 63) * 4
    assert work.kv_64_tile_slots == (64 + 64 + 128 + 64) * 4
    assert work.page_table_entries == 3
    _validate_pinned_read_handles(use)
    use.abort_before_submit()
    for owner in owners:
        owner.close()
    with pytest.raises(ValueError, match="generation changed"):
        _validate_pinned_read_handles(use)


def test_work_counters_reject_unvalidated_plan():
    with pytest.raises(TypeError, match="validated"):
        describe_packed_read(object())

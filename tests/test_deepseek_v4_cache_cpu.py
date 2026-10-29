"""Synthetic exact state checks; no checkpoint or GPU execution."""

import numpy as np
import pytest

from mlx2.runtime.models.deepseek_v4_cache import (
    DeepSeekV4LayerCache, DeepSeekV4PoolCache,
)
from mlx2.runtime.cache_operations import cache_operations, CacheOperationsUnsupported


def oracle(kv, gate, ape, ratio, dim):
    """Independent scalar window oracle from the official overlapping formula."""
    outputs = []
    for end in range(ratio, kv.shape[1] + 1, ratio):
        row = []
        for d in range(dim):
            values, scores = [], []
            if ratio == 4 and end >= 2 * ratio:
                for j in range(ratio):
                    values.append(kv[0, end - 2 * ratio + j, d])
                    scores.append(gate[0, end - 2 * ratio + j, d] + ape[j, d])
            for j in range(ratio):
                c = d + dim if ratio == 4 else d
                values.append(kv[0, end - ratio + j, c])
                scores.append(gate[0, end - ratio + j, c] + ape[j, c])
            weights = np.exp(np.array(scores, dtype=np.float64) - max(scores))
            row.append(np.dot(values, weights) / weights.sum())
        outputs.append(row)
    return np.asarray(outputs, dtype=np.float32)[None]


def data(ratio, n, dim=3):
    rng = np.random.default_rng(429)
    width = dim * (2 if ratio == 4 else 1)
    return (rng.normal(size=(1, n, width)).astype(np.float32),
            rng.normal(size=(1, n, width)).astype(np.float32),
            rng.normal(size=(ratio, width)).astype(np.float32))


@pytest.mark.parametrize("ratio,n,chunks", [
    (4, 37, [37]), (4, 37, [1] * 37), (4, 37, [3, 2, 4, 7, 21]),
    (128, 260, [260]), (128, 260, [127, 1, 1, 126, 5]),
])
def test_pool_chunking_matches_independent_reference(ratio, n, chunks):
    kv, gate, ape = data(ratio, n)
    cache = DeepSeekV4PoolCache(ratio, 3, xp=np)
    offset, positions = 0, []
    for count in chunks:
        pooled, starts = cache.compress(kv[:, offset:offset+count],
                                        gate[:, offset:offset+count], ape, offset)
        cache.append_pooled(pooled)
        positions.extend(starts)
        offset += count
    np.testing.assert_allclose(cache.pooled, oracle(kv, gate, ape, ratio, 3), rtol=2e-6, atol=2e-7)
    assert positions == list(range(0, n - n % ratio, ratio))
    assert cache.remainder == n % ratio


def test_overlap_previous_window_is_not_lost_at_decode_boundary():
    kv = np.zeros((1, 8, 2), np.float32)
    kv[:, :4, 0] = 12
    gate = np.zeros_like(kv)
    cache = DeepSeekV4PoolCache(4, 1, xp=np)
    for start in (0, 4):
        p, _ = cache.compress(kv[:, start:start+4], gate[:, start:start+4], np.zeros((4, 2)), start)
        cache.append_pooled(p)
    np.testing.assert_array_equal(cache.pooled, [[[0], [6]]])


def advance(cache, count):
    offset = cache.offset
    keys = np.arange(offset, offset + count, dtype=np.float32).reshape(1, 1, count, 1)
    cache.update_local(keys)
    for pool in (cache.compressor, cache.indexer):
        if pool is None:
            continue
        width = pool.head_dim * (2 if pool.ratio == 4 else 1)
        kv = np.broadcast_to(keys.reshape(1, count, 1), (1, count, width)).copy()
        p, _ = pool.compress(kv, np.zeros_like(kv), np.zeros((pool.ratio, width)), offset)
        pool.append_pooled(p)


def layer(ratio=4):
    return DeepSeekV4LayerCache(ratio, xp=np, head_dim=1, index_head_dim=1,
                                identity="synthetic@fixture/source-revision")


def assert_tree_equal(left, right):
    if isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    elif left is None:
        assert right is None
    else:
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("ratio,start,verify,keep", [(0, 129, 5, 1), (4, 131, 7, 2), (128, 255, 9, 3)])
def test_full_snapshot_rejected_suffix_replay_matches_ordinary(ratio, start, verify, keep):
    candidate, reference = layer(ratio), layer(ratio)
    advance(candidate, start)
    advance(reference, start)
    snapshot = candidate.snapshot()
    advance(candidate, verify)
    candidate.restore(snapshot)
    advance(candidate, keep)
    advance(reference, keep)
    assert candidate.meta_state == reference.meta_state
    assert_tree_equal(candidate.state, reference.state)
    # Snapshot owns the local and pooling arrays even under hostile in-place writes.
    saved = snapshot.arrays[0].copy()
    candidate.keys[...] = -999
    np.testing.assert_array_equal(snapshot.arrays[0], saved)
    candidate.restore(snapshot)
    np.testing.assert_array_equal(candidate.keys, saved)


def test_local_band_preserves_all_prefill_queries_and_retains_only_window():
    c = layer(0)
    advance(c, 200)
    values = np.arange(200, 220, dtype=np.float32).reshape(1, 1, 20, 1)
    band, empty = c.update_and_fetch(values, np.zeros((1, 1, 20, 0)))
    np.testing.assert_array_equal(band.ravel(), np.arange(73, 220))
    np.testing.assert_array_equal(c.keys.ravel(), np.arange(92, 220))
    assert empty.shape == (1, 1, 147, 0)
    assert c.offset == 220


def test_masks_and_pending_pool_refuse_wrong_sequence():
    c = DeepSeekV4PoolCache(4, 1, xp=np)
    kv = np.ones((1, 8, 2), np.float32)
    p, _ = c.compress(kv, kv, np.zeros((4, 2)), 0)
    with pytest.raises(ValueError, match="not appended"):
        c.compress(kv[:, :1], kv[:, :1], np.zeros((4, 2)), 8)
    c.append_pooled(p)
    np.testing.assert_array_equal(c.make_mask(8, 0),
        [[False, False]]*3 + [[True, False]]*4 + [[True, True]])
    with pytest.raises(ValueError, match="exact prefix"):
        c.compress(kv[:, :1], kv[:, :1], np.zeros((4, 2)), 9)


def test_persistence_and_contract_identity_and_no_unsafe_trim():
    c = layer()
    advance(c, 133)
    restored = DeepSeekV4LayerCache.from_state(c.state, c.meta_state)
    assert_tree_equal(restored.state, c.state)
    assert restored.meta_state == c.meta_state
    operations = cache_operations(c, capability="cpu-synthetic-test")
    assert operations.exact_prefix and operations.restores_at_checkpoints
    assert operations.rollback is None
    assert not c.is_trimmable()
    with pytest.raises(NotImplementedError, match="snapshot"):
        c.trim(1)
    foreign = DeepSeekV4LayerCache(4, xp=np, head_dim=1, index_head_dim=1, identity="different")
    with pytest.raises(ValueError, match="identity"):
        foreign.restore(c.snapshot())
    assert foreign.offset == 0


def test_snapshot_refuses_partial_layer_update():
    c = layer()
    c.update_local(np.ones((1, 1, 1, 1), np.float32))
    with pytest.raises(CacheOperationsUnsupported, match="incomplete"):
        c.snapshot()


def test_singleton_batch_lifecycle_clones_and_refuses_width_or_padding():
    source = layer()
    advance(source, 133)
    merged = source.merge([source])
    assert_tree_equal(merged.state, source.state)
    extracted = merged.extract(0)
    advance(merged, 1)
    assert extracted.offset == source.offset == 133
    merged.prepare(left_padding=[0], right_padding=[0], lengths=[1])
    merged.finalize()
    merged.filter([0])
    for operation in (lambda: source.merge([source, extracted]),
                      lambda: source.extract(1), lambda: source.filter([0, 0]),
                      lambda: source.prepare(left_padding=[1]),
                      lambda: source.prepare(right_padding=[1]),
                      lambda: source.extend(extracted)):
        with pytest.raises(ValueError):
            operation()
    merged.filter([])
    assert merged.nbytes == 0
    with pytest.raises(ValueError, match="closed"):
        merged.update_local(np.ones((1, 1, 1, 1)))
    assert source.offset == 133


def test_apcv2_descriptor_branch_uses_cpu_arrays_and_isolates_pools():
    import mlx.core as mx
    from mlx2.runtime.apc_v2 import inspect_apc_capabilities
    from mlx2.runtime.cow_cache import freeze_prompt_cache, restore_prompt_cache

    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    frozen = branch = None
    try:
        c = DeepSeekV4LayerCache(4, xp=mx, head_dim=1, index_head_dim=1,
                                identity="synthetic@cpu/source-revision")
        c.update_and_fetch(mx.ones((1, 1, 5, 1)), mx.zeros((1, 1, 5, 0)))
        for pool in (c.compressor, c.indexer):
            raw, _ = pool.compress(mx.ones((1, 5, 2)), mx.zeros((1, 5, 2)),
                                   mx.zeros((4, 2)), 0)
            pool.append_pooled(raw)
        caps = inspect_apc_capabilities([c])
        assert caps.exact_prefix and not caps.arbitrary_branch
        frozen, _ = freeze_prompt_cache([c], key="synthetic",
            tokens=(1, 2, 3, 4, 5), cache_type="deepseek_v4")
        branch = restore_prompt_cache(frozen)
        assert branch[0].meta_state == c.meta_state
        branch[0].compressor.previous_kv = mx.zeros_like(branch[0].compressor.previous_kv)
        assert c.compressor.previous_kv.tolist() == [[[[1], [1], [1], [1]]]]
        assert branch[0].compressor.previous_kv.tolist() == [[[[0], [0], [0], [0]]]]
    finally:
        if branch is not None:
            branch.close()
        if frozen is not None:
            frozen.close()
        mx.set_default_device(previous_device)

def test_retained_windows_own_compact_storage_after_large_prefill():
    c = DeepSeekV4LayerCache(4, xp=np, head_dim=3, index_head_dim=3,
                            identity="synthetic@cpu/source-revision")
    incoming = np.ones((1, 1, 4097, 3), np.float32)
    band = c.update_local(incoming)
    assert band.shape[2] == 4097
    assert c.keys.flags.owndata
    assert not np.shares_memory(c.keys, incoming)
    for pool in (c.compressor, c.indexer):
        kv, gate, ape = data(4, 4097)
        pooled, _ = pool.compress(kv, gate, ape, 0)
        pool.append_pooled(pooled)
        for retained in (pool.buf_kv, pool.buf_gate,
                         pool.previous_kv, pool.previous_gate):
            assert retained.flags.owndata
            assert retained.size <= 4 * 2 * 3
            assert not np.shares_memory(retained, kv)
            assert not np.shares_memory(retained, gate)
    snap = c.snapshot()
    assert snap.arrays[0].flags.owndata
    assert not np.shares_memory(snap.arrays[0], c.keys)

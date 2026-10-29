"""Opt-in synthetic packed cache state, replay, and accounting checks."""
import numpy as np
import pytest
from mlx2.runtime.models.deepseek_v4_cache import DeepSeekV4LayerCache
from mlx2.runtime.models.deepseek_v4_packed_cache import DeepSeekV4PackedLayerCache
from mlx2.runtime.models.deepseek_v4_packed_kv import pack_fp8_kv, unpack_fp8_kv
from mlx2.adapters.deepseek_v4_packed_memory import DeepSeekV4PackedCacheBudget


def qat(x):
    return unpack_fp8_kv(pack_fp8_kv(x.astype(np.float32), xp=np), xp=np)


def advance(cache, n):
    rng = np.random.default_rng(400 + cache.offset)
    offset = cache.offset
    band = cache.update_local(qat(rng.normal(size=(1,1,n,512))))
    for pool in (cache.compressor, cache.indexer):
        if pool is None: continue
        d = pool.head_dim; width = d * (2 if pool.ratio == 4 else 1)
        values = rng.normal(size=(1,n,width)).astype(np.float32)
        raw, _ = pool.compress(values, np.zeros_like(values), np.zeros((pool.ratio,width),np.float32), offset)
        pool.append_pooled(qat(raw) if d == 512 else raw)
    return band


def equal(a, b):
    if a is None: assert b is None
    elif isinstance(a, tuple):
        assert len(a) == len(b)
        for x,y in zip(a,b): equal(x,y)
    else: np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('ratio',[0,4,128])
def test_packed_storage_replay_snapshot_and_branch_isolation(ratio):
    base = DeepSeekV4LayerCache(ratio,xp=np,identity='synthetic-fp32')
    packed = DeepSeekV4PackedLayerCache(ratio,xp=np,identity='synthetic-fp8')
    for n in (3,126,9):
        np.testing.assert_array_equal(advance(base,n),advance(packed,n))
        np.testing.assert_array_equal(base.keys,packed.keys)
        if ratio: np.testing.assert_array_equal(base.compressor.pooled,packed.compressor.pooled)
    assert packed.nbytes < base.nbytes
    snap = packed.snapshot()
    branch = packed.extract(0)
    baseline_state = packed.snapshot()
    advance(branch,5)
    equal(packed.state,baseline_state.arrays)
    branch.restore(snap)
    equal(branch.state,packed.state)
    assert branch.meta_state == packed.meta_state
    assert branch.state[0][0].dtype == np.uint8
    branch.state[0][0][...] = 0
    assert np.any(packed.state[0][0] != 0)
    with pytest.raises(ValueError,match='revision'):
        packed.restore(base.snapshot())
    packed.close()
    assert packed.nbytes == 0


def test_packed_budget_charges_actual_storage_and_running_unpack():
    b = DeepSeekV4PackedCacheBudget((0,4,128))
    assert b.local_live_bytes == 3*128*732
    assert b.compression_fixed_bytes > 0
    resident = b.project_resident(4096)
    snapshot = b.snapshot_bytes(4096)
    unpack = b.unpack_workspace_bytes(4096)
    assert unpack > 3*128*512*4
    assert b.project(4096) == resident+snapshot+unpack
    assert b.project_pool(4096,resident_lanes=2,running_lanes=1) == 2*resident+snapshot+unpack
    for ratio in (0,4,128):
        c=DeepSeekV4PackedLayerCache(ratio,xp=np,identity='synthetic-fp8')
        advance(c,138)
        budget=DeepSeekV4PackedCacheBudget((ratio,))
        assert budget.project_resident(138) >= c.nbytes


def test_packed_apcv2_cpu_roundtrip():
    import mlx.core as mx
    from mlx2.runtime.cow_cache import freeze_prompt_cache, restore_prompt_cache
    old=mx.default_device(); mx.set_default_device(mx.cpu)
    frozen=branch=None
    try:
        c=DeepSeekV4PackedLayerCache(0,xp=mx,identity='synthetic-fp8')
        c.update_local(mx.ones((1,1,5,512)))
        frozen,_=freeze_prompt_cache([c],key='synthetic',tokens=(1,2,3,4,5),cache_type='deepseek_v4')
        branch=restore_prompt_cache(frozen)
        assert branch[0].meta_state == c.meta_state
        assert branch[0].state[0][0].dtype == mx.uint8
        assert np.array_equal(np.array(branch[0].keys),np.array(c.keys))
        branch[0].update_local(mx.zeros((1,1,1,512)))
        assert c.offset == 5 and branch[0].offset == 6
    finally:
        if branch is not None: branch.close()
        if frozen is not None: frozen.close()
        mx.set_default_device(old)


@pytest.mark.parametrize('window', [1, 128])
def test_incremental_retention_packs_only_new_rows_and_owns_planes(monkeypatch, window):
    import mlx2.runtime.models.deepseek_v4_packed_cache as module
    original = module.pack_fp8_kv
    packed_rows = []

    def tracked(value, *, xp):
        packed_rows.append(value.shape[2])
        return original(value, xp=xp)

    monkeypatch.setattr(module, 'pack_fp8_kv', tracked)
    cache = DeepSeekV4PackedLayerCache(0, xp=np, window_size=window, identity='synthetic-incremental')
    base = DeepSeekV4LayerCache(0, xp=np, window_size=window, identity='synthetic-incremental')
    rng = np.random.default_rng(921)
    for n in (3, 1, 126, 1, 4097, 1):
        incoming = qat(rng.normal(size=(1, 1, n, 512)))
        previous = cache.state[0]
        np.testing.assert_array_equal(cache.update_local(incoming), base.update_local(incoming))
        assert packed_rows[-1] == min(n, window)
        np.testing.assert_array_equal(cache.keys, base.keys)
        for plane in cache.state[0]:
            assert not np.shares_memory(plane, incoming)
            assert plane.flags.owndata
            if previous is not None:
                assert all(not np.shares_memory(plane, old) for old in previous)
        if previous is not None and n < window:
            count = min(previous[0].shape[2], window - n)
            for plane, old in zip(cache.state[0], previous):
                np.testing.assert_array_equal(plane[:, :, :count], old[:, :, -count:])
    assert len(packed_rows) == 6


def test_incremental_retention_preserves_raw_scale_encoding():
    cache = DeepSeekV4PackedLayerCache(0, xp=np, identity='synthetic-raw-retention')
    cache.update_local(np.ones((1, 1, 128, 512), np.float32))
    # A valid noncanonical code/scale pair decodes identically but a repack
    # canonicalizes it. Retention must preserve the serialized state verbatim.
    codes, scales, rope = cache.state[0]
    codes[...] = 0x38  # E4M3 value 1 with scale 1.
    scales[...] = 1
    before = tuple(a.copy() for a in (codes, scales, rope))
    cache.update_local(np.zeros((1, 1, 1, 512), np.float32))
    for after, old in zip(cache.state[0], before):
        np.testing.assert_array_equal(after[:, :, :-1], old[:, :, 1:])

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from mlx2.runtime.apc_v2 import inspect_apc_capabilities
from mlx2.runtime.cache_operations import cache_operations
from mlx2.runtime.cow_cache import snapshot_prompt_cache_descriptors
from mlx2.runtime.drafters.dflash_base import _fold_rms_norm
from mlx2.runtime.drafters.fold import FOLDKVCache, fold_attention
from mlx2.runtime.drafters.fold_reference import FOLDPolicy, fold_block
from mlx2.runtime.pld import _rewind, _snapshot


def test_fold_rms_norm_matches_stock_without_fused_metal_primitive():
    rng = np.random.default_rng(31)
    values = mx.array(rng.normal(size=(2, 7, 4, 16)).astype(np.float16))
    norm = nn.RMSNorm(16, eps=1e-6)
    norm.weight = mx.array(rng.normal(size=(16,)).astype(np.float16))
    expected = norm(values)
    actual = _fold_rms_norm(norm, values)
    mx.eval(expected, actual)
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=2e-3, atol=2e-3
    )


def test_live_fold_matches_source_equation_oracle():
    rng = np.random.default_rng(7)
    context, block, heads, dim = 9, 3, 2, 4
    arrays = [
        rng.normal(size=shape).astype(np.float32)
        for shape in (
            (block, heads, dim),
            (context, heads, dim),
            (context, heads, dim),
            (block, heads, dim),
            (block, heads, dim),
        )
    ]
    q, kc, vc, kb, vb = arrays
    policy = FOLDPolicy(window=3, alpha=2.0, sliding_layers="fold")
    rope = nn.RoPE(dim, traditional=False, base=10000.0)
    cache = FOLDKVCache()
    raw_k = mx.array(kc).transpose(1, 0, 2)[None]
    values = mx.array(vc).transpose(1, 0, 2)[None]
    cache.append_raw_keys(raw_k)
    cache.update_and_fetch(rope(raw_k, offset=0), values)
    actual, used_far = fold_attention(
        mx.array(q).transpose(1, 0, 2)[None],
        mx.array(kb).transpose(1, 0, 2)[None],
        mx.array(vb).transpose(1, 0, 2)[None],
        rope=rope,
        cache=cache,
        policy=policy,
        scale=dim**-0.5,
    )
    expected = fold_block(
        q,
        kc,
        vc,
        kb,
        vb,
        policy=policy,
        inv_freq=np.asarray([1.0, 0.01]),
        scale=dim**-0.5,
    )
    mx.eval(actual)
    assert used_far is True
    np.testing.assert_allclose(
        np.asarray(actual)[0].transpose(1, 0, 2),
        expected,
        rtol=2e-5,
        atol=2e-5,
    )


def test_fold_cache_state_includes_unrotated_keys_and_memory_bytes(tmp_path):
    cache = FOLDKVCache()
    keys = mx.ones((1, 2, 4, 8), mx.float16)
    values = mx.zeros_like(keys)
    cache.append_raw_keys(keys)
    cache.update_and_fetch(keys, values)
    assert len(cache.state) == 3
    assert (
        cache.nbytes == cache.keys.nbytes + cache.values.nbytes + cache.raw_keys.nbytes
    )
    assert cache.nbytes >= keys.nbytes * 3
    cache.trim(2)
    raw, retained_values = cache.raw_keys_and_values()
    assert raw.shape[2] == retained_values.shape[2] == cache.offset == 2

    from mlx2.runtime.models.cache import load_prompt_cache, save_prompt_cache

    path = tmp_path / "fold-cache.safetensors"
    save_prompt_cache(str(path), [cache])
    restored = load_prompt_cache(str(path))[0]
    assert isinstance(restored, FOLDKVCache)
    assert restored.offset == 2
    restored_raw, restored_values = restored.raw_keys_and_values()
    assert restored_raw.shape[2] == restored_values.shape[2] == 2


def test_fold_cache_declares_exact_publication_and_rollback_operations():
    cache = FOLDKVCache()
    keys = mx.ones((1, 2, 4, 8), mx.float16)
    values = mx.zeros_like(keys)
    cache.append_raw_keys(keys)
    cache.update_and_fetch(keys, values)

    operations = cache_operations(cache, capability="FOLD test")
    assert operations.family == "fold_kv"
    assert operations.topology == "kv"
    assert operations.planes[0].fields == ("keys", "values", "raw_keys")
    assert inspect_apc_capabilities([cache]).exact_prefix

    (snapshot,), _, receipt = snapshot_prompt_cache_descriptors([cache])
    assert snapshot is not cache
    assert snapshot.offset == cache.offset == 4
    assert receipt["snapshot_bytes"] == snapshot.nbytes
    np.testing.assert_array_equal(np.asarray(snapshot.keys), np.asarray(cache.keys))
    np.testing.assert_array_equal(np.asarray(snapshot.values), np.asarray(cache.values))
    np.testing.assert_array_equal(
        np.asarray(snapshot.raw_keys), np.asarray(cache.raw_keys)
    )

    rollback = _snapshot([cache])
    extra = mx.full((1, 2, 3, 8), 2, mx.float16)
    cache.append_raw_keys(extra)
    cache.update_and_fetch(extra, extra)
    assert cache.offset == 7
    _rewind([cache], rollback)
    assert cache.offset == 4
    raw, retained_values = cache.raw_keys_and_values()
    assert raw.shape[2] == retained_values.shape[2] == 4

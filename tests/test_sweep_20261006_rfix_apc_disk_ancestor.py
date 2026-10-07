"""A spilled draft checkpoint is still a sidecar ancestor.

``_sidecar_ancestor_locked`` considered resident entries only, so a sidecar
checkpoint spilled to the disk tier stayed hidden behind a deeper resident
plain entry even when the lookup was allowed to restore from disk: the
draft route refused the plain hit and prefilled cold.
"""

import tempfile

import mlx.core as mx
import pytest

from mlx2.runtime.apc_v2 import APCKey, APCv2, MTPAPCSidecar
from mlx2.runtime.models.cache import ArraysCache, KVCache


def _kv(length, seed=0):
    cache = KVCache()
    value = mx.arange(seed, seed + length, dtype=mx.float32).reshape(1, 1, length, 1)
    cache.update_and_fetch(value, value)
    mx.eval(cache.state)
    return cache


def _rec(length):
    cache = ArraysCache(1)
    cache[0] = mx.ones((1, 4), dtype=mx.float32) * length
    cache.lengths = mx.array([length], dtype=mx.int32)
    cache._host_lengths = (cache.lengths, [length])
    mx.eval(cache.state)
    return cache


def _sidecar(covered):
    return MTPAPCSidecar(
        ([_kv(covered - 1, seed=7)], mx.ones((1, 1, 4))), covered_tokens=covered
    )


def _spilled_ancestor(directory, *, hybrid=True):
    now = [0.0]
    apc = APCv2(
        max_size=8, layout_name=f"disk-ancestor-{hybrid}", idle_disk_seconds=1,
        idle_disk_dir=directory, now_fn=lambda: now[0],
    )
    key = APCKey("m")
    prompt = list(range(400))

    def caches(length):
        return [_rec(length), _kv(length)] if hybrid else [_kv(length)]

    assert apc.store(key, prompt[:100], caches(100), sidecar=_sidecar(100)).stored
    now[0] = 100.0
    assert apc.spill_idle_entries(now=100.0) == 1
    assert not apc._trie.get(key, prompt[:100]).prompt_cache
    assert apc.store(key, prompt[:300], caches(300)).stored
    return apc, key, prompt


@pytest.mark.parametrize("hybrid", [True, False])
def test_disk_sidecar_ancestor_is_restored_and_served(hybrid):
    with tempfile.TemporaryDirectory() as directory:
        apc, key, prompt = _spilled_ancestor(directory, hybrid=hybrid)
        hit = apc.lookup(key, prompt, require_sidecar=True)
        assert hit.sidecar is not None and hit.cached_tokens == 100
        assert apc.lifetime_stats()["sidecar_ancestor_hits"] == 1
        hit.cache.close()


def test_resident_only_probe_asks_admission_to_restore_it():
    with tempfile.TemporaryDirectory() as directory:
        apc, key, prompt = _spilled_ancestor(directory)
        probe = apc.lookup(key, prompt, require_sidecar=True, allow_disk_restore=False)
        assert probe.cache is None
        assert probe.miss_reason == "disk_restore_requires_admission"
        assert not apc._trie.get(key, prompt[:100]).prompt_cache
        apc.discard_lookup_credit(probe, "disk_restore_admission")
        hit = apc.lookup(key, prompt, require_sidecar=True)
        assert hit.sidecar is not None and hit.cached_tokens == 100
        hit.cache.close()


def test_plain_route_lookup_is_unchanged():
    with tempfile.TemporaryDirectory() as directory:
        apc, key, prompt = _spilled_ancestor(directory)
        hit = apc.lookup(key, prompt, allow_disk_restore=False)
        assert hit.cached_tokens == 300 and hit.sidecar is None
        hit.cache.close()
        assert not apc._trie.get(key, prompt[:100]).prompt_cache

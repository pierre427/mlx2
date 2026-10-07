"""APC-1: draft routes must see a shallower sidecar checkpoint.

A deeper sidecar-less entry (a plain-decode lane, a rolling checkpoint
without draft state) was the only shorter-prefix candidate APCv2 offered, so
a self-MTP route turned it into ``mtp_sidecar_missing`` and prefilled from 0
although a sidecar checkpoint was resident further up the same path.  On a
trimmable cache a sidecar-less store also pruned that sidecar entry.
"""

import mlx.core as mx

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
    return MTPAPCSidecar(([_kv(covered - 1, seed=7)], mx.ones((1, 1, 4))), covered_tokens=covered)


def _route_effective(hit, tokens):
    # serving.route_usable_hit for a self-MTP route.
    usable = hit.sidecar is not None or hit.cached_tokens in (0, len(tokens) - 1)
    return hit.cached_tokens if usable else 0


def test_deeper_plain_entry_does_not_hide_a_sidecar_ancestor():
    apc = APCv2(max_size=8, layout_name="sidecar-shadow-v1")
    key = APCKey("m")
    prompt = list(range(400))
    assert apc.store(key, prompt[:100], [_rec(100), _kv(100)], sidecar=_sidecar(100)).stored
    assert apc.store(key, prompt[:300], [_rec(300), _kv(300)]).stored
    plain = apc.lookup(key, prompt)
    assert plain.cached_tokens == 300 and plain.sidecar is None
    plain.cache.close()
    hit = apc.lookup(key, prompt, require_sidecar=True)
    assert hit.sidecar is not None and _route_effective(hit, prompt) == 100
    assert apc.lifetime_stats()["sidecar_ancestor_hits"] == 1
    hit.cache.close()


def test_plain_landing_at_len_minus_one_still_wins():
    apc = APCv2(max_size=8, layout_name="sidecar-shadow-v2")
    key = APCKey("m")
    prompt = list(range(400))
    assert apc.store(key, prompt[:100], [_rec(100), _kv(100)], sidecar=_sidecar(100)).stored
    assert apc.store(key, prompt[:399], [_rec(399), _kv(399)]).stored
    hit = apc.lookup(key, prompt, require_sidecar=True)
    assert hit.sidecar is None and hit.cached_tokens == 399
    hit.cache.close()


def test_sidecar_entry_is_not_subsumed_by_a_plain_entry():
    apc = APCv2(max_size=8, layout_name="sidecar-subsume-v1")
    key = APCKey("m")
    prompt = list(range(400))
    assert apc.store(key, prompt[:100], [_kv(100)], sidecar=_sidecar(100)).stored
    assert apc.store(key, prompt[:300], [_kv(300)]).stored
    assert apc._trie.get(key, prompt[:100]).sidecar is not None
    hit = apc.lookup(key, prompt, require_sidecar=True)
    assert hit.sidecar is not None and hit.cached_tokens == 100
    hit.cache.close()

"""APC-2: APCv2 lifetime counters are per request, not per lookup.

Serving looks up again for the same request (disk admission, preemption
replay, a same-prefix follower) and refuses some hits after APCv2 counted
them (``mtp_sidecar_missing``, media boundary, activation capsule).  Those
lookups were exported as hits, reused tokens and lookups, and a refused hit
still raised its entry's hit count, which promotes an interior checkpoint's
retention rank.
"""

import tempfile

import mlx.core as mx

from mlx2.runtime.apc_v2 import APCKey, APCv2
from mlx2.runtime.models.cache import ArraysCache, KVCache

STATS = ("lookups", "hits", "misses", "queried_tokens", "cached_tokens", "interior_hits")


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


def _delta(apc, base):
    after = apc.lifetime_stats()
    return {name: after[name] - base[name] for name in STATS}


def test_disk_admitted_request_counts_once():
    with tempfile.TemporaryDirectory() as directory:
        now = [0.0]
        apc = APCv2(
            max_size=8, layout_name="acct-v1", idle_disk_seconds=1,
            idle_disk_dir=directory, now_fn=lambda: now[0],
        )
        key = APCKey("m")
        prompt = list(range(64))
        assert apc.store(key, prompt[:48], [_rec(48), _kv(48)]).stored
        now[0] = 100.0
        assert apc.spill_idle_entries(now=100.0) >= 1
        base = apc.lifetime_stats()
        # Serving admission: a resident-only probe, then the gated restore.
        first = apc.lookup(key, prompt, allow_disk_restore=False)
        assert first.miss_reason == "disk_restore_requires_admission"
        assert apc.discard_lookup_credit(first, "disk_restore_admission")
        assert not apc.discard_lookup_credit(first, "disk_restore_admission")
        second = apc.lookup(key, prompt)
        assert second.hit and second.cached_tokens == 48
        second.cache.close()
        assert _delta(apc, base) == {
            "lookups": 1, "hits": 1, "misses": 0, "queried_tokens": 64,
            "cached_tokens": 48, "interior_hits": 0,
        }
        assert apc.apc_stats["discarded_lookups"] == {"disk_restore_admission": 1}


def test_refused_hit_credits_nothing_and_does_not_promote_retention():
    apc = APCv2(max_size=8, layout_name="acct-v2")
    key = APCKey("m")
    prompt = list(range(400))
    assert apc.store(
        key, prompt[:300], [_rec(300), _kv(300)], retention_role="interior_checkpoint"
    ).stored
    entry = apc._trie.get(key, prompt[:300])
    assert apc._entry_retention_rank(entry) == 0
    base = apc.lifetime_stats()
    hit = apc.lookup(key, prompt)
    assert hit.cached_tokens == 300 and apc._entry_retention_rank(entry) == 1
    hit.cache.close()
    apc.discard_lookup_credit(hit, "mtp_sidecar_missing")
    assert _delta(apc, base) == dict.fromkeys(STATS, 0)
    assert entry._apc_hit_count == 0 and apc._entry_retention_rank(entry) == 0
    assert apc._interior_reused_entries == 0

"""Withdrawing a lookup's credit restores everything the hit changed.

``discard_lookup_credit`` took back the hit count and decayed rate, but not
the served depth a hit records for value retention, so a refused hit kept
re-pricing the entry.  And the lifetime reused-interior counter was
decremented only through the credit that recorded the entry's first hit:
two refused hits withdrawn first-hit-first left it raised with the entry's
hit count back at zero.
"""

import mlx.core as mx

from mlx2.runtime.apc_v2 import APCKey, APCv2
from mlx2.runtime.models.cache import KVCache


def _kv(length):
    cache = KVCache()
    value = mx.zeros((1, 1, length, 8))
    cache.update_and_fetch(value, value)
    mx.eval(cache.state)
    return cache


def _interior(apc, key, length):
    assert apc.store(
        key, list(range(length)), [_kv(length)],
        retention_role="interior_checkpoint",
    ).stored
    return apc._trie.get(key, list(range(length)))


def _hit(apc, key, tokens):
    hit = apc.lookup(key, tokens)
    assert hit.hit
    hit.cache.close()
    return hit


def test_reused_interior_counter_unwinds_in_either_order():
    for order in ((0, 1), (1, 0)):
        apc = APCv2(max_size=8, layout_name=f"credit-order-{order}")
        key = APCKey("m")
        entry = _interior(apc, key, 32)
        hits = [_hit(apc, key, list(range(40))), _hit(apc, key, list(range(40)))]
        assert entry._apc_hit_count == 2 and apc._interior_reused_entries == 1
        for index in order:
            assert apc.discard_lookup_credit(hits[index], "refused")
        assert entry._apc_hit_count == 0
        assert apc._interior_reused_entries == 0, order


def test_withdrawn_hit_restores_the_served_depth():
    apc = APCv2(max_size=8, layout_name="credit-depth", retention_policy="value")
    key = APCKey("m")
    assert apc.store(key, list(range(64)), [_kv(64)]).stored
    entry = apc._trie.get(key, list(range(64)))
    _hit(apc, key, list(range(64)) + [900])  # serves 64
    assert entry._apc_served_depth == 64
    # A shorter prompt that branches the entry at 40, then refused.
    refused = _hit(apc, key, list(range(40)) + [901])
    assert entry._apc_served_depth == 40
    apc.discard_lookup_credit(refused, "refused")
    assert entry._apc_served_depth == 64


def test_served_depth_unwinds_out_of_order():
    apc = APCv2(max_size=8, layout_name="credit-depth-order", retention_policy="value")
    key = APCKey("m")
    assert apc.store(key, list(range(64)), [_kv(64)]).stored
    entry = apc._trie.get(key, list(range(64)))
    first = _hit(apc, key, list(range(64)) + [900])  # 64
    second = _hit(apc, key, list(range(40)) + [901])  # 40
    third = _hit(apc, key, list(range(50)) + [902])  # 50
    assert entry._apc_served_depth == 50
    # Withdraw the middle one, then the newest: the survivor's depth stands.
    apc.discard_lookup_credit(second, "refused")
    assert entry._apc_served_depth == 50
    apc.discard_lookup_credit(third, "refused")
    assert entry._apc_served_depth == 64
    apc.discard_lookup_credit(first, "refused")
    assert not getattr(entry, "_apc_served_depth", 0)

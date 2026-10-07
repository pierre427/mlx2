"""Admission's evictable estimate counts only bytes eviction can free.

``unleased_resident_nbytes`` summed the ledger charges of unleased entries.
The first holder of a shared checkpoint snapshot is charged for it, but
when another holder is leased, evicting the payer only moves that charge to
the leased holder: the snapshot stays resident.  Admission then counted
bytes it could never reclaim.
"""

import mlx.core as mx

from mlx2.runtime.apc_v2 import APCv2
from mlx2.runtime.models.cache import ArraysCache, KVCache


def _kv(cache, n):
    value = mx.zeros((1, 1, n, 64))
    cache.update_and_fetch(value, value)


def _state(seed):
    return mx.full((1, 8, 128, 128), float(seed))


def _two_publications(apc, key):
    kv, gdn = KVCache(), ArraysCache(1)
    _kv(kv, 4)
    gdn[0] = _state(1)
    mx.eval(kv.state, gdn[0])
    gdn.state_checkpoint([4], force=True)
    _kv(kv, 4)
    gdn[0] = _state(2)
    mx.eval(kv.state, gdn[0])
    gdn.state_checkpoint([8], force=True)
    mx.eval(gdn._checkpoints)
    assert apc.store(key, list(range(8)), [kv, gdn]).stored
    _kv(kv, 4)
    gdn[0] = _state(3)
    mx.eval(kv.state, gdn[0])
    assert apc.store(key, list(range(12)), [kv, gdn]).stored


def _drain(apc):
    before = apc.nbytes
    while apc.evict_oldest_unleased():
        pass
    return before - apc.nbytes


def test_a_leased_co_holder_keeps_the_payers_snapshots_unreclaimable():
    apc = APCv2(max_size=8, layout_name="reclaim-leased")
    key = apc.key("m")
    _two_publications(apc, key)
    lease = apc.lookup(key, list(range(12)) + [99])  # leases the 12-token entry
    assert lease.cached_tokens == 12
    estimate = apc.unleased_resident_nbytes()
    freed = _drain(apc)
    assert estimate == freed, (estimate, freed)
    lease.cache.close()


def test_snapshots_held_only_by_evictable_entries_count_once():
    apc = APCv2(max_size=8, layout_name="reclaim-free")
    key = apc.key("m")
    _two_publications(apc, key)
    estimate = apc.unleased_resident_nbytes()
    assert estimate == apc.nbytes
    assert estimate == _drain(apc)

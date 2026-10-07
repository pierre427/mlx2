"""COW-1: APCv2 charges shared recurrent checkpoint snapshots once.

A lane's ArraysCache keeps its state checkpoints across the prompt boundary
and the finished-lane publication, so both entries alias the same snapshot
buffers.  Each entry's ``nbytes`` counted them again, so the resident byte cap
and pressure eviction acted on bytes that did not exist.
"""

import gc

import mlx.core as mx

from mlx2.runtime.apc_v2 import APCv2
from mlx2.runtime.models.cache import ArraysCache, KVCache

MIB = 1 << 20


def _kv(cache, n):
    value = mx.zeros((1, 1, n, 64))
    cache.update_and_fetch(value, value)


def _state(seed):
    return mx.full((1, 8, 128, 128), float(seed))  # 0.5 MiB fp32


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
    assert apc.store(key, list(range(8)), [kv, gdn]).stored  # prompt boundary
    _kv(kv, 4)
    gdn[0] = _state(3)
    mx.eval(kv.state, gdn[0])
    assert apc.store(key, list(range(12)), [kv, gdn]).stored  # finished lane


def _entries(apc):
    return {len(tokens): entry for _key, tokens, entry in apc._entry_records_locked()}


def test_shared_checkpoints_are_charged_once():
    # Earlier tests' caches must not be freed inside the measurement.
    gc.collect()
    mx.clear_cache()
    baseline = mx.get_active_memory()
    apc = APCv2(max_size=8, layout_name="ledger-probe")
    _two_publications(apc, apc.key("m"))
    gc.collect()
    mx.clear_cache()
    held = mx.get_active_memory() - baseline
    charged = apc.nbytes
    assert charged <= held * 1.05, (charged / MIB, held / MIB)
    assert charged == sum(entry.nbytes for entry in _entries(apc).values())


def test_evicting_the_payer_moves_the_charge_to_the_remaining_holder():
    apc = APCv2(max_size=8, layout_name="ledger-evict")
    key = apc.key("m")
    _two_publications(apc, key)
    entries = _entries(apc)
    snapshot_bytes = 2 * _state(0).nbytes
    total = apc.nbytes
    first, second = entries[8], entries[12]
    freed_if_first_goes = first.nbytes - snapshot_bytes
    apc._drop_entry_locked(key, list(range(8)), first)
    # The snapshots are still held by the finished-lane entry: dropping the
    # boundary frees only its own bytes, and the survivor now pays for them.
    assert apc.nbytes == total - freed_if_first_goes
    assert apc.nbytes == second.nbytes
    apc._drop_entry_locked(key, list(range(12)), second)
    assert apc.nbytes == 0
    assert not apc._shared_buffers


def test_clear_resets_the_ledger():
    apc = APCv2(max_size=8, layout_name="ledger-clear")
    _two_publications(apc, apc.key("m"))
    apc.clear()
    assert apc.nbytes == 0 and not apc._shared_buffers


def test_a_lane_resumed_from_an_entry_shares_its_snapshots():
    apc = APCv2(max_size=8, layout_name="ledger-resume")
    key = apc.key("m")
    kv, gdn = KVCache(), ArraysCache(1)
    for position, seed in ((4, 1), (8, 2)):
        _kv(kv, 4)
        gdn[0] = _state(seed)
        mx.eval(kv.state, gdn[0])
        gdn.state_checkpoint([position], force=True)
    mx.eval(gdn._checkpoints)
    assert apc.store(key, list(range(8)), [kv, gdn]).stored
    stored_alone = apc.nbytes
    hit = apc.lookup(key, list(range(8)) + [50, 51, 52, 53])
    assert hit.cached_tokens == 8
    resumed = list(hit.cache)
    _kv(resumed[0], 12 - resumed[0].offset)
    resumed[1][0] = _state(9)
    mx.eval(resumed[0].state, resumed[1][0])
    assert apc.store(key, list(range(8)) + [50, 51, 52, 53], resumed).stored
    hit.cache.close()
    # The resumed lane carries the stored entry's snapshots at 4 and 8; only
    # its own KV and live state are new.
    assert apc.nbytes - stored_alone == resumed[0].nbytes + _state(0).nbytes
    assert apc.nbytes == sum(entry.nbytes for entry in _entries(apc).values())


def _pending_snapshot_store(apc, key):
    """Store a lane whose checkpoint snapshot has not been computed yet, as an
    ``async_eval``-scheduled capture is while the GPU still runs; returns the
    bytes the store made resident beyond what the lane already held."""
    kv, gdn = KVCache(), ArraysCache(1)
    _kv(kv, 4)
    gdn[0] = _state(1)
    mx.eval(kv.state, gdn[0])
    pending = mx.zeros((4 * MIB,)) + 1.0  # 16 MiB once evaluated
    gdn._checkpoints = [[(4, [pending])]]
    gc.collect()
    mx.clear_cache()
    before = mx.get_active_memory()
    assert apc.store(key, list(range(4)), [kv, gdn]).stored
    return mx.get_active_memory() - before, pending


def test_store_does_not_evaluate_pending_snapshots():
    # Keying the ledger by buffer address evaluated every snapshot on the
    # store path: a host wait on the GPU stream during prefill.
    apc = APCv2(max_size=8, layout_name="ledger-lazy")
    grown, pending = _pending_snapshot_store(apc, apc.key("m"))
    assert grown < 4 * MIB, grown / MIB
    assert apc.nbytes == sum(entry.nbytes for entry in _entries(apc).values())
    mx.eval(pending)


def test_pending_snapshots_shared_by_two_entries_are_charged_once():
    apc = APCv2(max_size=8, layout_name="ledger-lazy-shared")
    key = apc.key("m")
    kv, gdn = KVCache(), ArraysCache(1)
    _kv(kv, 4)
    gdn[0] = _state(1)
    mx.eval(kv.state, gdn[0])
    gdn._checkpoints = [[(4, [mx.zeros((MIB,)) + 1.0])]]  # 4 MiB, pending
    assert apc.store(key, list(range(4)), [kv, gdn]).stored
    alone = apc.nbytes
    _kv(kv, 4)
    gdn[0] = _state(2)
    mx.eval(kv.state, gdn[0])
    assert apc.store(key, list(range(8)), [kv, gdn]).stored
    entries = _entries(apc)
    assert apc.nbytes - alone == entries[8].nbytes
    assert entries[8].nbytes == kv.nbytes + _state(0).nbytes

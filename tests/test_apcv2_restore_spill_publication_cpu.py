"""A background spill published inside a disk restore must not drop the
placeholder being restored.

A disk restore reserves its resident bytes first, and the reservation waits
for in-flight background spills and publishes them
(``_spill_resident_budget_locked`` -> ``_collect_spills_locked(wait=True)``).
Publishing a spill can push the disk tier over its cap.  The synchronous spill
path then enforces the disk cap with ``exclude=entry`` so the placeholder being
restored survives until publication; background publication enforced it with
no exclusion.  The oldest disk-only placeholder, which is the one being
restored, was dropped under the restore: the lookup raised ``KeyError`` from
``PromptTrie.pop`` (an HTTP 500 at admission), or the restore published onto a
detached entry and left its bytes in ``_n_bytes`` until ``clear()``.
"""

import threading

import mlx.core as mx
import pytest

from mlx2.runtime import apc_v2
from mlx2.runtime.apc_v2 import APCKey, APCv2, _iter_trie_entries
from mlx2.runtime.models.cache import KVCache


def _state(length, seed=0):
    cache = KVCache()
    values = mx.arange(seed, seed + length, dtype=mx.float32).reshape(1, 1, length, 1)
    cache.update_and_fetch(values, values)
    mx.eval(cache.state)
    return [cache]


def _sizes(tmp_path):
    """(resident bytes, disk bytes) of one 8-token entry."""
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    probe = APCv2(
        max_size=8, layout_name="restore-spill-probe",
        idle_disk_seconds=3600, idle_disk_dir=str(probe_dir),
    )
    key = APCKey("probe")
    probe.store(key, [1] * 8, _state(8))
    resident = int(probe.nbytes)
    with probe._apc_lock:
        entry = probe._trie.get(key, [1] * 8)
        assert probe._spill_entry_locked(key, [1] * 8, entry, reason="idle")
    disk = int(probe._disk_bytes)
    probe.close()
    return resident, disk


@pytest.fixture
def writer_gates(monkeypatch):
    """Hold the n-th background write until gates[n] opens.  Synchronous
    spills (on the calling thread) write at once."""
    gates = [threading.Event(), threading.Event(), threading.Event()]
    calls = [0]
    real = apc_v2._write_spill_job
    caller = threading.current_thread()

    def write(job):
        if threading.current_thread() is caller:
            return real(job)
        gate = gates[min(calls[0], len(gates) - 1)]
        calls[0] += 1
        assert gate.wait(10), "test never released the writer"
        real(job)

    monkeypatch.setattr(apc_v2, "_write_spill_job", write)
    yield gates
    for gate in gates:
        gate.set()


def _release_writes_from_restore(monkeypatch, apc, gates):
    """Open gates[n] at the n-th restore reservation: each in-flight write
    lands while the restore is waiting for it, deterministically."""
    real = apc._reserve_restore_bytes_locked
    calls = [0]

    def reserve(required, *, entry):
        if calls[0] < len(gates):
            gates[calls[0]].set()
        calls[0] += 1
        return real(required, entry=entry)

    monkeypatch.setattr(apc, "_reserve_restore_bytes_locked", reserve)


def _spill_now(apc, key, tokens):
    with apc._apc_lock:
        entry = apc._trie.get(key, tokens)
        assert apc._spill_entry_locked(key, tokens, entry, reason="idle")
    assert not entry.prompt_cache and entry._apc_disk
    return entry


def _resident_bytes_in_trie(apc):
    with apc._apc_lock:
        return sum(int(entry.nbytes) for entry in _iter_trie_entries(apc._trie))


def test_publication_during_a_restore_keeps_the_placeholder(
    tmp_path, writer_gates, monkeypatch
):
    resident, disk = _sizes(tmp_path)
    now = [0.0]
    apc = APCv2(
        max_size=64,
        max_bytes=2 * resident,
        layout_name="restore-spill-v1",
        idle_disk_seconds=3600,
        idle_disk_dir=str(tmp_path),
        # Room for one snapshot: publishing a second must evict the oldest.
        idle_disk_max_bytes=disk + disk // 2,
        now_fn=lambda: now[0],
        background_spill=True,
    )
    key = APCKey("restore-spill")
    now[0] = 1.0
    apc.store(key, [5] * 8, _state(8, seed=5))
    _spill_now(apc, key, [5] * 8)  # the oldest disk-only snapshot
    for t, first in ((3.0, 10), (4.0, 20), (5.0, 30)):
        now[0] = t
        apc.store(key, [first] * 8, _state(8, seed=first))
    assert len(apc._spill_jobs) == 1  # [10]*8 is being written
    _release_writes_from_restore(monkeypatch, apc, writer_gates)
    now[0] = 6.0
    # Before the fix: KeyError from PromptTrie.pop in _drop_entry_locked.
    hit = apc.lookup(key, [5] * 8 + [1], allow_disk_restore=True)
    # The placeholder survives the publication and restores exactly.
    assert hit.hit and hit.cached_tokens == 8, hit.miss_reason
    keys, _ = hit.cache[0].state
    assert mx.array_equal(
        keys[..., :8, :].flatten(), mx.arange(5, 13, dtype=mx.float32)
    )
    hit.cache.close()
    assert apc._n_bytes == _resident_bytes_in_trie(apc)
    apc.clear(release_memory=False)


def test_a_restore_never_publishes_onto_a_detached_entry(
    tmp_path, writer_gates, monkeypatch
):
    resident, disk = _sizes(tmp_path)
    now = [0.0]
    apc = APCv2(
        max_size=2,
        max_bytes=1 << 40,
        layout_name="restore-spill-v1",
        idle_disk_seconds=3600,
        idle_disk_dir=str(tmp_path),
        # Room for two snapshots: the third publication evicts the oldest.
        idle_disk_max_bytes=2 * disk + disk // 2,
        now_fn=lambda: now[0],
        background_spill=True,
    )
    key = APCKey("restore-orphan")
    now[0] = 1.0
    apc.store(key, [5] * 8, _state(8, seed=5))
    restoring = _spill_now(apc, key, [5] * 8)
    for t, first in ((3.0, 10), (4.0, 20), (5.0, 30)):
        now[0] = t
        apc.store(key, [first] * 8, _state(8, seed=first))
    assert len(apc._spill_jobs) == 1  # count pool: [10]*8 is being written
    now[0] = 6.0
    reused = apc.lookup(key, [10] * 8 + [1], allow_disk_restore=False)
    assert reused.hit  # kept hot when its write lands
    reused.cache.close()
    _release_writes_from_restore(monkeypatch, apc, writer_gates)
    now[0] = 7.0
    hit = apc.lookup(key, [5] * 8 + [1], allow_disk_restore=True)
    assert hit.hit and hit.cached_tokens == 8, hit.miss_reason
    hit.cache.close()
    with apc._apc_lock:
        in_trie = any(entry is restoring for entry in _iter_trie_entries(apc._trie))
    # A restore publishes only onto the entry the trie still holds.
    assert in_trie and restoring.prompt_cache
    # Before the fix: _n_bytes kept the detached entry's bytes until clear().
    assert apc._n_bytes == _resident_bytes_in_trie(apc)
    apc.clear(release_memory=False)


def test_a_placeholder_dropped_under_its_restore_is_a_clean_miss(
    tmp_path, monkeypatch
):
    """Whatever removes the placeholder mid-restore, the restore must not
    publish onto the detached entry and the lookup must not raise."""
    resident, _disk = _sizes(tmp_path)
    now = [0.0]
    apc = APCv2(
        max_size=64,
        max_bytes=4 * resident,
        layout_name="restore-spill-v1",
        idle_disk_seconds=3600,
        idle_disk_dir=str(tmp_path),
        now_fn=lambda: now[0],
    )
    key = APCKey("restore-detached")
    now[0] = 1.0
    apc.store(key, [5] * 8, _state(8, seed=5))
    restoring = _spill_now(apc, key, [5] * 8)
    real = apc._reserve_restore_bytes_locked
    calls = [0]

    def reserve(required, *, entry):
        calls[0] += 1
        if calls[0] == 3:  # after the load, before publication
            apc._drop_entry_locked(key, [5] * 8, entry)
        return real(required, entry=entry)

    monkeypatch.setattr(apc, "_reserve_restore_bytes_locked", reserve)
    now[0] = 2.0
    hit = apc.lookup(key, [5] * 8 + [1], allow_disk_restore=True)
    assert calls[0] >= 3
    assert not hit.hit
    assert not restoring.prompt_cache
    assert apc._n_bytes == _resident_bytes_in_trie(apc) == 0
    apc.clear(release_memory=False)

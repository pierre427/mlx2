"""Store-time pressure spills write on a background thread.

A full APC used to write each evicted entry to disk inside the ``store`` that
evicted it, on the generation worker: about 1 s of TTFT per 27B 4K-token
request once the cache was full (qualification/runs/intake-probes-20261007,
prefillwarm-27b idle:12 cells).  With ``background_spill`` the entry stays
resident and servable while a writer thread writes it, and the next APC
operation publishes the disk record and drops the resident copy.
"""

import threading

import mlx.core as mx
import pytest

from mlx2.runtime import apc_v2
from mlx2.runtime.apc_v2 import APCKey, APCv2
from mlx2.runtime.models.cache import KVCache


def _state(length, seed=0):
    cache = KVCache()
    values = mx.arange(seed, seed + length, dtype=mx.float32).reshape(1, 1, length, 1)
    cache.update_and_fetch(values, values)
    mx.eval(cache.state)
    return [cache]


def _entry_bytes():
    probe = APCv2(max_size=8, layout_name="bg-spill-probe")
    probe.store(APCKey("probe"), list(range(8)), _state(8))
    nbytes = int(probe.nbytes)
    probe.clear(release_memory=False)
    return nbytes


@pytest.fixture
def gated_writer(monkeypatch):
    """Hold every background write until the test opens the gate."""
    gate = threading.Event()
    started = threading.Event()
    real = apc_v2._write_spill_job

    def write(job):
        started.set()
        assert gate.wait(10), "test never released the writer"
        real(job)

    monkeypatch.setattr(apc_v2, "_write_spill_job", write)
    return gate, started


def _apc(tmp_path, now, **kwargs):
    return APCv2(
        max_size=64,
        max_bytes=2 * _entry_bytes(),
        layout_name="bg-spill-v1",
        idle_disk_seconds=3600,
        idle_disk_dir=str(tmp_path),
        now_fn=lambda: now[0],
        background_spill=True,
        **kwargs,
    )


def _fill(apc, key, now):
    """Two resident entries, then a third whose store must evict the oldest."""
    for t, first in ((1.0, 10), (2.0, 20), (3.0, 30)):
        now[0] = t
        apc.store(key, [first] * 8, _state(8, seed=first))


def test_store_returns_before_the_write_and_the_entry_stays_servable(tmp_path, gated_writer):
    gate, started = gated_writer
    now = [0.0]
    apc = _apc(tmp_path, now)
    key = APCKey("bg")
    _fill(apc, key, now)
    assert started.wait(5)
    stats = apc.apc_stats["idle_disk"]
    assert stats["background_spills_submitted"] == 1
    assert stats["pressure_spills"] == 0  # not published yet
    assert stats["background_spill_inflight"] == 1
    gate.set()
    # Still resident (and not a hit, so it is not kept hot) until collected.
    apc._spill_jobs[0].done.wait(5)
    now[0] = 4.0
    miss = apc.lookup(key, [99] * 4, allow_disk_restore=False)
    assert not miss.hit
    stats = apc.apc_stats["idle_disk"]
    assert stats["background_spills_published"] == 1
    assert stats["pressure_spills"] == 1
    entry = apc._trie.get(key, [10] * 8)
    assert not entry.prompt_cache and entry._apc_disk
    # The disk copy restores exactly.
    now[0] = 5.0
    hit = apc.lookup(key, [10] * 8 + [1], allow_disk_restore=True)
    assert hit.hit and hit.cached_tokens == 8
    keys, _ = hit.cache[0].state
    assert mx.array_equal(keys[..., :8, :].flatten(), mx.arange(10, 18, dtype=mx.float32))
    hit.cache.close()
    apc.clear(release_memory=False)


def test_an_entry_reused_while_in_flight_stays_resident(tmp_path, gated_writer):
    gate, started = gated_writer
    now = [0.0]
    apc = _apc(tmp_path, now)
    key = APCKey("bg-hot")
    _fill(apc, key, now)
    assert started.wait(5)
    now[0] = 4.0
    hit = apc.lookup(key, [10] * 8 + [1], allow_disk_restore=False)
    assert hit.hit and hit.cached_tokens == 8  # served from the resident copy
    hit.cache.close()
    gate.set()
    apc._spill_jobs[0].done.wait(5)
    now[0] = 5.0
    apc.lookup(key, [99] * 4, allow_disk_restore=False)
    stats = apc.apc_stats["idle_disk"]
    assert stats["background_spills_kept_hot"] == 1
    entry = apc._trie.get(key, [10] * 8)
    assert entry.prompt_cache and entry._apc_disk
    apc.clear(release_memory=False)


def test_an_entry_dropped_while_in_flight_leaves_no_files(tmp_path, gated_writer):
    gate, started = gated_writer
    now = [0.0]
    apc = _apc(tmp_path, now)
    key = APCKey("bg-drop")
    _fill(apc, key, now)
    assert started.wait(5)
    with apc._apc_lock:
        apc._drop_entry_locked(key, [10] * 8, apc._trie.get(key, [10] * 8))
    gate.set()
    apc._spill_jobs[0].done.wait(5)
    now[0] = 4.0
    apc.lookup(key, [99] * 4, allow_disk_restore=False)
    assert apc.apc_stats["idle_disk"]["background_spills_discarded"] == 1
    assert not [p for p in tmp_path.iterdir() if "apc-idle" in p.name]
    apc.clear(release_memory=False)


def test_a_failed_background_write_falls_back_to_the_synchronous_path(tmp_path, monkeypatch):
    """A write that keeps failing is retried synchronously, whose failure
    handling drops the entry, so the cache does not stay over budget."""
    def fail(job):
        raise OSError("disk full")

    monkeypatch.setattr(apc_v2, "_write_spill_job", fail)
    now = [0.0]
    apc = _apc(tmp_path, now)
    key = APCKey("bg-fail")
    _fill(apc, key, now)
    apc._spill_jobs[0].done.wait(5)
    now[0] = 4.0
    apc.lookup(key, [99] * 4, allow_disk_restore=False)
    stats = apc.apc_stats["idle_disk"]
    assert stats["background_spill_failures"] == 1
    assert stats["spill_failures"] >= 2  # background, then the synchronous retry
    assert stats["background_spills_submitted"] == 1  # never resubmitted
    with apc._apc_lock:
        assert apc._n_bytes <= apc.max_bytes
    apc.clear(release_memory=False)


def test_in_flight_cap_falls_back_to_a_synchronous_write(tmp_path):
    now = [0.0]
    apc = _apc(tmp_path, now, background_spill_max_bytes=1)
    key = APCKey("bg-cap")
    _fill(apc, key, now)
    stats = apc.apc_stats["idle_disk"]
    assert stats["background_spill_sync_fallbacks"] >= 1
    assert stats["background_spills_submitted"] == 0
    assert stats["pressure_spills"] == 1  # written inside the store, as before
    apc.clear(release_memory=False)


def test_clear_waits_for_in_flight_writes(tmp_path, gated_writer):
    gate, started = gated_writer
    now = [0.0]
    apc = _apc(tmp_path, now)
    _fill(apc, APCKey("bg-clear"), now)
    assert started.wait(5)
    threading.Timer(0.2, gate.set).start()
    apc.clear(release_memory=False)
    assert not apc._spill_jobs
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp.safetensors")]


def test_a_victim_kept_resident_hands_its_eviction_to_another_entry(tmp_path, gated_writer):
    """Codex review: a kept-hot victim must not leave the cache over budget."""
    gate, started = gated_writer
    now = [0.0]
    apc = _apc(tmp_path, now)
    key = APCKey("bg-unmet")
    _fill(apc, key, now)
    assert started.wait(5)
    now[0] = 4.0
    hit = apc.lookup(key, [10] * 8 + [1], allow_disk_restore=False)
    hit.cache.close()
    gate.set()
    apc._spill_jobs[0].done.wait(5)
    now[0] = 5.0
    apc.lookup(key, [99] * 4, allow_disk_restore=False)  # collects, re-enforces
    for job in list(apc._spill_jobs):
        job.done.wait(5)
    now[0] = 6.0
    apc.lookup(key, [99] * 4, allow_disk_restore=False)
    with apc._apc_lock:
        assert apc._budget_nbytes_locked() <= apc._resident_entry_budget
        assert apc._n_bytes <= apc.max_bytes
    assert apc._trie.get(key, [10] * 8).prompt_cache  # the reused one stayed
    apc.clear(release_memory=False)


def test_a_replaced_in_flight_entry_stops_counting_as_reclaim(tmp_path, gated_writer):
    """Codex review: replacement via insert_cache must cancel the job."""
    gate, started = gated_writer
    now = [0.0]
    apc = _apc(tmp_path, now)
    key = APCKey("bg-replace")
    _fill(apc, key, now)
    assert started.wait(5)
    job = apc._spill_jobs[0]
    now[0] = 4.0
    apc.store(key, [10] * 8, _state(8, seed=77))  # republish the victim's tokens
    assert job.cancelled
    with apc._apc_lock:
        assert not apc._job_entry_current_locked(job)
    gate.set()
    job.done.wait(5)
    now[0] = 5.0
    apc.lookup(key, [99] * 4, allow_disk_restore=False)
    assert job.created and not any(p.exists() for p in job.created)
    apc.clear(release_memory=False)


def test_close_stops_the_writer_thread(tmp_path):
    now = [0.0]
    apc = _apc(tmp_path, now)
    _fill(apc, APCKey("bg-close"), now)
    writer = apc._spill_writer
    assert writer is not None
    apc.close()
    assert not writer._thread.is_alive()

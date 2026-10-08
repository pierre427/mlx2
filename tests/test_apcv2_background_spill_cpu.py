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


class _OpenGateOnWait(threading.Event):
    """job.done whose first wait() releases the held writer, so the victim's
    write lands exactly inside the waiting collect (the reservation's)."""

    def __init__(self, gate):
        super().__init__()
        self._gate = gate

    def wait(self, timeout=None):
        self._gate.set()
        return super().wait(timeout)


def test_a_restore_replaces_a_kept_hot_victim_synchronously(tmp_path, monkeypatch):
    """A victim kept hot inside a restore's reservation must not turn the
    reservation's room-making into background spills under the lowered limit
    (they stay in _n_bytes, so a restore that fits was deferred)."""
    gate = threading.Event()
    real = apc_v2._write_spill_job

    def held_first_write(job):
        assert gate.wait(10), "the reservation never waited for the write"
        real(job)

    now = [0.0]
    apc = _apc(tmp_path, now)
    key = APCKey("bg-restore")
    # D: a healthy disk checkpoint.
    now[0] = 1.0
    apc.store(key, [5] * 8, _state(8, seed=5))
    with apc._apc_lock:
        entry = apc._trie.get(key, [5] * 8)
        assert apc._spill_entry_locked(key, [5] * 8, entry, reason="idle")
    monkeypatch.setattr(apc_v2, "_write_spill_job", held_first_write)
    # A and B resident; storing C evicts A on the writer thread (held).
    for t, first in ((3.0, 10), (4.0, 20), (5.0, 30)):
        now[0] = t
        apc.store(key, [first] * 8, _state(8, seed=first))
    assert len(apc._spill_jobs) == 1
    victim_job = apc._spill_jobs[0]
    victim_job.done = _OpenGateOnWait(gate)
    # A is reused while its write is in flight, so it will be kept hot.
    now[0] = 6.0
    reuse = apc.lookup(key, [10] * 8 + [1], allow_disk_restore=True)
    assert reuse.hit and reuse.cached_tokens == 8
    reuse.cache.close()
    assert not victim_job.done.is_set()
    # Later writes are not held.
    monkeypatch.setattr(apc_v2, "_write_spill_job", real)

    now[0] = 7.0
    hit = apc.lookup(key, [5] * 8 + [1], allow_disk_restore=True)
    stats = apc.apc_stats["idle_disk"]
    assert stats["background_spills_kept_hot"] == 1
    try:
        # The restore fits once the other residents leave; it must not be
        # deferred because their bytes were sent to the background writer.
        assert hit.hit and hit.cached_tokens == 8, hit.miss_reason
        assert stats["restore_budget_deferrals"] == 0
        # Only the original store-time spill went to the writer thread.
        assert stats["background_spills_submitted"] == 1
        with apc._apc_lock:
            assert apc._n_bytes <= apc.max_bytes
    finally:
        if hit.cache is not None:
            hit.cache.close()
        gate.set()
        apc.close()


def _held_payload_bytes(apc):
    """Evaluated spill payload bytes still referenced by unfinished jobs."""
    held = 0
    for job in list(apc._spill_jobs):
        if job.done.is_set():
            continue
        for spill in job.files:
            for source in (spill.arrays, spill.cache):
                if isinstance(source, dict):
                    held += sum(int(a.nbytes) for a in source.values())
                elif source:
                    held += sum(int(getattr(c, "nbytes", 0)) for c in source)
    return held


def test_cancelled_in_flight_payloads_count_against_the_cap(tmp_path, monkeypatch):
    """A cancelled job's evaluated payload stays queued until the writer
    reaches it; it must keep counting against background_spill_max_bytes, or
    every cancellation admits another spill while the writer is busy."""
    gate = threading.Event()
    started = threading.Event()
    real = apc_v2._write_spill_job

    def write(job):
        if threading.current_thread().name == "apcv2-spill-writer":
            started.set()
            assert gate.wait(10), "test never released the writer"
        real(job)

    monkeypatch.setattr(apc_v2, "_write_spill_job", write)
    now = [0.0]
    # One entry's payload in flight.
    apc = _apc(tmp_path, now, background_spill_max_bytes=_entry_bytes())
    key = APCKey("bg-payload-cap")
    try:
        _fill(apc, key, now)
        assert started.wait(5)  # the writer is busy with the first victim
        assert _held_payload_bytes(apc) <= apc._background_spill_max_bytes
        republished = 0
        for round_ in range(3):
            victim = next(
                (job for job in reversed(apc._spill_jobs) if not job.cancelled), None
            )
            if victim is None:  # later victims were written synchronously
                break
            now[0] += 1.0
            # Republish the victim's tokens: the replacement cancels its job.
            apc.store(key, list(victim.tokens), _state(8, seed=100 + round_))
            assert victim.cancelled
            republished += 1
            assert _held_payload_bytes(apc) <= apc._background_spill_max_bytes, (
                f"round {round_}: writer queue holds {_held_payload_bytes(apc)} "
                f"payload bytes, cap {apc._background_spill_max_bytes}"
            )
        assert republished >= 1  # the first victim was always in flight
    finally:
        gate.set()
        for job in list(apc._spill_jobs):
            job.done.wait(5)
        apc.clear(release_memory=False)
        apc.close()


@pytest.mark.parametrize("mode", ["replace", "drop"])
def test_cancelled_spills_keep_memory_within_budget_and_cap(tmp_path, monkeypatch, mode):
    """While the writer is stalled, republishing or dropping in-flight
    victims must not grow live memory past the resident budget plus the
    in-flight cap (a replaced victim keeps its state through its job)."""
    import gc

    gate = threading.Event()
    started = threading.Event()
    real = apc_v2._write_spill_job

    def write(job):
        if threading.current_thread().name == "apcv2-spill-writer":
            started.set()
            assert gate.wait(10), "test never released the writer"
        real(job)

    def wide(seed):
        cache = KVCache()
        values = mx.full((1, 1, 8, 1024), float(seed), dtype=mx.float32)
        cache.update_and_fetch(values, values)
        mx.eval(cache.state)
        return [cache]

    monkeypatch.setattr(apc_v2, "_write_spill_job", write)
    probe = APCv2(max_size=8, layout_name="bg-spill-probe")
    probe.store(APCKey("probe"), [1] * 8, wide(1))
    entry = int(probe.nbytes)
    probe.clear(release_memory=False)
    now = [0.0]
    apc = APCv2(
        max_size=64,
        max_bytes=2 * entry,
        layout_name="bg-spill-v1",
        idle_disk_seconds=3600,
        idle_disk_dir=str(tmp_path),
        now_fn=lambda: now[0],
        background_spill=True,
        background_spill_max_bytes=2 * entry,
    )
    key = APCKey(f"bg-cancel-{mode}")
    try:
        for t, first in ((1.0, 10), (2.0, 20), (3.0, 30)):
            now[0] = t
            apc.store(key, [first] * 8, wide(first))
        assert started.wait(5)
        gc.collect()
        base = mx.get_active_memory()
        for round_ in range(8):
            victim = next(
                (job for job in reversed(apc._spill_jobs) if not job.cancelled), None
            )
            if victim is None:  # later victims were written synchronously
                break
            now[0] += 1.0
            if mode == "replace":
                apc.store(key, list(victim.tokens), wide(100 + round_))
            else:
                with apc._apc_lock:
                    apc._drop_entry_locked(key, list(victim.tokens), victim.entry)
                apc.store(key, [100 + round_] * 8, wide(100 + round_))
            assert victim.cancelled
            del victim
        gc.collect()
        grown = mx.get_active_memory() - base
        assert grown <= apc._background_spill_max_bytes, (
            f"{grown / entry:.1f} entries of live memory beyond the fill"
        )
    finally:
        gate.set()
        for job in list(apc._spill_jobs):
            job.done.wait(5)
        apc.clear(release_memory=False)
        apc.close()


def _shared_checkpoint_pair(apc, key):
    """A prompt boundary (8 tokens) and its finished lane (12 tokens) that
    alias the lane's two recurrent checkpoint snapshots: the boundary pays
    for them and the finished lane is charged only its increment."""
    from mlx2.runtime.models.cache import ArraysCache

    kv, gdn = KVCache(), ArraysCache(1)
    for position, seed in ((4, 1), (8, 2)):
        values = mx.zeros((1, 1, 4, 64))
        kv.update_and_fetch(values, values)
        gdn[0] = mx.full((1, 8, 128, 128), float(seed))  # 0.5 MiB fp32
        mx.eval(kv.state, gdn[0])
        gdn.state_checkpoint([position], force=True)
    mx.eval(gdn._checkpoints)
    assert apc.store(key, list(range(8)), [kv, gdn]).stored
    values = mx.zeros((1, 1, 4, 64))
    kv.update_and_fetch(values, values)
    gdn[0] = mx.full((1, 8, 128, 128), 3.0)
    mx.eval(kv.state, gdn[0])
    assert apc.store(key, list(range(12)), [kv, gdn]).stored


def test_a_cancelled_child_spill_charges_the_shared_checkpoints_it_keeps(
    tmp_path, monkeypatch
):
    """Codex review: a COW child's charge excludes the checkpoint snapshots
    its boundary pays for, but its queued payload holds them.  Cancelled
    while the writer is stalled, and with the paying boundary gone, the
    queue alone keeps them alive: the in-flight cap must count them, or
    further spills are admitted past background_spill_max_bytes.  While the
    boundary still holds them they are counted once, in its resident
    charge."""
    import gc

    gate = threading.Event()
    started = threading.Event()
    real = apc_v2._write_spill_job

    def write(job):
        if threading.current_thread().name == "apcv2-spill-writer":
            started.set()
            assert gate.wait(10), "test never released the writer"
        real(job)

    monkeypatch.setattr(apc_v2, "_write_spill_job", write)
    small = _entry_bytes()
    gc.collect()
    base = mx.get_active_memory()
    apc = APCv2(
        max_size=8,
        layout_name="bg-spill-cow",
        idle_disk_seconds=3600,
        idle_disk_dir=str(tmp_path),
        background_spill=True,
    )
    key = APCKey("bg-cow")
    try:
        _shared_checkpoint_pair(apc, key)
        with apc._apc_lock:
            payer = apc._trie.get(key, list(range(8)))
            child = apc._trie.get(key, list(range(12)))
            shared = int(child._apc_unpaid_nbytes)
            assert shared >= 1 << 20  # both snapshots are charged to the payer
            # Room for the child's charge plus one small spill, not for the
            # snapshots as well.
            apc._background_spill_max_bytes = int(child.nbytes) + small
            job = apc._prepare_spill_job_locked(key, list(range(12)), child)
            assert apc._submit_background_spill_locked(job)
        assert started.wait(5)  # the writer holds the child's whole payload
        with apc._apc_lock:
            apc._drop_entry_locked(key, list(range(12)), child)
            assert job.cancelled
        gc.collect()
        held = mx.get_active_memory() - base
        with apc._apc_lock:
            accounted = apc.nbytes + apc._queued_spill_nbytes_locked()
        # The payer still holds and pays for the snapshots: counted once.
        assert abs(held - accounted) <= 256 << 10, (held, accounted)
        with apc._apc_lock:
            apc._drop_entry_locked(key, list(range(8)), payer)
            assert apc.nbytes == 0 and not apc._shared_buffers
        del payer, child
        gc.collect()
        held = mx.get_active_memory() - base
        with apc._apc_lock:
            queued = apc._queued_spill_nbytes_locked()
        assert held <= apc.nbytes + queued + (256 << 10), (
            f"queue keeps {held} bytes alive, ledger counts {queued}"
        )
        apc.store(key, [70] * 8, _state(8, seed=70))
        with apc._apc_lock:
            entry = apc._trie.get(key, [70] * 8)
            further = apc._prepare_spill_job_locked(key, [70] * 8, entry)
            assert not apc._submit_background_spill_locked(further), (
                "a spill was admitted past the in-flight cap"
            )
    finally:
        gate.set()
        for pending in list(apc._spill_jobs):
            pending.done.wait(5)
        apc.clear(release_memory=False)
        apc.close()

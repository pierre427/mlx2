"""Real serving I/O ownership under hangs; all tensor operations use CPU."""

import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx2.runtime import apc_v2
from mlx2.runtime import weight_stream as ws
from mlx2.runtime.apc_v2 import APCKey, APCv2
from mlx2.runtime.io_ownership import storage_owner
from mlx2.runtime.models.cache import KVCache


def _state(seed):
    cache = KVCache()
    values = mx.arange(seed, seed + 8, dtype=mx.float32).reshape(1, 1, 8, 1)
    cache.update_and_fetch(values, values)
    mx.eval(cache.state)
    return [cache]


@pytest.mark.parametrize("collect", [False, True])
def test_spill_timeout_retains_payload_publishes_late_write_and_does_not_retry(
    tmp_path, monkeypatch, collect
):
    gate, started = threading.Event(), threading.Event()
    real_write = apc_v2._write_spill_job
    writes = []

    def blocked_write(job):
        writes.append(job)
        started.set()
        assert gate.wait(5)
        real_write(job)

    monkeypatch.setattr(apc_v2, "_write_spill_job", blocked_write)
    monkeypatch.setattr(apc_v2, "_SPILL_SETTLE_SECONDS", 0.02)
    apc = APCv2(max_size=8, max_bytes=1 << 20, layout_name="deadline",
                idle_disk_dir=str(tmp_path), idle_disk_seconds=3600,
                background_spill=True)
    key, tokens = APCKey("deadline"), [10] * 8
    apc.store(key, tokens, _state(10))
    entry = apc._trie.get(key, tokens)
    try:
        with apc._apc_lock:
            assert apc._spill_entry_locked(key, tokens, entry, reason="pressure",
                                           background=True)
        assert started.wait(1)
        job = apc._spill_job_of(entry)
        began = time.monotonic()
        with apc._apc_lock:
            if collect:
                assert apc._collect_spills_locked(wait=True, reenforce=False)
            else:
                assert not apc._spill_entry_locked(key, tokens, entry, reason="pressure")
            assert apc._budget_nbytes_locked() == apc._n_bytes
            assert not apc._spill_entry_locked(key, tokens, entry, reason="pressure")
        assert time.monotonic() - began < 0.5
        assert job.timed_out and job.cancelled and job.payload_held()
        assert entry.prompt_cache and not getattr(entry, "_apc_disk", None)
        assert len(writes) == 1
        assert apc.apc_stats["idle_disk"]["background_spill_timeouts"] == 1
        gate.set()
        assert job.done.wait(2)
        with apc._apc_lock:
            apc._collect_spills_locked()
        # The timeout only stopped the wait: the one finished write is
        # published (Sweep 2026-10-16 APC-1), never redone or discarded.
        assert getattr(entry, "_apc_disk", None) and not entry.prompt_cache
        assert len(writes) == 1
        stats = apc.apc_stats["idle_disk"]
        assert stats["late_spills_published"] == 1
        assert stats["background_spills_discarded"] == 0
        assert not stats["background_spill_writes_blocked"]
        hit = apc.lookup(key, tokens + [11])
        assert hit.hit
        hit.cache.close()
    finally:
        gate.set()
        apc.close()


class _BlockedHandles:
    def __init__(self):
        self.started = threading.Event()
        self.gate = threading.Event()
        self.finished = threading.Event()
        self.closed = threading.Event()
        self.late_view = None

    def check(self, path):
        assert not self.closed.is_set()

    def pread(self, path, offset, length):
        self.started.set()
        assert self.gate.wait(5)
        assert not self.closed.is_set(), "descriptor closed while owned by a reader"
        self.finished.set()
        return bytes(length)

    def readinto(self, path, offset, view):
        self.late_view = view
        self.pread(path, offset, len(view))
        view[:] = b"\x7f" * len(view)

    def close(self):
        self.closed.set()


_POOL_OWNER = storage_owner("weights", "deadline-test")


def _pool(owner=_POOL_OWNER):
    return ws._ReadPool(max_workers=2, thread_name_prefix="deadline-test", owner=owner)


def test_single_expert_read_times_out_and_manager_close_defers_handles(monkeypatch):
    monkeypatch.setattr(ws, "READ_TIMEOUT_SECONDS", 0.02)
    handles, pool = _BlockedHandles(), _pool()
    location = ws.TensorLocation(Path("shard"), "F32", (2, 4), 0, 32)
    reader = ws.ExpertSliceReader([ws.ExpertSliceSpec.single("weight", location)],
                                 handles=handles, pool=pool, window=2)
    manager = ws.ExpertStreamManager(plan=None, stats=ws.StreamStats(), pool=pool,
                                    caches={}, handles=handles)
    try:
        with pytest.raises(ws.WeightReadTimeout):
            reader.read_many([0])
        assert pool.counters()["weight_read_timeouts_total"] == 1
        assert handles.started.is_set()
        assert reader.outstanding_bytes() == 16
        with pytest.raises(ws.StreamingUnavailable, match="not drained"):
            reader.read_many([1])
        began = time.monotonic()
        manager.close()
        assert time.monotonic() - began < 0.5
        assert not handles.closed.is_set()
        with pytest.raises(ws.WeightReadTimeout, match="not drained"):
            _pool()
        _pool(owner=storage_owner("weights", "unrelated")).close()
        handles.gate.set()
        assert handles.closed.wait(2)
        assert pool.close(handles)
        assert reader.outstanding_bytes() == 0
        replacement = _pool()
        replacement.close()
    finally:
        handles.gate.set()
        pool.close(handles)


def test_router_hint_timeout_fails_the_route_and_close_does_not_block(monkeypatch):
    # The hint lane read on a private executor: a hung hint read neither
    # failed the route nor counted a timeout, the next decode queued behind
    # it, and close() joined the hung syscall (sweep 2026-10-16 APC-14).
    monkeypatch.setattr(ws, "READ_TIMEOUT_SECONDS", 0.05)
    handles, pool = _BlockedHandles(), ws._ReadPool(2, "hint-test", timeout=0.05)
    location = ws.TensorLocation(Path("shard"), "F32", (8, 4), 0, 128)
    reader = ws.ExpertSliceReader([ws.ExpertSliceSpec.single("weight", location)],
                                 handles=handles, pool=pool, window=2)
    stats = ws.StreamStats()
    cache = ws.ExpertLRU(reader, capacity_experts=4, stats=stats,
                         group="layers.0", path="layers.0.down_proj")
    manager = ws.ExpertStreamManager(
        plan=SimpleNamespace(top_k=2), stats=stats, pool=pool,
        caches={cache.path: cache}, handles=handles, router_hint="next_layer_reuse",
    )
    try:
        manager._router_hints.schedule("layers.0", [3])
        assert handles.started.wait(1)
        with pytest.raises(ws.WeightReadTimeout):
            manager._router_hints.consume("layers.0", cache.path, [3])
        counters = pool.counters()
        assert counters["weight_read_failed"] == 1
        assert counters["weight_read_timeouts_total"] == 1
        began = time.monotonic()
        with pytest.raises(ws.WeightReadTimeout):
            cache.acquire([5])
        manager.close()
        assert time.monotonic() - began < 0.5
        assert not handles.closed.is_set()
        handles.gate.set()
        assert handles.closed.wait(2)
    finally:
        handles.gate.set()
        pool.close(handles)


def test_spill_quarantine_survives_clear_and_is_scoped_to_its_directory(tmp_path, monkeypatch):
    gate, started = threading.Event(), threading.Event()
    real_write = apc_v2._write_spill_job

    def blocked_write(job):
        started.set()
        assert gate.wait(5)
        real_write(job)

    monkeypatch.setattr(apc_v2, "_write_spill_job", blocked_write)
    monkeypatch.setattr(apc_v2, "_SPILL_SETTLE_SECONDS", 0.01)
    monkeypatch.setattr(apc_v2, "_SPILL_DRAIN_SECONDS", 0.01)
    first = APCv2(max_size=8, max_bytes=1 << 20, layout_name="quarantine",
                  idle_disk_dir=str(tmp_path / "first"), idle_disk_seconds=3600,
                  background_spill=True)
    second = APCv2(max_size=8, max_bytes=1 << 20, layout_name="quarantine",
                   idle_disk_dir=str(tmp_path / "second"), idle_disk_seconds=3600,
                   background_spill=True)
    key, tokens = APCKey("quarantine"), list(range(8))
    job = None
    try:
        first.store(key, tokens, _state(0))
        entry = first._trie.get(key, tokens)
        with first._apc_lock:
            assert first._spill_entry_locked(key, tokens, entry, reason="pressure",
                                             background=True)
        assert started.wait(1)
        job = first._spill_job_of(entry)
        with first._apc_lock:
            assert not first._settle_spill_locked(entry)
        first.clear(release_memory=False)
        assert job.abandoned and job.payload_held()
        # Same directory: the quarantine survives clear(), so no second write
        # queues behind the stuck one, and a replacement owner is refused.
        first.store(key, tokens, _state(0))
        with first._apc_lock:
            assert not first._spill_entry_locked(
                key, tokens, first._trie.get(key, tokens), reason="pressure",
                background=True,
            )
        assert first.apc_stats["idle_disk"]["background_spill_writes_blocked"]
        with pytest.raises(ws.WeightReadTimeout, match="not drained"):
            APCv2(max_size=8, max_bytes=1 << 20, layout_name="quarantine",
                  idle_disk_dir=str(tmp_path / "first"), idle_disk_seconds=3600)
        # Another directory keeps spilling (Sweep 2026-10-16 APC-1).
        second.store(key, tokens, _state(0))
        with second._apc_lock:
            assert second._spill_entry_locked(
                key, tokens, second._trie.get(key, tokens), reason="pressure",
                background=True,
            )
        assert not second.apc_stats["idle_disk"]["background_spill_writes_blocked"]
        gate.set()
        assert job.done.wait(2)
        assert not list((tmp_path / "first").glob("apc-idle-*"))
    finally:
        gate.set()
        if job is not None:
            assert job.done.wait(2)
        first.close()
        second.close()


def test_dense_timeout_never_materializes_or_reuses_late_write_buffer(monkeypatch):
    monkeypatch.setattr(ws, "READ_TIMEOUT_SECONDS", 0.02)
    handles, pool = _BlockedHandles(), _pool()
    reader = ws.ProjectionReader(
        {"weight": ws.TensorLocation(Path("shard"), "F32", (8,), 0, 32)},
        handles=handles, pool=pool, stats=ws.StreamStats(),
    )
    try:
        with pytest.raises(ws.WeightReadTimeout):
            reader.load()
        owned = handles.late_view
        assert owned is not None
        with pytest.raises(ws.WeightReadTimeout):
            reader.load()
        assert handles.late_view is owned
        ws._close_read_pool(pool, handles)
        assert not handles.closed.is_set()
        handles.gate.set()
        assert handles.closed.wait(2)
        assert bytes(owned) == b"\x7f" * 32
    finally:
        handles.gate.set()
        pool.close(handles)


def test_abandoned_expert_iterator_accounts_running_reads_until_drain():
    handles, pool = _BlockedHandles(), _pool()
    location = ws.TensorLocation(Path("shard"), "F32", (2, 4), 0, 32)
    reader = ws.ExpertSliceReader([ws.ExpertSliceSpec.single("weight", location)],
                                 handles=handles, pool=pool, window=2)
    real = reader.read_bytes
    reader.read_bytes = lambda expert: ((bytes(16),),) if expert == 0 else real(expert)
    iterator = reader.iter_many([0, 1])
    try:
        assert next(iterator)[0] == 0
        assert handles.started.wait(1)
        iterator.close()
        assert reader.outstanding_bytes() == 16
        # Abandonment is not an I/O failure: the pool still owns the running
        # read under its deadline (sweep 2026-10-16 APC-13).
        assert pool.counters()["weight_read_failed"] == 0
        pool.close(handles)
        assert not handles.closed.is_set()
        handles.gate.set()
        assert handles.closed.wait(2)
        assert reader.outstanding_bytes() == 0
    finally:
        handles.gate.set()
        iterator.close()
        pool.close(handles)


def test_materialize_error_fails_only_the_request_not_the_shared_pool():
    # A model-thread error (e.g. a device OOM) while a sibling read ran used
    # to fail the pool every layer shares, so every later page-in raised
    # until reload though no I/O deadline was hit (sweep 2026-10-16 APC-13).
    handles, pool = _BlockedHandles(), _pool()
    location = ws.TensorLocation(Path("shard"), "F32", (4, 4), 0, 64)
    readers = [
        ws.ExpertSliceReader([ws.ExpertSliceSpec.single("weight", location)],
                             handles=handles, pool=pool, window=2)
        for _ in range(2)
    ]
    real = readers[0].read_bytes

    def read_bytes(expert):
        if expert == 0:  # return once the sibling read is running
            assert handles.started.wait(1)
            return ((bytes(16),),)
        return real(expert)

    readers[0].read_bytes = read_bytes
    readers[0].materialize = lambda payload: (_ for _ in ()).throw(MemoryError("oom"))
    for reader in readers[1:]:
        reader.materialize = lambda payload: payload
    stats = ws.StreamStats()
    (first, other) = [ws.ExpertLRU(reader, capacity_experts=4, stats=stats)
                      for reader in readers]
    try:
        with pytest.raises(MemoryError):
            first.acquire([0, 1])
        assert handles.started.wait(1)
        assert readers[0].outstanding_bytes() == 16
        assert pool.counters()["weight_read_failed"] == 0
        handles.gate.set()
        assert handles.finished.wait(1)
        deadline = time.monotonic() + 2
        while readers[0].outstanding_bytes() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert readers[0].outstanding_bytes() == 0
        assert other.acquire([2])[2] is not None
        readers[0].materialize = lambda payload: payload
        assert first.acquire([3])[3] is not None
    finally:
        handles.gate.set()
        pool.close(handles)


def test_permanently_hung_read_does_not_trap_interpreter_shutdown():
    # No device import or model work in this child: demonstrate the exact
    # executor exit behavior that ThreadPoolExecutor(wait=False) cannot give.
    code = """
import threading
from mlx2.runtime.weight_stream import _ReadPool
started = threading.Event()
def hang():
    started.set()
    threading.Event().wait()
pool = _ReadPool(1, 'permanent-hang')
pool.submit(hang)
assert started.wait(1)
pool.fail('injected timeout')
pool.close()
"""
    result = subprocess.run([sys.executable, "-c", code], timeout=5, check=False,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_pool_failure_cancels_queued_work_and_closes_owner_exactly_once():
    started, gate, closed = threading.Event(), threading.Event(), threading.Event()
    calls = []

    class Owner:
        def close(self):
            calls.append("close")
            closed.set()

    owner = Owner()
    pool = ws._ReadPool(1, "idempotent-close", owner=_POOL_OWNER)

    def running():
        started.set()
        assert gate.wait(5)
        assert calls == []

    first = pool.submit(running)
    assert started.wait(1)
    queued = pool.submit(lambda: calls.append("queued ran"))
    try:
        pool.fail(ws.WeightReadTimeout("injected"))
        pool.fail(ws.WeightReadTimeout("injected again"))
        pool.close(owner)
        pool.close(owner)
        assert queued.cancelled() and not first.done()
        assert pool.counters()["weight_read_timeouts_total"] == 1
        with pytest.raises(ws.WeightReadTimeout, match="not drained"):
            _pool()
        gate.set()
        assert closed.wait(2)
        pool.close(owner)
        assert calls == ["close"]
    finally:
        gate.set()
        pool.close(owner)


def test_initial_spill_uses_evaluated_worker_payload_and_publishes_late_write(tmp_path, monkeypatch):
    gate, started = threading.Event(), threading.Event()
    real = apc_v2._write_spill_job
    jobs = []
    owner_thread = threading.current_thread()

    def blocked(job):
        assert threading.current_thread() is not owner_thread
        assert all(spill.cache is None for spill in job.files)
        jobs.append(job)
        started.set()
        assert gate.wait(5)
        real(job)

    monkeypatch.setattr(apc_v2, "_write_spill_job", blocked)
    monkeypatch.setattr(apc_v2, "_SPILL_WRITE_SECONDS", 0.02)
    apc = APCv2(max_size=8, max_bytes=1 << 20, layout_name="initial-write",
                idle_disk_dir=str(tmp_path), idle_disk_seconds=3600)
    key, tokens = APCKey("initial-write"), [3] * 8
    apc.store(key, tokens, _state(3))
    entry = apc._trie.get(key, tokens)
    try:
        began = time.monotonic()
        with apc._apc_lock:
            assert not apc._spill_entry_locked(key, tokens, entry, reason="idle")
        assert time.monotonic() - began < 0.5
        assert started.is_set() and jobs[0].payload_held()
        assert jobs[0].timed_out and entry.prompt_cache
        assert not getattr(entry, "_apc_disk", None)
        gate.set()
        assert jobs[0].done.wait(2)
        with apc._apc_lock:
            apc._collect_spills_locked()
        # Published once it returned, still the stored entry (APC-1).
        assert getattr(entry, "_apc_disk", None) and not entry.prompt_cache
        assert len(jobs) == 1
    finally:
        gate.set()
        apc.close()


def test_refault_deadline_retains_staging_and_never_decodes_late_payload(tmp_path, monkeypatch):
    gate, started = threading.Event(), threading.Event()
    apc = APCv2(max_size=8, max_bytes=1 << 20, layout_name="refault-read",
                idle_disk_dir=str(tmp_path), idle_disk_seconds=3600)
    key, tokens = APCKey("refault-read"), [4] * 8
    apc.store(key, tokens, _state(4))
    entry = apc._trie.get(key, tokens)
    with apc._apc_lock:
        assert apc._spill_entry_locked(key, tokens, entry, reason="idle")
    real_read = apc._read_restore_payload
    reads = []

    def blocked(*args):
        reads.append(args)
        started.set()
        assert gate.wait(5)
        return real_read(*args)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("late bytes reached model deserialization")

    monkeypatch.setattr(apc, "_read_restore_payload", blocked)
    monkeypatch.setattr(apc_v2, "_RESTORE_READ_SECONDS", 0.02)
    monkeypatch.setattr(apc_v2, "load_prompt_cache", forbidden)
    try:
        began = time.monotonic()
        assert not apc.lookup(key, tokens + [5]).hit
        assert started.is_set() and time.monotonic() - began < 0.5
        assert apc._restore_staging_bytes > 0
        assert not entry.prompt_cache and entry._apc_disk
        assert not apc.lookup(key, tokens + [5]).hit
        assert len(reads) == 1
        with pytest.raises(ws.WeightReadTimeout):
            ws._ReadPool(1, "replacement-during-refault", owner=apc._disk_owner)
        gate.set()
        deadline = time.monotonic() + 2
        while apc._restore_staging_bytes and time.monotonic() < deadline:
            time.sleep(0.005)
        assert apc._restore_staging_bytes == 0
        assert not entry.prompt_cache and entry._apc_disk
        assert not apc.lookup(key, tokens + [5]).hit
        assert len(reads) == 1
    finally:
        gate.set()
        apc.close()


def test_refault_deserializes_on_caller_and_rejects_oversize_bytes(tmp_path, monkeypatch):
    apc = APCv2(max_size=8, max_bytes=1 << 20, layout_name="refault-cap",
                idle_disk_dir=str(tmp_path), idle_disk_seconds=3600,
                persistent_block_bytes=0)
    key, tokens = APCKey("refault-cap"), [6] * 8
    apc.store(key, tokens, _state(6))
    entry = apc._trie.get(key, tokens)
    with apc._apc_lock:
        assert apc._spill_entry_locked(key, tokens, entry, reason="idle")
    real_load = apc_v2.load_prompt_cache
    owner = threading.current_thread()
    decoded = []

    def load(payload):
        assert threading.current_thread() is owner
        assert hasattr(payload, "read")
        decoded.append(True)
        return real_load(payload)

    monkeypatch.setattr(apc_v2, "load_prompt_cache", load)
    try:
        hit = apc.lookup(key, tokens + [7])
        assert hit.hit and decoded
        hit.cache.close()
        with apc._apc_lock:
            assert apc._spill_entry_locked(key, tokens, entry, reason="idle")
        Path(entry._apc_disk["target"]).write_bytes(b"x" * 4096)
        monkeypatch.setattr(apc_v2, "_RESTORE_HEADER_BYTES", 1)
        decoded.clear()
        assert not apc.lookup(key, tokens + [7]).hit
        assert not decoded and apc._restore_staging_bytes == 0
    finally:
        apc.close()


def test_expert_deadline_fires_while_consumer_is_paused(monkeypatch):
    monkeypatch.setattr(ws, "READ_TIMEOUT_SECONDS", 0.02)
    gate, started = threading.Event(), threading.Event()
    pool = ws._ReadPool(2, "paused-consumer", timeout=ws.READ_TIMEOUT_SECONDS)
    reader = object.__new__(ws.ExpertSliceReader)
    reader._pool, reader.window, reader._outstanding = pool, 2, 0
    reader.expert_bytes = 4

    def read(expert):
        if expert:
            started.set()
            assert gate.wait(5)
        return (bytes(4),)

    reader.read_bytes = read
    iterator = reader.iter_many([0, 1])
    try:
        assert next(iterator)[0] == 0
        assert started.wait(1)
        end = time.monotonic() + 1
        while pool.failure is None and time.monotonic() < end:
            time.sleep(0.002)
        assert pool.failure
        gate.set()
        with pytest.raises(ws.WeightReadTimeout):
            next(iterator)
    finally:
        gate.set()
        iterator.close()
        pool.close()
    end = time.monotonic() + 1
    while pool.counters()["weight_read_inflight"] and time.monotonic() < end:
        time.sleep(0.002)
    assert pool.counters()["weight_read_inflight"] == 0


def test_late_worker_completion_refuses_result_even_if_watchdog_is_delayed(monkeypatch):
    monkeypatch.setattr(ws._ReadPool, "_watch", lambda _self: None)
    gate = threading.Event()
    pool = ws._ReadPool(1, "delayed-watchdog", timeout=0.01)
    future = pool.submit(lambda: gate.wait(5))
    try:
        time.sleep(0.02)
        gate.set()
        with pytest.raises(ws.WeightReadTimeout):
            future.result(timeout=1)
        assert pool.failure
    finally:
        gate.set()
        pool.close()


def test_failed_spill_worker_drops_traceback_payload_without_a_second_job(tmp_path, monkeypatch):
    import gc
    import weakref
    from types import SimpleNamespace

    class Payload:
        pass

    payload = Payload()
    reference = weakref.ref(payload)
    spill = apc_v2._SpillFile("target", tmp_path / "unused", "identity", arrays={"held": payload})
    job = apc_v2._SpillJob(None, (), SimpleNamespace(), [spill], False, 0, None, None, 0, 0)

    def fail(job):
        local_payload = job.files[0].arrays["held"]
        assert local_payload is not None
        raise OSError("injected failed serialization")

    monkeypatch.setattr(apc_v2, "_write_spill_job", fail)
    writer = apc_v2._SpillWriter()
    try:
        writer.submit(job)
        assert job.done.wait(1)
        assert "injected failed serialization" in str(job.error)
        del payload, spill, job
        gc.collect()
        assert reference() is None
    finally:
        assert writer.close(timeout=1)

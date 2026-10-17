"""Reachable startup, metadata and final-close waits retain their real owners."""

import json
import os
import subprocess
import sys
import threading
import time

import pytest
from test_peer_survey_persistence_cpu import _fixture, _persistent

from mlx2.runtime import apc_v2, expert_atlas, io_ownership, weight_stream
from mlx2.runtime.io_ownership import (
    CloseAction,
    QuarantineCapReached,
    ReadPool,
    ReadTimeout,
    storage_owner,
)
from mlx2.runtime.models import qwen4_ple_nvme as ple


def _eventually(predicate):
    until = time.monotonic() + 3
    while not predicate() and time.monotonic() < until:
        threading.Event().wait(0.002)
    assert predicate()


def _drained():
    with io_ownership._READ_POOLS_LOCK:
        return not io_ownership._QUARANTINED_READ_POOLS


def _shard(tmp_path):
    path = tmp_path / "model.safetensors"
    header = json.dumps({"weight": {"dtype": "F32", "shape": [2, 4],
                                    "data_offsets": [0, 32]}}).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header + bytes(32))
    return path


def test_late_final_close_is_never_retried_or_replaced(tmp_path, monkeypatch):
    monkeypatch.setattr(io_ownership, "FINALIZE_TIMEOUT_SECONDS", 0.02)
    fd = os.open(tmp_path / "owner", os.O_CREAT | os.O_RDWR, 0o600)
    entered, release = threading.Event(), threading.Event()
    calls = []

    def close():
        calls.append(fd)
        os.close(fd)
        entered.set()
        assert release.wait(3)

    pool = ReadPool(1, "close-deadline", owner=("close", "deadline"))
    owner = CloseAction(close)
    try:
        began = time.monotonic()
        assert not pool.close(owner)
        assert time.monotonic() - began < 0.5 and entered.is_set()
        # A close may already have released its number before hanging. A
        # repeated close must never call the finalizer on that number again.
        replacement_fd = os.open(tmp_path / "new-owner", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            assert not pool.close(owner)
            os.fstat(replacement_fd)
            assert calls == [fd]
            with pytest.raises(ReadTimeout, match="not drained"):
                ReadPool(1, "forbidden-replacement", owner=("close", "deadline"))
            ReadPool(1, "unrelated-owner", owner=("close", "other")).close()
        finally:
            os.close(replacement_fd)
    finally:
        release.set()
        assert pool.close(owner)
        _eventually(_drained)


def _stuck_pool(owner, release, workers=1):
    pool = ReadPool(workers, f"stuck-{owner}", timeout=0.02, owner=owner)
    for _ in range(workers):
        pool.submit(release.wait, 5)
    _eventually(lambda: pool.failure is not None)
    return pool


def test_stuck_pool_blocks_only_its_own_owner():
    # Sweep 2026-10-16 APC-1/APC-3 (Pierre's decision): quarantine is scoped
    # to the storage owner, not the process.
    release = threading.Event()
    try:
        _stuck_pool("storage-a", release)
        with pytest.raises(ReadTimeout, match="not drained for owner 'storage-a'"):
            ReadPool(1, "same-owner", owner="storage-a")
        ReadPool(1, "unrelated-owner", owner="storage-b").close()
        ReadPool(1, "anonymous").close()
    finally:
        release.set()
        _eventually(_drained)
    ReadPool(1, "after-drain", owner="storage-a").close()


def test_hung_worker_cap_refuses_every_owner(monkeypatch):
    monkeypatch.setattr(io_ownership, "MAX_QUARANTINED_WORKERS", 3)
    release = threading.Event()
    try:
        _stuck_pool("storage-a", release, workers=2)
        ReadPool(1, "below-cap", owner="storage-c").close()
        _stuck_pool("storage-b", release)
        assert io_ownership.quarantined_workers() == 3
        with pytest.raises(QuarantineCapReached, match="MAX_QUARANTINED_WORKERS=3"):
            ReadPool(1, "unrelated-at-cap", owner="storage-c")
        with pytest.raises(QuarantineCapReached):
            ReadPool(1, "anonymous-at-cap")
    finally:
        release.set()
        _eventually(_drained)
    ReadPool(1, "after-drain", owner="storage-c").close()


def test_permanently_blocked_finalizer_does_not_block_process_exit():
    code = """
import threading
from mlx2.runtime.io_ownership import CloseAction, ReadPool
pool = ReadPool(1, 'permanent-close')
assert not pool.close(CloseAction(lambda: threading.Event().wait()), timeout=.01)
print('caller-returned', flush=True)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, timeout=5, check=True)
    assert "caller-returned" in result.stdout


def test_failed_finalizer_is_not_retried_and_refuses_replacement():
    code = """
from mlx2.runtime.io_ownership import CloseAction, ReadPool, ReadTimeout
calls = []
def uncertain_close():
    calls.append(1)
    raise OSError('close result is uncertain')
pool = ReadPool(1, 'failed-close', owner='storage')
owner = CloseAction(uncertain_close)
assert not pool.close(owner)
assert not pool.close(owner)
assert calls == [1]
ReadPool(1, 'unrelated', owner='other-storage').close()
try:
    ReadPool(1, 'replacement', owner='storage')
except ReadTimeout:
    print('replacement-refused', flush=True)
else:
    raise AssertionError('uncertain descriptor owner was discarded')
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, timeout=5, check=True)
    assert "replacement-refused" in result.stdout


def test_bound_shard_startup_timeout_closes_late_open(tmp_path, monkeypatch):
    path = _shard(tmp_path)
    real_open = os.open
    entered, release = threading.Event(), threading.Event()
    descriptors = []
    monkeypatch.setattr(weight_stream, "READ_TIMEOUT_SECONDS", 0.02)

    def open_file(name, *args, **kwargs):
        fd = real_open(name, *args, **kwargs)
        if str(name) == str(path):
            descriptors.append(fd)
            entered.set()
            assert release.wait(3)
        return fd

    monkeypatch.setattr(weight_stream.os, "open", open_file)
    try:
        with pytest.raises(ReadTimeout):
            weight_stream.BoundShards(tmp_path, [path.name])
        assert entered.is_set()
        os.fstat(descriptors[0])
        with pytest.raises(ReadTimeout, match="not drained"):
            ReadPool(1, "startup-replacement", owner=storage_owner("weights", tmp_path))
    finally:
        release.set()
        _eventually(_drained)
    with pytest.raises(OSError):
        os.fstat(descriptors[0])


@pytest.mark.parametrize("post_read", [False, True])
def test_projection_fstat_deadline_prevents_publication(tmp_path, monkeypatch, post_read):
    path = _shard(tmp_path)
    monkeypatch.setattr(weight_stream, "READ_TIMEOUT_SECONDS", 0.02)
    shards = weight_stream.BoundShards(tmp_path, [path.name])
    pool = ReadPool(1, "projection-payload", timeout=0.2)
    stats = weight_stream.StreamStats()
    reader = weight_stream.ProjectionReader({"weight": shards.index.get("weight")},
        handles=shards, pool=pool, stats=stats)
    real_stat = os.fstat
    fd = shards._fd(path)
    entered, release = threading.Event(), threading.Event()
    calls = []

    def blocked_stat(number):
        if number == fd:
            calls.append(number)
            if len(calls) == (2 if post_read else 1):
                entered.set()
                assert release.wait(3)
        return real_stat(number)

    monkeypatch.setattr(weight_stream.os, "fstat", blocked_stat)
    try:
        with pytest.raises(ReadTimeout):
            reader.load()
        assert entered.is_set()
        assert stats.page_ins == 0
        shards.close()
        real_stat(fd)
    finally:
        release.set()
        pool.close()
        _eventually(_drained)
        shards.close()
    with pytest.raises(OSError):
        real_stat(fd)


@pytest.mark.parametrize("stage", ["_acquire_persist_lock", "_secure_persist_tree",
                                    "_rescan_persisted_entries"])
def test_apc_startup_deadline_keeps_unpublished_owner_until_drain(tmp_path, monkeypatch, stage):
    key, *_ = _fixture()
    entered, release = threading.Event(), threading.Event()
    original = getattr(apc_v2.APCv2, stage)
    owners = []

    def blocked(owner):
        owners.append(owner)
        entered.set()
        assert release.wait(3)
        return original(owner)

    monkeypatch.setattr(apc_v2, "_CONTROL_IO_SECONDS", 0.02)
    monkeypatch.setattr(apc_v2.APCv2, stage, blocked)
    try:
        with pytest.raises(ReadTimeout):
            _persistent(tmp_path, key)
        assert entered.is_set() and owners[0]._closed
        with pytest.raises(ReadTimeout, match="not drained"):
            _persistent(tmp_path, key)
    finally:
        release.set()
        _eventually(_drained)
    assert owners[0]._persist_lock_file is None
    monkeypatch.setattr(apc_v2.APCv2, stage, original)
    replacement = _persistent(tmp_path, key)
    replacement.close(release_memory=False)


def test_apc_final_unlock_deadline_retains_lock_until_return(tmp_path, monkeypatch):
    key, *_ = _fixture()
    apc = _persistent(tmp_path, key)
    entered, release = threading.Event(), threading.Event()
    original = apc_v2.fcntl.flock
    calls = []

    def unlock(fd, flags):
        if flags == apc_v2.fcntl.LOCK_UN:
            calls.append(fd)
            entered.set()
            assert release.wait(3)
        return original(fd, flags)

    monkeypatch.setattr(io_ownership, "FINALIZE_TIMEOUT_SECONDS", 0.02)
    monkeypatch.setattr(apc_v2.fcntl, "flock", unlock)
    try:
        began = time.monotonic()
        apc.close(release_memory=False)
        assert time.monotonic() - began < 0.5 and entered.is_set()
        apc.close(release_memory=False)
        assert apc._metadata_lock_retained and apc._persist_lock_file is not None
        assert len(calls) == 1
        with pytest.raises(ReadTimeout, match="not drained"):
            _persistent(tmp_path, key)
    finally:
        release.set()
        _eventually(_drained)
    assert apc._persist_lock_file is None


@pytest.mark.parametrize("stage", ["fstat", "fdopen"])
def test_apc_partial_lock_open_failure_closes_its_descriptor(tmp_path, monkeypatch, stage):
    key, *_ = _fixture()
    original_open, original_stat = os.open, os.fstat
    original_stage = getattr(os, stage)
    descriptors = []

    def record_open(path, *args, **kwargs):
        fd = original_open(path, *args, **kwargs)
        if str(path).endswith(".mlx2-apcv2.lock"):
            descriptors.append(fd)
        return fd

    def fail(fd, *args, **kwargs):
        if fd in descriptors:
            raise OSError("injected partial lock open")
        return original_stage(fd, *args, **kwargs)

    monkeypatch.setattr(apc_v2.os, "open", record_open)
    monkeypatch.setattr(apc_v2.os, stage, fail)
    with pytest.raises(RuntimeError, match="lock is already owned"):
        _persistent(tmp_path, key)
    assert len(descriptors) == 1
    with pytest.raises(OSError):
        original_stat(descriptors[0])
    assert _drained()


def test_atlas_flush_deadline_cannot_credit_late_persistence(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = expert_atlas.write_atlas
    monkeypatch.setattr(expert_atlas, "ATLAS_IO_SECONDS", 0.02)
    collector = expert_atlas.AtlasCollector(sink=tmp_path / "atlas", checkpoint_every=0)
    collector.bind(num_layers=1, num_units=2)
    collector.observe(0, [1])

    def blocked(*args):
        entered.set()
        assert release.wait(3)
        return original(*args)

    monkeypatch.setattr(expert_atlas, "write_atlas", blocked)
    try:
        with pytest.raises(ReadTimeout):
            collector.flush()
        assert entered.is_set() and collector.persisted_observations == 0
        assert collector.counters()["atlas_io_failed"] == 1
        collector.close()
        with pytest.raises(ReadTimeout, match="not drained"):
            ReadPool(1, "atlas-replacement",
                     owner=storage_owner("expert-atlas", tmp_path / "atlas"))
    finally:
        release.set()
        _eventually(_drained)
    assert collector.persisted_observations == 0


def test_atlas_trace_close_deadline_is_bounded_and_owned(tmp_path, monkeypatch):
    collector = expert_atlas.AtlasCollector(trace_path=tmp_path / "trace")
    collector.bind(num_layers=1, num_units=2)
    trace = collector._trace
    entered, release = threading.Event(), threading.Event()
    closes = []

    class BlockedClose:
        write = staticmethod(trace.write)

        def close(self):
            closes.append(1)
            entered.set()
            assert release.wait(3)
            trace.close()

    collector._trace = BlockedClose()
    monkeypatch.setattr(io_ownership, "FINALIZE_TIMEOUT_SECONDS", 0.02)
    try:
        began = time.monotonic()
        collector.close()
        assert time.monotonic() - began < 0.5 and entered.is_set()
        collector.close()
        assert closes == [1] and not trace.closed
        assert collector.counters()["atlas_io_failed"] == 1
        with pytest.raises(ReadTimeout, match="not drained"):
            ReadPool(1, "trace-replacement",
                     owner=storage_owner("expert-atlas", tmp_path / "trace"))
    finally:
        release.set()
        _eventually(_drained)
    assert trace.closed and collector._trace is None


def test_ple_startup_deadline_closes_late_descriptor(tmp_path, monkeypatch):
    path = tmp_path / "rows.bin"
    path.write_bytes(bytes(1024 * 20))
    monkeypatch.setenv("MLX_QWEN4_PLE_NVME_READ_POLICY", "pooled")
    monkeypatch.setattr(ple, "READ_TIMEOUT_SECONDS", 0.02)
    real_open = os.open
    entered, release = threading.Event(), threading.Event()
    descriptors = []

    def blocked(name, *args, **kwargs):
        fd = real_open(name, *args, **kwargs)
        if str(name) == str(path):
            descriptors.append(fd)
            entered.set()
            assert release.wait(3)
        return fd

    monkeypatch.setattr(ple.os, "open", blocked)
    try:
        with pytest.raises(ReadTimeout):
            ple.FileBackedShardedEmbedding(str(path), 1024, 32, 2)
        assert entered.is_set()
        os.fstat(descriptors[0])
        with pytest.raises(ReadTimeout, match="not drained"):
            ple.FileBackedShardedEmbedding(str(path), 1024, 32, 2)
    finally:
        release.set()
        _eventually(_drained)
    with pytest.raises(OSError):
        os.fstat(descriptors[0])


def test_ple_final_close_does_not_hold_replacement_constructor_lock(tmp_path, monkeypatch):
    path = tmp_path / "rows.bin"
    path.write_bytes(bytes(1024 * 20))
    monkeypatch.setenv("MLX_QWEN4_PLE_NVME_READ_POLICY", "pooled")
    table = ple.FileBackedShardedEmbedding(str(path), 1024, 32, 2)
    fd = table._fd
    real_close = os.close
    entered, release = threading.Event(), threading.Event()
    closes = []

    def blocked(number):
        if number == fd:
            closes.append(number)
            entered.set()
            assert release.wait(3)
        return real_close(number)

    monkeypatch.setattr(ple.os, "close", blocked)
    monkeypatch.setattr(io_ownership, "FINALIZE_TIMEOUT_SECONDS", 0.02)
    try:
        began = time.monotonic()
        table.close()
        assert time.monotonic() - began < 0.5 and entered.is_set()
        began = time.monotonic()
        with pytest.raises(ReadTimeout, match="not drained"):
            ple.FileBackedShardedEmbedding(str(path), 1024, 32, 2)
        assert time.monotonic() - began < 0.5
        table.close()
        assert closes == [fd]
        os.fstat(fd)
    finally:
        release.set()
        _eventually(_drained)
    with pytest.raises(OSError):
        os.fstat(fd)


def test_ple_hot_rows_inventory_has_deadline(tmp_path, monkeypatch):
    path = tmp_path / "rows.bin"
    path.write_bytes(bytes(1024 * 20))
    hot = tmp_path / "hot.txt"
    hot.write_text("0\n1\n")
    monkeypatch.setenv("MLX_QWEN4_PLE_NVME_READ_POLICY", "pooled")
    monkeypatch.setenv("MLX_QWEN4_PLE_NVME_LRU_MB", "1")
    monkeypatch.setattr(ple, "READ_TIMEOUT_SECONDS", 0.02)
    table = ple.FileBackedShardedEmbedding(str(path), 1024, 32, 2)
    original = ple._read_hot_row_ids
    entered, release = threading.Event(), threading.Event()

    def blocked(*args):
        entered.set()
        assert release.wait(3)
        return original(*args)

    monkeypatch.setattr(ple, "_read_hot_row_ids", blocked)
    try:
        with pytest.raises(ReadTimeout):
            table.preheat_from_file(str(hot))
        assert entered.is_set() and not table._lru
    finally:
        table.close()
        release.set()
        _eventually(_drained)
    assert not table._lru


def test_bounded_pool_accepts_the_next_call_once_the_previous_result_returned():
    # Future.result() wakes the caller before the pool's done-callback has
    # retired the future. A finished call must not hold the pending window.
    # Two workers: the delayed callback runs on the first one.
    pool = ReadPool(2, "back-to-back", timeout=5.0, max_pending=1)
    callback_entered, release = threading.Event(), threading.Event()
    retire = pool._completed

    def delayed_retire(future):
        callback_entered.set()
        assert release.wait(3)
        retire(future)

    pool._completed = delayed_retire
    try:
        assert pool.submit(lambda: 1).result(timeout=2) == 1
        assert callback_entered.wait(2)
        pool._completed = retire
        assert pool.call(lambda: 2) == 2
    finally:
        release.set()
        pool.close()
        _eventually(_drained)

"""Independent ownership races at an indeterminate APC manifest boundary."""

import threading
import time
from pathlib import Path

import pytest
from test_peer_apc_metadata_cpu import _assert_directory_owner_retained
from test_peer_survey_persistence_cpu import _cache, _fixture, _persistent

from mlx2.runtime import apc_v2
from mlx2.runtime.apc_v2 import APCMetadataIndeterminate


def test_concurrent_close_cannot_unlock_before_detached_writer_is_registered(tmp_path, monkeypatch):
    key, raw, _draft, _hidden, _sidecar = _fixture()
    apc = _persistent(tmp_path, key)
    apc._background_spill = True
    apc._background_spill_max_bytes = 1 << 20
    first, second = [1] * 8, [2] * 8
    payload_gate, manifest_gate, close_gate = (threading.Event() for _ in range(3))
    payload_started, close_started = threading.Event(), threading.Event()
    write, fsync = apc_v2._write_spill_job, apc._fsync_directory
    closer = None
    failures = []

    def blocked_write(job):
        if job.tokens == second:
            payload_started.set()
            assert payload_gate.wait(5)
        write(job)

    def blocked_fsync(path):
        assert manifest_gate.wait(5)
        fsync(path)

    monkeypatch.setattr(apc_v2, "_write_spill_job", blocked_write)
    monkeypatch.setattr(apc, "_fsync_directory", blocked_fsync)
    monkeypatch.setattr(apc_v2, "_MANIFEST_WRITE_SECONDS", .03)
    monkeypatch.setattr(apc_v2, "_SPILL_DRAIN_SECONDS", .03)
    try:
        for tokens in (first, second):
            apc.store(key, tokens, [_cache(raw, raw)])
        with apc._apc_lock:
            for tokens in (first, second):
                assert apc._spill_entry_locked(
                    key, tokens, apc._trie.get(key, tokens),
                    reason="pressure", background=True,
                )
        assert payload_started.wait(1)
        with apc._apc_lock, pytest.raises(APCMetadataIndeterminate):
            apc._collect_spills_locked()
        writer = apc._spill_writer
        original_close = writer.close

        def pause_close(*args, **kwargs):
            if threading.current_thread().name == "review-close-1":
                close_started.set()
                assert close_gate.wait(5)
            return original_close(*args, **kwargs)

        monkeypatch.setattr(writer, "close", pause_close)

        def first_close():
            try:
                apc.close()
            except BaseException as error:  # noqa: BLE001 - assert thread failures on caller
                failures.append(error)

        closer = threading.Thread(target=first_close, name="review-close-1")
        closer.start()
        assert close_started.wait(1)
        apc.close()
        manifest_gate.set()
        deadline = time.monotonic() + 1
        while not apc._metadata_future.done():
            assert time.monotonic() < deadline
            time.sleep(.005)
        waiter = apc._metadata_release_waiter
        if waiter is not None:
            waiter.join(.1)
        assert writer._thread.is_alive()
        assert apc._persist_lock_file is not None
        _assert_directory_owner_retained(tmp_path, key)
    finally:
        manifest_gate.set()
        payload_gate.set()
        close_gate.set()
        if closer is not None:
            closer.join(3)
            assert not closer.is_alive()
        apc.close()
        deadline = time.monotonic() + 2
        while apc._persist_lock_file is not None:
            assert time.monotonic() < deadline
            time.sleep(.005)
        assert not failures


def test_queued_resident_prefetch_cannot_mutate_after_manifest_timeout(tmp_path, monkeypatch):
    key, raw, _draft, _hidden, _sidecar = _fixture()
    apc = _persistent(tmp_path, key)
    gate = threading.Event()
    first, second = [3] * 8, [4] * 8
    fsync = apc._fsync_directory

    def blocked_fsync(path):
        assert gate.wait(5)
        fsync(path)

    monkeypatch.setattr(apc, "_fsync_directory", blocked_fsync)
    monkeypatch.setattr(apc_v2, "_MANIFEST_WRITE_SECONDS", .03)
    try:
        apc.store(key, first, [_cache(raw, raw)], session_tag=("a", "park"))
        apc.store(key, second, [_cache(raw, raw)], session_tag=("a", "queued"))
        apc.resume_session("a", "queued", ttl_seconds=300)
        pending = apc._pending_prefetch
        entry = apc._trie.get(key, second)
        pins = dict(getattr(entry, "_apc_resident_pin_expiries", {}))
        ok = apc._disk_stats["prefetch_restores_ok"]
        with pytest.raises(APCMetadataIndeterminate):
            apc.park_session("a", "park", ttl_seconds=300)
        with pytest.raises(APCMetadataIndeterminate):
            apc.service_pending_prefetch()
        assert apc._pending_prefetch is pending
        assert getattr(entry, "_apc_resident_pin_expiries", {}) == pins
        assert apc._disk_stats["prefetch_restores_ok"] == ok
    finally:
        gate.set()
        apc.close()
        deadline = time.monotonic() + 2
        while apc._persist_lock_file is not None:
            assert time.monotonic() < deadline
            time.sleep(.005)


def test_failed_payload_unlink_cannot_release_live_disk_charge(tmp_path, monkeypatch):
    key, raw, _draft, _hidden, _sidecar = _fixture()
    apc = _persistent(tmp_path, key)
    tag, tokens = ("a", "session"), list(range(8))
    try:
        apc.store(key, tokens, [_cache(raw, raw)], session_tag=tag)
        charged = apc.park_session(*tag, ttl_seconds=300)["disk_bytes"]
        entry = apc._trie.get(key, tokens)
        payload = Path(entry._apc_disk["target"])
        unlink = Path.unlink

        def refuse_payload(path, *args, **kwargs):
            if path == payload:
                raise PermissionError("injected payload unlink failure")
            return unlink(path, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "unlink", refuse_payload)
            with pytest.raises(apc_v2.APCMetadataIncomplete):
                apc.delete_session(*tag)
        assert payload.is_file()
        assert apc._disk_bytes == charged
        assert apc._disk_stats["session_deletes"] == 0
        assert apc.apc_stats["persistence"]["metadata_transaction"]["completion"] == "incomplete"
    finally:
        apc.close()

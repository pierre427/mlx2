"""Hot session accounting uses the immutable payload writer's receipt."""

import threading
import time
from pathlib import Path

import pytest
from test_peer_survey_persistence_cpu import _cache, _fixture, _persistent

from mlx2.runtime import apc_v2
from mlx2.runtime.apc_v2 import APCMetadataIndeterminate, APCv2


@pytest.mark.parametrize("mode", ["ephemeral", "persistent", "restart"])
def test_session_status_and_pin_accounting_need_no_filesystem_metadata(tmp_path, monkeypatch, mode):
    key, raw, _draft, _hidden, _sidecar = _fixture()
    tag = ("a", "session")
    tokens = list(range(8))
    apc = (_persistent(tmp_path, key) if mode != "ephemeral" else APCv2(
        max_size=8, max_bytes=1 << 20, layout_name="sidecar-refault",
        idle_disk_seconds=3600, idle_disk_dir=str(tmp_path),
    ))
    try:
        apc.store(key, tokens, [_cache(raw, raw)], session_tag=tag)
        parked = apc.park_session(*tag, ttl_seconds=300)
        expected = parked["disk_bytes"]
        assert expected > int(raw.nbytes) * 2
        if mode == "restart":
            apc.close()
            apc = _persistent(tmp_path, key)
        entry = apc._trie.get(key, tokens)
        assert entry._apc_disk["physical_bytes"] == expected

        def refuse_metadata(*args, **kwargs):
            raise AssertionError("hot session accounting touched filesystem metadata")

        with monkeypatch.context() as patch:
            patch.setattr(Path, "stat", refuse_metadata)
            patch.setattr(apc, "_disk_paths", refuse_metadata)
            assert apc.session_state(*tag)["disk_bytes"] == expected
            with apc._apc_lock:
                assert apc._entry_projected_disk_bytes(entry) == expected
                assert apc._tenant_pin_bytes_locked("a", resident=False) == expected
                assert apc._global_pin_bytes_locked() == expected
    finally:
        apc.close()


def test_external_payload_removal_does_not_release_a_live_disk_reservation(tmp_path):
    key, raw, _draft, _hidden, _sidecar = _fixture()
    apc = _persistent(tmp_path, key)
    tag = ("a", "session")
    tokens = list(range(8))
    try:
        apc.store(key, tokens, [_cache(raw, raw)], session_tag=tag)
        expected = apc.park_session(*tag, ttl_seconds=300)["disk_bytes"]
        entry = apc._trie.get(key, tokens)
        Path(entry._apc_disk["target"]).unlink()
        assert apc.session_state(*tag)["disk_bytes"] == expected
        assert not apc.lookup(key, tokens + [99]).hit
        assert apc.apc_stats["idle_disk"]["restore_digest_failures"] == 1
    finally:
        apc.close()


@pytest.mark.parametrize("stage", ["file_fsync", "replace", "directory_fsync"])
def test_manifest_deadline_is_indeterminate_retains_payload_and_directory_owner(
    tmp_path, monkeypatch, stage
):
    key, raw, _draft, _hidden, _sidecar = _fixture()
    apc = _persistent(tmp_path, key)
    tag, tokens = ("a", "session"), list(range(8))
    gate, started = threading.Event(), threading.Event()
    calls = []

    def stall():
        calls.append(threading.current_thread().name)
        started.set()
        assert gate.wait(5)

    if stage == "file_fsync":
        original = apc_v2.os.fsync

        def fsync(fd):
            if threading.current_thread().name.startswith("apcv2-metadata") and not calls:
                stall()
            return original(fd)

        monkeypatch.setattr(apc_v2.os, "fsync", fsync)
    elif stage == "replace":
        original = apc_v2.os.replace

        def replace(source, destination):
            if str(destination).endswith(".manifest.json"):
                stall()
            return original(source, destination)

        monkeypatch.setattr(apc_v2.os, "replace", replace)
    else:
        original = apc._fsync_directory

        def directory_fsync(path):
            stall()
            return original(path)

        monkeypatch.setattr(apc, "_fsync_directory", directory_fsync)
    monkeypatch.setattr(apc_v2, "_MANIFEST_WRITE_SECONDS", 0.03)
    try:
        apc.store(key, tokens, [_cache(raw, raw)], session_tag=tag)
        before = time.monotonic()
        with pytest.raises(APCMetadataIndeterminate, match="indeterminate"):
            apc.park_session(*tag, ttl_seconds=300)
        assert time.monotonic() - before < 0.5
        assert started.is_set()
        entry = apc._trie.get(key, tokens)
        assert entry.prompt_cache and entry.nbytes > 0
        payload = Path(entry._apc_disk["target"])
        assert payload.is_file()
        status = apc.apc_stats
        assert status["persistence"]["metadata_transaction"]["worker_pending"]
        assert status["persistence"]["metadata_transaction"]["reload_required"]
        assert status["idle_disk"]["persisted_writes"] == 0
        assert status["idle_disk"]["bytes_written"] == 0
        assert status["idle_disk"]["parks"] == 0
        for operation in (
            lambda: apc.lookup(key, tokens + [99]),
            lambda: apc.store(key, tokens, [_cache(raw, raw)]),
            lambda: apc.park_session(*tag, ttl_seconds=300),
            lambda: apc.delete_session(*tag),
            lambda: apc.clear(),
        ):
            with pytest.raises(APCMetadataIndeterminate):
                operation()
        report = apc.close()
        assert report["metadata_transaction"]["completion"] == "indeterminate"
        assert report["metadata_transaction"]["directory_lock_retained"]
        assert payload.is_file()
        # The global read quarantine may refuse replacement before the
        # retained directory lock is probed; both are truthful ownership gates.
        with pytest.raises(
            RuntimeError,
            match="lock is already owned|previous I/O reads have not drained",
        ):
            _persistent(tmp_path, key)
        assert len(calls) == 1
        gate.set()
        deadline = time.monotonic() + 2
        while apc._persist_lock_file is not None:
            assert time.monotonic() < deadline
            time.sleep(0.005)
        assert not apc.apc_stats["persistence"]["metadata_transaction"]["worker_pending"]
        with pytest.raises(APCMetadataIndeterminate):
            apc.lookup(key, tokens + [99])
        # The caller got an explicitly ambiguous outcome. Its late manifest
        # may be present, but only a new owner can inventory and use it.
        replacement = _persistent(tmp_path, key)
        try:
            hit = replacement.lookup(key, tokens + [99])
            assert hit.hit
            hit.cache.close()
        finally:
            replacement.close()
    finally:
        gate.set()
        apc.close()


def test_repeated_close_retains_both_metadata_and_other_spill_writer_owners(tmp_path, monkeypatch):
    key, raw, _draft, _hidden, _sidecar = _fixture()
    apc = _persistent(tmp_path, key)
    apc._background_spill = True
    apc._background_spill_max_bytes = 1 << 20
    first, second = [1] * 8, [2] * 8
    payload_gate, metadata_gate = threading.Event(), threading.Event()
    payload_started = threading.Event()
    original_write = apc_v2._write_spill_job
    original_sync = apc._fsync_directory

    def write(job):
        if job.tokens == second:
            payload_started.set()
            assert payload_gate.wait(5)
        original_write(job)

    def fsync(path):
        assert metadata_gate.wait(5)
        original_sync(path)

    monkeypatch.setattr(apc_v2, "_write_spill_job", write)
    monkeypatch.setattr(apc, "_fsync_directory", fsync)
    monkeypatch.setattr(apc_v2, "_MANIFEST_WRITE_SECONDS", 0.03)
    monkeypatch.setattr(apc_v2, "_SPILL_DRAIN_SECONDS", 0.03)
    try:
        for tokens in (first, second):
            apc.store(key, tokens, [_cache(raw, raw)])
        with apc._apc_lock:
            for tokens in (first, second):
                entry = apc._trie.get(key, tokens)
                assert apc._spill_entry_locked(key, tokens, entry, reason="pressure", background=True)
        assert payload_started.wait(1)
        with apc._apc_lock, pytest.raises(APCMetadataIndeterminate):
            apc._collect_spills_locked()
        assert len(apc._spill_jobs) == 1  # Unvisited work survives publication failure.
        apc.close()
        metadata_gate.set()
        deadline = time.monotonic() + 1
        while not apc._metadata_future.done():
            assert time.monotonic() < deadline
            time.sleep(0.005)
        apc.close()
        assert apc._persist_lock_file is not None
        with pytest.raises(RuntimeError, match="lock is already owned"):
            _persistent(tmp_path, key)
        payload_gate.set()
        deadline = time.monotonic() + 2
        while apc._persist_lock_file is not None:
            assert time.monotonic() < deadline
            time.sleep(0.005)
        replacement = _persistent(tmp_path, key)
        replacement.close()
    finally:
        payload_gate.set()
        metadata_gate.set()
        apc.close()


def test_plain_spill_timeout_releases_directory_only_after_late_writer_drains(tmp_path, monkeypatch):
    key, raw, _draft, _hidden, _sidecar = _fixture()
    apc = _persistent(tmp_path, key)
    apc._background_spill = True
    apc._background_spill_max_bytes = 1 << 20
    gate, started = threading.Event(), threading.Event()
    original = apc_v2._write_spill_job

    def write(job):
        started.set()
        assert gate.wait(5)
        original(job)

    monkeypatch.setattr(apc_v2, "_write_spill_job", write)
    monkeypatch.setattr(apc_v2, "_SPILL_DRAIN_SECONDS", 0.03)
    tokens = list(range(8))
    try:
        apc.store(key, tokens, [_cache(raw, raw)])
        with apc._apc_lock:
            assert apc._spill_entry_locked(
                key, tokens, apc._trie.get(key, tokens), reason="pressure", background=True,
            )
        assert started.wait(1)
        apc.close()
        apc.close()
        assert apc._persist_lock_file is not None
        with pytest.raises(RuntimeError, match="lock is already owned"):
            _persistent(tmp_path, key)
        gate.set()
        deadline = time.monotonic() + 2
        while apc._persist_lock_file is not None:
            assert time.monotonic() < deadline
            time.sleep(0.005)
        replacement = _persistent(tmp_path, key)
        replacement.close()
    finally:
        gate.set()
        apc.close()


@pytest.mark.parametrize("stage", ["discovery", "payload_unlink", "manifest_unlink", "directory_fsync"])
def test_delete_deadline_never_acknowledges_or_releases_charge_before_completion(
    tmp_path, monkeypatch, stage
):
    key, raw, _draft, _hidden, _sidecar = _fixture()
    apc = _persistent(tmp_path, key)
    tag, tokens = ("a", "session"), list(range(8))
    gate = threading.Event()
    apc.store(key, tokens, [_cache(raw, raw)], session_tag=tag)
    parked = apc.park_session(*tag, ttl_seconds=300)
    charged = parked["disk_bytes"]
    original_paths = apc_v2.block_file_paths
    original_unlink = Path.unlink
    original_fsync = apc._fsync_directory

    def stall():
        assert threading.current_thread().name.startswith("apcv2-metadata")
        assert gate.wait(5)

    def paths(path):
        # Even the removed-file count must use the captured inventory; only
        # the bounded deletion worker may discover payload files again.
        assert threading.current_thread().name.startswith("apcv2-metadata")
        if stage == "discovery":
            stall()
        return original_paths(path)

    def unlink(path, *args, **kwargs):
        if ((stage == "payload_unlink" and str(path).endswith(".target.safetensors"))
                or (stage == "manifest_unlink" and str(path).endswith(".manifest.json"))):
            stall()
        return original_unlink(path, *args, **kwargs)

    def fsync(path):
        if stage == "directory_fsync":
            stall()
        return original_fsync(path)

    # remove_block_file resolves its own module's discovery helper.
    from mlx2.runtime import persistent_blocks

    monkeypatch.setattr(apc_v2, "block_file_paths", paths)
    monkeypatch.setattr(persistent_blocks, "block_file_paths", paths)
    monkeypatch.setattr(Path, "unlink", unlink)
    monkeypatch.setattr(apc, "_fsync_directory", fsync)
    monkeypatch.setattr(apc_v2, "_MANIFEST_WRITE_SECONDS", 0.03)
    try:
        began = time.monotonic()
        with pytest.raises(APCMetadataIndeterminate):
            apc.delete_session(*tag)
        assert time.monotonic() - began < 0.5
        assert apc._disk_bytes == charged
        assert apc.apc_stats["idle_disk"]["session_deletes"] == 0
        assert apc.apc_stats["persistence"]["metadata_transaction"]["operation"] == "payload_delete"
        apc.close()
        # The global read quarantine may refuse replacement before the
        # retained directory lock is probed; both are truthful ownership gates.
        with pytest.raises(
            RuntimeError,
            match="lock is already owned|previous I/O reads have not drained",
        ):
            _persistent(tmp_path, key)
        gate.set()
        deadline = time.monotonic() + 2
        while apc._persist_lock_file is not None:
            assert time.monotonic() < deadline
            time.sleep(0.005)
        assert not list(tmp_path.glob("apc-idle-*"))
    finally:
        gate.set()
        apc.close()

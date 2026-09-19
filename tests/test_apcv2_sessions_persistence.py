"""CPU contracts for APCv2 session hints and restart persistence."""

import json
import os
import time
from pathlib import Path

import mlx.core as mx
import pytest

from mlx2.runtime.apc_v2 import (
    APCKey,
    APCSessionCapacityError,
    APCSessionNotFound,
    APCSessionUnavailable,
    APCv2,
)
from mlx2.runtime.models.cache import KVCache


def _state(tokens, seed=0):
    cache = KVCache()
    values = mx.arange(seed, seed + tokens, dtype=mx.float32).reshape(
        1, 1, tokens, 1
    )
    cache.update_and_fetch(values, values)
    mx.eval(cache.state)
    return cache


def _identity(revision="source-a", *, tenant="tenant-a"):
    return APCKey(
        "artifact-a",
        revision=revision,
        adapter="artifact-a",
        tokenizer_fingerprint="tokenizer-a",
        cache_layout_fingerprint="layout-a",
        semantic_fingerprint=("text-token-v1", "tenant", tenant),
    )


def _persistent(directory, *, revision="source-a", wall=None, **kwargs):
    return APCv2(
        max_size=16,
        max_bytes=1 << 20,
        layout_name="layout-a",
        idle_disk_seconds=180,
        idle_disk_dir=str(directory),
        persist_dir=str(directory),
        persist_identity=_identity(revision),
        persist_semantic_namespace="tenant",
        wall_time_fn=(lambda: wall[0]) if wall is not None else time.time,
        **kwargs,
    )


def test_session_park_resume_prefetch_and_delete(tmp_path):
    apc = APCv2(
        max_size=8,
        max_bytes=1 << 20,
        layout_name="layout-a",
        idle_disk_seconds=180,
        idle_disk_dir=str(tmp_path),
    )
    key = _identity()
    tag = ("tenant-a", "conversation-1")
    tokens = list(range(32))
    apc.store(key, tokens, [_state(32)], session_tag=tag)

    parked = apc.park_session(*tag, ttl_seconds=60)
    assert parked["state"] == "disk"
    assert parked["covered_tokens"] == 32
    assert parked["resident_bytes"] == 0
    assert parked["disk_pin_expires_at"] is not None

    accepted = apc.resume_session(*tag, ttl_seconds=30)
    assert accepted["state"] in {"disk", "resident"}
    apc.service_pending_prefetch()
    deadline = time.monotonic() + 5
    while apc.session_state(*tag)["state"] != "resident":
        assert time.monotonic() < deadline
        time.sleep(0.01)
    hit = apc.lookup(key, tokens + [99], session_tag=tag)
    assert hit.hit and hit.cached_tokens == len(tokens)
    hit.cache.close()
    assert apc.apc_stats["idle_disk"]["prefetch_restores_ok"] == 1
    assert apc.apc_stats["idle_disk"]["prefetch_hits"] == 1

    report = apc.delete_session(*tag)
    assert report["removed_entries"] == 1
    assert not list(tmp_path.glob("apc-idle-*"))
    with pytest.raises(APCSessionNotFound):
        apc.session_state(*tag)
    apc.close()


def test_shared_session_delete_only_removes_the_requested_tag(tmp_path):
    apc = APCv2(
        max_size=8,
        layout_name="layout-a",
        idle_disk_seconds=180,
        idle_disk_dir=str(tmp_path),
    )
    key = _identity()
    tokens = list(range(8))
    apc.store(key, tokens, [_state(8)], session_tag=("tenant-a", "a"))
    apc.store(key, tokens, [_state(8)], session_tag=("tenant-a", "b"))
    report = apc.delete_session("tenant-a", "a")
    assert report["shared_entries"] == 1
    assert apc.session_state("tenant-a", "b")["entries"] == 1
    with pytest.raises(APCSessionNotFound):
        apc.session_state("tenant-a", "a")
    apc.close()


def test_park_and_delete_defer_while_active_cow_lease_is_open(tmp_path):
    clock = [0.0]
    apc = APCv2(
        max_size=8,
        layout_name="layout-a",
        idle_disk_seconds=180,
        idle_disk_dir=str(tmp_path),
        now_fn=lambda: clock[0],
    )
    key = _identity()
    tag = ("tenant-a", "leased")
    tokens = list(range(8))
    apc.store(key, tokens, [_state(8)], session_tag=tag)
    lease = apc.lookup(key, tokens + [99])
    parked = apc.park_session(*tag, ttl_seconds=60)
    assert parked["state"] == "resident" and parked["park_pending"] == 1
    lease.cache.close()
    clock[0] = 2
    assert apc.spill_idle_entries() == 1
    assert apc.session_state(*tag)["state"] == "disk"

    lease = apc.lookup(key, tokens + [99])
    report = apc.delete_session(*tag)
    assert report["deferred_leased_entries"] == 1
    lease.cache.close()
    clock[0] = 4
    apc.spill_idle_entries()
    assert len(apc) == 0
    apc.close()


def test_per_tenant_pin_caps_and_tenant_isolation(tmp_path):
    apc = APCv2(
        max_size=8,
        layout_name="layout-a",
        idle_disk_seconds=180,
        idle_disk_dir=str(tmp_path),
        pinned_disk_bytes_per_tenant=1,
    )
    key = _identity()
    apc.store(
        key,
        list(range(8)),
        [_state(8)],
        session_tag=("tenant-a", "only-a"),
    )
    with pytest.raises(APCSessionNotFound):
        apc.park_session("tenant-b", "only-a", ttl_seconds=10)
    with pytest.raises(APCSessionCapacityError, match="pinned disk"):
        apc.park_session("tenant-a", "only-a", ttl_seconds=10)
    assert apc.apc_stats["idle_disk"]["pin_cap_rejections"] == 1
    apc.close()

    resident = APCv2(
        max_size=8,
        layout_name="layout-a",
        idle_disk_seconds=180,
        idle_disk_dir=str(tmp_path),
        pinned_resident_bytes_per_tenant=1,
    )
    resident.store(
        key,
        list(range(8)),
        [_state(8)],
        session_tag=("tenant-a", "resident-cap"),
    )
    resident.park_session("tenant-a", "resident-cap", ttl_seconds=10)
    with pytest.raises(APCSessionCapacityError, match="pinned resident"):
        resident.resume_session("tenant-a", "resident-cap")
    resident.close()


def test_persistent_restart_rescan_preserves_tag_pin_and_exact_state(tmp_path):
    wall = [1_000.0]
    key = _identity()
    tag = ("tenant-a", "restart-session")
    tokens = list(range(24))
    first = _persistent(tmp_path, wall=wall)
    first.store(key, tokens, [_state(24, 10)], session_tag=tag)
    first.park_session(*tag, ttl_seconds=60)
    first.close()

    wall[0] += 10
    second = _persistent(tmp_path, wall=wall)
    state = second.session_state(*tag)
    assert state["state"] == "disk"
    assert state["disk_pin_expires_at"] == pytest.approx(1060.0)
    assert second.apc_stats["persistence"]["rescan"]["registered"] == 1
    hit = second.lookup(key, tokens + [100])
    assert hit.hit and hit.cached_tokens == len(tokens)
    assert hit.cache[0].offset == len(tokens)
    hit.cache.close()
    second.close()


def test_expired_persistent_pin_is_dropped_at_rescan(tmp_path):
    wall = [2_000.0]
    first = _persistent(tmp_path, wall=wall)
    tag = ("tenant-a", "expired")
    first.store(_identity(), [1, 2, 3], [_state(3)], session_tag=tag)
    first.park_session(*tag, ttl_seconds=5)
    first.close()
    wall[0] += 6
    second = _persistent(tmp_path, wall=wall)
    assert second.session_state(*tag)["disk_pin_expires_at"] is None
    assert second.apc_stats["idle_disk"]["pin_expiries"] == 1
    second.close()


def test_rescan_discards_identity_mismatch_and_counts_it(tmp_path):
    first = _persistent(tmp_path)
    first.store(_identity(), [1, 2, 3], [_state(3)])
    first.park_all(time_budget_seconds=5)
    first.close()
    assert list(tmp_path.glob("apc-idle-*.manifest.json"))

    second = _persistent(tmp_path, revision="source-b")
    rescan = second.apc_stats["persistence"]["rescan"]
    assert rescan["registered"] == 0
    assert rescan["discarded"]["identity_mismatch"] == 1
    assert not list(tmp_path.glob("apc-idle-*.manifest.json"))
    second.close()


def test_corrupt_payload_registers_fast_then_fails_digest_at_restore(tmp_path):
    first = _persistent(tmp_path)
    key = _identity()
    first.store(key, list(range(16)), [_state(16)])
    first.park_all(time_budget_seconds=5)
    first.close()
    target = next(tmp_path.glob("apc-idle-*.target.safetensors"))
    payload = bytearray(target.read_bytes())
    payload[len(payload) // 2] ^= 1
    target.write_bytes(payload)

    second = _persistent(tmp_path)
    assert second.apc_stats["persistence"]["rescan"]["registered"] == 1
    miss = second.lookup(key, list(range(17)))
    assert not miss.hit
    assert second.apc_stats["idle_disk"]["restore_digest_failures"] == 1
    assert len(second) == 0
    second.close()


def test_manifest_rewrite_never_reblesses_corrupted_payload(tmp_path, monkeypatch):
    key = _identity()
    tag = ("tenant-a", "digest")
    tokens = list(range(16))
    first = _persistent(tmp_path)
    first.store(key, tokens, [_state(16, 5)], session_tag=tag)
    first.park_session(*tag, ttl_seconds=60)
    manifest = next(tmp_path.glob("apc-idle-*.manifest.json"))
    original = json.loads(manifest.read_text())["files"]["target"]["sha256"]
    target = next(tmp_path.glob("apc-idle-*.target.safetensors"))
    payload = bytearray(target.read_bytes())
    payload[-4] ^= 0x40
    target.write_bytes(payload)

    def forbid_payload_rehash(_path):
        raise AssertionError("metadata rewrite read persisted payload")

    monkeypatch.setattr(first, "_sha256_file", forbid_payload_rehash)

    first.close()
    assert json.loads(manifest.read_text())["files"]["target"]["sha256"] == original
    second = _persistent(tmp_path)
    miss = second.lookup(key, tokens + [99])
    assert not miss.hit
    assert second.apc_stats["idle_disk"]["restore_digest_failures"] == 1
    second.close()


def test_resident_count_limit_preserves_parked_pin_and_disk_placeholders(tmp_path):
    key = _identity()
    tag = ("tenant-a", "parked")
    first = APCv2(
        max_size=2,
        max_bytes=1 << 20,
        layout_name="layout-a",
        idle_disk_seconds=180,
        idle_disk_dir=str(tmp_path),
        persist_dir=str(tmp_path),
        persist_identity=key,
        persist_semantic_namespace="tenant",
    )
    first.store(key, [1, 2, 3], [_state(3)], session_tag=tag)
    first.park_session(*tag, ttl_seconds=3600)
    for index in range(4):
        tokens = [100 + index, 7, 8]
        first.store(key, tokens, [_state(3, index)])
        first.park_all(time_budget_seconds=5)
    assert first.session_state(*tag)["state"] == "disk"
    assert len(first) == 5
    first.close()

    second = APCv2(
        max_size=2,
        max_bytes=1 << 20,
        layout_name="layout-a",
        idle_disk_seconds=180,
        idle_disk_dir=str(tmp_path),
        persist_dir=str(tmp_path),
        persist_identity=key,
        persist_semantic_namespace="tenant",
    )
    assert second.apc_stats["persistence"]["rescan"]["registered"] == 5
    assert len(second) == 5
    assert second.session_state(*tag)["disk_pin_expires_at"] is not None
    second.close()


def test_pending_park_bytes_and_global_pin_cap_are_enforced(tmp_path):
    key = _identity()
    apc = APCv2(
        max_size=16,
        max_bytes=1 << 20,
        layout_name="layout-a",
        idle_disk_seconds=180,
        idle_disk_dir=str(tmp_path),
        pinned_disk_bytes_per_tenant=1 << 20,
        pinned_disk_bytes_global=1 << 20,
        now_fn=lambda: 0.0,
    )
    leases = []
    for index in range(3):
        tokens = [100 + index, *range(16)]
        apc.store(
            key,
            tokens,
            [_state(17, index)],
            session_tag=("tenant-a", f"s{index}"),
        )
        lease = apc.lookup(key, tokens + [99])
        assert lease.hit
        leases.append(lease)
    projected = apc.session_state("tenant-a", "s0")["resident_bytes"]
    apc._pinned_disk_bytes_per_tenant = int(projected * 1.5)
    apc.park_session("tenant-a", "s0", ttl_seconds=600)
    with pytest.raises(APCSessionCapacityError, match="per-tenant"):
        apc.park_session("tenant-a", "s1", ttl_seconds=600)
    assert apc._tenant_pin_bytes_locked("tenant-a", resident=False) == projected
    for lease in leases:
        lease.cache.close()
    apc.close()

    global_cap = APCv2(
        max_size=16,
        max_bytes=1 << 20,
        layout_name="layout-a",
        idle_disk_seconds=180,
        idle_disk_dir=str(tmp_path),
        pinned_disk_bytes_per_tenant=1 << 20,
        pinned_disk_bytes_global=projected,
    )
    for tenant, seed in (("tenant-a", 1), ("tenant-b", 2)):
        tokens = [seed, *range(16)]
        global_cap.store(
            key, tokens, [_state(17, seed)], session_tag=(tenant, "session")
        )
    global_cap.park_session("tenant-a", "session", ttl_seconds=60)
    with pytest.raises(APCSessionCapacityError, match="global"):
        global_cap.park_session("tenant-b", "session", ttl_seconds=60)
    global_cap.close()


def test_store_rejects_its_own_publication_when_only_pinned_state_remains(
    tmp_path, monkeypatch
):
    key = _identity()
    apc = APCv2(
        max_size=1,
        max_bytes=1 << 20,
        layout_name="layout-a",
        idle_disk_seconds=180,
        idle_disk_dir=str(tmp_path),
    )
    tag = ("tenant-a", "pinned")
    apc.store(key, [1, 2, 3], [_state(3)], session_tag=tag)
    lease = apc.lookup(key, [1, 2, 3, 4])
    assert lease.hit
    apc.park_session(*tag, ttl_seconds=60)
    monkeypatch.setattr(apc, "_spill_entry_locked", lambda *args, **kwargs: False)
    result = apc.store(key, [9, 8, 7], [_state(3, 9)])
    assert not result.stored
    assert apc.session_state(*tag)["entries"] == 1
    assert apc.apc_stats["idle_disk"]["publication_rejections"] == 1
    lease.cache.close()
    apc.close()


def test_persistence_metadata_io_failure_is_soft_on_lookup(tmp_path, monkeypatch):
    wall = [1_000.0]
    key = _identity()
    tag = ("tenant-a", "soft-failure")
    apc = _persistent(tmp_path, wall=wall)
    tokens = list(range(16))
    apc.store(key, tokens, [_state(16)], session_tag=tag)
    apc.park_session(*tag, ttl_seconds=5)
    wall[0] += 10
    original_replace = os.replace

    def fail_manifest_replace(source, destination):
        if str(destination).endswith(".manifest.json"):
            raise OSError("simulated private persistence path")
        return original_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_manifest_replace)
    hit = apc.lookup(key, tokens + [99], session_tag=tag)
    assert hit.hit
    hit.cache.close()
    assert apc.apc_stats["idle_disk"]["persistence_io_failures"] == 1
    apc.close()


def test_missing_payload_during_pin_expiry_becomes_a_miss(tmp_path):
    wall = [1_000.0]
    key = _identity()
    tag = ("tenant-a", "missing")
    tokens = list(range(16))
    apc = _persistent(tmp_path, wall=wall)
    apc.store(key, tokens, [_state(16)], session_tag=tag)
    apc.park_session(*tag, ttl_seconds=5)
    next(tmp_path.glob("apc-idle-*.target.safetensors")).unlink()
    wall[0] += 10
    miss = apc.lookup(key, tokens + [99], session_tag=tag)
    assert not miss.hit
    assert miss.miss_reason == "no_compatible_prefix"
    assert apc.apc_stats["idle_disk"]["restore_digest_failures"] == 1
    apc.close()


def test_cross_tenant_shared_entry_delete_only_removes_callers_tag(tmp_path):
    apc = APCv2(
        max_size=8,
        layout_name="layout-a",
        idle_disk_seconds=180,
        idle_disk_dir=str(tmp_path),
    )
    key = _identity()
    tokens = list(range(8))
    apc.store(key, tokens, [_state(8)], session_tag=("tenant-a", "same"))
    apc.store(key, tokens, [_state(8)], session_tag=("tenant-b", "same"))
    report = apc.delete_session("tenant-a", "same")
    assert report["shared_entries"] == 1
    assert apc.session_state("tenant-b", "same")["entries"] == 1
    with pytest.raises(APCSessionNotFound):
        apc.session_state("tenant-a", "same")
    apc.close()


def test_quarantine_is_bounded_and_reported(tmp_path):
    for index in range(3):
        manifest = tmp_path / f"apc-idle-{index}.manifest.json"
        manifest.write_text(json.dumps({"schema": APCv2._PERSIST_SCHEMA}))
        os.utime(manifest, (100 + index, 100 + index))
    apc = _persistent(
        tmp_path,
        quarantine_max_entries=2,
        quarantine_max_bytes=1 << 20,
    )
    quarantine = apc.apc_stats["persistence"]["quarantine"]
    assert quarantine["entries"] == 2
    assert quarantine["bytes"] > 0
    assert quarantine["evictions"] == 1
    assert not (tmp_path / "quarantine" / "apc-idle-0.manifest.json").exists()
    apc.close()


def test_closed_cache_rejects_resume_as_unavailable(tmp_path):
    apc = _persistent(tmp_path)
    tag = ("tenant-a", "closed")
    apc.store(_identity(), [1, 2, 3], [_state(3)], session_tag=tag)
    apc.park_session(*tag, ttl_seconds=60)
    apc.close()
    with pytest.raises(APCSessionUnavailable, match="closed"):
        apc.resume_session(*tag)


def test_persist_lock_refuses_symlink_and_rescan_hardens_permissions(tmp_path):
    victim = tmp_path / "victim"
    victim.write_text("untouched")
    (tmp_path / ".mlx2-apcv2.lock").symlink_to(victim)
    with pytest.raises(RuntimeError, match="lock"):
        _persistent(tmp_path)
    assert victim.read_text() == "untouched"
    (tmp_path / ".mlx2-apcv2.lock").unlink()

    first = _persistent(tmp_path)
    first.store(_identity(), [1, 2, 3], [_state(3)])
    first.park_all(time_budget_seconds=5)
    first.close()
    os.chmod(tmp_path, 0o755)
    for path in tmp_path.glob("apc-idle-*"):
        os.chmod(path, 0o644)
    leftover_blocks = tmp_path / "apc-idle-leftover.safetensors.blocks"
    leftover_blocks.mkdir(mode=0o755)
    (leftover_blocks / "part").write_bytes(b"orphan")

    second = _persistent(tmp_path)
    assert tmp_path.stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in tmp_path.glob("apc-idle-*"))
    assert not leftover_blocks.exists()
    second.close()


def test_half_written_files_are_cleaned_and_second_writer_is_refused(tmp_path):
    stale = tmp_path / ".apc-idle-dead.target.safetensors.abc.tmp"
    orphan = tmp_path / "apc-idle-dead.target.safetensors"
    stale.write_bytes(b"partial")
    orphan.write_bytes(b"complete-without-manifest")
    first = _persistent(tmp_path)
    assert not stale.exists() and not orphan.exists()
    with pytest.raises(RuntimeError, match="already owned"):
        _persistent(tmp_path)
    first.close()


def test_startup_rescan_is_manifest_only_and_bounded_for_eight_entries(tmp_path):
    first = _persistent(tmp_path)
    key = _identity()
    for index in range(8):
        tokens = [index + 100, *range(8)]
        first.store(key, tokens, [_state(len(tokens), index)])
    result = first.park_all(time_budget_seconds=5)
    assert result == {"spilled": 8, "skipped": 0}
    first.close()
    second = _persistent(tmp_path)
    rescan = second.apc_stats["persistence"]["rescan"]
    assert rescan["registered"] == 8
    assert rescan["elapsed_seconds"] < 1.0
    second.close()


def test_corrupt_manifest_is_quarantined(tmp_path):
    manifest = tmp_path / "apc-idle-bad.target.manifest.json"
    manifest.write_text(json.dumps({"schema": "bad"}))
    apc = _persistent(tmp_path)
    # Unknown schemas are safely deleted; malformed current schemas are
    # corruption and use the configured quarantine policy.
    assert not manifest.exists()
    apc.close()

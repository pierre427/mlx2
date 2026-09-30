"""Disk restores: transient errors defer, only corruption drops, and a
budget-blocked restore does not hash the payload first."""

import os
import tempfile

import mlx.core as mx
import pytest

from mlx2.runtime import apc_v2 as apc_mod
from mlx2.runtime.apc_v2 import APCKey, APCSessionNotFound, APCv2
from mlx2.runtime.models.cache import KVCache


def _state(n, seed=0):
    cache = KVCache()
    values = mx.arange(seed, seed + n, dtype=mx.float32).reshape(1, 1, n, 1)
    cache.update_and_fetch(values, values)
    mx.eval(cache.state)
    return cache


KEY = APCKey(
    "artifact-a", revision="source-a", adapter="artifact-a",
    tokenizer_fingerprint="tokenizer-a", cache_layout_fingerprint="layout-a",
    semantic_fingerprint=("text-token-v1", "tenant", "tenant-a"),
)
TAG = ("tenant-a", "conversation-1")
TOKENS = list(range(32))


@pytest.fixture
def parked(tmp_path):
    apc = APCv2(
        max_size=8, max_bytes=1 << 20, layout_name="layout-a",
        idle_disk_seconds=180, idle_disk_dir=str(tmp_path), persist_dir=str(tmp_path),
        persist_identity=KEY, persist_semantic_namespace="tenant",
    )
    apc.store(KEY, TOKENS, [_state(32)], session_tag=TAG)
    assert apc.park_session(*TAG, ttl_seconds=600)["state"] == "disk"
    try:
        yield apc, tmp_path
    finally:
        apc.close()


def _snapshots(path):
    return sorted(n for n in os.listdir(path) if n.startswith("apc-idle-"))


def _failing_loader(monkeypatch, error, times):
    real = apc_mod.load_prompt_cache
    left = {"n": times}

    def flaky(path, *args, **kwargs):
        if left["n"]:
            left["n"] -= 1
            raise error
        return real(path, *args, **kwargs)

    monkeypatch.setattr(apc_mod, "load_prompt_cache", flaky)


def test_transient_restore_error_keeps_the_parked_session(parked, monkeypatch):
    apc, path = parked
    before = _snapshots(path)
    _failing_loader(monkeypatch, MemoryError("heap momentarily full"), times=1)
    miss = apc.lookup(KEY, TOKENS + [99], session_tag=TAG)
    assert not miss.hit
    assert _snapshots(path) == before
    assert apc.session_state(*TAG)["state"] == "disk"
    again = apc.lookup(KEY, TOKENS + [99], session_tag=TAG)
    assert again.hit
    again.cache.close()
    stats = apc.apc_stats["idle_disk"]
    assert stats["restore_transient_deferrals"] == 1
    assert stats["restore_digest_failures"] == 0


def test_repeated_transient_errors_eventually_drop_the_entry(parked, monkeypatch):
    apc, path = parked
    _failing_loader(monkeypatch, OSError(24, "EMFILE"), times=10)
    for _ in range(APCv2._RESTORE_TRANSIENT_STRIKES):
        assert not apc.lookup(KEY, TOKENS + [99], session_tag=TAG).hit
    assert _snapshots(path) == []
    with pytest.raises(APCSessionNotFound):
        apc.session_state(*TAG)


def test_missing_payload_still_drops_immediately(parked, monkeypatch):
    apc, path = parked
    _failing_loader(monkeypatch, FileNotFoundError(2, "gone"), times=1)
    assert not apc.lookup(KEY, TOKENS + [99], session_tag=TAG).hit
    assert _snapshots(path) == []


def test_corrupt_content_still_drops_immediately(parked, monkeypatch):
    apc, path = parked
    _failing_loader(monkeypatch, ValueError("invalid safetensors"), times=1)
    assert not apc.lookup(KEY, TOKENS + [99], session_tag=TAG).hit
    assert _snapshots(path) == []


def test_budget_blocked_restore_does_not_hash_the_payload(monkeypatch):
    hashed = {"bytes": 0}
    original = APCv2._sha256_file

    def counting(path):
        hashed["bytes"] += os.path.getsize(path)
        return original(path)

    monkeypatch.setattr(APCv2, "_sha256_file", staticmethod(counting))
    with tempfile.TemporaryDirectory() as tmp:
        clock = [0.0]
        first = [_state(4096)]
        nbytes = sum(c.nbytes for c in first)
        apc = APCv2(
            max_size=8, max_bytes=int(nbytes * 1.5), layout_name="layout-a",
            idle_disk_seconds=1.0, idle_disk_dir=tmp, persist_dir=tmp,
            persist_identity=KEY, persist_semantic_namespace="tenant",
            now_fn=lambda: clock[0],
        )
        apc.store(KEY, list(range(4096)), first)
        clock[0] = 10.0
        assert apc.spill_idle_entries() == 1
        # A leased resident pins the budget, so the snapshot cannot come back.
        apc.store(KEY, list(range(10000, 14096)), [_state(4096, seed=5000)])
        lease = apc.lookup(KEY, list(range(10000, 14097)))
        assert lease.hit and lease.cache is not None
        hashed["bytes"] = 0
        blocked = apc.lookup(KEY, list(range(4097)))
        assert not blocked.hit
        assert blocked.miss_reason == "disk_restore_budget_unavailable"
        assert hashed["bytes"] == 0
        lease.cache.close()
        apc.close()


def test_rescan_skips_a_manifest_it_cannot_read_instead_of_discarding(parked, monkeypatch):
    apc, path = parked
    apc.close()
    manifests = sorted(p for p in os.listdir(path) if p.endswith(".manifest.json"))
    snapshots = _snapshots(path)
    assert manifests and snapshots
    from pathlib import Path

    real = Path.read_text

    def flaky(self, *args, **kwargs):
        if self.name.endswith(".manifest.json"):
            raise OSError(5, "EIO")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky)
    second = APCv2(
        max_size=8, max_bytes=1 << 20, layout_name="layout-a",
        idle_disk_seconds=180, idle_disk_dir=str(path), persist_dir=str(path),
        persist_identity=KEY, persist_semantic_namespace="tenant",
    )
    try:
        assert sorted(p for p in os.listdir(path) if p.endswith(".manifest.json")) == manifests
        assert second.apc_stats["persistence"]["rescan"]["skipped"] == len(manifests)
        # The payloads a skipped manifest names were never seen, so the
        # orphan sweep must leave them for the next clean rescan.
        assert _snapshots(path) == snapshots
    finally:
        second.close()


def test_strikes_are_consecutive_a_successful_restore_clears_them(parked, monkeypatch):
    apc, path = parked
    real = apc_mod.load_prompt_cache
    plan = iter([True, False, True, True])  # fail, succeed, fail, fail

    def flaky(p, *args, **kwargs):
        if next(plan, False):
            raise MemoryError("transient")
        return real(p, *args, **kwargs)

    monkeypatch.setattr(apc_mod, "load_prompt_cache", flaky)
    assert not apc.lookup(KEY, TOKENS + [99], session_tag=TAG).hit
    hit = apc.lookup(KEY, TOKENS + [99], session_tag=TAG)
    assert hit.hit
    hit.cache.close()
    assert apc.park_session(*TAG, ttl_seconds=600)["state"] == "disk"
    assert not apc.lookup(KEY, TOKENS + [99], session_tag=TAG).hit
    assert not apc.lookup(KEY, TOKENS + [99], session_tag=TAG).hit
    # Three transient errors, but never three in a row: the park survives.
    assert _snapshots(path)
    assert apc.session_state(*TAG)["state"] == "disk"

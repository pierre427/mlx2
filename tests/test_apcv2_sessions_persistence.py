"""CPU contracts for APCv2 session hints and restart persistence."""

import json
import os
import time
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from mlx2.runtime.apc_v2 import (
    APCKey,
    APCSessionCapacityError,
    APCSessionNotFound,
    APCSessionUnavailable,
    APCv2,
)
from mlx2.runtime.models.cache import KVCache

from test_apc_hits_hybrid_gdn_self_mtp import (  # noqa: F401 - fixture
    host,
    make_adapter,
    tiny_qwen38_mtp,
)
from test_apc_interior_placement import _ChatTemplateTokenizer, _words


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


def _resumed_two_turn_session(tmp_path):
    apc = APCv2(
        max_size=8,
        max_bytes=1 << 20,
        layout_name="layout-a",
        idle_disk_seconds=180,
        idle_disk_dir=str(tmp_path),
    )
    key = _identity()
    tag = ("tenant-a", "branched")
    base = list(range(1, 9))
    apc.store(key, base, [_state(8)], session_tag=tag)
    apc.store(key, base + list(range(20, 28)), [_state(16)], session_tag=tag)
    apc.park_session(*tag, ttl_seconds=60)
    apc.resume_session(*tag, ttl_seconds=30)  # prefetches the deepest entry
    apc.service_pending_prefetch()
    assert apc.apc_stats["idle_disk"]["prefetch_restores_ok"] == 1
    return apc, key, tag, base


def test_prefetch_is_settled_once_when_another_entry_serves_the_lookup(tmp_path):
    apc, key, tag, base = _resumed_two_turn_session(tmp_path)
    for index in range(4):
        # The conversation was edited after turn one: the 8-token entry,
        # restored from disk by the lookup itself, serves every request.
        hit = apc.lookup(key, base + [99, index], session_tag=tag)
        assert hit.hit and hit.cached_tokens == 8
        hit.cache.close()
    disk = apc.apc_stats["idle_disk"]
    assert (disk["prefetch_hits"], disk["prefetch_misses"]) == (0, 1)
    apc.close()


def test_prefetch_miss_on_a_cold_lookup_is_not_counted_again(tmp_path):
    apc, key, tag, base = _resumed_two_turn_session(tmp_path)
    miss = apc.lookup(key, [500, 501, 502], session_tag=tag)
    assert not miss.hit
    hit = apc.lookup(key, base + list(range(20, 28)) + [99], session_tag=tag)
    assert hit.hit and hit.cached_tokens == 16
    hit.cache.close()
    disk = apc.apc_stats["idle_disk"]
    assert (disk["prefetch_hits"], disk["prefetch_misses"]) == (0, 1)
    apc.close()


@pytest.mark.parametrize("query", ([10, 20, 30], [10, 20, 99]))
def test_prefetch_hit_keeps_persisted_snapshot_path_on_restart(tmp_path, query):
    key = _identity()
    tokens = [10, 20, 30]
    tag = ("tenant-a", "exact-resume")
    first = _persistent(tmp_path)
    first.store(key, tokens, [_state(len(tokens))], session_tag=tag)
    first.park_session(*tag, ttl_seconds=60)
    first.resume_session(*tag, ttl_seconds=60)
    first.service_pending_prefetch()
    deadline = time.monotonic() + 5
    while first.session_state(*tag)["state"] != "resident":
        assert time.monotonic() < deadline
        time.sleep(0.01)
    hit = first.lookup(key, query, session_tag=tag)
    assert hit.hit and hit.cached_tokens == len(tokens) - 1
    hit.cache.close()
    manifest = json.loads(next(tmp_path.glob("apc-idle-*.manifest.json")).read_text())
    assert manifest["tokens"] == tokens
    first.close()

    second = _persistent(tmp_path)
    assert second.apc_stats["persistence"]["rescan"]["registered"] == 1
    hit = second.lookup(key, tokens + [40])
    assert hit.hit and hit.cached_tokens == len(tokens)
    hit.cache.close()
    second.close()


def test_longer_store_preserves_parked_shorter_session(tmp_path):
    key = _identity()
    tag = ("tenant-a", "short-prefix")
    apc = _persistent(tmp_path)
    apc.store(key, [1, 2, 3], [_state(3)], session_tag=tag)
    parked = apc.park_session(*tag, ttl_seconds=60)
    assert parked["state"] == "disk"
    apc.store(key, [1, 2, 3, 4], [_state(4)])
    assert apc.session_state(*tag)["state"] == "disk"
    assert apc._trie.get(key, [1, 2, 3]) is not None
    assert next(tmp_path.glob("apc-idle-*.manifest.json")).exists()
    apc.close()

    restarted = _persistent(tmp_path)
    assert restarted.session_state(*tag)["state"] == "disk"
    restarted.close()


def test_republishing_a_parked_boundary_persists_the_replacement_before_a_crash(tmp_path):
    key = _identity()
    tag = ("tenant-a", "regenerate")
    tokens = list(range(16))
    first = _persistent(tmp_path)
    first.store(key, tokens, [_state(16, 0)], session_tag=tag)
    assert first.park_session(*tag, ttl_seconds=3600)["state"] == "disk"
    old_manifest = next(tmp_path.glob("apc-idle-*.manifest.json"))
    # A retried request republishes the same boundary with different content.
    first.store(key, tokens, [_state(16, 500)], session_tag=tag)
    state = first.session_state(*tag)
    assert state["state"] == "resident" and state["disk_pin_expires_at"]
    manifests = list(tmp_path.glob("apc-idle-*.manifest.json"))
    assert len(manifests) == 1 and manifests[0] != old_manifest
    # Crash: the process dies without close(); only its lock is released.
    first._release_persist_lock()

    second = _persistent(tmp_path)
    assert second.session_state(*tag)["state"] == "disk"
    hit = second.lookup(key, tokens + [99], session_tag=tag)
    assert hit.hit and hit.cached_tokens == 16
    keys, _values = hit.cache[0].state
    assert keys[0, 0, :16, 0].tolist() == list(range(500, 516))
    hit.cache.close()
    second.close()


def test_prefix_subsumption_preserves_committed_boundary_but_prunes_plain_cache(tmp_path):
    apc = _persistent(tmp_path)
    key = _identity()
    apc.store(key, [1, 2], [_state(2)], retention_role="committed_prompt_boundary")
    apc.store(key, [1, 2, 3], [_state(3)])
    assert apc._trie.get(key, [1, 2]) is not None
    apc.store(key, [5, 6], [_state(2)])
    apc.store(key, [5, 6, 7], [_state(3)])
    with pytest.raises(KeyError):
        apc._trie.get(key, [5, 6])
    apc.close()


def test_oversized_restore_rejects_before_payload_hash(tmp_path, monkeypatch):
    key = _identity()
    tokens = [1, 2, 3]
    apc = _persistent(tmp_path)
    apc.store(key, tokens, [_state(3)])
    apc.park_all(time_budget_seconds=5)
    entry = apc._trie.get(key, tokens)
    entry._apc_disk["resident_nbytes"] = apc.max_bytes + 1

    def forbid_hash(_path):
        raise AssertionError("oversized snapshot was hashed")

    monkeypatch.setattr(apc, "_sha256_file", forbid_hash)
    assert not apc.lookup(key, tokens + [4]).hit
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


class _Int8Policy:
    """Stand-in for an enabled ``Int8PrefillPolicy`` (only the fields used)."""

    enabled = True

    def __init__(self, revision):
        self.revision = revision


def _serving_semantic(tenant, *, scope=None, int8_revision=None):
    """Build a semantic fingerprint exactly as serving's ``cache_key_for`` does."""
    from mlx2.runtime.int8_prefill import apc_semantic_fingerprint
    from mlx2.serving import cache_semantic_fingerprint

    semantic = cache_semantic_fingerprint(tenant)
    if scope:
        semantic = f"{semantic}:media:{scope}"
    policy = _Int8Policy(int8_revision) if int8_revision else None
    return apc_semantic_fingerprint(semantic, policy)


def _namespace_server(directory, namespace, *, int8_revision=None):
    template = _serving_semantic(
        "__tenant_template__" if namespace == "tenant" else None,
        int8_revision=int8_revision,
    )
    return APCv2(
        max_size=16,
        max_bytes=1 << 20,
        layout_name="layout-a",
        idle_disk_seconds=180,
        idle_disk_dir=str(directory),
        persist_dir=str(directory),
        persist_identity=replace(_identity(), semantic_fingerprint=template),
        persist_semantic_namespace=namespace,
    )


def _park_under(directory, namespace, semantic, *, int8_revision=None):
    apc = _namespace_server(directory, namespace, int8_revision=int8_revision)
    key = replace(_identity(), semantic_fingerprint=semantic)
    tag = ("tenant-a", "parked")
    apc.store(key, list(range(8)), [_state(8)], session_tag=tag)
    assert apc.park_session(*tag, ttl_seconds=600)["state"] == "disk"
    apc.close()
    return key, tag


@pytest.mark.parametrize("namespace", ("shared", "tenant"))
@pytest.mark.parametrize(
    "scope, int8_revision",
    (
        (None, "int8-rev-1"),
        ("image-sha", None),
        ("image-sha|lora:adapter-sha", "int8-rev-1"),
        ("|lora:adapter-sha", None),
        (("image-sha", "hyper-directory", "memory-sha"), None),
    ),
)
def test_restart_keeps_int8_and_scoped_parked_sessions(
    tmp_path, namespace, scope, int8_revision
):
    tenant = "tenant-a" if namespace == "tenant" else None
    semantic = _serving_semantic(tenant, scope=scope, int8_revision=int8_revision)
    key, tag = _park_under(tmp_path, namespace, semantic, int8_revision=int8_revision)

    restarted = _namespace_server(tmp_path, namespace, int8_revision=int8_revision)
    rescan = restarted.apc_stats["persistence"]["rescan"]
    assert rescan["registered"] == 1 and rescan["discarded"] == {}
    assert restarted.session_state(*tag)["state"] == "disk"
    hit = restarted.lookup(key, list(range(8)) + [99], session_tag=tag)
    assert hit.hit and hit.cached_tokens == 8
    hit.cache.close()
    restarted.close()


@pytest.mark.parametrize(
    "stored_namespace, restart_namespace, stored_revision, restart_revision",
    (
        ("shared", "tenant", None, None),
        ("tenant", "shared", None, None),
        ("shared", "shared", "int8-rev-1", "int8-rev-2"),
        ("tenant", "tenant", "int8-rev-1", None),
        ("shared", "shared", None, "int8-rev-1"),
    ),
)
def test_restart_discards_other_namespace_mode_or_int8_revision(
    tmp_path, stored_namespace, restart_namespace, stored_revision, restart_revision
):
    tenant = "tenant-a" if stored_namespace == "tenant" else None
    semantic = _serving_semantic(
        tenant, scope="image-sha", int8_revision=stored_revision
    )
    _park_under(tmp_path, stored_namespace, semantic, int8_revision=stored_revision)

    restarted = _namespace_server(
        tmp_path, restart_namespace, int8_revision=restart_revision
    )
    rescan = restarted.apc_stats["persistence"]["rescan"]
    assert rescan["registered"] == 0
    assert rescan["discarded"] == {"identity_mismatch": 1}
    assert not list(tmp_path.glob("apc-idle-*.manifest.json"))
    restarted.close()


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
        # Room for one real snapshot (its projection plus the file header),
        # not for two.
        pinned_disk_bytes_global=int(projected * 1.5),
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


def _projected_park_apc(tmp_path, clock, cap):
    return APCv2(
        max_size=16,
        max_bytes=1 << 20,
        layout_name="layout-a",
        idle_disk_seconds=180,
        idle_disk_dir=str(tmp_path),
        pinned_disk_bytes_per_tenant=cap,
        now_fn=lambda: clock[0],
    )


def test_park_fails_closed_when_the_real_snapshot_outgrows_its_projection(tmp_path):
    """The precheck projects a resident entry at its nbytes; the written file
    carries a safetensors header on top.  A cap between the two must fail the
    park, not report success and rewrite the snapshot on every idle scan."""
    clock = [0.0]
    key = _identity()
    tag = ("tenant-a", "tight")
    apc = _projected_park_apc(tmp_path, clock, _state(32).nbytes)
    apc.store(key, list(range(32)), [_state(32)], session_tag=tag)
    with pytest.raises(APCSessionCapacityError, match="per-tenant"):
        apc.park_session(*tag, ttl_seconds=600)
    state = apc.session_state(*tag)
    assert state["state"] == "resident"
    assert state["disk_pin_expires_at"] is None and state["park_pending"] == 0
    for _ in range(3):
        clock[0] += 2.0
        apc.spill_idle_entries(now=clock[0])
    disk = apc.apc_stats["idle_disk"]
    assert disk["spill_failures"] == 1
    assert disk["bytes_written"] == 0
    assert not list(tmp_path.glob("apc-idle-*"))
    apc.close()


def test_failed_repark_restores_the_pins_it_replaced(tmp_path):
    clock = [0.0]
    key = _identity()
    tag = ("tenant-a", "grown")
    probe = _projected_park_apc(tmp_path / "probe", clock, 1 << 30)
    probe.store(key, list(range(16)), [_state(16)], session_tag=tag)
    probe.park_session(*tag, ttl_seconds=60)
    parked_file_bytes = probe.session_state(*tag)["disk_bytes"]
    probe.close()

    apc = _projected_park_apc(
        tmp_path, clock, parked_file_bytes + _state(32).nbytes
    )
    apc.store(key, list(range(16)), [_state(16)], session_tag=tag)
    first_expiry = apc.park_session(*tag, ttl_seconds=60)["disk_pin_expires_at"]
    apc.store(key, list(range(32)), [_state(32)], session_tag=tag)
    with pytest.raises(APCSessionCapacityError):
        apc.park_session(*tag, ttl_seconds=600)
    parked = apc._trie.get(key, list(range(16)))
    assert parked._apc_disk_pin_expiries == {tag: first_expiry}
    grown = apc._trie.get(key, list(range(32)))
    assert grown.prompt_cache and not grown._apc_disk_pin_expiries
    apc.close()


def test_deferred_park_that_cannot_fit_is_not_rewritten_every_scan(tmp_path):
    clock = [0.0]
    key = _identity()
    tag = ("tenant-a", "leased")
    tokens = list(range(32))
    apc = _projected_park_apc(tmp_path, clock, _state(32).nbytes)
    apc.store(key, tokens, [_state(32)], session_tag=tag)
    lease = apc.lookup(key, tokens + [99])
    assert apc.park_session(*tag, ttl_seconds=600)["park_pending"] == 1
    lease.cache.close()
    for _ in range(3):
        clock[0] += 2.0
        apc.spill_idle_entries(now=clock[0])
    state = apc.session_state(*tag)
    assert state["disk_pin_expires_at"] is None and state["park_pending"] == 0
    disk = apc.apc_stats["idle_disk"]
    assert disk["spill_failures"] == 1 and disk["bytes_written"] == 0
    assert disk["pin_cap_rejections"] == 1
    apc.close()


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


# ------------------------------------ serving: a resume prefetches what serves


def _session_engine(adapter, *, mtp, cache_dir=None):
    from mlx2 import serving

    engine = serving.ServingEngine(
        "tiny",
        adapter_factory=adapter,
        qualification_mode=True,
        mtp=mtp,
        max_lanes=1,
        prefill_step=16,
        cache_dir=None if cache_dir is None else str(cache_dir),
    )
    assert engine.ready.wait(60), engine.error
    return engine


def _park_and_resume(engine, session_id):
    state = engine.apc_session_park("default", session_id, ttl_seconds=60)
    deadline = time.monotonic() + 10
    while state["state"] != "disk":
        assert time.monotonic() < deadline, state
        time.sleep(0.02)
        state = engine.apc_session_state("default", session_id)
    engine.apc_session_resume("default", session_id)
    # The whole prefetch runs under one APC lock hold on the model worker, so
    # the deepest entry reading resident means every restore it made landed.
    while state["state"] != "resident":
        assert time.monotonic() < deadline, state
        time.sleep(0.02)
        state = engine.apc_session_state("default", session_id)


def _resumed_turn(engine, session_id, seed, request):
    """Seed a session, park and resume it, then send ``request(output)``."""
    first, _receipt, _job = _run_request_body(engine, seed(session_id))
    _park_and_resume(engine, session_id)
    disk = engine.apc.apc_stats["idle_disk"]
    restores_before = disk["restores"]
    body = request(first)
    out, receipt, job = _run_request_body(engine, dict(body, session_id=session_id))
    disk = engine.apc.apc_stats["idle_disk"]
    return body, out, receipt, job, dict(disk), disk["restores"] - restores_before


def _run_request_body(engine, body):
    job = engine.submit(dict(body))
    text = ""
    while True:
        event = job.events.get(timeout=120)
        if "error" in event:
            raise AssertionError(event)
        if "delta" in event:
            text += event["delta"].get("content", "")
        if "finish_reason" in event:
            return [int(t) for t in text.split()], event.get("receipt") or {}, job


@pytest.mark.parametrize("mtp", [False, True], ids=["ordinary", "mtp"])
@pytest.mark.parametrize("followup", ["resend", "continuation"])
def test_resumed_session_request_is_served_from_the_prefetch(
    host, tmp_path, mtp, followup
):
    """PF: on the MTP route a resume prefetched only the deepest entry.

    That finished lane's draft sidecar is state at its own end, so a re-send
    of the parked prompt cannot use it: the ``P-1`` boundary served it after
    a second, on-path disk restore, and the prefetch was a wasted restore
    and a miss.  The resume now also prefetches the boundary, so either
    follow-up is served from resident state with nothing restored on its
    own path, the prefetch is credited once, and output equals a cold run.
    """
    model, vocab = tiny_qwen38_mtp()
    adapter = make_adapter(model, vocab)
    prompt = [(7 * i + 3) % (vocab - 2) + 1 for i in range(18)]
    seed = lambda session: {  # noqa: E731
        "tokens": prompt, "max_tokens": 5, "temperature": 0, "session_id": session,
    }

    def request(first):
        tokens = prompt if followup == "resend" else prompt + first + [9, 8, 7, 6]
        return {"tokens": tokens, "max_tokens": 5, "temperature": 0}

    engine = _session_engine(adapter, mtp=mtp, cache_dir=tmp_path)
    try:
        body, out, receipt, job, disk, restored_on_path = _resumed_turn(
            engine, "pf", seed, request
        )
    finally:
        engine.close()
    assert restored_on_path == 0, disk
    assert (disk["parks"], disk["resumes"]) == (1, 1)
    assert disk["prefetch_restores_ok"] >= 1
    assert (disk["prefetch_hits"], disk["prefetch_misses"]) == (1, 0), disk
    cached = int(job.cached_tokens)
    if followup == "resend":
        assert cached == len(prompt) - 1
        if mtp:
            assert receipt["cache_checkpoint_role"] == "committed_prompt_boundary"
    else:
        assert cached >= len(prompt) + 5 - 1
    cold = _session_engine(adapter, mtp=mtp)
    try:
        cold_out, _receipt, cold_job = _run_request_body(cold, body)
    finally:
        cold.close()
    assert int(cold_job.cached_tokens or 0) == 0
    assert out == cold_out


@pytest.mark.parametrize("mtp", [False, True], ids=["ordinary", "mtp"])
def test_resumed_next_turn_uses_the_prefetched_generation_prompt_boundary(
    host, tmp_path, mtp
):
    """PF: a template that drops the generation prompt (Qwen3.6) from history.

    Its next turn resumes at the boundary before that suffix, which the
    finished lane cannot land on: on the MTP route it was restored from disk
    on the next turn's own path (a prefetch miss), and on the ordinary route
    the prefetched lane landed short of it.  The resume now prefetches it,
    so the next turn resumes exactly where an unparked session would.
    """
    model, vocab = tiny_qwen38_mtp()
    base = make_adapter(model, vocab)
    template = _ChatTemplateTokenizer(vocab, keeps_think_block=False)

    class Adapter(base):
        def __init__(self, path):
            super().__init__(path)
            detokenizer = type(self).tokenizer

            class Tokenizer:
                vocab_size = vocab
                eos_token_ids = []
                all_special_ids = template.all_special_ids
                apply_chat_template = staticmethod(template.apply_chat_template)

                @property
                def detokenizer(self):
                    return detokenizer.detokenizer

            self.tokenizer = Tokenizer()

        def prompt_tokens(self, request):
            return template.apply_chat_template(
                request["messages"], add_generation_prompt=True
            )

        def cache_budget(self, *, mtp):
            from mlx2.adapters.qwen38_memory import Qwen38CacheBudget

            return Qwen38CacheBudget.from_config(dict(vars(model.args)), mtp=mtp)

    messages = [
        {"role": "system", "content": _words(vocab, 1, 60)},
        {"role": "user", "content": _words(vocab, 2, 20)},
    ]
    seed = lambda session: {  # noqa: E731
        "messages": messages, "max_tokens": 6, "temperature": 0,
        "session_id": session,
    }

    def request(first):
        return {
            "messages": messages + [
                {"role": "assistant", "content": " ".join(map(str, first))},
                {"role": "user", "content": _words(vocab, 3, 20)},
            ],
            "max_tokens": 6,
            "temperature": 0,
        }

    engine = _session_engine(Adapter, mtp=mtp, cache_dir=tmp_path)
    try:
        body, out, receipt, job, disk, restored_on_path = _resumed_turn(
            engine, "pf-chat", seed, request
        )
    finally:
        engine.close()
    first_prompt = template.apply_chat_template(messages, add_generation_prompt=True)
    suffix = 7  # thinking off: <start> assistant \n <think> \n\n </think> \n\n
    assert int(job.cached_tokens) == len(first_prompt) - suffix
    assert receipt["cache_checkpoint_role"] == "interior_checkpoint"
    assert restored_on_path == 0, disk
    assert (disk["prefetch_hits"], disk["prefetch_misses"]) == (1, 0), disk
    cold = _session_engine(Adapter, mtp=mtp)
    try:
        cold_out, _receipt, cold_job = _run_request_body(cold, body)
    finally:
        cold.close()
    assert int(cold_job.cached_tokens or 0) == 0
    assert out == cold_out

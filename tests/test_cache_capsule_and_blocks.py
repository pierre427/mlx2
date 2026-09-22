# SPDX-License-Identifier: MIT
import json
import threading
from types import SimpleNamespace
from dataclasses import replace
import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from mlx2.runtime.apc_v2 import APCKey, APCv2
from mlx2.runtime.cache_capsule import (
    CacheCapsuleError,
    CacheCapsulePool,
    CacheCapsuleProduct,
    KVCacheCapsulePayload,
    StaleCacheCapsule,
    build_kv_cache_capsule_cpu,
    capture_kv_cache_plane,
    prepare_prompt_cache_capsules,
)
from mlx2.runtime.generate import BatchGenerator
from mlx2.runtime.models.cache import ArraysCache, KVCache, RotatingKVCache
from mlx2.runtime.persistent_blocks import (
    block_file_paths,
    encode_block_file,
    materialize_block_file,
    remove_block_file,
)


def _cache(length=5):
    cache = KVCache()
    keys = mx.arange(length * 2, dtype=mx.float32).astype(mx.bfloat16).reshape(1, 1, length, 2)
    cache.update_and_fetch(keys, keys)
    mx.eval(cache.state)
    return cache


def _key():
    return APCKey("model", "revision", "adapter", "tokenizer", "layout", "semantic")


class _Reservation:
    def __init__(self): self.releases = 0
    def release(self): self.releases += 1


def _external_product(source, backing=None):
    product = build_kv_cache_capsule_cpu(source)
    return CacheCapsuleProduct(
        replace(product.payload, backend="external"), backing_owner=backing
    )


def test_capsule_requires_exact_identity_and_reserves_before_publication():
    apc = APCv2(max_bytes=1 << 20, layout_name="layout")
    signature = apc.capsule_compatibility_signature(_key(), apc.layout_name)
    source = capture_kv_cache_plane(
        _cache(), generation=apc.capsule_generation.current, source_id="entry:plane:0",
        compatibility_signature=signature, target_batch=3, verify_raw_bits=True,
    )
    reservations = []
    def reserve(_nbytes):
        reservations.append(_Reservation()); return reservations[-1]
    pool = CacheCapsulePool(
        apc.capsule_generation, enabled=True, verify_raw_bits=True, reserve=reserve
    )
    receipt = pool.prepare(source, primary="cpu", fallback=None)
    lease = receipt.owner.lease()
    restored = lease.restore_batch_kv_cache(lambda payload: mx.eval(payload.keys, payload.values))
    assert restored.keys.shape[0] == 3
    assert reservations and reservations[0].releases == 0
    receipt.owner.release()
    assert reservations[0].releases == 0
    lease.close()
    assert reservations[0].releases == 1
    assert pool.counters["primary_successes"] == 1


def test_capsule_generation_double_check_rejects_reservation_side_effect():
    apc = APCv2(max_bytes=1 << 20, layout_name="layout")
    source = capture_kv_cache_plane(
        _cache(), generation=apc.capsule_generation.current, source_id="entry:plane:0",
        compatibility_signature=apc.capsule_compatibility_signature(_key(), "layout"),
    )
    reservation = _Reservation()
    def reserve(_nbytes):
        apc.capsule_generation.advance()
        return reservation
    pool = CacheCapsulePool(apc.capsule_generation, enabled=True, reserve=reserve)
    with pytest.raises(StaleCacheCapsule):
        pool.prepare(source, primary="cpu")
    assert reservation.releases == 1
    assert pool.counters["stale"] == 1


def test_apcv2_capsule_capacity_is_hard_reserved_and_released():
    apc = APCv2(max_bytes=1024, layout_name="layout")
    first = apc.reserve_capsule_bytes(800)
    assert first is not None
    assert apc.reserve_capsule_bytes(300) is None
    assert apc.apc_stats["cache_capsules"] == {
        "reservations": 1,
        "reservation_rejections": 1,
        "reserved_bytes_peak": 800,
        "reserved_bytes": 800,
    }
    first.release()
    assert apc.apc_stats["cache_capsules"]["reserved_bytes"] == 0


def test_raw_safetensors_detection_never_reads_the_whole_payload(tmp_path, monkeypatch):
    """Persistent raw payloads are not JSON block manifests.

    Pin accounting and restore call this detector repeatedly. A whole-file
    UTF-8 probe made park O(entries * payload bytes), and restore read every
    payload once before its required digest/deserialization passes.
    """
    path = tmp_path / "raw.safetensors"
    path.write_bytes(
        (16).to_bytes(8, "little") + b'{"fake":"header"}' + b"x" * (4 << 20)
    )

    original_read_text = type(path).read_text
    original_read_bytes = type(path).read_bytes

    def forbid_text(self, *args, **kwargs):
        if self == path:
            raise AssertionError("raw safetensors payload was decoded as text")
        return original_read_text(self, *args, **kwargs)

    def forbid_bytes(self, *args, **kwargs):
        if self == path:
            raise AssertionError("raw safetensors payload was read whole for detection")
        return original_read_bytes(self, *args, **kwargs)

    monkeypatch.setattr(type(path), "read_text", forbid_text)
    monkeypatch.setattr(type(path), "read_bytes", forbid_bytes)
    assert block_file_paths(path) == (path,)
    with materialize_block_file(path, expected_signature="unused") as materialized:
        assert materialized == path


def test_batch_generator_consumes_prepared_capsule_without_remerge():
    apc = APCv2(max_bytes=1 << 20, layout_name="layout")
    signature = apc.capsule_compatibility_signature(_key(), "layout")
    reservations = []
    def reserve(_nbytes):
        reservations.append(_Reservation()); return reservations[-1]
    pool = CacheCapsulePool(
        apc.capsule_generation, enabled=True, reserve=reserve
    )
    source_cache = _cache()
    prepared = prepare_prompt_cache_capsules(
        [source_cache], target_batch=2, generation=0, pool=pool,
        compatibility_signature=signature, backend="cpu", fallback=None,
    )
    generator = BatchGenerator(None, completion_batch_size=2, prefill_batch_size=2)
    try:
        uids = generator.insert(
            [[9], [9]], caches=[[_cache()], [_cache()]], all_tokens=[[1], [1]]
        )
        assert not generator.bind_cache_capsule("g", uids[0], prepared, 2)
        assert generator.bind_cache_capsule("g", uids[1], prepared, 2)
        prompt_batch = generator._make_batch(2)
        assert prompt_batch.prompt_cache[0] is prepared.prompt_cache[0]
        assert generator.scheduler_stats["cache_capsule_engaged"] == 1
        receipt = generator.pop_cache_capsule_receipt(uids[0])
        assert receipt["status"] == "engaged"
        assert receipt["planes"] == 1
        assert receipt["capsule_planes"] == 1
        assert receipt["ordinary_planes"] == 0
        assert len(reservations) == 1 and reservations[0].releases == 0
        generator.remove([uids[0]])
        assert reservations[0].releases == 0
        generator.remove([uids[1]])
        assert reservations[0].releases == 1
        assert prepared.prompt_cache == ()
        assert all(receipt.owner.released for receipt in prepared.receipts)
    finally:
        generator.close(); pool.close()


def test_mixed_cache_capsules_plain_planes_and_ordinarily_merges_rotating_planes():
    apc = APCv2(max_bytes=1 << 20, layout_name="layout")
    reservations = []
    def reserve(_nbytes):
        reservations.append(_Reservation()); return reservations[-1]
    pool = CacheCapsulePool(
        apc.capsule_generation, enabled=True, reserve=reserve
    )
    rotating = RotatingKVCache(max_size=8, keep=0)
    values = mx.arange(6, dtype=mx.float32).reshape(1, 1, 3, 2)
    rotating.update_and_fetch(values, values + 10)
    prepared = prepare_prompt_cache_capsules(
        [_cache(3), rotating], target_batch=2, generation=0, pool=pool,
        compatibility_signature=apc.capsule_compatibility_signature(_key(), "layout"),
        backend="cpu", fallback=None,
    )
    generator = BatchGenerator(None, completion_batch_size=2, prefill_batch_size=2)
    try:
        assert len(prepared.prompt_cache) == 2
        assert len(prepared.receipts) == 1
        assert prepared.ordinary_planes == 1
        assert prepared.prompt_cache[0].keys.shape[0] == 2
        assert prepared.prompt_cache[1].keys.shape[0] == 2
        assert mx.all(prepared.prompt_cache[1].keys[0] == values[0]).item()
        assert mx.all(prepared.prompt_cache[1].keys[1] == values[0]).item()
        # One transaction reserves the full mixed Bn result. Capsule planes
        # must not take a second per-plane reservation.
        assert len(reservations) == 1 and reservations[0].releases == 0
        uids = generator.insert(
            [[9], [9]], caches=[[_cache()], [_cache()]], all_tokens=[[1], [1]]
        )
        assert not generator.bind_cache_capsule("mixed", uids[0], prepared, 2)
        assert generator.bind_cache_capsule("mixed", uids[1], prepared, 2)
        generator._make_batch(2)
        receipt = generator.pop_cache_capsule_receipt(uids[0])
        assert receipt["planes"] == 2
        assert receipt["capsule_planes"] == 1
        assert receipt["ordinary_planes"] == 1
        generator.remove(uids)
    finally:
        generator.close(); prepared.close(); pool.close()
    assert reservations[0].releases == 1


def test_mixed_capsule_capacity_rejection_precedes_build_merge_or_publication():
    apc = APCv2(max_bytes=1 << 20, layout_name="layout")
    requested, builds = [], []
    class OrdinaryPlane:
        nbytes = 96
        def __init__(self): self.merges = 0
        def merge(self, _planes):
            self.merges += 1
            raise AssertionError("capacity rejection must precede merge")
    ordinary = OrdinaryPlane()
    plain = _cache(3)
    def reject(nbytes):
        requested.append(nbytes)
        return None
    pool = CacheCapsulePool(
        apc.capsule_generation, enabled=True, reserve=reject
    )
    original_build = pool._build
    def observed_build(*args, **kwargs):
        builds.append((args, kwargs))
        return original_build(*args, **kwargs)
    pool._build = observed_build
    with pytest.raises(CacheCapsuleError, match="capacity reservation rejected"):
        prepare_prompt_cache_capsules(
            [plain, ordinary], target_batch=2, generation=0, pool=pool,
            compatibility_signature=apc.capsule_compatibility_signature(
                _key(), "layout"
            ),
            backend="cpu", fallback=None,
        )
    assert requested == [(plain.nbytes + ordinary.nbytes) * 2]
    assert builds == [] and ordinary.merges == 0
    assert pool.counters["capacity_rejections"] == 1
    assert pool.counters["primary_successes"] == 0
    pool.close()


def test_mixed_capsule_generation_change_during_full_reservation_fails_before_build():
    apc = APCv2(max_bytes=1 << 20, layout_name="layout")
    reservation, builds = _Reservation(), []
    class OrdinaryPlane:
        nbytes = 96
        def __init__(self): self.merges = 0
        def merge(self, _planes): self.merges += 1
    ordinary = OrdinaryPlane()
    def reserve(_nbytes):
        apc.capsule_generation.advance()
        return reservation
    pool = CacheCapsulePool(
        apc.capsule_generation, enabled=True, reserve=reserve
    )
    original_build = pool._build
    def observed_build(*args, **kwargs):
        builds.append((args, kwargs)); return original_build(*args, **kwargs)
    pool._build = observed_build
    with pytest.raises(StaleCacheCapsule):
        prepare_prompt_cache_capsules(
            [_cache(3), ordinary], target_batch=2, generation=0, pool=pool,
            compatibility_signature=apc.capsule_compatibility_signature(
                _key(), "layout"
            ),
            backend="cpu", fallback=None,
        )
    assert reservation.releases == 1
    assert builds == [] and ordinary.merges == 0
    assert pool.counters["stale"] == 1
    pool.close()


def test_prepared_reservation_cannot_undercharge_a_capsule_source():
    apc = APCv2(max_bytes=1 << 20, layout_name="layout")
    backing, builds = _Reservation(), []
    pool = CacheCapsulePool(
        apc.capsule_generation, enabled=True, reserve=lambda _nbytes: backing
    )
    token = pool.reserve_prepared(0, 1)
    source = capture_kv_cache_plane(
        _cache(3), generation=0, source_id="too-large",
        compatibility_signature=("exact",), target_batch=2,
    )
    original_build = pool._build
    def observed_build(*args, **kwargs):
        builds.append((args, kwargs)); return original_build(*args, **kwargs)
    pool._build = observed_build
    with pytest.raises(CacheCapsuleError, match="exceeds capacity"):
        pool.prepare(
            source, primary="cpu", fallback=None,
            prepared_reservation=token,
        )
    assert builds == []
    token.release()
    assert backing.releases == 1
    pool.close()


def test_capsule_changed_compatibility_signature_fails_closed():
    apc = APCv2(max_bytes=1 << 20, layout_name="layout")
    source = capture_kv_cache_plane(
        _cache(), generation=0, source_id="entry", compatibility_signature=("exact",)
    )
    good = build_kv_cache_capsule_cpu(source).payload
    changed = KVCacheCapsulePayload(
        good.keys, good.values, good.offset, good.source_generation, good.source_id,
        ("different",), good.layout_fingerprint, good.backend, good.raw_digest,
    )
    pool = CacheCapsulePool(apc.capsule_generation, enabled=True)
    with pytest.raises(CacheCapsuleError, match="compatibility"):
        pool._accept(CacheCapsuleProduct(changed), source, "cpu", None)


def test_external_timeout_falls_back_and_late_result_is_disposed():
    apc = APCv2(max_bytes=1 << 20, layout_name="layout")
    source = capture_kv_cache_plane(
        _cache(), generation=0, source_id="entry", compatibility_signature=("exact",)
    )
    release = threading.Event()
    discarded = []
    class Adapter:
        def stage(self, _value): return {"host_only": True}
        def build(self, value): release.wait(1); return value
        def adopt(self, _built, adopted_source): return _external_product(adopted_source)
        def discard(self, product): discarded.append(product)
    pool = CacheCapsulePool(apc.capsule_generation, adapter=Adapter(), enabled=True)
    receipt = pool.prepare(source, primary="external", fallback="cpu", timeout_s=0)
    assert receipt.fallback_reason == "external_timeout"
    release.set()
    for _ in range(1000):
        if discarded: break
        threading.Event().wait(0.001)
    assert discarded
    assert pool.counters["late_disposals"] == 1
    receipt.owner.release(); pool.close()


def test_external_adopt_failure_discards_built_and_terminalizes_ticket():
    apc = APCv2(max_bytes=1 << 20, layout_name="layout")
    source = capture_kv_cache_plane(
        _cache(), generation=0, source_id="entry", compatibility_signature=("exact",)
    )
    discarded = []
    class Adapter:
        def stage(self, _value): return {"host_only": True}
        def build(self, value): return value
        def adopt(self, _built, _source): raise RuntimeError("adopt failed")
        def discard(self, product): discarded.append(product)
    pool = CacheCapsulePool(apc.capsule_generation, adapter=Adapter(), enabled=True)
    with pytest.raises(RuntimeError, match="adopt failed"):
        pool.prepare(source, primary="external", fallback=None, timeout_s=1)
    assert len(discarded) == 1 and discarded[0] == {"host_only": True}
    assert not pool._tickets and pool._active is None
    assert pool.counters["errors"] == 1


def test_external_stage_rejects_mlx_arrays_before_worker_launch():
    apc = APCv2(max_bytes=1 << 20, layout_name="layout")
    source = capture_kv_cache_plane(
        _cache(), generation=0, source_id="entry", compatibility_signature=("exact",)
    )
    class UnsafeAdapter:
        def stage(self, value): return {"keys": value.keys}
        def build(self, staged): return staged
    pool = CacheCapsulePool(
        apc.capsule_generation, adapter=UnsafeAdapter(), enabled=True
    )
    with pytest.raises(CacheCapsuleError, match="host-only"):
        pool.submit(source)
    assert not pool._tickets and pool._active is None


def test_external_stage_rejects_unknown_wrapper_hiding_mlx_array():
    apc = APCv2(max_bytes=1 << 20, layout_name="layout")
    source = capture_kv_cache_plane(
        _cache(), generation=0, source_id="entry", compatibility_signature=("exact",)
    )
    class UnsafeAdapter:
        def stage(self, value): return SimpleNamespace(keys=value.keys)
        def build(self, staged): return staged
    pool = CacheCapsulePool(
        apc.capsule_generation, adapter=UnsafeAdapter(), enabled=True
    )
    with pytest.raises(CacheCapsuleError, match="host-only"):
        pool.submit(source)
    assert not pool._tickets and pool._active is None


def test_external_stale_before_await_terminalizes_and_disposes_late_result():
    apc = APCv2(max_bytes=1 << 20, layout_name="layout")
    source = capture_kv_cache_plane(
        _cache(), generation=0, source_id="entry", compatibility_signature=("exact",)
    )
    release, discarded = threading.Event(), []
    class Adapter:
        def stage(self, _value): return {"host_only": True}
        def build(self, value): release.wait(1); return value
        def adopt(self, *_args): raise AssertionError("stale work must not adopt")
        def discard(self, product): discarded.append(product)
    pool = CacheCapsulePool(apc.capsule_generation, adapter=Adapter(), enabled=True)
    ticket = pool.submit(source)
    apc.capsule_generation.advance()
    with pytest.raises(StaleCacheCapsule):
        ticket.await_adopt(timeout_s=1, fallback=None)
    assert not pool._tickets
    release.set()
    for _ in range(1000):
        if discarded: break
        threading.Event().wait(0.001)
    assert discarded == [{"host_only": True}]


def test_external_stale_after_build_disposes_before_adopt():
    apc = APCv2(max_bytes=1 << 20, layout_name="layout")
    source = capture_kv_cache_plane(
        _cache(), generation=0, source_id="entry", compatibility_signature=("exact",)
    )
    release, discarded = threading.Event(), []
    class Adapter:
        def stage(self, _value): return {"host_only": True}
        def build(self, value):
            release.wait(1)
            apc.capsule_generation.advance()
            return value
        def adopt(self, *_args): raise AssertionError("stale work must not adopt")
        def discard(self, product): discarded.append(product)
    pool = CacheCapsulePool(apc.capsule_generation, adapter=Adapter(), enabled=True)
    ticket = pool.submit(source)
    threading.Timer(0.01, release.set).start()
    with pytest.raises(StaleCacheCapsule):
        ticket.await_adopt(timeout_s=1, fallback=None)
    assert discarded == [{"host_only": True}]
    assert not pool._tickets and pool._active is None


def test_external_accept_failure_releases_adopted_product_once():
    apc = APCv2(max_bytes=1 << 20, layout_name="layout")
    source = capture_kv_cache_plane(
        _cache(), generation=0, source_id="entry", compatibility_signature=("exact",)
    )
    backing = _Reservation()
    discarded = []
    class Adapter:
        def stage(self, _value): return {"host_only": True}
        def build(self, value): return value
        def adopt(self, _built, adopted_source):
            payload = _external_product(adopted_source).payload
            bad = KVCacheCapsulePayload(
                payload.keys, payload.values, payload.offset,
                payload.source_generation, payload.source_id, ("wrong",),
                payload.layout_fingerprint, "external", payload.raw_digest,
            )
            return CacheCapsuleProduct(bad, backing)
        def discard(self, product): discarded.append(product)
    pool = CacheCapsulePool(apc.capsule_generation, adapter=Adapter(), enabled=True)
    with pytest.raises(CacheCapsuleError, match="compatibility"):
        pool.prepare(source, primary="external", fallback=None, timeout_s=1)
    assert backing.releases == 1
    assert not discarded
    assert not pool._tickets and pool._active is None
    assert pool.counters["errors"] == 1


def test_block_file_roundtrip_identity_and_corruption_sidecars(tmp_path):
    path = tmp_path / "cache.safetensors"
    payload = bytes(range(251)) * 5
    path.write_bytes(payload)
    paths = encode_block_file(path, block_bytes=128, signature="exact-id:tokens=17")
    assert len(paths) > 2
    with materialize_block_file(path, expected_signature="exact-id:tokens=17") as restored:
        assert restored.read_bytes() == payload
    with pytest.raises(ValueError, match="signature"):
        with materialize_block_file(path, expected_signature="wrong"):
            pass
    block = block_file_paths(path)[1]
    block.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        with materialize_block_file(path, expected_signature="exact-id:tokens=17"):
            pass
    assert remove_block_file(path) > 0


def test_failed_reconstruction_cleans_partial_restore_and_startup_orphans(tmp_path):
    path = tmp_path / "apc-idle-test.safetensors"
    path.write_bytes(b"abcdefghijkl")
    encode_block_file(path, block_bytes=4, signature="exact")
    block = block_file_paths(path)[-1]
    block.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        with materialize_block_file(path, expected_signature="exact"):
            pass
    assert not list(tmp_path.glob(".apc-idle-*.restore.safetensors"))

    orphan = tmp_path / ".apc-idle-abandoned.restore.safetensors"
    orphan.write_bytes(b"partial")
    APCv2(max_bytes=1 << 20, layout_name="layout", idle_disk_seconds=1,
          idle_disk_dir=str(tmp_path))
    assert not orphan.exists()


def test_block_manifest_paths_fail_closed_for_materialize_and_remove(tmp_path):
    path = tmp_path / "cache.safetensors"
    path.write_bytes(b"payload")
    encode_block_file(path, block_bytes=4, signature="exact")
    outside = tmp_path / "outside.block"
    outside.write_bytes(b"must-survive")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["blocks"][0]["name"] = "../outside.block"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="basename"):
        with materialize_block_file(path, expected_signature="exact"):
            pass
    remove_block_file(path)
    assert outside.read_bytes() == b"must-survive"
    assert not path.exists()


def test_apcv2_startup_removes_orphaned_block_directories(tmp_path):
    orphan = tmp_path / "apc-idle-orphan.safetensors.blocks"
    orphan.mkdir()
    (orphan / "00000000-dead.block").write_bytes(b"orphan")
    unrelated = tmp_path / "other-product.safetensors.blocks"
    unrelated.mkdir()
    (unrelated / "keep.block").write_bytes(b"owned elsewhere")
    APCv2(
        max_bytes=1 << 20, layout_name="layout", idle_disk_seconds=1,
        idle_disk_dir=str(tmp_path), persistent_block_bytes=64,
    )
    assert not orphan.exists()
    assert (unrelated / "keep.block").read_bytes() == b"owned elsewhere"


def test_pending_capsule_member_removal_releases_whole_group():
    apc = APCv2(max_bytes=1 << 20, layout_name="layout")
    pool = CacheCapsulePool(apc.capsule_generation, enabled=True)
    prepared = prepare_prompt_cache_capsules(
        [_cache()], target_batch=2, generation=0, pool=pool,
        compatibility_signature=apc.capsule_compatibility_signature(_key(), "layout"),
        backend="cpu", fallback=None,
    )
    generator = BatchGenerator(None, completion_batch_size=2, prefill_batch_size=2)
    try:
        uids = generator.insert(
            [[9], [9]], caches=[[_cache()], [_cache()]], all_tokens=[[1], [1]]
        )
        assert not generator.bind_cache_capsule("pending", uids[0], prepared, 2)
        generator.remove([uids[0]])
        assert not generator._cache_capsule_pending
        assert all(receipt.owner.released for receipt in prepared.receipts)
        receipt = generator.pop_cache_capsule_receipt(uids[0])
        assert receipt["reason"] == "pending_member_removed"
    finally:
        generator.close(); pool.close()


def test_capsule_cleanup_is_safe_for_partially_initialized_generator():
    generator = BatchGenerator.__new__(BatchGenerator)
    generator._old_wired_limit = None
    generator._release_cache_capsule_uid(123)
    assert generator.pop_cache_capsule_receipt(123) is None
    generator.close()


def test_apcv2_block_persistence_validates_token_count_and_exact_nontrimmable(tmp_path):
    now = [0.0]
    apc = APCv2(
        max_size=4, max_bytes=1 << 20, layout_name="layout",
        idle_disk_seconds=1, idle_disk_dir=str(tmp_path),
        persistent_block_bytes=64, now_fn=lambda: now[0],
    )
    key, tokens = _key(), list(range(5))
    apc.store(key, tokens, [_cache(5)])
    now[0] = 2.0
    assert apc.spill_idle_entries(now=now[0]) == 1
    hit = apc.lookup(key, tokens)
    # PrefixIndex deliberately leaves the final prompt token for authoritative
    # evaluation, even on an exact token-list lookup.
    assert hit.hit and hit.cached_tokens == len(tokens) - 1
    assert apc.apc_stats["idle_disk"]["restores"] == 1
    hit.cache.close()

    # Re-spill and poison only the recorded logical token count: restore must
    # fail closed before the snapshot is exposed.
    now[0] = 4.0
    assert apc.spill_idle_entries(now=now[0]) == 1
    entry = apc._trie.get(key, tokens)
    entry._apc_disk["token_count"] += 1
    miss = apc.lookup(key, tokens)
    assert not miss.hit

    recurrent = ArraysCache(1)
    recurrent[0] = mx.ones((1, 2), dtype=mx.float32)
    recurrent.lengths = mx.array([3], dtype=mx.int32)
    recurrent._host_lengths = (recurrent.lengths, [3])
    exact_tokens = [7, 8, 9]
    apc.store(key, exact_tokens, [recurrent])
    exact = apc.lookup(key, exact_tokens)
    assert not exact.hit and exact.miss_reason.startswith("untrimmable_")
    longer = apc.lookup(key, [7, 8, 0])
    assert not longer.hit


def test_persistent_block_manifest_is_authenticated_not_just_checksummed(tmp_path):
    """A neighbour that swaps a block payload can recompute SHA-256s and
    rewrite the manifest; it cannot forge the per-process MAC."""
    import json

    import pytest

    from mlx2.runtime.persistent_blocks import (
        _digest,
        encode_block_file,
        materialize_block_file,
    )

    path = tmp_path / "snap.safetensors"
    path.write_bytes(b"A" * 10000)
    encode_block_file(path, block_bytes=4096, signature="sig")
    with materialize_block_file(path, expected_signature="sig") as restored:
        assert restored.read_bytes() == b"A" * 10000
    # Forge: swap the payload and recompute every plain digest.
    directory = path.with_suffix(path.suffix + ".blocks")
    forged = b"B" * 10000
    for block in directory.glob("*.block"):
        block.unlink()
    manifest = json.loads(path.read_text())
    manifest["blocks"] = []
    for index, start in enumerate(range(0, len(forged), 4096)):
        payload = forged[start:start + 4096]
        name = f"{index:08d}-{_digest(payload)}.block"
        (directory / name).write_bytes(payload)
        manifest["blocks"].append({"name": name, "size": len(payload), "sha256": _digest(payload)})
    manifest["sha256"] = _digest(forged)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="authentication failed"):
        with materialize_block_file(path, expected_signature="sig"):
            pass

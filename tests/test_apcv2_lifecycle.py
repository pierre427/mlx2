# Adapted from unified tests/test_apc_v2.py at 1e2bc604, MIT.
"""APCv2 model opt-in, layer segmentation, and atomic restore contracts."""

import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import mlx.core as mx
import pytest
from mlx2.runtime.apc_v2 import APCKey, APCv2, MTPAPCSidecar
from mlx2.runtime.cache_planes import (
    CachePlaneKind,
    TranscriptLedgerPlane,
    TranscriptLedgerSegment,
)
from mlx2.runtime.models.cache import (
    ArraysCache,
    KVCache,
    PrefixIndex,
    RotatingKVCache,
    _copy_prompt_cache_for_restore,
    achievable_trim,
)


def _state(cache, length, seed=0):
    values = mx.arange(seed, seed + length, dtype=mx.float32).reshape(1, 1, length, 1)
    cache.update_and_fetch(values, values)
    mx.eval(cache.state)
    return cache


def _recurrent(length):
    cache = ArraysCache(1)
    cache[0] = mx.ones((1, 4), dtype=mx.float32)
    cache.lengths = mx.array([length], dtype=mx.int32)
    cache._host_lengths = (cache.lengths, [length])
    mx.eval(cache.state)
    return cache


def test_prefix_index_removal_delta_matches_full_trie_scan():
    from mlx2.runtime.apc_v2 import _iter_trie_entries

    def insert(index, tokens, *, expected_reason):
        before = {id(entry) for entry in _iter_trie_entries(index._trie)}
        removed = []
        assert index.insert_cache(
            "model", tokens, [_state(KVCache(), len(tokens))],
            removed_entries=removed,
        )
        after = {id(entry) for entry in _iter_trie_entries(index._trie)}
        assert {id(entry) for _, _, _, entry in removed} == before - after
        assert len(removed) == len(before - after)
        assert [reason for reason, _, _, _ in removed] == expected_reason

    index = PrefixIndex(max_size=2)
    insert(index, [1, 2], expected_reason=[])
    insert(index, [1, 2, 3], expected_reason=["subsumed"])
    insert(index, [1, 2, 3], expected_reason=["replaced"])
    insert(index, [8, 9], expected_reason=[])
    insert(index, [10, 11], expected_reason=["size_limit"])

    byte_limited = PrefixIndex(
        max_size=10, max_bytes=_state(KVCache(), 1).nbytes
    )
    insert(byte_limited, [1], expected_reason=[])
    insert(byte_limited, [2], expected_reason=["byte_limit"])


def test_apcv2_store_cleanup_needs_no_full_trie_scan(monkeypatch):
    import mlx2.runtime.apc_v2 as module

    apc = APCv2(max_size=8, layout_name="no-store-scan")
    key = APCKey("model")
    apc.store(key, [1, 2], [_state(KVCache(), 2)])

    def forbid_scan(_trie):
        raise AssertionError("store traversed the whole trie")

    monkeypatch.setattr(module, "_iter_trie_entries", forbid_scan)
    apc.store(key, [1, 2, 3], [_state(KVCache(), 3)])
    apc.store(key, [1, 2, 3], [_state(KVCache(), 3, seed=10)])
    assert apc._trie.get(key, [1, 2, 3]) is not None


def test_third_party_cache_without_checkpoint_api_fails_closed():
    class ExternalRotatingCache:
        offset = 600
        state = ()

        @staticmethod
        def is_trimmable():
            return False

    assert achievable_trim([ExternalRotatingCache()], 1) is None


def test_apcv2_records_layers_token_segments_and_mtp_plane():
    apc = APCv2(max_size=4, layout_name="test-hybrid-v1")
    target = [
        _recurrent(600),
        _state(KVCache(), 600),
        _state(RotatingKVCache(max_size=128), 600, seed=1000),
    ]
    draft = [_state(KVCache(), 599, seed=2000)]
    sidecar = MTPAPCSidecar(
        (draft, mx.ones((1, 1, 4), dtype=mx.float32)), covered_tokens=600
    )
    apc.store(APCKey("qwen4"), list(range(600)), target, sidecar=sidecar)
    hit = apc.lookup(APCKey("qwen4"), list(range(601)))
    assert hit.hit_kind == "mtp_sidecar"
    assert hit.cached_tokens == 600
    assert hit.segment_manifest["schema"] == "apcv2.layer-segments.v1"
    assert hit.segment_manifest["layers"] == 3
    assert hit.segment_manifest["by_plane"]["gdn_recurrent"]["segments"] == 1
    assert hit.segment_manifest["by_plane"]["attention_kv"]["segments"] == 2
    assert hit.segment_manifest["by_plane"]["attention_ring"]["segments"] == 1
    assert hit.segment_manifest["by_plane"]["mtp_draft"]["segments"] == 2
    stats = apc.apc_stats
    assert stats["version"] == 2
    assert stats["layout_name"] == "test-hybrid-v1"
    assert stats["layer_segments"]["disk_entries"] == 0
    assert stats["layer_segments"]["segments"] == 6


def test_apcv2_reuse_telemetry_tracks_hit_age_idle_and_hit_count():
    """sglang#38559: reuse observations stay host-only and fixed-bucketed."""
    now = [0.0]
    apc = APCv2(
        max_size=1,
        layout_name="reuse-telemetry-v1",
        now_fn=lambda: now[0],
    )
    first = APCKey("first")
    apc.store(first, [1, 2], [_state(KVCache(), 2)])
    now[0] = 6.0
    hit = apc.lookup(first, [1, 2, 3])
    assert hit.hit
    hit.cache.close()
    now[0] = 40.0
    apc.store(APCKey("second"), [4, 5], [_state(KVCache(), 2)])

    reuse = apc.apc_stats["reuse_telemetry"]
    assert reuse["hit_age_seconds"]["count"] == 1
    assert reuse["hit_age_seconds"]["sum"] == 6.0
    assert reuse["hit_age_seconds"]["bucket_counts"] == (0, 0, 1, 1, 1, 1)
    assert reuse["eviction_age_seconds"]["count"] == 1
    assert reuse["eviction_age_seconds"]["sum"] == 40.0
    assert reuse["eviction_idle_seconds"]["sum"] == 34.0
    assert reuse["eviction_hit_count"]["bucket_counts"] == (0, 1, 1, 1, 1)
    apc.clear(release_memory=False)


@pytest.mark.parametrize(
    ("query", "cached_tokens"),
    [([1, 2, 3, 4], 3), ([1, 2, 9], 2)],
)
def test_apcv2_records_hits_on_the_entry_selected_for_trim(query, cached_tokens):
    apc = APCv2(max_size=4, layout_name="selected-entry-hit-v1")
    key = APCKey("selected")
    apc.store(
        key,
        [1, 2, 3, 4],
        [_state(KVCache(), 4)],
        retention_role="committed_prompt_boundary",
    )

    hit = apc.lookup(key, query)
    assert hit.hit and hit.cached_tokens == cached_tokens
    assert hit.retention_role == "committed_prompt_boundary"
    entry = apc._trie.get(key, [1, 2, 3, 4])
    assert entry._apc_hit_count == 1
    assert apc.apc_stats["reuse_telemetry"]["hit_age_seconds"]["count"] == 1
    hit.cache.close()
    apc.clear(release_memory=False)


def test_apcv2_republish_preserves_retention_age_and_hits_without_eviction():
    now = [0.0]
    apc = APCv2(
        max_size=4,
        layout_name="same-boundary-republish-v1",
        now_fn=lambda: now[0],
    )
    key = APCKey("republish")
    tokens = [1, 2, 3, 4]
    apc.store(key, tokens, [_state(KVCache(), 4)])
    now[0] = 6.0
    hit = apc.lookup(key, tokens + [5])
    assert hit.hit
    hit.cache.close()

    now[0] = 10.0
    apc.store(key, tokens, [_state(KVCache(), 4, seed=10)])
    entry = apc._trie.get(key, tokens)
    assert entry._apc_inserted_at == 0.0
    assert entry._apc_last_access_at == 6.0
    assert entry._apc_hit_count == 1
    reuse = apc.apc_stats["reuse_telemetry"]
    assert reuse["eviction_age_seconds"]["count"] == 0
    apc.clear(release_memory=False)


def test_apcv2_evicts_interior_checkpoint_before_default_and_boundary():
    """Under the shared byte budget, an unreused interior checkpoint goes first."""
    probe = APCv2(max_size=8, layout_name="retention-order-v1")
    probe.store(APCKey("probe"), [9], [_state(KVCache(), 1)])
    entry_bytes = int(probe.nbytes)
    probe.clear(release_memory=False)
    apc = APCv2(
        max_size=8, max_bytes=2 * entry_bytes, layout_name="retention-order-v1"
    )
    key = APCKey("retention")
    apc.store(
        key,
        [1],
        [_state(KVCache(), 1)],
        retention_role="committed_prompt_boundary",
    )
    apc.store(
        key,
        [3],
        [_state(KVCache(), 1)],
        retention_role="interior_checkpoint",
    )
    apc.store(key, [2], [_state(KVCache(), 1)])
    assert apc.lookup(key, [3, 4]).hit is False
    default = apc.lookup(key, [2, 4])
    boundary = apc.lookup(key, [1, 4])
    assert default.hit and default.retention_role == "default"
    assert boundary.hit and boundary.retention_role == "committed_prompt_boundary"
    default.cache.close()
    boundary.cache.close()
    apc.clear(release_memory=False)


def test_apcv2_interior_checkpoint_survives_a_count_full_cache():
    """Regression (rm04 GPU smoke): with ``max_size`` shared, a full cache of
    prompt boundaries/finished lanes evicted every fresh interior checkpoint
    at its own publication (captured 4/request, published 0, no RAG hits)."""
    apc = APCv2(max_size=2, layout_name="interior-pool-v1")
    key = APCKey("pool")
    apc.store(key, [1], [_state(KVCache(), 1)], retention_role="committed_prompt_boundary")
    apc.store(key, [2], [_state(KVCache(), 1)])
    stored = apc.store(
        key, [3], [_state(KVCache(), 1)], retention_role="interior_checkpoint"
    )
    assert stored.stored is True
    hit = apc.lookup(key, [3, 4])
    assert hit.hit and hit.retention_role == "interior_checkpoint"
    hit.cache.close()
    # The ordinary pool is still capped at max_size, and interiors do not
    # relieve it: a third ordinary entry evicts the least-recent ordinary one.
    apc.store(key, [5], [_state(KVCache(), 1)])
    assert apc._resident_entry_count_locked(interior=False) == 2
    assert apc.lookup(key, [3, 4]).hit
    assert not apc.lookup(key, [2, 4]).hit
    assert apc.apc_stats["interior"]["max_entries"] == 2
    apc.clear(release_memory=False)


def test_apcv2_interior_pool_has_its_own_count_cap():
    apc = APCv2(max_size=8, max_interior_entries=2, layout_name="interior-pool-cap-v1")
    key = APCKey("pool-cap")
    for token in (11, 12, 13):
        apc.store(key, [token], [_state(KVCache(), 1)], retention_role="interior_checkpoint")
    assert apc._resident_entry_count_locked(interior=True) == 2
    # Least-recent unreused interior goes first; the newest publication stays.
    assert not apc.lookup(key, [11, 0]).hit
    assert apc.lookup(key, [13, 0]).hit
    with pytest.raises(ValueError):
        APCv2(max_size=2, max_interior_entries=-1, layout_name="bad-v1")
    apc.clear(release_memory=False)


@pytest.mark.parametrize("prompt_length", [2047, 2048, 2049])
def test_apcv2_hybrid_mtp_boundary_hits_around_prefill_page(prompt_length):
    """P-1 target/P-2 draft coverage must not collapse at a 2048 boundary."""
    covered = prompt_length - 1
    prompt = list(range(prompt_length))
    target = [_recurrent(covered), _state(KVCache(), covered)]
    draft = [_state(KVCache(), covered - 1, seed=prompt_length)]
    sidecar = MTPAPCSidecar(
        (draft, mx.ones((1, 1, 4), dtype=mx.float32)), covered_tokens=covered
    )
    apc = APCv2(max_size=2, layout_name="qwen4-exp-layer-segments-v1")
    apc.store(APCKey("qwen4"), prompt[:covered], target, sidecar=sidecar)
    hit = apc.lookup(APCKey("qwen4"), prompt)
    assert hit.hit
    assert hit.hit_kind == "mtp_sidecar"
    assert hit.cached_tokens == covered
    assert hit.remaining_tokens == [prompt[-1]]
    assert hit.sidecar.covered_tokens == covered
    assert hit.sidecar.state[0][0].offset == covered - 1
    assert hit.cache[0].lengths.item() == covered
    assert hit.cache[1].offset == covered
    assert hit.segment_manifest["by_plane"]["gdn_recurrent"]["segments"] == 1
    assert hit.segment_manifest["by_plane"]["mtp_draft"]["segments"] == 2


def test_apcv2_spills_idle_target_and_mtp_sidecar_then_restores_exactly():
    with tempfile.TemporaryDirectory() as directory:
        now = [0.0]
        apc = APCv2(
            max_size=4,
            layout_name="qwen4-exp-layer-segments-v1",
            idle_disk_seconds=180,
            idle_disk_dir=directory,
            idle_disk_max_bytes=1 << 30,
            now_fn=lambda: now[0],
        )
        target = [_recurrent(3), _state(KVCache(), 3)]
        draft = [_state(KVCache(), 2, seed=100)]
        sidecar = MTPAPCSidecar(
            (draft, mx.ones((1, 1, 4), dtype=mx.float32)),
            covered_tokens=3,
            rng_key=mx.array([7, 11], dtype=mx.uint32),
            rng_draws=5,
        )
        key = APCKey("qwen4")
        apc.store(key, [1, 2, 3], target, sidecar=sidecar)
        now[0] = 179
        assert apc.spill_idle_entries() == 0
        now[0] = 180
        assert apc.spill_idle_entries() == 1
        assert apc.nbytes == 0
        assert apc.apc_stats["idle_disk"]["disk_entries"] == 1
        assert len(list(Path(directory).glob("apc-idle-*.safetensors"))) == 3
        hit = apc.lookup(key, [1, 2, 3, 4])
        assert hit.hit_kind == "mtp_sidecar"
        assert hit.cached_tokens == 3
        assert hit.remaining_tokens == [4]
        assert hit.cache[0].lengths.item() == 3
        assert hit.cache[1].offset == 3
        assert hit.sidecar.state[0][0].offset == 2
        assert hit.sidecar.covered_tokens == 3
        assert hit.sidecar.rng_draws == 5
        assert mx.array_equal(hit.sidecar.rng_key, mx.array([7, 11], dtype=mx.uint32))
        assert apc.apc_stats["idle_disk"]["restores"] == 1
        hit.cache.close()


def test_apcv2_pending_prefetch_cancellation_releases_slot_and_stays_on_disk(
    tmp_path,
):
    apc = APCv2(
        max_size=4,
        layout_name="cancel-prefetch-v1",
        idle_disk_seconds=180,
        idle_disk_dir=str(tmp_path),
    )
    key = APCKey("model")
    tag = ("tenant", "session")
    apc.store(
        key,
        [1, 2, 3],
        [_state(KVCache(), 3)],
        session_tag=tag,
    )
    assert apc.park_session(*tag, ttl_seconds=60)["state"] == "disk"
    apc.resume_session(*tag, ttl_seconds=60)
    assert apc.has_pending_prefetch
    assert apc.cancel_pending_prefetch()
    assert not apc.cancel_pending_prefetch()
    assert not apc.has_pending_prefetch
    assert apc.session_state(*tag)["state"] == "disk"
    assert apc.nbytes == 0
    entry = apc._trie.get(key, [1, 2, 3])
    assert tag not in getattr(entry, "_apc_prefetch_expected", set())
    assert not apc.service_pending_prefetch()
    stats = apc.apc_stats["idle_disk"]
    assert stats["prefetch_restores_cancelled"] == 1
    assert stats["prefetch_restores_abandoned"] == 1

    # Cancellation returns the single-slot semaphore, so a later serving
    # resume can queue normally instead of reporting a permanently full queue.
    apc.resume_session(*tag, ttl_seconds=60)
    assert apc.service_pending_prefetch()
    assert apc.session_state(*tag)["state"] == "resident"
    assert apc.apc_stats["idle_disk"]["prefetch_restores_cancelled"] == 1
    apc.close()


def test_apcv2_suspend_spills_every_resident_entry_and_restores_exactly(tmp_path):
    apc = APCv2(
        max_size=8,
        max_bytes=1 << 30,
        layout_name="qwen4-exp-layer-segments-v1",
        idle_disk_seconds=180,
        idle_disk_dir=str(tmp_path),
    )
    key = APCKey("qwen4-suspend")
    apc.store(
        key,
        [1, 2, 3],
        [_recurrent(3), _state(KVCache(), 3)],
        sidecar=MTPAPCSidecar(
            ([_state(KVCache(), 2, seed=100)], mx.ones((1, 1, 4))),
            covered_tokens=3,
        ),
        retention_role="committed_prompt_boundary",
    )
    apc.store(
        key,
        [7, 8],
        [_state(KVCache(), 2, seed=200)],
        retention_role="interior_checkpoint",
        session_tag=("tenant", "session"),
    )
    before = apc.nbytes
    report = apc.suspend_resident()
    assert report["entries"] == 2
    assert report["failures"] == 0
    assert report["bytes"] == before
    assert report["resident_entries_after"] == 0
    assert report["resident_bytes_after"] == 0
    assert apc.apc_stats["idle_disk"]["disk_entries"] == 2
    assert apc.session_state("tenant", "session")["state"] == "disk"

    hit = apc.lookup(key, [1, 2, 3, 4])
    assert hit.hit_kind == "mtp_sidecar"
    assert hit.cached_tokens == 3
    assert hit.cache[0].lengths.item() == 3
    assert hit.cache[1].offset == 3
    assert hit.sidecar.state[0][0].offset == 2
    hit.cache.close()
    apc.close()


def test_apcv2_suspend_drops_one_failed_entry_and_continues(tmp_path, monkeypatch):
    apc = APCv2(
        max_size=8,
        layout_name="suspend-failure-v1",
        idle_disk_seconds=180,
        idle_disk_dir=str(tmp_path),
    )
    key = APCKey("failure")
    apc.store(key, [1], [_state(KVCache(), 1)])
    apc.store(key, [2], [_state(KVCache(), 1, seed=10)])
    original = apc._spill_entry_locked

    def fail_one(entry_key, tokens, entry, *, reason):
        if tokens == [1]:
            return False
        return original(entry_key, tokens, entry, reason=reason)

    monkeypatch.setattr(apc, "_spill_entry_locked", fail_one)
    report = apc.suspend_resident()
    assert report["entries"] == 1
    assert report["failures"] == 1
    assert not apc.lookup(key, [1, 9]).hit
    restored = apc.lookup(key, [2, 9])
    assert restored.hit
    restored.cache.close()
    apc.close()


def test_apcv2_suspend_refuses_active_lease_without_partial_spill(tmp_path):
    apc = APCv2(
        max_size=8,
        layout_name="suspend-lease-v1",
        idle_disk_seconds=180,
        idle_disk_dir=str(tmp_path),
    )
    key = APCKey("leased")
    apc.store(key, [1, 2], [_state(KVCache(), 2)])
    lease = apc.lookup(key, [1, 2, 3])
    before = apc.nbytes
    with pytest.raises(RuntimeError, match="zero active leases"):
        apc.suspend_resident()
    assert apc.nbytes == before
    assert apc.apc_stats["idle_disk"]["disk_entries"] == 0
    lease.cache.close()
    apc.close()


def test_apcv2_resident_byte_pressure_spills_instead_of_dropping_entry():
    with tempfile.TemporaryDirectory() as directory:
        apc = APCv2(
            max_size=4,
            max_bytes=1,
            layout_name="qwen4-exp-layer-segments-v1",
            idle_disk_seconds=180,
            idle_disk_dir=directory,
        )
        key = APCKey("qwen4")
        apc.store(key, [1, 2, 3], [_state(KVCache(), 3)])
        assert len(apc) == 1
        assert apc.nbytes == 0
        disk = apc.apc_stats["idle_disk"]
        assert disk["pressure_spills"] == 1
        assert disk["disk_entries"] == 1
        entry = apc._trie.get(key, [1, 2, 3])
        apc.max_bytes = int(entry._apc_disk["resident_nbytes"])
        hit = apc.lookup(key, [1, 2, 3, 4])
        assert hit.hit
        assert hit.cached_tokens == 3
        hit.cache.close()


def test_apcv2_rejects_recorded_restore_larger_than_hard_cap_before_io(
    tmp_path, monkeypatch
):
    apc = APCv2(
        max_size=4,
        max_bytes=1 << 20,
        layout_name="test-kv-v1",
        idle_disk_seconds=1,
        idle_disk_dir=str(tmp_path),
    )
    key, tokens = APCKey("oversize-recorded"), [1, 2, 3]
    apc.store(key, tokens, [_state(KVCache(), len(tokens))])
    entry = apc._trie.get(key, tokens)
    with apc._apc_lock:
        assert apc._spill_entry_locked(key, tokens, entry, reason="pressure")
    entry._apc_disk["resident_nbytes"] = 16_384
    apc.max_bytes = 16_383

    def forbidden_load(_path):
        raise AssertionError("oversize recorded restore reached disk I/O")

    monkeypatch.setattr("mlx2.runtime.apc_v2.load_prompt_cache", forbidden_load)
    miss = apc.lookup(key, tokens + [4])
    assert not miss.hit
    assert apc.nbytes == 0
    assert len(apc) == 0
    assert apc.apc_stats["idle_disk"]["restore_failures"] == 1
    assert apc.apc_stats["idle_disk"]["disk_bytes"] == 0
    assert not list(tmp_path.glob("apc-idle-*.safetensors"))
    assert apc.apc_stats["cow"]["active_leases"] == 0


def test_apcv2_reports_checkpoint_above_current_restore_budget(tmp_path, caplog):
    apc = APCv2(
        max_size=4,
        max_bytes=1,
        layout_name="test-kv-v1",
        idle_disk_seconds=1,
        idle_disk_dir=str(tmp_path),
    )
    key, tokens = APCKey("oversize-spill"), [1, 2, 3]
    apc.store(key, tokens, [_state(KVCache(), len(tokens))])

    disk = apc.apc_stats["idle_disk"]
    assert disk["oversize_spills"] == 1
    assert disk["disk_entries"] == 1
    assert "reuse requires a larger cap" in caplog.text
    assert apc.nbytes <= apc.max_bytes


def test_apcv2_logs_unexpected_disk_restore_exception(tmp_path, monkeypatch, caplog):
    apc = APCv2(
        max_size=4,
        max_bytes=1 << 20,
        layout_name="test-kv-v1",
        idle_disk_seconds=1,
        idle_disk_dir=str(tmp_path),
    )
    key, tokens = APCKey("restore-error"), [1, 2, 3]
    apc.store(key, tokens, [_state(KVCache(), len(tokens))])
    entry = apc._trie.get(key, tokens)
    with apc._apc_lock:
        assert apc._spill_entry_locked(key, tokens, entry, reason="pressure")

    def fail_load(_path):
        raise RuntimeError("synthetic restore failure")

    monkeypatch.setattr("mlx2.runtime.apc_v2.load_prompt_cache", fail_load)
    assert not apc.lookup(key, tokens + [4]).hit
    assert apc.apc_stats["idle_disk"]["restore_failures"] == 1
    assert "synthetic restore failure" in caplog.text
    assert "3-token checkpoint" in caplog.text


def test_apcv2_rejects_actual_restore_larger_than_hard_cap_before_publication(
    tmp_path, monkeypatch
):
    apc = APCv2(
        max_size=4,
        max_bytes=1 << 20,
        layout_name="test-kv-v1",
        idle_disk_seconds=1,
        idle_disk_dir=str(tmp_path),
    )
    key, tokens = APCKey("oversize-actual"), [1, 2, 3]
    apc.store(key, tokens, [_state(KVCache(), len(tokens))])
    entry = apc._trie.get(key, tokens)
    with apc._apc_lock:
        assert apc._spill_entry_locked(key, tokens, entry, reason="pressure")
    recorded = int(entry._apc_disk["resident_nbytes"])
    apc.max_bytes = recorded
    oversized = [_state(KVCache(), 257, seed=100)]
    assert sum(item.nbytes for item in oversized) > apc.max_bytes
    monkeypatch.setattr(
        "mlx2.runtime.apc_v2.load_prompt_cache", lambda _path: oversized
    )

    miss = apc.lookup(key, tokens + [4])
    assert not miss.hit
    assert apc.nbytes <= apc.max_bytes
    assert len(apc) == 0
    assert apc.apc_stats["idle_disk"]["restore_failures"] == 1
    assert apc.apc_stats["idle_disk"]["disk_bytes"] == 0
    assert not list(tmp_path.glob("apc-idle-*.safetensors"))
    assert apc.apc_stats["cow"]["active_leases"] == 0


def test_apcv2_concurrent_disk_restore_stays_capped_and_releases_leases(tmp_path):
    apc = APCv2(
        max_size=4,
        max_bytes=1 << 20,
        layout_name="test-kv-v1",
        idle_disk_seconds=1,
        idle_disk_dir=str(tmp_path),
        idle_disk_max_bytes=1 << 20,
    )
    key, tokens = APCKey("concurrent-restore"), list(range(32))
    apc.store(key, tokens, [_state(KVCache(), len(tokens))])
    entry = apc._trie.get(key, tokens)
    with apc._apc_lock:
        assert apc._spill_entry_locked(key, tokens, entry, reason="pressure")
    apc.max_bytes = int(entry._apc_disk["resident_nbytes"])

    def restore_and_close(_worker):
        hit = apc.lookup(key, tokens + [100])
        if hit.cache is not None:
            hit.cache.close()
        return hit.hit, hit.cached_tokens

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(restore_and_close, range(32)))
    assert results == [(True, len(tokens))] * 32
    assert apc.nbytes <= apc.max_bytes
    assert apc.apc_stats["cow"]["active_leases"] == 0
    assert apc.apc_stats["idle_disk"]["disk_bytes"] <= (1 << 20)
    apc.clear()
    assert not list(tmp_path.glob("apc-idle-*.safetensors"))


def _prepare_restore_disk_enforcement_race(tmp_path):
    """Build the actual-size reserve race from the independent review."""
    apc = APCv2(
        max_size=4,
        max_bytes=1 << 20,
        layout_name="test-kv-v1",
        idle_disk_seconds=1,
        idle_disk_dir=str(tmp_path),
        idle_disk_max_bytes=1 << 20,
    )
    key = APCKey("restore-disk-enforcement-race")
    restoring_tokens = [1, 2, 3]
    resident_tokens = [8, 9, 10]
    apc.store(key, restoring_tokens, [_state(KVCache(), 1)])
    restoring_entry = apc._trie.get(key, restoring_tokens)
    assert restoring_entry.nbytes == 2048
    with apc._apc_lock:
        assert apc._spill_entry_locked(
            key, restoring_tokens, restoring_entry, reason="pressure"
        )
    restoring_disk_bytes = sum(
        path.stat().st_size for path in apc._disk_paths(restoring_entry)
    )
    # The recorded estimate admits without spilling B.  Re-reserving the
    # actual 2048-byte footprint must spill B and enforce a disk cap containing
    # only A's files, without evicting the A placeholder being restored.
    restoring_entry._apc_disk["resident_nbytes"] = 1
    apc.store(key, resident_tokens, [_state(KVCache(), 1, seed=100)])
    assert apc.nbytes == 2048
    apc.max_bytes = 2049
    apc._idle_disk_max_bytes = restoring_disk_bytes
    return apc, key, restoring_tokens, resident_tokens


def test_apcv2_restore_survives_disk_enforcement_of_actual_size_reserve(tmp_path):
    apc, key, restoring_tokens, resident_tokens = (
        _prepare_restore_disk_enforcement_race(tmp_path)
    )

    hit = apc.lookup(key, restoring_tokens + [4])
    assert hit.hit
    assert hit.cached_tokens == len(restoring_tokens)
    hit.cache.close()
    assert apc._trie.search(key, restoring_tokens).exact == restoring_tokens
    assert apc._trie.search(key, resident_tokens).exact is None
    assert apc.nbytes <= apc.max_bytes
    assert apc.apc_stats["idle_disk"]["disk_bytes"] <= apc._idle_disk_max_bytes
    assert apc.apc_stats["cow"]["active_leases"] == 0
    apc.clear()
    assert not list(tmp_path.glob("apc-idle-*.safetensors"))


def test_apcv2_concurrent_restore_survives_disk_enforcement_race(tmp_path):
    apc, key, restoring_tokens, _resident_tokens = (
        _prepare_restore_disk_enforcement_race(tmp_path)
    )
    # First exercise the exact actual-size/disk-enforcement race, then place
    # the surviving entry back on disk and race concurrent restore callers.
    first = apc.lookup(key, restoring_tokens + [4])
    assert first.hit
    first.cache.close()
    entry = apc._trie.get(key, restoring_tokens)
    with apc._apc_lock:
        assert apc._spill_entry_locked(key, restoring_tokens, entry, reason="pressure")

    def restore_and_close(_worker):
        hit = apc.lookup(key, restoring_tokens + [4])
        if hit.cache is not None:
            hit.cache.close()
        return hit.hit, hit.cached_tokens

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(restore_and_close, range(32)))
    assert results == [(True, len(restoring_tokens))] * 32
    assert apc.nbytes <= apc.max_bytes
    assert apc.apc_stats["idle_disk"]["disk_bytes"] <= apc._idle_disk_max_bytes
    assert apc.apc_stats["cow"]["active_leases"] == 0
    apc.clear()
    assert not list(tmp_path.glob("apc-idle-*.safetensors"))


def test_apcv2_neighbor_restore_does_not_select_respilled_placeholder():
    """A restored shorter checkpoint may re-spill its longer neighbor."""
    def checkpointed_hybrid(length, seed):
        recurrent = ArraysCache(1)
        recurrent[0] = mx.ones((1, 4), dtype=mx.float32) * seed
        recurrent.lengths = mx.array([length], dtype=mx.int32)
        recurrent._host_lengths = (recurrent.lengths, [length])
        recurrent.state_checkpoint([length], force=True)
        attention = KVCache()
        values = mx.arange(
            seed, seed + length, dtype=mx.float32
        ).reshape(1, 1, length, 1)
        attention.update_and_fetch(values, values)
        mx.eval(recurrent.state, attention.state)
        return [recurrent, attention]

    with tempfile.TemporaryDirectory() as directory:
        apc = APCv2(
            max_size=4,
            max_bytes=1 << 30,
            layout_name="test-checkpointed-hybrid-v1",
            idle_disk_seconds=1,
            idle_disk_dir=directory,
        )
        key = APCKey("model")
        shorter = [1, 2, 3, 4]
        longer = [1, 2, 3, 4, 5, 9]
        query = [1, 2, 3, 4, 5]
        apc.store(key, shorter, checkpointed_hybrid(len(shorter), 1))
        apc.store(key, longer, checkpointed_hybrid(len(longer), 10))
        with apc._apc_lock:
            resident_sizes = []
            for model, path, entry in apc._entry_records_locked():
                resident_sizes.append(int(entry.nbytes))
                assert apc._spill_entry_locked(
                    model, path, entry, reason="pressure"
                )
            assert all(
                not entry.prompt_cache and entry._apc_disk
                for _, _, entry in apc._entry_records_locked()
            )
        apc.max_bytes = max(resident_sizes)
        hit = apc.lookup(key, query)
        assert hit.hit
        assert hit.hit_kind == "prefix"
        assert hit.cached_tokens == len(shorter)
        assert hit.remaining_tokens == [5]
        assert len(hit.cache) == 2
        assert isinstance(hit.cache[0], ArraysCache)
        assert isinstance(hit.cache[1], KVCache)
        assert hit.cache[0].lengths.item() == len(shorter)
        assert hit.cache[1].offset == len(shorter)
        hit.cache.close()


def test_apcv2_malformed_returned_topology_fails_closed_and_releases_lease(
    monkeypatch,
):
    apc = APCv2(max_size=2, layout_name="test-kv-v1")
    key = APCKey("model")
    tokens = [1, 2, 3]
    apc.store(key, tokens, [_state(KVCache(), len(tokens))])
    entry = apc._trie.get(key, tokens)
    malformed = _copy_prompt_cache_for_restore(entry.prompt_cache)
    malformed.cow_prep_telemetry["target_segments"] += 1
    assert apc.apc_stats["cow"]["active_leases"] == 1

    monkeypatch.setattr(
        PrefixIndex,
        "fetch_nearest_cache",
        lambda self, model, query: (malformed, query[-1:]),
    )
    miss = apc.lookup(key, tokens + [4])
    assert not miss.hit
    assert miss.cache is None
    assert miss.remaining_tokens == tokens + [4]
    assert miss.miss_reason == "malformed_cache_topology"
    assert malformed.closed
    assert apc.apc_stats["cow"]["active_leases"] == 0


def test_apcv2_idle_spill_waits_for_live_cow_branch_to_close():
    with tempfile.TemporaryDirectory() as directory:
        now = [0.0]
        apc = APCv2(
            max_size=4,
            layout_name="qwen4-exp-layer-segments-v1",
            idle_disk_seconds=180,
            idle_disk_dir=directory,
            now_fn=lambda: now[0],
        )
        key = APCKey("qwen4")
        apc.store(key, [1, 2, 3], [_state(KVCache(), 3)])
        live = apc.lookup(key, [1, 2, 3, 4]).cache
        now[0] = 180
        assert apc.spill_idle_entries() == 0
        assert apc.nbytes > 0
        live.close()
        now[0] = 181
        assert apc.spill_idle_entries() == 1
        assert apc.nbytes == 0


def test_apcv2_missing_disk_snapshot_fails_closed_as_cache_miss():
    with tempfile.TemporaryDirectory() as directory:
        now = [0.0]
        apc = APCv2(
            max_size=4,
            layout_name="qwen4-exp-layer-segments-v1",
            idle_disk_seconds=180,
            idle_disk_dir=directory,
            now_fn=lambda: now[0],
        )
        key = APCKey("qwen4")
        tokens = [1, 2, 3]
        apc.store(key, tokens, [_state(KVCache(), 3)])
        now[0] = 180
        assert apc.spill_idle_entries() == 1
        entry = apc._trie.get(key, tokens)
        Path(entry._apc_disk["target"]).unlink()
        miss = apc.lookup(key, tokens + [4])
        assert not miss.hit
        assert len(apc) == 0
        assert apc.apc_stats["idle_disk"]["restore_failures"] == 1


def test_apcv2_disk_budget_evicts_oldest_disk_only_entry():
    with tempfile.TemporaryDirectory() as directory:
        apc = APCv2(
            max_size=4,
            max_bytes=1,
            layout_name="qwen4-exp-layer-segments-v1",
            idle_disk_seconds=180,
            idle_disk_dir=directory,
            idle_disk_max_bytes=1,
        )
        apc.store(APCKey("qwen4"), [1, 2, 3], [_state(KVCache(), 3)])
        disk = apc.apc_stats["idle_disk"]
        assert len(apc) == 0
        assert disk["disk_entries"] == 0
        assert disk["disk_evictions"] == 1
        assert not list(Path(directory).glob("apc-idle-*.safetensors"))


def test_apcv2_target_segment_invalidation_rejects_atomic_restore():
    apc = APCv2(max_size=2, layout_name="test-kv-v1")
    key = APCKey("model")
    apc.store(key, list(range(300)), [_state(KVCache(), 300)])
    entry = apc._trie.get(key, list(range(300)))
    owner = entry.prompt_cache.cow_owner
    target_segment = next(
        (
            segment
            for segment in owner.segment_manifest.segments
            if segment.key.kind == CachePlaneKind.ATTENTION_KV
        )
    )
    assert owner.invalidate_segment(target_segment.key, "test-stale")
    miss = apc.lookup(key, list(range(301)))
    assert not miss.hit
    assert miss.miss_reason == "stale_cow_generation"


def test_apcv2_transcript_segments_are_optional_and_independently_invalidatable():
    apc = APCv2(max_size=2, layout_name="test-kv-v1")
    key = APCKey("model")
    ledger = TranscriptLedgerPlane(
        "test-tokenizer",
        "rev-a",
        "transcript-a",
        (
            TranscriptLedgerSegment("turn:1", 0, 3, (7, 8, 9)),
            TranscriptLedgerSegment("turn:2", 3, 5, (10, 11)),
        ),
    )
    tokens = [1, 2, 3]
    apc.store(key, tokens, [_state(KVCache(), len(tokens))], transcript_ledger=ledger)
    hit = apc.lookup(key, tokens + [4])
    assert hit.hit
    assert hit.transcript_ledger is ledger
    assert hit.segment_manifest["by_plane"]["transcript_ledger"]["segments"] == 2
    entry = apc._trie.get(key, tokens)
    owner = entry.prompt_cache.cow_owner
    transcript_segment = next(
        (
            segment
            for segment in owner.segment_manifest.segments
            if segment.key.kind == CachePlaneKind.TRANSCRIPT_LEDGER
        )
    )
    assert owner.invalidate_segment(transcript_segment.key, "ledger-stale")
    target_only_hit = apc.lookup(key, tokens + [4])
    assert target_only_hit.hit
    assert target_only_hit.transcript_ledger is None


def test_paired_sidecar_reuse_refreshes_lru_checkpoint_priority():
    apc = APCv2(max_size=2, layout_name="test-hybrid-v1")
    key = APCKey("lru", revision="v1")
    def store(prefix):
        target = [_recurrent(3), _state(KVCache(), 3)]
        draft = [_state(KVCache(), 2)]
        apc.store(key, prefix, target, sidecar=MTPAPCSidecar(
            (draft, mx.ones((1, 1, 4))), covered_tokens=3))
    store([1, 2, 3])
    store([8, 9, 10])
    first = apc.lookup(key, [1, 2, 3, 4])
    assert first.hit_kind == "mtp_sidecar"
    first.cache.close()
    store([20, 21, 22])
    reused = apc.lookup(key, [1, 2, 3, 4])
    assert reused.hit_kind == "mtp_sidecar"
    reused.cache.close()
    assert not apc.lookup(key, [8, 9, 10, 11]).hit
    apc.clear()


def test_pressure_eviction_preserves_leased_checkpoint_generation():
    apc = APCv2(max_size=4, layout_name="test-kv-v1")
    key = APCKey("pressure")
    apc.store(key, [1, 2, 3], [_state(KVCache(), 3)])
    hit = apc.lookup(key, [1, 2, 3, 4], allow_disk_restore=False)
    owner, generation = hit.cache.cow_owner, hit.cache.cow_generation
    apc.store(key, [8, 9, 10], [_state(KVCache(), 3)])
    assert apc.evict_oldest_unleased()
    assert owner.generation == generation
    assert not apc.evict_oldest_unleased()
    assert apc.lookup(key, [8, 9, 10, 11]).hit is False
    hit.cache.close()
    assert apc.evict_oldest_unleased()
    assert len(apc) == 0


def test_pressure_prefers_terminal_tail_and_retains_exact_mtp_prompt_boundary():
    """A P+tail checkpoint cannot replace the exact P-1/P-2 MTP boundary."""
    apc = APCv2(max_size=4, layout_name="test-hybrid-v1")
    key = APCKey("near-limit", revision="v1")
    prompt = list(range(64))
    covered = len(prompt) - 1

    boundary_target = [_recurrent(covered), _state(KVCache(), covered)]
    boundary_draft = [_state(KVCache(), covered - 1, seed=100)]
    apc.store(
        key,
        prompt[:covered],
        boundary_target,
        sidecar=MTPAPCSidecar(
            (boundary_draft, mx.ones((1, 1, 4))), covered_tokens=covered
        ),
        retention_role="committed_prompt_boundary",
    )

    terminal = prompt + list(range(100, 163))
    terminal_target = [_recurrent(len(terminal)), _state(KVCache(), len(terminal))]
    terminal_draft = [_state(KVCache(), len(terminal) - 1, seed=200)]
    apc.store(
        key,
        terminal,
        terminal_target,
        sidecar=MTPAPCSidecar(
            (terminal_draft, mx.ones((1, 1, 4))), covered_tokens=len(terminal)
        ),
    )

    assert apc.evict_oldest_unleased()
    assert len(apc) == 1
    hit = apc.lookup(key, prompt)
    assert hit.hit_kind == "mtp_sidecar"
    assert hit.cached_tokens == covered
    assert hit.remaining_tokens == [prompt[-1]]
    hit.cache.close()
    assert apc.apc_stats["cow"]["active_leases"] == 0
    # A sole unleased boundary is the final pressure candidate, so admission
    # can still make progress when retaining it would deadlock reclamation.
    assert apc.evict_oldest_unleased()
    assert len(apc) == 0
    apc.clear()


def test_max_size_assigns_role_before_limits_and_repeated_mtp_reuse_has_no_leases():
    apc = APCv2(max_size=1, layout_name="test-hybrid-v1")
    key = APCKey("role-before-limit", revision="v1")
    prompt = list(range(9))
    covered = len(prompt) - 1

    for seed in range(3):
        apc.store(
            key,
            prompt[:covered],
            [_recurrent(covered), _state(KVCache(), covered, seed=seed * 100)],
            sidecar=MTPAPCSidecar(
                ([_state(KVCache(), covered - 1, seed=seed * 1000)],
                 mx.ones((1, 1, 4))),
                covered_tokens=covered,
            ),
            retention_role="committed_prompt_boundary",
        )
        apc.store(
            key,
            prompt + [100 + seed],
            [_recurrent(len(prompt) + 1),
             _state(KVCache(), len(prompt) + 1, seed=5000 + seed)],
            sidecar=MTPAPCSidecar(
                ([_state(KVCache(), len(prompt), seed=6000 + seed)],
                 mx.ones((1, 1, 4))),
                covered_tokens=len(prompt) + 1,
            ),
        )
        assert len(apc) == 1
        hit = apc.lookup(key, prompt)
        assert hit.hit_kind == "mtp_sidecar"
        assert hit.cached_tokens == covered
        hit.cache.close()
        assert apc.apc_stats["cow"]["active_leases"] == 0
    apc.clear()


def test_b20_capacity_retains_one_warm_prompt_boundary_per_lane():
    """A B20 cohort can keep all 20 distinct primed prompts warm at once."""
    apc = APCv2(max_size=20, layout_name="test-kv-v1")
    key = APCKey("b20-warm-boundaries", revision="v1")
    prompts = [[lane + 1, 1000 + lane] for lane in range(20)]

    for lane, prompt in enumerate(prompts):
        apc.store(
            key,
            prompt,
            [_state(KVCache(), len(prompt), seed=lane * 100)],
            retention_role="committed_prompt_boundary",
        )

    assert len(apc) == 20
    for prompt in prompts:
        hit = apc.lookup(key, prompt)
        assert hit.hit
        # Decode keeps the final prompt token as the uncached continuation;
        # every lane still has a nonzero APCv2 warm prefix.
        assert hit.cached_tokens == len(prompt) - 1
        assert hit.remaining_tokens == [prompt[-1]]
        hit.cache.close()
    assert apc.apc_stats["cow"]["active_leases"] == 0
    apc.clear()


def test_sole_prompt_boundary_is_last_resort_for_hard_resident_byte_cap():
    apc = APCv2(max_size=4, max_bytes=1, layout_name="test-hybrid-v1")
    key = APCKey("hard-byte-cap")
    apc.store(
        key,
        [1, 2, 3],
        [_state(KVCache(), 3)],
        sidecar=MTPAPCSidecar(
            ([_state(KVCache(), 2, seed=20)], mx.ones((1, 1, 4))),
            covered_tokens=3,
        ),
        retention_role="committed_prompt_boundary",
    )
    assert apc.nbytes <= apc.max_bytes
    assert len(apc) == 0


def test_disk_budget_keeps_boundary_ahead_of_terminal_after_restore(tmp_path):
    apc = APCv2(
        max_size=4,
        max_bytes=1 << 30,
        layout_name="test-hybrid-v1",
        idle_disk_seconds=180,
        idle_disk_dir=str(tmp_path),
        idle_disk_max_bytes=1 << 30,
    )
    key = APCKey("disk-role-order")
    boundary_tokens = [1, 2, 3]
    apc.store(
        key,
        boundary_tokens,
        [_state(KVCache(), 3)],
        sidecar=MTPAPCSidecar(
            ([_state(KVCache(), 2, seed=30)], mx.ones((1, 1, 4))),
            covered_tokens=3,
        ),
        retention_role="committed_prompt_boundary",
    )
    terminal_tokens = [8, 9, 10, 11]
    apc.store(key, terminal_tokens, [_state(KVCache(), 4, seed=40)])
    with apc._apc_lock:
        records = list(apc._entry_records_locked())
        for model, tokens, entry in records:
            assert apc._spill_entry_locked(model, tokens, entry, reason="pressure")
    hit = apc.lookup(key, boundary_tokens + [4])
    assert hit.hit_kind == "mtp_sidecar"
    hit.cache.close()
    boundary_entry = apc._trie.get(key, boundary_tokens)
    boundary_disk_bytes = sum(
        path.stat().st_size for path in apc._disk_paths(boundary_entry)
    )
    with apc._apc_lock:
        apc._idle_disk_max_bytes = boundary_disk_bytes
        apc._enforce_disk_limit_locked()
    assert apc._trie.search(key, terminal_tokens).exact is None
    assert getattr(boundary_entry, "_apc_disk", None)
    assert apc.apc_stats["idle_disk"]["disk_bytes"] <= boundary_disk_bytes
    apc.clear()


def test_disk_full_does_not_write_disposable_checkpoint_then_evict_it(tmp_path):
    apc = APCv2(
        max_size=4,
        max_bytes=1 << 20,
        layout_name="test-kv-v1",
        idle_disk_seconds=180,
        idle_disk_dir=str(tmp_path),
        idle_disk_max_bytes=1 << 20,
    )
    key = APCKey("disk-write-economics")
    boundary = [1, 2, 3]
    rolling = [8, 9, 10]
    apc.store(
        key, boundary, [_state(KVCache(), 3)],
        retention_role="committed_prompt_boundary",
    )
    assert apc.evict_oldest_unleased()
    boundary_disk_bytes = apc.apc_stats["idle_disk"]["disk_bytes"]
    assert boundary_disk_bytes > 0
    apc._idle_disk_max_bytes = boundary_disk_bytes

    apc.store(
        key, rolling, [_state(KVCache(), 3, seed=33)],
        retention_role="prefill_rolling",
    )
    written_before = apc.apc_stats["idle_disk"]["bytes_written"]
    assert apc.evict_oldest_unleased()
    disk = apc.apc_stats["idle_disk"]
    assert disk["bytes_written"] == written_before
    assert disk["spill_capacity_skips"] == 1
    assert disk["disk_evictions"] == 0
    assert apc._trie.search(key, rolling).exact is None
    assert apc._trie.search(key, boundary).exact is not None

    # Same-rank newer boundaries are still allowed to replace an older one.
    newer_boundary = [20, 21, 22]
    apc.store(
        key, newer_boundary, [_state(KVCache(), 3, seed=44)],
        retention_role="committed_prompt_boundary",
    )
    assert apc.evict_oldest_unleased()
    disk = apc.apc_stats["idle_disk"]
    assert disk["bytes_written"] > written_before
    assert disk["spill_capacity_skips"] == 1
    assert apc._trie.search(key, newer_boundary).exact is not None


def test_explicit_pressure_reclaims_sole_unleased_prompt_boundary():
    apc = APCv2(max_size=4, layout_name="test-hybrid-v1")
    key = APCKey("sole-boundary-pressure")
    apc.store(
        key,
        [1, 2, 3],
        [_state(KVCache(), 3)],
        retention_role="committed_prompt_boundary",
    )
    assert apc.evict_oldest_unleased()
    assert len(apc) == 0


def test_explicit_pressure_spills_sole_mtp_boundary_and_restores_exactly(tmp_path):
    """Serving pressure must preserve the only reusable P-1 target+MTP state."""
    apc = APCv2(
        max_size=4,
        layout_name="test-hybrid-v1",
        idle_disk_seconds=180,
        idle_disk_dir=str(tmp_path),
        idle_disk_max_bytes=1 << 30,
    )
    key = APCKey("sole-boundary-pressure-disk", revision="v1")
    prompt = [1, 2, 3, 4]
    covered = len(prompt) - 1
    target = [_recurrent(covered), _state(KVCache(), covered, seed=10)]
    draft = [_state(KVCache(), covered - 1, seed=100)]
    sidecar = MTPAPCSidecar(
        (draft, mx.arange(4, dtype=mx.float32).reshape(1, 1, 4)),
        covered_tokens=covered,
        rng_key=mx.array([7, 11], dtype=mx.uint32),
        rng_draws=5,
    )
    target_attention = tuple(mx.array(row) for row in target[1].state)
    draft_attention = tuple(mx.array(row) for row in draft[0].state)
    apc.store(
        key,
        prompt[:covered],
        target,
        sidecar=sidecar,
        retention_role="committed_prompt_boundary",
    )

    resident_bytes = apc.nbytes
    assert resident_bytes > 0
    assert apc.evict_oldest_unleased()
    assert apc.nbytes == 0
    assert len(apc) == 1
    disk = apc.apc_stats["idle_disk"]
    assert disk["pressure_spills"] == 1
    assert disk["disk_entries"] == 1

    # Production Qwen target+MTP snapshots are about 8.18 GiB.  Exercise the
    # same admission arithmetic without allocating that payload in a CPU test;
    # concrete restored arrays remain authoritative for publication/accounting.
    entry = apc._trie.get(key, prompt[:covered])
    entry._apc_disk["resident_nbytes"] = int(8.18 * (1 << 30))
    apc.max_bytes = 16 << 30

    cold = apc.lookup(key, prompt, allow_disk_restore=False)
    assert not cold.hit
    assert cold.miss_reason == "disk_restore_requires_admission"

    hit = apc.lookup(key, prompt)
    assert hit.hit_kind == "mtp_sidecar"
    assert hit.cached_tokens == covered
    assert hit.remaining_tokens == [prompt[-1]]
    assert hit.cache[0].lengths.item() == covered
    assert hit.cache[1].offset == covered
    assert hit.sidecar.covered_tokens == covered
    assert hit.sidecar.state[0][0].offset == covered - 1
    assert hit.sidecar.rng_draws == 5
    assert mx.array_equal(hit.sidecar.rng_key, mx.array([7, 11], dtype=mx.uint32))
    assert all(
        mx.array_equal(actual, expected)
        for actual, expected in zip(hit.cache[1].state, target_attention)
    )
    assert all(
        mx.array_equal(actual, expected)
        for actual, expected in zip(hit.sidecar.state[0][0].state, draft_attention)
    )
    hit.cache.close()
    assert apc.apc_stats["cow"]["active_leases"] == 0
    apc.clear()


def test_explicit_pressure_spill_failure_drops_and_progresses(tmp_path, monkeypatch):
    apc = APCv2(
        max_size=4,
        layout_name="test-hybrid-v1",
        idle_disk_seconds=180,
        idle_disk_dir=str(tmp_path),
    )
    key = APCKey("pressure-spill-failure")
    apc.store(
        key,
        [1, 2, 3],
        [_state(KVCache(), 3)],
        retention_role="committed_prompt_boundary",
    )

    def fail_spill(*_args, **_kwargs):
        raise OSError("simulated full disk")

    monkeypatch.setattr(apc, "_atomic_save_cache", fail_spill)
    assert apc.evict_oldest_unleased()
    assert apc.nbytes == 0
    assert len(apc) == 0
    assert apc.apc_stats["idle_disk"]["spill_failures"] == 1
    assert apc.apc_stats["cow"]["active_leases"] == 0


def test_default_role_preserves_prefix_index_mixed_cache_type_order():
    apc = APCv2(max_size=3, layout_name="test-kv-v1")
    key = APCKey("legacy-cache-type-order")
    rows = (
        ([1, 1], "assistant"),
        ([2, 2], "user"),
        ([3, 3], "system"),
        ([4, 4], "assistant"),
    )
    for seed, (tokens, cache_type) in enumerate(rows):
        apc.store(
            key,
            tokens,
            [_state(KVCache(), len(tokens), seed=seed * 10)],
            cache_type=cache_type,
        )
    assert apc._trie.search(key, [1, 1]).exact is None
    assert all(apc._trie.get(key, tokens) is not None for tokens, _ in rows[1:])
    apc.clear()


def test_resident_budget_spills_terminal_tail_before_exact_mtp_boundary(tmp_path):
    apc = APCv2(
        max_size=4,
        max_bytes=1 << 30,
        layout_name="test-hybrid-v1",
        idle_disk_seconds=180,
        idle_disk_dir=str(tmp_path),
    )
    key = APCKey("near-limit-budget", revision="v1")
    prompt = list(range(64))
    covered = len(prompt) - 1
    boundary_sidecar = MTPAPCSidecar(
        ([_state(KVCache(), covered - 1, seed=300)], mx.ones((1, 1, 4))),
        covered_tokens=covered,
    )
    apc.store(
        key,
        prompt[:covered],
        [_recurrent(covered), _state(KVCache(), covered)],
        sidecar=boundary_sidecar,
        retention_role="committed_prompt_boundary",
    )
    boundary_bytes = apc.nbytes

    terminal = prompt + list(range(100, 163))
    terminal_sidecar = MTPAPCSidecar(
        ([_state(KVCache(), len(terminal) - 1, seed=400)], mx.ones((1, 1, 4))),
        covered_tokens=len(terminal),
    )
    terminal_target = [_recurrent(len(terminal)), _state(KVCache(), len(terminal))]
    terminal_bytes = sum((cache.nbytes for cache in terminal_target)) + terminal_sidecar.nbytes
    apc.max_bytes = boundary_bytes + terminal_bytes - 1
    apc.store(key, terminal, terminal_target, sidecar=terminal_sidecar)

    stats = apc.apc_stats
    assert stats["idle_disk"]["pressure_spills"] == 1
    assert stats["idle_disk"]["disk_entries"] == 1
    assert stats["idle_disk"]["resident_bytes"] == boundary_bytes
    for _ in range(2):
        hit = apc.lookup(key, prompt)
        assert hit.hit_kind == "mtp_sidecar"
        assert hit.cached_tokens == covered
        hit.cache.close()
    assert apc.apc_stats["cow"]["active_leases"] == 0
    apc.clear()


def test_resident_lookup_cannot_restore_disk_before_admission(tmp_path, monkeypatch):
    apc = APCv2(max_size=4, layout_name="test-kv-v1", idle_disk_dir=str(tmp_path),
                idle_disk_seconds=1)
    key, tokens = APCKey("disk-admission"), [1, 2, 3]
    apc.store(key, tokens, [_state(KVCache(), 3)])
    entry = apc._trie.get(key, tokens)
    with apc._apc_lock:
        assert apc._spill_entry_locked(key, tokens, entry, reason="pressure")
    def unexpected_restore(*args):
        raise AssertionError("disk restoration happened before admission")
    monkeypatch.setattr(apc, "_restore_entry_locked", unexpected_restore)
    hit = apc.lookup(key, tokens + [4], allow_disk_restore=False)
    assert not hit.hit and hit.cache is None
    assert hit.miss_reason == "disk_restore_requires_admission"
    apc.clear()


def test_budget_blocked_disk_restore_keeps_the_snapshot(tmp_path):
    """Leased residents can pin the byte cap so a healthy disk snapshot has no
    room; that is a transient miss, not corruption, and must not delete it."""
    import mlx.core as mx

    from mlx2.runtime.apc_v2 import APCKey, APCv2
    from mlx2.runtime.models.cache import KVCache

    def state(length, seed=0):
        cache = KVCache()
        values = mx.arange(seed, seed + length, dtype=mx.float32).reshape(1, 1, length, 1)
        cache.update_and_fetch(values, values)
        mx.eval(cache.state)
        return cache

    clock = [0.0]
    key = APCKey("m")
    a = [state(512)]
    a_bytes = sum(c.nbytes for c in a)
    apc = APCv2(
        max_size=8, max_bytes=int(a_bytes * 1.5), layout_name="l",
        idle_disk_seconds=1.0, idle_disk_dir=str(tmp_path), now_fn=lambda: clock[0],
    )
    apc.store(key, list(range(512)), a)
    clock[0] = 10.0
    assert apc.spill_idle_entries() == 1
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files
    apc.store(key, list(range(1000, 1512)), [state(512, seed=5000)])
    lease = apc.lookup(key, list(range(1000, 1513)))
    assert lease.hit and lease.cache is not None

    blocked = apc.lookup(key, list(range(513)))
    assert not blocked.hit
    assert blocked.miss_reason == "disk_restore_budget_unavailable"
    disk = apc.apc_stats["idle_disk"]
    assert disk["restore_failures"] == 0
    assert disk["restore_budget_deferrals"] == 1
    assert disk["disk_entries"] == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == files

    lease.cache.close()
    restored = apc.lookup(key, list(range(513)))
    assert restored.hit and restored.cached_tokens == 512
    assert apc.apc_stats["idle_disk"]["restores"] == 1
    restored.cache.close()


def test_pressure_eviction_is_least_recently_accessed_not_fifo():
    import mlx.core as mx

    from mlx2.runtime.apc_v2 import APCKey, APCv2
    from mlx2.runtime.models.cache import KVCache

    def state(length, seed=0):
        cache = KVCache()
        values = mx.arange(seed, seed + length, dtype=mx.float32).reshape(1, 1, length, 1)
        cache.update_and_fetch(values, values)
        mx.eval(cache.state)
        return cache

    clock = [0.0]
    key = APCKey("m")
    apc = APCv2(max_size=8, layout_name="l", now_fn=lambda: clock[0])
    hot = list(range(100, 164))
    cold = list(range(200, 264))
    apc.store(key, hot, [state(64)])
    clock[0] += 1
    apc.store(key, cold, [state(64, seed=1)])
    # Reuse ``hot`` through the ordinary prefix path (no draft sidecar).
    for _ in range(3):
        clock[0] += 1
        hit = apc.lookup(key, hot + [999])
        assert hit.hit and hit.hit_kind == "prefix"
        hit.cache.close()
    assert apc.evict_oldest_unleased()
    survivor = apc.lookup(key, hot + [999])
    assert survivor.hit, "the reused checkpoint must outlive the never-reused one"
    survivor.cache.close()
    assert not apc.lookup(key, cold + [999]).hit


def test_startup_sweeps_orphaned_temporary_spill_files(tmp_path):
    from mlx2.runtime.apc_v2 import APCv2

    stale = tmp_path / "apc-idle-deadbeef.target.safetensors"
    orphan = tmp_path / ".apc-idle-deadbeef.target.safetensors.0123abcd.tmp.safetensors"
    unrelated = tmp_path / "keep.txt"
    for path in (stale, orphan, unrelated):
        path.write_bytes(b"x")
    APCv2(max_size=2, layout_name="l", idle_disk_seconds=1, idle_disk_dir=str(tmp_path))
    assert not stale.exists() and not orphan.exists()
    assert unrelated.exists()


def test_disk_restore_rejects_non_empty_placeholder_shapes(tmp_path):
    """The zero-size placeholder table must not become an allocation bomb."""
    import json

    import mlx.core as mx

    from mlx2.runtime.apc_v2 import APCKey, APCv2
    from mlx2.runtime.models.cache import KVCache, load_prompt_cache

    def state(length):
        cache = KVCache()
        values = mx.arange(0, length, dtype=mx.float32).reshape(1, 1, length, 1)
        cache.update_and_fetch(values, values)
        mx.eval(cache.state)
        return cache

    clock = [0.0]
    apc = APCv2(max_size=4, layout_name="l", idle_disk_seconds=1.0,
                idle_disk_dir=str(tmp_path), now_fn=lambda: clock[0])
    key = APCKey("m")
    apc.store(key, list(range(8)), [state(8)])
    clock[0] = 10.0
    assert apc.spill_idle_entries() == 1
    target = next(tmp_path.glob("apc-idle-*.target.safetensors"))
    arrays, metadata = mx.load(str(target), return_metadata=True)
    arrays = {name: mx.array(value) for name, value in arrays.items()}
    mx.eval(*arrays.values())
    # Forge a placeholder table claiming a huge non-empty array.
    metadata = {**metadata, "3": json.dumps({"bogus": ["float32", [1 << 30, 1 << 10]]})}
    forged = target.with_name("forged.safetensors")
    mx.save_safetensors(str(forged), arrays, metadata)
    forged.replace(target)
    with pytest.raises((ValueError, KeyError, TypeError)):
        load_prompt_cache(str(target))
    # The server path turns that into a miss, never an allocation.
    miss = apc.lookup(key, list(range(9)))
    assert not miss.hit
    assert apc.apc_stats["idle_disk"]["restore_failures"] == 1

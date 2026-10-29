"""Tiled bank bytes, bounded replacement, and late-owner contracts; CPU only."""

from concurrent.futures import ThreadPoolExecutor
import threading
import time

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters import deepseek_v41_iq1_arena as arena
from mlx2.adapters import deepseek_v41_mlx_weights as native
from mlx2.runtime.gguf.v41_iq1_tiled import unpack
from mlx2.runtime.io_ownership import ReadTimeout
from test_deepseek_v41_iq1_weights_cpu import artifact, write_compact_artifact
from test_deepseek_v41_mlx_weights_cpu import matrix

mx.set_default_device(mx.cpu)


def store_for(art, layers=2, **kwargs):
    payload = 3*256*50
    cache = layers*(3*payload+3*4)
    return arena.TiledExpertStore(art, cache, **kwargs)


def check_bank(bank, packed, layer):
    assert bank.capacity == len(bank.experts)
    slots = np.array(bank.slot_map)
    for position, expert in enumerate(bank.experts):
        assert slots[expert] == position
        for projection, name in enumerate(arena.FIELDS):
            # K=256 has eight groups of32 for qs and qh.
            lengths = (256*8*4, 256*8*2, 256*2)
            planes = tuple(np.array(bank.planes[3*projection+z])[position*n:(position+1)*n]
                           for z, n in enumerate(lengths))
            assert unpack(planes, (256, 256)).tobytes() == packed[layer][expert][name].tobytes()
    count = len(packed[layer])
    for expert in set(range(count))-set(bank.experts):
        assert slots[expert] == arena.MISSING_SLOT
    assert bank.nbytes == bank.capacity*3*256*50+4*count


def test_plan_counts_cache_map_replacement_inputs_and_all_readers(artifact):
    art, _ = artifact
    bound = 3*3*256*50+12
    plan = arena.tiled_weight_plan(art, bound, read_workers=3)
    assert plan["expert_bank_bound_bytes"] == bound
    assert plan["expert_active_bound_bytes"] == 4*bound+24+16
    assert plan["expert_staging_bound_bytes"] == 3*(2*max(
        art["manifest"]["files"][p]["size"] for paths in arena._inventory(art)[1].values() for p in paths)+native.CHUNK_BYTES)
    with pytest.raises(ValueError, match="largest layer"):
        arena.tiled_weight_plan(art, bound-1)
    for workers in (0, 33, True):
        with pytest.raises(ValueError, match="readers"):
            arena.tiled_weight_plan(art, bound, read_workers=workers)


def test_public_plan_and_direct_receipt_identify_physical_bank_capacity(artifact):
    art, _ = artifact
    store = store_for(art, layers=1)
    try:
        # Serving /status reads plan.as_dict directly, not store.receipt().
        plan = store.plan.as_dict()
        assert plan == store.receipt()["plan"]
        assert plan["allocation"] == "immutable_layer_bank_lru"
        assert plan["capacity_scope"] == "physical_layer_banks"
        assert plan["capacity_includes_slot_maps"] is True
        assert plan["max_epoch_layers"] == 1
        assert plan["resident_ceiling_bytes"] == store.cache_budget_bytes
        assert plan["reserved_bytes"] == store.reserved_bytes()
        assert "capacity_experts_global" not in plan
        assert "per_layer_equivalent_capacity" not in plan
        assert plan["theoretical_global_payload_equivalent_experts"] == 3
        assert plan["theoretical_per_layer_payload_equivalent_experts"] == 1
        assert plan["capacity_experts"] == 1
        assert plan["capacity_experts_semantics"] == "theoretical_uniform_payload_equivalent_not_per_layer_limit"
        assert plan["per_layer_bank_expert_limit"] == 3
    finally:
        store.close()


def test_empty_bank_and_exact_rebuild_without_raw_cache(artifact):
    art, packed = artifact
    store = store_for(art)
    try:
        with store.window():
            empty = store.bank("blk.0")
            assert empty.capacity == 0 and all(p.size == 0 for p in empty.planes)
            assert (np.array(empty.slot_map) == arena.MISSING_SLOT).all()
            bank = store.ensure("blk.0", np.array([2, 0, 2], np.uint32))
            assert bank.experts == (0, 2)
            check_bank(bank, packed, 0)
            assert not store._pending_uploads and not store._pending_host and not store._read_futures
            assert store.live_bytes() == bank.nbytes+empty.nbytes
        assert store.stats["uploaded_experts"] == 2
        assert store.stats["bank_rebuilds"] == 1
        assert store.receipt()["bank_concat"] == "bank_rebuild_only"
    finally:
        store.close()
    assert store.live_bytes() == 0 and store.pool.retirement_receipt()["ownership_drained"]
    assert store.retirement_receipt()["ownership_drained"]


def test_warm_bank_never_reads_routes_sources_or_concatenates(artifact, monkeypatch):
    art, _ = artifact
    store = store_for(art)
    try:
        with store.window():
            expected = store.ensure("blk.0", np.array([1], np.uint32))
        def forbidden(*args, **kwargs):
            raise AssertionError("warm bank touched cold path")
        monkeypatch.setattr(arena.np, "asarray", forbidden)
        monkeypatch.setattr(arena, "_verify_file", forbidden)
        monkeypatch.setattr(arena, "_read_tiled", forbidden)
        monkeypatch.setattr(mx, "concatenate", forbidden)
        before = store.stats.copy()
        with store.window():
            assert store.bank("blk.0") is expected
            assert store.bank("blk.0") is expected
        assert store.stats["bytes_read"] == before["bytes_read"]
        assert store.stats["bank_rebuilds"] == before["bank_rebuilds"]
    finally:
        store.close()


def test_replacement_keeps_old_bank_exact_until_epoch_fence(artifact):
    art, packed = artifact
    store = store_for(art)
    try:
        with store.window():
            old = store.ensure("blk.0", np.array([2], np.uint32))
        store.begin_epoch()
        assert store.bank("blk.0") is old
        new = store.ensure("blk.0", np.array([1], np.uint32))
        assert new.experts == (2, 1)
        check_bank(old, packed, 0)
        check_bank(new, packed, 0)
        assert store.live_bytes() == old.nbytes+new.nbytes
        assert store.live_bytes() <= store.reserved_bytes()
        store.end_epoch()
        assert store.live_bytes() == new.nbytes
    finally:
        store.close()


def test_coarse_bank_lru_uses_physical_allocations(artifact):
    art, packed = artifact
    store = store_for(art, layers=1)
    try:
        for layer in range(2):
            with store.window():
                bank = store.ensure(f"blk.{layer}", np.arange(3, dtype=np.uint32))
                check_bank(bank, packed, layer)
        assert tuple(store._cache) == ("blk.1",)
        assert store._cache_bytes == store.cache_budget_bytes
        assert store.stats["evictions"] == 1
        assert store.live_bytes() == store._cache_bytes
    finally:
        store.close()


def test_parallel_reads_are_bounded_and_all_results_join(artifact, monkeypatch):
    art, _ = artifact
    original = arena._read_tiled
    barrier, lock = threading.Barrier(2), threading.Lock()
    counts = {"active": 0, "peak": 0}
    def blocked(*args):
        with lock:
            counts["active"] += 1
            counts["peak"] = max(counts["peak"], counts["active"])
        try:
            barrier.wait(timeout=2)
            return original(*args)
        finally:
            with lock:
                counts["active"] -= 1
    monkeypatch.setattr(arena, "_read_tiled", blocked)
    store = store_for(art, read_workers=2)
    try:
        with store.window():
            store.ensure("blk.0", np.array([0, 1], np.uint32))
        assert counts == {"active": 0, "peak": 2}
        assert store.stats["upload_batches"] == 1
        assert not store._read_futures
    finally:
        store.close()


def test_invalid_ids_and_cross_layer_epoch_are_refused(artifact):
    art, _ = artifact
    store = store_for(art)
    try:
        store.begin_epoch()
        for ids in (np.array([-1]), np.array([3]), np.array([1.0]), np.array([], np.uint32), np.zeros(6145, np.uint32)):
            with pytest.raises(ValueError, match="host expert IDs"):
                store.ensure("blk.0", ids)
        store.bank("blk.0")
        with pytest.raises(ValueError, match="only one layer"):
            store.bank("blk.1")
        store.ensure("blk.0", np.array([0], np.uint32))
        with pytest.raises(ValueError, match="one bank replacement"):
            store.ensure("blk.0", np.array([1], np.uint32))
        store.end_epoch()
        with ThreadPoolExecutor(1) as pool:
            with pytest.raises(ValueError, match="another evaluation thread"):
                pool.submit(store.begin_epoch).result()
    finally:
        store.close()


def test_source_replacement_poison_preserves_previous_bank(artifact):
    art, packed = artifact
    store = store_for(art)
    try:
        with store.window():
            old = store.ensure("blk.0", np.array([0], np.uint32))
        path = art["root"]/store.tables["blk.0"][0]
        path.write_bytes(path.read_bytes())  # Same bytes, changed admitted seal.
        with pytest.raises(ValueError):
            with store.window():
                store.ensure("blk.0", np.array([1], np.uint32))
        assert store._cache["blk.0"] is old
        check_bank(old, packed, 0)
        with pytest.raises(ValueError, match="failed"):
            store.begin_epoch()
    finally:
        store.close()


def test_failed_upload_and_fence_keep_all_owners_until_retry(artifact, monkeypatch):
    art, _ = artifact
    store = store_for(art)
    original = mx.eval
    def fail(*args):
        raise RuntimeError("injected upload failure")
    monkeypatch.setattr(mx, "eval", fail)
    store._synchronize = lambda: (_ for _ in ()).throw(RuntimeError("injected fence failure"))
    try:
        with pytest.raises(RuntimeError, match="fence"):
            with store.window():
                store.ensure("blk.0", np.array([0, 1], np.uint32))
        assert store._active and store._pending_uploads and store._pending_host
        assert store.live_bytes() > 0
        with pytest.raises(RuntimeError, match="retained|retain"):
            store_for(art)
        with pytest.raises(RuntimeError, match="fence"):
            store.close()
    finally:
        monkeypatch.setattr(mx, "eval", original)
        store._synchronize = lambda: None
        store.close()
    assert store.live_bytes() == 0


def test_read_timeout_retains_futures_and_refuses_park_until_drain(artifact, monkeypatch):
    art, _ = artifact
    release, entered = threading.Event(), threading.Event()
    original = arena._read_tiled
    def late(*args):
        entered.set()
        assert release.wait(2)
        return original(*args)
    monkeypatch.setattr(arena, "_read_tiled", late)
    store = store_for(art, timeout=.02, read_workers=1)
    try:
        with pytest.raises(ReadTimeout):
            with store.window():
                store.ensure("blk.0", np.array([0], np.uint32))
        assert entered.is_set() and store._read_futures
        with pytest.raises(ReadTimeout):
            store.park()
        with pytest.raises(ReadTimeout, match="late owned read"):
            store.close()
        assert store.reserved_bytes() > 0
    finally:
        release.set()
        for future in tuple(store._read_futures):
            try:
                future.result(timeout=2)
            except ReadTimeout:
                pass
        deadline = time.monotonic()+2
        while store.pool.counters()["weight_read_inflight"] and time.monotonic() < deadline:
            time.sleep(.001)
        store.close()
    assert not store._read_futures and store.pool.retirement_receipt()["ownership_drained"]


def test_repeated_park_activate_and_shrink_are_accounted(artifact):
    art, _ = artifact
    store = store_for(art)
    try:
        for _ in range(2):
            for layer in range(2):
                with store.window():
                    store.ensure(f"blk.{layer}", np.arange(3, dtype=np.uint32))
            assert store.live_bytes() <= store.reserved_bytes()
            store.resize_cache_budget(store.minimum_cache_bytes)
            assert len(store._cache) == 1
            store.park()
            assert store.live_bytes() == store.reserved_bytes() == 0
            with pytest.raises(ValueError, match="parked"):
                store.begin_epoch()
            store.activate()
        store.park()
        path = art["root"]/store.tables["blk.1"][2]
        path.write_bytes(path.read_bytes())
        with pytest.raises(ValueError):
            store.activate()
        assert store._parked
    finally:
        store.close()


def test_deferred_epoch_holds_replaced_bank_without_global_fence(artifact):
    art, _ = artifact
    store = store_for(art)
    entered, release = threading.Event(), threading.Event()
    marker = object()
    def wait(value):
        assert value is marker
        entered.set()
        assert release.wait(2)
    try:
        with store.window():
            old = store.ensure("blk.0", np.array([0], np.uint32))
        def forbidden():
            raise AssertionError("deferred epoch performed a global fence")
        store._synchronize = forbidden
        with store.window():
            store.bank("blk.0")
            new = store.ensure("blk.0", np.array([1], np.uint32))
            store.defer_epoch(marker, waiter=wait)
        assert entered.wait(1)
        assert not store._active and len(store._epochs) == 1
        assert store.live_bytes() == old.nbytes+new.nbytes
        assert store.stats["deferred_epochs"] == 1
        release.set()
        store.drain_epochs()
        assert store.live_bytes() == new.nbytes
        assert store.stats["completed_epochs"] == 1
        assert store.stats["completion_marker_ns"] > 0
    finally:
        release.set()
        store._synchronize = lambda: None
        store.close()


def test_two_pending_epochs_bound_worker_and_capacity_eviction(artifact):
    art, _ = artifact
    store = store_for(art, layers=1)
    release, entered = threading.Event(), threading.Event()
    def wait(_):
        entered.set()
        assert release.wait(2)
    timer = None
    try:
        with store.window():
            first = store.ensure("blk.0", np.arange(3, dtype=np.uint32))
            store.defer_epoch(object(), waiter=wait)
        assert entered.wait(1)
        # A second same-bank epoch is legal and shares, rather than doubles,
        # the physical bank allocation in accounting.
        with store.window():
            assert store.bank("blk.0") is first
            store.defer_epoch(object(), waiter=wait)
        assert len(store._epochs) == 2 and store.live_bytes() == first.nbytes
        timer = threading.Timer(.025, release.set)
        timer.start()
        # The only cache bank is pinned. A third layer cannot evict it until
        # its exact markers return, then the new allocation fits the budget.
        with store.window():
            store.ensure("blk.1", np.arange(3, dtype=np.uint32))
            store.defer_epoch(object(), waiter=wait)
        store.drain_epochs()
        assert store.stats["max_pending_epochs"] == 2
        assert store.stats["completion_capacity_drains"] >= 1
        assert store.stats["evictions"] == 1
        assert tuple(store._cache) == ("blk.1",)
        assert store.live_bytes() <= store.reserved_bytes()
    finally:
        release.set()
        if timer:
            timer.join()
        store.close()


def test_completion_failure_quarantines_until_successful_close_fence(artifact):
    art, _ = artifact
    store = store_for(art)
    def fail(_):
        raise RuntimeError("injected completion failure")
    try:
        with store.window():
            bank = store.ensure("blk.0", np.array([0], np.uint32))
            store.defer_epoch(object(), waiter=fail)
        with pytest.raises(RuntimeError, match="completion failure"):
            store.drain_epochs()
        assert store._epochs and store.live_bytes() == bank.nbytes
        store._synchronize = lambda: (_ for _ in ()).throw(RuntimeError("still unfenced"))
        with pytest.raises(RuntimeError, match="unfenced"):
            store.close()
        assert store._epochs and store.live_bytes() > 0
    finally:
        store._synchronize = lambda: None
        store.close()
    assert not store._epochs and store.live_bytes() == 0


def test_completion_timeout_never_releases_running_marker_owner(artifact):
    art, _ = artifact
    store = store_for(art, timeout=.02)
    release = threading.Event()
    def wait(_):
        assert release.wait(2)
    try:
        with store.window():
            bank = store.ensure("blk.0", np.array([0], np.uint32))
            store.defer_epoch(object(), waiter=wait)
        with pytest.raises(ReadTimeout, match="marker"):
            store.drain_epochs()
        with pytest.raises(ReadTimeout, match="marker"):
            store.close()
        assert store.live_bytes() == bank.nbytes and store.reserved_bytes() > 0
        assert len(store._epochs) == 1 and store._completion_worker.thread.is_alive()
        assert not store.retirement_receipt()["ownership_drained"]
    finally:
        release.set()
        store.close()
    assert store.live_bytes() == 0


def test_real_cpu_marker_retires_only_its_dependency_graph(artifact):
    art, _ = artifact
    store = store_for(art)
    try:
        with store.window():
            bank = store.ensure("blk.0", np.array([0], np.uint32))
            marker = sum(mx.sum(p.astype(mx.float32)) for p in bank.planes)
            mx.async_eval(marker)
            store.defer_epoch(marker)
        store.drain_epochs()
        assert store.stats["deferred_epochs"] == store.stats["completed_epochs"] == 1
        assert not store._active and not store._epochs
    finally:
        store.close()


def test_marker_shape_and_supplied_admission_cannot_expand_budget(artifact):
    art, _ = artifact
    store = store_for(art)
    try:
        store.begin_epoch()
        with pytest.raises(ValueError, match="scalar"):
            store.defer_epoch(mx.zeros((2,)))
        store.end_epoch()
        altered = dict(store.bounds)
        altered["expert_staging_bound_bytes"] -= 1
        with pytest.raises(ValueError, match="admitted inventory"):
            arena.TiledExpertStore(art, store.cache_budget_bytes, _bounds=altered)
    finally:
        store.close()


@pytest.mark.parametrize("requested,effective", [(1, 1), (8, 8), (9, 8), (32, 8)])
def test_public_reader_request_normalizes_plan_store_pool_and_receipt(artifact, requested, effective):
    art, _ = artifact
    bound = 3*3*256*50+12
    plan = arena.tiled_weight_plan(art, bound, read_workers=requested)
    baseline = arena.tiled_weight_plan(art, bound, read_workers=effective)
    assert plan["expert_requested_read_workers"] == requested
    assert plan["expert_read_workers"] == effective
    assert plan["total_reservation_bytes"] == baseline["total_reservation_bytes"]
    assert plan["expert_staging_bound_bytes"] == baseline["expert_staging_bound_bytes"]
    store = arena.TiledExpertStore(art, bound, read_workers=requested, _bounds=plan)
    try:
        assert store.read_workers == store.pool._max_workers == store.pool._max_pending == effective
        assert store.plan.staging_window == effective
        assert store.receipt()["requested_read_workers"] == requested
        assert store.receipt()["effective_read_workers"] == effective
        with store.window():
            bank = store.ensure("blk.0", np.arange(3, dtype=np.uint32))
        assert bank.capacity == 3
        assert store.stats["upload_batches"] == (3+effective-1)//effective
    finally:
        store.close()


@pytest.mark.parametrize("requested", [2, 8, 32])
def test_host_batch_upload_preserves_order_across_remainder_and_replacement(artifact, monkeypatch, requested):
    base, source = artifact
    packed = {0: [{kind: source[0][e % 3][kind].copy() for kind in arena.FIELDS} for e in range(11)]}
    for e, expert in enumerate(packed[0]):
        for data in expert.values():
            data[2] ^= e  # Each stored expert has distinguishable source bytes.
    rng = np.random.default_rng(51)
    dense = {"token_embd.weight": matrix(7, 256, 8, rng), "output_norm.weight": np.ones(256, np.float32)}
    art = write_compact_artifact(base["root"].parent/"larger", dense, packed, base["args"])
    store = arena.TiledExpertStore(art, 11*3*256*50+44, read_workers=requested)
    original, uploads = mx.array, []
    def observed(value, *args, **kwargs):
        if isinstance(value, np.ndarray) and value.dtype == np.uint8:
            uploads.append(value.nbytes)
        return original(value, *args, **kwargs)
    monkeypatch.setattr(mx, "array", observed)
    try:
        requested_ids = np.array([10, 0, 9, 8, 7, 5, 6, 3, 1], np.uint32)
        with store.window():
            old = store.ensure("blk.0", requested_ids)
            check_bank(old, packed, 0)
        first_batches = (9+store.read_workers-1)//store.read_workers
        assert len(uploads) == 9*first_batches
        assert sum(uploads) == 9*3*256*50
        assert store.stats["uploaded_arrays"] == len(uploads)
        assert store.host_owned_bytes() == 0
        with store.window():
            new = store.ensure("blk.0", np.array([4, 2], np.uint32))
            assert new.experts == old.experts+(2, 4)
            check_bank(old, packed, 0)
            check_bank(new, packed, 0)
        assert len(uploads) == 9*(first_batches+1)
        assert not store._pending_uploads and not store._pending_host
        assert store.stats["host_batch_concat_bytes"] > 0
        assert store.stats["bank_concat_bytes"] > 0
        assert store.live_bytes() == new.nbytes
    finally:
        store.close()


def test_combined_host_owner_survives_partial_upload_and_failed_fence(artifact, monkeypatch):
    art, _ = artifact
    store = store_for(art, read_workers=2)
    original, calls = mx.array, []
    def interrupted(value, *args, **kwargs):
        if isinstance(value, np.ndarray) and value.dtype == np.uint8:
            calls.append(value.nbytes)
            if len(calls) == 4:
                raise RuntimeError("injected combined upload failure")
        return original(value, *args, **kwargs)
    monkeypatch.setattr(mx, "array", interrupted)
    store._synchronize = lambda: (_ for _ in ()).throw(RuntimeError("still pending GPU work"))
    try:
        with pytest.raises(RuntimeError, match="pending GPU"):
            with store.window():
                store.ensure("blk.0", np.array([2, 0], np.uint32))
        assert len(store._pending_uploads) == 3
        assert tuple(store._pending_host) == (("batch", 0),)
        assert len(store._pending_host[("batch", 0)]) == 9
        assert store.host_owned_bytes() == 2*3*256*50
        assert store.host_owned_bytes() <= store.bounds["expert_staging_bound_bytes"]
        with pytest.raises(RuntimeError, match="pending GPU"):
            store.close()
        assert store._pending_host and not store.retirement_receipt()["ownership_drained"]
    finally:
        monkeypatch.setattr(mx, "array", original)
        store._synchronize = lambda: None
        store.close()
    assert store.host_owned_bytes() == store.live_bytes() == 0

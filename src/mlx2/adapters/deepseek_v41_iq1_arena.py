# SPDX-License-Identifier: MIT
"""Immutable tiled IQ1_S layer banks with bounded, verified cold reads.

The source remains the standalone compact artifact. A bank owns nine tiled U8
planes and one GPU route table, never an additional raw expert GPU cache.
Only misses rebuild a bank. Per-layer epochs retain replaced banks until a
successful completion fence. See deepseek-v41-iq1-tiled-arena provenance.
"""

from collections import OrderedDict
from concurrent.futures import Future, TimeoutError as FutureTimeout
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import queue
import threading
import time

import numpy as np

from .deepseek_v41_iq1_weights import FIELDS, _inventory, compact_weight_plan
from .deepseek_v41_mlx_weights import (
    CHUNK_BYTES, NativeExpertPlan, NativeExpertStore, _payload_bytes,
    _quarantine_upload, _refuse_quarantined_uploads, _release_upload,
    _verify_file, read_native_file,
)
from ..runtime.io_ownership import ReadPool, ReadTimeout, storage_owner

REVISION = "mlx2.v41-iq1-tiled-immutable-bank-marker.v2"
MISSING_SLOT = np.uint32(0xFFFFFFFF)
MAX_PENDING_EPOCHS = 2


class TiledExpertPlan(NativeExpertPlan):
    """Describe physical bank capacity identically in serving and direct receipts."""

    def as_dict(self):
        result = super().as_dict()
        result.pop("capacity_experts_global")
        result.pop("per_layer_equivalent_capacity")
        result.update(
            allocation="immutable_layer_bank_lru", capacity_scope="physical_layer_banks",
            capacity_includes_slot_maps=True, max_epoch_layers=1,
            capacity_experts_semantics="theoretical_uniform_payload_equivalent_not_per_layer_limit",
            theoretical_global_payload_equivalent_experts=self.ceiling_bytes//max(1, self.expert_bytes),
            theoretical_per_layer_payload_equivalent_experts=self.capacity_experts,
            per_layer_bank_expert_limit=self.experts_per_layer,
        )
        return result


@dataclass(frozen=True, eq=False)
class TiledBank:
    prefix: str
    experts: tuple
    planes: tuple
    slot_map: object

    @property
    def capacity(self):
        return len(self.experts)

    @property
    def nbytes(self):
        return self.slot_map.nbytes + sum(p.nbytes for p in self.planes)


def tiled_weight_plan(artifact, cache_bytes, *, read_workers=4):
    """Admission counts physical banks, one retired bank, upload inputs and I/O.

The cache must hold a whole largest layer. No tensor changes precision. The
    extra layer bounds cover two deferred old banks, one active old bank and
    pending per-expert inputs while concatenation writes its cache allocation.
    Cached banks pinned by deferred epochs cannot be evicted until they drain.
    Host workers each own at most a source plus its tiled copy and read scratch.
    Once those reads finish, one equally bounded combined host batch replaces
    their tiled copies before the next reads start; no additional budget grows.
"""
    if type(read_workers) is not int or not 1 <= read_workers <= 32:
        raise ValueError("tiled expert readers must be between one and thirty-two")
    requested_read_workers = read_workers
    read_workers = min(8, read_workers)
    result = compact_weight_plan(artifact, cache_bytes)
    _, tables, _ = _inventory(artifact)
    files = artifact["manifest"]["files"]
    maximum = max(sum(_payload_bytes(files[p]) for p in paths) + 4*len(paths)
                  for paths in tables.values())
    if cache_bytes < maximum:
        raise ValueError("tiled expert cache must hold one complete largest layer bank")
    source = max(files[p]["size"] for paths in tables.values() for p in paths)
    # Empty maps stay available for every layer, even if its bank is evicted.
    maps = sum(4*len(paths) for paths in tables.values())
    staging = read_workers*(2*source + CHUNK_BYTES)
    result.update(minimum_expert_cache_bytes=maximum,
                  expert_active_bound_bytes=(MAX_PENDING_EPOCHS+2)*maximum + maps + 8*MAX_PENDING_EPOCHS,
                  expert_staging_bound_bytes=staging,
                  expert_bank_bound_bytes=maximum,
                  expert_empty_map_bound_bytes=maps,
                  expert_completion_bound_bytes=8*MAX_PENDING_EPOCHS,
                  expert_requested_read_workers=requested_read_workers,
                  expert_read_workers=read_workers)
    result["total_reservation_bytes"] = sum(result[k] for k in (
        "dense_payload_bytes", "dense_prepared_bound_bytes", "dense_staging_bound_bytes",
        "expert_cache_bytes", "expert_active_bound_bytes", "expert_staging_bound_bytes"))
    return result


class _NoPersistentDescriptors:
    def close(self):
        # read_native_file opens and closes each descriptor in the worker.
        pass


@dataclass(eq=False)
class _Epoch:
    marker: object
    banks: tuple
    future: Future
    submitted_ns: int
    ended_ns: int = 0


class _CompletionWorker:
    """One daemon GPU-completion waiter, never an I/O pool or graph producer.

The main evaluation thread must have scheduled marker and all of its graph
already. The callback waits for that exact marker only. A timeout does not
cancel GPU work, release its banks, or permit a replacement store.
"""
    def __init__(self, owner):
        self.owner = owner
        self.queue = queue.SimpleQueue()
        self.thread = threading.Thread(target=self._run, daemon=True, name="v41-tiled-completion")
        self.thread.start()

    def _run(self):
        while True:
            item = self.queue.get()
            if item is None:
                return
            epoch, waiter = item
            try:
                waiter(epoch.marker)
            except BaseException as error:
                _quarantine_upload(self.owner)
                epoch.ended_ns = time.perf_counter_ns()
                epoch.future.set_exception(error.with_traceback(None))
            else:
                epoch.ended_ns = time.perf_counter_ns()
                epoch.future.set_result(None)
            del item, epoch, waiter

    def close(self):
        self.queue.put(None)
        self.thread.join(timeout=1.0)
        if self.thread.is_alive():
            raise ReadTimeout("tiled completion worker has not stopped")


def _read_tiled(root, relative, record, shapes):
    """Pure CPU worker: full hash/seal verification before exact byte transpose."""
    from ..runtime.gguf.v41_iq1_tiled import pack
    source = read_native_file(root, relative, record)
    planes = tuple(p for kind in FIELDS for p in pack(source[kind+".data"], shapes[kind]))
    _verify_file(root, relative, record)
    if sum(p.nbytes for p in planes) != _payload_bytes(record):
        raise ValueError("tiled packing changed the expert payload size")
    return planes


class TiledExpertStore(NativeExpertStore):
    """Coarse layer-bank LRU; one layer and at most one rebuild per epoch.

bank() never reads routing IDs or source files. Its resident bytes were verified
when built and are immutable. ensure() is the explicit host miss boundary; it
checks every source represented by a rebuilt bank to avoid mixing generations.
The integration must discard provisional missing-route outputs before retrying.
"""

    def __init__(self, artifact, cache_bytes, *, read_workers=4, timeout=60.0,
                 _owned_manifest=None, _bounds=None):
        _refuse_quarantined_uploads()
        if _owned_manifest is not None and _owned_manifest is not artifact["manifest"]:
            raise ValueError("tiled manifest owner differs from weights")
        self.root = Path(artifact["root"]).resolve()
        self.manifest = deepcopy(artifact["manifest"]) if _owned_manifest is None else _owned_manifest
        self.files = self.manifest["files"]
        _, self.tables, self.row_tensors = _inventory({"manifest": self.manifest})
        expected = tiled_weight_plan({"manifest": self.manifest}, cache_bytes, read_workers=read_workers)
        if _bounds is not None and any(_bounds.get(k) != v for k, v in expected.items()):
            raise ValueError("tiled bank bounds differ from admitted inventory")
        self.bounds = expected
        self.cache_budget_bytes = self.maximum_cache_bytes = cache_bytes
        self.minimum_cache_bytes = expected["minimum_expert_cache_bytes"]
        self.binding = expected["manifest_binding_sha256"]
        self._cache, self._leased, self._empty = OrderedDict(), {}, {}
        self._pending_uploads, self._pending_host, self._read_futures = {}, {}, set()
        self._pending_bank = None
        self._cache_bytes = 0
        self._active = self._closed = self._parked = False
        self._owner = self._error = self._synchronize = None
        self._epoch_prefix = None
        self._epoch_rebuilt = False
        self._epochs = []
        self._completion_worker = None
        self._lock = threading.RLock()
        self.ple_tables, self._ple_bound = {}, False
        self.requested_read_workers = read_workers
        self.read_workers, self.timeout = expected["expert_read_workers"], timeout
        self._handles = _NoPersistentDescriptors()
        self.pool = ReadPool(self.read_workers, "v41-tiled-experts", timeout=timeout,
                             max_pending=self.read_workers, owner=storage_owner("v41-tiled-experts", self.root))
        self.stats = dict(cache_hits=0, cache_misses=0, bytes_read=0, evictions=0,
                          uploaded_experts=0, upload_batches=0, demand_wait_ns=0,
                          lookups=0, windows=0, parks=0, activations=0,
                          bank_rebuilds=0, bank_concat_bytes=0, source_checks=0,
                          uploaded_arrays=0, host_batch_concat_bytes=0,
                          deferred_epochs=0, completed_epochs=0, max_pending_epochs=0,
                          completion_wait_ns=0, completion_marker_ns=0,
                          completion_capacity_drains=0, synchronous_fences=0)
        count = max(map(len, self.tables.values()))
        largest = max(_payload_bytes(self.files[p]) for paths in self.tables.values() for p in paths)
        self.plan = TiledExpertPlan(layers=len(self.tables), experts_per_layer=count,
            expert_bytes=largest, table_bytes=expected["expert_table_bytes"],
            capacity_experts=min(count, cache_bytes//(len(self.tables)*largest)),
            ceiling_bytes=cache_bytes, projections=FIELDS, load_phase="pre_materialization",
            top_k=getattr(artifact.get("args"), "num_experts_per_tok", 6),
            transient_bound_bytes=expected["expert_active_bound_bytes"]+expected["expert_staging_bound_bytes"],
            staging_window=self.read_workers, manifest_sha256=self.binding,
            table_kinds=("tiled_IQ1_S",), page_cache="bypass")

    def _layer(self, prefix):
        self._ready()
        if not self._active or prefix not in self.tables:
            raise ValueError("tiled bank requires an active epoch and admitted layer")
        if self._epoch_prefix not in (None, prefix):
            raise ValueError("tiled epoch may lease only one layer")
        self._epoch_prefix = prefix

    def begin_epoch(self):
        with self._lock:
            self._ready()
            self._collect_completed()
            if self._active:
                raise ValueError("tiled expert epoch already active")
            self._active = True
            self._epoch_prefix = None
            self._epoch_rebuilt = False
            self.stats["windows"] += 1

    def end_epoch(self):
        with self._lock:
            if not self._active:
                raise ValueError("tiled expert epoch is not active")
            try:
                self._fence()
                self.stats["synchronous_fences"] += 1
            except BaseException as error:
                self._error = str(error)
                _quarantine_upload(self)
                raise
            self._leased.clear()
            # A failed rebuild may still have CPU futures. Their exact owners
            # remain until close drains them; never forget an unreturned result.
            if not self._error:
                self._pending_uploads.clear()
                self._pending_host.clear()
                self._pending_bank = None
            self._active = False
            self._epoch_prefix = None

    @contextmanager
    def window(self):
        self.begin_epoch()
        try:
            yield
        except BaseException as error:
            self._error = str(error)
            raise
        finally:
            if self._active:
                self.end_epoch()

    def _finish_epoch(self, epoch, *, wait):
        if not wait and not epoch.future.done():
            return False
        begun = time.perf_counter_ns()
        try:
            epoch.future.result(timeout=self.timeout if wait else 0)
        except BaseException as error:
            failure = ReadTimeout("tiled completion marker exceeded its deadline") if isinstance(error, FutureTimeout) else error
            self._error = str(failure)
            _quarantine_upload(self)
            raise failure
        finally:
            if wait:
                self.stats["completion_wait_ns"] += time.perf_counter_ns()-begun
        self._epochs.remove(epoch)
        self.stats["completed_epochs"] += 1
        self.stats["completion_marker_ns"] += epoch.ended_ns-epoch.submitted_ns
        return True

    def _collect_completed(self):
        # Ordered retirement also bounds state and keeps a failed earlier epoch
        # from being hidden by successful later completions.
        while self._epochs and self._finish_epoch(self._epochs[0], wait=False):
            pass

    def drain_epochs(self):
        """Wait for exact submitted markers; used before park/retire/publication."""
        with self._lock:
            while self._epochs:
                self._finish_epoch(self._epochs[0], wait=True)

    def defer_epoch(self, marker, *, waiter=None):
        """Transfer the active banks to a bounded, already-scheduled GPU marker.

        Caller contract: marker depends on the final layer output and every use
        of these banks, and mx.async_eval(marker, output, state) was called on
        the owning model thread before this method. Missing-route provisional
        outputs cannot be committed. This method schedules no model operations.
        """
        import mlx.core as mx
        with self._lock:
            self._ready()
            if (not self._active or self._pending_uploads or self._pending_host
                    or self._pending_bank is not None or self._read_futures):
                raise ValueError("tiled defer requires a clean active epoch")
            if marker is None or (waiter is not None and not callable(waiter)):
                raise ValueError("tiled defer requires a completion marker and waiter")
            if waiter is None and (not isinstance(marker, mx.array) or marker.size != 1 or marker.nbytes > 8):
                raise ValueError("tiled completion marker must be one scalar of at most eight bytes")
            self._collect_completed()
            while len(self._epochs) >= MAX_PENDING_EPOCHS:
                self.stats["completion_capacity_drains"] += 1
                self._finish_epoch(self._epochs[0], wait=True)
            epoch = _Epoch(marker, tuple(self._leased.values()), Future(), time.perf_counter_ns())
            self._epochs.append(epoch)
            # Publish ownership before starting the waiter. If thread creation
            # fails, keep both the active and epoch references for close.
            try:
                if self._completion_worker is None:
                    self._completion_worker = _CompletionWorker(self)
                self._completion_worker.queue.put((epoch, mx.eval if waiter is None else waiter))
            except BaseException as error:
                self._error = str(error)
                epoch.future.set_exception(error.with_traceback(None))
                _quarantine_upload(self)
                raise
            self._leased.clear()
            self._active = False
            self._epoch_prefix = None
            self.stats["deferred_epochs"] += 1
            self.stats["max_pending_epochs"] = max(self.stats["max_pending_epochs"], len(self._epochs))

    def bank(self, prefix):
        import mlx.core as mx
        with self._lock:
            self._layer(prefix)
            bank = self._cache.get(prefix)
            if bank is None:
                bank = self._empty.get(prefix)
                if bank is None:
                    # Publish owners before evaluation, even for the empty map.
                    slot_map = mx.array(np.full(len(self.tables[prefix]), MISSING_SLOT, np.uint32))
                    planes = tuple(mx.array([], mx.uint8) for _ in range(9))
                    bank = self._empty[prefix] = TiledBank(prefix, (), planes, slot_map)
                    mx.eval(slot_map, *planes)
            else:
                self._cache.move_to_end(prefix)
                self.stats["cache_hits"] += 1
            self._leased[id(bank)] = bank
            self.stats["lookups"] += 1
            return bank

    def _evict_for(self, prefix, new_bytes):
        old = self._cache.get(prefix)
        needed = self._cache_bytes - (old.nbytes if old is not None else 0) + new_bytes
        while needed > self.cache_budget_bytes:
            self._collect_completed()
            pinned = {id(b) for e in self._epochs for b in e.banks} | set(self._leased)
            candidate = next((key for key, bank in self._cache.items()
                              if key != prefix and id(bank) not in pinned), None)
            if candidate is None:
                if self._epochs:
                    self.stats["completion_capacity_drains"] += 1
                    self._finish_epoch(self._epochs[0], wait=True)
                    continue
                raise ValueError("tiled bank exceeds admitted cache")
            victim = self._cache.pop(candidate)
            self._cache_bytes -= victim.nbytes
            needed -= victim.nbytes
            self.stats["evictions"] += 1

    def ensure(self, prefix, host_ids):
        """Build a verified immutable replacement, only at the host miss boundary."""
        import mlx.core as mx
        with self._lock:
            self._layer(prefix)
            values = np.asarray(host_ids)
            if (values.dtype.kind not in "iu" or not values.size or values.size > 1024*6
                    or values.min() < 0 or values.max() >= len(self.tables[prefix])):
                raise ValueError("tiled ensure requires bounded valid host expert IDs")
            old = self._cache.get(prefix)
            current = () if old is None else old.experts
            missing = tuple(int(e) for e in np.unique(values) if int(e) not in current)
            if not missing:
                return self.bank(prefix)
            if self._epoch_rebuilt:
                raise ValueError("tiled epoch permits at most one bank replacement")
            self._epoch_rebuilt = True
            experts = current + missing
            size = sum(_payload_bytes(self.files[self.tables[prefix][e]]) for e in experts) + 4*len(self.tables[prefix])
            if size > self.bounds["expert_bank_bound_bytes"]:
                raise ValueError("tiled bank exceeds admitted layer geometry")
            if old is not None:
                self._leased[id(old)] = old
            self._evict_for(prefix, size)
            shapes = {k: (self.row_tensors[prefix][k].shape[1], self.row_tensors[prefix][k].shape[0]) for k in FIELDS}
            try:
                # Validate resident inputs too: a rebuild cannot combine old
                # verified bytes with sources replaced since admission.
                for e in experts:
                    path = self.tables[prefix][e]
                    _verify_file(self.root, path, self.files[path])
                    self.stats["source_checks"] += 1
                uploaded = []
                for start in range(0, len(missing), self.read_workers):
                    batch = []
                    begun = time.perf_counter_ns()
                    try:
                        for e in missing[start:start+self.read_workers]:
                            path = self.tables[prefix][e]
                            future = self.pool.submit(_read_tiled, self.root, path, self.files[path], shapes)
                            self._read_futures.add(future)
                            batch.append((e, path, future))
                        for e, path, future in batch:
                            try:
                                host = future.result(timeout=self.timeout)
                            except FutureTimeout as error:
                                failure = ReadTimeout("tiled expert read exceeded its deadline")
                                self.pool.fail(failure)
                                raise failure from error
                            self._pending_host[e] = host
                            self._read_futures.remove(future)
                            self.stats["bytes_read"] += self.files[path]["size"]
                            self.stats["cache_misses"] += 1
                        # All batch reads have ended. Coalesce each plane on the
                        # CPU before uploading: nine MLX arrays per read batch,
                        # instead of nine per expert. Publish the partial host
                        # owner before concatenation/upload can fail.
                        combined = self._pending_host[("batch", start)] = []
                        for z in range(9):
                            parts = [self._pending_host[e][z] for e, _, _ in batch]
                            plane = parts[0] if len(parts) == 1 else np.concatenate(parts)
                            combined.append(plane)
                            if len(parts) > 1:
                                self.stats["host_batch_concat_bytes"] += plane.nbytes
                        for e, _, _ in batch:
                            self._pending_host.pop(e)
                        batch.clear()
                        del parts, host, future
                        item = []
                        for z, plane in enumerate(combined):
                            value = mx.array(plane)
                            self._pending_uploads[("batch", start, z)] = value
                            item.append(value)
                            self.stats["uploaded_arrays"] += 1
                        uploaded.append(tuple(item))
                        mx.eval(item)
                        self._pending_host.pop(("batch", start))
                        del combined, plane
                        self.stats["upload_batches"] += 1
                    finally:
                        self.stats["demand_wait_ns"] += time.perf_counter_ns()-begun
                planes = []
                for z in range(9):
                    parts = ([old.planes[z]] if old is not None else []) + [item[z] for item in uploaded]
                    # Always materialize a distinct final bank for multiple
                    # inputs; a singleton is already a complete immutable plane.
                    value = parts[0] if len(parts) == 1 else mx.concatenate(parts)
                    if len(parts) > 1:
                        self.stats["bank_concat_bytes"] += value.nbytes
                    self._pending_uploads[("bank", z)] = value
                    planes.append(value)
                slots = np.full(len(self.tables[prefix]), MISSING_SLOT, np.uint32)
                slots[list(experts)] = np.arange(len(experts), dtype=np.uint32)
                slot_map = mx.array(slots)
                self._pending_uploads[("bank", "map")] = slot_map
                candidate = self._pending_bank = TiledBank(prefix, experts, tuple(planes), slot_map)
                mx.eval(slot_map, *planes)
                for e in experts:
                    path = self.tables[prefix][e]
                    _verify_file(self.root, path, self.files[path])
                if candidate.nbytes != size:
                    raise ValueError("tiled bank physical payload differs from reservation")
                self._cache[prefix] = candidate
                self._cache.move_to_end(prefix)
                self._cache_bytes += size - (old.nbytes if old is not None else 0)
                self._leased[id(candidate)] = candidate
                self.stats["uploaded_experts"] += len(missing)
                self.stats["bank_rebuilds"] += 1
                # Evaluation completed all concatenation reads. Final bank and
                # old epoch bank remain; raw and per-expert arrays are dropped.
                self._pending_uploads.clear()
                self._pending_bank = None
                return candidate
            except BaseException as error:
                self._error = str(error)
                _quarantine_upload(self)
                raise

    def _device_arrays(self):
        banks = (*self._cache.values(), *self._leased.values(), *self._empty.values())
        banks += tuple(b for epoch in self._epochs for b in epoch.banks)
        if self._pending_bank is not None:
            banks += (self._pending_bank,)
        arrays = {id(a): a for b in banks for a in (*b.planes, b.slot_map)}
        arrays.update({id(a): a for a in self._pending_uploads.values()})
        arrays.update({id(e.marker): e.marker for e in self._epochs if hasattr(e.marker, "nbytes")})
        return arrays

    def host_owned_bytes(self):
        arrays = {id(a): a for planes in self._pending_host.values() for a in planes}
        for future in self._read_futures:
            if future.done() and not future.cancelled() and future.exception() is None:
                arrays.update({id(a): a for a in future.result()})
        return sum(a.nbytes for a in arrays.values())

    def live_bytes(self):
        return (sum(a.nbytes for a in self._device_arrays().values()) + self.host_owned_bytes()
                + sum(len(t._cache)*t.row_bytes for t in self.ple_tables.values()))

    resident_bytes = live_bytes

    def apc_revision(self):
        return REVISION

    def receipt(self):
        plan = self.plan.as_dict()
        return dict(mode=self.mode, mechanism_revision=REVISION, allocation="immutable_layer_bank_lru",
                    source_binding=self.binding, plan=plan, live_bytes=self.live_bytes(),
                    reserved_bytes=self.reserved_bytes(), cache_payload_bytes=self._cache_bytes,
                    cache_budget_bytes=self.cache_budget_bytes, leased_items=len(self._leased),
                    active_window=self._active, pending_upload_items=len(self._pending_uploads),
                    pending_completion_epochs=len(self._epochs),
                    requested_read_workers=self.requested_read_workers,
                    effective_read_workers=self.read_workers,
                    pending_host_source_bytes=self.host_owned_bytes(),
                    required_read_inflight=self.pool.counters()["weight_read_inflight"],
                    read_failure=self.pool.failure, parked=self._parked, counters=self.counters(),
                    host_route_reads="miss_only", bank_concat="bank_rebuild_only",
                    upload_granularity="nine_planes_per_verified_host_batch",
                    source_checks="full_hash_on_read; full_seals_before_and_after_bank_rebuild",
                    source_read_byte_scope="whole expert safetensors files including headers",
                    qualification="unqualified")

    def retirement_receipt(self):
        """Include GPU-completion owners, which are deliberately not ReadPools."""
        with self._lock:
            io = self.pool.retirement_receipt()
            completion_drained = not self._epochs and self._completion_worker is None
            drained = bool(self._closed and self._parked and not self._active
                           and completion_drained and not self._read_futures
                           and not self._pending_uploads and not self._pending_host
                           and not self._leased and not self._cache and not self._empty
                           and io["ownership_drained"])
            return dict(ownership_drained=drained, retained_bytes=self.live_bytes(),
                        completion_owners_drained=completion_drained,
                        pending_completion_epochs=len(self._epochs), read_pool=io)

    def resize_cache_budget(self, budget):
        with self._lock:
            self._ready()
            if self._active or type(budget) is not int or not self.minimum_cache_bytes <= budget <= self.maximum_cache_bytes:
                raise ValueError("tiled resize requires an idle admitted budget")
            self.drain_epochs()
            self._fence()
            while self._cache_bytes > budget:
                _, bank = self._cache.popitem(last=False)
                self._cache_bytes -= bank.nbytes
                self.stats["evictions"] += 1
            self.cache_budget_bytes = self.plan.ceiling_bytes = budget
            return budget

    def park(self):
        with self._lock:
            if self._parked:
                return self.receipt()
            if self._active or self._error or self.pool.counters()["weight_read_inflight"] or self._read_futures:
                raise ReadTimeout("tiled leases, failed work or reads prevent parking")
            self.drain_epochs()
            self._fence()
            self._cache.clear()
            self._empty.clear()
            self._cache_bytes = 0
            self._parked = True
            self.stats["parks"] += 1
            return self.receipt()

    def activate(self):
        with self._lock:
            if self._closed or self._error or self.pool.failure:
                raise ValueError("failed tiled bank store cannot reactivate")
            # No source descriptor is reopened implicitly. All future bank
            # fills perform source seal/hash checks; activation checks inventory.
            for paths in self.tables.values():
                for path in paths:
                    _verify_file(self.root, path, self.files[path])
            self._parked = False
            self.stats["activations"] += 1
            return self.receipt()

    def close(self):
        with self._lock:
            self._closed = True
            self.pool.close(self._handles)
            try:
                try:
                    self.drain_epochs()
                except BaseException:
                    # A waiter which returned an error cannot prove completion.
                    # A successful explicit global fence may retire it; a live
                    # waiter still owns its marker/banks and is never forgotten.
                    if any(not epoch.future.done() for epoch in self._epochs):
                        raise
                    self._fence()
                    self._epochs.clear()
                self._fence()
                if self.pool.counters()["weight_read_inflight"] or any(not f.done() for f in self._read_futures):
                    raise ReadTimeout("tiled close retains a late owned read")
                if self._completion_worker is not None:
                    self._completion_worker.close()
                    self._completion_worker = None
            except BaseException:
                _quarantine_upload(self)
                raise
            self._cache.clear()
            self._empty.clear()
            self._leased.clear()
            self._pending_uploads.clear()
            self._pending_host.clear()
            self._read_futures.clear()
            self._pending_bank = None
            self._cache_bytes = 0
            self._active = False
            self._parked = True
            _release_upload(self)

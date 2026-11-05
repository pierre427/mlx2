"""Exact IQ1_S expert-row streaming through the generic early-load contract.

No whole expert tensor is uploaded. Native packed bytes and top-k order are
preserved. One bounded I/O window and per-layer LRUs own every live buffer.
"""

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, replace

import numpy as np

from ..runtime.gguf.packed_metal import PackedMatrix
from ..runtime.gguf.reader import GGUFError
from ..runtime.io_ownership import CloseAction, ReadPool, ReadTimeout, storage_owner
from ..runtime.models.deepseek_v41_engram import file_stamp
from ..runtime.weight_stream import ExpertStreamPlan

REVISION = "v41-iq1-expert-tickets-v2"


def read_rows(fd, entries):
    result = []
    for expert, spans in entries:
        planes = []
        for offset, size in spans:
            data = os.pread(fd, size, offset)
            if len(data) != size:
                parts, received = [data], len(data)
                while received < size:
                    part = os.pread(fd, size-received, offset+received)
                    if not part:
                        raise OSError("short V4.1 expert row read")
                    parts.append(part)
                    received += len(part)
                data = b"".join(parts)
            planes.append(data)
        result.append((expert, tuple(planes)))
    return result


def read_coalesced_rows(fd, entries):
    """Read adjacent exact source spans once; never fetch speculative gap bytes.

    Views retain each bounded read allocation until the evaluation owner has
    uploaded its rows. No three-plane repack or selected-weight stack is made.
    """
    spans = sorted((offset, size, expert, plane) for expert, planes in entries
                   for plane, (offset, size) in enumerate(planes))
    groups = []
    for span in spans:
        if groups and groups[-1][-1][0]+groups[-1][-1][1] == span[0] and span[0]+span[1]-groups[-1][0][0] <= 16 << 20:
            groups[-1].append(span)
        else:
            groups.append([span])
    found = {expert: [None]*len(planes) for expert, planes in entries}
    for group in groups:
        start, size = group[0][0], sum(s[1] for s in group)
        raw = read_rows(fd, [(0, ((start, size),))])[0][1][0]
        view = memoryview(raw)
        for offset, length, expert, plane in group:
            found[expert][plane] = view[offset-start:offset-start+length]
    return [(expert, tuple(found[expert])) for expert, _ in entries]


@dataclass
class ExpertReadTicket:
    owner: object
    thread_id: int
    prefix: str
    values: object
    unique: object
    inverse: object
    found: dict
    missing: list
    futures: list
    hints: list
    keep: object = None
    closed: bool = False


class PackedExpertStream:
    mode = "moe_experts"
    elastic = None

    def __init__(self, path, *, tensors, data_offset, expected_stat, ceiling_bytes, workers=8, top_k=6,
                 coalesced=False, adaptive=False, atlas=None, direct_only=False):
        if (type(workers) is not int or not 1 <= workers <= 32
                or type(top_k) is not int or not 1 <= top_k <= 6
                or type(ceiling_bytes) is not int or ceiling_bytes < 1):
            raise ValueError("invalid expert read/cache bounds")
        self.path, self.expected_stat = str(path), dict(expected_stat)
        self._lock = threading.RLock()
        self._closed = False
        self._parked = False
        self._execution_synchronize = self._synchronize_resize
        self.forced_stock = {"native_iq1_s": True}
        self.coalesced, self.atlas = coalesced, atlas
        self._ticket = None
        self._keys, self._previous = (), None
        self._hints = OrderedDict()
        self._retired_hints = []
        self._hint_disabled = False
        self.hint_budget = (128 << 20) if atlas else 0
        # Bound Python counters, keys and persisted-record decoding separately
        # from the raw hint buffers. These are host physical allocations too.
        self.atlas_metadata_reserve = (128 << 20) if atlas else 0
        self.tables = {}
        for name in tensors:
            if ".ffn_gate_exps.weight" not in name:
                continue
            prefix = name.removesuffix(".ffn_gate_exps.weight")
            group = tuple(tensors[prefix+".ffn_"+kind+"_exps.weight"] for kind in ("gate", "up", "down"))
            if any(t.ggml_type != 19 or len(t.shape) != 3 for t in group):
                raise GGUFError("streamed V4.1 experts require IQ1_S rank3 planes")
            gate, up, down = group
            if up.shape != gate.shape or down.shape != (gate.shape[1], gate.shape[0], gate.shape[2]):
                raise GGUFError("streamed expert planes disagree")
            self.tables[prefix] = group
        sizes = {sum(t.nbytes//t.shape[2] for t in group) for group in self.tables.values()}
        experts = {group[0].shape[2] for group in self.tables.values()}
        if not self.tables or len(sizes) != 1 or len(experts) != 1 or ceiling_bytes < 1:
            raise ValueError("streamed expert cache requires uniform bounded geometry")
        self.row_bytes, count = sizes.pop(), experts.pop()
        if atlas is not None and (atlas.layers != len(self.tables) or atlas.experts != count
                                  or atlas.top_k != min(top_k, count)):
            raise ValueError("expert atlas geometry differs from admitted stream")
        self.capacity = min(count, ceiling_bytes//(len(self.tables)*self.row_bytes))
        if self.capacity < min(top_k, count):
            raise ValueError("expert cache cannot hold one routed step per layer")
        self.caches = {p: OrderedDict() for p in self.tables}
        self.ceiling = self.capacity*len(self.tables)*self.row_bytes
        self.maximum_cache_bytes = self.ceiling
        self.ple_tables = {}
        self._ple_bound = False
        self.dense_provider = None
        self.data_offset = data_offset
        # One whole selected layer stack plus one replacement stack and a
        # bounded raw read batch. This is reserved even at decode width one.
        transient = (1 if direct_only else 2)*count*self.row_bytes + 2*min(32, count)*self.row_bytes + self.hint_budget + self.atlas_metadata_reserve
        self.plan = ExpertStreamPlan(
            layers=len(self.tables), experts_per_layer=count, expert_bytes=self.row_bytes,
            table_bytes=sum(t.nbytes for g in self.tables.values() for t in g),
            capacity_experts=self.capacity, ceiling_bytes=self.ceiling,
            projections=("gate", "up", "down"), load_phase="pre_materialization",
            top_k=min(top_k, count), transient_bound_bytes=transient, staging_window=min(32, count),
            manifest_sha256=hashlib.sha256(json.dumps(expected_stat, sort_keys=True).encode()).hexdigest(),
            table_kinds=("native_IQ1_S",), page_cache="bypass",
        )
        self.stats = {"lookups": 0, "cache_hits": 0, "cache_misses": 0, "bytes_read": 0, "evictions": 0,
                      "upload_batches": 0, "uploaded_experts": 0, "demand_wait_ns": 0,
                      "hint_issued": 0, "hint_used": 0, "hint_discarded": 0, "hint_requested_bytes": 0,
                      "hint_useful_bytes": 0, "hint_failed": 0, "hint_ready": 0, "ephemeral": 0,
                      "hint_prediction_calls": 0, "hint_prediction_candidates": 0,
                      "hint_prediction_selected": 0, "hint_prediction_truncated": 0,
                      "hint_prediction_empty": 0, "hint_prediction_all_excluded": 0,
                      "hint_prediction_all_resident": 0, "hint_prediction_all_inflight": 0,
                      "hint_skip_resident": 0, "hint_skip_inflight": 0, "hint_skip_budget": 0,
                      "hint_skip_disabled": 0, "hint_budget_exhausted": 0, "hint_submit_failed": 0,
                      "parks": 0, "activations": 0}
        from ..runtime.expert_residency import AdaptiveExpertResidency, AdaptiveResidencySettings

        self.adaptive = {p: AdaptiveExpertResidency(count, AdaptiveResidencySettings(enabled=True)) for p in self.tables} if adaptive else {}
        self.fd = os.open(self.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        self.pool = self.hint_pool = self.hint_fd = None
        try:
            self.verify()
            if os.uname().sysname == "Darwin":
                import fcntl

                fcntl.fcntl(self.fd, 48, 1)
                fcntl.fcntl(self.fd, 45, 0)
            self.workers = min(8, workers)
            self.pool = ReadPool(self.workers, "v41-experts", timeout=60.0,
                                 max_pending=self.workers, owner=storage_owner("v41-experts", self.path))
            self.hint_fd = os.dup(self.fd) if atlas else None
            self.hint_pool = (ReadPool(2, "v41-expert-hints", timeout=60.0,
                                      max_pending=max(1, self.hint_budget//self.row_bytes),
                                      owner=storage_owner("v41-expert-hints", self.path)) if atlas else None)
        except BaseException:
            if self.hint_pool is not None:
                self.hint_pool.close(CloseAction(lambda: os.close(self.hint_fd)))
            elif self.hint_fd is not None:
                os.close(self.hint_fd)
            if self.pool is not None:
                self.pool.close(CloseAction(lambda: os.close(self.fd)))
            else:
                os.close(self.fd)
            raise

    def verify(self):
        if file_stamp(os.fstat(self.fd)) != self.expected_stat or file_stamp(os.stat(self.path, follow_symlinks=False)) != self.expected_stat:
            raise ValueError("streamed expert source changed after admission")

    def _available(self):
        if self._closed:
            raise ValueError("streamed expert cache is closed")
        if self._parked:
            raise ValueError("streamed expert cache is parked")
        if self.pool.failure is not None:
            raise ReadTimeout("streamed expert reader failed; reload required: "+self.pool.failure)

    def begin_context(self, keys):
        with self._lock:
            self._available()
            if self._ticket is not None:
                raise ValueError("expert ticket still owns previous context")
            self._keys, self._previous = tuple(keys), None
            self._discard_hints()
            if self.atlas:
                self._predict(0)
                self._predict(1)

    def end_context(self):
        with self._lock:
            self._keys, self._previous = (), None

    def hint_owned_bytes(self):
        # A consumed hint remains a raw allocation owned by its demand ticket
        # until resolution/cancellation. Removing it from _hints is not release.
        acquired = len(self._ticket.hints) if self._ticket is not None else 0
        self._retired_hints[:] = [future for future in self._retired_hints if not future.done()]
        return (len(self._hints)+acquired+len(self._retired_hints))*self.row_bytes

    def _retire_ticket_hints(self, ticket):
        pending = [future for _, future in ticket.hints if not future.done() and not future.cancel()]
        if pending:
            # Optional failures may fall back to demand, but a running syscall
            # still owns its descriptor and raw-buffer reservation until drain.
            self._retired_hints.extend(pending)
            self._hint_disabled = True
            self.hint_pool.fail(ReadTimeout("abandoned expert hint still owns its read"))

    def _discard_hints(self, *, prefix=None):
        for key, future in list(self._hints.items()):
            if (prefix is None or key[0] == prefix) and future.done():
                del self._hints[key]
                self.stats["hint_discarded"] += 1
                # Observe errors even for unused hints. A failed hint cannot
                # provide weight bytes; the ordinary demand lane remains valid.
                try:
                    future.result()
                except Exception:
                    self.stats["hint_failed"] += 1

    def _entries(self, prefix, experts):
        return [(i, tuple((self.data_offset+t.offset+i*(t.nbytes//t.shape[2]), t.nbytes//t.shape[2])
                          for t in self.tables[prefix])) for i in experts]

    def _predict(self, layer, previous=None):
        prefix = f"blk.{layer}"
        if not self.atlas or not self._keys or prefix not in self.tables:
            return
        if self._hint_disabled:
            self.stats["hint_skip_disabled"] += 1
            return
        self.verify()
        resident = set(self.caches[prefix])
        inflight = {expert for hinted_prefix, expert in self._hints if hinted_prefix == prefix}
        diagnostics = {}
        predicted = self.atlas.predict(layer, self._keys, previous=previous,
                                       excluded=resident | inflight, diagnostics=diagnostics)
        skipped = set(diagnostics["excluded_experts"])
        resident_count = len(skipped & resident)
        inflight_count = len((skipped & inflight) - resident)
        candidates = diagnostics["candidates_total"]
        self.stats["hint_prediction_calls"] += 1
        self.stats["hint_prediction_candidates"] += candidates
        self.stats["hint_prediction_selected"] += len(predicted)
        self.stats["hint_prediction_truncated"] += diagnostics["truncated_total"]
        self.stats["hint_prediction_empty"] += int(candidates == 0)
        self.stats["hint_prediction_all_excluded"] += int(candidates > 0 and not diagnostics["eligible_total"])
        self.stats["hint_prediction_all_resident"] += int(candidates > 0 and resident_count == candidates)
        self.stats["hint_prediction_all_inflight"] += int(candidates > 0 and inflight_count == candidates)
        self.stats["hint_skip_resident"] += resident_count
        self.stats["hint_skip_inflight"] += inflight_count
        for index, expert in enumerate(predicted):
            key = prefix, expert
            if self.hint_owned_bytes()+self.row_bytes > self.hint_budget:
                self.stats["hint_skip_budget"] += len(predicted)-index
                self.stats["hint_budget_exhausted"] += 1
                break
            read = read_coalesced_rows if self.coalesced else read_rows
            try:
                self._hints[key] = self.hint_pool.submit(read, self.hint_fd, self._entries(prefix, [expert]))
            except Exception:
                self._hint_disabled = True
                self.stats["hint_failed"] += 1
                self.stats["hint_submit_failed"] += 1
                break
            self.stats["hint_issued"] += 1
            self.stats["hint_requested_bytes"] += self.row_bytes

    def _submit(self, prefix, missing):
        entries = self._entries(prefix, missing)
        groups = [[entries[int(i)] for i in chunk] for chunk in
                  np.array_split(np.arange(len(entries)), min(self.workers, len(entries)))] if entries else []
        read = read_coalesced_rows if self.coalesced else read_rows
        try:
            return [self.pool.submit(read, self.fd, chunk) for chunk in groups]
        except BaseException as error:
            # Some bounded submissions may already own the descriptor. Keep
            # them owned by the pool and refuse reuse after a partial failure.
            self.pool.fail(error)
            raise

    def prepare(self, prefix, ids):
        """Read back exact routes and start one bounded miss batch, no GPU upload."""
        with self._lock:
            self._available()
            if self._ticket is not None:
                raise ValueError("expert read ticket already owned")
            group = self.tables[prefix]
            values = np.asarray(ids)
            if values.dtype != np.uint32 or values.ndim != 2 or values.shape[0] > 1024 or not 1 <= values.shape[1] <= 6 or values.size == 0:
                raise ValueError("streamed experts require bounded uint32 routing rows")
            unique, inverse = np.unique(values, return_inverse=True)
            if int(unique[-1]) >= group[0].shape[2]:
                raise ValueError("streamed expert ID outside bound plane")
            self.verify()
            cache = self.caches[prefix]
            found = {int(i): cache[int(i)] for i in unique if int(i) in cache}
            missing = [int(i) for i in unique if int(i) not in found]
            # Hits belong to this working set. Protect them before inserting
            # misses, otherwise an old hit can be evicted during this gather.
            for expert in found:
                cache.move_to_end(expert)
            self.stats["cache_hits"] += len(unique)-len(missing)
            self.stats["cache_misses"] += len(missing)
            hints = []
            for expert in tuple(missing):
                future = self._hints.pop((prefix, expert), None)
                if future is not None:
                    hints.append((expert, future))
                    missing.remove(expert)
            keep = None
            if prefix in self.adaptive:
                policy = self.adaptive[prefix]
                wanted = list(map(int, unique))
                policy.observe(wanted, prefill=len(values)>1)
                keep = policy.retain(cache, wanted, self.capacity)
            ticket = ExpertReadTicket(self, threading.get_ident(), prefix, values.copy(), unique, inverse, found, missing,
                                      self._submit(prefix, missing[:32]), hints, keep)
            self._ticket = ticket
            try:
                if self.atlas and len(self._keys) == len(values):
                    layer = int(prefix.split(".")[1])
                    self.atlas.observe(layer, self._keys, values, previous=self._previous)
                    self._previous = values.copy()
                    self._discard_hints(prefix=prefix)
                    self._predict(layer+1, values)
            except BaseException:
                self.cancel(ticket)
                raise
            return ticket

    def cancel(self, ticket):
        """Drain an abandoned read window without publishing device buffers."""
        with self._lock:
            if ticket.owner is not self or ticket.thread_id != threading.get_ident():
                raise ValueError("foreign expert ticket evaluation owner")
            if ticket.closed:
                return
            if ticket is not self._ticket:
                raise ValueError("foreign expert ticket")
            deadline = time.monotonic()+60.0
            try:
                for future in ticket.futures:
                    if not future.cancel():
                        future.result(timeout=max(0.0, deadline-time.monotonic()))
                for _, future in ticket.hints:
                    if not future.cancel():
                        try:
                            future.result(timeout=max(0.0, deadline-time.monotonic()))
                        except Exception:
                            self._hint_disabled = True
                            self.stats["hint_failed"] += 1
            except BaseException as error:
                self.pool.fail(error)
                raise
            finally:
                self._retire_ticket_hints(ticket)
                ticket.closed = True
                ticket.futures, ticket.hints, ticket.found = [], [], {}
                self._ticket = None

    def resolve(self, ticket, *, row_buffers=False, host_ids=False):
        import mlx.core as mx

        with self._lock:
            if (self._closed or ticket.owner is not self or ticket is not self._ticket or ticket.closed
                    or ticket.thread_id != threading.get_ident()):
                raise ValueError("foreign or closed expert ticket")
            if self.pool.failure is not None:
                self.cancel(ticket)
                raise ReadTimeout("streamed expert reader failed; reload required: "+self.pool.failure)
            if row_buffers and ticket.values.shape[1] > 6:
                raise ValueError("direct expert row buffers require at most six routes per token")
            if host_ids and not row_buffers:
                raise ValueError("host expert IDs require direct row buffers")
            prefix, values, unique, inverse, found, missing = (ticket.prefix, ticket.values, ticket.unique,
                                                             ticket.inverse, ticket.found, ticket.missing)
            group, cache = self.tables[prefix], self.caches[prefix]
            try:
                for start in range(0, max(1, len(missing)), 32):
                    futures = ticket.futures if start == 0 else self._submit(prefix, missing[start:start+32])
                    pending = []
                    failed_hints = []
                    if start == 0:
                        for expert, future in ticket.hints:
                            ready = future.done()
                            then = time.perf_counter_ns()
                            try:
                                rows = future.result(timeout=60.0)
                                self.stats["hint_used"] += 1
                                self.stats["hint_ready"] += int(ready)
                                self.stats["hint_useful_bytes"] += self.row_bytes
                            except Exception:
                                self.stats["hint_failed"] += 1
                                self._hint_disabled = True
                                failed_hints.append(expert)
                                rows = []
                            self.stats["demand_wait_ns"] += time.perf_counter_ns()-then
                            for expert, planes in rows:
                                pending.append((expert, tuple(mx.array(np.frombuffer(raw, np.uint8)) for raw in planes), 0))
                    for future in futures:
                        then = time.perf_counter_ns()
                        rows = future.result(timeout=60.0)
                        self.stats["demand_wait_ns"] += time.perf_counter_ns()-then
                        for expert, planes in rows:
                            arrays = tuple(mx.array(np.frombuffer(raw, np.uint8)) for raw in planes)
                            pending.append((expert, arrays, sum(len(raw) for raw in planes)))
                    # Demand workers have drained before a failed optional
                    # hint falls back, preserving the bounded queue contract.
                    for future in self._submit(prefix,failed_hints):
                        then=time.perf_counter_ns()
                        rows=future.result(timeout=60.0)
                        self.stats["demand_wait_ns"] += time.perf_counter_ns()-then
                        for expert,planes in rows:
                            pending.append((expert,tuple(mx.array(np.frombuffer(raw,np.uint8)) for raw in planes),sum(len(raw) for raw in planes)))
                    if pending:
                        mx.eval([a for _, arrays, _ in pending for a in arrays])
                        self.stats["upload_batches"] += 1
                    self.stats["uploaded_experts"] += len(pending)
                    for expert, arrays, size in pending:
                        found[expert] = arrays
                        if ticket.keep is None or expert in ticket.keep:
                            cache[expert] = arrays
                        else:
                            self.stats["ephemeral"] += 1
                        if ticket.keep is not None:
                            for victim in tuple(cache):
                                if victim not in ticket.keep:
                                    del cache[victim]
                                    self.stats["evictions"] += 1
                        while len(cache) > self.capacity:
                            cache.popitem(last=False)
                            self.stats["evictions"] += 1
                        self.stats["bytes_read"] += size
                    del pending
                self.verify()
            except FutureTimeout as error:
                failure = ReadTimeout("V4.1 expert read exceeded its deadline")
                self.pool.fail(failure)
                raise failure from error
            except BaseException as error:
                self.pool.fail(error)
                raise
            finally:
                self._retire_ticket_hints(ticket)
                ticket.closed = True
                ticket.futures, ticket.hints, ticket.found = [], [], {}
                self._ticket = None
            for i in unique:
                if int(i) in cache:
                    cache.move_to_end(int(i))
            matrices = []
            for plane, tensor in enumerate(group):
                if row_buffers:
                    row_tensor = replace(tensor, shape=(*tensor.shape[:2], 1), offset=0,
                                         nbytes=tensor.nbytes//tensor.shape[2])
                    matrices.append(tuple(PackedMatrix(found[int(i)][plane], row_tensor) for i in unique))
                    continue
                packed = mx.concatenate([found[int(i)][plane] for i in unique])
                tensor = replace(tensor, shape=(*tensor.shape[:2], len(unique)), offset=0,
                                 nbytes=len(unique)*(tensor.nbytes//tensor.shape[2]))
                matrices.append(PackedMatrix(packed, tensor))
            self.stats["lookups"] += 1
            remapped = inverse.reshape(values.shape).astype(np.uint32)
            return (remapped if host_ids else mx.array(remapped)), tuple(matrices)

    def gather(self, prefix, ids, *, row_buffers=False, host_ids=False):
        if host_ids and not row_buffers:
            raise ValueError("host expert IDs require direct row buffers")
        return self.resolve(self.prepare(prefix, ids), row_buffers=row_buffers, host_ids=host_ids)

    def resume_reservation_bytes(self):
        return (self.ceiling + self.plan.transient_bound_bytes
                + sum(t.resume_reservation_bytes() for t in self.ple_tables.values())
                + (self.dense_provider.resume_reservation_bytes() if self.dense_provider is not None else 0))

    def reserved_bytes(self):
        return ((0 if self._parked else self.ceiling+self.plan.transient_bound_bytes)
                + sum(t.reserved_bytes() for t in self.ple_tables.values())
                + (self.dense_provider.reserved_bytes() if self.dense_provider is not None else 0))

    def live_bytes(self):
        return (sum(len(c) for c in self.caches.values())*self.row_bytes
                + sum(len(t._cache)*t.row_bytes for t in self.ple_tables.values())
                + (self.dense_provider.resident_bytes() if self.dense_provider is not None else 0))

    def bind_dense_provider(self, provider):
        """Register one adapter-owned dense pager without changing its lifecycle."""
        from ..runtime.elastic_memory import byte_count

        with self._lock:
            if self._closed or self._parked or self._ticket is not None or self.elastic is not None:
                raise ValueError("cannot bind dense pages after residency ownership")
            if self.dense_provider is not None and provider is not self.dense_provider:
                raise ValueError("dense page provider already bound")
            for name in ("reserved_bytes", "resident_bytes", "resume_reservation_bytes", "resize_cache_budget"):
                if not callable(getattr(provider, name, None)):
                    raise ValueError("dense page provider has no "+name)
            floor=byte_count(provider.minimum_cache_bytes,"dense minimum cache")
            ceiling=byte_count(provider.maximum_cache_bytes,"dense maximum cache")
            current=byte_count(provider.cache_budget_bytes,"dense current cache")
            if not floor <= current <= ceiling:
                raise ValueError("dense page cache geometry mismatch")
            self.dense_provider = provider

    def bind_ple_providers(self, engram_tables):
        """Count fixed and elastic PLE payload through the same owned manager."""
        from ..runtime.elastic_memory import byte_count

        with self._lock:
            if self._closed or self._parked or self._ticket is not None or self.elastic is not None:
                raise ValueError("cannot bind Engram providers after residency ownership")
            tables = dict(engram_tables)
            if (any(type(key) not in (int, str) or not str(key) for key in tables)
                    or len({str(key) for key in tables}) != len(tables)
                    or len({id(table) for table in tables.values()}) != len(tables)):
                raise ValueError("Engram providers must have distinct ownership and resource keys")
            if self._ple_bound and (tables.keys() != self.ple_tables.keys()
                                    or any(table is not self.ple_tables[key] for key,table in tables.items())):
                raise ValueError("Engram providers already bound")
            for table in tables.values():
                for name in ("reserved_bytes", "resume_reservation_bytes", "resize_cache_budget"):
                    if not callable(getattr(table,name,None)):
                        raise ValueError("Engram provider has no "+name)
                maximum = byte_count(table.maximum_cache_bytes,"Engram maximum cache")
                current = byte_count(table.resume_reservation_bytes(),"Engram current cache")
                current -= byte_count(getattr(table,'carry_staging_bound_bytes',0),'Engram carry staging')
                row_bytes = byte_count(table.row_bytes,"Engram row bytes")
                if not row_bytes or not 0 <= current <= maximum or not isinstance(getattr(table,"_cache",None),dict):
                    raise ValueError("Engram provider cache geometry mismatch")
            self.ple_tables, self._ple_bound = tables, True

    def enable_elastic(self, engram_tables, dense_provider=None):
        """One physical budget for immutable expert/PLE pages and APCv2 loans."""
        from ..runtime.elastic_memory import ElasticMemoryCoordinator, ReclaimableMemory

        if self.elastic is not None or self._ticket is not None or self._closed or self._parked:
            raise ValueError("cannot replace an owned residency coordinator")
        if dense_provider is not None:
            self.bind_dense_provider(dense_provider)
        self.bind_ple_providers(engram_tables)
        tables = self.ple_tables
        minimum = self.plan.top_k*len(self.tables)*self.row_bytes
        resources = {"experts": ReclaimableMemory(minimum,self.maximum_cache_bytes,self._resize_cache_budget,self.ceiling)}
        resources.update({f"ple.{key}": ReclaimableMemory(0,table.maximum_cache_bytes,table.resize_cache_budget,
                                                        table.resume_reservation_bytes()-getattr(table,'carry_staging_bound_bytes',0))
                          for key,table in tables.items()})
        if self.dense_provider is not None:
            dense=self.dense_provider
            resources["dense"]=ReclaimableMemory(dense.minimum_cache_bytes,dense.maximum_cache_bytes,
                                                  dense.resize_cache_budget,dense.cache_budget_bytes)
        self.elastic = ElasticMemoryCoordinator(sum(r.maximum_bytes for r in resources.values()),resources,
                                                 synchronize=self._synchronize_resize)

    @staticmethod
    def _synchronize_resize():
        import mlx.core as mx
        mx.synchronize()

    def bind_execution_synchronizer(self, synchronize):
        if self._closed or self.elastic is None or not callable(synchronize):
            raise ValueError("elastic native pages require an execution synchronizer")
        self.elastic.bind_owner()
        self.elastic.synchronize = synchronize
        self._execution_synchronize = synchronize

    def _resize_cache_budget(self, budget):
        with self._lock:
            if self._ticket is not None or self._closed:
                raise ValueError("cannot resize acquired expert rows")
            if type(budget) is not int or not self.plan.top_k*len(self.tables)*self.row_bytes <= budget <= self.maximum_cache_bytes:
                raise ValueError("native expert budget outside admission")
            self.capacity = budget//(len(self.tables)*self.row_bytes)
            for cache in self.caches.values():
                while len(cache)>self.capacity:
                    cache.popitem(last=False);self.stats["evictions"] += 1
            self.ceiling = self.capacity*len(self.tables)*self.row_bytes
            import mlx.core as mx
            mx.clear_cache()
            return self.ceiling

    def rebalance_headroom(self, *, available_bytes, protected_bytes):
        from ..runtime.elastic_memory import byte_count

        if self.elastic is None or self._closed or self._parked:
            raise ValueError("native residency coordination not selected")
        byte_count(available_bytes,"available_bytes");byte_count(protected_bytes,"protected_bytes")
        floor=sum(r.minimum_bytes for r in self.elastic.resources.values())
        fixed=self.plan.transient_bound_bytes
        fixed+=sum(getattr(t,'carry_staging_bound_bytes',0) for t in self.ple_tables.values())
        if self.dense_provider is not None:
            fixed += self.dense_provider.resume_reservation_bytes()-self.dense_provider.cache_budget_bytes
        ceiling=available_bytes+self.live_bytes()-fixed-protected_bytes
        ceiling=max(floor,min(self.elastic.total_bytes,ceiling))
        return self.elastic.set_claim("execution",self.elastic.total_bytes-ceiling)

    def unfilled_reserve_bytes(self):
        return max(0, self.reserved_bytes()-self.live_bytes())

    def counters(self):
        return {"v41_expert_"+k: v for k, v in self.stats.items()}

    def receipt(self):
        return {"mode": self.mode, "mechanism_revision": REVISION, "load_peak_bounded": True,
                "reserved_bytes": self.reserved_bytes(), "live_bytes": self.live_bytes(),
                "read_workers": self.workers, "raw_batch_experts": 32,
                "coalesced_source_reads": self.coalesced, "adaptive_retention": bool(self.adaptive),
                "atlas": self.atlas.manifest() if self.atlas else None,
                "atlas_stats": dict(self.atlas.stats) if self.atlas else {},
                "hint_reserved_bytes": self.hint_budget,
                "hint_owned_bytes": self.hint_owned_bytes(),
                "atlas_metadata_reserved_bytes": self.atlas_metadata_reserve,
                "retained_host_metadata_bound_bytes": self.atlas_metadata_reserve,
                "parked": self._parked,
                "resume_reservation_bytes": self.resume_reservation_bytes(),
                "read_failure": self.pool.failure,
                "required_read_inflight": self.pool.counters()["weight_read_inflight"],
                "hint_read_inflight": (self.hint_pool.counters()["weight_read_inflight"] if self.hint_pool else 0),
                "dense_pages": ({"reserved_bytes": self.dense_provider.reserved_bytes(),
                                 "resume_reservation_bytes": self.dense_provider.resume_reservation_bytes(),
                                 "resident_bytes": self.dense_provider.resident_bytes(),
                                 "cache_budget_bytes": self.dense_provider.cache_budget_bytes}
                                if self.dense_provider is not None else None),
                "elastic": self.elastic.receipt() if self.elastic is not None else None,
                "qualification": "unqualified", "math": "unchanged native IQ1_S bytes/top-k order"}

    def apc_revision(self):
        return REVISION

    def park(self, *, timeout=60.0):
        """Release immutable rows at a quiescent boundary, keeping source FDs.

        Cache budgets and source identities survive. Admission must reserve
        resume_reservation_bytes() before activation; a logical budget is not
        a claim that the corresponding pages were physically resident.
        """
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 60:
            raise ValueError("expert park timeout must be in (0, 60]")
        with self._lock:
            if self._closed or self._ticket is not None:
                raise ValueError("cannot park a closed/acquired expert stream")
            if self.pool.counters()["weight_read_inflight"]:
                raise ValueError("cannot park while required expert reads remain owned")
            if self._parked:
                return self.receipt()
            deadline = time.monotonic()+timeout
            for future in (*self._hints.values(), *self._retired_hints):
                if future.cancel():
                    continue
                try:
                    future.result(timeout=max(0.0, deadline-time.monotonic()))
                except FutureTimeout as error:
                    failure = ReadTimeout("expert hints did not drain before parking")
                    self._hint_disabled = True
                    self.hint_pool.fail(failure)
                    raise failure from error
                except Exception:
                    self._hint_disabled = True
                    self.stats["hint_failed"] += 1
            self._execution_synchronize()
            self.stats["hint_discarded"] += len(self._hints)
            self._hints.clear()
            self._retired_hints.clear()
            for cache in self.caches.values():
                cache.clear()
            self._keys, self._previous = (), None
            self._parked = True
            self.stats["parks"] += 1
            return self.receipt()

    def activate(self):
        """Revalidate the admitted source; never reopen or silently rebind it."""
        with self._lock:
            if self._closed or self._ticket is not None:
                raise ValueError("cannot activate a closed/acquired expert stream")
            if self.pool.failure is not None:
                raise ReadTimeout("streamed expert reader failed; reload required: "+self.pool.failure)
            self.verify()
            if self._parked:
                self._parked = False
                self.stats["activations"] += 1
            return self.receipt()

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.caches.clear()
            if self.hint_pool is not None:
                self.hint_pool.close(CloseAction(lambda: os.close(self.hint_fd)))
            self._hints.clear()
            self._retired_hints.clear()
            if self._ticket is not None:
                self._ticket.closed = True
                self._ticket.futures, self._ticket.hints, self._ticket.found = [], [], {}
            self._ticket = None
            self.pool.close(CloseAction(lambda: os.close(self.fd)))
            if self.atlas:
                self.atlas.save()

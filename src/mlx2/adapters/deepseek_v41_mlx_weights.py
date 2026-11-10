# SPDX-License-Identifier: MIT
"""Standalone native V4.1 affine weights and globally bounded expert residency.

Only the admitted derived files are opened. I/O workers own bounded host reads;
the evaluation thread owns MLX arrays and keeps expert leases through a fence.
Native F32 QMM is a distinct numerical profile from the original GGUF kernels.
"""

from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import threading
import time
from types import SimpleNamespace

import numpy as np

from ..runtime.immutable_weight_pages import read_page
from ..runtime.io_ownership import ReadPool, ReadTimeout, storage_owner
from ..runtime.weight_stream import ExpertStreamPlan

REVISION = "mlx2.v41-native-expert-global-lru-f32.v1"
MANIFEST = "mlx2-v41-native-manifest.json"
CHUNK_BYTES = 16 << 20
_DTYPES = {"F32": "<f4", "F16": "<f2", "U32": "<u4", "I32": "<i4",
           "U8": "u1", "I8": "i1", "BF16": "<u2"}
_UPLOAD_QUARANTINE = {}
_QUARANTINE_LOCK = threading.RLock()


def _quarantine_upload(owner):
    # A failed adapter constructor can itself become unreachable. Keep uncertain
    # device/staging owners alive independently until an explicit successful fence.
    with _QUARANTINE_LOCK:
        _UPLOAD_QUARANTINE[id(owner)] = owner


def _release_upload(owner):
    with _QUARANTINE_LOCK:
        _UPLOAD_QUARANTINE.pop(id(owner), None)


def _refuse_quarantined_uploads():
    with _QUARANTINE_LOCK:
        if _UPLOAD_QUARANTINE:
            raise RuntimeError("native uploads retain failed-completion owners; release them before reloading")


def release_quarantined_native_uploads():
    """Explicitly retry completion/cleanup; failed owners remain strongly held."""
    with _QUARANTINE_LOCK:
        owners = tuple(_UPLOAD_QUARANTINE.values())
    first = None
    for owner in owners:
        try:
            owner.close()
        except BaseException as error:
            first = first or error
    if first is not None:
        raise first
    return len(owners)


def _stamp(value):
    return dict(dev=value.st_dev, ino=value.st_ino, size=value.st_size,
                mtime_ns=value.st_mtime_ns, ctime_ns=value.st_ctime_ns)


def _seal(record):
    seal = record.get("stat", record.get("source_stat"))
    if (not isinstance(seal, dict) or set(seal) != {"dev", "ino", "size", "mtime_ns", "ctime_ns"}
            or any(type(v) is not int for v in seal.values())
            or seal["size"] != record["size"]
            or "stat" in record and "source_stat" in record and record["stat"] != record["source_stat"]):
        raise ValueError("native file is missing its immutable stat seal")
    return seal


def _path(root, relative):
    item = Path(relative)
    if (not isinstance(relative, str) or item.is_absolute() or ".." in item.parts
            or item.as_posix() != relative or not item.parts):
        raise ValueError("native file path escapes its artifact")
    target = root / item
    if not target.resolve().is_relative_to(root):
        raise ValueError("native file path escapes its artifact")
    return target


def _verify_file(root, relative, record):
    target = _path(root, relative)
    current = target.stat(follow_symlinks=False)
    if not stat.S_ISREG(current.st_mode) or _stamp(current) != _seal(record):
        raise ValueError("native weight file changed after admission: " + relative)
    return target


def _payload_bytes(record):
    return sum(row["data_offsets"][1]-row["data_offsets"][0] for row in record["tensors"].values())


def _manifest_binding(manifest):
    digest, size = hashlib.sha256(), 0
    for part in json.JSONEncoder(sort_keys=True, separators=(",", ":")).iterencode(manifest):
        raw = part.encode()
        digest.update(raw)
        size += len(raw)
    return digest.hexdigest(), size


# Content verification of reduced expert-bank files (Q1/Q2), the same law the
# original Source shards use: hash each file's bytes once, then trust the
# immutable stat seal (dev, ino, size, mtime_ns, ctime_ns) on later reads. The
# key carries the path, admitted sha256 and the seal, so a replaced or touched
# file never matches; every read still checks the seal before and after.
_BANK_VERIFIED = set()
_BANK_VERIFIED_LOCK = threading.Lock()
BANK_READ_STATS = {"content_hashes": 0, "seal_verified_reads": 0}


def _bank_seal_key(target, record):
    seal = _seal(record)
    return (str(target), record["sha256"], record["size"],
            tuple(seal[k] for k in ("dev", "ino", "size", "mtime_ns", "ctime_ns")))


def clear_bank_verification():
    with _BANK_VERIFIED_LOCK:
        _BANK_VERIFIED.clear()


def read_sealed_bank_file(root, relative, record):
    """One sealed reduced-bank safetensors file -> {name: host array}; one open.

    The header is parsed from the same bytes (no second open). The SHA-256 is
    computed on the first read of each (file, sha256, seal) and cached; later
    reads skip it and rely on the stat seal checked on the descriptor before
    and after the read and on the path afterwards. Fail-closed on any change.
    """
    if not isinstance(record.get("sha256"), str) or len(record["sha256"]) != 64:
        raise ValueError("reduced bank record lacks its admitted sha256")
    return read_native_file(root, relative, record, hash_once=True)


def read_native_file(root, relative, record, *, hash_once=False):
    """Read and verify one bounded file; no MLX work or retained mmap on workers.

    Without ``header_bytes``/``tensors`` in ``record`` the safetensors header
    is parsed from the bytes read (bounded) instead of compared to an admitted
    layout. ``hash_once`` caches content verification by stat seal (banks).
    """
    target = _verify_file(root, relative, record)
    key = _bank_seal_key(target, record) if hash_once else None
    fd = os.open(target, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        if _stamp(os.fstat(fd)) != _seal(record):
            raise ValueError("native source descriptor differs from admission")
        if os.uname().sysname == "Darwin":
            import fcntl
            fcntl.fcntl(fd, 48, 1)  # F_NOCACHE
            fcntl.fcntl(fd, 45, 0)  # F_RDAHEAD
        raw = read_page(fd, 0, record["size"], CHUNK_BYTES)
        with _BANK_VERIFIED_LOCK:
            verified = key is not None and key in _BANK_VERIFIED
        if not verified:
            if hashlib.sha256(memoryview(raw)).hexdigest() != record["sha256"]:
                raise ValueError("native weight content differs from admitted hash")
        if _stamp(os.fstat(fd)) != _seal(record):
            raise ValueError("native weight content differs from admitted hash")
        _verify_file(root, relative, record)
        if key is not None:
            with _BANK_VERIFIED_LOCK:
                _BANK_VERIFIED.add(key)
                BANK_READ_STATS["seal_verified_reads" if verified else "content_hashes"] += 1
        admitted = "tensors" in record
        base = record["header_bytes"] if admitted else (
            int.from_bytes(raw[:8], "little")+8 if len(raw) >= 8 else None)
        low, high = (8, CHUNK_BYTES) if admitted else (10, (1 << 20)+8)
        if (type(base) is not int or not low <= base <= min(record["size"], high)
                or int.from_bytes(raw[:8], "little")+8 != base):
            raise ValueError("invalid native safetensors header bound")
        header = json.loads(raw[8:base].tobytes())
        if not isinstance(header, dict):
            raise ValueError("invalid native safetensors header")
        header.pop("__metadata__", None)
        if admitted and header != record["tensors"]:
            raise ValueError("native safetensors header differs from admitted layout")
        result, intervals = {}, []
        for name, row in header.items():
            if not admitted and (not isinstance(row, dict) or set(row) != {"dtype", "shape", "data_offsets"}
                    or not isinstance(row["shape"], list) or not isinstance(row["data_offsets"], list)
                    or len(row["data_offsets"]) != 2):
                raise ValueError("invalid native safetensors tensor row")
            dtype = _DTYPES.get(row["dtype"])
            shape, (first, last) = row["shape"], row["data_offsets"]
            if (dtype is None or any(type(n) is not int or n < 0 for n in shape)
                    or type(first) is not int or type(last) is not int
                    or not 0 <= first <= last <= record["size"]-base
                    or math.prod(shape)*np.dtype(dtype).itemsize != last-first):
                raise ValueError("invalid native safetensors tensor extent")
            intervals.append((first, last))
            result[name] = np.frombuffer(raw, dtype=dtype, count=math.prod(shape),
                                        offset=base+first).reshape(shape)
        intervals.sort()
        if (not intervals or intervals[0][0] != 0 or intervals[-1][1] != record["size"]-base
                or any(a[1] != b[0] for a, b in zip(intervals, intervals[1:]))):
            raise ValueError("native safetensors payload has gaps or overlapping tensors")
        return result
    finally:
        os.close(fd)


def _inventory(artifact):
    manifest = artifact["manifest"]
    if (not manifest.get("finished") or manifest.get("numeric_profile") != "native_f32_qmm"
            or manifest.get("schema") != "mlx2.deepseek-v41-native.v1"):
        raise ValueError("native weights require a completed native_f32_qmm artifact")
    files, tensors = manifest["files"], manifest["tensors"]
    dense, expert = set(), {}
    for name, row in tensors.items():
        storage = row["storage"]
        if storage["kind"] in ("raw", "affine"):
            dense.add(storage["file"])
        elif storage["kind"] == "expert-affine":
            if (storage.get("bits"), storage.get("group_size"), storage.get("mode")) != (2, 32, "affine"):
                raise ValueError("unsupported native expert quantization")
            prefix = name.split(".ffn_", 1)[0]
            paths = tuple(storage["expert_files"])
            if prefix in expert and expert[prefix] != paths:
                raise ValueError("native expert projection inventories disagree")
            expert[prefix] = paths
        elif storage["kind"] != "engram":
            raise ValueError("unsupported native weight storage")
    if not dense or not expert or any(not paths for paths in expert.values()):
        raise ValueError("native artifact lacks dense or expert inventory")
    names = dense | {path for paths in expert.values() for path in paths}
    if any(path not in files for path in names):
        raise ValueError("native file inventory is incomplete")
    for path in names:
        _seal(files[path])
    return sorted(dense), expert


def native_weight_plan(artifact, expert_cache_bytes):
    """CPU-only admission bounds, including active aliases and owned read staging."""
    dense, experts = _inventory(artifact)
    files = artifact["manifest"]["files"]
    expert_files = {path for paths in experts.values() for path in paths}
    largest = max(_payload_bytes(files[path]) for path in expert_files)
    if type(expert_cache_bytes) is not int or expert_cache_bytes < largest:
        raise ValueError("native expert cache must hold one complete expert item")
    dense_bytes = sum(_payload_bytes(files[path]) for path in dense)
    # Model-owned HC/raw anchors may retain F32 casts beside their native F16
    # storage. Count all non-F32 raw casts conservatively before constructing it.
    prepared = sum(4*math.prod(row["numpy_shape"]) for row in artifact["manifest"]["tensors"].values()
                   if row["storage"]["kind"] == "raw" and row["ggml_type"] != 0)
    dense_staging = max(files[path]["size"] for path in dense)+CHUNK_BYTES
    active = max(sum(_payload_bytes(files[path]) for path in paths) for paths in experts.values())
    expert_staging = max(files[path]["size"] for path in expert_files)+CHUNK_BYTES
    binding, manifest_bytes = _manifest_binding(artifact["manifest"])
    return dict(dense_payload_bytes=dense_bytes, dense_prepared_bound_bytes=prepared,
                dense_staging_bound_bytes=dense_staging, expert_cache_bytes=expert_cache_bytes,
                minimum_expert_cache_bytes=largest, expert_active_bound_bytes=active,
                expert_staging_bound_bytes=expert_staging,
                manifest_binding_sha256=binding, manifest_canonical_bytes=manifest_bytes,
                metadata_bound_bytes=max(256 << 20, 16*manifest_bytes),
                expert_table_bytes=sum(_payload_bytes(files[path]) for path in expert_files),
                total_reservation_bytes=dense_bytes+prepared+dense_staging+expert_cache_bytes+active+expert_staging)


class NativeExpertPlan(ExpertStreamPlan):
    def as_dict(self):
        result = super().as_dict()
        result.update(allocation="global_byte_lru", capacity_scope="global",
                      capacity_experts_global=self.ceiling_bytes//max(1, self.expert_bytes),
                      per_layer_equivalent_capacity=self.capacity_experts,
                      resident_ceiling_bytes=self.ceiling_bytes)
        if getattr(self,'reservation_policy',None)=='adaptive_working_set':
            result.update(configured_capacity_bytes=self.ceiling_bytes,
                          reserved_bytes=getattr(self,'working_reservation_bytes',0),
                          reservation_policy='adaptive_working_set',
                          reservation_observation='last live provider reading; capacity is a maximum')
        return result


class NativeExpertStore:
    """A global byte LRU; leases retain evicted active items through evaluation."""
    mode = "moe_experts"
    atlas = elastic = None

    def __init__(self, artifact, cache_bytes, *, read_workers=2, timeout=60.0,
                 _owned_manifest=None, _bounds=None):
        _refuse_quarantined_uploads()
        self.root = Path(artifact["root"]).resolve()
        if _owned_manifest is not None and _owned_manifest is not artifact["manifest"]:
            raise ValueError("native expert manifest owner differs from weights")
        self.manifest = deepcopy(artifact["manifest"]) if _owned_manifest is None else _owned_manifest
        self.files = self.manifest["files"]
        _, self.tables = _inventory({"manifest": self.manifest})
        self.bounds = dict(_bounds) if _bounds is not None else native_weight_plan({"manifest": self.manifest}, cache_bytes)
        if self.bounds["expert_cache_bytes"] != cache_bytes:
            raise ValueError("native expert bounds differ from admitted cache")
        if type(read_workers) is not int or not 1 <= read_workers <= 32:
            raise ValueError("invalid native expert read worker bound")
        self.cache_budget_bytes = self.maximum_cache_bytes = cache_bytes
        self.minimum_cache_bytes = self.bounds["minimum_expert_cache_bytes"]
        self._cache, self._leased = OrderedDict(), {}
        self._pending_uploads, self._pending_host = {}, {}
        self._cache_bytes = 0
        self.ple_tables, self._ple_bound = {}, False
        self._active = self._closed = self._parked = False
        self._owner = None
        self._error = None
        self._lock = threading.RLock()
        self._synchronize = None
        self.stats = dict(cache_hits=0, cache_misses=0, bytes_read=0, evictions=0,
                          uploaded_experts=0, upload_batches=0, demand_wait_ns=0,
                          lookups=0, windows=0, parks=0, activations=0)
        self.binding = self.bounds["manifest_binding_sha256"]
        counts = {len(paths) for paths in self.tables.values()}
        if len(counts) != 1:
            raise ValueError("native expert layers have inconsistent expert counts")
        count = counts.pop()
        self.plan = NativeExpertPlan(layers=len(self.tables), experts_per_layer=count,
            expert_bytes=self.minimum_cache_bytes, table_bytes=self.bounds["expert_table_bytes"],
            capacity_experts=min(count, cache_bytes//(len(self.tables)*self.minimum_cache_bytes)),
            ceiling_bytes=cache_bytes, projections=("gate", "up", "down"),
            load_phase="pre_materialization", top_k=artifact["args"].num_experts_per_tok,
            transient_bound_bytes=self.bounds["expert_active_bound_bytes"]+self.bounds["expert_staging_bound_bytes"],
            staging_window=1, manifest_sha256=self.binding,
            table_kinds=("native_affine2_group32_f32",), page_cache="bypass")
        self.pool = ReadPool(min(8, read_workers), "v41-native-experts", timeout=timeout, max_pending=1,
                             owner=storage_owner("v41-native-experts", self.root))

    def _ready(self):
        if self._closed or self._parked or self._error:
            raise ValueError("native expert provider is closed, parked or failed")
        if self.pool.failure is not None:
            raise ReadTimeout("native expert reader failed; reload required: " + self.pool.failure)
        owner = threading.get_ident()
        if self._owner is not None and self._owner != owner:
            raise ValueError("native experts belong to another evaluation thread")
        self._owner = owner

    def _fence(self):
        import mlx.core as mx
        (self._synchronize or mx.synchronize)()

    @contextmanager
    def window(self):
        with self._lock:
            self._ready()
            if self._active:
                raise ValueError("native expert execution window already active")
            if getattr(self,'adaptive_residency_enabled',False):self.adaptive_residency_update()
            self._active = True
            self.stats["windows"] += 1
        try:
            yield
        except BaseException as error:
            self._error = str(error)
            raise
        finally:
            # Failed completion retains active ownership and every lease.
            try:
                self._fence()
            except BaseException as error:
                self._error = str(error)
                _quarantine_upload(self)
                raise
            with self._lock:
                self._leased.clear()
                self._pending_uploads.clear()
                self._pending_host.clear()
                self._active = False
                if getattr(self,'adaptive_residency_enabled',False):self.adaptive_residency_update()

    def acquire(self, prefix, expert):
        import mlx.core as mx
        with self._lock:
            self._ready()
            if not self._active or prefix not in self.tables or type(expert) is not int:
                raise ValueError("native expert acquire requires an admitted execution window")
            if not 0 <= expert < len(self.tables[prefix]):
                raise ValueError("native expert index outside admitted layer")
            path = self.tables[prefix][expert]
            record = self.files[path]
            _verify_file(self.root, path, record)
            if path in self._leased:
                return self._leased[path]
            leased_bytes = sum(sum(v.nbytes for v in values.values()) for values in self._leased.values())
            if leased_bytes+_payload_bytes(record) > self.bounds["expert_active_bound_bytes"]:
                raise ValueError("native expert leases exceed one admitted layer")
            if path in self._cache:
                arrays = self._cache[path]
                self._cache.move_to_end(path)
                self.stats["cache_hits"] += 1
            else:
                started = time.perf_counter_ns()
                try:
                    raw = self.pool.call(read_native_file, self.root, path, record)
                finally:
                    self.stats["demand_wait_ns"] += time.perf_counter_ns()-started
                expected = {kind+"."+field for kind in ("gate", "up", "down")
                            for field in ("qweight", "scales", "biases")}
                if set(raw) != expected:
                    raise ValueError("native expert item must contain exactly nine arrays")
                self._pending_host[path] = raw
                arrays = self._pending_uploads[path] = {}
                for key, value in raw.items():
                    arrays[key] = mx.array(value)
                mx.eval(arrays)
                self._pending_host.pop(path)
                del raw
                self.stats["cache_misses"] += 1
                self.stats["bytes_read"] += record["size"]
                self.stats["uploaded_experts"] += 1
                self.stats["upload_batches"] += 1
                size = sum(value.nbytes for value in arrays.values())
                if size != _payload_bytes(record) or size > self.cache_budget_bytes:
                    raise ValueError("native expert payload exceeds its admitted cache bound")
                self._cache[path] = arrays
                self._cache_bytes += size
                while self._cache_bytes > self.cache_budget_bytes:
                    _, victim = self._cache.popitem(last=False)
                    self._cache_bytes -= sum(value.nbytes for value in victim.values())
                    self.stats["evictions"] += 1
            self._leased[path] = arrays
            self._pending_uploads.pop(path, None)
            return arrays

    def begin_context(self, keys):
        self._ready()

    def end_context(self):
        pass

    def live_bytes(self):
        unique = {id(v): v for item in (*self._cache.values(), *self._leased.values(), *self._pending_uploads.values()) for v in item.values()}
        return sum(v.nbytes for v in unique.values())+sum(len(t._cache)*t.row_bytes for t in self.ple_tables.values())

    resident_bytes = live_bytes

    @property
    def adaptive_residency_enabled(self):
        return getattr(self,'_adaptive_residency_enabled',False)

    @property
    def memory_admission_policy(self):
        return 'attempt_after_reclaim' if self.adaptive_residency_enabled else 'bounded_reservation'

    @property
    def adaptive_state(self):
        return dict(getattr(self,'_adaptive_last',{}),enabled=self.adaptive_residency_enabled,
            target_cache_bytes=getattr(self,'_adaptive_target',self.cache_budget_bytes),
            target_payload_bytes=getattr(self,'_adaptive_payload_ceiling',self.cache_budget_bytes),
            working_set_experts=getattr(self,'_adaptive_working_set',None))

    def configure_adaptive_residency(self,enabled=True,*,available_bytes=None,headroom_bytes=0,
                                     working_set_experts=6,staging_bytes=None,staging_within_cache=False):
        """Adapter opt-in: a configured cache is a maximum, never prepaid RAM.

        Other adapters remain on their existing reservation contract. Source
        may supply a measured availability callback or a planner-owned target.
        Pressure is advisory: immutable active leases survive a smaller target.
        """
        staging=max(self.minimum_cache_bytes,self.bounds.get('expert_staging_bound_bytes',self.minimum_cache_bytes))if staging_bytes is None else staging_bytes
        if (type(enabled)is not bool or available_bytes is not None and not callable(available_bytes)
                or type(headroom_bytes)is not int or headroom_bytes<0
                or type(working_set_experts)is not int or not 1<=working_set_experts<=384
                or type(staging)is not int or staging<self.minimum_cache_bytes
                or type(staging_within_cache)is not bool):
            raise ValueError('Invalid adaptive expert residency contract')
        self._adaptive_residency_enabled=enabled
        self._adaptive_available=available_bytes;self._adaptive_headroom=headroom_bytes
        self._adaptive_working_set=working_set_experts;self._adaptive_staging=staging
        self._adaptive_staging_within_cache=staging_within_cache
        self._adaptive_override=None;self._adaptive_target=self.maximum_cache_bytes
        self._adaptive_protected_bytes=0
        self._adaptive_last={}
        self.plan.reservation_policy='adaptive_working_set' if enabled else None
        if enabled:self.plan.working_reservation_bytes=self._adaptive_reservation_bytes()
        return self.adaptive_residency_update()if enabled else {'enabled':False}

    def _adaptive_victim(self):
        protected=set(self._leased)|set(self._pending_uploads)
        return next((key for key in self._cache if key not in protected),None)

    def _adaptive_evict_key(self,key):
        arrays=self._cache.pop(key)
        amount=getattr(arrays,'allocation_nbytes',None)
        if amount is None:amount=sum(v.nbytes for v in arrays.values())
        self._cache_bytes-=amount;self.stats['evictions']+=1
        return amount

    def adaptive_residency_update(self,*,incoming_bytes=0,measured_available_bytes=None,protected_bytes=None):
        if not self.adaptive_residency_enabled:return {'enabled':False}
        if type(incoming_bytes)is not int or incoming_bytes<0:raise ValueError('Invalid imminent expert bytes')
        with self._lock:
            available=measured_available_bytes
            try:
                if available is not None:pass
                elif self._adaptive_available is None:
                    from ..runtime.os_memory import host_memory_snapshot
                    host=host_memory_snapshot();available=None if host is None else host.available_bytes
                else:available=self._adaptive_available()
            except Exception:
                available=None
            known=type(available)is int and available>=0
            available=available if known else 0
            if protected_bytes is not None:
                if type(protected_bytes)is not int or protected_bytes<0:raise ValueError('Invalid protected working bytes')
                self._adaptive_protected_bytes=protected_bytes
            staging=self._adaptive_staging if self._adaptive_staging_within_cache else 0
            floor=min(self.maximum_cache_bytes,self.minimum_cache_bytes*self._adaptive_working_set+staging)
            live_target=min(self.maximum_cache_bytes,max(floor,self._cache_bytes+available
                -max(self._adaptive_headroom,self._adaptive_protected_bytes)))
            target=(live_target if self._adaptive_override is None else min(live_target,max(floor,self._adaptive_override)))
            self._adaptive_target=target
            payload=max(self.minimum_cache_bytes,target-staging)
            self._adaptive_payload_ceiling=payload
            evicted=0
            # Each iteration removes one unleased object. Exhausted pins defer
            # shrinking instead of spinning or recycling an encoded resource.
            for _ in range(len(self._cache)):
                if self._cache_bytes+incoming_bytes<=payload:break
                victim=self._adaptive_victim()
                if victim is None:break
                evicted+=self._adaptive_evict_key(victim)
            self._adaptive_last=dict(enabled=True,configured_maximum_bytes=self.maximum_cache_bytes,
                effective_cache_target_bytes=target,effective_payload_target_bytes=payload,
                measured_available_bytes=available,availability_known=known,evicted_bytes=evicted,
                deferred_for_pins=self._cache_bytes+incoming_bytes>payload,
                imminent_expert_window_bytes=self.minimum_cache_bytes*self._adaptive_working_set,
                staging_bytes=self._adaptive_staging,precision_selection='unchanged')
            return dict(self._adaptive_last)

    def rebalance_headroom(self,*,available_bytes,protected_bytes):
        if not self.adaptive_residency_enabled:raise ValueError('Adaptive expert residency is not enabled')
        if any(type(v)is not int or v<0 for v in (available_bytes,protected_bytes)):
            raise ValueError('Headroom rebalance requires measured nonnegative bytes')
        result=self.adaptive_residency_update(measured_available_bytes=available_bytes,protected_bytes=protected_bytes)
        # Dropping a tensor is not physical free-memory credit. Release only
        # allocator-owned unused blocks; the caller MUST remeasure afterward.
        if result['evicted_bytes']:
            self._adaptive_reclaim_allocator()
        result['free_memory_credit_bytes']=0
        return result

    def _adaptive_reclaim_allocator(self):
        import mlx.core as mx
        mx.clear_cache()

    def release_completed_experts(self,keys,*,roots,wait=None):
        """Release a sequential expert after all its output consumers complete.

        Bounded completion uses key-based epoch counters and cannot safely mix
        early release/reacquisition generations. Its caller must choose an
        ordinary provider window for this sequential route.
        """
        if not self.adaptive_residency_enabled:raise ValueError('Adaptive expert residency is not enabled')
        keys=tuple(keys);roots=tuple(roots)
        if not keys or len(set(keys))!=len(keys)or not roots or any(v is None for v in roots):
            raise ValueError('Completed expert release requires distinct keys and every output root')
        with self._lock:
            if not self._active or getattr(self,'_completion_collector',None)is not None:
                raise ValueError('Sequential expert release requires an ordinary active provider window')
            if any(key not in self._leased or key in self._pending_uploads or key in self._pending_host for key in keys):
                raise ValueError('Cannot release an absent or pending expert upload')
            owners=tuple(self._leased[key]for key in keys)
            if wait is None:
                import mlx.core as mx
                mx.eval(*roots)
            else:wait(roots)
            if any(self._leased.get(key)is not owner for key,owner in zip(keys,owners)):
                raise RuntimeError('Completed expert ownership generation changed during wait')
            for key in keys:del self._leased[key]
            self.stats['adaptive_completed_expert_releases']=self.stats.get('adaptive_completed_expert_releases',0)+len(keys)
            return dict(released=len(keys),rebalanced=self.adaptive_residency_update(),free_memory_credit_bytes=0)

    def set_adaptive_cache_target(self,cache_bytes,*,working_set_experts=None):
        if not self.adaptive_residency_enabled:raise ValueError('Adaptive expert residency is not enabled')
        if cache_bytes is not None and(type(cache_bytes)is not int or cache_bytes<self.minimum_cache_bytes):
            raise ValueError('Adaptive target must admit one expert')
        if working_set_experts is not None:
            if type(working_set_experts)is not int or not 1<=working_set_experts<=384:raise ValueError('Invalid imminent expert count')
            self._adaptive_working_set=working_set_experts
        self._adaptive_override=cache_bytes
        return self.adaptive_residency_update()

    def record_pressure_attempt(self,required_bytes,available_bytes):
        if any(type(v)is not int or v<0 for v in (required_bytes,available_bytes)):
            raise ValueError('Pressure accounting requires actual nonnegative byte counts')
        self.last_pressure_admission=dict(required_bytes=required_bytes,available_bytes=available_bytes,
            shortfall_bytes=max(0,required_bytes-available_bytes),policy=self.memory_admission_policy,
            free_memory_credit_bytes=0)
        return dict(self.last_pressure_admission)

    def _adaptive_reservation_bytes(self):
        live=self.live_bytes()
        ple_live=sum(len(t._cache)*t.row_bytes for t in self.ple_tables.values())
        host=sum(getattr(value,'host_allocation_nbytes',self.files.get(key,{}).get('size',0))
                 for key,value in self._pending_host.items())
        return (max(0,live-ple_live,self.minimum_cache_bytes*self._adaptive_working_set)
                +max(host,self._adaptive_staging)
                +sum(getattr(t,'resume_reservation_bytes',t.reserved_bytes)()for t in self.ple_tables.values()))

    def resume_reservation_bytes(self):
        if self.adaptive_residency_enabled:return self._adaptive_reservation_bytes()
        return self.cache_budget_bytes+self.plan.transient_bound_bytes+sum(t.resume_reservation_bytes() for t in self.ple_tables.values())

    def reserved_bytes(self):
        if self.adaptive_residency_enabled:
            value=(sum(t.reserved_bytes()for t in self.ple_tables.values())if self._parked else self._adaptive_reservation_bytes())
            self.plan.working_reservation_bytes=value
            return value
        return ((0 if self._parked else self.cache_budget_bytes+self.plan.transient_bound_bytes)
                +sum(t.reserved_bytes() for t in self.ple_tables.values()))

    def bind_ple_providers(self, engram_tables):
        if self._active or self._closed or self._parked:
            raise ValueError("cannot bind PLE during native expert ownership")
        tables = dict(engram_tables)
        if len({id(t) for t in tables.values()}) != len(tables):
            raise ValueError("native PLE providers must have distinct ownership")
        if self._ple_bound and (tables.keys() != self.ple_tables.keys()
                               or any(t is not self.ple_tables[k] for k, t in tables.items())):
            raise ValueError("native PLE providers already bound")
        for table in tables.values():
            if (not callable(getattr(table, "reserved_bytes", None))
                    or not callable(getattr(table, "resume_reservation_bytes", None))
                    or not isinstance(getattr(table, "_cache", None), dict)
                    or type(table.row_bytes) is not int or table.row_bytes < 1):
                raise ValueError("invalid native PLE provider")
        self.ple_tables, self._ple_bound = tables, True

    def unfilled_reserve_bytes(self):
        return max(0, self.reserved_bytes()-self.live_bytes())

    def counters(self):
        return {"v41_expert_"+k: v for k, v in self.stats.items()}

    def apc_revision(self):
        return REVISION

    def receipt(self):
        reservation=self.reserved_bytes()
        result=dict(mode=self.mode, mechanism_revision=REVISION, allocation="global_byte_lru",
                    numeric_profile="native_f32_qmm", source_binding=self.binding,
                    source_read_byte_scope="whole expert safetensors files including headers",
                    plan=self.plan.as_dict(), live_bytes=self.live_bytes(), reserved_bytes=reservation,
                    cache_payload_bytes=self._cache_bytes, cache_budget_bytes=self.cache_budget_bytes,
                    leased_items=len(self._leased), active_window=self._active,
                    pending_upload_items=len(self._pending_uploads),
                    pending_host_source_bytes=sum(self.files[path]["size"] for path in self._pending_host),
                    required_read_inflight=self.pool.counters()["weight_read_inflight"],
                    read_failure=self.pool.failure, parked=self._parked,
                    counters=self.counters(), qualification="unqualified")
        if self.adaptive_residency_enabled:
            result.update(adaptive_residency=dict(self._adaptive_last),memory_admission_policy=self.memory_admission_policy,
                          last_pressure_admission=getattr(self,'last_pressure_admission',None))
        return result

    def resize_cache_budget(self, budget):
        if self.adaptive_residency_enabled:return self.set_adaptive_cache_target(budget)['effective_cache_target_bytes']
        self._ready()
        if self._active or type(budget) is not int or not self.minimum_cache_bytes <= budget <= self.maximum_cache_bytes:
            raise ValueError("native expert resize requires an idle admitted budget")
        self._fence()
        while self._cache_bytes > budget:
            _, victim = self._cache.popitem(last=False)
            self._cache_bytes -= sum(v.nbytes for v in victim.values())
            self.stats["evictions"] += 1
        self.cache_budget_bytes = self.plan.ceiling_bytes = budget
        return budget

    def park(self):
        if self._parked:
            return self.receipt()
        if self._active or self.pool.counters()["weight_read_inflight"]:
            raise ReadTimeout("native expert active leases or reads prevent parking")
        self._fence()
        self._cache.clear()
        self._cache_bytes = 0
        self._parked = True
        self.stats["parks"] += 1
        return self.receipt()

    def activate(self):
        if self._closed or self._error or self.pool.failure:
            raise ValueError("failed native expert provider cannot reactivate")
        # Every required item is checked against its seal before reuse/read.
        self._parked = False
        self.stats["activations"] += 1
        return self.receipt()

    def close(self):
        self.pool.close()
        self._closed = True
        try:
            self._fence()
        except BaseException:
            _quarantine_upload(self)
            raise
        self._leased.clear()
        self._pending_uploads.clear()
        self._pending_host.clear()
        self._active = False
        self._cache.clear()
        self._cache_bytes = 0
        if self.pool.counters()["weight_read_inflight"]:
            raise ReadTimeout("native expert close retains a late owned read")
        self._parked = True
        _release_upload(self)


class NativeV41Weights:
    compute_policy = "native_f32_qmm:"+REVISION
    dense_prefill_policy = expert_prefill_policy = "reference"
    expert_io_policy = "blocking"
    shared_policy = "reference"
    dense_pages = None

    def __init__(self, artifact, *, expert_cache_bytes, expert_read_workers=2):
        _refuse_quarantined_uploads()
        self.artifact = {**artifact, "manifest": deepcopy(artifact["manifest"])}
        self.root = Path(artifact["root"]).resolve()
        self.metadata = deepcopy(artifact["metadata"])
        self.bounds = native_weight_plan(self.artifact, expert_cache_bytes)
        self.tensors = {name: SimpleNamespace(name=name, **row)
                        for name, row in self.artifact["manifest"]["tensors"].items()}
        self._dense, self._quant, self._packed = {}, {}, {}
        self._pending_dense, self._pending_dense_host = {}, None
        self._loaded = False
        self._error = None
        self._parked = self._closed = False
        self.scheduled_expert_calls = {}
        self.scheduled_dense_calls = {}
        self.scheduled_grouped_calls = {}
        self.scheduled_expert_buffer_operations = self.expert_buffer_counters = {}
        self._dense_files, _ = _inventory(self.artifact)
        self._read_pool = ReadPool(1, "v41-native-dense", timeout=60.0, max_pending=1,
                                   owner=storage_owner("v41-native-dense", self.root))
        self.expert_stream = None
        try:
            self.expert_stream = NativeExpertStore(self.artifact, expert_cache_bytes,
                read_workers=expert_read_workers, _owned_manifest=self.artifact["manifest"], _bounds=self.bounds)
        except BaseException:
            self._read_pool.close()
            self._dense.clear()
            self._quant.clear()
            raise

    def _ready(self):
        if self._parked or self._closed or self._error:
            raise ValueError("native weights are parked, closed or failed")

    def _ensure_dense(self):
        self._ready()
        if not self._loaded:
            self._load_dense()

    def _load_dense(self):
        import mlx.core as mx
        files = self.artifact["manifest"]["files"]
        try:
            for path in self._dense_files:
                raw = self._read_pool.call(read_native_file, self.root, path, files[path])
                self._pending_dense_host = raw
                arrays = self._pending_dense
                for name, value in raw.items():
                    arrays[name] = mx.array(value)
                    if files[path]["tensors"][name]["dtype"] == "BF16":
                        arrays[name] = arrays[name].view(mx.bfloat16)
                mx.eval(arrays)
                if set(arrays) & set(self._dense):
                    raise ValueError("native dense tensor appears in multiple files")
                self._dense.update(arrays)
                self._pending_dense.clear()
                self._pending_dense_host = None
                del arrays, raw
            for name, tensor in self.tensors.items():
                storage = tensor.storage
                if storage["kind"] == "affine":
                    if (storage["bits"], storage["group_size"], storage["mode"]) != (8, 32, "affine"):
                        raise ValueError("unsupported native dense quantization")
                    self._quant[name] = tuple(self._dense[storage["keys"][field]] for field in ("qweight", "scales", "biases"))
                    _validate_affine(self._quant[name], tensor.numpy_shape, bits=8)
                elif storage["kind"] == "raw":
                    value = self._dense[storage["key"]]
                    if tuple(value.shape) != tuple(tensor.numpy_shape):
                        raise ValueError("native raw tensor shape differs from manifest")
            self._loaded = True
        except BaseException as error:
            self._error = str(error)
            # The adapter already owns this object before lazy loading starts.
            # Keep both staging and device arrays if completion cannot be proven.
            try:
                mx.synchronize()
            except BaseException:
                _quarantine_upload(self)
                raise
            self._dense.clear()
            self._quant.clear()
            self._pending_dense.clear()
            self._pending_dense_host = None
            raise

    def verify_source(self):
        self._ready()
        files = self.artifact["manifest"]["files"]
        for path in self._dense_files:
            _verify_file(self.root, path, files[path])
        # Expert files are verified on every acquisition, including cache hits.
        return True

    def array(self, name):
        self._ensure_dense()
        storage = self.tensors[name].storage
        if storage["kind"] != "raw":
            raise ValueError("native array access cannot expand packed or disk-only weights")
        return self._dense[storage["key"]]

    def packed(self, name):
        self._ensure_dense()
        if name not in self._quant:
            raise ValueError("native packed lookup requires a resident dense affine matrix")
        return self._quant[name]

    def can_fuse_dense(self, x, *names):
        return False

    def linear(self, name):
        self._ensure_dense()
        import mlx.core as mx
        if name in self._quant:
            def apply(x):
                self._ready()
                q, s, b = self._quant[name]
                self.scheduled_dense_calls["native_affine8_f32"] = self.scheduled_dense_calls.get("native_affine8_f32", 0)+1
                return mx.quantized_matmul(x.astype(mx.float32), q, s, b, group_size=32, bits=8).astype(mx.float32)
        else:
            def apply(x):
                self._ready()
                return x.astype(mx.float32) @ self.array(name).astype(mx.float32).T
        return apply

    def gate_up(self, x, gate_name, up_name):
        return self.linear(gate_name)(x), self.linear(up_name)(x)

    def grouped(self, x, name, *, groups=8):
        import mlx.core as mx
        self._ensure_dense()
        shape = tuple(self.tensors[name].numpy_shape)
        if len(shape) != 2 or shape[0] % groups or x.ndim != 4 or x.shape[2:] != (groups, shape[1]):
            raise ValueError("native grouped projection shape differs from admitted matrix")
        n = shape[0]//groups
        if name not in self._quant:
            return mx.einsum("btgi,goi->btgo", x.astype(mx.float32), self.array(name).astype(mx.float32).reshape(groups, n, shape[1]))
        q, s, b = self._quant[name]
        value = mx.quantized_matmul(x.transpose(0, 2, 1, 3).astype(mx.float32),
            q.reshape(groups, n, q.shape[-1]), s.reshape(groups, n, s.shape[-1]),
            b.reshape(groups, n, b.shape[-1]), group_size=32, bits=8)
        self.scheduled_grouped_calls["native_affine8_f32"] = self.scheduled_grouped_calls.get("native_affine8_f32", 0)+1
        return value.transpose(0, 2, 1, 3).astype(mx.float32)

    def prepare_embedding(self, name="token_embd.weight"):
        self._ensure_dense()
        return self.packed(name) if name in self._quant else self.array(name)

    def embedding(self, ids, name="token_embd.weight"):
        import mlx.core as mx
        self._ensure_dense()
        if name not in self._quant:
            return self.array(name)[ids].astype(mx.float32)
        q, s, b = self._quant[name]
        flat = ids.reshape(-1)
        rows = mx.dequantize(q[flat], s[flat], b[flat], group_size=32, bits=8, dtype=mx.float32)
        return rows.reshape(*ids.shape, self.tensors[name].numpy_shape[-1])

    def moe(self, x, ids, scores, prefix, *, ticket=None):
        import mlx.core as mx
        from ..runtime.models.deepseek_v4 import limited_swiglu
        self._ready()
        if ticket is not None:
            raise ValueError("native expert provider does not accept packed-GGUF tickets")
        values = np.asarray(ids)
        if (values.dtype != np.uint32 or values.ndim != 2 or not 0 < values.shape[0] <= 1024
                or not 0 < values.shape[1] <= self.artifact["args"].num_experts_per_tok
                or x.ndim != 2 or x.shape != (values.shape[0], self.artifact["args"].hidden_size)
                or scores.shape != values.shape):
            raise ValueError("native expert route shape differs from admitted execution")
        flat = values.reshape(-1)
        order = np.argsort(flat, kind="stable")
        sorted_ids = flat[order]
        if prefix not in self.expert_stream.tables or int(sorted_ids[-1]) >= len(self.expert_stream.tables[prefix]):
            raise ValueError("native expert routes outside admitted inventory")
        owned = not self.expert_stream._active
        with self.expert_stream.window() if owned else nullcontext():
            parts = []
            rounded = lambda v: v.astype(mx.bfloat16).astype(mx.float32)
            for expert in np.unique(sorted_ids):
                locations = order[sorted_ids == expert]
                row_indices = mx.array(locations//values.shape[1], mx.uint32)
                slot_indices = mx.array(locations, mx.uint32)
                data = self.expert_stream.acquire(prefix, int(expert))
                def project(value, kind):
                    params = tuple(data[kind+"."+field] for field in ("qweight", "scales", "biases"))
                    target = ((self.artifact["args"].hidden_size, self.artifact["args"].moe_intermediate_size)
                              if kind == "down" else (self.artifact["args"].moe_intermediate_size, self.artifact["args"].hidden_size))
                    _validate_affine(params, target, bits=2)
                    return mx.quantized_matmul(value.astype(mx.float32), *params, group_size=32, bits=2).astype(mx.float32)
                selected = x[row_indices]
                gate, up = rounded(project(selected, "gate")), rounded(project(selected, "up"))
                middle = rounded(limited_swiglu(gate, up)*scores.reshape(-1)[slot_indices, None])
                parts.append(rounded(project(middle, "down")))
            # Only activation rows are concatenated; expert weight arrays remain
            # separate immutable items. Restore original route slots before sum.
            sorted_output = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)
            restored = sorted_output[mx.array(np.argsort(order), mx.uint32)].reshape(*values.shape, x.shape[-1])
            output = mx.zeros(x.shape, mx.float32)
            for slot in range(values.shape[1]):
                output = output+restored[:, slot]
            self.scheduled_expert_calls["native_affine2_f32_per_expert"] = self.scheduled_expert_calls.get("native_affine2_f32_per_expert", 0)+1
            self.expert_stream.stats["lookups"] += 1
            if owned:
                mx.eval(output)
            return output

    def dense_window(self):
        self._ready()
        return self.expert_stream.window()

    def evaluate_dense_window(self, *values):
        import mlx.core as mx
        def arrays(value):
            if isinstance(value, mx.array):
                yield value
            elif isinstance(value, dict):
                for item in value.values():
                    yield from arrays(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    yield from arrays(item)
        mx.eval(list(arrays(values)))

    def preload_native_linear_fp32(self):
        self._ready()

    def native_fp32_parameters(self):
        return {}

    def resident_bytes(self):
        unique = {id(v): v for v in (*self._dense.values(), *self._pending_dense.values())}
        return sum(v.nbytes for v in unique.values())

    def resume_reservation_bytes(self):
        return self.bounds["dense_payload_bytes"]+self.bounds["dense_prepared_bound_bytes"]+self.bounds["dense_staging_bound_bytes"]

    def reserved_bytes(self):
        return 0 if self._parked and not self.resident_bytes() and self._pending_dense_host is None else self.resume_reservation_bytes()

    def dense_receipt(self):
        return dict(mode="native_affine_resident", numeric_profile="native_f32_qmm",
                    retained_payload_bytes=self.resident_bytes(), reserved_bytes=self.reserved_bytes(),
                    pending_upload_arrays=len(self._pending_dense),
                    parked=self._parked, qualification="unqualified", source_gguf_required=False)

    def park(self):
        if self._parked:
            return self.dense_receipt()
        self.expert_stream.park()
        if self._read_pool.counters()["weight_read_inflight"]:
            raise ReadTimeout("native dense reads prevent parking")
        self._dense.clear()
        self._quant.clear()
        self._pending_dense.clear()
        self._pending_dense_host = None
        self._loaded = False
        self._parked = True
        return self.dense_receipt()

    def activate(self):
        if self._closed or self._error or self._read_pool.failure:
            raise ValueError("failed native weights cannot reactivate")
        if self._parked:
            self._load_dense()
            self.expert_stream.activate()
            self._parked = False
        return self.dense_receipt()

    def close(self):
        first = None
        try:
            if self.expert_stream is not None:
                self.expert_stream.close()
        except BaseException as error:
            first = error
        self._read_pool.close()
        if first is None and self._read_pool.counters()["weight_read_inflight"]:
            first = ReadTimeout("native dense close retains a late owned read")
        if first is None:
            self._dense.clear()
            self._quant.clear()
            self._pending_dense.clear()
            self._pending_dense_host = None
            self._loaded = False
            self._closed = self._parked = True
            _release_upload(self)
        if first is not None:
            _quarantine_upload(self)
            raise first


def _validate_affine(params, shape, *, bits):
    import mlx.core as mx
    q, s, b = params
    shape = tuple(shape)
    if (len(shape) != 2 or shape[-1] % 32 or q.dtype != mx.uint32
            or s.dtype != mx.float32 or b.dtype != mx.float32
            or q.shape != (shape[0], shape[1]//(32//bits))
            or s.shape != (shape[0], shape[1]//32) or b.shape != s.shape):
        raise ValueError("native affine matrix layout or precision differs from profile")

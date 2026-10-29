# SPDX-License-Identifier: MIT
"""Standalone compact IQ1_S experts using the existing direct-row Metal math.

Dense affine weights and ownership machinery are shared with the native provider.
Expert files contain only three immutable packed byte arrays, never decoded or
concatenated weight stacks. See deepseek-v41-standalone-iq1-provider provenance.
"""

from collections import OrderedDict
from contextlib import nullcontext
from copy import deepcopy
import math
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np

from .deepseek_v41_mlx_weights import (
    CHUNK_BYTES, NativeExpertPlan, NativeExpertStore, NativeV41Weights,
    _manifest_binding, _payload_bytes, _refuse_quarantined_uploads,
    _seal, _verify_file, read_native_file,
)
from ..runtime.gguf.reader import GGUFTensor
from ..runtime.io_ownership import ReadPool, storage_owner

SCHEMA = "mlx2.deepseek-v41-iq1.v1"
PROFILE = "native_iq1_s_packed"
MANIFEST = "mlx2-v41-iq1-manifest.json"
REVISION = "mlx2.v41-standalone-iq1-row-exact-r1decode-r4multi-g10-s4.v2"
FIELDS = ("gate", "up", "down")


def _inventory(artifact):
    manifest = artifact["manifest"]
    if (manifest.get("schema") != SCHEMA or manifest.get("numeric_profile") != PROFILE
            or not manifest.get("finished")):
        raise ValueError("compact weights require a completed packed IQ1_S artifact")
    files, tensors = manifest["files"], manifest["tensors"]
    dense, experts, planes = set(), {}, {}
    for name, row in tensors.items():
        storage = row["storage"]
        kind = storage["kind"]
        if kind in ("raw", "affine"):
            dense.add(storage["file"])
        elif kind == "expert-iq1_s":
            if row["ggml_type"] != 19 or len(row["numpy_shape"]) != 3:
                raise ValueError("compact expert plane must declare IQ1_S rank three")
            prefix, suffix = name.split(".ffn_", 1)
            projection = suffix.removesuffix("_exps.weight")
            count, rows, width = row["numpy_shape"]
            paths = tuple(storage["expert_files"])
            if (projection not in FIELDS or len(paths) != count or not 1 <= count <= 384
                    or rows < 1 or width < 1 or rows % 256 or width % 256
                    or storage["keys"] != {"data": projection+".data"}
                    or len(set(paths)) != len(paths)):
                raise ValueError("invalid compact expert projection geometry")
            if prefix in experts and experts[prefix] != paths:
                raise ValueError("compact expert projection inventories disagree")
            experts[prefix] = paths
            planes.setdefault(prefix, {})[projection] = GGUFTensor(
                name, (width, rows, 1), 19, 0, rows*(width//256)*50)
        elif kind != "engram":
            raise ValueError("unsupported compact weight storage")
    if not dense or not experts:
        raise ValueError("compact artifact is missing dense or expert files")
    for prefix, group in planes.items():
        if set(group) != set(FIELDS):
            raise ValueError("compact layer requires all three expert projections")
        gate, up, down = (group[k] for k in FIELDS)
        if up.shape != gate.shape or down.shape != (gate.shape[1], gate.shape[0], 1):
            raise ValueError("compact expert gate/up/down dimensions disagree")
        expected = {k+".data": group[k].nbytes for k in FIELDS}
        for path in experts[prefix]:
            if path not in files or set(files[path]["tensors"]) != set(expected):
                raise ValueError("compact item requires exactly three byte arrays")
            for key, nbytes in expected.items():
                row = files[path]["tensors"][key]
                if (row["dtype"] != "U8" or row["shape"] != [nbytes]
                        or row["data_offsets"][1]-row["data_offsets"][0] != nbytes):
                    raise ValueError("compact expert byte extent differs from IQ1_S geometry")
    paths = dense | {p for layer in experts.values() for p in layer}
    if any(path not in files for path in paths):
        raise ValueError("compact file inventory is incomplete")
    for path in paths:
        _seal(files[path])
    return sorted(dense), experts, planes


def compact_weight_plan(artifact, expert_cache_bytes):
    """CPU-only native-dense/compact-expert admission with active layer leases."""
    dense, experts, _ = _inventory(artifact)
    files = artifact["manifest"]["files"]
    expert_files = {p for paths in experts.values() for p in paths}
    largest = max(_payload_bytes(files[p]) for p in expert_files)
    if type(expert_cache_bytes) is not int or expert_cache_bytes < largest:
        raise ValueError("compact expert cache must hold one complete packed item")
    dense_bytes = sum(_payload_bytes(files[p]) for p in dense)
    prepared = sum(4*math.prod(row["numpy_shape"]) for row in artifact["manifest"]["tensors"].values()
                   if row["storage"]["kind"] == "raw" and row["ggml_type"] != 0)
    dense_staging = max(files[p]["size"] for p in dense)+CHUNK_BYTES
    expert_staging = max(files[p]["size"] for p in expert_files)+CHUNK_BYTES
    active = max(sum(_payload_bytes(files[p]) for p in paths) for paths in experts.values())
    binding, size = _manifest_binding(artifact["manifest"])
    return dict(dense_payload_bytes=dense_bytes, dense_prepared_bound_bytes=prepared,
                dense_staging_bound_bytes=dense_staging, expert_cache_bytes=expert_cache_bytes,
                minimum_expert_cache_bytes=largest, expert_active_bound_bytes=active,
                expert_staging_bound_bytes=expert_staging,
                expert_table_bytes=sum(_payload_bytes(files[p]) for p in expert_files),
                manifest_binding_sha256=binding, manifest_canonical_bytes=size,
                metadata_bound_bytes=max(256 << 20, 16*size),
                total_reservation_bytes=dense_bytes+prepared+dense_staging+expert_cache_bytes+active+expert_staging)


class CompactExpertStore(NativeExpertStore):
    """Reuse the native provider's fence/quarantine/LRU lifecycle for packed bytes."""

    def __init__(self, artifact, cache_bytes, *, read_workers=2, timeout=60.0,
                 _owned_manifest=None, _bounds=None):
        _refuse_quarantined_uploads()
        self.root = Path(artifact["root"]).resolve()
        if _owned_manifest is not None and _owned_manifest is not artifact["manifest"]:
            raise ValueError("compact expert manifest owner differs from weights")
        self.manifest = deepcopy(artifact["manifest"]) if _owned_manifest is None else _owned_manifest
        self.files = self.manifest["files"]
        _, self.tables, self.row_tensors = _inventory({"manifest": self.manifest})
        self.bounds = dict(_bounds) if _bounds is not None else compact_weight_plan({"manifest": self.manifest}, cache_bytes)
        if self.bounds["expert_cache_bytes"] != cache_bytes:
            raise ValueError("compact expert bounds differ from admitted cache")
        if type(read_workers) is not int or not 1 <= read_workers <= 32:
            raise ValueError("invalid compact expert reader bound")
        self.cache_budget_bytes = self.maximum_cache_bytes = cache_bytes
        self.minimum_cache_bytes = self.bounds["minimum_expert_cache_bytes"]
        self._cache, self._leased = OrderedDict(), {}
        self._pending_uploads, self._pending_host = {}, {}
        self._cache_bytes = 0
        self.ple_tables, self._ple_bound = {}, False
        self._active = self._closed = self._parked = False
        self._owner = self._error = self._synchronize = None
        self._lock = threading.RLock()
        self.stats = dict(cache_hits=0, cache_misses=0, bytes_read=0, evictions=0,
                          uploaded_experts=0, upload_batches=0, demand_wait_ns=0,
                          lookups=0, windows=0, parks=0, activations=0)
        self.binding = self.bounds["manifest_binding_sha256"]
        counts = {len(paths) for paths in self.tables.values()}
        if len(counts) != 1:
            raise ValueError("compact expert layers have inconsistent expert counts")
        count = counts.pop()
        self.plan = NativeExpertPlan(layers=len(self.tables), experts_per_layer=count,
            expert_bytes=self.minimum_cache_bytes, table_bytes=self.bounds["expert_table_bytes"],
            capacity_experts=min(count, cache_bytes//(len(self.tables)*self.minimum_cache_bytes)),
            ceiling_bytes=cache_bytes, projections=FIELDS, load_phase="pre_materialization",
            top_k=artifact["args"].num_experts_per_tok,
            transient_bound_bytes=self.bounds["expert_active_bound_bytes"]+self.bounds["expert_staging_bound_bytes"],
            staging_window=1, manifest_sha256=self.binding, table_kinds=("native_IQ1_S",), page_cache="bypass")
        self.pool = ReadPool(min(8, read_workers), "v41-compact-experts", timeout=timeout,
                             max_pending=1, owner=storage_owner("v41-compact-experts", self.root))

    def acquire(self, prefix, expert):
        import mlx.core as mx
        with self._lock:
            self._ready()
            if (not self._active or prefix not in self.tables or type(expert) is not int
                    or not 0 <= expert < len(self.tables[prefix])):
                raise ValueError("compact expert acquire requires an admitted window and expert")
            path = self.tables[prefix][expert]
            record = self.files[path]
            _verify_file(self.root, path, record)
            if path in self._leased:
                return self._leased[path]
            size = _payload_bytes(record)
            leased = sum(sum(v.nbytes for v in item.values()) for item in self._leased.values())
            if leased+size > self.bounds["expert_active_bound_bytes"]:
                raise ValueError("compact expert leases exceed one admitted layer")
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
                self.stats["cache_misses"] += 1
                self.stats["bytes_read"] += record["size"]
                self._pending_host[path] = raw
                arrays = self._pending_uploads[path] = {}
                for key, value in raw.items():
                    if value.dtype != np.uint8 or value.ndim != 1:
                        raise ValueError("compact expert upload requires flat packed U8 bytes")
                    arrays[key] = mx.array(value)
                mx.eval(arrays)
                self._pending_host.pop(path)
                del raw
                self.stats["uploaded_experts"] += 1
                self.stats["upload_batches"] += 1
                if sum(v.nbytes for v in arrays.values()) != size or size > self.cache_budget_bytes:
                    raise ValueError("compact expert exceeds admitted payload bound")
                self._cache[path] = arrays
                self._cache_bytes += size
                while self._cache_bytes > self.cache_budget_bytes:
                    _, victim = self._cache.popitem(last=False)
                    self._cache_bytes -= sum(v.nbytes for v in victim.values())
                    self.stats["evictions"] += 1
            self._leased[path] = arrays
            self._pending_uploads.pop(path, None)
            return arrays

    def gather(self, prefix, ids):
        from ..runtime.gguf.packed_metal import PackedMatrix
        values = np.asarray(ids)
        if (values.dtype != np.uint32 or values.ndim != 2 or not 1 <= values.shape[0] <= 1024
                or not 1 <= values.shape[1] <= 6 or prefix not in self.tables
                or int(values.max()) >= len(self.tables[prefix])):
            raise ValueError("compact gather requires bounded uint32 expert routes")
        unique, inverse = np.unique(values, return_inverse=True)
        items = [self.acquire(prefix, int(expert)) for expert in unique]
        planes = tuple(tuple(PackedMatrix(item[kind+".data"], self.row_tensors[prefix][kind])
                             for item in items) for kind in FIELDS)
        self.stats["lookups"] += 1
        return inverse.reshape(values.shape).astype(np.uint32), planes

    def apc_revision(self):
        return REVISION

    def receipt(self):
        result = super().receipt()
        result.update(mechanism_revision=REVISION, numeric_profile=PROFILE,
                      weight_math="packed IQ1_S, direct-row BF16 store/route-slot law",
                      host_dequantization=False, expert_weight_concatenation=False)
        return result


class CompactV41Weights(NativeV41Weights):
    compute_policy = PROFILE+":"+REVISION

    def __init__(self, artifact, *, expert_cache_bytes, expert_read_workers=2,
                 expert_prefill_policy="reference"):
        _refuse_quarantined_uploads()
        if expert_prefill_policy not in ("reference", "segmented_nax_fp16"):
            raise ValueError("unsupported compact expert prefill profile")
        self.expert_prefill_policy = expert_prefill_policy
        self.artifact = {**artifact, "manifest": deepcopy(artifact["manifest"])}
        self.root = Path(artifact["root"]).resolve()
        self.metadata = deepcopy(artifact["metadata"])
        self.bounds = compact_weight_plan(self.artifact, expert_cache_bytes)
        self.tensors = {name: SimpleNamespace(name=name, **row)
                        for name, row in self.artifact["manifest"]["tensors"].items()}
        self._dense, self._quant, self._packed = {}, {}, {}
        self._pending_dense, self._pending_dense_host = {}, None
        self._loaded = self._parked = self._closed = False
        self._error = None
        self.scheduled_expert_calls, self.scheduled_dense_calls, self.scheduled_grouped_calls = {}, {}, {}
        self.scheduled_expert_buffer_operations = self.expert_buffer_counters = {}
        self._dense_files, _, _ = _inventory(self.artifact)
        self._read_pool = ReadPool(1, "v41-compact-dense", timeout=60.0, max_pending=1,
                                   owner=storage_owner("v41-compact-dense", self.root))
        self.expert_stream = None
        try:
            self.expert_stream = CompactExpertStore(self.artifact, expert_cache_bytes,
                read_workers=expert_read_workers, _owned_manifest=self.artifact["manifest"], _bounds=self.bounds)
        except BaseException:
            self._read_pool.close()
            raise

    def moe(self, x, ids, scores, prefix, *, ticket=None):
        import mlx.core as mx
        from ..runtime.gguf.segmented_prefill import segmented_prefill_active
        self._ready()
        if ticket is not None:
            raise ValueError("compact provider does not accept foreign read tickets")
        if (x.ndim != 2 or x.shape[-1] != self.artifact["args"].hidden_size
                or ids.ndim != 2 or ids.shape[0] != x.shape[0] or scores.shape != ids.shape
                or not 1 <= ids.shape[1] <= self.artifact["args"].num_experts_per_tok):
            raise ValueError("compact MoE activation/routing shape mismatch")
        owned = not self.expert_stream._active
        with self.expert_stream.window() if owned else nullcontext():
            remapped, planes = self.expert_stream.gather(prefix, ids)
            if self.expert_prefill_policy == "segmented_nax_fp16" and segmented_prefill_active(x.shape[0]):
                from ..runtime.gguf.v41_row_prefill import moe
                route = "v41_iq1_segmented_row_buffers_fp16"
                result = moe(x, remapped, scores, *planes, counters=self.expert_buffer_counters, group_experts=12)
            else:
                from ..runtime.gguf.v41_row_exact import moe
                route = "v41_iq1_row_exact"
                result = moe(x, remapped, scores, *planes, reuse=1 if x.shape[0] == 1 else 4, broadcast=False,
                             group_experts=10, simd_groups=4, counters=self.expert_buffer_counters)
            self.scheduled_expert_calls[route] = self.scheduled_expert_calls.get(route, 0)+1
            if owned:
                mx.eval(result)
            return result

    def dense_receipt(self):
        result = super().dense_receipt()
        result.update(numeric_profile=PROFILE, expert_prefill_policy=self.expert_prefill_policy,
                      dense_math="native affine8/group32/F32 metadata and F32 QMM")
        return result

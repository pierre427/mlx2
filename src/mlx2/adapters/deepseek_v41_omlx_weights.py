# SPDX-License-Identifier: MIT
"""Resident affine dense weights and a bounded expert LRU for omlx-layout V4.1.

Dense projections run MLX affine QMM with the checkpoint's per-module bits
and group size. ``quantize_input`` modules first apply the official FP8
activation law (deepseek_v41_quant.quantize), which is the same rule omlx
applies. Routed experts are read one expert at a time as exact slices of the
stacked tensors and follow the Source reference expert law: FP8 input and
middle, BF16-rounded gate/up/middle/down, expert-ID-ordered accumulation.
I/O workers never touch MLX. See provenance/deepseek-v41-omlx.json.
"""
import os
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import numpy as np

from ..runtime.immutable_weight_pages import file_stamp, read_page
from ..runtime.io_ownership import ReadPool, ReadTimeout, storage_owner
from .deepseek_v41_mlx_weights import (
    NativeExpertPlan,
    NativeExpertStore,
    _quarantine_upload,
    _refuse_quarantined_uploads,
    _release_upload,
)
from .deepseek_v41_omlx_artifact import DTYPES, read_tensors
from .deepseek_v41_source_artifact import checked_file

REVISION = "mlx2.v41-omlx-affine-expert-global-lru.v1"
NUMERIC_PROFILE = "omlx_affine_act_fp8_f32_qmm"
CHUNK_BYTES = 16 << 20
_FIELDS = ("weight", "scales", "biases")
_KIND_FIELD = {"weight": "qweight", "scales": "scales", "biases": "biases"}


def layer_bank_bytes(artifact):
    return artifact["inventory"]["expert_item_bytes"]*artifact["args"].n_routed_experts


def omlx_weight_plan(artifact, expert_cache_bytes, bank_layers=()):
    """CPU-only admission bounds; same keys as the native affine planner.

    ``bank_layers`` keep their whole stacked expert tensors resident for the
    device-side gather route; the LRU cache serves only the other layers.
    """
    tensors, args = artifact["tensors"], artifact["args"]
    item = artifact["inventory"]["expert_item_bytes"]
    names = [*artifact["dense"].values()]
    for spec in artifact["quant"].values():
        names += [spec["module"]+"."+field for field in _FIELDS]
    dense_bytes = sum(tensors[name]["size"] for name in names)
    if type(expert_cache_bytes) is not int or expert_cache_bytes < item:
        raise ValueError("omlx V4.1 expert cache must hold one complete expert")
    table = artifact["inventory"]["storage_bytes"]["expert"]
    bank_layers = tuple(sorted(bank_layers))
    if (len(set(bank_layers)) != len(bank_layers)
            or any(type(n) is not int or not 0 <= n < args.num_hidden_layers for n in bank_layers)):
        raise ValueError("omlx expert bank layers outside the model")
    bank = layer_bank_bytes(artifact)*len(bank_layers)
    # Host staging while one bank kind uploads: _load_dense reads its weight,
    # scales and biases in one read_tensors call. Reported in full below.
    bank_staging = max(sum(tensors[spec["module"]+"."+field]["size"] for field in _FIELDS)
                       for spec in artifact["experts"][0].values()) if bank_layers else 0
    # Host staging of the largest dense tensor while it uploads (embed.weight,
    # 1.23 GiB on the pinned checkpoint). Reported, not reserved: see the total.
    dense_staging = max(tensors[name]["size"] for name in names)+CHUNK_BYTES
    expert_staging = item+CHUNK_BYTES
    # The reservation keeps this route's original terms: two read chunks, one
    # expert, and only the packed weight of a bank kind. Load-time staging
    # (dense_staging, full bank_staging) never coexists with a full expert
    # cache, and reserving it as the native planner does (DS41-2, f196622fb)
    # only shrank the LRU cache 11.45 -> 10.08 GiB at the served shape and cost
    # warm decode -8% / 3K prefill -16% with identical output
    # (qualification/runs/ds41-omlx-plan-ab-20261106). Reverted on Pierre's
    # call, 2026-11-06.
    reserved_bank_staging = max(tensors[spec["module"]+".weight"]["size"]
                                for spec in artifact["experts"][0].values()) if bank_layers else 0
    # One streamed layer's experts may be leased in one window.
    active = 0 if len(bank_layers) == args.num_hidden_layers else item*args.n_routed_experts
    return {"dense_payload_bytes": dense_bytes, "dense_prepared_bound_bytes": 0,
            "dense_staging_bound_bytes": dense_staging,
            "expert_cache_bytes": expert_cache_bytes, "minimum_expert_cache_bytes": item,
            "expert_active_bound_bytes": active, "expert_staging_bound_bytes": expert_staging,
            "manifest_binding_sha256": artifact["identity"]["fingerprint"], "manifest_canonical_bytes": 0,
            "metadata_bound_bytes": 256 << 20, "expert_table_bytes": table,
            "expert_bank_layers": list(bank_layers), "expert_bank_bytes": bank,
            "expert_bank_staging_bound_bytes": bank_staging,
            "total_reservation_bytes": (dense_bytes+2*CHUNK_BYTES+expert_cache_bytes+active+item
                                        +bank+reserved_bank_staging)}


def bank_expert_views(bank, experts):
    """Per-expert arrays as as_strided views of a resident stacked bank.

    Unlike slices, evaluated as_strided views share the bank's buffer (0 bytes
    added, checked on GPU 2026-11-04) and quantized_matmul on them is
    bit-equal to the slice result, so the exact per-expert law reads the bank
    in place.
    """
    import mlx.core as mx
    names = ("qweight", "scales", "biases")
    views = []
    for expert in range(experts):
        item = {}
        for kind, arrays in bank.items():
            for name, array in zip(names, arrays):
                rows, cols = array.shape[1:]
                item[kind+"."+name] = mx.as_strided(array, (rows, cols), (cols, 1), expert*rows*cols)
        views.append(item)
    mx.eval(views)
    return views


def expert_extents(artifact, layer, expert):
    """(shard, offset, size, dtype, shape, key) for one expert's nine arrays."""
    result = []
    for kind, spec in artifact["experts"][layer].items():
        for field in _FIELDS:
            row = artifact["tensors"][spec["module"]+"."+field]
            count = row["shape"][0]
            if not 0 <= expert < count or row["size"] % count:
                raise ValueError("omlx expert slice outside its stacked tensor")
            size = row["size"]//count
            result.append((row["file"], row["offset"]+expert*size, size, row["dtype"],
                           row["shape"][1:], kind+"."+_KIND_FIELD[field]))
    return result


def read_expert(root, seals, extents):
    """Bounded sealed reads on an I/O worker; returns host arrays only."""
    by_shard = {}
    for extent in extents:
        by_shard.setdefault(extent[0], []).append(extent)
    result = {}
    for shard, rows in by_shard.items():
        seal = seals[shard]
        path = checked_file(root, shard, seal)
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            if file_stamp(os.fstat(fd)) != seal:
                raise ValueError("omlx expert shard changed before read")
            if os.uname().sysname == "Darwin":
                import fcntl
                fcntl.fcntl(fd, 48, 1)  # F_NOCACHE
                fcntl.fcntl(fd, 45, 0)  # F_RDAHEAD
            for _, offset, size, dtype, shape, key in rows:
                raw = read_page(fd, offset, size, CHUNK_BYTES)
                result[key] = raw.view(np.dtype(DTYPES[dtype])).reshape(shape)
            if file_stamp(os.fstat(fd)) != seal:
                raise ValueError("omlx expert shard changed during read")
        finally:
            os.close(fd)
    return result


class OmlxExpertStore(NativeExpertStore):
    """Global byte LRU over (layer, expert) slices of stacked expert tensors."""

    def __init__(self, artifact, cache_bytes, *, bounds, read_workers=2, timeout=60.0):
        _refuse_quarantined_uploads()
        args = artifact["args"]
        self.artifact = artifact
        self.root = artifact["root"]
        self.seals = artifact["seals"]
        self.bounds = dict(bounds)
        if self.bounds["expert_cache_bytes"] != cache_bytes:
            raise ValueError("omlx expert bounds differ from admitted cache")
        if type(read_workers) is not int or not 1 <= read_workers <= 32:
            raise ValueError("invalid omlx expert read worker bound")
        item = self.bounds["minimum_expert_cache_bytes"]
        self.tables = {f"blk.{layer}": [(layer, e) for e in range(args.n_routed_experts)]
                       for layer in range(args.num_hidden_layers)}
        self.files = {key: {"size": item} for keys in self.tables.values() for key in keys}
        self.cache_budget_bytes = self.maximum_cache_bytes = cache_bytes
        self.minimum_cache_bytes = item
        self._cache, self._leased = OrderedDict(), {}
        self._pending_uploads, self._pending_host = {}, {}
        self._cache_bytes = 0
        self.ple_tables, self._ple_bound = {}, False
        self._active = self._closed = self._parked = False
        self._owner = None
        self._error = None
        self._lock = threading.RLock()
        self._synchronize = None
        self.stats = dict.fromkeys(("cache_hits", "cache_misses", "bytes_read", "evictions",
                                    "uploaded_experts", "upload_batches", "demand_wait_ns",
                                    "lookups", "windows", "parks", "activations"), 0)
        self.binding = self.bounds["manifest_binding_sha256"]
        self.plan = NativeExpertPlan(layers=args.num_hidden_layers, experts_per_layer=args.n_routed_experts,
            expert_bytes=item, table_bytes=self.bounds["expert_table_bytes"],
            capacity_experts=min(args.n_routed_experts, cache_bytes//(args.num_hidden_layers*item)),
            ceiling_bytes=cache_bytes, projections=("gate", "up", "down"),
            load_phase="pre_materialization", top_k=args.num_experts_per_tok,
            transient_bound_bytes=self.bounds["expert_active_bound_bytes"]+self.bounds["expert_staging_bound_bytes"],
            staging_window=1, manifest_sha256=self.binding,
            table_kinds=("omlx_affine_stacked_slice",), page_cache="bypass")
        self.pool = ReadPool(min(8, read_workers), "v41-omlx-experts", timeout=timeout, max_pending=1,
                             owner=storage_owner("v41-omlx-experts", self.root))

    def acquire(self, prefix, expert, *, admit=True):
        """Lease one expert; ``admit=False`` never evicts a cached expert for it.

        A full cache then serves the miss for this window only, so a prefill
        sweep over more experts than fit cannot churn the resident set (LRU's
        worst case on a cyclic scan).
        """
        import mlx.core as mx
        with self._lock:
            self._ready()
            if not self._active or prefix not in self.tables or type(expert) is not int:
                raise ValueError("omlx expert acquire requires an admitted execution window")
            if not 0 <= expert < len(self.tables[prefix]):
                raise ValueError("omlx expert index outside admitted layer")
            key = self.tables[prefix][expert]
            if key in self._leased:
                return self._leased[key]
            if (len(self._leased)+1)*self.minimum_cache_bytes > self.bounds["expert_active_bound_bytes"]:
                raise ValueError("omlx expert leases exceed one admitted layer")
            if key in self._cache:
                arrays = self._cache[key]
                self._cache.move_to_end(key)
                self.stats["cache_hits"] += 1
            else:
                started = time.perf_counter_ns()
                try:
                    raw = self.pool.call(read_expert, self.root, self.seals,
                                         expert_extents(self.artifact, *key))
                finally:
                    self.stats["demand_wait_ns"] += time.perf_counter_ns()-started
                self._pending_host[key] = raw
                arrays = self._pending_uploads[key] = {}
                for name, value in raw.items():
                    array = mx.array(value)
                    arrays[name] = array.view(mx.bfloat16) if value.dtype == np.uint16 else array
                mx.eval(arrays)
                self._pending_host.pop(key)
                del raw
                self.stats["cache_misses"] += 1
                self.stats["bytes_read"] += self.minimum_cache_bytes
                self.stats["uploaded_experts"] += 1
                self.stats["upload_batches"] += 1
                size = sum(value.nbytes for value in arrays.values())
                if size != self.minimum_cache_bytes or size > self.cache_budget_bytes:
                    raise ValueError("omlx expert payload differs from its admitted size")
                if not admit and self._cache_bytes + size > self.cache_budget_bytes:
                    self.stats["scan_bypass"] = self.stats.get("scan_bypass", 0) + 1
                    self._leased[key] = arrays
                    self._pending_uploads.pop(key, None)
                    return arrays
                self._cache[key] = arrays
                self._cache_bytes += size
                while self._cache_bytes > self.cache_budget_bytes:
                    victim = next((k for k in self._cache if k not in self._leased and k != key), None)
                    if victim is None:
                        break
                    self._cache_bytes -= sum(v.nbytes for v in self._cache.pop(victim).values())
                    self.stats["evictions"] += 1
            self._leased[key] = arrays
            self._pending_uploads.pop(key, None)
            return arrays

    # False under layer_eval=async_v1: layers end without a device fence, so
    # the CPU builds layer n+1 while the GPU runs layer n. Leases are still
    # released per layer; MLX keeps every array a pending graph reads alive,
    # so an LRU eviction cannot free a buffer still in use.
    layer_fence = True

    @contextmanager
    def window(self):
        if self.layer_fence:
            with NativeExpertStore.window(self):
                yield
            return
        with self._lock:
            self._ready()
            if self._active:
                raise ValueError("omlx expert execution window already active")
            self._active = True
            self.stats["windows"] += 1
        try:
            yield
        except BaseException as error:
            self._error = str(error)
            try:
                self._fence()
            except BaseException:  # noqa: BLE001 - keep the original failure; quarantine the owner
                _quarantine_upload(self)
            raise
        finally:
            with self._lock:
                self._leased.clear()
                self._pending_uploads.clear()
                self._pending_host.clear()
                self._active = False

    def apc_revision(self):
        return REVISION

    def receipt(self):
        result = super().receipt()
        result.update(mechanism_revision=REVISION, numeric_profile=NUMERIC_PROFILE,
                      source_read_byte_scope="exact per-expert slices of stacked expert tensors")
        return result


class OmlxV41Weights:
    compute_policy = NUMERIC_PROFILE+":"+REVISION
    dense_prefill_policy = expert_prefill_policy = "reference"
    expert_io_policy = "blocking"
    shared_policy = "reference"
    dense_pages = None

    def __init__(self, artifact, *, expert_cache_bytes, expert_read_workers=2, bounds=None,
                 bf16_loads_policy="direct_f32", bank_layers=(), expert_math="reference",
                 dense_qat="reference", layer_eval="sync", woa_storage="reference", bank_decode="gather_qmm",
                 prefill_admission="lru"):
        _refuse_quarantined_uploads()
        if (bf16_loads_policy not in ("reference", "direct_f32") or expert_math not in ("reference", "hoisted_v1")
                or dense_qat not in ("reference", "fused_fp32") or layer_eval not in ("sync", "async_v1")
                or woa_storage not in ("reference", "mxfp8_gemv_v1")
                or bank_decode not in ("gather_qmm", "routed_v1")
                or prefill_admission not in ("lru", "scan_resistant_v1")):
            raise ValueError("unsupported omlx weights policy")
        # hoisted_v1: one FP8 input quantize per layer and one SwiGLU/middle
        # quantize over all selected experts' rows; per-row arithmetic is
        # unchanged, so it is bit-exact with the reference loop.
        self.expert_math, self.dense_qat, self.layer_eval = expert_math, dense_qat, layer_eval
        # Resident bank layers: "routed_v1" runs one-row decode through the
        # device-routed kernels and every other width through the per-expert
        # path on bank slices; "gather_qmm" is the stock gather (opt-in, slower).
        self.bank_decode = bank_decode
        # scan_resistant_v1: multi-row (prefill) misses never evict from a full
        # expert cache; one-row decode admits as before.
        self.prefill_admission = prefill_admission
        # mxfp8_gemv_v1: the checkpoint's BF16 wo_a lies on the MXFP8 grid
        # (verify_report wo_a_mxfp8_grid); keep exact MXFP8 codes instead,
        # proven by a bitwise round trip at load, and read them in registers.
        self.woa_storage = woa_storage
        self.woa_storage_receipt = {}
        self._last_input = None  # (source array, prepared) for consecutive same-input projections
        self.artifact, self.root, self.metadata = artifact, artifact["root"], {}
        self.bank_layers = tuple(sorted(bank_layers))
        self.bounds = (omlx_weight_plan(artifact, expert_cache_bytes, self.bank_layers)
                       if bounds is None else dict(bounds))
        if list(self.bank_layers) != list(self.bounds.get("expert_bank_layers", [])):
            raise ValueError("omlx expert bank layers differ from admission")
        self._banks, self._bank_views = {}, {}
        self.bf16_loads_policy = bf16_loads_policy
        self.tensors = {}
        for alias, name in artifact["aliases"].items():
            spec = artifact["quant"].get(alias)
            shape = spec["logical"] if spec else artifact["tensors"][name]["shape"]
            self.tensors[alias] = SimpleNamespace(name=alias, numpy_shape=tuple(shape), shape=tuple(shape)[::-1],
                                                  ggml_type=0, storage={"kind": "omlx", "key": alias})
        self._dense, self._quant = {}, {}
        self._pending_dense, self._pending_dense_host = {}, None
        self._loaded = self._parked = self._closed = False
        self._error = None
        self.scheduled_expert_calls, self.scheduled_dense_calls, self.scheduled_grouped_calls = {}, {}, {}
        self.scheduled_expert_buffer_operations = self.expert_buffer_counters = {}
        self._read_pool = ReadPool(1, "v41-omlx-dense", timeout=60, max_pending=1,
                                   owner=storage_owner("v41-omlx-dense", self.root))
        self.expert_stream = None
        try:
            self.expert_stream = OmlxExpertStore(artifact, expert_cache_bytes, bounds=self.bounds,
                                                 read_workers=expert_read_workers)
            self.expert_stream.layer_fence = layer_eval == "sync"
        except BaseException:
            self._read_pool.close()
            raise

    def _count(self, table, key):
        table[key] = table.get(key, 0)+1

    def _ready(self):
        if self._parked or self._closed or self._error:
            raise ValueError("omlx V4.1 weights are parked, closed or failed")

    def _ensure_dense(self):
        self._ready()
        if not self._loaded:
            self._load_dense()

    def _load_dense(self):
        import mlx.core as mx

        def device(value):
            array = mx.array(value)
            return array.view(mx.bfloat16) if value.dtype == np.uint16 else array
        try:
            for alias, name in self.artifact["dense"].items():
                raw = self._read_pool.call(read_tensors, self.artifact, (name,))
                self._pending_dense_host = raw
                self._pending_dense[alias] = device(raw[name])
                mx.eval(self._pending_dense)
                self._dense.update(self._pending_dense)
                self._pending_dense.clear()
                self._pending_dense_host = None
                if self.woa_storage == "mxfp8_gemv_v1" and alias.endswith("attn_output_a.weight"):
                    self._encode_woa(alias)
            for alias, spec in self.artifact["quant"].items():
                names = tuple(spec["module"]+"."+field for field in _FIELDS)
                raw = self._read_pool.call(read_tensors, self.artifact, names)
                self._pending_dense_host = raw
                arrays = tuple(device(raw[name]) for name in names)
                mx.eval(arrays)
                self._quant[alias] = arrays
                self._pending_dense_host = None
            for layer in self.bank_layers:
                bank = {}
                for kind, spec in self.artifact["experts"][layer].items():
                    names = tuple(spec["module"]+"."+field for field in _FIELDS)
                    raw = self._read_pool.call(read_tensors, self.artifact, names)
                    self._pending_dense_host = raw
                    bank[kind] = tuple(device(raw[name]) for name in names)
                    mx.eval(bank[kind])
                    self._pending_dense_host = None
                    del raw
                self._banks[layer] = bank
                self._bank_views[layer] = bank_expert_views(bank, self.artifact["args"].n_routed_experts)
            self._loaded = True
        except BaseException as error:
            self._error = str(error)
            try:
                mx.synchronize()
            except BaseException:
                _quarantine_upload(self)
                raise
            self._dense.clear()
            self._quant.clear()
            self._banks.clear()
            self._bank_views.clear()
            self._pending_dense.clear()
            self._pending_dense_host = None
            raise

    def _encode_woa(self, alias):
        """Replace one BF16 wo_a with MXFP8 codes; refuse unless the round trip is bitwise."""
        import mlx.core as mx

        from ..runtime.gguf.source_dense_mxfp8 import validate_scale_bytes
        weight = self._dense[alias]
        q, scales = mx.quantize(weight, mode="mxfp8")[:2]
        expanded = mx.dequantize(q, scales=scales, mode="mxfp8", dtype=mx.bfloat16)
        exact = mx.array_equal(expanded.view(mx.uint16), weight.view(mx.uint16)).item()
        if not exact:
            raise ValueError("omlx wo_a is not on the MXFP8 grid; select deepseek_source_woa_storage=reference")
        validate_scale_bytes(np.array(scales))
        mx.eval(q, scales)
        self._dense[alias+".q"], self._dense[alias+".s"] = q, scales
        del self._dense[alias]
        self.woa_storage_receipt[alias] = {"exact_round_trip": True, "codes_bytes": q.nbytes + scales.nbytes,
                                           "bf16_bytes": weight.nbytes}

    def verify_source(self):
        for name, seal in self.artifact["seals"].items():
            checked_file(self.root, name, seal)
        return True

    def array(self, name):
        self._ensure_dense()
        if name not in self._dense:
            raise ValueError("omlx array access cannot expand a quantized projection: " + name)
        return self._dense[name]

    def packed(self, name):
        raise ValueError("omlx V4.1 has no packed-GGUF tensors")

    def can_fuse_dense(self, x, *names):
        return False

    def _fp8(self, x, table, prefix):
        """FP8 activation law; the fused kernel self-checks against the reference."""
        import mlx.core as mx

        from ..runtime.models.deepseek_v41_quant import fused_quantize, quantize
        value = x.astype(mx.float32)
        if self.dense_qat == "fused_fp32" and mx.default_device() == mx.gpu and value.shape[-1] % 32 == 0:
            self._count(table, prefix+"_fp8_fused")
            return fused_quantize(value, "fp8").astype(mx.float32)
        self._count(table, prefix+"_fp8_reference")
        return quantize(value, "fp8").astype(mx.float32)

    def _input(self, x, spec):
        import mlx.core as mx
        if not spec["quantize_input"]:
            return x.astype(mx.float32)
        last = self._last_input
        if last is not None and last[0] is x:
            # Same array object as the previous projection's input (wq_a/wkv,
            # shared gate/up): its quantized operand is identical.
            self._count(self.scheduled_dense_calls, "omlx_input_fp8_reused")
            return last[1]
        prepared = self._fp8(x, self.scheduled_dense_calls, "omlx_input")
        self._last_input = (x, prepared)
        return prepared

    def linear(self, name):
        self._ensure_dense()
        import mlx.core as mx
        spec = self.artifact["quant"].get(name)

        def apply(x):
            self._ready()
            if spec is not None:
                q, s, b = self._quant[name]
                self._count(self.scheduled_dense_calls, f"omlx_affine{spec['bits']}_g{spec['group_size']}_f32")
                return mx.quantized_matmul(self._input(x, spec), q, s, b, group_size=spec["group_size"],
                                           bits=spec["bits"]).astype(mx.float32)
            weight = self.array(name)
            if (name == "output.weight" and self.bf16_loads_policy == "direct_f32"
                    and x.ndim == 3 and x.shape[1] == 1):
                from ..runtime.gguf.source_dense import maybe_project
                value = maybe_project(x.astype(mx.float32), weight, "head")
                if value is not None:
                    self._count(self.scheduled_dense_calls, "omlx_bf16_head_direct_f32")
                    return value
            return x.astype(mx.float32) @ weight.astype(mx.float32).T
        return apply

    def gate_up(self, x, gate_name, up_name, **_):
        return self.linear(gate_name)(x), self.linear(up_name)(x)

    def grouped(self, x, name, *, groups=8):
        import mlx.core as mx
        self._ensure_dense()
        shape = tuple(self.tensors[name].numpy_shape)
        if (name in self.artifact["quant"] or len(shape) != 2 or shape[0] % groups
                or x.ndim != 4 or x.shape[2:] != (groups, shape[1])):
            raise ValueError("omlx grouped projection requires its admitted BF16 matrix")
        if name+".q" in self._dense:
            from ..runtime.gguf import source_dense_mxfp8
            q, scales = self._dense[name+".q"], self._dense[name+".s"]
            if x.shape[1] == 1:
                value = source_dense_mxfp8.maybe_project(x.astype(mx.float32), q, scales)
                if value is not None:
                    self._count(self.scheduled_grouped_calls, "omlx_mxfp8_grouped_gemv_f32")
                    return value
            self._count(self.scheduled_grouped_calls, "omlx_mxfp8_grouped_bf16_expansion")
            weight = source_dense_mxfp8.expand_bf16(q, scales)
            return mx.einsum("btgi,goi->btgo", x.astype(mx.float32),
                             weight.astype(mx.float32).reshape(groups, shape[0]//groups, shape[1]))
        weight = self.array(name)
        if self.bf16_loads_policy == "direct_f32" and x.shape[1] == 1:
            from ..runtime.gguf.source_dense import maybe_project
            value = maybe_project(x.astype(mx.float32), weight, "wo_a")
            if value is not None:
                self._count(self.scheduled_grouped_calls, "omlx_bf16_grouped_direct_f32")
                return value
        self._count(self.scheduled_grouped_calls, "omlx_bf16_grouped_f32")
        return mx.einsum("btgi,goi->btgo", x.astype(mx.float32),
                         weight.astype(mx.float32).reshape(groups, shape[0]//groups, shape[1]))

    def prepare_embedding(self, name="token_embd.weight"):
        return self.array(name)

    def embedding(self, ids, name="token_embd.weight"):
        import mlx.core as mx
        return self.array(name)[ids].astype(mx.float32)

    def _bank_slice(self, prefix, expert):
        """One expert's arrays: load-time as_strided views of its resident bank."""
        layer = int(prefix.split(".")[1])
        views = self._bank_views.get(layer)
        if views is not None:
            return views[expert]
        bank = self._banks[layer]
        names = ("qweight", "scales", "biases")
        return {kind+"."+name: bank[kind][i][expert] for kind in bank for i, name in enumerate(names)}

    def _routed_supported(self, layer):
        from ..runtime.models import deepseek_v41_routed_decode as routed
        args = self.artifact["args"]
        return routed.supports(args.hidden_size, args.moe_intermediate_size, args.num_experts_per_tok,
                               self.artifact["experts"][layer])

    def _routed_moe(self, x, ids, scores, layer):
        """One decode row over a resident bank; route IDs stay on the GPU.

        Per-row arithmetic of the per-expert path (MLX qmv_fast / qmv in the
        routed kernels, then the same BF16 rounds, clamped SwiGLU, FP8 middle
        and expert-ID-ordered F32 sum).
        """
        import mlx.core as mx

        from ..runtime.models import deepseek_v41_routed_decode as routed
        from ..runtime.models.deepseek_v4 import limited_swiglu
        args = self.artifact["args"]
        top = args.num_experts_per_tok
        if tuple(ids.shape) != (1, top) or tuple(scores.shape) != (1, top) or tuple(x.shape) != (1, args.hidden_size):
            raise ValueError("omlx routed decode requires one admitted route row")
        specs, bank = self.artifact["experts"][layer], self._banks[layer]
        rounded = lambda v: v.astype(mx.bfloat16).astype(mx.float32)

        def prepared(value, kind):
            if not specs[kind]["quantize_input"]:
                return value.astype(mx.float32)
            return self._fp8(value, self.scheduled_expert_calls, "omlx_expert")
        index = ids.reshape(top).astype(mx.uint32)
        gate, up = routed.gate_up(prepared(x, "gate"), index, bank["gate"], bank["up"], specs["gate"])
        middle = rounded(limited_swiglu(rounded(gate), rounded(up))*scores.astype(mx.float32).reshape(top, 1))
        down = rounded(routed.down(prepared(middle, "down"), index, bank["down"], specs["down"]))
        ordered = down[mx.argsort(index)]
        output = mx.zeros((1, args.hidden_size), mx.float32)
        for slot in range(top):
            output = output + ordered[slot:slot+1]
        self._count(self.scheduled_expert_calls, "omlx_affine_experts_bank_routed_decode")
        return output

    def _bank_moe(self, x, ids, scores, layer):
        """Device-side routed experts over one resident stacked bank.

        Same law as the per-expert path below: FP8 input (per row, so one
        quantize for all selected experts is exact) and middle, BF16-rounded
        stages, expert-ID-ordered sum. Route IDs never leave the GPU.
        """
        import mlx.core as mx

        from ..runtime.models.deepseek_v4 import limited_swiglu
        from ..runtime.models.deepseek_v41_quant import quantize
        args = self.artifact["args"]
        if (len(ids.shape) != 2 or ids.shape[1] != args.num_experts_per_tok or not 0 < ids.shape[0] <= 1024
                or tuple(scores.shape) != tuple(ids.shape) or tuple(x.shape) != (ids.shape[0], args.hidden_size)):
            raise ValueError("omlx expert route geometry differs from admitted execution")
        rows, top = ids.shape
        specs, bank = self.artifact["experts"][layer], self._banks[layer]
        index = ids.astype(mx.uint32)
        rounded = lambda v: v.astype(mx.bfloat16).astype(mx.float32)

        def prepared(value, kind):
            return (quantize(value, "fp8") if specs[kind]["quantize_input"] else value).astype(mx.float32)

        def project(value, kind):
            q, s, b = bank[kind]
            return mx.gather_qmm(value, q, s, b, rhs_indices=index, transpose=True,
                                 group_size=specs[kind]["group_size"], bits=specs[kind]["bits"]).astype(mx.float32)
        selected = prepared(x, "gate").reshape(rows, 1, 1, args.hidden_size)
        gate, up = rounded(project(selected, "gate")), rounded(project(selected, "up"))
        middle = rounded(limited_swiglu(gate, up)*scores.astype(mx.float32).reshape(rows, top, 1, 1))
        down = rounded(project(prepared(middle, "down"), "down")).reshape(rows, top, args.hidden_size)
        ordered = mx.take_along_axis(down, mx.argsort(index, axis=1)[:, :, None], axis=1)
        output = ordered[:, 0]
        for slot in range(1, top):
            output = output + ordered[:, slot]
        self._count(self.scheduled_expert_calls, "omlx_affine_experts_bank_gather_qmm")
        return output

    def moe(self, x, ids, scores, prefix, *, ticket=None, **_):
        import mlx.core as mx

        from ..runtime.models.deepseek_v4 import limited_swiglu
        from ..runtime.models.deepseek_v41_quant import quantize
        self._ready()
        if ticket is not None:
            raise ValueError("omlx expert provider does not accept early-load tickets")
        bank_layer = int(prefix.split(".")[1]) if prefix.startswith("blk.") else None
        acquire = self.expert_stream.acquire
        if self.prefill_admission == "scan_resistant_v1" and len(ids.shape) == 2 and ids.shape[0] > 1:
            stream = self.expert_stream
            acquire = lambda prefix, expert: stream.acquire(prefix, expert, admit=False)
        if bank_layer in self._banks:
            if self.bank_decode == "gather_qmm":
                return self._bank_moe(x, ids, scores, bank_layer)
            if len(ids.shape) == 2 and ids.shape[0] == 1 and self._routed_supported(bank_layer):
                return self._routed_moe(x, ids, scores, bank_layer)
            acquire = self._bank_slice
        args = self.artifact["args"]
        values = np.asarray(ids)
        if (values.dtype != np.uint32 or values.ndim != 2 or not 0 < values.shape[0] <= 1024
                or values.shape[1] != args.num_experts_per_tok or scores.shape != values.shape
                or x.shape != (values.shape[0], args.hidden_size)):
            raise ValueError("omlx expert route geometry differs from admitted execution")
        flat = values.reshape(-1)
        order = np.argsort(flat, kind="stable")
        sorted_ids = flat[order]
        if prefix not in self.expert_stream.tables or int(sorted_ids[-1]) >= args.n_routed_experts:
            raise ValueError("omlx expert routes outside admitted inventory")
        layer = int(prefix.split(".")[1])
        specs = self.artifact["experts"][layer]
        owned = not self.expert_stream._active
        with self.expert_stream.window() if owned else nullcontext():
            parts = []
            rounded = lambda v: v.astype(mx.bfloat16).astype(mx.float32)

            def project(value, data, kind):
                spec = specs[kind]
                return mx.quantized_matmul(value, data[kind+".qweight"], data[kind+".scales"],
                                           data[kind+".biases"], group_size=spec["group_size"],
                                           bits=spec["bits"]).astype(mx.float32)

            def prepared(value, kind):
                if not specs[kind]["quantize_input"]:
                    return value.astype(mx.float32)
                if self.expert_math == "hoisted_v1":
                    return self._fp8(value, self.scheduled_expert_calls, "omlx_expert")
                return quantize(value, "fp8").astype(mx.float32)
            if self.expert_math == "hoisted_v1":
                return self._hoisted_moe(x, scores, values, order, sorted_ids, prefix, specs, owned,
                                         project, prepared, rounded, limited_swiglu, acquire)
            for expert in np.unique(sorted_ids):
                slots = order[sorted_ids == expert]
                rows = mx.array(slots//values.shape[1], mx.uint32)
                data = acquire(prefix, int(expert))
                selected = prepared(x[rows], "gate")
                gate, up = rounded(project(selected, data, "gate")), rounded(project(selected, data, "up"))
                middle = rounded(limited_swiglu(gate, up)*scores.reshape(-1)[mx.array(slots, mx.uint32), None])
                parts.append(rounded(project(prepared(middle, "down"), data, "down")))
            sorted_output = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)
            restored = sorted_output[mx.array(np.argsort(order), mx.uint32)].reshape(*values.shape, x.shape[-1])
            output = mx.zeros(x.shape, mx.float32)
            # Same accumulation order as the Source reference: expert-ID order.
            ranking = mx.array(np.argsort(values, axis=1, kind="stable"), mx.uint32)
            for slot in range(values.shape[1]):
                output = output + mx.take_along_axis(restored, ranking[:, slot:slot+1, None], axis=1)[:, 0]
            self._count(self.scheduled_expert_calls, "omlx_affine_experts_act_fp8_per_expert")
            self.expert_stream.stats["lookups"] += 1
            if owned:
                mx.eval(output)
            return output

    def _hoisted_moe(self, x, scores, values, order, sorted_ids, prefix, specs, owned,
                     project, prepared, rounded, limited_swiglu, acquire):
        """Reference law with whole-layer pointwise stages (one launch each)."""
        import mlx.core as mx
        experts, counts = np.unique(sorted_ids, return_counts=True)
        rows = mx.array(order//values.shape[1], mx.uint32)  # expert-sorted route slots
        selected = prepared(x, "gate")[rows]
        data, gates, ups, start = [], [], [], 0
        for expert, count in zip(experts, counts):
            data.append(acquire(prefix, int(expert)))
            part = selected[start:start+count]
            gates.append(project(part, data[-1], "gate"))
            ups.append(project(part, data[-1], "up"))
            start += count
        concat = lambda items: items[0] if len(items) == 1 else mx.concatenate(items, axis=0)
        gate, up = rounded(concat(gates)), rounded(concat(ups))
        middle = prepared(rounded(limited_swiglu(gate, up)*scores.reshape(-1)[mx.array(order, mx.uint32), None]),
                          "down")
        parts, start = [], 0
        for item, count in zip(data, counts):
            parts.append(project(middle[start:start+count], item, "down"))
            start += count
        sorted_output = rounded(concat(parts))
        restored = sorted_output[mx.array(np.argsort(order), mx.uint32)].reshape(*values.shape, x.shape[-1])
        output = mx.zeros(x.shape, mx.float32)
        ranking = mx.array(np.argsort(values, axis=1, kind="stable"), mx.uint32)
        for slot in range(values.shape[1]):
            output = output + mx.take_along_axis(restored, ranking[:, slot:slot+1, None], axis=1)[:, 0]
        self._count(self.scheduled_expert_calls, "omlx_affine_experts_hoisted_v1")
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
        if self.layer_eval == "async_v1":
            mx.async_eval(list(arrays(values)))
        else:
            mx.eval(list(arrays(values)))

    def preload_native_linear_fp32(self):
        self._ready()

    def native_fp32_parameters(self):
        return {}

    def resident_bytes(self):
        unique = {id(v): v for v in (*self._dense.values(), *self._pending_dense.values(),
                                     *(a for t in self._quant.values() for a in t),
                                     *(a for bank in self._banks.values() for t in bank.values() for a in t))}
        return sum(v.nbytes for v in unique.values())

    def resume_reservation_bytes(self):
        return (self.bounds["dense_payload_bytes"]+self.bounds["dense_staging_bound_bytes"]
                + self.bounds.get("expert_bank_bytes", 0))

    def reserved_bytes(self):
        return 0 if self._parked and not self.resident_bytes() else self.resume_reservation_bytes()

    def dense_receipt(self):
        return {"mode": "omlx_affine_resident", "numeric_profile": NUMERIC_PROFILE,
                "retained_payload_bytes": self.resident_bytes(), "reserved_bytes": self.reserved_bytes(),
                "pending_upload_arrays": len(self._pending_dense), "bf16_loads": self.bf16_loads_policy,
                "expert_bank_layers": list(self.bank_layers), "woa_storage": self.woa_storage,
                "woa_storage_receipt_layers": len(self.woa_storage_receipt), "parked": self._parked, "qualification": "unqualified"}

    def park(self):
        if self._parked:
            return self.dense_receipt()
        self.expert_stream.park()
        if self._read_pool.counters()["weight_read_inflight"]:
            raise ReadTimeout("omlx dense reads prevent parking")
        self._dense.clear()
        self._quant.clear()
        self._banks.clear()
        self._bank_views.clear()
        self._pending_dense.clear()
        self._pending_dense_host = None
        self._loaded = False
        self._parked = True
        return self.dense_receipt()

    def activate(self):
        if self._closed or self._error or self._read_pool.failure:
            raise ValueError("failed omlx V4.1 weights cannot reactivate")
        if self._parked:
            self._parked = False
            self._load_dense()
            self.expert_stream.activate()
        return self.dense_receipt()

    def close(self):
        first = None
        try:
            if self.expert_stream is not None:
                self.expert_stream.close()
        except BaseException as error:  # noqa: BLE001 - retain every uncertain owner
            first = error
        self._read_pool.close()
        if first is None and self._read_pool.counters()["weight_read_inflight"]:
            first = ReadTimeout("omlx dense close retains a late owned read")
        if first is None:
            self._dense.clear()
            self._quant.clear()
            # Resident expert banks (up to 32 layers, ~2 GiB each) are owned
            # here as well; park() already drops them, close() must too.
            self._banks.clear()
            self._bank_views.clear()
            self._pending_dense.clear()
            self._pending_dense_host = None
            self._loaded = False
            self._closed = self._parked = True
            _release_upload(self)
        if first is not None:
            _quarantine_upload(self)
            raise first

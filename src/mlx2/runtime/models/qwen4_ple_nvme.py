# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; see docs/PROVENANCE.md and provenance/flashnext.json.
from __future__ import annotations
import fnmatch
import hashlib
import json
import os
import threading
import time
import warnings
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from .. import round_levers as _lv

MANIFEST_FORMAT = "qwen4-ple-rows"
MANIFEST_VERSION = 1
PREFILL_ID_THRESHOLD = 512
DECODE_WORKERS = 16
PREFILL_WORKERS = 64
PREFETCH_WORKERS = 16
_FORK_RESOURCE_LOCK = threading.RLock()


@dataclass(frozen=True)
class LookupStats:
    """Foreground lookup counters (prefetch reads are not counted).

    ``bytes_read`` counts disk preads only; an LRU hit reads no bytes.
    ``cache_evictions`` includes evictions caused by prefetch inserts.
    """

    lookups: int
    rows: int
    unique_rows: int
    bytes_read: int
    elapsed_seconds: float
    cache_hits: int
    cache_misses: int
    cache_evictions: int


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    """View uint16 bfloat16 bits as float32 values."""
    return (bits.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    """Round float32 to bfloat16 bits with round-to-nearest-even."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(32767) + (bits >> 16 & 1)
    return (rounded >> 16).astype(np.uint16)


def dequant_rows_numpy(row_bytes: np.ndarray, dims: int) -> np.ndarray:
    """Dequantize packed q4/g32 rows to bfloat16 bits.

    Args:
        row_bytes: uint8 array of shape ``[n, row_bytes]`` in sidecar layout.
        dims: row width in elements (must be a multiple of 32).

    Returns:
        uint16 array of shape ``[n, dims]`` holding bfloat16 bits.
    """
    n = row_bytes.shape[0]
    weight_bytes = dims // 2
    group_bytes = dims // 32 * 2
    words = np.ascontiguousarray(row_bytes[:, :weight_bytes]).view(np.uint32)
    scales = np.ascontiguousarray(
        row_bytes[:, weight_bytes : weight_bytes + group_bytes]
    ).view(np.uint16)
    biases = np.ascontiguousarray(
        row_bytes[:, weight_bytes + group_bytes : weight_bytes + 2 * group_bytes]
    ).view(np.uint16)
    shifts = np.uint32(4) * np.arange(8, dtype=np.uint32)
    q = (words[..., None] >> shifts & np.uint32(15)).astype(np.float32)
    q = q.reshape(n, dims)
    s = np.repeat(bf16_bits_to_f32(scales), 32, axis=1)
    b = np.repeat(bf16_bits_to_f32(biases), 32, axis=1)
    return f32_to_bf16_bits(q * s + b)


def _read_safetensors_header(path):
    """Return (tensor header dict, data section offset) for a safetensors file."""
    import struct

    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    return (header, 8 + header_len)


def _source_shard_refs(model_path, manifest):
    """Map shard index -> {part: (file, start, row_nbytes)} for the source tensors."""
    model_path = Path(model_path)
    with open(model_path / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]
    prefix = manifest["tensor_prefix"]
    files = sorted(
        {v for (k, v) in weight_map.items() if k.startswith(prefix + ".shard_")}
    )
    refs = {}
    for file_name in files:
        (header, data_offset) = _read_safetensors_header(model_path / file_name)
        file_size = os.path.getsize(model_path / file_name)
        for name, info in header.items():
            if not name.startswith(prefix + ".shard_"):
                continue
            shard_index = int(name.split(".shard_")[1].split(".")[0])
            part = name.rsplit(".", 1)[1]
            (start, end) = info["data_offsets"]
            rows = info["shape"][0]
            if rows <= 0 or end <= start:
                raise ValueError(f"{name} has an empty tensor in {file_name}")
            if start < 0 or data_offset + end > file_size:
                raise ValueError(
                    f"{name} byte range [{start}, {end}) exceeds {file_name} ({file_size} bytes)"
                )
            refs.setdefault(shard_index, {})[part] = (
                model_path / file_name,
                data_offset + start,
                (end - start) // rows,
            )
    return refs


def spot_check_sidecar_rows(
    sidecar_path: str, model_path, manifest: dict, num_random: int = 256
) -> int:
    """Compare sampled sidecar rows byte-for-byte against the source tensors.

    The manifest's index digest only proves the sidecar was built against an
    artifact with the same ``model.safetensors.index.json``; this binds the
    check to actual tensor content. Deterministic edge rows (first/last row
    of the first and last shard, plus each shard boundary neighborhood of
    the first shard) and ``num_random`` freshly-drawn random rows are read
    from both the sidecar and the safetensors source. Raises ``ValueError``
    on the first mismatch: a bit-flipped sidecar or a source whose shard
    bytes changed under an unchanged index must both refuse to load.

    Returns the number of rows checked.
    """
    rows_per_shard = manifest["rows_per_shard"]
    total_rows = manifest["total_rows"]
    row_bytes = manifest["row_bytes"]
    refs = _source_shard_refs(model_path, manifest)
    if sorted(refs) != list(range(manifest["num_shards"])):
        raise ValueError(
            f"artifact has shard indices {sorted(refs)[:3]}..., manifest expects 0..{manifest['num_shards'] - 1}"
        )
    edges = {
        0,
        rows_per_shard - 1,
        min(rows_per_shard, total_rows - 1),
        total_rows - rows_per_shard,
        total_rows - 1,
    }
    rng = np.random.default_rng()
    picks = sorted(
        edges | {int(r) for r in rng.integers(0, total_rows, size=num_random)}
    )
    handles = {}

    def fh(path):
        if path not in handles:
            handles[path] = open(path, "rb")
        return handles[path]

    try:
        with open(sidecar_path, "rb") as sidecar:
            for global_row in picks:
                (shard_index, local) = divmod(global_row, rows_per_shard)
                expected = b""
                for part in ("weight", "scales", "biases"):
                    (path, start, nbytes) = refs[shard_index][part]
                    f = fh(path)
                    f.seek(start + local * nbytes)
                    expected += f.read(nbytes)
                sidecar.seek(manifest["data_offset"] + global_row * row_bytes)
                if sidecar.read(row_bytes) != expected:
                    raise ValueError(
                        f"PLE sidecar row {global_row} (shard {shard_index}, local {local}) does not match the artifact's shard tensors: the sidecar is stale or corrupt. Rebuild it with scripts/build_qwen4_ple_sidecar.py."
                    )
    finally:
        for f in handles.values():
            f.close()
    return len(picks)


def manifest_path(sidecar_path: str) -> str:
    return str(sidecar_path) + ".manifest.json"


def load_manifest(sidecar_path: str) -> dict:
    with open(manifest_path(sidecar_path), "r") as f:
        manifest = json.load(f)
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError(
            f"{manifest_path(sidecar_path)} is not a {MANIFEST_FORMAT} manifest"
        )
    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError(
            f"unsupported PLE sidecar manifest version {manifest.get('version')}"
        )
    _validate_manifest_geometry(manifest)
    return manifest


def _validate_manifest_geometry(manifest: dict) -> None:
    """Reject a manifest whose declared layout is internally inconsistent.

    The digest and spot checks bind the sidecar to the artifact; this binds
    the addressing arithmetic (row width, strides, shard split) before any
    field is used to compute a file offset.
    """
    quant = {key: manifest.get(key) for key in ("bits", "group_size", "mode")}
    if quant != {"bits": 4, "group_size": 32, "mode": "affine"}:
        raise ValueError(f"unsupported PLE sidecar quantization: {quant}")
    dims = manifest.get("dims", 0)
    if dims <= 0 or dims % 32:
        raise ValueError(f"PLE sidecar dims={dims} must be a positive multiple of 32")
    groups = dims // 32
    expected = {
        "weight_bytes": dims // 2,
        "scales_bytes": groups * 2,
        "biases_bytes": groups * 2,
        "row_bytes": dims // 2 + 2 * groups * 2,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(
                f"PLE sidecar {field}={manifest.get(field)} does not match dims={dims} (expected {value})"
            )
    num_shards = manifest.get("num_shards", 0)
    rows_per_shard = manifest.get("rows_per_shard", 0)
    if num_shards <= 0 or rows_per_shard <= 0:
        raise ValueError("PLE sidecar shard counts must be positive")
    if manifest.get("total_rows") != num_shards * rows_per_shard:
        raise ValueError(
            f"PLE sidecar total_rows={manifest.get('total_rows')} != {num_shards} shards x {rows_per_shard} rows"
        )
    if manifest.get("data_offset", -1) < 0:
        raise ValueError("PLE sidecar data_offset must be non-negative")
    if len(manifest.get("shard_sha256", [])) != num_shards:
        raise ValueError("PLE sidecar manifest needs one sha256 per shard")


def index_json_sha256(model_path) -> str:
    with open(Path(model_path) / "model.safetensors.index.json", "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def verify_sidecar_against_artifact(sidecar_path: str, model_path) -> dict:
    """Check the sidecar manifest and file against the source artifact.

    Verifies the recorded ``model.safetensors.index.json`` digest and the
    sidecar file size. Raises ``ValueError`` on any mismatch. Returns the
    parsed manifest.
    """
    manifest = load_manifest(sidecar_path)
    expected = manifest["source_index_sha256"]
    actual = index_json_sha256(model_path)
    if expected != actual:
        raise ValueError(
            f"PLE sidecar was built from a different artifact: manifest source_index_sha256={expected} but {model_path} has {actual}. Rebuild the sidecar with scripts/build_qwen4_ple_sidecar.py."
        )
    expected_size = (
        manifest["data_offset"] + manifest["total_rows"] * manifest["row_bytes"]
    )
    actual_size = os.path.getsize(sidecar_path)
    if actual_size != expected_size:
        raise ValueError(
            f"PLE sidecar {sidecar_path} has size {actual_size}, manifest expects {expected_size}"
        )
    return manifest


def assert_sidecar_not_in_weight_files(sidecar_path: str) -> None:
    """The sidecar must never enter ``load_model``'s weight_files glob.

    ``load_model`` globs ``model*.safetensors`` both to load weights and to
    build the ``MLX_LM_UBC_EVICT`` eviction list. The sidecar backs live
    lookups, so evicting its UBC pages would reintroduce cold reads; keeping
    it out of that glob is a hard invariant of the sidecar naming scheme.
    """
    name = os.path.basename(str(sidecar_path))
    if fnmatch.fnmatch(name, "model*.safetensors"):
        raise ValueError(
            f"PLE sidecar name {name!r} matches the model*.safetensors weight glob; it would be loaded as weights and UBC-evicted. Name it ple_rows.bin."
        )


class FileBackedShardedEmbedding(nn.Module):
    """Drop-in for ``ShardedEmbedding.lookup_numpy`` reading rows from NVMe.

    Holds no public MLX parameters. The optional private verify shards keep
    lazy packed checkpoint arrays for a width-3 device route. Other shapes
    deduplicate host ids, ``pread`` selected rows, and dequantize them once.
    """

    is_file_backed = True

    def __init__(
        self,
        sidecar_path: str,
        vocab_size: int,
        dims: int,
        num_shards: int,
        data_offset: int = 0,
        verify_shards=None,
    ):
        super().__init__()
        if vocab_size % num_shards:
            raise ValueError("PLE vocabulary must split evenly across shards")
        if dims % 32:
            raise ValueError("PLE row width must be a multiple of group size 32")
        self.sidecar_path = str(sidecar_path)
        self.vocab_size = vocab_size
        self.dims = dims
        self.num_shards = num_shards
        self.rows_per_shard = vocab_size // num_shards
        self.row_bytes = dims // 2 + 2 * (dims // 32) * 2
        self.data_offset = data_offset
        self._verify_shards = verify_shards
        self._verify_owner_pid = os.getpid()
        self._verify_device_lookups = 0
        self._verify_fallback_lookups = 0
        self._verify_device_prepared = False
        self.dequant_backend = os.getenv("MLX_QWEN4_PLE_NVME_DEQUANT", "numpy")
        if self.dequant_backend not in {"numpy", "mx"}:
            raise ValueError("MLX_QWEN4_PLE_NVME_DEQUANT must be numpy or mx")
        self.stats_timing = os.getenv("MLX_QWEN4_PLE_NVME_STATS_TIMING") == "1"
        lru_mb = float(os.getenv("MLX_QWEN4_PLE_NVME_LRU_MB", "0"))
        if lru_mb < 0:
            raise ValueError("MLX_QWEN4_PLE_NVME_LRU_MB must be non-negative")
        self.lru_capacity_rows = int(lru_mb * 2**20) // self.row_bytes
        self.preheated_rows = 0
        self.decode_workers = int(
            os.getenv("MLX_QWEN4_PLE_NVME_DECODE_WORKERS", str(DECODE_WORKERS))
        )
        self.prefill_workers = int(
            os.getenv("MLX_QWEN4_PLE_NVME_PREFILL_WORKERS", str(PREFILL_WORKERS))
        )
        self._lifecycle_lock = threading.Lock()
        self._closed = False
        self._stat_lookups = 0
        self._stat_rows = 0
        self._stat_unique_rows = 0
        self._stat_bytes = 0
        self._stat_elapsed = 0.0
        self._stat_cache_hits = 0
        self._stat_cache_misses = 0
        self._stat_cache_evictions = 0
        with _FORK_RESOURCE_LOCK:
            self._open_resources()

    def _open_resources(self):
        fd = os.open(self.sidecar_path, os.O_RDONLY)
        pool = None
        try:
            pool = ThreadPoolExecutor(
                max_workers=max(self.decode_workers, self.prefill_workers),
                thread_name_prefix="ple-nvme",
            )
            prefetch_pool = ThreadPoolExecutor(
                max_workers=PREFETCH_WORKERS, thread_name_prefix="ple-nvme-prefetch"
            )
        except BaseException:
            if pool is not None:
                pool.shutdown(wait=True)
            os.close(fd)
            raise
        self._lru = OrderedDict()
        self._lru_lock = threading.Lock()
        self._dq_cache = {}
        self._dq_lock = threading.Lock()
        self._fd = fd
        self._pool = pool
        self._prefetch_pool = prefetch_pool
        self._owner_pid = os.getpid()

    def _check_owner(self, *, reopen: bool = True) -> None:
        """Rebuild fd/pools/LRU after a fork before touching any of them.

        A forked child inherits the parent's dict and possibly a lock held
        at fork time; every LRU-touching entry point calls this first (not
        only ``_submit``) so a complete cache hit or a prefetch membership
        filter can never use the inherited state. The module lock is reset
        in the child; inherited instance locks must never be acquired here.
        """
        if os.getpid() != self._owner_pid:
            with _FORK_RESOURCE_LOCK:
                if os.getpid() != self._owner_pid:
                    inherited_fd = self._fd
                    self._lifecycle_lock = threading.Lock()
                    self._lru_lock = threading.Lock()
                    self._dq_lock = threading.Lock()
                    self._lru = OrderedDict()
                    self._dq_cache = {}
                    self._fd = self._pool = self._prefetch_pool = None
                    try:
                        if reopen and (not self._closed):
                            self._open_resources()
                        else:
                            self._closed = True
                            self._owner_pid = os.getpid()
                    finally:
                        if inherited_fd is not None:
                            os.close(inherited_fd)

    def _submit(self, use_prefetch_pool: bool, fns, required: bool):
        """Submit ``fns`` atomically with the closed/fork check.

        Fork safety: a forked child inherits executor bookkeeping but none
        of the worker threads, so submissions in the child would hang.
        Detect the pid change and rebuild fd + pools in the child (the
        parent's descriptors stay untouched). Submitting under the
        lifecycle lock makes the closed check atomic with the submission,
        so close() either sees the futures (and drains them before closing
        the fd) or the submission observes the closed flag. ``required``
        submissions raise when closed; optional (prefetch) ones no-op.

        Each fn is called as ``fn(fd)``; the fd stays valid while the
        returned futures may still run because close() drains the pools
        before closing it.
        """
        self._check_owner()
        with self._lifecycle_lock:
            if self._closed:
                if required:
                    raise RuntimeError("FileBackedShardedEmbedding is closed")
                return []
            pool = self._prefetch_pool if use_prefetch_pool else self._pool
            return [pool.submit(fn, self._fd) for fn in fns]

    def close(self):
        self._check_owner(reopen=False)
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            (fd, pool, prefetch_pool) = (self._fd, self._pool, self._prefetch_pool)
        pool.shutdown(wait=True)
        prefetch_pool.shutdown(wait=True)
        with _FORK_RESOURCE_LOCK:
            os.close(fd)
            self._fd = self._pool = self._prefetch_pool = None
        with self._lru_lock:
            self._lru.clear()
        with self._dq_lock:
            self._dq_cache.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _workers_for(self, num_ids: int) -> int:
        if num_ids >= PREFILL_ID_THRESHOLD:
            return self.prefill_workers
        return self.decode_workers

    def _pread_rows(self, row_ids: np.ndarray, workers: int) -> np.ndarray:
        """Read ``row_ids`` from disk on the pool; no cache involved."""
        n = int(row_ids.size)
        out = np.empty((n, self.row_bytes), dtype=np.uint8)
        if n == 0:
            return out
        row_bytes = self.row_bytes
        base = self.data_offset

        def read_span(start, stop):

            def task(fd):
                for i in range(start, stop):
                    offset = base + int(row_ids[i]) * row_bytes
                    data = os.pread(fd, row_bytes, offset)
                    if len(data) != row_bytes:
                        raise IOError(
                            f"short pread of PLE row {int(row_ids[i])} ({len(data)}/{row_bytes} bytes)"
                        )
                    out[i] = np.frombuffer(data, dtype=np.uint8)

            return task

        workers = max(1, min(workers, n))
        bounds = np.linspace(0, n, workers + 1, dtype=np.int64)
        futures = self._submit(
            False,
            [
                read_span(int(bounds[w]), int(bounds[w + 1]))
                for w in range(workers)
                if bounds[w] < bounds[w + 1]
            ],
            required=True,
        )
        wait(futures)
        for future in futures:
            future.result()
        return out

    def _cache_put(self, row_ids, rows: np.ndarray) -> None:
        """Insert packed rows; evict LRU entries over the byte budget."""
        if not self.lru_capacity_rows:
            return
        self._check_owner()
        with self._lru_lock:
            if self._closed:
                return
            for row_id, row in zip(np.asarray(row_ids).tolist(), rows):
                self._lru[int(row_id)] = bytes(row)
                self._lru.move_to_end(int(row_id))
            while len(self._lru) > self.lru_capacity_rows:
                self._lru.popitem(last=False)
                self._stat_cache_evictions += 1

    def _read_rows(self, row_ids: np.ndarray, workers: int) -> np.ndarray:
        """Serve ``row_ids`` from the LRU where possible, disk otherwise."""
        n = int(row_ids.size)
        if not self.lru_capacity_rows or n == 0:
            self._stat_bytes += n * self.row_bytes
            return self._pread_rows(row_ids, workers)
        self._check_owner()
        out = np.empty((n, self.row_bytes), dtype=np.uint8)
        missing_positions = []
        with self._lru_lock:
            for i in range(n):
                cached = self._lru.get(int(row_ids[i]))
                if cached is None:
                    missing_positions.append(i)
                else:
                    self._lru.move_to_end(int(row_ids[i]))
                    out[i] = np.frombuffer(cached, dtype=np.uint8)
        self._stat_cache_hits += n - len(missing_positions)
        self._stat_cache_misses += len(missing_positions)
        if missing_positions:
            missing_ids = row_ids[np.asarray(missing_positions, dtype=np.int64)]
            rows = self._pread_rows(missing_ids, workers)
            out[np.asarray(missing_positions, dtype=np.int64)] = rows
            self._cache_put(missing_ids, rows)
            self._stat_bytes += len(missing_positions) * self.row_bytes
        return out

    def _dequant(self, row_bytes: np.ndarray) -> mx.array:
        if self.dequant_backend == "numpy":
            bits = dequant_rows_numpy(row_bytes, self.dims)
            return mx.array(bits).view(mx.bfloat16)
        weight_bytes = self.dims // 2
        group_bytes = self.dims // 32 * 2
        w = mx.array(np.ascontiguousarray(row_bytes[:, :weight_bytes]).view(np.uint32))
        s = mx.array(
            np.ascontiguousarray(
                row_bytes[:, weight_bytes : weight_bytes + group_bytes]
            ).view(np.uint16)
        ).view(mx.bfloat16)
        b = mx.array(
            np.ascontiguousarray(row_bytes[:, weight_bytes + group_bytes :]).view(
                np.uint16
            )
        ).view(mx.bfloat16)
        return mx.dequantize(w, s, b, group_size=32, bits=4, mode="affine")

    def lookup_numpy(self, indices: np.ndarray) -> mx.array:
        self._check_owner()
        if self._closed:
            raise RuntimeError("FileBackedShardedEmbedding is closed")
        _ht_t0 = 0.0
        started = time.perf_counter() if self.stats_timing else None
        shape = indices.shape
        flat = np.asarray(indices, dtype=np.int64).reshape(-1)
        (unique, inverse) = np.unique(flat, return_inverse=True)
        staged = None
        dq_cache = getattr(self, "_dq_cache", None)
        if dq_cache and self.dequant_backend == "numpy":
            with self._dq_lock:
                staged = [dq_cache.pop(int(i), None) for i in unique.tolist()]
            hits = sum((1 for s in staged if s is not None))
            _lv.bump("ple_dq_hits", hits)
            _lv.bump("ple_dq_misses", int(unique.size) - hits)
            if hits == 0:
                staged = None
        if staged is None:
            rows = self._read_rows(unique, self._workers_for(flat.size))
            values = self._dequant(rows)
        else:
            bits = np.empty((unique.size, self.dims), dtype=np.uint16)
            miss = [j for (j, s) in enumerate(staged) if s is None]
            for j, s in enumerate(staged):
                if s is not None:
                    bits[j] = s
            if miss:
                miss_idx = np.asarray(miss, dtype=np.int64)
                rows = self._read_rows(unique[miss_idx], self._workers_for(len(miss)))
                bits[miss_idx] = dequant_rows_numpy(rows, self.dims)
            values = mx.array(bits).view(mx.bfloat16)
        result = values[mx.array(inverse.astype(np.int64))].reshape(*shape, self.dims)
        self._stat_lookups += 1
        self._stat_rows += flat.size
        self._stat_unique_rows += unique.size
        if started is not None:
            self._stat_elapsed += time.perf_counter() - started
        return result

    @property
    def verify_device_available(self) -> bool:
        return (
            self._verify_shards is not None
            and (not self._closed)
            and (os.getpid() == self._verify_owner_pid)
        )

    def record_verify_route(self, route: str) -> None:
        if route == "device":
            self._verify_device_lookups += 1
        elif route == "fallback":
            self._verify_fallback_lookups += 1
        else:
            raise ValueError(f"unknown PLE verify route {route!r}")

    @property
    def verify_status(self):
        return {
            "device_available": self.verify_device_available,
            "device_prepared": self._verify_device_prepared,
            "device_lookups": self._verify_device_lookups,
            "fallback_lookups": self._verify_fallback_lookups,
        }

    def prepare_verify_device(self) -> None:
        """Stage private packed shards before the first verify command buffer."""
        if not self.verify_device_available or self._verify_device_prepared:
            return
        for shard in self._verify_shards:
            mx.eval(*shard)
        self._verify_device_prepared = True

    def lookup_verify_device(self, indices: mx.array) -> mx.array:
        """Gather width-3 verify rows from private packed shard arrays."""
        if not self.verify_device_available:
            raise RuntimeError("PLE device verify table is unavailable")
        shape = indices.shape
        flat = indices.reshape(-1).astype(mx.int64)
        output = None
        for shard_index, (weight, scales, biases) in enumerate(self._verify_shards):
            start = shard_index * self.rows_per_shard
            stop = start + self.rows_per_shard
            active = (flat >= start) & (flat < stop)
            local = mx.clip(flat - start, 0, self.rows_per_shard - 1)
            values = mx.dequantize(
                mx.take(weight, local, axis=0),
                mx.take(scales, local, axis=0),
                mx.take(biases, local, axis=0),
                group_size=32,
                bits=4,
                mode="affine",
            )
            if output is None:
                output = mx.zeros_like(values)
            output = mx.where(active[:, None], values, output)
        return output.reshape(*shape, self.dims)

    @property
    def stats(self) -> LookupStats:
        return LookupStats(
            int(self._stat_lookups),
            int(self._stat_rows),
            int(self._stat_unique_rows),
            int(self._stat_bytes),
            float(self._stat_elapsed),
            int(self._stat_cache_hits),
            int(self._stat_cache_misses),
            int(self._stat_cache_evictions),
        )

    def __call__(self, indices: mx.array) -> mx.array:
        mx.eval(indices)
        return self.lookup_numpy(np.asarray(indices, dtype=np.int64))

    def prefetch_rows(self, indices: np.ndarray) -> list:
        """Warm the page cache (and LRU, when enabled) without blocking.

        Fire-and-forget. With the LRU on, already-cached rows are skipped
        and freshly-read rows are inserted, so prefetched rows survive
        page-cache eviction under memory pressure.
        """
        flat = np.unique(np.asarray(indices, dtype=np.int64).reshape(-1))
        if self.lru_capacity_rows and flat.size:
            self._check_owner()
            with self._lru_lock:
                flat = np.asarray(
                    [i for i in flat.tolist() if i not in self._lru], dtype=np.int64
                )
        if flat.size == 0:
            return []
        row_bytes = self.row_bytes
        base = self.data_offset
        cache_put = self._cache_put if self.lru_capacity_rows else None
        bounds = np.linspace(
            0, flat.size, min(PREFETCH_WORKERS, flat.size) + 1, dtype=np.int64
        )

        def warm(start, stop):

            def task(fd):
                span = []
                for i in range(start, stop):
                    span.append(
                        os.pread(fd, row_bytes, base + int(flat[i]) * row_bytes)
                    )
                if cache_put is not None:
                    complete = [
                        (int(flat[start + j]), data)
                        for (j, data) in enumerate(span)
                        if len(data) == row_bytes
                    ]
                    if complete:
                        cache_put(
                            [row_id for (row_id, _) in complete],
                            [
                                np.frombuffer(data, dtype=np.uint8)
                                for (_, data) in complete
                            ],
                        )

            return task

        return self._submit(
            True,
            [
                warm(int(bounds[w]), int(bounds[w + 1]))
                for w in range(len(bounds) - 1)
                if bounds[w] < bounds[w + 1]
            ],
            required=False,
        )

    _DQ_CACHE_MAX_ROWS = 8192

    def prefetch_dequant_rows(self, indices: np.ndarray) -> int:
        """Lever (a): read + dequantize ``indices`` on the prefetch pool.

        Rows already staged are skipped. Reads go through the packed LRU
        (and populate it) like ``prefetch_rows``; the dequantization is
        ``dequant_rows_numpy``, the function the foreground applies, so a
        staged row is bit-identical to the one it replaces. Only the numpy
        dequant backend is supported (the ``mx`` backend is not bit-exact
        with it and stays foreground-only). Returns the rows submitted.
        """
        if self.dequant_backend != "numpy":
            return 0
        flat = np.unique(np.asarray(indices, dtype=np.int64).reshape(-1))
        if flat.size == 0:
            return 0
        self._check_owner()
        with self._dq_lock:
            missing = [i for i in flat.tolist() if i not in self._dq_cache]
        if not missing:
            return 0
        ids = np.asarray(missing, dtype=np.int64)
        (row_bytes, base, dims) = (self.row_bytes, self.data_offset, self.dims)
        lru = self.lru_capacity_rows > 0

        def task(fd):
            rows = np.empty((ids.size, row_bytes), dtype=np.uint8)
            need = []
            if lru:
                with self._lru_lock:
                    for j, row_id in enumerate(ids.tolist()):
                        cached = self._lru.get(row_id)
                        if cached is None:
                            need.append(j)
                        else:
                            rows[j] = np.frombuffer(cached, dtype=np.uint8)
            else:
                need = list(range(ids.size))
            (fresh_ids, fresh_rows) = ([], [])
            for j in need:
                data = os.pread(fd, row_bytes, base + int(ids[j]) * row_bytes)
                if len(data) != row_bytes:
                    return
                rows[j] = np.frombuffer(data, dtype=np.uint8)
                fresh_ids.append(int(ids[j]))
                fresh_rows.append(rows[j])
            if lru and fresh_ids:
                self._cache_put(fresh_ids, fresh_rows)
            bits = dequant_rows_numpy(rows, dims)
            with self._dq_lock:
                if len(self._dq_cache) >= self._DQ_CACHE_MAX_ROWS:
                    self._dq_cache.clear()
                for j, row_id in enumerate(ids.tolist()):
                    self._dq_cache[row_id] = bits[j]
            _lv.bump("ple_prefetch_rows", int(ids.size))

        self._submit(True, [task], required=False)
        return int(ids.size)

    def submit_prefetch(self, fn) -> None:
        """Run ``fn`` (id hashing + ``prefetch_rows``) on the prefetch pool.

        ``fn`` takes no arguments; a closed embedding drops it silently.
        """
        self._submit(True, [lambda _fd: fn()], required=False)

    def preheat_from_file(self, path: str) -> int:
        """Load hot-row ids (one per line, ``#`` comments) into the LRU.

        The file is ordered hottest-first (see
        ``scripts/build_qwen4_ple_hot_rows.py``); ids beyond the LRU byte
        budget are dropped. Synchronous and fail-closed: an id outside the
        table refuses. Returns the number of rows cached.
        """
        if not self.lru_capacity_rows:
            raise ValueError(
                "MLX_QWEN4_PLE_NVME_PREHEAT requires MLX_QWEN4_PLE_NVME_LRU_MB to be set to a positive budget"
            )
        ids = []
        with open(path) as f:
            for line in f:
                text = line.split("#", 1)[0].strip()
                if not text:
                    continue
                row_id = int(text)
                if not 0 <= row_id < self.vocab_size:
                    raise ValueError(
                        f"hot-row id {row_id} in {path} is outside the PLE table [0, {self.vocab_size})"
                    )
                ids.append(row_id)
        ordered = list(dict.fromkeys(ids))[: self.lru_capacity_rows]
        if not ordered:
            return 0
        id_array = np.asarray(ordered, dtype=np.int64)
        rows = self._pread_rows(id_array, self.prefill_workers)
        self._cache_put(id_array[::-1], rows[::-1])
        return len(ordered)


def _iter_ple_embeddings(model):
    """Yield ``(weight_key_prefix, NGramEmbedding)`` for every PLE layer."""
    language_model = getattr(model, "language_model", model)
    for index, layer in enumerate(language_model.model.layers):
        ple = getattr(layer, "ple", None)
        if ple is not None:
            prefix = (
                f"language_model.model.layers.{index}.ple.ple_embedding.ngram_embedding"
            )
            yield (prefix, ple.ple_embedding)


def install_file_backed_ple(
    model, weights: dict, sidecar_path: str, model_path, *, _owned_tables=None
):
    """Replace the matching resident ``ShardedEmbedding`` with the sidecar.

    Called from ``load_model`` after ``sanitize`` and before quantization.
    Verifies the manifest against the artifact, swaps in a
    ``FileBackedShardedEmbedding`` (removing the shard modules from the
    module tree so the quantization predicate never visits them), drops the
    ``shard_*`` tensors from ``weights``, and forces the CPU hash backend.

    Returns the pruned weights dict.
    """
    assert_sidecar_not_in_weight_files(sidecar_path)
    manifest = verify_sidecar_against_artifact(sidecar_path, model_path)
    spot_check_sidecar_rows(
        sidecar_path,
        model_path,
        manifest,
        num_random=int(os.getenv("MLX_QWEN4_PLE_NVME_SPOT_CHECK_ROWS", "256")),
    )
    installed = False
    for prefix, ngram_embedding in _iter_ple_embeddings(model):
        if prefix != manifest["tensor_prefix"]:
            continue
        resident = ngram_embedding.ngram_embedding
        if getattr(resident, "is_file_backed", False):
            raise ValueError(f"PLE sidecar already installed at {prefix}")
        for field, expected in (
            ("total_rows", resident.vocab_size),
            ("dims", resident.dims),
            ("num_shards", resident.num_shards),
            ("rows_per_shard", resident.rows_per_shard),
        ):
            if manifest[field] != expected:
                raise ValueError(
                    f"PLE sidecar manifest {field}={manifest[field]} does not match the model ({expected}) at {prefix}"
                )
        verify_shards = []
        for shard_index in range(resident.num_shards):
            shard = f"{prefix}.shard_{shard_index}"
            keys = tuple(
                (f"{shard}.{field}" for field in ("weight", "scales", "biases"))
            )
            if not all((key in weights for key in keys)):
                verify_shards = None
                break
            verify_shards.append(tuple((weights[key] for key in keys)))
        table = FileBackedShardedEmbedding(
            sidecar_path,
            vocab_size=manifest["total_rows"],
            dims=manifest["dims"],
            num_shards=manifest["num_shards"],
            data_offset=manifest["data_offset"],
            verify_shards=None if verify_shards is None else tuple(verify_shards),
        )
        try:
            if _owned_tables is not None:
                _owned_tables.append(table)
            if (
                os.getenv("MLX_QWEN4_PLE_VERIFY_DEVICE") == "1"
                and mx.metal.is_available()
                and (mx.default_device() == mx.gpu)
            ):
                table.prepare_verify_device()
            ngram_embedding.ngram_embedding = table
            if preheat := os.environ.get("MLX_QWEN4_PLE_NVME_PREHEAT"):
                ngram_embedding.ngram_embedding.preheated_rows = (
                    ngram_embedding.ngram_embedding.preheat_from_file(preheat)
                )
            if ngram_embedding.hash_backend in ("metal", "metal_prefill"):
                warnings.warn(
                    f"MLX_QWEN4_PLE_NVME forces CPU n-gram hashing; overriding MLX_QWEN4_PLE_HASH_BACKEND={ngram_embedding.hash_backend} to routed_cpu"
                )
            ngram_embedding.hash_backend = "routed_cpu"
            shard_prefix = f"{prefix}.shard_"
            for key in [k for k in weights if k.startswith(shard_prefix)]:
                del weights[key]
            installed = True
        except BaseException:
            try:
                table.close()
            except BaseException:
                pass
            raise
    if not installed:
        raise ValueError(
            f"PLE sidecar tensor_prefix {manifest['tensor_prefix']!r} matches no PLE layer in this model"
        )
    return weights

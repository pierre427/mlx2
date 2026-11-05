# SPDX-License-Identifier: Apache-2.0
"""Opt-in CPU experiment for immutable sparse KV pages and logical reductions.

This module never imports MLX. It is a request-owned scratch store, not an
APCv2 replacement or a serving route. The two-pass NumPy oracle specifies its
own FP32 operation order and BF16 partial rounding: it does NOT certify Metal
``fast::exp``, SIMD reduction order, or an installed MLX build. See
``provenance/ironmlx-sparse-kv-20261104.json`` for the design source.
"""

from __future__ import annotations

import hashlib
import math
import re
import shutil
import struct
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_MAGIC = b"MLX2SKV1"
_HEADER = struct.Struct("<8sQIII")


def _positive(value: int, name: str, *, zero: bool = False) -> int:
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError(
            f"{name} must be {'nonnegative' if zero else 'positive'} integer"
        )
    return value


def _cpu_array(values, name: str) -> np.ndarray:
    # Device arrays can execute from __array__; reject before conversion.
    if type(values) is not np.ndarray or values.dtype != np.float32:
        raise ValueError(f"{name} must be a NumPy float32 ndarray")
    return values


def bf16_round(values) -> np.ndarray:
    """Round finite FP32 inputs to BF16, represented as FP32, ties to even."""
    array = _cpu_array(values, "values")
    if not np.isfinite(array).all():
        raise ValueError("KV and attention values must be finite")
    bits = array.view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    result = rounded.view(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("BF16 conversion overflow")
    return result


@dataclass(frozen=True)
class SparseKVConfig:
    artifact_sha256: str
    source_revision: str
    heads: int
    head_dim: int
    page_tokens: int = 4
    max_tokens: int = 65536
    max_disk_bytes: int = 64 << 20
    max_gather_bytes: int = 8 << 20
    read_cache_bytes: int = 1 << 20

    def __post_init__(self):
        if not isinstance(self.artifact_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.artifact_sha256
        ):
            raise ValueError("artifact_sha256 must be a lowercase SHA256")
        if (
            not isinstance(self.source_revision, str)
            or not self.source_revision.strip()
        ):
            raise ValueError("source_revision is required")
        for name in (
            "heads",
            "head_dim",
            "page_tokens",
            "max_tokens",
            "max_disk_bytes",
            "max_gather_bytes",
        ):
            _positive(getattr(self, name), name)
        _positive(self.read_cache_bytes, "read_cache_bytes", zero=True)
        if self.page_tokens > self.max_tokens:
            raise ValueError("page_tokens exceeds max_tokens")


@dataclass(frozen=True)
class SelectedKV:
    positions: np.ndarray
    keys: np.ndarray
    values: np.ndarray
    logical_length: int
    artifact_sha256: str
    source_revision: str


@dataclass
class _Record:
    path: Path
    tokens: int
    size: int
    references: int = 1


class SparseKVSnapshot:
    def __init__(self, store: SparseKVStore, pages: tuple[int, ...], length: int):
        self._store = store
        self._pages = pages
        self.length = length
        self.closed = False
        store._retain(pages)

    def close(self):
        if not self.closed:
            self.closed = True
            self._store._release(self._pages)

    def __enter__(self):
        if self.closed:
            raise ValueError("snapshot is closed")
        return self

    def __exit__(self, *_):
        self.close()


class SparseKVStore:
    """One explicit, bounded CPU scratch session. Close histories/snapshots first.

    Immutable records are reference counted across histories and snapshots.
    There is no reopen or crash-resume contract. A session directory is private
    and disposable; no model or prefix-cache files are read or modified.
    The owner serializes calls, just as an adapter-owned execution actor does.
    """

    def __init__(self, config: SparseKVConfig, *, directory: str | Path | None = None):
        self.config = config
        self.root = Path(tempfile.mkdtemp(prefix="mlx2-sparse-kv-", dir=directory))
        self._records: dict[int, _Record] = {}
        self._pending_delete: set[int] = set()
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        self._next = 0
        self._disk = 0
        self._cached = 0
        self._closed = False
        self.read_bytes = 0
        self.cache_hits = 0
        self.cleanup_failures = 0

    def _open(self):
        if self._closed:
            raise ValueError("store is closed")

    def history(self) -> SparseKVHistory:
        self._open()
        return SparseKVHistory(self)

    def fork(self, snapshot: SparseKVSnapshot) -> SparseKVHistory:
        self._check_snapshot(snapshot)
        self._retain(snapshot._pages)
        return SparseKVHistory(self, snapshot._pages, snapshot.length)

    def _check_snapshot(self, snapshot):
        self._open()
        if (
            not isinstance(snapshot, SparseKVSnapshot)
            or snapshot._store is not self
            or snapshot.closed
        ):
            raise ValueError("foreign or closed snapshot")

    def _retain(self, pages):
        self._open()
        if any(
            page not in self._records or self._records[page].references == 0
            for page in pages
        ):
            raise ValueError("released page")
        for page in pages:
            self._records[page].references += 1

    def _release(self, pages):
        for page in pages:
            record = self._records[page]
            record.references -= 1
            if record.references == 0:
                if page in self._cache:
                    self._cached -= len(self._cache.pop(page))
                self._pending_delete.add(page)
                self._collect_page(page)

    def _collect_page(self, page):
        record = self._records[page]
        try:
            record.path.unlink(missing_ok=True)
        except OSError:
            # A logical transaction must not stop halfway through releasing
            # several pages. Keep cleanup debt and quota, then retry explicitly.
            self.cleanup_failures += 1
            return
        self._disk -= record.size
        self._pending_delete.remove(page)
        del self._records[page]

    def collect(self):
        """Retry deferred file cleanup without resurrecting released pages."""
        self._open()
        for page in tuple(self._pending_delete):
            self._collect_page(page)
        return len(self._pending_delete)

    def _write(self, keys: np.ndarray, values: np.ndarray) -> int:
        token_count = len(keys)
        generation = self._next
        self._next += 1
        packed = np.stack((keys, values), axis=1).view(np.uint32)
        payload = (packed >> 16).astype("<u2").tobytes()
        body = (
            _HEADER.pack(
                _MAGIC, generation, token_count, self.config.heads, self.config.head_dim
            )
            + payload
        )
        raw = hashlib.sha256(body).digest() + body
        if self._disk + len(raw) > self.config.max_disk_bytes:
            raise ValueError("sparse KV disk quota exceeded")
        path = self.root / f"{generation:016x}.page"
        try:
            with path.open("xb") as handle:
                path.chmod(0o600)
                handle.write(raw)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        self._records[generation] = _Record(path, token_count, len(raw))
        self._disk += len(raw)
        return generation

    def _read(self, page: int) -> tuple[np.ndarray, np.ndarray]:
        record = self._records[page]
        if page in self._cache:
            raw = self._cache.pop(page)
            self._cache[page] = raw
            self.cache_hits += 1
        else:
            # Limit reads even if a file was replaced by an oversized one.
            with record.path.open("rb") as handle:
                raw = handle.read(record.size + 1)
            self.read_bytes += len(raw)
        if (
            len(raw) != record.size
            or hashlib.sha256(memoryview(raw)[32:]).digest() != raw[:32]
        ):
            raise ValueError("sparse KV checksum or size mismatch")
        magic, generation, tokens, heads, dim = _HEADER.unpack_from(raw, 32)
        if (magic, generation, tokens, heads, dim) != (
            _MAGIC,
            page,
            record.tokens,
            self.config.heads,
            self.config.head_dim,
        ):
            raise ValueError("sparse KV record identity mismatch")
        if page not in self._cache and len(raw) <= self.config.read_cache_bytes:
            while (
                self._cache and self._cached + len(raw) > self.config.read_cache_bytes
            ):
                _, old = self._cache.popitem(last=False)
                self._cached -= len(old)
            self._cache[page] = raw
            self._cached += len(raw)
        words = np.frombuffer(raw, dtype="<u2", offset=32 + _HEADER.size).astype(
            np.uint32
        )
        arrays = (words << 16).view(np.float32).reshape(tokens, 2, heads, dim)
        return arrays[:, 0], arrays[:, 1]

    def receipt(self):
        return {
            "schema": "mlx2.ironmlx_sparse_kv.cpu.v1",
            "artifact_sha256": self.config.artifact_sha256,
            "source_revision": self.config.source_revision,
            "implemented": "cpu_experiment",
            "qualified": False,
            "selected": False,
            "observed_used": False,
            "execution_device": "cpu",
            "gpu_used": False,
            "live_pages": len(self._records) - len(self._pending_delete),
            "pending_delete_pages": len(self._pending_delete),
            "cleanup_failures": self.cleanup_failures,
            "disk_bytes": self._disk,
            "read_cache_bytes": self._cached,
            "application_read_bytes": self.read_bytes,
            "read_cache_hits": self.cache_hits,
        }

    def close(self):
        if not self._closed:
            self.collect()
            if len(self._records) > len(self._pending_delete):
                raise ValueError("close live histories and snapshots before store")
            if self._pending_delete:
                raise OSError(
                    "sparse KV cleanup pending; retry close after filesystem recovery"
                )
            shutil.rmtree(self.root)
            self._closed = True

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *_):
        self.close()


class SparseKVHistory:
    def __init__(self, store: SparseKVStore, pages=(), length=0):
        self.store = store
        self._pages = tuple(pages)
        self.length = length
        self.closed = False

    def _open(self):
        self.store._open()
        if self.closed:
            raise ValueError("history is closed")

    def append(self, keys, values):
        self._open()
        keys, values = _cpu_array(keys, "keys"), _cpu_array(values, "values")
        expected = (self.store.config.heads, self.store.config.head_dim)
        if (
            keys.ndim != 3
            or keys.shape[1:] != expected
            or keys.shape != values.shape
            or len(keys) == 0
        ):
            raise ValueError("KV must have equal nonempty [tokens, heads, dim] shapes")
        if self.length + len(keys) > self.store.config.max_tokens:
            raise ValueError("sparse KV logical token cap exceeded")
        page_tokens = self.store.config.page_tokens
        written_tokens = len(keys) + self.length % page_tokens
        needed = (
            math.ceil(written_tokens / page_tokens) * (32 + _HEADER.size)
            + written_tokens * self.store.config.heads * self.store.config.head_dim * 4
        )
        # Snapshots and the previous tail stay charged until publication.
        # Reject before BF16 conversion/stacking can allocate a large copy.
        if self.store._disk + needed > self.store.config.max_disk_bytes:
            raise ValueError("sparse KV disk quota exceeded")
        keys, values = bf16_round(keys), bf16_round(values)
        kept = self._pages
        if self.length % page_tokens:
            tail = self.length % page_tokens
            old_keys, old_values = self.store._read(self._pages[-1])
            keys = np.concatenate((old_keys[:tail], keys))
            values = np.concatenate((old_values[:tail], values))
            kept = self._pages[:-1]
        created = []
        try:
            for start in range(0, len(keys), page_tokens):
                created.append(
                    self.store._write(
                        keys[start : start + page_tokens],
                        values[start : start + page_tokens],
                    )
                )
        except BaseException:
            self.store._release(created)
            raise
        old = self._pages[len(kept) :]
        self._pages = kept + tuple(created)
        self.length = sum(self.store._records[p].tokens for p in self._pages)
        self.store._release(old)

    def snapshot(self) -> SparseKVSnapshot:
        self._open()
        return SparseKVSnapshot(self.store, self._pages, self.length)

    def restore(self, snapshot: SparseKVSnapshot):
        self._open()
        self.store._check_snapshot(snapshot)
        self.store._retain(snapshot._pages)
        self.store._release(self._pages)
        self._pages, self.length = snapshot._pages, snapshot.length

    def trim(self, length: int):
        self._open()
        _positive(length, "length", zero=True)
        if length > self.length:
            raise ValueError("rollback exceeds logical history")
        count = math.ceil(length / self.store.config.page_tokens)
        self.store._release(self._pages[count:])
        self._pages, self.length = self._pages[:count], length

    def gather(self, page_indices) -> SelectedKV:
        self._open()
        indices = tuple(page_indices)
        if any(type(i) is not int or not 0 <= i < len(self._pages) for i in indices):
            raise ValueError("selected page outside history")
        if tuple(sorted(set(indices))) != indices:
            raise ValueError("selected pages must be unique and increasing")
        pt = self.store.config.page_tokens
        counts = [min(pt, self.length - i * pt) for i in indices]
        tokens = sum(counts)
        # Charge returned arrays plus one page's decode scratch (two FP32
        # buffers and its encoded bytes). Read-cache storage has its own cap.
        size = tokens * (self.store.config.heads * self.store.config.head_dim * 8 + 8)
        scratch = max(
            (self.store._records[self._pages[i]].tokens for i in indices), default=0
        )
        scratch = scratch * self.store.config.heads * self.store.config.head_dim * 20
        if indices:
            scratch += 32 + _HEADER.size
        if size + scratch > self.store.config.max_gather_bytes:
            raise ValueError("selected KV gather exceeds RAM budget")
        shape = (tokens, self.store.config.heads, self.store.config.head_dim)
        k, v = np.empty(shape, dtype=np.float32), np.empty(shape, dtype=np.float32)
        pos = np.empty(tokens, dtype=np.int64)
        cursor = 0
        for index, count in zip(indices, counts):
            page_keys, page_values = self.store._read(self._pages[index])
            k[cursor : cursor + count] = page_keys[:count]
            v[cursor : cursor + count] = page_values[:count]
            pos[cursor : cursor + count] = np.arange(index * pt, index * pt + count)
            cursor += count
            del page_keys, page_values
        for array in (k, v, pos):
            array.flags.writeable = False
        return SelectedKV(
            pos,
            k,
            v,
            self.length,
            self.store.config.artifact_sha256,
            self.store.config.source_revision,
        )

    def close(self):
        if not self.closed:
            self.store._release(self._pages)
            self.closed, self._pages, self.length = True, (), 0

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *_):
        self.close()


def partitioned_attention(
    query,
    selected: SelectedKV,
    *,
    blocks: int,
    scale: float = 1.0,
    preserve_logical_positions: bool = True,
    max_workspace_bytes: int = 64 << 20,
) -> np.ndarray:
    """CPU two-pass oracle with a deliberate compact-renumbering negative arm.

    The partition count must come from the reference contract at the ORIGINAL
    logical history length. The caller must not resolve it from gathered size.
    This CPU operation order is independent of Metal SIMD/exp implementations.
    """
    _positive(blocks, "blocks")
    _positive(max_workspace_bytes, "max_workspace_bytes")
    if blocks > 4096:
        raise ValueError("partition workspace exceeds bounded reference contract")
    if type(preserve_logical_positions) is not bool:
        raise ValueError("preserve_logical_positions must be boolean")
    if not isinstance(selected, SelectedKV):
        raise TypeError("selected must be a SelectedKV")
    _positive(selected.logical_length, "logical_length", zero=True)
    q = _cpu_array(query, "query")
    k, v = _cpu_array(selected.keys, "keys"), _cpu_array(selected.values, "values")
    if (
        q.ndim != 2
        or not all(q.shape)
        or k.ndim != 3
        or not all(k.shape[1:])
        or k.shape != v.shape
        or q.shape[1] != k.shape[2]
        or q.shape[0] % k.shape[1]
    ):
        raise ValueError(
            "attention expects query [QH,D], selected KV [T,KVH,D] with integral GQA"
        )
    # Include BF16 rounding's integer temporaries and output rounding. NumPy
    # allocator caches and the caller-owned input arrays are outside this cap.
    if blocks * (q.shape[1] * 24 + 16) + q.nbytes * 3 > max_workspace_bytes:
        raise ValueError("attention partials exceed workspace budget")
    if (
        type(scale) not in (int, float)
        or not math.isfinite(scale)
        or not np.isfinite(q).all()
        or not np.isfinite(k).all()
        or not np.isfinite(v).all()
    ):
        raise ValueError("attention operands and scale must be finite")
    positions = selected.positions
    if (
        type(positions) is not np.ndarray
        or positions.shape != (len(k),)
        or positions.dtype.kind not in "iu"
        or np.any(positions < 0)
        or np.any(positions >= selected.logical_length)
        or np.any(positions[1:] <= positions[:-1])
    ):
        raise ValueError(
            "selected logical positions must be unique, ordered and in history"
        )
    logical = positions if preserve_logical_positions else np.arange(len(k))
    output = np.zeros(q.shape, dtype=np.float32)
    ratio = q.shape[0] // k.shape[1]
    for head in range(q.shape[0]):
        numerators = np.zeros((blocks, q.shape[1]), dtype=np.float32)
        sums = np.zeros(blocks, dtype=np.float32)
        maximums = np.full(blocks, -np.inf, dtype=np.float32)
        for row, position in enumerate(logical):
            block = int(position) % blocks
            score = np.float32(0)
            for dim in range(q.shape[1]):
                product = np.float32(
                    np.float32(q[head, dim] * np.float32(scale))
                    * k[row, head // ratio, dim]
                )
                score = np.float32(score + product)
            maximum = np.maximum(maximums[block], score)
            factor = np.float32(np.exp(np.float32(maximums[block] - maximum)))
            weight = np.float32(np.exp(np.float32(score - maximum)))
            sums[block] = np.float32(sums[block] * factor + weight)
            numerators[block] = np.asarray(
                numerators[block] * factor + weight * v[row, head // ratio],
                dtype=np.float32,
            )
            maximums[block] = maximum
        if len(k):
            partials = bf16_round(numerators)
            maximum = np.max(maximums)
            numerator = np.zeros(q.shape[1], dtype=np.float32)
            denominator = np.float32(0)
            for block in range(blocks):
                factor = np.float32(np.exp(np.float32(maximums[block] - maximum)))
                denominator = np.float32(denominator + factor * sums[block])
                numerator = np.asarray(
                    numerator + factor * partials[block], dtype=np.float32
                )
            output[head] = numerator / denominator
        del numerators, sums, maximums
        if len(k):
            del partials, numerator
    return bf16_round(output)

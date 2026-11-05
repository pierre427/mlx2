# SPDX-License-Identifier: MIT
"""CPU-only exact Engram hashing and bounded, source-precision SSD row reads.

Mechanisms: official V4.1 and ds4; lifecycle: mlx2 Qwen PLE/ReadPool.
No full-table mmap, materialization, or device call occurs in this module.
See provenance/deepseek-v41-adapter.json and retained notices.
"""

import itertools
import math
import os
import stat
import threading
from collections import OrderedDict
from concurrent.futures import TimeoutError as FutureTimeout
from functools import lru_cache

import numpy as np

from ..io_ownership import CloseAction, ReadPool, ReadTimeout, storage_owner


def file_stamp(st):
    return {
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "inode": st.st_ino,
        "device": st.st_dev,
    }


class EngramHasher:
    def __init__(self, metadata):
        p = "deepseek41.engram."
        self.token_map = np.asarray(metadata[p + "token_map"], dtype=np.int64)
        self.rows = tuple(metadata[p + "rows"])
        self.primes = np.asarray(metadata[p + "primes"], dtype=np.int64).reshape(
            len(self.rows), 24
        )
        self.multipliers = np.asarray(
            metadata[p + "multipliers"], dtype=np.int64
        ).reshape(len(self.rows), 4)
        self.vocab = metadata[p + "compressed_vocab_size"]
        self.pad_id = metadata[p + "pad_id"]
        if (
            self.token_map.ndim != 1
            or not len(self.token_map)
            or self.vocab < 1
            or np.any(self.token_map < 0)
            or np.any(self.token_map >= self.vocab)
            or not 0 <= self.pad_id < self.vocab
            or np.any(self.primes < 2)
            or tuple(self.primes.sum(-1)) != self.rows
            or np.any(self.multipliers <= 0)
            or np.any(self.multipliers % 2 != 1)
            or np.any(self.multipliers > np.iinfo(np.int64).max // self.vocab)
        ):
            raise ValueError("invalid sealed V4.1 Engram hash layout")
        self.offsets = np.cumsum(
            np.concatenate(
                (np.zeros((len(self.rows), 1), np.int64), self.primes[:, :-1]), axis=-1
            ),
            axis=-1,
        )

    def __call__(self, ids, tail, mask=None):
        ids, tail = np.asarray(ids), np.asarray(tail)
        if (
            ids.ndim != 1
            or ids.dtype.kind not in "iu"
            or np.any(ids < 0)
            or np.any(ids >= len(self.token_map))
        ):
            raise ValueError("Engram requires in-vocabulary integer token IDs")
        if (
            tail.shape != (3,)
            or tail.dtype.kind not in "iu"
            or np.any(tail < -1)
            or np.any(tail >= self.vocab)
        ):
            raise ValueError("invalid compressed n-gram tail")
        compressed = self.token_map[ids].copy()
        if mask is not None:
            mask = np.asarray(mask)
            if mask.shape != ids.shape or mask.dtype != np.bool_:
                raise ValueError("Engram token mask must be boolean and match IDs")
            compressed[~mask] = -1
        history = np.concatenate((tail[::-1], compressed))
        positions = np.arange(len(ids)) + 3
        blocked = np.zeros(len(ids), bool)
        products = []
        for shift in range(4):
            current = history[positions - shift]
            blocked |= current == -1
            products.append(
                np.where(blocked, self.pad_id, current)[:, None]
                * self.multipliers[:, shift]
            )
        rolling, output = products[0], []
        for shift in range(1, 4):
            rolling = rolling ^ products[shift]
            output.append(
                rolling[..., None] % self.primes[:, (shift - 1) * 8 : shift * 8]
            )
        hashes = np.concatenate(output, axis=-1) + self.offsets
        return hashes.astype(np.uint32), history[-3:][::-1].astype(np.int32).copy()


def decode_engram_rows(raw):
    """264-byte E4M3FN/E8M0 rows -> BF16-rounded FP32, rejecting nonfinite data."""
    raw = np.asarray(raw)
    if raw.dtype != np.uint8 or raw.ndim != 2 or raw.shape[1] != 264:
        raise ValueError("Engram requires packed 264-byte rows")
    codes = raw[:, :256]
    exponent = ((codes >> 3) & 15).astype(np.int32)
    fraction = (codes & 7).astype(np.float32)
    values = np.where(
        exponent == 0, np.ldexp(fraction, -9), np.ldexp(8 + fraction, exponent - 10)
    ).astype(np.float32)
    values = np.copysign(values, np.where(codes & 128, -1.0, 1.0)).astype(np.float32)
    scales = np.repeat(raw[:, 256:], 32, axis=-1)
    if np.any((codes & 127) == 127) or np.any(scales == 255):
        raise ValueError("nonfinite Engram code or scale")
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.ldexp(values, scales.astype(np.int32) - 127)
    bits = np.ascontiguousarray(values).view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    result = rounded.view(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite decoded Engram row")
    return result


@lru_cache(maxsize=1)
def engram_bf16_lut():
    """128KiB format table; same F32 scaling and BF16 ties-to-even as reference."""
    codes = np.arange(256, dtype=np.uint8)[None, :]
    exponent = ((codes >> 3) & 15).astype(np.int32)
    fraction = (codes & 7).astype(np.float32)
    values = np.where(exponent == 0, np.ldexp(fraction, -9),
                      np.ldexp(8 + fraction, exponent - 10)).astype(np.float32)
    values = np.copysign(values, np.where(codes & 128, -1.0, 1.0)).astype(np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.ldexp(values, np.arange(256, dtype=np.int32)[:, None] - 127)
    bits = np.ascontiguousarray(values).view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16
    invalid = (((codes & 127) == 127) | (np.arange(256)[:, None] == 255)
               | ((rounded & 0x7F80) == 0x7F80))
    result = np.where(invalid, 0xFFFF, rounded).astype(np.uint16)
    result.flags.writeable = False
    return result


def decode_engram_bf16(raw):
    raw = np.asarray(raw)
    if raw.dtype != np.uint8 or raw.ndim != 2 or raw.shape[1] != 264:
        raise ValueError("Engram requires packed 264-byte rows")
    result = engram_bf16_lut()[np.repeat(raw[:, 256:], 32, axis=-1), raw[:, :256]]
    if np.any(result == 0xFFFF):
        raise ValueError("nonfinite decoded Engram row")
    return result


AFFINE4_ROW_PLANES = (128, 16, 16)  # U32 codes, BF16 scales, BF16 biases


def decode_affine4_rows(codes, scales, biases, compact=False):
    """MLX affine 4-bit/group-32 rows -> BF16-rounded FP32 (or BF16 bits).

    Codes are packed low nibble first in little-endian U32 words. Each value
    is scale*q+bias in F32 (scale*q is exact: 4-bit by BF16), rounded once
    to BF16 ties-to-even. This is the Metal mx.dequantize law that omlx
    serves with (GPU oracle on real rows of both tables, 2026-11-04). MLX's
    CPU kernel instead rounds after the product and after the sum; it
    differs on about 45% of values and is not used.
    """
    codes, scales, biases = (np.asarray(v) for v in (codes, scales, biases))
    if (codes.dtype != np.uint32 or codes.ndim != 2 or codes.shape[1] != 32
            or scales.dtype != np.uint16 or biases.dtype != np.uint16
            or scales.shape != (codes.shape[0], 8) or biases.shape != scales.shape):
        raise ValueError("Engram affine rows require U32[32] codes and BF16[8] scales/biases")
    shifts = np.arange(0, 32, 4, dtype=np.uint32)
    q = ((codes[:, :, None] >> shifts) & np.uint32(15)).reshape(codes.shape[0], 256).astype(np.float32)
    as_f32 = lambda v: (v.astype(np.uint32) << 16).view(np.float32)

    def bf16(value):
        bits = np.ascontiguousarray(value, np.float32).view(np.uint32)
        return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)).view(np.float32)

    with np.errstate(over="ignore", invalid="ignore"):
        values = bf16(q.reshape(-1, 8, 32)*as_f32(scales)[:, :, None]
                      + as_f32(biases)[:, :, None]).reshape(-1, 256)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite decoded Engram row")
    return (values.view(np.uint32) >> 16).astype(np.uint16) if compact else values


def _read_affine_chunk(fd, bases, rows, compact=False):
    planes = []
    for base, size in zip(bases, AFFINE4_ROW_PLANES):
        data = bytearray()
        for row in rows:
            offset, consumed = base + int(row)*size, 0
            while consumed < size:
                block = os.pread(fd, size-consumed, offset+consumed)
                if not block:
                    raise OSError("short Engram row read")
                data.extend(block)
                consumed += len(block)
        planes.append(bytes(data))
    count = len(rows)
    return decode_affine4_rows(np.frombuffer(planes[0], "<u4").reshape(count, 32),
                               np.frombuffer(planes[1], "<u2").reshape(count, 8),
                               np.frombuffer(planes[2], "<u2").reshape(count, 8), compact)


def _read_chunk(fd, base, rows, compact=False, scale_offset=None):
    result = []
    for row in rows:
        data = bytearray()
        spans = ((base + int(row)*264, 264),) if scale_offset is None else (
            (base + int(row)*256, 256), (scale_offset + int(row)*8, 8))
        for offset, size in spans:
            consumed = 0
            while consumed < size:
                block = os.pread(fd, size-consumed, offset+consumed)
                if not block:
                    raise OSError("short Engram row read")
                data.extend(block)
                consumed += len(block)
        result.append(bytes(data))
    raw = np.frombuffer(b"".join(result), np.uint8).reshape(-1, 264)
    return decode_engram_bf16(raw) if compact else decode_engram_rows(raw)


class EngramRows:
    """One table, one bounded prefetch window, bounded decoded-row LRU.

    Worker futures retain the file descriptor until they drain, even after a
    timeout/unload. Cached rows stay source-precision and are never device arrays.
    A mixed hit/miss lookup reads only its unique misses, then restores order.
    """

    def __init__(
        self,
        path,
        *,
        offset,
        rows,
        expected_stat,
        cache_bytes=8 << 20,
        workers=8,
        max_rows=24576,
        timeout=60.0,
        cache_format="fp32",
        scale_offset=None,
        carry_window=False,
        bias_offset=None,
    ):
        # bias_offset selects the MLX affine 4-bit/group-32 three-plane layout
        # (weight at offset, scales at scale_offset, biases at bias_offset).
        if (
            type(offset) is not int or offset < 0
            or type(carry_window) is not bool
            or (scale_offset is not None and (type(scale_offset) is not int or scale_offset < 0))
            or (bias_offset is not None and (type(bias_offset) is not int or bias_offset < 0
                                             or scale_offset is None))
            or type(rows) is not int or rows < 1
            or type(cache_bytes) is not int or cache_bytes < 0
            or type(workers) is not int or not 1 <= workers <= 32
            or type(max_rows) is not int or max_rows < 1 or cache_format not in ("fp32", "bf16_lut")
            or isinstance(timeout, bool) or not isinstance(timeout, (int,float))
            or not math.isfinite(timeout) or timeout <= 0
        ):
            raise ValueError("invalid Engram I/O bounds")
        self.path, self.offset, self.rows = os.fspath(path), offset, rows
        self.scale_offset = scale_offset
        self.bias_offset = bias_offset
        self.affine = bias_offset is not None
        self.source_row_bytes = sum(AFFINE4_ROW_PLANES) if self.affine else 264
        self.expected_stat = dict(expected_stat)
        self.compact = cache_format == "bf16_lut"
        self.row_bytes = 512 if self.compact else 1024
        self.carry_enabled = carry_window
        self.carry_staging_bound_bytes = max_rows*(self.row_bytes+24) if carry_window else 0
        self.cache_rows, self.max_rows = cache_bytes // self.row_bytes, max_rows
        self.maximum_cache_bytes = self.cache_rows*self.row_bytes
        if self.compact:
            engram_bf16_lut()
        self.workers, self.timeout = workers, timeout
        self._lock = threading.RLock()
        self._cache, self._pending, self._carry = OrderedDict(), None, None
        self._closed = False
        self._parked = False
        self.stats = {
            "lookups": 0,
            "requested_rows": 0,
            "unique_rows": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "bytes_read": 0,
            "evictions": 0,
            "parks": 0,
            "activations": 0,
        }
        self.fd = os.open(
            self.path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            self.verify()
            if self.affine:
                spans = sorted((start, start+rows*size) for start, size in
                               zip((offset, scale_offset, bias_offset), AFFINE4_ROW_PLANES))
                if spans[-1][1] > os.fstat(self.fd).st_size:
                    raise ValueError("Engram table exceeds sealed file")
                if any(a[1] > b[0] for a, b in itertools.pairwise(spans)):
                    raise ValueError("Engram affine planes overlap")
            else:
                end = offset+rows*264 if scale_offset is None else max(offset+rows*256, scale_offset+rows*8)
                if end > os.fstat(self.fd).st_size:
                    raise ValueError("Engram table exceeds sealed file")
                if scale_offset is not None and not (scale_offset >= offset+rows*256 or offset >= scale_offset+rows*8):
                    raise ValueError("Engram weight and scale extents overlap")
            if os.uname().sysname == "Darwin":
                import fcntl

                fcntl.fcntl(
                    self.fd, 48, 1
                )  # F_NOCACHE; hashed reads must not displace resident weights
                fcntl.fcntl(self.fd, 45, 0)  # F_RDAHEAD
            self.pool = ReadPool(
                workers,
                "v41-engram",
                timeout=timeout,
                max_pending=(2 if carry_window else 1) * workers,
                owner=storage_owner("v41-engram", self.path),
            )
        except BaseException:
            os.close(self.fd)
            raise

    def verify(self):
        st = os.fstat(self.fd)
        if (
            not stat.S_ISREG(st.st_mode)
            or file_stamp(st) != self.expected_stat
            or file_stamp(os.stat(self.path, follow_symlinks=False))
            != self.expected_stat
        ):
            raise ValueError("Engram source changed after sealing")

    def prefetch(self, ids):
        with self._lock:
            if self._closed:
                raise ValueError("Engram reader is closed")
            if self._parked:
                raise ValueError("Engram reader is parked")
            if self.pool.failure is not None:
                raise ReadTimeout("Engram reader failed; reload required: "+self.pool.failure)
            values = np.asarray(ids)
            if (
                values.dtype.kind not in "iu"
                or values.size > self.max_rows
                or np.any(values < 0)
                or np.any(values >= self.rows)
            ):
                raise ValueError("Engram lookup exceeds row identity/budget")
            values = values.astype(np.int64, copy=True)
            if self._pending is not None:
                if not np.array_equal(values, self._pending[0]):
                    raise ValueError("Engram prefetch window is already owned")
                return
            if self._carry is not None:
                carry, self._carry = self._carry, None
                if np.array_equal(values, carry[0]):
                    self.verify()
                    self._pending = carry
                    self.stats["carry_hits"] = self.stats.get("carry_hits", 0) + 1
                    return
                self._drain_window(carry)
                self.stats["carry_discards"] = self.stats.get("carry_discards", 0) + 1
            self.verify()
            unique = np.unique(values)
            hits = {
                int(row): self._cache[int(row)]
                for row in unique
                if int(row) in self._cache
            }
            # Touch hits before installing misses. Never copy/reinsert a hit
            # evicted by this over-capacity lookup: that churns the entire LRU.
            for row in hits:
                self._cache.move_to_end(row)
            missing = np.asarray(
                [row for row in unique if int(row) not in hits], np.int64
            )
            futures = []
            try:
                for group in np.array_split(
                    missing, min(self.workers, max(1, len(missing)))
                ):
                    if len(group):
                        futures.append(
                            (
                                group,
                                self.pool.submit(
                                    _read_affine_chunk, self.fd,
                                    (self.offset, self.scale_offset, self.bias_offset), group, self.compact,
                                ) if self.affine else self.pool.submit(
                                    _read_chunk, self.fd, self.offset, group, self.compact, self.scale_offset
                                ),
                            )
                        )
            except BaseException as error:
                self.pool.fail(error)
                raise
            self._pending = values, unique, hits, missing, futures

    def prefetch_next(self, ids):
        """Own one copied next-chunk window, independently of the active read.

        Never steal a current window or publish next rows in its place. The
        queue admits at most two worker sets; stale carries drain before reuse.
        """
        with self._lock:
            if not self.carry_enabled:
                raise ValueError("Engram carry window was not admitted")
            active, self._pending = self._pending, None
            try:
                self.prefetch(ids)
                self._carry, self._pending = self._pending, None
                self.stats["carry_windows"] = self.stats.get("carry_windows", 0) + 1
            finally:
                self._pending = active

    def _drain_window(self, pending):
        if pending is None:
            return
        for _, future in pending[4]:
            try:
                future.result(timeout=self.timeout)
            except FutureTimeout:
                self.pool.fail(ReadTimeout("Engram prefetch exceeded its deadline"))
            except BaseException as error:
                self.pool.fail(error)

    def discard_prefetch(self):
        """Release a window a failed forward opened but never read.

        In-flight chunks still occupy the bounded queue, so wait for them and
        drop their rows. A failed or late read still poisons the pool exactly
        as read() would; the caller's original error is never replaced.
        """
        with self._lock:
            pending, carry = self._pending, self._carry
            self._pending = self._carry = None
            if pending is None and carry is None:
                return False
            self._drain_window(pending)
            self._drain_window(carry)
            self.stats["discarded_prefetches"] = self.stats.get("discarded_prefetches", 0) + 1
            return True

    def read(self, ids):
        with self._lock:
            self.prefetch(ids)
            values, unique, found, missing, futures = self._pending
            try:
                for group, future in futures:
                    decoded = future.result(timeout=self.timeout)
                    found.update(zip(map(int, group), decoded))
                self.verify()
            except FutureTimeout as error:
                failure = ReadTimeout("Engram prefetch exceeded its deadline")
                self.pool.fail(failure)
                raise failure from error
            except BaseException as error:
                self.pool.fail(error)
                raise
            finally:
                self._pending = None
            output = (
                np.stack([found[int(row)] for row in values.flat]).reshape(
                    *values.shape, 256
                )
                if values.size
                else np.empty((*values.shape, 256), np.float32)
            )
            if self.compact:
                output = (output.astype(np.uint32) << 16).view(np.float32)
            for row in missing:
                key = int(row)
                if self.cache_rows:
                    self._cache[key] = found[key].copy()
                    self._cache.move_to_end(key)
                    while len(self._cache) > self.cache_rows:
                        self._cache.popitem(last=False)
                        self.stats["evictions"] += 1
            self.stats["lookups"] += 1
            self.stats["requested_rows"] += values.size
            self.stats["unique_rows"] += len(unique)
            self.stats["cache_hits"] += len(unique) - len(missing)
            self.stats["cache_misses"] += len(missing)
            self.stats["bytes_read"] += self.source_row_bytes * len(missing)
            return output

    def diagnostics(self):
        with self._lock:
            return {
                **self.stats,
                "storage": ("pread-affine4-g32-bf16-three-planes" if self.affine
                            else "pread-e4m3-e8m0-row264" if self.scale_offset is None
                            else "pread-e4m3-e8m0-split-planes"),
                "table_bytes": self.rows * self.source_row_bytes,
                "resident_table_bytes": 0,
                "cached_bytes": len(self._cache) * self.row_bytes,
                "cache_budget_bytes": self.cache_rows * self.row_bytes,
                "cache_format": "bf16_lut" if self.compact else "fp32",
                "format_lut_bytes": 131072 if self.compact else 0,
                "max_lookup_rows": self.max_rows,
                "prefetch_pending": self._pending is not None,
                "carry_pending": self._carry is not None,
                "prefetch_window_limit": 2 if self.carry_enabled else 1,
                "carry_staging_bound_bytes": self.carry_staging_bound_bytes,
                "parked": self._parked,
                "reserved_bytes": self.reserved_bytes(),
                "resume_reservation_bytes": self.resume_reservation_bytes(),
                "read_failure": self.pool.failure,
                "read_inflight": self.pool.counters()["weight_read_inflight"],
            }

    def resume_reservation_bytes(self):
        return self.cache_rows*self.row_bytes + self.carry_staging_bound_bytes

    def reserved_bytes(self):
        return 0 if self._parked else self.resume_reservation_bytes()

    def park(self):
        with self._lock:
            if self._closed or self._pending is not None:
                raise ValueError("cannot park a closed/owned Engram read window")
            # A cancelled request may never consume its known-tail carry.
            # Parking an otherwise idle provider drains and retires it.
            carry,self._carry=self._carry,None
            self._drain_window(carry)
            if self.pool.counters()["weight_read_inflight"]:
                raise ValueError("cannot park while Engram reads remain owned")
            if not self._parked:
                self._cache.clear()
                self._parked = True
                self.stats["parks"] += 1
            return self.diagnostics()

    def activate(self):
        with self._lock:
            if self._closed or self._pending is not None or self._carry is not None:
                raise ValueError("cannot activate a closed/owned Engram read window")
            if self.pool.failure is not None:
                raise ReadTimeout("Engram reader failed; reload required: "+self.pool.failure)
            self.verify()
            if self._parked:
                self._parked = False
                self.stats["activations"] += 1
            return self.diagnostics()

    def resize_cache_budget(self, budget):
        with self._lock:
            if self._closed or self._pending is not None or self._carry is not None:
                raise ValueError("cannot resize an owned/closed Engram read window")
            if type(budget) is not int or not 0 <= budget <= self.maximum_cache_bytes:
                raise ValueError("Engram cache budget outside admitted bounds")
            self.cache_rows = budget//self.row_bytes
            while len(self._cache) > self.cache_rows:
                self._cache.popitem(last=False)
                self.stats["evictions"] += 1
            return self.cache_rows*self.row_bytes

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._cache.clear()
            self._pending = None
            self._carry = None
            fd = self.fd
            self.pool.close(CloseAction(lambda: os.close(fd)))

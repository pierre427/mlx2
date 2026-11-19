# SPDX-License-Identifier: Apache-2.0
# Original mlx2 code; pattern follows deepseek_v41_engram.EngramRows.
"""SSD row reader for Hy4-flash Engram tables (PLE-style offload, the default).

Each Engram layer's table (``engram_layer.multi_head_embedding.embedding``,
about 165M rows of 96 values, 31.7 GB in bf16) stays in its safetensors
shard.  A step reads only the 16 hashed rows per token: ``os.pread`` of one
row's bytes at ``data_start + row * row_bytes``, de-duplicated, through a
bounded LRU of decoded rows.  Nothing here touches the GPU; rows return as
host fp32 and the model moves them to the device once per layer and step.

Engram tables are hashed per-layer embeddings, so they take mlx2's PLE
offload I/O contract (``qwen4_ple_nvme.FileBackedShardedEmbedding``): reads
run on an owned ``io_ownership.ReadPool`` (daemon workers, a deadline per
read, storage-owner quarantine of a hung read, descriptor closed by the pool
once its reads drain), decode-sized and prefill-sized worker counts split at
``PREFILL_ROW_THRESHOLD`` rows, and a bounded LRU of decoded rows.  Offload is
the default and only mode for real checkpoints (Pierre, 2026-11-18: always
PLE-offload by default); the resident reader exists for tests.

Table formats (``TABLE_FORMATS``; converter ``--engram-formats``, adapter
policy ``engram_table_format``, default ``bf16``).  The two packed formats
are one ``U8 [rows, row_bytes]`` tensor per table, so a row is still one
``pread``; the shard's ``__metadata__`` declares the format and its layout
and the reader refuses a format it was not asked for.  Rows are decoded on
the PLE pool workers.

* ``bf16``: the checkpoint's table (BF16, or F32/F16 in tests), unchanged.
* ``fp8_e4m3_row``: per row, ``dim`` E4M3 codes then one scale in the
  declared dtype (BF16 for a BF16 source: 98 bytes for 96 values).  Written
  with ``hy_v4_flash.fp8_quantize``, so ``codes * scale`` equals the
  ``transformers`` profile's per-vector round trip of the bf16 row exactly
  when the model runs in that dtype (``fp8_round_trip_dtype``).
* ``affine4_g32``: MLX affine 4-bit, group 32: per row ``dim / 8`` little-
  endian U32 code words (low nibble first), then ``dim / 32`` scales, then
  ``dim / 32`` biases in the declared dtype (60 bytes for 96 values).  Each
  value is ``scale * q + bias`` in fp32, the law of ``deepseek_v41_engram.
  decode_affine4_rows`` (the model's cast to bf16 then rounds it once, as
  Metal ``mx.dequantize`` does).  Approximate and unqualified.
"""

from __future__ import annotations

import json
import os
import struct
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

import time
from concurrent.futures import TimeoutError as FutureTimeout

from ..io_ownership import CloseAction, ReadPool, ReadTimeout, call_io, storage_owner

# PLE offload sizing (qwen4_ple_nvme): decode/prefill worker pools, deadline.
DECODE_WORKERS = 16
PREFILL_WORKERS = 64
PREFILL_ROW_THRESHOLD = 512
READ_TIMEOUT_SECONDS = 60.0

_DTYPES = {"BF16": (2, "bf16"), "F32": (4, "<f4"), "F16": (2, "<f2"), "F8_E4M3": (1, "e4m3"),
           "U8": (1, "u1")}

FORMAT_KEY = "mlx2_engram_format"
FP8_ROW = "fp8_e4m3_row"
AFFINE4 = "affine4_g32"
TABLE_FORMATS = ("bf16", FP8_ROW, AFFINE4)
AFFINE4_GROUP = 32
# Declared exactness of each format, relative to reading the bf16 table.
FORMAT_RELATION = {
    "bf16": "source",
    FP8_ROW: ("exact vs the transformers-profile E4M3 round trip in its declared dtype "
              "(CPU-verified; Metal parity unverified)"),
    AFFINE4: "approximate (affine 4-bit, group 32)",
}
_SCALE_BYTES = {"BF16": 2, "F32": 4, "F16": 2}


def packed_row_bytes(table_format: str, dim: int, scale_dtype: str) -> int:
    """Bytes of one packed row of ``table_format`` (``dim`` values)."""
    width = _SCALE_BYTES[scale_dtype]
    if table_format == FP8_ROW:
        return dim + width
    if table_format == AFFINE4:
        if dim % AFFINE4_GROUP:
            raise ValueError(f"affine4_g32 needs a row width divisible by {AFFINE4_GROUP}")
        return dim // 2 + 2 * width * (dim // AFFINE4_GROUP)
    raise ValueError(f"{table_format} is not a packed Engram format")


def _as_f32(raw: np.ndarray, dtype: str) -> np.ndarray:
    """Little-endian scale/bias bytes ``[..., k*width]`` -> fp32 ``[..., k]``."""
    raw = np.ascontiguousarray(raw)
    if dtype == "BF16":
        return (raw.view("<u2").astype(np.uint32) << 16).view(np.float32)
    return raw.view("<f4" if dtype == "F32" else "<f2").astype(np.float32)


def decode_fp8_row(raw: np.ndarray, dim: int, scale_dtype: str) -> np.ndarray:
    """Packed ``fp8_e4m3_row`` rows ``uint8 [n, dim + w]`` -> fp32 ``[n, dim]``.

    For BF16/F16 scales ``e4m3 * scale`` is exact in fp32 (a 4-bit by an at
    most 11-bit significand), so the model's cast to its dtype rounds the
    product once, as MLX's ``from_fp8(q) * scale`` does; for F32 scales the
    product rounds once in fp32, which is again what MLX computes.
    Nonfinite codes or results are refused.
    """
    raw = np.asarray(raw, np.uint8)
    if raw.ndim != 2 or raw.shape[1] != packed_row_bytes(FP8_ROW, dim, scale_dtype):
        raise ValueError("fp8_e4m3_row rows have the wrong width")
    codes = raw[:, :dim]
    if np.any((codes & 127) == 127):
        raise ValueError("nonfinite Engram E4M3 code")
    scale = _as_f32(raw[:, dim:], scale_dtype)
    with np.errstate(over="ignore", invalid="ignore"):
        values = e4m3_decode(codes) * scale
    if not np.isfinite(values).all():
        raise ValueError("nonfinite decoded Engram row")
    return values.astype(np.float32, copy=False)


def decode_affine4_row(raw: np.ndarray, dim: int, scale_dtype: str) -> np.ndarray:
    """Packed ``affine4_g32`` rows ``uint8 [n, row_bytes]`` -> fp32 ``[n, dim]``.

    Same law as ``deepseek_v41_engram.decode_affine4_rows`` (whose layout is
    fixed at 256 values in three planes): ``scale * q`` exact, plus ``bias``
    rounded once in fp32.
    """
    raw = np.asarray(raw, np.uint8)
    if raw.ndim != 2 or raw.shape[1] != packed_row_bytes(AFFINE4, dim, scale_dtype):
        raise ValueError("affine4_g32 rows have the wrong width")
    n, groups, width = raw.shape[0], dim // AFFINE4_GROUP, _SCALE_BYTES[scale_dtype]
    words = np.ascontiguousarray(raw[:, : dim // 2]).view("<u4")
    shifts = np.arange(0, 32, 4, dtype=np.uint32)
    q = ((words[:, :, None] >> shifts) & np.uint32(15)).reshape(n, groups, AFFINE4_GROUP).astype(np.float32)
    at = dim // 2
    scales = _as_f32(raw[:, at: at + groups * width], scale_dtype)
    biases = _as_f32(raw[:, at + groups * width:], scale_dtype)
    with np.errstate(over="ignore", invalid="ignore"):
        values = (q * scales[:, :, None] + biases[:, :, None]).reshape(n, dim)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite decoded Engram row")
    return values.astype(np.float32, copy=False)


def table_layout(path, name) -> dict:
    """Declared layout of Engram table ``name`` in ``path`` (no data read).

    ``{"format", "start", "rows", "dim", "row_bytes", "dtype", "scale_dtype"}``;
    a table without format metadata is ``bf16`` (the checkpoint's own).
    """
    header = safetensors_header(path)
    meta = header.get("__metadata__") or {}
    start, dtype, shape = safetensors_tensor(path, name)
    if len(shape) != 2:
        raise ValueError("Engram table must be 2-D")
    table_format = meta.get(FORMAT_KEY, "bf16")
    if table_format not in TABLE_FORMATS:
        raise ValueError(f"unknown Engram table format {table_format!r}")
    if table_format == "bf16":
        if dtype in ("U8", "F8_E4M3"):
            # A pre-quantized table needs its row scales; only the packed
            # fp8_e4m3_row layout carries them.
            raise ValueError(f"{dtype} Engram tables need declared row scales (fp8_e4m3_row)")
        return {"format": "bf16", "start": start, "rows": shape[0], "dim": shape[1],
                "row_bytes": _DTYPES[dtype][0] * shape[1], "dtype": dtype, "scale_dtype": None}
    scale_dtype = meta.get("mlx2_engram_scale_dtype")
    if dtype != "U8" or scale_dtype not in _SCALE_BYTES:
        raise ValueError(f"{table_format} table must be U8 with a declared scale dtype")
    dim = int(meta.get("mlx2_engram_dim", -1))
    if dim < 1 or packed_row_bytes(table_format, dim, scale_dtype) != shape[1]:
        raise ValueError(f"{table_format} row width disagrees with its declared dim")
    if table_format == AFFINE4 and int(meta.get("mlx2_engram_group_size", -1)) != AFFINE4_GROUP:
        raise ValueError("affine4_g32 table must declare group size 32")
    if table_format == FP8_ROW and meta.get("mlx2_engram_round_trip_dtype") != scale_dtype:
        raise ValueError("fp8_e4m3_row table must declare its round trip in its scale dtype")
    return {"format": table_format, "start": start, "rows": shape[0], "dim": dim,
            "row_bytes": shape[1], "dtype": dtype, "scale_dtype": scale_dtype}


def safetensors_header(path) -> dict:
    with open(path, "rb") as handle:
        (size,) = struct.unpack("<Q", handle.read(8))
        if not 2 <= size <= 1 << 28:
            raise ValueError("implausible safetensors header")
        header = json.loads(handle.read(size))
    header["__header_bytes__"] = 8 + size
    return header


def safetensors_tensor(path, name):
    """``(data_start, dtype, shape)`` of ``name`` in a safetensors shard."""
    header = safetensors_header(path)
    size = header.pop("__header_bytes__") - 8
    entry = header.get(name)
    if not isinstance(entry, dict):
        raise KeyError(f"{name} not in {path}")
    begin, end = entry["data_offsets"]
    dtype, shape = entry["dtype"], tuple(int(s) for s in entry["shape"])
    if dtype not in _DTYPES:
        raise ValueError(f"unsupported Engram table dtype {dtype}")
    width = _DTYPES[dtype][0]
    if end - begin != width * int(np.prod(shape)):
        raise ValueError("safetensors byte range disagrees with shape")
    return 8 + size + begin, dtype, shape


def _decode(raw: bytes, dtype: str, count: int, dim: int) -> np.ndarray:
    if dtype == "BF16":
        bits = np.frombuffer(raw, "<u2").astype(np.uint32) << 16
        return bits.view(np.float32).reshape(count, dim)
    if dtype == "F8_E4M3":
        codes = np.frombuffer(raw, np.uint8)
        return e4m3_decode(codes).reshape(count, dim)
    return np.frombuffer(raw, _DTYPES[dtype][1]).astype(np.float32).reshape(count, dim)


def e4m3_decode(codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes, np.uint8)
    exponent = ((codes >> 3) & 15).astype(np.int32)
    fraction = (codes & 7).astype(np.float32)
    values = np.where(exponent == 0, np.ldexp(fraction, -9),
                      np.ldexp(8 + fraction, exponent - 10)).astype(np.float32)
    values = np.where((codes & 127) == 127, np.nan, values)
    return np.copysign(values, np.where(codes & 128, -1.0, 1.0)).astype(np.float32)


class EngramTableRows:
    """Read rows of one ``[rows, dim]`` table straight from its shard (PLE offload)."""

    def __init__(self, path, name, *, cache_rows=1 << 16, decode_workers=DECODE_WORKERS,
                 prefill_workers=PREFILL_WORKERS, timeout=READ_TIMEOUT_SECONDS, table_format="bf16"):
        self.path = os.fspath(path)
        self.name = name
        self.owner = storage_owner("hy4-engram", self.path)
        if table_format not in TABLE_FORMATS:
            raise ValueError(f"Engram table format must be one of {TABLE_FORMATS}")
        layout = call_io("hy4-engram-header", table_layout, self.path, name, owner=self.owner)
        if layout["format"] != table_format:
            raise ValueError(f"Engram table {name} is {layout['format']}, policy asked for {table_format}")
        self.format = table_format
        self.start, self.dtype = layout["start"], layout["dtype"]
        self.scale_dtype = layout["scale_dtype"]
        self.rows, self.dim = layout["rows"], layout["dim"]
        self.row_bytes = layout["row_bytes"]
        self.decode_workers, self.prefill_workers = int(decode_workers), int(prefill_workers)
        self.timeout = float(timeout)
        self._lock = threading.Lock()
        self._closed = False
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.cache_rows = int(cache_rows)
        self.stats = {"lookups": 0, "requested_rows": 0, "unique_rows": 0,
                      "cache_hits": 0, "bytes_read": 0, "prefill_reads": 0, "decode_reads": 0}
        self._pool = ReadPool(max(self.decode_workers, self.prefill_workers), "hy4-engram",
                              timeout=self.timeout, owner=self.owner)
        # The descriptor is owned by a holder from the moment it exists, so a
        # failed fstat or a late open after the deadline still closes it (the
        # PLE pattern, qwen4_ple_nvme._open_resources).
        holder = {"fd": None}
        expected = self.start + self.rows * self.row_bytes

        def open_file():
            holder["fd"] = os.open(self.path, os.O_RDONLY)
            size = os.fstat(holder["fd"]).st_size
            if size < expected:
                raise ValueError(f"Engram shard has {size} bytes, table needs {expected}")
            return holder["fd"]

        def close_holder():
            if holder["fd"] is not None:
                os.close(holder["fd"])
                holder["fd"] = None

        self._close_action = CloseAction(close_holder)
        try:
            self._fd = self._pool.call(open_file)
        except BaseException:
            self._pool.close(self._close_action)
            self._pool = None
            raise

    def _read_chunk(self, fd, rows):
        out = []
        for row in rows:
            offset = self.start + int(row) * self.row_bytes
            data = bytearray()
            while len(data) < self.row_bytes:
                block = os.pread(fd, self.row_bytes - len(data), offset + len(data))
                if not block:
                    raise OSError("short Engram row read")
                data.extend(block)
            out.append(bytes(data))
        raw = b"".join(out)
        if self.format == FP8_ROW:
            return decode_fp8_row(np.frombuffer(raw, np.uint8).reshape(len(rows), -1), self.dim, self.scale_dtype)
        if self.format == AFFINE4:
            return decode_affine4_row(np.frombuffer(raw, np.uint8).reshape(len(rows), -1), self.dim, self.scale_dtype)
        return _decode(raw, self.dtype, len(rows), self.dim)

    @property
    def approximate(self) -> bool:
        return self.format == AFFINE4

    @property
    def fp8_round_trip_dtype(self):
        """MLX dtype in which this table's rows already took the E4M3 round trip."""
        if self.format != FP8_ROW:
            return None
        import mlx.core as mx

        return {"BF16": mx.bfloat16, "F32": mx.float32, "F16": mx.float16}[self.scale_dtype]

    def _fetch(self, missing):
        if not missing:
            return {}
        with self._lock:
            if self._closed:
                raise RuntimeError("Engram table reader is closed")
            pool, fd = self._pool, self._fd
        prefill = len(missing) >= PREFILL_ROW_THRESHOLD
        workers = self.prefill_workers if prefill else self.decode_workers
        chunks = [c for c in np.array_split(np.asarray(missing, np.int64), min(workers, len(missing))) if c.size]
        futures = [pool.submit(self._read_chunk, fd, chunk) for chunk in chunks]
        deadline = time.monotonic() + self.timeout  # one deadline for the whole lookup
        out = {}
        try:
            for chunk, future in zip(chunks, futures):
                values = future.result(timeout=max(0.0, deadline - time.monotonic()))
                for row, value in zip(chunk.tolist(), values):
                    out[row] = value.copy()  # a cached row must not pin its chunk
        except FutureTimeout:
            failure = ReadTimeout("Engram row read exceeded its deadline")
            pool.fail(failure)
            raise failure from None
        with self._lock:
            self.stats["prefill_reads" if prefill else "decode_reads"] += 1
        return out

    def read_host(self, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, np.int64).reshape(-1)
        if rows.size and (rows.min() < 0 or rows.max() >= self.rows):
            raise IndexError("Engram row outside its table")
        unique = np.unique(rows)
        out = {}
        missing = []
        with self._lock:
            for row in unique.tolist():
                hit = self._cache.get(row)
                if hit is None:
                    missing.append(row)
                else:
                    self._cache.move_to_end(row)
                    out[row] = hit
        out.update(self._fetch(missing))
        with self._lock:
            for row in missing:
                self._cache[row] = out[row]
            while len(self._cache) > self.cache_rows:
                self._cache.popitem(last=False)
            self.stats["lookups"] += 1
            self.stats["requested_rows"] += int(rows.size)
            self.stats["unique_rows"] += int(unique.size)
            self.stats["cache_hits"] += int(unique.size) - len(missing)
            self.stats["bytes_read"] += len(missing) * self.row_bytes
        if not rows.size:
            return np.zeros((0, self.dim), np.float32)
        return np.stack([out[r] for r in rows.tolist()])

    def read(self, rows: np.ndarray):
        import mlx.core as mx

        return mx.array(self.read_host(rows))

    def diagnostics(self):
        return {"path": self.path, "tensor": self.name, "rows": self.rows, "dim": self.dim,
                "dtype": self.dtype, "format": self.format, "scale_dtype": self.scale_dtype,
                "row_bytes": self.row_bytes, "table_bytes": self.rows * self.row_bytes,
                "approximate": self.approximate, "relation": FORMAT_RELATION[self.format],
                "qualified": False, "default": self.format == "bf16",
                "cache_rows": self.cache_rows, "offload": "ple_file_backed",
                "decode_workers": self.decode_workers, "prefill_workers": self.prefill_workers,
                "pool": self._pool.counters() if self._pool is not None else None,
                "closed": self._closed, **self.stats}

    def close(self):
        lock = getattr(self, "_lock", None)
        if lock is None:
            return
        with lock:
            if self._closed:
                return
            self._closed = True
            pool, self._pool = self._pool, None
            self._fd = None
        if pool is not None:
            # The holder closes the descriptor once in-flight reads drain.
            pool.close(self._close_action)


def table_shards(weight_map: dict, formats: dict | None, table_format: str) -> dict:
    """Tensor name -> shard of every Engram table in ``table_format``.

    ``formats`` is the converter's ``config["mlx2_engram_formats"]``
    (format -> {tensor: shard}); an artifact without it carries only its
    checkpoint tables, which are ``bf16`` and listed in the weight map.
    """
    if table_format not in TABLE_FORMATS:
        raise ValueError(f"Engram table format must be one of {TABLE_FORMATS}")
    if formats is None:
        if table_format != "bf16":
            raise ValueError(f"artifact carries no {table_format} Engram tables (only bf16)")
        return weight_map
    if table_format not in formats:
        raise ValueError(f"artifact carries no {table_format} Engram tables "
                         f"(it has {sorted(formats)}); select one with engram_table_format")
    return formats[table_format]


def open_engram_tables(root, weight_map: dict, layer_ids: dict, *, table_format="bf16",
                       formats=None, **kwargs) -> dict:
    """Decoder layer index -> ``EngramTableRows`` for every Engram layer."""
    out = {}
    shards = table_shards(weight_map, formats, table_format)
    try:
        for index in layer_ids:
            name = f"model.layers.{index}.engram_layer.multi_head_embedding.embedding.weight"
            shard = shards.get(name)
            if shard is None:
                raise ValueError(f"Engram table {name} ({table_format}) is not in the artifact")
            out[index] = EngramTableRows(Path(root) / shard, name, table_format=table_format, **kwargs)
    except BaseException:
        for reader in out.values():
            reader.close()
        raise
    return out

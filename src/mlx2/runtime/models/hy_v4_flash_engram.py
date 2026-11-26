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
* ``iq4_nl``: ggml IQ4_NL (GGUF tensor type 20), read from a GGUF file that
  the config binds per Engram layer (``text_config.engram_gguf_tables``:
  ``{layer_id: {"file", "tensor"}}``; the third-party Nace.AI pack's layout,
  provenance ``hy-v4-flash.json``).  A row of ``dim`` values is ``dim / 32``
  blocks of 18 bytes: one little-endian F16 scale ``d``, then 16 bytes of
  4-bit codes, value ``j`` the low nibble of byte ``j`` and value ``j + 16``
  its high nibble, each an index into the fixed 16-level codebook
  ``IQ4NL_LEVELS``; a value is ``d * level`` in fp32 (ggml
  ``dequantize_row_iq4_nl``).  The table carries no E4M3 round trip: the
  ``transformers`` profile still applies it at runtime after the cast to
  the activation dtype.  Approximate and unqualified.
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
from ..read_advise import ReadAdvisor
from ..technologies.embedding_tables import (Affine4RowSpec, decode_packed_affine4, pread_exact,
                                             read_safetensors_header_file, scale_bytes_to_f32)

# PLE offload sizing (qwen4_ple_nvme): decode/prefill worker pools, deadline.
DECODE_WORKERS = 16
PREFILL_WORKERS = 64
PREFILL_ROW_THRESHOLD = 512
READ_TIMEOUT_SECONDS = 60.0
# Opt-in read-ahead hint for a verify round's Engram rows (omlx #5274 /
# #5464 analogue; runtime/read_advise.py).  ``Model.engram_inputs`` hashes
# every Engram layer of the round first, then hints the rows of every layer
# after the first one it reads, so those layers' SSD reads are already queued
# while the first layer is read.  Default OFF, unqualified: on PLE reads the
# same hint measured slower beside a multi-worker pool (recon 2026-11-13) and
# no Hy4 A/B exists yet.  Decode/verify-sized lookups only (fewer than
# PREFILL_ROW_THRESHOLD missing rows); prefill keeps its 64-worker pool.
READ_ADVISE_ENV = "MLX_HY4_ENGRAM_READ_ADVISE"

_DTYPES = {"BF16": (2, "bf16"), "F32": (4, "<f4"), "F16": (2, "<f2"), "F8_E4M3": (1, "e4m3"),
           "U8": (1, "u1")}

FORMAT_KEY = "mlx2_engram_format"
FP8_ROW = "fp8_e4m3_row"
AFFINE4 = "affine4_g32"
IQ4NL = "iq4_nl"
TABLE_FORMATS = ("bf16", FP8_ROW, AFFINE4, IQ4NL)
# Formats scripts/convert_hy_v4_flash_mlx.py writes (safetensors shards);
# ``iq4_nl`` is read from a GGUF file only.
WRITABLE_FORMATS = ("bf16", FP8_ROW, AFFINE4)
AFFINE4_GROUP = 32
# ggml ``kvalues_iq4nl``: the fixed IQ4_NL codebook.
IQ4NL_LEVELS = np.array([-127, -104, -83, -65, -49, -35, -22, -10,
                         1, 13, 25, 38, 53, 69, 89, 113], dtype=np.float32)
IQ4NL_BLOCK = 32
IQ4NL_BLOCK_BYTES = 18
GGUF_TYPE_IQ4_NL = 20
# Declared exactness of each format, relative to reading the bf16 table.
FORMAT_RELATION = {
    "bf16": "source",
    FP8_ROW: ("exact vs the transformers-profile E4M3 round trip in its declared dtype "
              "(CPU-verified; Metal parity unverified)"),
    AFFINE4: "approximate (affine 4-bit, group 32)",
    IQ4NL: "approximate (ggml IQ4_NL, 32-value blocks; third-party GGUF table)",
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
    if table_format == IQ4NL:
        if dim % IQ4NL_BLOCK:
            raise ValueError(f"iq4_nl needs a row width divisible by {IQ4NL_BLOCK}")
        return dim // IQ4NL_BLOCK * IQ4NL_BLOCK_BYTES
    raise ValueError(f"{table_format} is not a packed Engram format")


def _as_f32(raw: np.ndarray, dtype: str) -> np.ndarray:
    """Little-endian scale/bias bytes ``[..., k*width]`` -> fp32 ``[..., k]``."""
    return scale_bytes_to_f32(raw, dtype)


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
    spec = Affine4RowSpec(dim=dim, group_size=AFFINE4_GROUP, scale_dtype=scale_dtype,
                          rounding="none", nonfinite="refuse", output="f32")
    return decode_packed_affine4(spec, raw)


def decode_iq4nl_row(raw: np.ndarray, dim: int) -> np.ndarray:
    """Packed ``iq4_nl`` rows ``uint8 [n, dim / 32 * 18]`` -> fp32 ``[n, dim]``.

    ggml ``dequantize_row_iq4_nl``: per 32-value block an F16 scale ``d``,
    then 16 code bytes; value ``j`` is ``d * LEVELS[b[j] & 15]`` and value
    ``j + 16`` is ``d * LEVELS[b[j] >> 4]``.  The product of an F16 scale
    and a 7-bit level is exact in fp32.  Nonfinite results are refused.
    """
    raw = np.asarray(raw, np.uint8)
    if raw.ndim != 2 or raw.shape[1] != packed_row_bytes(IQ4NL, dim, "F16"):
        raise ValueError("iq4_nl rows have the wrong width")
    blocks = raw.reshape(raw.shape[0], dim // IQ4NL_BLOCK, IQ4NL_BLOCK_BYTES)
    scales = np.ascontiguousarray(blocks[:, :, :2]).view("<f2").astype(np.float32)  # [n, b, 1]
    codes = blocks[:, :, 2:]
    codes = np.concatenate((codes & 15, codes >> 4), axis=-1)
    with np.errstate(over="ignore", invalid="ignore"):
        values = (IQ4NL_LEVELS[codes] * scales).reshape(raw.shape[0], dim)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite decoded Engram row")
    return values.astype(np.float32, copy=False)


GGUF_MAGIC = b"GGUF"
# GGUF metadata value types -> fixed byte width (8 string, 9 array: variable).
_GGUF_SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def is_gguf(path) -> bool:
    with open(path, "rb") as handle:
        return handle.read(4) == GGUF_MAGIC


def gguf_tensors(path) -> dict:
    """``{name: (absolute data offset, ggml type, dims)}`` of a GGUF v2/v3 file.

    Reads the header only.  ``dims`` is in GGUF order (``ne[0]`` first, the
    innermost, contiguous dimension).  ``general.alignment`` sets the data
    alignment (default 32).
    """
    with open(path, "rb") as handle:
        size = os.fstat(handle.fileno()).st_size

        def read(n):
            data = handle.read(n)
            if len(data) != n:
                raise ValueError("truncated GGUF header")
            return data

        def u32():
            return struct.unpack("<I", read(4))[0]

        def u64():
            return struct.unpack("<Q", read(8))[0]

        def string():
            n = u64()
            if n > 1 << 16:
                raise ValueError("implausible GGUF string")
            return read(n).decode("utf-8")

        def value(kind, depth=0):
            if kind in _GGUF_SCALAR:
                raw = read(_GGUF_SCALAR[kind])
                return struct.unpack({1: "<B", 2: "<H", 4: "<I", 8: "<Q"}[len(raw)], raw)[0] \
                    if kind in (0, 2, 4, 10) else raw
            if kind == 8:
                return string()
            if kind == 9 and depth == 0:
                inner, count = u32(), u64()
                if count > 1 << 24:
                    raise ValueError("implausible GGUF array")
                return [value(inner, 1) for _ in range(count)]
            raise ValueError(f"unsupported GGUF metadata type {kind}")

        if read(4) != GGUF_MAGIC:
            raise ValueError("not a GGUF file")
        version = u32()
        if version not in (2, 3):
            raise ValueError(f"unsupported GGUF version {version}")
        n_tensors, n_kv = u64(), u64()
        if n_tensors > 1 << 16 or n_kv > 1 << 16:
            raise ValueError("implausible GGUF header counts")
        alignment = 32
        for _ in range(n_kv):
            key = string()
            kind = u32()
            got = value(kind)
            if key == "general.alignment":
                if kind != 4 or not got or got & (got - 1):
                    raise ValueError("GGUF general.alignment must be a power-of-two u32")
                alignment = got
        infos = {}
        for _ in range(n_tensors):
            name = string()
            ndim = u32()
            if not 1 <= ndim <= 4:
                raise ValueError("GGUF tensor rank out of range")
            dims = tuple(u64() for _ in range(ndim))
            kind, offset = u32(), u64()
            if name in infos:
                raise ValueError(f"duplicate GGUF tensor {name}")
            infos[name] = (kind, dims, offset)
        data0 = -(-handle.tell() // alignment) * alignment
    out = {}
    for name, (kind, dims, offset) in infos.items():
        if offset % alignment:
            raise ValueError(f"GGUF tensor {name} is misaligned")
        if data0 + offset > size:
            raise ValueError(f"GGUF tensor {name} starts past the end of the file")
        out[name] = (data0 + offset, kind, dims)
    return out


def gguf_table_layout(path, name) -> dict:
    """Layout of an IQ4_NL Engram table ``name`` in GGUF file ``path``."""
    tensors = gguf_tensors(path)
    entry = tensors.get(name)
    if entry is None:
        raise KeyError(f"{name} not in {path}")
    start, kind, dims = entry
    if kind != GGUF_TYPE_IQ4_NL:
        raise ValueError(f"GGUF Engram table {name} has ggml type {kind}; only IQ4_NL (20) is implemented")
    if len(dims) != 2:
        raise ValueError("Engram table must be 2-D")
    dim, rows = int(dims[0]), int(dims[1])
    row_bytes = packed_row_bytes(IQ4NL, dim, "F16")
    if start + rows * row_bytes > os.path.getsize(path):
        raise ValueError(f"GGUF Engram table {name} runs past the end of the file")
    ends = sorted(s for s, _, _ in tensors.values() if s > start)
    if ends and start + rows * row_bytes > ends[0]:
        raise ValueError(f"GGUF Engram table {name} overlaps the next tensor")
    return {"format": IQ4NL, "start": start, "rows": rows, "dim": dim, "row_bytes": row_bytes,
            "dtype": "IQ4_NL", "scale_dtype": "F16"}


def table_layout(path, name) -> dict:
    """Declared layout of Engram table ``name`` in ``path`` (no data read).

    ``{"format", "start", "rows", "dim", "row_bytes", "dtype", "scale_dtype"}``;
    a table without format metadata is ``bf16`` (the checkpoint's own).
    """
    if is_gguf(path):
        return gguf_table_layout(path, name)
    header = safetensors_header(path)
    meta = header.get("__metadata__") or {}
    start, dtype, shape = safetensors_tensor(path, name)
    if len(shape) != 2:
        raise ValueError("Engram table must be 2-D")
    table_format = meta.get(FORMAT_KEY, "bf16")
    if table_format not in WRITABLE_FORMATS:
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
    header, data_offset = read_safetensors_header_file(
        path, length_bounds=(2, 1 << 28), bound_message="implausible safetensors header")
    header["__header_bytes__"] = data_offset
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
                      "cache_hits": 0, "bytes_read": 0, "prefill_reads": 0, "decode_reads": 0,
                      "prefetch_rows_submitted": 0, "prefetch_rows_consumed": 0,
                      "prefetch_rows_landed": 0, "prefetch_rows_declined": 0,
                      "foreground_rows_read": 0}
        # Rows a prefetch put in flight: row -> (future of its chunk, index in it).
        self._inflight: dict = {}
        self.advisor = ReadAdvisor(READ_ADVISE_ENV)
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

    def _read_row(self, fd, row):
        return pread_exact(fd, self.row_bytes, self.start + int(row) * self.row_bytes)

    def _read_chunk(self, fd, rows):
        size, base, pread = self.row_bytes, self.start, os.pread
        # One pread per row; a short read (rare) is completed row by row.
        out = [pread(fd, size, base + row * size) for row in np.asarray(rows, np.int64).tolist()]
        for i, data in enumerate(out):
            if len(data) != size:
                out[i] = self._read_row(fd, rows[i])
        raw = b"".join(out)
        if self.format == FP8_ROW:
            return decode_fp8_row(np.frombuffer(raw, np.uint8).reshape(len(rows), -1), self.dim, self.scale_dtype)
        if self.format == AFFINE4:
            return decode_affine4_row(np.frombuffer(raw, np.uint8).reshape(len(rows), -1), self.dim, self.scale_dtype)
        if self.format == IQ4NL:
            return decode_iq4nl_row(np.frombuffer(raw, np.uint8).reshape(len(rows), -1), self.dim)
        return _decode(raw, self.dtype, len(rows), self.dim)

    @property
    def approximate(self) -> bool:
        return self.format in (AFFINE4, IQ4NL)

    @property
    def fp8_round_trip_dtype(self):
        """MLX dtype in which this table's rows already took the E4M3 round trip."""
        if self.format != FP8_ROW:
            return None
        import mlx.core as mx

        return {"BF16": mx.bfloat16, "F32": mx.float32, "F16": mx.float16}[self.scale_dtype]

    def _fetch(self, missing):
        if not len(missing):
            return {}
        with self._lock:
            if self._closed:
                raise RuntimeError("Engram table reader is closed")
            pool, fd = self._pool, self._fd
        missing = np.asarray(missing, np.int64)
        prefill = len(missing) >= PREFILL_ROW_THRESHOLD
        workers = self.prefill_workers if prefill else self.decode_workers
        chunks = [c for c in np.array_split(missing, min(workers, len(missing))) if c.size]
        futures = [pool.submit(self._read_chunk, fd, chunk) for chunk in chunks]
        deadline = time.monotonic() + self.timeout  # one deadline for the whole lookup
        out = {}
        try:
            for chunk, future in zip(chunks, futures):
                values = future.result(timeout=max(0.0, deadline - time.monotonic()))
                # Views of the chunk: read_host copies each row it caches
                # (``_cache_put``), so the LRU never pins a whole chunk.
                out.update(zip(chunk.tolist(), values))
        except FutureTimeout:
            failure = ReadTimeout("Engram row read exceeded its deadline")
            pool.fail(failure)
            raise failure from None
        with self._lock:
            self.stats["prefill_reads" if prefill else "decode_reads"] += 1
        return out

    @property
    def read_advise(self) -> bool:
        return self.advisor.enabled

    def advise(self, rows: np.ndarray) -> int:
        """Queue a read-ahead hint for the uncached ``rows`` of a later lookup.

        Returns the number of rows hinted (0 when off, cached, empty or
        prefill-sized).  The hint runs on this reader's pool (no filesystem
        call on the model thread) and is not awaited; a refused or
        unsubmittable hint changes nothing but the ``failures`` counter.
        """
        if not self.advisor.enabled:
            return 0
        unique = np.unique(np.asarray(rows, np.int64).reshape(-1))
        if unique.size and (unique[0] < 0 or unique[-1] >= self.rows):
            return 0  # the read itself reports the bad row
        with self._lock:
            if self._closed:
                return 0
            pool, fd = self._pool, self._fd
            missing = [r for r in unique.tolist() if r not in self._cache]
        if not missing or len(missing) >= PREFILL_ROW_THRESHOLD:
            return 0
        spans = [(self.start + r * self.row_bytes, self.row_bytes) for r in missing]
        try:
            pool.submit(self.advisor.advise, fd, spans, len(missing))
        except Exception:
            self.advisor.record_failure(len(spans))
            return 0
        return len(missing)

    # At most this many rows in flight from speculative prefetches; further
    # ones are declined rather than queued without bound.  A ``required``
    # prefetch (the forward's own rows, read right after) is never declined:
    # declining one only moved its rows onto the critical path.  Until
    # 2026-11-21 the forward's plan was capped too, so a prefill chunk of more
    # than 4096 tokens (16 heads x 4096 = this cap) read the rest of every
    # table in the foreground, one table after another.
    MAX_INFLIGHT_ROWS = 1 << 16

    def _land(self):
        """Move finished prefetch rows into the LRU (caller holds the lock)."""
        done = [row for row, (future, _i) in self._inflight.items() if future.done()]
        for row in done:
            future, index = self._inflight.pop(row)
            try:
                values = future.result(timeout=0)
            except BaseException:
                continue  # read_host will refetch it (and the pool reports the failure)
            self._cache[row] = values[index].copy()
            self.stats["prefetch_rows_landed"] += 1
        while len(self._cache) > self.cache_rows:
            self._cache.popitem(last=False)

    def prefetch(self, rows, *, required: bool = False) -> int:
        """Start reading ``rows`` that are neither cached nor in flight; never blocks.

        ``required``: the caller reads every row next (a forward's plan), so
        ``MAX_INFLIGHT_ROWS`` does not apply.  Exact: a later ``read_host``
        returns the same values whether the rows came from a prefetch or a
        foreground read.  Returns the rows submitted.
        """
        rows = np.unique(np.asarray(rows, np.int64).reshape(-1))
        if rows.size and (rows.min() < 0 or rows.max() >= self.rows):
            raise IndexError("Engram row outside its table")
        with self._lock:
            if self._closed:
                raise RuntimeError("Engram table reader is closed")
            self._land()
            todo = [r for r in rows.tolist() if r not in self._cache and r not in self._inflight]
            room = len(todo) if required else self.MAX_INFLIGHT_ROWS - len(self._inflight)
            if len(todo) > room:
                self.stats["prefetch_rows_declined"] += len(todo) - max(room, 0)
                todo = todo[: max(room, 0)]
            if not todo:
                return 0
            pool, fd = self._pool, self._fd
            workers = self.prefill_workers if len(todo) >= PREFILL_ROW_THRESHOLD else self.decode_workers
            chunks = [c for c in np.array_split(np.asarray(todo, np.int64), min(workers, len(todo))) if c.size]
            for chunk in chunks:
                future = pool.submit(self._read_chunk, fd, chunk)
                for i, row in enumerate(chunk.tolist()):
                    self._inflight[row] = (future, i)
            self.stats["prefetch_rows_submitted"] += len(todo)
            self.stats["bytes_read"] += len(todo) * self.row_bytes
        return len(todo)

    def read_host(self, rows: np.ndarray) -> np.ndarray:
        """Decoded rows ``[n, dim]`` fp32 of ``rows`` (any order, repeats allowed).

        Rows this lookup needs are claimed from the in-flight prefetches
        before any other prefetch lands in the LRU: landing first could evict
        a requested row that a prefetch had already read (whenever one
        forward's rows outnumber ``cache_rows``), and the read would fetch it
        again in the foreground.
        """
        rows = np.asarray(rows, np.int64).reshape(-1)
        if rows.size and (rows.min() < 0 or rows.max() >= self.rows):
            raise IndexError("Engram row outside its table")
        if not rows.size:
            return np.zeros((0, self.dim), np.float32)
        unique, inverse = np.unique(rows, return_inverse=True)
        out = {}
        missing = []
        waiting = {}  # future -> [(row, index in its chunk)]
        with self._lock:
            for row in unique.tolist():
                pending = self._inflight.pop(row, None)
                if pending is not None:
                    waiting.setdefault(pending[0], []).append((row, pending[1]))
                    continue
                hit = self._cache.get(row)
                if hit is not None:
                    self._cache.move_to_end(row)
                    out[row] = hit
                else:
                    missing.append(row)
            self._land()
        deadline = time.monotonic() + self.timeout
        consumed = 0
        for future, members in waiting.items():
            try:
                values = future.result(timeout=max(0.0, deadline - time.monotonic()))
            except FutureTimeout:
                failure = ReadTimeout("Engram prefetch read exceeded its deadline")
                self._pool.fail(failure)
                raise failure from None
            index = [i for _row, i in members]
            # Rows of one gathered copy; cached rows are copied again, one by one.
            out.update(zip((row for row, _i in members), values[index]))
            consumed += len(members)
        with self._lock:
            self.stats["prefetch_rows_consumed"] += consumed
        fetched = self._fetch(missing)
        out.update(fetched)
        with self._lock:
            # LRU order: this lookup's hits (re-inserted when ``_land`` above
            # evicted them), then the rows it read.  Only the newest
            # ``cache_rows`` can survive, so only those are copied.
            waited = [row for members in waiting.values() for row, _i in members]
            read = set(missing).union(waited)
            hits = [r for r in unique.tolist() if r not in read]
            self._cache_put(hits + missing + waited, out)
            self.stats["lookups"] += 1
            self.stats["requested_rows"] += int(rows.size)
            self.stats["unique_rows"] += int(unique.size)
            self.stats["cache_hits"] += int(unique.size) - len(missing) - consumed
            self.stats["bytes_read"] += len(missing) * self.row_bytes
            self.stats["foreground_rows_read"] += len(missing)
        table = np.stack([out[r] for r in unique.tolist()])
        return table[inverse.reshape(-1)]

    def _cache_put(self, order, values):
        """Insert ``order`` (oldest first) into the LRU, then evict to ``cache_rows``.

        Each new entry is its own contiguous copy (``.base is None``), so the
        LRU holds at most ``cache_rows * dim`` floats whatever buffer the row
        came from (a chunk, a gathered prefetch).  Caller holds the lock.
        """
        if not self.cache_rows:
            self._cache.clear()
            return
        order = order[-self.cache_rows:]
        for row in order:
            hit = self._cache.get(row)
            if hit is not None and hit.base is None:
                self._cache.move_to_end(row)
            else:
                self._cache[row] = np.array(values[row], copy=True)
                self._cache.move_to_end(row)
        while len(self._cache) > self.cache_rows:
            self._cache.popitem(last=False)

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
                "closed": self._closed, "read_advise": self.advisor.status(), **self.stats}

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
    if table_format not in WRITABLE_FORMATS:
        raise ValueError(f"Engram table format must be one of {WRITABLE_FORMATS}")
    if formats is None:
        if table_format != "bf16":
            raise ValueError(f"artifact carries no {table_format} Engram tables (only bf16)")
        return weight_map
    if table_format not in formats:
        raise ValueError(f"artifact carries no {table_format} Engram tables "
                         f"(it has {sorted(formats)}); select one with engram_table_format")
    return formats[table_format]


def gguf_bindings(gguf_tables, layer_ids: dict) -> dict:
    """Decoder layer index -> ``(file, tensor)`` from ``text_config.engram_gguf_tables``.

    ``gguf_tables`` maps each config Engram layer id (decoder index + 1) to
    ``{"file": <path inside the artifact>, "tensor": <GGUF tensor name>}``;
    every Engram layer needs exactly one binding.
    """
    if not isinstance(gguf_tables, dict):
        raise ValueError("engram_gguf_tables must map Engram layer ids to {file, tensor}")
    by_id = {str(layer_id): index for index, layer_id in layer_ids.items()}
    if set(gguf_tables) != set(by_id):
        raise ValueError("engram_gguf_tables must bind exactly the Engram layers")
    out = {}
    for key, binding in gguf_tables.items():
        if (not isinstance(binding, dict) or set(binding) != {"file", "tensor"}
                or not all(isinstance(binding[k], str) and binding[k] for k in ("file", "tensor"))):
            raise ValueError(f"engram_gguf_tables[{key!r}] must be {{file, tensor}}")
        name = binding["file"]
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"Engram GGUF file must stay within the artifact: {name}")
        out[by_id[key]] = (name, binding["tensor"])
    return out


def open_engram_tables(root, weight_map: dict, layer_ids: dict, *, table_format="bf16",
                       formats=None, gguf_tables=None, **kwargs) -> dict:
    """Decoder layer index -> ``EngramTableRows`` for every Engram layer.

    ``iq4_nl`` tables come from the GGUF bindings ``gguf_tables``
    (``text_config.engram_gguf_tables``); every other format from the
    weight map or the converter's ``mlx2_engram_formats``.
    """
    out = {}
    if table_format == IQ4NL:
        if gguf_tables is None:
            raise ValueError("artifact binds no iq4_nl Engram tables (no engram_gguf_tables)")
        located = gguf_bindings(gguf_tables, layer_ids)
    else:
        shards = table_shards(weight_map, formats, table_format)
        located = {}
        for index in layer_ids:
            name = f"model.layers.{index}.engram_layer.multi_head_embedding.embedding.weight"
            if shards.get(name) is None:
                raise ValueError(f"Engram table {name} ({table_format}) is not in the artifact")
            located[index] = (shards[name], name)
    try:
        for index in layer_ids:
            shard, name = located[index]
            out[index] = EngramTableRows(Path(root) / shard, name, table_format=table_format, **kwargs)
    except BaseException:
        for reader in out.values():
            reader.close()
        raise
    return out

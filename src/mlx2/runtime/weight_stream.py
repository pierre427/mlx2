"""Disk streaming of quantized weights: MoE expert rows and dense MLP projections.

Experts of a routed MoE layer are contiguous rows of a stacked ``[E, ...]``
tensor inside the model's own safetensors shards.  This module reads one
expert's rows by byte range on demand, keeps a bounded per-layer LRU of the
experts a run actually touches, and remaps ``rhs_indices`` so the untouched
``mx.gather_qmm`` kernel computes on the resident subset.  Dense paging
(``install_dense_streaming``) reads a whole quantized projection per call,
runs the stock ``mx.quantized_matmul`` and evaluates its output before the
projection's bytes are released.

Two install phases exist and receipts name them:

* ``pre_materialization`` -- :mod:`mlx2.runtime.streamed_load` replaces the
  streamable modules while the model tree still holds unevaluated shard
  arrays, so the streamed tables are never materialized.  Source addressing
  is *proven*: a module tensor must be the very array object ``mx.load``
  returned (a rename keeps identity) or an output-axis concatenation recorded
  in the context-scoped ledger (:func:`record_concat`).  Descriptors and
  headers are bound once (:class:`BoundShards`).
* ``post_materialization`` -- the legacy engine path for adapters that do not
  declare streaming: the full model was already evaluated, so the load-time
  peak is NOT bounded; only the steady-state residency is.

Deliberate non-goals, each measured or argued in the design note:

* **No cross-layer speculative prefetch.**  Adjacent-layer expert overlap sits
  at chance on three architectures and an end-to-end A/B gave -2.9 % decode.
  Fetch is reactive: a miss is resolved when the router names it.
* **No mmap demand paging.**  Metal wires the whole buffer on first kernel use,
  so a 2.6 MiB expert would fault its entire ~1.2 GiB stacked tensor.
* **No pinned hot set.**  The atlas in :mod:`mlx2.runtime.expert_atlas` collects
  access counts but must not influence residency in this iteration.
* **No chunking of wide gathers.**  Splitting a call changes its geometry and
  therefore MLX's kernel choice and bits.  A wide call is served whole and its
  worst-case transient is *reserved* (``transient_bound_bytes``).

Correctness invariant: paging changes only *where* a byte sits, never its
value.  The same quantized rows reach the same kernel with the same
``group_size``/``bits``/``mode``; only the index vector differs, and the remap
is order preserving so a sorted gather stays sorted.

Accounting is *tracked weight bytes* -- counter arithmetic over the arrays this
module creates.  It is not measured process memory: the OS page cache, the
MLX allocator cache, activations and KV caches are outside it.

Any throughput measured on a streamed configuration is NOT a benchmark.
"""

from __future__ import annotations

import contextvars
import gc
import hashlib
import json
import logging
import os
import stat as stat_module
import threading
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Concurrency sweet spot measured on the M3 Pro's internal NVMe: random
# ~900 KiB reads run 2.14 GB/s at one thread and 7.46 GB/s at sixteen.  Never
# issue single-threaded small reads; never multi-thread one sequential stream.
DEFAULT_READ_WORKERS = 16
# Expert payloads in flight (read or awaiting materialization) per worker.
# Host staging is bounded by ``window x expert_bytes`` rather than by the
# number of missing experts in a wide prefill call.
STAGING_WINDOW_PER_WORKER = 2
# Dense projections are read in chunks so a large tensor still uses the pool.
DENSE_READ_CHUNK_BYTES = 8 << 20
# Bumped whenever the streamed arithmetic or its addressing contract changes;
# bound into the APCv2 numerics namespace and the settings receipt.
MECHANISM_REVISION = "weight-stream-v2"

STREAM_MODES = ("moe_experts", "dense_mlp")

_SAFETENSORS_DTYPES = {
    "BOOL": ("u1", "bool_"),
    "U8": ("u1", "uint8"),
    "I8": ("i1", "int8"),
    "U16": ("u2", "uint16"),
    "I16": ("i2", "int16"),
    "F16": ("f2", "float16"),
    "BF16": ("u2", "bfloat16"),
    "U32": ("u4", "uint32"),
    "I32": ("i4", "int32"),
    "F32": ("f4", "float32"),
    "U64": ("u8", "uint64"),
    "I64": ("i8", "int64"),
    "F64": ("f8", "float64"),
}

_SAFETENSORS_HEADER_LIMIT = 64 << 20
_MAX_TENSOR_ELEMENTS = (1 << 63) - 1
_SAFETENSORS_DTYPE_BYTES = {
    name: int(numpy_code[1:])
    for name, (numpy_code, _) in _SAFETENSORS_DTYPES.items()
}
# These are valid safetensors dtypes even when an expert reader cannot
# materialize them. The index validates the whole shard, not only experts.
_SAFETENSORS_DTYPE_BYTES.update({
    "F8_E4M3FN": 1, "F8_E5M2": 1,
    "F8_E4M3FNUZ": 1, "F8_E5M2FNUZ": 1,
})


class StreamingUnavailable(RuntimeError):
    """Streaming cannot be installed for this model; run fully resident."""


class WorkingSetTooSmall(RuntimeError):
    """The configured ceiling cannot hold one step's experts.

    Refusing with a printed budget beats a run that thrashes for six hours and
    then fails; Splash's ``WorkingSetTooSmall`` is the precedent.
    """


class WeightSourceChanged(RuntimeError):
    """A bound weight file changed (size, times or inode) after binding.

    Raised from a page-in, so a streamed forward fails before any state it
    produced can be published.  The engine treats it as fatal for the worker:
    the weights it serves are no longer the weights it bound.  The check is a
    ``stat`` tuple -- a mutation detector, not a content hash.
    """


# ---------------------------------------------------------------------------
# safetensors byte-range addressing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorLocation:
    path: Path
    dtype: str
    shape: Tuple[int, ...]
    begin: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.begin

    def row_bytes(self) -> int:
        if not self.shape or self.shape[0] <= 0:
            raise StreamingUnavailable(f"not a stacked tensor: {self.shape}")
        nbytes = self.nbytes
        rows = int(self.shape[0])
        if nbytes % rows:
            raise StreamingUnavailable("stacked tensor rows are not byte aligned")
        return nbytes // rows

    def row_range(self, row: int) -> Tuple[int, int]:
        stride = self.row_bytes()
        start = self.begin + row * stride
        return (start, stride)


def _decode_header(first8: bytes, read_payload, size: int, path) -> Dict[str, dict]:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise StreamingUnavailable(f"duplicate safetensors key {key!r}: {path}")
            result[key] = value
        return result

    if len(first8) != 8:
        raise StreamingUnavailable(f"truncated safetensors header: {path}")
    length = int.from_bytes(first8, "little")
    if not 0 < length <= min(_SAFETENSORS_HEADER_LIMIT, size - 8):
        raise StreamingUnavailable(f"invalid safetensors header length: {path}")
    raw = read_payload(length)
    try:
        header = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as error:
        raise StreamingUnavailable(f"invalid safetensors JSON header: {path}") from error
    if not isinstance(header, dict) or "__data_offset__" in header:
        raise StreamingUnavailable(f"invalid safetensors header object: {path}")
    header.pop("__metadata__", None)
    return {"__data_offset__": 8 + length, **header}


def read_safetensors_header(path: Path) -> Dict[str, dict]:
    with open(path, "rb") as handle:
        size = os.fstat(handle.fileno()).st_size
        return _decode_header(handle.read(8), handle.read, size, path)


def _pread_all(fd: int, offset: int, length: int, path) -> bytes:
    chunks = []
    remaining = length
    while remaining > 0:
        chunk = os.pread(fd, remaining, offset)
        if not chunk:
            raise StreamingUnavailable(f"short read on {path} at {offset}")
        chunks.append(chunk)
        offset += len(chunk)
        remaining -= len(chunk)
    if not chunks:
        return b""
    return chunks[0] if len(chunks) == 1 else b"".join(chunks)


def _read_header_fd(fd: int, size: int, path) -> Dict[str, dict]:
    first8 = os.pread(fd, 8, 0)
    return _decode_header(
        first8, lambda length: _pread_all(fd, 8, length, path), size, path
    )


class SafetensorsIndex:
    """Name -> byte range across every shard of a checkpoint."""

    def __init__(self, tensors: Dict[str, TensorLocation]):
        self._tensors = tensors

    @classmethod
    def from_model_path(cls, model_path) -> "SafetensorsIndex":
        root = Path(model_path).expanduser()
        shards = sorted(root.glob("*.safetensors"))
        if not shards:
            raise StreamingUnavailable(f"no safetensors shards under {root}")
        entries = []
        for shard in shards:
            header = read_safetensors_header(shard)
            entries.append((shard, header, shard.stat().st_size))
        return cls.from_headers(entries)

    @classmethod
    def from_headers(cls, entries) -> "SafetensorsIndex":
        """Index ``(path, header, file_size)`` triples, validating each shard."""
        tensors: Dict[str, TensorLocation] = {}
        for (shard, header, file_size) in entries:
            header = dict(header)
            base = header.pop("__data_offset__")
            payload_bytes = int(file_size) - base
            ranges = []
            for name, spec in header.items():
                if name in tensors:
                    raise StreamingUnavailable(f"duplicate safetensors tensor {name!r}")
                if not isinstance(spec, dict):
                    raise StreamingUnavailable(f"invalid safetensors tensor {name!r}")
                dtype = spec.get("dtype")
                shape = spec.get("shape")
                offsets = spec.get("data_offsets")
                # The index covers unrelated model tensors too. Preserve
                # their existing dtype acceptance; the materializers still
                # refuse formats they cannot reconstruct.
                if not isinstance(dtype, str) or not dtype:
                    raise StreamingUnavailable(f"invalid safetensors dtype for {name!r}")
                if not isinstance(shape, list) or any(
                    type(dim) is not int or not 0 <= dim <= _MAX_TENSOR_ELEMENTS
                    for dim in shape
                ):
                    raise StreamingUnavailable(f"invalid safetensors shape for {name!r}")
                if not (isinstance(offsets, list) and len(offsets) == 2
                        and all(type(offset) is int for offset in offsets)
                        and 0 <= offsets[0] <= offsets[1] <= payload_bytes):
                    raise StreamingUnavailable(f"invalid safetensors offsets for {name!r}")
                elements = 1
                for dim in shape:
                    elements *= dim
                    if elements > _MAX_TENSOR_ELEMENTS:
                        raise StreamingUnavailable(f"invalid safetensors shape for {name!r}")
                itemsize = _SAFETENSORS_DTYPE_BYTES.get(dtype)
                if itemsize is not None and offsets[1] - offsets[0] != elements * itemsize:
                    raise StreamingUnavailable(f"safetensors byte count mismatch for {name!r}")
                tensors[name] = TensorLocation(
                    path=Path(shard),
                    dtype=dtype,
                    shape=tuple(shape),
                    begin=base + int(offsets[0]),
                    end=base + int(offsets[1]),
                )
                if offsets[0] != offsets[1]:
                    ranges.append((offsets[0], offsets[1], name))
            for previous, current in zip(sorted(ranges), sorted(ranges)[1:]):
                if previous[1] > current[0]:
                    raise StreamingUnavailable(
                        f"overlapping safetensors tensor ranges: {previous[2]!r}, {current[2]!r}"
                    )
        return cls(tensors)

    def __contains__(self, name: str) -> bool:
        return name in self._tensors

    def __getitem__(self, name: str) -> TensorLocation:
        return self._tensors[name]

    def get(self, name: str) -> Optional[TensorLocation]:
        return self._tensors.get(name)

    def names(self) -> Iterable[str]:
        return self._tensors.keys()


def _readinto(fd: int, offset: int, view: memoryview, path) -> None:
    """Fill ``view`` from ``fd`` at ``offset`` without an intermediate copy."""
    filled = 0
    total = len(view)
    preadv = getattr(os, "preadv", None)
    while filled < total:
        if preadv is not None:
            count = preadv(fd, [view[filled:]], offset + filled)
            if count <= 0:
                raise StreamingUnavailable(f"short read on {path} at {offset + filled}")
        else:  # pragma: no cover - platforms without preadv
            chunk = os.pread(fd, total - filled, offset + filled)
            if not chunk:
                raise StreamingUnavailable(f"short read on {path} at {offset + filled}")
            count = len(chunk)
            view[filled:filled + count] = chunk
        filled += count


class _FileHandles:
    """One shared read-only descriptor per shard, re-opened after a fork.

    Used by the legacy post-materialization install, whose shards were found
    by directory glob.  The early-load path binds descriptors up front with
    :class:`BoundShards` instead.

    Reads are positional (``os.pread``), so the read pool and the model
    thread share one descriptor per file.  Per-thread tables opened one per
    shard for every read worker plus the model thread (17 x shards by
    default) and never closed them; a many-shard checkpoint then crossed
    macOS's default soft limit of 256 and EMFILE inside a page-in killed the
    worker.  A forked child inherits the table and the descriptors; keying on
    the owning pid makes the child open its own rather than share them.

    :meth:`close` must run after every reader has stopped: a closed number
    is reused by the next ``open`` and a late ``pread`` would read that file.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._table: Dict[Path, int] = {}
        self._owner = os.getpid()
        self._closed = False

    def get(self, path: Path) -> int:
        if os.getpid() != self._owner:
            self._lock = threading.Lock()
            self._table = {}
            self._owner = os.getpid()
        fd = self._table.get(path)
        if fd is None:
            with self._lock:
                if self._closed:
                    raise StreamingUnavailable("expert weight files are closed")
                fd = self._table.get(path)
                if fd is None:
                    fd = os.open(str(path), os.O_RDONLY)
                    self._table[path] = fd
        return fd

    def check(self, path: Path) -> None:
        """Legacy handles carry no binding; nothing to verify."""

    def sources_receipt(self) -> List[list]:
        return []

    def close(self) -> None:
        with self._lock:
            table, self._table = self._table, {}
            self._closed = True
        for fd in table.values():
            os.close(fd)

    def pread(self, path: Path, offset: int, length: int) -> bytes:
        return _pread_all(self.get(path), offset, length, path)

    def readinto(self, path: Path, offset: int, view: memoryview) -> None:
        _readinto(self.get(path), offset, view, path)


class BoundShards:
    """Descriptors and headers for an artifact's indexed shards, bound once.

    Every shard named by the artifact's weight map is opened at construction;
    its ``fstat`` must agree with the adapter's identity record
    ``(name, size, mtime_ns)`` and its header is parsed from that same
    descriptor.  The full ``(dev, ino, size, mtime_ns, ctime_ns)`` tuple is
    pinned and re-checked before and after every page-in batch, and
    :meth:`verify_paths` confirms the paths still name the bound inodes (a
    replacement during load is refused).  An in-place rewrite, a truncation,
    or an atomic rename-replace (unlinking the bound inode moves its
    ``ctime``) raises :class:`WeightSourceChanged`; the bound inode stays
    readable throughout, so one read never mixes two revisions.

    This is a mutation detector, not a content hash: model artifact pinning
    remains the adapter identity's job.
    """

    def __init__(self, root, names: Sequence[str], *, records=None, weight_map=None):
        self.root = Path(root).expanduser().resolve()
        self._lock = threading.Lock()
        self._closed = False
        self._fds: Dict[Path, int] = {}
        self._pins: Dict[Path, Tuple[int, int, int, int, int]] = {}
        self._names: Dict[Path, str] = {}
        self.weight_map = dict(weight_map) if weight_map else None
        record_map = None
        if records is not None:
            record_map = {}
            for record in records:
                record_map[str(record[0])] = (int(record[1]), int(record[2]))
        names = list(dict.fromkeys(str(name) for name in names))
        if not names:
            raise StreamingUnavailable(f"no indexed safetensors shards under {self.root}")
        entries = []
        try:
            for name in names:
                relative = Path(name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise StreamingUnavailable(f"shard path escapes the artifact: {name}")
                path = self.root / relative
                if path in self._fds:
                    # "./s" and "s" are one file: a second descriptor would
                    # overwrite the first, which then never closes.
                    raise StreamingUnavailable(
                        f"shard {name} aliases already-bound shard {self._names[path]}"
                    )
                fd = os.open(str(path), os.O_RDONLY)
                self._fds[path] = fd
                self._names[path] = name
                info = os.fstat(fd)
                if not stat_module.S_ISREG(info.st_mode):
                    raise StreamingUnavailable(f"shard is not a regular file: {name}")
                if record_map is not None:
                    expected = record_map.get(name)
                    if expected is None:
                        raise StreamingUnavailable(
                            f"shard {name} is not in the artifact identity record"
                        )
                    if expected != (info.st_size, info.st_mtime_ns):
                        raise WeightSourceChanged(
                            f"shard {name} changed since the artifact was inspected"
                        )
                self._pins[path] = (
                    info.st_dev, info.st_ino, info.st_size,
                    info.st_mtime_ns, info.st_ctime_ns,
                )
                entries.append((path, _read_header_fd(fd, info.st_size, path), info.st_size))
            self.index = SafetensorsIndex.from_headers(entries)
        except BaseException:
            self.close()
            raise

    @property
    def paths(self) -> Tuple[Path, ...]:
        return tuple(self._fds)

    def name_of(self, path: Path) -> str:
        return self._names[Path(path)]

    def _fd(self, path: Path) -> int:
        fd = self._fds.get(Path(path))
        if fd is None or self._closed:
            raise StreamingUnavailable(f"weight shard is not bound or closed: {path}")
        return fd

    def check(self, path: Path) -> None:
        path = Path(path)
        info = os.fstat(self._fd(path))
        pinned = self._pins[path]
        observed = (
            info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns
        )
        if observed != pinned:
            raise WeightSourceChanged(
                f"bound weight shard {self._names[path]} changed after binding"
            )

    def check_all(self) -> None:
        for path in self._fds:
            self.check(path)

    def verify_paths(self) -> None:
        """The artifact paths still name the bound files (no replacement)."""
        for (path, pinned) in self._pins.items():
            try:
                info = os.stat(path)
            except OSError as error:
                raise WeightSourceChanged(
                    f"weight shard {self._names[path]} disappeared"
                ) from error
            observed = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
            if observed != pinned[:4]:
                raise WeightSourceChanged(
                    f"weight shard {self._names[path]} was replaced during load"
                )
            self.check(path)

    def sources_receipt(self) -> List[list]:
        """``[name, size, mtime_ns]`` per bound shard; no host-local inodes."""
        return sorted(
            [self._names[path], pin[2], pin[3]] for (path, pin) in self._pins.items()
        )

    def pread(self, path: Path, offset: int, length: int) -> bytes:
        return _pread_all(self._fd(path), offset, length, path)

    def readinto(self, path: Path, offset: int, view: memoryview) -> None:
        _readinto(self._fd(path), offset, view, path)

    def close(self) -> None:
        with self._lock:
            fds, self._fds = dict(self._fds), {}
            self._closed = True
        for fd in fds.values():
            try:
                os.close(fd)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# source origin: identity through sanitize, plus a concat ledger
# ---------------------------------------------------------------------------

_CONCAT_LEDGER: "contextvars.ContextVar[Optional[dict]]" = contextvars.ContextVar(
    "mlx2_weight_stream_concat_ledger", default=None
)


def record_concat(result, parts, axis) -> None:
    """Record ``result = concatenate(parts, axis)`` while a ledger is active.

    Called by load-time transforms that fuse checkpoint tensors.  Outside a
    streamed load no ledger is set and this is a no-op.  The result object is
    held so its ``id`` cannot be reused while the ledger lives.
    """
    ledger = _CONCAT_LEDGER.get()
    if ledger is not None:
        ledger[id(result)] = (result, tuple(parts), int(axis))


@contextmanager
def concat_ledger(target: dict):
    token = _CONCAT_LEDGER.set(target)
    try:
        yield target
    finally:
        _CONCAT_LEDGER.reset(token)


def _mx_dtype(dtype_name: str):
    entry = _SAFETENSORS_DTYPES.get(dtype_name)
    if entry is None:
        return None
    import mlx.core as mx

    return getattr(mx, entry[1])


def _check_location(where: str, location: TensorLocation, array) -> None:
    target = _mx_dtype(location.dtype)
    if target is None:
        raise StreamingUnavailable(
            f"{where}: safetensors dtype {location.dtype} cannot be materialized"
        )
    if tuple(array.shape) != tuple(location.shape):
        raise StreamingUnavailable(
            f"{where}: live shape {tuple(array.shape)} != checkpoint {location.shape}"
        )
    if array.dtype != target:
        raise StreamingUnavailable(
            f"{where}: live dtype {array.dtype} != checkpoint {location.dtype}"
        )


class TensorOrigins:
    """Where each lazily loaded raw array came from, by object identity.

    Python keeps the object identity of an array through a dict rename, and
    ``Module.load_weights``/``update`` stores the very object it is given.  A
    live module tensor that *is* a recorded raw array is therefore proven to
    be that checkpoint tensor, byte for byte.  The only other accepted
    derivation is an output-axis concatenation recorded in :attr:`concats`.
    Strong references are held, so recorded ``id`` values stay unique.
    """

    def __init__(self, index: SafetensorsIndex, *, weight_map=None, name_of=None):
        self._index = index
        self._weight_map = dict(weight_map) if weight_map else None
        self._name_of = name_of
        self._arrays: Dict[int, Tuple[object, TensorLocation]] = {}
        self.concats: dict = {}

    def record_shard(self, shard_path: Path, arrays: dict) -> None:
        shard_path = Path(shard_path)
        shard_name = self._name_of(shard_path) if self._name_of else shard_path.name
        for (name, array) in arrays.items():
            location = self._index.get(name)
            if location is None or Path(location.path) != shard_path:
                continue
            if self._weight_map is not None and self._weight_map.get(name) != shard_name:
                continue  # unindexed tensor: never addressable
            self._arrays[id(array)] = (array, location)

    def locate(self, array) -> Optional[TensorLocation]:
        entry = self._arrays.get(id(array))
        if entry is None or entry[0] is not array:
            return None
        return entry[1]

    def resolve_expert(self, path: str, key: str, array) -> "ExpertSliceSpec":
        where = f"{path}.{key}"
        location = self.locate(array)
        if location is not None:
            _check_location(where, location, array)
            return ExpertSliceSpec.single(key, location)
        concat = self.concats.get(id(array))
        if concat is not None and concat[0] is array:
            (_, parts, axis) = concat
            ndim = len(array.shape)
            if ndim < 3 or axis % ndim != ndim - 2:
                raise StreamingUnavailable(
                    f"{where}: concatenation along axis {axis} is not the "
                    "expert output axis; it cannot be addressed by rows"
                )
            located = []
            for part in parts:
                part_location = self.locate(part)
                if part_location is None:
                    raise StreamingUnavailable(
                        f"{where}: a concatenated part was itself transformed"
                    )
                _check_location(where, part_location, part)
                located.append(part_location)
            spec = ExpertSliceSpec(key=key, parts=tuple(located), concat_axis=0)
            if tuple(array.shape) != (spec.num_experts, *spec.expert_shape_concat()):
                raise StreamingUnavailable(f"{where}: concatenated geometry mismatch")
            return spec
        raise StreamingUnavailable(
            f"{where} was transformed during sanitize (not a rename and not a "
            "ledgered output-axis concatenation); it cannot be streamed"
        )

    def resolve_tensor(self, path: str, key: str, array) -> TensorLocation:
        where = f"{path}.{key}"
        location = self.locate(array)
        if location is None:
            raise StreamingUnavailable(
                f"{where} is not an untransformed checkpoint tensor; dense "
                "streaming accepts renames only"
            )
        _check_location(where, location, array)
        return location


# ---------------------------------------------------------------------------
# per-expert slice reader
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpertSliceSpec:
    """Where one projection's experts live, and how to rebuild one.

    ``parts`` is normally a single stacked checkpoint tensor.  It holds more
    than one when the live module's projection is a load-time *fusion* of
    several checkpoint tensors -- Qwen3-Next-family adapters concatenate
    ``gate_proj`` and ``up_proj`` into ``gate_up_proj`` along the output axis.
    Fusing one expert is still pure byte-range addressing: read each part's
    row for that expert and concatenate them in the same order, which
    reproduces the fused row exactly.
    """

    key: str
    parts: Tuple[TensorLocation, ...]
    concat_axis: int = 0

    @classmethod
    def single(cls, key: str, location: TensorLocation) -> "ExpertSliceSpec":
        return cls(key=key, parts=(location,))

    @property
    def location(self) -> TensorLocation:
        return self.parts[0]

    @property
    def num_experts(self) -> int:
        return int(self.parts[0].shape[0])

    def row_bytes(self) -> int:
        return sum(part.row_bytes() for part in self.parts)

    @property
    def expert_shape(self) -> Tuple[int, ...]:
        return tuple(self.location.shape[1:])

    def expert_shape_concat(self) -> Tuple[int, ...]:
        """One expert's live row shape after the parts are concatenated."""
        shapes = [tuple(part.shape[1:]) for part in self.parts]
        head = list(shapes[0])
        if not head:
            raise StreamingUnavailable("expert rows have no axes")
        axis = self.concat_axis % len(head)
        for shape in shapes[1:]:
            if len(shape) != len(head) or any(
                shape[i] != head[i] for i in range(len(head)) if i != axis
            ):
                raise StreamingUnavailable("concatenated expert parts disagree in shape")
            head[axis] += shape[axis]
        return tuple(head)


class ExpertSliceReader:
    """Byte-range reads of one MoE projection's expert rows.

    Disk I/O is split from ``mx.array`` construction on purpose: MLX arrays
    must be built on the thread that runs the model, so the pool returns plain
    bytes and :meth:`materialize` runs on the caller's thread.
    """

    def __init__(
        self,
        specs: Sequence[ExpertSliceSpec],
        *,
        handles,
        pool: Optional[ThreadPoolExecutor] = None,
        window: int = 0,
    ):
        if not specs:
            raise StreamingUnavailable("expert reader needs at least one tensor")
        self.specs = tuple(specs)
        self._handles = handles
        self._pool = pool
        self.window = int(window) if window else 0
        counts = {part.shape[0] for spec in self.specs for part in spec.parts}
        if len(counts) != 1:
            raise StreamingUnavailable("projection tensors disagree on expert count")
        self.num_experts = int(counts.pop())
        self.expert_bytes = sum(spec.row_bytes() for spec in self.specs)
        self.source_paths = tuple(
            dict.fromkeys(part.path for spec in self.specs for part in spec.parts)
        )
        self._outstanding = 0

    def check_sources(self) -> None:
        check = getattr(self._handles, "check", None)
        if check is not None:
            for path in self.source_paths:
                check(path)

    def outstanding_bytes(self) -> int:
        """Host staging currently held for this reader (in flight or ready)."""
        return self._outstanding * self.expert_bytes

    def read_bytes(self, expert: int) -> Tuple[Tuple[bytes, ...], ...]:
        """Raw rows for ``expert``.  Safe to run off the model thread."""
        out = []
        for spec in self.specs:
            blobs = []
            for part in spec.parts:
                (offset, length) = part.row_range(expert)
                blobs.append(self._handles.pread(part.path, offset, length))
            out.append(tuple(blobs))
        return tuple(out)

    def read_many(self, experts: Sequence[int]) -> Dict[int, Tuple[bytes, ...]]:
        return dict(self.iter_many(experts))

    def iter_many(self, experts: Sequence[int]) -> Iterator[Tuple[int, tuple]]:
        """Yield ``(expert, payload)`` as reads finish, at most ``window`` held.

        A wide prefill can miss every expert of a layer; submitting them all
        at once kept every payload in host memory until the last arrived.
        The caller materializes and drops each payload as it is yielded.
        """
        experts = list(experts)
        if self._pool is None or len(experts) <= 1:
            for expert in experts:
                self._outstanding = 1
                try:
                    yield (expert, self.read_bytes(expert))
                finally:
                    self._outstanding = 0
            return
        window = self.window or len(experts)
        queue = iter(experts)
        pending = {}

        def submit_next():
            for expert in queue:
                pending[self._pool.submit(self.read_bytes, expert)] = expert
                return

        try:
            for _ in range(min(window, len(experts))):
                submit_next()
            while pending:
                self._outstanding = len(pending)
                (done, not_done) = wait(tuple(pending), return_when=FIRST_COMPLETED)
                # Take ONE completed read and drop every other handle to it
                # before yielding: a retained ``done`` set or future would keep
                # its payload alive outside the window.  Other completed reads
                # stay in ``pending`` and are counted there.
                future = next(iter(done))
                del done, not_done
                expert = pending.pop(future)
                payload = future.result()
                del future
                self._outstanding = len(pending) + 1
                yield (expert, payload)
                del payload
                # Refill only after the consumer dropped this payload, so
                # in flight + ready never exceeds the window.
                submit_next()
        finally:
            for future in pending:
                future.cancel()
            pending.clear()
            self._outstanding = 0

    def materialize(self, payload: Sequence[bytes]) -> Tuple["object", ...]:
        """Build the expert's concrete ``mx.array`` rows.  Model thread only.

        Evaluated before return: a lazy ``view``/``concatenate`` entry would
        keep its pieces alive and double a fused row until first use.
        """
        import mlx.core as mx
        import numpy as np

        arrays = []
        for (spec, blobs) in zip(self.specs, payload):
            pieces = []
            for (part, blob) in zip(spec.parts, blobs):
                dtype = _SAFETENSORS_DTYPES.get(part.dtype)
                if dtype is None:
                    raise StreamingUnavailable(
                        f"unsupported safetensors dtype {part.dtype}"
                    )
                (numpy_code, mlx_name) = dtype
                flat = np.frombuffer(blob, dtype=np.dtype(numpy_code))
                array = mx.array(flat)
                target = getattr(mx, mlx_name)
                if array.dtype != target:
                    array = array.view(target)
                pieces.append(array.reshape(tuple(part.shape[1:])))
            arrays.append(
                pieces[0]
                if len(pieces) == 1
                else mx.concatenate(pieces, axis=spec.concat_axis)
            )
        mx.eval(*arrays)
        return tuple(arrays)


# ---------------------------------------------------------------------------
# accounting
# ---------------------------------------------------------------------------

# Counters that split by phase.  During ``load`` they land in ``load_*`` so a
# load-time diagnostic forward (the dtype probe) can never satisfy the
# serving qualification gate, which reads the unprefixed serving counters.
_PHASED = frozenset(
    {"page_ins", "page_in_bytes", "hits", "misses", "evictions", "overflows"}
)


@dataclass
class StreamStats:
    page_ins: int = 0
    page_in_bytes: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    resident_bytes: int = 0
    refusals: int = 0
    overflows: int = 0
    load_page_ins: int = 0
    load_page_in_bytes: int = 0
    load_hits: int = 0
    load_misses: int = 0
    load_evictions: int = 0
    load_overflows: int = 0
    peak_resident_bytes: int = 0
    peak_tracked_bytes: int = 0
    peak_transient_bytes: int = 0
    peak_staging_bytes: int = 0
    phase: str = "serve"
    _group: object = field(default=None, repr=False)
    _group_transient: int = field(default=0, repr=False)
    _group_members: set = field(default_factory=set, repr=False)

    def bump(self, name: str, amount: int = 1) -> None:
        if self.phase == "load" and name in _PHASED:
            name = "load_" + name
        setattr(self, name, getattr(self, name) + int(amount))

    def _track(self) -> None:
        tracked = self.resident_bytes + self._group_transient
        if tracked > self.peak_tracked_bytes:
            self.peak_tracked_bytes = tracked

    def add_resident(self, nbytes: int) -> None:
        self.resident_bytes += int(nbytes)
        if self.resident_bytes > self.peak_resident_bytes:
            self.peak_resident_bytes = self.resident_bytes
        self._track()

    def remove_resident(self, nbytes: int) -> None:
        self.resident_bytes -= int(nbytes)

    def note_transient(self, group, nbytes: int, member=None) -> None:
        """Device bytes a call holds beyond the steady cache.

        Calls of one layer (one ``group``) accumulate: their gather copies stay
        referenced by the lazy graph until the next layer's router sync.  The
        window restarts at a real evaluated boundary: a different group (the
        next layer, after its router sync), or the same ``member`` -- one
        projection -- being called again, which only happens in a later
        forward whose host index read evaluated the earlier one.
        """
        if group != self._group or member in self._group_members:
            self._group = group
            self._group_transient = 0
            self._group_members = set()
        self._group_members.add(member)
        self._group_transient += int(nbytes)
        if self._group_transient > self.peak_transient_bytes:
            self.peak_transient_bytes = self._group_transient
        self._track()

    def note_staging(self, nbytes: int) -> None:
        if nbytes > self.peak_staging_bytes:
            self.peak_staging_bytes = int(nbytes)

    def as_dict(self) -> Dict[str, int]:
        return {
            "stream_page_ins_total": self.page_ins,
            "stream_page_in_bytes_total": self.page_in_bytes,
            "stream_expert_hits_total": self.hits,
            "stream_expert_misses_total": self.misses,
            "stream_evictions_total": self.evictions,
            "stream_resident_bytes": self.resident_bytes,
            "stream_admission_refusals_total": self.refusals,
            "stream_working_set_overflows_total": self.overflows,
            "stream_load_page_ins_total": self.load_page_ins,
            "stream_load_page_in_bytes_total": self.load_page_in_bytes,
            "stream_peak_resident_bytes": self.peak_resident_bytes,
            "stream_peak_tracked_bytes": self.peak_tracked_bytes,
            "stream_peak_transient_bytes": self.peak_transient_bytes,
            "stream_peak_staging_bytes": self.peak_staging_bytes,
        }


# ---------------------------------------------------------------------------
# the cache
# ---------------------------------------------------------------------------


class ExpertLRU:
    """Bounded, dict-keyed LRU over one layer's experts.

    Dict-keyed, **not** a fixed slot array.  Every index in a forward pass is
    resolved before the gather runs, so an eviction triggered later in the same
    pass would otherwise clobber a slot an earlier expert already resolved to.
    Keying on the expert id and holding the pass's working set removes that
    class of bug entirely.
    """

    def __init__(self, reader: ExpertSliceReader, *, capacity_experts: int,
                 stats: StreamStats, group=None):
        if capacity_experts < 1:
            raise WorkingSetTooSmall("expert cache capacity must be at least 1")
        self.reader = reader
        self.capacity = int(min(capacity_experts, reader.num_experts))
        self.stats = stats
        self.group = group
        self._entries: "OrderedDict[int, Tuple[object, ...]]" = OrderedDict()

    @property
    def resident(self) -> int:
        return len(self._entries)

    def _drop(self, expert: int) -> None:
        self._entries.pop(expert)
        self.stats.remove_resident(self.reader.expert_bytes)
        self.stats.bump("evictions")

    def acquire(self, experts: Sequence[int]) -> Dict[int, Tuple["object", ...]]:
        """Resident rows for every id in ``experts``, fetching what is missing.

        The whole request set is held for the duration of the call: eviction
        only ever considers entries outside it.
        """
        wanted = list(dict.fromkeys(int(expert) for expert in experts))
        # A wide batch can route to more experts in one call than the steady
        # ceiling holds.  Every index must resolve before the gather runs, so
        # the only correct answers are "fetch them all" or "fail the request".
        # We fetch, count the overshoot, and trim back to the ceiling before
        # returning; the overshoot is part of the reserved transient bound.
        overflow = len(wanted) > self.capacity
        if overflow:
            self.stats.bump("overflows")
        missing = []
        for expert in wanted:
            if expert in self._entries:
                self._entries.move_to_end(expert)
                self.stats.bump("hits")
            else:
                self.stats.bump("misses")
                missing.append(expert)
        try:
            if missing:
                # Drop unrelated rows before the temporary working set is
                # materialized, including when that set exceeds capacity.
                self._evict_for(len(missing), keep=set(wanted))
                self.reader.check_sources()
                fetched = {}
                for (expert, payload) in self.reader.iter_many(missing):
                    self.stats.note_staging(self.reader.outstanding_bytes())
                    fetched[expert] = self.reader.materialize(payload)
                    del payload
                # Re-check after the reads: an in-place rewrite during I/O
                # must fail before any row it produced can be consumed.
                self.reader.check_sources()
                # Insert in request order so LRU order (and so page-in counts)
                # does not depend on which read finished first.
                for expert in missing:
                    self._entries[expert] = fetched.pop(expert)
                    self.stats.add_resident(self.reader.expert_bytes)
                    self.stats.bump("page_ins")
                    self.stats.bump("page_in_bytes", self.reader.expert_bytes)
            return {expert: self._entries[expert] for expert in wanted}
        finally:
            # An OOM/read/materialization error must not leave an oversized
            # bank resident after the request has failed.
            self._trim_to_capacity()

    def note_gather(self, unique: int) -> None:
        """Account one call's gather copy plus any over-capacity rows it holds."""
        expert_bytes = self.reader.expert_bytes
        transient = unique * expert_bytes + max(0, unique - self.capacity) * expert_bytes
        self.stats.note_transient(self.group, transient, member=id(self))

    def _evict_for(self, incoming: int, *, keep) -> None:
        while self.resident + incoming > self.capacity:
            victim = None
            for expert in self._entries:
                if expert not in keep:
                    victim = expert
                    break
            if victim is None:  # Only the current working set remains.
                break
            self._drop(victim)

    def _trim_to_capacity(self) -> None:
        while self.resident > self.capacity:
            self._drop(next(iter(self._entries)))

    def clear(self) -> None:
        while self._entries:
            expert = next(iter(self._entries))
            self._entries.pop(expert)
            self.stats.remove_resident(self.reader.expert_bytes)

    def seed(self, experts: Sequence[int]) -> None:
        """Warm the cache through the fetch path.

        Never by slicing an already-loaded stacked weight: a prefix slice keeps
        the whole parent buffer alive and frees nothing.
        """
        for expert in experts:
            self.acquire([expert])


# ---------------------------------------------------------------------------
# the streamed modules
# ---------------------------------------------------------------------------


_STREAMED_CLASS = None


def _streamed_class():
    """Build the streamed module class once, at module scope.

    Deliberately NOT a closure over the module being replaced.  A class
    defined per call captures the original ``QuantizedSwitchLinear`` in a
    closure cell, which pins its stacked weight/scales/biases for the life of
    the process -- so installing streaming would free nothing at all.  That
    is not hypothetical: it was measured on an M3, where active memory after
    install stayed at 19.68 GiB instead of dropping by the table.
    """
    global _STREAMED_CLASS
    if _STREAMED_CLASS is not None:
        return _STREAMED_CLASS

    import mlx.core as mx
    import numpy as np

    from .models.switch_layers import (
        QuantizedSwitchLinear,
        _quantized_gather_tail_policy,
    )

    class StreamedQuantizedSwitchLinear(QuantizedSwitchLinear):
        def __init__(self, *, group_size, bits, mode, bias, input_dims,
                     output_dims, reader, cache, layer_index, collector):
            # Bypass the base class' random-init constructor entirely.
            object.__setattr__(self, "_no_grad", set())
            object.__setattr__(self, "_training", False)
            dict.__init__(self)
            self.group_size = group_size
            self.bits = bits
            self.mode = mode
            if bias is not None:
                self["bias"] = bias
            self._stream_reader = reader
            self._stream_cache = cache
            self._stream_layer = layer_index
            self._stream_collector = collector
            self._stream_num_experts = reader.num_experts
            self._stream_input_dims = int(input_dims)
            self._stream_output_dims = int(output_dims)
            self.freeze(recurse=False)

        @property
        def input_dims(self):
            return self._stream_input_dims

        @property
        def output_dims(self):
            return self._stream_output_dims

        @property
        def num_experts(self):
            return self._stream_num_experts

        def __call__(self, x, indices, sorted_indices=False):
            # The host read of ``indices`` evaluates the graph up to this
            # layer's router, which releases the previous layer's gather
            # copies: retention is bounded to about one layer.
            host = np.asarray(indices, dtype=np.int64)
            unique = np.unique(host)  # ascending: the remap stays monotonic,
            # so a sorted gather is still sorted afterwards.
            try:
                entries = self._stream_cache.acquire([int(e) for e in unique])
            except BaseException:
                if self._stream_collector is not None:
                    self._stream_collector.note_lost_call(self._stream_layer)
                raise
            self._stream_cache.note_gather(int(unique.size))
            if self._stream_collector is not None:
                self._stream_collector.observe(self._stream_layer, unique)
            table = np.zeros(self._stream_num_experts, dtype=np.uint32)
            table[unique] = np.arange(unique.size, dtype=np.uint32)
            remapped = mx.array(table[host].reshape(host.shape))
            rows = [entries[int(e)] for e in unique]
            weight = mx.stack([row[0] for row in rows])
            scales = mx.stack([row[1] for row in rows])
            biases = mx.stack([row[2] for row in rows]) if len(rows[0]) > 2 else None
            tail_policy = _quantized_gather_tail_policy(
                self.mode, int(x.shape[-1]), sorted_indices
            )
            if tail_policy == "dense":
                quantized = [weight, scales]
                if biases is not None:
                    quantized.append(biases)
                dense = mx.dequantize(
                    *quantized,
                    group_size=self.group_size,
                    bits=self.bits,
                    mode=self.mode,
                )
                out = mx.gather_mm(
                    x,
                    dense.swapaxes(-1, -2),
                    rhs_indices=remapped,
                    sorted_indices=sorted_indices,
                )
            else:
                out = mx.gather_qmm(
                    x,
                    weight,
                    scales,
                    biases,
                    rhs_indices=remapped,
                    transpose=True,
                    group_size=self.group_size,
                    bits=self.bits,
                    mode=self.mode,
                    sorted_indices=(
                        False if tail_policy == "unsorted" else sorted_indices
                    ),
                )
            if "bias" in self:
                out = out + mx.expand_dims(self["bias"][indices], -2)
            return out

    _STREAMED_CLASS = StreamedQuantizedSwitchLinear
    return _STREAMED_CLASS


def _streamed_switch_linear(base, reader, cache, *, layer_index, unit_offset, collector):
    """Return a drop-in replacement for one ``QuantizedSwitchLinear``.

    Everything the replacement needs is copied out of ``base`` here; no
    reference to ``base`` survives this call, so its stacked tensors become
    collectable the moment the parent drops it.
    """
    return _streamed_class()(
        group_size=base.group_size,
        bits=base.bits,
        mode=base.mode,
        bias=base["bias"] if "bias" in base else None,
        input_dims=base.input_dims,
        output_dims=base.output_dims,
        reader=reader,
        cache=cache,
        layer_index=layer_index,
        collector=collector,
    )


def is_streamed_module(module) -> bool:
    return getattr(module, "_stream_reader", None) is not None


# ---------------------------------------------------------------------------
# sizing
# ---------------------------------------------------------------------------

# Resident fraction of the expert table, as a function of table/RAM ratio.
# Community measurement, reproduced in the design note: decode throughput
# peaks and then collapses as the cache squeezes the kernel page cache
# (60 GB -> 35 GB was +65 % decode on one 128 GB box).  Bigger is not better,
# so the budget is sized from this curve rather than from "as much as fits".
_RESIDENT_FRACTION_CURVE = ((1.05, 0.45), (1.8, 0.30), (2.6, 0.20))


def resident_fraction(table_bytes: int, budget_bytes: int) -> float:
    """Fraction of the expert table worth holding resident."""
    if budget_bytes <= 0 or table_bytes <= 0:
        return 0.0
    ratio = table_bytes / float(budget_bytes)
    points = _RESIDENT_FRACTION_CURVE
    if ratio <= 1.0:
        # The whole table fits inside the budget: there is nothing to stream
        # and no page cache to squeeze.  The curve starts at the first ratio
        # where the table does *not* fit (1.05 -> 0.45), and the step between
        # the two is real, not an artefact: it is the measured collapse.
        return 1.0
    if ratio <= points[0][0]:
        return points[0][1]
    if ratio >= points[-1][0]:
        # Extrapolating a collapse curve is how you get a made-up constant;
        # clamp to the last measured point instead.
        return points[-1][1]
    for ((x0, y0), (x1, y1)) in zip(points, points[1:]):
        if x0 <= ratio <= x1:
            span = x1 - x0
            return y0 + (y1 - y0) * ((ratio - x0) / span)
    return points[-1][1]



def plan_cache_experts(
    *,
    table_bytes: int,
    expert_bytes: int,
    num_layers: int,
    top_k: int,
    ceiling_bytes: int,
) -> int:
    """Per-layer expert capacity for an enforced ``ceiling_bytes``."""
    if expert_bytes <= 0 or num_layers <= 0:
        raise WorkingSetTooSmall("cannot size a cache for an empty model")
    floor_bytes = num_layers * max(top_k, 1) * expert_bytes
    if ceiling_bytes < floor_bytes:
        raise WorkingSetTooSmall(
            f"expert cache ceiling {ceiling_bytes} B is below the floor of one "
            f"step's experts ({floor_bytes} B = {num_layers} layers x {top_k} "
            f"experts x {expert_bytes} B)"
        )
    fraction = resident_fraction(table_bytes, ceiling_bytes)
    by_curve = int((table_bytes * fraction) // (num_layers * expert_bytes))
    by_ceiling = int(ceiling_bytes // (num_layers * expert_bytes))
    return max(max(top_k, 1), min(by_curve, by_ceiling))


def expert_transient_bound(groups: Dict[str, List[Tuple[int, int]]], *,
                           capacity: int, staging_bytes: int) -> int:
    """Worst-case device bytes one layer's gathers hold beyond the cache.

    ``groups`` maps a layer (the switch module's parent path) to
    ``(num_experts, expert_bytes)`` per streamed projection.  The worst call
    routes to every expert (``U = E``: a prefill chunk of ``rows x top_k``
    already does so on the target models): the gather stacks ``E`` rows and
    holds ``E - capacity`` rows beyond the steady cache.  Calls of one layer
    accumulate until the next router sync.  Host staging is added on top.
    """
    worst = 0
    for projections in groups.values():
        total = 0
        for (experts, expert_bytes) in projections:
            total += experts * expert_bytes
            total += max(0, experts - min(capacity, experts)) * expert_bytes
        worst = max(worst, total)
    return int(worst + staging_bytes)


# ---------------------------------------------------------------------------
# the managers
# ---------------------------------------------------------------------------


@dataclass
class ExpertStreamPlan:
    layers: int = 0
    experts_per_layer: int = 0
    expert_bytes: int = 0
    table_bytes: int = 0
    capacity_experts: int = 0
    ceiling_bytes: int = 0
    projections: Tuple[str, ...] = ()
    resident_paths: Tuple[str, ...] = ()
    load_phase: str = "post_materialization"
    top_k: int = 1
    transient_bound_bytes: int = 0
    staging_window: int = 0
    manifest_sha256: str = ""
    resident_remainder_bytes: Optional[int] = None

    def as_dict(self) -> Dict[str, object]:
        extra = {"resident_paths": list(self.resident_paths)} if self.resident_paths else {}
        return {
            **extra,
            "layers": self.layers,
            "experts_per_layer": self.experts_per_layer,
            "expert_bytes": self.expert_bytes,
            "table_bytes": self.table_bytes,
            "capacity_experts": self.capacity_experts,
            "ceiling_bytes": self.ceiling_bytes,
            "resident_ceiling_bytes": self.capacity_experts
            * self.layers
            * self.expert_bytes,
            "projections": list(self.projections),
            "load_phase": self.load_phase,
            "load_peak_bounded": self.load_phase == "pre_materialization",
            "top_k": self.top_k,
            "excluded_bytes": self.table_bytes,
            "transient_bound_bytes": self.transient_bound_bytes,
            "reserved_bytes": self.ceiling_bytes + self.transient_bound_bytes,
            "staging_window": self.staging_window,
            "manifest_sha256": self.manifest_sha256,
            "resident_remainder_bytes": self.resident_remainder_bytes,
        }


class _ManagerBase:
    mode = ""

    def reserved_bytes(self) -> int:
        raise NotImplementedError

    def live_bytes(self) -> int:
        return 0

    def unfilled_reserve_bytes(self) -> int:
        """The reservation not already visible as resident device memory.

        Admission measures free memory after resident arrays, so charging the
        full reservation again would count the filled part of the cache twice.
        """
        return max(0, int(self.reserved_bytes()) - int(self.live_bytes()))

    def begin_serving(self) -> None:
        """End the load phase: later page-ins are serving evidence."""
        self.stats.phase = "serve"

    def receipt(self) -> Dict[str, object]:
        plan = self.plan.as_dict()
        return {
            "mode": self.mode,
            "mechanism_revision": MECHANISM_REVISION,
            "load_phase": plan["load_phase"],
            "load_peak_bounded": plan["load_phase"] == "pre_materialization",
            "reserved_bytes": int(self.reserved_bytes()),
            "manifest_sha256": plan.get("manifest_sha256", ""),
            "forced_stock": dict(self.forced_stock or {}),
            "bound_kind": "tracked-weight-bytes",
            "excludes": [
                "os_page_cache", "mlx_allocator_cache", "activations", "kv_cache",
            ],
            "qualification": "unqualified",
        }

    def apc_revision(self) -> str:
        return f"{MECHANISM_REVISION}:{self.mode}:{self.plan.load_phase}"


class ExpertStreamManager(_ManagerBase):
    """Installs and owns per-layer expert caches for one model."""

    mode = "moe_experts"

    def __init__(
        self,
        *,
        plan: ExpertStreamPlan,
        stats: StreamStats,
        pool: Optional[ThreadPoolExecutor],
        caches: Dict[str, ExpertLRU],
        collector=None,
        handles=None,
        streamed_keys: Sequence[str] = (),
    ):
        self.plan = plan
        self.stats = stats
        self.caches = caches
        self.collector = collector
        self.streamed_keys = tuple(streamed_keys)
        self.forced_stock: Dict[str, int] = {}
        self._pool = pool
        self._handles = handles
        self._closed = False

    def reserved_bytes(self) -> int:
        return int(self.plan.ceiling_bytes) + int(self.plan.transient_bound_bytes)

    def live_bytes(self) -> int:
        return sum(
            cache.resident * cache.reader.expert_bytes for cache in self.caches.values()
        )

    def counters(self) -> Dict[str, int]:
        self.stats.resident_bytes = self.live_bytes()
        counters = self.stats.as_dict()
        if self.collector is not None:
            counters.update(self.collector.counters())
        return counters

    def close(self) -> None:
        if self._pool is not None:
            # Running page-ins finish before their descriptors close (see
            # ``_FileHandles.close``); queued ones are dropped.
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
        if self._handles is not None:
            self._handles.close()
            self._handles = None
        if not self._closed:
            self._closed = True
            for cache in self.caches.values():
                cache.clear()
            if self.collector is not None:
                # Engine shutdown closes only this manager: the atlas sink
                # and the trace tail are persisted here or never.  Telemetry
                # must not break shutdown.
                try:
                    self.collector.close()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("expert atlas not persisted at close: %s", exc)


# Load-time fusions this addressing layer can reassemble from checkpoint
# rows by NAME (legacy post-materialization path only).  The fused projection
# is a concatenation along the output axis, so one fused expert row is the
# concatenation of each part's row for that same expert, in this order.
_FUSED_PROJECTIONS = {"gate_up_proj": ("gate_proj", "up_proj")}


def _resolve_spec(index: SafetensorsIndex, path: str, key: str):
    """Byte ranges backing ``<path>.<key>``, direct or reassembled."""
    location = index.get(f"{path}.{key}")
    if location is not None:
        return ExpertSliceSpec.single(key, location)
    (prefix, _, leaf) = path.rpartition(".")
    parts = _FUSED_PROJECTIONS.get(leaf)
    if not parts:
        return None
    located = [index.get(f"{prefix}.{part}.{key}") for part in parts]
    if any(part is None for part in located):
        return None
    # Concatenation is along the output axis (-2 of the stacked tensor,
    # which is axis 0 once the leading expert axis is dropped).
    return ExpertSliceSpec(key=key, parts=tuple(located), concat_axis=0)


def _module_parent(model, path: str):
    parts = path.split(".")
    node = model
    for part in parts[:-1]:
        node = node[int(part)] if isinstance(node, list) else node[part]
    return (node, parts[-1])


def _swap_module(model, path: str, replacement) -> None:
    (parent, leaf) = _module_parent(model, path)
    if isinstance(parent, list):
        parent[int(leaf)] = replacement
    else:
        parent[leaf] = replacement


def is_mtp_path(path: str) -> bool:
    """True for a module under an embedded MTP draft head (``mtp.*``/``*.mtp.*``)."""
    return "mtp" in path.split(".")


def _check_expert_geometry(path: str, module, specs: Sequence[ExpertSliceSpec]) -> None:
    """Checkpoint rows must rebuild the live table's exact rows and dtype."""
    for spec in specs:
        live = module.get(spec.key)
        if live is None:
            continue
        where = f"{path}.{spec.key}"
        if tuple(live.shape[1:]) != spec.expert_shape_concat():
            raise StreamingUnavailable(
                f"{where}: live expert row shape {tuple(live.shape[1:])} != "
                f"checkpoint rows {spec.expert_shape_concat()}"
            )
        for part in spec.parts:
            target = _mx_dtype(part.dtype)
            if target is None:
                raise StreamingUnavailable(
                    f"{where}: safetensors dtype {part.dtype} cannot be materialized"
                )
            if live.dtype != target:
                raise StreamingUnavailable(
                    f"{where}: live dtype {live.dtype} != checkpoint {part.dtype}"
                )


def _manifest(entries, handles) -> str:
    sources = []
    receipt = getattr(handles, "sources_receipt", None)
    if callable(receipt):
        sources = receipt()
    payload = json.dumps(
        {"revision": MECHANISM_REVISION, "tensors": entries, "sources": sources},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _location_entry(location: TensorLocation) -> list:
    return [
        location.path.name, location.begin, location.end, location.dtype,
        list(location.shape),
    ]


def install_expert_streaming(
    model,
    model_path,
    *,
    ceiling_bytes: int,
    top_k: int = 1,
    read_workers: int = DEFAULT_READ_WORKERS,
    collector=None,
    mtp_resident: bool = False,
    shards: Optional[BoundShards] = None,
    resolver: Optional[Callable] = None,
) -> ExpertStreamManager:
    """Replace every streamable ``QuantizedSwitchLinear`` with a streamed one.

    With ``shards`` and ``resolver`` (the early-load path from
    :mod:`mlx2.runtime.streamed_load`) the model still holds unevaluated
    arrays, each tensor's source is proven by ``resolver`` and the manager
    owns ``shards``.  Without them (legacy) shards are found by directory glob
    and tensors by name, after the adapter already materialized everything.

    ``mtp_resident`` keeps the expert tables of an embedded MTP draft head
    (any module path with an ``mtp`` component, e.g. Flash-Next's
    ``mtp.layers.0.mlp.switch_mlp``) resident: a native-MTP route drafts from
    that head every cycle, so streaming it puts page-ins on the draft path
    (omlx #3935). Kept tables are not in the plan's layers or ceiling; they
    are charged as ordinary resident weights, and listed in
    ``plan.resident_paths``.

    All replacements are built before any is swapped in; on failure the read
    pool and descriptors are closed and the model is left untouched.

    Raises :class:`StreamingUnavailable` when a tensor cannot be addressed
    exactly, and :class:`WorkingSetTooSmall` when the ceiling cannot hold one
    step.
    """
    import mlx.core as mx

    from .models.switch_layers import QuantizedSwitchLinear

    early = shards is not None
    if early and resolver is None:
        raise ValueError("early-load expert streaming needs a source resolver")
    handles = shards if early else None
    pool = None
    try:
        if early:
            index = shards.index
        else:
            index = SafetensorsIndex.from_model_path(model_path)
            handles = _FileHandles()

        targets: List[Tuple[str, object, List[ExpertSliceSpec]]] = []
        resident_paths: List[str] = []
        for (path, module) in model.named_modules():
            if not isinstance(module, QuantizedSwitchLinear):
                continue
            if is_streamed_module(module):
                raise StreamingUnavailable(
                    f"{path!r} is already streamed; streaming installs once"
                )
            if mtp_resident and is_mtp_path(path):
                resident_paths.append(path)
                continue
            if early and getattr(module, "mode", "affine") == "nvfp4":
                raise StreamingUnavailable(
                    f"{path!r}: nvfp4 gathers may take a dequantized dense tail "
                    "whose transient is not reserved"
                )
            specs = []
            for key in ("weight", "scales", "biases"):
                value = module.get(key)
                if value is None:
                    continue
                spec = resolver(path, key, value) if early else _resolve_spec(index, path, key)
                if spec is None:
                    specs = []
                    break
                specs.append(spec)
            if not specs:
                raise StreamingUnavailable(
                    f"checkpoint has no byte-addressable expert rows for {path!r}; "
                    "this model's expert tensors are renamed, or fused from parts "
                    "this addressing layer does not know how to reassemble"
                )
            expected = int(module.num_experts)
            for spec in specs:
                if spec.num_experts != expected:
                    raise StreamingUnavailable(
                        f"{path!r} has {expected} experts but the checkpoint rows "
                        f"for {spec.key} have {spec.num_experts}; a load-time "
                        "transform (for example folding the shared expert into "
                        "the gather) has changed the table and byte ranges no "
                        "longer address it"
                    )
            _check_expert_geometry(path, module, specs)
            targets.append((path, module, specs))

        if not targets:
            raise StreamingUnavailable("model has no quantized routed experts")

        expert_bytes = max(
            sum(spec.row_bytes() for spec in specs) for (_, _, specs) in targets
        )
        experts_per_layer = max(specs[0].num_experts for (_, _, specs) in targets)
        table_bytes = sum(
            specs[0].num_experts * sum(spec.row_bytes() for spec in specs)
            for (_, _, specs) in targets
        )
        capacity = plan_cache_experts(
            table_bytes=table_bytes,
            expert_bytes=expert_bytes,
            num_layers=len(targets),
            top_k=top_k,
            ceiling_bytes=ceiling_bytes,
        )
        workers = max(1, int(read_workers))
        window = workers * STAGING_WINDOW_PER_WORKER
        groups: Dict[str, List[Tuple[int, int]]] = {}
        for (path, _, specs) in targets:
            groups.setdefault(path.rpartition(".")[0], []).append(
                (specs[0].num_experts, sum(spec.row_bytes() for spec in specs))
            )
        transient = expert_transient_bound(
            groups, capacity=capacity, staging_bytes=window * expert_bytes
        )
        manifest = _manifest(
            [
                [path, spec.key, [_location_entry(p) for p in spec.parts], spec.concat_axis]
                for (path, _, specs) in targets
                for spec in specs
            ],
            handles,
        )

        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="expert-stream")
        stats = StreamStats(phase="load" if early else "serve")
        caches: Dict[str, ExpertLRU] = {}
        replacements = []
        streamed_keys = []
        for (layer_index, (path, module, specs)) in enumerate(targets):
            reader = ExpertSliceReader(specs, handles=handles, pool=pool, window=window)
            cache = ExpertLRU(
                reader, capacity_experts=capacity, stats=stats,
                group=path.rpartition(".")[0],
            )
            caches[path] = cache
            replacements.append((path, _streamed_switch_linear(
                module,
                reader,
                cache,
                layer_index=layer_index,
                unit_offset=0,
                collector=collector,
            )))
            streamed_keys.extend(f"{path}.{spec.key}" for spec in specs)
        for (path, replacement) in replacements:
            _swap_module(model, path, replacement)
        projections = tuple(spec.key for spec in targets[0][2])
        # Drop the stacked tables; the streamed modules own their bytes now.
        # The replaced modules sit in reference cycles (nn.Module holds its
        # parameters in a dict that the parent module also references), so
        # refcounting alone does not release them and the tables stay resident
        # for an arbitrary time.  Collect explicitly before clearing, or the
        # whole point of installing -- dropping the table -- silently does not
        # happen.  Measured on an M3: without this, active memory after install
        # was unchanged at 19.68 GiB.  Loop variables hold a replaced module
        # too, so they are dropped with the lists.
        module = specs = value = None
        del targets, replacements
        gc.collect()
        mx.clear_cache()

        plan = ExpertStreamPlan(
            layers=len(caches),
            experts_per_layer=experts_per_layer,
            expert_bytes=expert_bytes,
            table_bytes=table_bytes,
            capacity_experts=capacity,
            ceiling_bytes=int(ceiling_bytes),
            projections=projections,
            resident_paths=tuple(sorted(resident_paths)),
            load_phase="pre_materialization" if early else "post_materialization",
            top_k=int(top_k),
            transient_bound_bytes=transient,
            staging_window=window,
            manifest_sha256=manifest,
        )
        if collector is not None:
            collector.bind(num_layers=plan.layers, num_units=plan.experts_per_layer)
        return ExpertStreamManager(
            plan=plan,
            stats=stats,
            pool=pool,
            caches=caches,
            collector=collector,
            handles=handles,
            streamed_keys=streamed_keys,
        )
    except BaseException:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        if handles is not None:
            handles.close()
        raise


# ---------------------------------------------------------------------------
# dense projection paging
# ---------------------------------------------------------------------------


class ProjectionReader:
    """Whole-tensor reads of one quantized projection, into one buffer each."""

    def __init__(self, specs: Dict[str, TensorLocation], *, handles, pool,
                 stats: StreamStats, chunk_bytes: int = DENSE_READ_CHUNK_BYTES):
        for (key, location) in specs.items():
            if _SAFETENSORS_DTYPES.get(location.dtype) is None:
                raise StreamingUnavailable(
                    f"{key}: safetensors dtype {location.dtype} cannot be materialized"
                )
        self.specs = dict(specs)
        self._handles = handles
        self._pool = pool
        self._stats = stats
        self._chunk = max(1, int(chunk_bytes))
        self.nbytes = sum(location.nbytes for location in self.specs.values())
        self.source_paths = tuple(dict.fromkeys(l.path for l in self.specs.values()))

    def load(self) -> Dict[str, object]:
        """Concrete arrays for every tensor of the projection.  Model thread."""
        import mlx.core as mx
        import numpy as np

        for path in self.source_paths:
            self._handles.check(path)
        buffers = {
            key: np.empty(location.nbytes, dtype=np.uint8)
            for (key, location) in self.specs.items()
        }
        self._stats.note_staging(self.nbytes)
        jobs = []
        for (key, location) in self.specs.items():
            view = memoryview(buffers[key])
            for start in range(0, location.nbytes, self._chunk):
                stop = min(location.nbytes, start + self._chunk)
                jobs.append((location.path, location.begin + start, view[start:stop]))
        if self._pool is None or len(jobs) <= 1:
            for job in jobs:
                self._handles.readinto(*job)
        else:
            futures = [self._pool.submit(self._handles.readinto, *job) for job in jobs]
            wait(futures)
            for future in futures:
                future.result()
        # Re-check after the reads: a mutation during I/O fails here, before
        # the bytes become arrays any forward can consume.
        for path in self.source_paths:
            self._handles.check(path)
        arrays = {}
        for (key, location) in self.specs.items():
            (numpy_code, mlx_name) = _SAFETENSORS_DTYPES[location.dtype]
            array = mx.array(buffers[key].view(np.dtype(numpy_code)))
            target = getattr(mx, mlx_name)
            if array.dtype != target:
                array = array.view(target)
            arrays[key] = array.reshape(location.shape)
        mx.eval(*arrays.values())
        # Host buffer plus device copy coexist until here: 2x the projection.
        # Each call is its own window: the previous projection's output was
        # evaluated before its arrays were released.
        self._stats.note_transient(id(self), 2 * self.nbytes, member=id(self))
        del buffers
        self._stats.bump("page_ins")
        self._stats.bump("page_in_bytes", self.nbytes)
        return arrays


_DENSE_CLASS = None


def _dense_class():
    """A plain module, deliberately NOT a ``QuantizedLinear`` subclass.

    Every alternate consumer of projection storage (fused GDN/QSA tables,
    lane matmul, sp_qmm, int8 prefill, row-exact verify, TensorFold, LoRA)
    selects on ``QuantizedLinear`` type or instance; a plain module is skipped
    by all of them rather than read as a resident table that is not there.
    """
    global _DENSE_CLASS
    if _DENSE_CLASS is not None:
        return _DENSE_CLASS

    import mlx.core as mx
    import mlx.nn as nn

    class StreamedQuantizedProjection(nn.Module):
        def __init__(self, *, reader, group_size, bits, mode, input_dims,
                     output_dims, bias):
            super().__init__()
            self.group_size = group_size
            self.bits = bits
            self.mode = mode
            if bias is not None:
                self["bias"] = bias
            self._stream_reader = reader
            self._stream_input_dims = int(input_dims)
            self._stream_output_dims = int(output_dims)
            self.freeze(recurse=False)

        @property
        def input_dims(self):
            return self._stream_input_dims

        @property
        def output_dims(self):
            return self._stream_output_dims

        def __call__(self, x):
            arrays = self._stream_reader.load()
            # Exactly nn.QuantizedLinear.__call__ on the same bytes.
            out = mx.quantized_matmul(
                x,
                arrays["weight"],
                scales=arrays["scales"],
                biases=arrays.get("biases"),
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
                mode=self.mode,
            )
            if "bias" in self:
                out = out + self["bias"]
            # Evaluate before the projection is released: otherwise the lazy
            # graph would retain every projection of the forward.
            mx.eval(out)
            del arrays
            return out

    _DENSE_CLASS = StreamedQuantizedProjection
    return _DENSE_CLASS


@dataclass
class DenseStreamPlan:
    projections: int = 0
    max_projection_bytes: int = 0
    excluded_bytes: int = 0
    staging_ceiling_bytes: int = 0
    transient_bound_bytes: int = 0
    read_workers: int = 0
    manifest_sha256: str = ""
    load_phase: str = "pre_materialization"
    resident_remainder_bytes: Optional[int] = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "projections": self.projections,
            "max_projection_bytes": self.max_projection_bytes,
            "excluded_bytes": self.excluded_bytes,
            "staging_ceiling_bytes": self.staging_ceiling_bytes,
            "transient_bound_bytes": self.transient_bound_bytes,
            "reserved_bytes": self.staging_ceiling_bytes,
            "read_workers": self.read_workers,
            "manifest_sha256": self.manifest_sha256,
            "load_phase": self.load_phase,
            "load_peak_bounded": self.load_phase == "pre_materialization",
            "resident_remainder_bytes": self.resident_remainder_bytes,
        }


class DenseStreamManager(_ManagerBase):
    mode = "dense_mlp"

    def __init__(self, *, plan: DenseStreamPlan, stats: StreamStats, pool, handles,
                 streamed_keys: Sequence[str]):
        self.plan = plan
        self.stats = stats
        self.streamed_keys = tuple(streamed_keys)
        self.forced_stock: Dict[str, int] = {}
        self._pool = pool
        self._handles = handles

    def reserved_bytes(self) -> int:
        return int(self.plan.staging_ceiling_bytes)

    def counters(self) -> Dict[str, int]:
        stats = self.stats
        return {
            "dense_stream_page_ins_total": stats.page_ins,
            "dense_stream_page_in_bytes_total": stats.page_in_bytes,
            "dense_stream_load_page_ins_total": stats.load_page_ins,
            "dense_stream_load_page_in_bytes_total": stats.load_page_in_bytes,
            "dense_stream_peak_tracked_bytes": stats.peak_tracked_bytes,
            "dense_stream_peak_staging_bytes": stats.peak_staging_bytes,
        }

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
        if self._handles is not None:
            self._handles.close()
            self._handles = None


def install_dense_streaming(
    model,
    *,
    targets: Sequence[str],
    shards: BoundShards,
    resolver: Callable,
    staging_bytes: int,
    read_workers: int = DEFAULT_READ_WORKERS,
) -> DenseStreamManager:
    """Replace the named quantized projections with page-per-call modules.

    ``targets`` are module paths chosen by the adapter.  Each must be a plain
    ``nn.QuantizedLinear`` (not one an installer already wrapped) whose every
    tensor resolves to an untransformed checkpoint tensor.  The staging budget
    must hold the largest projection twice (host buffer + device copy).
    """
    import mlx.core as mx
    import mlx.nn as nn

    pool = None
    try:
        if not targets:
            raise StreamingUnavailable("no dense projections were selected")
        modules = dict(model.named_modules())
        resolved = []
        for path in targets:
            module = modules.get(path)
            if module is None:
                raise StreamingUnavailable(f"{path!r} is not a module of this model")
            if type(module) is not nn.QuantizedLinear:
                raise StreamingUnavailable(
                    f"{path!r} is {type(module).__name__}, not a plain QuantizedLinear"
                )
            if getattr(module, "mode", "affine") == "nvfp4":
                raise StreamingUnavailable(f"{path!r}: nvfp4 projections are not streamed")
            specs = {}
            for key in ("weight", "scales", "biases"):
                value = module.get(key)
                if value is None:
                    continue
                specs[key] = resolver(path, key, value)
            if "weight" not in specs or "scales" not in specs:
                raise StreamingUnavailable(f"{path!r} has no quantized weight/scales")
            resolved.append((path, module, specs))
        largest = max(sum(l.nbytes for l in specs.values()) for (_, _, specs) in resolved)
        if int(staging_bytes) < 2 * largest:
            raise WorkingSetTooSmall(
                f"dense staging budget {int(staging_bytes)} B is below twice the "
                f"largest selected projection ({2 * largest} B)"
            )
        excluded = sum(sum(l.nbytes for l in specs.values()) for (_, _, specs) in resolved)
        manifest = _manifest(
            [
                [path, key, [_location_entry(location)], 0]
                for (path, _, specs) in resolved
                for (key, location) in sorted(specs.items())
            ],
            shards,
        )
        workers = max(1, int(read_workers))
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dense-stream")
        stats = StreamStats(phase="load")
        cls = _dense_class()
        replacements = []
        streamed_keys = []
        for (path, module, specs) in resolved:
            reader = ProjectionReader(specs, handles=shards, pool=pool, stats=stats)
            (out_dims, packed) = module["weight"].shape
            replacements.append((path, cls(
                reader=reader,
                group_size=module.group_size,
                bits=module.bits,
                mode=getattr(module, "mode", "affine"),
                input_dims=packed * 32 // module.bits,
                output_dims=out_dims,
                bias=module["bias"] if "bias" in module else None,
            )))
            streamed_keys.extend(f"{path}.{key}" for key in specs)
        for (path, replacement) in replacements:
            _swap_module(model, path, replacement)
        module = specs = value = None
        del resolved, replacements, modules
        gc.collect()
        mx.clear_cache()
        plan = DenseStreamPlan(
            projections=len(targets),
            max_projection_bytes=largest,
            excluded_bytes=excluded,
            staging_ceiling_bytes=int(staging_bytes),
            transient_bound_bytes=2 * largest,
            read_workers=workers,
            manifest_sha256=manifest,
        )
        return DenseStreamManager(
            plan=plan, stats=stats, pool=pool, handles=shards, streamed_keys=streamed_keys
        )
    except BaseException:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        shards.close()
        raise


# ---------------------------------------------------------------------------
# APCv2 numerics identity
# ---------------------------------------------------------------------------

APC_WRAPPER_TAG = "weight-stream"


def apc_weight_stream_fingerprint(semantic, manager):
    """Wrap an APCv2 semantic namespace when weights are streamed.

    A streamed table makes table-reading kernels take the stock path, so
    states computed under streaming must not be shared with states computed
    by a resident route's default kernels.  Unchanged when nothing streams.
    """
    if manager is None:
        return semantic
    return (semantic, APC_WRAPPER_TAG, manager.apc_revision())

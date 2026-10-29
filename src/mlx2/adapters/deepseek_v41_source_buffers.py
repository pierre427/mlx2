# SPDX-License-Identifier: MIT
"""One immutable expert backing, exact typed views and bounded host staging.

Provenance: provenance/deepseek-v41-source-buffers.json. No MLX work on import
or reader workers. The provider owns pending host/upload references and must
evaluate uploads before releasing host staging, then retain expert leases until
all GPU consumers complete. This module does not change those lifetimes.
"""
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
import os
import weakref

import numpy as np

ORIGINAL_BYTES = 18_800_640
Q1_BYTES = 6_635_520
_PARTS = (("gate", "w1"), ("up", "w3"), ("down", "w2"))
_COMPACT_UPLOAD_ISSUER = object()


def _schema(kind):
    if kind not in ("original", "q1"):
        raise ValueError("unknown packed expert format")
    result = {}
    for part, _ in _PARTS:
        n, k = (5120, 2304) if part == "down" else (2304, 5120)
        result[part + ".qweight"] = ("<u4", (n, k // (8 if kind == "original" else 32)))
        result[part + ".scales"] = ("|u1", (n, k // 32)) if kind == "original" else ("<f2", (n, k // 64))
        if kind == "q1":
            result[part + ".biases"] = ("<f2", (n, k // 64))
    return result


@dataclass(frozen=True)
class TensorSlice:
    name: str
    offset: int
    dtype: str
    shape: tuple

    @property
    def nbytes(self):
        return int(np.prod(self.shape)) * np.dtype(self.dtype).itemsize


@dataclass(frozen=True)
class ExpertLayout:
    kind: str
    tensors: tuple

    def __post_init__(self):
        schema = _schema(self.kind)
        if (type(self.tensors) is not tuple or len(self.tensors) != len(schema)
                or any(not isinstance(t, TensorSlice) for t in self.tensors)
                or {t.name for t in self.tensors} != set(schema)):
            raise ValueError("packed expert tensor inventory mismatch")
        cursor = 0
        for t in sorted(self.tensors, key=lambda t: t.offset if type(t.offset) is int else -1):
            dtype, shape = schema[t.name]
            if (type(t.offset) is not int or t.offset != cursor
                    or type(t.shape) is not tuple or t.shape != shape
                    or any(type(n) is not int for n in t.shape)
                    or t.dtype != dtype or t.offset % np.dtype(dtype).itemsize):
                raise ValueError("packed expert tensor geometry/alignment mismatch")
            cursor += t.nbytes
        if cursor != self.nbytes:
            raise ValueError("packed expert payload size mismatch")

    @property
    def nbytes(self):
        return ORIGINAL_BYTES if self.kind == "original" else Q1_BYTES

    def tensor(self, name):
        return next(t for t in self.tensors if t.name == name)


def canonical_layout(kind):
    """CPU descriptor; offsets refer to the payload-only backing."""
    cursor, tensors = 0, []
    for name, (dtype, shape) in _schema(kind).items():
        t = TensorSlice(name, cursor, dtype, shape)
        tensors.append(t)
        cursor += t.nbytes
    return ExpertLayout(kind, tuple(tensors))


def _owner(array):
    while isinstance(array.base, np.ndarray):
        array = array.base
    if (array.base is not None or not array.flags.owndata
            or array.dtype != np.uint8 or array.ndim != 1 or not array.flags.c_contiguous):
        raise ValueError("packed host tensors require one owned contiguous uint8 slab")
    return array


class HostExpert:
    """Payload view retaining exactly one host allocation, including its header."""
    __slots__ = ("_data", "_layout")

    def __init__(self, data, layout):
        if (not isinstance(layout, ExpertLayout) or not isinstance(data, np.ndarray)
                or data.dtype != np.uint8 or data.ndim != 1
                or not data.flags.c_contiguous or data.nbytes != layout.nbytes):
            raise ValueError("packed host payload geometry mismatch")
        owner = _owner(data)
        first = data.ctypes.data - owner.ctypes.data
        if first < 0 or first + data.nbytes != owner.nbytes:
            raise ValueError("packed host payload must cover its allocation tail")
        if (layout.kind == "original" and first != 0) or not 0 <= first <= (1 << 20) + 8:
            raise ValueError("packed host header exceeds admitted layout")
        if any((data.ctypes.data + t.offset) % np.dtype(t.dtype).itemsize for t in layout.tensors):
            raise ValueError("packed host tensor alignment mismatch")
        self._data = data.view()
        self._data.flags.writeable = False
        self._layout = layout

    @property
    def data(self):
        return self._data

    @property
    def layout(self):
        return self._layout

    @property
    def host_allocation_nbytes(self):
        return _owner(self._data).nbytes

    def views(self):
        return {t.name: self.data[t.offset:t.offset+t.nbytes].view(t.dtype).reshape(t.shape)
                for t in self.layout.tensors}


def borrow_q1_packed(tensors):
    """Borrow read_native_file's payload views; never concatenate/copy them.

    Input must already have passed the native file seal/hash/header checks.
    This validates tensor geometry and storage ownership, not file provenance.
    """
    schema = _schema("q1")
    if not isinstance(tensors, Mapping) or set(tensors) != set(schema):
        raise ValueError("Q1 packed tensor inventory mismatch")
    owner, rows = None, []
    for name, (dtype, shape) in schema.items():
        a = tensors[name]
        if (not isinstance(a, np.ndarray) or a.dtype.str != dtype or a.shape != shape
                or not a.flags.c_contiguous):
            raise ValueError("Q1 packed tensor dtype/shape mismatch")
        root = _owner(a)
        if owner is not None and root is not owner:
            raise ValueError("Q1 tensors do not share the same owner")
        owner = root
        rows.append((name, a.ctypes.data - root.ctypes.data, dtype, shape))
    first = min(row[1] for row in rows)
    layout = ExpertLayout("q1", tuple(TensorSlice(n, off-first, dt, sh) for n, off, dt, sh in rows))
    return HostExpert(owner[first:first+layout.nbytes], layout)


def _read_into(fd, offset, destination):
    """Read into the existing slab; neither backend creates a payload bytes copy."""
    if hasattr(os, "preadv"):
        return os.preadv(fd, [destination], offset)
    # Descriptor belongs only to this reader. Unbuffered readinto handles Darwin,
    # where Python may not expose preadv, without os.pread's temporary bytes.
    with os.fdopen(fd, "rb", buffering=0, closefd=False) as stream:
        stream.seek(offset)
        return stream.readinto(destination)


def read_original_packed(artifact, layer, expert, *, read_into=None):
    """Fill one expert slab directly from exact, sealed official source extents.

    Optional read_into(fd, offset, memoryview) returns the count actually filled;
    the provider hook must not retain destination views or allocate another slab.
    Short reads are retried; zero/invalid progress refuses. No tensor bytes are
    returned before descriptor and path seals have both been rechecked.
    """
    from .deepseek_v41_source_artifact import checked_file
    from ..runtime.immutable_weight_pages import file_stamp

    if type(layer) is not int or not 0 <= layer < 40 or type(expert) is not int or not 0 <= expert < 384:
        raise ValueError("packed original expert identity out of range")
    layout, shards = canonical_layout("original"), {}
    for part, source_part in _PARTS:
        for field, source_field, dtype in (("qweight", "weight", "I8"), ("scales", "scale", "F8_E8M0")):
            key = f"layers.{layer}.ffn.experts.{expert}.{source_part}.{source_field}"
            row = artifact["tensors"][key]
            target = layout.tensor(part + "." + field)
            shape = (target.shape[0], target.shape[1]*4) if field == "qweight" else target.shape
            if (row["dtype"] != dtype or tuple(row["shape"]) != shape
                    or type(row["size"]) is not int or row["size"] != target.nbytes
                    or type(row["offset"]) is not int or row["offset"] < 0
                    or row["offset"] + row["size"] > artifact["seals"][row["file"]]["size"]):
                raise ValueError("packed original source extent geometry mismatch")
            shards.setdefault(row["file"], []).append((row, target))
    # Refuse overlapping source extents before allocating staging.
    for rows in shards.values():
        ordered = sorted((r["offset"], r["offset"]+r["size"]) for r, _ in rows)
        if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
            raise ValueError("packed original source extents overlap")
    slab = np.empty(layout.nbytes, np.uint8)
    reader = _read_into if read_into is None else read_into
    for shard, rows in shards.items():
        seal = artifact["seals"][shard]
        path = checked_file(artifact["root"], shard, seal)
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            if file_stamp(os.fstat(fd)) != seal:
                raise ValueError("source changed before packed read")
            if os.uname().sysname == "Darwin":
                import fcntl
                fcntl.fcntl(fd, 48, 1)
                fcntl.fcntl(fd, 45, 0)
            for row, target in rows:
                cursor = 0
                while cursor < target.nbytes:
                    stop = min(cursor+(1 << 20), target.nbytes)
                    destination = memoryview(slab)[target.offset+cursor:target.offset+stop]
                    try:
                        count = reader(fd, row["offset"]+cursor, destination)
                    finally:
                        destination.release()
                    if type(count) is not int or not 0 < count <= stop-cursor:
                        raise OSError("short or invalid packed expert read")
                    cursor += count
            if file_stamp(os.fstat(fd)) != seal:
                raise ValueError("source changed during packed read")
            checked_file(artifact["root"], shard, seal)
        finally:
            os.close(fd)
    return HostExpert(slab, layout)


class PackedExpert(Mapping):
    """Only tensor entries participate in values()/items() and byte summation."""
    __slots__ = ("_backing", "_layout", "_tensors", "_compact_issuer", "_compact_backing")

    def __init__(self, backing, layout, tensors, *, _compact_issuer=None):
        self._backing, self._layout, self._tensors = backing, layout, tensors
        self._compact_issuer, self._compact_backing = None, None
        if _compact_issuer is _COMPACT_UPLOAD_ISSUER:
            try:
                self._compact_backing = weakref.ref(backing)
                self._compact_issuer = _COMPACT_UPLOAD_ISSUER
            except TypeError:
                # Foreign CPU test backends may lack weakrefs. Ordinary APIs
                # remain valid, but a no-copy pointer kernel must refuse them.
                pass

    def __getitem__(self, name):
        return self._tensors[name]

    def __iter__(self):
        return iter(self._tensors)

    def __len__(self):
        return len(self._tensors)

    @property
    def backing(self):
        return self._backing

    @property
    def layout(self):
        return self._layout

    @property
    def allocation_nbytes(self):
        return self._layout.nbytes

    @property
    def has_compact_backing(self):
        """Loader proof only; no strides readback, new buffer or strong owner.

        Bound to the exact fresh upload object, so replacing the private backing
        or reusing a released Python address cannot inherit this capability.
        This is an internal ownership contract, not an adversarial sandbox.
        Provenance: deepseek-v41-source-down-worklist.json.
        """
        return bool(self._compact_issuer is _COMPACT_UPLOAD_ISSUER
                    and self._compact_backing is not None
                    and self._compact_backing() is self._backing)


def materialize(host, *, stream=None):
    """Schedule one MLX backing and typed views, without evaluation or host refs.

    Install the returned Mapping in provider pending_uploads before evaluating
    mx.eval(packed.backing, *packed.values()) (do not rely on generic Mapping
    traversal by the evaluator);
    retain host in pending_host until that evaluation succeeds. Failure/cancel
    follows the provider's existing quarantine/completion contract. CPU tests
    pass mx.cpu explicitly; default uses the caller's current stream.
    """
    if not isinstance(host, HostExpert):
        raise ValueError("materialize requires validated HostExpert")
    import mlx.core as mx
    with mx.stream(stream) if stream is not None else nullcontext():
        backing = mx.array(host.data)
        dtypes = {"<u4": mx.uint32, "|u1": mx.uint8, "<f2": mx.float16}
        tensors = {t.name: backing[t.offset:t.offset+t.nbytes].view(dtypes[t.dtype]).reshape(t.shape)
                   for t in host.layout.tensors}
    return PackedExpert(backing, host.layout, tensors, _compact_issuer=_COMPACT_UPLOAD_ISSUER)

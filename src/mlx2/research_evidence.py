# SPDX-License-Identifier: Apache-2.0 AND MIT
"""Typed byte evidence and bounded research archives, independent of devices.

Callers own materialization, numerical laws, readable state and model schemas.
This module neither imports MLX nor publishes serving/cache state. Array archive
decoding uses the caller's NumPy backend only after all headers are checked.
Provenance: provenance/library-alignment-a2-20261027.json.
"""
import ast
import hashlib
import io
import math
import os
import re
import stat
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OracleAtom:
    kind: str
    dtype: str = ""
    shape: tuple = ()
    payload: object = None

    def summary(self):
        if self.kind == "array":
            return f"array(dtype={self.dtype}, shape={self.shape}, bytes={len(self.payload)})"
        return f"{self.kind}({self.payload!r})"


def array_atom(*, dtype, shape, payload):
    """Snapshot already-materialized logical C-order bytes without a cast."""
    shape = tuple(shape)
    if not isinstance(dtype, str) or not dtype or any(type(v) is not int or v < 0 for v in shape):
        raise ValueError("raw array evidence needs dtype and nonnegative integer shape")
    return OracleAtom("array", dtype, shape, bytes(payload))


def host_atom(value):
    """Exact host scalar evidence; callers unwrap library-specific scalars."""
    if value is None:
        return OracleAtom("none")
    payload = struct.pack("!d", value) if isinstance(value, float) else value
    if not isinstance(value, (float, bool, int, str, bytes)):
        payload = repr(value)
    return OracleAtom("host", type(value).__name__, payload=payload)


def diff_atoms(expected, actual):
    """Return deterministic field-level differences; no numeric tolerance.

    Array payload comparisons distinguish signed zero and NaN payload bits.
    Model-specific traversal, masking and near-tie interpretation stay outside.
    """
    differences = []
    for path in sorted(set(expected) | set(actual)):
        if path not in expected:
            differences.append((path, "unexpected"))
        elif path not in actual:
            differences.append((path, "missing"))
        else:
            left, right = expected[path], actual[path]
            for field in ("kind", "dtype", "shape", "payload"):
                if getattr(left, field) != getattr(right, field):
                    differences.append((path, field))
    return tuple(differences)


def _read_v3_header(value, backend):
    """Bounded NPY 3.0 public format: little-endian length and UTF-8 dict."""
    raw_length = value.read(4)
    if len(raw_length) != 4:
        raise ValueError("Array archive truncated NPY header length")
    length = struct.unpack('<I', raw_length)[0]
    if length > 4096:
        raise ValueError("Array archive NPY header exceeds bound")
    raw = value.read(length)
    if len(raw) != length:
        raise ValueError("Array archive truncated NPY header")
    try:
        header = ast.literal_eval(raw.decode('utf-8'))
    except (UnicodeError, SyntaxError, ValueError, RecursionError) as error:
        raise ValueError("Array archive invalid NPY header") from error
    if (type(header) is not dict or set(header) != {'descr', 'fortran_order', 'shape'}
            or type(header['fortran_order']) is not bool or type(header['shape']) is not tuple
            or len(header['shape']) > 32
            or any(type(v) is not int or v < 0 for v in header['shape'])):
        raise ValueError("Array archive invalid NPY header fields")
    try:
        dtype = backend.lib.format.descr_to_dtype(header['descr'])
    except TypeError as error:
        raise ValueError("Array archive invalid NPY header dtype") from error
    return header['shape'], header['fortran_order'], dtype


def load_array_archive(path, *, specs, sha256, identity, expected_identity,
                       max_archive_bytes, max_array_bytes, max_container_bytes, backend,
                       allowed_versions=((1, 0), (2, 0))):
    """Load a sealed NPZ after identity, inventory and every NPY header pass.

    ``specs`` maps exact array names to ``(shape, dtype)``. The caller provides
    its existing NumPy backend and performs semantic/finite/route checks after
    return. Reading once into bounded immutable bytes prevents hash/load races.
    """
    if (not isinstance(allowed_versions, tuple) or not allowed_versions
            or any(version not in ((1, 0), (2, 0), (3, 0)) for version in allowed_versions)):
        raise ValueError("Array archive NPY version policy invalid")
    if not identity or not expected_identity or any(
        not identity.get(key) or identity.get(key) != value
        for key, value in expected_identity.items()
    ):
        raise ValueError("Array archive identity mismatch")
    for limit in (max_archive_bytes, max_array_bytes, max_container_bytes):
        if type(limit) is not int or limit < 0:
            raise ValueError("Array archive byte budget must be nonnegative integer")
    if not specs or not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise ValueError("Array archive inventory/hash missing")
    expected = {}
    array_bytes = 0
    for key, (shape, dtype) in specs.items():
        shape, dtype = tuple(shape), backend.dtype(dtype)
        if (not re.fullmatch(r"[A-Za-z0-9_.-]+", key) or len(shape) > 32
                or any(type(v) is not int or v < 0 for v in shape)
                or dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None
                or dtype.kind not in "biufc" or dtype.itemsize <= 0):
            raise ValueError("Array archive NPY header specification invalid")
        size = math.prod(shape) * dtype.itemsize
        array_bytes += size
        expected[key + ".npy"] = (shape, dtype, size)
    if array_bytes > max_array_bytes:
        raise ValueError("Array archive NPY header exceeds array budget")
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Array archive symlink")
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("Array archive must be a regular file")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(fd, "rb") as source:
        metadata = os.fstat(source.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("Array archive must remain a regular file")
        if metadata.st_size > max_archive_bytes:
            raise ValueError("Array archive file budget")
        raw = source.read(max_archive_bytes + 1)
    if len(raw) > max_archive_bytes or hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("Array archive file budget/hash mismatch")
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entries = archive.infolist()
            if (len(entries) != len(expected) or {v.filename for v in entries} != set(expected)
                    or sum(v.file_size for v in entries) > max_container_bytes):
                raise ValueError("Array archive decompression inventory/budget")
            for entry in entries:
                if entry.flag_bits & 1 or entry.compress_type not in (
                    zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED, zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA
                ):
                    raise ValueError("Array archive encrypted/unsupported ZIP entry")
                shape, dtype, size = expected[entry.filename]
                with archive.open(entry) as value:
                    version = backend.lib.format.read_magic(value)
                    if version not in allowed_versions:
                        raise ValueError("Array archive unsupported NPY header version")
                    if version == (1, 0):
                        header = backend.lib.format.read_array_header_1_0(value, max_header_size=4096)
                    elif version == (2, 0):
                        header = backend.lib.format.read_array_header_2_0(value, max_header_size=4096)
                    else:
                        header = _read_v3_header(value, backend)
                    if header[0] != shape or header[2] != dtype or entry.file_size - value.tell() != size:
                        raise ValueError("Array archive NPY header geometry/dtype/payload")
    except (zipfile.BadZipFile, EOFError) as error:
        raise ValueError("Array archive invalid container/NPY header") from error
    with backend.load(io.BytesIO(raw), allow_pickle=False) as arrays:
        return {key: backend.array(arrays[key], copy=True) for key in specs}

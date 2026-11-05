"""Preflight NPZ/NPY geometry before NumPy allocates a declared array."""

from __future__ import annotations

import io
import math
import zipfile

import numpy as np


def bounded_npz_arrays(raw, *, maximum_bytes=64 * 1024**2):
    if not isinstance(raw, bytes) or len(raw) > maximum_bytes:
        raise ValueError("NPZ input exceeds its byte bound")
    if type(maximum_bytes) is not int or maximum_bytes < 1:
        raise ValueError("positive NPZ byte bound required")
    metadata = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        if (not 1 <= len(entries) <= 128 or sum(entry.file_size for entry in entries) > maximum_bytes
                or len({entry.filename for entry in entries}) != len(entries)):
            raise ValueError("NPZ archive exceeds expanded bounds or repeats entries")
        for entry in entries:
            name = entry.filename
            if (not name.endswith(".npy") or not name[:-4] or "/" in name or "\\" in name
                    or entry.flag_bits & 1):
                raise ValueError("NPZ entries must be unique flat unencrypted NPY arrays")
            key = name[:-4]
            if key in metadata:
                raise ValueError("duplicate normalized NPZ array key")
            with archive.open(entry) as stream:
                version = np.lib.format.read_magic(stream)
                readers = {(1, 0): np.lib.format.read_array_header_1_0,
                           (2, 0): np.lib.format.read_array_header_2_0}
                if version not in readers:
                    raise ValueError("only numeric NPY format versions 1 and 2 are supported")
                # Read the header BEFORE np.load can allocate its claimed shape.
                shape, fortran, dtype = readers[version](stream, max_header_size=10000)
                if (dtype.kind not in "biuf" or dtype.hasobject or dtype.fields is not None
                        or dtype.subdtype is not None or dtype.itemsize > 16
                        or len(shape) > 4 or any(type(size) is not int or size < 0 for size in shape)):
                    raise ValueError("NPZ arrays must have bounded plain numeric geometry")
                size = math.prod(shape) * dtype.itemsize
                if size > maximum_bytes or size != entry.file_size - stream.tell():
                    raise ValueError("NPY declared geometry exceeds its actual bounded payload")
                metadata[key] = (shape, fortran, dtype)
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        if set(archive.files) != set(metadata):
            raise ValueError("NPZ keys changed after header validation")
        arrays = {name: archive[name] for name in archive.files}
        if any(array.shape != metadata[name][0] or array.dtype != metadata[name][2]
               for name, array in arrays.items()):
            raise ValueError("NPZ array differs from its validated header")
        return arrays

"""Bounded GGUF metadata/index reader and zero-copy packed host views.

The reader does not instantiate MLX arrays or dequantize model weights. Tensor
shape is GGUF order (contiguous dimension first); numpy_shape reverses it.
Outstanding packed views must be released before closing their GGUFFile.
"""

from __future__ import annotations

import math
import mmap
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


class GGUFError(ValueError):
    pass


# Explicitly implemented formats only: reject an unknown mixed tensor instead
# of silently assigning it affine quantization semantics.
FORMATS = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    8: ("Q8_0", 32, 34),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    16: ("IQ2_XXS", 256, 66),
    19: ("IQ1_S", 256, 50),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    30: ("BF16", 1, 2),
}
SCALARS = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "?",
    10: "Q",
    11: "q",
    12: "d",
}


@dataclass(frozen=True)
class GGUFTensor:
    name: str
    shape: tuple[int, ...]
    ggml_type: int
    offset: int
    nbytes: int

    @property
    def format(self) -> str:
        return FORMATS[self.ggml_type][0]

    @property
    def numpy_shape(self) -> tuple[int, ...]:
        return self.shape[::-1]


class GGUFFile:
    """Parse an index without faulting in its potentially huge tensor payload."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_metadata_bytes: int = 1 << 30,
        max_items: int = 2_000_000,
    ):
        self.path = Path(path)
        self._file: BinaryIO = self.path.open("rb")
        self._mapping: mmap.mmap | None = None
        self._size = self.path.stat().st_size
        self._metadata_limit = min(self._size, max_metadata_bytes)
        self._max_items = max_items
        self.metadata: dict[str, object] = {}
        self.tensors: dict[str, GGUFTensor] = {}
        try:
            self._read_index()
        except BaseException:
            self._file.close()
            raise

    def _read(self, n: int) -> bytes:
        if n < 0 or self._file.tell() + n > self._metadata_limit:
            raise GGUFError("truncated or oversized GGUF metadata/index")
        data = self._file.read(n)
        if len(data) != n:
            raise GGUFError("truncated GGUF metadata/index")
        return data

    def _unpack(self, fmt: str):
        return struct.unpack("<" + fmt, self._read(struct.calcsize("<" + fmt)))[0]

    def _count(self) -> int:
        value = self._unpack("Q")
        if value > self._max_items:
            raise GGUFError("GGUF item count exceeds configured bound")
        return value

    def _string(self) -> str:
        n = self._unpack("Q")
        try:
            return self._read(n).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GGUFError("invalid UTF-8 GGUF string") from exc

    def _value(self, kind: int, *, array: bool = False):
        if kind in SCALARS:
            return self._unpack(SCALARS[kind])
        if kind == 8:
            return self._string()
        if kind == 9 and not array:
            subtype, count = self._unpack("I"), self._count()
            return tuple(self._value(subtype, array=True) for _ in range(count))
        raise GGUFError(f"unsupported GGUF metadata value type {kind}")

    def _read_index(self):
        if self._read(4) != b"GGUF":
            raise GGUFError("not little-endian GGUF")
        self.version = self._unpack("I")
        if self.version not in (2, 3):
            raise GGUFError(f"unsupported GGUF version {self.version}")
        tensor_count, kv_count = self._count(), self._count()
        for _ in range(kv_count):
            key = self._string()
            if key in self.metadata:
                raise GGUFError(f"duplicate metadata key {key!r}")
            self.metadata[key] = self._value(self._unpack("I"))
        alignment = self.metadata.get("general.alignment", 32)
        if (
            type(alignment) is not int
            or alignment < 1
            or alignment > 65536
            or alignment & (alignment - 1)
        ):
            raise GGUFError("invalid GGUF data alignment")
        for _ in range(tensor_count):
            name, ndim = self._string(), self._unpack("I")
            if name in self.tensors or not 1 <= ndim <= 4:
                raise GGUFError("duplicate tensor or invalid dimension count")
            shape = tuple(self._unpack("Q") for _ in range(ndim))
            kind, offset = self._unpack("I"), self._unpack("Q")
            if kind not in FORMATS:
                raise GGUFError(f"unsupported GGML type {kind} for {name}")
            _, block, size = FORMATS[kind]
            if any(d < 1 for d in shape) or shape[0] % block:
                raise GGUFError(f"invalid block alignment/shape for {name}")
            if offset % alignment:
                raise GGUFError(f"misaligned tensor offset for {name}")
            self.tensors[name] = GGUFTensor(
                name, shape, kind, offset, math.prod(shape) // block * size
            )
        self.data_offset = (self._file.tell() + alignment - 1) // alignment * alignment
        end = self.data_offset
        for tensor in sorted(self.tensors.values(), key=lambda t: t.offset):
            start = self.data_offset + tensor.offset
            if start < end or start + tensor.nbytes > self._size:
                raise GGUFError(f"overlapping or truncated tensor {tensor.name}")
            end = start + tensor.nbytes

    def packed_view(self, name: str) -> memoryview:
        tensor = self.tensors[name]
        if self._mapping is None:
            self._mapping = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        start = self.data_offset + tensor.offset
        return memoryview(self._mapping)[start : start + tensor.nbytes]

    def inventory(self) -> dict:
        formats: dict[str, dict[str, int]] = {}
        for t in self.tensors.values():
            row = formats.setdefault(
                t.format, {"tensors": 0, "packed_bytes": 0, "elements": 0}
            )
            row["tensors"] += 1
            row["packed_bytes"] += t.nbytes
            row["elements"] += math.prod(t.shape)
        return {
            "path": str(self.path),
            "version": self.version,
            "file_bytes": self._size,
            "data_offset": self.data_offset,
            "tensor_count": len(self.tensors),
            "formats": formats,
        }

    def release_mapping(self):
        """Unmap payload pages after upload; retain the small parsed index."""
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None

    def close(self):
        if self._mapping is not None:
            self._mapping.close()  # BufferError deliberately exposes outstanding views.
            self._mapping = None
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

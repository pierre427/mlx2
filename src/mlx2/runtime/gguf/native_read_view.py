"""Lossless APFS-cloned safetensors header over the original packed payload.

Only metadata pages are changed. Packed codes, scales and native tensors stay
byte-identical, allowing MLX's native asynchronous file reader to upload them
without a Python/NumPy intermediate. Original project implementation.
"""

from __future__ import annotations

import ctypes
import json
import os
import struct
from pathlib import Path

from ..immutable_weight_pages import same_stamp
from .mlx_storage import atomic_json, stamp
from .reader import GGUFError, GGUFFile

SCHEMA = "mlx2.gguf-native-read-view.v1"


def expected_header(file):
    header = {}
    cursor = 0
    for name, t in sorted(file.tensors.items(), key=lambda pair: pair[1].offset):
        if t.offset > cursor:
            header[f"__mlx2_padding_{cursor}"] = {
                "dtype": "U8",
                "shape": [t.offset - cursor],
                "data_offsets": [cursor, t.offset],
            }
        native = {
            0: "F32",
            1: "F16",
            24: "I8",
            25: "I16",
            26: "I32",
            27: "I64",
            30: "BF16",
        }
        if t.ggml_type in native:
            dtype, shape = native[t.ggml_type], list(t.numpy_shape)
        else:
            dtype, shape = "U8", [t.nbytes]
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [t.offset, t.offset + t.nbytes],
        }
        cursor = t.offset + t.nbytes
    end = file.path.stat().st_size - file.data_offset
    if cursor < end:
        header["__mlx2_trailing_padding"] = {
            "dtype": "U8",
            "shape": [end - cursor],
            "data_offsets": [cursor, end],
        }
    return header


def prepare_read_view(source, destination, *, source_sha256):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    seal = destination.with_suffix(destination.suffix + ".json")
    if destination.exists() or seal.exists():
        validate_read_view(source, destination, source_sha256=source_sha256)
        return destination
    with GGUFFile(source) as file:
        before = stamp(source)
        header = expected_header(file)
        raw = json.dumps(header, separators=(",", ":")).encode()
        size = file.data_offset - 8
        if len(raw) > size or size % 8:
            raise GGUFError(
                "GGUF metadata region cannot hold native packed read header"
            )
        # clonefile fails closed on unsupported filesystems: never silently
        # duplicate an additional 73 GiB or fall back to floating conversion.
        lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        lib.clonefile.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        lib.clonefile.restype = ctypes.c_int
        temporary = destination.with_suffix(destination.suffix + ".partial")
        if lib.clonefile(os.fsencode(source), os.fsencode(temporary), 0):
            raise OSError(ctypes.get_errno(), "APFS clonefile failed", str(destination))
        try:
            with temporary.open("r+b") as out:
                out.write(struct.pack("<Q", size))
                out.write(raw)
                out.write(b" " * (size - len(raw)))
                out.flush()
                os.fsync(out.fileno())
            if stamp(source) != before:
                raise GGUFError("source changed while preparing packed read view")
            temporary.replace(destination)
            atomic_json(
                seal,
                {
                    "schema": SCHEMA,
                    "source_sha256": source_sha256,
                    "source_stat": before,
                    "view_stat": stamp(destination),
                    "header_bytes": file.data_offset,
                    "payload": "APFS clone; unchanged original GGUF tensor bytes",
                    "physical_copy": "metadata pages only; no requantization",
                },
            )
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    validate_read_view(source, destination, source_sha256=source_sha256)
    return destination


def validate_read_view(source, destination, *, source_sha256):
    source, destination = Path(source), Path(destination)
    receipt = json.loads(
        destination.with_suffix(destination.suffix + ".json").read_text()
    )
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("source_sha256") != source_sha256
        or not same_stamp(stamp(source), receipt.get("source_stat"))
        or not same_stamp(stamp(destination), receipt.get("view_stat"))
    ):
        raise GGUFError("native packed read view source/seal changed")
    with GGUFFile(source) as file, destination.open("rb") as view:
        size = struct.unpack("<Q", view.read(8))[0]
        if size + 8 != file.data_offset or json.loads(
            view.read(size)
        ) != expected_header(file):
            raise GGUFError("native packed read view metadata changed")
    return receipt


class NativeReadViewFile(GGUFFile):
    def __init__(self, source, view, *, source_sha256):
        validate_read_view(source, view, source_sha256=source_sha256)
        super().__init__(source)
        self.view = Path(view)
        self.source_stat = stamp(self.path)
        self.view_stat = stamp(self.view)
        self._arrays = None

    def verify(self):
        if stamp(self.path) != self.source_stat or stamp(self.view) != self.view_stat:
            raise GGUFError("native read view source changed during load")

    def device_array(self, name):
        import mlx.core as mx

        self.verify()
        if self._arrays is None:
            self._arrays = mx.load(self.view)
        return self._arrays[name]

    def close(self):
        self._arrays = None
        super().close()

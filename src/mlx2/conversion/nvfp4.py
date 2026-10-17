"""Bounded CPU import of ModelOpt W4A16 NVFP4 checkpoints.

Preserves the published E2M1 values, E4M3 block scales and FP32 tensor
scales. No accelerator or model implementation is imported here.
Provenance: provenance/nemotron35-modelopt-nvfp4-generalized.json.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
import uuid
from itertools import pairwise
from pathlib import Path

import numpy as np

NATIVE_QUANTIZATION = {
    "bits": 4, "group_size": 16, "mode": "nvfp4",
    "global_scale": "per_tensor_fp32",
}
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1, "U32": 4, "I64": 8}
FP4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_header(path):
    path = Path(path)
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError("Truncated safetensors prefix")
        length = struct.unpack("<Q", raw)[0]
        if not 0 < length <= min(64 << 20, path.stat().st_size - 8):
            raise ValueError("Invalid safetensors header length")

        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"Duplicate tensor/header field: {key}")
                result[key] = value
            return result

        header = json.loads(handle.read(length), object_pairs_hook=unique)
    spans = []
    payload_size = path.stat().st_size - 8 - length
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        shape, dtype, offsets = entry["shape"], entry["dtype"], entry["data_offsets"]
        if dtype not in DTYPE_BYTES or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError(f"Unsupported tensor schema: {name}")
        start, end = offsets
        if (type(start) is not int or type(end) is not int
                or not 0 <= start <= end <= payload_size
                or end - start != math.prod(shape) * DTYPE_BYTES[dtype]):
            raise ValueError(f"Invalid tensor span: {name}")
        spans.append((start, end))
    spans.sort()
    if (not spans or spans[0][0] != 0 or spans[-1][1] != payload_size
            or any(a[1] != b[0] for a, b in pairwise(spans))):
        raise ValueError("Safetensors payload has gaps, overlaps or trailing bytes")
    return header, length + 8


def decode_e4m3(raw):
    codes = np.asarray(raw, dtype=np.uint8)
    exponent, mantissa = (codes >> 3) & 15, codes & 7
    if np.any((codes & 128) != 0) or np.any((exponent == 15) & (mantissa == 7)):
        raise ValueError("NVFP4 scales must be finite and nonnegative")
    return np.where(exponent == 0, mantissa.astype(np.float32) / 512,
                    (1 + mantissa.astype(np.float32) / 8) * np.exp2(exponent.astype(np.int32) - 7)).astype(np.float32)


def decode_rows(packed, scales, global_scale):
    packed = np.asarray(packed, dtype=np.uint8)
    scales = np.asarray(scales, dtype=np.uint8)
    if (packed.ndim != 2 or scales.shape != (packed.shape[0], packed.shape[1] // 8)
            or packed.shape[1] % 8):
        raise ValueError("NVFP4 requires a row-major block-16 scale grid")
    scalar = np.asarray(global_scale, dtype=np.float32)
    if scalar.size != 1 or not np.isfinite(scalar).all() or scalar.item() <= 0:
        raise ValueError("NVFP4 tensor scale must be finite and positive")
    codes = np.empty((packed.shape[0], packed.shape[1] * 2), np.uint8)
    codes[:, 0::2], codes[:, 1::2] = packed & 15, packed >> 4
    # Keep the global scale separate; rounding its product back to E4M3 loses data.
    decoded = FP4[codes].reshape(packed.shape[0], -1, 16)
    decoded *= decode_e4m3(scales)[:, :, None]
    decoded *= scalar.item()
    return decoded.reshape(packed.shape[0], -1)


def pack_group16(packed, scales, *, tile_rows=8):
    """Persistently tile ModelOpt bytes without changing their group-16 law.

    Every logical group retains its eight source E2M1 bytes and one source
    E4M3 byte.  Tail rows are zero padded and excluded again by
    :func:`unpack_group16`; no affine scale/bias layout is reused.
    """

    packed = np.asarray(packed, dtype=np.uint8)
    scales = np.asarray(scales, dtype=np.uint8)
    if (
        packed.ndim != 2
        or packed.shape[1] % 8
        or scales.shape != (packed.shape[0], packed.shape[1] // 8)
        or type(tile_rows) is not int
        or tile_rows < 1
    ):
        raise ValueError("NVFP4 persistent pack requires row-major group-16 bytes")
    rows, packed_columns = packed.shape
    groups = packed_columns // 8
    tiles = (rows + tile_rows - 1) // tile_rows
    packed_tiles = np.zeros((tiles, groups, tile_rows, 8), dtype=np.uint8)
    scale_tiles = np.zeros((tiles, groups, tile_rows), dtype=np.uint8)
    for row in range(rows):
        tile, lane = divmod(row, tile_rows)
        packed_tiles[tile, :, lane] = packed[row].reshape(groups, 8)
        scale_tiles[tile, :, lane] = scales[row]
    return packed_tiles, scale_tiles, rows


def unpack_group16(packed_tiles, scale_tiles, rows):
    """Invert :func:`pack_group16` byte-for-byte for its logical rows."""

    packed_tiles = np.asarray(packed_tiles, dtype=np.uint8)
    scale_tiles = np.asarray(scale_tiles, dtype=np.uint8)
    if (
        packed_tiles.ndim != 4
        or packed_tiles.shape[-1] != 8
        or scale_tiles.shape != packed_tiles.shape[:-1]
        or type(rows) is not int
        or not 0 <= rows <= packed_tiles.shape[0] * packed_tiles.shape[2]
    ):
        raise ValueError("Invalid NVFP4 persistent group-16 pack")
    groups, tile_rows = packed_tiles.shape[1:3]
    packed = np.empty((rows, groups * 8), dtype=np.uint8)
    scales = np.empty((rows, groups), dtype=np.uint8)
    for row in range(rows):
        tile, lane = divmod(row, tile_rows)
        packed[row] = packed_tiles[tile, :, lane].reshape(-1)
        scales[row] = scale_tiles[tile, :, lane]
    return packed, scales


def bf16_bits(values):
    bits = np.asarray(values, dtype="<f4").view("<u4")
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype("<u2")


def convert_checkpoint(source, destination, *, mode="bf16", reserve_bytes=50_000_000_000):
    """Write an exclusive, atomic single-shard artifact, without loading a model.

    ``native`` changes byte packaging only; BF16 rounds the decoded published
    quantized values once. Neither mode recreates pre-quantization weights.
    """
    if mode not in {"bf16", "native"}:
        raise ValueError("NVFP4 import mode must be bf16 or native")
    source, destination = Path(source).resolve(), Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if type(reserve_bytes) is not int or reserve_bytes < 0:
        raise ValueError("Invalid free-space reserve")
    config = json.loads((source / "config.json").read_text())
    quant = config.get("quantization_config", {})
    if (quant.get("quant_algo") != "W4A16_NVFP4" or quant.get("group_size") != 16
            or quant.get("quant_method") not in {"modelopt", "modelopt_fp4", "modelopt_mixed"}):
        raise ValueError("Expected ModelOpt W4A16_NVFP4 block-16 artifact")
    weight_file = source / "model.safetensors"
    header, payload = read_header(weight_file)
    prefixes = sorted(k.removesuffix(".weight_scale_2") for k in header if k.endswith(".weight_scale_2"))
    if not prefixes:
        raise ValueError("No complete NVFP4 matrix triples")
    owned = set()
    for prefix in prefixes:
        names = [prefix + suffix for suffix in (".weight", ".weight_scale", ".weight_scale_2")]
        if any(name not in header for name in names):
            raise ValueError(f"Incomplete NVFP4 matrix: {prefix}")
        weight, scale, scalar = (header[name] for name in names)
        if (weight["dtype"] != "U8" or len(weight["shape"]) != 2
                or weight["shape"][1] % 8 or scale["dtype"] != "F8_E4M3"
                or scale["shape"] != [weight["shape"][0], weight["shape"][1] // 8]
                or scalar["dtype"] != "F32" or math.prod(scalar["shape"]) != 1):
            raise ValueError(f"Invalid NVFP4 matrix geometry: {prefix}")
        owned.update(names)
    output = {}
    for name, entry in header.items():
        if name == "__metadata__" or name in owned:
            continue
        if entry["dtype"] != "BF16":
            raise ValueError(f"Unrecognized non-BF16 tensor: {name}")
        output[name] = {"dtype": entry["dtype"], "shape": entry["shape"]}
    for prefix in prefixes:
        shape = list(header[prefix + ".weight"]["shape"])
        if mode == "bf16":
            shape[-1] *= 2
            output[prefix + ".weight"] = {"dtype": "BF16", "shape": shape}
        else:
            shape[-1] //= 4
            output[prefix + ".weight"] = {"dtype": "U32", "shape": shape}
            output[prefix + ".scales"] = {"dtype": "U8", "shape": header[prefix + ".weight_scale"]["shape"]}
            output[prefix + ".global_scale"] = {"dtype": "F32", "shape": []}
    offset = 0
    for entry in output.values():
        size = math.prod(entry["shape"]) * DTYPE_BYTES[entry["dtype"]]
        entry["data_offsets"] = [offset, offset + size]
        offset += size
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < offset + reserve_bytes:
        raise ValueError("Insufficient space for output plus reserve")
    source_stat = weight_file.stat()
    source_sha = sha256(weight_file)
    staging = destination.with_name(destination.name + ".incomplete-" + uuid.uuid4().hex)
    staging.mkdir()
    output_path = staging / "model.safetensors"
    raw_header = json.dumps(output, separators=(",", ":")).encode()
    raw_header += b" " * ((-len(raw_header)) % 8)
    with weight_file.open("rb") as inp, output_path.open("wb") as out:
        out.write(struct.pack("<Q", len(raw_header)))
        out.write(raw_header)

        def copy(name):
            start, end = header[name]["data_offsets"]
            inp.seek(payload + start)
            remaining = end - start
            while remaining:
                block = inp.read(min(8 << 20, remaining))
                if not block:
                    raise ValueError("Source truncated during conversion")
                if name.endswith(".weight_scale"):
                    decode_e4m3(np.frombuffer(block, np.uint8))
                elif name.endswith(".weight_scale_2"):
                    scalar = np.frombuffer(block, "<f4")
                    if scalar.size != 1 or not np.isfinite(scalar).all() or scalar.item() <= 0:
                        raise ValueError("NVFP4 tensor scale must be finite and positive")
                out.write(block)
                remaining -= len(block)

        for name in output:
            if name in header and name not in owned:
                copy(name)
                continue
            prefix = name.rsplit(".", 1)[0]
            if mode == "native":
                copy(prefix + {"weight": ".weight", "scales": ".weight_scale", "global_scale": ".weight_scale_2"}[name.rsplit(".", 1)[1]])
                continue
            w = header[prefix + ".weight"]
            rows, packed_cols = w["shape"]
            scale_entry = header[prefix + ".weight_scale"]
            inp.seek(payload + header[prefix + ".weight_scale_2"]["data_offsets"][0])
            scalar = struct.unpack("<f", inp.read(4))[0]
            chunk_rows = max(1, (8 << 20) // (packed_cols * 16))
            for first in range(0, rows, chunk_rows):
                count = min(chunk_rows, rows - first)
                inp.seek(payload + w["data_offsets"][0] + first * packed_cols)
                packed = np.frombuffer(inp.read(count * packed_cols), np.uint8).reshape(count, packed_cols)
                inp.seek(payload + scale_entry["data_offsets"][0] + first * (packed_cols // 8))
                scales = np.frombuffer(inp.read(count * (packed_cols // 8)), np.uint8).reshape(count, packed_cols // 8)
                out.write(bf16_bits(decode_rows(packed, scales, scalar)).tobytes())
        out.flush()
        os.fsync(out.fileno())
    after = weight_file.stat()
    if (source_stat.st_size, source_stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Source changed during conversion")
    read_header(output_path)
    config.pop("quantization_config")
    config.pop("quantization", None)
    if mode == "native":
        config["quantization"] = {**NATIVE_QUANTIZATION, "modules": prefixes}
    (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    receipt = {"schema": "mlx2.modelopt-nvfp4-import.v1", "mode": mode,
               "source_config_sha256": sha256(source / "config.json"),
               "source_weight_sha256": source_sha, "output_weight_sha256": sha256(output_path),
               "quantized_matrices": len(prefixes), "output_tensor_bytes": offset,
               "global_scale_preserved": True, "qualified": False, "selected": False}
    (staging / "mlx2-conversion.json").write_text(json.dumps(receipt, indent=2) + "\n")
    os.rename(staging, destination)
    return receipt

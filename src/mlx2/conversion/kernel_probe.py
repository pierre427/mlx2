"""CPU preparation of real LiLiCorr matrices for bounded component probes.

The dense reference decodes the published NVFP4 values. It is not the
unavailable original pre-quantization checkpoint, or a full-model baseline.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

import numpy as np

from .affine import quantize_rows
from .nvfp4 import decode_rows, read_header, sha256

SOURCE_WEIGHT_SHA256 = "eec5efdf1b838020c2842f52d560c187515e84889c51f99d70b75c773c286599"
SOURCE_CONFIG_SHA256 = "880d77cb17f6f45925ba78ba94a04b92b5a88b6046e7b6e376d2ab794b02ba04"
MATRICES = {"fc": (4096, 20480), "layers.0.mlp.up_proj": (12288, 4096)}
SCHEMA = "mlx2.nemotron35-nvfp4-kernel-inputs.v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def read_matrix(path, prefix, *, expected_shape):
    """Read one complete NVFP4 triple, with explicit bounded geometry."""
    header, payload = read_header(path)
    rows, columns = expected_shape
    if rows * columns > 100_000_000 or rows < 1 or columns < 64 or columns % 64:
        raise ValueError("Kernel matrix exceeds the component geometry bound")
    result = []
    with Path(path).open("rb") as stream:
        for suffix, dtype, shape, numpy_dtype in (
            (".weight", "U8", [rows, columns // 2], "u1"),
            (".weight_scale", "F8_E4M3", [rows, columns // 16], "u1"),
            (".weight_scale_2", "F32", [1], "<f4"),
        ):
            entry = header[prefix + suffix]
            if entry["dtype"] != dtype or entry["shape"] not in (shape, [] if suffix.endswith("_2") else shape):
                raise ValueError("Kernel matrix differs from its declared source geometry")
            start, end = entry["data_offsets"]
            stream.seek(payload + start)
            raw = stream.read(end - start)
            if len(raw) != end - start:
                raise ValueError("Truncated kernel source tensor")
            result.append(np.frombuffer(raw, dtype=numpy_dtype).copy().reshape(shape))
    packed, scales, scalar = result
    dense = decode_rows(packed, scales, scalar)
    return {"native_weight": packed.view("<u4"), "native_scales": scales,
            "global_scale": scalar.reshape(()), "dense_fp32": dense}


def prepare(source, destination, *, reserve_bytes=50_000_000_000, progress=None):
    source, destination = Path(source).resolve(), Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    weight = source / "model.safetensors"
    if (weight.is_symlink() or sha256(source / "config.json") != SOURCE_CONFIG_SHA256
            or sha256(weight) != SOURCE_WEIGHT_SHA256):
        raise ValueError("Kernel inputs require the exact published LiLiCorr source")
    before = weight.stat()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < reserve_bytes + 1_000_000_000:
        raise OSError("Kernel preparation would consume the free-space reserve")
    staging = destination.with_name(destination.name + ".incomplete-" + uuid.uuid4().hex)
    staging.mkdir()
    matrices = []
    for index, (prefix, shape) in enumerate(MATRICES.items()):
        arrays = read_matrix(weight, prefix, expected_shape=shape)
        errors = {}
        # Bound fitting scratch independently of the real matrix size.
        for bits in (4, 8):
            chunks = [[], [], []]
            error = 0.
            for first in range(0, shape[0], 16):
                packed, scales, biases, receipt = quantize_rows(arrays["dense_fp32"][first:first + 16], bits=bits)
                for target, chunk in zip(chunks, (packed, scales, biases), strict=True):
                    target.append(chunk)
                error += receipt["error_after"]
            for suffix, parts in zip(("weight", "scales", "biases"), chunks, strict=True):
                arrays[f"affine{bits}_{suffix}"] = np.concatenate(parts)
            errors[str(bits)] = {"weight_mse": error / (shape[0] * shape[1]), "activation_calibrated": False}
        file = staging / f"matrix-{index}.npz"
        with file.open("xb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        matrices.append({"prefix": prefix, "shape": list(shape), "file": file.name,
                         "sha256": sha256(file), "arrays": {key: {"dtype": str(value.dtype), "shape": list(value.shape)}
                                                               for key, value in arrays.items()}, "fitting": errors})
        if progress:
            progress({"prepared_matrix": prefix, "shape": shape})
        del arrays
    after = weight.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Kernel source changed during preparation")
    manifest = {"schema": SCHEMA, "source_weight_sha256": SOURCE_WEIGHT_SHA256,
                "source_config_sha256": SOURCE_CONFIG_SHA256, "matrices": matrices,
                "reference": "Decoded published NVFP4 weights; not original BF16 or full-model performance",
                "rows": [1, 8, 128], "seed": 20261015, "qualified": False}
    manifest["fingerprint"] = hashlib.sha256(canonical(manifest)).hexdigest()
    (staging / "manifest.json").write_bytes(canonical(manifest) + b"\n")
    os.rename(staging, destination)
    return manifest


def verify(path):
    path = Path(path).resolve()
    manifest = json.loads((path / "manifest.json").read_text())
    fingerprint = manifest.pop("fingerprint")
    if (manifest.get("schema") != SCHEMA or manifest.get("source_weight_sha256") != SOURCE_WEIGHT_SHA256
            or manifest.get("source_config_sha256") != SOURCE_CONFIG_SHA256
            or hashlib.sha256(canonical(manifest)).hexdigest() != fingerprint
            or {item["prefix"]: tuple(item["shape"]) for item in manifest["matrices"]} != MATRICES
            or len(manifest["matrices"]) != len(MATRICES) or manifest.get("rows") != [1, 8, 128]):
        raise ValueError("Kernel manifest source/geometry binding differs")
    for item in manifest["matrices"]:
        name = item["file"]
        file = path / name
        if Path(name).name != name or file.is_symlink() or sha256(file) != item["sha256"]:
            raise ValueError("Kernel matrix payload differs")
        with np.load(file, allow_pickle=False) as archive:
            if set(archive.files) != set(item["arrays"]):
                raise ValueError("Kernel matrix array closure differs")
            for key, schema in item["arrays"].items():
                array = archive[key]
                if str(array.dtype) != schema["dtype"] or list(array.shape) != schema["shape"]:
                    raise ValueError("Kernel matrix array geometry differs")
    manifest["fingerprint"] = fingerprint
    return manifest


def error_metrics(actual, expected):
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Kernel correctness reference is nonfinite or has different geometry")
    delta = actual - expected
    return {"relative_rmse": float(np.sqrt(np.mean(delta**2) / max(np.mean(expected**2), 1e-30))),
            "max_absolute_error": float(np.max(np.abs(delta)))}

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Resumable CPU-only conversion of the sealed Maya V4.1 image to native MLX.

Original project writer; numerical packing is reused from
deepseek_v41_native_layer.py (its local source/license provenance is retained).
No source deletion or GPU execution occurs. A completed manifest proves storage
conversion and file integrity, not native matmul equivalence or qualification.
"""
from __future__ import annotations

import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import subprocess
import time

import numpy as np

from deepseek_v41_native_layer import affine_spec, convert_affine, validate_affine
from mlx2.runtime.gguf.reader import FORMATS, GGUFFile

MANIFEST = "mlx2-v41-native-manifest.json"
SCHEMA = "mlx2.deepseek-v41-native.v1"
RAW_DTYPES = {0: ("F32", 4), 1: ("F16", 2)}
CHUNK_ELEMENTS = 1 << 20
COPY_BYTES = 16 << 20


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(COPY_BYTES):
            h.update(chunk)
    return h.hexdigest()


def source_hash(fd, size):
    """One final sequential pass over the original sealed file, including gaps."""
    digest = hashlib.sha256()
    for first in range(0, size, COPY_BYTES):
        digest.update(read_exact(fd, min(COPY_BYTES, size - first), first))
    return digest.hexdigest()


def stamp(value):
    s = os.fstat(value) if isinstance(value, int) else Path(value).stat(follow_symlinks=False)
    return dict(dev=s.st_dev, ino=s.st_ino, size=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def safe_path(root, relative):
    part = PurePosixPath(relative)
    if part.is_absolute() or not part.parts or any(p in (".", "..") for p in part.parts):
        raise ValueError("invalid relative artifact path")
    path = Path(root)
    if path.is_symlink():
        raise ValueError("artifact root is a symlink")
    for p in part.parts:
        path = path / p
        if path.is_symlink():
            raise ValueError("artifact path contains a symlink")
    return path


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if path.is_symlink() or temporary.is_symlink():
        raise ValueError("refusing symlink metadata path")
    with temporary.open("wb") as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    parent = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def read_exact(fd, size, offset):
    chunks = []
    total = 0
    while total < size:
        data = os.pread(fd, size - total, offset + total)
        if not data:
            raise EOFError("short source tensor read")
        chunks.append(data)
        total += len(data)
    return chunks[0] if len(chunks) == 1 else b"".join(chunks)


def write_at(fd, data, offset):
    view = memoryview(data).cast("B")
    while view:
        count = os.pwrite(fd, view, offset)
        if count <= 0:
            raise OSError("short converted tensor write")
        offset += count
        view = view[count:]


def check_space(destination, size, reserve_bytes):
    if type(size) is not int or size < 0 or type(reserve_bytes) is not int or reserve_bytes < 0:
        raise ValueError("invalid disk allocation bound")
    if shutil.disk_usage(destination).free < size + reserve_bytes:
        raise OSError(f"free-space reserve would be crossed: need file {size} + reserve {reserve_bytes} bytes")


def recipe():
    directory = Path(__file__).resolve().parent
    root = directory.parents[1]
    sources = [directory / "deepseek_v41_native_layer.py", root / "src/mlx2/runtime/gguf/blocks.py",
               root / "src/mlx2/runtime/gguf/iq1_codebook.py"]
    return dict(version=1, group_size=32, scales="F32", biases="F32", source_axis_order="numpy_shape",
                iq1_s=dict(bits=2, q="ternary+1", scale="half_d*(2*((qh>>12)&7)+1)",
                           bias="scale*(signed_delta_0.125-1)"),
                q8_0=dict(bits=8, q="int8+128", scale="half_d", bias="-128*half_d"),
                signed_zero="numerically exact; count IEEE zero-sign differences",
                raw="F16/F32 and packed I8 PLE bytes unchanged",
                source_sha256={str(p.relative_to(root)): file_hash(p) for p in sources})


def algebra_selftest():
    """Exhaust all finite half scales and possible affine scalar values."""
    half = np.arange(1 << 16, dtype=np.uint16).view(np.float16)
    half = half[np.isfinite(half)].astype(np.float32)
    count = signs = 0
    for first in range(0, len(half), 1024):
        d = half[first:first + 1024, None]
        for factor in range(1, 16, 2):
            scale = d * np.float32(factor)
            for delta in (-.125, .125):
                values = np.array([-1, 0, 1], np.float32)[None, :]
                expected = scale * (values + np.float32(delta))
                actual = scale * (values + 1) + scale * np.float32(delta - 1)
                if not np.array_equal(expected, actual):
                    raise ValueError("IQ1 affine algebra is not exact")
                count += expected.size
                signs += int(np.count_nonzero(expected.view(np.uint32) != actual.view(np.uint32)))
        q = np.arange(-128, 128, dtype=np.float32)[None, :]
        expected = d * q
        actual = d * (q + 128) + d * np.float32(-128)
        if not np.array_equal(expected, actual):
            raise ValueError("Q8 affine algebra is not exact")
        count += expected.size
        signs += int(np.count_nonzero(expected.view(np.uint32) != actual.view(np.uint32)))
    # Independently verify every eight-lane codebook entry through decode_blocks.
    from deepseek_v41_native_layer import IQ1_GRID
    n = len(IQ1_GRID)
    raw = np.zeros((n, 50), np.uint8)
    raw[:, :2] = np.frombuffer(np.float16(1).tobytes(), np.uint8)
    ix = np.arange(n, dtype=np.uint16)
    raw[:, 2:34] = (ix & 255).astype(np.uint8)[:, None]
    hi = ix >> 8
    high = (hi[:, None] | hi[:, None] << 3 | hi[:, None] << 6 | hi[:, None] << 9)
    high = high | np.arange(8, dtype=np.uint16)[None, :] << 12
    high = high | (np.arange(8, dtype=np.uint16)[None, :] % 2) << 15
    raw[:, 34:] = high.astype("<u2").view(np.uint8)
    validate_affine(raw, 19, (n, 256), require_bitwise=True)
    return dict(passed=True, finite_half_scales=len(half), scalar_cases=count,
                synthetic_signed_zero_differences=signs, iq1_codebook_entries=n)


def build_plan(tensors):
    """Return original tensor map and bounded per-file conversion jobs."""
    result, jobs, experts = {}, {}, {}
    for name, tensor in sorted(tensors.items()):
        entry = dict(shape=list(tensor.shape), numpy_shape=list(tensor.numpy_shape),
                     ggml_type=tensor.ggml_type, offset=tensor.offset, nbytes=tensor.nbytes)
        layer_match = re.match(r"blk\.(\d+)\.", name)
        layer = int(layer_match.group(1)) if layer_match else None
        expert = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight", name)
        if expert:
            if tensor.ggml_type != 19 or len(tensor.numpy_shape) != 3:
                raise ValueError("unexpected expert tensor representation")
            total, rows, width = tensor.numpy_shape
            keys = {suffix: expert.group(2) + "." + suffix for suffix in ("qweight", "scales", "biases")}
            paths = [f"experts/layer-{layer:03d}/expert-{i:03d}.safetensors" for i in range(total)]
            entry["storage"] = dict(kind="expert-affine", expert_files=paths, keys=keys,
                                    bits=2, group_size=32, mode="affine")
            experts.setdefault(layer, set()).add(expert.group(2))
            for i, path in enumerate(paths):
                jobs.setdefault(path, []).append(dict(name=name, expert=i, keys=keys))
        elif tensor.ggml_type == 24:
            if layer is None or tensor.numpy_shape[-1] != 264 or not name.endswith("engram_embd.weight"):
                raise ValueError("unrecognized I8 tensor")
            path = f"engram/layer-{layer:03d}.bin"
            entry["storage"] = dict(kind="engram", file=path, row_bytes=264, rows=tensor.numpy_shape[0])
            jobs[path] = [dict(name=name, raw_file=True)]
        else:
            path = f"dense/layer-{layer:03d}.safetensors" if layer is not None else "weights.safetensors"
            if tensor.ggml_type == 8:
                keys = {s: name + "." + s for s in ("qweight", "scales", "biases")}
                entry["storage"] = dict(kind="affine", file=path, keys=keys, bits=8, group_size=32, mode="affine")
                job = dict(name=name, keys=keys)
            elif tensor.ggml_type in RAW_DTYPES:
                entry["storage"] = dict(kind="raw", file=path, key=name)
                job = dict(name=name, key=name)
            else:
                raise ValueError(f"unsupported conversion type {tensor.ggml_type}: {name}")
            jobs.setdefault(path, []).append(job)
        result[name] = entry
    if any(kinds != {"gate", "up", "down"} for kinds in experts.values()):
        raise ValueError("incomplete expert triplet")
    for path, entries in jobs.items():
        if path.startswith("experts/") and len(entries) != 3:
            raise ValueError("expert count differs across projections")
    return result, jobs


def tensor_layout(entries, tensors):
    layout, position = {}, 0
    for job in entries:
        tensor = tensors[job["name"]]
        shape = tensor.numpy_shape[1:] if "expert" in job else tensor.numpy_shape
        if "keys" in job:
            spec = affine_spec(tensor.ggml_type, shape)
            arrays = [(job["keys"][s], "U32" if s == "qweight" else "F32", spec[k])
                      for s, k in (("qweight", "packed_shape"), ("scales", "scales_shape"), ("biases", "biases_shape"))]
        else:
            arrays = [(job["key"], RAW_DTYPES[tensor.ggml_type][0], shape)]
        for key, dtype, out_shape in arrays:
            size = math.prod(out_shape) * (2 if dtype == "F16" else 4)
            if key in layout:
                raise ValueError("duplicate native tensor key")
            layout[key] = dict(dtype=dtype, shape=list(out_shape), data_offsets=[position, position + size])
            position += size
    header = canonical({**layout, "__metadata__": {"format": "mlx", "quantization": "affine/group32/F32-metadata"}})
    header += b" " * (-len(header) % 8)
    return layout, struct.pack("<Q", len(header)) + header, position


def zero_sign_differences(raw, ggml_type):
    """Inspect only sign-sensitive blocks; ordinary nonzero IQ1 has no zeros."""
    _, elements, size = FORMATS[ggml_type]
    blocks = np.frombuffer(raw, np.uint8).reshape(-1, size)
    d = blocks[:, :2].view("<f2").reshape(-1)
    sensitive = d == 0
    if ggml_type == 8:
        sensitive |= (d < 0) & np.any(blocks[:, 2:] == 0, axis=1)
    subset = blocks[sensitive]
    total = 0
    for first in range(0, len(subset), CHUNK_ELEMENTS // elements):
        part = subset[first:first + CHUNK_ELEMENTS // elements]
        total += validate_affine(part, ggml_type, (len(part), elements))["signed_zero_differences"]
    return total


def write_safetensors(source_fd, data_offset, tensors, entries, target, *, rows_per_chunk=64):
    layout, header, payload_size = tensor_layout(entries, tensors)
    output_fd = os.open(target, os.O_CREAT | os.O_TRUNC | os.O_RDWR, 0o600)
    proof = dict(sampled_rows=0, signed_zero_differences=0, sampled_signed_zero_differences=0,
                 source_bytes=0, quantized_elements=0)
    try:
        os.ftruncate(output_fd, len(header) + payload_size)
        write_at(output_fd, header, 0)
        for job in entries:
            tensor = tensors[job["name"]]
            shape = tensor.numpy_shape[1:] if "expert" in job else tensor.numpy_shape
            start = data_offset + tensor.offset
            size = tensor.nbytes
            if "expert" in job:
                size //= tensor.numpy_shape[0]
                start += job["expert"] * size
            proof["source_bytes"] += size
            if "keys" not in job:
                destination = len(header) + layout[job["key"]]["data_offsets"][0]
                for first in range(0, size, COPY_BYTES):
                    write_at(output_fd, read_exact(source_fd, min(COPY_BYTES, size - first), start + first), destination + first)
                continue
            width = shape[-1]
            rows = math.prod(shape[:-1])
            _, block, block_bytes = FORMATS[tensor.ggml_type]
            row_bytes = width // block * block_bytes
            step = min(rows_per_chunk, CHUNK_ELEMENTS // width)
            if step < 1:
                raise ValueError("one source row exceeds conversion bound")
            sample_rows = {0, rows // 2, rows - 1}
            for first in range(0, rows, step):
                last = min(rows, first + step)
                raw = read_exact(source_fd, (last - first) * row_bytes, start + first * row_bytes)
                arrays = convert_affine(raw, tensor.ggml_type, (last - first, width))
                for suffix, array in zip(("qweight", "scales", "biases"), arrays):
                    item = layout[job["keys"][suffix]]
                    stride = math.prod(item["shape"][1:]) * 4 if len(item["shape"]) == 2 else item["shape"][-1] * 4
                    write_at(output_fd, array, len(header) + item["data_offsets"][0] + first * stride)
                proof["signed_zero_differences"] += zero_sign_differences(raw, tensor.ggml_type)
                for row in sample_rows:
                    if first <= row < last:
                        offset = (row - first) * row_bytes
                        stored = []
                        for suffix, dtype in (("qweight", "<u4"), ("scales", "<f4"), ("biases", "<f4")):
                            item = layout[job["keys"][suffix]]
                            columns = item["shape"][-1]
                            saved = read_exact(output_fd, columns * 4,
                                               len(header) + item["data_offsets"][0] + row * columns * 4)
                            stored.append(np.frombuffer(saved, dtype=dtype).reshape(1, columns))
                        checked = validate_affine(raw[offset:offset + row_bytes], tensor.ggml_type, (1, width), stored)
                        proof["sampled_rows"] += 1
                        proof["sampled_signed_zero_differences"] += checked["signed_zero_differences"]
            proof["quantized_elements"] += math.prod(shape)
        os.fsync(output_fd)
    finally:
        os.close(output_fd)
    return dict(size=len(header) + payload_size, header_bytes=len(header), tensors=layout, proof=proof)


def copy_range(source_fd, start, size, target):
    expected = hashlib.sha256()
    with Path(target).open("wb") as stream:
        for first in range(0, size, COPY_BYTES):
            raw = read_exact(source_fd, min(COPY_BYTES, size - first), start + first)
            expected.update(raw)
            stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return expected.hexdigest()


def verify_file(root, relative, record):
    path = safe_path(root, relative)
    before = stamp(path)
    if before["size"] != record["size"] or file_hash(path) != record["sha256"] or stamp(path) != before:
        raise ValueError(f"converted file failed verification: {relative}")
    if record.get("header_bytes"):
        with path.open("rb") as stream:
            header_length = struct.unpack("<Q", stream.read(8))[0]
            if header_length + 8 != record["header_bytes"] or header_length > 16 << 20:
                raise ValueError("invalid safetensors header length")
            header = json.loads(stream.read(header_length))
        header.pop("__metadata__", None)
        if header != record["tensors"]:
            raise ValueError("safetensors inventory differs from receipt")
        end = 0
        for item in sorted(header.values(), key=lambda x: x["data_offsets"][0]):
            first, last = item["data_offsets"]
            element_size = {"F16": 2, "F32": 4, "U32": 4}[item["dtype"]]
            if first != end or last - first != math.prod(item["shape"]) * element_size:
                raise ValueError("safetensors payload has holes/overlap/shape mismatch")
            end = last
        if end + record["header_bytes"] != record["size"]:
            raise ValueError("safetensors payload size mismatch")
    return before


def verify_manifest(root, manifest, *, hash_files=True):
    expected = set(manifest["assets"])
    expected_arrays = {}
    coverage = 0
    for tensor in manifest["tensors"].values():
        storage = tensor["storage"]
        if storage["kind"] == "expert-affine":
            expected.update(storage["expert_files"])
            if len(storage["expert_files"]) != tensor["numpy_shape"][0]:
                raise ValueError("incomplete expert storage coverage")
        else:
            expected.add(storage["file"])
        if storage["kind"] in ("affine", "expert-affine"):
            expected_bits = 2 if tensor["ggml_type"] == 19 else 8
            if storage.get("bits") != expected_bits or storage.get("group_size") != 32 or storage.get("mode") != "affine":
                raise ValueError("native affine recipe changed")
            paths = storage["expert_files"] if storage["kind"] == "expert-affine" else [storage["file"]]
            shape = tensor["numpy_shape"][1:] if storage["kind"] == "expert-affine" else tensor["numpy_shape"]
            spec = affine_spec(tensor["ggml_type"], shape)
            for path in paths:
                for suffix, shape_key in (("qweight", "packed_shape"), ("scales", "scales_shape"), ("biases", "biases_shape")):
                    key = storage["keys"][suffix]
                    arrays = expected_arrays.setdefault(path, {})
                    if key in arrays:
                        raise ValueError("native output aliases original tensors")
                    arrays[key] = dict(dtype="U32" if suffix == "qweight" else "F32", shape=list(spec[shape_key]))
        elif storage["kind"] == "raw":
            arrays = expected_arrays.setdefault(storage["file"], {})
            if storage["key"] in arrays:
                raise ValueError("native raw output aliases original tensors")
            arrays[storage["key"]] = dict(dtype=RAW_DTYPES[tensor["ggml_type"]][0], shape=tensor["numpy_shape"])
        elif storage["kind"] == "engram":
            if (tensor["ggml_type"] != 24 or storage["row_bytes"] != 264
                    or storage["rows"] * 264 != tensor["nbytes"]
                    or manifest["files"][storage["file"]]["size"] != tensor["nbytes"]):
                raise ValueError("PLE byte coverage changed")
        else:
            raise ValueError("unknown storage representation")
        coverage += tensor["nbytes"]
    if expected != set(manifest["files"]):
        raise ValueError("manifest file inventory incomplete or unexpected")
    if len(manifest["tensors"]) != manifest["inventory"]["source_tensor_count"]:
        raise ValueError("original tensor inventory incomplete")
    if coverage != manifest["inventory"]["source_tensor_bytes"]:
        raise ValueError("original tensor byte coverage incomplete")
    total = 0
    for relative, record in manifest["files"].items():
        if relative in expected_arrays:
            described = {key: dict(dtype=item["dtype"], shape=item["shape"]) for key, item in record["tensors"].items()}
            if described != expected_arrays[relative]:
                raise ValueError("native tensor inventory differs from original source geometry")
        before = verify_file(root, relative, record) if hash_files else stamp(safe_path(root, relative))
        if before != record["stat"]:
            raise ValueError(f"converted file stat seal changed: {relative}")
        total += record["size"]
    return dict(passed=True, file_count=len(expected), source_tensor_count=len(manifest["tensors"]),
                source_tensor_bytes=coverage, converted_bytes=total, all_file_hashes_checked=hash_files)


@contextmanager
def conversion_lock(root):
    path = safe_path(root, ".conversion.lock")
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another converter owns this artifact") from error
        yield


def convert(source, output, *, reserve_bytes=100 << 30, max_files=None, plan_only=False, phase=print):
    from mlx2.adapters.deepseek_v41_artifact import FILENAME, inspect_artifact
    source, output = Path(source).resolve(), Path(output)
    if output.is_symlink():
        raise ValueError("output root is a symlink")
    output = output.resolve()
    if source == output or source in output.parents:
        raise ValueError("native artifact must be separate from source directory")
    admitted = inspect_artifact(source)
    source_path = source / FILENAME
    source_stamp = stamp(source_path)
    asset_sources = {"tokenizer.reference.json": source / "tokenizer.reference.json",
                     "source-config.json": source / "config.json",
                     "source-integrity.json": source / "mlx2-v41-integrity.json",
                     "encoding_dsv41.py": Path(__file__).resolve().parents[2] / "src/mlx2/runtime/models/deepseek_v41_encoding.py"}
    for name in ("README.upstream.md", "acquisition.json", FILENAME + ".receipt.json"):
        if (source / name).is_file():
            asset_sources[name] = source / name
    asset_bindings = {}
    for relative, path in asset_sources.items():
        if path.is_symlink():
            raise ValueError("asset source is a symlink")
        before = stamp(path)
        asset_bindings[relative] = dict(stat=before, sha256=file_hash(path))
        if stamp(path) != before:
            raise ValueError("source asset changed during binding")
    with GGUFFile(source_path) as gguf:
        mapped, jobs = build_plan(gguf.tensors)
        specs = {}
        for path, entries in jobs.items():
            if entries[0].get("raw_file"):
                specs[path] = gguf.tensors[entries[0]["name"]].nbytes
            else:
                _, header, payload_size = tensor_layout(entries, gguf.tensors)
                specs[path] = len(header) + payload_size
        summary = dict(file_count=len(jobs), tensor_count=len(mapped), output_weight_bytes=sum(specs.values()),
                       source_tensor_bytes=sum(t.nbytes for t in gguf.tensors.values()),
                       minimum_free_reserve_bytes=reserve_bytes, largest_file_bytes=max(specs.values()))
        if plan_only:
            return summary
        output.mkdir(parents=True, exist_ok=True)
        with conversion_lock(output):
            recipe_value = recipe()
            recipe_hash = hashlib.sha256(canonical(recipe_value)).hexdigest()
            source_info = dict(checkpoint=FILENAME + "@" + admitted["identity"]["fingerprint"],
                               fingerprint=admitted["identity"]["fingerprint"], gguf_sha256=admitted["identity"]["weight_sha256"],
                               filename=FILENAME, size=source_stamp["size"], stat=source_stamp)
            manifest_path = output / MANIFEST
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                if (manifest.get("schema") != SCHEMA or manifest.get("source") != source_info
                        or manifest.get("recipe_sha256") != recipe_hash or manifest.get("tensors") != mapped
                        or manifest.get("source_asset_bindings") != asset_bindings):
                    raise ValueError("existing artifact has different source or conversion recipe")
                if manifest.get("finished"):
                    return verify_manifest(output, manifest)
            else:
                if any(p.name != ".conversion.lock" for p in output.iterdir()):
                    raise ValueError("refusing unrelated nonempty output directory")
                manifest = dict(schema=SCHEMA, model_type="deepseek41_mlx", format="mlx2-v41-affine-v1",
                                numeric_profile="native_f32_qmm", finished=False, qualification="unqualified",
                                source=source_info, metadata=gguf.metadata, tensors=mapped, files={}, assets=[],
                                source_asset_bindings=asset_bindings, tokenizer_binding=admitted.get("tokenizer_binding", {}),
                                recipe=recipe_value, recipe_sha256=recipe_hash, inventory=summary,
                                source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                                writer_sha256=file_hash(__file__), started_unix=time.time())
                manifest["inventory"]["source_tensor_count"] = len(mapped)
                atomic_json(manifest_path, manifest)
            proof = algebra_selftest()
            source_fd = gguf._file.fileno()
            def source_check():
                if stamp(source_fd) != source_stamp or stamp(source_path) != source_stamp:
                    raise ValueError("sealed GGUF changed during conversion")
            source_check()
            completed, records = 0, {}
            # Tensor files first; largest PLE copies run last so stop/resume is
            # useful while CPU workers and runtime integration are developed.
            order = sorted(jobs, key=lambda p: (p.startswith("engram/"), p))
            for relative in order:
                source_check()
                target = safe_path(output, relative)
                receipt_path = safe_path(output, "conversion-state/" + hashlib.sha256(relative.encode()).hexdigest() + ".json")
                if receipt_path.exists():
                    receipt = json.loads(receipt_path.read_text())
                    if receipt.get("path") != relative or receipt.get("recipe_sha256") != recipe_hash:
                        raise ValueError("resume receipt binding changed")
                    record = receipt["record"]
                    verify_file(output, relative, record)
                    if stamp(target) != record["stat"]:
                        raise ValueError("resume file stat seal changed")
                else:
                    if max_files is not None and completed >= max_files:
                        break
                    check_space(output, specs[relative], reserve_bytes)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    partial = safe_path(output, relative + ".partial")
                    entries = jobs[relative]
                    if entries[0].get("raw_file"):
                        tensor = gguf.tensors[entries[0]["name"]]
                        expected_hash = copy_range(source_fd, gguf.data_offset + tensor.offset, tensor.nbytes, partial)
                        record = dict(size=tensor.nbytes, header_bytes=0, tensors={},
                                      proof=dict(byte_exact_raw=True, source_sha256=expected_hash))
                    else:
                        expected_hash = None
                        record = write_safetensors(source_fd, gguf.data_offset, gguf.tensors, entries, partial)
                    source_check()
                    record["sha256"] = file_hash(partial)
                    if expected_hash is not None and expected_hash != record["sha256"]:
                        raise ValueError("raw output differs from source bytes")
                    os.replace(partial, target)
                    record["stat"] = stamp(target)
                    verify_file(output, relative, record)
                    atomic_json(receipt_path, dict(path=relative, recipe_sha256=recipe_hash, record=record))
                    completed += 1
                records[relative] = record
                if len(records) % 32 == 0 or completed == 1:
                    progress = dict(completed_files=len(records), total_files=len(jobs), converted_bytes=sum(r["size"] for r in records.values()),
                                    last_file=relative, updated_unix=time.time())
                    atomic_json(output / "conversion-progress.json", progress)
                    phase(json.dumps(progress))
            if len(records) != len(jobs):
                return dict(finished=False, completed_files=len(records), total_files=len(jobs), newly_written=completed)
            for relative, original in asset_sources.items():
                binding = asset_bindings[relative]
                if original.is_symlink() or stamp(original) != binding["stat"]:
                    raise ValueError("source asset changed during conversion")
                target = safe_path(output, relative)
                check_space(output, original.stat().st_size, reserve_bytes)
                temporary = safe_path(output, relative + ".partial")
                with original.open("rb") as stream:
                    digest = copy_range(stream.fileno(), 0, original.stat().st_size, temporary)
                if digest != binding["sha256"] or stamp(original) != binding["stat"]:
                    raise ValueError("source asset bytes changed during conversion")
                os.replace(temporary, target)
                records[relative] = dict(size=target.stat().st_size, sha256=digest, stat=stamp(target), header_bytes=0, tensors={})
            native_config = dict(model_type="deepseek41_mlx", mlx2_weight_format="mlx2-v41-affine-v1", manifest_file=MANIFEST)
            atomic_json(output / "config.json", native_config)
            records["config.json"] = dict(size=(output / "config.json").stat().st_size, sha256=file_hash(output / "config.json"),
                                          stat=stamp(output / "config.json"), header_bytes=0, tensors={})
            manifest["assets"] = sorted([*asset_sources, "config.json"])
            manifest["files"] = records
            manifest["algebra_validation"] = proof
            phase(json.dumps(dict(phase="verify_all_converted_file_hashes", files=len(records), updated_unix=time.time())))
            manifest["verification"] = verify_manifest(output, manifest)
            source_check()
            phase(json.dumps(dict(phase="verify_full_original_gguf_sha256", bytes=source_stamp["size"], updated_unix=time.time())))
            original_sha = source_hash(source_fd, source_stamp["size"])
            source_check()
            if original_sha != source_info["gguf_sha256"]:
                raise ValueError("full original GGUF SHA256 differs from admitted identity")
            manifest["verification"].update(original_full_sha256=original_sha, original_full_sha256_checked=True)
            manifest["finished"] = True
            manifest["finished_unix"] = time.time()
            manifest["derived_identity_sha256"] = hashlib.sha256(canonical(dict(source=source_info["fingerprint"], recipe=recipe_hash,
                                                    files={p: r["sha256"] for p, r in records.items()}))).hexdigest()
            atomic_json(manifest_path, manifest)
            return dict(finished=True, **manifest["verification"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--reserve-gib", type=int, default=100)
    parser.add_argument("--max-files", type=int)
    args = parser.parse_args()
    if args.reserve_gib < 0 or (args.max_files is not None and args.max_files < 0):
        parser.error("reserve and max-files must be nonnegative")
    if args.verify_only:
        manifest = json.loads((args.output / MANIFEST).read_text())
        if not manifest.get("finished"):
            raise ValueError("artifact is not complete")
        result = verify_manifest(args.output, manifest)
    else:
        if args.source is None:
            parser.error("--source is required for conversion")
        result = convert(args.source, args.output, reserve_bytes=args.reserve_gib << 30,
                         max_files=args.max_files, plan_only=args.plan)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

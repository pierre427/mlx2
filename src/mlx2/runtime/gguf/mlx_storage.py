"""Lossless packed safetensors storage; quantization semantics remain GGUF's.

The CPU converter streams payloads without creating MLX arrays. Serving loads
ordinary safetensors arrays directly, preserving uint8 packed buffers and native
scalar dtypes. This container is intentionally not affine MLX quantization.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from dataclasses import asdict, replace
from pathlib import Path

from .k_blocks import SIZES as K_FORMATS
from .reader import FORMATS, GGUFError, GGUFFile, GGUFTensor

SCHEMA = "mlx2.packed-safetensors.v1"
MANIFEST = "mlx2-packed.json"
SEAL = "mlx2-packed-integrity.json"
NATIVE = {0: ("F32", "<f4"), 1: ("F16", "<f2"), 24: ("I8", "i1"),
          25: ("I16", "<i2"), 26: ("I32", "<i4"), 27: ("I64", "<i8"),
          30: ("BF16", "<u2")}


def stamp(path):
    s = Path(path).stat()
    return dict(size=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns,
                inode=s.st_ino, device=s.st_dev)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    with path.with_suffix(path.suffix + ".tmp").open("x") as f:
        json.dump(value, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    path.with_suffix(path.suffix + ".tmp").replace(path)


def local_file(root, name):
    if not isinstance(name, str) or Path(name).name != name or name in (".", ".."):
        raise GGUFError("packed storage requires a local basename")
    p = Path(root) / name
    if p.is_symlink() or not p.is_file():
        raise GGUFError("packed storage asset must be a regular nonsymlink file")
    return p


def read_header(path):
    with path.open("rb") as f:
        nbytes = f.read(8)
        if len(nbytes) != 8:
            raise GGUFError("truncated safetensors header")
        size = struct.unpack("<Q", nbytes)[0]
        if not 2 <= size <= 16 << 20:
            raise GGUFError("oversized safetensors header")
        header = json.loads(f.read(size))
    end = 0
    for name, t in sorted(header.items(), key=lambda item: item[1]["data_offsets"][0]):
        begin, stop = t["data_offsets"]
        if begin != end or stop < begin or stop + size + 8 > path.stat().st_size:
            raise GGUFError("overlapping/truncated safetensors payload")
        end = stop
    if end + size + 8 != path.stat().st_size:
        raise GGUFError("safetensors payload extent differs")
    return header, size + 8


def inspect_storage(root, *, verify_payload=False):
    root = Path(root)
    manifest_path = local_file(root, MANIFEST)
    seal = json.loads(local_file(root, SEAL).read_text())
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema") != SCHEMA or seal.get("schema") != SCHEMA
            or seal.get("manifest_sha256") != sha256(manifest_path)):
        raise GGUFError("packed manifest differs from integrity seal")
    headers = {}
    if set(seal.get("shards", {})) != set(manifest["shards"]):
        raise GGUFError("packed shard seal inventory differs")
    for name, record in manifest["shards"].items():
        path = local_file(root, name)
        before = stamp(path)
        if seal["shards"][name] != before or before["size"] != record["bytes"]:
            raise GGUFError("packed shard changed after verification: " + name)
        if verify_payload and sha256(path) != record["sha256"]:
            raise GGUFError("packed shard SHA256 differs: " + name)
        headers[name] = read_header(path)
        if stamp(path) != before:
            raise GGUFError("packed shard changed during inspection")
    seen = set()
    for name, record in manifest["tensors"].items():
        kind, shape = record["ggml_type"], record["shape"]
        if kind not in FORMATS or not 1 <= len(shape) <= 4 or any(type(d) is not int or d < 1 for d in shape):
            raise GGUFError("invalid packed tensor geometry")
        _, block, size = FORMATS[kind]
        if shape[0] % block or record["nbytes"] != math.prod(shape) // block * size:
            raise GGUFError("invalid packed tensor byte geometry")
        key, shard = record["key"], record["shard"]
        h = headers[shard][0][key]
        expected_dtype = NATIVE[kind][0] if kind in NATIVE else "U8"
        expected_shape = shape[::-1] if kind in NATIVE else [record["nbytes"]]
        if h["dtype"] != expected_dtype or h["shape"] != expected_shape or h["data_offsets"][1]-h["data_offsets"][0] != record["nbytes"]:
            raise GGUFError("packed safetensors descriptor differs")
        if (shard, key) in seen:
            raise GGUFError("aliased packed tensor storage")
        seen.add((shard, key))
    for name, record in manifest.get("linear_fp32", {}).items():
        t = manifest["tensors"][name]
        h = headers[record["shard"]][0][record["key"]]
        layout = record.get("layout", "contiguous_operand")
        if layout not in ("base_transposed_view", "contiguous_operand"):
            raise GGUFError("unsupported prepared FP32 layout")
        expected_shape = t["shape"][::-1] if layout == "base_transposed_view" else t["shape"]
        if t["ggml_type"] not in (1, 30) or len(t["shape"]) != 2 or h["dtype"] != "F32" or h["shape"] != expected_shape:
            raise GGUFError("invalid prepared FP32 linear operand")
        if h["data_offsets"][1]-h["data_offsets"][0] != math.prod(t["shape"])*4:
            raise GGUFError("invalid prepared FP32 linear extent")
        seen.add((record["shard"], record["key"]))
    if seen != {(s, k) for s, (h, _) in headers.items() for k in h}:
        raise GGUFError("unregistered packed storage payload")
    return manifest, seal, headers


def convert_gguf(source, destination, *, source_sha256, linear_names=(), shard_bytes=2 << 30, progress=None):
    """Create a new container, fully verify source and output, publish seal last.

    Existing nonempty destinations are refused. A failed conversion stays
    incomplete and cannot load; original source bytes are never modified.
    """
    import numpy as np

    source, destination = Path(source), Path(destination)
    if source.is_symlink() or not source.is_file() or type(shard_bytes) is not int or shard_bytes < 1:
        raise GGUFError("regular source and positive shard bound required")
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise GGUFError("refusing nonempty conversion destination")
    before = stamp(source)
    if sha256(source) != source_sha256 or stamp(source) != before:
        raise GGUFError("conversion source SHA256/stat differs")
    manifest = dict(schema=SCHEMA, source_sha256=source_sha256, metadata={}, tensors={}, linear_fp32={}, shards={})
    with GGUFFile(source) as file:
        manifest["metadata"] = file.metadata
        groups, pending, total = [], [], 0
        for name, t in file.tensors.items():
            if t.ggml_type not in NATIVE and t.ggml_type not in (8,10,16) and t.ggml_type not in K_FORMATS:
                raise GGUFError("unsupported packed MLX tensor format")
            # Embeddings stay in an independent shard for bounded file-row reads.
            if pending and (total + t.nbytes > shard_bytes or name == "token_embd.weight"):
                groups.append(pending); pending, total = [], 0
            pending.append(name); total += t.nbytes
            if name == "token_embd.weight":
                groups.append(pending); pending, total = [], 0
        if pending:
            groups.append(pending)
        for number, names in enumerate(groups, 1):
            shard = f"model-{number:05d}.safetensors"
            entries, header, position = [], {}, 0
            for name in names:
                t = file.tensors[name]
                dtype = NATIVE[t.ggml_type][0] if t.ggml_type in NATIVE else "U8"
                shape = list(t.numpy_shape) if t.ggml_type in NATIVE else [t.nbytes]
                header[name] = dict(dtype=dtype, shape=shape, data_offsets=[position, position+t.nbytes])
                position += t.nbytes
                entries.append((name, None))
                manifest["tensors"][name] = {**asdict(t), "shape": list(t.shape), "offset": 0, "shard": shard, "key": name}
                if name in linear_names and t.ggml_type in (1, 30):
                    view = file.packed_view(name)
                    try:
                        raw = np.frombuffer(view, dtype=NATIVE[t.ggml_type][1]).reshape(t.numpy_shape)
                        fp32 = raw.astype(np.float32) if t.ggml_type == 1 else (raw.astype(np.uint32) << 16).view(np.float32)
                        # MLX's transpose-then-cast retains column-major strides.
                        # Save the base in C order; loading then taking a view
                        # preserves that operand layout without a runtime copy.
                        prepared = np.ascontiguousarray(fp32)
                    finally:
                        del raw
                        view.release()
                    key = "__linear_fp32__." + name
                    header[key] = dict(dtype="F32", shape=list(t.numpy_shape), data_offsets=[position, position+prepared.nbytes])
                    position += prepared.nbytes
                    entries.append((key, prepared))
                    manifest["linear_fp32"][name] = dict(shard=shard, key=key, layout="base_transposed_view")
            encoded = json.dumps(header, separators=(",", ":")).encode()
            encoded += b" " * (-len(encoded) % 8)
            h = hashlib.sha256()
            with (destination/shard).open("xb") as output:
                for value in (struct.pack("<Q", len(encoded)), encoded):
                    output.write(value); h.update(value)
                for key, prepared in entries:
                    view = memoryview(prepared).cast("B") if prepared is not None else file.packed_view(key)
                    payload = hashlib.sha256()
                    try:
                        for start in range(0, len(view), 8 << 20):
                            chunk = view[start:start+(8 << 20)]
                            output.write(chunk); h.update(chunk); payload.update(chunk)
                            chunk.release()
                    finally:
                        view.release()
                    if prepared is None:
                        manifest["tensors"][key]["sha256"] = payload.hexdigest()
                    else:
                        manifest["linear_fp32"][key.removeprefix("__linear_fp32__.")]["sha256"] = payload.hexdigest()
                output.flush(); os.fsync(output.fileno())
            file.release_mapping()
            if sha256(destination/shard) != h.hexdigest():
                raise GGUFError("converted shard verification failed")
            manifest["shards"][shard] = dict(bytes=(destination/shard).stat().st_size, sha256=h.hexdigest())
            if progress:
                progress(dict(completed_shards=number, total_shards=len(groups), completed_bytes=sum(r["bytes"] for r in manifest["shards"].values())))
        if stamp(source) != before:
            raise GGUFError("conversion source changed")
    atomic_json(destination/MANIFEST, manifest)
    atomic_json(destination/SEAL, dict(schema=SCHEMA, manifest_sha256=sha256(destination/MANIFEST),
                                     shards={n:stamp(destination/n) for n in manifest["shards"]}))
    inspect_storage(destination)
    return manifest


class MLXPackedFile:
    """Small sealed index plus lazy native MLX shard loads, owned by weights."""
    def __init__(self, root):
        self.path = Path(root)
        self.manifest, self.seal, self.headers = inspect_storage(root)
        self.metadata = self.manifest["metadata"]
        self.tensors = {n:GGUFTensor(n, tuple(t["shape"]), t["ggml_type"], 0, t["nbytes"]) for n,t in self.manifest["tensors"].items()}
        self._loaded = {}
        self._mapping = None

    def _get(self, record):
        import mlx.core as mx
        name = record["shard"]
        path = local_file(self.path, name)
        if stamp(path) != self.seal["shards"][name]:
            raise GGUFError("packed shard changed before load")
        if name not in self._loaded:
            self._loaded[name] = mx.load(str(path))
        if stamp(path) != self.seal["shards"][name]:
            raise GGUFError("packed shard changed during load")
        return self._loaded[name][record["key"]]

    def device_array(self, name):
        return self._get(self.manifest["tensors"][name])

    def linear_fp32(self, name):
        record = self.manifest.get("linear_fp32", {}).get(name)
        if record is None:
            return None
        import mlx.core as mx

        value = self._get(record)
        if record.get("layout") == "base_transposed_view":
            return value.T
        # Earlier diagnostic containers saved the operand in C order. Restore
        # the reference layout when reading them; new conversions need no copy.
        value = mx.contiguous(value.T).T
        self._loaded[record["shard"]][record["key"]] = value
        return value

    def embedding_location(self, name):
        r = self.manifest["tensors"][name]
        h, offset = self.headers[r["shard"]]
        return self.path/r["shard"], replace(self.tensors[name], offset=0), offset+h[r["key"]]["data_offsets"][0], self.seal["shards"][r["shard"]]

    def packed_view(self, name):
        import mmap

        self.release_mapping()
        path, tensor, start, expected = self.embedding_location(name)
        if stamp(path) != expected:
            raise GGUFError("packed shard changed before host read")
        with path.open("rb") as f:
            self._mapping = mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
        return memoryview(self._mapping)[start:start+tensor.nbytes]

    def release_mapping(self):
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None

    def close(self):
        self._loaded.clear()
        self.release_mapping()

    def verify(self):
        manifest, seal, headers = inspect_storage(self.path)
        if manifest != self.manifest or seal != self.seal or headers != self.headers:
            raise GGUFError("packed storage binding changed during preload")

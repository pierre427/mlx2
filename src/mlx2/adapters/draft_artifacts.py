"""Metadata-only inspection shared by external draft artifact loaders.

Reads safetensors headers (never payloads), checks the exact expected schema,
dtype, byte sizes, shard/index agreement and payload overlap, and returns a
fingerprint binding config, index, shard stats and header digests.  Mirrors
the DFlash2 inspector (``adapters/dflash2.py``) with the schema as a
parameter so each drafter family owns only its expected shapes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .dflash2 import _decode_unique_json, _read_safetensors_header

_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4}
_DTYPES = {"bfloat16": "BF16", "float16": "F16", "float32": "F32"}


def validate_headers(paths, mapping, expected, dtype, *, label):
    observed, digests = {}, []
    for path in paths:
        raw, header, payload = _read_safetensors_header(path)
        digests.append(hashlib.sha256(raw).hexdigest())
        ranges = []
        for name, record in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(record, dict) or set(record) != {"dtype", "shape", "data_offsets"}:
                raise ValueError(f"Invalid tensor metadata for {name!r}")
            if name in observed:
                raise ValueError(f"Duplicate {label} tensor across shards: {name}")
            shape, offsets = record["shape"], record["data_offsets"]
            if record["dtype"] != dtype or not isinstance(shape, list) or not all(
                type(v) is int and v >= 0 for v in shape
            ):
                raise ValueError(f"{label} tensor dtype/shape mismatch: {name}")
            if not (
                isinstance(offsets, list) and len(offsets) == 2
                and all(type(v) is int for v in offsets)
                and 0 <= offsets[0] <= offsets[1] <= payload
            ):
                raise ValueError(f"Invalid {label} tensor offsets: {name}")
            elements = 1
            for value in shape:
                elements *= value
            if offsets[1] - offsets[0] != elements * _DTYPE_BYTES[dtype]:
                raise ValueError(f"{label} tensor byte size mismatch: {name}")
            ranges.append((offsets[0], offsets[1], name))
            observed[name] = shape
            if mapping is not None and mapping.get(name) != path.name:
                raise ValueError(f"{label} weight index/shard mismatch: {name}")
        ranges.sort()
        for previous, current in zip(ranges, ranges[1:]):
            if previous[1] > current[0]:
                raise ValueError(f"Overlapping {label} tensor payloads: {previous[2]}, {current[2]}")
    if mapping is not None and set(mapping) != set(observed):
        raise ValueError(f"{label} weight index does not match shard headers")
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        wrong = sorted(n for n in set(expected) & set(observed) if expected[n] != observed[n])
        raise ValueError(
            f"{label} checkpoint schema mismatch (missing={missing}, extra={extra}, wrong_shapes={wrong})"
        )
    return digests


def inspect_files(path, config_bytes, *, expected, dtype, label):
    """Resolve shards (index or single file), validate headers, fingerprint."""
    path = Path(path).expanduser().resolve()
    dtype_code = _DTYPES.get(str(dtype).lower())
    if dtype_code is None:
        raise ValueError(f"Unsupported {label} checkpoint dtype: {dtype!r}")
    digest = hashlib.sha256()
    digest.update(config_bytes)
    index = path / "model.safetensors.index.json"
    if index.exists():
        content = index.read_bytes()
        digest.update(content)
        mapping = _decode_unique_json(content, f"{label} weight index").get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"Invalid {label} weight index")
        names = sorted(set(mapping.values()))
    else:
        mapping, names = None, ["model.safetensors"]
    files, paths = [], []
    for name in names:
        # HF-cache snapshots are symlinks into blobs/, so the shard name (not
        # its resolved target) is what must stay inside the artifact.
        if (
            not isinstance(name, str) or "/" in name or "\\" in name
            or name.startswith(".") or not name.endswith(".safetensors")
            or not (path / name).is_file()
        ):
            raise ValueError(f"Invalid {label} shard path")
        stat = (path / name).stat()
        record = (name, stat.st_size, stat.st_mtime_ns)
        files.append(record)
        paths.append(path / name)
        digest.update(json.dumps(record).encode())
    header_digests = validate_headers(paths, mapping, expected, dtype_code, label=label)
    digest.update(json.dumps(header_digests).encode())
    return {
        "path": str(path),
        "fingerprint": digest.hexdigest(),
        "files": files,
        "header_sha256": header_digests,
    }


__all__ = ["inspect_files", "validate_headers"]

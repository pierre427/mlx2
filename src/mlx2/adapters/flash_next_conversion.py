# SPDX-License-Identifier: MIT
"""CPU-only proof for preserving a direct-table PLE sidecar while reindexing.

The runtime PLE verifier remains the authority at load time. This offline
operation may update its index binding only after proving every PLE tensor
and every sidecar row is unchanged. Tensor-free converted sidecars require
their original builder and are deliberately not admitted here.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np

from ..runtime.models.ngram_table_keys import check_ngram_table_tensors

MANIFEST_NAME = "ple_rows.bin.manifest.json"
SIDECAR_NAME = "ple_rows.bin"
_REBUILD = "Rebuild the PLE sidecar against the converted artifact; automatic rebinding was refused."
_HEX = re.compile(r"^[0-9a-f]{64}$")


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_hash(tensor):
    digest = hashlib.sha256()
    raw = tensor.raw()
    for start in range(0, tensor.nbytes, 8 << 20):
        digest.update(raw[start:start + (8 << 20)])
    return digest.hexdigest()


def _geometry(manifest):
    if manifest.get("format") != "qwen4-ple-rows" or manifest.get("version") != 1:
        raise ValueError("unsupported PLE manifest format/version. " + _REBUILD)
    if {k: manifest.get(k) for k in ("bits", "group_size", "mode")} != {"bits": 4, "group_size": 32, "mode": "affine"}:
        raise ValueError("unsupported PLE quantization. " + _REBUILD)
    for name in ("dims", "num_shards", "rows_per_shard", "total_rows", "row_bytes"):
        if type(manifest.get(name)) is not int or manifest[name] <= 0:
            raise ValueError(f"invalid PLE {name}. " + _REBUILD)
    dims = manifest["dims"]
    if dims % 32 or type(manifest.get("data_offset")) is not int or manifest["data_offset"] < 0:
        raise ValueError("invalid PLE dimensions/data offset. " + _REBUILD)
    lengths = {"weight": dims // 2, "scales": dims // 32 * 2, "biases": dims // 32 * 2}
    if any(manifest.get(part + "_bytes") != width for part, width in lengths.items()) or manifest["row_bytes"] != sum(lengths.values()):
        raise ValueError("PLE row geometry differs from the supported layout. " + _REBUILD)
    if manifest["total_rows"] != manifest["num_shards"] * manifest["rows_per_shard"]:
        raise ValueError("PLE total row count differs. " + _REBUILD)
    hashes = manifest.get("shard_sha256")
    if not isinstance(hashes, list) or len(hashes) != manifest["num_shards"] or any(not isinstance(x, str) or not _HEX.fullmatch(x) for x in hashes):
        raise ValueError("PLE logical shard digests are missing or invalid. " + _REBUILD)
    if not isinstance(manifest.get("tensor_prefix"), str) or not manifest["tensor_prefix"]:
        raise ValueError("PLE tensor prefix is missing. " + _REBUILD)
    return lengths


def prepare_ple_rebind(source):
    """Validate source binding and all direct-table rows before output creation."""
    path = source.path
    manifest_file, sidecar = path / MANIFEST_NAME, path / SIDECAR_NAME
    if not manifest_file.is_file() or not sidecar.is_file():
        raise ValueError("PLE sidecar and manifest must both exist. " + _REBUILD)
    manifest = json.loads(manifest_file.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("PLE manifest must be an object. " + _REBUILD)
    lengths = _geometry(manifest)
    actual_index = _sha_file(path / "model.safetensors.index.json")
    if manifest.get("source_index_sha256") != actual_index:
        raise ValueError("source PLE manifest index binding is stale. " + _REBUILD)
    if manifest.get("conversion") is not None:
        raise ValueError("already-converted PLE sidecars need their original conversion builder. " + _REBUILD)
    prefix = manifest["tensor_prefix"]
    check_ngram_table_tensors(source.tensors, prefix=prefix, num_shards=manifest["num_shards"])
    required = {f"{prefix}.shard_{i}.{part}" for i in range(manifest["num_shards"]) for part in lengths}
    present = {name for name in source.tensors if name.startswith(prefix + ".shard_")}
    if present != required:
        raise ValueError("direct PLE table tensors are absent or incomplete. " + _REBUILD)
    rows, dims = manifest["rows_per_shard"], manifest["dims"]
    descriptors = {}
    for name in sorted(required):
        tensor = source.tensors[name]
        part = name.rsplit(".", 1)[1]
        expected_dtype = "U32" if part == "weight" else "BF16"
        expected_shape = (rows, dims // (8 if part == "weight" else 32))
        if tensor.dtype != expected_dtype or tensor.shape != expected_shape or tensor.nbytes != rows * lengths[part]:
            raise ValueError(f"PLE tensor layout differs: {name}. " + _REBUILD)
        descriptors[name] = {"dtype": tensor.dtype, "shape": list(tensor.shape), "nbytes": tensor.nbytes}
    expected_size = manifest["data_offset"] + manifest["total_rows"] * manifest["row_bytes"]
    if sidecar.stat().st_size != expected_size:
        raise ValueError("source PLE sidecar size differs from the manifest. " + _REBUILD)
    full = hashlib.sha256()
    tensor_hashes = {}
    # Full row comparison binds the sidecar to actual source bytes, rather
    # than trusting an index or a sampled row population to authorize rebinding.
    with sidecar.open("rb") as stream:
        remaining = manifest["data_offset"]
        while remaining:
            chunk = stream.read(min(remaining, 8 << 20))
            if not chunk:
                raise ValueError("truncated PLE sidecar prefix. " + _REBUILD)
            full.update(chunk); remaining -= len(chunk)
        batch = max(1, (8 << 20) // manifest["row_bytes"])
        for i in range(manifest["num_shards"]):
            parts = {part: source.tensors[f"{prefix}.shard_{i}.{part}"].raw() for part in lengths}
            hashes = {part: hashlib.sha256() for part in lengths}
            shard = hashlib.sha256()
            for start in range(0, rows, batch):
                count = min(batch, rows - start)
                joined = np.empty((count, manifest["row_bytes"]), dtype=np.uint8)
                cursor = 0
                for part, width in lengths.items():
                    raw = parts[part][start * width:(start + count) * width]
                    hashes[part].update(raw)
                    joined[:, cursor:cursor + width] = raw.reshape(count, width)
                    cursor += width
                data = joined.tobytes()
                observed = stream.read(len(data))
                if observed != data:
                    raise ValueError(f"source PLE sidecar differs from table rows in shard {i}. " + _REBUILD)
                shard.update(data); full.update(observed)
            if shard.hexdigest() != manifest["shard_sha256"][i]:
                raise ValueError(f"source PLE logical shard {i} digest is stale. " + _REBUILD)
            for part, digest in hashes.items():
                tensor_hashes[f"{prefix}.shard_{i}.{part}"] = digest.hexdigest()
    full_hash = full.hexdigest()
    if manifest.get("file_sha256") is not None and manifest["file_sha256"] != full_hash:
        raise ValueError("source PLE full-file digest is stale. " + _REBUILD)
    return {"schema": "mlx2.sc-mxfp4-ple-rebind.v1", "source_path": str(path),
            "source_index_sha256": actual_index, "manifest": manifest,
            "manifest_sha256": _sha_file(manifest_file), "sidecar_sha256": full_hash,
            "protected_tensors": sorted(required), "tensor_descriptors": descriptors,
            "tensor_sha256": tensor_hashes}


def finish_ple_rebind(source, output, binding, *, conversion_schema):
    """Rebind after proving every copied PLE payload remains byte-identical."""
    path = output.path
    if _sha_file(source.path / MANIFEST_NAME) != binding["manifest_sha256"] or _sha_file(source.path / "model.safetensors.index.json") != binding["source_index_sha256"]:
        raise ValueError("source PLE metadata changed during conversion. " + _REBUILD)
    manifest = binding["manifest"]
    check_ngram_table_tensors(output.tensors, prefix=manifest["tensor_prefix"], num_shards=manifest["num_shards"])
    present = {name for name in output.tensors if name.startswith(manifest["tensor_prefix"] + ".shard_")}
    if present != set(binding["protected_tensors"]):
        raise ValueError("converted PLE table tensor set changed. " + _REBUILD)
    for name, expected in binding["tensor_descriptors"].items():
        original, copied = source.tensors.get(name), output.tensors.get(name)
        if original is None or copied is None or any(getattr(copied, key) != (tuple(value) if key == "shape" else value) for key, value in expected.items()):
            raise ValueError(f"converted PLE tensor layout changed: {name}. " + _REBUILD)
        if _tensor_hash(original) != binding["tensor_sha256"][name] or _tensor_hash(copied) != binding["tensor_sha256"][name]:
            raise ValueError(f"converted PLE tensor bytes changed: {name}. " + _REBUILD)
    for sidecar in (source.path / SIDECAR_NAME, path / SIDECAR_NAME):
        if _sha_file(sidecar) != binding["sidecar_sha256"]:
            raise ValueError("copied PLE sidecar bytes changed. " + _REBUILD)
    new_index = _sha_file(path / "model.safetensors.index.json")
    provenance = {"schema": binding["schema"], "operation": "unchanged-direct-table-reindex",
                  "conversion_schema": conversion_schema, "source_model_path": str(source.path),
                  "source_index_sha256": binding["source_index_sha256"], "output_index_sha256": new_index,
                  "source_manifest_sha256": binding["manifest_sha256"],
                  "sidecar_sha256": binding["sidecar_sha256"], "table_tensor_sha256": binding["tensor_sha256"]}
    manifest = dict(binding["manifest"])
    manifest.update(source_index_sha256=new_index, source_model_path=str(path),
                    file_sha256=binding["sidecar_sha256"], sc_mxfp4_rebind=provenance)
    target = path / MANIFEST_NAME
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(tmp, target)
    return provenance


def rebound_ple_artifact_identity(path):
    """Use the serving adapter's exact fingerprint after its manifest changes."""
    from .flash_next import artifact_identity
    return {"scheme": "mlx2-flash-next", **artifact_identity(Path(path))}

#!/usr/bin/env python3
"""Convert an mlx-serve Qwen4-Exp pack into mlx2's fail-closed layout.

The source pack stores the PLE table as one safetensors-like, tensor-major
``ngram_table.bin``. mlx2 requires row-interleaved ``ple_rows.bin`` data plus
an artifact-bound manifest. Model shards are hard-linked into a new directory;
the source artifact is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import time

import numpy as np


ROWS_PER_SHARD = 2_500_012
NUM_SHARDS = 128
BLOCK_ROWS = 262_144


def read_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"truncated header length: {path}")
        (length,) = struct.unpack("<Q", raw)
        header = json.loads(handle.read(length))
    return header, 8 + length


def file_sha256(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: dict) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(partial, path)


def finalize_existing(source: Path, target: Path) -> None:
    """Bind an already converted sidecar to its receipt with a full-file hash."""
    manifest_path = target / "ple_rows.bin.manifest.json"
    receipt_path = target / "mlx2-conversion.json"
    manifest = json.loads(manifest_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema") != "mlx2.artifact-conversion.v1":
        raise ValueError("existing conversion receipt has an unsupported schema")
    if Path(receipt.get("source", "")).resolve() != source:
        raise ValueError("existing conversion receipt source does not match")
    if Path(receipt.get("target", "")).resolve() != target:
        raise ValueError("existing conversion receipt target does not match")
    index_sha256 = file_sha256(target / "model.safetensors.index.json")
    if manifest.get("source_index_sha256") != index_sha256:
        raise ValueError("existing manifest is not bound to the target model index")
    if receipt.get("source_index_sha256") != index_sha256:
        raise ValueError("existing receipt is not bound to the target model index")
    source_index = json.loads((source / "model.safetensors.index.json").read_text())
    headers = tensor_headers(source, source_index)
    config_path = target / "config.json"
    config = json.loads(config_path.read_text())
    config["quantization"] = quantization_config(headers, config.get("quantization", {}))
    normalize_eos_config(config, source)
    write_json_atomic(config_path, config)
    ple_sha256 = file_sha256(target / "ple_rows.bin")
    if receipt.get("ple_output_sha256") != ple_sha256:
        raise ValueError("existing PLE sidecar does not match its conversion receipt")
    manifest["file_sha256"] = ple_sha256
    manifest["conversion"] = {
        "schema": "mlx2.mlx-serve-ngram-import.v1",
        "receipt": "mlx2-conversion.json",
        "source_format": "mlx-serve-ngram",
    }
    write_json_atomic(manifest_path, manifest)
    write_json_atomic(receipt_path, receipt)


def validate_ngram(path: Path) -> tuple[dict, int]:
    header, data_offset = read_header(path)
    metadata = header.get("__metadata__", {})
    if metadata != {"format": "mlx-serve-ngram", "bits": "4", "group_size": "32"}:
        raise ValueError(f"unsupported ngram metadata: {metadata}")
    expected = {
        "weight": ("U32", [320_001_536, 20], [0, 25_600_122_880]),
        "scales": ("BF16", [320_001_536, 5], [25_600_122_880, 28_800_138_240]),
        "biases": ("BF16", [320_001_536, 5], [28_800_138_240, 32_000_153_600]),
    }
    for name, (dtype, shape, offsets) in expected.items():
        record = header.get(name, {})
        observed = (record.get("dtype"), record.get("shape"), record.get("data_offsets"))
        if observed != (dtype, shape, offsets):
            raise ValueError(f"unsupported {name} layout: {observed}")
    if path.stat().st_size != data_offset + expected["biases"][2][1]:
        raise ValueError("ngram file size does not match its declared tensors")
    return header, data_offset


def tensor_headers(source: Path, index: dict) -> dict:
    merged = {}
    for file_name in sorted(set(index["weight_map"].values())):
        header, _ = read_header(source / file_name)
        header.pop("__metadata__", None)
        overlap = set(merged).intersection(header)
        if overlap:
            raise ValueError(f"duplicate tensor names: {sorted(overlap)[:3]}")
        merged.update(header)
    return merged


def runtime_weight_prefix(prefix: str) -> str:
    """Mirror Qwen4 Model.sanitize naming for quantization predicates."""
    if prefix.startswith("model.language_model.mtp."):
        return prefix.replace("model.language_model.mtp.", "mtp.", 1)
    if prefix.startswith("language_model.mtp."):
        return prefix.replace("language_model.mtp.", "mtp.", 1)
    if prefix.startswith("model.mtp."):
        return prefix.removeprefix("model.")
    if prefix.startswith("model.language_model"):
        return prefix.replace("model.language_model", "language_model.model", 1)
    if not prefix.startswith(("language_model.", "mtp.")):
        return "language_model." + prefix
    return prefix


def normalize_eos_config(config: dict, artifact_path: Path) -> None:
    """Make mlx2 stop on both the tokenizer EOS and the model text EOS."""
    tokenizer_config = json.loads((artifact_path / "tokenizer_config.json").read_text())
    eos_token = tokenizer_config.get("eos_token")
    if isinstance(eos_token, dict):
        eos_token = eos_token.get("content")
    tokenizer = json.loads((artifact_path / "tokenizer.json").read_text())
    token_ids = {
        item.get("content"): item.get("id")
        for item in tokenizer.get("added_tokens", [])
        if isinstance(item.get("id"), int)
    }
    ids = []
    if isinstance(eos_token, str) and eos_token in token_ids:
        ids.append(token_ids[eos_token])
    for declared in (
        config.get("eos_token_id"),
        config.get("text_config", {}).get("eos_token_id"),
    ):
        for token_id in declared if isinstance(declared, list) else [declared]:
            if isinstance(token_id, int) and token_id not in ids:
                ids.append(token_id)
    if not ids:
        raise ValueError("could not resolve an EOS token id from the artifact")
    config["eos_token_id"] = ids if len(ids) > 1 else ids[0]


def quantization_config(headers: dict, base: dict) -> dict:
    result = {"group_size": 64, "bits": 4, "mode": "affine"}
    for key, weight in headers.items():
        if not key.endswith(".weight"):
            continue
        prefix = key.removesuffix(".weight")
        scales = headers.get(prefix + ".scales")
        if not scales:
            continue
        weight_width = weight["shape"][-1]
        scale_width = scales["shape"][-1]
        if scale_width <= 0 or weight_width % scale_width:
            raise ValueError(f"cannot infer quantization for {prefix}")
        ratio = weight_width // scale_width
        if ratio == 16:
            result[runtime_weight_prefix(prefix)] = {
                "group_size": 64,
                "bits": 8,
                "mode": "affine",
            }
        elif ratio != 8:
            raise ValueError(f"unsupported packed/scales ratio {ratio} for {prefix}")
    return result


def convert_ple(source_file: Path, output: Path, index_sha256: str) -> dict:
    header, data_offset = validate_ngram(source_file)
    offsets = {name: data_offset + header[name]["data_offsets"][0] for name in ("weight", "scales", "biases")}
    row_sizes = {"weight": 80, "scales": 10, "biases": 10}
    partial = output.with_suffix(output.suffix + ".partial")
    shard_hashes = []
    started = time.monotonic()
    descriptor = os.open(source_file, os.O_RDONLY)
    try:
        with partial.open("wb") as target:
            for shard in range(NUM_SHARDS):
                digest = hashlib.sha256()
                shard_row = shard * ROWS_PER_SHARD
                for local in range(0, ROWS_PER_SHARD, BLOCK_ROWS):
                    rows = min(BLOCK_ROWS, ROWS_PER_SHARD - local)
                    global_row = shard_row + local
                    parts = {}
                    for name, row_size in row_sizes.items():
                        parts[name] = os.pread(
                            descriptor,
                            rows * row_size,
                            offsets[name] + global_row * row_size,
                        )
                        if len(parts[name]) != rows * row_size:
                            raise ValueError(f"short {name} read at row {global_row}")
                    block = np.empty((rows, 100), dtype=np.uint8)
                    block[:, :80] = np.frombuffer(parts["weight"], dtype=np.uint8).reshape(rows, 80)
                    block[:, 80:90] = np.frombuffer(parts["scales"], dtype=np.uint8).reshape(rows, 10)
                    block[:, 90:] = np.frombuffer(parts["biases"], dtype=np.uint8).reshape(rows, 10)
                    data = block.tobytes()
                    target.write(data)
                    digest.update(data)
                shard_hashes.append(digest.hexdigest())
                if (shard + 1) % 8 == 0:
                    done = (shard + 1) * ROWS_PER_SHARD * 100
                    rate = done / max(time.monotonic() - started, 1e-9) / (1 << 20)
                    print(f"PLE {shard + 1}/{NUM_SHARDS}: {done / (1 << 30):.2f} GiB, {rate:.0f} MiB/s", flush=True)
    finally:
        os.close(descriptor)
    os.replace(partial, output)
    return {
        "format": "qwen4-ple-rows",
        "version": 1,
        "tensor_prefix": "language_model.model.layers.1.ple.ple_embedding.ngram_embedding",
        "dims": 160,
        "group_size": 32,
        "bits": 4,
        "mode": "affine",
        "weight_bytes": 80,
        "scales_bytes": 10,
        "biases_bytes": 10,
        "row_bytes": 100,
        "num_shards": NUM_SHARDS,
        "rows_per_shard": ROWS_PER_SHARD,
        "total_rows": NUM_SHARDS * ROWS_PER_SHARD,
        "data_offset": 0,
        "source_model_path": str(source_file.parent.resolve()),
        "source_index_sha256": index_sha256,
        "shard_sha256": shard_hashes,
        "created_unix": int(time.time()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="add the fail-closed full-file binding to an existing conversion",
    )
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    target = args.target.expanduser().resolve()
    if args.finalize_existing:
        if not target.is_dir():
            raise FileNotFoundError(f"converted target does not exist: {target}")
        finalize_existing(source, target)
        print(target)
        return 0
    if target.exists():
        raise FileExistsError(f"refusing existing target: {target}")
    temp = target.with_name(target.name + ".partial")
    if temp.exists():
        raise FileExistsError(f"refusing existing partial target: {temp}")
    index_path = source / "model.safetensors.index.json"
    index_bytes = index_path.read_bytes()
    index = json.loads(index_bytes)
    headers = tensor_headers(source, index)
    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "qwen4_exp" or "ngram_table" not in config:
        raise ValueError("source is not the expected mlx-serve qwen4_exp pack")
    temp.mkdir(parents=True)
    try:
        for item in source.iterdir():
            if not item.is_file() or item.name in {"config.json", "ngram_table.bin"}:
                continue
            os.link(item, temp / item.name)
        config.pop("ngram_table")
        config["quantization"] = quantization_config(headers, config.get("quantization", {}))
        normalize_eos_config(config, source)
        (temp / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        ple = temp / "ple_rows.bin"
        manifest = convert_ple(
            source / "ngram_table.bin",
            ple,
            hashlib.sha256(index_bytes).hexdigest(),
        )
        ple_sha256 = file_sha256(ple)
        manifest["file_sha256"] = ple_sha256
        manifest["conversion"] = {
            "schema": "mlx2.mlx-serve-ngram-import.v1",
            "receipt": "mlx2-conversion.json",
            "source_format": "mlx-serve-ngram",
        }
        write_json_atomic(temp / "ple_rows.bin.manifest.json", manifest)
        receipt = {
            "schema": "mlx2.artifact-conversion.v1",
            "source": str(source),
            "target": str(target),
            "source_completion_marker": json.loads((source / ".hf-download-complete.json").read_text()),
            "source_index_sha256": hashlib.sha256(index_bytes).hexdigest(),
            "ple_output_sha256": ple_sha256,
            "completed_unix": int(time.time()),
        }
        write_json_atomic(temp / "mlx2-conversion.json", receipt)
        os.replace(temp, target)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

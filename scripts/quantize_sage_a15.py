#!/usr/bin/env python3
"""CPU-only streamed affine candidates from the pinned Sage BF16 expansion.

Q8 minimizes further loss from the already Q4_K_M source. mixed-q4 uses Q4
routed experts, Q8 elsewhere and weight-MSE refinement; it is not calibrated
oQ4e. Recurrent parameters, routers, norms and convolutions are preserved.
See provenance/sage-a15-affine-requantization-20261022.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import shutil
import struct
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mlx2.conversion.affine import quantize_rows

FINGERPRINT = "ba62aaf8bbd40762c67b16fa012947b52b5f5e35f8d73d69101d401a0dc27b71"
GROUP_SIZE = 64


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def tensor_header(path):
    with path.open("rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        if length > 1 << 20:
            raise ValueError("Unexpected safetensors header size")
        header = json.loads(f.read(length))
    entries = {k: v for k, v in header.items() if k != "__metadata__"}
    if len(entries) != 1:
        raise ValueError("Expected one tensor per Sage source shard")
    key, entry = next(iter(entries.items()))
    if entry["dtype"] not in ("BF16", "F32"):
        raise ValueError("Source is not a floating-point Sage expansion")
    size = math.prod(entry["shape"]) * (2 if entry["dtype"] == "BF16" else 4)
    if entry["data_offsets"] != [0, size] or path.stat().st_size != 8 + length + size:
        raise ValueError("Malformed source tensor offsets")
    return key, entry, 8 + length


def policy(key, shape, profile):
    # nn.Linear / nn.Embedding / SwitchLinear only, never convolution or gates.
    eligible = key.endswith(".weight") and len(shape) in (2, 3) and shape[-1] % GROUP_SIZE == 0
    protected = (".conv1d.", ".mlp.gate.", ".shared_expert_gate.")
    if not eligible or any(part in key for part in protected):
        return None
    if not (key.endswith(("_proj.weight", ".embed_tokens.weight", ".lm_head.weight", ".mtp.fc.weight"))
            or ".linear_attn.in_proj_" in key):
        return None
    return 4 if profile == "mixed-q4" and ".switch_mlp." in key else 8


def quantize_tensor(source, output, key, entry, offset, bits, rows_per_chunk):
    shape = entry["shape"]
    rows, width = math.prod(shape[:-1]), shape[-1]
    specs = [(key, "U32", shape[:-1] + [width * bits // 32], 4),
             (key.removesuffix(".weight") + ".scales", "BF16", shape[:-1] + [width // GROUP_SIZE], 2),
             (key.removesuffix(".weight") + ".biases", "BF16", shape[:-1] + [width // GROUP_SIZE], 2)]
    header, data_size = {}, 0
    for name, dtype, stored_shape, item_size in specs:
        size = math.prod(stored_shape) * item_size
        header[name] = {"dtype": dtype, "shape": stored_shape, "data_offsets": [data_size, data_size + size]}
        data_size += size
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    temporary = output.with_suffix(".part")
    with temporary.open("wb") as f:
        f.write(struct.pack("<Q", len(raw))); f.write(raw); f.truncate(8 + len(raw) + data_size)
    dtype = "<u2" if entry["dtype"] == "BF16" else "<f4"
    values = np.memmap(source, mode="r", dtype=dtype, offset=offset, shape=(rows, width))
    maps = [np.memmap(temporary, mode="r+", dtype="<u4" if item_size == 4 else "<u2",
                      offset=8 + len(raw) + header[name]["data_offsets"][0],
                      shape=(rows, stored_shape[-1])) for name, _, stored_shape, item_size in specs]
    totals = {"error_before": 0., "error_after": 0., "elements": 0}
    for start in range(0, rows, rows_per_chunk):
        chunk = np.asarray(values[start:start + rows_per_chunk])
        chunk = (chunk.astype(np.uint32) << 16).view(np.float32) if dtype == "<u2" else chunk
        packed, scales, biases, error = quantize_rows(chunk, bits=bits, group_size=GROUP_SIZE, refine=bits < 8)
        for target, result in zip(maps, (packed, scales, biases), strict=True):
            target[start:start + len(chunk)] = result
        for field in totals:
            totals[field] += error[field]
    for target in maps:
        target.flush()
    del maps, values
    os.replace(temporary, output)
    return header, totals


def convert_shard(source, staging, filename, record, profile, rows_per_chunk):
    """One independently owned shard; never modify shared progress or metadata."""
    src, dst = source / filename, staging / filename
    if not record or sha256(src) != record["sha256"]:
        raise ValueError("Source shard hash mismatch: " + filename)
    key, entry, offset = tensor_header(src)
    bits = policy(key, entry["shape"], profile)
    error = None
    if bits is None:
        shutil.copy2(src, dst)
        header = {key: entry}
    else:
        header, error = quantize_tensor(src, dst, key, entry, offset, bits, rows_per_chunk)
    result = {"file": filename, "source_sha256": record["sha256"],
              "sha256": sha256(dst), "bytes": dst.stat().st_size, "bits": bits, "error": error}
    return key, header, result


def convert(source, output, profile="q8", rows_per_chunk=128, workers=1):
    from mlx2.adapters.qwen36_35b import inspect_artifact

    source = source.expanduser().resolve(strict=True)
    output = output.expanduser().absolute()
    if profile not in ("q8", "mixed-q4") or rows_per_chunk < 1 or type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError("Invalid conversion profile or row bound")
    if output.exists() or output.is_symlink() or output.is_relative_to(source):
        raise FileExistsError("Destination must be a new separate artifact")
    inspected = inspect_artifact(source)
    if inspected["identity"]["fingerprint"] != FINGERPRINT or inspected["config"].get("quantization"):
        raise ValueError("Source differs from the pinned Sage BF16 expansion")
    if shutil.disk_usage(output.parent).free < (40 << 30) + (80 << 30):
        raise OSError("Conversion would cross the 80 GiB free-space reserve")
    receipt = json.loads((source / "mlx2-gguf-conversion.json").read_text())
    records = {item["filename"]: item for item in receipt["files"].values()}
    staging = output.with_name(output.name + ".incomplete")
    staging.mkdir()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip()
    report = {"schema": "mlx2.sage-affine-requantization.v1", "source_checkpoint": source.name + "@sha256:" + FINGERPRINT,
              "source_revision": revision, "converter_sha256": sha256(Path(__file__)), "affine_sha256": sha256(Path(__file__).resolve().parents[1] / "src/mlx2/conversion/affine.py"),
              "profile": profile, "workers": workers, "activation_calibrated": False, "qualified": False,
              "source_already_quantized": True, "started_at": time.time(), "files": [], "modules": {}}
    weight_map, total_bytes = {}, 0
    try:
        filenames = sorted(set(inspected["weight_map"].values()))
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
        try:
            for index, (key, header, result) in enumerate(pool.map(convert_shard, repeat(source), repeat(staging), filenames,
                                                               (records.get(name) for name in filenames), repeat(profile), repeat(rows_per_chunk))):
                filename, bits = result["file"], result["bits"]
                if bits is not None:
                    report["modules"][key.removesuffix(".weight")] = {"bits": bits, "group_size": GROUP_SIZE, "mode": "affine"}
                weight_map.update({name: filename for name in header})
                total_bytes += sum(v["data_offsets"][1] - v["data_offsets"][0] for v in header.values())
                report["files"].append(result)
                report["shards_done"] = index + 1
                atomic_json(staging / "progress.json", {k: v for k, v in report.items() if k != "files"})
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        config = inspected["config"]
        quant = {"bits": 8 if profile == "q8" else 4, "group_size": GROUP_SIZE, "mode": "affine", **report["modules"]}
        config["quantization"] = config["quantization_config"] = quant
        config["mlx2_requantization"] = {"profile": profile, "source_checkpoint": report["source_checkpoint"], "activation_calibrated": False, "qualified": False}
        atomic_json(staging / "config.json", config)
        atomic_json(staging / "model.safetensors.index.json", {"metadata": {"total_size": total_bytes}, "weight_map": weight_map})
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json", "README.md", "LICENSE"):
            if (source / name).is_file():
                shutil.copy2(source / name, staging / name)
        report.update(finished_at=time.time(), tensor_bytes=total_bytes, checkpoint=output.name + "@sha256:" + inspect_artifact(staging)["identity"]["fingerprint"])
        atomic_json(staging / "mlx2-requantization.json", report)
        (staging / "progress.json").unlink()
        os.rename(staging, output)
        return report
    except BaseException:
        atomic_json(staging / "interrupted.json", {"qualified": False, "shards_done": len(report["files"]), "time": time.time()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("q8", "mixed-q4"), default="q8")
    parser.add_argument("--rows-per-chunk", type=int, default=128)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    result = convert(args.source, args.output, args.profile, args.rows_per_chunk, args.workers)
    print(json.dumps({key: result[key] for key in ("checkpoint", "profile", "tensor_bytes", "qualified")}))

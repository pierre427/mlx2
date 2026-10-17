"""Stream the pinned BF16 Super VL source to native affine MLX tensors.

No model or accelerator is imported. Each source shard is hashed while its
payload is consumed once. Expert fusion writes directly to final offsets,
so the 249 GB source and a full fused expert layer never occupy RAM.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import struct
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..adapters.nemotron35_super_config import (
    SOURCE_CONFIG_SHA256,
    SOURCE_INDEX_SHA256,
    SOURCE_REVISION,
    tensor_schema,
)
from .affine import quantize_rows
from .nemotron35_plan import make_plan
from .nvfp4 import DTYPE_BYTES, read_header, sha256

SOURCE_REPOSITORY = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
EXPERT = re.compile(r"(.*\.mixer\.)experts\.(\d+)\.(up_proj|down_proj)\.weight$")
METADATA = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
            "generation_config.json", "special_tokens_map.json", "vocab.json",
            "merges.txt", "preprocessor_config.json", "processor_config.json",
            "LICENSE", "NOTICE")


def fused_destination(name):
    match = EXPERT.fullmatch(name)
    if match is None:
        return name, None
    prefix, expert, direction = match.groups()
    return prefix + "switch_mlp." + ("fc1" if direction == "up_proj" else "fc2") + ".weight", int(expert)


def tensor_bytes(entry):
    return math.prod(entry["shape"]) * DTYPE_BYTES[entry["dtype"]]


class ShardWriter:
    """Exclusive seekable safetensors shards with fully planned spans."""
    def __init__(self, directory, schema, *, max_shard_bytes=4_000_000_000):
        self.directory, self.locations, self.handles = directory, {}, {}
        groups, group, count = [], {}, 0
        for name, entry in sorted(schema.items()):
            size = tensor_bytes(entry)
            if group and count + size > max_shard_bytes:
                groups.append(group)
                group, count = {}, 0
            group[name] = entry
            count += size
        if group:
            groups.append(group)
        for index, group in enumerate(groups, 1):
            filename = f"model-{index:05d}-of-{len(groups):05d}.safetensors"
            offset, header = 0, {"__metadata__": {"format": "mlx"}}
            for name, entry in group.items():
                size = tensor_bytes(entry)
                header[name] = {**entry, "data_offsets": [offset, offset + size]}
                offset += size
            raw = json.dumps(header, separators=(",", ":")).encode()
            raw += b" " * (-len(raw) % 8)
            handle = (directory / filename).open("xb+")
            handle.write(struct.pack("<Q", len(raw)) + raw)
            payload = 8 + len(raw)
            handle.truncate(payload + offset)
            self.handles[filename] = handle
            for name in group:
                self.locations[name] = (filename, payload + header[name]["data_offsets"][0], tensor_bytes(group[name]))

    def write(self, name, offset, data):
        filename, start, size = self.locations[name]
        if offset < 0 or offset + len(data) > size:
            raise ValueError(f"Output slice escapes planned tensor: {name}")
        handle = self.handles[filename]
        handle.seek(start + offset)
        if handle.write(data) != len(data):
            raise OSError("Short output write")

    def close(self):
        for handle in self.handles.values():
            if not handle.closed:
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()


def inspect_source(source):
    source = Path(source).resolve()
    completed = json.loads((source / ".hf-download-complete.json").read_text())
    manifest = json.loads((source / ".hf-download-manifest.json").read_text())
    state = json.loads((source / ".hf-download-state.json").read_text())
    for value in (completed, manifest, state):
        if value.get("repo") != SOURCE_REPOSITORY or value.get("revision") != SOURCE_REVISION:
            raise ValueError("Source acquisition is not complete at the pinned revision")
    if any(source.rglob("*.part")):
        raise ValueError("Source still has partial downloads")
    config = json.loads((source / "config.json").read_text())
    if sha256(source / "config.json") != SOURCE_CONFIG_SHA256 or sha256(source / "model.safetensors.index.json") != SOURCE_INDEX_SHA256:
        raise ValueError("Source config/index differs from pinned release")
    schema = tensor_schema(config)
    weight_map = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    if set(weight_map) != set(schema):
        raise ValueError("Source index differs from the complete pinned tensor topology")
    entries = {entry["path"]: entry for entry in manifest["files"]}
    files, observed = {}, set()
    for name in sorted(set(weight_map.values())):
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError("Source shard path escapes artifact")
        path = source / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("Missing source shard")
        info, record = entries[name], state["files"][name]
        stat = path.stat()
        if not info.get("sha256") or record.get("sha256") != info["sha256"] or stat.st_size != info["size"]:
            raise ValueError("Source acquisition hash/size record differs")
        header, payload = read_header(path)
        tensors = {key: value for key, value in header.items() if key != "__metadata__"}
        if set(tensors) != {key for key, file in weight_map.items() if file == name}:
            raise ValueError("Source shard/index mismatch")
        for key, entry in tensors.items():
            if {field: entry[field] for field in ("shape", "dtype")} != schema[key] or key in observed:
                raise ValueError(f"Source tensor differs: {key}")
            observed.add(key)
        files[name] = {"path": path, "tensors": tensors, "payload": payload,
                       "sha256": info["sha256"], "stat": stat}
    if observed != set(schema):
        raise ValueError("Incomplete source closure")
    return config, files


def metadata_snapshot(source):
    manifest = json.loads((source / ".hf-download-manifest.json").read_text())
    entries = {item["path"]: item for item in manifest["files"]}
    result = {}
    for filename in METADATA:
        if filename not in entries:
            continue
        entry, path = entries[filename], source / filename
        if path.is_symlink() or not path.is_file() or path.stat().st_size != entry["size"]:
            raise ValueError("Source metadata is absent, escaped or incomplete")
        raw = path.read_bytes()
        if entry.get("sha256"):
            valid = hashlib.sha256(raw).hexdigest() == entry["sha256"]
        else:
            valid = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == entry["blob_id"]
        if not valid:
            raise ValueError(f"Source metadata hash differs: {filename}")
        result[filename] = raw
    return result


def convert_checkpoint(source, destination, *, expert_bits=4, reserve_bytes=50_000_000_000,
                       importance_path=None, progress=None, workers=4):
    source, destination = Path(source).resolve(), Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if type(reserve_bytes) is not int or reserve_bytes < 0:
        raise ValueError("Invalid free-space reserve")
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("CPU fitting workers must be from one to eight")
    config, files = inspect_source(source)
    metadata = metadata_snapshot(source)
    source_config_hash, source_index_hash = sha256(source / "config.json"), sha256(source / "model.safetensors.index.json")
    plan = make_plan(config, expert_bits=expert_bits)
    importance = None
    calibration = None
    expert_policy = None
    if importance_path is not None:
        from .importance import ActivationMoments, statistics_specs

        importance = ActivationMoments.load(importance_path, specs=statistics_specs(plan))
        coverage = importance.require_image_samples()
        expert_policy = importance.expert_policy(minimum_count=coverage["minimum_required"])
        calibration = {"binding": importance.binding, "coverage": coverage,
                       "expert_policy": expert_policy,
                       "manifest_sha256": sha256(Path(importance_path) / "manifest.json"),
                       "moments_sha256": sha256(Path(importance_path) / "moments.npz"),
                       "scope": "covered low-bit experts use activation-weighted Q4; undercovered experts and protected 8-bit tensors retain stored-BF16 weight-MSE fitting"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < plan["tensor_bytes"] + reserve_bytes + 100_000_000:
        raise OSError("Mixed quant would consume the free-space reserve")
    staging = destination.with_name(destination.name + ".incomplete-" + uuid.uuid4().hex)
    staging.mkdir()
    writer, started = ShardWriter(staging, plan["schema"]), time.time()
    processed, errors, fit_errors, written = [], {}, {}, {}
    executor = ThreadPoolExecutor(max_workers=workers)

    def finish_chunk(task, output_name, row, fit_mode):
        packed, scales, biases, error = task.result()
        prefix = output_name.removesuffix(".weight")
        for target, array in ((output_name, packed), (prefix + ".scales", scales), (prefix + ".biases", biases)):
            data = array.tobytes()
            writer.write(target, row * array.shape[-1] * array.dtype.itemsize, data)
            written[target] = written.get(target, 0) + len(data)
        accumulator = errors.setdefault(prefix, {"error_before": 0., "error_after": 0., "elements": 0})
        for field in accumulator:
            accumulator[field] += error[field]
        if fit_mode is not None:
            mode = fit_errors.setdefault(prefix, {}).setdefault(
                fit_mode, {"error_before": 0., "error_after": 0., "elements": 0})
            for field in mode:
                mode[field] += error[field]

    try:
        for file_number, (filename, file) in enumerate(files.items(), 1):
            if shutil.disk_usage(destination.parent).free < reserve_bytes:
                raise OSError("Mixed quant crossed the free-space reserve")
            digest = hashlib.sha256()
            with file["path"].open("rb") as handle:
                digest.update(handle.read(file["payload"]))
                for name, entry in sorted(file["tensors"].items(), key=lambda item: item[1]["data_offsets"][0]):
                    output_name, expert = fused_destination(name)
                    quant = plan["modules"].get(output_name.removesuffix(".weight"))
                    rows = entry["shape"][0] if quant else 1
                    columns = entry["shape"][-1] if quant else 1
                    if quant:
                        prefix = output_name.removesuffix(".weight")
                        channel_importance = (importance.importance_for(
                            prefix, expert, minimum_count=expert_policy["minimum_observations"])
                            if importance is not None and quant["bits"] < 8 else None)
                        fit_mode = None
                        if importance is not None and quant["bits"] < 8:
                            fit_mode = ("activation_weighted" if channel_importance is not None
                                        else "stored_bf16_weight_mse_fallback")
                        chunk_rows = max(1, (1 << 18) // columns)
                        pending = deque()
                        for first in range(0, rows, chunk_rows):
                            count = min(chunk_rows, rows - first)
                            raw = handle.read(count * columns * 2)
                            if len(raw) != count * columns * 2:
                                raise ValueError("Truncated source during conversion")
                            digest.update(raw)
                            values = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32).reshape(count, columns)
                            row = first + (expert or 0) * rows
                            pending.append((executor.submit(quantize_rows, values, bits=quant["bits"], group_size=quant["group_size"],
                                                            importance=channel_importance), output_name, row, fit_mode))
                            if len(pending) >= 2 * workers:
                                finish_chunk(*pending.popleft())
                        while pending:
                            finish_chunk(*pending.popleft())
                    else:
                        size, consumed = tensor_bytes(entry), 0
                        while consumed < size:
                            raw = handle.read(min(8 << 20, size - consumed))
                            if not raw:
                                raise ValueError("Truncated source during conversion")
                            digest.update(raw)
                            writer.write(output_name, consumed, raw)
                            consumed += len(raw)
                        written[output_name] = written.get(output_name, 0) + consumed
                if handle.read(1):
                    raise ValueError("Unexpected source trailing payload")
            stat = file["path"].stat()
            if (digest.hexdigest() != file["sha256"] or stat.st_size != file["stat"].st_size
                    or stat.st_mtime_ns != file["stat"].st_mtime_ns):
                raise ValueError("Source changed or failed SHA-256 during conversion")
            processed.append({"file": filename, "sha256": digest.hexdigest()})
            event = {"state": "converting", "source_shards_done": file_number,
                     "source_shards_total": len(files), "elapsed_seconds": time.time() - started}
            (staging / "progress.json").write_text(json.dumps(event, indent=2) + "\n")
            if progress:
                progress(event)
        if written != {name: tensor_bytes(entry) for name, entry in plan["schema"].items()}:
            raise ValueError("Converted output did not fill every planned tensor exactly once")
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        writer.close()
    output_files = []
    for name in writer.handles:
        read_header(staging / name)
        output_files.append({"file": name, "sha256": sha256(staging / name), "bytes": (staging / name).stat().st_size})
    for filename, raw in metadata.items():
        (staging / filename).write_bytes(raw)
    if sha256(source / "config.json") != source_config_hash or sha256(source / "model.safetensors.index.json") != source_index_hash:
        raise ValueError("Source metadata changed during conversion")
    config["quantization"] = {"group_size": 64, "bits": expert_bits, "mode": "affine", **plan["modules"]}
    (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    index = {"metadata": {"total_size": plan["tensor_bytes"]},
             "weight_map": {name: item[0] for name, item in writer.locations.items()}}
    (staging / "model.safetensors.index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    receipt = {"schema": "mlx2.nemotron35-super-vl-conversion.v1", "recipe": plan["recipe"],
               "expert_bits": expert_bits, "tensor_bytes": plan["tensor_bytes"],
               "bytes_by_category": plan["bytes_by_category"],
               "source_repository": SOURCE_REPOSITORY, "source_revision": SOURCE_REVISION,
               "source_config_sha256": source_config_hash,
               "source_index_sha256": source_index_hash,
               "metadata_sha256": {name: hashlib.sha256(raw).hexdigest() for name, raw in metadata.items()},
               "source_shards": processed, "output_shards": output_files,
               "output_config_sha256": sha256(staging / "config.json"),
               "output_index_sha256": sha256(staging / "model.safetensors.index.json"),
               "weight_fit": ("stored-BF16 hybrid activation-weighted/fallback MSE clipping and alternating least squares"
                              if importance is not None else
                              "stored-BF16 MSE clipping and alternating least squares"),
               "calibration_mode": ("hybrid_activation_weighted_q4_with_weight_mse_q4_fallback"
                                    if importance is not None else None),
               "activation_aware": importance is not None,
               "fully_activation_weighted": bool(importance is not None and expert_policy["fallback_experts"] == 0),
               "calibration": calibration, "errors": errors, "fit_errors_by_mode": fit_errors,
               "elapsed_seconds": time.time() - started,
               "cpu_fitting_workers": workers,
               "qualified": False, "selected": False, "observed_used": False}
    (staging / "mlx2-conversion.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (staging / "progress.json").unlink()
    # Never replace another artifact, even if one appeared during conversion.
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    os.rename(staging, destination)
    return receipt

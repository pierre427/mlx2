"""Convert the pinned mixed-native Bolt checkpoint to native affine MLX tensors.

Every GGML IQ2_XXS / Q8_0 and ModelOpt NVFP4 tensor is decoded on the CPU
(bit-exact to the reference laws), rounded once to BF16, and refit with the
same stored-BF16 MSE clipping/LSQ affine fitter as Super.  No model or
accelerator is imported.  Expert fusion writes straight to final offsets.
See provenance/nemotron35-bolt.json.
"""
from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import re
import shutil
import time
import uuid
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .affine import bf16_float, quantize_rows
from .ggml import dequantize_iq2_xxs, dequantize_q8_0
from .nemotron35 import ShardWriter, fused_destination, tensor_bytes
from .nemotron35_bolt_plan import make_plan, tensor_schema
from .nvfp4 import bf16_bits, decode_rows, read_header, sha256

SOURCE_REPOSITORY = "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB"
SCHEMA = "mlx2.nemotron35-bolt-conversion.v1"
METADATA = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
            "generation_config.json", "preprocessor_config.json", "processor_config.json")
_SCALE = re.compile(r".*\.weight_scale(_2)?$")
_GEOMETRY = {"IQ2_XXS": (256, 66), "Q8_0": (32, 34)}


def _hf_lfs_sha256(source, name):
    """The download's recorded LFS SHA-256 (second line of the hub metadata)."""
    record = source / ".cache" / "huggingface" / "download" / (name + ".metadata")
    lines = record.read_text().splitlines()
    if len(lines) < 2 or len(lines[1]) != 64:
        raise ValueError(f"Missing hub LFS record: {name}")
    return lines[1]


def inspect_source(source, *, verify_payload=True):
    """Pinned metadata/header closure (Codex inspector) plus LFS payload hashes."""
    from ..adapters.nemotron35_bolt import inspect_source_artifact

    artifact = inspect_source_artifact(source)
    config = artifact["config"]
    quant = json.loads((source / "hf_quant_config.json").read_text())
    algos = {name + ".weight": item["quant_algo"]
             for name, item in quant["quantization"]["quantized_layers"].items()}
    index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    files = {}
    for name in sorted(set(index.values())):
        path = source / name
        header, prefix = read_header(path)
        expected = _hf_lfs_sha256(source, name)
        if verify_payload and sha256(path) != expected:
            raise ValueError(f"Bolt shard differs from its LFS SHA-256: {name}")
        files[name] = {"path": path, "payload": prefix, "sha256": expected,
                       "stat": path.stat(),
                       "tensors": {k: v for k, v in header.items() if k != "__metadata__"}}
    return artifact, config, algos, files


def _decode(kind, raw, rows, columns, extra=None):
    data = np.frombuffer(raw, dtype=np.uint8)
    if kind in _GEOMETRY:
        block, size = _GEOMETRY[kind]
        blocks = data.reshape(rows, columns // block, size)
        return (dequantize_iq2_xxs if kind == "IQ2_XXS" else dequantize_q8_0)(blocks)
    if kind == "W4A16_NVFP4":
        scales, global_scale = extra
        return decode_rows(data.reshape(rows, columns // 2),
                           np.frombuffer(scales, np.uint8).reshape(rows, columns // 16),
                           global_scale)
    if kind == "BF16":
        return (data.view("<u2").astype(np.uint32) << 16).view(np.float32).reshape(rows, columns)
    raise ValueError(f"Unsupported source law: {kind}")


def _row_bytes(kind, columns):
    if kind in _GEOMETRY:
        block, size = _GEOMETRY[kind]
        return columns // block * size
    if kind == "W4A16_NVFP4":
        return columns // 2
    return columns * 2


def _work(kind, raw, rows, columns, extra, quant):
    values = bf16_float(_decode(kind, raw, rows, columns, extra))
    if not np.isfinite(values).all():
        raise ValueError("Non-finite dequantized source values")
    if quant is None:
        return bf16_bits(values), None
    packed, scales, biases, error = quantize_rows(
        values, bits=quant["bits"], group_size=quant["group_size"])
    error["sum_squares"] = float(np.square(values, dtype=np.float64).sum())
    return (packed, scales, biases), error


def convert_checkpoint(source, destination, *, reserve_bytes=50_000_000_000,
                       workers=6, progress=None):
    source, destination = Path(source).resolve(), Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if type(workers) is not int or not 1 <= workers <= 12:
        raise ValueError("CPU fitting workers must be from one to twelve")
    started = time.time()
    artifact, config, algos, files = inspect_source(source)
    plan = make_plan(config)
    logical = tensor_schema(config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < plan["tensor_bytes"] + reserve_bytes:
        raise OSError("Bolt conversion would consume the free-space reserve")
    owner = {}
    for filename, file in files.items():
        for name in file["tensors"]:
            owner[name] = filename
    staging = destination.with_name(destination.name + ".incomplete-" + uuid.uuid4().hex)
    staging.mkdir()
    writer = ShardWriter(staging, plan["schema"])
    errors, written, laws = {}, {}, {}
    descriptors = {name: os.open(file["path"], os.O_RDONLY) for name, file in files.items()}
    # Processes, not threads: the affine refit is many small NumPy calls and
    # thread workers serialize on the GIL and allocator (measured ~14x slower).
    executor = ProcessPoolExecutor(max_workers=workers,
                                   mp_context=multiprocessing.get_context("spawn"))

    def pread(name, start, length):
        file = files[owner[name]]
        # nvfp4.read_header returns the header prefix length (8 + JSON).
        begin = file["payload"] + file["tensors"][name]["data_offsets"][0]
        data = os.pread(descriptors[owner[name]], length, begin + start)
        if len(data) != length:
            raise ValueError(f"Truncated Bolt source read: {name}")
        return data

    def finish(task, output_name, row):
        arrays, error = task.result()
        prefix = output_name.removesuffix(".weight")
        if error is None:
            data = arrays.tobytes()
            writer.write(output_name, row * arrays.shape[-1] * 2, data)
            written[output_name] = written.get(output_name, 0) + len(data)
            return
        for target, array in zip((output_name, prefix + ".scales", prefix + ".biases"), arrays):
            data = array.tobytes()
            writer.write(target, row * array.shape[-1] * array.dtype.itemsize, data)
            written[target] = written.get(target, 0) + len(data)
        total = errors.setdefault(prefix, {"error_before": 0., "error_after": 0.,
                                           "sum_squares": 0., "elements": 0})
        for field in total:
            total[field] += error[field]

    try:
        names = sorted((n for n in owner if not _SCALE.fullmatch(n)),
                       key=lambda n: (owner[n], files[owner[n]]["tensors"][n]["data_offsets"][0]))
        pending = deque()
        for number, name in enumerate(names, 1):
            entry = files[owner[name]]["tensors"][name]
            output_name, expert = fused_destination(name)
            kind = algos.get(name, entry["dtype"])
            laws[kind] = laws.get(kind, 0) + 1
            shape = logical[name]["shape"]
            quant = plan["modules"].get(output_name.removesuffix(".weight"))
            if kind in ("F32", "I64") or (kind == "BF16" and quant is None):
                raw = pread(name, 0, tensor_bytes(entry))
                writer.write(output_name, 0, raw)
                written[output_name] = written.get(output_name, 0) + len(raw)
                continue
            rows, columns = math.prod(shape[:-1]), shape[-1]
            extra = None
            if kind == "W4A16_NVFP4":
                prefix = name.removesuffix(".weight")
                scale_entry = files[owner[prefix + ".weight_scale"]]["tensors"][prefix + ".weight_scale"]
                scales = pread(prefix + ".weight_scale", 0, tensor_bytes(scale_entry))
                global_scale = np.frombuffer(pread(prefix + ".weight_scale_2", 0, 4), "<f4")[0]
            row_bytes = _row_bytes(kind, columns)
            chunk_rows = max(1, (1 << 18) // columns)
            for first in range(0, rows, chunk_rows):
                count = min(chunk_rows, rows - first)
                raw = pread(name, first * row_bytes, count * row_bytes)
                if kind == "W4A16_NVFP4":
                    extra = (scales[first * columns // 16:(first + count) * columns // 16],
                             global_scale)
                row = first + (expert or 0) * rows
                pending.append((executor.submit(_work, kind, raw, count, columns, extra, quant),
                                output_name, row))
                while len(pending) >= 3 * workers:
                    finish(*pending.popleft())
            if number % 500 == 0 or number == len(names):
                event = {"state": "converting", "source_tensors_done": number,
                         "source_tensors_total": len(names),
                         "elapsed_seconds": time.time() - started}
                (staging / "progress.json").write_text(json.dumps(event) + "\n")
                if progress:
                    progress(event)
                if shutil.disk_usage(destination.parent).free < reserve_bytes:
                    raise OSError("Bolt conversion crossed the free-space reserve")
        while pending:
            finish(*pending.popleft())
        if written != {name: tensor_bytes(entry) for name, entry in plan["schema"].items()}:
            raise ValueError("Converted output did not fill every planned tensor exactly once")
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        writer.close()
        for descriptor in descriptors.values():
            os.close(descriptor)
    for filename, file in files.items():
        stat = file["path"].stat()
        if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != (
                file["stat"].st_size, file["stat"].st_mtime_ns, file["stat"].st_ino):
            raise ValueError("Bolt source changed during conversion")
    output_files = []
    for name in writer.handles:
        read_header(staging / name)
        output_files.append({"file": name, "sha256": sha256(staging / name),
                             "bytes": (staging / name).stat().st_size})
    metadata = {}
    for filename in METADATA:
        raw = (source / filename).read_bytes()
        (staging / filename).write_bytes(raw)
        metadata[filename] = hashlib.sha256(raw).hexdigest()
    source_quantization = config.pop("quantization_config")
    config["quantization"] = {"group_size": 32, "bits": 4, "mode": "affine", **plan["modules"]}
    (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    index = {"metadata": {"total_size": plan["tensor_bytes"]},
             "weight_map": {name: item[0] for name, item in writer.locations.items()}}
    (staging / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n")
    relative = {prefix: math.sqrt(item["error_after"] / item["sum_squares"])
                for prefix, item in errors.items() if item["sum_squares"] > 0}
    receipt = {
        "schema": SCHEMA, "recipe": plan["recipe"], "expert_bits": 4,
        "expert_group_size": plan["expert_group_size"],
        "tensor_bytes": plan["tensor_bytes"], "bytes_by_category": plan["bytes_by_category"],
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": artifact["identity"]["source_revision"],
        "source_fingerprint": artifact["identity"]["fingerprint"],
        "source_config_sha256": sha256(source / "config.json"),
        "source_quant_sha256": sha256(source / "hf_quant_config.json"),
        "source_index_sha256": sha256(source / "model.safetensors.index.json"),
        "source_quantization_config": source_quantization,
        "source_laws": laws,
        "source_shards": [{"file": name, "sha256": file["sha256"]} for name, file in files.items()],
        "source_payload_verification": "every shard SHA-256 equals its hub LFS record before conversion",
        "metadata_sha256": metadata, "output_shards": output_files,
        "output_config_sha256": sha256(staging / "config.json"),
        "output_index_sha256": sha256(staging / "model.safetensors.index.json"),
        "dequantization": "GGML IQ2_XXS/Q8_0 (bit-exact fp32 vs gguf-py 0.19.0) and ModelOpt NVFP4, rounded once to BF16",
        "weight_fit": "stored-BF16 MSE clipping and alternating least squares",
        "activation_aware": False, "errors": errors,
        "relative_rms_vs_bf16_dequant": relative,
        "elapsed_seconds": time.time() - started, "cpu_fitting_workers": workers,
        "qualified": False, "selected": False, "observed_used": False,
    }
    (staging / "mlx2-conversion.json").write_text(json.dumps(receipt, indent=2) + "\n")
    progress_file = staging / "progress.json"
    if progress_file.exists():
        progress_file.unlink()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    os.rename(staging, destination)
    return receipt

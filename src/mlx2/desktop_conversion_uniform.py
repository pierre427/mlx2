"""Original bounded NumPy affine Q2/Q4/Q8 artifact conversion.

Matrix layout is standard MLX affine: packed U32 weights, BF16 scales and
biases, group size 64. No model code or GPU is executed. Conversion is a
candidate artifact operation, never evidence that an adapter can serve it.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import struct
import tempfile
from pathlib import Path

import numpy as np


def _bf16(values):
    words = np.asarray(values, dtype="<f4").view("<u4")
    rounded = ((words + np.uint32(0x7fff) + ((words >> 16) & 1)) >> 16).astype("<u2")
    return rounded, (rounded.astype("<u4") << 16).view("<f4")


def quantize(values, bits):
    if bits not in (2, 4, 8):
        raise ValueError("Uniform conversion supports 2, 4 or 8 bits")
    values = np.asarray(values, dtype="<f4")
    if values.shape[-1] % 64 or not np.isfinite(values).all():
        raise ValueError("Affine weights need finite 64-aligned input columns")
    groups = values.reshape(*values.shape[:-1], -1, 64)
    low, high = groups.min(-1), groups.max(-1)
    biases, bias = _bf16(low)
    scales, scale = _bf16(np.maximum((high - low) / ((1 << bits) - 1), np.float32(1e-12)))
    codes = np.clip(np.rint((groups - bias[..., None]) / scale[..., None]), 0, (1 << bits) - 1).astype("<u4")
    codes = codes.reshape(*values.shape[:-1], -1, 32 // bits)
    shifts = np.arange(32 // bits, dtype="<u4") * bits
    packed = np.bitwise_or.reduce(codes << shifts, axis=-1)
    return packed.astype("<u4"), scales, biases


def _eligible(name, tensor):
    # Preserve routers, convolutions, norms and recurrent state parameters.
    # Model adapters decide whether each emitted affine module is supported.
    return (name.endswith(".weight") and tensor["dtype"] in {"BF16", "F16", "F32"}
            and (len(tensor["shape"]) == 2 or (len(tensor["shape"]) == 3 and re.search(r"(experts|switch_mlp)", name))) and tensor["shape"][-1] % 64 == 0
            and not re.search(r"(^|\.)(gate|router|norm|.*norm|conv\w*|ssm\w*|A_log|dt_bias|vision_model|vision_tower|visual|audio_model|audio_tower|video_model)(\.|$)", name))


def convert(source, destination, bits, update):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    sources = sorted(source.glob("*.safetensors"))
    weight_map, total_output, count = {}, 0, 0
    for shard_index, shard in enumerate(sources):
        with shard.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(header_size))
        output_header, offset = {"__metadata__": {"format": "mlx"}}, 0
        descriptor, name = tempfile.mkstemp(dir=destination, prefix=".tensor-payload-")
        try:
            with os.fdopen(descriptor, "wb") as payload:
                def write(key, dtype, shape, arrays, output_header=output_header, shard=shard):
                    nonlocal offset
                    start = offset
                    for array in arrays:
                        raw = memoryview(array).cast("B")
                        payload.write(raw)
                        offset += len(raw)
                    output_header[key] = {"dtype": dtype, "shape": shape, "data_offsets": [start, offset]}
                    weight_map[key] = shard.name
                for tensor_index, (key, tensor) in enumerate((k, v) for k, v in header.items() if k != "__metadata__"):
                    update("quantizing", 100 * (shard_index + tensor_index / max(1, len(header) - 1)) / len(sources), f"Shard {shard_index + 1}/{len(sources)} · {key}")
                    start, end = tensor["data_offsets"]
                    raw = np.memmap(shard, mode="r", dtype="u1", offset=8 + header_size + start, shape=(end - start,))
                    shape = tensor["shape"]
                    if _eligible(key, tensor):
                        dtype = {"BF16": "<u2", "F16": "<f2", "F32": "<f4"}[tensor["dtype"]]
                        matrix = raw.view(dtype).reshape(-1, shape[-1])
                        row_count = matrix.shape[0]
                        rows = max(1, (4 << 20) // max(1, shape[-1] * 4))
                        # Write each table separately, re-reading only bounded
                        # CPU chunks so a very large matrix never fills RAM.
                        for table, suffix, dtype_out, columns in ((0, "weight", "U32", shape[-1] * bits // 32), (1, "scales", "BF16", shape[-1] // 64), (2, "biases", "BF16", shape[-1] // 64)):
                            def chunks(row_count=row_count, rows=rows, shard_index=shard_index, tensor_index=tensor_index, table=table, header=header, key=key, matrix=matrix, tensor=tensor):
                                for row in range(0, row_count, rows):
                                    update("quantizing", 100 * (shard_index + (tensor_index + (table + row / max(1, row_count)) / 3) / max(1, len(header) - 1)) / len(sources), f"{key} · rows {row}/{row_count}")
                                    chunk = matrix[row:row + rows]
                                    values = (chunk.astype("<u4") << 16).view("<f4") if tensor["dtype"] == "BF16" else chunk.astype("<f4")
                                    yield quantize(values, bits)[table]
                            write(key.rsplit(".", 1)[0] + "." + suffix, dtype_out, [*shape[:-1], columns], chunks())
                        count += 1
                    else:
                        write(key, tensor["dtype"], shape, (raw[row:row + (8 << 20)] for row in range(0, raw.size, 8 << 20)))
                    del raw
                payload.flush()
                os.fsync(payload.fileno())
            encoded = json.dumps(output_header, separators=(",", ":")).encode()
            encoded += b" " * (-len(encoded) % 8)
            output = destination / shard.name
            with output.open("xb") as stream, Path(name).open("rb") as payload:
                stream.write(struct.pack("<Q", len(encoded)))
                stream.write(encoded)
                shutil.copyfileobj(payload, stream, 8 << 20)
                stream.flush()
                os.fsync(stream.fileno())
            total_output += output.stat().st_size
        finally:
            Path(name).unlink(missing_ok=True)
    if not count:
        raise ValueError("No compatible affine matrices found in this checkpoint")
    config = json.loads((source / "config.json").read_text())
    config["quantization"] = {"group_size": 64, "bits": bits, "mode": "affine"}
    config.pop("quantization_config", None)
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    index = {"metadata": {"total_size": total_output}, "weight_map": weight_map}
    (destination / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    names = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json", "vocab.json", "merges.txt", "chat_template.jinja", "preprocessor_config.json", "processor_config.json", "added_tokens.json", "tokenizer.model", "tekken.json", "README.md", "LICENSE", "LICENSE.txt", "NOTICE"}
    for name in names:
        path = source / name
        if path.is_file() and not path.is_symlink():
            shutil.copyfile(path, destination / name)
    update("completed", 100, f"Converted {count} affine matrices · unqualified candidate")

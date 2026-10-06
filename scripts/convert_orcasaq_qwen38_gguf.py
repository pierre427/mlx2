#!/usr/bin/env python3
"""Convert the pinned OrcaSAQ Qwen3.8 GGUF text target to indexed MLX weights.

This is a CPU-only expansion of the GGUF's quantized values to BF16; the BF16
rounding adds error beyond the source quantization (F32 tensors stay F32).
It does not invent an MLX quantization policy. GGUF Qwen RMSNorm gammas are
already +1 folded; MLX-shaped conv1d weights keep the mlx2 sanitizer from
folding those gammas a second time. The sanitizer casts residual-stream F32
norms to BF16 for compute. GDN A_log/dt_bias/linear_attn.norm remain F32 by
its explicit policy.
Each source tensor is streamed in bounded rows to one safetensors file, making
an interrupted conversion resumable without a whole-model resident copy.
The 866-file output is a compatibility-first artifact; load time, capacity,
and throughput have not been optimized or qualified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import struct
from pathlib import Path

SOURCE_REPO = "orcarouter/OrcaSAQ-2-Cyber-27B-Uncensored-GGUF"
SOURCE_REVISION = "a0ebe1b5ad5c009cd382908585c04b7e9e0cf0c0"
SOURCE_NAME = "OrcaSAQ-2-27B-Uncensored.gguf"
SOURCE_BYTES = 15_676_553_472
SOURCE_SHA256 = "4ab6bdf8a9008869abf1630fb92d04b5439805f96b56692962fa240b1d439977"
LAYOUT_SHA256 = "5b0f52ac89b9839d8c947e1f30f141c7272332a28d247d78853ab422d436dadd"
EXPECTED_TYPES = {"F32": 360, "IQ4_XS": 439, "Q5_K": 65, "Q6_K": 2}
TENSOR_COUNT = 866
SUPPORT_FILES = (
    "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
    "generation_config.json", "vocab.json", "merges.txt",
)
PREFIX = "language_model."
BLOCK = re.compile(r"^blk\.(\d+)\.(.+)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gguf_layout(reader) -> tuple[dict, str]:
    layout = {
        t.name: {"shape": [int(n) for n in reversed(t.shape)], "type": t.tensor_type.name}
        for t in reader.tensors
    }
    digest = hashlib.sha256(json.dumps(layout, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return layout, digest


def inspect_source(path: Path, *, verify_bytes: bool = False):
    import gguf

    if path.name != SOURCE_NAME or not path.is_file() or path.stat().st_size != SOURCE_BYTES:
        raise ValueError("source filename or size differs from pinned OrcaSAQ GGUF")
    reader = gguf.GGUFReader(path)
    layout, digest = gguf_layout(reader)
    if len(layout) != TENSOR_COUNT or digest != LAYOUT_SHA256:
        raise ValueError("GGUF tensor layout differs from pinned OrcaSAQ revision")
    from collections import Counter

    if dict(Counter(item["type"] for item in layout.values())) != EXPECTED_TYPES:
        raise ValueError("GGUF tensor encodings differ from pinned revision")
    expected_meta = {
        "general.architecture": "qwen35", "qwen35.block_count": 65,
        "qwen35.nextn_predict_layers": 1, "qwen35.embedding_length": 5120,
        "qwen35.attention.head_count": 24, "qwen35.attention.head_count_kv": 4,
        "qwen35.ssm.inner_size": 6144, "qwen35.ssm.state_size": 128,
        "qwen35.ssm.group_count": 16, "qwen35.ssm.time_step_rank": 48,
        "qwen35.full_attention_interval": 4, "qwen35.ssm.conv_kernel": 4,
        "tokenizer.ggml.eos_token_id": 248046,
        "tokenizer.ggml.bos_token_id": 248044,
    }
    for key, expected in expected_meta.items():
        field = reader.get_field(key)
        if field is None or field.contents() != expected:
            raise ValueError(f"GGUF metadata mismatch: {key}")
    mapping = {t.name: map_tensor(t.name, tuple(int(n) for n in reversed(t.shape))) for t in reader.tensors}
    targets = [entry[0] for entry in mapping.values()]
    if len(set(targets)) != TENSOR_COUNT:
        raise ValueError("GGUF name mapping collides")
    if verify_bytes and sha256(path) != SOURCE_SHA256:
        raise ValueError("GGUF SHA-256 differs from pinned source revision")
    return reader, layout, digest, mapping


def map_tensor(name: str, shape: tuple[int, ...]) -> tuple[str, tuple[int, ...], str]:
    """Return MLX key, stored shape, and inverse llama.cpp conversion operation."""
    top = {
        "token_embd.weight": ("model.embed_tokens.weight", (248320, 5120), "identity"),
        "output.weight": ("lm_head.weight", (248320, 5120), "identity"),
        "output_norm.weight": ("model.norm.weight", (5120,), "identity"),
    }
    if name in top:
        key, expected, operation = top[name]
    else:
        match = BLOCK.fullmatch(name)
        if match is None:
            raise ValueError(f"unknown GGUF tensor: {name}")
        layer, part = int(match[1]), match[2]
        if not 0 <= layer <= 64:
            raise ValueError(f"unknown Qwen block: {layer}")
        prefix = f"model.layers.{layer}." if layer < 64 else "mtp.layers.0."
        common = {
            "attn_norm.weight": ("input_layernorm.weight", (5120,), "identity"),
            "post_attention_norm.weight": ("post_attention_layernorm.weight", (5120,), "identity"),
            "ffn_down.weight": ("mlp.down_proj.weight", (5120, 17408), "identity"),
            "ffn_gate.weight": ("mlp.gate_proj.weight", (17408, 5120), "identity"),
            "ffn_up.weight": ("mlp.up_proj.weight", (17408, 5120), "identity"),
        }
        attn = {
            "attn_k.weight": ("self_attn.k_proj.weight", (1024, 5120), "identity"),
            "attn_k_norm.weight": ("self_attn.k_norm.weight", (256,), "identity"),
            "attn_output.weight": ("self_attn.o_proj.weight", (5120, 6144), "identity"),
            "attn_q.weight": ("self_attn.q_proj.weight", (12288, 5120), "identity"),
            "attn_q_norm.weight": ("self_attn.q_norm.weight", (256,), "identity"),
            "attn_v.weight": ("self_attn.v_proj.weight", (1024, 5120), "identity"),
        }
        gdn = {
            "attn_gate.weight": ("linear_attn.in_proj_z.weight", (6144, 5120), "v_rows_128"),
            "attn_qkv.weight": ("linear_attn.in_proj_qkv.weight", (10240, 5120), "qkv_rows"),
            "ssm_a": ("linear_attn.A_log", (48,), "a_log"),
            "ssm_alpha.weight": ("linear_attn.in_proj_a.weight", (48, 5120), "v_rows_1"),
            "ssm_beta.weight": ("linear_attn.in_proj_b.weight", (48, 5120), "v_rows_1"),
            "ssm_conv1d.weight": ("linear_attn.conv1d.weight", (10240, 4, 1), "conv_rows"),
            "ssm_dt.bias": ("linear_attn.dt_bias", (48,), "v_rows_1"),
            "ssm_norm.weight": ("linear_attn.norm.weight", (128,), "identity"),
            "ssm_out.weight": ("linear_attn.out_proj.weight", (5120, 6144), "v_columns_128"),
        }
        nextn = {
            "nextn.eh_proj.weight": ("mtp.fc.weight", (5120, 10240), "identity"),
            "nextn.enorm.weight": ("mtp.pre_fc_norm_embedding.weight", (5120,), "identity"),
            "nextn.hnorm.weight": ("mtp.pre_fc_norm_hidden.weight", (5120,), "identity"),
            "nextn.shared_head_norm.weight": ("mtp.norm.weight", (5120,), "identity"),
        }
        if part in common:
            suffix, expected, operation = common[part]
            key = prefix + suffix
        elif layer == 64 and part in nextn:
            key, expected, operation = nextn[part]
        elif (layer == 64 or (layer + 1) % 4 == 0) and part in attn:
            suffix, expected, operation = attn[part]
            key = prefix + suffix
        elif layer < 64 and (layer + 1) % 4 != 0 and part in gdn:
            suffix, expected, operation = gdn[part]
            key = prefix + suffix
        else:
            raise ValueError(f"tensor in wrong layer or unknown: {name}")
    source_shape = (10240, 4) if operation == "conv_rows" else expected
    if shape != source_shape:
        raise ValueError(f"unexpected GGUF shape for {name}: {shape}")
    return PREFIX + key, expected, operation


def _source_head(grouped_head: int) -> int:
    """Inverse value-head layout: grouped index -> GGUF tiled index."""
    return (grouped_head % 3) * 16 + grouped_head // 3


def source_row(row: int, operation: str) -> int:
    if operation == "qkv_rows" and row >= 4096:
        return 4096 + _source_head((row - 4096) // 128) * 128 + (row - 4096) % 128
    if operation == "conv_rows" and row >= 4096:
        return 4096 + _source_head((row - 4096) // 128) * 128 + (row - 4096) % 128
    if operation == "v_rows_128":
        return _source_head(row // 128) * 128 + row % 128
    if operation in ("v_rows_1", "a_log"):
        return _source_head(row)
    return row


def source_columns(width: int):
    import numpy as np

    if width != 6144:
        raise ValueError("unexpected GDN out projection input width")
    return np.fromiter((_source_head(i // 128) * 128 + i % 128 for i in range(width)), dtype=np.intp)


def _bf16_bytes(values):
    import numpy as np

    values = np.asarray(values, dtype="<f4")
    bits = values.view("<u4")
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return (rounded >> 16).astype("<u2").tobytes()


def write_tensor(tensor, target: Path, key: str, shape: tuple[int, ...], operation: str,
                 *, rows_per_chunk: int = 128) -> dict:
    import gguf
    import numpy as np

    if rows_per_chunk < 1:
        raise ValueError("rows_per_chunk must be positive")
    output_dtype = "F32" if tensor.tensor_type.name == "F32" else "BF16"
    item_bytes = 4 if output_dtype == "F32" else 2
    count = int(np.prod(shape))
    header = json.dumps({key: {"dtype": output_dtype, "shape": list(shape),
                               "data_offsets": [0, count * item_bytes]}}, separators=(",", ":")).encode()
    header += b" " * ((-len(header)) % 8)
    temporary = target.with_suffix(target.suffix + ".part")
    temporary.unlink(missing_ok=True)
    column_index = source_columns(shape[-1]) if operation == "v_columns_128" else None
    try:
        with temporary.open("wb") as stream:
            stream.write(struct.pack("<Q", len(header)))
            stream.write(header)
            rows = int(shape[0])
            for start in range(0, rows, rows_per_chunk):
                stop = min(start + rows_per_chunk, rows)
                indices = [source_row(row, operation) for row in range(start, stop)]
                encoded = tensor.data[indices]
                values = np.asarray(gguf.dequantize(encoded, tensor.tensor_type), dtype=np.float32)
                values = values.reshape((stop - start, -1))
                if column_index is not None:
                    values = values[:, column_index]
                if operation == "a_log":
                    if not np.all(values < 0):
                        raise ValueError("GGUF ssm_a must contain negative A")
                    values = np.log(-values)
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"nonfinite dequantized values: {tensor.name}")
                stream.write(values.astype("<f4", copy=False).tobytes() if output_dtype == "F32" else _bf16_bytes(values))
            stream.flush()
            os.fsync(stream.fileno())
        if temporary.stat().st_size != 8 + len(header) + count * item_bytes:
            raise ValueError(f"converted tensor length mismatch: {tensor.name}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {"size": target.stat().st_size, "sha256": sha256(target), "dtype": output_dtype,
            "source_type": tensor.tensor_type.name}


def _support_config(support: Path) -> dict:
    config = json.loads((support / "config.json").read_text())
    text = config.get("text_config", {})
    required = {"num_hidden_layers": 64, "hidden_size": 5120, "intermediate_size": 17408,
                "num_attention_heads": 24, "num_key_value_heads": 4, "vocab_size": 248320,
                "mtp_num_hidden_layers": 1, "full_attention_interval": 4,
                "linear_num_key_heads": 16, "linear_num_value_heads": 48,
                "linear_key_head_dim": 128, "linear_value_head_dim": 128}
    if config.get("model_type") != "qwen3_5" or any(text.get(k) != v for k, v in required.items()):
        raise ValueError("support config is not the dense Qwen3.8 27B MTP topology")
    if text.get("tie_word_embeddings") or config.get("tie_word_embeddings"):
        raise ValueError("GGUF has independent token embedding and output head")
    if text.get("layer_types") != ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(64)]:
        raise ValueError("support layer order differs from GGUF")
    config.pop("quantization", None)
    config.pop("quantization_config", None)
    text["dtype"] = "bfloat16"
    return config


def _check_support_tokenizer(reader, support: Path) -> dict:
    """Bind the copied tokenizer to every active GGUF token and merge rule."""
    tokenizer = json.loads((support / "tokenizer.json").read_text())
    vocab = tokenizer["model"]["vocab"]
    added = tokenizer.get("added_tokens", [])
    id_to_token = {int(index): token for token, index in vocab.items()}
    for item in added:
        index = int(item["id"])
        if index in id_to_token and id_to_token[index] != item["content"]:
            raise ValueError("support tokenizer has a conflicting token ID")
        id_to_token[index] = item["content"]
    if set(id_to_token) != set(range(248077)):
        raise ValueError("support tokenizer active ID range differs from GGUF")
    gguf_tokens = reader.get_field("tokenizer.ggml.tokens").contents()
    if len(gguf_tokens) != 248320:
        raise ValueError("GGUF token table length differs from embedding rows")
    for index, token in id_to_token.items():
        if gguf_tokens[index] != token:
            raise ValueError(f"support tokenizer differs from GGUF at ID {index}")
    gguf_merges = reader.get_field("tokenizer.ggml.merges").contents()
    if gguf_merges != tokenizer["model"]["merges"]:
        raise ValueError("support tokenizer merge order differs from GGUF")
    return {"active_token_ids": len(id_to_token), "gguf_table_entries": len(gguf_tokens),
            "gguf_padding_entries": len(gguf_tokens) - len(id_to_token),
            "matched_merges": len(gguf_merges)}


def _write_json_atomic(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)


def convert(source: Path, output: Path, support: Path, *, rows_per_chunk: int = 128,
            max_tensors: int | None = None) -> dict:
    reader, layout, layout_hash, mapping = inspect_source(source, verify_bytes=True)
    config = _support_config(support)
    for name in SUPPORT_FILES:
        if not (support / name).is_file():
            raise ValueError(f"missing support file: {name}")
    tokenizer_check = _check_support_tokenizer(reader, support)
    config_bytes = (json.dumps(config, sort_keys=True, indent=2) + "\n").encode()
    support_identity = {
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "files": {name: {"sha256": sha256(support / name), "size": (support / name).stat().st_size}
                  for name in SUPPORT_FILES},
    }
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "mlx2-gguf-conversion.json"
    if receipt_path.is_file():
        existing = json.loads(receipt_path.read_text())
        if existing.get("source_sha256") != SOURCE_SHA256 or existing.get("layout_sha256") != layout_hash or existing.get("tensor_count") != TENSOR_COUNT:
            raise ValueError("existing conversion receipt belongs to another source")
        if existing.get("config_sha256") != support_identity["config_sha256"]:
            raise ValueError("existing conversion receipt belongs to another support config")
        if set(existing.get("files", {})) != set(layout):
            raise ValueError("existing conversion receipt is incomplete")
        for source_name, record in existing["files"].items():
            key = mapping[source_name][0]
            filename = record.get("filename")
            if record.get("key") != key or not isinstance(filename, str) or Path(filename).name != filename:
                raise ValueError("existing conversion receipt has invalid shard mapping")
            shard = output / filename
            if not shard.is_file() or shard.stat().st_size != record["size"] or sha256(shard) != record["sha256"]:
                raise ValueError(f"existing converted shard changed: {filename}")
        for name, expected in existing.get("support_files", {}).items():
            if name not in SUPPORT_FILES or sha256(output / name) != expected["sha256"] or sha256(support / name) != expected["sha256"]:
                raise ValueError(f"existing support file changed: {name}")
        if set(existing.get("support_files", {})) != set(SUPPORT_FILES):
            raise ValueError("existing support file list incomplete")
        for name, field in (("config.json", "config_sha256"),
                            ("model.safetensors.index.json", "index_sha256")):
            if sha256(output / name) != existing.get(field):
                raise ValueError(f"existing converted {name} changed")
        return {"state": "complete-verified", "tensor_count": TENSOR_COUNT,
                "receipt": str(receipt_path)}
    progress_path = output / ".gguf-conversion-progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
        "source_sha256": SOURCE_SHA256, "layout_sha256": layout_hash,
        "support_identity": support_identity, "files": {}}
    if (progress.get("source_sha256") != SOURCE_SHA256 or progress.get("layout_sha256") != layout_hash
            or progress.get("support_identity") != support_identity):
        raise ValueError("conversion progress belongs to another source or support revision")
    tensors = sorted(reader.tensors, key=lambda t: t.name)
    for index, tensor in enumerate(tensors):
        if max_tensors is not None and index >= max_tensors:
            break
        key, shape, operation = mapping[tensor.name]
        filename = f"model-{index + 1:05d}-of-{TENSOR_COUNT:05d}.safetensors"
        path = output / filename
        record = progress["files"].get(tensor.name)
        if path.exists():
            if not record or record["filename"] != filename or path.stat().st_size != record["size"] or sha256(path) != record["sha256"]:
                raise ValueError(f"unverified existing output shard: {path}")
            continue
        if record is not None:
            raise ValueError(f"recorded output shard missing: {filename}")
        converted = write_tensor(tensor, path, key, shape, operation, rows_per_chunk=rows_per_chunk)
        progress["files"][tensor.name] = {"filename": filename, "key": key, **converted}
        _write_json_atomic(progress_path, progress)
        print(f"converted {index + 1}/{TENSOR_COUNT}: {tensor.name}", flush=True)
    if len(progress["files"]) != TENSOR_COUNT:
        return {"state": "partial", "completed": len(progress["files"]), "expected": TENSOR_COUNT,
                "progress": str(progress_path)}
    support_hashes = {}
    for name in SUPPORT_FILES:
        source_file, dest_file = support / name, output / name
        digest = sha256(source_file)
        if dest_file.exists():
            if sha256(dest_file) != digest:
                raise ValueError(f"support file differs: {name}")
        else:
            shutil.copyfile(source_file, dest_file)
        support_hashes[name] = {"sha256": digest, "size": dest_file.stat().st_size}
    _write_json_atomic(output / "config.json", config)
    weight_map = {record["key"]: record["filename"] for record in progress["files"].values()}
    total_size = sum((4 if record["dtype"] == "F32" else 2) *
                     math.prod(mapping[name][1]) for name, record in progress["files"].items())
    _write_json_atomic(output / "model.safetensors.index.json", {"metadata": {"total_size": total_size},
                                                                "weight_map": weight_map})
    receipt = {"schema": "mlx2.orcasaq-qwen38-gguf-conversion.v1", "source_repository": SOURCE_REPO,
               "source_revision": SOURCE_REVISION, "source_file": source.name,
               "source_sha256": SOURCE_SHA256, "source_size": SOURCE_BYTES,
               "layout_sha256": layout_hash, "source_types": EXPECTED_TYPES,
               "tensor_count": TENSOR_COUNT, "output_format": "indexed-safetensors-per-tensor",
               "quantized_output_dtype": "BF16", "float_output_dtype": "F32",
               "files": progress["files"], "support_files": support_hashes,
               "tokenizer_check": tokenizer_check,
               "config_sha256": sha256(output / "config.json"),
               "index_sha256": sha256(output / "model.safetensors.index.json"),
               "qualification": "unqualified"}
    _write_json_atomic(receipt_path, receipt)
    progress_path.unlink()
    return {"state": "complete", "tensor_count": TENSOR_COUNT,
            "receipt": str(receipt_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--support-model", type=Path)
    parser.add_argument("--rows-per-chunk", type=int, default=128)
    parser.add_argument("--inspect", action="store_true", help="Header-only pinned layout check; no payload SHA")
    parser.add_argument("--max-tensors", type=int, help="Stop after this many tensors for a bounded partial probe")
    args = parser.parse_args()
    if args.inspect:
        _reader, layout, digest, mapping = inspect_source(args.gguf)
        print(json.dumps({"state": "header-validated", "tensor_count": len(layout),
                          "layout_sha256": digest, "mapped_keys": len(mapping),
                          "source_bytes_sha256_verified": False}, indent=2))
    else:
        if args.output is None or args.support_model is None:
            parser.error("conversion requires --output and --support-model")
        print(json.dumps(convert(args.gguf, args.output, args.support_model,
                                 rows_per_chunk=args.rows_per_chunk, max_tensors=args.max_tensors), indent=2))


if __name__ == "__main__":
    main()

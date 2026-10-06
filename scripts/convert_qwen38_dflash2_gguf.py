"""Convert the pinned Qwen3.8 DFlash2 Q8_0 GGUF head for mlx2.

This is an offline, CPU-only conversion. It does not change the source GGUF or
claim that the converted head has passed generation parity or qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from mlx2.adapters.dflash2 import _expected_weight_shapes, inspect_drafter
from mlx2.runtime.drafters.dflash2_config import DFlash2Config

SOURCE_NAME = "Qwen3.8-27B-DFlash2-Q8_0.gguf"
SOURCE_SHA256 = "c18e800daedc59ca68fd13b6a856d795746af6d399a9279ac6a277d1d422f87e"
SOURCE_REVISION = "2d9571f8ce46e151f61c6499c99dee6079e1d610"
TARGET_REPOSITORY = "orcarouter/OrcaSAQ-2-Cyber-27B-Uncensored-GGUF"
TARGET_REVISION = "a0ebe1b5ad5c009cd382908585c04b7e9e0cf0c0"
TARGET_SHA256 = "4ab6bdf8a9008869abf1630fb92d04b5439805f96b56692962fa240b1d439977"
SHARD_LIMIT = 512 << 20
ROW_CHUNK = 64

_TOP = {
    "enc.output_norm.weight": "hidden_norm.weight",
    "output_norm.weight": "norm.weight",
    "fc.weight": "fc.weight",
    "selector_hidden.weight": "candidate_selector.hidden_projection.weight",
    "selector_predecessor.weight": "candidate_selector.predecessor_codebook",
    "selector_successor.weight": "candidate_selector.successor_codebook",
}
_LAYER = {
    "attn_conv_base": "attention_conv.base_kernel",
    "attn_conv_proj.weight": "attention_conv.kernel_projection.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_norm.weight": "input_layernorm.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "ffn_conv_base": "mlp_conv.base_kernel",
    "ffn_conv_proj.weight": "mlp_conv.kernel_projection.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
}


def mapped_name(name: str) -> str:
    if name in _TOP:
        return _TOP[name]
    parts = name.split(".", 2)
    if len(parts) != 3 or parts[0] != "blk" or not parts[1].isdigit():
        raise ValueError(f"unsupported DFlash2 GGUF tensor: {name}")
    index = int(parts[1])
    if not 0 <= index < 5 or parts[2] not in _LAYER:
        raise ValueError(f"unsupported DFlash2 GGUF tensor: {name}")
    return f"layers.{index}.{_LAYER[parts[2]]}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _field(reader, name: str):
    item = reader.get_field(name)
    return None if item is None else item.contents()


def validate_metadata(reader, args: DFlash2Config) -> None:
    expected = {
        "dflash.block_count": args.num_hidden_layers,
        "dflash.context_length": args.max_position_embeddings,
        "dflash.embedding_length": args.hidden_size,
        "dflash.feed_forward_length": args.intermediate_size,
        "dflash.attention.head_count": args.num_attention_heads,
        "dflash.attention.head_count_kv": args.num_key_value_heads,
        "dflash.attention.causal": args.is_causal,
        "dflash.attention.key_length": args.head_dim,
        "dflash.attention.value_length": args.head_dim,
        "dflash.block_size": args.block_size,
        "dflash.conv_kernel_size": args.conv_kernel_size,
        "dflash.conv_group_size": args.conv_group_size,
        "dflash.selector_rank": args.selector_rank,
        "dflash.selector_top_k": args.selector_top_k,
        "dflash.target_layers": [index + 1 for index in args.target_layer_ids],
        "dflash.attention.sliding_window": args.sliding_window,
        "dflash.attention.sliding_window_pattern": [True] * args.num_hidden_layers,
        "tokenizer.ggml.mask_token_id": args.mask_token_id,
    }
    for key, value in expected.items():
        observed = _field(reader, key)
        if observed != value:
            raise ValueError(f"DFlash2 GGUF/template metadata mismatch: {key}: {observed!r} != {value!r}")
    for key, value in {
        "dflash.rope.freq_base": args.rope_theta,
        "dflash.attention.layer_norm_rms_epsilon": args.rms_norm_eps,
    }.items():
        observed = _field(reader, key)
        if not isinstance(observed, (float, int)) or not math.isclose(
            observed, value, rel_tol=1e-6
        ):
            raise ValueError(f"DFlash2 GGUF/template metadata mismatch: {key}")
    if _field(reader, "tokenizer.ggml.pre") != "qwen35":
        raise ValueError("DFlash2 GGUF tokenizer dialect is not qwen35")
    if len(_field(reader, "tokenizer.ggml.tokens") or []) != args.vocab_size:
        raise ValueError("DFlash2 GGUF vocabulary size differs from template")


def inventory(reader, expected: dict[str, list[int]]) -> list[tuple[object, str]]:
    """Fail closed on the exact GGUF shape, topology, type, and name dialect."""
    if _field(reader, "general.architecture") != "dflash":
        raise ValueError("expected dflash GGUF architecture")
    if _field(reader, "general.quantization_version") != 2:
        raise ValueError("expected GGUF quantization version 2")
    converted = []
    seen = set()
    for tensor in reader.tensors:
        output_name = mapped_name(tensor.name)
        if output_name in seen:
            raise ValueError(f"duplicate mapped tensor: {output_name}")
        seen.add(output_name)
        shape = [int(dimension) for dimension in reversed(tensor.shape)]
        if expected.get(output_name) != shape:
            raise ValueError(f"shape mismatch for {tensor.name}: {shape}")
        if tensor.tensor_type.name not in {"F32", "Q8_0"}:
            raise ValueError(f"unsupported GGUF type for {tensor.name}: {tensor.tensor_type.name}")
        expected_type = "Q8_0" if len(shape) == 2 else "F32"
        if tensor.tensor_type.name != expected_type:
            raise ValueError(f"unexpected GGUF type for {tensor.name}: {tensor.tensor_type.name}")
        converted.append((tensor, output_name))
    if seen != set(expected):
        raise ValueError(
            f"DFlash2 tensor coverage mismatch: missing={sorted(set(expected)-seen)}, "
            f"extra={sorted(seen-set(expected))}"
        )
    return converted


def convert(source_dir: Path, template_dir: Path, target_dir: Path, output_dir: Path) -> dict:
    import gguf
    import numpy as np
    import torch
    from safetensors.torch import save_file

    source_dir = source_dir.resolve()
    template_dir = template_dir.resolve()
    target_dir = target_dir.resolve()
    output_dir = output_dir.resolve()
    manifest_path = source_dir / "download-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    record = manifest.get("dflash2_drafter", {})
    if (record.get("revision"), record.get("file"), record.get("sha256")) != (
        SOURCE_REVISION, SOURCE_NAME, SOURCE_SHA256
    ):
        raise ValueError("source manifest differs from pinned DFlash2 revision")
    target_record = manifest.get("target", {})
    if tuple(target_record.get(key) for key in ("repository", "revision", "sha256")) != (
        TARGET_REPOSITORY, TARGET_REVISION, TARGET_SHA256
    ):
        raise ValueError("source manifest differs from pinned OrcaSAQ target identity")
    source = source_dir / SOURCE_NAME
    if source.stat().st_size != record.get("size_bytes") or sha256_file(source) != SOURCE_SHA256:
        raise ValueError("DFlash2 GGUF source bytes differ from pinned manifest")
    target_receipt_path = target_dir / "mlx2-gguf-conversion.json"
    if not target_receipt_path.is_file():
        raise ValueError("target must be the receipt-bound OrcaSAQ GGUF conversion")
    target_receipt = json.loads(target_receipt_path.read_text())
    if (
        target_receipt.get("source_sha256") != TARGET_SHA256
        or target_receipt.get("source_revision") != TARGET_REVISION
        or target_receipt.get("config_sha256") != sha256_file(target_dir / "config.json")
        or target_receipt.get("index_sha256") != sha256_file(target_dir / "model.safetensors.index.json")
    ):
        raise ValueError("converted GGUF target identity differs from pinned source")
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    config_bytes = (template_dir / "config.json").read_bytes()
    config = json.loads(config_bytes)
    if config.get("architectures") != ["DFlash2DraftModel"] or config.get("dtype") != "bfloat16":
        raise ValueError("expected BF16 DFlash2 MLX template")
    args = DFlash2Config.from_dict(config)
    if (args.num_hidden_layers, args.hidden_size, args.vocab_size) != (5, 5120, 248320):
        raise ValueError("template topology differs from Qwen3.8 DFlash2")
    expected = _expected_weight_shapes(args)
    reader = gguf.GGUFReader(source)
    validate_metadata(reader, args)
    tensors = inventory(reader, expected)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.partial-", dir=output_dir.parent))
    try:
        (staging / "config.json").write_bytes(config_bytes)
        pending = {}
        pending_size = 0
        shard_index = 0
        weight_map = {}
        hashes = {}

        def flush() -> None:
            nonlocal pending, pending_size, shard_index
            if not pending:
                return
            shard_index += 1
            name = f"model-{shard_index:05d}.safetensors"
            path = staging / name
            save_file(pending, str(path), metadata={"format": "pt"})
            hashes[name] = sha256_file(path)
            for tensor_name in pending:
                weight_map[tensor_name] = name
            pending = {}
            pending_size = 0

        for tensor, output_name in tensors:
            shape = tuple(expected[output_name])
            if tensor.tensor_type.name == "Q8_0":
                item = torch.empty(shape, dtype=torch.bfloat16)
                for start in range(0, shape[0], ROW_CHUNK):
                    end = min(start + ROW_CHUNK, shape[0])
                    values = gguf.dequantize(tensor.data[start:end], tensor.tensor_type)
                    if values.size != (end - start) * shape[1]:
                        raise ValueError(f"dequantized row size mismatch: {tensor.name}")
                    # The GGUF header dimensions are reversed, but its flat
                    # payload is already [out, in]. Never transpose it.
                    array = np.array(values.reshape(end - start, shape[1]), copy=True, order="C")
                    item[start:end] = torch.from_numpy(array).to(torch.bfloat16)
                    del values, array
            else:
                values = gguf.dequantize(tensor.data, tensor.tensor_type)
                if values.size != math.prod(shape):
                    raise ValueError(f"dequantized size mismatch: {tensor.name}")
                array = np.array(values.reshape(shape), copy=True, order="C")
                item = torch.from_numpy(array).to(torch.bfloat16).contiguous()
                del values, array
            if pending and pending_size + item.numel() * item.element_size() > SHARD_LIMIT:
                flush()
            pending[output_name] = item
            pending_size += item.numel() * item.element_size()
            del item
        flush()
        index = {
            "metadata": {"total_size": sum(2 * math.prod(shape) for shape in expected.values())},
            "weight_map": weight_map,
        }
        (staging / "model.safetensors.index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
        receipt = {
            "schema": "mlx2.qwen38-dflash2-gguf-conversion.v1",
            "source_repository": record["repository"],
            "source_revision": SOURCE_REVISION,
            "source_file": SOURCE_NAME,
            "source_sha256": SOURCE_SHA256,
            "source_target_repository": target_record["repository"],
            "source_target_revision": target_record["revision"],
            "source_target_sha256": target_record["sha256"],
            "source_tensor_types": dict(Counter(t.tensor_type.name for t, _ in tensors)),
            "source_tokenizer_eos_token_id": _field(reader, "tokenizer.ggml.eos_token_id"),
            "template_eos_token_id": config.get("eos_token_id"),
            "template_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "inspection_target_config_sha256": sha256_file(target_dir / "config.json"),
            "inspection_target_receipt_sha256": sha256_file(target_receipt_path),
            "output_dtype": "BF16",
            "output_shard_sha256": hashes,
            "output_tensor_count": len(weight_map),
            "parity": "unverified",
            "qualification": "unqualified",
        }
        (staging / "conversion-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        inspected = inspect_drafter(staging, target_dir)
        if len(inspected["header_sha256"]) != len(hashes):
            raise ValueError("mlx2 draft inspection did not cover all output shards")
        os.replace(staging, output_dir)
        return receipt
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--template-dir", required=True, type=Path)
    parser.add_argument("--target-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(convert(args.source_dir, args.template_dir, args.target_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()

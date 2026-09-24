#!/usr/bin/env python3
"""Convert the Qwen-Image-2.1 Q4_K_M GGUF transformer to MLX safetensors.

The conversion is deliberately CPU-only. It expands GGUF Q4_K/Q6_K blocks to
BF16, so the result preserves the source quantization error but is larger on
disk. No model is instantiated and no MLX device is touched.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
from pathlib import Path

SOURCE_REPO = "abenzerps/Qwen-Image-2.1-Uncensored-GGUF"
SOURCE_REVISION = "40319fb15542f0ad22921e0124a191a8a935a60a"
SOURCE_SHA256 = "e79c8a009f2ecbdb6c70fd663d9aea9ee304a0d91f347e4169a756b8ad141b41"
BASE_REPO = "Qwen/Qwen-Image-2.1"
BASE_REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
EXPECTED_TENSOR_COUNT = 297
# SHA-256 of sorted JSON {original tensor name: original shape}, checked against
# both the GGUF header and the two official safetensors headers at these commits.
EXPECTED_LAYOUT_SHA256 = "12842fba2c952af27a9297d2f61dddb48c6a07eb729a84860bb18f4e5a52f6b9"


def _layout(reader) -> dict[str, list[int]]:
    # GGUF stores dimensions innermost first; safetensors uses row-major order.
    return {tensor.name: list(map(int, reversed(tensor.shape))) for tensor in reader.tensors}


def _key(name: str) -> str:
    return name.replace(
        "time_text_embed.timestep_embedder.", "time_text_embed."
    ).replace("modulation.1.", "modulation.0.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_base_components(base: Path, output: Path) -> None:
    marker = base / ".hf-download-complete.json"
    if not marker.is_file():
        raise ValueError("base Qwen-Image-2.1 snapshot is not verified complete")
    proof = json.loads(marker.read_text())
    if proof.get("repo") != BASE_REPO or proof.get("revision") != BASE_REVISION:
        raise ValueError("base snapshot identity differs from the pinned revision")
    for dirname in ("processor", "scheduler", "text_encoder", "vae"):
        source = base / dirname
        if not source.is_dir():
            raise ValueError(f"base snapshot lacks {dirname}")
        for item in source.rglob("*"):
            if not item.is_file():
                continue
            target = output / item.relative_to(base)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.stat().st_size != item.stat().st_size:
                    raise ValueError(f"existing component differs: {target}")
                continue
            try:
                os.link(item, target)
            except OSError:
                shutil.copy2(item, target)
    shutil.copy2(base / "model_index.json", output / "model_index.json")


def link_base(base: Path, output: Path) -> dict:
    """Complete an already-converted transformer once the base snapshot lands."""
    receipt_path = output / "mlx2-conversion.json"
    if not receipt_path.is_file():
        raise ValueError("transformer conversion receipt is missing")
    proof = json.loads(receipt_path.read_text())
    if proof.get("source_revision") != SOURCE_REVISION or proof.get("tensor_count") != EXPECTED_TENSOR_COUNT:
        raise ValueError("transformer conversion identity differs")
    for name, record in proof["output_files"].items():
        candidate = output / "transformer" / name
        if candidate.stat().st_size != record["size"] or _sha256(candidate) != record["sha256"]:
            raise ValueError(f"converted shard changed: {name}")
    _copy_base_components(base, output)
    proof["base_repo"] = BASE_REPO
    proof["base_revision"] = BASE_REVISION
    receipt_path.write_text(json.dumps(proof, indent=2) + "\n")
    return proof


def convert(gguf_path: Path, output: Path, *, base: Path | None = None) -> dict:
    import gguf
    import numpy as np
    import torch
    from safetensors.torch import save_file

    if gguf_path.suffix != ".gguf" or not gguf_path.is_file():
        raise ValueError("input must be a completed .gguf file")
    reader = gguf.GGUFReader(gguf_path)
    layout = _layout(reader)
    layout_hash = hashlib.sha256(
        json.dumps(layout, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if len(layout) != EXPECTED_TENSOR_COUNT or layout_hash != EXPECTED_LAYOUT_SHA256:
        raise ValueError("GGUF tensor names/shapes differ from pinned Qwen-Image-2.1")
    if len({_key(name) for name in layout}) != len(layout):
        raise ValueError("MLX key remapping collides")

    source_hash = _sha256(gguf_path)
    if source_hash != SOURCE_SHA256:
        raise ValueError("GGUF SHA-256 differs from the pinned Q4_K_M artifact")
    output.mkdir(parents=True, exist_ok=True)
    completed = output / "mlx2-conversion.json"
    if completed.is_file():
        prior = json.loads(completed.read_text())
        if prior.get("source_sha256") != source_hash or prior.get("layout_sha256") != layout_hash:
            raise ValueError("existing conversion is bound to another GGUF")
        for name, record in prior["output_files"].items():
            item = output / "transformer" / name
            if item.stat().st_size != record["size"] or _sha256(item) != record["sha256"]:
                raise ValueError(f"existing converted shard changed: {name}")
        return link_base(base, output) if base is not None and prior.get("base_revision") is None else prior
    transformer = output / "transformer"
    transformer.mkdir(exist_ok=True)
    config = {
        "_class_name": "QwenImage21Transformer2DModel",
        "attention_head_dim": 128,
        "axes_dims_rope": [16, 56, 56],
        "context_in_dim": 4096,
        "in_channels": 64,
        "num_attention_heads": 32,
        "num_layers": 32,
        "out_channels": 64,
        "patch_size": 1,
        "mlp_ratio": 3,
        "eps": 1e-6,
        "causal_condition": True,
        "mlx_format": True,
    }
    (transformer / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    progress_path = output / ".mlx2-conversion-progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
        "source_sha256": source_hash,
        "layout_sha256": layout_hash,
        "output_files": {},
    }
    if progress["source_sha256"] != source_hash or progress["layout_sha256"] != layout_hash:
        raise ValueError("existing conversion progress is bound to another GGUF")

    groups: dict[str, list] = {}
    for tensor in reader.tensors:
        parts = tensor.name.split(".")
        group = f"block-{int(parts[1]):02d}" if parts[:1] == ["transformer_blocks"] else "global"
        groups.setdefault(group, []).append(tensor)
    output_files: dict[str, dict] = {}
    for group, tensors in groups.items():
        target = transformer / f"{group}.safetensors"
        if target.exists():
            record = progress["output_files"].get(target.name)
            if not record or target.stat().st_size != record["size"] or _sha256(target) != record["sha256"]:
                raise ValueError(f"unverified existing converted shard: {target}")
            output_files[target.name] = record
            continue
        arrays = {}
        for tensor in tensors:
            values = gguf.dequantize(tensor.data, tensor.tensor_type)
            if int(np.prod(tensor.shape)) != values.size:
                raise ValueError(f"dequantized shape mismatch: {tensor.name}")
            values = np.asarray(values).reshape(tuple(reversed(tensor.shape)))
            arrays[_key(tensor.name)] = torch.from_numpy(values.copy()).to(torch.bfloat16)
            del values
        temporary = target.with_suffix(".safetensors.part")
        temporary.unlink(missing_ok=True)
        save_file(arrays, str(temporary), metadata={"format": "pt"})
        os.replace(temporary, target)
        output_files[target.name] = {"size": target.stat().st_size, "sha256": _sha256(target)}
        progress["output_files"][target.name] = output_files[target.name]
        progress_path.write_text(json.dumps(progress, indent=2) + "\n")
        print(f"converted {group}: {len(arrays)} tensors", flush=True)
        del arrays
        gc.collect()
    del reader
    if base is not None:
        _copy_base_components(base, output)
    proof = {
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "source_file": gguf_path.name,
        "source_sha256": source_hash,
        "base_repo": BASE_REPO if base is not None else None,
        "base_revision": BASE_REVISION if base is not None else None,
        "tensor_count": EXPECTED_TENSOR_COUNT,
        "layout_sha256": layout_hash,
        "output_dtype": "bfloat16",
        "output_files": output_files,
        "execution_qualification": "pending",
    }
    (output / "mlx2-conversion.json").write_text(json.dumps(proof, indent=2) + "\n")
    progress_path.unlink(missing_ok=True)
    return proof


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--link-base-only", action="store_true")
    args = parser.parse_args()
    if args.link_base_only:
        if args.base_model is None or args.gguf is not None:
            parser.error("--link-base-only requires --base-model and no --gguf")
        result = link_base(args.base_model, args.output)
    else:
        if args.gguf is None:
            parser.error("--gguf is required for conversion")
        result = convert(args.gguf, args.output, base=args.base_model)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert pinned Recursive Flint BF16 weights to an MLX mixed-Q4 candidate.

The ``oq4e`` profile is deliberately fidelity-biased: routed expert matrices
use refined affine Q4/group-64; dense and shared MLPs use Q5; attention,
embeddings and the untied output head use Q6. Routers, norms and learned sink
logits stay BF16. This is stored-weight-MSE optimization, not activation-aware
calibration, and the receipt says so. A later calibrated image must use a new
profile/revision rather than relabel this artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlx2.adapters.flint import (
    SOURCE_INDEX_SHA256,
    SOURCE_REPOSITORY,
    SOURCE_REVISION,
    inspect_artifact,
)
from mlx2.conversion.affine import quantize_rows

GROUP_SIZE = 64
PROFILE = "flint-oq4e-weight-mse-v1"
SUPPORT_FILES = (
    ".gitattributes",
    "LICENSE",
    "README.md",
    "chat_template.jinja",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def sha256(path: Path, chunk_bytes: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _source_manifest(source: Path) -> tuple[dict, dict]:
    tree = source / ".cache" / "huggingface" / "trees" / f"{SOURCE_REVISION}.json"
    if not tree.is_file():
        raise ValueError("Flint source has no pinned Hugging Face tree receipt")
    manifest = json.loads(tree.read_text())
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if manifest.get("format_version") != 1 or not isinstance(files, dict):
        raise ValueError("Flint Hugging Face tree receipt is malformed")
    if any((source / ".cache" / "huggingface" / "download").rglob("*.incomplete")):
        raise ValueError("Flint source download is still incomplete")
    for name, record in files.items():
        path = source / name
        if not path.is_file() or path.stat().st_size != record.get("size"):
            raise ValueError(
                f"Flint source file is absent or has the wrong size: {name}"
            )
    if sha256(source / "model.safetensors.index.json") != SOURCE_INDEX_SHA256:
        raise ValueError("Flint source index differs from the pinned release")
    receipt_path = source / ".mlx2-hf-download.json"
    if not receipt_path.is_file():
        raise ValueError("Flint source has no verified download receipt")
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("complete") is not True
        or receipt.get("repository") != SOURCE_REPOSITORY
        or receipt.get("revision") != SOURCE_REVISION
        or receipt.get("file_count") != len(files)
        or receipt.get("bytes") != sum(record["size"] for record in files.values())
    ):
        raise ValueError("Flint source download receipt differs from the pinned tree")
    verified = {record.get("path"): record for record in receipt.get("files", ())}
    if set(verified) != set(files):
        raise ValueError("Flint source download receipt has different file closure")
    for name, record in files.items():
        field = "lfs_sha256" if record.get("lfs_sha256") else "git_blob_sha1"
        wanted = record.get("lfs_sha256") or record.get("blob_id")
        if verified[name].get("bytes") != record["size"] or verified[name].get(field) != wanted:
            raise ValueError(f"Flint source receipt differs for {name}")
    return manifest, receipt


def _module_name(key: str) -> str | None:
    return key.removesuffix(".weight") if key.endswith(".weight") else None


def _policy(key: str, value) -> tuple[int | None, str]:
    """Return bit width and receipt category for one source tensor."""
    module = _module_name(key)
    if module is None or value.ndim != 2 or value.shape[-1] % GROUP_SIZE:
        return None, "protected_bf16"
    if ".mlp.experts." in key:
        return 4, "routed_experts_q4"
    if key in {"model.embed_tokens.weight", "lm_head.weight"}:
        return 6, "embedding_and_head_q6"
    if ".self_attn." in key and key.endswith("_proj.weight"):
        return 6, "attention_q6"
    if ".mlp.shared_expert." in key:
        return 5, "shared_expert_q5"
    if key.startswith(("model.layers.0.mlp.", "model.layers.1.mlp.")):
        return 5, "dense_prefix_q5"
    # Router gates make the discrete top-k choice and remain BF16. Norms and
    # learned sink logits are one-dimensional and were already protected.
    return None, "protected_bf16"


def _runtime_module(source_module: str) -> str:
    if ".mlp.experts." not in source_module:
        return source_module
    prefix, tail = source_module.split(".mlp.experts.", 1)
    _expert, projection = tail.split(".", 1)
    return f"{prefix}.mlp.switch_mlp.{projection}"


def _quant_config(index: dict) -> dict:
    quant = {"group_size": GROUP_SIZE, "bits": 4, "mode": "affine"}
    for key in index:
        if not key.endswith(".weight"):
            continue
        # Shapes decide whether a tensor is quantized; policy categories are
        # repeated during conversion. Overrides here bind every non-Q4 module
        # in the runtime tree after expert packing.
        if (
            key in {"model.embed_tokens.weight", "lm_head.weight"}
            or ".self_attn." in key
            and key.endswith("_proj.weight")
        ):
            quant[_runtime_module(key.removesuffix(".weight"))] = {
                "group_size": GROUP_SIZE,
                "bits": 6,
                "mode": "affine",
            }
        elif ".mlp.shared_expert." in key or key.startswith(
            ("model.layers.0.mlp.", "model.layers.1.mlp.")
        ):
            quant[_runtime_module(key.removesuffix(".weight"))] = {
                "group_size": GROUP_SIZE,
                "bits": 5,
                "mode": "affine",
            }
    return quant


def convert(
    source: Path, destination: Path, *, reserve_bytes: int = 80_000_000_000
) -> dict:
    source = source.expanduser().resolve(strict=True)
    destination = destination.expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if not isinstance(reserve_bytes, int) or reserve_bytes < 0:
        raise ValueError("reserve_bytes must be a nonnegative integer")
    manifest, download_receipt = _source_manifest(source)
    inspected = inspect_artifact(source)
    if download_receipt.get("artifact_fingerprint") != inspected["identity"]["fingerprint"]:
        raise ValueError("Flint source identity changed after download verification")
    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "flint" or config.get("quantization") is not None:
        raise ValueError("source is not the pinned BF16 Flint checkpoint")
    index = json.loads((source / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    if not isinstance(index, dict) or not index:
        raise ValueError("Flint source index is empty")
    if shutil.disk_usage(destination.parent).free < 25_000_000_000 + reserve_bytes:
        raise OSError("Flint conversion would cross the requested free-space reserve")

    staging = destination.with_name(
        destination.name + ".incomplete-" + uuid.uuid4().hex
    )
    staging.mkdir(parents=True)
    download_receipt = {
        **download_receipt,
        "conversion_started": True,
        "conversion_profile": PROFILE,
        "conversion_destination": str(destination),
    }
    _write_json_atomic(source / ".mlx2-hf-download.json", download_receipt)
    mx.set_default_device(mx.cpu)
    output_map = {}
    outputs = []
    errors = defaultdict(
        lambda: {"error_before": 0.0, "error_after": 0.0, "elements": 0}
    )
    bytes_by_category = defaultdict(int)
    started = time.time()
    source_shards = sorted(set(index.values()))
    try:
        for number, filename in enumerate(source_shards, 1):
            if shutil.disk_usage(destination.parent).free < reserve_bytes:
                raise OSError("Flint conversion crossed the free-space reserve")
            source_path = source / filename
            source_record = manifest["files"].get(filename, {})
            tensors = mx.load(str(source_path))
            converted = {}
            for key in sorted(tensors):
                value = tensors[key]
                bits, category = _policy(key, value)
                if bits is None:
                    converted[key] = value
                    bytes_by_category[category] += value.nbytes
                    output_map[key] = filename
                    continue
                values = np.asarray(value.astype(mx.float32))
                packed, scales, biases, error = quantize_rows(
                    values, bits=bits, group_size=GROUP_SIZE
                )
                base = key.removesuffix(".weight")
                converted[key] = mx.array(packed)
                converted[base + ".scales"] = mx.array(
                    (scales.astype(np.uint32) << 16).view(np.float32)
                ).astype(mx.bfloat16)
                converted[base + ".biases"] = mx.array(
                    (biases.astype(np.uint32) << 16).view(np.float32)
                ).astype(mx.bfloat16)
                for output_key in (key, base + ".scales", base + ".biases"):
                    output_map[output_key] = filename
                    bytes_by_category[category] += converted[output_key].nbytes
                for field in errors[category]:
                    errors[category][field] += error[field]
                del values
            target = staging / filename
            # MLX appends ``.safetensors`` when the supplied name does not
            # already carry that suffix.  Keep it last on the temporary name
            # so the atomic rename refers to the file MLX actually wrote.
            temporary = target.with_name(target.stem + ".part.safetensors")
            mx.eval(list(converted.values()))
            mx.save_safetensors(str(temporary), converted, metadata={"format": "mlx"})
            os.replace(temporary, target)
            outputs.append(
                {
                    "file": filename,
                    "bytes": target.stat().st_size,
                    "sha256": sha256(target),
                    "source_lfs_sha256": source_record.get("lfs_sha256"),
                }
            )
            tensors.clear()
            converted.clear()
            mx.clear_cache()
            _write_json_atomic(
                staging / "progress.json",
                {
                    "state": "converting",
                    "source_shards_done": number,
                    "source_shards_total": len(source_shards),
                    "elapsed_seconds": time.time() - started,
                },
            )

        output_config = json.loads(json.dumps(config))
        output_config["quantization"] = _quant_config(index)
        output_config["quantization_config"] = output_config["quantization"]
        output_config["mlx2_conversion"] = {
            "profile": PROFILE,
            "source_repository": SOURCE_REPOSITORY,
            "source_revision": SOURCE_REVISION,
            "activation_aware": False,
            "weight_fit": "stored-BF16 MSE clipping and alternating least squares",
            "qualification": "unqualified",
        }
        total_size = sum(tensor_bytes for tensor_bytes in bytes_by_category.values())
        _write_json_atomic(staging / "config.json", output_config)
        _write_json_atomic(
            staging / "model.safetensors.index.json",
            {"metadata": {"total_size": total_size}, "weight_map": output_map},
        )
        for filename in SUPPORT_FILES:
            path = source / filename
            if path.is_file() and filename not in {
                "config.json",
                "model.safetensors.index.json",
            }:
                target = staging / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        receipt = {
            "schema": "mlx2.flint-oq4e-conversion.v1",
            "profile": PROFILE,
            "source_repository": SOURCE_REPOSITORY,
            "source_revision": SOURCE_REVISION,
            "source_index_sha256": SOURCE_INDEX_SHA256,
            "source_artifact_fingerprint": inspected["identity"]["fingerprint"],
            "source_shards": [
                {
                    "file": name,
                    "bytes": manifest["files"][name]["size"],
                    "lfs_sha256": manifest["files"][name].get("lfs_sha256"),
                }
                for name in source_shards
            ],
            "output_shards": outputs,
            "quantization": output_config["quantization"],
            "bytes_by_category": dict(bytes_by_category),
            "tensor_bytes": total_size,
            "fit_error_by_category": dict(errors),
            "activation_aware": False,
            "calibration": None,
            "calibration_boundary": (
                "weight-MSE candidate only; do not call activation-aware oQ4e or promote "
                "without source-bound representative calibration and exact-artifact validation"
            ),
            "implemented": True,
            "qualified": False,
            "selected": False,
            "observed_used": False,
            "elapsed_seconds": time.time() - started,
        }
        _write_json_atomic(staging / "mlx2-conversion.json", receipt)
        (staging / "progress.json").unlink(missing_ok=True)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        os.rename(staging, destination)
        return receipt
    except Exception:  # noqa: TRY203 - staging is intentionally retained verbatim
        # Preserve the nonempty staging directory and progress receipt for a
        # distinguishable, inspectable retry; never overwrite it implicitly.
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--reserve-gib", type=float, default=80.0)
    args = parser.parse_args()
    if args.reserve_gib < 0:
        parser.error("--reserve-gib must be nonnegative")
    receipt = convert(
        args.source,
        args.destination,
        reserve_bytes=int(args.reserve_gib * (1 << 30)),
    )
    print(
        json.dumps({"destination": str(args.destination), "receipt": receipt}, indent=2)
    )


if __name__ == "__main__":
    main()

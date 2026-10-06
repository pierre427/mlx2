#!/usr/bin/env python3
"""CPU-only affine repack of the pinned OrcaSAQ GGUF-to-MLX expansion.

This adds a second, lossy quantization step after GGUF dequantization and BF16
rounding. The default profile favors fidelity over minimum disk size:
IQ4_XS -> MLX 5-bit, Q5_K -> 6-bit, Q6_K -> 8-bit, group 64 affine.
F32 source tensors are copied unchanged. Output remains experimental and
unqualified until strict load, output parity, and controlled performance runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from convert_orcasaq_qwen38_gguf import (
    SOURCE_SHA256,
    SUPPORT_FILES,
    TENSOR_COUNT,
    sha256,
)

DEFAULT_PROFILE = {"IQ4_XS": 5, "Q5_K": 6, "Q6_K": 8}
GROUP_SIZE = 64
NOTICE = (
    "Experimental, unqualified repack. GGUF dequantization, BF16 rounding, and "
    "this MLX affine quantization are distinct transformations. The second "
    "quantization adds error; numerical parity and speed are unmeasured."
)


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    if temporary.is_symlink():
        raise ValueError(f"temporary output must not be a symlink: {temporary.name}")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)


def _resolved_roots(source: Path, output: Path) -> tuple[Path, Path]:
    """Return disjoint physical roots without accepting symlink aliases."""
    source = source.expanduser()
    output = output.expanduser()
    if source.is_symlink() or output.is_symlink():
        raise ValueError("source and output roots must not be symlinks")
    try:
        source = source.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError("source root does not exist") from error
    if not source.is_dir():
        raise ValueError("source root is not a directory")
    output = output.resolve(strict=False)
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("source and output roots must be disjoint")
    return source, output


def _flat_child(root: Path, filename: object, *, suffix: str | None = None) -> Path:
    """Resolve a receipt filename without allowing traversal or symlink escape."""
    if not isinstance(filename, str) or not filename or filename in {".", ".."}:
        raise ValueError("receipt filename must be a non-empty plain filename")
    relative = Path(filename)
    if relative.is_absolute() or relative.name != filename or len(relative.parts) != 1:
        raise ValueError(f"receipt filename is not flat: {filename!r}")
    if suffix is not None and relative.suffix != suffix:
        raise ValueError(f"receipt filename has an unexpected suffix: {filename!r}")
    candidate = root / relative
    if candidate.is_symlink():
        raise ValueError(f"receipt path must not be a symlink: {filename!r}")
    try:
        resolved = candidate.resolve(strict=False)
        resolved_root = root.resolve(strict=True)
    except (FileNotFoundError, RuntimeError) as error:
        raise ValueError(f"receipt path cannot be resolved safely: {filename!r}") from error
    if resolved.parent != resolved_root:
        raise ValueError(f"receipt path escapes its artifact root: {filename!r}")
    return candidate


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _expected_output_keys(source_record: dict) -> list[str]:
    key = source_record["key"]
    if source_record["source_type"] == "F32":
        return [key]
    base = key.removesuffix(".weight")
    return sorted((key, base + ".scales", base + ".biases"))


def _validate_repack_records(
    root: Path,
    records: object,
    source_files: dict,
    profile: dict[str, int],
    *,
    require_complete: bool,
) -> dict[str, str]:
    """Validate replay records before trusting any receipt-selected path."""
    if not isinstance(records, dict):
        raise TypeError("repack file records must be an object")
    if require_complete and set(records) != set(source_files):
        raise ValueError("existing repack receipt is incomplete")
    if not set(records) <= set(source_files):
        raise ValueError("repack records contain unknown source tensors")
    filenames: set[str] = set()
    weight_map: dict[str, str] = {}
    for name, entry in records.items():
        if not isinstance(entry, dict):
            raise TypeError(f"repack record is not an object: {name}")
        original = source_files[name]
        filename = entry.get("filename")
        path = _flat_child(root, filename, suffix=".safetensors")
        if filename != original["filename"]:
            raise ValueError(f"repack shard filename differs from source mapping: {name}")
        if filename in filenames:
            raise ValueError(f"repack records reuse a shard filename: {filename}")
        filenames.add(filename)
        expected_bits = None if original["source_type"] == "F32" else profile[original["source_type"]]
        if entry.get("source_type") != original["source_type"] or entry.get("bits") != expected_bits:
            raise ValueError(f"repack record policy differs from source mapping: {name}")
        expected_keys = _expected_output_keys(original)
        if entry.get("keys") != expected_keys:
            raise ValueError(f"repack record tensor keys differ from source mapping: {name}")
        if (
            not isinstance(entry.get("size"), int)
            or entry["size"] <= 0
            or not isinstance(entry.get("data_bytes"), int)
            or entry["data_bytes"] <= 0
            or not _valid_sha256(entry.get("sha256"))
        ):
            raise ValueError(f"repack record has invalid file metadata: {name}")
        for key in expected_keys:
            if key in weight_map:
                raise ValueError(f"duplicate output tensor: {key}")
            weight_map[key] = path.name
    return weight_map


def _profile_from_args(args) -> dict[str, int]:
    profile = {"IQ4_XS": args.iq4_bits, "Q5_K": args.q5_bits, "Q6_K": args.q6_bits}
    if any(bits not in (2, 3, 4, 5, 6, 8) for bits in profile.values()):
        raise ValueError("MLX affine bits must be one of 2, 3, 4, 5, 6, 8")
    return profile


def _source_record(source: Path) -> tuple[dict, str]:
    receipt_path = _flat_child(source, "mlx2-gguf-conversion.json")
    receipt = json.loads(receipt_path.read_text())
    if (not isinstance(receipt, dict)
            or receipt.get("schema") != "mlx2.orcasaq-qwen38-gguf-conversion.v1"
            or receipt.get("source_sha256") != SOURCE_SHA256
            or receipt.get("tensor_count") != TENSOR_COUNT
            or receipt.get("qualification") != "unqualified"):
        raise ValueError("source is not the complete pinned OrcaSAQ GGUF expansion")
    files = receipt.get("files", {})
    if not isinstance(files, dict) or len(files) != TENSOR_COUNT:
        raise ValueError("source expansion receipt has incomplete shards")
    if any(not isinstance(entry, dict) for entry in files.values()):
        raise ValueError("source expansion receipt has malformed shard records")
    filenames = [entry.get("filename") for entry in files.values()]
    keys = [entry.get("key") for entry in files.values()]
    if (
        any(not isinstance(value, str) or not value for value in filenames + keys)
        or len(set(filenames)) != TENSOR_COUNT
        or len(set(keys)) != TENSOR_COUNT
    ):
        raise ValueError("source expansion receipt has incomplete or colliding shards")
    for entry in files.values():
        _flat_child(source, entry.get("filename"), suffix=".safetensors")
        if (
            not isinstance(entry.get("key"), str)
            or not entry["key"]
            or not isinstance(entry.get("size"), int)
            or entry["size"] <= 0
            or not _valid_sha256(entry.get("sha256"))
        ):
            raise ValueError("source expansion receipt has invalid shard metadata")
    if Counter(entry.get("source_type") for entry in files.values()) != Counter(
            {"F32": 360, "IQ4_XS": 439, "Q5_K": 65, "Q6_K": 2}):
        raise ValueError("source expansion tensor encodings differ from pinned target")
    if any(entry.get("dtype") != ("F32" if entry["source_type"] == "F32" else "BF16")
           for entry in files.values()):
        raise ValueError("source expansion tensor dtype contradicts GGUF encoding")
    support_files = receipt.get("support_files")
    if not isinstance(support_files, dict) or set(support_files) != set(SUPPORT_FILES):
        raise ValueError("source expansion support file set differs from pinned target")
    for name, record in support_files.items():
        path = _flat_child(source, name)
        if (
            not isinstance(record, dict)
            or not _valid_sha256(record.get("sha256"))
            or not isinstance(record.get("size"), int)
            or record["size"] <= 0
            or not path.is_file()
            or path.stat().st_size != record["size"]
            or sha256(path) != record["sha256"]
        ):
            raise ValueError(f"source expansion support file changed: {name}")
    for name, field in (("config.json", "config_sha256"),
                        ("model.safetensors.index.json", "index_sha256")):
        path = _flat_child(source, name)
        if not _valid_sha256(receipt.get(field)) or sha256(path) != receipt[field]:
            raise ValueError(f"source expansion {name} changed")
    index = json.loads(_flat_child(source, "model.safetensors.index.json").read_text())["weight_map"]
    if index != {entry["key"]: entry["filename"] for entry in files.values()}:
        raise ValueError("source expansion weight index disagrees with receipt")
    return receipt, sha256(receipt_path)


def _quant_config(source_config: dict, files: dict, profile: dict[str, int]) -> dict:
    config = json.loads(json.dumps(source_config))
    if config.get("model_type") != "qwen3_5" or config.get("text_config", {}).get("num_hidden_layers") != 64:
        raise ValueError("source config is not Qwen3.8 27B")
    if "quantization" in config or "quantization_config" in config:
        raise ValueError("source expansion config must be unquantized")
    quant = {"group_size": GROUP_SIZE, "bits": profile["IQ4_XS"], "mode": "affine"}
    modules = set()
    for record in files.values():
        kind = record["source_type"]
        if kind == "F32":
            continue
        key = record["key"]
        if not key.endswith(".weight"):
            raise ValueError(f"quantized GGUF tensor is not a module weight: {key}")
        module = key.removesuffix(".weight")
        if module in modules:
            raise ValueError(f"quantized module occurs twice: {module}")
        modules.add(module)
        # Explicit entries bind every packed layer to the per-GGUF-type
        # policy. nn.quantize's class_predicate reads these exact names.
        quant[module] = {"group_size": GROUP_SIZE, "bits": profile[kind], "mode": "affine"}
    config["quantization"] = quant
    return config


def _pack_one(source_file: Path, target_file: Path, record: dict, bits: int | None) -> dict:
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    tensors = mx.load(str(source_file))
    if set(tensors) != {record["key"]}:
        raise ValueError(f"source shard has unexpected tensor keys: {source_file}")
    key = record["key"]
    value = tensors[key]
    if bits is None:
        if record["source_type"] != "F32":
            raise ValueError("only F32 source tensors may bypass repack")
        temporary = target_file.with_suffix(target_file.suffix + ".part")
        if temporary.is_symlink():
            raise ValueError(f"temporary output must not be a symlink: {temporary.name}")
        shutil.copyfile(source_file, temporary)
        os.replace(temporary, target_file)
        return {"size": target_file.stat().st_size, "sha256": sha256(target_file),
                "data_bytes": value.nbytes, "keys": [key], "bits": None}
    if value.ndim != 2 or value.shape[-1] % GROUP_SIZE or not key.endswith(".weight"):
        raise ValueError(f"unsupported affine tensor geometry: {key} {value.shape}")
    packed, scales, biases = mx.quantize(value, group_size=GROUP_SIZE, bits=bits, mode="affine")
    mx.eval(packed, scales, biases)
    base = key.removesuffix(".weight")
    output = {key: packed, base + ".scales": scales, base + ".biases": biases}
    temporary = target_file.with_name(target_file.stem + ".part.safetensors")
    temporary.unlink(missing_ok=True)
    try:
        mx.save_safetensors(str(temporary), output)
        os.replace(temporary, target_file)
    finally:
        temporary.unlink(missing_ok=True)
    return {"size": target_file.stat().st_size, "sha256": sha256(target_file),
            "data_bytes": sum(tensor.nbytes for tensor in output.values()),
            "keys": sorted(output), "bits": bits}


def repack(source: Path, output: Path, *, profile: dict[str, int] | None = None,
           max_tensors: int | None = None) -> dict:
    source, output = _resolved_roots(source, output)
    profile = dict(DEFAULT_PROFILE if profile is None else profile)
    if set(profile) != set(DEFAULT_PROFILE) or any(bits not in (2, 3, 4, 5, 6, 8) for bits in profile.values()):
        raise ValueError("affine profile must assign supported bit widths to IQ4_XS, Q5_K, and Q6_K")
    source_receipt, source_receipt_sha = _source_record(source)
    source_config = json.loads(_flat_child(source, "config.json").read_text())
    output_config = _quant_config(source_config, source_receipt["files"], profile)
    config_digest = hashlib.sha256((json.dumps(output_config, sort_keys=True, indent=2) + "\n").encode()).hexdigest()
    identity = {"source_receipt_sha256": source_receipt_sha,
                "source_model_config_sha256": source_receipt["config_sha256"],
                "profile": profile, "output_config_sha256": config_digest}
    output.mkdir(parents=True, exist_ok=True)
    completion = _flat_child(output, "mlx2-affine-repack.json")
    if completion.is_file():
        result = json.loads(completion.read_text())
        if (
            not isinstance(result, dict)
            or result.get("schema") != "mlx2.orcasaq-qwen38-affine-repack.v1"
            or result.get("qualification") != "experimental-unqualified"
            or result.get("source_gguf_sha256") != SOURCE_SHA256
            or result.get("source_conversion_receipt") != str(source / "mlx2-gguf-conversion.json")
            or result.get("quantization") != {
                "group_size": GROUP_SIZE,
                "mode": "affine",
                "bits_by_gguf_type": profile,
            }
            or result.get("added_quantization_error") != "present-unmeasured"
            or result.get("notice") != NOTICE
        ):
            raise ValueError("existing repack receipt has invalid provenance or state")
        if result.get("identity") != identity:
            raise ValueError("existing repack receipt belongs to another source or profile")
        expected_weight_map = _validate_repack_records(
            output, result.get("files"), source_receipt["files"], profile,
            require_complete=True,
        )
        for name, entry in result["files"].items():
            path = _flat_child(output, entry["filename"], suffix=".safetensors")
            if not path.is_file() or path.stat().st_size != entry["size"] or sha256(path) != entry["sha256"]:
                raise ValueError(f"existing repack shard changed: {name}")
        config_path = _flat_child(output, "config.json")
        index_path = _flat_child(output, "model.safetensors.index.json")
        if (
            result.get("config_sha256") != identity["output_config_sha256"]
            or not _valid_sha256(result.get("index_sha256"))
            or sha256(config_path) != result["config_sha256"]
            or sha256(index_path) != result["index_sha256"]
        ):
            raise ValueError("existing repack config or index changed")
        index = json.loads(index_path.read_text())
        expected_index = {
            "metadata": {"total_size": sum(entry["data_bytes"] for entry in result["files"].values())},
            "weight_map": expected_weight_map,
        }
        if index != expected_index:
            raise ValueError("existing repack weight index disagrees with receipt")
        if result.get("support_files") != source_receipt["support_files"]:
            raise ValueError("existing repack support file records changed")
        for name, record in source_receipt["support_files"].items():
            path = _flat_child(output, name)
            if not path.is_file() or path.stat().st_size != record["size"] or sha256(path) != record["sha256"]:
                raise ValueError(f"existing repack support file changed: {name}")
        return {"state": "complete-verified", "receipt": str(completion)}
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    progress_path = _flat_child(output, ".mlx2-affine-repack-progress.json")
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {"identity": identity, "files": {}}
    if not isinstance(progress, dict) or progress.get("identity") != identity:
        raise ValueError("existing repack progress belongs to another source or profile")
    _validate_repack_records(
        output, progress.get("files"), source_receipt["files"], profile,
        require_complete=False,
    )
    ordered = sorted(source_receipt["files"].items(), key=lambda item: item[1]["filename"])
    for count, (name, original) in enumerate(ordered):
        if max_tensors is not None and count >= max_tensors:
            break
        source_file = _flat_child(source, original["filename"], suffix=".safetensors")
        if not source_file.is_file() or source_file.stat().st_size != original["size"] or sha256(source_file) != original["sha256"]:
            raise ValueError(f"source tensor shard changed: {name}")
        target_file = _flat_child(output, original["filename"], suffix=".safetensors")
        prior = progress["files"].get(name)
        if target_file.exists():
            if not prior or prior["filename"] != target_file.name or target_file.stat().st_size != prior["size"] or sha256(target_file) != prior["sha256"]:
                raise ValueError(f"unverified existing repack shard: {target_file}")
            continue
        if prior:
            raise ValueError(f"recorded repack shard missing: {target_file.name}")
        kind = original["source_type"]
        bits = None if kind == "F32" else profile[kind]
        converted = _pack_one(source_file, target_file, original, bits)
        progress["files"][name] = {"filename": target_file.name, "source_type": kind, **converted}
        _write_json_atomic(progress_path, progress)
        print(f"CPU repacked {count + 1}/{TENSOR_COUNT}: {name}", flush=True)
    if len(progress["files"]) != TENSOR_COUNT:
        return {"state": "partial", "completed": len(progress["files"]), "expected": TENSOR_COUNT,
                "progress": str(progress_path)}
    weight_map = _validate_repack_records(
        output, progress["files"], source_receipt["files"], profile,
        require_complete=True,
    )
    for name, support in source_receipt["support_files"].items():
        input_file = _flat_child(source, name)
        target_file = _flat_child(output, name)
        if sha256(input_file) != support["sha256"]:
            raise ValueError(f"source support file changed: {name}")
        if target_file.exists():
            if sha256(target_file) != support["sha256"]:
                raise ValueError(f"existing output support file changed: {name}")
        else:
            shutil.copyfile(input_file, target_file)
    config_path = _flat_child(output, "config.json")
    index_path = _flat_child(output, "model.safetensors.index.json")
    _write_json_atomic(config_path, output_config)
    total_size = sum(entry["data_bytes"] for entry in progress["files"].values())
    _write_json_atomic(index_path, {"metadata": {"total_size": total_size}, "weight_map": weight_map})
    receipt = {"schema": "mlx2.orcasaq-qwen38-affine-repack.v1", "identity": identity,
               "source_gguf_sha256": SOURCE_SHA256, "source_conversion_receipt": str(source / "mlx2-gguf-conversion.json"),
               "quantization": {"group_size": GROUP_SIZE, "mode": "affine", "bits_by_gguf_type": profile},
               "files": progress["files"], "config_sha256": sha256(config_path),
               "index_sha256": sha256(index_path),
               "support_files": source_receipt["support_files"], "added_quantization_error": "present-unmeasured",
               "qualification": "experimental-unqualified", "notice": NOTICE}
    _write_json_atomic(completion, receipt)
    progress_path.unlink()
    return {"state": "complete", "receipt": str(completion), "output_tensors": len(weight_map)}


def make_dflash_policy(target: Path, draft: Path) -> dict:
    """Build exact config/index pins only after both artifacts pass inspection."""
    from mlx2.adapters.dflash2 import content_revision as draft_revision
    from mlx2.adapters.dflash2 import inspect_drafter
    from mlx2.adapters.qwen38_27b import content_revision as target_revision
    from mlx2.adapters.qwen38_27b import inspect_artifact

    if not (target / "mlx2-affine-repack.json").is_file():
        raise ValueError("target repack is incomplete")
    inspected = inspect_artifact(target)
    if not inspected["has_mtp"]:
        raise ValueError("repacked target lost its MTP head")
    drafter = inspect_drafter(draft, target)
    return {"draft_model": str(draft.expanduser().resolve()), "num_draft": 7,
            "pairwise_selection": "batched", "draft_revision": draft_revision(drafter),
            "target_revision": target_revision(target)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iq4-bits", type=int, default=5)
    parser.add_argument("--q5-bits", type=int, default=6)
    parser.add_argument("--q6-bits", type=int, default=8)
    parser.add_argument("--max-tensors", type=int, help="Bounded partial run; writes no complete model index")
    parser.add_argument("--draft-model", type=Path, help="After completion, write exact DFlash2 policy beside target")
    args = parser.parse_args()
    result = repack(args.source, args.output, profile=_profile_from_args(args),
                    max_tensors=args.max_tensors)
    if args.draft_model is not None and result["state"].startswith("complete"):
        policy = make_dflash_policy(args.output, args.draft_model)
        policy_path = args.output / "dflash2-policy.json"
        if policy_path.is_file() and json.loads(policy_path.read_text()) != policy:
            raise ValueError("existing DFlash2 policy pins another target or drafter")
        _write_json_atomic(policy_path, policy)
        result["dflash2_policy"] = str(policy_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

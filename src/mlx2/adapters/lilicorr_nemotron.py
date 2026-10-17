"""Pinned Nemotron LiLiCorr export inspection and BF16 execution contract.

The BF16 export is executable only after strict payload, owned-head, trained
sliding-window and target-state binding. Native NVFP4 and ModelOpt packed
exports remain metadata-only and fail before allocation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from ..runtime.io_ownership import call_io, storage_owner
from .dflash2 import _decode_unique_json, _read_safetensors_header, _source_binding
from .lilicorr import LiLiCorrConfig, _file_sha256, expected_weight_shapes
from .nemotron35_super_vl import _metadata_bytes, inspect_artifact, runtime_config

SOURCE_REVISION = "7c7221aeb4cb3139307aa457f7dce11a8fb6ceb2"
EXPORTS = {
    "e2b7ee2ea7ea4d78ee11ce67a15e7c113621950ec59972d18a0776cd8e3247db": (
        "bf16",
        "f0a24bb4b7a85b7943c690364094c9d531965adc440eb78fe3c7b1eb663fa739",
    ),
    "3f0bbfd0ed1190ef652adc1e707b4037f0bf8f37cb88cb194a5d4c3b4f878a50": (
        "native",
        "202c45c903765066c47c772a3921706054823c33115f7a96607063052f670a5c",
    ),
    "880d77cb17f6f45925ba78ba94a04b92b5a88b6046e7b6e376d2ab794b02ba04": (
        "modelopt",
        "eec5efdf1b838020c2842f52d560c187515e84889c51f99d70b75c773c286599",
    ),
}
SOURCE_CONFIG_SHA256 = (
    "880d77cb17f6f45925ba78ba94a04b92b5a88b6046e7b6e376d2ab794b02ba04"
)
SOURCE_WEIGHT_SHA256 = (
    "eec5efdf1b838020c2842f52d560c187515e84889c51f99d70b75c773c286599"
)
TRAINED_SLIDING_WINDOW = 8192


def expected_export_tensors(config, variant):
    """Extend the existing trained head's *shape* oracle, never its loader."""
    normalized = copy.deepcopy(config)
    normalized.pop("quantization", None)
    normalized.pop("quantization_config", None)
    normalized["has_own_lm_head"] = False
    normalized["use_sliding_window"] = False
    normalized["sliding_window"] = None
    normalized["layer_types"] = ["full_attention"] * config["num_hidden_layers"]
    for key in ("causal", "projector_type", "use_swa", "swa_window_size"):
        normalized["dflash_config"].pop(key, None)
    args = LiLiCorrConfig.from_dict(normalized)
    shapes = expected_weight_shapes(args)
    shapes["lm_head.weight"] = [args.vocab_size, args.hidden_size]
    modules = {
        "fc",
        "lm_head",
        "lilicorr.context_proj",
        "lilicorr.pass_hidden_proj",
        "lilicorr.token_proj",
    }
    modules.update(
        f"layers.{i}.mlp.{name}_proj"
        for i in range(args.num_hidden_layers)
        for name in ("down", "gate", "up")
    )
    modules.update(
        f"lilicorr.layers.{i}.{name}"
        for i in range(args.lilicorr_num_layers)
        for name in ("attn.out_proj", "mlp.down_proj", "mlp.up_proj")
    )
    if variant == "native":
        quant = config.get("quantization", {})
        if quant != {
            "bits": 4,
            "group_size": 16,
            "mode": "nvfp4",
            "global_scale": "per_tensor_fp32",
            "modules": sorted(modules),
        }:
            raise ValueError("LiLiCorr native quantization modules/exclusions differ")
    elif variant not in ("bf16", "modelopt") or config.get("quantization") is not None:
        raise ValueError("Unsupported LiLiCorr export recipe")
    expected = {}
    for key, shape in shapes.items():
        module = key.removesuffix(".weight")
        if variant == "bf16" or not key.endswith(".weight") or module not in modules:
            expected[key] = ("BF16", shape)
            continue
        if shape[-1] % 16:
            raise ValueError("LiLiCorr NVFP4 input width is not group aligned")
        expected[key] = (
            "U32" if variant == "native" else "U8",
            [*shape[:-1], shape[-1] // (8 if variant == "native" else 2)],
        )
        scale = ".scales" if variant == "native" else ".weight_scale"
        global_scale = ".global_scale" if variant == "native" else ".weight_scale_2"
        expected[module + scale] = (
            "U8" if variant == "native" else "F8_E4M3",
            [*shape[:-1], shape[-1] // 16],
        )
        expected[module + global_scale] = ("F32", [])
    return expected


def executable_config(config):
    """Decode only the pinned BF16 owned-head/sliding execution contract."""
    if not isinstance(config, dict):
        raise TypeError("Nemotron LiLiCorr config must be an object")
    nested = config.get("dflash_config")
    if (
        config.get("architectures") != ["LiLiCorrDraftModel"]
        or config.get("model_type") != "qwen3"
        or config.get("has_own_lm_head") is not True
        or config.get("use_sliding_window") is not True
        or not isinstance(nested, dict)
        or nested.get("causal") is not False
        or nested.get("projector_type") != "lilicorr"
        or nested.get("use_swa") is not True
        or nested.get("swa_window_size") != config.get("sliding_window")
        or config.get("sliding_window") != TRAINED_SLIDING_WINDOW
    ):
        raise ValueError("Unsupported Nemotron LiLiCorr executable contract")
    fields = LiLiCorrConfig.__dataclass_fields__
    values = {key: value for key, value in config.items() if key in fields}
    values.update({key: value for key, value in nested.items() if key in fields})
    values["has_own_lm_head"] = True
    return LiLiCorrConfig(**values)


def _inspect_export(path, target):
    path = Path(path).expanduser().resolve()
    raw_config = _metadata_bytes(path / "config.json")
    config_hash = hashlib.sha256(raw_config).hexdigest()
    if config_hash not in EXPORTS:
        raise ValueError("Unknown Nemotron LiLiCorr export revision")
    variant, weight_hash = EXPORTS[config_hash]
    config = _decode_unique_json(raw_config, "LiLiCorr export config")
    text = target["text_config"]
    nested = config["dflash_config"]
    taps = nested["target_layer_ids"]
    if (
        config["hidden_size"] != text["hidden_size"]
        or config["vocab_size"] != text["vocab_size"]
        or config["num_target_layers"] != len(text["layers_block_type"])
        or not taps
        or taps != sorted(set(taps))
        or any(
            type(i) is not int or not 0 <= i < config["num_target_layers"] for i in taps
        )
    ):
        raise ValueError("LiLiCorr target feature/vocabulary/layer contract differs")
    if (
        config.get("has_own_lm_head") is not True
        or config.get("use_sliding_window") is not True
        or nested.get("causal") is not False
        or nested.get("projector_type") != "lilicorr"
        or nested.get("use_swa") is not True
        or config["sliding_window"] != nested.get("swa_window_size")
        or config["layer_types"] != ["sliding_attention"] * config["num_hidden_layers"]
    ):
        raise ValueError(
            "LiLiCorr owned-head/sliding-attention export contract differs"
        )
    expected = expected_export_tensors(config, variant)
    file = path / "model.safetensors"
    binding = _source_binding(file)
    raw_header, header, size = _read_safetensors_header(file)
    actual = {k: v for k, v in header.items() if k != "__metadata__"}
    if actual.keys() != expected.keys():
        raise ValueError("LiLiCorr export tensor inventory differs")
    dtype_bytes = {"BF16": 2, "U32": 4, "U8": 1, "F8_E4M3": 1, "F32": 4}
    ranges = []
    for key, (dtype, shape) in expected.items():
        item = actual[key]
        if (
            set(item) != {"dtype", "shape", "data_offsets"}
            or item["dtype"] != dtype
            or item["shape"] != shape
        ):
            raise ValueError(f"LiLiCorr export dtype/shape mismatch: {key}")
        span = item["data_offsets"]
        if (
            not isinstance(span, list)
            or len(span) != 2
            or any(type(x) is not int for x in span)
            or not 0 <= span[0] <= span[1] <= size
            or span[1] - span[0] != math.prod(shape) * dtype_bytes[dtype]
        ):
            raise ValueError(f"LiLiCorr export extent mismatch: {key}")
        ranges.append(span)
    end = 0
    for start, stop in sorted(ranges):
        if start != end:
            raise ValueError("LiLiCorr export payload overlaps or has gaps")
        end = stop
    if end != size or binding != _source_binding(file):
        raise ValueError("LiLiCorr export payload incomplete or changed")
    if variant != "modelopt":
        receipt = _decode_unique_json(
            _metadata_bytes(path / "mlx2-conversion.json"), "LiLiCorr conversion"
        )
        required = {
            "schema": "mlx2.modelopt-nvfp4-import.v1",
            "mode": variant,
            "source_config_sha256": SOURCE_CONFIG_SHA256,
            "source_weight_sha256": SOURCE_WEIGHT_SHA256,
            "output_weight_sha256": weight_hash,
            "global_scale_preserved": True,
            "quantized_matrices": 26,
            "output_tensor_bytes": size,
        }
        if any(receipt.get(key) != value for key, value in required.items()):
            raise ValueError("LiLiCorr export conversion receipt mismatch")
    header_hash = hashlib.sha256(raw_header).hexdigest()
    fingerprint = hashlib.sha256(
        json.dumps(
            [config_hash, header_hash, binding, target["identity"]["fingerprint"]],
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return {
        "schema": "mlx2.nemotron-lilicorr-export-plan.v1",
        "variant": variant,
        "path": str(path),
        "fingerprint": fingerprint,
        "config_sha256": config_hash,
        "header_sha256": header_hash,
        "expected_weight_sha256": weight_hash,
        "weight_payload_rehashed": False,
        "source_revision": SOURCE_REVISION,
        "tensor_count": len(actual),
        "weight_bytes": size,
        "target_fingerprint": target["identity"]["fingerprint"],
        "target_layer_ids": taps,
        "target_feature_width": len(taps) * text["hidden_size"],
        "target_and_draft_weight_bytes": target["load_plan"]["resident_weight_bytes"]
        + size,
        "owned_lm_head": True,
        "trained_sliding_window": config["sliding_window"],
        "artifact_schema_supported": True,
        "runtime_load_supported": False,
        "qualified": False,
        "selected": False,
        "observed_used": False,
        "remaining": [
            "owned draft LM head must survive target binding",
            "trained sliding attention must preserve absolute positions and source mask convention",
            "Nemotron target taps and recurrent/draft rollback must be bound through serving",
            "NVFP4 global scales/exclusions need source numerical parity before native loading",
        ]
        + (
            ["ModelOpt byte packing must be converted before MLX loading"]
            if variant == "modelopt"
            else []
        ),
    }


def inspect_export_plan(draft_path, target_path):
    """Inspect both exact artifacts without importing MLX or reading tensors."""
    target = inspect_artifact(target_path)
    return call_io(
        "nemotron-lilicorr-export-plan", _inspect_export, str(draft_path), target,
        owner=storage_owner("weights", draft_path),
    )


def _inspect_drafter(path, target):
    plan = _inspect_export(path, target)
    if plan["variant"] != "bf16":
        raise ValueError(
            "Nemotron LiLiCorr native/modelopt quantized exports are not executable"
        )
    path = Path(path).expanduser().resolve()
    raw_config = _metadata_bytes(path / "config.json")
    config = _decode_unique_json(raw_config, "LiLiCorr export config")
    args = executable_config(config)
    file = path / "model.safetensors"
    return {
        **plan,
        "schema": "mlx2.nemotron-lilicorr-drafter.v1",
        "config": config,
        "args": args,
        "files": [(file.name, file.stat().st_size, file.stat().st_mtime_ns)],
        "source_binding": _source_binding(file),
        "target": copy.deepcopy(target),
        "runtime_load_supported": True,
        "remaining": [
            "full-artifact CPU/Metal load and decode are not qualified",
            "native/modelopt quantized exports remain refused",
        ],
    }


def inspect_drafter(path, *, target_path=None, target_artifact=None):
    """Inspect the executable pinned BF16 pair without reading tensor payloads."""
    if (target_path is None) == (target_artifact is None):
        raise ValueError("provide exactly one Nemotron target path or artifact")
    target = (
        inspect_artifact(target_path)
        if target_artifact is None
        else copy.deepcopy(target_artifact)
    )
    # The full Super adapter and the deliberately stripped text-only adapter
    # expose different load plans over the same language geometry.  Normalize
    # the full artifact here so the owned LiLiCorr head binds to the adapter's
    # actual target identity while retaining one inspection implementation.
    if "text_config" not in target:
        target["text_config"] = runtime_config(target["config"])
    if "load_plan" not in target:
        target["load_plan"] = {
            "resident_weight_bytes": int(target["conversion"]["tensor_bytes"]),
        }
    return call_io("nemotron-lilicorr-drafter", _inspect_drafter, str(path), target,
                   owner=storage_owner("weights", path))


def content_revision(record):
    return record["fingerprint"]


def _validate_target_contract(record, target_contract):
    required = {
        "schema": "mlx2.nemotron-lilicorr-target.v1",
        "target_model_type": "nemotron_h_omni",
        "target_fingerprint": record["target_fingerprint"],
        "target_layer_ids": tuple(record["args"].target_layer_ids),
        "target_feature_width": (
            len(record["args"].target_layer_ids) * record["args"].hidden_size
        ),
        "capture": "post-layer-pre-final-norm",
        "ordinary_reference_preserved": True,
        "speculative_rollback": "exact",
        "owned_lm_head": True,
        "owned_final_norm": True,
        "qualified": False,
    }
    if not isinstance(target_contract, dict) or target_contract != required:
        raise ValueError("Nemotron LiLiCorr target contract differs")


def load_drafter(
    record,
    target_model,
    *,
    target_contract,
    runtime_quantization=None,
    draft_attention_windows=None,
):
    """Load the pinned BF16 export and an optional adapter-owned Q8 profile."""
    from .nemotron35_draft_quantization import normalize_runtime_quantization

    runtime_quantization = normalize_runtime_quantization(runtime_quantization)
    if draft_attention_windows is not None:
        raise ValueError("Nemotron LiLiCorr trained sliding windows are immutable")
    if record.get("variant") != "bf16" or record.get("runtime_load_supported") is not True:
        raise ValueError("Nemotron LiLiCorr export is not executable")
    _validate_target_contract(record, target_contract)
    if record.get("args") != executable_config(record.get("config")):
        raise ValueError("Nemotron LiLiCorr load configuration differs")
    refreshed = _inspect_drafter(record["path"], record["target"])
    for key in (
        "fingerprint", "config_sha256", "header_sha256", "source_binding",
        "files", "target_fingerprint", "expected_weight_sha256",
    ):
        if refreshed.get(key) != record.get(key):
            raise ValueError("Nemotron LiLiCorr artifact changed since inspection")
    file = Path(record["path"]) / "model.safetensors"
    if _file_sha256(file) != record["expected_weight_sha256"]:
        raise ValueError("Nemotron LiLiCorr tensor bytes differ from pinned export")

    import mlx.core as mx

    from ..runtime.drafters.lilicorr import LiLiCorrDraftModel

    model = LiLiCorrDraftModel(record["args"])
    weights = model.sanitize(mx.load(str(file)))
    mx.eval(list(weights.values()))
    model.load_weights(list(weights.items()), strict=True)
    if runtime_quantization is not None:
        from .nemotron35_draft_quantization import apply_runtime_quantization

        apply_runtime_quantization(model, runtime_quantization)
    else:
        model.runtime_weight_quantization = None
    model.eval()
    mx.eval(model.parameters())
    weights.clear()
    mx.clear_cache()
    return model.bind(target_model, target_contract=target_contract)

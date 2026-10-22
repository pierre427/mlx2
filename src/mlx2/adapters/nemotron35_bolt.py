"""Nemotron 3.5 Bolt adapter: pinned source contract and converted-artifact serving.

Bolt reuses the Nemotron-H/RADIO model family, but its published 24 GB
checkpoint mixes ModelOpt NVFP4, GGML Q8_0 and GGML IQ2_XXS tensors over a
heterogeneous Puzzle-NAS MoE geometry.  The source is inspected only; the
executable artifact is its CPU conversion to native affine MLX tensors
(scripts/convert_nemotron35_bolt.py).  Serving is text-only, has no native
MTP head, and takes the companion LiLiCorr export as its external drafter.
Every route is unqualified.  See provenance/nemotron35-bolt.json.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from ..sampling_defaults import MODEL_CARD, SamplingDefaults, VendorSampling
from .dflash2 import _decode_unique_json, _read_safetensors_header
from .nemotron35_super import Nemotron35SuperAdapter

SOURCE_REVISION = "c6cc2e8ffa353403213d6b9b18aaf8eb684d705a"
SOURCE_CONFIG_SHA256 = "cf4e46f4d6d241e67a91ed949681531d3f378ca0a7e04e3f8af66324098d9587"
SOURCE_QUANT_SHA256 = "0c533a01ce840e63f4ed732d781b5fda6150daf5faa84feae7b906b7fd05a2d0"
SOURCE_INDEX_SHA256 = "8e26109aa40707bb4aa92ef3263320810a63c1223af936e9949bc0ff4d6030b6"
SOURCE_TENSOR_BYTES = 24_079_678_824
SOURCE_SHARD_BYTES = 24_083_745_496
SOURCE_TENSOR_COUNT = 28_885
SOURCE_SHARDS = 90
SOURCE_HEADER_SET_SHA256 = "2edee4dac15750b6ee8656049b8acd85fc979c6c2729b689f599a8bd4ea25462"
SOURCE_METADATA_SHA256 = {
    "tokenizer.json": "d0432ad8b1c2984a2e2dca82e1933e3fd5628df6e95c51fba4c3ee9950dce299",
    "tokenizer_config.json": "c6067a79d81c7bf7b1485588b2a407ac9fe124056f4f759f09fc211d7114d123",
    "chat_template.jinja": "6521afc75afb0661e3558e7e6dee5f2d15de8be1c4dece8171935e778ea283f4",
    "generation_config.json": "c67c90fcadf0e7ef78b663eb159664b36a35ffb9dd2523096e226e5c3d1b0e5f",
}

LILICORR_REVISION = "abd478065678ee83a93821c4b3f512da24e4cecb"
LILICORR_CONFIG_SHA256 = "006c3f02f39cbaa10b6b6cabff88d64a9e40ff56b57d5aec9b452abe9dbc10e0"
LILICORR_QUANT_SHA256 = "848259a76e0dce8e4dd48102c049a2b854b6fcea203068c329ffeee76f12633d"
LILICORR_WEIGHT_SHA256 = "850c121c5fa444dfdebbc15d8f23a1ab2beda55863460ab44203e3cb63808562"
LILICORR_TENSOR_BYTES = 962_467_000
LILICORR_HEADER_SHA256 = "0b3bed1f9d9030d6b97ce9c5ffd1fca921cc7b8910e4ca7a8a083f4e2a0f8eb6"

CACHE_LAYOUT = "nemotron35-bolt-hybrid-fp32-router-v1"
CONVERSION_SCHEMA = "mlx2.nemotron35-bolt-conversion.v1"
# The BF16 import of the pinned NVFP4 LiLiCorr (conversion/nvfp4.py, mode bf16).
LILICORR_EXPORT_CONFIG_SHA256 = "404f0e8e4ccf6218c48c13fba7c116853193eb8bb2c43ac516fc60e8e0f6b434"
LILICORR_EXPORT_HEADER_SHA256 = "0a88b1c4c27a6169b3b44d7015afd5c62c60f23c8656672712ff32e9247bac74"
LILICORR_EXPORT_WEIGHT_SHA256 = "97ea2a78a446a61b330e811c954bebf7fc8d5b0924e1879b207da0c27e7a86fa"
LILICORR_TAPS = (1, 18, 35, 51, 68, 85)

DESCRIPTOR = ModelDescriptor(
    model_type="nemotron_h_omni",
    family="nemotron35-bolt-74b-a9b",
    variant="mixed-affine-experts4g32",
    cache_layout=CACHE_LAYOUT,
    state_planes=frozenset(
        {
            StatePlane.ATTENTION_KV,
            StatePlane.RECURRENT,
            StatePlane.RNG,
            StatePlane.TRANSCRIPT,
            StatePlane.DRAFT,
        }
    ),
    capabilities=frozenset(
        {
            Capability.TEXT,
            Capability.STREAMING,
            Capability.TOOLS,
            Capability.REASONING,
            Capability.CONTINUOUS_BATCH,
            Capability.PREFIX_REUSE,
            Capability.APC_V2,
            Capability.LAYERED_CACHE,
            Capability.GRAMMAR,
            # Row-exact B1 verify (nemotron_h.Model.pld_verify_forward).
            Capability.PROMPT_LOOKUP,
        }
    ),
    metadata={
        "execution": "mlx2.adapters.nemotron35_bolt.Nemotron35BoltAdapter",
        "qualification": "unqualified",
        "ordinary_reference": "implemented",
        "vision": "absent-text-only-images-fail-closed",
        "native_mtp": "absent",
        "external_draft": "lilicorr-bf16-import-attention-sinks",
    },
)
SAMPLING = VendorSampling.single(
    SamplingDefaults(temperature=1.0, top_p=0.95, source=MODEL_CARD),
    model="NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> tuple[bytes, dict]:
    raw = path.read_bytes()
    value = _decode_unique_json(raw, label)
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return raw, value


def _is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


_DTYPE_BYTES = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "I64": 8, "U8": 1}


def _source_binding(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or escaped Nemotron 3.5 Bolt {label}: {path.name}")


def _validate_safetensors_file(path: Path) -> tuple[bytes, dict, int, tuple[int, ...]]:
    """Validate complete header geometry and stable, contiguous payload extent."""
    _require_regular_file(path, "safetensors payload")
    before = _source_binding(path)
    raw_header, header, payload = _read_safetensors_header(path)
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    ranges = []
    for name, item in tensors.items():
        if not isinstance(item, dict) or set(item) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(f"Invalid Nemotron 3.5 Bolt tensor header: {name}")
        dtype, shape, span = item["dtype"], item["shape"], item["data_offsets"]
        if (
            dtype not in _DTYPE_BYTES
            or not isinstance(shape, list)
            or any(type(value) is not int or value < 0 for value in shape)
            or not isinstance(span, list)
            or len(span) != 2
            or any(type(value) is not int for value in span)
            or not 0 <= span[0] <= span[1] <= payload
            or span[1] - span[0] != math.prod(shape) * _DTYPE_BYTES[dtype]
        ):
            raise ValueError(f"Invalid Nemotron 3.5 Bolt tensor extent: {name}")
        ranges.append(span)
    end = 0
    for start, stop in sorted(ranges):
        if start != end:
            raise ValueError("Nemotron 3.5 Bolt payload overlaps or has gaps")
        end = stop
    if end != payload or before != _source_binding(path):
        raise ValueError("Nemotron 3.5 Bolt payload is incomplete or changed")
    return raw_header, tensors, payload, before


def validate_target_config(config: dict) -> dict:
    """Validate the immutable architecture and native storage declarations."""
    if not isinstance(config, dict):
        raise TypeError("Nemotron 3.5 Bolt config must be an object")
    text = config.get("llm_config")
    vision = config.get("vision_config")
    blocks = text.get("block_configs") if isinstance(text, dict) else None
    if (
        config.get("architectures") != ["NemotronH_Omni_Reasoning_V3"]
        or config.get("model_type") != "nemotron_h_omni"
        or not isinstance(text, dict)
        or text.get("model_type") != "nemotron_h_puzzle"
        or text.get("hidden_size") != 4096
        or text.get("vocab_size") != 131072
        or text.get("max_position_embeddings") != 262144
        or text.get("num_nextn_predict_layers") != 0
        or text.get("mtp_block_configs") != []
        or text.get("mtp_layers_block_type") != []
        or not isinstance(blocks, list)
        or any(not isinstance(item, dict) for item in blocks)
        or text.get("layers_block_type") != [item.get("block_type") for item in blocks]
    ):
        raise ValueError("Unsupported Nemotron 3.5 Bolt target geometry")
    counts = Counter(text["layers_block_type"])
    moe = [item for item in blocks if item.get("block_type") == "moe"]
    if (
        len(blocks) != 88
        or counts != {"mamba": 40, "moe": 40, "attention": 8}
        or {item.get("n_routed_experts") for item in moe}
        != {240, 304, 368, 432, 496}
        or {item.get("moe_intermediate_size") for item in moe}
        != {1536, 2048, 2304, 2560}
        or {item.get("num_experts_per_tok") for item in moe} != {9}
        or text.get("mamba_num_heads") != 128
        or text.get("mamba_head_dim") != 64
        or text.get("ssm_state_size") != 128
        or text.get("n_groups") != 8
        or text.get("num_attention_heads") != 32
        or text.get("num_key_value_heads") != 2
    ):
        raise ValueError("Nemotron 3.5 Bolt Puzzle-NAS blocks differ")
    if (
        not isinstance(vision, dict)
        or vision.get("model_type") != "radio"
        or vision.get("hidden_size") != 1280
        or vision.get("num_hidden_layers") != 32
        or vision.get("patch_size") != 16
        or config.get("projector_hidden_size") != 20480
        or config.get("sound_config") is not None
    ):
        raise ValueError("Nemotron 3.5 Bolt RADIO geometry differs")
    quantization_config = config.get("quantization_config")
    groups = (
        quantization_config.get("config_groups")
        if isinstance(quantization_config, dict)
        else None
    )
    if (
        not isinstance(groups, dict)
        or set(groups) != {"group_0", "group_1", "group_2"}
        or any(not isinstance(item, dict) for item in groups.values())
    ):
        raise ValueError("Nemotron 3.5 Bolt mixed quantization groups differ")
    q8, nvfp4, iq2 = groups["group_0"], groups["group_1"], groups["group_2"]
    if (
        {k: q8.get(k) for k in ("quant_algo", "group_size", "packing", "block_payload_bytes")}
        != {"quant_algo": "Q8_0", "group_size": 32, "packing": "ggml", "block_payload_bytes": 34}
        or len(q8.get("targets", [])) != 113
        or nvfp4.get("weights")
        != {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16}
        or len(nvfp4.get("targets", [])) != 80
        or {k: iq2.get(k) for k in ("quant_algo", "group_size", "packing", "block_payload_bytes")}
        != {"quant_algo": "IQ2_XXS", "group_size": 256, "packing": "ggml", "block_payload_bytes": 66}
        or len(iq2.get("targets", [])) != 27_728
    ):
        raise ValueError("Nemotron 3.5 Bolt native packing contract differs")
    return text


def inspect_source_artifact(model_path) -> dict:
    """Pinned source checkpoint (mixed GGML/ModelOpt); inspection only."""
    path = Path(model_path).expanduser().resolve()
    for name, digest in SOURCE_METADATA_SHA256.items():
        item = path / name
        _require_regular_file(item, "metadata")
        if _sha256(item) != digest:
            raise ValueError(f"Unknown Nemotron 3.5 Bolt metadata revision: {name}")
    config_path = path / "config.json"
    quant_path = path / "hf_quant_config.json"
    index_path = path / "model.safetensors.index.json"
    for item, label in (
        (config_path, "config"),
        (quant_path, "quantization metadata"),
        (index_path, "tensor index"),
    ):
        _require_regular_file(item, label)
    config_raw, config = _read_json(config_path, "Bolt config")
    quant_raw, quant = _read_json(quant_path, "Bolt quantization")
    index_raw, index = _read_json(index_path, "Bolt index")
    if (
        hashlib.sha256(config_raw).hexdigest() != SOURCE_CONFIG_SHA256
        or hashlib.sha256(quant_raw).hexdigest() != SOURCE_QUANT_SHA256
        or hashlib.sha256(index_raw).hexdigest() != SOURCE_INDEX_SHA256
    ):
        raise ValueError("Unknown Nemotron 3.5 Bolt source revision")
    text = validate_target_config(config)
    q = quant.get("quantization")
    quantized_layers = q.get("quantized_layers", {}) if isinstance(q, dict) else {}
    if (
        quant.get("producer") != {"name": "modelopt", "version": "0.37.0"}
        or not isinstance(q, dict)
        or q.get("quant_algo") != "MIXED_PRECISION"
        or q.get("kv_cache_quant_algo") is not None
        or not isinstance(quantized_layers, dict)
        or any(not isinstance(item, dict) for item in quantized_layers.values())
        or Counter(
            item.get("quant_algo") for item in quantized_layers.values()
        )
        != {"IQ2_XXS": 27_728, "Q8_0": 113, "W4A16_NVFP4": 80}
    ):
        raise ValueError("Nemotron 3.5 Bolt ModelOpt manifest differs")
    mapping = index.get("weight_map")
    if (
        index.get("metadata") != {"total_size": SOURCE_TENSOR_BYTES}
        or not isinstance(mapping, dict)
        or any(not isinstance(key, str) or not isinstance(value, str) for key, value in mapping.items())
        or len(mapping) != SOURCE_TENSOR_COUNT
        or len(set(mapping.values())) != SOURCE_SHARDS
    ):
        raise ValueError("Nemotron 3.5 Bolt tensor index differs")
    files = []
    bindings = []
    header_set = hashlib.sha256()
    payload_bytes = 0
    for name in sorted(set(mapping.values())):
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError("Invalid Nemotron 3.5 Bolt shard path")
        shard = path / name
        raw_header, tensors, payload, binding = _validate_safetensors_file(shard)
        if set(tensors) != {key for key, owner in mapping.items() if owner == name}:
            raise ValueError(f"Nemotron 3.5 Bolt shard/index differs: {name}")
        header_set.update(name.encode())
        header_set.update(raw_header)
        payload_bytes += payload
        files.append((name, binding[2], binding[3]))
        bindings.append((name, *binding))
    if sum(size for _, size, _ in files) != SOURCE_SHARD_BYTES:
        raise ValueError("Nemotron 3.5 Bolt shard bytes differ")
    if payload_bytes != SOURCE_TENSOR_BYTES or header_set.hexdigest() != SOURCE_HEADER_SET_SHA256:
        raise ValueError("Unknown Nemotron 3.5 Bolt tensor schema revision")
    identity = hashlib.sha256(
        json.dumps(
            [
                SOURCE_REVISION,
                SOURCE_CONFIG_SHA256,
                SOURCE_QUANT_SHA256,
                SOURCE_INDEX_SHA256,
                SOURCE_HEADER_SET_SHA256,
                SOURCE_METADATA_SHA256,
                bindings,
            ],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {
        "schema": "mlx2.nemotron35-bolt-artifact.v1",
        "path": str(path),
        "identity": {"fingerprint": identity, "source_revision": SOURCE_REVISION},
        "identity_kind": "pinned-metadata-and-exact-headers-with-file-bindings-not-payload-rehash",
        "config": config,
        "text_config": text,
        "files": files,
        "tensor_count": SOURCE_TENSOR_COUNT,
        "tensor_bytes": SOURCE_TENSOR_BYTES,
        "quantization": ["Q8_0", "W4A16_NVFP4", "IQ2_XXS"],
        "runtime_load_supported": False,
        "implemented": "artifact-inspection-contract-only",
        "qualified": False,
        "selected": False,
        "observed_used": False,
        "blocking_execution_laws": ["ggml_q8_0", "ggml_iq2_xxs"],
    }


def validate_lilicorr_config(config: dict, target: dict) -> dict:
    if not isinstance(config, dict) or not isinstance(target, dict):
        raise TypeError("Bolt LiLiCorr config and target must be objects")
    nested = config.get("dflash_config")
    taps = [1, 18, 35, 51, 68, 85]
    if (
        config.get("architectures") != ["LiLiCorrDraftModel"]
        or config.get("model_type") != "qwen3"
        or config.get("hidden_size") != 4096
        or config.get("vocab_size") != 131072
        or config.get("num_hidden_layers") != 5
        or config.get("block_size") != 8
        or config.get("sliding_window") != 1024
        or config.get("use_sliding_window") is not True
        or config.get("layer_types") != ["sliding_attention"] * 5
        or config.get("attention_sink_bias") is not True
        or not isinstance(nested, dict)
        or nested.get("target_layer_ids") != taps
        or nested.get("attention_sink_bias") is not True
        or nested.get("lilicorr_candidate_topk") != 8
        or nested.get("lilicorr_hidden_size") != 1024
        or nested.get("conv_kernel_size") != 2
        or nested.get("conv_group_size") != 16
        or nested.get("swa_window_size") != 1024
    ):
        raise ValueError("Unsupported Nemotron 3.5 Bolt LiLiCorr geometry")
    identity = target.get("identity")
    text = target.get("text_config")
    if (
        target.get("schema") != "mlx2.nemotron35-bolt-artifact.v1"
        or not isinstance(identity, dict)
        or identity.get("source_revision") != SOURCE_REVISION
        or not _is_sha256(identity.get("fingerprint"))
        or not isinstance(text, dict)
    ):
        raise ValueError("Bolt LiLiCorr target artifact contract differs")
    if (
        text.get("hidden_size") != config["hidden_size"]
        or text.get("vocab_size") != config["vocab_size"]
        or len(text.get("layers_block_type", [])) != 88
        or any(type(tap) is not int or not 0 <= tap < 88 for tap in nested["target_layer_ids"])
    ):
        raise ValueError("Bolt LiLiCorr target feature contract differs")
    quant = config.get("quantization_config")
    if (
        not isinstance(quant, dict)
        or quant.get("quant_algo") != "W4A16_NVFP4"
        or quant.get("group_size") != 16
        or quant.get("quant_method") != "modelopt_fp4"
    ):
        raise ValueError("Bolt LiLiCorr NVFP4 contract differs")
    return {**config, "num_target_layers": 88}


def inspect_lilicorr(draft_path, *, target_artifact: dict) -> dict:
    """Pinned source NVFP4 LiLiCorr bound to the source target; inspection only."""
    path = Path(draft_path).expanduser().resolve()
    config_path = path / "config.json"
    quant_path = path / "hf_quant_config.json"
    _require_regular_file(config_path, "LiLiCorr config")
    _require_regular_file(quant_path, "LiLiCorr quantization metadata")
    config_raw, config = _read_json(config_path, "Bolt LiLiCorr config")
    quant_raw, quant = _read_json(quant_path, "Bolt LiLiCorr quantization")
    if (
        hashlib.sha256(config_raw).hexdigest() != LILICORR_CONFIG_SHA256
        or hashlib.sha256(quant_raw).hexdigest() != LILICORR_QUANT_SHA256
    ):
        raise ValueError("Unknown Nemotron 3.5 Bolt LiLiCorr source revision")
    normalized = validate_lilicorr_config(config, target_artifact)
    q = quant.get("quantization")
    if (
        quant.get("producer") != {"name": "modelopt", "version": "0.0.0"}
        or not isinstance(q, dict)
        or q.get("quant_algo") != "W4A16_NVFP4"
        or q.get("group_size") != 16
    ):
        raise ValueError("Bolt LiLiCorr ModelOpt manifest differs")
    file = path / "model.safetensors"
    raw_header, tensors, payload, binding = _validate_safetensors_file(file)
    if hashlib.sha256(raw_header).hexdigest() != LILICORR_HEADER_SHA256:
        raise ValueError("Unknown Nemotron 3.5 Bolt LiLiCorr tensor schema revision")
    dtypes = Counter(item["dtype"] for item in tensors.values())
    if len(tensors) != 179 or dtypes != {"BF16": 104, "F32": 25, "F8_E4M3": 25, "U8": 25}:
        raise ValueError("Bolt LiLiCorr tensor inventory differs")
    required = {
        "layers.0.self_attn.attention_sink_bias": ("BF16", [32]),
        "lilicorr.relative_slot_bias": ("BF16", [8, 15]),
        "lilicorr.token_proj.weight": ("U8", [1024, 2048]),
        "lilicorr.token_proj.weight_scale": ("F8_E4M3", [1024, 256]),
        "lilicorr.token_proj.weight_scale_2": ("F32", []),
    }
    for name, (dtype, shape) in required.items():
        item = tensors.get(name, {})
        if item.get("dtype") != dtype or item.get("shape") != shape:
            raise ValueError(f"Bolt LiLiCorr tensor differs: {name}")
    if binding[2] != LILICORR_TENSOR_BYTES + len(raw_header) + 8 or payload != LILICORR_TENSOR_BYTES:
        raise ValueError("Bolt LiLiCorr payload bytes differ")
    if _sha256(file) != LILICORR_WEIGHT_SHA256 or binding != _source_binding(file):
        raise ValueError("Unknown or changed Nemotron 3.5 Bolt LiLiCorr weights")
    fingerprint = hashlib.sha256(
        json.dumps(
            [
                LILICORR_REVISION,
                LILICORR_CONFIG_SHA256,
                LILICORR_QUANT_SHA256,
                LILICORR_HEADER_SHA256,
                binding,
                target_artifact["identity"]["fingerprint"],
            ],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {
        "schema": "mlx2.nemotron35-bolt-lilicorr-artifact.v1",
        "path": str(path),
        "fingerprint": fingerprint,
        "source_revision": LILICORR_REVISION,
        "config": normalized,
        "tensor_count": len(tensors),
        "tensor_bytes": payload,
        "target_fingerprint": target_artifact["identity"]["fingerprint"],
        "target_layer_ids": normalized["dflash_config"]["target_layer_ids"],
        "runtime_load_supported": False,
        "implemented": "artifact-inspection-and-target-binding-contract-only",
        "qualified": False,
        "selected": False,
        "observed_used": False,
        "blocking_execution_laws": ["attention_sink_bias", "target_ggml_q8_0", "target_ggml_iq2_xxs"],
        "weight_sha256": LILICORR_WEIGHT_SHA256,
        "weight_payload_rehashed": True,
    }


def inspect_artifact(model_path) -> dict:
    """Converted affine artifact, bound to the pinned source and its recipe."""
    from ..conversion.nemotron35_bolt_plan import language_config, make_plan
    from ..conversion.nvfp4 import read_header, sha256

    path = Path(model_path).expanduser().resolve()
    receipt_path = path / "mlx2-conversion.json"
    if not receipt_path.is_file():
        raise ValueError(
            "Nemotron 3.5 Bolt serves only its converted affine artifact; convert the "
            "pinned source with scripts/convert_nemotron35_bolt.py"
        )
    _, config = _read_json(path / "config.json", "Bolt converted config")
    _, receipt = _read_json(receipt_path, "Bolt conversion receipt")
    if (
        receipt.get("schema") != CONVERSION_SCHEMA
        or receipt.get("source_revision") != SOURCE_REVISION
        or receipt.get("source_config_sha256") != SOURCE_CONFIG_SHA256
        or receipt.get("source_quant_sha256") != SOURCE_QUANT_SHA256
        or receipt.get("source_index_sha256") != SOURCE_INDEX_SHA256
        or receipt.get("output_config_sha256") != sha256(path / "config.json")
        or receipt.get("output_index_sha256") != sha256(path / "model.safetensors.index.json")
    ):
        raise ValueError("Nemotron 3.5 Bolt requires a source-bound completed conversion")
    source_config = {k: v for k, v in config.items() if k != "quantization"}
    source_config["quantization_config"] = receipt.get("source_quantization_config")
    validate_target_config(source_config)
    text = language_config(config)
    plan = make_plan(config)
    if (
        config.get("quantization")
        != {"group_size": 32, "bits": 4, "mode": "affine", **plan["modules"]}
        or receipt.get("recipe") != plan["recipe"]
        or receipt.get("tensor_bytes") != plan["tensor_bytes"]
    ):
        raise ValueError("Nemotron 3.5 Bolt mixed precision differs from its recipe")
    mapping = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    if set(mapping) != set(plan["schema"]):
        raise ValueError("Nemotron 3.5 Bolt index has missing or unexpected tensors")
    digest, records = hashlib.sha256(), []
    for name in ("config.json", "model.safetensors.index.json", "mlx2-conversion.json",
                 *SOURCE_METADATA_SHA256):
        item = path / name
        _require_regular_file(item, "metadata")
        if name in SOURCE_METADATA_SHA256 and _sha256(item) != SOURCE_METADATA_SHA256[name]:
            raise ValueError(f"Nemotron 3.5 Bolt metadata differs from source: {name}")
        digest.update(name.encode())
        digest.update(item.read_bytes())
    shards = {item["file"]: item for item in receipt["output_shards"]}
    if set(shards) != set(mapping.values()):
        raise ValueError("Nemotron 3.5 Bolt conversion receipt shard closure differs")
    for name in sorted(shards):
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError("Nemotron 3.5 Bolt shard path escapes artifact")
        item = path / name
        _require_regular_file(item, "shard")
        if item.stat().st_size != shards[name]["bytes"]:
            raise ValueError("Nemotron 3.5 Bolt shard is truncated")
        header, _ = read_header(item)
        tensors = {k: v for k, v in header.items() if k != "__metadata__"}
        if set(tensors) != {k for k, shard in mapping.items() if shard == name}:
            raise ValueError("Nemotron 3.5 Bolt shard/index differs")
        for key, entry in tensors.items():
            if {f: entry[f] for f in ("shape", "dtype")} != plan["schema"][key]:
                raise ValueError(f"Nemotron 3.5 Bolt tensor geometry differs: {key}")
        stat = item.stat()
        record = (name, stat.st_size, stat.st_mtime_ns)
        records.append(record)
        digest.update(json.dumps(record).encode())
        digest.update(json.dumps(header, sort_keys=True).encode())
    return {
        "schema": "mlx2.nemotron35-bolt-converted-artifact.v1",
        "config": config,
        "text_config": text,
        "weight_map": mapping,
        "conversion": receipt,
        "has_mtp": False,
        "quantization": plan["modules"],
        "identity": {
            "path": str(path),
            "fingerprint": digest.hexdigest(),
            "files": records,
            "source_revision": SOURCE_REVISION,
        },
        "qualified": False,
        "selected": False,
        "observed_used": False,
    }


def lilicorr_executable_config(config: dict):
    """LiLiCorr args for the BF16 import: shared target head, sinks, sliding 1024."""
    from .lilicorr import LiLiCorrConfig

    nested = config.get("dflash_config")
    if (
        not isinstance(nested, dict)
        or config.get("quantization_config") is not None
        or config.get("quantization") is not None
        or config.get("has_own_lm_head") is not None
        or config.get("tie_word_embeddings") is not False
    ):
        raise ValueError("Unsupported Nemotron 3.5 Bolt LiLiCorr export")
    fields = LiLiCorrConfig.__dataclass_fields__
    values = {k: v for k, v in config.items() if k in fields}
    values.update({k: v for k, v in nested.items() if k in fields})
    values.update(num_target_layers=88, has_own_lm_head=False, attention_sink_bias=True)
    args = LiLiCorrConfig(**values)
    if (
        tuple(args.target_layer_ids) != LILICORR_TAPS
        or args.block_size != 8
        or args.sliding_window != 1024
        or args.num_key_value_heads != 2
    ):
        raise ValueError("Nemotron 3.5 Bolt LiLiCorr export geometry differs")
    return args


def inspect_lilicorr_drafter(draft_path, *, target_artifact: dict) -> dict:
    """Pinned BF16 LiLiCorr import bound to the converted target identity."""
    path = Path(draft_path).expanduser().resolve()
    if not (path / "mlx2-conversion.json").is_file() and (path / "hf_quant_config.json").is_file():
        raise ValueError(
            "Nemotron 3.5 Bolt LiLiCorr ModelOpt-packed source is not loadable as "
            "shipped; serve its pinned BF16 import (conversion/nvfp4.py --mode bf16). "
            "No native NVFP4 import of the Bolt head is pinned"
        )
    config_raw, config = _read_json(path / "config.json", "Bolt LiLiCorr export config")
    _, receipt = _read_json(path / "mlx2-conversion.json", "Bolt LiLiCorr import receipt")
    if hashlib.sha256(config_raw).hexdigest() != LILICORR_EXPORT_CONFIG_SHA256:
        raise ValueError("Unknown Nemotron 3.5 Bolt LiLiCorr export revision")
    required = {
        "schema": "mlx2.modelopt-nvfp4-import.v1",
        "mode": "bf16",
        "source_config_sha256": LILICORR_CONFIG_SHA256,
        "source_weight_sha256": LILICORR_WEIGHT_SHA256,
        "output_weight_sha256": LILICORR_EXPORT_WEIGHT_SHA256,
        "quantized_matrices": 25,
        "global_scale_preserved": True,
    }
    if any(receipt.get(k) != v for k, v in required.items()):
        raise ValueError("Nemotron 3.5 Bolt LiLiCorr import receipt differs")
    file = path / "model.safetensors"
    raw_header, tensors, payload, binding = _validate_safetensors_file(file)
    if (
        hashlib.sha256(raw_header).hexdigest() != LILICORR_EXPORT_HEADER_SHA256
        or {item["dtype"] for item in tensors.values()} != {"BF16"}
        or payload != receipt.get("output_tensor_bytes")
    ):
        raise ValueError("Nemotron 3.5 Bolt LiLiCorr export tensors differ")
    args = lilicorr_executable_config(config)
    identity = target_artifact.get("identity", {})
    if (
        target_artifact.get("schema") != "mlx2.nemotron35-bolt-converted-artifact.v1"
        or identity.get("source_revision") != SOURCE_REVISION
        or not _is_sha256(identity.get("fingerprint"))
        or target_artifact["text_config"]["hidden_size"] != args.hidden_size
        or target_artifact["text_config"]["vocab_size"] != args.vocab_size
    ):
        raise ValueError("Bolt LiLiCorr target artifact contract differs")
    fingerprint = hashlib.sha256(
        json.dumps(
            [LILICORR_EXPORT_CONFIG_SHA256, LILICORR_EXPORT_HEADER_SHA256, binding,
             identity["fingerprint"]],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {
        "schema": "mlx2.nemotron35-bolt-lilicorr-drafter.v1",
        "path": str(path),
        "fingerprint": fingerprint,
        "config": config,
        "args": args,
        "source_binding": binding,
        "target_fingerprint": identity["fingerprint"],
        "target_layer_ids": list(LILICORR_TAPS),
        "expected_weight_sha256": LILICORR_EXPORT_WEIGHT_SHA256,
        "attention_sinks": "implemented-gpt-oss-sink-law",
        "owned_lm_head": False,
        "runtime_load_supported": True,
        "qualified": False,
        "selected": False,
        "observed_used": False,
    }


def load_lilicorr_drafter(record, target_model, *, target_contract):
    refreshed = inspect_lilicorr_drafter(
        record["path"], target_artifact=record["_target_artifact"]
    )
    if any(refreshed[k] != record[k] for k in ("fingerprint", "source_binding")):
        raise ValueError("Nemotron 3.5 Bolt LiLiCorr changed since inspection")
    file = Path(record["path"]) / "model.safetensors"
    if _sha256(file) != record["expected_weight_sha256"]:
        raise ValueError("Nemotron 3.5 Bolt LiLiCorr bytes differ from the pinned import")

    import mlx.core as mx

    from ..runtime.drafters.lilicorr import LiLiCorrDraftModel

    model = LiLiCorrDraftModel(record["args"])
    weights = model.sanitize(mx.load(str(file)))
    model.load_weights(list(weights.items()), strict=True)
    model.runtime_weight_quantization = None
    model.eval()
    mx.eval(model.parameters())
    weights.clear()
    mx.clear_cache()
    return model.bind(target_model, target_contract=target_contract)


def routed_exact_geometries(text: dict) -> list:
    """Bolt's Puzzle-NAS MoE geometries, admitted to the exact routed batch.

    One entry per distinct (experts, width) pair in ``block_configs``, in
    Nemotron-H's (hidden, experts, top-k, latent, width, shared) order.  The
    declaration only admits a layer to the strip; each layer's router,
    projections, gather and combine still pass their own parity canaries.
    """
    return sorted(
        {
            (
                int(text["hidden_size"]),
                int(block["n_routed_experts"]),
                int(block["num_experts_per_tok"]),
                int(text["moe_latent_size"]),
                int(block["moe_intermediate_size"]),
                int(text["moe_shared_expert_intermediate_size"]),
            )
            for block in text["block_configs"]
            if block["block_type"] == "moe"
        }
    )


class Nemotron35BoltAdapter(Nemotron35SuperAdapter):
    """Super's serving contract over Bolt's converted Puzzle-NAS artifact."""

    descriptor = DESCRIPTOR
    sampling_defaults = SAMPLING
    artifact_inspector = staticmethod(inspect_artifact)
    default_route = "ordinary"
    multimodal_routes = frozenset()
    # With no drafter flag the server selects latched prompt lookup
    # (capability default); its verify rows then run the exact short-segment
    # kernels Super's native route uses (nemotron_h.Model.pld_verify_forward).
    # pld-families-20261021 at c7aa1e237: greedy output identical to
    # --ordinary on code, copy, 1.5K-token copy and Spomin 4K prompts;
    # decode 1.90x (copy) and 2.62x (long copy) of ordinary, 1.02x code,
    # 0.77x Spomin (latch probes).  Without these kernels the verify is
    # tokenwise and PLD runs at 0.82-1.01x.  An operator policy naming any
    # key wins.  Not a qualification.
    adapter_owned_default_policy_keys = frozenset({
        "mamba_conv", "mamba_projection", "mamba_segment", "mamba_ssm",
    })
    default_route_execution_policy = {
        "prompt_lookup": {
            "mamba_conv": "fused_short",
            "mamba_projection": "short_row_exact",
            "mamba_segment": "layerwise_short_exact_routed_batch",
            "mamba_ssm": "fused_short_gate",
        },
    }
    cache_layout = CACHE_LAYOUT
    # llm_config declares </s> (2); generation_config adds <|im_end|> (11).
    eos_token_ids = (2, 11)
    EXTERNAL_ROUTE_TAG = "bolt-external-v1"
    EXTERNAL_PROFILE = "nemotron35-bolt-apcv2-external"
    EXTERNAL_POLICY_EXTENSIONS = frozenset()
    incremental_tokenizer_renderer_revision = "nemotron35-bolt-renderer-v1"

    @staticmethod
    def _runtime_language_config(config):
        from ..conversion.nemotron35_bolt_plan import language_config

        return language_config(config)

    def _declare_model_contracts(self, model, text):
        model.declare_routed_exact_geometries(routed_exact_geometries(text))

    @staticmethod
    def _runtime_model_class():
        from ..runtime.models.nemotron35_bolt import Model

        return Model

    def _external_drafter(self, model_path, artifact, draft_weight_quantization):
        if draft_weight_quantization is not None:
            raise ValueError("Nemotron 3.5 Bolt LiLiCorr has no runtime quantization profile")
        record = inspect_lilicorr_drafter(
            self.external_policy["draft_model"], target_artifact=artifact
        )
        record["_target_artifact"] = copy.deepcopy(artifact)
        contract = self.nemotron_lilicorr_target_contract(record)

        def loader(item, target_model):
            return load_lilicorr_drafter(item, target_model, target_contract=contract)

        return record, loader

    def prepare_multimodal_request(self, request, *, file_loader=None):
        for message in request.get("messages", ()):
            content = message.get("content")
            if isinstance(content, list) and any(
                part.get("type") != "text" for part in content
            ):
                raise ValueError("Nemotron 3.5 Bolt serving is text-only")
        return super().prepare_multimodal_request(request, file_loader=file_loader)

    def profile_name(self, mtp):
        if mtp:
            raise ValueError("Nemotron 3.5 Bolt has no native MTP head")
        return "nemotron35-bolt-affine4g32-apcv2-ordinary"

    def execution_numerics_contract(self):
        return {
            "router_compute": "float32",
            "recurrent_state": "float32",
            "time_step_limit": [0.0, "infinity"],
            "source_revision": SOURCE_REVISION,
            "quantization_recipe": self.conversion["recipe"],
            "qualification": "unqualified",
        }

    def nemotron_lilicorr_target_contract(self, draft_record):
        if draft_record["target_fingerprint"] != self.identity["fingerprint"]:
            raise ValueError("Nemotron 3.5 Bolt LiLiCorr target revision changed")
        taps = tuple(draft_record["args"].target_layer_ids)
        return {
            "schema": "mlx2.nemotron-lilicorr-target.v1",
            "target_model_type": "nemotron_h_omni",
            "target_fingerprint": self.identity["fingerprint"],
            "target_layer_ids": taps,
            "target_feature_width": len(taps) * draft_record["args"].hidden_size,
            "capture": "post-layer-pre-final-norm",
            "ordinary_reference_preserved": True,
            "speculative_rollback": "exact",
            "owned_lm_head": False,
            "owned_final_norm": True,
            "qualified": False,
        }

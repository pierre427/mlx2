"""GLM-5.2 MoE DSA ordinary artifact candidate, gated on complete weights.

The pinned unified GLM class inherits DeepSeek V3.2's per-layer indexer and
does not implement this checkpoint's shared-indexer schedule. Inspection is
CPU-only; this module intentionally exposes no serving adapter constructor.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane

SOURCE_REVISION = "1104ced19ed98800bdaf4ebcdca14bbdeb597c23"
MODEL_TYPE = "glm_moe_dsa"
EXPECTED = {
    "model_type": MODEL_TYPE, "architectures": ["GlmMoeDsaForCausalLM"],
    "vocab_size": 154880, "hidden_size": 6144,
    "num_hidden_layers": 78, "num_attention_heads": 64,
    "num_key_value_heads": 64, "head_dim": 192,
    "index_head_dim": 128, "index_n_heads": 32,
    "index_topk": 2048, "index_topk_freq": 4,
    "index_skip_topk_offset": 3, "indexer_rope_interleave": True,
    "kv_lora_rank": 512, "q_lora_rank": 2048,
    "qk_nope_head_dim": 192, "qk_rope_head_dim": 64,
    "v_head_dim": 256, "n_routed_experts": 256,
    "n_shared_experts": 1, "num_experts_per_tok": 8,
    "first_k_dense_replace": 3, "moe_layer_freq": 1,
    "intermediate_size": 12288, "moe_intermediate_size": 2048,
    "routed_scaling_factor": 2.5, "topk_method": "noaux_tc",
    "scoring_func": "sigmoid", "num_nextn_predict_layers": 1,
    "index_share_for_mtp_iteration": True,
}

DESCRIPTOR = ModelDescriptor(
    model_type=MODEL_TYPE, family="glm-5.2-moe-dsa",
    variant="ordinary-artifact-candidate",
    state_planes=frozenset({StatePlane.ATTENTION_KV, StatePlane.RNG,
                            StatePlane.TRANSCRIPT}),
    capabilities=frozenset({Capability.TEXT}),
    cache_layout="glm52-dsa-shared-indexer-unimplemented-v1",
    metadata={"qualification": "blocked", "source_revision": SOURCE_REVISION,
              "source_gap": "unified DeepSeek V3.2 inherits per-layer indexer and ignores shared indexer schedule",
              "mtp": "separate-unqualified"},
)


def _json(path: Path) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r} in {path.name}")
            result[key] = value
        return result

    value = json.loads(path.read_text(), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    return value


def validate_topology(config: dict) -> None:
    if any(config.get(key) != value for key, value in EXPECTED.items()):
        raise ValueError("GLM-5.2 DSA topology mismatch")
    expected_indexers = ["full" if i < 3 or i % 4 == 2 else "shared"
                         for i in range(78)]
    if config.get("indexer_types") != expected_indexers:
        raise ValueError("GLM-5.2 shared-indexer schedule mismatch")
    if config.get("mlp_layer_types") != ["dense"] * 3 + ["sparse"] * 75:
        raise ValueError("GLM-5.2 dense/MoE schedule mismatch")
    if config.get("rope_parameters") != {"rope_theta": 8000000,
                                          "rope_type": "default"}:
        raise ValueError("GLM-5.2 rotary geometry mismatch")


def _required_tensors() -> set[str]:
    required = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    attention = (
        "q_a_proj.weight", "q_a_layernorm.weight", "q_b_proj.weight",
        "kv_a_proj_with_mqa.weight", "kv_a_layernorm.weight",
        "kv_b_proj.weight", "o_proj.weight",
    )
    indexer = ("k_norm.bias", "k_norm.weight", "weights_proj.weight",
               "wk.weight", "wq_b.weight")
    for layer in range(78):
        prefix = f"model.layers.{layer}."
        required.update({prefix + "input_layernorm.weight",
                         prefix + "post_attention_layernorm.weight"})
        required.update(prefix + "self_attn." + suffix for suffix in attention)
        if layer < 3 or layer % 4 == 2:
            required.update(prefix + "self_attn.indexer." + suffix
                            for suffix in indexer)
        if layer < 3:
            required.update(prefix + "mlp." + projection + ".weight"
                            for projection in ("gate_proj", "up_proj", "down_proj"))
        else:
            required.update(prefix + "mlp.gate." + suffix for suffix in
                            ("weight", "e_score_correction_bias"))
            required.update(prefix + "mlp.shared_experts." + projection + ".weight"
                            for projection in ("gate_proj", "up_proj", "down_proj"))
            required.update(prefix + f"mlp.experts.{expert}.{projection}.weight"
                            for expert in range(256)
                            for projection in ("gate_proj", "up_proj", "down_proj"))
    return required


def _header(path: Path) -> dict:
    size = path.stat().st_size
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError("truncated GLM safetensors shard")
        length = struct.unpack("<Q", raw)[0]
        if not 0 < length <= min(64 << 20, size - 8):
            raise ValueError("invalid GLM safetensors header")
        header = json.loads(stream.read(length))
    if not isinstance(header, dict):
        raise ValueError("invalid GLM safetensors header object")
    intervals = []
    data_bytes = size - 8 - length
    for key, value in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(value, dict) or not isinstance(value.get("dtype"), str):
            raise ValueError(f"invalid GLM tensor header: {key}")
        shape = value.get("shape")
        offsets = value.get("data_offsets")
        if (not isinstance(shape, list) or not all(type(v) is int and v >= 0 for v in shape)
                or not isinstance(offsets, list) or len(offsets) != 2
                or not all(type(v) is int for v in offsets)
                or not 0 <= offsets[0] < offsets[1] <= data_bytes):
            raise ValueError(f"invalid GLM tensor offsets or shape: {key}")
        intervals.append(tuple(offsets))
    intervals.sort()
    if any(a[1] > b[0] for a, b in zip(intervals, intervals[1:])):
        raise ValueError("overlapping GLM safetensors payloads")
    return header


def inspect_artifact(model_path: str | Path) -> dict:
    path = Path(model_path).expanduser().resolve()
    config = _json(path / "config.json")
    validate_topology(config)
    index_path = path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise ValueError("GLM-5.2 requires a complete indexed checkpoint")
    weight_map = _json(index_path).get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("GLM-5.2 weight index is empty")
    if not (path / "tokenizer.json").is_file() or not (path / "tokenizer_config.json").is_file():
        raise ValueError("GLM-5.2 tokenizer metadata is incomplete")
    missing = _required_tensors() - weight_map.keys()
    if missing:
        raise ValueError(f"GLM-5.2 ordinary trunk is incomplete: {len(missing)} tensors")
    shards = sorted(set(weight_map.values()))
    actual = {}
    records = []
    for name in shards:
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError("unsafe GLM-5.2 shard name")
        item = path / name
        if not item.is_file() or not item.resolve().is_relative_to(path):
            raise ValueError(f"missing or foreign GLM-5.2 shard: {name}")
        header = _header(item)
        for key in header:
            if key == "__metadata__":
                continue
            if key in actual:
                raise ValueError(f"duplicate GLM-5.2 tensor across shards: {key}")
            actual[key] = name
        stat = item.stat()
        records.append((name, stat.st_size, stat.st_mtime_ns))
    if actual != weight_map:
        raise ValueError("GLM-5.2 index and safetensors headers disagree")
    digest = hashlib.sha256()
    for name in ("config.json", "model.safetensors.index.json",
                 "tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        item = path / name
        if item.is_file():
            digest.update(name.encode())
            digest.update(item.read_bytes())
    for record in records:
        digest.update(json.dumps(record).encode())
    return {"config": config, "identity": {"path": str(path),
            "fingerprint": digest.hexdigest(), "files": records},
            "weight_map": weight_map, "ordinary_tensor_count": len(_required_tensors()),
            "embedded_mtp_tensor_count": sum(key.startswith("model.layers.78.")
                                             for key in weight_map),
            "qualified": False, "selected": False}

"""Qwen3.5 122B ordinary text route and separate MTP candidate loader."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from .qwen36_35b import Qwen3635BA3BAdapter, configure_environment
from .qwen38_27b import resolve_eos_token_ids

CACHE_LAYOUT = "qwen35-122b-a10b-hybrid-layer-segments-v1"
EXPECTED = {
    "num_hidden_layers": 48, "hidden_size": 3072,
    "num_attention_heads": 32, "num_key_value_heads": 2,
    "head_dim": 256, "full_attention_interval": 4,
    "vocab_size": 248320, "num_experts": 256,
    "num_experts_per_tok": 8, "moe_intermediate_size": 1024,
    "shared_expert_intermediate_size": 1024,
    "linear_num_key_heads": 16, "linear_num_value_heads": 64,
    "linear_key_head_dim": 128, "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "attn_output_gate": True, "rms_norm_eps": 1e-6,
}

QWEN35_122B = ModelDescriptor(
    model_type="qwen3_5_moe", family="qwen3.5-122b-a10b", variant="ordinary-text",
    state_planes=frozenset({
        StatePlane.ATTENTION_KV, StatePlane.RECURRENT,
        StatePlane.RNG, StatePlane.TRANSCRIPT,
    }),
    capabilities=frozenset({
        Capability.TEXT, Capability.STREAMING, Capability.CONTINUOUS_BATCH,
        Capability.PREFIX_REUSE, Capability.APC_V2, Capability.LAYERED_CACHE,
    }),
    cache_layout=CACHE_LAYOUT,
    metadata={
        "execution": "mlx2.adapters.qwen35_122b.Qwen35122BA10BAdapter",
        "qualification": "pending", "scope": "text-only ordinary decode",
        "embedded_mtp": "offline-candidate-unqualified", "vision": "direct-candidate-unqualified",
    },
)


def inspect_artifact(model_path: str | Path) -> dict:
    path = Path(model_path).expanduser().resolve()
    config = json.loads((path / "config.json").read_text())
    text = config.get("text_config")
    if config.get("model_type") != "qwen3_5_moe" or not isinstance(text, dict):
        raise ValueError("Qwen3.5 122B requires qwen3_5_moe text configuration")
    if any(text.get(key) != value for key, value in EXPECTED.items()):
        raise ValueError("artifact topology does not match Qwen3.5 122B-A10B")
    expected_layers = [
        "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
        for i in range(48)
    ]
    if text.get("layer_types") != expected_layers:
        raise ValueError("artifact layer order does not match Qwen3.5 122B-A10B")
    rope = text.get("rope_parameters")
    if not isinstance(rope, dict) or any(rope.get(k) != v for k, v in {
        "type": "default", "rope_theta": 10000000,
        "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10],
        "mrope_interleaved": True,
    }.items()):
        raise ValueError("artifact rotary geometry does not match Qwen3.5 122B-A10B")
    if text.get("mtp_num_hidden_layers") != 1 or text.get("mtp_use_dedicated_embeddings") is not False:
        raise ValueError("unexpected embedded MTP topology")
    quant = config.get("quantization")
    if not isinstance(quant, dict) or any(quant.get(k) != v for k, v in {
        "bits": 4, "group_size": 64, "mode": "affine",
    }.items()):
        raise ValueError("unsupported Qwen3.5 122B quantization")
    for name, setting in quant.items():
        if name in {"bits", "group_size", "mode"}:
            continue
        if not isinstance(setting, dict) or setting.get("group_size") not in {64, 128} or setting.get("mode") != "affine" or setting.get("bits") not in {4, 5, 6, 8}:
            raise ValueError(f"unsupported quantization override: {name}")
    index_path = path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text()).get("weight_map")
    if not isinstance(index, dict) or not index:
        raise ValueError("artifact has no indexed weights")
    required = {
        "language_model.model.embed_tokens.weight",
        "language_model.model.norm.weight", "language_model.lm_head.weight",
    }
    for layer in range(48):
        stem = f"language_model.model.layers.{layer}."
        required.update({
            stem + "input_layernorm.weight", stem + "post_attention_layernorm.weight",
            stem + "mlp.gate.weight", stem + "mlp.switch_mlp.down_proj.weight",
            stem + "mlp.shared_expert.gate_proj.weight",
        })
        required.add(stem + ("self_attn.q_proj.weight" if (layer + 1) % 4 == 0 else "linear_attn.in_proj_qkv.weight"))
    if not required.issubset(index):
        raise ValueError("artifact index lacks required 122B text tensors")
    mtp_keys = [k for k in index if k.startswith("language_model.mtp.")]
    vision_keys = [k for k in index if k.startswith("vision_tower.")]
    if not mtp_keys or not vision_keys:
        raise ValueError("artifact sidecar tensor layout differs from pinned 122B checkpoint")
    mtp_required = {
        "language_model.mtp.fc.weight", "language_model.mtp.norm.weight",
        "language_model.mtp.pre_fc_norm_embedding.weight",
        "language_model.mtp.pre_fc_norm_hidden.weight",
        "language_model.mtp.layers.0.self_attn.q_proj.weight",
        "language_model.mtp.layers.0.self_attn.k_proj.weight",
        "language_model.mtp.layers.0.self_attn.v_proj.weight",
        "language_model.mtp.layers.0.self_attn.o_proj.weight",
        "language_model.mtp.layers.0.mlp.gate.weight",
        "language_model.mtp.layers.0.mlp.switch_mlp.down_proj.weight",
    }
    if not mtp_required.issubset(index):
        raise ValueError("artifact index lacks embedded 122B MTP tensors")
    vision = config.get("vision_config")
    if not isinstance(vision, dict) or any(vision.get(k) != v for k, v in {
        "depth": 27, "hidden_size": 1152, "num_heads": 16,
        "out_hidden_size": 3072, "patch_size": 16,
        "spatial_merge_size": 2, "temporal_patch_size": 2,
    }.items()):
        raise ValueError("unsupported embedded 122B vision topology")
    vision_required = {
        "vision_tower.blocks.0.attn.qkv.weight",
        "vision_tower.blocks.26.attn.qkv.weight",
    }
    if not vision_required.issubset(index):
        raise ValueError("artifact index lacks embedded 122B vision tensors")
    names = sorted(set(index.values()))
    records = []
    for name in names:
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError("unsafe 122B weight shard path")
        item = path / name
        if not item.is_file() or item.stat().st_size < 8:
            raise ValueError(f"missing weight shard: {name}")
        stat = item.stat()
        records.append((name, stat.st_size, stat.st_mtime_ns))
    digest = hashlib.sha256()
    for name in (
        "config.json", "model.safetensors.index.json", "tokenizer.json",
        "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
    ):
        item = path / name
        if item.is_file():
            digest.update(name.encode())
            digest.update(item.read_bytes())
    for record in records:
        digest.update(json.dumps(record).encode())
    return {
        "config": config, "weight_map": index, "has_mtp": False,
        "embedded_mtp_tensor_count": len(mtp_keys),
        "embedded_vision_tensor_count": len(vision_keys),
        "qualification": "pending",
        "identity": {"path": str(path), "fingerprint": digest.hexdigest(), "files": records},
    }


class Qwen35122BA10BAdapter(Qwen3635BA3BAdapter):
    default_route = "ordinary"
    descriptor = QWEN35_122B
    sampling_defaults = None
    default_route_execution_policy = {}
    default_mtp_ordinary_handoff_max_width = None

    def __init__(self, model_path: str, *, execution_policy=None, require_mtp=False):
        if require_mtp:
            raise ValueError("Qwen3.5 122B embedded MTP is not implemented")
        if execution_policy not in (None, {}):
            raise ValueError("Qwen3.5 122B has no qualified execution policy")
        artifact = inspect_artifact(model_path)
        self.identity = artifact["identity"]
        self.config = artifact["config"]
        self.descriptor = QWEN35_122B
        self.environment = configure_environment()
        self.layout = CACHE_LAYOUT
        self._tables = []
        self._num_draft = 0
        self._kernels = {}
        self.mtp_norm_repairs = []
        self.mtp_norm_means = {}
        path = Path(self.identity["path"])
        config = json.loads(json.dumps(artifact["config"]))
        config["text_config"]["mtp_num_hidden_layers"] = 0
        import mlx.core as mx
        import mlx.nn as nn
        from transformers import AutoTokenizer
        from ..runtime.models.qwen35_122b import Model, ModelArgs
        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper
        from ..runtime.ubc_evict import load_shards_evicting

        self.model = Model(ModelArgs.from_dict(config))
        files = [path / name for name in sorted(set(artifact["weight_map"].values()))]
        weights = self.model.sanitize(
            load_shards_evicting(files, sanitize=self.model.shard_prune)
        )
        quant = config["quantization"]
        def predicate(name, module):
            if name in quant:
                return quant[name]
            return hasattr(module, "to_quantized") and f"{name}.scales" in weights
        nn.quantize(
            self.model, group_size=quant["group_size"], bits=quant["bits"],
            mode=quant["mode"], class_predicate=predicate,
        )
        self.model.load_weights(list(weights.items()), strict=True)
        self.model.eval()
        mx.eval(self.model.parameters())
        weights.clear()
        mx.clear_cache()
        self._record_load_dtype()
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        # Some tokenizer classes rebuild the pre-tokenizer instead of reading tokenizer.json.
        from ..runtime.tokenizer_integrity import repair_loaded_tokenizer

        self.pretokenizer_receipt = repair_loaded_tokenizer(tokenizer, path)
        eos = resolve_eos_token_ids(config, tokenizer)
        self.tokenizer = TokenizerWrapper(
            tokenizer, detokenizer_class=BPEStreamingDetokenizer, eos_token_ids=eos
        )
        self.max_context = int(config["text_config"]["max_position_embeddings"])

    def profile_name(self, mtp):
        if mtp:
            raise ValueError("Qwen3.5 122B MTP route is unavailable")
        return "qwen35-122b-a10b-apcv2-ordinary"

    def cache_budget(self, *, mtp):
        if mtp:
            raise ValueError("Qwen3.5 122B MTP route is unavailable")
        from .qwen38_memory import Qwen38CacheBudget
        # Unqualified conservative planning charge, not a measured bound.
        text = self.config["text_config"]
        return replace(
            Qwen38CacheBudget.from_config(text, mtp=False),
            transient_gib_per_lane=8.0,
        )

    def approximate_kv_operations(self):
        return {}

    def diagnostics(self):
        result = super().diagnostics()
        result.update({
            "architecture": "qwen3.5-122b-a10b-hybrid-moe",
            "layout": self.layout, "embedded_mtp_selected": False,
            "vision_selected": False, "qualification": "pending",
        })
        return result



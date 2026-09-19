"""GPU-free inspection plus the Qwen3.6 35B-A3B serving adapter."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from .qwen38_27b import Qwen3827BAdapter

CACHE_LAYOUT = "qwen36-35b-a3b-hybrid-layer-segments-v1"


def descriptor_for(*, has_mtp: bool) -> ModelDescriptor:
    capabilities = {
        Capability.TEXT,
        Capability.STREAMING,
        Capability.TOOLS,
        Capability.REASONING,
        Capability.CONTINUOUS_BATCH,
        Capability.PREFIX_REUSE,
        Capability.APC_V2,
        Capability.LAYERED_CACHE,
        Capability.GRAMMAR,
        Capability.PROMPT_LOOKUP,
    }
    planes = {
        StatePlane.ATTENTION_KV,
        StatePlane.RECURRENT,
        StatePlane.RNG,
        StatePlane.TRANSCRIPT,
        StatePlane.GRAMMAR,
    }
    if has_mtp:
        capabilities.update({Capability.MTP, Capability.SEGMENTED_MTP})
        planes.add(StatePlane.DRAFT)
    return ModelDescriptor(
        model_type="qwen3_5_moe",
        family="qwen3.6-35b-a3b",
        variant="native-mtp" if has_mtp else "ordinary",
        state_planes=frozenset(planes),
        capabilities=frozenset(capabilities),
        cache_layout=CACHE_LAYOUT,
        metadata={
            "execution": "mlx2.adapters.qwen36_35b.Qwen3635BA3BAdapter",
            "qualification": "pending",
            "scope": "text-only",
            "compiled_decode": "implemented-upstream-not-selected",
            "moe_kernels": "implemented-shared-not-selected",
        },
    )


QWEN36_35B = descriptor_for(has_mtp=False)


def inspect_artifact(model_path: str | Path) -> dict:
    path = Path(model_path).expanduser().resolve()
    config = json.loads((path / "config.json").read_text())
    text = config.get("text_config", config)
    if config.get("model_type") != "qwen3_5_moe":
        raise ValueError("Qwen3.6 35B-A3B requires qwen3_5_moe")
    expected = {
        "num_hidden_layers": 40,
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "full_attention_interval": 4,
        "vocab_size": 248320,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
    }
    if any(text.get(key) != value for key, value in expected.items()):
        raise ValueError("artifact topology does not match Qwen3.6 35B-A3B")
    if text.get("mtp_num_hidden_layers", 0) not in (0, 1):
        raise ValueError("only the single-layer Qwen3.6 MTP head is implemented")
    weight_map = json.loads((path / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("artifact has no indexed weights")
    shards = sorted(set(weight_map.values()))
    for name in shards:
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("weight shard paths must stay within the artifact")
        if not (path / name).is_file():
            raise ValueError(f"missing weight shard: {name}")
    mtp_keys = [key for key in weight_map if key.startswith(("language_model.mtp.", "mtp."))]
    has_mtp = bool(mtp_keys)
    if has_mtp and text.get("mtp_num_hidden_layers") != 1:
        raise ValueError("MTP tensors and configured head count disagree")
    if has_mtp:
        normalized = {key.removeprefix("language_model.") for key in mtp_keys}
        required = {
            "mtp.fc.weight",
            "mtp.norm.weight",
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight",
            "mtp.layers.0.self_attn.q_proj.weight",
            "mtp.layers.0.self_attn.k_proj.weight",
            "mtp.layers.0.self_attn.v_proj.weight",
            "mtp.layers.0.self_attn.o_proj.weight",
            "mtp.layers.0.mlp.gate.weight",
            "mtp.layers.0.mlp.switch_mlp.down_proj.weight",
        }
        if not required <= normalized or not (
            "mtp.layers.0.mlp.switch_mlp.gate_up_proj.weight" in normalized
            or {
                "mtp.layers.0.mlp.switch_mlp.gate_proj.weight",
                "mtp.layers.0.mlp.switch_mlp.up_proj.weight",
            } <= normalized
        ):
            raise ValueError("embedded MTP head is incomplete")
    digest = hashlib.sha256()
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
    ):
        item = path / name
        if item.is_file():
            digest.update(name.encode())
            digest.update(item.read_bytes())
    records = []
    for name in shards:
        stat = (path / name).stat()
        record = (name, stat.st_size, stat.st_mtime_ns)
        records.append(record)
        digest.update(json.dumps(record).encode())
    return {
        "config": config,
        "weight_map": weight_map,
        "has_mtp": has_mtp,
        "mtp_tensor_count": len(mtp_keys),
        "identity": {"path": str(path), "fingerprint": digest.hexdigest(), "files": records},
    }


def configure_environment() -> dict[str, str]:
    profile = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "MLX_ENABLE_TF32": "0",
        "MLX_GDN_PACKED": "1",
        "MLX_GDN_CORE": "0",
        "MLX_QWEN36_FUSED_GDN_DECODE": "0",
        "MLX_LM_COMPILED_DECODE": "0",
        "MLX_QWEN4_MOE_FUSED_GATE_UP": "0",
        "MLX_QWEN4_MOE_ROUTER_KERNEL": "0",
        "MLX_QWEN4_FUSED_EXPERT_KERNEL": "stock",
        "MLX_LM_SEGMENTED_SELF_MTP": "1",
        "MLX_LM_TRUE_BATCHED_SEGMENTED_MTP": "1",
        "MLX_LM_SHARED_QSA_SUFFIX": "0",
    }
    for name in tuple(os.environ):
        if name.startswith(("MLX_QWEN", "MLX_LM_", "MLXUAG_", "MLX_GDN_")):
            del os.environ[name]
    os.environ.update(profile)
    return profile


class Qwen3635BA3BAdapter(Qwen3827BAdapter):
    descriptor = QWEN36_35B

    def __init__(self, model_path: str, *, require_mtp: bool = False, execution_policy=None):
        if execution_policy is not None and not isinstance(execution_policy, dict):
            raise ValueError("execution policy must be a JSON object")
        policy = {} if execution_policy is None else dict(execution_policy)
        if set(policy) - {"num_draft"}:
            raise ValueError("Qwen3.6 execution policy supports only num_draft")
        self._num_draft = policy.get("num_draft", 2)
        if type(self._num_draft) is not int or not 1 <= self._num_draft <= 3:
            raise ValueError("num_draft must be 1, 2, or 3")
        artifact = inspect_artifact(model_path)
        if require_mtp and not artifact["has_mtp"]:
            raise ValueError("requested MTP requires embedded head weights")
        self.identity = artifact["identity"]
        self.descriptor = descriptor_for(has_mtp=artifact["has_mtp"])
        self.environment = configure_environment()
        self.layout = CACHE_LAYOUT
        self._tables = []
        path = Path(self.identity["path"])
        config = dict(artifact["config"])
        config["text_config"] = dict(config.get("text_config", config))
        if not artifact["has_mtp"]:
            config["text_config"]["mtp_num_hidden_layers"] = 0

        import mlx.core as mx
        import mlx.nn as nn
        from transformers import AutoTokenizer
        from ..runtime.models.qwen36_35b import Model, ModelArgs
        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper
        from ..runtime.ubc_evict import ubc_evict_paths

        self.model = Model(ModelArgs.from_dict(config))
        weights = {}
        files = [path / name for name in sorted(set(artifact["weight_map"].values()))]
        for file in files:
            weights.update(mx.load(str(file)))
        weights = self.model.sanitize(weights)
        quant = config.get("quantization", config.get("quantization_config"))
        if quant:
            def predicate(name, module):
                if name in quant:
                    return quant[name]
                return hasattr(module, "to_quantized") and f"{name}.scales" in weights
            nn.quantize(
                self.model,
                group_size=quant["group_size"],
                bits=quant["bits"],
                mode=quant.get("mode", "affine"),
                class_predicate=predicate,
            )
        self.model.load_weights(list(weights.items()), strict=True)
        self.model.eval()
        mx.eval(self.model.parameters())
        weights.clear()
        ubc_evict_paths([str(file) for file in files])
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        eos = config.get("eos_token_id", config["text_config"].get("eos_token_id"))
        if isinstance(eos, int):
            eos = [eos]
        self.tokenizer = TokenizerWrapper(
            tokenizer, detokenizer_class=BPEStreamingDetokenizer, eos_token_ids=eos
        )
        self.max_context = int(config["text_config"]["max_position_embeddings"])

    def profile_name(self, mtp):
        if mtp and Capability.MTP not in self.descriptor.capabilities:
            raise ValueError("requested MTP requires embedded head weights")
        return f"qwen36-35b-a3b-apcv2-{'mtp' + str(self._num_draft) if mtp else 'ordinary'}"

    def diagnostics(self):
        result = super().diagnostics()
        result["architecture"] = "sparse-moe-hybrid-gdn-gqa"
        result["layout"] = self.layout
        result["optimized_moe_selected"] = False
        result["compiled_decode_selected"] = False
        return result

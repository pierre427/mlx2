"""HY V3 full and REAP-pruned ordinary text decode candidates."""

from __future__ import annotations

import os
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from .ordinary_artifact import inspect_indexed_artifact
from .ordinary_text import OrdinaryTextAdapter
from ..sampling_defaults import GENERATION_CONFIG, SamplingDefaults, VendorSampling
from ..process_env import (
    PROCESS_NUMERICS,
    clear_inherited_profile,
    require_process_numerics,
)

CACHE_LAYOUT = "hy-v3-full-kv-v1"
# Hy3 generation_config.json: do_sample true, temperature 0.9, top_p 1,
# top_k -1 (off, the neutral 0 here).
SAMPLING = VendorSampling.single(
    SamplingDefaults(temperature=0.9, top_p=1.0, source=GENERATION_CONFIG,
                     note="do_sample=true; top_k -1 (off)"),
    model="Hy3 local artifacts",
)


def descriptor_for(*, reap: bool) -> ModelDescriptor:
    return ModelDescriptor(
        model_type="hy_v3", family="hy-v3", variant="reap50-4bit-ordinary" if reap else "6bit-ordinary",
        state_planes=frozenset({StatePlane.ATTENTION_KV, StatePlane.RNG,
                                StatePlane.TRANSCRIPT}),
        capabilities=frozenset({Capability.TEXT, Capability.STREAMING,
                                Capability.CONTINUOUS_BATCH, Capability.PREFIX_REUSE,
                                Capability.APC_V2}),
        cache_layout=CACHE_LAYOUT,
        metadata={"execution": "mlx2.adapters.hy_v3.HYV3Adapter",
                  "qualification": "pending", "scope": "text-only", "mtp": "absent"},
    )


def inspect_artifact(model_path: str | Path) -> dict:
    artifact = inspect_indexed_artifact(model_path)
    config = artifact["config"]
    if config.get("model_type") != "hy_v3":
        raise ValueError("artifact is not HY V3")
    expected = {"hidden_size": 4096, "num_hidden_layers": 80,
                "intermediate_size": 13312, "num_attention_heads": 64,
                "num_key_value_heads": 8, "head_dim": 128,
                "num_experts_per_tok": 8, "num_shared_experts": 1,
                "expert_hidden_dim": 1536, "first_k_dense_replace": 1,
                "qk_norm": True, "route_norm": True,
                "moe_router_use_sigmoid": True,
                "moe_router_enable_expert_bias": True}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("HY V3 topology mismatch")
    if (config.get("hidden_act", "silu") != "silu" or
            config.get("tie_word_embeddings", False)):
        raise ValueError("HY V3 activation or embedding topology is unsupported")
    rope = config.get("rope_parameters")
    if (not isinstance(rope, dict) or
            rope.get("rope_type", "default") != "default" or
            not isinstance(rope.get("rope_theta"), (int, float)) or
            rope["rope_theta"] <= 0):
        raise ValueError("HY V3 RoPE topology is unsupported")
    # An omitted key takes the reference HYV3Config default (2.826 / True),
    # which the runtime ModelArgs now share; a declared one must be sane.
    scale = config.get("router_scaling_factor", 2.826)
    if (isinstance(scale, bool) or not isinstance(scale, (int, float)) or scale <= 0
            or type(config.get("enable_moe_fp32_combine", True)) is not bool):
        raise ValueError("HY V3 router scaling or MoE combine config is unsupported")
    if config.get("enable_attention_fp32_softmax"):
        raise ValueError("HY V3 fp32 attention softmax is not implemented")
    n_experts = config.get("num_experts")
    reap = n_experts == 96 and config.get("mtp_num_experts") == 192
    if not reap and (n_experts != 192 or config.get("mtp_num_experts") not in (None, 192)):
        raise ValueError("HY V3 expert topology is neither full nor REAP50")
    quant = config.get("quantization")
    if not isinstance(quant, dict) or any(quant.get(k) != v for k, v in
          {"group_size": 64, "bits": 4 if reap else 6, "mode": "affine"}.items()):
        raise ValueError("HY V3 quantization does not match expert topology")
    weights = artifact["weight_map"]
    required = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    for i in range(80):
        prefix = f"model.layers.{i}."
        required.update({prefix + "self_attn.q_proj.weight",
                         prefix + "self_attn.k_proj.weight",
                         prefix + "self_attn.v_proj.weight",
                         prefix + "self_attn.o_proj.weight",
                         prefix + "self_attn.q_norm.weight",
                         prefix + "self_attn.k_norm.weight",
                         prefix + "input_layernorm.weight",
                         prefix + "post_attention_layernorm.weight"})
        if i:
            required.update({prefix + "mlp.router.gate.weight",
                             prefix + "mlp.router.expert_bias",
                             prefix + "mlp.shared_mlp.gate_proj.weight",
                             prefix + "mlp.shared_mlp.up_proj.weight",
                             prefix + "mlp.shared_mlp.down_proj.weight",
                             prefix + "mlp.switch_mlp.gate_proj.weight",
                             prefix + "mlp.switch_mlp.up_proj.weight",
                             prefix + "mlp.switch_mlp.down_proj.weight"})
        else:
            required.update({prefix + "mlp.gate_proj.weight",
                             prefix + "mlp.up_proj.weight",
                             prefix + "mlp.down_proj.weight"})
    if not required <= weights.keys():
        missing = sorted(required - weights.keys())
        raise ValueError(f"HY V3 ordinary trunk tensors are incomplete: {missing[0]}")
    # These indexed snapshots are fully affine-quantized. A missing companion
    # makes the loader choose a different module layout after reading shards.
    projections = (
        key for key in required
        if key.endswith(".weight") and (
            (".self_attn." in key and not key.endswith("_norm.weight")) or
            ".mlp." in key or key == "lm_head.weight"
        )
    )
    for weight in projections:
        stem = weight[:-len(".weight")]
        if stem + ".scales" not in weights or stem + ".biases" not in weights:
            raise ValueError(f"HY V3 quantized projection is incomplete: {stem}")
    mtp_count = sum(k.startswith(("mtp.", "model.layers.80.")) for k in weights)
    if reap and mtp_count == 0:
        raise ValueError("HY V3 REAP50 artifact is missing its declared sidecar")
    artifact.update(reap=reap, mtp_tensor_count=mtp_count, has_mtp=False)
    return artifact


def configure_environment() -> dict[str, str]:
    # Refuse an explicit TF32 value before the profile overwrites it
    # (sweep 1002 review item 6).
    require_process_numerics("the Hy-V3 profile")
    profile = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               **PROCESS_NUMERICS, "MLX_LM_COMPILED_DECODE": "0"}
    clear_inherited_profile(("MLX_QWEN", "MLX_LM_", "MLXUAG_", "MLX_GDN_"))
    os.environ.update(profile)
    return profile


class HYV3Adapter(OrdinaryTextAdapter):
    descriptor = descriptor_for(reap=False)
    sampling_defaults = SAMPLING
    artifact_inspector = staticmethod(inspect_artifact)
    profile = "hy-v3-apcv2-ordinary"

    def __init__(self, model_path: str, *, execution_policy=None):
        from .process_globals import guarded_construction

        guarded_construction(
            self, lambda: self._init_hy_v3(model_path, execution_policy=execution_policy)
        )

    def _init_hy_v3(self, model_path: str, *, execution_policy=None):
        from .process_globals import claim_stock_moe

        if execution_policy:
            raise ValueError("HY V3 ordinary decode accepts no model execution policy")
        artifact = inspect_artifact(model_path)
        self.descriptor = descriptor_for(reap=artifact["reap"])
        self.identity = artifact["identity"]
        self.layout = CACHE_LAYOUT
        self.environment = configure_environment()
        claim_stock_moe(self, "the HY V3 adapter")
        path = Path(self.identity["path"])
        import mlx.core as mx
        from mlx import nn
        from transformers import AutoTokenizer
        from ..runtime.models.hy_v3 import Model, ModelArgs
        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper
        from ..runtime.ubc_evict import load_shards_evicting

        config = artifact["config"]
        self.model = Model(ModelArgs.from_dict(config))
        files = [path / name for name in sorted(set(artifact["weight_map"].values()))]
        weights = self.model.sanitize(load_shards_evicting(files))

        def predicate(name, module):
            override = config["quantization"].get(name)
            if isinstance(override, dict):
                return override
            return hasattr(module, "to_quantized") and f"{name}.scales" in weights

        quant = config["quantization"]
        nn.quantize(self.model, group_size=quant["group_size"], bits=quant["bits"],
                    mode=quant["mode"], class_predicate=predicate)
        self.model.load_weights(list(weights.items()), strict=True)
        self.model.eval()
        mx.eval(self.model.parameters())
        weights.clear()
        mx.clear_cache()
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True,
                                                  trust_remote_code=False)
        # Some tokenizer classes rebuild the pre-tokenizer instead of reading tokenizer.json.
        from ..runtime.tokenizer_integrity import repair_loaded_tokenizer

        self.pretokenizer_receipt = repair_loaded_tokenizer(tokenizer, path)
        self.tokenizer = TokenizerWrapper(tokenizer,
                                          detokenizer_class=BPEStreamingDetokenizer,
                                          eos_token_ids=[int(config["eos_token_id"])])
        self.max_context = int(config["max_position_embeddings"])

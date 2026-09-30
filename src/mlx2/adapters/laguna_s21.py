"""Laguna S 2.1 ordinary route for indexed 4-bit and bf16 artifacts."""

from __future__ import annotations

from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from ..sampling_defaults import GENERATION_CONFIG, SamplingDefaults, VendorSampling
from .laguna_xs21 import LagunaXS21Adapter, _load_json, configure_environment
from .ordinary_artifact import inspect_indexed_artifact

CACHE_LAYOUT = "laguna-s21-layer-segments-v1"
LAGUNA_S21 = ModelDescriptor(
    model_type="laguna", family="laguna-s", variant="2.1-ordinary",
    state_planes=frozenset({StatePlane.ATTENTION_KV, StatePlane.RNG,
                            StatePlane.TRANSCRIPT}),
    capabilities=frozenset({Capability.TEXT, Capability.STREAMING, Capability.TOOLS,
                            Capability.REASONING, Capability.CONTINUOUS_BATCH,
                            Capability.PREFIX_REUSE, Capability.APC_V2,
                            Capability.LAYERED_CACHE, Capability.PROMPT_LOOKUP,
                            Capability.GRAMMAR}),
    cache_layout=CACHE_LAYOUT,
    metadata={"execution": "mlx2.adapters.laguna_s21.LagunaS21Adapter",
              "qualification": "pending", "scope": "text-only ordinary decode",
              "speculation": "absent"},
)
LAGUNA_S21_SAMPLING = VendorSampling.single(
    SamplingDefaults(temperature=1.0, top_p=1.0, top_k=20, min_p=0.0,
                     source=GENERATION_CONFIG), model="poolside/Laguna-S-2.1"
)


def inspect_artifact(model_path: str | Path) -> dict:
    artifact = inspect_indexed_artifact(model_path)
    path = Path(artifact["identity"]["path"])
    config = _load_json(path / "config.json")
    expected = {"model_type": "laguna", "architectures": ["LagunaForCausalLM"],
                "vocab_size": 100352, "hidden_size": 3072,
                "intermediate_size": 12288, "num_hidden_layers": 48,
                "num_attention_heads": 48, "num_key_value_heads": 8,
                "head_dim": 128, "sliding_window": 512,
                "num_experts": 256, "num_experts_per_tok": 10,
                "moe_intermediate_size": 1024,
                "shared_expert_intermediate_size": 1024,
                "mlp_only_layers": [0], "moe_routed_scaling_factor": 2.5,
                "norm_topk_prob": True, "tie_word_embeddings": False,
                "gating": "per-head"}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("artifact topology does not match Laguna S 2.1")
    if (
        config.get("moe_router_score_func", "sigmoid") != "sigmoid"
        or config.get("moe_router_use_sigmoid", True) is not True
    ):
        # The model computes sqrtsoftplus routing (transformers #48119), but
        # no such checkpoint has been qualified here.
        raise ValueError("only sigmoid Laguna router scoring is qualified")
    layers = ["full_attention" if i % 4 == 0 else "sliding_attention"
              for i in range(48)]
    if config.get("layer_types") != layers:
        raise ValueError("Laguna S layer order mismatch")
    if config.get("mlp_layer_types") != ["dense", *("sparse" for _ in range(47))]:
        raise ValueError("Laguna S MLP order mismatch")
    if config.get("num_attention_heads_per_layer") != [48 if i % 4 == 0 else 72
                                                        for i in range(48)]:
        raise ValueError("Laguna S attention-head order mismatch")
    quant = config.get("quantization")
    if quant is not None and any(quant.get(k) != v for k, v in
                                 {"group_size": 64, "bits": 4, "mode": "affine"}.items()):
        raise ValueError("Laguna S quantization mismatch")
    weights = artifact["weight_map"]
    if any(marker in key.lower() for key in weights for marker in ("mtp.", "eagle", "draft")):
        raise ValueError("speculative tensors are not part of Laguna S ordinary artifact")
    normalized = {key.removeprefix("language_model.") for key in weights}
    required = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    for i in range(48):
        prefix = f"model.layers.{i}."
        required.add(prefix + "self_attn.q_proj.weight")
        required.add(prefix + "self_attn.o_proj.weight")
        if i:
            required.add(prefix + "mlp.shared_expert.gate_proj.weight")
            required.add(prefix + ("mlp.gate.proj.weight" if quant else "mlp.gate.weight"))
            if quant:
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    for suffix in ("weight", "scales", "biases"):
                        required.add(prefix + f"mlp.switch_mlp.{projection}.{suffix}")
            else:
                for expert in range(256):
                    for projection in ("gate_proj", "up_proj", "down_proj"):
                        required.add(prefix + f"mlp.experts.{expert}.{projection}.weight")
        else:
            required.add(prefix + "mlp.gate_proj.weight")
    if not required <= normalized:
        raise ValueError("Laguna S ordinary trunk tensors are incomplete")
    artifact.update(config=config, quantized=quant is not None,
                    layers=48, global_layers=12, sliding_layers=36,
                    sliding_window=512, supports_native_mtp=False,
                    qualification="pending")
    return artifact


class LagunaS21Adapter(LagunaXS21Adapter):
    descriptor = LAGUNA_S21
    sampling_defaults = LAGUNA_S21_SAMPLING
    artifact_inspector = staticmethod(inspect_artifact)

    @staticmethod
    def profile_name(mtp):
        if mtp:
            raise ValueError("Laguna S 2.1 has no native MTP route")
        return "laguna-s21-apcv2-ordinary"

    def __init__(self, model_path: str, *, execution_policy=None):
        if execution_policy:
            raise ValueError("Laguna S ordinary decode accepts no draft execution policy")
        artifact = inspect_artifact(model_path)
        self.identity = artifact["identity"]
        self.config = artifact["config"]
        self.environment = configure_environment()
        self.layout = CACHE_LAYOUT
        self.draft_model = None
        path = Path(self.identity["path"])
        import mlx.core as mx
        from mlx import nn
        from transformers import AutoTokenizer
        from ..runtime.models.laguna import Model, ModelArgs
        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper
        from ..runtime.ubc_evict import load_shards_evicting

        self.model = Model(ModelArgs.from_dict(self.config))
        files = [path / name for name in sorted(set(artifact["weight_map"].values()))]
        weights = self.model.sanitize(load_shards_evicting(files))
        quant = self.config.get("quantization")
        if quant:
            def predicate(name, module):
                override = quant.get(name)
                if isinstance(override, dict):
                    return override
                return hasattr(module, "to_quantized") and f"{name}.scales" in weights
            nn.quantize(self.model, group_size=quant["group_size"], bits=quant["bits"],
                        mode=quant["mode"], class_predicate=predicate)
        self.model.load_weights(list(weights.items()), strict=True)
        self.model.eval()
        mx.eval(self.model.parameters())
        weights.clear()
        mx.clear_cache()
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True,
                                                  trust_remote_code=False,
                                                  fix_mistral_regex=True)
        eos = self.config["eos_token_id"]
        self.tokenizer = TokenizerWrapper(tokenizer,
                                          detokenizer_class=BPEStreamingDetokenizer,
                                          eos_token_ids=[eos] if isinstance(eos, int) else list(eos))
        self.max_context = int(self.config["max_position_embeddings"])

    def diagnostics(self):
        return {"architecture": "laguna-sparse-moe-gqa", "cache_layout": self.layout,
                "sliding_layers": 36, "global_layers": 12,
                "sliding_window": 512, "speculation": "absent",
                "qualification": "pending", "fused_moe": self.model.fused_moe_stats()}

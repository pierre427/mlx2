"""Agnes 3 Flash text-only, ordinary-decode candidate."""

from __future__ import annotations

import os
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from .ordinary_artifact import inspect_indexed_artifact
from .ordinary_text import OrdinaryTextAdapter

CACHE_LAYOUT = "agnes-hybrid-layer-segments-v1"
DESCRIPTOR = ModelDescriptor(
    model_type="agnes", family="agnes-3-flash", variant="preview-6bit-text-ordinary",
    state_planes=frozenset({StatePlane.ATTENTION_KV, StatePlane.RECURRENT,
                            StatePlane.RNG, StatePlane.TRANSCRIPT}),
    capabilities=frozenset({Capability.TEXT, Capability.STREAMING,
                            Capability.CONTINUOUS_BATCH, Capability.PREFIX_REUSE,
                            Capability.APC_V2, Capability.LAYERED_CACHE}),
    cache_layout=CACHE_LAYOUT,
    metadata={"execution": "mlx2.adapters.agnes_3_flash.Agnes3FlashAdapter",
              "qualification": "pending", "scope": "text-only", "mtp": "absent",
              "vision": "absent"},
)


def inspect_artifact(model_path: str | Path) -> dict:
    artifact = inspect_indexed_artifact(model_path)
    config = artifact["config"]
    text = config.get("text_config")
    if config.get("model_type") != "agnes" or not isinstance(text, dict):
        raise ValueError("artifact is not an Agnes conditional-generation checkpoint")
    expected = {"model_type": "agnes_text", "hidden_size": 5120,
                "num_hidden_layers": 72, "intermediate_size": 17408,
                "parallel_ffn_intermediate_size": 2048, "vocab_size": 248320,
                "num_attention_heads": 24, "num_key_value_heads": 4,
                "head_dim": 256, "linear_num_key_heads": 16,
                "linear_num_value_heads": 48, "linear_key_head_dim": 128,
                "linear_value_head_dim": 128, "attn_output_gate": True,
                "output_gate_type": "swish"}
    if any(text.get(key) != value for key, value in expected.items()):
        raise ValueError("Agnes 3 Flash text topology mismatch")
    layers = ["agnes_global_attention" if (i + 1) % 4 == 0
              else "agnes_delta_attention" for i in range(72)]
    if text.get("layer_types") != layers:
        raise ValueError("Agnes 3 Flash layer order mismatch")
    if config.get("quantization") != {"group_size": 64, "bits": 6, "mode": "affine"}:
        raise ValueError("Agnes 3 Flash requires affine 6-bit weights")
    weights = artifact["weight_map"]
    if any(key.startswith(("mtp.", "language_model.mtp.")) for key in weights):
        raise ValueError("embedded Agnes MTP is outside the ordinary text contract")
    required = {"language_model.model.embed_tokens.weight", "language_model.lm_head.weight",
                "language_model.model.norm.weight"}
    for i, kind in enumerate(layers):
        prefix = f"language_model.model.layers.{i}."
        required.add(prefix + ("global_attn.q_proj.weight" if kind.endswith("global_attention")
                               else "delta_attn.in_proj_qkv.weight"))
        required.add(prefix + "mlp.parallel_ffn.gate_proj.weight")
    if not required <= weights.keys():
        raise ValueError("Agnes 3 Flash text tensors are incomplete")
    artifact["has_mtp"] = False
    artifact["text_weight_count"] = sum(key.startswith("language_model.") for key in weights)
    return artifact


def configure_environment() -> dict[str, str]:
    profile = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "MLX_ENABLE_TF32": "0", "MLX_GDN_PACKED": "1",
               "MLX_GDN_CORE": "0", "MLX_LM_COMPILED_DECODE": "0"}
    for name in tuple(os.environ):
        if name.startswith(("MLX_QWEN", "MLX_LM_", "MLXUAG_", "MLX_GDN_", "MLX_AGNES_")):
            del os.environ[name]
    os.environ.update(profile)
    return profile


class Agnes3FlashAdapter(OrdinaryTextAdapter):
    descriptor = DESCRIPTOR
    artifact_inspector = staticmethod(inspect_artifact)
    profile = "agnes-3-flash-6bit-apcv2-ordinary"

    def __init__(self, model_path: str, *, execution_policy=None):
        if execution_policy:
            raise ValueError("Agnes ordinary decode accepts no model execution policy")
        artifact = inspect_artifact(model_path)
        self.identity = artifact["identity"]
        self.layout = CACHE_LAYOUT
        self.environment = configure_environment()
        path = Path(self.identity["path"])
        import mlx.core as mx
        from mlx import nn
        from transformers import AutoTokenizer
        from ..runtime.models.agnes import Model, ModelArgs
        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper
        from ..runtime.ubc_evict import load_shards_evicting

        config = artifact["config"]
        self.model = Model(ModelArgs.from_dict(config))
        files = [path / name for name in sorted(set(artifact["weight_map"].values()))]
        weights = self.model.sanitize(load_shards_evicting(files))
        quant = config["quantization"]
        nn.quantize(self.model, group_size=quant["group_size"], bits=quant["bits"],
                    mode=quant["mode"], class_predicate=lambda name, module:
                    hasattr(module, "to_quantized") and f"{name}.scales" in weights)
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
        eos = config["eos_token_id"]
        self.tokenizer = TokenizerWrapper(tokenizer,
                                          detokenizer_class=BPEStreamingDetokenizer,
                                          eos_token_ids=[eos] if isinstance(eos, int) else list(eos))
        self.max_context = int(config["text_config"]["max_position_embeddings"])

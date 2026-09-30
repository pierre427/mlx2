"""HiLS Attention 7B candidate; its landmark cache is not APCv2 qualified."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .artifact_paths import shard_within_artifact
from ..contracts import Capability, ModelDescriptor, StatePlane
from .ordinary_text import OrdinaryTextAdapter
from ..process_env import PROCESS_NUMERICS


DESCRIPTOR = ModelDescriptor(
    model_type="olmo_hils", family="hils-attention", variant="7b-ordinary-b1",
    state_planes=frozenset({StatePlane.ATTENTION_KV, StatePlane.RNG, StatePlane.TRANSCRIPT}),
    capabilities=frozenset({Capability.TEXT, Capability.STREAMING}),
    cache_layout=None,
    metadata={"execution": "mlx2.adapters.olmo_hils.OlmoHiLSAdapter",
              "qualification": "pending", "scope": "plain text batch width one; no APCv2 publication"},
)


def inspect_artifact(model_path: str | Path) -> dict:
    path = Path(model_path).expanduser().resolve()
    config = json.loads((path / "config.json").read_text())
    if not isinstance(config, dict):
        raise ValueError("HiLS config must be an object")
    expected = {
        "model_type": "olmo_hils", "architectures": ["HiLSForCausalLM"],
        "hidden_size": 4096, "intermediate_size": 11008,
        "num_hidden_layers": 32, "num_attention_heads": 32,
        "num_key_value_heads": 32, "vocab_size": 100278,
        "max_position_embeddings": 131072, "sliding_window": 512,
        "hils_sliding_window": 512, "chunk_size": 64, "hils_topk": 32,
        "full_attn_interleave": 4, "layerwise_qk_norm": True,
        "layerwise_lmkq_norm": True, "apply_hils_rope": True,
        "enable_inrange_rope": True, "rope_context_length": 8192,
        "rope_period_multiplier": 2.0, "tie_word_embeddings": False,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("HiLS 7B landmark topology does not match")
    if config.get("num_swa_layers") not in (None, 0):
        raise ValueError("unsupported HiLS layer schedule")
    quant = config.get("quantization")
    if quant is not None and quant != {"group_size": 64, "bits": 6, "mode": "affine"}:
        raise ValueError("unsupported HiLS quantization")
    index = json.loads((path / "model.safetensors.index.json").read_text())
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("HiLS requires an indexed checkpoint")
    required = {"model.embed_tokens.weight", "model.lmk_embed", "model.norm.weight", "lm_head.weight"}
    for layer in range(32):
        prefix = f"model.layers.{layer}."
        required.update(prefix + suffix for suffix in (
            "self_attn.q_proj.weight", "self_attn.k_proj.weight",
            "self_attn.v_proj.weight", "self_attn.o_proj.weight",
            "self_attn.q_norm.weight", "self_attn.k_norm.weight",
            "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
        ))
        if layer % 4 == 3:
            required.update({prefix + "self_attn.lmk_q_proj.0.weight",
                             prefix + "self_attn.lmk_q_proj.1.weight",
                             prefix + "self_attn.lmk_q_norm.weight"})
    if not required <= weight_map.keys():
        raise ValueError("HiLS indexed landmark tensor topology is incomplete")
    names = sorted(set(weight_map.values()))
    records = []
    for name in names:
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts or Path(name).suffix != ".safetensors":
            raise ValueError("unsafe HiLS shard path")
        item = path / name
        if not item.is_file() or not shard_within_artifact(path, item.resolve()):
            raise ValueError(f"missing HiLS shard: {name}")
        stat = item.stat(); records.append((name, stat.st_size, stat.st_mtime_ns))
    digest = hashlib.sha256()
    for name in ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"):
        item = path / name
        if item.is_file():
            digest.update(name.encode()); digest.update(item.read_bytes())
    for record in records:
        digest.update(json.dumps(record).encode())
    return {"config": config, "weight_map": weight_map, "quantized": quant is not None,
            "identity": {"path": str(path), "fingerprint": digest.hexdigest(), "files": records},
            "hils_layers": 8, "swa_layers": 24, "apcv2_qualified": False}


class OlmoHiLSAdapter(OrdinaryTextAdapter):
    descriptor = DESCRIPTOR
    profile = "olmo-hils-7b-b1-ordinary"

    def __init__(self, model_path: str, *, execution_policy=None):
        if execution_policy not in (None, {}):
            raise ValueError("HiLS supports ordinary B1 execution only")
        artifact = inspect_artifact(model_path)
        self.identity = artifact["identity"]
        self.config = artifact["config"]
        self.layout = "hils-landmark-custom-cache-v1"
        self.environment = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", **PROCESS_NUMERICS}
        os.environ.update(self.environment)
        path = Path(self.identity["path"])
        import mlx.core as mx
        from mlx import nn
        from transformers import AutoTokenizer
        from ..runtime.models.olmo_hils import Model, ModelArgs
        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper
        from ..runtime.ubc_evict import load_shards_evicting
        self.model = Model(ModelArgs.from_dict(self.config))
        files = [path / name for name in sorted(set(artifact["weight_map"].values()))]
        weights = load_shards_evicting(files)
        quant = self.config.get("quantization")
        if quant:
            nn.quantize(self.model, group_size=quant["group_size"], bits=quant["bits"],
                        mode=quant["mode"], class_predicate=lambda name, module:
                        hasattr(module, "to_quantized") and f"{name}.scales" in weights)
        self.model.load_weights(list(weights.items()), strict=True)
        self.model.eval(); mx.eval(self.model.parameters())
        weights.clear(); mx.clear_cache()
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        # Some tokenizer classes rebuild the pre-tokenizer instead of reading tokenizer.json.
        from ..runtime.tokenizer_integrity import repair_loaded_tokenizer

        self.pretokenizer_receipt = repair_loaded_tokenizer(tokenizer, path)
        self.tokenizer = TokenizerWrapper(tokenizer, detokenizer_class=BPEStreamingDetokenizer,
                                          eos_token_ids=[int(self.config["eos_token_id"])])
        self.max_context = int(self.config["max_position_embeddings"])

    @staticmethod
    def lane_projection_groups():
        """Offered to the lane installer; stacked only under ``declared_groups``."""
        from ..runtime.models.olmo_hils import lane_projection_groups

        return lane_projection_groups()

    def prompt_tokens(self, request):
        if "messages" in request or request.get("tools"):
            raise ValueError("HiLS candidate accepts plain text prompts only")
        return self.tokenizer.encode(request["prompt"], add_special_tokens=False)

    def execution_config(self, *, max_lanes, prefill_step):
        if max_lanes != 1:
            raise ValueError("HiLS landmark bookkeeping supports one lane")
        return super().execution_config(max_lanes=max_lanes, prefill_step=prefill_step)

    def close(self):
        self.model = None
        self.tokenizer = None

"""CPU-safe artifact inspection and the dense Qwen3.8 27B serving adapter.

Import/inspect performs no tensor imports or model loads. Instantiating the
adapter loads weights and is reserved for a separately authorized GPU window.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from .flash_next import FlashNextAdapter

CACHE_LAYOUT = "qwen38-27b-hybrid-layer-segments-v1"


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
        Capability.PROMPT_LOOKUP,
    }
    planes = {
        StatePlane.ATTENTION_KV,
        StatePlane.RECURRENT,
        StatePlane.RNG,
        StatePlane.TRANSCRIPT,
    }
    if has_mtp:
        capabilities.update({Capability.MTP, Capability.SEGMENTED_MTP})
        planes.add(StatePlane.DRAFT)
    return ModelDescriptor(
        model_type="qwen3_5",
        family="qwen3.8-27b",
        variant="27b-mtp" if has_mtp else "27b-ordinary",
        state_planes=frozenset(planes),
        capabilities=frozenset(capabilities),
        cache_layout=CACHE_LAYOUT,
        metadata={
            "execution": "mlx2.adapters.qwen38_27b.Qwen3827BAdapter",
            "qualification": "pending",
            "scope": "text-only",
            "true_batched_segmented_mtp": "implemented-cpu-oracle-gpu-unqualified",
        },
    )


QWEN38_27B = descriptor_for(has_mtp=True)
QWEN38_27B_ORDINARY = descriptor_for(has_mtp=False)


def inspect_artifact(model_path: str | Path) -> dict:
    """Validate local metadata without importing MLX or opening tensor payloads."""
    path = Path(model_path).expanduser().resolve()
    config = json.loads((path / "config.json").read_text())
    text = config.get("text_config", config)
    if config.get("model_type") != "qwen3_5" or text.get("num_experts", 0):
        raise ValueError("Qwen3.8 27B requires the dense qwen3_5 artifact layout")
    expected = {
        "num_hidden_layers": 64,
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "full_attention_interval": 4,
        "vocab_size": 248320,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
    }
    if any(text.get(k) != v for k, v in expected.items()):
        raise ValueError("artifact topology does not match Qwen3.8 27B")
    if text.get("layer_types") not in (
        None,
        [
            "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
            for i in range(64)
        ],
    ):
        raise ValueError("artifact layer order does not match Qwen3.8 27B")
    if text.get("mtp_num_hidden_layers", 0) not in (0, 1):
        raise ValueError("only the single-layer Qwen3.8 MTP head is implemented")
    index = json.loads((path / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    if not isinstance(index, dict) or not index:
        raise ValueError("artifact has no indexed weights")
    names = sorted(set(index.values()))
    for name in names:
        if (
            not isinstance(name, str)
            or Path(name).is_absolute()
            or ".." in Path(name).parts
        ):
            raise ValueError("weight shard paths must stay within the artifact")
        if not (path / name).is_file():
            raise ValueError(f"missing weight shard: {name}")
    mtp_keys = [k for k in index if k.startswith(("language_model.mtp.", "mtp."))]
    has_mtp = bool(mtp_keys)
    if has_mtp and text.get("mtp_num_hidden_layers") != 1:
        raise ValueError("MTP tensors and configured head count disagree")
    if has_mtp:
        normalized = {k.removeprefix("language_model.") for k in mtp_keys}
        required = {
            "mtp.fc.weight",
            "mtp.norm.weight",
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight",
            "mtp.layers.0.self_attn.q_proj.weight",
            "mtp.layers.0.self_attn.k_proj.weight",
            "mtp.layers.0.self_attn.v_proj.weight",
            "mtp.layers.0.self_attn.o_proj.weight",
            "mtp.layers.0.mlp.gate_proj.weight",
            "mtp.layers.0.mlp.up_proj.weight",
            "mtp.layers.0.mlp.down_proj.weight",
        }
        if not required <= normalized:
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
    for name in names:
        stat = (path / name).stat()
        record = (name, stat.st_size, stat.st_mtime_ns)
        records.append(record)
        digest.update(json.dumps(record).encode())
    return {
        "config": config,
        "weight_map": index,
        "has_mtp": has_mtp,
        "mtp_tensor_count": len(mtp_keys),
        "identity": {
            "path": str(path),
            "fingerprint": digest.hexdigest(),
            "files": records,
        },
    }


def configure_environment() -> dict[str, str]:
    """Candidate dense profile; flags confer no qualification by themselves."""
    profile = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "MLX_ENABLE_TF32": "0",
        "MLX_GDN_PACKED": "1",
        "MLX_GDN_CORE": "0",
        "MLX_LM_COMPILED_DECODE": "0",
        "MLX_LM_SEGMENTED_SELF_MTP": "1",
        "MLX_LM_TRUE_BATCHED_SEGMENTED_MTP": "1",
        "MLX_LM_SHARED_QSA_SUFFIX": "0",
    }
    for name in tuple(os.environ):
        if name.startswith(("MLX_QWEN", "MLX_LM_", "MLXUAG_", "MLX_GDN_")):
            del os.environ[name]
    os.environ.update(profile)
    return profile


class Qwen3827BAdapter(FlashNextAdapter):
    """Dense text adapter using shared chat parsing and modern runtime state."""

    descriptor = QWEN38_27B

    def __init__(
        self, model_path: str, *, require_mtp: bool = False, execution_policy=None
    ):
        if execution_policy is not None and not isinstance(execution_policy, dict):
            raise ValueError("execution policy must be a JSON object")
        policy = {} if execution_policy is None else dict(execution_policy)
        if set(policy) - {"num_draft"}:
            raise ValueError("Qwen3.8 27B execution policy supports only num_draft")
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
        config = artifact["config"]
        import mlx.core as mx
        import mlx.nn as nn
        from transformers import AutoTokenizer
        from ..runtime.models.qwen38_27b import Model, ModelArgs
        from ..runtime.tokenizer_utils import TokenizerWrapper, BPEStreamingDetokenizer
        from ..runtime.ubc_evict import ubc_evict_paths

        # Conversion configs may advertise a head that was stripped from weights.
        config = dict(config)
        config["text_config"] = dict(config.get("text_config", config))
        if not artifact["has_mtp"]:
            config["text_config"]["mtp_num_hidden_layers"] = 0
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
        tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
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
        return (
            f"qwen38-27b-apcv2-mtp{getattr(self, '_num_draft', 2)}"
            if mtp
            else "qwen38-27b-apcv2-ordinary"
        )

    def execution_config(self, *, max_lanes, prefill_step):
        return {
            "persistent": True,
            "num_draft": getattr(self, "_num_draft", 2)
            if Capability.MTP in self.descriptor.capabilities
            else 0,
            "rate_gate": False,
            "prefill_step_size": prefill_step,
            "segment_aware_live_tip": True,
            "segment_aware_cohort_size": max_lanes,
        }

    def cache_budget(self, *, mtp):
        from .qwen38_memory import Qwen38CacheBudget

        return Qwen38CacheBudget.from_config(
            self.model.args.text_config, mtp=mtp
        )

    def diagnostics(self):
        from ..runtime.segmented_self_mtp import segmented_self_mtp_stats

        return {
            "architecture": "dense-hybrid-gdn-gqa",
            "layout": self.layout,
            "mtp_head_present": Capability.MTP in self.descriptor.capabilities,
            "segmented_mtp": segmented_self_mtp_stats(),
        }

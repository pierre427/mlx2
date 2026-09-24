"""Nemotron 3 Super 120B-A12B 5-bit hybrid/MTP decode adapter.

Inspection is CPU-only and reads safetensors headers, never tensor payloads.
Model construction is separate; production route selection needs a qualification receipt.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from ..sampling_defaults import GENERATION_CONFIG, SamplingDefaults, VendorSampling
from .flash_next import FlashNextAdapter

CACHE_LAYOUT = "nemotron3-super-hybrid-mamba-kv-v1"
SAMPLING = VendorSampling.single(
    SamplingDefaults(temperature=1.0, top_p=0.95, source=GENERATION_CONFIG),
    model="NVIDIA-Nemotron-3-Super-120B-A12B-5bit-MTP",
)
DESCRIPTOR = ModelDescriptor(
    model_type="nemotron_h",
    family="nemotron3-super-120b-a12b",
    variant="5bit-mtp",
    state_planes=frozenset({StatePlane.ATTENTION_KV, StatePlane.RECURRENT, StatePlane.RNG, StatePlane.TRANSCRIPT, StatePlane.DRAFT}),
    capabilities=frozenset({Capability.TEXT, Capability.STREAMING, Capability.TOOLS, Capability.REASONING, Capability.CONTINUOUS_BATCH, Capability.PREFIX_REUSE, Capability.APC_V2, Capability.LAYERED_CACHE, Capability.GRAMMAR, Capability.MTP}),
    cache_layout=CACHE_LAYOUT,
    metadata={"execution": "mlx2.adapters.nemotron3_super.Nemotron3SuperAdapter", "qualification": "pending", "scope": "text-only", "embedded_mtp": "implemented-candidate"},
)


def _shard_header(path: Path) -> dict:
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"truncated safetensors shard: {path.name}")
        header_size = struct.unpack("<Q", raw)[0]
        if header_size > 64 * 1024 * 1024 or header_size > path.stat().st_size - 8:
            raise ValueError(f"invalid safetensors header: {path.name}")
        header = json.loads(f.read(header_size))
    if not isinstance(header, dict):
        raise TypeError(f"invalid safetensors header: {path.name}")
    payload_size = path.stat().st_size - 8 - header_size
    spans = []
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= payload_size):
            raise ValueError(f"incomplete safetensors payload: {path.name}:{key}")
        spans.append((start, end))
    if not spans or min(spans)[0] != 0 or max(spans)[1] != payload_size:
        raise ValueError(f"incomplete safetensors payload: {path.name}")
    if any(left[1] != right[0] for left, right in zip(sorted(spans), sorted(spans)[1:])):
        raise ValueError(f"safetensors payload has gaps or overlaps: {path.name}")
    return header


def inspect_artifact(model_path: str | Path) -> dict:
    path = Path(model_path).expanduser().resolve()
    config = json.loads((path / "config.json").read_text())
    expected = {
        "model_type": "nemotron_h", "num_hidden_layers": 88,
        "hidden_size": 4096, "vocab_size": 131072,
        "num_attention_heads": 32, "num_key_value_heads": 2, "head_dim": 128,
        "mamba_num_heads": 128, "mamba_head_dim": 64,
        "ssm_state_size": 128, "conv_kernel": 4, "n_groups": 8,
        "n_routed_experts": 512, "num_experts_per_tok": 22,
        "moe_latent_size": 1024, "moe_intermediate_size": 2688,
        "num_nextn_predict_layers": 1, "mtp_hybrid_override_pattern": "*E",
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("artifact topology does not match Nemotron 3 Super 120B-A12B")
    pattern = config.get("hybrid_override_pattern")
    if not isinstance(pattern, str) or len(pattern) != 88 or any(pattern.count(k) != n for k, n in (("M", 40), ("E", 40), ("*", 8))):
        raise ValueError("artifact hybrid layer order is invalid")
    if config.get("quantization") != {"group_size": 64, "bits": 5, "mode": "affine"}:
        raise ValueError("artifact is not affine 5-bit")
    if config.get("eos_token_id") != [2, 11]:
        raise ValueError("Nemotron 3 Super EOS token contract changed")
    tokenizer_config = json.loads((path / "tokenizer_config.json").read_text())
    if tokenizer_config.get("tool_parser_type") != "qwen3_coder":
        raise ValueError("Nemotron 3 Super tool parser contract changed")
    template = (path / "chat_template.jinja").read_text()
    if not all(marker in template for marker in ("<tool_call>", "<function=", "<parameter=", "enable_thinking", "</think>")):
        raise ValueError("Nemotron 3 Super chat template contract changed")
    index = json.loads((path / "model.safetensors.index.json").read_text())
    weights = index.get("weight_map")
    if not isinstance(weights, dict) or len(weights) != 1511:
        raise ValueError("Nemotron 3 Super weight index is incomplete")
    mtp_tensor_count = sum(key.startswith("mtp.") for key in weights)
    if mtp_tensor_count != 40:
        raise ValueError("Nemotron 3 Super embedded MTP index is incomplete")
    names = sorted(set(weights.values()))
    if len(names) != 17:
        raise ValueError("Nemotron 3 Super requires 17 complete shards")
    required = {
        "backbone.embeddings.weight", "backbone.layers.0.mixer.conv1d.weight",
        "backbone.layers.87.norm.weight", "lm_head.weight",
        "mtp.layers.0.eh_proj.weight", "mtp.layers.1.mixer.switch_mlp.fc1.weight",
    }
    if not required <= weights.keys():
        raise ValueError("required backbone or MTP tensors are absent")
    for i, block in enumerate(pattern):
        prefix = f"backbone.layers.{i}.mixer."
        key = {"M": "conv1d.weight", "E": "switch_mlp.fc1.weight", "*": "q_proj.weight"}[block]
        if prefix + key not in weights:
            raise ValueError(f"layer {i} tensor layout does not match hybrid pattern")
    digest = hashlib.sha256()
    for filename in ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
        item = path / filename
        if not item.is_file():
            raise ValueError(f"missing artifact metadata: {filename}")
        digest.update(filename.encode()); digest.update(item.read_bytes())
    records = []
    observed = set()
    for name in names:
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("weight shard paths must stay within the artifact")
        item = (path / name).resolve()
        if not item.is_relative_to(path) or not item.is_file():
            raise ValueError(f"missing or escaped weight shard: {name}")
        header = _shard_header(item)
        tensor_names = set(header) - {"__metadata__"}
        indexed = {key for key, shard in weights.items() if shard == name}
        if tensor_names != indexed or observed & tensor_names:
            raise ValueError(f"safetensors index/header mismatch: {name}")
        observed.update(tensor_names)
        stat = item.stat()
        record = (name, stat.st_size, stat.st_mtime_ns)
        records.append(record); digest.update(json.dumps(record).encode())
    return {"config": config, "weight_map": weights, "has_mtp": True,
            "mtp_tensor_count": mtp_tensor_count,
            "identity": {"path": str(path), "fingerprint": digest.hexdigest(), "files": records}}


class Nemotron3SuperAdapter(FlashNextAdapter):
    descriptor = DESCRIPTOR
    sampling_defaults = SAMPLING
    artifact_inspector = staticmethod(inspect_artifact)
    # This adapter is not well tuned yet. Its exact B1 MTP verifier is slower
    # than ordinary decode at the tested 8K and 32K contexts; batched MTP,
    # cached-prefix parity, and longer contexts still need qualification.
    default_route = "ordinary"

    @staticmethod
    def spomin_backend(model, prompt_cache):
        return None

    def __init__(self, model_path: str, *, execution_policy=None):
        if execution_policy is not None and (
            not isinstance(execution_policy, dict)
            or set(execution_policy) - {"num_draft"}
        ):
            raise ValueError("Nemotron 3 Super execution policy supports only num_draft")
        from .mtp_depth_cap import validate_self_mtp_num_draft

        self._num_draft = validate_self_mtp_num_draft(
            (execution_policy or {}).get("num_draft", 3)
        )
        if self._num_draft > 3:
            raise ValueError("Nemotron 3 Super MTP depth above 3 is not qualified")
        artifact = inspect_artifact(model_path)
        self.identity = artifact["identity"]
        self.layout = CACHE_LAYOUT
        self.environment = {
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "MLX_ENABLE_TF32": "0",
        }
        os.environ.update(self.environment)
        path = Path(self.identity["path"])
        import mlx.core as mx
        from mlx import nn
        from transformers import AutoTokenizer

        from ..runtime.models.nemotron_h import Model, ModelArgs
        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper
        from ..runtime.ubc_evict import load_shards_evicting

        config = dict(artifact["config"])
        self.model = Model(ModelArgs.from_dict(config))
        files = [path / name for name in sorted(set(artifact["weight_map"].values()))]
        weights = self.model.sanitize(load_shards_evicting(files))
        quant = config["quantization"]
        nn.quantize(self.model, group_size=quant["group_size"], bits=quant["bits"],
                    mode=quant["mode"], class_predicate=lambda name, module: hasattr(module, "to_quantized") and f"{name}.scales" in weights)
        self.model.load_weights(list(weights.items()), strict=True)
        self.model.eval(); mx.eval(self.model.parameters())
        weights.clear(); mx.clear_cache()
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.tokenizer = TokenizerWrapper(tokenizer, detokenizer_class=BPEStreamingDetokenizer,
                                          eos_token_ids=[2, 11])
        self.max_context = config["max_position_embeddings"]

    def thinking_enabled(self, request):
        return bool(request.get("enable_thinking", True))

    def prompt_tokens(self, request):
        if "messages" in request:
            from .flash_next import chat_template
            return chat_template(self.tokenizer, {**request, "enable_thinking": self.thinking_enabled(request)}, tokenize=True)
        return self.tokenizer.encode(request["prompt"], add_special_tokens=False)

    def render_prompt(self, request):
        if "messages" in request:
            from .flash_next import chat_template
            return chat_template(self.tokenizer, {**request, "enable_thinking": self.thinking_enabled(request)}, tokenize=False)
        return request["prompt"]

    def output_parser(self, request):
        return super().output_parser({**request, "enable_thinking": self.thinking_enabled(request)})

    def profile_name(self, mtp):
        return (
            f"nemotron3-super-5bit-apcv2-mtp{self._num_draft}"
            if mtp else "nemotron3-super-5bit-apcv2-ordinary"
        )

    def execution_config(self, *, max_lanes, prefill_step):
        return {"persistent": True, "num_draft": self._num_draft, "rate_gate": False,
                "prefill_step_size": prefill_step, "segment_aware_live_tip": True,
                "segment_aware_cohort_size": max_lanes}

    def cache_budget(self, *, mtp):
        from .nemotron3_super_memory import NemotronCacheBudget
        return NemotronCacheBudget.from_config(self.model.args, mtp=mtp)

    def diagnostics(self):
        return {"architecture": "nemotron-h-mamba-moe-gqa", "layout": self.layout,
                "mtp_head_present": True, "mtp_candidate": True}

    def close(self):
        self.model = None

"""CPU-safe artifact gate and ordinary serving adapters for standard decoders."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane

FAMILIES = frozenset({"qwen3", "qwen3_moe", "qwen2", "llama"})
CACHE_LAYOUT = "standard-full-kv-layer-segments-v1"


def descriptor_for(family: str) -> ModelDescriptor:
    if family not in FAMILIES:
        raise ValueError(f"unsupported standard decoder family: {family}")
    return ModelDescriptor(
        model_type=family,
        family=family.replace("_", "-"),
        variant="ordinary",
        state_planes=frozenset({StatePlane.ATTENTION_KV, StatePlane.RNG, StatePlane.TRANSCRIPT}),
        capabilities=frozenset({
            Capability.TEXT, Capability.STREAMING, Capability.CONTINUOUS_BATCH,
            Capability.PREFIX_REUSE, Capability.APC_V2, Capability.LAYERED_CACHE,
        }),
        cache_layout=CACHE_LAYOUT,
        metadata={
            "execution": "mlx2.adapters.standard_decoder.StandardDecoderAdapter",
            "qualification": "pending",
            "scope": "text-only ordinary decode",
        },
    )


def _json(path: Path) -> dict:
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"duplicate key {key!r} in {path.name}")
            out[key] = value
        return out
    value = json.loads(path.read_text(), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must be an object")
    return value


def _positive(config: dict, name: str) -> int:
    value = config.get(name)
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def inspect_artifact(model_path: str | Path, *, expected: str | None = None) -> dict:
    """Inspect topology, index closure and identity with no tensor imports."""
    path = Path(model_path).expanduser().resolve()
    config = _json(path / "config.json")
    family = config.get("model_type")
    if family not in FAMILIES or (expected is not None and family != expected):
        raise ValueError(f"unsupported artifact family: {family!r}")
    for key in (
        "hidden_size", "num_hidden_layers", "intermediate_size",
        "num_attention_heads", "vocab_size", "max_position_embeddings",
    ):
        _positive(config, key)
    heads = _positive(config, "num_attention_heads")
    kv_heads = config.get("num_key_value_heads", heads)
    if type(kv_heads) is not int or kv_heads < 1 or heads % kv_heads:
        raise ValueError("invalid grouped-query head geometry")
    if type(config.get("rms_norm_eps")) not in {float, int} or config["rms_norm_eps"] <= 0:
        raise ValueError("invalid RMS normalization epsilon")
    if type(config.get("rope_theta")) not in {float, int} or config["rope_theta"] <= 0:
        raise ValueError("invalid rotary base")
    if type(config.get("tie_word_embeddings")) is not bool:
        raise ValueError("tie_word_embeddings must be declared")
    if family in {"qwen3", "qwen3_moe"}:
        _positive(config, "head_dim")
    elif config.get("head_dim") is not None:
        _positive(config, "head_dim")
    elif config["hidden_size"] % heads:
        raise ValueError("head dimension is not integral")
    if family == "qwen3_moe":
        experts = _positive(config, "num_experts")
        top_k = _positive(config, "num_experts_per_tok")
        _positive(config, "moe_intermediate_size")
        _positive(config, "decoder_sparse_step")
        if top_k > experts or not isinstance(config.get("mlp_only_layers"), list):
            raise ValueError("invalid MoE routing geometry")
        if any(type(i) is not int or i < 0 or i >= config["num_hidden_layers"] for i in config["mlp_only_layers"]):
            raise ValueError("invalid dense-only layer list")
        if type(config.get("norm_topk_prob")) is not bool:
            raise ValueError("norm_topk_prob must be declared")
    if family == "llama" and config.get("layer_types") not in (None, ["full_attention"] * config["num_hidden_layers"]):
        raise ValueError("sliding Llama requires a separate cache layout")
    if config.get("sliding_window") and family == "llama":
        raise ValueError("sliding Llama requires a separate cache layout")
    quant = config.get("quantization", config.get("quantization_config"))
    if quant is not None and (
        not isinstance(quant, dict) or type(quant.get("group_size")) is not int
        or type(quant.get("bits")) is not int
        or quant["group_size"] <= 0 or quant["bits"] not in {4, 6, 8}
    ):
        raise ValueError("unsupported quantization configuration")
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        weights = _json(index_path).get("weight_map")
        if not isinstance(weights, dict) or not weights:
            raise ValueError("empty safetensors index")
    else:
        only = path / "model.safetensors"
        if not only.is_file():
            raise ValueError("missing model weights")
        # Indexless BF16 artifacts are accepted; strict tensor names are
        # checked by load_weights(strict=True) at the model-load boundary.
        weights = None
    names = sorted(set(weights.values())) if weights is not None else ["model.safetensors"]
    records = []
    for name in names:
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError("weight index contains an unsafe shard path")
        # Hugging Face snapshots link shards into their sibling blobs tree.
        # The basename gate above prevents traversal while preserving those
        # legitimate local links.
        item = path / name
        if not item.is_file() or item.stat().st_size < 8:
            raise ValueError(f"missing or empty weight shard: {name}")
        stat = item.stat()
        records.append((name, stat.st_size, stat.st_mtime_ns))
    if weights is not None:
        if any(k.startswith(("mtp.", "draft.")) for k in weights):
            raise ValueError("speculative head is not an ordinary target artifact")
        required = {"model.embed_tokens.weight", "model.norm.weight", "model.layers.0.self_attn.q_proj.weight"}
        if not required.issubset(weights):
            raise ValueError("index lacks required decoder tensors")
        if family == "qwen3_moe" and "model.layers.0.mlp.gate.weight" not in weights:
            raise ValueError("MoE index lacks router tensors")
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
    normalized = dict(config)
    normalized["num_key_value_heads"] = kv_heads
    normalized.setdefault("head_dim", config["hidden_size"] // heads)
    normalized.setdefault("rope_traditional", False)
    normalized.setdefault("attention_bias", False)
    normalized.setdefault("mlp_bias", False)
    return {
        "config": normalized,
        "weight_map": weights,
        "identity": {"path": str(path), "fingerprint": digest.hexdigest(), "files": records},
        "qualification": "pending", "has_mtp": False,
    }


class StandardDecoderAdapter:
    default_route = "ordinary"
    descriptor = descriptor_for("qwen3")

    def profile_name(self, mtp):
        if mtp:
            raise ValueError("standard decoder has no native MTP route")
        return f"{self.descriptor.model_type}-apcv2-ordinary"

    def execution_config(self, *, max_lanes, prefill_step):
        return {
            "persistent": True, "num_draft": 0, "backend": "ordinary",
            "rate_gate": False, "prefill_step_size": prefill_step,
            "segment_aware_live_tip": False,
            "segment_aware_cohort_size": max_lanes,
        }

    def close(self):
        pass

    def __init__(self, model_path: str, *, execution_policy=None):
        if execution_policy not in (None, {}):
            raise ValueError("ordinary standard decoder has no execution policy")
        artifact = inspect_artifact(model_path)
        self.config = artifact["config"]
        self.identity = artifact["identity"]
        self.descriptor = descriptor_for(self.config["model_type"])
        self.layout = CACHE_LAYOUT
        self.environment = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        path = Path(self.identity["path"])
        import mlx.core as mx
        from mlx import nn
        from transformers import AutoTokenizer
        from ..runtime.models.standard_decoder import Model, ModelArgs
        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper
        from ..runtime.ubc_evict import load_shards_evicting

        self.model = Model(ModelArgs.from_dict(self.config))
        files = [path / name for name, _, _ in self.identity["files"]]
        weights = self.model.sanitize(load_shards_evicting(files))
        quant = self.config.get("quantization", self.config.get("quantization_config"))
        if quant:
            def predicate(name, module):
                override = quant.get(name)
                if isinstance(override, dict):
                    return override
                return hasattr(module, "to_quantized") and f"{name}.scales" in weights
            nn.quantize(
                self.model, group_size=quant["group_size"], bits=quant["bits"],
                mode=quant.get("mode", "affine"), class_predicate=predicate,
            )
        self.model.load_weights(list(weights.items()), strict=True)
        self.model.eval()
        mx.eval(self.model.parameters())
        weights.clear()
        mx.clear_cache()
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        eos = self.config.get("eos_token_id")
        self.tokenizer = TokenizerWrapper(
            tokenizer, detokenizer_class=BPEStreamingDetokenizer,
            eos_token_ids=[eos] if isinstance(eos, int) else list(eos or []),
        )
        self.max_context = self.config["max_position_embeddings"]

    def prompt_tokens(self, request: dict) -> list[int]:
        if "messages" in request:
            return self.tokenizer.apply_chat_template(
                request["messages"], add_generation_prompt=True, tokenize=True,
                enable_thinking=request.get("enable_thinking", self.tokenizer.has_thinking),
            )
        return self.tokenizer.encode(request["prompt"], add_special_tokens=False)

    def render_prompt(self, request: dict) -> str:
        if "messages" in request:
            return self.tokenizer.apply_chat_template(
                request["messages"], add_generation_prompt=True, tokenize=False,
                enable_thinking=request.get("enable_thinking", self.tokenizer.has_thinking),
            )
        return request["prompt"]

    def output_parser(self, request):
        from ..output import OutputParser
        if request.get("tools") or request.get("tool_choice") not in (None, "none", "auto"):
            raise ValueError("tool calling is not qualified for this adapter")
        return OutputParser(
            chat="messages" in request,
            thinking=self.config["model_type"] in {"qwen3", "qwen3_moe"}
            and request.get("enable_thinking", self.tokenizer.has_thinking),
            stops=request.get("stop", ()),
        )

    def diagnostics(self):
        return {
            "architecture": self.config["model_type"],
            "cache_layout": self.layout,
            "qualification": "pending",
            "route": "ordinary",
        }

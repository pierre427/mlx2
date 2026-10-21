"""Strict artifact gate and ordinary serving adapter for Recursive Flint."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from ..sampling_defaults import GENERATION_CONFIG, SamplingDefaults, VendorSampling
from .artifact_paths import shard_within_artifact

SOURCE_REPOSITORY = "RecursiveSI/flint"
SOURCE_REVISION = "2a247be0c0aec895ccb0ac7ca92229ec6df37288"
SOURCE_INDEX_SHA256 = "e5e9c492685f84c8d2d8369f36c74674b03410715f23f6179313f6f1feb90311"
SGLANG_WHEEL_SHA256 = "f9d1ba4ec8ddfb1d09d386e7995f56732431066587a73e0fb458d80de6e605cb"
CACHE_LAYOUT = "flint-full-swa-layer-segments-v1"


DESCRIPTOR = ModelDescriptor(
    model_type="flint",
    family="flint",
    variant="40b-a2.5b-ordinary",
    state_planes=frozenset(
        {
            StatePlane.ATTENTION_KV,
            StatePlane.RNG,
            StatePlane.TRANSCRIPT,
            StatePlane.GRAMMAR,
        }
    ),
    capabilities=frozenset(
        {
            Capability.TEXT,
            Capability.STREAMING,
            Capability.CONTINUOUS_BATCH,
            Capability.PREFIX_REUSE,
            Capability.APC_V2,
            Capability.LAYERED_CACHE,
            Capability.PROMPT_LOOKUP,
            Capability.TOOLS,
            Capability.REASONING,
        }
    ),
    cache_layout=CACHE_LAYOUT,
    metadata={
        "execution": "mlx2.adapters.flint.FlintAdapter",
        "qualification": "pending",
        "scope": "ordinary text decode; model-selected optional reasoning and tools",
    },
)

SAMPLING = VendorSampling.single(
    SamplingDefaults(
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        source=GENERATION_CONFIG,
    ),
    model="RecursiveSI/flint",
)

_EXPECTED = {
    "architectures": ["FlintForCausalLM"],
    "model_type": "flint",
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "num_hidden_layers": 49,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "max_position_embeddings": 262144,
    "sliding_window": 513,
    "moe_intermediate_size": 512,
    "shared_expert_intermediate_size": 1024,
    "num_experts": 256,
    "num_experts_per_tok": 6,
    "vocab_size": 150144,
    "mlp_only_layers": [0, 1],
    "tie_word_embeddings": False,
    "attention_bias": False,
    "hidden_act": "silu",
    "norm_topk_prob": True,
    "router_score_function": "sigmoid",
    "softmax_type": "learnable",
    "rope_theta": 10000.0,
}
_FULL_LAYERS = (0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48)
_LAYER_TYPES = [
    "full_attention" if index in _FULL_LAYERS else "sliding_attention"
    for index in range(49)
]


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
        raise TypeError(f"{path.name} must be an object")
    return value


def _quantization(config: dict, weights: dict) -> dict | None:
    quant = config.get("quantization")
    if quant is None:
        if any(name.endswith((".scales", ".biases")) for name in weights):
            raise ValueError("Flint packed weights require quantization metadata")
        return None
    if not isinstance(quant, dict):
        raise TypeError("Flint quantization must be an object")
    base = {name: quant.get(name) for name in ("group_size", "bits", "mode")}
    if (
        base["group_size"] not in (32, 64, 128)
        or base["bits"] not in (2, 3, 4, 5, 6, 8)
        or base["mode"] != "affine"
    ):
        raise ValueError("unsupported Flint affine quantization default")
    for name, value in quant.items():
        if name in base:
            continue
        if not isinstance(value, dict):
            raise TypeError(f"Flint quantization override is not an object: {name}")
        merged = {**base, **value}
        if (
            merged["group_size"] not in (32, 64, 128)
            or merged["bits"] not in (2, 3, 4, 5, 6, 8)
            or merged["mode"] != "affine"
        ):
            raise ValueError(f"unsupported Flint quantization override: {name}")
    return quant


def inspect_artifact(model_path: str | Path) -> dict:
    path = Path(model_path).expanduser().resolve()
    config = _json(path / "config.json")
    if any(config.get(key) != value for key, value in _EXPECTED.items()):
        raise ValueError("Flint topology differs from the pinned 40B-A2.5B release")
    if config.get("layer_types") != _LAYER_TYPES:
        raise ValueError("Flint full/sliding attention order differs")
    if config.get("no_rope_layer_indices") != list(_FULL_LAYERS):
        raise ValueError("Flint RoPE-free layer order differs")
    if config.get("use_sliding_window") is not True:
        raise ValueError("Flint sliding attention must be enabled")
    if config.get("attention_dropout") not in (0, 0.0):
        raise ValueError("Flint attention dropout must be zero at inference")
    if config.get("rms_norm_eps") != 1e-6:
        raise ValueError("Flint RMSNorm epsilon differs")
    index_path = path / "model.safetensors.index.json"
    index = _json(index_path)
    weights = index.get("weight_map")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("Flint weight index is empty")
    files = []
    for name in sorted(set(weights.values())):
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".safetensors")
        ):
            raise ValueError("Flint shard path is unsafe")
        shard = path / name
        if not shard.is_file() or not shard_within_artifact(path, shard.resolve()):
            raise ValueError(f"Flint shard is missing or escaped: {name}")
        stat = shard.stat()
        files.append((name, stat.st_size, stat.st_mtime_ns))

    required = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    for layer in range(49):
        prefix = f"model.layers.{layer}."
        required.update(
            prefix + suffix
            for suffix in (
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
                "self_attn.softmax_offset",
            )
        )
        if layer < 2:
            required.update(
                prefix + f"mlp.{projection}_proj.weight"
                for projection in ("gate", "up", "down")
            )
        else:
            required.update(
                {
                    prefix + "mlp.gate.weight",
                    prefix + "mlp.shared_expert.gate_proj.weight",
                    prefix + "mlp.shared_expert.up_proj.weight",
                    prefix + "mlp.shared_expert.down_proj.weight",
                }
            )
            source = prefix + "mlp.experts.0.gate_proj.weight"
            packed = prefix + "mlp.switch_mlp.gate_proj.weight"
            if source in weights:
                for expert in range(256):
                    required.update(
                        prefix + f"mlp.experts.{expert}.{projection}_proj.weight"
                        for projection in ("gate", "up", "down")
                    )
            elif packed in weights:
                required.update(
                    prefix + f"mlp.switch_mlp.{projection}_proj.weight"
                    for projection in ("gate", "up", "down")
                )
            else:
                raise ValueError(
                    f"Flint layer {layer} has no complete expert representation"
                )
    missing = required - weights.keys()
    if missing:
        raise ValueError(
            "Flint indexed tensor topology is incomplete: "
            + ", ".join(sorted(missing)[:5])
        )
    quant = _quantization(config, weights)
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
    for record in files:
        digest.update(json.dumps(record).encode())
    return {
        "config": config,
        "weight_map": weights,
        "quantization": quant,
        "source_layout": "packed"
        if any(".switch_mlp." in key for key in weights)
        else "hf-per-expert",
        "identity": {
            "path": str(path),
            "fingerprint": digest.hexdigest(),
            "files": files,
        },
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": config.get("mlx2_conversion", {}).get(
            "source_revision", SOURCE_REVISION
        ),
        "qualification": "pending",
    }


class FlintAdapter:
    default_route = "ordinary"
    descriptor = DESCRIPTOR
    sampling_defaults = SAMPLING
    artifact_inspector = staticmethod(inspect_artifact)
    profile = "flint-apcv2-ordinary"
    tool_call_open_marker = "<tool_call>"

    @staticmethod
    def spomin_backend(model, prompt_cache):
        return None

    def approximate_kv_operations(self):
        return {}

    def profile_name(self, mtp):
        if mtp:
            raise ValueError("Flint has no native MTP route")
        return self.profile

    def execution_config(self, *, max_lanes, prefill_step):
        return {
            "persistent": True,
            "num_draft": 0,
            "rate_gate": False,
            "prefill_step_size": prefill_step,
            "segment_aware_live_tip": False,
            "segment_aware_cohort_size": max_lanes,
        }

    def prefill_step_default(self):
        return 512

    def __init__(self, model_path: str, *, execution_policy=None):
        from .process_globals import guarded_construction

        guarded_construction(
            self,
            lambda: self._init_flint(model_path, execution_policy=execution_policy),
        )

    def _init_flint(self, model_path: str, *, execution_policy=None):
        if execution_policy not in (None, {}):
            raise ValueError("Flint accepts no adapter-owned execution policy")
        artifact = inspect_artifact(model_path)
        self.identity = artifact["identity"]
        self.config = artifact["config"]
        self.layout = CACHE_LAYOUT
        from ..process_env import (
            PROCESS_NUMERICS,
            assert_process_profile_applied,
            clear_inherited_profile,
            require_process_numerics,
        )

        require_process_numerics("the Flint profile")
        clear_inherited_profile(())
        self.environment = {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            **PROCESS_NUMERICS,
        }
        os.environ.update(self.environment)
        assert_process_profile_applied("the Flint profile")
        from .process_globals import claim_stock_moe

        claim_stock_moe(self, "the Flint adapter")

        import mlx.core as mx
        from mlx import nn
        from transformers import AutoTokenizer

        from ..runtime.models.flint import Model, ModelArgs
        from ..runtime.tokenizer_integrity import repair_loaded_tokenizer
        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper
        from ..runtime.ubc_evict import load_shards_evicting

        path = Path(self.identity["path"])
        self.model = Model(ModelArgs.from_dict(self.config))
        files = [path / name for name, _, _ in self.identity["files"]]
        weights = load_shards_evicting(files)
        weights = self.model.sanitize(weights, eager=True)
        quant = artifact["quantization"]
        if quant:
            keep = self.model.quant_predicate

            def predicate(name, module):
                override = quant.get(name)
                if isinstance(override, dict):
                    return override
                return (
                    hasattr(module, "to_quantized")
                    and keep(name, module)
                    and f"{name}.scales" in weights
                )

            nn.quantize(
                self.model,
                group_size=quant["group_size"],
                bits=quant["bits"],
                mode=quant["mode"],
                class_predicate=predicate,
            )
        self.model.load_weights(list(weights.items()), strict=True)
        self.model.eval()
        mx.eval(self.model.parameters())
        weights.clear()
        mx.clear_cache()
        tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        self.pretokenizer_receipt = repair_loaded_tokenizer(tokenizer, path)
        eos = self.config.get("eos_token_id", tokenizer.eos_token_id)
        eos_ids = eos if isinstance(eos, list) else [eos]
        self.tokenizer = TokenizerWrapper(
            tokenizer,
            detokenizer_class=BPEStreamingDetokenizer,
            eos_token_ids=eos_ids,
        )
        self.max_context = int(self.config["max_position_embeddings"])

    def prompt_tokens(self, request: dict) -> list[int]:
        if "messages" not in request:
            return self.tokenizer.encode(request["prompt"], add_special_tokens=False)
        messages = copy.deepcopy(request["messages"])
        for message in messages:
            for call in message.get("tool_calls", []):
                function = call["function"]
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise TypeError("Flint tool arguments must be a JSON object")
                function["arguments"] = arguments
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            tools=request.get("tools")
            if request.get("tool_choice") != "none"
            else None,
        )

    def render_prompt(self, request: dict) -> str:
        if "messages" not in request:
            return request["prompt"]
        messages = copy.deepcopy(request["messages"])
        for message in messages:
            for call in message.get("tool_calls", []):
                function = call["function"]
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                function["arguments"] = arguments
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            tools=request.get("tools")
            if request.get("tool_choice") != "none"
            else None,
        )

    @staticmethod
    def thinking_enabled(request: dict) -> bool:
        # Flint chooses per turn; never tell serving that the generation starts
        # inside a think block.  OutputParser still recognizes a leading one.
        return False

    def thinking_close_token_ids(self):
        ids = self.tokenizer.encode("</think>", add_special_tokens=False)
        return (int(ids[0]),) if len(ids) == 1 else None

    def output_parser(self, request):
        from ..output import OutputParser, constrained_tool_choice
        from ..runtime.tool_parsers.laguna import parse_tool_call

        return OutputParser(
            chat="messages" in request,
            thinking=False,
            tools=request.get("tools")
            if request.get("tool_choice") != "none"
            else None,
            parse_tool=parse_tool_call,
            stops=request.get("stop", ()),
            constrained_tools=constrained_tool_choice(request),
            parallel_tool_calls=request.get("parallel_tool_calls", True),
            tolerant_tool_markers=request.get("_tolerant_tool_markers", False),
        )

    def diagnostics(self):
        return {
            "architecture": "flint-sparse-moe-gqa",
            "cache_layout": self.layout,
            "attention": {
                "full_layers": 14,
                "sliding_layers": 35,
                "sliding_window": 513,
            },
            "experts": {"total": 256, "active": 6, "shared": 1, "packed": True},
            "ordinary_reference": True,
            "qualification": "pending",
            "adapter_implemented": True,
            "adapter_default_route": "ordinary",
            "target_forward_observed": getattr(self.model, "completed_forwards", 0) > 0,
        }

    def close(self):
        from .process_globals import release

        release(self)
        self.model = None
        self.tokenizer = None

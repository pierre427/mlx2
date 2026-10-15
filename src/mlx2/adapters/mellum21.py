"""CPU-safe Mellum 2.1 artifact gate and optimized ordinary adapter."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from ..process_env import (
    PROCESS_NUMERICS,
    assert_process_profile_applied,
    clear_inherited_profile,
    require_process_numerics,
)
from ..sampling_defaults import MODEL_CARD, SamplingDefaults, VendorSampling

CACHE_LAYOUT = "mellum21-mixed-kv-layer-segments-v1"
SOURCE_REPOSITORY = "JetBrains/Mellum2.1-12B-A2.5B-Thinking"
SOURCE_REVISION = "92ddae9fc7665e9f801d141d2e5a6b2caf2460c4"
MTP_SOURCE_LAYER = 28
MTP_REQUIRED_TENSORS = frozenset(
    {
        "mtp.fc.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.norm.weight",
        "mtp.layers.28.self_attn.q_proj.weight",
        "mtp.layers.28.self_attn.k_proj.weight",
        "mtp.layers.28.self_attn.v_proj.weight",
        "mtp.layers.28.self_attn.o_proj.weight",
        "mtp.layers.28.self_attn.q_norm.weight",
        "mtp.layers.28.self_attn.k_norm.weight",
        "mtp.layers.28.input_layernorm.weight",
        "mtp.layers.28.post_attention_layernorm.weight",
        "mtp.layers.28.mlp.gate.weight",
    }
    | {
        f"mtp.layers.28.mlp.experts.{expert}.{projection}.weight"
        for expert in range(64)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
)

MELLUM21_THINKING = ModelDescriptor(
    model_type="mellum",
    family="mellum2.1",
    variant="thinking-ordinary",
    state_planes=frozenset(
        {StatePlane.ATTENTION_KV, StatePlane.RNG, StatePlane.TRANSCRIPT}
    ),
    capabilities=frozenset(
        {
            Capability.TEXT,
            Capability.STREAMING,
            Capability.TOOLS,
            Capability.REASONING,
            Capability.CONTINUOUS_BATCH,
            Capability.PREFIX_REUSE,
            Capability.APC_V2,
            Capability.LAYERED_CACHE,
            Capability.PROMPT_LOOKUP,
            Capability.GRAMMAR,
        }
    ),
    cache_layout=CACHE_LAYOUT,
    metadata={
        "execution": "mlx2.adapters.mellum21.Mellum21ThinkingAdapter",
        "qualification": "pending",
        "scope": "text-only ordinary decode",
        "speculation": "checkpoint has no MTP tensors",
        "ordinary_reference": "implemented",
        "prompt_lookup": "target-verified-exact-rollback",
        "external_draft": "unsupported-no-bound-draft-artifact",
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
    },
)

MELLUM21_SAMPLING = VendorSampling.single(
    SamplingDefaults(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        source=f"{MODEL_CARD} (JetBrains Mellum2.1 Thinking quickstart)",
    ),
    model=SOURCE_REPOSITORY,
)


def _load_json(path: Path) -> dict:
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r} in {path.name}")
            value[key] = item
        return value

    result = json.loads(path.read_text(), object_pairs_hook=unique)
    if not isinstance(result, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return result


def _expected_layers() -> list[str]:
    return [
        "full_attention" if index % 4 == 3 else "sliding_attention"
        for index in range(28)
    ]


def inspect_mtp_schema(config: dict, weight_map: dict) -> dict | None:
    """Validate the published one-layer Mellum MTP *artifact* contract.

    This deliberately does not make the route loadable.  It gives the ordinary
    artifact inspector an exact, adapter-owned boundary for future weights so
    an unfamiliar or partial head is refused before model construction.
    """
    primary = config.get("num_mtp_layers")
    alias = config.get("num_nextn_predict_layers")
    for name, value in (
        ("num_mtp_layers", primary),
        ("num_nextn_predict_layers", alias),
    ):
        if value is not None and type(value) is not int:
            raise ValueError(f"Mellum {name} must be an integer")
    if primary is not None and alias is not None and primary != alias:
        raise ValueError("Mellum MTP layer-count aliases disagree")
    raw_count = primary if primary is not None else alias
    if raw_count is None:
        count = 0
    else:
        count = raw_count
    mtp_keys = {key for key in weight_map if key.startswith("mtp.")}
    if count == 0 and not mtp_keys:
        if any(config.get(name) for name in ("mtp_layer_types", "mtp_mlp_layer_types")):
            raise ValueError("Mellum MTP geometry requires exactly one declared layer")
        return None
    if count != 1:
        raise ValueError("Mellum MTP artifacts must declare exactly one layer")
    if config.get("mtp_layer_types", ["sliding_attention"]) != [
        "sliding_attention"
    ] or config.get("mtp_mlp_layer_types", ["sparse"]) != ["sparse"]:
        raise ValueError("Mellum MTP requires one sliding-attention sparse-MoE layer")
    missing = sorted(MTP_REQUIRED_TENSORS - mtp_keys)
    if missing:
        raise ValueError(f"Mellum MTP artifact is incomplete: missing {missing[0]}")
    extra = sorted(mtp_keys - MTP_REQUIRED_TENSORS)
    if extra:
        raise ValueError(f"Mellum MTP artifact has unexpected tensor: {extra[0]}")
    layer_pattern = re.compile(r"^mtp\.layers\.(\d+)\.")
    source_layers = {
        int(match.group(1))
        for key in mtp_keys
        if (match := layer_pattern.match(key)) is not None
    }
    if source_layers != {MTP_SOURCE_LAYER}:
        raise ValueError(
            "Mellum MTP tensors must use the target-depth source layer 28"
        )
    return {
        "schema": "mlx2.mellum21-mtp-artifact.v1",
        "layers": 1,
        "source_layer": MTP_SOURCE_LAYER,
        "attention": "sliding_attention",
        "mlp": "sparse",
        "projection_order": ("embedding", "previous_hidden"),
        "cache": "private_rotating_kv",
        "post_norm": "mtp.norm",
        "shared_embedding": True,
        "shared_lm_head": True,
        "tensor_count": len(mtp_keys),
        "runtime_load_supported": False,
    }


def inspect_artifact(model_path: str | Path) -> dict:
    """Validate Mellum 2.1 topology, shard closure and identity without MLX."""
    path = Path(model_path).expanduser().resolve()
    config = _load_json(path / "config.json")
    expected = {
        "architectures": ["MellumForCausalLM"],
        "model_type": "mellum",
        "hidden_size": 2304,
        "num_hidden_layers": 28,
        "intermediate_size": 7168,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "num_experts": 64,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 896,
        "vocab_size": 98304,
        "max_position_embeddings": 131072,
        "sliding_window": 1024,
        "norm_topk_prob": True,
        "tie_word_embeddings": False,
        "rms_norm_eps": 1e-6,
        "attention_bias": False,
        "hidden_act": "silu",
        "eos_token_id": 28,
        "use_sliding_window": True,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("artifact topology does not match Mellum 2.1 Thinking")
    if config.get("layer_types") != _expected_layers() or config.get(
        "mlp_layer_types"
    ) != ["sparse"] * 28:
        raise ValueError("artifact layer order does not match Mellum 2.1")
    rope = config.get("rope_parameters")
    full = rope.get("full_attention") if isinstance(rope, dict) else None
    sliding = rope.get("sliding_attention") if isinstance(rope, dict) else None
    if full != {
        "rope_type": "yarn",
        "rope_theta": 500000.0,
        "factor": 16.0,
        "original_max_position_embeddings": 8192,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "attention_factor": 1.2772588722239782,
    } or sliding != {"rope_type": "default", "rope_theta": 500000.0}:
        raise ValueError("artifact RoPE topology does not match Mellum 2.1")
    template_path = path / "chat_template.jinja"
    if not template_path.is_file():
        raise ValueError("Mellum Thinking artifact requires chat_template.jinja")
    template = template_path.read_text()
    if any(marker not in template for marker in ("<think>", "<tool_call>", "enable_thinking")):
        raise ValueError("artifact chat template is not the Mellum Thinking contract")

    index = _load_json(path / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Mellum artifact must have a nonempty weight index")
    mtp_contract = inspect_mtp_schema(config, weight_map)
    if mtp_contract is not None:
        raise ValueError(
            "Mellum MTP artifact schema is recognized, but the route remains "
            "disabled until trained weights receive real-artifact parity and rollback validation"
        )
    extra_decoder_layer = re.compile(r"^(?:model\.)?layers\.(\d+)\.")
    if any(
        key.startswith(("mtp.", "model.mtp.", "language_model.mtp.", "draft."))
        or (
            (match := extra_decoder_layer.match(key)) is not None
            and int(match.group(1)) >= expected["num_hidden_layers"]
        )
        for key in weight_map
    ):
        raise ValueError("embedded speculative tensors are not an ordinary Mellum target")
    required = {
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.mlp.gate.weight",
        "model.norm.weight",
        "lm_head.weight",
    }
    if not required <= set(weight_map):
        raise ValueError("Mellum artifact tensor topology is incomplete")
    individual_experts = {
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.27.mlp.experts.63.down_proj.weight",
    }
    packed_experts = {
        "model.layers.0.mlp.switch_mlp.gate_proj.weight",
        "model.layers.27.mlp.switch_mlp.down_proj.weight",
    }
    if not (individual_experts <= set(weight_map) or packed_experts <= set(weight_map)):
        raise ValueError("Mellum artifact expert tensor topology is incomplete")
    records = []
    for name in sorted(set(weight_map.values())):
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".safetensors")
        ):
            raise ValueError("weight index contains an unsafe shard path")
        item = path / name
        if not item.is_file() or item.stat().st_size < 8:
            raise ValueError(f"missing or empty Mellum weight shard: {name}")
        stat = item.stat()
        records.append((name, stat.st_size, stat.st_mtime_ns))

    quant = config.get("quantization", config.get("quantization_config"))
    if quant is not None and (
        not isinstance(quant, dict)
        or type(quant.get("group_size")) is not int
        or type(quant.get("bits")) is not int
        or quant["group_size"] <= 0
        or quant["bits"] not in {4, 6, 8}
    ):
        raise ValueError("unsupported Mellum quantization configuration")
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
    for record in records:
        digest.update(json.dumps(record).encode())
    revision = path.name if re.fullmatch(r"[0-9a-f]{40}", path.name) else None
    return {
        "config": config,
        "weight_map": weight_map,
        "identity": {
            "path": str(path),
            "fingerprint": digest.hexdigest(),
            "files": records,
            "source_repository": SOURCE_REPOSITORY,
            "source_revision": revision or "unattested-local-copy",
        },
        "layers": 28,
        "sliding_layers": 21,
        "global_layers": 7,
        "sliding_window": 1024,
        "supports_native_mtp": False,
        "qualification": "pending",
    }


def configure_environment() -> dict[str, str]:
    require_process_numerics("the Mellum 2.1 profile")
    clear_inherited_profile(("MLX_QWEN", "MLX_LM_", "MLXUAG_", "MLX_GDN_"))
    profile = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        **PROCESS_NUMERICS,
        "MLX_LM_COMPILED_DECODE": "0",
        "MLX_LM_SEGMENTED_SELF_MTP": "0",
        "MLX_LM_TRUE_BATCHED_SEGMENTED_MTP": "0",
    }
    os.environ.update(profile)
    assert_process_profile_applied("the Mellum 2.1 profile")
    return profile


class Mellum21ThinkingAdapter:
    default_route = "ordinary"
    default_mtp_ordinary_handoff_max_width = None
    descriptor = MELLUM21_THINKING
    sampling_defaults = MELLUM21_SAMPLING
    reasoning_effort_semantics = "thinking_toggle"
    tool_call_open_marker = "<tool_call>"
    weight_streaming_modes = frozenset({"moe_experts"})

    def __init__(self, model_path: str, *, execution_policy=None, weight_streaming=None):
        from .process_globals import guarded_construction

        try:
            guarded_construction(
                self,
                lambda: self._initialize(
                    model_path,
                    execution_policy=execution_policy,
                    weight_streaming=weight_streaming,
                ),
            )
        except BaseException:
            self.close()
            raise

    def _initialize(self, model_path: str, *, execution_policy=None, weight_streaming=None):
        from ..runtime.streamed_load import require_declared

        stream_request = require_declared(type(self), weight_streaming)
        if execution_policy is not None and (
            not isinstance(execution_policy, dict) or execution_policy
        ):
            raise ValueError("Mellum 2.1 has no adapter execution-policy overrides")
        from .process_globals import claim_stock_moe

        artifact = inspect_artifact(model_path)
        self.identity = artifact["identity"]
        self.config = artifact["config"]
        self.layout = CACHE_LAYOUT
        self.environment = configure_environment()
        claim_stock_moe(self, "the Mellum 2.1 ordinary adapter")
        path = Path(self.identity["path"])

        import mlx.core as mx
        from mlx import nn
        from transformers import AutoTokenizer

        from ..runtime.models.mellum import Model, ModelArgs
        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper
        from ..runtime.ubc_evict import load_shards_evicting

        self.model = Model(ModelArgs.from_dict(self.config))
        files = [path / name for name in sorted(set(artifact["weight_map"].values()))]
        quant = self.config.get("quantization", self.config.get("quantization_config"))
        def quantize(weights):
            if not quant:
                return

            base_predicate = self.model.quant_predicate

            def predicate(name, module):
                override = quant.get(name)
                if isinstance(override, dict):
                    return override
                if f"{name}.scales" not in weights:
                    return False
                return base_predicate(name, module)

            nn.quantize(
                self.model,
                group_size=quant["group_size"],
                bits=quant["bits"],
                mode=quant.get("mode", "affine"),
                class_predicate=predicate,
            )

        self.weight_stream = None
        if stream_request is None:
            weights = self.model.sanitize(load_shards_evicting(files), eager=True)
            quantize(weights)
            self.model.load_weights(list(weights.items()), strict=True)
        else:
            from ..runtime.streamed_load import (
                force_stock_expert_arithmetic,
                load_streamed,
            )

            def installed(manager):
                manager.forced_stock = force_stock_expert_arithmetic(self.model)

            loaded = load_streamed(
                self.model,
                path,
                [file.name for file in files],
                request=stream_request,
                sanitize=lambda weights: self.model.sanitize(weights, eager=False),
                quantize=quantize,
                records=self.identity["files"],
                weight_map=artifact["weight_map"],
                top_k=int(self.config["num_experts_per_tok"]),
                on_installed=installed,
            )
            weights = loaded.weights
            self.weight_stream = loaded.manager
        self.model.eval()
        mx.eval(self.model.parameters())
        weights.clear()
        mx.clear_cache()

        tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        from ..runtime.tokenizer_integrity import repair_loaded_tokenizer

        self.pretokenizer_receipt = repair_loaded_tokenizer(tokenizer, path)
        from .eos import artifact_eos_token_ids

        self.tokenizer = TokenizerWrapper(
            tokenizer,
            detokenizer_class=BPEStreamingDetokenizer,
            eos_token_ids=artifact_eos_token_ids(path, self.config, tokenizer),
        )
        self.max_context = int(self.config["max_position_embeddings"])
        if self.weight_stream is not None:
            self.weight_stream.begin_serving()

    def close(self):
        stream = getattr(self, "weight_stream", None)
        if stream is not None:
            stream.close()
            self.weight_stream = None
        if getattr(self, "_process_claim", None) is not None:
            from .process_globals import release

            release(self)

    @staticmethod
    def profile_name(mtp):
        if mtp:
            raise ValueError("Mellum 2.1 checkpoint has no implemented native MTP route")
        return "mellum21-thinking-apcv2-ordinary"

    @staticmethod
    def execution_config(*, max_lanes, prefill_step):
        return {
            "persistent": True,
            "num_draft": 0,
            "backend": "ordinary",
            "rate_gate": False,
            "prefill_step_size": prefill_step,
            "segment_aware_live_tip": False,
            "segment_aware_cohort_size": max_lanes,
        }

    @staticmethod
    def thinking_enabled(request: dict) -> bool:
        return "messages" in request and request.get("enable_thinking", True) is not False

    def thinking_close_token_ids(self):
        try:
            ids = list(self.tokenizer.encode("</think>", add_special_tokens=False))
            return (int(ids[0]),) if len(ids) == 1 else None
        except (AttributeError, TypeError, ValueError):
            return None

    def prompt_tokens(self, request: dict) -> list[int]:
        if "messages" not in request:
            return self.tokenizer.encode(request["prompt"], add_special_tokens=False)
        messages = copy.deepcopy(request["messages"])
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=self.thinking_enabled(request),
            tools=request.get("tools") if request.get("tool_choice") != "none" else None,
        )

    def render_prompt(self, request: dict) -> str:
        if "messages" not in request:
            return request["prompt"]
        return self.tokenizer.apply_chat_template(
            copy.deepcopy(request["messages"]),
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=self.thinking_enabled(request),
            tools=request.get("tools") if request.get("tool_choice") != "none" else None,
        )

    def output_parser(self, request):
        from ..output import OutputParser, constrained_tool_choice
        from .xing_output import parse_tool_block

        return OutputParser(
            chat="messages" in request,
            thinking=self.thinking_enabled(request),
            tools=request.get("tools") if request.get("tool_choice") != "none" else None,
            parse_tool=parse_tool_block,
            stops=request.get("stop", ()),
            constrained_tools=constrained_tool_choice(request),
            parallel_tool_calls=request.get("parallel_tool_calls", True),
            tolerant_tool_markers=request.get("_tolerant_tool_markers", False),
        )

    def route_capability_contract(self):
        """Declare Mellum's ordinary-only exact serving boundary."""
        return {
            "schema": "mlx2.mellum21-route-capability-contract.v1",
            "family": self.descriptor.family,
            "variant": self.descriptor.variant,
            "default_route": "ordinary",
            "ordinary_reference_preserved": True,
            "qualification": "unqualified",
            "apcv2": {
                "implemented": True,
                "exact_only": True,
                "qualified": False,
                "selected": None,
                "selection_authority": "serving_route_receipt",
                "cache_layout": self.descriptor.cache_layout,
                "state_planes": tuple(
                    sorted(plane.value for plane in self.descriptor.state_planes)
                ),
            },
            "prompt_lookup": {
                "implemented": True,
                "target_verified": True,
                "rollback": "exact-mixed-kv-layer-segments",
                "qualified": False,
                "selected": None,
                "selection_authority": "serving_route_receipt",
            },
            "moe_expert_streaming": {
                "implemented": True,
                "qualified": False,
                "eligible": True,
                "selected": None,
                "observed_used": False,
                "default": False,
                "selection_authority": "server_weight_stream_receipt",
                "observation_authority": "server_weight_stream_receipt",
            },
            "native_mtp": {
                "implemented": False,
                "reason": "released artifact contains no MTP tensors",
            },
            "external_draft": {
                "implemented": False,
                "reason": "no revision-bound draft artifact",
            },
        }

    def diagnostics(self):
        return {
            "architecture": "mellum",
            "family": "mellum2.1",
            "cache_layout": self.layout,
            "attention": {
                "sliding_layers": 21,
                "global_layers": 7,
                "sliding_window": 1024,
            },
            "experts": {"total": 64, "active": 8, "packed": True},
            "qualification": "unqualified",
            "adapter_default_route": "ordinary",
            "adapter_implemented": True,
            "selection_authority": "serving_route_receipt",
            "target_forward_observed": getattr(self.model, "completed_forwards", 0) > 0,
            "route_capability_contract": self.route_capability_contract(),
        }

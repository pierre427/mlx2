"""CPU-safe Laguna XS 2.1 artifact inspection and ordinary serving adapter."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from .external_draft_policy import ExternalDraftAdapterMixin
from ..sampling_defaults import GENERATION_CONFIG, SamplingDefaults, VendorSampling

CACHE_LAYOUT = "laguna-xs21-layer-segments-v1"

LAGUNA_XS21 = ModelDescriptor(
    model_type="laguna",
    family="laguna-xs",
    variant="2.1-ordinary",
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
        "execution": "mlx2.adapters.laguna_xs21.LagunaXS21Adapter",
        "qualification": "pending",
        "scope": "text-only ordinary decode",
        "speculation": "none-qualified",
    },
)


def _load_json(path: Path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r} in {path.name}")
            value[key] = item
        return value

    return json.loads(path.read_text(), object_pairs_hook=unique)


def inspect_artifact(model_path: str | Path) -> dict:
    """Validate the supported Laguna topology and shard closure without MLX."""
    path = Path(model_path).expanduser().resolve()
    config = _load_json(path / "config.json")
    expected = {
        "model_type": "laguna",
        "architectures": ["LagunaForCausalLM"],
        "vocab_size": 100352,
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_hidden_layers": 40,
        "num_attention_heads": 48,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "sliding_window": 512,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "mlp_only_layers": [0],
        "moe_routed_scaling_factor": 2.5,
        "norm_topk_prob": True,
        "tie_word_embeddings": False,
        "gating": "per-head",
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("artifact topology does not match Laguna XS 2.1")
    layers = config.get("layer_types")
    required_layers = [
        "full_attention" if index % 4 == 0 else "sliding_attention"
        for index in range(40)
    ]
    if layers != required_layers:
        raise ValueError("artifact layer order does not match Laguna XS 2.1")
    mlp_layers = config.get("mlp_layer_types")
    if mlp_layers != ["dense", *("sparse" for _ in range(39))]:
        raise ValueError("artifact MLP order does not match Laguna XS 2.1")
    heads = config.get("num_attention_heads_per_layer")
    required_heads = [48 if index % 4 == 0 else 64 for index in range(40)]
    if heads != required_heads:
        raise ValueError("artifact attention-head order does not match Laguna XS 2.1")
    index_value = _load_json(path / "model.safetensors.index.json")
    weight_map = index_value.get("weight_map") if isinstance(index_value, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Laguna artifact must have a nonempty weight index")
    if any(marker in key.lower() for key in weight_map for marker in ("mtp.", "eagle", "draft")):
        raise ValueError("embedded speculative tensors are not a Laguna target artifact")
    normalized = {key.removeprefix("language_model.") for key in weight_map}
    required = {
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.1.mlp.gate.proj.weight",
        "model.layers.1.mlp.gate.e_score_correction_bias",
        "model.layers.1.mlp.switch_mlp.gate_proj.weight",
        "model.layers.1.mlp.shared_expert.gate_proj.weight",
        "model.norm.weight",
        "lm_head.weight",
    }
    if not required <= normalized:
        raise ValueError("Laguna artifact tensor topology is incomplete")
    records = []
    for name in sorted(set(weight_map.values())):
        if not isinstance(name, str):
            raise TypeError("weight shard path must be a string")
        item = path / name
        resolved = item.resolve()
        repository_root = path.parent.parent
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or not item.absolute().is_relative_to(path)
            or not resolved.is_relative_to(repository_root)
            or item.suffix != ".safetensors"
            or not item.is_file()
        ):
            raise ValueError("weight index must reference local safetensors files")
        stat = item.stat()
        records.append((name, stat.st_size, stat.st_mtime_ns))
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
    return {
        "config": config,
        "weight_map": weight_map,
        "identity": {
            "path": str(path),
            "fingerprint": digest.hexdigest(),
            "files": records,
        },
        "layers": 40,
        "sliding_layers": 30,
        "global_layers": 10,
        "sliding_window": 512,
        "supports_native_mtp": False,
        "qualification": "pending",
    }


def configure_environment() -> dict[str, str]:
    profile = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "MLX_ENABLE_TF32": "0",
        "MLX_LM_COMPILED_DECODE": "0",
        "MLX_LM_SEGMENTED_SELF_MTP": "0",
        "MLX_LM_TRUE_BATCHED_SEGMENTED_MTP": "0",
        "MLX_LAGUNA_FUSED_DOWN": "stock",
        "MLX_LAGUNA_FUSED_ROUTER": "stock",
    }
    for name in tuple(os.environ):
        if name.startswith(("MLX_QWEN", "MLX_LM_", "MLXUAG_", "MLX_GDN_")):
            del os.environ[name]
    os.environ.update(profile)
    return profile


# Vendor sampling defaults.  generation_config.json: do_sample true,
# temperature 1.0, top_p 1.0, min_p 0.0.  Model card "Best practices"
# (poolside's benchmark settings): temperature 1.0, top_k 20, top_p 1.0.
LAGUNA_XS21_SAMPLING = VendorSampling.single(
    SamplingDefaults(
        temperature=1.0,
        top_p=1.0,
        top_k=20,
        min_p=0.0,
        source=GENERATION_CONFIG,
        field_sources={"top_k": "model card (poolside/Laguna-XS-2.1, Best practices)"},
    ),
    model="poolside/Laguna-XS-2.1",
)


class LagunaXS21Adapter(ExternalDraftAdapterMixin):
    default_route = "ordinary"
    sampling_defaults = LAGUNA_XS21_SAMPLING
    descriptor = LAGUNA_XS21
    reasoning_effort_semantics = "thinking_toggle"

    @staticmethod
    def spomin_backend(model, prompt_cache):
        """Adapter-owned approximate KV surgery for full and sliding attention."""
        from ..runtime.spomin_standard_surgery import StandardAttentionSpominBackend

        return StandardAttentionSpominBackend(model, prompt_cache)

    @staticmethod
    def thinking_enabled(request: dict) -> bool:
        return "messages" in request and request.get("enable_thinking", True) is not False

    def thinking_close_token_ids(self):
        try:
            ids = list(self.tokenizer.encode("</think>", add_special_tokens=False))
            return (int(ids[0]),) if len(ids) == 1 else None
        except (AttributeError, TypeError, ValueError):
            return None

    @staticmethod
    def profile_name(mtp):
        if mtp:
            raise ValueError("Laguna XS 2.1 has no implemented native MTP route")
        return "laguna-xs21-apcv2-ordinary"

    # Candidate external route: poolside's causal DFlash block drafter.  The
    # card measures K=15; every verify row is a full MoE forward on the
    # target, so the rm06 GPU sweep (qualification/runs/rm06-laguna-20260919/
    # greedy-merged.json, B1 greedy, 3 interleaved reps against ordinary
    # 99.0 tok/s) picks the shallowest depth: k3 107.3 tok/s (1.083x), k5
    # 1.007x, k7 0.920x, k11 0.80x.  The previous default of 7 was slower
    # than ordinary.  No qualified receipt pins this route.
    EXTERNAL_DEFAULT_NUM_DRAFT = 3
    EXTERNAL_ROUTE_TAG = "external-laguna-dflash-v1"
    EXTERNAL_PROFILE = "laguna-xs21-apcv2-laguna-dflash"

    def execution_config(self, *, max_lanes, prefill_step):
        if getattr(self, "draft_model", None) is not None:
            return self._external_execution_config(max_lanes=max_lanes, prefill_step=prefill_step)
        return {
            "persistent": True,
            "num_draft": 0,
            "rate_gate": False,
            "prefill_step_size": prefill_step,
            "segment_aware_live_tip": False,
            "segment_aware_cohort_size": max_lanes,
        }

    def __init__(self, model_path: str, *, execution_policy=None):
        external = self._parse_external_policy(execution_policy, family="Laguna")
        artifact = inspect_artifact(model_path)
        draft_record = None
        if external:
            # Header-only drafter inspection before any target tensor loads.
            from .laguna_dflash import inspect_drafter

            draft_record = inspect_drafter(
                self.external_policy["draft_model"], target_config=artifact["config"]
            )
            self._check_num_draft(draft_record)
        self.identity = artifact["identity"]
        self.config = artifact["config"]
        self.environment = configure_environment()
        self.layout = CACHE_LAYOUT
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
        quant = self.config.get("quantization", self.config.get("quantization_config"))
        if quant:
            normalized_quant = {
                key.removeprefix("language_model."): value
                for key, value in quant.items()
            }

            def predicate(name, module):
                override = normalized_quant.get(name)
                if isinstance(override, dict):
                    return override
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
        mx.clear_cache()
        tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False, fix_mistral_regex=True
        )
        eos = self.config.get("eos_token_id")
        self.tokenizer = TokenizerWrapper(
            tokenizer,
            detokenizer_class=BPEStreamingDetokenizer,
            eos_token_ids=[eos] if isinstance(eos, int) else list(eos or []),
        )
        self.max_context = int(self.config["max_position_embeddings"])
        if draft_record is not None:
            from .laguna_dflash import load_drafter

            self._bind_external_drafter(draft_record, load_drafter, LAGUNA_XS21)

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
                    raise TypeError("Laguna tool arguments must be a JSON object")
                function["arguments"] = arguments
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            enable_thinking=self.thinking_enabled(request),
            tools=request.get("tools") if request.get("tool_choice") != "none" else None,
        )

    def output_parser(self, request):
        from ..output import OutputParser, constrained_tool_choice
        from ..runtime.tool_parsers.laguna import parse_tool_call

        return OutputParser(
            chat="messages" in request,
            thinking=self.thinking_enabled(request),
            tools=request.get("tools") if request.get("tool_choice") != "none" else None,
            parse_tool=parse_tool_call,
            stops=request.get("stop", ()),
            constrained_tools=constrained_tool_choice(request),
            parallel_tool_calls=request.get("parallel_tool_calls", True),
            tolerant_tool_markers=request.get("_tolerant_tool_markers", False),
        )

    def diagnostics(self):
        kernel_stats = self.model.fused_moe_stats()
        return {
            "architecture": "laguna-sparse-moe-gqa",
            "cache_layout": self.layout,
            "sliding_layers": 30,
            "global_layers": 10,
            "sliding_window": 512,
            "speculation": (
                "external-laguna-dflash-implemented-unqualified"
                if getattr(self, "draft_model", None) is not None
                else "none-qualified"
            ),
            "fused_moe": {
                "status": "implemented-candidate-unselected",
                "candidate_token_widths": [1, 2, 4, 8],
                "qualified_token_widths": [],
                "selection_reason": (
                    "exact-artifact full-model qualification was slower and did not "
                    "preserve the reference top logit"
                ),
                "qualification_receipt": (
                    "qualification/runs/laguna-fused-moe-20260918/full-model.json"
                ),
                **kernel_stats,
            },
            "qualification": "pending",
        }

    def set_fused_moe_modes(self, *, down: bool, router: bool):
        """Select benchmark arms without reloading weights."""
        self.model.set_fused_moe_modes(down=down, router=router)

    def close(self):
        had_resources = any(getattr(self, name, None) is not None for name in ("model", "tokenizer"))
        self.model = None
        self.draft_model = None
        self.tokenizer = None
        if had_resources:
            import mlx.core as mx
            mx.clear_cache()

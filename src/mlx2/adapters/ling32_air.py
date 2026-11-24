"""Ling-3.2-air text target on the Ling hybrid KDA + gated-MLA runtime.

Ling-3.2-air (inclusionAI, MIT) shares Ling-3.0-flash-VL's architecture:
35 KDA + 7 head-wise gated MLA layers (5:1), 512 routed experts (top-8,
sigmoid group routing) + 1 shared expert, 2 dense layers, one MTP layer and a
Qwen3-MoE ViT.  Relative to the Ling 3 Flash oQ4 adapter this target adds:

* the vendor's per-layer clamped SwiGLU (``expert_swiglu_limit_list`` /
  ``share_expert_swiglu_limit_list``; post-activation law, see
  ``bailing_hybrid.clamped_swiglu``) — live on layers 34-41;
* an fp32, unquantized router (``router_dtype: fp32``);
* tool calls in the template's ``<tool_call>``/``<arg_key>`` XML format;
* the model card's sampling defaults (T=1.0, top_p=0.95, top_k=20).

The served artifact is the mlx2 conversion of the vendor block-FP8 checkpoint
(``mlx2.conversion.ling32_air``; recipe ``ling32-air-affine-e4g64-r8g64-v1``).
Scope is text-only; the vision shard is carried but not served.
Provenance: provenance/ling32-air.json.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace

from ..contracts import Capability, ModelDescriptor, StatePlane
from ..sampling_defaults import MODEL_CARD, SamplingDefaults, VendorSampling
from . import ling3_flash
from .ordinary_artifact import inspect_indexed_artifact

RECIPE = "ling32-air-affine-e4g64-r8g64-v1"
SOURCE_REVISION = "1613b1a4dcc6aa0d8f346606628cd9a92061a525"
CACHE_LAYOUT = "ling32-air-hybrid-kda-gated-mla-v1"
EXPERT_LIMITS = [0] * 35 + [4] * 7
SHARED_LIMITS = [0] * 34 + [5] * 6 + [7] * 2

DESCRIPTOR = replace(
    ling3_flash.DESCRIPTOR,
    family="ling-3.2-air",
    variant="mlx2-e4r8-ordinary",
    capabilities=ling3_flash.DESCRIPTOR.capabilities | {Capability.TOOLS},
    cache_layout=CACHE_LAYOUT,
    metadata={
        **ling3_flash.DESCRIPTOR.metadata,
        "execution": "mlx2.adapters.ling32_air.Ling32AirAdapter",
        "qualification": "pending",
        "scope": "text-only",
        "qualified": False,
        "selected": False,
        "observed_used": False,
    },
)

def descriptor_for(*, has_mtp: bool) -> ModelDescriptor:
    """Native MTP adds the draft plane; it stays unqualified and unselected."""
    if not has_mtp:
        return DESCRIPTOR
    return replace(
        DESCRIPTOR,
        variant="mlx2-e4r8-mtp",
        state_planes=DESCRIPTOR.state_planes | {StatePlane.DRAFT},
        capabilities=DESCRIPTOR.capabilities | {Capability.MTP},
        metadata={**DESCRIPTOR.metadata, "native_mtp": "implemented-unqualified"},
    )


# Keys of the native-MTP tree-tile route on top of Ling's tree controls
# (``draft_model`` is the target artifact itself).
NATIVE_TREE_KEYS = frozenset({
    "ast_topology", "pairwise_selection", "tree_node_budget_by_lanes",
    "tensorfold_cohort_limit",
})


SAMPLING = VendorSampling.single(
    SamplingDefaults(
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        source=MODEL_CARD,
        note="inclusionAI/Ling-3.2-air model card (enable_thinking true)",
    ),
    model="inclusionAI/Ling-3.2-air",
)


def is_ling32_air(config: dict) -> bool:
    return str((config.get("mlx2_conversion") or {}).get("recipe", "")).startswith("ling32-air")


def inspect_artifact(model_path):
    artifact = inspect_indexed_artifact(model_path)
    config = artifact["config"]
    conversion = config.get("mlx2_conversion") or {}
    if conversion.get("recipe") != RECIPE or conversion.get("source_revision") != SOURCE_REVISION:
        raise ValueError("artifact is not the pinned mlx2 Ling-3.2-air conversion")
    expected = {
        "model_type": "bailing_hybrid",
        "architectures": ["BailingMoeV3ForCausalLM"],
        "vocab_size": 157184,
        "hidden_size": 2560,
        "intermediate_size": 6144,
        "moe_intermediate_size": 768,
        "moe_shared_expert_intermediate_size": 768,
        "num_hidden_layers": 42,
        "num_nextn_predict_layers": 1,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "num_experts": 512,
        "num_experts_per_tok": 8,
        "num_shared_experts": 1,
        "n_group": 8,
        "topk_group": 4,
        "first_k_dense_replace": 2,
        "layer_group_size": 6,
        "max_position_embeddings": 262144,
        "head_dim": 128,
        "kv_lora_rank": 512,
        "q_lora_rank": None,
        "qk_rope_head_dim": 64,
        "qk_nope_head_dim": 128,
        "v_head_dim": 128,
        "rope_theta": 6000000,
        "routed_scaling_factor": 2.5,
        "score_function": "sigmoid",
        "router_dtype": "fp32",
        "short_conv_kernel_size": 4,
        "no_kda_lora": True,
        "kda_safe_gate": True,
        "kda_lower_bound": -5.0,
        "mtp_use_kda": False,
        "gated_attention_proj_granularity_type": "head_wise",
        "expert_swiglu_limit_list": EXPERT_LIMITS,
        "share_expert_swiglu_limit_list": SHARED_LIMITS,
        "tie_word_embeddings": False,
    }
    wrong = sorted(key for key, value in expected.items() if config.get(key) != value)
    if wrong:
        raise ValueError(f"artifact topology does not match Ling-3.2-air: {wrong[:6]}")
    # Off-default architecture switches the runtime does not implement must stay off.
    for key in ("use_nGPT", "use_mla_nope", "value_norm", "up_proj_norm", "scale_router_input"):
        if config.get(key):
            raise ValueError(f"Ling-3.2-air switch {key} is not supported")
    quant = config.get("quantization")
    if not isinstance(quant, dict) or (quant.get("bits"), quant.get("group_size"), quant.get("mode")) != (
        8, 64, "affine"
    ):
        raise ValueError("Ling-3.2-air target requires the mlx2 e4r8 affine artifact")
    weights = artifact["weight_map"]
    for index in range(2, 43):
        prefix = f"model.layers.{index}." if index < 42 else "mtp.layers.0."
        for projection in ("gate_proj", "up_proj", "down_proj"):
            module = f"{prefix}mlp.switch_mlp.{projection}"
            if quant.get(module) != {"group_size": 64, "bits": 4} or module + ".scales" not in weights:
                raise ValueError(f"Ling-3.2-air routed experts are not 4-bit g64: {module}")
        if f"{prefix}mlp.gate.weight" not in weights or f"{prefix}mlp.gate.scales" in weights:
            raise ValueError("Ling-3.2-air router must be present and unquantized")
    required = {
        "model.word_embeddings.weight", "model.norm.weight", "lm_head.weight",
        "mtp.layers.0.eh_proj.weight", "mtp.layers.0.enorm.weight",
        "mtp.layers.0.hnorm.weight", "mtp.layers.0.final_layernorm.weight",
        "mtp.layers.0.attention.kv_b_proj.weight", "mtp.layers.0.attention.g_proj.weight",
    }
    for index in range(42):
        prefix = f"model.layers.{index}.attention."
        if (index + 1) % 6 == 0:
            required |= {prefix + name for name in (
                "q_proj.weight", "kv_a_proj_with_mqa.weight", "kv_a_layernorm.weight",
                "kv_b_proj.weight", "g_proj.weight", "dense.weight")}
        else:
            required |= {prefix + name for name in (
                "q_proj.weight", "k_proj.weight", "v_proj.weight", "f_proj.weight",
                "g_proj.weight", "b_proj.weight", "q_conv1d.weight", "k_conv1d.weight",
                "v_conv1d.weight", "A_log", "dt_bias", "o_norm.weight", "o_proj.weight")}
    missing = sorted(required - weights.keys())
    if missing:
        raise ValueError(f"Ling-3.2-air tensors are incomplete: {missing[:8]}")
    artifact["qualification"] = "pending"
    artifact["has_mtp"] = True
    return artifact


class Ling32AirAdapter(ling3_flash.Ling3FlashAdapter):
    descriptor = descriptor_for(has_mtp=True)
    default_route = "ordinary"
    EXTERNAL_POLICY_EXTENSIONS = ling3_flash.Ling3FlashAdapter.EXTERNAL_POLICY_EXTENSIONS | NATIVE_TREE_KEYS
    sampling_defaults = SAMPLING
    artifact_inspector = staticmethod(inspect_artifact)
    profile = "ling32-air-e4r8-apcv2-ordinary"
    family_label = "Ling-3.2-air"
    # Pierre 2026-11-23: kernels that pass the checkpoint smoke go default on.
    # kda_fused passed (bitwise, engaged, not slower); routed_moe_decode was
    # slower and stays off (see ling3_flash.optimization_policy).
    # {"ling_optimizations": {"kda_fused": false}} selects the eager path.
    optimization_overrides = {"kda_fused": True}
    cache_layout_base = CACHE_LAYOUT
    tree_variant_prefix = "mlx2-e4r8-mtp-ast"
    # No Ling-3.2-air UBTree drafter exists; the 3.0 drafter is a different
    # target. The external route stays fail-closed until one is bound.
    EXTERNAL_ROUTE_TAG = "external-ubtree-ling32-air-v1"
    EXTERNAL_PROFILE = "ling32-air-e4r8-apcv2-mtp-tree"
    EXTERNAL_DEFAULT_NUM_DRAFT = 5

    def __init__(self, model_path, *, execution_policy=None, weight_streaming=None):
        from .mtp_depth_cap import validate_self_mtp_num_draft

        policy = dict(execution_policy or {})
        # Native MTP chain route: ``num_draft`` without a draft_model (the
        # vLLM recipe on the card uses 3; sglang NEXTN defaults lower).
        self._num_draft = 2
        if "draft_model" not in policy and "num_draft" in policy:
            self._num_draft = validate_self_mtp_num_draft(policy.pop("num_draft"))
        elif "draft_model" in policy:
            # Native-MTP tree tiles: anchor + the default 3x5 topology (depth 5).
            policy.setdefault("num_draft", self.EXTERNAL_DEFAULT_NUM_DRAFT)
        super().__init__(model_path, execution_policy=policy, weight_streaming=weight_streaming)
        self.page_in_receipt = None
        if os.environ.get("MLX2_PAGE_IN_WEIGHTS") == "1" and self.weight_stream is None:
            self.page_in_receipt = self._page_in_weights()

    def _page_in_weights(self):
        """Read every resident parameter once (opt-in ``MLX2_PAGE_IN_WEIGHTS=1``).

        The shards are memory-mapped, so a 70 GB artifact otherwise pages in on
        first use, during the first requests. A night window's swap guard
        needs that paging (and the idle-page eviction it causes) to finish
        before the server answers /health. Residency only: no math changes, so
        identity and cache layout are unchanged.
        """
        import time

        import mlx.core as mx
        from mlx.utils import tree_flatten

        started, touched = time.time(), 0
        for _name, value in tree_flatten(self.model.parameters()):
            mx.eval(mx.sum(value.view(mx.uint8)))
            touched += value.nbytes
        return {"seconds": round(time.time() - started, 1), "bytes": touched}

    def _model_kwargs(self):
        return {"mtp": True}

    def _model_arg_overrides(self):
        # Clamped SwiGLU, fp32 router + route weights, -inf group masking.
        return {"vendor_v32_laws": True}

    def _bind_optimizations(self):
        # The fp32 router computes x32 @ W32.T itself (Gate.router_fp32) and
        # never calls its Linear, so no lane law can or should cover it. Its
        # row invariance is a GPU probe (decode row vs verify rows), not assumed.
        for _name, module in self.model.named_modules():
            if getattr(module, "router_fp32", False):
                module.gate_proj._ling_lane_exempt = "fp32-router-direct-matmul"
        super()._bind_optimizations()

    def profile_name(self, mtp):
        if mtp:
            return f"ling32-air-e4r8-apcv2-mtp{self._num_draft}"
        return self.profile

    def execution_config(self, *, max_lanes, prefill_step):
        if self.draft_model is not None:
            return super().execution_config(max_lanes=max_lanes, prefill_step=prefill_step)
        return {
            "persistent": True,
            "num_draft": self._num_draft,
            "rate_gate": False,
            "prefill_step_size": prefill_step,
            "segment_aware_live_tip": True,
            "segment_aware_cohort_size": max_lanes,
        }

    def cache_budget(self, *, mtp):
        if getattr(self, "draft_model", None) is not None:
            return None
        from .ling_memory import LingCacheBudget

        return LingCacheBudget.from_config(self.config, mtp=mtp)

    def _inspect_external_drafter(self, model_path):
        from .qwen38_27b import _names_target

        if not _names_target(self.external_policy.get("draft_model"), model_path):
            raise ValueError(
                "Ling-3.2-air has no external drafter family; draft_model must be the "
                "target artifact itself (its native MTP head drafts the AST tree tiles)"
            )
        from ..runtime.ast_tree_proposals import DEFAULT_TOPOLOGY, parse_topology

        policy = self.external_policy
        topology = parse_topology(policy.setdefault("ast_topology", DEFAULT_TOPOLOGY))
        deepest = max(topology) if isinstance(topology, tuple) else 15
        if policy["num_draft"] < deepest:
            raise ValueError("Ling-3.2-air num_draft must cover the AST topology depth")
        policy.setdefault("pairwise_selection", "host")
        return "native_mtp", None

    def _bind_loaded_drafter(self, draft_record, draft_format):
        if draft_record != "native_mtp":
            raise ValueError("Ling-3.2-air binds only its native MTP head")
        from .qwen38_27b import bind_tree_tile_drafter

        bind_tree_tile_drafter(self, self.identity["path"], prompt_lookup=False,
                               descriptor_for=descriptor_for)
        self.descriptor = replace(
            self.descriptor,
            variant="mlx2-e4r8-mtp-ast-tree",
            metadata={**self.descriptor.metadata, "qualification": "unqualified",
                      "implemented": True, "selected": True, "observed_used": False,
                      "target_tree_executor": self.external_policy.get("ast_target_executor", "packed")},
        )
        if self.external_policy.get("ast_graft_prompt_lookup", False):
            from .qwen38_27b import graft_prompt_lookup_drafter

            graft_prompt_lookup_drafter(self, prompt_lookup=False)
        self._bind_tree_controls()

    def create_external_batch(self, **kwargs):
        draft = self.draft_model
        base = getattr(draft, "base", draft)
        if getattr(base, "receipt_kind", None) != "external_native_mtp":
            raise ValueError("Ling-3.2-air AST requires its bound native MTP head")
        from ..runtime.hybrid_tree_executor import HybridTreeExecutor
        from ..runtime.ling_tree_executor import LingTreeExecutor

        if kwargs.get("dynamic_tree_overflow", "cohort") != "cohort":
            raise ValueError("Ling AST requires bounded tree cohort overflow")
        kwargs.setdefault("dynamic_tree_overflow", "cohort")
        kwargs.setdefault("dynamic_singleton_tree", True)
        kwargs.setdefault("dynamic_tree_max_width", 4)
        kwargs.setdefault("tensorfold_cohort_limit", 4)
        rows, adaptive = ling3_flash._tree_controls(self.external_policy)
        kwargs.setdefault("tree_node_budget_by_lanes", {
            width: max(1, rows // width - 1) for width in range(1, 5)
        })
        if adaptive is not None:
            kwargs.setdefault("ast_adaptive_rows", adaptive)
        serial = self.external_policy.get("ast_target_executor", "packed") == "serial"
        if "target_tree_executor_provider" not in kwargs:
            kwargs["target_tree_executor_provider"] = (
                HybridTreeExecutor.create() if serial
                else LingTreeExecutor(optimizations=getattr(self, "optimizations", None))
            )
        # Skip Ling3Flash's UBTree-only guard; the shared mixin builds the batch.
        return super(ling3_flash.Ling3FlashAdapter, self).create_external_batch(**kwargs)

    def _inspect(self, model_path):
        return inspect_artifact(model_path)

    def eos_token_ids(self):
        # generation_config.json: [156892 (<|endoftext|>), 156895 (<|role_end|>)].
        from pathlib import Path

        ids = [int(self.config["eos_token_id"])]
        path = Path(self.identity["path"]) / "generation_config.json"
        declared = json.loads(path.read_text()).get("eos_token_id", []) if path.is_file() else []
        for value in declared if isinstance(declared, list) else [declared]:
            if int(value) not in ids:
                ids.append(int(value))
        return ids

    def _template_messages(self, request):
        from .ordinary_text import _template_history

        # Decodes OpenAI argument strings and refuses non-object arguments.
        return _template_history(request["messages"])

    def _template_tools(self, request):
        from ..output import callable_tools

        return callable_tools(request)

    def prompt_tokens(self, request):
        if "messages" not in request:
            return self.tokenizer.encode(request["prompt"], add_special_tokens=False)
        return self.tokenizer.apply_chat_template(
            self._template_messages(request), add_generation_prompt=True, tokenize=True,
            enable_thinking=self.thinking_enabled(request), tools=self._template_tools(request),
        )

    def render_prompt(self, request):
        if "messages" not in request:
            return request["prompt"]
        return self.tokenizer.apply_chat_template(
            self._template_messages(request), add_generation_prompt=True, tokenize=False,
            enable_thinking=self.thinking_enabled(request), tools=self._template_tools(request),
        )

    def output_parser(self, request):
        from ..output import OutputParser, callable_tools, constrained_tool_choice
        # Same <tool_call>name / <arg_key> / <arg_value> XML (and tojson for
        # non-string values) as the Laguna template; see provenance/ling32-air.json.
        from ..runtime.tool_parsers.laguna import parse_tool_call

        return OutputParser(
            chat="messages" in request,
            thinking=self.thinking_enabled(request),
            tools=callable_tools(request),
            parse_tool=parse_tool_call,
            stops=request.get("stop", ()),
            constrained_tools=constrained_tool_choice(request),
            parallel_tool_calls=request.get("parallel_tool_calls", True),
            tolerant_tool_markers=request.get("_tolerant_tool_markers", False),
        )


__all__ = ["DESCRIPTOR", "Ling32AirAdapter", "inspect_artifact", "is_ling32_air"]

"""Ling 3 Flash hybrid target and its explicit UBTree AST route."""

from __future__ import annotations

import hashlib
import os
from dataclasses import replace
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from ..process_env import (
    PROCESS_NUMERICS,
    assert_process_profile_applied,
    clear_inherited_profile,
    require_process_numerics,
)
from ..sampling_defaults import MODEL_CARD, SamplingDefaults, VendorSampling
from .external_draft_policy import ExternalDraftAdapterMixin
from .ordinary_artifact import inspect_indexed_artifact
from .ordinary_text import OrdinaryTextAdapter

CACHE_LAYOUT = "ling3-flash-hybrid-kda-bounded-mla-v3"


def _tree_controls(policy):
    rows = policy.get("ast_tile_rows", 16)
    if type(rows) is not int or not 4 <= rows <= 32:
        raise ValueError("Ling AST tile rows must be an integer from 4 to 32")
    adaptive = policy.get("ast_adaptive_rows")
    if adaptive is not None and type(adaptive) is not bool:
        raise ValueError("Ling adaptive tree rows must be boolean")
    if policy.get("ast_target_executor", "packed") not in ("packed", "serial"):
        raise ValueError("Ling AST target executor must be packed or serial")
    if policy.get("ast_target_executor") == "serial" and rows > 16:
        raise ValueError("Ling serial reference supports at most 16 tile rows")
    return rows, adaptive


DESCRIPTOR = ModelDescriptor(
    model_type="bailing_hybrid",
    family="ling-3.0-flash",
    variant="oq4-ordinary",
    state_planes=frozenset(
        {
            StatePlane.ATTENTION_KV,
            StatePlane.RECURRENT,
            StatePlane.RNG,
            StatePlane.TRANSCRIPT,
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
        }
    ),
    cache_layout=CACHE_LAYOUT,
    metadata={
        "execution": "mlx2.adapters.ling3_flash.Ling3FlashAdapter",
        "qualification": "pending",
        "scope": "text-only",
        "implemented": True,
        "qualified": False,
        "selected": False,
        "observed_used": False,
    },
)

SAMPLING = VendorSampling.single(
    SamplingDefaults(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        source=MODEL_CARD,
        note="TensorFold Ling-3.0-flash-oQ4 model card",
    ),
    model="TensorFold/Ling-3.0-flash-oQ4",
)


def inspect_artifact(model_path):
    artifact = inspect_indexed_artifact(model_path)
    config = artifact["config"]
    expected = {
        "model_type": "bailing_hybrid",
        "architectures": ["BailingMoeV3ForCausalLM"],
        "vocab_size": 157184,
        "hidden_size": 2560,
        "intermediate_size": 6144,
        "moe_intermediate_size": 768,
        "num_hidden_layers": 42,
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
        "qk_rope_head_dim": 64,
        "qk_nope_head_dim": 128,
        "v_head_dim": 128,
        "short_conv_kernel_size": 4,
        "no_kda_lora": True,
        "kda_safe_gate": True,
        "kda_lower_bound": -5.0,
        "gated_attention_proj_granularity_type": "head_wise",
        "tie_word_embeddings": False,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("artifact topology does not match Ling 3 Flash")
    quant = config.get("quantization")
    if not isinstance(quant, dict) or any(
        quant.get(key) != value
        for key, value in {"group_size": 64, "bits": 4, "mode": "affine"}.items()
    ):
        raise ValueError("Ling 3 Flash target requires the oQ4 affine artifact")
    weights = artifact["weight_map"]
    required = {
        "model.word_embeddings.weight",
        "model.word_embeddings.scales",
        "model.word_embeddings.biases",
        "model.norm.weight",
        "lm_head.weight",
        "lm_head.scales",
        "lm_head.biases",
    }
    global_layers = {5, 11, 17, 23, 29, 35, 41}
    for index in range(42):
        prefix = f"model.layers.{index}."
        required.update(
            {
                prefix + "input_layernorm.weight",
                prefix + "post_attention_layernorm.weight",
            }
        )
        if index in global_layers:
            required.update(
                {
                    prefix + "attention.q_proj.weight",
                    prefix + "attention.kv_a_proj_with_mqa.weight",
                    prefix + "attention.kv_a_layernorm.weight",
                    prefix + "attention.embed_q.weight",
                    prefix + "attention.unembed_out.weight",
                    prefix + "attention.dense.weight",
                }
            )
        else:
            required.update(
                {
                    prefix + "attention.q_proj.weight",
                    prefix + "attention.k_proj.weight",
                    prefix + "attention.v_proj.weight",
                    prefix + "attention.q_conv1d.weight",
                    prefix + "attention.k_conv1d.weight",
                    prefix + "attention.v_conv1d.weight",
                    prefix + "attention.A_log",
                    prefix + "attention.dt_bias",
                    prefix + "attention.o_proj.weight",
                }
            )
    if not required <= weights.keys():
        missing = sorted(required - weights.keys())
        raise ValueError(f"Ling 3 Flash tensors are incomplete: {missing[:8]}")
    artifact["qualification"] = "pending"
    return artifact


def configure_environment():
    require_process_numerics("the Ling 3 Flash profile")
    profile = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        **PROCESS_NUMERICS,
        "MLX_LM_COMPILED_DECODE": "0",
    }
    clear_inherited_profile(("MLX_QWEN", "MLX_LM_", "MLXUAG_", "MLX_GDN_"))
    os.environ.update(profile)
    assert_process_profile_applied("the Ling 3 Flash profile")
    return profile


class Ling3FlashAdapter(ExternalDraftAdapterMixin, OrdinaryTextAdapter):
    descriptor = DESCRIPTOR
    sampling_defaults = SAMPLING
    artifact_inspector = staticmethod(inspect_artifact)
    profile = "ling3-flash-oq4-apcv2-ordinary"
    default_route = "ordinary"
    # Stacked affine expert rows retain checkpoint byte identity. The shared
    # loader installs bounded row caches before materializing any expert table.
    weight_streaming_modes = frozenset({"moe_experts"})
    ast_prompt_lookup_tiles = False
    ast_tile_rows_max = 32
    ast_tree_tiles = True
    # The generic header gate protects Qwen TensorFold kernels whose private
    # state is hard-coded to bf16. Ling's provider owns its KDA/MLA math and
    # uses the generic installed projection law, including TensorFold lanes.
    ast_tree_tile_dtype_gate = False
    EXTERNAL_DEFAULT_NUM_DRAFT = 15
    EXTERNAL_ROUTE_TAG = "external-ubtree-ling3-v1"
    EXTERNAL_PROFILE = "ling3-flash-oq4-apcv2-ubtree"
    ast_external_batch_size_route = "tree15_bounded_cohorts_v1"
    # ``server.apply_ast_invocation`` adds this exact tree-tile selection for
    # an explicit ``--ast-drafter`` artifact.
    EXTERNAL_POLICY_EXTENSIONS = frozenset({
        "batch_size_route", "ast_graft_prompt_lookup", "ast_tile_rows", "ast_adaptive_rows",
        "ast_target_executor",
    })

    def __init__(self, model_path, *, execution_policy=None, weight_streaming=None):
        from .process_globals import guarded_construction

        self.weight_stream = None
        try:
            guarded_construction(
                self,
                lambda: self._init_ling(
                    model_path, execution_policy=execution_policy,
                    weight_streaming=weight_streaming,
                ),
            )
        except BaseException:
            self.close()
            raise

    def _parse_external_policy(self, execution_policy, *, family):
        external = super()._parse_external_policy(execution_policy, family=family)
        if external:
            route = self.external_policy.get("batch_size_route")
            if route not in (
                None, "tree15_b1_b4_chain_b5plus_v1", self.ast_external_batch_size_route,
            ):
                raise ValueError("Ling UBTree supports only the bounded tree15 AST route")
            # Accept the former generic CLI selection as a compatibility alias;
            # expose the actual tree-only policy before identity/config binding.
            self.external_policy["batch_size_route"] = self.ast_external_batch_size_route
        return external

    def _init_ling(self, model_path, *, execution_policy=None, weight_streaming=None):
        from ..runtime.streamed_load import require_declared

        stream_request = require_declared(type(self), weight_streaming)
        external = self._parse_external_policy(execution_policy, family="Ling 3 Flash")
        _tree_controls(self.external_policy)
        if type(self.external_policy.get("ast_graft_prompt_lookup", False)) is not bool:
            raise ValueError("Ling AST prompt-lookup graft must be boolean")
        if external:
            # Current AST hardware tiles hold an anchor plus 15 nodes.  The
            # checkpoint's B64 search stays represented in its config/receipt;
            # serving uses an explicit prefix-closed B15 projection.
            self.external_policy.setdefault("num_draft", 15)
        artifact = inspect_artifact(model_path)
        self.config = artifact["config"]
        self.identity = artifact["identity"]
        self.layout = CACHE_LAYOUT
        self.environment = configure_environment()
        # bailing_hybrid builds mlx2 switch_layers experts (stock sorted
        # gather, no NAX scope): pin the MoE process globals for this load.
        from .process_globals import claim_stock_moe

        claim_stock_moe(self, "the Ling 3 Flash adapter")
        draft_record = draft_format = None
        if external:
            from .external_drafter_formats import drafter_format

            draft_format = drafter_format(self.external_policy["draft_model"])
            if draft_format.name != "ubtree":
                raise ValueError(
                    "Ling 3 Flash external route requires Ling3UBTreeModel"
                )
            draft_record = draft_format.inspect_drafter(
                self.external_policy["draft_model"], model_path
            )
            self._check_num_draft(draft_record)

        path = Path(self.identity["path"])
        import mlx.core as mx
        from mlx import nn
        from transformers import AutoTokenizer

        from ..runtime.models.bailing_hybrid import Model, ModelArgs
        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper
        from ..runtime.ubc_evict import load_shards_evicting

        self.model = Model(ModelArgs.from_dict(self.config))
        files = [path / name for name in sorted(set(artifact["weight_map"].values()))]
        quant = self.config["quantization"]

        def quantize(weights):
            def predicate(name, module):
                override = quant.get(name)
                if isinstance(override, dict):
                    return override
                return hasattr(module, "to_quantized") and f"{name}.scales" in weights

            nn.quantize(
                self.model,
                group_size=quant["group_size"],
                bits=quant["bits"],
                mode=quant["mode"],
                class_predicate=predicate,
            )

        if stream_request is None:
            weights = self.model.sanitize(load_shards_evicting(files))
            quantize(weights)
            self.model.load_weights(list(weights.items()), strict=True)
        else:
            from ..runtime.streamed_load import (
                force_stock_expert_arithmetic,
                load_streamed,
            )

            loaded = load_streamed(
                self.model, path, [file.name for file in files],
                request=stream_request,
                sanitize=self.model.sanitize,
                quantize=quantize,
                records=self.identity["files"],
                weight_map=artifact["weight_map"],
                top_k=int(self.config["num_experts_per_tok"]),
                on_installed=lambda manager: setattr(
                    manager, "forced_stock", force_stock_expert_arithmetic(self.model)
                ),
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
        self.tokenizer = TokenizerWrapper(
            tokenizer,
            detokenizer_class=BPEStreamingDetokenizer,
            eos_token_ids=[int(self.config["eos_token_id"])],
        )
        self.max_context = int(self.config["max_position_embeddings"])
        if draft_record is not None:
            self._bind_external_drafter(
                draft_record, draft_format.load_drafter, DESCRIPTOR
            )
            self.descriptor = replace(
                self.descriptor,
                variant="oq4-ubtree-ast-b15",
                metadata={
                    **self.descriptor.metadata,
                    "qualification": "unqualified",
                    "implemented": True,
                    "selected": True,
                    "observed_used": False,
                    "checkpoint_tree_budget": 64,
                    "served_tree_budget": 15,
                    "target_tree_executor": self.external_policy.get("ast_target_executor", "packed"),
                },
            )
            if self.external_policy.get("ast_graft_prompt_lookup", False):
                self._graft_prompt_lookup()
            self._bind_tree_controls()
        if self.weight_stream is not None:
            self.weight_stream.begin_serving()

    def close(self):
        try:
            stream = getattr(self, "weight_stream", None)
            if stream is not None:
                stream.close()
                self.weight_stream = None
        finally:
            super().close()

    def execution_config(self, *, max_lanes, prefill_step):
        if self.draft_model is not None:
            return {
                **self._external_execution_config(
                    max_lanes=max_lanes, prefill_step=prefill_step
                ),
                "batch_size_route": self.ast_external_batch_size_route,
                "dynamic_tree_overflow": "cohort",
                "tree_node_budget_by_lanes": {
                    width: max(1, _tree_controls(self.external_policy)[0] // width - 1)
                    for width in range(1, 5)
                },
            }
        return super().execution_config(
            max_lanes=max_lanes, prefill_step=prefill_step
        )

    def bind_serving_lane_matmul(self, receipt):
        """Complete raw TensorFold's coverage with the owned MPP format lane.

        The released target has q8-g128 shared experts. Raw TensorFold leaves
        these on MLX's width-dependent stock law; the owned lane covers them
        without changing any of TensorFold's already tiled weight storage.
        Both ordinary and AST calls receive the same installed arithmetic.
        """
        from mlx import nn

        from ..runtime.lane import install, installed_lane
        from ..runtime.tensorfold_qwen38 import lane_qmm

        raw_tensorfold = (receipt or {}).get("law_id") == "tensorfold-lane-qmm-v1+fused-groups"
        if raw_tensorfold:
            gaps = install(
                self.model, min_rows=1, max_rows=32,
                skip=lambda _name, module: not isinstance(module, (nn.QuantizedLinear, nn.Linear))
                or (isinstance(module, nn.QuantizedLinear) and lane_qmm.takes(module)),
            )
            receipt["format_completion"] = gaps
            receipt["law_id"] += ":ling-format-completion:" + gaps["law_id"]
        packed = getattr(self, "draft_model", None) is not None and (
            self.external_policy.get("ast_target_executor", "packed") == "packed"
        )
        if packed:
            rows, _adaptive = _tree_controls(self.external_policy)
            missing = []
            for name, module in self.model.named_modules():
                if not isinstance(module, (nn.QuantizedLinear, nn.Linear)):
                    continue
                raw = raw_tensorfold and isinstance(module, nn.QuantizedLinear) and lane_qmm.takes(module)
                owned = installed_lane(module) is not None and (
                    getattr(module, "_lane_min_rows", None) == 1
                    and getattr(module, "_lane_max_rows", 0) >= rows
                )
                if not (raw or owned):
                    missing.append(name)
            if not receipt or missing:
                raise ValueError(
                    "Packed Ling AST requires a complete row-invariant projection law; "
                    "use --lane-matmul exact or tensorfold, or ast_target_executor=serial. "
                    f"Uncovered projections: {missing[:4]}"
                )
        self.packed_projection_law = None if receipt is None else {
            "law_id": receipt.get("law_id"),
            "format_completion": receipt.get("format_completion"),
        }

    def lane_policy_defaults(self):
        if self.draft_model is not None and self.external_policy.get("ast_target_executor", "packed") == "packed":
            return {"mode": "exact", "moe": {"mode": "exact"}}
        return {}

    def _graft_prompt_lookup(self):
        from ..runtime.drafters.prompt_lookup_draft import PromptLookupDraftModel

        if getattr(self.draft_model, "receipt_kind", None) != "external_ubtree":
            raise ValueError("Ling prompt lookup graft requires its UBTree drafter")
        self.draft_model = PromptLookupDraftModel(
            self.model, num_draft=self._external_num_draft(), base=self.draft_model
        )
        digest = hashlib.sha256(
            (self.identity["fingerprint"] + ":ling-ubtree-pld-graft-v1").encode()
        ).hexdigest()
        self.identity = {**self.identity, "fingerprint": digest}
        self.layout += ":pld-graft-v1:" + digest
        self.descriptor = replace(
            self.descriptor, variant="oq4-ubtree-ast-b15-pld",
            cache_layout=self.layout,
            metadata={**self.descriptor.metadata, "prompt_lookup_graft": True},
        )

    def create_external_batch(self, **kwargs):
        from ..runtime.drafters.prompt_lookup_draft import PromptLookupDraftModel

        draft = self.draft_model
        if isinstance(draft, PromptLookupDraftModel):
            draft = draft.base
        if getattr(draft, "receipt_kind", None) != "external_ubtree":
            raise ValueError("Ling AST requires a bound UBTree drafter")
        from ..runtime.hybrid_tree_executor import HybridTreeExecutor
        from ..runtime.ling_tree_executor import LingTreeExecutor

        # UBTree supplies learned trees, not categorical chain proposal laws.
        # Keep excess sessions on bounded provider cohorts at every pressure.
        if kwargs.get("dynamic_tree_overflow", "cohort") != "cohort":
            raise ValueError("Ling UBTree requires bounded tree cohort overflow")
        kwargs.setdefault("dynamic_tree_overflow", "cohort")
        kwargs.setdefault("dynamic_singleton_tree", True)
        kwargs.setdefault("dynamic_tree_max_width", 4)
        kwargs.setdefault("tensorfold_cohort_limit", 4)
        rows, adaptive = _tree_controls(self.external_policy)
        kwargs.setdefault("tree_node_budget_by_lanes", {
            width: max(1, rows // width - 1) for width in range(1, 5)
        })
        if adaptive is not None:
            kwargs.setdefault("ast_adaptive_rows", adaptive)
        provider = (
            HybridTreeExecutor if self.external_policy.get("ast_target_executor", "packed") == "serial"
            else LingTreeExecutor
        )
        if provider is HybridTreeExecutor and rows > 16:
            raise ValueError("Ling serial reference supports at most 16 tile rows")
        kwargs.setdefault("target_tree_executor_provider", provider.create())
        return super().create_external_batch(**kwargs)

    def _bind_tree_controls(self):
        import json

        rows, adaptive = _tree_controls(self.external_policy)
        value = {"tile_rows": rows, "adaptive_rows": adaptive,
                 "target_executor": self.external_policy.get("ast_target_executor", "packed")}
        digest = hashlib.sha256((self.identity["fingerprint"] + json.dumps(
            value, sort_keys=True, separators=(",", ":")
        )).encode()).hexdigest()
        self.identity = {**self.identity, "fingerprint": digest}
        self.layout += ":tree-controls-v1:" + digest
        base = getattr(self.draft_model, "base", None) or self.draft_model
        base.served_tree_budget = rows - 1
        self.descriptor = replace(
            self.descriptor,
            variant=f"oq4-ubtree-ast-b{rows - 1}" + (
                "-pld" if self.external_policy.get("ast_graft_prompt_lookup", False) else ""
            ),
            cache_layout=self.layout,
            metadata={**self.descriptor.metadata, "served_tree_budget": rows - 1,
                      "tree_controls": value},
        )

    def diagnostics(self):
        if self.draft_model is None:
            return {"route": "ordinary", "qualification": "pending"}
        return {
            "route": "external_ubtree_ast",
            "qualification": "unqualified",
            "implemented": True,
            "selected": True,
            "observed_used": False,
            "draft": self.draft_model.receipt_settings,
            "tree_controls": dict(zip(("tile_rows", "adaptive_rows"), _tree_controls(self.external_policy))),
            "projection_law": getattr(self, "packed_projection_law", None),
            **({"prompt_lookup_graft": dict(self.draft_model.stats)}
               if self.external_policy.get("ast_graft_prompt_lookup", False) else {}),
        }


__all__ = ["DESCRIPTOR", "Ling3FlashAdapter", "inspect_artifact"]

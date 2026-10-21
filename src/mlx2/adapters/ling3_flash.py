"""Ling 3 Flash hybrid target and its explicit UBTree AST route."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from ..process_env import (
    PROCESS_NUMERICS,
    clear_inherited_profile,
    require_process_numerics,
)
from ..sampling_defaults import MODEL_CARD, SamplingDefaults, VendorSampling
from .external_draft_policy import ExternalDraftAdapterMixin
from .ordinary_artifact import inspect_indexed_artifact
from .ordinary_text import OrdinaryTextAdapter

CACHE_LAYOUT = "ling3-flash-hybrid-kda-mla-v1"
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
    return profile


class Ling3FlashAdapter(ExternalDraftAdapterMixin, OrdinaryTextAdapter):
    descriptor = DESCRIPTOR
    sampling_defaults = SAMPLING
    artifact_inspector = staticmethod(inspect_artifact)
    profile = "ling3-flash-oq4-apcv2-ordinary"
    default_route = "ordinary"
    ast_tree_tiles = True
    # The generic header gate protects Qwen TensorFold kernels whose private
    # state is hard-coded to bf16. Ling verifies ordinary quantized modules
    # through the exact serial mixed-state executor instead.
    ast_tree_tile_dtype_gate = False
    EXTERNAL_DEFAULT_NUM_DRAFT = 15
    EXTERNAL_ROUTE_TAG = "external-ubtree-ling3-v1"
    EXTERNAL_PROFILE = "ling3-flash-oq4-apcv2-ubtree"
    # ``server.apply_ast_invocation`` adds this exact tree-tile selection for
    # an explicit ``--ast-drafter`` artifact.
    EXTERNAL_POLICY_EXTENSIONS = frozenset({"batch_size_route"})

    def __init__(self, model_path, *, execution_policy=None):
        from .process_globals import guarded_construction

        guarded_construction(
            self,
            lambda: self._init_ling(model_path, execution_policy=execution_policy),
        )

    def _init_ling(self, model_path, *, execution_policy=None):
        external = self._parse_external_policy(execution_policy, family="Ling 3 Flash")
        if external:
            # Current AST hardware tiles hold an anchor plus 15 nodes.  The
            # checkpoint's B64 search stays represented in its config/receipt;
            # serving uses an explicit prefix-closed B15 projection.
            self.external_policy.setdefault("num_draft", 15)
            route = self.external_policy.get("batch_size_route")
            if route not in (None, "tree15_b1_b4_chain_b5plus_v1"):
                raise ValueError(
                    "Ling UBTree supports only the bounded tree15 AST route"
                )
        artifact = inspect_artifact(model_path)
        self.config = artifact["config"]
        self.identity = artifact["identity"]
        self.layout = CACHE_LAYOUT
        self.environment = configure_environment()
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
        weights = self.model.sanitize(load_shards_evicting(files))
        quant = self.config["quantization"]

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
        self.model.load_weights(list(weights.items()), strict=True)
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
                    "target_tree_executor": "serial-exact-reference",
                },
            )

    def create_external_batch(self, **kwargs):
        if getattr(self.draft_model, "receipt_kind", None) != "external_ubtree":
            raise ValueError("Ling AST requires a bound UBTree drafter")
        from ..runtime.hybrid_tree_executor import HybridTreeExecutor

        kwargs.setdefault("dynamic_singleton_tree", True)
        kwargs.setdefault("dynamic_tree_max_width", 4)
        kwargs.setdefault("tree_node_budget_by_lanes", {1: 15, 2: 7, 3: 4, 4: 3})
        kwargs.setdefault("target_tree_executor_provider", HybridTreeExecutor.create())
        return super().create_external_batch(**kwargs)

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
        }


__all__ = ["DESCRIPTOR", "Ling3FlashAdapter", "inspect_artifact"]

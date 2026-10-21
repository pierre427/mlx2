"""Strict runtime configuration for released Ling 3 UBTree drafters."""

from __future__ import annotations

from dataclasses import dataclass

from .base_config import DFlashConfig


@dataclass
class UBTreeConfig(DFlashConfig):
    maximum_proposal_length = 15
    model_type: str = "ubtree"
    hidden_act: str = "silu"
    input_embedding_scale: float = 1.0
    output_multiplier: float = 1.0
    selector_rank: int = 256
    factorizer_hidden_width: int = 512
    selector_top_k: int = 64
    tree_supertree_width: int = 12
    tree_budget: int = 64
    depth_condition_count: int = 15

    @classmethod
    def from_dict(cls, config):
        ub = config.get("ubtree_config")
        nested = config.get("dflash_config")
        if not isinstance(ub, dict) or not isinstance(nested, dict):
            raise TypeError("UBTree requires ubtree_config and dflash_config")
        expected = {
            "type": "residual_bigram",
            "version": 1,
            "architecture": "cp",
            "candidate_source": "frozen_dflash_unary_topk_ids_only",
            "depth_conditioning": "unary_correction_two_scale",
            "depth_scale_parameterization": "exp_log_scale",
            "token_factorizer": "target_embedding_mlp",
            "hidden_projection_type": "shared",
            "inference_hidden_projection": "materialized_per_depth_affine",
            "tree_builder": "pruned",
            "adds_unary_logits": True,
            "trained": True,
        }
        if any(ub.get(key) != value for key, value in expected.items()):
            raise ValueError("unsupported UBTree selector contract")
        merged = dict(config)
        merged["dflash_config"] = nested
        args = DFlashConfig.from_dict(merged)
        values = {
            name: getattr(args, name) for name in DFlashConfig.__dataclass_fields__
        }
        values.update(
            model_type="ubtree",
            hidden_act=config.get("hidden_act", "silu"),
            input_embedding_scale=float(config.get("input_embedding_scale", 1.0)),
            output_multiplier=float(config.get("output_multiplier", 1.0)),
            block_size=int(nested.get("block_size")),
            mask_token_id=int(nested.get("mask_token_id")),
            target_layer_ids=list(nested.get("target_layer_ids", ())),
            num_target_layers=int(nested.get("num_target_layers")),
            rope_theta=float(
                config.get("rope_parameters", {}).get(
                    "rope_theta", config.get("rope_theta", 10000.0)
                )
            ),
        )
        values.update(
            selector_rank=int(ub.get("rank")),
            factorizer_hidden_width=int(ub.get("factorizer_hidden_width")),
            selector_top_k=int(ub.get("candidate_k")),
            tree_supertree_width=int(ub.get("tree_supertree_width")),
            tree_budget=int(ub.get("tree_budget")),
            depth_condition_count=int(ub.get("depth_condition_count")),
        )
        result = cls(**values)
        if (
            result.hidden_act != "silu"
            or result.num_hidden_layers != 5
            or result.num_target_layers != 42
            or result.target_layer_ids != [1, 11, 23, 29, 35]
            or result.block_size != 16
            or result.depth_condition_count != result.block_size - 1
            or result.selector_rank != 256
            or result.factorizer_hidden_width != 512
            or result.selector_top_k != 64
            or result.tree_supertree_width != 12
            or result.tree_budget != 64
        ):
            raise ValueError("unsupported released UBTree geometry")
        return result


__all__ = ["UBTreeConfig"]

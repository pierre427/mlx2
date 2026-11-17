# SPDX-License-Identifier: MIT
"""Kimi Linear DFlash block drafter (Moonlight556 Kimi-Linear-48B-A3B-DFlash-240k).

The checkpoint is a plain z-lab-style ``DFlashDraftModel``: five bidirectional
Qwen3 attention layers (q/k RMSNorm, default full-head RoPE at theta 1e4,
head_dim 72) over ``hidden_norm(fc(concat(target taps)))`` context and an
anchor-plus-mask block, with the target's embedding and LM head.  That is
exactly ``dflash_base.DFlashDraftModel``'s math.  The math and external route
contract (context append, row batching, per-position laws, best-first trees)
live in ``plain_dflash.PlainDFlashDraftModel``; this module keeps the Kimi
strict config parser and the Kimi receipt names, which are unchanged.
Provenance: provenance/kimi-linear-speculative.json (source dflash.py read as
data; no remote code is imported).

Because the block inputs after the anchor are mask tokens and attention is
bidirectional, position ``j`` does not depend on the drafts at positions
``< j``: ``softmax(logits_j / T)`` is the exact law of the drafted token, and a
tree may combine per-depth candidates freely.  No capability is qualified by
importing or constructing this module.
"""
from __future__ import annotations

from collections.abc import Mapping

from .base_config import DFlashConfig
from .plain_dflash import PlainDFlashDraftModel, best_first_tree, expected_weight_shapes

ARCHITECTURE = "DFlashDraftModel"


def parse_config(config: Mapping) -> DFlashConfig:
    """Strict parse; refuse every geometry this drafter would silently ignore."""
    if config.get("architectures") != [ARCHITECTURE] or config.get("model_type") != "qwen3":
        raise ValueError("Kimi DFlash expects a qwen3 DFlashDraftModel artifact")
    dflash = config.get("dflash_config")
    if not isinstance(dflash, Mapping):
        raise ValueError("Kimi DFlash requires a dflash_config object")
    if bool(dflash.get("causal_head", False)):
        raise ValueError("Kimi DFlash serves the bidirectional (causal_head=false) block only")
    if config.get("attention_bias", False) or config.get("tie_word_embeddings", False):
        raise ValueError("Kimi DFlash requires bias-free attention and an untied target head")
    if config.get("hidden_act", "silu") != "silu":
        # The draft MLP is the SwiGLU of dflash_base, whatever the config says.
        raise ValueError("Kimi DFlash supports hidden_act='silu' only")
    layers = config.get("layer_types") or []
    if (len(layers) != config.get("num_hidden_layers")
            or any(kind != "full_attention" for kind in layers)
            or config.get("sliding_window") is not None or config.get("use_sliding_window")):
        raise ValueError("Kimi DFlash requires full-attention draft layers")
    rope = config.get("rope_parameters") or config.get("rope_scaling") or {}
    if not isinstance(rope, Mapping) or rope.get("rope_type", "default") != "default":
        raise ValueError("Kimi DFlash supports default full-head RoPE only")
    theta = rope.get("rope_theta", config.get("rope_theta"))
    if not isinstance(theta, (int, float)) or theta <= 0:
        raise ValueError("Kimi DFlash requires an explicit rope_theta")
    flat = dict(config)
    flat["rope_theta"] = float(theta)
    flat["rope_scaling"] = None
    args = DFlashConfig.from_dict(flat)
    ids = list(args.target_layer_ids)
    if (not ids or ids != sorted(set(ids)) or ids[0] < 0
            or ids[-1] >= args.num_target_layers):
        raise ValueError("Kimi DFlash target_layer_ids must be increasing target layers")
    if not 0 <= int(args.mask_token_id) < args.vocab_size:
        raise ValueError("Kimi DFlash mask_token_id must be inside the vocabulary")
    if args.block_size < 2:
        raise ValueError("Kimi DFlash block_size must exceed one")
    return args


class KimiDFlashDraftModel(PlainDFlashDraftModel):
    receipt_kind = "external_kimi_dflash"
    receipt_name = "kimi_dflash"
    label = "Kimi DFlash"


Model = KimiDFlashDraftModel

__all__ = ["ARCHITECTURE", "KimiDFlashDraftModel", "Model", "best_first_tree",
           "expected_weight_shapes", "parse_config"]

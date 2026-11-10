"""Source DS4.1 native routing mask; expert-store ownership stays unchanged."""

from __future__ import annotations

import hashlib
import json


def source_expert_mask_binding(adapter):
    from ..experimental.weight_space.experts import LayerExpertMask
    from ..runtime.expert_masks import ExpertMaskBinding
    from .deepseek_v41_source_weights import SourceExpertStore

    model, policy = adapter.model, adapter.kernel_policy
    store = getattr(adapter.weights, "expert_stream", None)
    if not isinstance(store, SourceExpertStore):
        raise TypeError("Source expert masks require the native SourceExpertStore")
    refused = {
        "deepseek_physical_batch": "b1", "deepseek_vision": "reference",
        "deepseek_source_dspark_ast": "off", "deepseek_source_dspark_prepare": "off",
        "deepseek_source_dspark_target_forward": "reference",
        "deepseek_source_ast_kv_preparation": "reference",
        "deepseek_source_resident_gather": "reference", "deepseek_source_resident_prefill": "reference",
        "deepseek_source_route_bias": "off", "deepseek_source_prefill_approx": "off",
    }
    if (any(policy.get(key, ordinary) != ordinary for key, ordinary in refused.items())
            or getattr(adapter, "draft_model", None) is not None
            or getattr(model, "source_route_bias", None) is not None
            or getattr(model, "source_approx_private_profile", None) is not None
            or getattr(adapter.weights, "source_resident_bank_session", None) is not None
            or policy.get("deepseek_source_experts", {}).get("residency") == "dynamic_precision_v1"
            or {"q1_on_miss", "tiers"} & set(policy.get("deepseek_source_experts") or {})):
        raise ValueError("Source expert masks require the ordinary fixed-precision B1 route")
    source_fingerprint = adapter.identity["fingerprint"]
    if len(source_fingerprint) != 64:
        raise ValueError("Source expert masks require the exact source weight SHA256")
    profile = "ds41-source-native-mask-v1:" + hashlib.sha256(json.dumps({
        "policy": policy, "model_identity": getattr(model, "identity", None),
        "numerics_revision": getattr(adapter, "numerics_revision", None),
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    layers = tuple(LayerExpertMask(block.layer, block.c.n_routed_experts,
        block.c.num_experts_per_tok, tuple(range(block.c.n_routed_experts)),
        block.c.norm_topk_prob, block.c.routed_scaling_factor) for block in model.layers)

    def install(manager):
        originals = []
        for block in model.layers:
            original = block.router
            originals.append((block, original))
            block.router = _router(block, original, manager)

        def restore():
            for block, original in originals:
                block.router = original

        return restore

    return ExpertMaskBinding(source_fingerprint, adapter.numeric_profile, profile, layers, install)


def _router(block, original, manager):
    def router(x, image_mask=None):
        import mlx.core as mx

        allowed = manager.selection(block.layer)
        if allowed is None:
            return original(x) if image_mask is None else original(x, image_mask=image_mask)
        if image_mask is not None or x.ndim != 3 or x.shape[0] != 1:
            raise ValueError("Source expert masks require text-only physical width one")
        from ..runtime.models.deepseek_v4 import route_experts

        model, config = block.model, block.c
        route = route_experts
        if getattr(model, "pointwise_policy", "reference") == "fused":
            from ..runtime.models.deepseek_v41_fused import router as fused_router
            from ..runtime.models.deepseek_v41_fused import self_check

            self_check()
            route = fused_router
        # The correction affects ranking only. Native scores, normalization,
        # original IDs, top-k order and expert combination remain native.
        correction = mx.where(allowed, block.correction, -float("inf"))
        return route(model.linear(x, block.prefix + "ffn_gate_inp.weight", rounded=False),
            top_k=config.num_experts_per_tok, scale=config.routed_scaling_factor,
            correction=correction, normalize=config.norm_topk_prob)

    return router

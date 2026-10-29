"""Pure DeepSeek execution identity derivation; no tensors or server trust."""

from __future__ import annotations

import hashlib
import json

NATIVE_NUMERICS_REVISION = "native-packed-fp32-qat-stable-index-v2"


def model_execution_identity(
    identity,
    *,
    attention_policy="reference",
    cache_policy="fp32",
    hc_policy="reference",
    hc_projection_policy="reference",
    expert_prefill_policy="reference",
    dense_prefill_policy="reference",
    rope_qat_policy="reference",
    index_qat_policy="reference",
):
    for name, value, default in (
        ("attention", attention_policy, "reference"),
        ("cache", cache_policy, "fp32"),
        ("hc", hc_policy, "reference"),
        ("hc_projection", hc_projection_policy, "reference"),
        ("expert_prefill", expert_prefill_policy, "reference"),
        ("dense_prefill", dense_prefill_policy, "reference"),
        ("rope_qat", rope_qat_policy, "reference"),
        ("index_qat", index_qat_policy, "reference"),
    ):
        if value != default:
            identity += ":" + name + "=" + value
    return identity


def native_target_identity(fingerprint, policy):
    return model_execution_identity(
        fingerprint + ":" + NATIVE_NUMERICS_REVISION + ":compute=" + policy["deepseek_packed"],
        attention_policy=policy["deepseek_attention"],
        cache_policy=policy["deepseek_kv"],
        hc_policy=policy["deepseek_hc"],
        hc_projection_policy=policy["deepseek_hc_projection"],
        expert_prefill_policy=policy["deepseek_expert_prefill"],
        dense_prefill_policy=policy["deepseek_dense_prefill"],
        rope_qat_policy=policy["deepseek_rope_qat"],
        index_qat_policy=policy.get("deepseek_index_qat", "reference"),
    )


def dspark_target_identity(
    identity, compute_policy, embedding_policy, verifier, confidence
):
    return json.dumps(
        [identity, compute_policy, embedding_policy, verifier]
        + ([{"draft_confidence": confidence}] if confidence else []),
        separators=(",", ":"),
    )


def dspark_execution_binding(backbone_identity, target_identity, verifier):
    return hashlib.sha256(
        json.dumps(
            [
                "deepseek-v4-dspark-chain-greedy-v2",
                verifier,
                backbone_identity,
                target_identity,
            ],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def dspark_artifact_fingerprint(execution_binding, num_draft):
    return hashlib.sha256(
        json.dumps([execution_binding, num_draft], separators=(",", ":")).encode()
    ).hexdigest()

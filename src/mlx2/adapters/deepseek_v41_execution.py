# SPDX-License-Identifier: MIT
"""Shared V4.1 policy admission, model wiring and truthful route receipts."""
from dataclasses import replace

# Contexts above this bound need native packed KV plus candidate-only scoring.
# ``context_gated_v1`` selects exactly that pair above the bound and the
# reference pair at or below it; it is an explicit opt-in, never a default.
CONTEXT_GATE_TOKENS = 65536
PACKED_ATTENTION_MATH = ("reference", "all_retained_broadcast", "omlx_nax_bf16_online_v1")

POLICY_CHOICES = {
    "deepseek_kv_storage": ("reference", "native_packed", "context_gated_v1"),
    "deepseek_candidate_score": ("reference", "candidate_only"),
    "deepseek_ced": ("reference", "dependency_exact"),
    # Opt-in, unqualified and not GPU-tested (2026-11-02): a dependency CED
    # prefill chunk with at least one first-decoder-layer cone of rows after
    # it in the same prompt call skips every layer above the barrier
    # (Sushi's state-only chunks). See deepseek_v41_ced.state_only_admitted.
    "deepseek_ced_state_only": ("reference", "after_cone_v1"),
}


def resolve_context_gate(policy):
    """Resolve the opt-in gate once, in place, to concrete storage/scoring."""
    if policy.get("deepseek_kv_storage") != "context_gated_v1":
        return policy
    limit = policy.get("deepseek_context_limit", CONTEXT_GATE_TOKENS)
    if type(limit) is not int:
        raise ValueError("V4.1 context gate requires an integer context limit")
    if limit > CONTEXT_GATE_TOKENS:
        policy["deepseek_kv_storage"] = "native_packed"
        policy["deepseek_candidate_score"] = "candidate_only"
    else:
        policy["deepseek_kv_storage"] = "reference"
    return policy


def validate_csa_policy(policy):
    resolve_context_gate(policy)
    if "deepseek_context_limit" in policy:
        limit = policy["deepseek_context_limit"]
        if type(limit) is not int or not 1 <= limit <= 1048576:
            raise ValueError("V4.1 context limit must be an integer in1..1048576")
        if limit > 65536 and (policy.get("deepseek_kv_storage") != "native_packed"
                              or policy.get("deepseek_candidate_score") != "candidate_only"):
            raise ValueError("V4.1 contexts above65536 require native packed KV and candidate-only scoring")
    if policy.get("deepseek_kv_storage", "reference") == "native_packed":
        # Reference and fused QAT/attention-epilogue producers both feed the
        # packed producer the same pre-QAT values; fused QAT selects the exact
        # fused Metal codec. Source short-decode laws have no packed consumer.
        if policy.get("deepseek_source_attention_math", "reference") not in PACKED_ATTENTION_MATH:
            raise ValueError("native packed KV requires reference, broadcast or oMLX selected attention math")
        if policy.get("deepseek_verify_rows", "serial") != "serial" or policy.get("deepseek_verify_commit", "replay") != "replay":
            raise ValueError("native packed KV supports serial snapshot/replay verification")
    if policy.get("deepseek_ced", "reference") != "reference" and (
        policy.get("ast_prompt_lookup") or policy.get("draft_model") == "embedded_dspark"
        or policy.get("deepseek_vision", "reference") != "reference"
    ):
        raise ValueError("dependency CED is a separate ordinary text prefill profile")
    if (policy.get("deepseek_ced_state_only", "reference") != "reference"
            and policy.get("deepseek_ced", "reference") != "dependency_exact"):
        raise ValueError("CED state-only chunks require deepseek_ced=dependency_exact")


def model_options(policy):
    return {"kv_storage_policy": policy.get("deepseek_kv_storage", "reference"),
            "candidate_score_policy": policy.get("deepseek_candidate_score", "reference"),
            "ced_policy": policy.get("deepseek_ced", "reference"),
            "ced_state_only": policy.get("deepseek_ced_state_only", "reference") == "after_cone_v1",
            "context_limit": policy.get("deepseek_context_limit")}


def bind_cache_layout(adapter):
    if adapter.model.kv_storage_policy == "native_packed":
        from ..runtime.models.deepseek_v41_packed_cache import CACHE_LAYOUT
        adapter.layout = CACHE_LAYOUT
        adapter.descriptor = replace(adapter.descriptor, cache_layout=CACHE_LAYOUT)


def execution_receipt(model):
    packed = getattr(model, "kv_storage_policy", "reference") == "native_packed"
    ced = getattr(model, "ced_policy", "reference")
    receipt = {"kv_storage": "native packed QAT codes/scales; bounded decoded compute scratch" if packed else "physical FP32; BF16/FP8/FP4-rounded values",
            "kv_storage_policy": getattr(model, "kv_storage_policy", "reference"),
            "candidate_score_policy": getattr(model, "candidate_score_policy", "reference"),
            "ced": ced, "full_depth_prefill": ced == "reference",
            "ced_state_only": "after_cone_v1" if getattr(model, "ced_state_only", False) else "reference",
            "supported_context": getattr(model, "context_limit", 65536),
            "long_context_evidence": "explicit unqualified profile" if getattr(model, "context_limit", 65536) > 65536 else "see exact checkpoint receipts",
            "approximate_replay": False,
            "scheduled_kv_calls": dict(getattr(model, "scheduled_kv_calls", {}))}
    if packed:
        receipt["packed_codec"] = getattr(model, "packed_codec_policy", "reference")
    return receipt

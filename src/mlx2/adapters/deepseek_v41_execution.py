# SPDX-License-Identifier: MIT
"""Shared V4.1 policy admission, model wiring and truthful route receipts."""
from dataclasses import replace

POLICY_CHOICES = {
    "deepseek_kv_storage": ("reference", "native_packed"),
    "deepseek_candidate_score": ("reference", "candidate_only"),
    "deepseek_ced": ("reference", "dependency_exact"),
}


def validate_csa_policy(policy):
    if "deepseek_context_limit" in policy:
        limit = policy["deepseek_context_limit"]
        if type(limit) is not int or not 1 <= limit <= 1048576:
            raise ValueError("V4.1 context limit must be an integer in1..1048576")
        if limit > 65536 and (policy.get("deepseek_kv_storage") != "native_packed"
                              or policy.get("deepseek_candidate_score") != "candidate_only"):
            raise ValueError("V4.1 contexts above65536 require native packed KV and candidate-only scoring")
    if policy.get("deepseek_kv_storage", "reference") == "native_packed":
        if policy.get("deepseek_source_attention_math", "reference") != "reference":
            raise ValueError("native packed KV requires reference selected attention math")
        for name in ("deepseek_qat", "deepseek_attention_epilogue"):
            if policy.get(name, "reference") != "reference":
                raise ValueError("native packed KV requires reference QAT/attention producers")
        if policy.get("deepseek_verify_rows", "serial") != "serial" or policy.get("deepseek_verify_commit", "replay") != "replay":
            raise ValueError("native packed KV supports serial snapshot/replay verification")
    if policy.get("deepseek_ced", "reference") != "reference" and (
        policy.get("ast_prompt_lookup") or policy.get("draft_model") == "embedded_dspark"
        or policy.get("deepseek_vision", "reference") != "reference"
    ):
        raise ValueError("dependency CED is a separate ordinary text prefill profile")


def model_options(policy):
    return {"kv_storage_policy": policy.get("deepseek_kv_storage", "reference"),
            "candidate_score_policy": policy.get("deepseek_candidate_score", "reference"),
            "ced_policy": policy.get("deepseek_ced", "reference"),
            "context_limit": policy.get("deepseek_context_limit")}


def bind_cache_layout(adapter):
    if adapter.model.kv_storage_policy == "native_packed":
        from ..runtime.models.deepseek_v41_packed_cache import CACHE_LAYOUT
        adapter.layout = CACHE_LAYOUT
        adapter.descriptor = replace(adapter.descriptor, cache_layout=CACHE_LAYOUT)


def execution_receipt(model):
    packed = getattr(model, "kv_storage_policy", "reference") == "native_packed"
    ced = getattr(model, "ced_policy", "reference")
    return {"kv_storage": "native packed QAT codes/scales; bounded decoded compute scratch" if packed else "physical FP32; BF16/FP8/FP4-rounded values",
            "kv_storage_policy": getattr(model, "kv_storage_policy", "reference"),
            "candidate_score_policy": getattr(model, "candidate_score_policy", "reference"),
            "ced": ced, "full_depth_prefill": ced == "reference",
            "supported_context": getattr(model, "context_limit", 65536),
            "long_context_evidence": "explicit unqualified profile" if getattr(model, "context_limit", 65536) > 65536 else "see exact checkpoint receipts",
            "approximate_replay": False,
            "scheduled_kv_calls": dict(getattr(model, "scheduled_kv_calls", {}))}

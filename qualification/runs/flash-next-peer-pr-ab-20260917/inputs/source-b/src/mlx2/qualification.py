"""Bind selectable serving routes to a tested artifact, runtime, and settings."""

import json
from pathlib import Path

from .contracts import Capability, Fidelity, QualifiedProfile, RouteRequest
from .routing import RoutePlanner


REQUIRED_CHECKS = frozenset(
    {
        "cold_text",
        "warm_prefix",
        "stream",
        "tools",
        "reasoning",
        "stop",
        "batch",
        "context",
        "cancel",
        "recovery",
        "unit_tests",
        "hermes_client",
        "sampling_controls",
        "logprobs",
        "cache_leases",
        "runtime_stable",
        "mixed_warm",
    }
)


# Independent trust anchor for the reviewed receipt producer. Updating
# qualify_serving.py intentionally requires a source/runtime re-freeze and an
# explicit update here; a receipt may not authorize its own producer.
APPROVED_QUALIFICATION_HARNESS = {
    "schema": "mlx2.qualification-harness.v1",
    "name": "scripts/qualify_serving.py",
    "sha256": "e11b88073643434ec4cb1b0c9a33079c8c7b5b7e2c2167b956c3a4ebc5f6be3f",
}


def _environment_mode_enabled(value):
    """Return whether a mode-style environment setting selects a feature."""
    if value is None:
        return False
    return str(value).strip().casefold() not in {"", "0", "false", "off", "no"}


def required_feature_checks(settings):
    """Demand observed execution for selected mechanisms in the served domain."""
    if settings.get("speculation") == "external_draft":
        return {"feature_external_draft", "feature_proposal_distribution", "feature_paired_draft_cache", "feature_segmented_transaction"}
    if settings.get("speculation") == "prompt_lookup":
        return {
            "feature_prompt_lookup",
            "feature_prompt_lookup_proposals",
            "feature_prompt_lookup_rollback",
        }
    policy = settings.get("execution_policy", {})
    env = settings.get("environment", {})
    context = settings.get("max_context", 0)
    features = set()
    if env.get("MLX_QWEN4_PLE_NVME"):
        features.add("file_backed_ple")
    if env.get("MLX_QWEN4_PLE_COMPILE") == "1":
        features.add("compiled_ple")
    if env.get("MLX_QWEN4_QSA_POOLED_KEY_CACHE") == "1":
        features.add("pooled_qsa")
    if env.get("MLX_QWEN4_QSA_SCATTER_CHOSEN") == "1":
        features.add("scatter_qsa")
    if env.get("MLX_QWEN4_FUSED_GDN_DECODE") == "1":
        features.add("fused_gdn_decode")
    if env.get("MLX_QWEN4_EAGER_DISPATCH") == "1":
        features.add("eager_dispatch")
    if (env.get("MLX_QWEN4_MOE_FUSED_GATE_UP") == "1"
            and env.get("MLX_QWEN4_FUSED_EXPERT_KERNEL", "stock") != "stock"):
        features.add("fused_moe")
    if not settings.get("mtp"):
        return {"feature_" + name for name in features}
    if env.get("MLX_QWEN4_FUSED_GDN_VERIFY") == "1":
        features.add("fused_gdn_verify")
    if policy.get("segment_aware_async_qsa_promotion"):
        features.add("async_promotion")
    if policy.get("prefetch_known_tail_ple"):
        features.add("known_tail_prefetch")
    if _environment_mode_enabled(env.get("MLX_LM_SHARED_QSA_SUFFIX")) and context > int(env.get("MLX_LM_SHARED_QSA_SUFFIX_MIN_CONTEXT", "16380")) + 100:
        features.add("shared_qsa")
    if _environment_mode_enabled(env.get("MLX_QWEN4_QSA_INDEXED")) and context > int(env.get("MLX_QWEN4_QSA_INDEXED_MIN_CONTEXT", "16384")) + 100:
        features.add("indexed_qsa")
    if "shared_qsa" in features and context > int(env.get("MLX_LM_QSA_PRIVATE_DELTA_MIN_CONTEXT_MN", "65536")) + 100:
        features.add("private_delta")
    if env.get("MLX_QWEN4_QSA_INDEXED_FUSED_MERGE") == "1":
        features.add("indexed_fused_merge")
    if env.get("MLX_QWEN4_QSA_INDEXED_FUSED_GATE") == "1":
        features.add("indexed_output_gate")
    return {"feature_" + name for name in features}


def load_qualified_route(
    path,
    *,
    runtime,
    artifact,
    settings,
    descriptor,
    name,
):
    record = json.loads(Path(path).read_text())
    if record.get("qualification_harness") != APPROVED_QUALIFICATION_HARNESS:
        raise ValueError("qualification does not match approved harness")
    if record.get("runtime") != runtime or record.get("artifact") != artifact:
        raise ValueError("qualification does not match runtime and artifact")
    if record.get("settings") != settings:
        raise ValueError("qualification does not match serving settings")
    checks = record.get("checks", {})
    required = set(REQUIRED_CHECKS) | (
        {"mtp_execution"} if settings["mtp"] else set()
    )
    if Capability.GRAMMAR in descriptor.capabilities:
        required.add("structured_output")
    if record.get("passed") is not True or any(
        checks.get(c, {}).get("passed") is not True for c in required
    ):
        raise ValueError("qualification checks are missing or failed")
    if any(checks.get(name, {}).get("passed") is not True for name in required_feature_checks(settings)):
        raise ValueError("selected execution mechanisms lack observed qualification")
    capabilities = {
        Capability.TEXT,
        Capability.STREAMING,
        Capability.CONTINUOUS_BATCH,
        Capability.PREFIX_REUSE,
        Capability.APC_V2,
        Capability.LAYERED_CACHE,
        Capability.TOOLS,
        Capability.REASONING,
    }
    if settings.get("speculation") == "external_draft":
        capabilities.add(Capability.EXTERNAL_DRAFT)
    if settings.get("speculation") == "prompt_lookup":
        capabilities.add(Capability.PROMPT_LOOKUP)
    if settings["mtp"]:
        capabilities.update({Capability.MTP, Capability.SEGMENTED_MTP})
    if Capability.GRAMMAR in descriptor.capabilities:
        capabilities.add(Capability.GRAMMAR)
    planner = RoutePlanner()
    planner.register_model(descriptor)
    planner.register_profile(
        descriptor.key,
        QualifiedProfile(
            name=name,
            capabilities=frozenset(capabilities),
            fidelity=Fidelity.NUMERICALLY_BOUNDED,
            evidence=(str(Path(path).resolve()),),
            implementation=descriptor.metadata["execution"],
        ),
    )
    return planner.decide(
        RouteRequest(
            descriptor.model_type,
            descriptor.variant,
            frozenset(capabilities),
            minimum_fidelity=Fidelity.NUMERICALLY_BOUNDED,
        )
    )

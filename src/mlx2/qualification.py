"""Bind selectable serving routes to a tested artifact, runtime, and settings."""

import json
from pathlib import Path

from .contracts import Capability, Fidelity, QualifiedProfile, RouteRequest
from .media_qualification import SMOL_MEDIA_CHECKS, evaluate_smol_media_report
from .qwen25_media_qualification import CHECKS as QWEN_MEDIA_CHECKS, evaluate_qwen25_media_report
from .lfm25_media_qualification import LFM_MEDIA_CHECKS, evaluate_lfm_media_report
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


def required_generic_checks(descriptor):
    """Require only generic probes the selected adapter can actually serve.

    The approved harness uses the route's selected capabilities for the same
    decision. Adapter-owned media checks remain separate and fail closed until
    a reviewed media producer exists.
    """
    required = set(REQUIRED_CHECKS)
    if Capability.TOOLS not in descriptor.capabilities:
        required.discard("tools")
    if Capability.REASONING not in descriptor.capabilities:
        required.discard("reasoning")
    if Capability.CONTINUOUS_BATCH not in descriptor.capabilities:
        required.difference_update({"batch", "mixed_warm"})
    return required


# Independent trust anchor for the reviewed receipt producer. Updating
# qualify_serving.py intentionally requires a source/runtime re-freeze and an
# explicit update here; a receipt may not authorize its own producer.
APPROVED_QUALIFICATION_HARNESS = {
    "schema": "mlx2.qualification-harness.v1",
    "name": "scripts/qualify_serving.py",
    "sha256": "a15d63311009b7eaf421568d11d0ba6afd23e64964a602669ae12faaf6b8eb40",
}

# The approved generic producer has no live adapter-owned media probes. A
# caller may edit a receipt file, so a bare {"passed": true} under an invented
# check name is not evidence that the reviewed producer ran that check. Add
# names here only with the corresponding reviewed producer implementation.
APPROVED_ADAPTER_CHECKS = SMOL_MEDIA_CHECKS | QWEN_MEDIA_CHECKS | LFM_MEDIA_CHECKS
APPROVED_MEDIA_HARNESS = {
    "name": "scripts/qualify_media_serving.py",
    "sha256": "796cb204b9927e53f74bf1d0186b8c24d385947db5a36e56f2f336675a7fc2b5",
}
APPROVED_MEDIA_PRODUCERS = {
    "smolvlm": (APPROVED_MEDIA_HARNESS, evaluate_smol_media_report, SMOL_MEDIA_CHECKS),
    "qwen2_5_vl": (
        {"name": "scripts/qualify_qwen25_media_serving.py",
         "sha256": "e430c5521cb2bd0cbb51d1458b98fa850d0ca2d7be305bf8013c2ef5915292e8"},
        evaluate_qwen25_media_report, QWEN_MEDIA_CHECKS,
    ),
    "lfm2_vl": (
        {"name": "scripts/qualify_lfm25_media_serving.py",
         "sha256": "706f232dc13ef062e6f15f8262b683edc6bf4f58c26fe5556ee50bf591cd7776"},
        evaluate_lfm_media_report, LFM_MEDIA_CHECKS,
    ),
}
PINNED_MEDIA_SOURCE_REVISION = "8a5e704e0fe43cd8654c144c4ecbd4c8aececeb5"


def validate_adapter_qualification(report, *, runtime, artifact, settings, descriptor=None):
    """Bind a companion producer's traces to the same exact serving route."""
    if not isinstance(report, dict):
        raise ValueError("adapter qualification producer is unavailable")
    model_type = report.get("model_type")
    if (descriptor is not None and model_type != descriptor.model_type) or model_type not in APPROVED_MEDIA_PRODUCERS:
        raise ValueError("adapter qualification producer is unavailable")
    harness, evaluator, expected_checks = APPROVED_MEDIA_PRODUCERS[model_type]
    if (report.get("schema") != "mlx2.media-serving-qualification.v1"
            or report.get("qualification_harness") != harness
            or report.get("source_revision") != PINNED_MEDIA_SOURCE_REVISION
            or (descriptor is not None and descriptor.metadata.get("source_revision") != PINNED_MEDIA_SOURCE_REVISION)
            or report.get("runtime") != runtime
            or report.get("artifact") != artifact):
        raise ValueError("adapter qualification does not match approved producer or artifact")
    reported_settings = report.get("settings")
    if not isinstance(reported_settings, dict) or {
        key: value for key, value in reported_settings.items()
        if key not in PROVENANCE_ONLY_SETTINGS
    } != {
        key: value for key, value in settings.items()
        if key not in PROVENANCE_ONLY_SETTINGS
    }:
        raise ValueError("adapter qualification does not match serving settings")
    observed = evaluator(report)
    checks = report.get("checks")
    if (report.get("passed") is not True or not isinstance(checks, dict)
            or set(checks) != expected_checks
            or any(not isinstance(checks.get(name), dict)
                   or checks[name].get("passed") is not value
                   for name, value in observed.items())
            or not all(observed.values())):
        raise ValueError("adapter qualification traces are missing or failed")
    return observed


def _environment_mode_enabled(value):
    """Return whether a mode-style environment setting selects a feature."""
    if value is None:
        return False
    return str(value).strip().casefold() not in {"", "0", "false", "off", "no"}


def required_feature_checks(settings):
    """Demand observed execution for selected mechanisms in the served domain."""
    features = _route_feature_checks(settings)
    if settings.get("disk_cache") is True:
        features.add("feature_apc_sessions")
    if settings.get("apc_persistence") is True:
        features.add("feature_apc_persistence")
    if (settings.get("int8_prefill") or {}).get("enabled") is True:
        # Any route: a selected int8 prefill policy must show engaged calls.
        features.add("feature_int8_prefill")
    if settings.get("sp_qmm"):
        # Patching eligible projections does not prove that the measured
        # shape policy actually routed any model calls through the kernel.
        features.add("feature_sp_qmm")
    verify = settings.get("qsdpa_verify_kernel") or {}
    if (
        (settings.get("approximate_kv") or {}).get("enabled") is True
        and verify.get("enabled") is True
        and type(verify.get("min_context")) is int
        and type(settings.get("max_context")) is int
        and settings["max_context"] >= verify["min_context"]
    ):
        # A selected long-context quantized attention route must actually
        # reach the fused kernel during this qualification run.
        features.add("feature_qsdpa_verify_kernel")
    if (settings.get("host_memory_signals") or {}).get("enabled") is True:
        # Any route: the selected host signal must be observed in telemetry.
        features.add("feature_host_memory_signals")
    if (settings.get("verify_bitexact") or {}).get("enabled") is True:
        # Any route: the bit-exact matmul route counter must have advanced.
        features.add("feature_verify_bitexact")
    if (settings.get("moe_expert_streaming") or {}).get("enabled") is True:
        # A streamed model must show real page-ins: a "streaming" run that
        # never faulted an expert was fully resident and proves nothing.
        features.add("feature_moe_expert_streaming")
    if (settings.get("memory_preemption") or {}).get("enabled") is True:
        # Any route: a selected preemption policy must show a lane that was
        # preempted and replayed to completion.
        features.add("feature_memory_preemption")
    # Item 12 (any route): selected tool-grammar extensions must be observed.
    if settings.get("constrained_tool_grammar_auto") is True:
        features.add("feature_tool_grammar_auto")
    if settings.get("tool_grammar_streaming") is True:
        features.add("feature_tool_grammar_streaming")
    return features


def _route_feature_checks(settings):
    env = settings.get("environment", {})
    features = set()
    if settings.get("speculation") == "external_draft":
        features.update({"external_draft", "proposal_distribution", "paired_draft_cache", "segmented_transaction"})
        if (settings.get("fly_verification") or {}).get("enabled") is True:
            features.add("fly_verification")
        if (settings.get("execution_policy") or {}).get("pairwise_selection") == "batched":
            features.add("external_pairwise_selection")
    elif settings.get("speculation") == "prompt_lookup":
        features.update({
            "prompt_lookup",
            "prompt_lookup_proposals",
            "prompt_lookup_rollback",
        })
        # The default-off rotating replay transaction is a selected mechanism
        # only when the served prompt-lookup policy turns it on.
        if (settings.get("prompt_lookup") or {}).get("rotating_replay") is True:
            features.add("prompt_lookup_rotating_replay")
    # Every speculation route still executes the target model.  Its selected
    # kernels and state mechanisms need evidence alongside proposal/rollback
    # checks; returning early here would silently waive those requirements.
    policy = settings.get("execution_policy", {})
    context = settings.get("max_context", 0)
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
    if (settings.get("spomin_live_surgery") or {}).get("enabled") is True:
        # Approximate compaction is selectable only with an observed edit.
        features.add("spomin_surgery")
    if settings.get("approximate_kv", {}).get("enabled") is True:
        # A selected approximate operation must show at least one lane it was
        # actually applied to, and a measured fidelity report for the same
        # operation and artifact that passes runtime/kv_quant_fidelity.py.
        features.add("approximate_kv")
        features.add("approximate_kv_fidelity")
        if settings["approximate_kv"].get("compose_mtp") is True:
            # Target-only quantization under self-MTP: observed MTP lanes.
            features.add("approximate_kv_mtp")
    if (settings.get("apc_interior_checkpoints") or {}).get("count", 0) > 0:
        # Selection is meaningful only when a request both captures and
        # publishes an exact interior checkpoint for later reuse.
        features.add("apc_interior_checkpoints")
    if (settings.get("apc_rolling_checkpoints") or {}).get("interval_tokens", 0) > 0:
        # A selected rolling policy must show a published progress point that
        # a later request resumed from (hybrid) or a cancel publication (KV).
        features.add("apc_rolling_checkpoints")
    if settings.get("apc_junction_checkpoints") is True:
        # A junction must be captured and published at an observed branch
        # point, then serve a later request that diverges there.
        features.add("apc_junction_checkpoints")
    if settings.get("prefill_scheduling"):
        # Present only when the server-owned policy is selected; the harness
        # must observe an SRPT reorder or bypass-capped service to qualify it.
        features.add("prefill_scheduling")
    if not settings.get("mtp") or settings.get("speculation") in {"external_draft", "prompt_lookup"}:
        return {"feature_" + name for name in features}
    if settings.get("adaptive_mtp_depth", {}).get("enabled") is True:
        features.add("adaptive_mtp_depth")
    if settings.get("mtp_ordinary_handoff", {}).get("enabled") is True:
        features.add("mtp_ordinary_handoff")
    if settings.get("fly_verification", {}).get("enabled") is True:
        features.add("fly_verification")
    if (settings.get("self_mtp_copy_draft") or {}).get("enabled") is True:
        # A selected copy-draft route must show verified copied spans.
        features.add("self_mtp_copy_draft")
    if env.get("MLX_QWEN4_FUSED_GDN_VERIFY") == "1":
        features.add("fused_gdn_verify")
    if (env.get("MLX_QWEN4_FUSED_GDN_VERIFY") == "1"
            and env.get("MLX_QWEN4_FUSED_GDN_REPLAY_ROLLBACK") == "1"):
        # Compact replay rollback is a distinct state path from verify; a
        # selected profile must show it actually rolled back at least once.
        features.add("fused_gdn_replay_rollback")
    if (env.get("MLX_QWEN4_FUSED_GDN_VERIFY") == "1"
            and env.get("MLX_QWEN4_FUSED_GDN_REPLAY_ROLLBACK") == "1"
            and env.get("MLX_QWEN4_FUSED_GDN_DYNAMIC_ACCEPT") == "1"):
        # Device-count reconstruction is a distinct kernel; a selected profile
        # must show at least one dynamic rollback.
        features.add("fused_gdn_dynamic_accept")
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


def required_descriptor_checks(descriptor):
    """Adapter-declared checks that a generic text receipt cannot satisfy."""
    checks = descriptor.metadata.get("required_qualification_checks", ())
    if not isinstance(checks, (list, tuple)) or any(
        not isinstance(check, str) or not check for check in checks
    ):
        raise ValueError("descriptor qualification checks must be nonempty strings")
    required = set(checks)
    for capability, check in (
        (Capability.VISION, "multimodal_image"),
        (Capability.VIDEO, "multimodal_video"),
        (Capability.AUDIO, "multimodal_audio_input"),
        (Capability.OUTPUT_AUDIO, "output_audio"),
    ):
        if capability in descriptor.capabilities:
            required.add(check)
    return required


# Settings that record *how* a value was chosen rather than the value itself.
# They are excluded from the receipt match; the values they explain are not.
#   route_selection_source: an explicit --ordinary qualification authorizes
#     the same resolved ordinary route when it comes from an adapter default.
#   cache_bytes_source / cache_bytes_clamped_from / cache_bytes_headroom: an
#     explicit --cache-bytes receipt authorizes a host default (or a
#     post-load clamp) that resolves to the same ``cache_bytes``, and a
#     receipt recorded before these fields existed still matches.
PROVENANCE_ONLY_SETTINGS = frozenset(
    {
        "route_selection_source",
        "cache_bytes_source",
        "cache_bytes_clamped_from",
        "cache_bytes_headroom",
    }
)


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
    # How a setting was chosen is observational provenance, not route
    # identity (see PROVENANCE_ONLY_SETTINGS); the chosen values stay bound.
    qualified_settings = record.get("settings")
    if not isinstance(qualified_settings, dict) or not isinstance(settings, dict):
        raise ValueError("qualification does not match serving settings")
    qualified_settings = {
        key: value
        for key, value in qualified_settings.items()
        if key not in PROVENANCE_ONLY_SETTINGS
    }
    serving_settings = {
        key: value
        for key, value in settings.items()
        if key not in PROVENANCE_ONLY_SETTINGS
    }
    if qualified_settings != serving_settings:
        raise ValueError("qualification does not match serving settings")
    checks = record.get("checks", {})
    required = required_generic_checks(descriptor) | (
        {"mtp_execution"} if settings["mtp"] else set()
    )
    if Capability.GRAMMAR in descriptor.capabilities:
        required.add("structured_output")
    descriptor_checks = required_descriptor_checks(descriptor)
    unsupported = descriptor_checks - APPROVED_ADAPTER_CHECKS
    if unsupported:
        raise ValueError(
            "approved qualification harness cannot produce adapter checks: "
            + ", ".join(sorted(unsupported))
        )
    if descriptor_checks:
        validate_adapter_qualification(
            record.get("adapter_qualification"), runtime=runtime,
            artifact=artifact, settings=settings, descriptor=descriptor,
        )
    if record.get("passed") is not True or any(
        checks.get(c, {}).get("passed") is not True for c in required
    ):
        raise ValueError("qualification checks are missing or failed")
    # Every recorded gate, not only the required and feature ones: the loader
    # trusted the top-level flag for capability scope, quiescence, APCv2
    # reuse and stores, context bound and the MTP aggregates, so a receipt
    # written with any of them failed (python -O skips the producer's
    # asserts) still selected the route.
    failed = sorted(
        name for name, value in checks.items()
        if not (isinstance(value, dict) and value.get("passed") is True)
    )
    if failed:
        raise ValueError("qualification records failed checks: " + ", ".join(failed))
    missing_features = sorted(
        name
        for name in required_feature_checks(settings)
        if checks.get(name, {}).get("passed") is not True
    )
    if missing_features:
        raise ValueError(
            "selected execution mechanisms lack observed qualification: "
            + ", ".join(missing_features)
        )
    capabilities = {
        Capability.TEXT,
        Capability.STREAMING,
        Capability.CONTINUOUS_BATCH,
        Capability.PREFIX_REUSE,
        Capability.APC_V2,
        Capability.LAYERED_CACHE,
        Capability.TOOLS,
        Capability.REASONING,
    } & set(descriptor.capabilities)
    capabilities.update(
        set(descriptor.capabilities)
        & {
            Capability.VISION,
            Capability.VIDEO,
            Capability.AUDIO,
            Capability.OUTPUT_AUDIO,
        }
    )
    if settings.get("speculation") == "external_draft":
        capabilities.add(Capability.EXTERNAL_DRAFT)
    if settings.get("speculation") == "prompt_lookup":
        capabilities.add(Capability.PROMPT_LOOKUP)
    if settings["mtp"]:
        capabilities.update({Capability.MTP, Capability.SEGMENTED_MTP})
    if Capability.GRAMMAR in descriptor.capabilities:
        capabilities.add(Capability.GRAMMAR)
    # A route that serves quantized KV is approximate by construction; never
    # let its receipt claim the numerically bounded tier.
    # Int8 (W8A8) prefill changes prefill numerics; same tier as quantized KV.
    fidelity = (
        Fidelity.APPROXIMATE
        if settings.get("approximate_kv", {}).get("enabled") is True
        or (settings.get("int8_prefill") or {}).get("enabled") is True
        else Fidelity.NUMERICALLY_BOUNDED
    )
    planner = RoutePlanner()
    planner.register_model(descriptor)
    planner.register_profile(
        descriptor.key,
        QualifiedProfile(
            name=name,
            capabilities=frozenset(capabilities),
            fidelity=fidelity,
            evidence=(str(Path(path).resolve()),),
            implementation=descriptor.metadata["execution"],
        ),
    )
    return planner.decide(
        RouteRequest(
            descriptor.model_type,
            descriptor.variant,
            frozenset(capabilities),
            minimum_fidelity=fidelity,
        )
    )

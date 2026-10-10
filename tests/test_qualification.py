import json
from pathlib import Path
import pytest

from mlx2.adapters.qwen import QWEN4_FLASH_NEXT
from mlx2.qualification import (
    APPROVED_QUALIFICATION_HARNESS,
    REQUIRED_CHECKS,
    load_qualified_route,
    required_generic_checks,
    required_descriptor_checks,
)


def test_text_batching_does_not_require_multimodal_batch_evidence():
    from mlx2.contracts import Capability
    from mlx2.adapters.gemma4 import GEMMA4_A4B
    from mlx2.adapters.lfm25_vl import LFM25_VL
    from mlx2.adapters.mlx_vlm import GEMMA3N, MINICPMO
    from mlx2.adapters.qwen25_vl import DESCRIPTOR as QWEN25_VL
    from mlx2.adapters.smolvlm2 import DESCRIPTOR as SMOLVLM

    # Text continuous batching is covered by the ordinary batch checks. Only
    # media adapters that declare this probe require multimodal batch evidence.
    assert {"batch", "mixed_warm"} <= required_generic_checks(QWEN4_FLASH_NEXT)
    assert "multimodal_continuous_batch" not in required_descriptor_checks(
        QWEN4_FLASH_NEXT
    )
    descriptors = (GEMMA3N, GEMMA4_A4B, MINICPMO, SMOLVLM, LFM25_VL, QWEN25_VL)
    for descriptor in descriptors:
        requires_media_batch = (
            Capability.CONTINUOUS_BATCH in descriptor.capabilities
        )
        assert (
            "multimodal_continuous_batch" in required_descriptor_checks(descriptor)
        ) is requires_media_batch


def test_generic_checks_follow_adapter_capabilities():
    from mlx2.adapters.lfm25_vl import LFM25_VL

    assert {"tools", "reasoning", "batch", "mixed_warm"} <= required_generic_checks(
        QWEN4_FLASH_NEXT
    )
    assert not {"tools", "reasoning", "batch", "mixed_warm"} & required_generic_checks(
        LFM25_VL
    )
    assert {"cold_text", "warm_prefix", "context", "recovery"} <= required_generic_checks(
        LFM25_VL
    )


def test_qualification_binds_artifact_runtime_settings_and_checks(tmp_path):
    path = tmp_path / "qualification.json"
    record = {
        "passed": True,
        "runtime": {"source": "abc"},
        "artifact": "weights",
        "settings": {
            "mtp": True,
            "max_context": 16384,
            "default_max_tokens": 65_536,
        },
        "qualification_harness": APPROVED_QUALIFICATION_HARNESS,
        "checks": {c: {"passed": True} for c in REQUIRED_CHECKS | {"mtp_execution", "structured_output"}},
    }
    args = dict(
        runtime=record["runtime"],
        artifact="weights",
        settings=record["settings"],
        descriptor=QWEN4_FLASH_NEXT,
        name="test",
    )
    path.write_text(json.dumps(record))
    assert "profile=test" in load_qualified_route(path, **args).receipt
    with pytest.raises(ValueError, match="serving settings"):
        load_qualified_route(
            path,
            **{
                **args,
                "settings": {**record["settings"], "default_max_tokens": 512},
            },
        )
    for key, bad in [
        ("artifact", "other"),
        ("runtime", {}),
        ("settings", {"mtp": False}),
        ("checks", {}),
    ]:
        path.write_text(json.dumps({**record, key: bad}))
        with pytest.raises(ValueError):
            load_qualified_route(path, **args)


@pytest.mark.parametrize("qualified_fp32", [False, True])
def test_fp32_head_receipt_cannot_select_opposite_head_mode(tmp_path, qualified_fp32):
    """The head precision is execution identity in both directions."""
    execution_policy = {"persistent": True}
    if qualified_fp32:
        execution_policy["fp32_head_logits"] = True
    settings = {
        "mtp": False,
        "max_context": 32768,
        "execution_policy": execution_policy,
    }
    record = {
        "passed": True,
        "runtime": {"source": "abc"},
        "artifact": "weights",
        "settings": settings,
        "qualification_harness": APPROVED_QUALIFICATION_HARNESS,
        "checks": {
            name: {"passed": True}
            for name in REQUIRED_CHECKS | {"structured_output"}
            # A selected fp32 head must also show its installed receipt.
            | ({"feature_fp32_head_logits"} if qualified_fp32 else set())
        },
    }
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(record))
    args = dict(
        runtime=record["runtime"], artifact=record["artifact"],
        descriptor=QWEN4_FLASH_NEXT, name="fp32-head-identity",
    )
    assert load_qualified_route(path, settings=settings, **args).profile
    opposite = dict(execution_policy)
    if qualified_fp32:
        del opposite["fp32_head_logits"]
    else:
        opposite["fp32_head_logits"] = True
    with pytest.raises(ValueError, match="serving settings"):
        load_qualified_route(
            path, settings={**settings, "execution_policy": opposite}, **args
        )


def test_multimodal_descriptor_requires_adapter_owned_live_checks(tmp_path):
    from mlx2.adapters.mlx_vlm import GEMMA3N

    required = required_descriptor_checks(GEMMA3N)
    assert required == {
        "multimodal_image",
        "multimodal_video",
        "multimodal_audio_input",
        "multimodal_encoder_batching",
        "multimodal_continuous_batch",
        "multimodal_apcv2_reuse",
    }
    path = tmp_path / "qualification.json"
    settings = {"mtp": False, "max_context": 32768}
    checks = {name: {"passed": True} for name in REQUIRED_CHECKS}
    record = {
        "passed": True,
        "runtime": {"source": "abc"},
        "artifact": "weights",
        "settings": settings,
        "qualification_harness": APPROVED_QUALIFICATION_HARNESS,
        "checks": checks,
    }
    path.write_text(json.dumps(record))
    args = dict(
        runtime=record["runtime"],
        artifact=record["artifact"],
        settings=settings,
        descriptor=GEMMA3N,
        name="mlx-vlm-apcv2-ordinary",
    )
    with pytest.raises(
        ValueError, match="adapter qualification producer is unavailable"
    ):
        load_qualified_route(path, **args)
    record["checks"].update({name: {"passed": True} for name in required})
    path.write_text(json.dumps(record))
    with pytest.raises(
        ValueError, match="adapter qualification producer is unavailable"
    ):
        load_qualified_route(path, **args)


def _rebind_to_current_producer(companion):
    """The recorded 2026-09-26 M3 runs name the producer revision that ran
    them; the producers have since tightened the media-reuse predicate, which
    by design invalidates those receipts until they are re-run.  These tests
    exercise the route-binding contract, so they re-bind the recorded traces
    to the current producer pin (the evaluator still recomputes every check
    from the real traces)."""
    from mlx2.qualification import APPROVED_MEDIA_PRODUCERS, PINNED_MEDIA_SOURCE_REVISION

    companion["qualification_harness"] = APPROVED_MEDIA_PRODUCERS[companion["model_type"]][0]
    # The runs also predate the mlx-vlm dependency-content contract: their
    # ``mlx_vlm`` identity names an install, not the executed source bytes.
    # Re-bind that identity the same way (the parity rows stay as recorded).
    identity = {"schema": "mlx2.vlm-dependencies.v1", "family": companion["model_type"],
                "source_sha256": "c" * 64, "dependency_files": 105,
                "reference_revision": PINNED_MEDIA_SOURCE_REVISION}
    companion["settings"]["mlx_vlm"] = dict(identity)
    for arm in companion["arms"].values():
        arm["mlx_vlm_runtime"] = dict(identity)
        arm["parity"]["source_sha256"] = identity["source_sha256"]
        if "binding" in arm:
            arm["binding"]["settings"]["mlx_vlm"] = dict(identity)
    for case in companion.get("text_source", {}).get("cases", {}).values():
        case["source_sha256"] = identity["source_sha256"]


def test_smol_companion_is_recomputed_before_route_selection(tmp_path):
    from mlx2.adapters.smolvlm2 import DESCRIPTOR

    companion = json.loads((Path(__file__).resolve().parents[1] / "docs/experiments"
                            / "SMOLVLM2-M3-LIVE-MEDIA-QUALIFICATION-2026-09-26.json").read_text())
    _rebind_to_current_producer(companion)
    record = {
        "passed": True, "runtime": companion["runtime"],
        "artifact": companion["artifact"], "settings": companion["settings"],
        "qualification_harness": APPROVED_QUALIFICATION_HARNESS,
        "checks": {name: {"passed": True} for name in required_generic_checks(DESCRIPTOR)},
        "adapter_qualification": companion,
    }
    path = tmp_path / "synthetic-generic.json"
    path.write_text(json.dumps(record))
    args = {"runtime": record["runtime"], "artifact": record["artifact"],
            "settings": record["settings"], "descriptor": DESCRIPTOR,
            "name": "synthetic-contract-check"}
    assert load_qualified_route(path, **args).profile.name == "synthetic-contract-check"

    forged = json.loads(json.dumps(record))
    forged["adapter_qualification"]["arms"]["video"]["parity"]["decode"][0]["max_abs"] = 1.0
    path.write_text(json.dumps(forged))
    with pytest.raises(ValueError, match="traces"):
        load_qualified_route(path, **args)


@pytest.mark.parametrize("family,module_name,descriptor_name", [
    ("QWEN25", "mlx2.adapters.qwen25_vl", "DESCRIPTOR"),
    ("LFM25", "mlx2.adapters.lfm25_vl", "LFM25_VL"),
])
def test_family_media_companion_binds_normal_route_contract(
    tmp_path, family, module_name, descriptor_name,
):
    import importlib

    descriptor = getattr(importlib.import_module(module_name), descriptor_name)
    companion = json.loads((Path(__file__).resolve().parents[1] / "docs/experiments"
                            / f"{family}-M3-LIVE-MEDIA-QUALIFICATION-2026-09-26.json").read_text())
    _rebind_to_current_producer(companion)
    record = {
        "passed": True, "runtime": companion["runtime"],
        "artifact": companion["artifact"], "settings": companion["settings"],
        "qualification_harness": APPROVED_QUALIFICATION_HARNESS,
        "checks": {name: {"passed": True} for name in required_generic_checks(descriptor)},
        "adapter_qualification": companion,
    }
    path = tmp_path / "synthetic-generic.json"
    path.write_text(json.dumps(record))
    args = {"runtime": record["runtime"], "artifact": record["artifact"],
            "settings": record["settings"], "descriptor": descriptor,
            "name": "synthetic-contract-check"}
    assert load_qualified_route(path, **args).profile.name == "synthetic-contract-check"

    forged = json.loads(json.dumps(record))
    forged["adapter_qualification"]["arms"]["video"]["parity"]["decode"][0]["max_abs"] = 1.0
    path.write_text(json.dumps(forged))
    with pytest.raises(ValueError, match="traces"):
        load_qualified_route(path, **args)


def test_advanced_qualified_domain_requires_actual_mechanisms():
    from mlx2.qualification import required_feature_checks
    settings = {"mtp": True, "max_context": 131072,
                "execution_policy": {"segment_aware_async_qsa_promotion": True, "prefetch_known_tail_ple": True},
                "environment": {"MLX_LM_SHARED_QSA_SUFFIX": "auto", "MLX_QWEN4_QSA_INDEXED": "auto"}}
    assert required_feature_checks(settings) == {"feature_async_promotion", "feature_known_tail_prefetch", "feature_shared_qsa", "feature_indexed_qsa", "feature_private_delta"}
    settings["mtp"] = False
    assert required_feature_checks(settings) == set()


def test_prompt_lookup_route_requires_observed_proposal_and_rollback():
    from mlx2.qualification import required_feature_checks
    from scripts.qualify_serving import feature_observations

    settings = {"speculation": "prompt_lookup"}
    assert required_feature_checks(settings) == {
        "feature_prompt_lookup",
        "feature_prompt_lookup_proposals",
        "feature_prompt_lookup_rollback",
    }
    observed = feature_observations(
        {
            "settings": settings,
            "scheduler": {
                "pld_retrieval_cycles": 2,
                "pld_proposed": 8,
                "pld_rollbacks": 1,
            },
        }
    )
    assert observed["prompt_lookup"] == 2
    assert observed["prompt_lookup_proposals"] == 8
    assert observed["prompt_lookup_rollback"] == 1
    assert observed["prompt_lookup_rotating_replay"] == 0


@pytest.mark.parametrize("speculation", ["prompt_lookup", "external_draft"])
def test_speculative_routes_preserve_target_mechanism_requirements(speculation):
    from mlx2.qualification import required_feature_checks

    route = {"speculation": speculation, "mtp": False}
    checks = required_feature_checks(route)
    assert required_feature_checks({
        **route,
        "apc_rolling_checkpoints": {"interval_tokens": 16},
        "prefill_scheduling": {"enabled": True},
        "environment": {
            "MLX_QWEN4_PLE_NVME": "/artifact/ple_rows.bin",
            "MLX_QWEN4_PLE_COMPILE": "1",
            "MLX_QWEN4_FUSED_GDN_DECODE": "1",
        },
    }) == checks | {
        "feature_file_backed_ple", "feature_compiled_ple", "feature_fused_gdn_decode",
        "feature_apc_rolling_checkpoints", "feature_prefill_scheduling",
    }


def test_prompt_lookup_cannot_qualify_without_selected_target_mechanism(tmp_path):
    from mlx2.qualification import required_feature_checks

    settings = {"speculation": "prompt_lookup", "mtp": False}
    checks = REQUIRED_CHECKS | required_feature_checks(settings) | {"structured_output"}
    settings["environment"] = {"MLX_QWEN4_PLE_NVME": "/artifact/ple_rows.bin"}
    record = {
        "passed": True,
        "runtime": {"source": "abc"},
        "artifact": "weights",
        "settings": settings,
        "qualification_harness": APPROVED_QUALIFICATION_HARNESS,
        "checks": {name: {"passed": True} for name in checks},
    }
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="feature_file_backed_ple"):
        load_qualified_route(
            path, runtime=record["runtime"], artifact="weights", settings=settings,
            descriptor=QWEN4_FLASH_NEXT, name="prompt-lookup",
        )


def test_rotating_replay_policy_requires_observed_transaction_rounds():
    from mlx2.qualification import required_feature_checks
    from scripts.qualify_serving import feature_observations

    base = {
        "feature_prompt_lookup",
        "feature_prompt_lookup_proposals",
        "feature_prompt_lookup_rollback",
    }
    for policy in ({}, {"rotating_replay": False}, None):
        settings = {"speculation": "prompt_lookup", "prompt_lookup": policy}
        assert required_feature_checks(settings) == base
    settings = {
        "speculation": "prompt_lookup",
        "prompt_lookup": {"rotating_replay": True},
    }
    assert required_feature_checks(settings) == base | {
        "feature_prompt_lookup_rotating_replay"
    }
    # The knob configures nothing on a route that does not run prompt lookup.
    assert "feature_prompt_lookup_rotating_replay" not in required_feature_checks(
        {"speculation": "ordinary", "prompt_lookup": {"rotating_replay": True}}
    )
    observed = feature_observations(
        {"settings": settings, "scheduler": {"pld_rotating_replay_rounds": 3}}
    )
    assert observed["prompt_lookup_rotating_replay"] == 3


def test_apc_persistence_and_sessions_require_lifecycle_observations():
    from mlx2.qualification import required_feature_checks
    from scripts.qualify_serving import feature_observations

    assert required_feature_checks(
        {"mtp": False, "disk_cache": False, "apc_persistence": False}
    ) == set()
    assert required_feature_checks(
        {"mtp": False, "disk_cache": True, "apc_persistence": False}
    ) == {"feature_apc_sessions"}
    assert required_feature_checks(
        {"mtp": False, "disk_cache": True, "apc_persistence": True}
    ) == {"feature_apc_persistence", "feature_apc_sessions"}

    observed = feature_observations(
        {
            "apcv2": {
                "idle_disk": {
                    "persisted_writes": 2,
                    "restores": 1,
                    "parks": 3,
                    "resumes": 2,
                    "prefetch_restores_ok": 1,
                    "prefetch_hits": 1,
                },
                "persistence": {"rescan": {"registered": 1}},
            }
        }
    )
    assert observed["apc_persistence"] == 1
    assert observed["apc_sessions"] == 1
    missing = feature_observations(
        {
            "apcv2": {
                "idle_disk": {"persisted_writes": 4, "restores": 0},
                "persistence": {"rescan": {"registered": 2}},
            }
        }
    )
    assert missing["apc_persistence"] == 0
    assert missing["apc_sessions"] == 0


@pytest.mark.parametrize(
    "disabled",
    [None, "", "   ", 0, False, "0", " false ", "FALSE", "Off", " NO "],
)
@pytest.mark.parametrize(
    "name",
    ["MLX_LM_SHARED_QSA_SUFFIX", "MLX_QWEN4_QSA_INDEXED"],
)
def test_mode_environment_disabled_tokens_do_not_require_features(name, disabled):
    from mlx2.qualification import required_feature_checks

    settings = {
        "mtp": True,
        "max_context": 262144,
        "execution_policy": {},
        "environment": {name: disabled},
    }
    assert required_feature_checks(settings) == set()


@pytest.mark.parametrize(
    "enabled",
    [1, True, "1", " true ", "ON", "yes", "enabled", " Auto "],
)
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "MLX_LM_SHARED_QSA_SUFFIX",
            {"feature_shared_qsa", "feature_private_delta"},
        ),
        ("MLX_QWEN4_QSA_INDEXED", {"feature_indexed_qsa"}),
    ],
)
def test_mode_environment_enabled_tokens_require_selected_features(
    name, expected, enabled
):
    from mlx2.qualification import required_feature_checks

    settings = {
        "mtp": True,
        "max_context": 262144,
        "execution_policy": {},
        "environment": {name: enabled},
    }
    assert required_feature_checks(settings) == expected


def test_qwen38_disabled_shared_qsa_environment_does_not_require_private_delta():
    from mlx2.qualification import required_feature_checks

    settings = {
        "mtp": True,
        "max_context": 262144,
        "execution_policy": {
            "persistent": True,
            "num_draft": 2,
            "rate_gate": False,
            "prefill_step_size": 2048,
            "segment_aware_live_tip": True,
            "segment_aware_cohort_size": 20,
        },
        "environment": {
            "MLX_LM_SHARED_QSA_SUFFIX": "0",
            "MLX_LM_SEGMENTED_SELF_MTP": "1",
            "MLX_LM_TRUE_BATCHED_SEGMENTED_MTP": "1",
        },
    }
    assert required_feature_checks(settings) == set()


@pytest.mark.parametrize("harness", [
    None,
    {},
    {"schema": "foreign"},
    {
        "schema": "mlx2.qualification-harness.v1",
        "name": "scripts/qualify_serving.py",
        "sha256": "stale",
    },
])
def test_qualification_rejects_missing_or_foreign_harness(tmp_path, harness):
    path = tmp_path / "qualification.json"
    record = {
        "passed": True,
        "runtime": {"source": "abc"},
        "artifact": "weights",
        "settings": {"mtp": True, "max_context": 16384},
        "checks": {c: {"passed": True} for c in REQUIRED_CHECKS | {"mtp_execution"}},
    }
    if harness is not None:
        record["qualification_harness"] = harness
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="approved harness"):
        load_qualified_route(
            path,
            runtime=record["runtime"],
            artifact=record["artifact"],
            settings=record["settings"],
            descriptor=QWEN4_FLASH_NEXT,
            name="test",
        )


def test_flash_parity_mechanisms_are_required_for_ordinary_and_mtp():
    from mlx2.qualification import required_feature_checks

    environment = {
        "MLX_QWEN4_PLE_NVME": "/model/ple_rows.bin",
        "MLX_QWEN4_PLE_COMPILE": "1",
        "MLX_QWEN4_QSA_POOLED_KEY_CACHE": "1",
        "MLX_QWEN4_QSA_SCATTER_CHOSEN": "1",
        "MLX_QWEN4_FUSED_GDN_DECODE": "1",
        "MLX_QWEN4_FUSED_GDN_VERIFY": "1",
        "MLX_QWEN4_EAGER_DISPATCH": "1",
        "MLX_QWEN4_MOE_FUSED_GATE_UP": "1",
        "MLX_QWEN4_FUSED_EXPERT_KERNEL": "auto",
        "MLX_LM_SHARED_QSA_SUFFIX": "off",
        "MLX_QWEN4_QSA_INDEXED": "off",
    }
    ordinary = required_feature_checks({
        "mtp": False, "max_context": 16384, "environment": environment,
        "execution_policy": {},
    })
    assert ordinary == {
        "feature_file_backed_ple", "feature_compiled_ple",
        "feature_pooled_qsa", "feature_scatter_qsa",
        "feature_fused_gdn_decode", "feature_eager_dispatch", "feature_fused_moe",
    }
    mtp = required_feature_checks({
        "mtp": True, "max_context": 16384, "environment": environment,
        "execution_policy": {},
    })
    assert mtp == ordinary | {"feature_fused_gdn_verify"}
    replay = required_feature_checks({
        "mtp": True, "max_context": 16384,
        "environment": {**environment, "MLX_QWEN4_FUSED_GDN_REPLAY_ROLLBACK": "1"},
        "execution_policy": {},
    })
    assert replay == mtp | {"feature_fused_gdn_replay_rollback"}
    # Replay rollback rides on the verify kernel; alone it is not a gate.
    verify_off = required_feature_checks({
        "mtp": True, "max_context": 16384,
        "environment": {**environment, "MLX_QWEN4_FUSED_GDN_VERIFY": "0",
                        "MLX_QWEN4_FUSED_GDN_REPLAY_ROLLBACK": "1"},
        "execution_policy": {},
    })
    assert "feature_fused_gdn_replay_rollback" not in verify_off


def test_selected_flash_parity_mechanism_without_check_fails_closed(tmp_path):
    path = tmp_path / "qualification.json"
    settings = {
        "mtp": False, "max_context": 16384,
        "environment": {"MLX_QWEN4_PLE_NVME": "/model/ple_rows.bin"},
        "execution_policy": {},
    }
    checks = {
        name: {"passed": True}
        for name in REQUIRED_CHECKS | {"structured_output"}
    }
    record = {"passed": True, "runtime": {"source": "abc"},
              "artifact": "weights", "settings": settings,
              "qualification_harness": APPROVED_QUALIFICATION_HARNESS,
              "checks": checks}
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="mechanisms lack observed qualification"):
        load_qualified_route(path, runtime=record["runtime"], artifact="weights",
                             settings=settings, descriptor=QWEN4_FLASH_NEXT, name="test")


def test_external_draft_status_counters_cover_all_required_features():
    from scripts.qualify_serving import feature_observations

    observed = feature_observations({"scheduler": {
        "external_rounds": 3, "proposed_tokens": 9,
        "paired_cache_resumes": 2, "segmented_transactions": 3,
    }})
    assert {name for name in ("external_draft", "proposal_distribution",
                               "paired_draft_cache", "segmented_transaction")
            if observed[name] > 0} == {
        "external_draft", "proposal_distribution",
        "paired_draft_cache", "segmented_transaction",
    }


def test_external_draft_fallback_invalidates_optimized_feature_evidence():
    from scripts.qualify_serving import feature_observations

    observed = feature_observations({"scheduler": {
        "external_rounds": 3, "proposed_tokens": 9,
        "paired_cache_resumes": 2, "segmented_transactions": 3,
        "draft_fallbacks": 1,
    }})
    assert observed["external_draft"] == 0


def test_native_mtp_feature_evidence_uses_segmented_execution_counters():
    from scripts.qualify_serving import feature_observations

    observed = feature_observations({
        "settings": {"mtp": True},
        "execution": {"segmented_mtp": {
            "transaction_branches": 12,
            "transaction_promotions": 12,
            "transaction_rejections": 0,
            "accepted_zero": 5,
            "accepted_partial": 4,
            "accepted_all": 3,
        }},
        "scheduler": {},
    })

    assert observed["segmented_transaction"] == 12
    assert observed["segmented_rollback"] == 9


def test_native_mtp_rollback_evidence_fails_closed_without_rejected_suffix():
    from scripts.qualify_serving import feature_observations

    observed = feature_observations({
        "settings": {"mtp": True},
        "execution": {"segmented_mtp": {
            "transaction_branches": 3,
            "transaction_promotions": 3,
            "transaction_rejections": 0,
            "accepted_zero": 0,
            "accepted_partial": 0,
            "accepted_all": 3,
        }},
        "scheduler": {"segmented_rollbacks": 99},
    })

    # Native MTP must prove its own segmented rollback.  Unrelated scheduler
    # evidence cannot satisfy the route gate.
    assert observed["segmented_transaction"] == 3
    assert observed["segmented_rollback"] == 0


def test_spomin_policy_requires_an_observed_surgery():
    from mlx2.qualification import required_feature_checks
    from scripts.qualify_serving import feature_observations

    assert "feature_spomin_surgery" not in required_feature_checks(
        {"mtp": False, "spomin_live_surgery": {"enabled": False}}
    )
    assert "feature_spomin_surgery" in required_feature_checks(
        {"mtp": False, "spomin_live_surgery": {"enabled": True, "capacity_tokens": 8}}
    )
    assert feature_observations({})["spomin_surgery"] == 0
    observed = feature_observations(
        {"spomin_live_surgery": {"counts": {"applied": 3, "declined": 1}}}
    )
    assert observed["spomin_surgery"] == 3


def test_interior_checkpoint_policy_requires_captured_and_published_observation():
    from mlx2.qualification import required_feature_checks
    from scripts.qualify_serving import feature_observations

    settings = {
        "mtp": False,
        "speculation": "ordinary",
        "apc_interior_checkpoints": {"count": 3, "min_stride": 64},
    }
    assert "feature_apc_interior_checkpoints" in required_feature_checks(settings)
    assert feature_observations({})["apc_interior_checkpoints"] == 0
    assert feature_observations(
        {
            "counts": {
                "apc_interior_checkpoints_captured": 3,
                "apc_interior_checkpoints_published": 2,
            }
        }
    )["apc_interior_checkpoints"] == 2
    assert feature_observations(
        {
            "counts": {
                "apc_interior_checkpoints_captured": 3,
                "apc_interior_checkpoints_published": 0,
            }
        }
    )["apc_interior_checkpoints"] == 0


def test_selected_interior_checkpoints_without_observation_fail_qualification(tmp_path):
    path = tmp_path / "qualification.json"
    settings = {
        "mtp": False,
        "max_context": 16384,
        "speculation": "ordinary",
        "execution_policy": {},
        "environment": {},
        "apc_interior_checkpoints": {"count": 2, "min_stride": 64},
    }
    checks = {
        name: {"passed": True}
        for name in REQUIRED_CHECKS | {"structured_output"}
    }
    record = {
        "passed": True,
        "runtime": {"source": "abc"},
        "artifact": "weights",
        "settings": settings,
        "qualification_harness": APPROVED_QUALIFICATION_HARNESS,
        "checks": checks,
    }
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="mechanisms lack observed qualification"):
        load_qualified_route(
            path,
            runtime=record["runtime"],
            artifact="weights",
            settings=settings,
            descriptor=QWEN4_FLASH_NEXT,
            name="test",
        )


def test_adaptive_mtp_requires_passing_bound_benchmark_observation(tmp_path):
    path = tmp_path / "qualification.json"
    settings = {
        "mtp": True,
        "max_context": 16384,
        "speculation": "native_mtp",
        "execution_policy": {"num_draft": 2},
        "environment": {},
        "adaptive_mtp_depth": {"enabled": True},
    }
    checks = {
        name: {"passed": True}
        for name in REQUIRED_CHECKS | {"structured_output", "mtp_execution"}
    }
    record = {
        "passed": True,
        "runtime": {"source": "abc"},
        "artifact": "weights",
        "settings": settings,
        "qualification_harness": APPROVED_QUALIFICATION_HARNESS,
        "checks": checks,
    }
    args = dict(
        runtime=record["runtime"],
        artifact="weights",
        settings=settings,
        descriptor=QWEN4_FLASH_NEXT,
        name="adaptive",
    )
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="mechanisms lack observed qualification"):
        load_qualified_route(path, **args)
    record["checks"]["feature_adaptive_mtp_depth"] = {
        "passed": True,
        "evidence": {
            "benchmark_sha256": "abc",
            "correctness": True,
            "throughput": True,
            "alternatives_measured": True,
            "recovery_implication": True,
        },
    }
    path.write_text(json.dumps(record))
    assert "profile=adaptive" in load_qualified_route(path, **args).receipt


def test_mtp_ordinary_handoff_requires_observed_feature_check(tmp_path):
    path = tmp_path / "qualification.json"
    settings = {
        "mtp": True,
        "max_context": 16384,
        "speculation": "native_mtp",
        "execution_policy": {"num_draft": 2},
        "environment": {},
        # Above the handoff width: at max_lanes <= 8 it can never engage.
        "max_lanes": 16,
        "mtp_ordinary_handoff": {"enabled": True, "max_mtp_width": 8},
    }
    checks = {
        name: {"passed": True}
        for name in REQUIRED_CHECKS | {"structured_output", "mtp_execution"}
    }
    record = {
        "passed": True,
        "runtime": {"source": "abc"},
        "artifact": "weights",
        "settings": settings,
        "qualification_harness": APPROVED_QUALIFICATION_HARNESS,
        "checks": checks,
    }
    args = dict(
        runtime=record["runtime"],
        artifact="weights",
        settings=settings,
        descriptor=QWEN4_FLASH_NEXT,
        name="handoff",
    )
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="feature_mtp_ordinary_handoff"):
        load_qualified_route(path, **args)
    record["checks"]["feature_mtp_ordinary_handoff"] = {
        "passed": True,
        "evidence": {
            "events": 1,
            "lanes": 16,
            "comparison_count": 16,
            "unsafe_divergences": [],
        },
    }
    path.write_text(json.dumps(record))
    assert "profile=handoff" in load_qualified_route(path, **args).receipt


def test_disabled_adaptive_settings_match_a_real_committed_record():
    from mlx2.runtime.adaptive_policy import AdaptiveMTPDepthPolicy

    path = Path(
        "qualification/runs/integration-gpu-20260918/"
        "muse-ordinary-qualification.json"
    )
    record = json.loads(path.read_text())
    current_settings = dict(record["settings"])
    current_settings["adaptive_mtp_depth"] = (
        AdaptiveMTPDepthPolicy.from_value(None).as_dict()
    )
    assert current_settings == record["settings"]


def test_prompt_lookup_keeps_common_environment_evidence_requirements():
    from mlx2.qualification import required_feature_checks

    settings = {
        "speculation": "prompt_lookup",
        "environment": {
            "MLX_QWEN4_PLE_NVME": "/fixture/ple_rows.bin",
            "MLX_QWEN4_PLE_COMPILE": "1",
        },
    }
    checks = required_feature_checks(settings)
    assert "feature_prompt_lookup" in checks
    assert "feature_file_backed_ple" in checks
    assert "feature_compiled_ple" in checks


@pytest.mark.parametrize("speculation", [None, "prompt_lookup", "external_draft"])
def test_selected_sp_qmm_requires_observed_routed_calls(speculation):
    from mlx2.qualification import required_feature_checks
    from scripts.qualify_serving import feature_observations

    settings = {"mtp": False, "sp_qmm": {"modules": 2, "policy": "measured"}}
    if speculation is not None:
        settings["speculation"] = speculation
    assert "feature_sp_qmm" in required_feature_checks(settings)
    initial = {
        "settings": settings,
        "sp_qmm": {"enabled": True, "modules": 2, "routed_calls": 7, "stock_calls": 4},
    }
    final = {
        "settings": settings,
        "sp_qmm": {"enabled": True, "modules": 2, "routed_calls": 7, "stock_calls": 19},
    }
    # A selected policy, eligible modules, prior routed calls, and stock
    # fallback calls cannot certify execution in this qualification run.
    assert feature_observations(final, initial=initial)["sp_qmm"] == 0
    assert feature_observations(final)["sp_qmm"] == 0
    final["sp_qmm"]["routed_calls"] = 9
    assert feature_observations(final, initial=initial)["sp_qmm"] == 2
    final["sp_qmm"]["enabled"] = False
    assert feature_observations(final, initial=initial)["sp_qmm"] == 0
    final["sp_qmm"]["enabled"] = True
    final["sp_qmm"]["modules"] = 0
    assert feature_observations(final, initial=initial)["sp_qmm"] == 0
    final["sp_qmm"]["modules"] = 2
    final["settings"] = {"mtp": False}
    assert feature_observations(final, initial=initial)["sp_qmm"] == 0


def test_sp_qmm_qualification_receipt_requires_feature_check(tmp_path):
    from mlx2.qualification import required_feature_checks

    settings = {"mtp": False, "max_context": 32768,
                "sp_qmm": {"modules": 2, "policy": "measured"}}
    record = {
        "passed": True,
        "runtime": {"source": "abc"},
        "artifact": "weights",
        "settings": settings,
        "qualification_harness": APPROVED_QUALIFICATION_HARNESS,
        "checks": {name: {"passed": True}
                   for name in REQUIRED_CHECKS | {"structured_output"}},
    }
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="feature_sp_qmm"):
        load_qualified_route(
            path, runtime=record["runtime"], artifact="weights", settings=settings,
            descriptor=QWEN4_FLASH_NEXT, name="sp-qmm-ordinary",
        )
    assert "feature_sp_qmm" in required_feature_checks(settings)


def test_quantized_verify_requires_run_local_multirow_engagement():
    from scripts.qualify_serving import feature_observations

    settings = {
        "max_context": 65536,
        "approximate_kv": {"enabled": True, "compose_mtp": True},
        "qsdpa_verify_kernel": {"enabled": True, "min_context": 32768},
    }
    initial = {
        "settings": settings,
        "qsdpa_verify": {
            "counts": {"verify_kernel_calls": 4},
            "by_rows": {"verify_kernel_L1": 2, "verify_kernel_L3": 2},
        },
    }
    final = {
        "settings": settings,
        "qsdpa_verify": {
            "counts": {"verify_kernel_calls": 5},
            "by_rows": {"verify_kernel_L1": 3, "verify_kernel_L3": 2},
        },
    }
    # Prior calls and a new single-row fallback cannot certify MTP verify.
    assert feature_observations(final, initial=initial)["qsdpa_verify_kernel"] == 0
    assert feature_observations(final)["qsdpa_verify_kernel"] == 0
    final["qsdpa_verify"]["counts"]["verify_kernel_calls"] = 6
    final["qsdpa_verify"]["by_rows"]["verify_kernel_L3"] = 3
    assert feature_observations(final, initial=initial)["qsdpa_verify_kernel"] == 1
    final["settings"] = {
        **settings,
        "qsdpa_verify_kernel": {"enabled": False, "min_context": 32768},
    }
    assert feature_observations(final, initial=initial)["qsdpa_verify_kernel"] == 0
    ordinary = {**settings, "approximate_kv": {"enabled": True}}
    initial["settings"] = final["settings"] = ordinary
    assert feature_observations(final, initial=initial)["qsdpa_verify_kernel"] == 2


def test_quantized_verify_qualification_receipt_requires_feature_check(tmp_path):
    settings = {
        "mtp": True,
        "max_context": 65536,
        "approximate_kv": {"enabled": True, "operation": "kv_q8", "compose_mtp": True},
        "qsdpa_verify_kernel": {"enabled": True, "min_context": 32768},
    }
    record = {
        "passed": True,
        "runtime": {"source": "abc"},
        "artifact": "weights",
        "settings": settings,
        "qualification_harness": APPROVED_QUALIFICATION_HARNESS,
        "checks": {name: {"passed": True}
                   for name in REQUIRED_CHECKS | {"structured_output", "mtp_execution"}
                   | required_descriptor_checks(QWEN4_FLASH_NEXT)},
    }
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="feature_qsdpa_verify_kernel"):
        load_qualified_route(
            path, runtime=record["runtime"], artifact="weights", settings=settings,
            descriptor=QWEN4_FLASH_NEXT, name="quantized-verify-mtp",
        )


def _selectable_feature_names():
    """Every feature ``required_feature_checks`` can demand, read from source.

    Reading the source rather than a hand-built settings matrix means a newly
    added selectable mechanism is covered the moment it is added, whatever
    settings shape selects it.
    """
    import ast
    import inspect

    import mlx2.qualification as qualification

    tree = ast.parse(inspect.getsource(qualification))
    names = set()
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef) or function.name not in {
            "required_feature_checks",
            "_route_feature_checks",
        }:
            continue
        for node in ast.walk(function):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"add", "update"}
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in {"features", "common"}
            ):
                values = node.args
                if node.func.attr == "update":
                    values = [
                        element for arg in node.args
                        if isinstance(arg, (ast.Set, ast.List, ast.Tuple))
                        for element in arg.elts
                    ]
                names.update(
                    arg.value for arg in values
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                )
            elif isinstance(node, ast.Set):
                names.update(
                    element.value for element in node.elts
                    if isinstance(element, ast.Constant)
                    and isinstance(element.value, str)
                    and element.value.startswith("feature_")
                )
    return {name.removeprefix("feature_") for name in names}


def test_every_selectable_feature_has_a_harness_observation():
    from scripts.qualify_serving import feature_observations, unobservable_features

    selectable = _selectable_feature_names()
    # The source scan must itself see the whole family, or it proves nothing.
    assert {
        "apc_junction_checkpoints", "apc_rolling_checkpoints",
        "apc_inflight_prefix_wait",
        "external_pairwise_selection", "fused_gdn_dynamic_accept",
        "host_memory_signals", "memory_preemption", "moe_expert_streaming",
        "prefill_scheduling", "tool_grammar_auto", "tool_grammar_streaming",
        "external_draft", "apc_sessions", "int8_prefill",
        "varlen_dense_mlp", "varlen_sparse_moe",
        "external_varlen_prefill", "ingress_cohort",
    } <= selectable
    assert sorted(selectable - set(feature_observations({}))) == []
    assert unobservable_features(selectable) == []
    # The startup guard names what the approved harness cannot observe.
    assert unobservable_features(selectable | {"not_a_mechanism"}) == [
        "not_a_mechanism"
    ]


@pytest.mark.parametrize(
    ("feature", "engaged", "idle"),
    [
        (
            "apc_rolling_checkpoints",
            {"counts": {"apc_rolling_checkpoints_published": 2},
             "apcv2": {"lifetime": {"rolling_hits": 1}}},
            # Published but never resumed from: the hybrid route proved nothing.
            {"counts": {"apc_rolling_checkpoints_published": 2},
             "apcv2": {"lifetime": {"rolling_hits": 0}}},
        ),
        (
            "apc_rolling_checkpoints",
            {"counts": {"apc_rolling_checkpoints_cancel_published": 1}},
            {"counts": {"apc_rolling_checkpoints_planned": 4}},
        ),
        (
            "apc_junction_checkpoints",
            {"counts": {"apc_junction_checkpoints_published": 1},
             "apcv2": {"lifetime": {"junction_hits": 1}}},
            {"counts": {"apc_junction_checkpoints_planned": 3,
                        "apc_junction_checkpoints_published": 1},
             "apcv2": {"lifetime": {"junction_hits": 0}}},
        ),
        (
            "apc_inflight_prefix_wait",
            {"counts": {"apc_inflight_checkpoints_published": 1,
                         "apc_inflight_prefix_hits": 3}},
            {"counts": {"apc_inflight_checkpoints_published": 1,
                         "apc_inflight_prefix_hits": 0}},
        ),
        (
            "external_pairwise_selection",
            {"scheduler": {"external_pairwise_selection_groups": 2}},
            {"scheduler": {"external_pairwise_selection_groups": 0}},
        ),
        (
            "fused_gdn_dynamic_accept",
            {"execution": {"fused_gdn": {"replay_dynamic_rollback_calls": 3}}},
            {"execution": {"fused_gdn": {"replay_rollback_calls": 3,
                                         "replay_dynamic_rollback_calls": 0}}},
        ),
        (
            "host_memory_signals",
            {"host_memory_pressure_level": 0,
             "host_memory_available_bytes": 64 << 30},
            # The policy is on but the host reading was never produced.
            {"settings": {"host_memory_signals": {"enabled": True}},
             "host_memory_pressure_level": 0},
        ),
        (
            "memory_preemption",
            {"counts": {"memory_preemptions": 1, "preempted_replays": 1}},
            {"counts": {"memory_preemptions": 1, "preempted_replays": 0}},
        ),
        (
            "moe_expert_streaming",
            {"counts": {"stream_page_ins_total": 12}},
            {"counts": {"stream_page_ins_total": 0,
                        "stream_expert_hits_total": 40}},
        ),
        (
            "prefill_scheduling",
            {"scheduler": {"prefill_scheduling_bypasses": 2}},
            {"scheduler": {"prefill_scheduling_bypasses": 0,
                           "prefill_scheduling_bypass_forced": 0,
                           "prefill_scheduling_one_slice_clamps": 5}},
        ),
        (
            "prefill_scheduling",
            {"scheduler": {"prefill_scheduling_bypass_forced": 1}},
            {"settings": {"prefill_scheduling": {"order": "srpt"}}},
        ),
        (
            "tool_grammar_auto",
            {"counts": {"constrained_tool_grammar_auto_engagements": 1}},
            {"counts": {"constrained_tool_grammar_engagements": 4,
                        "constrained_tool_grammar_auto_engagements": 0}},
        ),
        (
            "tool_grammar_streaming",
            {"counts": {"constrained_tool_grammar_streams": 1}},
            {"counts": {"constrained_tool_grammar_engagements": 4,
                        "constrained_tool_grammar_streams": 0}},
        ),
    ],
)
def test_selectable_feature_observations_need_the_mechanism_to_engage(
    feature, engaged, idle
):
    from scripts.qualify_serving import feature_observations

    assert feature_observations(engaged)[feature] > 0
    # A selected policy (or any counter short of engagement) is not evidence.
    assert feature_observations(idle)[feature] == 0
    assert feature_observations({})[feature] == 0


def test_prefill_candidates_require_observed_calls_during_qualification():
    from mlx2.qualification import required_feature_checks
    from scripts.qualify_serving import feature_observations

    settings = {"prefill_execution": {"projection": {"kernel": "native"}, "scan": {"chunk_size": 8}}}
    assert {"feature_prefill_projection", "feature_prefill_scan"} <= required_feature_checks(settings)
    initial = {"execution": {
        "tensorfold_prefill": {"counters": {"grouped_calls": 9, "swiglu_calls": 3}},
        "gdn_prefill_scan": {"counters": {"calls": 11}},
    }}
    assert feature_observations(initial)["prefill_projection"] == 0
    assert feature_observations(initial, initial=initial)["prefill_scan"] == 0
    final = {"execution": {
        "tensorfold_prefill": {"counters": {"grouped_calls": 10, "swiglu_calls": 5}},
        "gdn_prefill_scan": {"counters": {"calls": 12}},
    }}
    observed = feature_observations(final, initial=initial)
    assert observed["prefill_projection"] == 3
    assert observed["prefill_scan"] == 1
    assert not {"feature_prefill_projection", "feature_prefill_scan"} & required_feature_checks({})


def test_varlen_and_external_prefill_require_run_local_engagement():
    from mlx2.qualification import required_feature_checks
    from scripts.qualify_serving import feature_observations

    settings = {
        "speculation": "external_draft",
        "prefill_execution": {
            "varlen": {"schema": "mlx2.varlen-dense-mlp.v1"},
            "external_varlen_prefill": {
                "schema": "mlx2.external-varlen-prefill.v1"
            },
        },
        "execution_policy": {
            "pairwise_selection": "batched",
            "external_varlen_prefill": {
                "enabled": True,
                "schema": "mlx2.external-varlen-prefill.v1",
            },
            "ingress_cohort": {
                "enabled": True,
                "mechanism": "external_varlen_prefill",
                "target_lanes": 4,
            },
        },
    }
    assert {
        "feature_varlen_dense_mlp",
        "feature_external_varlen_prefill",
        "feature_ingress_cohort",
        "feature_external_pairwise_selection",
    } <= required_feature_checks(settings)
    initial = {
        "execution": {
            "varlen_dense_mlp": {
                "counters": {
                    "mlp_compaction_calls": 7,
                    "padding_token_rows": 100,
                }
            }
        },
        "scheduler": {
            "external_batched_prefill_rounds": 3,
            "external_batched_prefill_lanes": 6,
            "external_pairwise_selection_groups": 4,
        },
        "counts": {
            "ingress_cohort_observed_used": 2,
            "ingress_cohort_target_reached": 1,
        },
    }
    idle = feature_observations(initial, initial=initial)
    assert idle["varlen_dense_mlp"] == 0
    assert idle["external_varlen_prefill"] == 0
    assert idle["ingress_cohort"] == 0
    final = {
        "execution": {
            "varlen_dense_mlp": {
                "counters": {
                    "mlp_compaction_calls": 9,
                    "padding_token_rows": 105,
                }
            }
        },
        "scheduler": {
            "external_batched_prefill_rounds": 4,
            "external_batched_prefill_lanes": 8,
            "external_pairwise_selection_groups": 5,
        },
        "counts": {
            "ingress_cohort_observed_used": 3,
            "ingress_cohort_target_reached": 2,
        },
    }
    observed = feature_observations(final, initial=initial)
    assert observed["varlen_dense_mlp"] == 2
    assert observed["external_varlen_prefill"] == 1
    assert observed["ingress_cohort"] == 1
    partial = {
        **final,
        "execution": {
            "varlen_dense_mlp": {
                "counters": {
                    "mlp_compaction_calls": 9,
                    "padding_token_rows": 100,
                }
            }
        },
        "scheduler": {
            "external_batched_prefill_rounds": 4,
            "external_batched_prefill_lanes": 6,
        },
    }
    partial_observed = feature_observations(partial, initial=initial)
    assert partial_observed["varlen_dense_mlp"] == 0
    assert partial_observed["external_varlen_prefill"] == 0

    # A physical B2 packed slab proves use, but not a target_lanes=4 ingress
    # policy. Only the coalescer's target-reached counter can satisfy the gate.
    b2_only = {
        **final,
        "counts": {
            "ingress_cohort_observed_used": 3,
            "ingress_cohort_executed_lanes": 2,
            "ingress_cohort_max_execution_width": 2,
            "ingress_cohort_target_reached": 1,
        },
    }
    assert feature_observations(b2_only, initial=initial)["ingress_cohort"] == 0

    sparse = {
        "prefill_execution": {
            "varlen": {"schema": "mlx2.varlen-sparse-moe.v1"}
        }
    }
    assert "feature_varlen_sparse_moe" in required_feature_checks(sparse)
    sparse_initial = {
        "execution": {
            "varlen_sparse_moe": {
                "counters": {
                    "moe_compaction_calls": 10,
                    "padding_token_rows": 80,
                }
            }
        }
    }
    sparse_final = {
        "execution": {
            "varlen_sparse_moe": {
                "counters": {
                    "moe_compaction_calls": 13,
                    "padding_token_rows": 84,
                }
            }
        }
    }
    assert feature_observations(
        sparse_final, initial=sparse_initial
    )["varlen_sparse_moe"] == 3


def test_external_adaptive_verification_is_serialized_but_unqualifiable():
    from mlx2.qualification import unqualifiable_candidate

    assert unqualifiable_candidate({}) is None
    reason = unqualifiable_candidate(
        {
            "execution_policy": {
                "adaptive_verification": {
                    "verification_costs": [1.0, 2.0, 4.0]
                }
            }
        }
    )
    assert "adaptive_verification" in reason
    assert "cannot observe" in reason


def test_progressive_verification_requires_run_local_spanning_observation():
    from mlx2.qualification import required_feature_checks, unqualifiable_candidate
    from scripts.qualify_serving import feature_observations

    settings = {
        "speculation": "external_draft",
        "execution_policy": {"progressive_verification_tile": 3},
    }
    assert unqualifiable_candidate(settings) is None
    assert "feature_progressive_verification" in required_feature_checks(settings)

    initial = {
        "settings": settings,
        "scheduler": {
            "external_progressive_verify_tile": 3,
            "external_progressive_verify_rounds": 7,
            "external_progressive_verify_launches": 13,
            "external_progressive_verify_target_rows": 35,
            "external_progressive_verify_full_tiles": 6,
        },
    }
    final = {
        "settings": settings,
        "scheduler": {
            "external_progressive_verify_tile": 3,
            "external_progressive_verify_rounds": 8,
            "external_progressive_verify_launches": 15,
            "external_progressive_verify_target_rows": 41,
            "external_progressive_verify_full_tiles": 7,
        },
    }
    assert feature_observations(final, initial=initial)[
        "progressive_verification"
    ] == 1

    idle = {**final, "scheduler": dict(initial["scheduler"])}
    assert feature_observations(idle, initial=initial)[
        "progressive_verification"
    ] == 0
    no_span = {**final, "scheduler": {
        **final["scheduler"],
        "external_progressive_verify_full_tiles": 6,
    }}
    assert feature_observations(no_span, initial=initial)[
        "progressive_verification"
    ] == 0
    one_launch = {**final, "scheduler": {
        **final["scheduler"],
        "external_progressive_verify_launches": 14,
    }}
    assert feature_observations(one_launch, initial=initial)[
        "progressive_verification"
    ] == 0
    changed_tile = {**final, "settings": {
        **settings,
        "execution_policy": {"progressive_verification_tile": 4},
    }}
    assert feature_observations(changed_tile, initial=initial)[
        "progressive_verification"
    ] == 0
    mismatched_status = {**final, "scheduler": {
        **final["scheduler"],
        "external_progressive_verify_tile": 4,
    }}
    assert feature_observations(mismatched_status, initial=initial)[
        "progressive_verification"
    ] == 0


def test_progressive_multilane_cap_requires_run_local_engagement():
    from mlx2.qualification import required_feature_checks
    from scripts.qualify_serving import feature_observations

    settings = {
        "speculation": "external_draft",
        "execution_policy": {
            "progressive_verification_tile": 3,
            "progressive_multilane_draft_cap": 3,
        },
    }
    assert (
        "feature_progressive_multilane_draft_cap"
        in required_feature_checks(settings)
    )
    initial = {
        "settings": settings,
        "scheduler": {
            "external_multilane_draft_cap": 3,
            "external_multilane_draft_cap_rounds": 4,
            "external_multilane_draft_cap_lanes": 8,
        },
    }
    final = {
        "settings": settings,
        "scheduler": {
            "external_multilane_draft_cap": 3,
            "external_multilane_draft_cap_rounds": 5,
            "external_multilane_draft_cap_lanes": 10,
        },
    }
    assert feature_observations(final, initial=initial)[
        "progressive_multilane_draft_cap"
    ] == 1
    singleton_only = {
        **final,
        "scheduler": {
            **final["scheduler"],
            "external_multilane_draft_cap_lanes": 9,
        },
    }
    assert feature_observations(singleton_only, initial=initial)[
        "progressive_multilane_draft_cap"
    ] == 0
    drifted = {
        **final,
        "settings": {
            **settings,
            "execution_policy": {
                **settings["execution_policy"],
                "progressive_multilane_draft_cap": 4,
            },
        },
    }
    assert feature_observations(drifted, initial=initial)[
        "progressive_multilane_draft_cap"
    ] == 0


def test_loader_requires_progressive_verification_feature(tmp_path):
    from dataclasses import replace

    from mlx2.adapters.muse_glimmer import MUSE_GLIMMER
    from mlx2.contracts import Capability
    from mlx2.qualification import required_feature_checks

    descriptor = replace(
        MUSE_GLIMMER,
        capabilities=MUSE_GLIMMER.capabilities | {Capability.EXTERNAL_DRAFT},
    )
    settings = {
        "mtp": False,
        "speculation": "external_draft",
        "max_context": 131072,
        "max_lanes": 1,
        "execution_policy": {"progressive_verification_tile": 3},
    }
    checks = {
        name: {"passed": True}
        for name in (REQUIRED_CHECKS - {"batch", "mixed_warm"})
        | {"structured_output"}
        | required_feature_checks(settings)
    }
    checks.pop("feature_progressive_verification")
    record = {
        "passed": True,
        "runtime": {"source": "abc"},
        "artifact": "weights",
        "settings": settings,
        "qualification_harness": APPROVED_QUALIFICATION_HARNESS,
        "checks": checks,
    }
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="feature_progressive_verification"):
        load_qualified_route(
            path,
            runtime=record["runtime"],
            artifact="weights",
            settings=settings,
            descriptor=descriptor,
            name="muse-progressive",
        )
    checks["feature_progressive_verification"] = {"passed": True}
    path.write_text(json.dumps(record))
    decision = load_qualified_route(
        path,
        runtime=record["runtime"],
        artifact="weights",
        settings=settings,
        descriptor=descriptor,
        name="muse-progressive",
    )
    assert decision.profile.name == "muse-progressive"
    assert Capability.EXTERNAL_DRAFT in decision.profile.capabilities
    assert Capability.CONTINUOUS_BATCH not in decision.profile.capabilities


def test_two_lane_loader_requires_and_selects_continuous_batch(tmp_path):
    from dataclasses import replace

    from mlx2.adapters.muse_glimmer import MUSE_GLIMMER
    from mlx2.contracts import Capability
    from mlx2.qualification import required_feature_checks

    descriptor = replace(
        MUSE_GLIMMER,
        capabilities=MUSE_GLIMMER.capabilities | {Capability.EXTERNAL_DRAFT},
    )
    settings = {
        "mtp": False,
        "speculation": "external_draft",
        "max_context": 4096,
        "max_lanes": 2,
        "execution_policy": {},
    }
    checks = {
        name: {"passed": True}
        for name in REQUIRED_CHECKS
        | {"structured_output"}
        | required_feature_checks(settings)
    }
    checks.pop("batch")
    record = {
        "passed": True,
        "runtime": {"source": "abc"},
        "artifact": "weights",
        "settings": settings,
        "qualification_harness": APPROVED_QUALIFICATION_HARNESS,
        "checks": checks,
    }
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(record))
    args = {
        "runtime": record["runtime"],
        "artifact": "weights",
        "settings": settings,
        "descriptor": descriptor,
        "name": "muse-b2",
    }
    with pytest.raises(ValueError, match="missing or failed"):
        load_qualified_route(path, **args)
    checks["batch"] = {"passed": True}
    path.write_text(json.dumps(record))
    decision = load_qualified_route(path, **args)
    assert Capability.CONTINUOUS_BATCH in decision.profile.capabilities


def test_external_tree_batch_size_route_is_unqualifiable():
    # The artifact-bound default tree route is qualifiable (it needs
    # feature_external_tree); any other tree route stays a candidate.
    from mlx2.qualification import unqualifiable_candidate

    assert unqualifiable_candidate(
        {"execution_policy": {"batch_size_route": "tree15_b1_b4_chain_b5plus_v1"}}
    ) is None
    reason = unqualifiable_candidate(
        {
            "execution_policy": {
                "batch_size_route": "tree15_b1_chain_b2plus_v1"
            }
        }
    )
    assert "batch_size_route" in reason
    assert "route-specific qualification gate" in reason


def test_a_recorded_failed_check_refuses_the_route(tmp_path):
    """The loader re-checked only the required and feature gates and trusted
    the top-level flag for the rest (capability scope, quiescence, APCv2
    reuse/stores, context bound, MTP aggregates)."""
    record = {
        "passed": True, "runtime": {"source": "abc"}, "artifact": "weights",
        "settings": {"mtp": True, "max_context": 16384, "default_max_tokens": 65_536},
        "qualification_harness": APPROVED_QUALIFICATION_HARNESS,
        "checks": {c: {"passed": True}
                   for c in REQUIRED_CHECKS | {"mtp_execution", "structured_output"}},
    }
    args = dict(runtime=record["runtime"], artifact="weights", settings=record["settings"],
                descriptor=QWEN4_FLASH_NEXT, name="failed-gate")
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(record))
    assert load_qualified_route(path, **args)
    record["checks"]["quiescence"] = {"passed": False}
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="failed checks: quiescence"):
        load_qualified_route(path, **args)


def test_the_serving_producer_gates_without_assert():
    """Under python -O a bare assert is skipped: a failed check was recorded,
    the run went on, and the producer wrote passed: true and exited 0."""
    import ast

    source = (Path(__file__).resolve().parents[1] / "scripts" / "qualify_serving.py").read_text()
    assert not [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Assert)]

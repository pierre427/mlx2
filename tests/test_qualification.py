import json
from pathlib import Path
import pytest

from mlx2.adapters.qwen import QWEN4_FLASH_NEXT
from mlx2.qualification import (
    APPROVED_QUALIFICATION_HARNESS,
    REQUIRED_CHECKS,
    load_qualified_route,
    required_descriptor_checks,
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


def test_multimodal_descriptor_requires_adapter_owned_live_checks(tmp_path):
    from mlx2.adapters.mlx_vlm import GEMMA3N

    required = required_descriptor_checks(GEMMA3N)
    assert required == {
        "multimodal_image",
        "multimodal_video",
        "multimodal_audio_input",
        "multimodal_encoder_batching",
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
    with pytest.raises(ValueError, match="missing or failed"):
        load_qualified_route(path, **args)
    record["checks"].update({name: {"passed": True} for name in required})
    path.write_text(json.dumps(record))
    profile = load_qualified_route(path, **args).profile
    assert {"vision", "video", "audio"} <= {
        capability.value for capability in profile.capabilities
    }


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

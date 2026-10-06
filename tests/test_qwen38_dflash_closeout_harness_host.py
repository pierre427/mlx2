from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from mlx_blocker import block_mlx_imports, mlx_module_names

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_checkpoint_layer_summary_locates_exact_and_envelope_failures():
    module = load_script("qualify_qwen38_dflash_checkpoint_parity.py")
    layers = [
        {
            "index": 0,
            "structural_errors": [],
            "array_count": 2,
            "exact_array_count": 2,
            "passed": True,
        },
        {
            "index": 1,
            "structural_errors": [],
            "array_count": 2,
            "exact_array_count": 1,
            "passed": True,
        },
        {
            "index": 2,
            "structural_errors": [],
            "array_count": 2,
            "exact_array_count": 0,
            "passed": False,
        },
    ]
    assert module.layer_divergence_summary(layers) == {
        "first_exact_divergent_layer": 1,
        "first_tolerance_failed_layer": 2,
        "failed_layers": [2],
    }


def test_checkpoint_harness_requires_and_labels_explicit_gdn_reference():
    module = load_script("qualify_qwen38_dflash_checkpoint_parity.py")
    with pytest.raises(ValueError, match="explicit boolean fused_gdn"):
        module.require_explicit_gdn_reference_policy({})
    with pytest.raises(ValueError, match="explicit boolean fused_gdn"):
        module.require_explicit_gdn_reference_policy({"fused_gdn": "yes"})
    assert module.require_explicit_gdn_reference_policy({"fused_gdn": False}) == {
        "kind": "ordinary_unfused_gdn_with_selected_lane_projections",
        "policy_field": "fused_gdn",
        "policy_value": False,
        "ordinary_diagnostic": True,
        "claim": (
            "explicit serial GDN reference under the applied lane projection policy; "
            "not stock-projection parity or TensorFold qualification by itself"
        ),
    }
    selected = module.require_explicit_gdn_reference_policy({"fused_gdn": True})
    assert selected["kind"] == "selected_fused_gdn_with_selected_lane_projections"
    assert selected["ordinary_diagnostic"] is False


def test_checkpoint_harness_replays_cache_inputs_and_retains_numeric_gates():
    source = (
        ROOT / "scripts/qualify_qwen38_dflash_checkpoint_parity.py"
    ).read_text()
    assert "for token in committed_cache_inputs:" in source
    assert "for token in emitted_outputs:" not in source
    assert "capture_layers = tuple(range(len(adapter.model.layers)))" in source
    assert '"deepest_leaf_semantic_continuation"' in source
    assert "relative_l2_max\": 0.02" in source
    assert "normalized_max\": 0.02" in source
    assert '"recurrent": "tolerance"' in source
    assert '"recurrent_atol": 2e-6' in source
    assert '"recurrent_rtol": 2e-5' in source


def test_b4_phase_observation_is_round_local_and_complete():
    module = load_script("qualify_qwen38_dflash_fixed_cohort_abba.py")
    assert "tree_recovery_capture" in module.PHASE_NAMES
    assert "tree_draft_prelaunch" in module.PHASE_NAMES
    before_scheduler = {"external_phase_rounds": 7}
    after_scheduler = {"external_phase_rounds": 10}
    for index, name in enumerate(module.PHASE_NAMES, 1):
        key = f"external_phase_{name}_ns"
        before_scheduler[key] = index * 10
        after_scheduler[key] = index * 10 + index * 300
    observation = module.phase_observation(
        {"scheduler": before_scheduler}, {"scheduler": after_scheduler}
    )
    assert observation["physical_tree_rounds"] == 3
    assert observation["phase_ns"]["tree_target_wait"] > 0
    assert observation["phase_ns_per_tree_round"]["tree_target_wait"] == (
        observation["phase_ns"]["tree_target_wait"] / 3
    )
    combined = module.combine_phase_observations([observation, observation])
    assert combined["physical_tree_rounds"] == 6
    assert combined["phase_ns"]["tree_round"] == (
        2 * observation["phase_ns"]["tree_round"]
    )


def test_b4_phase_observation_fails_closed_on_missing_or_rolled_back_counters():
    module = load_script("qualify_qwen38_dflash_fixed_cohort_abba.py")
    with pytest.raises(RuntimeError, match="did not engage"):
        module.phase_observation(
            {"scheduler": {}},
            {"scheduler": {"external_phase_rounds": 1}},
        )
    with pytest.raises(RuntimeError, match="counter rollback"):
        module.numeric_delta({"external_phase_rounds": 2}, {"external_phase_rounds": 1})


def test_b4_phase_observation_allows_scheduler_gauges_to_decrease():
    module = load_script("qualify_qwen38_dflash_fixed_cohort_abba.py")
    before_scheduler = {
        "external_adaptive_round_depth": 8,
        "external_adaptive_verify_width": 9,
        "external_phase_rounds": 7,
        "external_tree_node_budget_last": 64,
        "reservation_bytes": 5_000_000_000,
    }
    after_scheduler = {
        "external_adaptive_round_depth": 4,
        "external_adaptive_verify_width": 5,
        "external_phase_rounds": 10,
        "external_tree_node_budget_last": 56,
        "reservation_bytes": 450_000_000,
    }
    for index, name in enumerate(module.PHASE_NAMES, 1):
        key = f"external_phase_{name}_ns"
        before_scheduler[key] = index * 10
        after_scheduler[key] = index * 310

    observation = module.phase_observation(
        {"scheduler": before_scheduler}, {"scheduler": after_scheduler}
    )

    assert observation["physical_tree_rounds"] == 3
    delta = module.numeric_delta(
        before_scheduler,
        after_scheduler,
        non_monotonic_gauges=module.NON_MONOTONIC_SCHEDULER_GAUGES,
    )
    assert delta["external_adaptive_round_depth"] == -4
    assert delta["external_adaptive_verify_width"] == -4
    assert delta["external_tree_node_budget_last"] == -8
    assert delta["reservation_bytes"] == -4_550_000_000


def test_scheduler_gauge_allowlist_does_not_mask_true_counter_rollback():
    module = load_script("qualify_qwen38_dflash_fixed_cohort_abba.py")
    with pytest.raises(RuntimeError, match="external_phase_rounds"):
        module.numeric_delta(
            {"external_phase_rounds": 2, "reservation_bytes": 5_000},
            {"external_phase_rounds": 1, "reservation_bytes": 4_000},
            non_monotonic_gauges=module.NON_MONOTONIC_SCHEDULER_GAUGES,
        )


def test_reverse_20x20_manifest_retains_seeded_stochastic_identity():
    module = load_script("qualify_qwen38_dflash_fixed_cohort_abba.py")
    assert module.ARM_ORDERS["baab"] == (
        "candidate",
        "control",
        "control",
        "candidate",
    )
    first = module.build_manifest(20, 5.0, sampling_profile="mixed-seeded")
    second = module.build_manifest(20, 5.0, sampling_profile="mixed-seeded")
    assert first["body_hashes_sha256"] == second["body_hashes_sha256"]
    assert first["timed_seeded_non_greedy_requests"] == 200
    timed = [row for row in first["rows"] if row["phase"] == "timed"]
    assert len(timed) == 400
    assert sum(row["body"]["temperature"] == 0.7 for row in timed) == 200
    assert len({row["body"]["seed"] for row in timed}) == 400


def test_pld_dflash_policy_pair_is_chain_only_and_one_field_different():
    module = load_script("qualify_qwen38_dflash_fixed_cohort_abba.py")
    control = {
        "draft_model": "/draft",
        "num_draft": 7,
        "pairwise_selection": "batched",
        "varlen_dense_mlp": True,
        "external_varlen_prefill": True,
    }
    candidate = {**control, "proposal_composition": module.PLD_COMPOSITION}
    module.validate_policies(
        control,
        candidate,
        qualification_kind="pld-dflash",
    )
    with pytest.raises(RuntimeError, match="chain route"):
        module.validate_policies(
            {**control, "batch_size_route": "tree15_b1_b4_chain_b5plus_v1"},
            {
                **candidate,
                "batch_size_route": "tree15_b1_b4_chain_b5plus_v1",
            },
            qualification_kind="pld-dflash",
        )
    with pytest.raises(RuntimeError, match="pinned prompt-lookup policy"):
        module.validate_policies(
            control,
            {
                **control,
                "proposal_composition": {**module.PLD_COMPOSITION, "lookback": 128},
            },
            qualification_kind="pld-dflash",
        )


def test_varlen_qualify_preflight_requires_corrected_matched_policy_pair():
    module = load_script("qualify_qwen38_dflash_fixed_cohort_abba.py")
    candidate = json.loads(
        (
            ROOT
            / "qualification/policies/qwen38-27b-dflash2-varlen-tensorfold.json"
        ).read_text()
    )
    control = {
        **candidate,
        "varlen_dense_mlp": {
            **candidate["varlen_dense_mlp"],
            "minimum_padding_fraction": 1.0,
        },
    }
    module.validate_policies(control, candidate, qualification_kind="varlen")

    with pytest.raises(RuntimeError, match="corrected p25"):
        module.validate_policies(
            control,
            {**candidate, "varlen_dense_mlp": True},
            qualification_kind="varlen",
        )
    with pytest.raises(RuntimeError, match="100%-padding"):
        module.validate_policies(
            {
                **control,
                "varlen_dense_mlp": {
                    **control["varlen_dense_mlp"],
                    "minimum_padding_fraction": 0.5,
                },
            },
            candidate,
            qualification_kind="varlen",
        )
    with pytest.raises(RuntimeError, match="tree_node_budget_by_lanes"):
        stale_budgets = {"1": 15, "2": 15, "3": 15, "4": 15}
        module.validate_policies(
            {**control, "tree_node_budget_by_lanes": stale_budgets},
            {**candidate, "tree_node_budget_by_lanes": stale_budgets},
            qualification_kind="varlen",
        )
    with pytest.raises(RuntimeError, match="differ outside varlen"):
        module.validate_policies(
            control,
            {**candidate, "target_revision": "f" * 64},
            qualification_kind="varlen",
        )


def test_pld_dflash_observation_requires_both_committed_sources():
    module = load_script("qualify_qwen38_dflash_fixed_cohort_abba.py")

    def row(prompt_lookup_accepted=3):
        return {
            "receipt": {
                "speculation": {
                    "proposal_composition": {
                        "selected": True,
                        "qualified": False,
                        "observed_used": True,
                        "committed_verification": {
                            "prompt_lookup": {
                                "verified_rounds": 2,
                                "proposed_tokens": 8,
                                "accepted_tokens": prompt_lookup_accepted,
                            },
                            "external": {
                                "verified_rounds": 4,
                                "proposed_tokens": 20,
                                "accepted_tokens": 12,
                            },
                        },
                    }
                }
            }
        }

    observed = module.proposal_composition_observation([row(), row()], required=True)
    assert observed["receipt_source"] == "mlx2.speculation.proposal_composition"
    assert observed["engaged"] == {"prompt_lookup": True, "external": True}
    assert observed["committed_verification"]["prompt_lookup"]["accepted_tokens"] == 6
    with pytest.raises(RuntimeError, match="both sources"):
        module.proposal_composition_observation(
            [row(prompt_lookup_accepted=0)], required=True
        )
    with pytest.raises(RuntimeError, match="control unexpectedly"):
        module.proposal_composition_observation([row()], required=False)


def test_pld_observation_rejects_legacy_top_level_composition_receipt():
    module = load_script("qualify_qwen38_dflash_fixed_cohort_abba.py")
    row = {
        "receipt": {
            "proposal_composition": {
                "selected": True,
                "qualified": False,
                "observed_used": True,
            }
        }
    }
    with pytest.raises(RuntimeError, match="selected on 0/1 responses"):
        module.proposal_composition_observation([row], required=True)


def test_completed_arm_is_persisted_before_mechanism_failure(tmp_path):
    module = load_script("qualify_qwen38_dflash_fixed_cohort_abba.py")
    path = tmp_path / "candidate.json"
    result = {
        "arm": "candidate",
        "results": [
            {
                "phase": "timed",
                "receipt": {"speculation": {}},
                "output_sha256": "completed-request",
            }
        ],
        "mechanism_validation": {"status": "pending"},
    }
    record = module.write_arm_result(path, result)
    assert record["report_sha256"] == module.sha256(path)
    with pytest.raises(RuntimeError, match="selected on 0/1 responses"):
        module.validate_persisted_arm_mechanism(
            result,
            path,
            qualification_kind="pld-dflash",
            arm="candidate",
        )
    persisted = json.loads(path.read_text())
    assert persisted["results"][0]["output_sha256"] == "completed-request"
    assert persisted["mechanism_validation"]["status"] == "failed"
    assert "selected on 0/1 responses" in persisted["mechanism_validation"]["error"]


def test_pld_policy_manifest_binds_inputs_not_runtime_derived_artifact(tmp_path):
    module = load_script("qualify_qwen38_dflash_fixed_cohort_abba.py")
    target_revision = "1" * 64
    draft_revision = "2" * 64
    runtime_draft = "3" * 64
    target_artifact = "4" * 64
    paths = {}
    records = []
    for arm in ("control", "candidate"):
        path = tmp_path / f"pld-{arm}-policy.json"
        policy = {
            "target_revision": target_revision,
            "draft_revision": draft_revision,
        }
        path.write_text(json.dumps(policy))
        paths[arm] = path
        records.append(
            {
                "role": f"pld-{arm}",
                "generated_path": str(path),
                "generated_sha256": module.sha256(path),
                "adapter_inspection": {
                    "target_revision": target_revision,
                    "draft_revision": draft_revision,
                    "runtime_draft_fingerprint": runtime_draft,
                },
            }
        )
    manifest_path = tmp_path / "policy-derivation-manifest.json"
    modules = {
        name: {"path": str(path), "sha256": module.sha256(path)}
        for name, path in {
            "mlx2.adapters.dflash2": ROOT / "src/mlx2/adapters/dflash2.py",
            "mlx2.adapters.qwen38_27b": ROOT / "src/mlx2/adapters/qwen38_27b.py",
            "mlx2.adapters.qwen38_tensorfold_source": (
                ROOT / "src/mlx2/adapters/qwen38_tensorfold_source.py"
            ),
        }.items()
    }
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "mlx2.qwen38-dflash-closeout-policy-derivation.v1",
                "source": {
                    "commit": "a" * 40,
                    "root": str(ROOT),
                    "modules": modules,
                },
                "real_adapter_inspection": {
                    "mlx_import_blocked": True,
                    "mlx_modules_after": [],
                },
                "target": {
                    "artifact_fingerprint": target_artifact,
                    "payload_revision": target_revision,
                },
                "draft": {"payload_revision": draft_revision},
                "policies": records,
            }
        )
    )
    binding = module.validate_policy_manifest(
        manifest_path,
        expected_source="a" * 40,
        model={"artifact_fingerprint": target_artifact},
        policies=paths,
        role_prefix="pld",
    )
    assert binding["target_artifact_fingerprint"] == target_artifact
    assert binding["policies"]["candidate"]["runtime_draft_fingerprint"] == (
        runtime_draft
    )
    modules["mlx2.adapters.dflash2"]["sha256"] = "5" * 64
    manifest_path.write_text(
        json.dumps(json.loads(manifest_path.read_text()) | {
            "source": {
                "commit": "a" * 40,
                "root": str(ROOT),
                "modules": modules,
            }
        })
    )
    with pytest.raises(RuntimeError, match="dflash2 digest differs"):
        module.validate_policy_manifest(
            manifest_path,
            expected_source="a" * 40,
            model={"artifact_fingerprint": target_artifact},
            policies=paths,
            role_prefix="pld",
        )
    modules["mlx2.adapters.dflash2"]["sha256"] = module.sha256(
        ROOT / "src/mlx2/adapters/dflash2.py"
    )
    records[1]["generated_sha256"] = "5" * 64
    manifest = json.loads(manifest_path.read_text())
    manifest["source"]["modules"] = modules
    manifest["policies"] = records
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="digest differs"):
        module.validate_policy_manifest(
            manifest_path,
            expected_source="a" * 40,
            model={"artifact_fingerprint": target_artifact},
            policies=paths,
            role_prefix="pld",
        )


def test_varlen_policy_manifest_binds_corrected_tensorfold_source(tmp_path):
    module = load_script("qualify_qwen38_dflash_fixed_cohort_abba.py")
    target_revision = "1" * 64
    draft_revision = "2" * 64
    runtime_draft = "3" * 64
    target_artifact = "4" * 64
    tensorfold = {
        "root": "/source/tensorfold",
        "revision": "5" * 40,
        "tree": "6" * 40,
        "tracked_source": "src/tensorfold",
        "tracked_diff": "",
        "module_sha256": {"src/tensorfold/example.py": "7" * 64},
    }
    paths = {}
    records = []
    for arm in ("control", "candidate"):
        path = tmp_path / f"abba-{arm}-policy.json"
        path.write_text(
            json.dumps(
                {
                    "target_revision": target_revision,
                    "draft_revision": draft_revision,
                }
            )
        )
        paths[arm] = path
        records.append(
            {
                "role": f"abba-{arm}",
                "generated_path": str(path),
                "generated_sha256": module.sha256(path),
                "adapter_inspection": {
                    "target_revision": target_revision,
                    "draft_revision": draft_revision,
                    "runtime_draft_fingerprint": runtime_draft,
                },
            }
        )
    modules = {
        name: {"path": str(path), "sha256": module.sha256(path)}
        for name, path in {
            "mlx2.adapters.dflash2": ROOT / "src/mlx2/adapters/dflash2.py",
            "mlx2.adapters.qwen38_27b": ROOT / "src/mlx2/adapters/qwen38_27b.py",
            "mlx2.adapters.qwen38_tensorfold_source": (
                ROOT / "src/mlx2/adapters/qwen38_tensorfold_source.py"
            ),
        }.items()
    }
    manifest_path = tmp_path / "policy-derivation-manifest.json"
    manifest = {
        "schema": "mlx2.qwen38-dflash-closeout-policy-derivation.v1",
        "source": {
            "commit": "a" * 40,
            "root": str(ROOT),
            "modules": modules,
        },
        "real_adapter_inspection": {
            "mlx_import_blocked": True,
            "mlx_modules_after": [],
        },
        "target": {
            "artifact_fingerprint": target_artifact,
            "payload_revision": target_revision,
        },
        "draft": {"payload_revision": draft_revision},
        "tensorfold": tensorfold,
        "policies": records,
    }
    manifest_path.write_text(json.dumps(manifest))
    binding = module.validate_policy_manifest(
        manifest_path,
        expected_source="a" * 40,
        model={"artifact_fingerprint": target_artifact},
        policies=paths,
        role_prefix="abba",
        tensorfold=tensorfold,
    )
    assert binding["tensorfold"] == tensorfold
    with pytest.raises(RuntimeError, match="TensorFold source binding differs"):
        module.validate_policy_manifest(
            manifest_path,
            expected_source="a" * 40,
            model={"artifact_fingerprint": target_artifact},
            policies=paths,
            role_prefix="abba",
            tensorfold={**tensorfold, "tree": "8" * 40},
        )


def test_varlen_runtime_artifact_defaults_are_not_historical_constants():
    module = load_script("qualify_qwen38_dflash_fixed_cohort_abba.py")
    assert not hasattr(module, "EXPECTED_ARM_ARTIFACTS")
    source = (ROOT / "scripts/qualify_qwen38_dflash_fixed_cohort_abba.py").read_text()
    assert "expected_arm_artifacts = dict(explicit_artifacts)" in source


def test_cross_arm_requires_stable_distinct_runtime_artifacts():
    module = load_script("qualify_qwen38_dflash_fixed_cohort_abba.py")
    row = {
        "phase": "timed",
        "round": 0,
        "request": 0,
        "body_sha256": "body",
        "output_sha256": "output",
    }
    manifest = {"rows": [row]}

    def arm(name, artifact):
        return {
            "arm": name,
            "occurrence": 1,
            "results": [dict(row)],
            "timed_wall_seconds": 1.0,
            "server_identity": {"artifact": artifact},
        }

    arms = [
        arm("control", "1" * 64),
        arm("candidate", "2" * 64),
        arm("candidate", "2" * 64),
        arm("control", "1" * 64),
    ]
    summary = module.validate_cross_arm(arms, manifest)
    assert summary["served_artifacts_stable_within_arm"] is True
    assert summary["served_artifacts_distinct_across_policies"] is True
    arms[2]["server_identity"]["artifact"] = "3" * 64
    with pytest.raises(RuntimeError, match="drifted within an arm"):
        module.validate_cross_arm(arms, manifest)
    arms[2]["server_identity"]["artifact"] = "1" * 64
    arms[1]["server_identity"]["artifact"] = "1" * 64
    with pytest.raises(RuntimeError, match="same served artifact"):
        module.validate_cross_arm(arms, manifest)


def test_cross_arm_records_seeded_policy_divergence_but_requires_repeatability():
    module = load_script("qualify_qwen38_dflash_fixed_cohort_abba.py")
    manifest = {
        "rows": [{
            "phase": "timed",
            "round": 0,
            "request": 1,
            "body_sha256": "sampled-body",
            "body": {"temperature": 0.7, "seed": 610001},
        }]
    }

    def arm(name, occurrence, output):
        return {
            "arm": name,
            "occurrence": occurrence,
            "results": [{
                "phase": "timed",
                "round": 0,
                "request": 1,
                "body_sha256": "sampled-body",
                "output_sha256": output,
            }],
            "timed_wall_seconds": 1.0,
            "server_identity": {
                "artifact": ("1" if name == "control" else "2") * 64
            },
        }

    arms = [
        arm("control", 1, "control-output"),
        arm("candidate", 2, "candidate-output"),
        arm("candidate", 3, "candidate-output"),
        arm("control", 4, "control-output"),
    ]
    summary = module.validate_cross_arm(arms, manifest)
    assert summary["canonical_outputs_identical"] is False
    assert summary["sampled_outputs_repeatable_within_policy"] is True
    assert summary["sampled_cross_policy_divergence_count"] == 1
    assert summary["sampled_cross_policy_divergences"][0]["seed"] == 610001

    arms[2]["results"][0]["output_sha256"] = "candidate-repeat-drift"
    with pytest.raises(RuntimeError, match="sampled output drift within candidate"):
        module.validate_cross_arm(arms, manifest)


def test_cross_arm_still_fails_on_greedy_policy_divergence():
    module = load_script("qualify_qwen38_dflash_fixed_cohort_abba.py")
    row = {
        "phase": "timed",
        "round": 0,
        "request": 0,
        "body_sha256": "greedy-body",
        "body": {"temperature": 0},
    }
    manifest = {"rows": [row]}

    def arm(name, occurrence, output):
        return {
            "arm": name,
            "occurrence": occurrence,
            "results": [{**row, "output_sha256": output}],
            "timed_wall_seconds": 1.0,
            "server_identity": {
                "artifact": ("1" if name == "control" else "2") * 64
            },
        }

    arms = [
        arm("control", 1, "control-output"),
        arm("candidate", 2, "candidate-output"),
        arm("candidate", 3, "candidate-output"),
        arm("control", 4, "control-output"),
    ]
    with pytest.raises(RuntimeError, match="greedy output drift"):
        module.validate_cross_arm(arms, manifest)


def test_receipt_schema_regression_keeps_drafter_identity_fields():
    source = (ROOT / "tests/test_external_dflash2_cpu.py").read_text()
    for field in (
        "trained_block_size",
        "runtime_block_size",
        "target_layer_ids",
        "proposal_distribution",
        "tree_ranking",
        "tree_max_nodes",
    ):
        assert f'settings["{field}"]' in source


def test_checkpoint_summary_locates_first_round_layer_and_step():
    module = load_script("summarize_qwen38_dflash_checkpoint_parity.py")
    result = {
        "schema": "old",
        "source_commit": "abc",
        "prompt_tokens": 8192,
        "passed": False,
        "rounds": [
            {
                "round": 4,
                "span": 16,
                "accepted_drafts": 0,
                "continuation_category": "zero_accept",
                "equal": False,
                "logical_state": {"failed_layers": [3, 4]},
                "transaction_commit_oracle": {"failed_layers": []},
            }
        ],
        "continuation_categories": {
            "zero_accept": {
                "thresholds": {"logit_relative_l2_max": 0.02},
                "steps": [
                    {
                        "step": 0,
                        "passed": False,
                        "argmax_equal": True,
                        "logits": {"relative_l2": 0.03},
                    }
                ],
            }
        },
        "planted_deepest_branch": {
            "draft_depth": 3,
            "maximum_draft_depth": 3,
            "transaction_commit_oracle": {"passed": True},
            "continuation": {"steps": []},
        },
    }
    summary = module.summarize(result)
    assert summary["first_failed_round"]["first_tolerance_failed_layer"] == 3
    assert summary["first_failed_round"]["transaction_oracle_failed_layers"] == []
    assert summary["continuation_categories"]["zero_accept"] == {
        "step": 0,
        "reasons": ["logit_relative_l2"],
        "argmax_equal": True,
        "hidden_first_exact_divergent_layer": None,
        "hidden_first_tolerance_failed_layer": None,
        "cache_first_exact_divergent_layer": None,
        "cache_first_tolerance_failed_layer": None,
    }
    assert summary["planted_deepest_branch"]["structure_passed"] is True


def test_closeout_policy_derivation_changes_only_revision_value_bytes():
    module = load_script("prepare_qwen38_dflash_closeout_policies.py")
    source = {
        "draft_model": "/draft",
        "num_draft": 7,
        "draft_revision": "a" * 64,
        "target_revision": "b" * 64,
        "varlen_dense_mlp": True,
    }
    source_raw = json.dumps(source, indent=4).encode() + b"\n"
    generated_raw, proof = module.derive_policy_bytes(
        source_raw,
        draft_revision="c" * 64,
        target_revision="d" * 64,
        label="stage1-parity",
    )
    generated = json.loads(generated_raw)
    assert generated == {
        **source,
        "draft_revision": "c" * 64,
        "target_revision": "d" * 64,
    }
    assert proof["changed_fields"] == ["draft_revision", "target_revision"]
    assert proof["canonical_non_identity_equal"] is True
    assert proof["normalized_non_identity_bytes_equal"] is True
    assert len(generated_raw) == len(source_raw)
    assert module.normalized_policy_bytes(generated_raw) == (
        module.normalized_policy_bytes(source_raw)
    )
    changed_lines = [
        line
        for line in proof["unified_diff"].splitlines()
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    ]
    assert len(changed_lines) == 4
    assert all("revision" in line for line in changed_lines)


def test_closeout_policy_derives_pld_control_by_removing_only_composition():
    module = load_script("prepare_qwen38_dflash_closeout_policies.py")
    candidate = {
        "draft_model": "/draft",
        "num_draft": 7,
        "draft_revision": "a" * 64,
        "target_revision": "b" * 64,
        "varlen_dense_mlp": True,
        "external_varlen_prefill": True,
        "proposal_composition": {
            "prompt_lookup": True,
            "ngram_min": 3,
            "ngram_max": 6,
            "lookback": 4096,
            "native_mtp": False,
            "mtp_max_history": 4096,
        },
    }
    raw = json.dumps(candidate, indent=2).encode() + b"\n"
    control = json.loads(module.pld_control_source_bytes(raw))
    assert control == {
        key: value for key, value in candidate.items() if key != "proposal_composition"
    }
    with pytest.raises(RuntimeError, match="unexpected composition"):
        module.pld_control_source_bytes(
            json.dumps(
                {**candidate, "proposal_composition": {"prompt_lookup": True}}
            ).encode()
        )


def test_closeout_policy_derives_matched_p25_varlen_control():
    module = load_script("prepare_qwen38_dflash_closeout_policies.py")
    candidate_path = (
        ROOT / "qualification/policies/qwen38-27b-dflash2-varlen-tensorfold.json"
    )
    candidate = json.loads(candidate_path.read_bytes())
    control = json.loads(module.abba_control_source_bytes(candidate_path.read_bytes()))
    assert candidate["varlen_dense_mlp"] == {
        "enabled": True,
        "minimum_padding_fraction": 0.25,
        "minimum_padding_rows": 1,
    }
    assert candidate["tree_node_budget_by_lanes"] == {
        "1": 15,
        "2": 7,
        "3": 4,
        "4": 3,
    }
    assert control == {
        **candidate,
        "varlen_dense_mlp": {
            **candidate["varlen_dense_mlp"],
            "minimum_padding_fraction": 1.0,
        },
    }
    with pytest.raises(RuntimeError, match="corrected p25 policy"):
        module.abba_control_source_bytes(
            json.dumps({**candidate, "varlen_dense_mlp": True}).encode()
        )
    with pytest.raises(RuntimeError, match="15/7/4/3"):
        module.abba_control_source_bytes(
            json.dumps(
                {
                    **candidate,
                    "tree_node_budget_by_lanes": {
                        "1": 15,
                        "2": 15,
                        "3": 15,
                        "4": 15,
                    },
                }
            ).encode()
        )


def test_closeout_real_p25_fixture_retains_required_legacy_identity_pins():
    module = load_script("prepare_qwen38_dflash_closeout_policies.py")
    canonical_path = (
        ROOT / "qualification/policies/qwen38-27b-dflash2-varlen-tensorfold.json"
    )
    identity_template_path = (
        ROOT
        / "qualification/runs/dflash-varlen-tensorfold-20x20-r2-20261005"
        / "policy-candidate.json"
    )
    canonical = json.loads(canonical_path.read_bytes())
    identity_template = json.loads(identity_template_path.read_bytes())
    transplanted = json.loads(
        module.abba_candidate_source_bytes(
            canonical_path.read_bytes(), identity_template_path.read_bytes()
        )
    )
    assert module.without_identity(transplanted) == module.without_identity(canonical)
    assert {
        field: transplanted[field] for field in module.IDENTITY_FIELDS
    } == {
        field: identity_template[field] for field in module.IDENTITY_FIELDS
    }
    assert transplanted["varlen_dense_mlp"]["minimum_padding_fraction"] == 0.25
    assert transplanted["tree_node_budget_by_lanes"] == {
        "1": 15,
        "2": 7,
        "3": 4,
        "4": 3,
    }
    control = json.loads(module.abba_control_source_bytes(json.dumps(transplanted).encode()))
    assert module.without_identity(control) == {
        **module.without_identity(canonical),
        "varlen_dense_mlp": {
            **canonical["varlen_dense_mlp"],
            "minimum_padding_fraction": 1.0,
        },
    }


def test_closeout_source_policy_hashes_and_harnesses_require_derived_policy():
    module = load_script("prepare_qwen38_dflash_closeout_policies.py")
    expected = {
        "stage1-parity": ROOT
        / "qualification/runs/dflash-varlen-context-ladder-20261005/policy.json",
        "abba-candidate": ROOT
        / "qualification/policies/qwen38-27b-dflash2-varlen-tensorfold.json",
    }
    for label, path in expected.items():
        assert module.sha256(path) == module.EXPECTED_SOURCE_POLICY_SHA256[label]
    assert module.sha256(module.DEFAULT_ABBA_IDENTITY_TEMPLATE) == (
        module.EXPECTED_SOURCE_POLICY_SHA256["abba-identity-template"]
    )
    stage1_raw = expected["stage1-parity"].read_bytes()
    stage0_raw = module.stage0_source_bytes(stage1_raw)
    assert module.sha256_bytes(stage0_raw) == (
        module.EXPECTED_SOURCE_POLICY_SHA256["stage0-b2"]
    )
    assert json.loads(stage0_raw)["tensorfold_cohort_limit"] == 2
    candidate_raw = expected["abba-candidate"].read_bytes()
    control = json.loads(module.abba_control_source_bytes(candidate_raw))
    candidate = json.loads(candidate_raw)
    assert control["tree_node_budget_by_lanes"] == candidate[
        "tree_node_budget_by_lanes"
    ]
    assert control["varlen_dense_mlp"] == {
        **candidate["varlen_dense_mlp"],
        "minimum_padding_fraction": 1.0,
    }
    for script in (
        "qualify_qwen38_dflash_current_head_smoke.py",
        "qualify_qwen38_dflash_checkpoint_parity.py",
    ):
        source = (ROOT / "scripts" / script).read_text()
        assert 'parser.add_argument("--policy", type=Path, required=True)' in source


def test_tree_policy_ordinary_inspection_is_host_only(monkeypatch, tmp_path):
    block_mlx_imports(monkeypatch, __name__)
    from mlx2.adapters import dflash2, qwen38_27b, qwen38_tensorfold_source

    checked = []
    monkeypatch.setattr(
        qwen38_tensorfold_source,
        "validate_source",
        lambda: checked.append("vendored") or {"revision": "pinned"},
    )
    monkeypatch.setattr(qwen38_27b, "_legacy_content_revision", lambda _path: "1" * 64)
    monkeypatch.setattr(
        qwen38_27b,
        "_inspect_target_content",
        lambda path: {
            "path": str(path),
            "revision": "2" * 64,
            "source_bindings": [],
        },
    )
    record = {
        "path": str(tmp_path / "draft"),
        "fingerprint": "5" * 64,
        "files": [],
        "header_sha256": [],
        "weight_sha256": [],
        "config_sha256": "6" * 64,
        "index_sha256": None,
        "source_bindings": [],
        "config": {},
        "args": SimpleNamespace(block_size=8),
    }
    monkeypatch.setattr(dflash2, "inspect_drafter", lambda *_args: record)
    monkeypatch.setattr(dflash2, "content_revision", lambda _record: "3" * 64)
    monkeypatch.setattr(
        dflash2, "_legacy_content_revision", lambda _record: "4" * 64
    )
    monkeypatch.setattr(dflash2, "validate_runtime_quantization", lambda value: value)
    policy = {
        "draft_model": str(tmp_path / "draft"),
        "draft_revision": "3" * 64,
        "target_revision": "2" * 64,
        "num_draft": 7,
        "batch_size_route": "tree15_b1_b4_chain_b5plus_v1",
    }
    inspected = qwen38_27b.inspect_external_policy(policy, tmp_path / "target")
    assert inspected["target_revision"] == "2" * 64
    assert inspected["draft_revision"] == "3" * 64
    assert checked == ["vendored"]
    assert mlx_module_names() == []


def test_tensorfold_validation_helper_rejects_changed_vendored_module(
    monkeypatch, tmp_path
):
    block_mlx_imports(monkeypatch, __name__)
    from mlx2.adapters import qwen38_tensorfold_source

    module = tmp_path / "module.py"
    module.write_text("changed = True\n")
    monkeypatch.setattr(qwen38_tensorfold_source, "VENDORED_ROOT", tmp_path)
    monkeypatch.setattr(
        qwen38_tensorfold_source,
        "QUALIFICATION_MODULE_SHA256",
        {"module.py": "0" * 64},
    )
    with pytest.raises(RuntimeError, match="vendored TensorFold module mismatch"):
        qwen38_tensorfold_source.validate_source()
    runtime_source = (ROOT / "src/mlx2/runtime/qwen38_tensorfold.py").read_text()
    assert "validate_source(root)" in runtime_source
    assert "import subprocess" not in runtime_source
    assert mlx_module_names() == []


def test_tensorfold_validation_refuses_an_external_source_root(monkeypatch, tmp_path):
    block_mlx_imports(monkeypatch, __name__)
    from mlx2.adapters import qwen38_tensorfold_source

    with pytest.raises(RuntimeError, match="must be the vendored mlx2 package"):
        qwen38_tensorfold_source.validate_source(tmp_path)
    assert mlx_module_names() == []


def test_tensorfold_qualification_identity_binds_tree_and_module_hashes(
    monkeypatch, tmp_path
):
    block_mlx_imports(monkeypatch, __name__)
    from mlx2.adapters import qwen38_tensorfold_source

    module_path = tmp_path / "example.py"
    module_path.write_text("bound = True\n")
    digest = hashlib.sha256(module_path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        qwen38_tensorfold_source,
        "VENDORED_ROOT",
        tmp_path,
    )
    monkeypatch.setattr(qwen38_tensorfold_source, "EXPECTED_TREE", "a" * 40)
    monkeypatch.setattr(
        qwen38_tensorfold_source,
        "QUALIFICATION_MODULE_SHA256",
        {"example.py": digest},
    )
    identity = qwen38_tensorfold_source.qualification_source_identity()
    assert identity["tree"] == "a" * 40
    assert identity["module_sha256"] == {"example.py": digest}
    module_path.write_text("bound = False\n")
    with pytest.raises(RuntimeError, match="vendored TensorFold module mismatch"):
        qwen38_tensorfold_source.qualification_source_identity()
    assert mlx_module_names() == []


def test_policy_tool_binds_imports_to_its_own_worktree(monkeypatch, tmp_path):
    module = load_script("prepare_qwen38_dflash_closeout_policies.py")
    assert module.SOURCE_ROOT == (ROOT / "src").resolve()
    assert str(module.SOURCE_ROOT) in sys.path
    fake = SimpleNamespace(__file__=str(tmp_path / "mlx2/adapters/fake.py"))
    monkeypatch.setitem(sys.modules, "mlx2.adapters.fake", fake)
    with pytest.raises(RuntimeError, match="not source-bound"):
        module.require_local_modules(("mlx2.adapters.fake",))


def test_abba_preflight_rejects_untracked_importable_sources():
    source = (ROOT / "scripts/qualify_qwen38_dflash_fixed_cohort_abba.py").read_text()
    assert '"--untracked-files=all", "--", "src", "scripts"' in source
    assert "qualification_source_identity" in source

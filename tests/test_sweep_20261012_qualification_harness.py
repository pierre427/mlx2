"""Sweep 2026-10-12 G3: qualification-harness regressions (CPU only)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location(
        f"_sweep1012_{name}", ROOT / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(ROOT / "scripts"))
    return module


# L8-2 / L5-02 -----------------------------------------------------------
# e3c19168e renamed the scheduler counters external_tensorfold_* to
# external_target_tree_*; /v1/status exposes the new names raw.

NEW, OLD = "external_target_tree_", "external_tensorfold_"


def _tree_status(prefix, *, tree_rounds, target_rounds):
    return {"scheduler": {
        "external_tree_rounds": tree_rounds,
        prefix + "target_rounds": target_rounds,
    }}


@pytest.mark.parametrize("prefix", (NEW, OLD))
def test_external_tree_is_observed_under_runtime_and_historical_names(prefix):
    observe = _script("qualify_serving").feature_observations
    initial = _tree_status(prefix, tree_rounds=0, target_rounds=0)
    final = _tree_status(prefix, tree_rounds=12, target_rounds=12)
    assert observe(final, initial=initial)["external_tree"] == 12
    assert observe(initial, initial=initial)["external_tree"] == 0


def test_runtime_emits_only_the_target_tree_scheduler_names():
    source = (ROOT / "src/mlx2/runtime/external_speculative.py").read_text()
    assert NEW + "target_rounds" in source
    assert OLD + "target_rounds" not in source


def _tree_scheduler(prefix, *, limit, target, rounds, width):
    return {
        prefix + "cohort_limit": limit,
        prefix + "target_rounds": target,
        prefix + "cohort_rounds": rounds,
        prefix + "cohort_max_width": width,
        "draft_fallbacks": 0,
        "recovery_checkpoint_restores": 0,
    }


@pytest.mark.parametrize("prefix", (NEW, OLD))
def test_tensorfold_cohort_arm_reads_runtime_and_historical_names(prefix):
    cohort = _script("qualify_qwen38_tensorfold_cohort")
    row = {
        "prompt": "code", "sampling": "greedy", "rep": 1, "lane": 0,
        "warmup": False, "completion_tokens": 64, "makespan_seconds": 2.0,
        "output_sha256": "same",
        "speculation": {"tensorfold_target": {"cohort_limit": 4}},
    }
    status = {"scheduler": _tree_scheduler(prefix, limit=4, target=3, rounds=2, width=4)}
    assert cohort.validate_arm(4, [dict(row, lane=i) for i in range(4)], status)


@pytest.mark.parametrize("prefix", (NEW, OLD))
def test_b1_discriminator_reads_runtime_and_historical_names(prefix):
    probe = _script("probe_qwen38_tree15_b1_discriminator")

    def status(tree, physical):
        return {"scheduler": {"external_auto_tree_rounds": tree,
                              prefix + "target_rounds": physical,
                              "external_auto_tree_max_width": 1}}

    row = probe.validate_counters(status(0, 0), status(3, 3), True)
    assert row["delta"]["external_tensorfold_target_rounds"] == 3


@pytest.mark.parametrize("prefix", (NEW, OLD))
def test_live_transition_reads_runtime_and_historical_names(prefix):
    gate = _script("probe_qwen38_tree15_live_transition")

    def snap(tree, width):
        return {"scheduler": {
            "external_auto_tree_rounds": tree,
            "external_auto_tree_max_width": width,
            prefix + "target_rounds": tree,
            prefix + "cohort_rounds": tree,
            prefix + "cohort_max_width": width,
        }}

    gate.verify_stage(snap(0, 0), snap(1, 1), 1)
    gate.verify_stage(snap(1, 1), snap(2, 4), 4)


@pytest.mark.parametrize("name", (
    "probe_qwen38_b2_crossover", "qualify_qwen38_dflash_fixed_cohort_abba",
))
@pytest.mark.parametrize("prefix", (NEW, OLD))
def test_cohort_probes_read_runtime_and_historical_names(name, prefix):
    module = _script(name)
    status = {"scheduler": _tree_scheduler(prefix, limit=4, target=3, rounds=2, width=4)}
    counters = module.tree_scheduler_counters(status)
    assert counters["external_tensorfold_cohort_limit"] == 4
    assert counters["external_tensorfold_cohort_rounds"] == 2
    assert counters["external_tensorfold_cohort_max_width"] == 4


# L1-F5 ------------------------------------------------------------------
# The per-lane cost-per-commit handoff (e3c19168e) fires at any width, so a
# route that selects it is not "unreachable" when max_lanes <= max_mtp_width.

COST = {"enabled": True, "ordinary_samples": 4, "speculative_samples": 4,
        "loss_ratio": 0.97, "loss_decisions": 3, "window": 16}


def _handoff_settings(cost=None, *, max_lanes=2):
    handoff = {"enabled": True, "max_mtp_width": 4}
    if cost is not None:
        handoff["cost_per_commit"] = cost
    return {"mtp": True, "speculation": "self_mtp", "max_lanes": max_lanes,
            "environment": {}, "mtp_ordinary_handoff": handoff}


def test_cost_per_commit_handoff_is_required_not_declared_unreachable():
    from mlx2.qualification import (
        _default_on_mechanisms,
        required_feature_checks,
        selected_not_observed_features,
    )

    settings = _handoff_settings(COST)
    assert "mtp_ordinary_handoff" not in _default_on_mechanisms(settings)[1]
    assert "feature_mtp_ordinary_handoff" not in selected_not_observed_features(settings)
    required = required_feature_checks(settings)
    assert "feature_mtp_ordinary_handoff_cost_per_commit" in required
    assert "feature_mtp_ordinary_handoff" in required


@pytest.mark.parametrize("cost", (None, {"enabled": False}))
def test_static_handoff_alone_stays_unreachable_at_or_below_its_width(cost):
    from mlx2.qualification import _default_on_mechanisms, required_feature_checks

    settings = _handoff_settings(cost)
    reason = _default_on_mechanisms(settings)[1]["mtp_ordinary_handoff"]
    assert "can never grow past the handoff width" in reason
    required = required_feature_checks(settings)
    assert "feature_mtp_ordinary_handoff" not in required
    assert "feature_mtp_ordinary_handoff_cost_per_commit" not in required


def test_cost_handoff_engagement_is_the_run_local_counter_delta():
    observe = _script("qualify_serving").feature_observations
    key = "mtp_ordinary_handoff_cost_per_committed_token"
    initial = {"scheduler": {key: 2}}
    assert observe({"scheduler": {key: 5}}, initial=initial)[
        "mtp_ordinary_handoff_cost_per_commit"] == 3
    assert observe(initial, initial=initial)["mtp_ordinary_handoff_cost_per_commit"] == 0


# L6-2 -------------------------------------------------------------------
# 086421f7b made the adaptive MoE pad fall back to the fixed floor rule
# (uncalibrated_{host,runtime,range}_*) off the fitted host/runtime/range.
# Those are the floor's decisions, not the adaptive law's.

def _pad_choices(monkeypatch, policy, *, host=True):
    from types import SimpleNamespace

    import mlx2.runtime.models.switch_layers as SL

    layer = SimpleNamespace(mode="affine", bits=4, group_size=64, num_experts=512,
                            output_dims=640, input_dims=2560)
    monkeypatch.setattr(SL, "_RHS_PAD_POLICY", policy)
    monkeypatch.setattr(SL, "_pad_cost_host", lambda: host)
    monkeypatch.setattr(SL, "_pad_cost_runtime", lambda: True)
    SL.moe_pad_status(reset=True)
    for rows in (160, 400, 1200, 2000):
        SL._record_pad_choice(SL._adaptive_pad(rows, layer)[1])
    return SL.moe_pad_status(reset=True)


def _pad_runs(status):
    final = {"execution": {"moe_pad": status}}
    initial = {"execution": {"moe_pad": {**status, "choices": {}}}}
    return final, initial


def test_uncalibrated_adaptive_pad_is_not_observed_used(monkeypatch):
    from mlx2.qualification import moe_rhs_pad_uncalibrated

    status = _pad_choices(monkeypatch, "adaptive", host=False)
    assert status["choices"] and all(
        reason.startswith("uncalibrated_host_") for reason in status["choices"])
    final, initial = _pad_runs(status)
    observe = _script("qualify_serving").feature_observations
    assert observe(final, initial=initial)["moe_rhs_pad"] == 0
    reason = moe_rhs_pad_uncalibrated(final["execution"], initial["execution"])
    assert "cost model did not run" in reason and "uncalibrated_host_" in reason
    # Identical snapshots: nothing grew this run, nothing to explain.
    assert moe_rhs_pad_uncalibrated(final["execution"], final["execution"]) is None


def test_calibrated_adaptive_pad_counts_only_cost_model_decisions(monkeypatch):
    from mlx2.qualification import moe_rhs_pad_uncalibrated

    status = _pad_choices(monkeypatch, "adaptive", host=True)
    cost = sum(v for k, v in status["choices"].items() if k.startswith("cost_"))
    assert cost > 0
    status["choices"]["mlx_streams"] = 7
    status["choices"]["uncalibrated_range_qmv"] = 5
    final, initial = _pad_runs(status)
    observe = _script("qualify_serving").feature_observations
    assert observe(final, initial=initial)["moe_rhs_pad"] == cost
    assert moe_rhs_pad_uncalibrated(final["execution"], initial["execution"]) is None


def test_always_pad_counts_its_own_decisions(monkeypatch):
    status = _pad_choices(monkeypatch, "always")
    assert status["choices"].get("always_rhs", 0) > 0
    final, initial = _pad_runs(status)
    observe = _script("qualify_serving").feature_observations
    assert observe(final, initial=initial)["moe_rhs_pad"] == status["choices"]["always_rhs"]


# L6-1 -------------------------------------------------------------------
# 086421f7b set Gemma3nVideoPolicy.frame_batch_size == max_frames (64) so one
# default video stays in one encoder batch.  multimodal_encoder_batching
# needs a call split into >1 chunk, which a single default video can never
# produce; it is required only when the policy can split one.

UNSPLIT = {"video_fps": 2.0, "video_max_frames": 64, "video_frame_batch_size": 64,
           "video_frame_batch_splits": False}
SPLIT = {"video_fps": 2.0, "video_max_frames": 64, "video_frame_batch_size": 16,
         "video_frame_batch_splits": True}
GEMMA3N_BASE = {"multimodal_image", "multimodal_video", "multimodal_audio_input",
                "multimodal_continuous_batch", "multimodal_apcv2_reuse"}


def test_gemma3n_video_policy_reports_whether_one_default_video_splits():
    from mlx2.adapters.multimodal import Gemma3nVideoPolicy

    assert Gemma3nVideoPolicy().as_dict() == UNSPLIT
    assert Gemma3nVideoPolicy(frame_batch_size=16).as_dict() == SPLIT
    assert Gemma3nVideoPolicy(max_frames=8, frame_batch_size=8).as_dict()[
        "video_frame_batch_splits"] is False


def test_gemma3n_encoder_check_follows_the_resolved_video_policy():
    from mlx2.adapters.mlx_vlm import GEMMA3N
    from mlx2.qualification import required_descriptor_checks

    assert "multimodal_encoder_batching" not in GEMMA3N.metadata[
        "required_qualification_checks"]
    unsplit = required_descriptor_checks(GEMMA3N, {"adapter_policy": UNSPLIT})
    split = required_descriptor_checks(GEMMA3N, {"adapter_policy": SPLIT})
    assert unsplit == GEMMA3N_BASE | {"multimodal_encoder_unsplit"}
    # Fail closed: a splitting policy requires the batching check again.
    assert split == GEMMA3N_BASE | {"multimodal_encoder_batching"}
    with pytest.raises(ValueError, match="settings are required"):
        required_descriptor_checks(GEMMA3N)
    with pytest.raises(ValueError, match="setting is missing"):
        required_descriptor_checks(GEMMA3N, {"adapter_policy": {}})


def test_gemma3n_adapter_binds_its_video_policy_into_route_identity(monkeypatch):
    from mlx2.adapters import mlx_vlm

    def fake_base_init(self, model_path, *, execution_policy=None):
        self.model = object()
        self.media_feature_cache = None

    monkeypatch.setattr(mlx_vlm._MLXVLMAdapter, "__init__", fake_base_init)
    monkeypatch.setattr(mlx_vlm, "install_gemma3n_vision_batching", lambda *a: None)
    monkeypatch.setattr(mlx_vlm, "install_media_feature_cache", lambda *a, **k: None)
    adapter = mlx_vlm.Gemma3nAdapter("unused")
    # serving.py records adapter.policy.as_dict() as settings["adapter_policy"].
    assert adapter.policy is adapter.video_policy
    assert adapter.policy.as_dict() == UNSPLIT


def _gemma3n_report(policy, observation, *, feature_parity=True):
    from mlx2.media_qualification import NATIVE_VLM_SOURCE_REVISIONS

    revision = NATIVE_VLM_SOURCE_REVISIONS["gemma3n"]
    runtime = {"schema": "mlx2.vlm-dependencies.v1", "family": "gemma3n",
               "source_sha256": "a" * 64, "dependency_files": 1,
               "reference_revision": revision}
    harness = {"name": "native", "sha256": "c" * 64}
    from mlx2.adapters.mlx_vlm import GEMMA3N

    rules = [dict(rule, setting=list(rule["setting"]))
             for rule in GEMMA3N.metadata["conditional_qualification_checks"]]
    parity = {"passed": feature_parity, "shapes_match": True, "tensor_count": 1,
              "max_abs": 0.0}
    report = {
        "model_type": "gemma3n", "qualification_harness": harness,
        "source_revision": revision, "source_runtime": runtime,
        "mlx_vlm_runtime": runtime, "settings": {"mlx_vlm": runtime,
                                                 "adapter_policy": policy},
        "conditional_qualification_checks": rules,
        "serving_encoder_batching": observation,
        "direct": {"video": {"encoder_batching": observation,
                             "parity": {"feature_parity": parity}}},
    }
    return report, harness, rules


def _evaluate(report, harness, rules):
    from mlx2.media_qualification import evaluate_native_vlm_report

    return evaluate_native_vlm_report(report, expected_family="gemma3n",
                                      expected_harness=harness,
                                      expected_policy_checks=rules)


def _audit_encoder_evidence():
    import json

    path = ROOT / "docs/audits/2026-10-12-regressions/gemma3n-video-default-repair.json"
    if not path.is_file():
        # The audit evidence lives outside the published code tree.
        pytest.skip(f"{path.relative_to(ROOT)} is not in this tree")
    return json.loads(path.read_text())["encoder"]


def test_default_gemma3n_policy_evidence_passes_the_unsplit_check():
    observation = _audit_encoder_evidence()  # 086421f7b's own: [17], [17]
    checks = _evaluate(*_gemma3n_report(UNSPLIT, observation))
    assert checks["multimodal_encoder_unsplit"] is True
    assert "multimodal_encoder_batching" not in checks


def test_unsplit_check_fails_closed():
    observation = _audit_encoder_evidence()
    # A split call, a limit other than the selected policy's, or failed
    # feature parity is not single-batch evidence.
    split = {**observation, "calls": [{"input_count": 17, "chunk_sizes": [16, 1]}]}
    assert _evaluate(*_gemma3n_report(UNSPLIT, split))["multimodal_encoder_unsplit"] is False
    other_limit = {**observation, "policy_limit": 32}
    assert _evaluate(*_gemma3n_report(UNSPLIT, other_limit))["multimodal_encoder_unsplit"] is False
    assert _evaluate(*_gemma3n_report(UNSPLIT, observation, feature_parity=False))[
        "multimodal_encoder_unsplit"] is False
    # Under a splitting policy the same single-batch evidence cannot pass:
    # the batching check is required instead.
    checks = _evaluate(*_gemma3n_report(SPLIT, observation))
    assert checks["multimodal_encoder_batching"] is False
    assert "multimodal_encoder_unsplit" not in checks


def test_vlm_campaign_spec_requires_the_resolved_gemma3n_encoder_feature(tmp_path):
    import json as _json

    routes = _script("qualify_vlm_routes")
    artifact = tmp_path / "gemma3n"
    artifact.mkdir()
    (artifact / "config.json").write_text(_json.dumps({"model_type": "gemma3n"}))
    (artifact / "weights.safetensors").write_bytes(b"fixture")

    def spec(settings):
        return set(routes.campaign_spec(
            "gemma3n", artifact=artifact,
            reference_revision=routes.FAMILIES["gemma3n"]["revision"],
            reference_source_sha256="a" * 64, producer_sha256="c" * 64,
            runtime={"python": "3.12"}, settings=settings,
        )["required_features"])

    assert "encoder_unsplit" in spec({"adapter_policy": UNSPLIT})
    assert "encoder_batching" not in spec({"adapter_policy": UNSPLIT})
    assert "encoder_batching" in spec({"adapter_policy": SPLIT})
    # Unknown policy: fail closed to the batching requirement.
    assert "encoder_batching" in spec({"route": "ordinary"})

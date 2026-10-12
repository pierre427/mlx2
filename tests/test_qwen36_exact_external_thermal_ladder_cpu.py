"""CPU/static contracts for the Qwen3.6 exact-external thermal ladder."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "qualification"
    / "runs"
    / "speculative-composition-qualification-20261012"
    / "diagnose_qwen36_exact_external_thermal_ladder.py"
)
pytestmark = pytest.mark.skipif(
    not SCRIPT.is_file(), reason="private Qwen3.6 thermal harness is not exported"
)


def _module():
    spec = importlib.util.spec_from_file_location("qwen36_exact_ladder", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    values = {
        "execute": False,
        "max_new_tokens": 128,
        "max_length": 131072,
        "cell_gap_seconds": 5.0,
        "output_dir": None,
        "i_own_the_gpu": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_plan_is_exact_six_cell_single_session_performance_only():
    module = _module()
    plan = module._plan(_args())
    assert plan["schema"] == "mlx2.qwen36-exact-external-thermal-context-ladder.v1"
    assert plan["diagnostic_only"] is True
    assert plan["performance_experiment"] is True
    assert plan["qualification_claim"] is False
    assert plan["preflight_run"] is False
    assert plan["production_gate_changed"] is False
    assert plan["measurement"]["one_model_and_draft_load"] is True
    assert plan["measurement"]["one_request_at_a_time"] is True
    assert plan["measurement"]["measured_repetitions_per_cell"] == 1
    assert plan["measurement"]["cell_gap_seconds"] == 5.0
    assert [cell["requested_context_tokens"] for cell in plan["cells"]] == [
        1024,
        4096,
        16384,
        32768,
        65536,
        131072,
    ]
    assert all(cell["width"] == 1 for cell in plan["cells"])
    assert all(cell["max_new_tokens"] == 128 for cell in plan["cells"])


def test_plan_pins_exact_external_route_and_excludes_progressive_composition():
    module = _module()
    plan = module._plan(_args())
    policy = plan["external_policy"]
    assert policy["num_draft"] == 3
    assert policy["pairwise_selection"] == "host"
    assert policy["proposal_composition"] is False
    assert policy["exact_verification"] == "token"
    assert not any(key.startswith("progressive_") for key in policy)
    assert len(policy["target_revision"]) == 64
    assert len(policy["draft_revision"]) == 64
    assert plan["feature"]["approximate_operation"] is False
    assert plan["feature"]["progressive_verification_selected"] is False


def test_plan_records_adapter_default_selected_features_and_off_candidates():
    module = _module()
    selected = module._plan(_args())["selected_target_features"]
    assert selected["moe_nax_gather"] == "fused"
    assert selected["eager_dispatch_stride"] == 2
    assert selected["eager_dispatch_max_rows"] == 64
    assert selected["moe_routed_decode"] == "gate_up_down_shared"
    assert selected["moe_topk_fold"] == "launch"
    assert selected["fused_gdn_decode"] is True
    assert selected["fused_gdn_verify"] is True
    assert selected["fused_gdn_batch_verify"] is False
    assert selected["moe_window"] is False
    assert selected["moe_window_batch_decode"] is False
    assert selected["moe_window_verify"] is False


def test_exact_batch_identity_fails_closed_on_progressive_or_composition():
    module = _module()
    draft = SimpleNamespace()
    batch = SimpleNamespace(
        exact_verification="token",
        progressive_verification_tile=None,
        progressive_verification_fidelity=None,
        progressive_target_protocol=None,
        progressive_protocol_binding=None,
        pairwise_selection="host",
        draft_topology="chain",
        target_execution="reference",
        num_draft=3,
        draft=draft,
    )
    assert module._validate_exact_batch(batch)["proposal_composition_bound"] is False
    batch.progressive_verification_tile = 2
    with pytest.raises(RuntimeError, match="route identity mismatch"):
        module._validate_exact_batch(batch)
    batch.progressive_verification_tile = None
    draft.last_proposal_sources = ()
    with pytest.raises(RuntimeError, match="route identity mismatch"):
        module._validate_exact_batch(batch)


def test_exact_engagement_requires_external_rounds_and_no_progressive_counters():
    module = _module()
    observed = module._exact_external_engagement(
        {
            "external_rounds": 8,
            "proposed_tokens": 24,
            "accepted_proposals": 18,
            "target_max_width": 1,
            "draft_max_width": 1,
        }
    )
    assert observed["acceptance_rate"] == 0.75
    assert observed["progressive_observed"] is False
    assert observed["observed_used"] is True
    contaminated = module._exact_external_engagement(
        {
            "external_rounds": 8,
            "proposed_tokens": 24,
            "external_progressive_verify_rounds": 1,
        }
    )
    assert contaminated["observed_used"] is False


def test_target_feature_engagement_keeps_windows_distinct():
    module = _module()
    report = module._target_feature_engagement(
        fused_b1={"mode": "fused", "fused_calls": 3, "fallbacks": 0},
        decode_wins={
            "gdn": {"verify": {"calls": 5, "fallbacks": 0}},
            "moe": {
                "routed_gate_up_calls": 7,
                "topk_launch_calls": 7,
                "shared_calls": 7,
                "window_calls": 0,
            },
        },
        nax={"mode": "fused", "calls": {"gather": 2}, "fallbacks": {}},
        eager_delta={"eager_dispatch_forwards": 4},
    )
    assert report["fused_gdn_decode"]["observed_used"] is True
    assert report["decode_wins"]["fused_gdn_verify_observed"] is True
    assert report["decode_wins"]["routed_moe_observed"] is True
    assert report["decode_wins"]["topk_fold_observed"] is True
    assert report["moe_nax_gather"]["observed_used"] is True
    assert report["eager_dispatch"]["observed_used"] is True
    assert report["excluded_default_off_features_remained_off"] is True


def test_execute_fails_closed_without_explicit_gpu_ownership():
    module = _module()
    args = _args(execute=True, output_dir=Path("unused"))
    with pytest.raises(ValueError, match="--i-own-the-gpu"):
        module._execute(args, module._plan(args))

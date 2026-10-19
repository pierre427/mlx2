"""Static admission checks for bounded Qwen3.6 performance probes."""

import json
from pathlib import Path

import pytest
from mlx2.adapters.external_draft_policy import normalize_progressive_external_policy
from mlx2.runtime.cost_per_commit_handoff import CostPerCommitHandoffSettings


POLICIES = Path(__file__).resolve().parents[1] / (
    "qualification/runs/ast-smokes-20261018/policies"
)


@pytest.mark.parametrize(
    "depth,tile,filename",
    [
        (2, 1, "qwen36-35b-ast-q8-moe-attn-k2-primary.json"),
        (3, 2, "qwen36-35b-ast-q8-moe-attn-k3-tile2-primary.json"),
        (4, 1, "qwen36-35b-ast-q8-moe-attn-k4-primary.json"),
        (4, "auto", "qwen36-35b-ast-q8-moe-attn-k4-auto-primary.json"),
    ],
)
def test_qwen36_primary_progressive_depth_probe_is_exact_and_branchless(
    depth, tile, filename
):
    path = POLICIES / filename
    policy = json.loads(path.read_text())
    normalized = normalize_progressive_external_policy(
        policy,
        num_draft=depth,
        minimum_proposals=min(3, depth),
    )
    assert normalized["num_draft"] == depth
    assert normalized["proposal_composition"] is False
    assert normalized["progressive_verification_tile"] == tile
    assert normalized["progressive_multilane_draft_cap"] == depth
    assert normalized.get("progressive_alternative_branches", 1) == 1
    assert "continuation_pool" not in normalized
    assert normalized["target_verify_row_exact"] is True
    assert normalized["moe_window_row_exact"] is True
    assert normalized["attention_window_row_exact"] is True


def test_qwen36_k4_auto_cost_guard_only_adds_validated_reversible_handoff():
    base = json.loads(
        (POLICIES / "qwen36-35b-ast-q8-moe-attn-k4-auto-primary.json").read_text()
    )
    guarded = json.loads(
        (
            POLICIES
            / "qwen36-35b-ast-q8-moe-attn-k4-auto-primary-cost-guard.json"
        ).read_text()
    )
    cost = guarded.pop("cost_per_commit_handoff")
    assert guarded == base
    assert CostPerCommitHandoffSettings.from_value(cost).as_dict() == {
        "enabled": True,
        "ordinary_samples": 2,
        "speculative_samples": 2,
        "loss_ratio": 0.97,
        "loss_decisions": 1,
        "window": 4,
        "reversible": True,
        "reprobe_tokens": 16,
        "reprobe_max_tokens": 64,
        "deferred_context_cap": 64,
        "settle_rounds": 1,
        "pipelined_park": True,
    }
    normalized = normalize_progressive_external_policy(
        {**guarded, "cost_per_commit_handoff": cost},
        num_draft=4,
        minimum_proposals=3,
    )
    assert normalized["progressive_verification_tile"] == "auto"
    assert normalized["progressive_multilane_draft_cap"] == 4
    assert normalized["cost_per_commit_handoff"] == cost


def test_qwen36_q8_one_way_guard_selects_latest_round_loss():
    policy = json.loads(
        (
            POLICIES
            / "qwen36-35b-ast-q8-moe-attn-k4-auto-primary-one-way-cost-guard.json"
        ).read_text()
    )
    settings = CostPerCommitHandoffSettings.from_value(
        policy["cost_per_commit_handoff"]
    )
    assert settings.loss_basis == "current_round"
    assert settings.reversible is False
    assert settings.as_dict()["loss_basis"] == "current_round"

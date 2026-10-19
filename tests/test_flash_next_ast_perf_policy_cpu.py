"""Static admission checks for the bounded Flash-Next AST candidate."""

import json
from pathlib import Path

from mlx2.adapters.external_draft_policy import normalize_progressive_external_policy
from mlx2.runtime.cost_per_commit_handoff import CostPerCommitHandoffSettings


POLICIES = Path(__file__).resolve().parents[1] / (
    "qualification/runs/ast-smokes-20261018/policies"
)


def test_flash_next_k4_auto_guard_is_exact_branchless_and_one_way():
    base = json.loads(
        (POLICIES / "flash-next-native-mtp-ast-k4-auto-primary.json").read_text()
    )
    guarded = json.loads(
        (
            POLICIES
            / "flash-next-native-mtp-ast-k4-auto-primary-one-way-cost-guard.json"
        ).read_text()
    )
    cost = guarded.pop("cost_per_commit_handoff")
    assert guarded == base
    normalized = normalize_progressive_external_policy(
        {**guarded, "cost_per_commit_handoff": cost},
        num_draft=4,
        minimum_proposals=1,
    )
    assert normalized["row_exact_verify"] is True
    assert normalized["progressive_verification_tile"] == "auto"
    assert normalized["progressive_multilane_draft_cap"] == 3
    assert normalized.get("progressive_alternative_branches", 1) == 1
    assert "continuation_pool" not in normalized
    settings = CostPerCommitHandoffSettings.from_value(cost)
    assert settings.loss_basis == "current_round"
    assert settings.reversible is False
    assert settings.as_dict()["loss_basis"] == "current_round"


def test_flash_next_rolling_guard_prices_cumulative_route_cost():
    base = json.loads(
        (POLICIES / "flash-next-native-mtp-ast-k4-auto-primary.json").read_text()
    )
    guarded = json.loads(
        (
            POLICIES
            / "flash-next-native-mtp-ast-k4-auto-primary-rolling-cost-guard.json"
        ).read_text()
    )
    cost = guarded.pop("cost_per_commit_handoff")
    assert guarded == base
    settings = CostPerCommitHandoffSettings.from_value(cost)
    assert settings.loss_basis == "rolling_window"
    assert settings.speculative_samples == settings.window == 4
    assert settings.reversible is False
    assert settings.one_way_ordinary_settle_rounds == 1
    assert settings.as_dict()["one_way_ordinary_settle_rounds"] == 1
    assert "loss_basis" not in settings.as_dict()


def test_flash_next_reversible_guard_excludes_warmup_and_reprobes():
    base = json.loads(
        (POLICIES / "flash-next-native-mtp-ast-k4-auto-primary.json").read_text()
    )
    guarded = json.loads(
        (
            POLICIES
            / "flash-next-native-mtp-ast-k4-auto-primary-cost-guard.json"
        ).read_text()
    )
    cost = guarded.pop("cost_per_commit_handoff")
    assert guarded == base
    settings = CostPerCommitHandoffSettings.from_value(cost)
    assert settings.reversible is True
    assert settings.settle_rounds == 1
    assert settings.pipelined_park is True
    assert settings.reprobe_tokens == 16
    assert settings.reprobe_max_tokens == 64
    assert settings.deferred_context_cap == 64


def test_flash_next_multibranch_ast_has_a_one_way_cost_guard():
    base = json.loads(
        (POLICIES / "flash-next-native-mtp-ast.json").read_text()
    )
    guarded = json.loads(
        (POLICIES / "flash-next-native-mtp-ast-cost-guard.json").read_text()
    )
    cost = guarded.pop("cost_per_commit_handoff")
    assert guarded == base
    settings = CostPerCommitHandoffSettings.from_value(cost)
    assert settings.reversible is False
    assert settings.loss_basis == "rolling_window"
    assert settings.one_way_ordinary_settle_rounds == 1

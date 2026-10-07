"""rfix-kad 2026-10-07: the external-draft qualification observations read
lifetime counters, so a run that never engaged the route could pass on
history (and historical draft fallbacks could fail a healthy run). Every
required external-route feature is now a final - initial delta."""

from __future__ import annotations

import copy

import pytest

from mlx2.qualification import required_feature_checks
from scripts.qualify_serving import feature_observations

SETTINGS = {
    "mtp": False,
    "speculation": "external_draft",
    "fly_verification": {"enabled": True},
    "execution_policy": {
        "pairwise_selection": "batched",
        "batch_size_route": "tree15_b1_b4_chain_b5plus_v1",
        "external_varlen_prefill": {"enabled": True},
        "ingress_cohort": {"enabled": True},
    },
}
HISTORY = {
    "settings": SETTINGS,
    "scheduler": {
        "external_rounds": 9, "proposed_tokens": 30, "paired_cache_resumes": 4,
        "segmented_transactions": 5, "segmented_rollbacks": 2,
        "external_pairwise_selection_groups": 3, "draft_fallbacks": 0,
        "fly_relaxed_accepts": 6, "external_tree_rounds": 7,
        "external_tensorfold_target_rounds": 7,
        "external_batched_prefill_rounds": 2, "external_batched_prefill_lanes": 4,
    },
    "counts": {"ingress_cohort_target_reached": 2},
    "recent_receipts": [{"speculation": {"verification": "fly", "relaxed_accepts": 3}}],
}
EXTERNAL = (
    "external_draft", "proposal_distribution", "paired_draft_cache",
    "segmented_transaction", "segmented_rollback", "external_pairwise_selection",
    "fly_verification",
)


def test_identical_snapshots_observe_no_required_external_feature():
    required = {name.removeprefix("feature_") for name in required_feature_checks(SETTINGS)}
    assert set(EXTERNAL) - {"segmented_rollback"} <= required
    observed = feature_observations(HISTORY, initial=copy.deepcopy(HISTORY))
    assert {name: observed[name] for name in sorted(required | set(EXTERNAL))
            if observed[name]} == {}


def test_run_growth_is_observed_as_a_delta():
    final = copy.deepcopy(HISTORY)
    for key in ("external_rounds", "proposed_tokens", "paired_cache_resumes",
                "segmented_transactions", "segmented_rollbacks",
                "external_pairwise_selection_groups", "fly_relaxed_accepts"):
        final["scheduler"][key] += 2
    observed = feature_observations(final, initial=HISTORY)
    assert {name: observed[name] for name in EXTERNAL} == dict.fromkeys(EXTERNAL, 2)


def test_historical_fallbacks_do_not_fail_a_clean_run():
    initial = copy.deepcopy(HISTORY)
    initial["scheduler"]["draft_fallbacks"] = 1
    final = copy.deepcopy(initial)
    final["scheduler"]["external_rounds"] += 4
    assert feature_observations(final, initial=initial)["external_draft"] == 4
    final["scheduler"]["draft_fallbacks"] += 1
    assert feature_observations(final, initial=initial)["external_draft"] == 0


def test_native_segmented_counters_are_deltas():
    native = {
        "settings": {"mtp": True},
        "execution": {"segmented_mtp": {
            "transaction_branches": 12, "accepted_zero": 5, "accepted_partial": 4,
        }},
        "scheduler": {},
    }
    observed = feature_observations(native, initial=copy.deepcopy(native))
    assert (observed["segmented_transaction"], observed["segmented_rollback"]) == (0, 0)
    grown = copy.deepcopy(native)
    grown["execution"]["segmented_mtp"]["transaction_branches"] += 3
    grown["execution"]["segmented_mtp"]["accepted_zero"] += 1
    observed = feature_observations(grown, initial=native)
    assert (observed["segmented_transaction"], observed["segmented_rollback"]) == (3, 1)

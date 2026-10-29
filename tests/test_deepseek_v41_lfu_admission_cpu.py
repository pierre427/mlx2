"""Source LFU admission uses its actual profile; synthetic CPU bounds only."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from mlx2.adapters import deepseek_v41_source_lfu as lfu
from mlx2.adapters.deepseek_v41_source import source_policy
from mlx2.runtime.streamed_load import WeightStreamRequest

GIB = 1 << 30


@pytest.mark.parametrize("value", [[], False, 0, "", [("deepseek_context_limit", 4096)], "invalid"])
def test_source_policy_rejects_non_dictionary_input(value):
    with pytest.raises(ValueError, match="must be a dictionary"):
        source_policy(value)


def test_source_policy_preserves_none_and_empty_dictionary_defaults():
    assert source_policy(None) == source_policy({})


@pytest.fixture
def profile(monkeypatch):
    calls = []

    def planner(artifact, budget):
        calls.append((artifact, budget))
        return dict(
            dense_payload_bytes=GIB,
            dense_prepared_bound_bytes=0,
            dense_staging_bound_bytes=GIB,
            expert_cache_bytes=budget,
            minimum_expert_cache_bytes=lfu.ORIGINAL_BYTES,
            expert_active_bound_bytes=8 * GIB,
            expert_staging_bound_bytes=GIB,
            total_reservation_bytes=budget + 11 * GIB,
        )

    monkeypatch.setattr(lfu, "weight_plan", planner)
    artifact = {"args": SimpleNamespace(engram_layer_ids=(1, 7))}
    policy = lfu.parse_policy({"mode": "original_lfu", "budget_bytes": 8 * GIB})
    request = WeightStreamRequest(
        "moe_experts", policy["budget_bytes"], admission_limit_bytes=20 * GIB
    )
    return artifact, policy, request, calls


def test_lfu_admits_imminent_progress_without_full_cache_reservation(profile):
    artifact, policy, request, calls = profile
    plan = lfu.admission(artifact, request, policy)
    assert calls == [(artifact, 8 * GIB)]
    assert plan['required_bytes'] == 2 * lfu.ORIGINAL_BYTES
    assert plan['estimated_full_reservation_bytes'] == 19 * GIB
    assert plan['expert_active_bound_bytes'] == policy['budget_bytes']
    assert plan['expert_staging_bound_bytes'] == lfu.ORIGINAL_BYTES
    assert plan['estimates_are_advisory'] and not plan['full_cache_preallocation']


@pytest.mark.parametrize('limit', [None, 0, 1, 20 * GIB])
def test_lfu_demand_attempt_does_not_require_fictitious_full_profile_fit(profile, limit):
    artifact, policy, request, _ = profile
    plan = lfu.admission(artifact, replace(request, admission_limit_bytes=limit), policy)
    assert plan['admission_limit_bytes'] == limit
    assert plan['required_bytes'] == plan['imminent_working_reserve_bytes'] == 2 * lfu.ORIGINAL_BYTES
    assert plan['headroom_bytes'] == plan['metadata_reserve_bytes'] == 0


@pytest.mark.parametrize('changes', [
    {'mode': 'dense_mlp'}, {'mtp_resident': True}, {'collector': object()},
    {'prefill_admission': 'terminal_hot'}, {'router_hint': 'next_layer_reuse'},
    {'expert_overlap': True},
])
def test_lfu_rejects_foreign_ownership_options(profile, changes):
    artifact, policy, request, calls = profile
    with pytest.raises(ValueError, match='synchronous ordinary'):
        lfu.admission(artifact, replace(request, **changes), policy)
    assert not calls


def test_lfu_requires_explicit_budget_to_match_policy(profile):
    artifact, policy, request, calls = profile
    with pytest.raises(ValueError, match='requested expert maximum disagree'):
        lfu.admission(artifact, replace(request, budget_bytes=9 * GIB), policy)
    assert not calls


def test_lfu_auto_uses_policy_maximum_without_reserving_it(profile):
    artifact, policy, request, calls = profile
    plan = lfu.admission(artifact, replace(request, budget_bytes=0, budget_mode='auto'), policy)
    assert calls == [(artifact, policy['budget_bytes'])]
    assert plan['expert_cache_bytes'] == policy['budget_bytes']
    assert plan['required_bytes'] == 2 * lfu.ORIGINAL_BYTES


def test_lfu_preserves_source_ceiling_above_native_lru_cap(profile):
    artifact, _, request, calls = profile
    policy = lfu.parse_policy({'mode': 'original_lfu', 'budget_bytes': 80 * GIB})
    request = replace(request, budget_bytes=policy['budget_bytes'], admission_limit_bytes=24 * GIB)
    plan = lfu.admission(artifact, request, policy)
    assert calls == [(artifact, 80 * GIB)]
    assert plan['required_bytes'] < request.admission_limit_bytes
    assert plan['estimated_full_reservation_bytes'] > request.admission_limit_bytes

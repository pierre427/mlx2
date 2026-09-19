import math

import pytest

from mlx2.runtime.async_mtp_scheduler import (
    AspireDepthPolicy,
    CostModel,
    FeedbackDepthPolicy,
    FixedDepthPolicy,
    RequestContext,
    TraceRecord,
    replay_trace,
)

COST = CostModel(beta_model=0.2, beta_mlp=0.01, beta_attn=0.00003)
CONTEXT = RequestContext("a", 32_768, 2_048, 4)


def test_cost_model_matches_homogeneous_batch_equations():
    assert COST.draft_cost(CONTEXT) == pytest.approx(0.48576)
    assert COST.verify_cost(CONTEXT, 2) == pytest.approx(4.25216)


def test_feedback_policy_moves_one_step_and_stays_bounded():
    policy = FeedbackDepthPolicy(initial_depth=2, minimum_depth=1, maximum_depth=3)
    assert policy.choose_depth(CONTEXT) == 2
    policy.observe(2, 2)
    assert policy.choose_depth(CONTEXT) == 3
    policy.observe(3, 0)
    policy.observe(2, 0)
    policy.observe(1, 0)
    assert policy.choose_depth(CONTEXT) == 1


def test_aspire_policy_updates_alpha_using_successes_then_failure():
    policy = AspireDepthPolicy(COST, maximum_depth=8, initial_depth=3, smoothing=0.8)
    assert policy.choose_depth(CONTEXT) == 3
    policy.observe(3, 1)
    assert policy.alpha == pytest.approx(0.82)
    assert 0 <= policy.choose_depth(CONTEXT) <= 8
    policy.observe(0, 0)
    assert policy.alpha == pytest.approx(0.82)


def test_aspire_policy_adapts_depth_to_request_local_history():
    low = AspireDepthPolicy(COST)
    high = AspireDepthPolicy(COST)
    for _ in range(12):
        low.observe(3, 0)
        high.observe(3, 3)
    assert low.choose_depth(CONTEXT) == 0
    assert high.choose_depth(CONTEXT) == 8


def test_replay_is_deterministic_and_request_local():
    records = [
        TraceRecord("a", 0, 16_384, 1_024, 2, 4, 4),
        TraceRecord("b", 0, 20_000, 1_024, 2, 4, 0),
        TraceRecord("a", 1, 16_386, 1_024, 2, 4, 1),
        TraceRecord("b", 1, 20_002, 1_024, 2, 4, 4),
    ]
    result = replay_trace(
        records,
        policy_name="fixed_k2",
        policy_factory=lambda: FixedDepthPolicy(2),
        cost_model=COST,
    )
    assert result.records == 4
    assert result.drafted_tokens == 8
    assert result.accepted_tokens == 5
    assert result.committed_tokens == 9
    assert result.full_accept_rounds == 2
    assert math.isfinite(result.throughput)
    assert result.as_dict()["mean_depth"] == 2


def test_replay_rejects_non_monotonic_rounds_and_policy_above_trace_cap():
    duplicate = [
        TraceRecord("a", 0, 10, 2, 1, 2, 0),
        TraceRecord("a", 0, 10, 2, 1, 2, 0),
    ]
    with pytest.raises(ValueError, match="round_index"):
        replay_trace(
            duplicate,
            policy_name="fixed",
            policy_factory=lambda: FixedDepthPolicy(1),
            cost_model=COST,
        )
    with pytest.raises(ValueError, match="outside trace cap"):
        replay_trace(
            [TraceRecord("a", 0, 10, 2, 1, 1, 0)],
            policy_name="fixed",
            policy_factory=lambda: FixedDepthPolicy(2),
            cost_model=COST,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"request_id": "", "round_index": 0, "context_length": 10, "sparse_context_length": 2, "batch_size": 1, "max_depth": 2, "accepted_prefix": 0},
        {"request_id": "a", "round_index": 0, "context_length": 10, "sparse_context_length": 11, "batch_size": 1, "max_depth": 2, "accepted_prefix": 0},
        {"request_id": "a", "round_index": 0, "context_length": 10, "sparse_context_length": 2, "batch_size": 1, "max_depth": 2, "accepted_prefix": 3},
    ],
)
def test_trace_validation(kwargs):
    with pytest.raises(ValueError):
        TraceRecord(**kwargs)

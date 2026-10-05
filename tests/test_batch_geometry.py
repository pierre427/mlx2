import pytest

from mlx2.runtime.batch_geometry import (
    GeometryCapabilities,
    GeometryLane,
    GeometryPolicy,
    GeometrySchedulerPolicy,
    plan_batch_geometry,
    trim_rectangular_cohort,
)


def test_skewed_prefill_selects_packed_and_records_avoided_padding():
    lanes = [GeometryLane(i, "prefill", n) for i, n in enumerate((1536, 4096, 6982))]
    plan = plan_batch_geometry(
        lanes,
        GeometryCapabilities(
            packed_prefill=True, max_lanes=3, token_budget=24000, prefill_chunk=8192
        ),
    )
    assert plan.geometry == "packed"
    assert plan.real_rows == 12614
    assert plan.padding_rows == 8332
    assert plan.padding_fraction == pytest.approx(8332 / 20946)
    assert plan.receipt()["selected_by_default"] is False


def test_no_packed_capability_buckets_skewed_rows():
    lanes = [GeometryLane(i, "prefill", n) for i, n in enumerate((1000, 1100, 4000))]
    plan = plan_batch_geometry(
        lanes,
        GeometryCapabilities(max_lanes=3, token_budget=16000, prefill_chunk=8192),
        GeometryPolicy(max_bucket_ratio=1.5),
    )
    assert plan.geometry == "bucketed_rectangular"
    assert plan.prefill_uids == (0, 1)
    assert plan.charged_rows == 2200
    assert plan.padding_rows == 100


def test_decode_first_mixed_round_reserves_speculative_verification_rows():
    lanes = [
        GeometryLane(10, "decode", 1, draft_tokens=2),
        GeometryLane(11, "decode", 1),
        GeometryLane(20, "prefill", 300),
    ]
    plan = plan_batch_geometry(
        lanes,
        GeometryCapabilities(mixed_forward=True, max_lanes=4, token_budget=512),
    )
    assert plan.geometry == "mixed_decode_first"
    assert plan.decode_uids == (10, 11)
    assert plan.verification_rows == 2
    assert plan.prefill_rows == 300
    assert plan.real_rows == plan.charged_rows == 304


def test_mixed_round_enforces_max_lanes_across_decode_and_prefill():
    lanes = [
        GeometryLane(10, "decode", 1),
        GeometryLane(11, "decode", 1),
        GeometryLane(20, "prefill", 4),
        GeometryLane(21, "prefill", 4),
        GeometryLane(22, "prefill", 4),
        GeometryLane(23, "prefill", 4),
    ]
    plan = plan_batch_geometry(
        lanes,
        GeometryCapabilities(
            mixed_forward=True, max_lanes=4, token_budget=100, prefill_chunk=4
        ),
    )
    assert plan.geometry == "mixed_decode_first"
    assert plan.decode_uids == (10, 11)
    assert plan.prefill_uids == (20, 21)
    assert len(plan.lane_uids) == 4


def test_mixed_round_does_not_overflow_exhausted_prefill_budget():
    lanes = [
        GeometryLane(10, "decode", 1, draft_tokens=3),
        GeometryLane(20, "prefill", 100),
    ]
    plan = plan_batch_geometry(
        lanes,
        GeometryCapabilities(
            mixed_forward=True, max_lanes=4, token_budget=4, prefill_chunk=100
        ),
    )
    assert plan.geometry == "rectangular"
    assert plan.decode_uids == (10,)
    assert plan.prefill_uids == ()
    assert plan.charged_rows == plan.token_budget == 4


def test_prefill_only_round_keeps_first_lane_overflow_for_progress():
    plan = plan_batch_geometry(
        [GeometryLane(20, "prefill", 100)],
        GeometryCapabilities(max_lanes=4, token_budget=4, prefill_chunk=100),
    )
    assert plan.prefill_uids == (20,)
    assert plan.charged_rows == 100
    assert plan.charged_rows > plan.token_budget


def test_decode_only_does_not_claim_packed_without_verify_capability():
    lanes = [GeometryLane(1, "decode", 1, draft_tokens=3)]
    ordinary = plan_batch_geometry(lanes, GeometryCapabilities(token_budget=8))
    packed = plan_batch_geometry(
        lanes, GeometryCapabilities(packed_verify=True, token_budget=8)
    )
    assert ordinary.geometry == "rectangular"
    assert packed.geometry == "packed"
    assert ordinary.verification_rows == packed.verification_rows == 3


def test_server_policy_cannot_declare_execution_capabilities():
    parsed = GeometrySchedulerPolicy.from_value({"token_budget": 4096})
    assert parsed.enabled and parsed.as_dict()["token_budget"] == 4096
    with pytest.raises(ValueError, match="unknown batch_geometry"):
        GeometrySchedulerPolicy.from_value({"packed_prefill": True})


def test_rectangular_budget_keeps_head_and_drops_costliest_companion():
    selected, receipt = trim_rectangular_cohort(
        (2048, 256, 2048, 512), (0, 1, 2, 3), token_budget=6144
    )
    assert selected[0] == 0
    assert selected == (0, 2, 3)
    assert receipt == {
        "candidate_rows": 4,
        "selected_rows": 3,
        "deferred_rows": 1,
        "real_rows": 4608,
        "charged_rows": 6144,
        "padding_rows": 1536,
        "original_charged_rows": 8192,
    }


def test_rectangular_budget_always_allows_one_row_to_progress():
    selected, receipt = trim_rectangular_cohort(
        (8192, 64), (0, 1), active_lengths=(1024,), token_budget=512
    )
    assert selected == (0,)
    assert receipt["charged_rows"] > 512


@pytest.mark.parametrize(
    "lane",
    [
        lambda: GeometryLane(1, "prefill", 0),
        lambda: GeometryLane(1, "decode", 1, draft_tokens=-1),
        lambda: GeometryLane(1, "prefill", 1, draft_tokens=1),
    ],
)
def test_invalid_lane_is_refused(lane):
    with pytest.raises((TypeError, ValueError)):
        lane()

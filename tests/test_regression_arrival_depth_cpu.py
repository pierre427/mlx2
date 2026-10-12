"""An idle static ordinary cohort must not reclaim for unused MTP rows."""

import pytest

from mlx2.runtime.adaptive_policy import MTPOrdinaryHandoffPolicy
from mlx2.runtime.memory_policy import SelfMTPLaneAdmissionController
from mlx2.serving import (
    admit_lane_headroom,
    lane_admission_required_gib,
    native_mtp_arrival_depth,
)


@pytest.mark.parametrize("width,active,park,enabled,expected", [
    (4, False, False, True, 0),
    (3, False, False, True, 2),
    (4, True, False, True, 2),
    (4, False, True, True, 2),
    (4, False, False, False, 2),
])
def test_arrival_prices_only_the_selected_static_idle_route(
    width, active, park, enabled, expected
):
    policy = MTPOrdinaryHandoffPolicy(enabled, 3)
    assert native_mtp_arrival_depth(
        2, handoff_policy=policy, projected_width=width,
        decode_active=active, adaptive_park_active=park,
    ) == expected


def test_ordinary_arrival_does_not_reclaim_to_buy_unused_verify_workspace():
    controller = SelfMTPLaneAdmissionController(cache_estimator=lambda _: 0)
    options = dict(context_tokens=32768, cache_gib=0.0)
    floor = lane_admission_required_gib(controller, draft_depth=0, **options)
    full = lane_admission_required_gib(controller, draft_depth=2, **options)
    reclaims = []
    depth = native_mtp_arrival_depth(
        2, handoff_policy=MTPOrdinaryHandoffPolicy(True, 3), projected_width=4,
    )
    admitted, used_floor, required = admit_lane_headroom(
        controller, draft_depth=depth, **options,
        headroom=lambda: int((full + floor) / 2 * (1 << 30)),
        reclaim=lambda: reclaims.append(True), evict=lambda: False,
    )
    assert admitted and not used_floor
    assert required == pytest.approx(floor)
    assert reclaims == []

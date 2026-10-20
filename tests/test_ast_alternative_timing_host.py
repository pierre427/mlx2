"""Backend-local timing feedback; synthetic host costs are not GPU evidence."""

from dataclasses import replace

import pytest

from mlx2.runtime.ast_planner import LaunchPlan, ReadyRow
from test_ast_planner_host import alternative, planner, profile


def timed(backend, width, seconds):
    return LaunchPlan((0,), width, (width,), 1, 1, 1, 1, backend, (1024,)), seconds


def test_alternative_measurements_change_selection_without_training_primary():
    primary = profile(row_costs=(2.0,) * 32, launch_cost=0)
    bound = replace(primary, alternative_backends=(alternative(primary),))
    p = planner(bound)
    ready = [ReadyRow(0, 7, 2, 1024)]
    assert (
        p.plan_launch(ready, terminal=True, max_lanes=1).backend
        == "target_tree_single_session"
    )
    for _ in range(8):
        p.observe_launch(*timed("target_tree_single_session", 3, 0.020))
    assert p.cost_samples == 0
    assert p.launch_curve_receipt()["launch_fit_samples"] == 0
    # A real ordinary reference measurement establishes comparable units.
    p.observe_launch(*timed(primary.backend, 3, 0.010))
    assert p.plan_launch(ready, terminal=True, max_lanes=1).backend == primary.backend
    receipt = p.receipt()["alternative_backends"][0]
    assert receipt["cost_samples"] == 8
    assert receipt["timing_authority"] == "backend_local_observed_target_invocations_v1"
    assert receipt["observed_used"] is True
    assert receipt["qualified"] is False
    fresh = planner(bound)
    assert fresh.receipt()["alternative_backends"][0]["cost_samples"] == 0
    assert (
        fresh.plan_launch(ready, terminal=True, max_lanes=1).backend
        == "target_tree_single_session"
    )


def test_alternative_shapes_learn_independently_in_shared_primary_units():
    primary = profile(row_costs=(2.0,) * 32, launch_cost=0)
    p = planner(replace(primary, alternative_backends=(alternative(primary),)))
    p.observe_launch(*timed(primary.backend, 3, 0.010))
    for _ in range(24):
        p.observe_launch(*timed("target_tree_single_session", 2, 0.002))
        p.observe_launch(*timed("target_tree_single_session", 4, 0.040))
    assert p.backend_cost("target_tree_single_session", (2,), (1024,)) < p.cost(
        (2,), (1024,)
    )
    assert p.backend_cost("target_tree_single_session", (4,), (1024,)) > p.cost(
        (4,), (1024,)
    )
    assert p.receipt()["alternative_backends"][0]["observed_shape_count"] == 2
    assert p.cost_samples == 1
    assert p.backend_cost("target_tree_single_session", (1,), (1024,)) == float("inf")
    assert p.backend_cost("target_tree_single_session", (2, 2), (1024, 1024)) == float(
        "inf"
    )


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf")])
def test_invalid_alternative_timing_does_not_change_cost(seconds):
    primary = profile()
    p = planner(replace(primary, alternative_backends=(alternative(primary),)))
    prior = p.backend_cost("target_tree_single_session", (3,), (1024,))
    p.observe_launch(*timed("target_tree_single_session", 3, seconds))
    assert p.backend_cost("target_tree_single_session", (3,), (1024,)) == prior
    assert p.receipt()["alternative_backends"][0]["cost_samples"] == 0


def test_successor_cost_prior_is_not_replaced_by_forward_only_time():
    primary = profile()
    candidate = replace(
        alternative(primary),
        timing_boundary="target_successor",
        cost_authority="offline_paired_successor_timing",
    )
    p = planner(replace(primary, alternative_backends=(candidate,)))
    before = p.backend_cost(candidate.backend, (3,), (1024,))
    p.observe_launch(*timed(candidate.backend, 3, 0.000001))
    assert p.backend_cost(candidate.backend, (3,), (1024,)) == before
    receipt = p.receipt()["alternative_backends"][0]
    assert receipt["launches"] == 1 and receipt["observed_used"]
    assert receipt["cost_samples"] == 0
    assert receipt["timing_boundary"] == "target_successor"
    assert receipt["live_timing_withheld"]
    assert receipt["timing_authority"] == "offline_paired_successor_timing"
    assert p.cost_samples == 0
    with pytest.raises(ValueError, match="timing boundary"):
        replace(candidate, timing_boundary="unknown")

"""Host-only measured source decisions; synthetic times are not GPU evidence."""

from dataclasses import FrozenInstanceError
import json

import pytest

from mlx2.runtime.ast_tree_tuning import ASTTreeTuner


def _observe(tuner, plan, *, seconds=0.1, accepted=None, **flags):
    accepted = plan.depth if accepted is None else accepted
    return tuner.observe(
        plan,
        seconds=seconds,
        emitted=accepted + 1,
        accepted=accepted,
        unique_rows=1 + plan.depth * plan.branches,
        launches=plan.expected_tiles,
        **flags,
    )


def test_startup_preserves_legal_four_path_source_and_root_inclusive_capacity():
    tuner = ASTTreeTuner(("source-model-profile", 17), max_depth=7)
    plan = tuner.plan(remaining_tokens=64)
    assert (plan.depth, plan.branches, plan.max_tile_rows) == (7, 4, 16)
    assert plan.expected_tiles == plan.max_tiles == 2
    assert plan.reason == "startup_measurement"
    with pytest.raises(FrozenInstanceError):
        plan.depth = 2
    json.dumps(plan.receipt())
    json.dumps(tuner.receipt())
    _observe(tuner, plan)
    assert tuner.receipt()["last_plan"] == plan.receipt()


def test_real_completed_round_throughput_reduces_slow_wide_branches():
    tuner = ASTTreeTuner("scope", max_depth=7)
    plans = []
    for _ in range(64):
        plan = tuner.plan(remaining_tokens=128)
        plans.append(plan)
        # All proposals reject: valid economics, not a discarded round.
        assert _observe(tuner, plan, seconds=0.01 * plan.branches, accepted=0)
    assert sum(plan.branches == 1 for plan in plans[-16:]) >= 12
    assert tuner.receipt()["cost_samples"] == 64
    assert not tuner.receipt()["skipped"]


def test_beneficial_deep_continuation_is_retained_over_shorter_work():
    tuner = ASTTreeTuner("scope", max_depth=7)
    plans = []
    for _ in range(64):
        plan = tuner.plan(remaining_tokens=128)
        plans.append(plan)
        assert _observe(tuner, plan, seconds=0.04 + 0.001 * plan.depth)
    assert sum(plan.depth == 7 for plan in plans[-16:]) >= 12
    assert any(plan.depth == 6 for plan in plans)  # Measured, not assumed.


def test_pressure_growth_limits_source_work_not_target_capacity_and_release_recovers():
    tuner = ASTTreeTuner("scope", max_depth=7)
    for _ in range(2):
        _observe(tuner, tuner.plan(remaining_tokens=64))
    pressured = tuner.plan(remaining_tokens=64, ready_sessions=4, target_sessions=4)
    assert pressured.depth == 7 and pressured.branches == 1
    assert pressured.max_tile_rows == 4 and pressured.max_tiles == 2
    assert pressured.ready_sessions == 4
    _observe(tuner, pressured, seconds=10)
    released = tuner.plan(remaining_tokens=64, ready_sessions=1)
    assert released.depth * released.branches > pressured.depth * pressured.branches
    assert released.max_tile_rows == 16 and released.max_tiles == 2
    assert {row["ready_sessions"] for row in tuner.receipt()["measurements"]} == {1, 4}


def test_terminal_failed_and_discarded_rounds_do_not_train_or_duplicate():
    tuner = ASTTreeTuner("scope", max_depth=7)
    short = tuner.plan(remaining_tokens=3)
    assert short.depth == 2 and short.terminal_censored
    assert not _observe(tuner, short)
    for flag in ({"terminal": True}, {"completed": False}, {"rejected": True}):
        assert not _observe(tuner, tuner.plan(remaining_tokens=64), **flag)
    valid = tuner.plan(remaining_tokens=64)
    assert _observe(tuner, valid)
    assert not _observe(tuner, valid)
    assert tuner.receipt()["cost_samples"] == 1
    assert tuner.plan(remaining_tokens=1).depth == 0
    assert tuner.plan(remaining_tokens=0).expected_tiles == 0


def test_scope_isolation_hysteresis_and_fixed_policy_bypass():
    tuner = ASTTreeTuner("source-a", max_depth=7)
    other = ASTTreeTuner("source-b", max_depth=7)
    foreign = other.plan(remaining_tokens=64)
    assert not _observe(tuner, foreign)
    # A five-percent advantage does not defeat the ten-percent hysteresis.
    plans = []
    for _ in range(20):
        plan = tuner.plan(remaining_tokens=64)
        plans.append(plan)
        _observe(tuner, plan, seconds=1 if plan.branches == 4 else 0.95)
    incumbent = [
        plan for plan in plans if plan.reason == "measured_throughput_hysteresis"
    ]
    assert incumbent and all(plan.branches == 4 for plan in incumbent)
    fixed = ASTTreeTuner("fixed", max_depth=7, fixed_depth=5, fixed_branches=4)
    plan = fixed.plan(remaining_tokens=64, ready_sessions=20)
    assert (plan.depth, plan.branches, plan.reason) == (5, 4, "explicit_fixed")
    assert not _observe(fixed, plan) and fixed.receipt()["cost_samples"] == 0


def test_bounds_invalid_measurements_and_history_are_bounded():
    tuner = ASTTreeTuner("scope", max_depth=15)
    for pressure in range(1, 20):
        for _ in range(8):
            plan = tuner.plan(remaining_tokens=128, ready_sessions=pressure)
            assert 1 <= plan.depth <= 15 and 1 <= plan.branches <= 4
            assert 1 <= plan.expected_tiles <= plan.max_tiles <= 2
            assert plan.max_tile_rows == 16
            _observe(tuner, plan, seconds=plan.branches + plan.depth / 10)
    assert len(tuner.receipt()["measurements"]) <= 32
    for seconds in (0, -1, float("nan"), float("inf"), True):
        assert not _observe(tuner, tuner.plan(remaining_tokens=128), seconds=seconds)
    with pytest.raises(ValueError):
        ASTTreeTuner("scope", max_depth=16)
    with pytest.raises(ValueError):
        ASTTreeTuner(([],), max_depth=7)
    with pytest.raises(ValueError):
        tuner.plan(remaining_tokens=3, ready_sessions=0)


def test_long_source_explores_coarse_depths_without_linear_microsteps():
    tuner = ASTTreeTuner("scope", max_depth=15)
    depths = []
    for _ in range(24):
        plan = tuner.plan(remaining_tokens=128)
        depths.append(plan.depth)
        # Flat progress, so shortening costly source work is beneficial.
        _observe(tuner, plan, seconds=0.01 * plan.depth, accepted=0)
    assert 7 in depths
    assert depths.index(7) < 12


def test_minimum_depth_survives_pressure_and_exploration_except_censored_tail():
    tuner = ASTTreeTuner("scope", max_depth=15, minimum_depth=4)
    for pressure in (1, 4, 16, 1):
        for _ in range(24):
            plan = tuner.plan(remaining_tokens=128, ready_sessions=pressure)
            assert 4 <= plan.depth <= 15
            assert plan.max_tile_rows == 16
            assert plan.expected_tiles <= plan.max_tiles
            assert _observe(tuner, plan, seconds=plan.depth, accepted=0)
    tail = tuner.plan(remaining_tokens=3, ready_sessions=16)
    assert tail.depth == 2 and tail.terminal_censored
    assert not _observe(tuner, tail)
    short_cap = ASTTreeTuner("short", max_depth=3, minimum_depth=4)
    assert short_cap.plan(remaining_tokens=128, ready_sessions=16).depth == 3
    assert short_cap.receipt()["minimum_depth"] == 3
    with pytest.raises(ValueError):
        ASTTreeTuner("invalid", max_depth=7, minimum_depth=0)


def test_physical_cohort_quota_includes_each_private_root_and_isolates_costs():
    tuner = ASTTreeTuner("cohort", max_depth=15, minimum_depth=3)
    for width, depth, rows in ((2, 15, 8), (3, 9, 5), (4, 7, 4)):
        plan = tuner.plan(
            remaining_tokens=128, ready_sessions=width, target_sessions=width
        )
        assert (plan.depth, plan.branches, plan.max_tile_rows) == (depth, 1, rows)
        assert plan.max_tiles == plan.expected_tiles == 2
        assert width * plan.max_tile_rows <= 16
        assert plan.receipt()["target_sessions"] == width
        assert _observe(tuner, plan)
    serial = tuner.plan(remaining_tokens=128, ready_sessions=4, target_sessions=1)
    assert serial.max_tile_rows == 16 and serial.depth == 15
    assert serial.cost_samples == 0
    assert _observe(tuner, serial, seconds=1)
    shared = tuner.plan(remaining_tokens=128, ready_sessions=4, target_sessions=4)
    assert shared.cost_samples == 1
    assert {
        (row["ready_sessions"], row["target_sessions"])
        for row in tuner.receipt()["measurements"]
    } == {(2, 2), (3, 3), (4, 4), (4, 1)}


def test_physical_cohort_refuses_a_quota_below_semantic_source_floor():
    tuner = ASTTreeTuner("floor", max_depth=15, minimum_depth=4)
    with pytest.raises(ValueError, match="proposal floor"):
        tuner.plan(remaining_tokens=128, ready_sessions=4, target_sessions=4)
    plan = tuner.plan(remaining_tokens=128, ready_sessions=4, target_sessions=3)
    assert (plan.depth, plan.max_tile_rows) == (9, 5)
    tail = tuner.plan(remaining_tokens=2, ready_sessions=4, target_sessions=3)
    assert tail.depth == 1 and tail.terminal_censored
    assert not _observe(tuner, tail)

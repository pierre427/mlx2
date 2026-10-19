"""Host contract tests: physical plans are not performance qualification."""
import math
from dataclasses import replace

import pytest

from mlx2.runtime.ast_planner import (
    AlternativeExecutionBackend,
    LAUNCH_BUCKET_MIN_SAMPLES,
    LAUNCH_FIT_MIN_SAMPLES,
    ASTExecutionPlanner,
    ExecutionProfile,
    LaunchPlan,
    NumericalEquivalenceOracle,
    ReadyRow,
    RequestWork,
)


def profile(**kwargs):
    return replace(ExecutionProfile(
        binding="artifact-and-source", device="declared-device", build="declared-build",
        max_lanes=8, tile_widths=(1, 2, 3), terminal_widths=(2, 3, 4),
        packed_min_width=1, row_costs=tuple(float(n) for n in range(1, 33)),
        launch_cost=1, context_row_cost=0, draft_cost=1,
    ), **kwargs)


def planner(p=None, **kwargs):
    return ASTExecutionPlanner(p or profile(), tile_cap=3, **kwargs)


def alternative(primary, *, costs=(0.5, 0.5, 0.5, 0.5)):
    oracle = NumericalEquivalenceOracle(
        profile_binding=primary.binding,
        device=primary.device,
        build=primary.build,
        source_revision_token="provider-source",
        evidence="qualification/example/exact-oracle.json",
        equivalent=True,
    )
    return AlternativeExecutionBackend(
        backend="target_tree_single_session",
        profile_binding=primary.binding,
        device=primary.device,
        build=primary.build,
        source_revision_token="provider-source",
        input_widths=(2, 3, 4),
        row_costs=costs,
        oracle=oracle,
    )


def test_single_session_alternative_requires_exact_oracle_and_strictly_lower_cost():
    primary = profile(row_costs=(2.0,) * 32, launch_cost=0)
    with pytest.raises(ValueError, match="numerical oracle"):
        AlternativeExecutionBackend(
            backend="target_tree_single_session",
            profile_binding=primary.binding,
            device=primary.device,
            build=primary.build,
            source_revision_token="provider-source",
            input_widths=(2,),
            row_costs=(1.0, 1.0),
        )
    admitted = replace(primary, alternative_backends=(alternative(primary),))
    selected = planner(admitted).plan_launch(
        [ReadyRow(0, 7, 2, 0)], terminal=True, max_lanes=1
    )
    assert selected.backend == "target_tree_single_session"
    tied = replace(
        primary,
        alternative_backends=(alternative(primary, costs=(2.0,) * 4),),
    )
    assert planner(tied).plan_launch(
        [ReadyRow(0, 7, 2, 0)], terminal=True, max_lanes=1
    ).backend == primary.backend


def test_target_tree_alternative_never_cohorts_sessions_and_reports_actual_use():
    primary = profile(row_costs=(2.0,) * 32, launch_cost=0)
    p = planner(replace(primary, alternative_backends=(alternative(primary),)))
    single = p.plan_launch(
        [ReadyRow(0, 7, 2, 0)], terminal=True, max_lanes=4
    )
    p.observe_launch(single, 0.001)
    two = p.plan_launch(
        [ReadyRow(0, 7, 2, 0), ReadyRow(1, 8, 2, 0)],
        terminal=True,
        max_lanes=4,
    )
    assert single.backend == "target_tree_single_session"
    assert two.backend == "target_tree_single_session"
    assert len(two.indices) == 1
    assert two.batch_count == 2
    receipt = p.receipt()["alternative_backends"][0]
    assert receipt["selected"] is True
    assert receipt["observed_used"] is True
    assert receipt["launches"] == 1
    assert receipt["qualified"] is False


def test_geometry_prior_changes_cold_source_plan_and_explicit_tile_is_pinned():
    serial = planner()
    shared = planner(profile(row_costs=(1.0,) * 32))
    work = [RequestWork(1, 7, 128, 4, acceptance=.8)]
    a, b = serial.source_budgets(work)[0], shared.source_budgets(work)[0]
    assert a != b
    assert a.tile == b.tile == 3
    assert b.branches > a.branches
    assert shared.receipt()["cost_samples"] == 0
    assert shared.receipt()["qualified"] is False


def test_each_request_keeps_its_own_remaining_depth_and_fair_capacity():
    p = planner(profile(row_costs=(1.0,) * 32))
    plans = p.source_budgets([
        RequestWork(1, 1, 20, 4), RequestWork(2, 7, 20, 4, .99),
        RequestWork(3, 0, 20, 4), RequestWork(4, 5, 20, 4, .99),
    ])
    assert [x.uid for x in plans] == [1, 2, 3, 4]
    assert plans[0].depth == 1 and plans[0].branches == 1
    assert plans[1].depth > 1 and plans[1].branches <= 2
    assert plans[2].depth == 0 and plans[2].node_budget == 0
    assert plans[3].depth > 1


def test_short_terminal_reprobe_has_a_primary_plan():
    assert planner().source_budgets([RequestWork(1, 1, 0, 4, reprobe=True)])[0].branches == 1


def test_reprobe_explores_deeper_block_and_learns_its_own_distribution():
    p = planner()
    assert p.source_budgets([RequestWork(1, 7, 0, 4, reprobe=True)])[0].depth == 7
    p.observe_commit(1, 0, 3, source_depth=3)
    p.observe_commit(1, 7, 7, censored=True, source_depth=7)
    assert p.acceptance(1, 3) == 1 / 3
    assert p.acceptance(1, 7) == 8 / 9
    p.release(1)
    assert not p._depth_survival


def test_sliding_attention_cost_stops_growing_at_its_actual_window():
    p = profile(context_windows=((128, .5),))
    assert p.cost([3], [128]) == p.cost([3], [4096])


def test_path_inventory_limit_does_not_reduce_the_drafter_depth_floor():
    p = planner(source_limit=1, minimum_depth=3)
    assert p.source_budgets([RequestWork(1, 7, 0, 4, reprobe=True)])[0].depth >= 3


def test_padding_sensitive_partition_and_oldest_row_fairness():
    p = planner(profile(launch_cost=.1))
    ready = [ReadyRow(i, i, d, 10) for i, d in enumerate([1, 3, 1, 3])]
    plan = p.plan_launch(ready, terminal=True, max_lanes=4)
    assert plan.indices == (0, 2)
    assert plan.input_lengths == (2, 2)
    assert plan.receipt()["padding_rows"] == 0
    assert plan.batch_count == 2
    # Retirement/refill: the oldest survivor is never displaced by arrivals.
    rest = [r for r in ready if r.index not in plan.indices] + [ReadyRow(4, 4, 1, 10)]
    following = p.plan_launch(rest, terminal=True, max_lanes=4)
    assert 1 in following.indices


def test_context_work_is_balanced_across_physical_batches():
    p = planner(profile(context_row_cost=.1))
    ready = [ReadyRow(i, i, 3, c) for i, c in enumerate([1, 2, 100, 101])]
    plan = p.plan_launch(ready, terminal=True, max_lanes=2)
    assert len(plan.indices) == 2
    assert sum(ready[i].context_length for i in plan.indices) == 102
    assert plan.batch_count == 2


def test_equal_cost_nine_row_frontier_balances_occupancy_instead_of_eight_plus_one():
    p = planner(profile(row_costs=(1.0,) * 32))
    plan = p.plan_launch([ReadyRow(i, i, 3, 0) for i in range(9)], terminal=True, max_lanes=8)
    assert sorted(plan.batch_row_counts) == [4, 5]
    assert 0 in plan.indices


def test_adapter_minimum_width_splits_narrow_nonterminal_trunks():
    p = planner(profile(packed_min_width=2))
    ready = [ReadyRow(i, i, 1, 0) for i in range(4)]
    plan = p.plan_launch(ready, terminal=False, max_lanes=4)
    assert plan.indices == (0,) and plan.batch_count == 4
    assert math.isinf(p.profile.cost([1, 1]))
    assert len(p.plan_launch(ready, terminal=True, max_lanes=4).indices) == 4


def test_auto_skips_an_over_capacity_tile_without_discarding_legal_smaller_tiles():
    p = planner(profile(max_position_rows=2), fixed_tile=False)
    ready = [ReadyRow(0, 0, 3, 0)]
    assert p.plan_launch(ready, terminal=True, max_lanes=1).tile <= 2
    pinned = planner(profile(max_position_rows=2))
    with pytest.raises(ValueError, match="no feasible exact physical plan"):
        pinned.plan_launch(ready, terminal=True, max_lanes=1)


def test_physical_tile_prices_survival_not_only_raw_row_throughput():
    p = planner(profile(row_costs=tuple(1 + n * .1 for n in range(1, 33)), launch_cost=0), fixed_tile=False)
    cold = p.plan_launch([ReadyRow(0, 0, 3, 0, .05)], terminal=True, max_lanes=1)
    warm = p.plan_launch([ReadyRow(0, 0, 3, 0, .99)], terminal=True, max_lanes=1)
    assert cold.tile == 1 and warm.tile == 3
    assert cold.acceptance_priors == (.05,)
    assert warm.expected_cohort_progress > cold.expected_cohort_progress


def test_sibling_rows_cannot_multiply_guaranteed_request_progress():
    p = planner()
    siblings = p.plan_launch([ReadyRow(i, 1, 3, 0, 0) for i in range(3)], terminal=True, max_lanes=4)
    requests = p.plan_launch([ReadyRow(i, i, 3, 0, 0) for i in range(3)], terminal=True, max_lanes=4)
    assert siblings.expected_cohort_progress == 1
    assert requests.expected_cohort_progress == 3


def test_compaction_prices_only_supported_row_independent_work():
    raw = profile(launch_cost=0)
    compact = replace(raw, valid_row_costs=tuple(n / 2 for n in range(1, 33)))
    assert raw.cost([1, 3]) == 6
    assert compact.cost([1, 3]) == 5  # four live projection rows, six state rows


def test_measured_shape_cost_refines_later_plans_and_remains_source_local():
    p = planner(profile(row_costs=(1.0,) * 32))
    ready = [ReadyRow(i, i, 3, 10) for i in range(4)]
    plan = p.plan_launch(ready, terminal=True, max_lanes=4)
    for _ in range(8):
        p.observe_launch(plan, .001)
    baseline = p.cost(plan.input_lengths, plan.context_lengths)
    # A different shape calibrates global seconds/work independently.
    other = p.plan_launch(ready[:1], terminal=True, max_lanes=1)
    for _ in range(16):
        p.observe_launch(other, .01)
    assert p.cost(plan.input_lengths, plan.context_lengths) < baseline
    assert p.receipt()["observed_shape_count"] == 2
    assert planner().receipt()["observed_shape_count"] == 0


def test_censored_commits_do_not_invent_failures_and_retirement_clears_state():
    p = planner()
    p.observe_commit(1, 3, 7, censored=True)
    assert p.acceptance(1) == 4 / 5
    p.observe_commit(1, 0, 7)
    assert p.acceptance(1) == 4 / 6
    p.release(1)
    assert p.acceptance(1) == .5


def test_invalid_geometry_fails_closed():
    with pytest.raises(ValueError):
        planner(profile(terminal_widths=(2, 3)))
    with pytest.raises(ValueError):
        planner().plan_launch([ReadyRow(0, 1, 0, 0)], terminal=True, max_lanes=4)
    with pytest.raises(ValueError):
        planner().source_budgets([RequestWork(1, 3, 0, 2, float("nan"))])


def _timed(lengths, seconds, backend="padded_row_exact"):
    cost = float(sum(lengths))
    return LaunchPlan((0,), max(lengths), tuple(lengths), cost, cost, cost, 1, backend), seconds


def test_launch_curve_makes_one_wide_launch_cheaper_than_repeated_narrow_ones():
    p = planner(profile(launch_cost=0), fixed_tile=False)
    row = [ReadyRow(0, 0, 3, 0, .99)]
    # Cold: linear memory work and no curve, so three one-row launches price
    # like one three-row launch and the first law's certainty wins.
    assert p.plan_launch(row, terminal=True, max_lanes=1).tile == 1
    assert p.receipt()["launch_curve"] is None
    # Timed launches of two row counts reveal a 9 ms dispatch floor over a
    # 1 ms-per-unit slope; the curve is observed, never read from tensors.
    for _ in range(LAUNCH_FIT_MIN_SAMPLES):
        p.observe_launch(*_timed([1], .010))
        p.observe_launch(*_timed([2], .011))
    receipt = p.receipt()
    assert receipt["launch_estimator"] == "per_row_count_median_isotonic_v1"
    assert [(c["rows"], c["samples"]) for c in receipt["launch_curve"]] == [(1, 8), (2, 8)]
    assert abs(receipt["fitted_launch_seconds"] - .009) < 1e-9
    assert abs(receipt["fitted_seconds_per_cost_unit"] - .001) < 1e-9
    assert receipt["launch_fit_samples"] == 2 * LAUNCH_FIT_MIN_SAMPLES
    # The never-launched three-row shape is priced on the curve's last
    # segment, in the same seconds-per-unit basis as the launched shapes.
    assert abs(p.cost([3], [0]) * p.observed_scale - .012) < 1e-9
    assert abs(p.cost([1], [0]) * p.observed_scale - .010) < 1e-9
    assert p.plan_launch(row, terminal=True, max_lanes=1).tile == 3


def test_launch_curve_uses_medians_and_needs_two_informative_row_counts():
    p = planner(profile(launch_cost=0), fixed_tile=False)
    for _ in range(4 * LAUNCH_FIT_MIN_SAMPLES):
        p.observe_launch(*_timed([1], .010))
    assert p.receipt()["launch_curve"] is None
    # Without a curve the observed shape keeps its prior-shrunk law.
    scale, count = p._shape_costs[p._shape_key((1,), ())]
    weight = count / (count + 4)
    assert p.cost([1], ()) == pytest.approx(
        1.0 * (1 + weight * (min(4.0, max(0.25, scale / p.observed_scale)) - 1)))
    assert p.cost([3], ()) == 3.0
    # A bucket below the per-bucket minimum contributes no point.
    for _ in range(LAUNCH_BUCKET_MIN_SAMPLES - 1):
        p.observe_launch(*_timed([2], .011))
    assert p.receipt()["launch_curve"] is None
    p.observe_launch(*_timed([2], .011))
    assert [c["rows"] for c in p.receipt()["launch_curve"]] == [1, 2]
    # Heavy-tailed outliers do not move a bucket's median.
    q = planner(profile(launch_cost=0), fixed_tile=False)
    for _ in range(LAUNCH_FIT_MIN_SAMPLES):
        q.observe_launch(*_timed([1], .010))
        q.observe_launch(*_timed([2], .011))
    for _ in range(3):
        q.observe_launch(*_timed([1], .500))
    assert abs(q.receipt()["launch_curve"][0]["seconds"] - .010) < 1e-9
    # Wider launches measured faster than narrow ones pool into one flat
    # level: no shape information, so the curve is not used.
    r = planner(profile(launch_cost=0), fixed_tile=False)
    for _ in range(LAUNCH_FIT_MIN_SAMPLES):
        r.observe_launch(*_timed([1], .020))
        r.observe_launch(*_timed([2], .010))
    assert r.receipt()["launch_curve"] is None
    assert r.plan_launch([ReadyRow(0, 0, 3, 0, .99)], terminal=True, max_lanes=1).tile == 1
    # Three buckets with one violator: the violating pair is pooled, the
    # curve stays nondecreasing, and interpolation between points is linear.
    t = planner(profile(launch_cost=0), fixed_tile=False)
    for _ in range(LAUNCH_FIT_MIN_SAMPLES):
        t.observe_launch(*_timed([1], .010))
        t.observe_launch(*_timed([2], .014))
        t.observe_launch(*_timed([4], .012))
    curve = t.receipt()["launch_curve"]
    assert [c["rows"] for c in curve] == [1, [2, 4]]
    assert abs(curve[1]["seconds"] - .013) < 1e-9
    assert abs(t._curve_seconds(0.5) - .010) < 1e-9
    assert abs(t._curve_seconds(3.0) - .013) < 1e-9


def test_shared_trunk_is_priced_once_and_learned_uplift_owns_branch_width():
    # Shared weight reads make extra rows cheap, so branches are affordable
    # and only their observed uplift decides whether they are worth it.
    p = planner(profile(row_costs=(1.0,) * 32))
    # Four branches over depth 4 with a two-token trunk: one trunk-plus-chain
    # row, one suffix row, two one-token rows. Never four full chains.
    assert p._branch_rows(4, 2, 0, 3) == [3, 1, 1, 1]
    assert p._branch_rows(4, 2, 3, 4) == [1, 1]
    assert p._branch_rows(1, 2, 0, 3) == [3]
    # The former law charged four full chains: [3,3,3,3] then the terminal
    # [2,2,2,2] slab. Trunk sharing leaves [3,1,1,1] then [2,2].
    linear = planner(profile(launch_cost=0))
    former = linear.cost([3] * 4, [0] * 4) + linear.cost([2] * 4, [3] * 4)
    assert linear._round_cost(4, 4, 3, 0, 1.0, trunk=2) == linear.cost([3, 1, 1, 1], [0] * 4) + linear.cost([2, 2], [3] * 2)
    assert linear._round_cost(4, 4, 3, 0, 1.0, trunk=2) < former
    # The union prior lets branches in cold; exact multi-branch commits that
    # bought nothing teach the planner to stop paying for them.
    work = RequestWork(7, 6, 0, 4, acceptance=.6, trunk=2)
    cold = p.source_budgets([work])[0]
    assert cold.branches > 1 and cold.reason == "hardware_cohort_cost_prior"
    for _ in range(12):
        p.observe_branches(7, extra_rows=6, uplift=0)
    taught = p.source_budgets([work])[0]
    assert taught.branches == 1 and taught.reason == "learned_branch_uplift"
    assert p.branch_uplift(7) == (0.0, 12)
    # Branches that keep paying off stay selected; a reprobe keeps its width.
    for _ in range(40):
        p.observe_branches(7, extra_rows=6, uplift=3)
    assert p.source_budgets([work])[0].branches > 1
    assert p.source_budgets([replace(work, reprobe=True)])[0].branches == 4
    # Primary-only rounds carry no branch evidence and retirement clears it.
    p.observe_branches(7, extra_rows=0, uplift=0)
    p.release(7)
    assert p.branch_uplift(7) == (None, 0)
    assert p.receipt()["branch_authority"] == "planner_learned_uplift_v1"
    with pytest.raises(ValueError):
        p.observe_branches(7, extra_rows=-1, uplift=0)


def test_collision_survival_is_per_position_lagged_and_request_owned():
    p = planner(profile(row_costs=(1.0,) * 32))
    work = RequestWork(7, 4, 0, 1, acceptance=.2)
    ordinary = p.source_budgets([work])[0]
    assert ordinary.survival_authority == "lagged_acceptance"
    assert ordinary.survival_priors == (.2,) * ordinary.depth

    explicit = planner(profile(row_costs=(1.0,) * 32)).source_budgets([
        replace(work, collision_probabilities=(.6, .5, .4, .3))
    ])[0]
    assert explicit.survival_priors == (.6, .5, .4, .3)[:explicit.depth]

    p.observe_collision_probabilities(7, (.9, .8, .7))
    refined = p.source_budgets([work])[0]
    assert refined.survival_authority == (
        "proposal_collision_probability_lagged_v1"
    )
    # The previous round only reached three positions. The unobserved suffix
    # keeps the ordinary lagged acceptance prior instead of being fabricated.
    assert refined.survival_priors[:3] == (.9, .8, .7)
    if refined.depth > 3:
        assert refined.survival_priors[3:] == (.2,) * (refined.depth - 3)
    receipt = p.receipt()
    assert receipt["collision_feature"] == "sum_q_squared_per_position_v1"
    assert receipt["collision_feature_observations"] == 1
    assert receipt["collision_feature_phase"] == (
        "prior_completed_round_before_current_draw"
    )
    p.release(7)
    assert p.receipt()["collision_feature_requests"] == 0
    with pytest.raises(ValueError, match="collision probabilities"):
        p.observe_collision_probabilities(7, (.8, 0))


def test_collision_history_can_outlive_a_short_terminal_budget():
    p = planner()
    p.observe_collision_probabilities(9, (.9, .8, .7, .6))
    terminal = p.source_budgets([RequestWork(9, 0, 0, 1)])[0]
    assert terminal.depth == 0 and terminal.survival_priors == ()

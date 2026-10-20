"""Host feedback exercise; no real-artifact capability or timing is minted."""

from mlx2.runtime.ast_planner import (
    LAUNCH_FIT_MIN_SAMPLES,
    ASTExecutionPlanner,
    ExecutionProfile,
    LaunchPlan,
    ReadyRow,
    RequestWork,
)


def test_observed_costs_and_request_acceptance_choose_five_or_three():
    profile = ExecutionProfile(
        binding="synthetic-host-feedback-not-serving-evidence",
        device="cpu",
        build="host-test",
        max_lanes=4,
        tile_widths=(1, 2, 3, 4, 5),
        terminal_widths=(2, 3, 4, 5, 6),
        packed_min_width=1,
        row_costs=tuple(float(n) for n in range(1, 65)),
        launch_cost=0,
        context_row_cost=0,
        draft_cost=1,
    )
    planner = ASTExecutionPlanner(profile, tile_cap=5, fixed_tile=False, source_limit=1)
    # Synthetic launch seeds derived by dividing the e54dabcb static single
    # chain total timings by its launch counts. These are NOT measured
    # per-launch costs and are never imported into serving or qualification.
    seeds = {
        3: (0.19708620899473317 + 0.19918887497624382) / 2 / 5,
        5: (0.1565269170096144 + 0.15522829099791124) / 2 / 3,
    }
    for _ in range(LAUNCH_FIT_MIN_SAMPLES):
        for width, seconds in seeds.items():
            plan = LaunchPlan(
                (0,), width, (width,), width, width, width, 1, profile.backend
            )
            planner.observe_launch(plan, seconds)
    for _ in range(8):
        planner.observe_commit(10, 15, 15, censored=True)
    for _ in range(64):
        planner.observe_commit(11, 0, 1)
    good = planner.acceptance(10)
    poor = planner.acceptance(11)
    assert good > 0.98 and poor < 0.04
    high = planner.plan_launch(
        [ReadyRow(0, 10, 15, 0, good)], terminal=False, max_lanes=1
    )
    low = planner.plan_launch(
        [ReadyRow(0, 11, 15, 0, poor)], terminal=False, max_lanes=1
    )
    assert (high.tile, low.tile) == (5, 3)
    assert high.input_lengths == (5,)  # No nonterminal bonus.
    terminal = planner.plan_launch(
        [ReadyRow(0, 10, 2, 0, good)], terminal=True, max_lanes=1
    )
    assert terminal.input_lengths == (3,)  # Exactly one terminal bonus row.
    budgets = [
        planner.source_budgets(
            [RequestWork(uid, 7, 0, 1, acceptance=planner.acceptance(uid))]
        )[0]
        for uid in (10, 11)
    ]
    assert budgets[0].tile == 5 and budgets[1].tile == 3
    assert budgets[0].depth != budgets[1].depth
    assert all(budget.branches == 1 and budget.depth <= 7 for budget in budgets)
    planner.release(10)
    assert planner.acceptance(10) == 0.5
    assert planner.acceptance(11) == poor

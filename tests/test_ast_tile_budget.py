"""Measured tree-tile size from counterfactual acceptance and fitted cost."""

import pytest

from mlx2.runtime.ast_tile_budget import TileBudgetController


def _drive(controller, *, cost, path_len, rounds=120):
    chosen = []
    for _ in range(rounds):
        n = controller.choose(1, 15)
        chosen.append(n)
        # Best-first ranks: the committed path takes ranks 0, 1, 2, ...
        ranks = list(range(min(path_len, n)))
        controller.observe(1, n, ranks, cost(n))
    return chosen


def test_steep_cost_and_shallow_paths_choose_a_small_tile():
    controller = TileBudgetController()
    chosen = _drive(controller, cost=lambda n: 0.010 + 0.004 * (n + 1), path_len=2)
    assert chosen[:6] == [15] * 6                  # warmup at the knob maximum
    assert controller.decisions[1] <= 5
    assert chosen[-1] == 15 or chosen[-1] <= 5      # refresh rounds return to max


def test_flat_cost_keeps_the_full_tile():
    controller = TileBudgetController()
    _drive(controller, cost=lambda n: 0.040 + 0.0002 * (n + 1), path_len=15)
    assert controller.decisions[1] == 15


def test_counterfactual_tokens_count_only_ranks_below_each_size():
    controller = TileBudgetController()
    controller.observe(1, 15, [0, 1, 4, 9], 0.05)
    tokens = controller.receipt()["1"]["tokens_per_round"]
    assert tokens["3"] == 3 and tokens["5"] == 4 and tokens["11"] == 5 and tokens["15"] == 5


@pytest.mark.parametrize("counterfactual", [True, False])
def test_snapshot_restore_rolls_back_an_observation_and_stays_reusable(counterfactual):
    controller = TileBudgetController(counterfactual=counterfactual)
    _drive(controller, cost=lambda n: 0.010 + 0.004 * (n + 1), path_len=2, rounds=20)
    before = controller.receipt()
    choice = controller.choose(1, 15)
    snapshot = controller.snapshot()
    controller.observe(1, 15, [0, 1, 2], 0.5)
    controller.observe(2, 7, [0], 0.3)
    assert controller.receipt() != before
    controller.restore(snapshot)
    assert controller.receipt() == before
    assert controller.choose(1, 15) == choice
    # The snapshot is independent of the restored state.
    controller.observe(1, 15, [0], 0.5)
    controller.restore(snapshot)
    assert controller.receipt() == before


@pytest.mark.parametrize("counterfactual", [True, False])
def test_deferred_commit_cost_keeps_prelaunched_budget_and_counts_acceptance_once(counterfactual):
    controller = TileBudgetController(counterfactual=counterfactual)
    _drive(controller, cost=lambda n: 0.01 + n * 0.004, path_len=2, rounds=20)
    controller.observe(1, 15, [0, 1], None)
    before = controller.receipt()["1"]
    pending = controller.choose(1, 15)
    snapshot = controller.snapshot()
    controller.observe_cost(1, 15, 9.0)
    after = controller.receipt()["1"]
    assert after["rounds"] == before["rounds"]
    assert after["tokens_per_round"] == before["tokens_per_round"]
    assert after["cost_ms"] != before["cost_ms"]
    assert controller.choose(1, 15) == pending
    controller.restore(snapshot)
    assert controller.receipt()["1"] == before


def test_grafted_tree_yield_uses_only_the_budget_actually_verified():
    controller = TileBudgetController(counterfactual=False)
    controller.observe(1, 15, [0, 2, 6], 0.1)
    assert controller.receipt()["1"]["tokens_per_round"] == {"15": 4.0}
    assert controller.receipt()["1"]["yield_observation"] == "verified_budget_only"
    controller.observe(1, 5, [0, 1], 0.05)
    assert controller.receipt()["1"]["tokens_per_round"] == {"5": 3.0, "15": 4.0}


@pytest.mark.parametrize("maximum", [15, 31])
def test_measured_yields_explore_all_budgets_and_find_the_fastest(maximum):
    controller = TileBudgetController(counterfactual=False)
    visited = set()
    for _ in range(500):
        nodes = controller.choose(1, maximum)
        visited.add(nodes)
        controller.observe(1, nodes, [0, 1], 0.010 + 0.004 * (nodes + 1))
    expected = {3, 5, 7, 9, 11, 15}
    if maximum == 31:
        expected |= {23, 31}
    assert visited == expected
    # Three emitted tokens at every size: n=3 costs26ms versus42ms at n=7.
    assert controller.decisions[1] == 3


def test_measured_yield_refresh_revisits_candidates_with_four_lane_observations():
    controller = TileBudgetController(counterfactual=False)
    for _ in range(150):
        nodes = controller.choose(4, 15)
        for _lane in range(4):
            controller.observe(4, nodes, [0, 1], 0.010 + 0.004 * (nodes + 1))
    verified = controller.receipt()["4"]["verified_nodes"]
    # Initial exploration needs only one four-lane cohort per size. Every
    # smaller size must later be sampled again, even when it loses to n=3.
    assert all(verified[str(n)] >= 8 for n in (3, 5, 7, 9, 11))


def test_restore_preserves_a_refresh_probe_without_repeating_it():
    controller = TileBudgetController(counterfactual=False, warmup=0, probe=1, refresh=8)
    _drive(controller, cost=lambda n: 0.010 + 0.004 * (n + 1), path_len=2, rounds=17)
    pending = controller.choose(1, 15)
    assert pending == 7
    snapshot = controller.snapshot()
    controller.observe(1, pending, [0, 1], 0.010 + 0.004 * (pending + 1))
    expected = controller.choose(1, 15)
    assert expected == 3
    controller.restore(snapshot)
    assert controller.choose(1, 15) == pending
    controller.observe(1, pending, [0, 1], 0.010 + 0.004 * (pending + 1))
    assert controller.choose(1, 15) == expected


def test_smaller_cost_probe_collects_the_configured_sample_count():
    controller = TileBudgetController()
    for _ in range(controller.warmup):
        controller.observe(1, 15, [0, 1], 0.104)
    for _ in range(controller.probe):
        assert controller.choose(1, 15) == 7
        controller.observe(1, 7, [0, 1], 0.102)
    # The smaller tile improves the fitted rate by only 2.97%, below 8%.
    assert controller.choose(1, 15) == 15


def test_reduction_margin_compares_against_the_maximum_rate():
    for margin, expected in ((1.0, 3), (1.08, 15), (100.0, 15)):
        controller = TileBudgetController(probe=1, margin=margin)
        for _ in range(controller.warmup):
            controller.observe(1, 15, [0, 1], 0.104)
        assert controller.choose(1, 15) == 7
        controller.observe(1, 7, [0, 1], 0.102)
        assert controller.choose(1, 15) == expected


def test_reduction_margin_does_not_compound_across_intermediate_sizes():
    controller = TileBudgetController(warmup=0, probe=0, refresh=10000, alpha=1)
    for index in range(100):
        ranks = [0, 9, 10]
        if index < 38:
            ranks.append(1)
        if index < 6:
            ranks.append(8)
        if index < 56:
            ranks.append(14)
        controller.observe(1, 15, ranks, 0.05)
    controller.observe(1, 7, [0], 0.03)
    # Expected rates: maximum 100, eleven nodes 111, three nodes 118.81.
    # Three nodes clears the maximum's margin and beats every alternative.
    assert controller.choose(1, 15) == 3

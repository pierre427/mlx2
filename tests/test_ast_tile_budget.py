"""Measured tree-tile size from counterfactual acceptance and fitted cost."""

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


def test_snapshot_restore_rolls_back_an_observation_and_stays_reusable():
    controller = TileBudgetController()
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

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


# Robust cost mode (MLX2_AST_TILE_BUDGET_COST=robust, opt-in).

def _steep(n):
    return 0.010 + 0.004 * (n + 1)


def _flat(n):
    return 0.076 + 0.0013 * (n + 1)


def test_unknown_cost_mode_fails_closed():
    with pytest.raises(ValueError, match="cost_mode"):
        TileBudgetController(cost_mode="median")
    with pytest.raises(ValueError, match="cost_skip"):
        TileBudgetController(cost_mode="robust", cost_window=0)


def test_robust_drops_first_samples_and_takes_the_window_median():
    controller = TileBudgetController(cost_mode="robust", cost_skip=2, cost_window=3, cost_warmup_rounds=0)
    for seconds in (0.9, 0.5, 0.10, 0.30, 0.11, 0.12):
        controller.observe_cost(1, 15, seconds)
    receipt = controller.receipt()["1"]
    assert receipt["cost_skipped"] == {"15": 2}
    assert receipt["cost_samples"] == {"15": 4}
    # Window keeps the last three kept samples (0.30, 0.11, 0.12).
    assert receipt["cost_ms"] == {"15": 120.0}
    assert receipt["cost_mode"] == "robust"


def _with_compile_outlier(cost, outlier_size, outlier=0.94):
    """``cost`` with one first-use stall at ``outlier_size``."""

    hit = {"done": False}

    def wrapped(n):
        if n == outlier_size and not hit["done"]:
            hit["done"] = True
            return outlier
        return cost(n)
    return wrapped


def test_first_use_stall_at_the_probe_flattens_the_ewma_fit_but_not_robust():
    # On arm, 2026-11-12: one 940 ms first sample at a small size turned the
    # slope negative (clamped flat), which hides a steep cost curve.
    ewma = TileBudgetController()
    _drive(ewma, cost=_with_compile_outlier(_steep, 7), path_len=2, rounds=60)
    assert ewma.cost_curve(1)[1] == 0.0
    assert ewma.decisions[1] == 15
    robust = TileBudgetController(cost_mode="robust")
    _drive(robust, cost=_with_compile_outlier(_steep, 7), path_len=2, rounds=60)
    assert robust.decisions[1] <= 5
    assert robust.cost_curve(1)[1] > 0.003


def test_one_slow_maximum_sample_sends_ewma_small_but_not_robust():
    # Off arm, 2026-11-12: a flat curve (15 nodes worth ~2x the tokens of 3)
    # where one slow maximum-size round should not leave the full tile.
    def run(controller):
        chosen = []
        for index in range(400):
            n = controller.choose(1, 15)
            chosen.append(n)
            stall = index == 20 and n == 15
            ranks = [rank for rank in (0, 1, 2, 4, 6, 9, 12) if rank < n]
            controller.observe(1, n, ranks, 0.6 if stall else _flat(n))
        return chosen

    ewma = TileBudgetController()
    ewma_chosen = run(ewma)
    assert sum(1 for n in ewma_chosen if n < 15) > 50
    robust = TileBudgetController(cost_mode="robust")
    robust_chosen = run(robust)
    # Only the cost-probe rounds (initial burst, one per refresh epoch) leave
    # the maximum.
    assert sum(1 for n in robust_chosen if n < 15) <= (
        robust.cost_skip + robust.probe + 400 // robust.refresh
    )


def test_robust_fit_ignores_sizes_with_too_few_kept_samples():
    controller = TileBudgetController(cost_mode="robust", cost_skip=0, min_cost_samples=2, cost_warmup_rounds=0)
    for _ in range(3):
        controller.observe_cost(1, 15, 0.096)
        controller.observe_cost(1, 7, 0.089)
    controller.observe_cost(1, 4, 0.9)   # one end-of-generation truncated round
    a, b = controller.cost_curve(1)
    assert b == pytest.approx(0.007 / 8)
    assert a == pytest.approx(0.096 - b * 16)


def test_robust_probe_collects_kept_samples_after_the_warmup_drop():
    controller = TileBudgetController(cost_mode="robust", cost_warmup_rounds=0)
    for _ in range(controller.warmup):
        controller.observe(1, 15, [0, 1], 0.104)
    probes = 0
    while controller.choose(1, 15) == 7:
        controller.observe(1, 7, [0, 1], 0.102)
        probes += 1
        assert probes < 20
    assert probes == controller.cost_skip + controller.probe


@pytest.mark.parametrize("counterfactual", [True, False])
def test_robust_snapshot_restore_covers_window_and_skips(counterfactual):
    controller = TileBudgetController(counterfactual=counterfactual, cost_mode="robust")
    _drive(controller, cost=_steep, path_len=2, rounds=20)
    before = controller.receipt()
    choice = controller.choose(1, 15)
    snapshot = controller.snapshot()
    controller.observe(1, 15, [0, 1, 2], 0.5)
    controller.observe(1, 9, [0], 0.3)
    controller.observe(1, 9, [0], 0.3)
    controller.observe(1, 9, [0], 0.3)
    assert controller.receipt() != before
    controller.restore(snapshot)
    assert controller.receipt() == before
    assert controller.choose(1, 15) == choice


def test_trace_sink_receives_every_cost_sample():
    rows = []
    controller = TileBudgetController(cost_mode="robust", trace=rows.append, cost_warmup_rounds=0)
    controller.observe(1, 15, [0], 0.05)
    controller.observe(1, 15, [0], None)
    controller.observe_cost(2, 7, 0.0125)
    assert rows == [(1, 1, 15, 50.0, None), (2, 0, 7, 12.5, None)]


def test_robust_probe_completes_from_commit_scope_samples_alone():
    # Serving falls back to commit-scope seconds whenever no clean round
    # period exists (interleaved cohorts), so kept samples always arrive.
    controller = TileBudgetController(cost_mode="robust")
    chosen = _drive(controller, cost=_flat, path_len=12, rounds=200)
    initial = controller.cost_skip + controller.probe
    assert initial <= chosen.count(7) <= initial + 200 // controller.refresh


def test_robust_fit_never_mixes_period_and_commit_scope_costs():
    controller = TileBudgetController(cost_mode="robust", cost_skip=0, cost_warmup_rounds=0)
    for _ in range(3):
        controller.observe_cost(1, 15, 0.120, scope="period")   # + host work
        controller.observe_cost(1, 15, 0.096)
        controller.observe_cost(1, 7, 0.089)                    # probe: commit only
    receipt = controller.receipt()["1"]
    assert receipt["fit_scope"] == "commit"                     # the period covers one size
    assert receipt["cost_ms"] == {"7": 89.0, "15": 96.0}
    assert receipt["cost_samples_by_scope"] == {"period": {"15": 3}, "commit": {"7": 3, "15": 3}}
    for _ in range(2):
        controller.observe_cost(1, 7, 0.111, scope="period")
    receipt = controller.receipt()["1"]
    assert receipt["fit_scope"] == "period"                     # period first once it fits
    assert receipt["cost_ms"] == {"7": 111.0, "15": 120.0}
    with pytest.raises(ValueError, match="scope"):
        controller.observe_cost(1, 7, 0.1, scope="wall")


def test_robust_probe_continues_until_the_fit_scope_covers_it():
    # The probe burst overlapped a prefill (commit fallbacks) while the
    # maximum's samples are clean periods: probe until the period covers 7.
    controller = TileBudgetController(cost_mode="robust", cost_skip=0, cost_warmup_rounds=0)
    for _ in range(controller.warmup):
        controller.observe(1, 15, [0, 1, 2, 3, 4, 5], None)
        controller.observe_cost(1, 15, 0.120, scope="period")
    for _ in range(controller.probe):
        assert controller.choose(1, 15) == 7
        controller.observe(1, 7, [0, 1, 2, 3, 4, 5], None)
        controller.observe_cost(1, 7, 0.080)
    assert controller.choose(1, 15) == 7                         # period lacks 7
    for _ in range(2):
        controller.observe(1, 7, [0, 1, 2, 3, 4, 5], None)
        controller.observe_cost(1, 7, 0.110, scope="period")
    assert controller.receipt()["1"]["fit_scope"] == "period"
    assert controller.cost_curve(1) is not None
    assert controller.choose(1, 15) in (7, 15)


def test_robust_probe_extension_is_bounded():
    controller = TileBudgetController(cost_mode="robust", cost_skip=0, cost_warmup_rounds=0)
    for _ in range(controller.warmup):
        controller.observe(1, 15, [0, 1], None)
        controller.observe_cost(1, 15, 0.120, scope="period")
    probes = 0
    while controller.choose(1, 15) == 7:
        controller.observe(1, 7, [0, 1], None)
        controller.observe_cost(1, 7, 0.080)                     # never a clean period
        probes += 1
        assert probes <= 4 * controller.probe
    assert probes == 4 * controller.probe


def test_robust_waits_out_process_warmup_before_probing():
    # 2026-11-12 trace: ~130 ms rounds at every size for the first rounds,
    # ~97 ms after.  A probe inside that window priced 7 above 15.
    def cost(n, index):
        return (0.130 if index < 20 else 0.076) + 0.0013 * (n + 1)

    controller = TileBudgetController(cost_mode="robust")
    chosen = []
    for index in range(300):
        n = controller.choose(1, 15)
        chosen.append(n)
        controller.observe(1, n, [0, 1, 2, 3, 4, 5, 6, 8], cost(n, index))
    receipt = controller.receipt()["1"]
    assert all(n == 15 for n in chosen[:controller.cost_warmup_rounds])
    assert receipt["cost_warmup_dropped"] == controller.cost_warmup_rounds - 1
    a, b = controller.cost_curve(1)
    assert b == pytest.approx(0.0013, rel=1e-6)    # steady-state slope recovered
    assert a == pytest.approx(0.076, rel=1e-6)


def test_robust_reprobes_the_probe_size_once_per_refresh_epoch():
    controller = TileBudgetController(cost_mode="robust", cost_warmup_rounds=0)
    chosen = _drive(controller, cost=_flat, path_len=12, rounds=48 * 6)
    initial = controller.cost_skip + controller.probe
    # The initial burst, then one fresh sample per refresh epoch.
    assert initial < chosen.count(7) <= initial + 6


def test_robust_measured_yields_explore_after_warmup_and_fit():
    # counterfactual=False probes every candidate; after the process warm-up
    # each needs kept cost samples, or the fit would stay empty (maximum).
    controller = TileBudgetController(counterfactual=False, cost_mode="robust")
    chosen = []
    for index in range(600):
        n = controller.choose(1, 15)
        chosen.append(n)
        cost = 0.110 if index < 24 else 0.030 + 0.004 * (n + 1)
        controller.observe(1, n, [0, 1], cost)
    assert controller.cost_curve(1) is not None
    assert controller.decisions[1] == 3
    assert chosen[-100:].count(3) > 80

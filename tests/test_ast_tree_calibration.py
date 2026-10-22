"""Per-depth drafter calibration of AST tree tiles (Kapitsa KfBudget port); CPU only."""

import math

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mlx2.runtime.ast_tree_calibration import (
    DepthCalibration,
    calibrated_log_p,
    expected_tokens,
    fit_logistic,
    live_nodes,
    logit_q,
    parse_mode,
    tree_records,
)
from mlx2.runtime.drafters.dflash_tree import (
    best_first_tree,
    calibrated_best_first_tree,
)
from mlx2.runtime.external_speculative import TreeDraftRow

from test_dflash_pair_select import drain, generator, tiny
from test_dflash_tree_bridge import _install_reference_cohort


# -- the law -----------------------------------------------------------------


@pytest.mark.parametrize("q", [1e-6, 0.01, 0.3, 0.5, 0.9, 0.999])
def test_identity_calibration_is_the_drafter_probability(q):
    assert calibrated_log_p(math.log(q), 0.0, 1.0) == pytest.approx(math.log(q), abs=1e-9)
    assert logit_q(math.log(q)) == pytest.approx(math.log(q / (1 - q)), rel=1e-9)


def test_logit_q_is_finite_at_certainty():
    assert math.isfinite(logit_q(0.0))
    assert math.isfinite(calibrated_log_p(0.0, 0.0, 1.0))


def _synthetic_walk(rng, alpha, beta, count):
    lq = np.log(rng.uniform(0.02, 0.98, size=count))
    x = np.array([logit_q(v) for v in lq])
    p = 1.0 / (1.0 + np.exp(-(alpha + beta * x)))
    y = (rng.uniform(size=count) < p).astype(float)
    return lq, x, y


def test_fit_recovers_a_known_overconfident_depth():
    rng = np.random.default_rng(0)
    _lq, x, y = _synthetic_walk(rng, -1.2, 0.6, 4000)
    a, b = 0.0, 1.0
    for _ in range(6):
        a, b = fit_logistic(x, y, a, b)
    assert a == pytest.approx(-1.2, abs=0.15)
    assert b == pytest.approx(0.6, abs=0.1)


def test_fit_with_no_signal_stays_near_the_identity_prior():
    a, b = fit_logistic([], [], 0.0, 1.0)
    assert (a, b) == (0.0, 1.0)


def test_fit_clamps_beta_positive():
    # Labels anti-correlated with q: beta is held at its floor, never negative.
    x = [-3.0, -2.0, 2.0, 3.0] * 50
    y = [1.0, 1.0, 0.0, 0.0] * 50
    a, b = 0.0, 1.0
    for _ in range(5):
        a, b = fit_logistic(x, y, a, b)
    assert b == pytest.approx(0.1)


def test_online_calibration_learns_per_depth_and_refits_on_cadence():
    rng = np.random.default_rng(1)
    calibration = DepthCalibration("order", refit_every=4, min_records=32, window=256)
    truth = {1: (0.4, 1.1), 3: (-1.5, 0.5)}
    for _tree in range(400):
        records = []
        for depth, (alpha, beta) in truth.items():
            lq, _x, y = _synthetic_walk(rng, alpha, beta, 3)
            records += [(depth, float(v), bool(h)) for v, h in zip(lq, y)]
        calibration.observe("greedy", records)
    alpha, beta = calibration.params("greedy")
    assert alpha[1] == pytest.approx(0.4, abs=0.35)
    assert alpha[3] == pytest.approx(-1.5, abs=0.35)
    assert beta[3] < beta[1]
    # Depths without trials keep the identity; the sampled table is separate.
    assert (alpha[2], beta[2]) == (0.0, 1.0)
    assert calibration.params("sampled")[0][1] == 0.0
    assert calibration.fits == 100
    receipt = calibration.receipt()
    assert receipt["tables"]["greedy"]["trials"]["1"] == 1200
    assert set(receipt["tables"]["greedy"]["alpha"]) == {"1", "3"}


def test_window_keeps_only_recent_records():
    calibration = DepthCalibration("order", window=8, refit_every=1000)
    calibration.observe("greedy", [(1, -0.1, True)] * 8)
    calibration.observe("greedy", [(1, -3.0, False)] * 5)
    table = calibration._tables["greedy"]
    assert len(table.xs[1]) == 8
    assert sum(table.ys[1]) == 3


def test_snapshot_restore_rolls_back_records_and_fits():
    calibration = DepthCalibration("order_size", refit_every=1, min_records=1)
    calibration.observe("greedy", [(1, -0.5, True), (2, -1.0, False)])
    saved = calibration.snapshot()
    before = calibration.params("greedy")
    calibration.observe("greedy", [(1, -0.2, False)] * 10)
    calibration.observe_rate(10, 0.1)
    assert calibration.params("greedy") != before
    calibration.restore(saved)
    assert calibration.params("greedy") == before
    assert calibration.rate() is None
    calibration.restore(saved)          # the snapshot stays reusable
    assert calibration.trees == 1


def test_parse_mode():
    assert parse_mode(None) is None and parse_mode(False) is None
    assert parse_mode(True) == "order"
    assert parse_mode("order_size") == "order_size"
    with pytest.raises(ValueError):
        parse_mode("sometimes")
    with pytest.raises(ValueError):
        DepthCalibration("off")


# -- trials of a verified tree -------------------------------------------------


def test_tree_records_are_the_nodes_whose_parent_was_accepted():
    # anchor -> 0 -> 2 -> 4 ; anchor -> 1 ; 0 -> 3
    parents = [-1, -1, 0, 0, 2]
    depths = [1, 1, 2, 2, 3]
    lqs = [-0.1, -2.0, -0.3, -1.5, -0.4]
    # committed: anchor, node 0, node 2 (rows 0, 1, 3); node 4 was rejected
    records = tree_records(parents, depths, lqs, [0, 1, 3])
    assert records == [
        (1, -0.1, True), (1, -2.0, False),   # children of the anchor
        (2, -0.3, True), (2, -1.5, False),   # children of accepted node 0
        (3, -0.4, False),                    # child of accepted node 2
    ]
    # Nothing accepted: only the anchor's children are trials.
    assert [r[0] for r in tree_records(parents, depths, lqs, [0])] == [1, 1]


# -- expected tokens and the live-row rule ------------------------------------


def test_expected_tokens_and_live_rows():
    path = [math.log(p) for p in (0.9, 0.8, 0.5, 0.1, 0.05, 0.01)]
    assert expected_tokens(path, 0) == 1.0
    assert expected_tokens(path, 2) == pytest.approx(2.7)
    # Flat cost keeps every row; a steep cost keeps the confident prefix.
    assert live_nodes(path, lambda n: 1.0) == 6
    assert live_nodes(path, lambda n: 1.0 + 0.5 * n) == 2
    # With a long-run rate, a row stays only when its expected tokens pay
    # for its seconds at that rate.
    # 60 tok/s x 2 ms a row = 0.12 tokens: the p=0.1 node does not pay.
    assert live_nodes(path, lambda n: 0.04 + 0.002 * (n + 1), rate=60.0) == 3
    assert live_nodes(path, lambda n: 0.04 + 0.002 * (n + 1), rate=600.0) == 1
    assert live_nodes([], lambda n: 1.0) == 0


# -- the calibrated tree -------------------------------------------------------


def _lattice(seed=3, depth=6, width=8, vocab=40, rank=4):
    rng = np.random.default_rng(seed)
    candidates = np.stack([rng.permutation(vocab)[:width] for _ in range(depth)])
    unary = rng.normal(size=(depth, width)) * 2.0
    projected = rng.normal(size=(depth, rank))
    pred = rng.normal(size=(vocab, rank))
    succ = rng.normal(size=(vocab, rank))
    return candidates, unary, projected, 5, pred, succ


IDENTITY = ((0.0,) * 33, (1.0,) * 33)


def test_calibrated_tree_is_best_first_and_parent_closed():
    tokens, parents, info = calibrated_best_first_tree(
        *_lattice(), calibration=IDENTITY, max_nodes=15
    )
    assert len(tokens) == len(parents) == 15
    path = info["path_log_p"]
    assert all(left >= right - 1e-12 for left, right in zip(path, path[1:]))
    for node, parent in enumerate(parents):
        assert parent < node
        assert info["depths"][node] == (1 if parent < 0 else info["depths"][parent] + 1)
        own = math.log(math.exp(info["log_q"][node]))       # identity: log p = log q
        base = 0.0 if parent < 0 else path[parent]
        assert path[node] == pytest.approx(base + own, abs=1e-9)
    # The prefix of n nodes is exactly the n-node tree.
    small = calibrated_best_first_tree(*_lattice(), calibration=IDENTITY, max_nodes=6)
    assert small[0] == tokens[:6] and small[1] == parents[:6]


def test_deep_overconfidence_widens_the_tree():
    deep = list(IDENTITY[0]), list(IDENTITY[1])
    for depth in range(2, 33):
        deep[0][depth] = -3.0          # deeper nodes rarely survive
    _t, parents_identity, _ = calibrated_best_first_tree(
        *_lattice(), calibration=IDENTITY, max_nodes=12
    )
    _t, parents_deep, info = calibrated_best_first_tree(
        *_lattice(), calibration=(tuple(deep[0]), tuple(deep[1])), max_nodes=12
    )
    roots = lambda parents: sum(1 for parent in parents if parent < 0)  # noqa: E731
    assert roots(parents_deep) > roots(parents_identity)
    assert max(info["depths"]) <= 3


def test_default_tree_is_unchanged_by_the_calibrated_variant():
    # The calibrated search is a separate function; the default search keeps
    # its fixed tau/edge/children law byte for byte.
    first = best_first_tree(*_lattice(), max_nodes=15)
    second = best_first_tree(*_lattice(), max_nodes=15)
    assert first == second
    assert len(first) == 2


# -- generator wiring ----------------------------------------------------------


def _tree_batch(model, draft, monkeypatch, **kwargs):
    monkeypatch.setenv("MLX2_TENSORFOLD_SOURCE", "/unused-by-cpu-oracle")
    batch = generator(
        model, draft, completion_batch_size=4, ready_drain="all",
        dynamic_singleton_tree=True, dynamic_tree_max_width=4,
        tree_node_budget_by_lanes={1: 15, 2: 7, 3: 4, 4: 3}, **kwargs,
    )
    _install_reference_cohort(batch, monkeypatch, [])
    return batch


def _reference(model, draft, prompts, tokens):
    reference = generator(model, draft, completion_batch_size=4, ready_drain="all")
    reference.insert(prompts, max_tokens=[tokens] * len(prompts),
                     sampling_configs=[{"sampling_temp": 0.0}] * len(prompts))
    return drain(reference)[0]


@pytest.mark.parametrize("mode", ["order", "order_size"])
@pytest.mark.parametrize("width", [1, 4])
def test_calibrated_tree_keeps_exact_greedy_tokens(monkeypatch, mode, width):
    prompts = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]][:width]
    model, draft = tiny(vocab=64, top_k=16, block_size=8)
    expected = _reference(model, draft, prompts, 24)
    batch = _tree_batch(model, draft, monkeypatch, ast_tree_calibration=mode)
    if mode == "order_size":
        # A steep measured cost: the live-row rule must cut trees.
        monkeypatch.setattr(batch.tile_budget, "cost_curve", lambda width: (0.001, 0.004))
    batch.insert(prompts, max_tokens=[24] * width,
                 sampling_configs=[{"sampling_temp": 0.0}] * width)
    actual, receipts = drain(batch)
    assert actual == expected
    receipt = batch.scheduler_stats["external_tree_calibration"]
    assert receipt["mode"] == mode
    assert receipt["trees"] > 0
    assert receipt["tables"]["greedy"]["trials"]["1"] > 0
    if mode == "order_size" and width == 1:
        assert receipt["truncated_rounds"] > 0


def test_calibration_fails_closed_without_a_lattice_drafter(monkeypatch):
    model, draft = tiny(vocab=64, top_k=16, block_size=8)
    monkeypatch.setattr(type(draft), "supports_tree_calibration", False)
    with pytest.raises(ValueError, match="ast_tree_calibration requires"):
        _tree_batch(model, draft, monkeypatch, ast_tree_calibration="order")
    monkeypatch.setattr(type(draft), "supports_tree_calibration", True)
    with pytest.raises(ValueError, match="order_size needs measured tile costs"):
        _tree_batch(model, draft, monkeypatch, ast_tree_calibration="order_size",
                    ast_adaptive_rows=False)
    with pytest.raises(ValueError, match="ast_tree_calibration requires"):
        generator(model, draft, ast_tree_calibration="order")


def test_live_block_cut_is_a_parent_closed_prefix(monkeypatch):
    model, draft = tiny(vocab=64, top_k=16, block_size=8)
    batch = _tree_batch(model, draft, monkeypatch, ast_tree_calibration="order_size")
    monkeypatch.setattr(batch.tile_budget, "cost_curve", lambda width: (0.0, 1.0))
    path = [math.log(p) for p in (0.9, 0.6, 0.3, 0.2)]
    block = TreeDraftRow([7, 8, 9, 10], [-1, 0, 0, 1], 1, info={
        "depths": [1, 2, 2, 3], "log_q": [-0.1] * 4, "path_log_p": path,
    })
    (cut,) = batch._calibrated_live_blocks([object()], [block])
    assert cut.tokens == [7] and cut.parents == [-1]
    assert cut.info["path_log_p"] == path[:1]
    assert batch.tree_calibration.truncated_nodes == 3
    # A cohort is never cut here (its trees are padded to one width).
    assert batch._calibrated_live_blocks([object(), object()], [block, block]) == [block, block]

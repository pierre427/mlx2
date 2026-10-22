"""Selectable AST tile rows (8 or 16) and the calibration switch; CPU only."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from mlx2.adapters.qwen38_27b import Qwen3827BAdapter, _tile_node_budgets
from mlx2.runtime.ast_tile_budget import TileBudgetController
from mlx2.server import apply_ast_invocation

from test_dflash_pair_select import drain, generator, tiny
from test_dflash_tree_bridge import _install_reference_cohort


def _args(**overrides):
    values = dict(model="/m/target", ast=True, ast_drafter="/m/dflash", ast_topology=None,
                  ast_graft_pld=False, draft_model=None, external_draft=False,
                  native_mtp=False, prompt_lookup=False)
    values.update(overrides)
    return SimpleNamespace(**values)


def _resolution(*, tiles=True, mtp=True):
    adapter = type("Adapter", (), {"ast_tree_tiles": tiles})
    return SimpleNamespace(adapter_type=adapter, artifact={"has_mtp": mtp},
                           default_route="native_mtp" if mtp else "ordinary")


def test_cli_selects_eight_rows_and_calibration():
    args = _args(ast_tile_rows=8, ast_tree_calibration="order_size")
    policy = apply_ast_invocation(args, {}, _resolution())
    assert policy["ast_tile_rows"] == 8
    assert policy["ast_tree_calibration"] == "order_size"
    assert args.ast_route["tile_rows"] == 8
    assert args.ast_route["tree_calibration"] == "order_size"


def test_cli_calibration_fails_closed():
    with pytest.raises(ValueError, match="require --ast"):
        apply_ast_invocation(_args(ast=False, ast_tree_calibration="order"), None, _resolution())
    with pytest.raises(ValueError, match="lattice drafter"):
        apply_ast_invocation(_args(ast_drafter="mtp", ast_tree_calibration="order"), None,
                             _resolution())
    with pytest.raises(ValueError, match="lattice drafter"):
        apply_ast_invocation(_args(ast_tree_calibration="order"), None, _resolution(tiles=False))


def test_tile_rows_map_to_lane_budgets_with_the_adapter_default():
    assert Qwen3827BAdapter.default_ast_tile_rows == 16
    assert _tile_node_budgets({}, 4) == {1: 15, 2: 7, 3: 4, 4: 3}
    assert _tile_node_budgets({"ast_tile_rows": 8}, 4) == {1: 7, 2: 3, 3: 1, 4: 1}
    # The adapter default applies only when the operator names no rows.
    assert _tile_node_budgets({}, 4, default_rows=8) == {1: 7, 2: 3, 3: 1, 4: 1}
    assert _tile_node_budgets({"ast_tile_rows": 16}, 4, default_rows=8)[1] == 15
    # An 8-row lone lane can keep the 16-row shared cohort tile.
    assert _tile_node_budgets(
        {"ast_tile_rows": 8, "ast_multilane_tile_rows": 16}, 4
    ) == {1: 7, 2: 7, 3: 4, 4: 3}
    with pytest.raises(ValueError, match="4 to 16"):
        _tile_node_budgets({"ast_tile_rows": 32}, 4)


def test_adaptive_controller_stays_within_eight_rows():
    controller = TileBudgetController(warmup=1, probe=1)
    seen = set()
    for _ in range(12):
        nodes = controller.choose(1, 7)
        seen.add(nodes)
        controller.observe(1, nodes, list(range(min(nodes, 3))), 0.02 + 0.001 * nodes)
    assert seen <= {3, 5, 7} and 7 in seen


def test_eight_row_tile_keeps_exact_greedy_tokens_and_a_short_lattice(monkeypatch):
    prompts = [[1, 2, 3]]
    model, draft = tiny(vocab=64, top_k=16, block_size=16)
    reference = generator(model, draft, completion_batch_size=4, ready_drain="all")
    reference.insert(prompts, max_tokens=[24], sampling_configs=[{"sampling_temp": 0.0}])
    expected = drain(reference)[0]

    monkeypatch.setenv("MLX2_TENSORFOLD_SOURCE", "/unused-by-cpu-oracle")
    batch = generator(
        model, draft, completion_batch_size=4, ready_drain="all",
        dynamic_singleton_tree=True, dynamic_tree_max_width=4,
        tree_node_budget_by_lanes=_tile_node_budgets({"ast_tile_rows": 8}, 4),
    )
    _install_reference_cohort(batch, monkeypatch, [])
    counts = []
    real = draft.start_tree
    monkeypatch.setattr(draft, "start_tree",
                        lambda *a, **k: counts.append(a[3]) or real(*a, **k))
    batch.insert(prompts, max_tokens=[24], sampling_configs=[{"sampling_temp": 0.0}])
    actual, _receipts = drain(batch)
    assert actual == expected
    assert counts and max(counts) <= 7
    assert all(int(n) <= 7 for n in batch.scheduler_stats["external_tree_node_budget_histogram"])


def test_rows_and_calibration_are_in_the_route_receipt(monkeypatch):
    from mlx2.runtime import external_speculative

    adapter = object.__new__(Qwen3827BAdapter)
    adapter.model = object()
    adapter.draft_model = object()
    adapter.identity = {"fingerprint": "test"}
    monkeypatch.setattr(Qwen3827BAdapter, "_external_num_draft", lambda self: 15)
    monkeypatch.setattr(Qwen3827BAdapter, "_initialize_external_feedback", lambda self: None)
    monkeypatch.setattr(Qwen3827BAdapter, "_external_execution_config",
                        lambda self, **kwargs: {"backend": "external_draft"})
    monkeypatch.setenv("MLX2_TENSORFOLD_SOURCE", "/pinned/source")
    calls = []
    monkeypatch.setattr(external_speculative, "ExternalDraftBatchGenerator",
                        lambda *args, **kwargs: calls.append(kwargs) or object())

    adapter.external_policy = {"batch_size_route": "tree15_b1_b4_chain_b5plus_v1"}
    config = adapter.execution_config(max_lanes=4, prefill_step=16)
    assert config["ast_tile_rows"] == {
        "rows": 16, "authority": "adapter_default", "adaptive_rows": True,
    }
    assert "ast_tree_calibration" not in config
    adapter.create_external_batch()
    assert "ast_tree_calibration" not in calls[-1]

    adapter.external_policy = {
        "batch_size_route": "tree15_b1_b4_chain_b5plus_v1",
        "ast_tile_rows": 8, "ast_tree_calibration": "order",
    }
    config = adapter.execution_config(max_lanes=4, prefill_step=16)
    assert config["ast_tile_rows"]["rows"] == 8
    assert config["ast_tile_rows"]["authority"] == "operator_policy"
    assert config["tree_node_budget_by_lanes"] == {1: 7, 2: 3, 3: 1, 4: 1}
    assert config["ast_tree_calibration"]["mode"] == "order"
    adapter.create_external_batch()
    assert calls[-1]["ast_tree_calibration"] == "order"
    assert calls[-1]["tree_node_budget_by_lanes"] == {1: 7, 2: 3, 3: 1, 4: 1}

    # A subclass that declares another default tile owns it (checked below).
    _subclass_default(adapter, monkeypatch)


def test_tile_receipts_are_flat_metrics_with_row_mode_and_counts():
    controller = TileBudgetController(warmup=1, probe=1)
    for _ in range(5):
        nodes = controller.choose(1, 7)
        controller.observe(1, nodes, [0, 1], 0.02)
    receipt = controller.receipt()["1"]
    assert receipt["mode"] == "adaptive"
    assert receipt["maximum_nodes"] == 7 and receipt["tile_rows"] == 8
    assert sum(receipt["verified_nodes"].values()) == receipt["rounds"] == 5
    assert {"rounds", "chosen", "cost_ms", "tokens_per_round"} <= set(receipt)
    assert set(receipt["fit"]) == {"intercept_ms", "slope_ms_per_row"}
    saved = controller.snapshot()
    controller.observe(1, 3, [], 0.01)
    controller.restore(saved)
    assert controller.receipt()["1"]["verified_nodes"] == receipt["verified_nodes"]


def test_generator_publishes_the_tile_mode_from_the_start(monkeypatch):
    monkeypatch.setenv("MLX2_TENSORFOLD_SOURCE", "/unused-by-cpu-oracle")
    model, draft = tiny(vocab=64, top_k=16, block_size=16)
    batch = generator(
        model, draft, completion_batch_size=4, dynamic_singleton_tree=True,
        dynamic_tree_max_width=4,
        tree_node_budget_by_lanes=_tile_node_budgets({"ast_tile_rows": 8}, 4),
        ast_tree_calibration="order",
    )
    mode = batch.scheduler_stats["external_tree_tile_mode"]
    assert mode == {
        "mode": "adaptive", "tile_rows": 8,
        "maximum_nodes_by_lanes": {"1": 7, "2": 3, "3": 1, "4": 1},
        "calibration": "order", "batch_row_sharing": False,
    }
    assert batch.scheduler_stats["external_tree_calibration"]["enabled"] is True


def _subclass_default(adapter, monkeypatch):
    monkeypatch.setattr(Qwen3827BAdapter, "default_ast_tile_rows", 8)
    adapter.external_policy = {"batch_size_route": "tree15_b1_b4_chain_b5plus_v1"}
    config = adapter.execution_config(max_lanes=4, prefill_step=16)
    assert config["ast_tile_rows"]["rows"] == 8
    assert config["tree_node_budget_by_lanes"][1] == 7

"""AST invocation/default/status regressions; no model artifacts are loaded."""

from types import SimpleNamespace

import pytest

from mlx2.adapters.qwen38_27b import QWEN38_27B, Qwen3827BAdapter, _tile_node_budgets
from mlx2.adapters.registry import AdapterResolution
from mlx2.ast_observability import route_status
from mlx2.best_route_defaults import explicit_route_options
from mlx2.server import (
    RouteSelection, apply_ast_invocation, build_parser, resolve_execution_policy_defaults,
)


@pytest.mark.parametrize("knob,value", [("ast_tiles", 2), ("ast_tile_rows", 8),
                                       ("ast_multilane_tile_rows", 8)])
@pytest.mark.parametrize("explicit_route", [False, True])
def test_tile_shorthand_owns_default_geometry(knob, value, explicit_route):
    resolution = AdapterResolution(Qwen3827BAdapter, QWEN38_27B, {})
    policy = {"draft_model": "/fixture/draft", knob: value,
              **Qwen3827BAdapter.default_external_route_binding}
    if explicit_route:
        policy["batch_size_route"] = "tree15_b1_b4_chain_b5plus_v1"
    resolved = resolve_execution_policy_defaults(
        policy, RouteSelection("external_draft", "explicit_flag"), resolution)
    assert "tree_node_budget_by_lanes" not in resolved
    budgets = _tile_node_budgets(resolved, 4)
    assert budgets[1] == (31 if knob == "ast_tiles" else 7 if knob == "ast_tile_rows" else 15)
    assert budgets[2] == (15 if knob == "ast_tiles" else 3)


@pytest.mark.parametrize("flags", [["--ast-topology", "3x5"], ["--ast-graft-pld"]])
def test_chain_routes_refuse_tree_only_options(flags):
    args = build_parser().parse_args(["--model", "/fixture", "--ast", *flags])
    resolution = SimpleNamespace(
        adapter_type=type("Chain", (), {"ast_tree_tiles": False}),
        artifact={"has_mtp": False}, default_route="ordinary")
    with pytest.raises(ValueError, match="require tree tiles"):
        apply_ast_invocation(args, None, resolution)


def test_status_uses_effective_policy_despite_omitted_cli_placeholders():
    args = build_parser().parse_args(
        ["--model", "/fixture", "--ast", "--ast-drafter", "/fixture/draft"])
    adapter = type("Tree", (), {"ast_tree_tiles": True, "ast_tree_tile_dtype_gate": False})
    resolution = SimpleNamespace(adapter_type=adapter, artifact={}, default_route="ordinary")
    policy = apply_ast_invocation(args, {
        "ast_topology": "drafter", "ast_tiles": 2, "ast_tile_rows": 8,
        "ast_tree_calibration": "order", "ast_batch_row_sharing": True,
        "ast_graft_prompt_lookup": True, "ast_adaptive_rows": False,
    }, resolution)
    status = route_status(args.ast_route, policy, adapter)
    assert {k: status[k] for k in (
        "topology", "tiles", "tile_rows", "tree_calibration", "batch_row_sharing",
        "graft_pld", "tile_rows_mode",
    )} == {"topology": "drafter", "tiles": 2, "tile_rows": "8", "tree_calibration": "order",
           "batch_row_sharing": True, "graft_pld": True, "tile_rows_mode": "fixed"}


@pytest.mark.parametrize("flag,key", [("--ast-tree-calibration", "ast_tree_calibration"),
                                     ("--ast-batch-row-sharing", "ast_batch_row_sharing")])
def test_tree_controls_prevent_an_automatic_route_from_inventing_ast(flag, key):
    flags = [flag, "order"] if key == "ast_tree_calibration" else [flag]
    args = build_parser().parse_args(["--model", "/fixture", *flags])
    assert key in explicit_route_options(args)
    with pytest.raises(ValueError, match="require --ast"):
        apply_ast_invocation(args, None, SimpleNamespace())

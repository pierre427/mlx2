"""AST tree-tile topology parsing, budget fitting and path folding."""

import pytest

from mlx2.runtime.ast_tree_proposals import (
    parse_topology,
    paths_to_tree,
    pattern_for_budget,
    tree_receipt,
)


def test_topology_forms():
    assert parse_topology("3x5") == (5, 5, 5)
    assert parse_topology("4/4/4/3") == (4, 4, 4, 3)
    assert parse_topology([2, 3]) == (2, 3)
    assert parse_topology("chain") == "chain"
    assert parse_topology(None) == (5, 5, 5)


@pytest.mark.parametrize("bad", ["4x4", "0x5", "a/b", "", 7, [0, 2]])
def test_topology_rejects_overfull_or_malformed(bad):
    with pytest.raises(ValueError):
        parse_topology(bad)


@pytest.mark.parametrize("budget, expected", [
    (15, (5, 5, 5)), (7, (3, 2, 2)), (4, (2, 2)), (3, (3,)), (1, (1,)), (0, ()),
])
def test_pattern_fits_lane_budgets(budget, expected):
    pattern = pattern_for_budget("3x5", budget)
    assert pattern == expected
    assert sum(pattern) <= budget


def test_chain_takes_the_whole_budget():
    assert pattern_for_budget("chain", 7) == (7,)


def test_paths_fold_shared_prefixes_once_and_parents_precede_children():
    tokens, parents = paths_to_tree([(1, 2, 3), (1, 2, 4), (5, 6)], 15)
    assert tokens == [1, 2, 3, 4, 5, 6]
    assert parents == [-1, 0, 1, 1, -1, 4]
    assert all(parent < index for index, parent in enumerate(parents))
    assert tree_receipt(tokens, parents) == {"nodes": 6, "root_branches": 2, "max_depth": 3}


def test_budget_truncates_lower_ranked_paths_first():
    tokens, parents = paths_to_tree([(1, 2, 3, 4, 5), (6, 7, 8, 9, 10), (11, 12)], 7)
    assert tokens == [1, 2, 3, 4, 5, 6, 7]
    assert parents == [-1, 0, 1, 2, 3, -1, 5]


def test_forbidden_root_drops_the_whole_path():
    tokens, parents = paths_to_tree([(9, 1), (2, 3)], 15, forbidden_root=[9])
    assert tokens == [2, 3] and parents == [-1, 0]


def test_tile_knobs_derive_lane_budgets():
    from mlx2.adapters.qwen38_27b import _tile_node_budgets

    assert _tile_node_budgets({}, 4) == {1: 15, 2: 7, 3: 4, 4: 3}
    assert _tile_node_budgets({"ast_tiles": 2}, 4) == {1: 31, 2: 15, 3: 9, 4: 7}
    assert _tile_node_budgets({"ast_tile_rows": 8}, 1) == {1: 7}
    with pytest.raises(ValueError, match="ast_tiles"):
        _tile_node_budgets({"ast_tiles": 3}, 4)
    with pytest.raises(ValueError, match="conflicts"):
        _tile_node_budgets({"ast_tiles": 2, "tree_node_budget_by_lanes": {"1": 15}}, 1)


def test_multilane_tile_rows_shrink_shared_rounds_only():
    from mlx2.adapters.qwen38_27b import _tile_node_budgets

    assert _tile_node_budgets({}, 4, 8) == {1: 15, 2: 3, 3: 1, 4: 1}
    assert _tile_node_budgets({"ast_multilane_tile_rows": 16}, 4, 8) == {1: 15, 2: 7, 3: 4, 4: 3}
    with pytest.raises(ValueError, match="multilane"):
        _tile_node_budgets({"ast_multilane_tile_rows": 2}, 4)

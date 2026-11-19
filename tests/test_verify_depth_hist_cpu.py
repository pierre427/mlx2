"""Per-round drafted depth, the denominator for per-position acceptance."""
from types import SimpleNamespace

from mlx2.runtime.external_speculative import TreeDraftRow, _draft_round_depth
from mlx2.runtime.prompt_lookup import HybridStats


def test_chain_block_depth_is_its_drafted_count():
    assert _draft_round_depth(SimpleNamespace(tokens=[5, 6, 7]), 3) == 3
    assert _draft_round_depth(SimpleNamespace(tokens=[]), 0) == 0


def test_tree_block_depth_is_the_longest_path_not_the_node_count():
    # Root children 0 and 1; 2 under 0; 3 under 2; 4 under 1.  Five nodes,
    # longest path 0 -> 2 -> 3 = depth 3.
    tree = TreeDraftRow(tokens=[10, 11, 12, 13, 14], parents=[-1, -1, 0, 2, 1], width=2)
    assert _draft_round_depth(tree, 5) == 3
    # Parents need not precede their children.
    shuffled = TreeDraftRow(tokens=[13, 12, 10], parents=[1, 2, -1])
    assert _draft_round_depth(shuffled, 3) == 3


def test_stats_carry_a_depth_histogram():
    assert HybridStats().verify_depth_hist == {}

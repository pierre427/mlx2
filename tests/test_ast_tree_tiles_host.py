"""Pure scheduling geometry; synthetic long paths are not a drafter capability."""

from dataclasses import dataclass, replace

import pytest

from mlx2.runtime.ast_tree_tiles import (
    ASTProposalTrie,
    ASTTreeTile,
    plan_tree_tile,
)


# Test-only reference walk.  Production commits through ast_tree_verify's
# own walker; this executable specification keeps the tile-geometry tests.
@dataclass(frozen=True)
class ASTTreeAdvance:
    commit_local_indices: tuple[int, ...]
    commit_global_indices: tuple[int, ...]
    emitted: tuple[int, ...]
    next_root: int | None
    reason: str

    @property
    def finished(self):
        return self.next_root is None


def advance_tree_tile(trie, tile, next_tokens, *, stop_tokens=(), max_emitted=None):
    """Walk actual processed target choices, never the draft confidence.

    Commit exactly the visited input rows. A matching unexecuted child becomes
    the next tile's root; its already-decided token is not emitted twice.
    No root is reverified. A correction, stop, limit or leaf bonus closes the
    slate, so unrelated or lower-ranked pending alternatives cannot run later.
    """
    next_tokens = tuple(next_tokens)
    if (
        not isinstance(trie, ASTProposalTrie)
        or not isinstance(tile, ASTTreeTile)
        or len(next_tokens) != len(tile.tokens)
        or any(type(token) is not int or token < 0 for token in next_tokens)
        or (
            max_emitted is not None
            and (type(max_emitted) is not int or max_emitted < 1)
        )
    ):
        raise ValueError("AST tree advance requires one target choice per tile row")
    local = {node: index for index, node in enumerate(tile.global_indices)}
    if (
        not tile.global_indices
        or tile.global_indices[0] != tile.root
        or len(local) != len(tile.global_indices)
        or len(tile.parents) != len(tile.tokens)
        or any(
            type(node) is not int or not 0 <= node < len(trie.nodes)
            for node in tile.global_indices
        )
        or tile.tokens != tuple(trie.nodes[node].token for node in tile.global_indices)
        or tile.parents
        != tuple(
            -1 if index == 0 else local.get(trie.nodes[node].parent, -2)
            for index, node in enumerate(tile.global_indices)
        )
        or any(
            parent != -1 if index == 0 else not 0 <= parent < index
            for index, parent in enumerate(tile.parents)
        )
    ):
        raise ValueError("AST tile does not belong to its exact proposal trie")
    stops = frozenset(stop_tokens)
    current, commits, emitted = tile.root, [], []
    while True:
        index = local[current]
        commits.append(index)
        token = next_tokens[index]
        emitted.append(token)
        child = trie.matching_child(current, token)
        if token in stops:
            reason, pending = "stop", None
        elif max_emitted is not None and len(emitted) >= max_emitted:
            reason, pending = "limit", None
        elif child is None:
            reason = "leaf_bonus" if not trie.children[current] else "correction"
            pending = None
        elif child not in local:
            reason, pending = "reachable_continuation", child
        else:
            current = child
            continue
        return ASTTreeAdvance(
            tuple(commits),
            tuple(tile.global_indices[index] for index in commits),
            tuple(emitted),
            pending,
            reason,
        )



def _choices(trie, tile, path, bonus=999):
    successor = dict(zip(path, (*path[1:], bonus)))
    return tuple(successor.get(token, 998) for token in tile.tokens)


def test_four_disjoint_depth_four_paths_count_root_inside_sixteen():
    paths = tuple(tuple(range(10 * lane + 1, 10 * lane + 5)) for lane in range(4))
    trie = ASTProposalTrie.from_paths(0, paths)
    tile = plan_tree_tile(trie)
    assert len(tile.tokens) == 16
    assert tile.path_depths == (4, 4, 4, 3)
    assert tile.parents[0] == -1
    assert all(parent < index for index, parent in enumerate(tile.parents))
    assert len(set(tile.global_indices)) == 16
    assert tile.reason == "balanced_unknown_acceptance"
    assert tile.receipt()["proposal_rows"] == 15
    assert tile.shared_rows_saved == 3  # Three duplicate anchors avoided.


def test_shared_prefix_is_one_node_but_equal_suffix_under_other_parent_is_not():
    paths = ((1, 2, 3, 9), (1, 2, 4, 9), (1, 2, 3, 9))
    trie = ASTProposalTrie.from_paths(0, paths)
    tile = plan_tree_tile(trie)
    assert len(tile.tokens) == 7
    assert len([node for node in trie.nodes if node.token == 9]) == 2
    assert trie.path_nodes[0] == trie.path_nodes[2]
    assert trie.complete_path_indices(trie.path_nodes[0][-1]) == (0, 2)
    assert trie.nodes[trie.path_nodes[0][2]].path_indices == (0, 1, 2)
    assert tile.path_depths == (4, 4, 4)
    assert tile.shared_rows_saved == 8


def test_confident_fifteen_chain_fills_first_tile_and_cancels_all_alternatives():
    primary = tuple(range(1, 16))
    others = tuple(tuple(range(100 * lane, 100 * lane + 15)) for lane in (1, 2, 3))
    trie = ASTProposalTrie.from_paths(
        0,
        (primary, *others),
        conditional_acceptance=((0.99,) * 15, *((0.01,) * 15,) * 3),
        score_authority="synthetic_cpu_matching_slate",
    )
    tile = plan_tree_tile(trie)
    assert tile.tokens == (0, *primary)
    assert tile.path_depths == (15, 0, 0, 0)
    result = advance_tree_tile(trie, tile, (*primary, 999))
    assert result.finished and result.reason == "leaf_bonus"
    assert result.emitted == (*primary, 999)
    assert result.commit_global_indices == trie.path_nodes[0]
    assert len(result.commit_local_indices) == 16
    assert result.next_root is None


def test_continuation_never_reverifies_root_or_emits_intermediate_bonus():
    path = tuple(range(1, 16))
    trie = ASTProposalTrie.from_paths(0, (path,))
    root, visited, emitted, committed_tokens = 0, set(), [], []
    while root is not None:
        tile = plan_tree_tile(trie, root=root, max_rows=5)
        assert not visited.intersection(tile.global_indices)
        visited.update(tile.global_indices)
        result = advance_tree_tile(trie, tile, _choices(trie, tile, (0, *path)))
        committed_tokens.extend(
            tile.tokens[index] for index in result.commit_local_indices
        )
        emitted.extend(result.emitted)
        if not result.finished:
            assert result.reason == "reachable_continuation"
            assert result.emitted[-1] == trie.nodes[result.next_root].token
        root = result.next_root
    assert visited == set(range(16))
    assert committed_tokens == [0, *path]
    assert emitted == [*path, 999]
    # A successor sees the identical logical token state as one full-chain tile.
    full = plan_tree_tile(trie)
    full_result = advance_tree_tile(trie, full, (*path, 999))
    assert committed_tokens == [
        full.tokens[i] for i in full_result.commit_local_indices
    ]
    assert emitted == list(full_result.emitted)


def test_shared_ten_prefix_continues_only_the_matching_five_token_tail():
    prefix = tuple(range(1, 11))
    paths = ((*prefix, 11, 12, 13, 14, 15), (*prefix, 21, 22, 23, 24, 25))
    trie = ASTProposalTrie.from_paths(0, paths)
    initial = plan_tree_tile(trie, max_rows=11)
    assert initial.tokens == (0, *prefix)
    step = advance_tree_tile(trie, initial, (*prefix, 21))
    assert step.reason == "reachable_continuation"
    tail = plan_tree_tile(trie, root=step.next_root)
    assert tail.tokens == (21, 22, 23, 24, 25)
    assert tail.parents == (-1, 0, 1, 2, 3)
    assert set(initial.global_indices).isdisjoint(tail.global_indices)
    final = advance_tree_tile(trie, tail, (22, 23, 24, 25, 999))
    assert (*step.emitted, *final.emitted) == (*paths[1], 999)
    assert trie.complete_path_indices(final.commit_global_indices[-1]) == (1,)


def test_root_rejection_stop_and_terminal_limit_prune_all_pending_work():
    trie = ASTProposalTrie.from_paths(0, ((1, 2, 3, 4), (1, 2, 7, 8)))
    tile = plan_tree_tile(trie, max_rows=3)
    reject = advance_tree_tile(trie, tile, (9, 9, 9))
    assert reject.finished and reject.reason == "correction"
    assert reject.commit_local_indices == (0,) and reject.emitted == (9,)
    stop = advance_tree_tile(trie, tile, (1, 2, 3), stop_tokens=(1,))
    assert stop.finished and stop.reason == "stop" and stop.emitted == (1,)
    limited = advance_tree_tile(trie, tile, (1, 2, 3), max_emitted=2)
    assert limited.finished and limited.reason == "limit"
    assert limited.emitted == (1, 2)


def test_lagged_acceptance_is_explicit_and_shared_node_support_is_not_summed():
    paths = ((1, 2), (1, 3))
    trie = ASTProposalTrie.from_paths(
        0,
        paths,
        conditional_acceptance=((0.8, 0.5), (0.7, None)),
        lagged_acceptance=0.25,
        score_authority="matching_slate_and_completed_rounds",
    )
    shared = trie.path_nodes[0][1]
    assert trie.nodes[shared].reach == 0.8
    assert trie.nodes[trie.path_nodes[0][-1]].reach == 0.4
    assert trie.nodes[trie.path_nodes[1][-1]].reach == 0.175
    assert plan_tree_tile(trie, max_rows=3).tokens == (0, 1, 2)
    unknown = ASTProposalTrie.from_paths(
        0, paths, conditional_acceptance=((0.8, 0.5), None)
    )
    assert plan_tree_tile(unknown).reason == "balanced_unknown_acceptance"


def test_invalid_probabilities_shapes_and_foreign_tile_fail_closed():
    for confidence in (((1.1,),), ((float("nan"),),), ((True,),), ((0.5, 0.5),)):
        with pytest.raises(ValueError, match="confidence"):
            ASTProposalTrie.from_paths(0, ((1,),), conditional_acceptance=confidence)
    with pytest.raises(ValueError, match="lagged"):
        ASTProposalTrie.from_paths(0, ((1,),), lagged_acceptance=-1)
    trie = ASTProposalTrie.from_paths(0, ((1, 2),))
    for cap in (0, 17, True):
        with pytest.raises(ValueError, match="total rows"):
            plan_tree_tile(trie, max_rows=cap)
    tile = plan_tree_tile(trie)
    for foreign in (
        replace(tile, tokens=(3, 1, 2)),
        replace(tile, parents=(-1, 0, 0)),
        replace(tile, global_indices=(0, 1, 1)),
        replace(tile, global_indices=(0, 2), tokens=(0, 2), parents=(-1, -2)),
    ):
        with pytest.raises(ValueError, match="exact proposal trie"):
            advance_tree_tile(trie, foreign, (1,) * len(foreign.tokens))


def test_actual_short_source_paths_are_never_extended_to_synthetic_fifteen():
    paths = tuple(tuple(range(10 * lane + 1, 10 * lane + 8)) for lane in range(4))
    trie = ASTProposalTrie.from_paths(0, paths, lagged_acceptance=1.0)
    tile = plan_tree_tile(trie)
    assert tile.path_depths == (4, 4, 4, 3)
    assert max(node.depth for node in trie.nodes) == 7
    assert len(tile.tokens) <= 16

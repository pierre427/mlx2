"""Host-only DFlash path allocation; no MLX device or model required."""

from types import SimpleNamespace

import pytest

from mlx2.adapters.proposal_path_sources import ExternalContinuationSource


def source(tokens, parents):
    class TreeDraft:
        proposal_distribution = "stochastic_exact_law"
        fold_policy = None
        config = SimpleNamespace(block_size=8)
        calls = 0

        def propose_tree(self, _anchors, _pending, cache, nodes, **options):
            self.calls += 1
            assert nodes == 15
            assert options["lattice_positions"] == 5
            cache["private_mutation"] = True
            return [(tokens, parents)]

    draft = TreeDraft()
    context = SimpleNamespace(
        depth=4,
        anchor=7,
        pending_features=None,
        draft_cache={"committed": True},
    )
    return ExternalContinuationSource(draft), context, draft


@pytest.mark.parametrize("limit", [2, 4])
def test_source_offers_siblings_before_spending_slots_on_ancestors(limit):
    provider, context, draft = source(
        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        [-1, 0, 1, 2, 1, 4, 1, 6, 1, 8],
    )
    paths = provider(context, limit)
    expected = [
        (10, 11, 12, 13),
        (10, 11, 14, 15),
        (10, 11, 16, 17),
        (10, 11, 18, 19),
    ]
    assert [path.tokens for path in paths] == expected[:limit]
    # The first formerly offered full-depth primary is unchanged. Retain its
    # exact ordinal score, rather than renumbering the selected frontier.
    assert paths[0].tokens == (10, 11, 12, 13)
    assert [path.ranking_score for path in paths] == [-3.0, -5.0, -7.0, -9.0][:limit]
    assert all(
        path.ranking_convention == "dflash2_best_first_node_ordinal_proxy"
        for path in paths
    )
    assert draft.calls == 1
    assert context.draft_cache == {"committed": True}
    receipt = provider.last_diagnostics
    assert receipt["nodes"] == receipt["candidate_paths"] == 10
    assert receipt["node_budget"] == 15
    assert receipt["prefix_dominated_paths_removed"] == 6
    assert receipt["nondominated_paths"] == 4
    assert receipt["offered_paths"] == limit
    assert receipt["offered_node_ordinals"] == [3, 5, 7, 9][:limit]


def test_single_path_control_preserves_legacy_first_node():
    provider, context, _draft = source(
        [10, 11, 12, 13, 14],
        [-1, 0, 1, 2, 1],
    )
    paths = provider(context, 1)
    assert [path.tokens for path in paths] == [(10,)]
    assert paths[0].ranking_score == 0.0
    assert provider.last_diagnostics["path_selection"] == "legacy_single_path"
    assert provider.last_diagnostics["prefix_dominated_paths_removed"] == 0
    assert provider.last_diagnostics["offered_paths"] == 1


def test_ragged_frontier_keeps_real_short_leaves_and_never_fills_missing_paths():
    provider, context, _draft = source(
        [10, 11, 12, 13, 20, 21, 30],
        [-1, 0, 1, 2, -1, 4, -1],
    )
    paths = provider(context, 4)
    assert [path.tokens for path in paths] == [
        (10, 11, 12, 13),
        (20, 21),
        (30,),
    ]
    assert [path.ranking_score for path in paths] == [-3.0, -5.0, -6.0]
    assert provider.last_diagnostics["offered_paths"] == 3
    assert provider.last_diagnostics["offered_depth_hist"] == {
        "4": 1,
        "2": 1,
        "1": 1,
    }

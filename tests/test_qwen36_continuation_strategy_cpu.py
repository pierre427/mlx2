"""GPU-free Qwen3.6 longest-first proposal-strategy contracts."""

from types import SimpleNamespace

import pytest

from mlx2.adapters.proposal_path_sources import ExternalContinuationSource
from mlx2.adapters.qwen36_35b import CACHE_LAYOUT, Qwen3635BA3BAdapter
from mlx2.runtime.continuation_strategy import (
    ContinuationStrategy,
    LongestFirstExactPrefix,
)
from mlx2.runtime.proposal_providers import ContinuationContext


@pytest.mark.parametrize("fingerprint,expected", [
    ("other-checkpoint", 2048),
    ("a2fa43d0942e24f2e0ef3041a7ffe88ae0e4d3f39de56f0f4c8e8c94664b144c", 256),
    ("204ca8991ab03cb1d78066b295185c78ab3b07367f3fc453680fe36d3058c83c", 256),
    ("ba62aaf8bbd40762c67b16fa012947b52b5f5e35f8d73d69101d401a0dc27b71", 256),
])
def test_qwen36_owns_measured_prefill_step_instead_of_dense_autoscale(fingerprint, expected):
    adapter = object.__new__(Qwen3635BA3BAdapter)
    adapter.identity = {"fingerprint": fingerprint}
    assert adapter.prefill_step_default() == expected
    # AST binds target + drafter + numerical laws into a new route identity;
    # its target's operator-selected chunk preference must survive that bind.
    adapter.identity = {"fingerprint": "composite-route", "target_fingerprint": fingerprint}
    assert adapter.prefill_step_default() == expected


@pytest.mark.parametrize("layout", [CACHE_LAYOUT, CACHE_LAYOUT + "-residual-bfloat16-v1"])
def test_qwen36_strategy_pins_exact_target_and_routed_expert_cost(layout):
    adapter = object.__new__(Qwen3635BA3BAdapter)
    adapter.layout = layout
    adapter.external_policy = {
        "continuation_strategy": "longest_first_exact_prefix"
    }
    strategy = adapter.continuation_verification_strategy()
    assert strategy.algorithm == "longest_first_exact_prefix_v1"
    assert strategy.cache_layout == layout
    assert strategy.proposal_state == "bound_source_only"
    assert strategy.target_state == "authoritative_exact"
    assert strategy.routed_experts_per_token == 8
    assert strategy.routed_moe_layers == 40


def test_longest_first_prunes_impossible_siblings_and_reuses_frontier():
    paths = (
        (1, 2, 3, 4, 5, 6, 7),
        (0, 2, 3, 4, 9, 6),
        (1, 2, 0, 4, 9, 6),
        (1, 2, 3, 4, 9, 6),
    )
    frontier = LongestFirstExactPrefix(paths, maximum=8)
    first = frontier.next_attempt()
    assert first.path_index == 0 and first.suffix == paths[0]
    observed = frontier.observe(first, (1, 2, 3, 4, 9))
    assert set(observed.pruned_indices) == {1, 2}
    second = frontier.next_attempt()
    assert second.path_index == 3 and second.suffix == (6,)
    assert second.start == 5


def test_strategy_accounting_geometry_fails_closed():
    with pytest.raises(ValueError, match="proposal state"):
        ContinuationStrategy(
            proposal_state="authoritative",
            cache_layout=CACHE_LAYOUT,
        )
    with pytest.raises(ValueError, match="geometry must be complete"):
        ContinuationStrategy(routed_experts_per_token=8)


def test_dflash2_is_a_bound_complete_path_source_not_target_state():
    class Draft:
        proposal_distribution = "categorical_distribution"
        config = SimpleNamespace(block_size=8)

        def propose_tree(self, inputs, features, cache, nodes, lattice_positions):
            assert inputs == [17]
            assert features == "request-private-features"
            assert cache == ["request-private-cache"]
            assert nodes == 15
            assert lattice_positions == 4
            return ([10, 11, 12, 20, 21, 22], [-1, 0, 1, -1, 3, 4]), None

    source = ExternalContinuationSource(Draft())
    paths = source(
        SimpleNamespace(
            depth=3,
            anchor=17,
            pending_features="request-private-features",
            draft_cache=["request-private-cache"],
        ),
        limit=8,
    )
    assert source.mechanism == "dflash2"
    # Multi-path offers keep only prefix-nondominated paths (e54dabcb9): a
    # longer supplied path already covers every ancestor prefix under
    # target-draw-then-prefix-match verification. All six tree nodes are
    # still enumerated as prefix-closed candidates first.
    assert [path.tokens for path in paths] == [(10, 11, 12), (20, 21, 22)]
    assert [path.ranking_score for path in paths] == [-2.0, -5.0]
    receipt = source.last_diagnostics
    assert receipt["path_selection"] == "prefix_nondominated"
    assert receipt["nodes"] == receipt["candidate_paths"] == 6
    assert receipt["prefix_dominated_paths_removed"] == 4
    assert all(
        path.confidence_features == (None,) * len(path.tokens) for path in paths
    )


def test_dflash2_enumerates_all_fifteen_prefix_closed_tree_nodes_before_limit():
    class Draft:
        proposal_distribution = "categorical_distribution"
        config = SimpleNamespace(block_size=8)

        def propose_tree(
            self, _inputs, _features, _cache, nodes, *, lattice_positions
        ):
            assert nodes == 15
            assert lattice_positions == 4
            tokens = list(range(10, 25))
            parents = [-1, 0, 1, -1, 3, 4, -1, 6, 7, -1, 9, 10, -1, 12, 13]
            return (tokens, parents), None

    source = ExternalContinuationSource(Draft())
    paths = source(
        SimpleNamespace(depth=3, anchor=17, pending_features=None, draft_cache=[]),
        limit=15,
    )
    # All fifteen prefix-closed nodes are enumerated before the limit is
    # applied; the multi-path offer then drops the ten ancestor prefixes that
    # the five full-depth leaves already cover (e54dabcb9).
    receipt = source.last_diagnostics
    assert receipt["nodes"] == receipt["candidate_paths"] == 15
    assert receipt["depth_hist"] == {"1": 5, "2": 5, "3": 5}
    assert receipt["prefix_dominated_paths_removed"] == 10
    assert receipt["offered_paths"] == len(paths) == 5
    assert [len(path.tokens) for path in paths] == [3] * 5
    assert paths[0].tokens == (10, 11, 12)
    assert paths[-1].tokens == (22, 23, 24)
    assert receipt["offered_node_ordinals"] == [2, 5, 8, 11, 14]

    # The single-path control still sees the first prefix-closed node.
    single = ExternalContinuationSource(Draft())(
        SimpleNamespace(depth=3, anchor=17, pending_features=None, draft_cache=[]),
        limit=1,
    )
    assert [path.tokens for path in single] == [(10,)]


def test_fold_keeps_dflash2_complete_path_protocol_with_distinct_identity():
    class Draft:
        proposal_distribution = "categorical_distribution"
        fold_policy = SimpleNamespace(window=4096)
        config = SimpleNamespace(block_size=8)

        def propose_tree(self, *_args, **_kwargs):
            return ([10, 11], [-1, 0]), None

    source = ExternalContinuationSource(Draft())
    paths = source(
        SimpleNamespace(
            depth=2,
            anchor=17,
            pending_features=None,
            draft_cache=[],
        ),
        limit=8,
    )
    assert source.mechanism == "dflash2_fold"
    # Prefix-nondominated multi-path offer (e54dabcb9): (10, 11) covers (10,).
    assert [path.tokens for path in paths] == [(10, 11)]
    assert source.last_diagnostics["candidate_paths"] == 2


def test_dspark_complete_path_bridge_is_a_supported_provider_contract():
    class RenamedDSparkWrapper:
        proposal_distribution = "deterministic_point_mass"
        continuation_protocol = "dspark"
        config = SimpleNamespace(block_size=8)

        @staticmethod
        def propose_complete_paths(context, limit):
            context.draft_cache.append("private-mutation")
            return [
                {
                    "tokens": [3, 4, 5],
                    "confidence_features": [0.1, 0.2, 0.3],
                    "ranking_score": 0.75,
                }
            ][:limit]

    source = ExternalContinuationSource(RenamedDSparkWrapper())
    shared_cache = []
    paths = source(
        ContinuationContext((), 17, 3, shared_cache, None),
        limit=5,
    )
    assert source.mechanism == "dspark"
    assert paths[0].tokens == (3, 4, 5)
    assert paths[0].ranking_convention == "dspark_complete_path_private_proxy"
    assert shared_cache == []

"""GPU-free Qwen3.6 longest-first proposal-strategy contracts."""

from types import SimpleNamespace

import pytest

from mlx2.adapters.proposal_path_sources import ExternalContinuationSource
from mlx2.adapters.qwen36_35b import CACHE_LAYOUT, Qwen3635BA3BAdapter
from mlx2.runtime.continuation_strategy import (
    ContinuationStrategy,
    LongestFirstExactPrefix,
)


def test_qwen36_owns_measured_prefill_step_instead_of_dense_autoscale():
    adapter = object.__new__(Qwen3635BA3BAdapter)
    assert adapter.prefill_step_default() == 2048


def test_qwen36_strategy_pins_exact_target_and_routed_expert_cost():
    adapter = object.__new__(Qwen3635BA3BAdapter)
    adapter.external_policy = {
        "continuation_strategy": "longest_first_exact_prefix"
    }
    strategy = adapter.continuation_verification_strategy()
    assert strategy.algorithm == "longest_first_exact_prefix_v1"
    assert strategy.cache_layout == CACHE_LAYOUT
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
    assert [path.tokens for path in paths] == [(10, 11, 12), (20, 21, 22)]
    assert all(path.confidence_features == (None, None, None) for path in paths)

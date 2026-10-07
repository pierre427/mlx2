"""SPEC-03: the bounded tree route's B5+ chain leg keeps the tree row budget."""
import pytest
from test_dflash_tree_bridge import generator, tiny

import mlx2.runtime.external_speculative as external

BUDGETS = {1: 15, 2: 7, 3: 4, 4: 3}
b_num_draft = 3  # the shared test generator's num_draft


@pytest.mark.parametrize(
    "lanes, cap",
    [(1, None), (4, None), (5, 2), (6, 1), (8, 1), (9, 0), (16, 0)],
)
def test_chain_cap_keeps_the_sixteen_row_budget(lanes, cap):
    assert external._chain_draft_cap(BUDGETS, 4, lanes) == cap
    if cap is not None:
        assert lanes * (cap + 1) <= 16


def test_b1_only_tree_route_keeps_its_chain_depth():
    assert external._chain_draft_cap({1: 15}, 1, 5) is None


def _five_lanes(monkeypatch, *, capped):
    if not capped:
        monkeypatch.setattr(external, "_chain_draft_cap", lambda *_args: None)
    m, d = tiny(vocab=64, top_k=16, block_size=8)
    b = generator(m, d, pairwise_selection="host", completion_batch_size=6,
                  dynamic_singleton_tree=True, dynamic_tree_max_width=4,
                  tree_node_budget_by_lanes=dict(BUDGETS))
    # Singleton reference tree rounds below B5 (TensorFold is Metal-only).
    b.target_execution = "reference"
    lengths = []
    real = d.draft_distributions

    def spy(anchors, hidden, cache, count, *args, **kwargs):
        lengths.append((b._auto_active_width, count))
        return real(anchors, hidden, cache, count, *args, **kwargs)

    monkeypatch.setattr(d, "draft_distributions", spy)
    prompts = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]
    b.insert(prompts, max_tokens=[12] * 5, sampling_configs=[{"sampling_temp": 0.0}] * 5)
    tokens = {}
    for _ in range(400):
        for response in b.next()[1]:
            tokens.setdefault(response.uid, []).append(response.token)
        if not b.lanes:
            break
    return tokens, lengths, b.scheduler_stats


def test_b5_chain_rounds_draft_at_most_the_row_budget_and_stay_exact(monkeypatch):
    with monkeypatch.context() as patch:
        uncapped, wide, _ = _five_lanes(patch, capped=False)
    capped, lengths, stats = _five_lanes(monkeypatch, capped=True)
    assert stats["external_auto_chain_rounds"] > 0
    chain = [count for width, count in lengths if width == 5]
    assert chain and max(chain) == 2
    assert max(count for width, count in wide if width == 5) == b_num_draft
    assert capped == uncapped

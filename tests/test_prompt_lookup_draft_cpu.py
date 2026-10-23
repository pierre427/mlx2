"""Prompt lookup as an AST tree-tile drafter (model-free, and grafted)."""

from types import SimpleNamespace

import pytest

from mlx2.runtime.drafters.prompt_lookup_draft import PromptLookupDraftModel


def _target():
    args = SimpleNamespace(vocab_size=100, hidden_size=8)
    return SimpleNamespace(speculative_args=args, layers=[object()] * 4)


def test_copy_paths_prefer_specific_ngrams_and_recent_sites():
    draft = PromptLookupDraftModel(_target(), num_draft=5)
    history = [1, 2, 3, 9, 9, 7, 2, 3, 4, 4, 8, 1]
    paths, best = draft.copy_paths(history, 2, 3, 3)
    # Context (..., 1, 2): the "1 2" bigram matched once (-> 3 9 9).
    assert paths[0] == (3, 9, 9) and best == 2


def test_tree_tile_from_several_copy_sources_and_incremental_index():
    draft = PromptLookupDraftModel(_target(), num_draft=5)
    history = [5, 6, 1, 2, 5, 6, 3, 4, 5]
    tree = draft.propose_tree([6], None, [], 15, processor_histories=[history], topology="3x5")
    tokens, parents = tree[0]
    # Two distinct sites follow "5 6": (3, 4, 5, 6) and (1, 2, 5, 6, 3).
    assert tokens[:4] == [3, 4, 5, 6] and parents[:4] == [-1, 0, 1, 2]
    assert -1 in parents[4:]
    draft.propose_tree([7], None, [], 15, processor_histories=[history + [6]], topology="3x5")
    assert draft.stats["index_rebuilds"] == 1 and draft.stats["index_extends"] == 1


@pytest.mark.parametrize("extension", [[], [9, 10]])
def test_matching_suffix_does_not_reuse_an_unrelated_prefix(extension):
    draft = PromptLookupDraftModel(_target(), num_draft=5, ngram_min=3, ngram_max=3)
    suffix = [7] * 29 + [1, 2, 3]
    first = [1, 2, 3, 42] + suffix
    second = [8, 8, 8, 77] + suffix + extension
    draft.copy_paths(first[:-1], first[-1], 5, 1)
    fresh = PromptLookupDraftModel(_target(), num_draft=5, ngram_min=3, ngram_max=3)
    assert draft.copy_paths(second[:-1], second[-1], 5, 1) == fresh.copy_paths(
        second[:-1], second[-1], 5, 1,
    ) == ([], 0)
    assert draft.stats["index_rebuilds"] == 2


def test_bogus_suffix_match_cannot_displace_the_learned_graft():
    class Base:
        config = SimpleNamespace(block_size=16)

        def propose_tree(self, *args, **kwargs):
            return [([50, 51], [-1, 0])]

    draft = PromptLookupDraftModel(_target(), num_draft=5, base=Base(), ngram_min=3, ngram_max=3)
    suffix = [7] * 29 + [1, 2, 3]
    first, second = [1, 2, 3, 42] + suffix, [8, 8, 8, 77] + suffix
    draft.copy_paths(first[:-1], first[-1], 5, 1)
    assert draft.propose_tree(
        [second[-1]], None, [], 5, processor_histories=[second[:-1]],
    ) == [([50, 51], [-1, 0])]
    assert draft.stats["copy_first"] == 0


def test_grafted_copy_branch_leads_when_the_match_is_specific():
    class Base:
        config = SimpleNamespace(block_size=16, target_layer_ids=(3,))

        def make_cache(self):
            return []

        def propose_tree(self, anchors, hidden, cache, max_nodes, **_):
            return [([50, 51, 52], [-1, 0, 0]) for _ in anchors]

    draft = PromptLookupDraftModel(_target(), num_draft=5, base=Base())
    history = [10, 11, 12, 13, 14, 20, 10, 11]
    tokens, parents = draft.propose_tree([12], None, [], 15, processor_histories=[history])[0]
    assert tokens[:5] == [13, 14, 20, 10, 11]  # "10 11 12" trigram copy first
    assert tokens[5:] == [50, 51, 52] and parents[5:] == [-1, 5, 5]
    tokens, _ = draft.propose_tree([99], None, [], 15, processor_histories=[history])[0]
    assert tokens == [50, 51, 52]               # no match: base tree only


def test_graft_forwards_metadata_to_a_declaring_base():
    class Base:
        config = SimpleNamespace(block_size=16)
        tree_topology_aware = True
        requires_processor_histories = True

        def propose_tree(self, anchors, hidden, cache, max_nodes, **options):
            assert options["topology"] == "3x5"
            assert options["processor_histories"] == [[1, 2]]
            return [([50], [-1])]

    graft = PromptLookupDraftModel(_target(), num_draft=5, base=Base())
    tokens, _ = graft.propose_tree(
        [99], None, [], 15, processor_histories=[[1, 2]], topology="3x5",
    )[0]
    assert tokens == [50]

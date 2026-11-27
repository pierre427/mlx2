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


# -- draft_lazy (speculative_device_round ``lazy_draft``) ---------------------

# A prompt that shares n-grams with the tiny Inkling target's greedy cycle but
# continues them differently, so copies are both accepted and rejected.
_REPEATS = [30, 18, 33, 12, 4, 19, 35, 32, 1, 6, 5, 17, 2, 33, 8, 21, 4, 11, 26, 9, 14, 3]


def _pld_decode(sampling=None, num_draft=4, steps=40, **options):
    import mlx.core as mx
    from test_inkling_mtp_cpu import target

    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    mx.set_default_device(mx.cpu)
    model = target()
    draft = PromptLookupDraftModel(model, num_draft=num_draft)
    batch = ExternalDraftBatchGenerator(model, draft_model=draft, binding="test",
                                        num_draft=num_draft, prefill_step_size=64, **options)
    try:
        uid = batch.insert([list(_REPEATS)], max_tokens=[steps],
                           sampling_configs=[sampling or {"sampling_temp": 0}])[0]
        got = []
        for _ in range(400):
            _, responses = batch.next()
            got += [r.token for r in responses if r.uid == uid]
            if not batch.lanes:
                break
        else:
            raise AssertionError("scheduler stalled")
        # The drafter's own counters ride along under a private key.
        return got, {**batch.scheduler_stats, "_drafter_stats": dict(draft.stats)}
    finally:
        batch.close()


@pytest.mark.parametrize("depth", [1, 3, 5])
def test_draft_lazy_equals_the_host_copy(depth):
    import mlx.core as mx

    draft = PromptLookupDraftModel(_target(), num_draft=5)
    for history, anchor in (([1, 2, 3, 9, 9, 7, 2, 3, 4, 4, 8, 1], 2), ([5, 6, 7], 8)):
        tokens, laws = draft.draft_distributions([anchor], None, [], depth, [None], [0.0],
                                                 processor_histories=[history])
        if not tokens[0]:
            # No copy: the host runs the anchor-only round as its plain
            # single-token step, so the lazy draft hands it back.
            from mlx2.runtime.proposal_composition import LazyDecline

            with pytest.raises(LazyDecline) as caught:
                draft.draft_lazy(anchor, None, [], depth, processor_history=list(history))
            assert caught.value.reason == "prompt_lookup_no_copy"
            continue
        lazy = draft.draft_lazy(anchor, None, [], depth, processor_history=list(history))
        assert isinstance(lazy, mx.array) and lazy.dtype == mx.int32
        assert [int(t) for t in lazy.tolist()] == tokens[0]
        assert [law.token for law in laws[0]] == tokens[0]
    # A copy reads no logits: processors do not change it (nor on the host).
    history = [1, 2, 3, 9, 9, 7, 2, 3, 4, 4, 8, 1]
    assert draft.draft_lazy(2, None, [], 3, processor_history=history,
                            logits_processors=[lambda t, x: x]).tolist() == [3, 9, 9]


def test_draft_lazy_declines_before_touching_anything():
    import mlx.core as mx

    from mlx2.runtime.proposal_composition import LazyDecline

    draft = PromptLookupDraftModel(_target(), num_draft=5)
    history = [1, 2, 3, 9, 9, 7, 2, 3, 4, 4, 8, 1]
    cases = [
        ((mx.array([2]), None, [], 3), {"processor_history": history},
         "prompt_lookup_device_anchor"),
        ((2, None, [], 3, lambda logits: logits[0, :1]), {"processor_history": history},
         "prompt_lookup_sampled_drafts"),
        ((2, None, [], 3), {}, "prompt_lookup_history"),
        ((2, None, [], 3), {"processor_history": object()}, "prompt_lookup_device_history"),
    ]
    for args, kwargs, reason in cases:
        with pytest.raises(LazyDecline) as caught:
            draft.draft_lazy(*args, **kwargs)
        assert caught.value.reason == reason
    assert draft.stats["index_rebuilds"] == draft.stats["index_extends"] == 0

    class Base:
        config = SimpleNamespace(block_size=16)

    graft = PromptLookupDraftModel(_target(), num_draft=5, base=Base())
    with pytest.raises(LazyDecline) as caught:
        graft.draft_lazy(2, None, [], 3, processor_history=history)
    assert caught.value.reason == "prompt_lookup_graft"


def test_draft_lazy_admission_declines_no_copy_and_graft_and_books_stats_once():
    draft = PromptLookupDraftModel(_target(), num_draft=5)
    history = [1, 2, 3, 9, 9, 7, 2, 3, 4, 4, 8, 1]
    host = PromptLookupDraftModel(_target(), num_draft=5)
    # A copy: admitted; the lazy draft reuses the lookup and books it once.
    assert draft.draft_lazy_admission(2, history, 3) is None
    assert draft.stats == host.stats  # nothing booked by the admission itself
    assert draft.draft_lazy(2, None, [], 3, processor_history=history).tolist() == [3, 9, 9]
    host.draft_distributions([2], None, [], 3, [None], [0.0], processor_histories=[history])
    assert draft.stats == host.stats
    # No copy: declined; the host path that serves the round books the lookup once.
    assert draft.draft_lazy_admission(8, [5, 6, 7], 3) == "prompt_lookup_no_copy"
    tokens, _ = draft.draft_distributions([8], None, [], 3, [None], [0.0],
                                          processor_histories=[[5, 6, 7]])
    host.draft_distributions([8], None, [], 3, [None], [0.0], processor_histories=[[5, 6, 7]])
    assert tokens == [[]] and draft.stats == host.stats
    # ``lookup=False`` (a prompt-copy round) runs no lookup.
    assert draft.draft_lazy_admission(8, [5, 6, 7, 8], 3, lookup=False) is None
    assert draft.stats == host.stats and draft._pending_copy is None

    class Base:
        config = SimpleNamespace(block_size=16)

    graft = PromptLookupDraftModel(_target(), num_draft=5, base=Base())
    assert graft.draft_lazy_admission(2, history, 3) == "prompt_lookup_graft"
    assert graft.draft_lazy_admission(2, history, 3, lookup=False) == "prompt_lookup_graft"


@pytest.mark.parametrize(("options", "sampling"), [
    ({"lazy_draft": True, "decide": True}, None),
    ({"lazy_draft": True}, None),
    ({"lazy_draft": True, "decide": True, "processors": True},
     {"sampling_temp": 0, "presence_penalty": 1.5}),
])
def test_device_round_prompt_lookup_equals_the_host_round(options, sampling):
    from mlx2.runtime.speculative_device_round import DECLINE_REASONS

    off, off_stats = _pld_decode(sampling)
    on, stats = _pld_decode(sampling, device_round=options)
    assert on == off
    rounds = stats["external_device_round_rounds"]
    assert rounds > 0 and stats["external_device_round_lazy_draft_lanes"] == rounds
    # Rounds with no copy are declined at admission (before the snapshot) and
    # run as the host's plain single-token decode step, as with the option off.
    reasons = stats[DECLINE_REASONS]
    assert set(reasons) == {"prompt_lookup_no_copy"}
    assert reasons["prompt_lookup_no_copy"] == stats["external_device_round_declines"] > 0
    for key in ("proposed_tokens", "accepted_proposals", "external_plain_single_token_rounds",
                "external_cohort_target_single_lane_plain_single_token_rounds",
                "_drafter_stats"):
        assert stats[key] == off_stats[key], key
    assert off_stats["external_plain_single_token_rounds"] > 0
    # Real copies: some accepted, some rejected.
    assert 0 < stats["accepted_proposals"] < stats["proposed_tokens"]


def test_device_round_prompt_lookup_sampled_drafts_decline_counted():
    from mlx2.runtime.speculative_device_round import DECLINE_REASONS

    sampling = {"sampling_temp": 0.7, "seed": 11}
    # Point-mass copies on a sampled lane: lazy drafts, host verifier.
    off, _ = _pld_decode(sampling)
    lazy, lazy_stats = _pld_decode(sampling, device_round={"lazy_draft": True})
    assert lazy == off and lazy_stats["external_device_round_rounds"] > 0
    # ``decide`` draws its sampled rows with its own key (same law, other
    # tokens), so the declined arm is compared with a decide-only engine.
    decided, _ = _pld_decode(sampling, device_round={"decide": True})
    on, stats = _pld_decode(
        sampling, device_round={"lazy_draft": True, "decide": True, "sampled_drafts": True})
    assert on == decided
    reasons = stats[DECLINE_REASONS]
    # No-copy rounds decline at admission first (host plain single-token step).
    assert set(reasons) == {"prompt_lookup_sampled_drafts", "prompt_lookup_no_copy"}
    assert reasons["prompt_lookup_sampled_drafts"] > 0
    assert sum(reasons.values()) == stats["external_device_round_declines"]
    assert stats["external_device_round_rounds"] == 0

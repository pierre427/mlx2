"""Synthetic continuation-stash isolation; no checkpoint or device execution."""

from types import SimpleNamespace

from mlx2.runtime.drafters.native_mtp import NativeMTPDraftModel
from mlx2.runtime.proposal_providers import ContinuationContext


def _draft():
    draft = NativeMTPDraftModel.__new__(NativeMTPDraftModel)
    draft.revision = "a" * 64
    draft._stash = {}
    draft.stats = {"complete_path_calls": 0, "complete_path_stash_misses": 0}
    return draft


def _context(history, *, processors=()):
    return ContinuationContext(tuple(history), 3, 1, None, None, processors)


def test_stash_does_not_reuse_a_foreign_prefix_with_the_same_last_64_tokens():
    draft = _draft()
    first, second = [1, *([5] * 64)], [2, *([5] * 64)]
    draft._remember(first, 3, [7], [((7, -0.1), (8, -2.0))])
    assert draft.propose_complete_paths(_context(first), 2)[0].tokens == (7,)
    assert draft.propose_complete_paths(_context(second), 2) == []
    draft._remember(second, 3, [9], [((9, -0.2), (10, -3.0))])
    assert draft.propose_complete_paths(_context(first), 2)[0].tokens == (7,)
    assert draft.propose_complete_paths(_context(second), 2)[0].tokens == (9,)


def test_identical_histories_keep_each_processor_owners_ranked_proposals():
    draft = _draft()
    history = [1, 2]
    # Processors need not be hashable, and their equality is not ownership.
    first, second = SimpleNamespace(kind="mask"), SimpleNamespace(kind="mask")
    draft._remember(history, 3, [7], [((7, -0.1),)], processors=[first])
    draft._remember(history, 3, [9], [((9, -0.2),)], processors=[second])
    assert draft.propose_complete_paths(_context(history, processors=(first,)), 1)[0].tokens == (7,)
    assert draft.propose_complete_paths(_context(history, processors=(second,)), 1)[0].tokens == (9,)
    assert draft.propose_complete_paths(_context(history), 1) == []


def test_stash_binds_processor_order():
    draft = _draft()
    first, second = object(), object()
    draft._remember([1, 2], 3, [7], [((7, -0.1),)], processors=[first, second])
    assert draft.propose_complete_paths(_context([1, 2], processors=(second, first)), 1) == []


def test_batched_drafting_publishes_only_each_lanes_processed_paths():
    import mlx.core as mx
    from test_apc_hits_hybrid_gdn_self_mtp import tiny_qwen4_mtp

    def only(token):
        def mask(_history, values):
            return mx.where(mx.arange(values.shape[-1]) == token, values, -mx.inf)

        return mask

    with mx.stream(mx.cpu):
        model, _ = tiny_qwen4_mtp()
        model.select_external_feature_source("mtp_seed")
        draft = NativeMTPDraftModel(model, num_draft=1, revision="a" * 64)
        history = [1, 2]
        seeds = model.prefill_body(
            mx.array([history]), model.make_cache(), draft.config.target_layer_ids
        )
        first, second = only(7), only(9)
        tokens, _ = draft.draft_distributions(
            [3, 3], mx.repeat(seeds, 2, axis=0),
            draft.batch_caches([draft.make_cache(), draft.make_cache()]),
            1, [None, None], [0.0, 0.0],
            context_tokens=[[2, 3], [2, 3]],
            processor_histories=[history, history],
            logits_processors=[[first], [second]],
        )
        assert tokens == [[7], [9]]
        assert draft.propose_complete_paths(_context(history, processors=(first,)), 1)[0].tokens == (7,)
        assert draft.propose_complete_paths(_context(history, processors=(second,)), 1)[0].tokens == (9,)

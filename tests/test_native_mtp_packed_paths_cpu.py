"""Full conditioned MTP proposals; no target TensorFold capability claimed."""
import copy

import mlx.core as mx
import numpy as np
import pytest
from test_apc_hits_hybrid_gdn_self_mtp import tiny_qwen4_mtp

from mlx2.runtime.drafters.native_mtp import NativeMTPDraftModel


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def setup():
    model, _ = tiny_qwen4_mtp()
    model.select_external_feature_source("mtp_seed")
    prompt = [1, 7, 3, 9, 2, 8]
    seeds = model.prefill_body(mx.array([prompt]), model.make_cache(), (1,))
    return model, seeds, [prompt[1:] + [5]]


@pytest.mark.parametrize("pattern", [(5, 5, 5), (3, 3, 3, 3, 3), (4, 4, 4, 3)])
def test_each_branch_matches_an_independent_conditioned_head_unroll(pattern):
    model, seeds, pairing = setup()
    draft = NativeMTPDraftModel(model, num_draft=max(pattern), revision="a" * 64)
    cache = draft.make_cache()
    paths = draft.draft_packed_paths(
        [5], seeds, cache, pattern, context_tokens=pairing,
    )[0]
    assert [len(path.tokens) for path in paths] == list(pattern)
    assert len({path.tokens[0] for path in paths}) == len(pattern)
    assert draft.stats["committed_positions"] == seeds.shape[1]
    # Exactly one primary unroll, then only the alternative suffixes.
    assert draft.stats["chain_steps"] == sum(depth - 1 for depth in pattern)
    reference = model.make_mtp_cache()
    model.mtp_step(seeds[:, :-1], mx.array([pairing[0][:-1]], mx.uint32), reference)
    logits, seed = model.mtp_step(seeds[:, -1:], mx.array([[5]], mx.uint32), reference)
    root_values = np.asarray(logits[0, -1]).astype(np.float64)
    expected_roots = np.argsort(-root_values, kind="stable")[:len(pattern)].tolist()
    assert [path.tokens[0] for path in paths] == expected_roots
    for path in paths:
        state, hidden = copy.deepcopy(reference), seed[:, -1:]
        expected = [path.tokens[0]]
        for _ in range(len(path.tokens) - 1):
            top, hidden = model.mtp_step(hidden, mx.array([[expected[-1]]]), state)
            expected.append(int(mx.argmax(top[0, -1]).item()))
        assert tuple(expected) == path.tokens
        assert len(path.confidence_features) == len(path.tokens)
        assert all(np.isfinite(value) for value in path.confidence_features)
    # Proposal generation must not write any unverified token to the live head.
    for actual, expected in zip(cache, reference):
        assert actual.offset == expected.offset
        for lhs, rhs in zip(actual.state, expected.state):
            np.testing.assert_array_equal(np.asarray(lhs), np.asarray(rhs))


@pytest.mark.parametrize("pattern", [(), (16,), (5, 5, 6), (True,), (0,), (-1,)])
def test_invalid_packing_is_refused_before_committed_head_mutation(pattern):
    model, seeds, pairing = setup()
    draft = NativeMTPDraftModel(model, num_draft=5, revision="a" * 64)
    with pytest.raises(ValueError, match="packed MTP"):
        draft.draft_packed_paths([5], seeds, draft.make_cache(), pattern, context_tokens=pairing)
    assert draft.stats["committed_positions"] == 0


def test_declared_depth_is_not_silently_extended():
    model, seeds, pairing = setup()
    draft = NativeMTPDraftModel(model, num_draft=3, revision="a" * 64)
    with pytest.raises(ValueError, match="capability"):
        draft.draft_packed_paths([5], seeds, draft.make_cache(), (5, 5, 5), context_tokens=pairing)
    assert draft.stats["committed_positions"] == 0


def test_branch_failure_keeps_live_head_at_verified_context(monkeypatch):
    model, seeds, pairing = setup()
    draft = NativeMTPDraftModel(model, num_draft=3, revision="a" * 64)
    cache = draft.make_cache()
    original = draft._step
    calls = 0

    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 5:  # context2 + primary2, then first alternative
            raise RuntimeError("injected branch failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(draft, "_step", fail)
    with pytest.raises(RuntimeError, match="injected"):
        draft.draft_packed_paths([5], seeds, cache, (3, 3, 3, 3, 3), context_tokens=pairing)
    assert all(layer.offset == seeds.shape[1] for layer in cache)


@pytest.mark.parametrize("topology", ["3x5", "5x3", "4/4/4/3", "chain"])
def test_lazy_tree_tile_matches_synced_packed_paths(topology):
    from mlx2.runtime.ast_tree_proposals import paths_to_tree, pattern_for_budget

    model, seeds, pairing = setup()
    pattern = pattern_for_budget(topology, 15)
    depth = max(pattern)
    synced = NativeMTPDraftModel(model, num_draft=depth, revision="a" * 64)
    lazy = NativeMTPDraftModel(model, num_draft=depth, revision="a" * 64)
    synced_cache, lazy_cache = synced.make_cache(), lazy.make_cache()
    if len(pattern) > 1:
        reference = [path.tokens for path in synced.draft_packed_paths(
            [5], seeds, synced_cache, pattern, context_tokens=pairing)[0]]
    else:
        reference = [tuple(synced.draft_distributions(
            [5], seeds, synced_cache, depth, [None], [0.0], context_tokens=pairing)[0][0])]
    tree = lazy.propose_tree([5], seeds, lazy_cache, 15, context_tokens=pairing,
                             topology=topology)[0]
    assert tree == paths_to_tree(reference, 15)
    assert lazy.stats["committed_positions"] == seeds.shape[1]
    for actual, expected in zip(lazy_cache, synced_cache):
        assert actual.offset == expected.offset
        for lhs, rhs in zip(actual.state, expected.state):
            np.testing.assert_array_equal(np.asarray(lhs), np.asarray(rhs))


@pytest.mark.parametrize("topology", ["3x5", "5x3", "4/4/4/3"])
def test_batched_branches_on_a_plain_kv_head_match_independent_unrolls(topology):
    """A plain KV head (Qwen3.8) drafts every branch in one step per depth."""
    from route_harness import tiny_qwen38_mtp

    from mlx2.runtime.ast_tree_proposals import paths_to_tree, pattern_for_budget

    model, _ = tiny_qwen38_mtp()
    object.__setattr__(model, "external_feature_source", "mtp_seed")
    object.__setattr__(model, "external_feature_width", 64)
    prompt = [1, 7, 3, 9, 2, 8]
    seeds = model.model(mx.array([prompt]), model.make_cache())
    pairing = [prompt[1:] + [5]]
    pattern = pattern_for_budget(topology, 15)
    draft = NativeMTPDraftModel(model, num_draft=max(pattern), revision="a" * 64)
    tree = draft.propose_tree([5], seeds, draft.make_cache(), 15, context_tokens=pairing,
                              topology=topology)[0]
    assert draft.stats["batched_tree_calls"] == 1
    reference = model.make_mtp_cache()
    model.mtp_step(seeds[:, :-1], mx.array([pairing[0][:-1]], mx.uint32), reference)
    logits, seed = model.mtp_step(seeds[:, -1:], mx.array([[5]], mx.uint32), reference)
    roots = np.argsort(-np.asarray(logits[0, -1]).astype(np.float64), kind="stable")[:len(pattern)]
    paths = []
    for root, depth in zip(roots.tolist(), pattern):
        state, hidden, path = copy.deepcopy(reference), seed[:, -1:], [root]
        while len(path) < depth:
            top, hidden = model.mtp_step(hidden, mx.array([[path[-1]]], mx.uint32), state)
            path.append(int(mx.argmax(top[0, -1]).item()))
        paths.append(tuple(path))
    assert tree == paths_to_tree(paths, 15)

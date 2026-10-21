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

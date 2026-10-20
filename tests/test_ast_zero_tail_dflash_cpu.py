"""Committed-context scheduling equivalence on tiny CPU DFlash2 models."""

import copy

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from test_dflash_long_block import tiny

from mlx2.runtime.speculative_sampling import RequestRNG


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("offset", [0, 7])
def test_preappend_then_empty_b4_preserves_context_and_continuation(dtype, offset):
    """Exercise empty/mixed tails, nonzero RoPE, and a wrapped sliding ring.

    The fixture has one sliding layer and one full-attention layer. Context
    projection remains per lane; only the proposal call becomes B4. That
    shape can change float32 rounding, so compare its law numerically while
    requiring exact tokens, resident KV and subsequent same-shape decoding.
    """
    target, draft = tiny()
    target.set_dtype(dtype)
    draft.set_dtype(dtype)
    tails = [mx.random.normal((1, length, 16)).astype(dtype) for length in (0, 1, 2, 4)]
    initial = [draft.make_cache() for _ in tails]
    if offset:
        for cache in initial:
            draft.append_context(mx.random.normal((1, offset, 16)).astype(dtype), cache)
    reference = copy.deepcopy(initial)
    candidate = copy.deepcopy(initial)
    reference_rngs = [RequestRNG(100 + row) for row in range(4)]
    candidate_rngs = [RequestRNG(100 + row) for row in range(4)]
    reference_tokens, reference_laws = [], []
    for row, (tail, cache) in enumerate(zip(tails, reference, strict=True)):
        tokens, laws = draft.draft_distributions(
            [row + 1],
            tail,
            draft.batch_caches([cache]),
            3,
            [reference_rngs[row]],
            [0.7],
        )
        reference_tokens.append(tokens[0])
        reference_laws.append(laws[0])
    for tail, cache in zip(tails, candidate, strict=True):
        if tail.shape[1]:
            draft.append_context(tail, cache)
    tokens, laws = draft.draft_distributions(
        [1, 2, 3, 4],
        mx.zeros((4, 0, 16), dtype=dtype),
        draft.batch_caches(candidate),
        3,
        candidate_rngs,
        [0.7] * 4,
    )
    assert tokens == reference_tokens
    assert [rng.draws for rng in reference_rngs] == [
        rng.draws for rng in candidate_rngs
    ]
    np.testing.assert_allclose(laws, reference_laws, atol=1e-5, rtol=0)
    for original, preappended in zip(reference, candidate, strict=True):
        for expected, actual in zip(original, preappended, strict=True):
            assert actual.offset == expected.offset
            if hasattr(expected, "_idx"):
                assert actual._idx == expected._idx
            for a, b in zip(
                expected.keys_and_values(), actual.keys_and_values(), strict=True
            ):
                np.testing.assert_array_equal(
                    np.asarray(a.astype(mx.float32)),
                    np.asarray(b.astype(mx.float32)),
                )
    next_hidden = mx.random.normal((4, 1, 16)).astype(dtype)
    continuations = [
        draft.draft_distributions(
            [4, 3, 2, 1],
            next_hidden,
            draft.batch_caches(cache_rows),
            3,
            [RequestRNG(200 + row) for row in range(4)],
            [0.7] * 4,
        )
        for cache_rows in (reference, candidate)
    ]
    assert continuations[0][0] == continuations[1][0]
    np.testing.assert_array_equal(continuations[0][1], continuations[1][1])

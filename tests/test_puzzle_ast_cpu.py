"""Synthetic CPU Puzzle capture and AST contracts; no checkpoint qualification."""

import copy

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.attention_tree_executor import AttentionTreeExecutor
from mlx2.runtime.models.gpt_oss_puzzle import Model, ModelArgs


def tiny_puzzle():
    mx.random.seed(17)
    args = ModelArgs(
        num_hidden_layers=3, num_experts_per_tok=2, vocab_size=64, hidden_size=32,
        intermediate_size=32, head_dim=8, num_attention_heads=4, num_key_value_heads=2,
        block_configs=[{"num_local_experts": n, "sliding_window": w}
                       for n, w in ((4, 3), (2, None), (4, 5))],
    )
    model = Model(args)
    model.eval()
    mx.eval(model.parameters())
    return model


def test_puzzle_taps_match_ordinary_and_body_skips_head(monkeypatch):
    model = tiny_puzzle()
    tokens = mx.array([[1, 2, 3, 4]])
    expected = model(tokens, cache=model.make_cache())
    logits, features = model.forward_with_taps(tokens, model.make_cache(), (0, 2))
    np.testing.assert_array_equal(np.asarray(logits), np.asarray(expected))
    assert features.shape == (1, 4, 64)

    def forbidden_head(_):
        raise AssertionError("body-only prefill projected vocabulary logits")

    monkeypatch.setattr(model, "lm_head", forbidden_head)
    body = model.prefill_body(tokens, model.make_cache(), (0, 2))
    np.testing.assert_array_equal(np.asarray(body), np.asarray(features))
    assert model.speculative_args is model.args


@pytest.mark.parametrize("layers", [(), (2, 0), (0, 0), (-1,), (3,)])
def test_puzzle_refuses_invalid_capture_layers(layers):
    model = tiny_puzzle()
    with pytest.raises(ValueError, match="capture layers"):
        model.prefill_body(mx.array([[1, 2]]), model.make_cache(), layers)


def test_puzzle_tree_rows_and_committed_continuation_match_serial():
    model = tiny_puzzle()
    cache = model.make_cache()
    model(mx.array([[1, 2, 3, 4, 5, 6]]), cache=cache)
    mx.eval([c.state for c in cache])
    original = copy.deepcopy(cache)
    logits, features, txn = AttentionTreeExecutor.create().forward(
        model, [7, 8, 9, 10], [-1, 0, 1, 0], cache, (0, 2), cached=True)
    assert features.shape == (1, 4, 64)
    for row, path in enumerate(([7], [7, 8], [7, 8, 9], [7, 10])):
        reference = copy.deepcopy(original)
        for token in path:
            expected = model(mx.array([[token]]), cache=reference)
        np.testing.assert_allclose(np.asarray(logits[0, row]), np.asarray(expected[0, -1]),
                                   atol=2e-4, rtol=2e-4)
    txn.commit_paths([[0, 3]])
    actual = model(mx.array([[11]]), cache=cache)
    reference = copy.deepcopy(original)
    for token in (7, 10, 11):
        expected = model(mx.array([[token]]), cache=reference)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=2e-4, rtol=2e-4)

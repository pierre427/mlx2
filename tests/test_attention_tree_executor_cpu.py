"""Attention-only tree tiles (gpt-oss style: sliding + full layers, sinks)."""

import copy

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.attention_tree_executor import AttentionTreeExecutor
from mlx2.runtime.models.gpt_oss import Model, ModelArgs


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def tiny(window=6):
    args = ModelArgs(
        num_hidden_layers=4, num_local_experts=4, num_experts_per_tok=2,
        vocab_size=64, hidden_size=32, intermediate_size=32, head_dim=8,
        num_attention_heads=4, num_key_value_heads=2, sliding_window=window,
    )
    mx.random.seed(3)
    model = Model(args)
    model.eval()
    model.update(model.parameters())
    # Non-zero sinks exercise the sink path through the tree mask.
    for layer in model.model.layers:
        layer.self_attn.sinks = mx.random.normal((args.num_attention_heads,))
    mx.eval(model.parameters())
    return model


def _prefill(model, prompt):
    cache = model.make_cache()
    model(mx.array([prompt]), cache=cache)
    mx.eval([c.state for c in cache])
    return cache


def _serial_last_logits(model, cache, tokens):
    cache = copy.deepcopy(cache)
    logits = None
    for token in tokens:
        logits = model(mx.array([[token]]), cache=cache)[0, -1]
    return np.asarray(logits.astype(mx.float32)), cache


TREES = [
    ([7, 3, 9, 11, 2], [-1, 0, 1, 2, 3]),                    # chain
    ([7, 3, 9, 11, 2, 5, 8], [-1, 0, 1, 0, 3, 4, 0]),        # three branches
    ([7, 3], [-1, 0]),
]


@pytest.mark.parametrize("prompt_len", [4, 13])          # inside and past the window
@pytest.mark.parametrize("tokens, parents", TREES)
def test_every_tile_row_matches_serial_decoding_of_its_path(prompt_len, tokens, parents):
    model = tiny()
    prompt = [(5 * i + 1) % 60 for i in range(prompt_len)]
    cache = _prefill(model, prompt)
    reference_cache = copy.deepcopy(cache)
    executor = AttentionTreeExecutor.create()
    logits, features, txn = executor.forward(model, tokens, parents, cache, (3,), cached=True)
    mx.eval(logits, features)
    assert features.shape == (1, len(tokens), 32)
    for row in range(len(tokens)):
        path, node = [], row
        while node >= 0:
            path.append(tokens[node])
            node = parents[node]
        expected, _ = _serial_last_logits(model, reference_cache, list(reversed(path)))
        np.testing.assert_allclose(np.asarray(logits[0, row].astype(mx.float32)), expected,
                                   atol=2e-4, rtol=2e-4)
    txn.abort()
    for item, original in zip(cache, reference_cache):
        assert item.offset == original.offset


@pytest.mark.parametrize("prompt_len", [4, 13])
def test_commit_keeps_the_path_and_continues_like_serial_decoding(prompt_len):
    model = tiny()
    prompt = [(3 * i + 2) % 60 for i in range(prompt_len)]
    cache = _prefill(model, prompt)
    reference = copy.deepcopy(cache)
    tokens, parents = TREES[1]
    executor = AttentionTreeExecutor.create()
    logits, _features, txn = executor.forward(model, tokens, parents, cache, (3,), cached=True)
    mx.eval(logits)
    path = [0, 3, 4, 5]                                    # anchor -> 11 -> 2 -> 5
    txn.commit_paths([path])
    after = model(mx.array([[17]]), cache=cache)[0, -1]
    expected, serial_cache = _serial_last_logits(model, reference, [tokens[i] for i in path] + [17])
    np.testing.assert_allclose(np.asarray(after.astype(mx.float32)), expected, atol=2e-4, rtol=2e-4)
    for item, serial in zip(cache, serial_cache):
        assert item.offset == serial.offset


def test_cohort_lanes_stay_independent():
    model = tiny()
    caches = [_prefill(model, [1, 2, 3, 4, 5]), _prefill(model, [9, 8, 7])]
    references = [copy.deepcopy(c) for c in caches]
    tokens = [[7, 3, 9], [4, 4, 1]]
    parents = [[-1, 0, 0], [-1, 0, 1]]
    executor = AttentionTreeExecutor.create()
    logits, _features, txn = executor.forward_many(
        model, tokens, parents, caches, (3,), cached=True, configured_limit=4)
    mx.eval(logits)
    for lane in range(2):
        expected, _ = _serial_last_logits(model, references[lane], [tokens[lane][0], tokens[lane][2]]
                                          if parents[lane][2] == 0 else tokens[lane])
        np.testing.assert_allclose(np.asarray(logits[lane, 2].astype(mx.float32)), expected,
                                   atol=2e-4, rtol=2e-4)
    txn.abort()

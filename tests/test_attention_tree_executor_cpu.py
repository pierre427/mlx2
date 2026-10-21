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


def _greedy(model, prompt, count):
    cache = model.make_cache()
    logits = model(mx.array([prompt]), cache=cache)[0, -1]
    out = []
    for _ in range(count):
        token = int(mx.argmax(logits).item())
        out.append(token)
        logits = model(mx.array([[token]]), cache=cache)[0, -1]
    return out


def test_external_route_with_prompt_lookup_tiles_matches_greedy_under_admission():
    """gpt-oss --ast end to end: PLD tree drafter, attention tree target,
    admission memory accounting on (the model-free drafter has no draft KV)."""
    from mlx2.runtime.attention_tree_executor import AttentionTreeExecutor
    from mlx2.runtime.drafters.prompt_lookup_draft import PromptLookupDraftModel
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    model = tiny(window=6)
    prompt = [3, 9, 4, 3, 9, 4, 3, 9, 4, 11, 3, 9]
    reference = _greedy(model, prompt, 24)
    draft = PromptLookupDraftModel(model, num_draft=15)
    batch = ExternalDraftBatchGenerator(
        model, draft_model=draft, binding="tiny-gpt-oss", num_draft=15,
        completion_batch_size=4, dynamic_singleton_tree=True, dynamic_tree_max_width=4,
        tree_node_budget_by_lanes={1: 15, 2: 7, 3: 4, 4: 3},
        target_tree_executor_provider=AttentionTreeExecutor.create(),
        memory_headroom=lambda: 1 << 40,
    )
    uid = batch.insert([prompt], max_tokens=[24], sampling_configs=[{"sampling_temp": 0.0}])[0]
    tokens = []
    for _ in range(200):
        _, responses = batch.next()
        tokens.extend(r.token for r in responses if r.uid == uid)
        if uid not in batch.lanes:
            break
    assert tokens == reference
    assert batch.scheduler_stats["external_tree_rounds"] > 0


def test_pad_cohort_trees_adds_unused_anchor_leaves():
    from mlx2.runtime.external_speculative import _pad_cohort_trees

    inputs = [[7, 0, 1, 2], [9, 0]]
    parents = [[-1, 0, 1, 0], [-1, 0]]
    padded, tree = _pad_cohort_trees(inputs, parents)
    assert padded[0] == inputs[0] and tree[0] == parents[0]
    assert padded[1] == [9, 0, 1, 2] and tree[1] == [-1, 0, 0, 0]
    same = [[1, 2], [3, 4]]
    assert _pad_cohort_trees(same, [[-1, 0], [-1, 0]])[0] is same


class _UniformCohortExecutor(AttentionTreeExecutor):
    """The TensorFold cohort contract: one width for every lane of a tile."""

    def forward_many(self, model, inputs, parents, *args, **kwargs):
        if len({len(row) for row in inputs}) != 1:
            raise ValueError("TensorFold cohort requires uniform token/parent widths")
        return super().forward_many(model, inputs, parents, *args, **kwargs)


def test_ragged_cohort_trees_are_padded_to_one_uniform_tile():
    """Lanes of one tree round may propose trees of different sizes (prompt
    lookup finds fewer copies for one lane: the Qwen3.6-35B --ast B4 crash of
    2026-10-21). The cohort tile must stay uniform and every real row must
    still equal serial decoding of its path."""
    from mlx2.runtime.drafters.prompt_lookup_draft import PromptLookupDraftModel
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    model = tiny(window=6)
    prompts = [[3, 9, 4, 3, 9, 4, 3, 9, 4, 11, 3, 9], [5, 17, 29, 41, 2, 13]]
    batch = ExternalDraftBatchGenerator(
        model, draft_model=PromptLookupDraftModel(model, num_draft=15),
        binding="tiny-gpt-oss", num_draft=15, completion_batch_size=4,
        dynamic_singleton_tree=True, dynamic_tree_max_width=4,
        tree_node_budget_by_lanes={1: 15, 2: 7, 3: 4, 4: 3},
        target_tree_executor_provider=_UniformCohortExecutor.create(),
        memory_headroom=lambda: 1 << 40,
    )
    batch.insert(prompts, max_tokens=[20, 20],
                 sampling_configs=[{"sampling_temp": 0.0}] * 2)
    for _ in range(20):
        batch.next()
        if len(batch.lanes) == 2 and all(lane.anchor is not None for lane in batch.lanes.values()):
            break
    cohort = list(batch.lanes.values())
    references = [copy.deepcopy(lane.cache) for lane in cohort]
    inputs = [[cohort[0].anchor, 1, 2, 3], [cohort[1].anchor, 4]]
    parents = [[-1, 0, 1, 2], [-1, 0]]
    logits, _features, transaction = batch._target_tree_forward_many(cohort, inputs, parents)
    mx.eval(logits)
    assert logits.shape[1] == 4
    for lane, (tokens, tree) in enumerate(zip(inputs, parents)):
        for row in range(len(tokens)):
            path, node = [], row
            while node >= 0:
                path.append(tokens[node])
                node = tree[node]
            expected, _ = _serial_last_logits(model, references[lane], list(reversed(path)))
            np.testing.assert_allclose(np.asarray(logits[lane, row].astype(mx.float32)), expected,
                                       atol=2e-4, rtol=2e-4)
    transaction.abort()

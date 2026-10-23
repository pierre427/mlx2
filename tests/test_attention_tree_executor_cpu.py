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


def _assert_committed_cache_equal(actual, expected):
    for item, reference in zip(actual, expected, strict=True):
        assert item.offset == reference.offset
        assert getattr(item, "_idx", None) == getattr(reference, "_idx", None)
        for value, want in zip(item.keys_and_values(), reference.keys_and_values(), strict=True):
            np.testing.assert_array_equal(np.asarray(value), np.asarray(want))


@pytest.mark.parametrize("window", [None, 1, 3, 6])
@pytest.mark.parametrize("prefix", [0, 3, 17])
def test_tree_mask_matches_absolute_position_oracle(window, prefix):
    from mlx2.runtime.attention_tree_executor import _TileScope
    from mlx2.runtime.models.cache import KVCache

    parents = [-1, 0, 1, 2, 1, 4, 5, 6]
    cache = KVCache()
    cache.offset = prefix
    scope = _TileScope(None, [cache], parents)
    actual = scope._mask_for(cache)(len(parents), window_size=window)
    expected = []
    for node in range(len(parents)):
        path, cursor = [], node
        while cursor >= 0:
            path.append(cursor)
            cursor = parents[cursor]
        position = prefix + len(path) - 1
        visible = [window is None or position - key < window for key in range(prefix)]
        for key in range(len(parents)):
            depth, cursor = 0, key
            while parents[cursor] >= 0:
                depth += 1
                cursor = parents[cursor]
            visible.append(key in path and (window is None or position - prefix - depth < window))
        expected.append(visible)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_large_tree_mask_keeps_python_array_construction_bounded(monkeypatch):
    from mlx2.runtime.attention_tree_executor import _TileScope
    from mlx2.runtime.models.cache import KVCache

    cache = KVCache()
    cache.offset = 131072
    original = mx.array
    constructed = []

    def bounded_array(value, *args, **kwargs):
        def entries(part):
            return sum(entries(v) for v in part) if isinstance(part, (list, tuple)) else 1
        if isinstance(value, (list, tuple)):
            size = entries(value)
            constructed.append(size)
            assert size <= 32 * 32
        return original(value, *args, **kwargs)

    monkeypatch.setattr(mx, "array", bounded_array)
    scope = _TileScope(None, [cache], [-1] + [0] * 31)
    mask = scope._mask_for(cache)(32)
    assert mask.shape == (32, 131104)
    assert max(constructed) == 32 * 32
    assert bool(mx.all(mask[:, :cache.offset]).item())
    np.testing.assert_array_equal(np.asarray(mask[-1, cache.offset:]),
                                  [True] + [False] * 30 + [True])


def test_scope_shares_depth_tensor_and_restores_existing_mask():
    from mlx2.runtime.attention_tree_executor import _TileScope, attention_modules

    model = tiny()
    cache = model.make_cache()
    original = [module.rope for module in attention_modules(model)]
    override = lambda *args, **kwargs: None
    cache[0].make_mask = override
    with _TileScope(model, cache, [-1, 0, 0]) as scope:
        assert all(module.rope.depths is scope.depths for module in attention_modules(model))
    assert all(module.rope is rope for module, rope in zip(attention_modules(model), original))
    assert cache[0].make_mask is override
    assert all("make_mask" not in item.__dict__ for item in cache[1:])


def test_partial_scope_entry_restores_all_installed_overrides(monkeypatch):
    from mlx2.runtime.attention_tree_executor import _TileScope, attention_modules

    model = tiny()
    cache = model.make_cache()
    original_rope = [module.rope for module in attention_modules(model)]
    override = lambda *args, **kwargs: None
    cache[0].make_mask = override
    original_mask_for = _TileScope._mask_for

    def fail_second(self, item):
        if item is cache[1]:
            raise RuntimeError("injected mask setup failure")
        return original_mask_for(self, item)

    monkeypatch.setattr(_TileScope, "_mask_for", fail_second)
    with pytest.raises(RuntimeError, match="injected mask setup failure"):
        with _TileScope(model, cache, [-1, 0]):
            pytest.fail("entry should fail")
    assert all(module.rope is rope for module, rope in zip(attention_modules(model), original_rope))
    assert cache[0].make_mask is override
    assert all("make_mask" not in item.__dict__ for item in cache[1:])


@pytest.mark.parametrize("extra_steps", [0, 2])
def test_one_row_full_ring_abort_restores_values_and_continuation(extra_steps):
    model = tiny(window=6)
    cache = _prefill(model, [1, 2, 3, 4, 5, 6])
    for token in range(10, 10 + extra_steps):
        mx.eval(model(mx.array([[token]]), cache=cache))
    reference = copy.deepcopy(cache)
    executor = AttentionTreeExecutor.create()
    logits, features, transaction = executor.forward(model, [17], [-1], cache, (3,), cached=True)
    mx.eval(logits, features)
    transaction.abort()
    _assert_committed_cache_equal(cache, reference)
    actual = model(mx.array([[19]]), cache=cache)[0, -1]
    expected, _ = _serial_last_logits(model, reference, [19])
    np.testing.assert_allclose(np.asarray(actual), expected, atol=2e-4, rtol=2e-4)


def test_forward_exception_restores_ring_and_model_scope(monkeypatch):
    from mlx2.runtime.attention_tree_executor import attention_modules

    model = tiny()
    cache = _prefill(model, [1, 2, 3, 4, 5, 6])
    reference = copy.deepcopy(cache)
    original_rope = [module.rope for module in attention_modules(model)]
    original_forward = model.forward_with_taps

    def fail_after_forward(*args, **kwargs):
        outputs = original_forward(*args, **kwargs)
        mx.eval(*outputs)
        raise RuntimeError("injected target failure")

    monkeypatch.setattr(model, "forward_with_taps", fail_after_forward)
    with pytest.raises(RuntimeError, match="injected target failure"):
        AttentionTreeExecutor.create().forward(model, [17], [-1], cache, (3,), cached=True)
    _assert_committed_cache_equal(cache, reference)
    assert all(module.rope is rope for module, rope in zip(attention_modules(model), original_rope))
    assert all("make_mask" not in item.__dict__ for item in cache)


def test_cohort_commit_failure_rolls_back_already_committed_lanes(monkeypatch):
    from mlx2.runtime import attention_tree_executor as tree

    model = tiny()
    caches = [_prefill(model, list(range(1, 7))), _prefill(model, list(range(7, 13)))]
    references = copy.deepcopy(caches)
    logits, features, transaction = AttentionTreeExecutor.create().forward_many(
        model, [[17], [19]], [[-1], [-1]], caches, (3,), cached=True, configured_limit=2)
    mx.eval(logits, features)
    original_compact = tree._compact
    calls = 0

    def fail_second(*args):
        nonlocal calls
        original_compact(*args)
        calls += 1
        if calls == 2:
            raise RuntimeError("injected compaction failure")

    monkeypatch.setattr(tree, "_compact", fail_second)
    with pytest.raises(RuntimeError, match="injected compaction failure"):
        transaction.commit_paths([[0], [0]])
    assert transaction.closed and all(tx.closed for tx in transaction.transactions)
    for cache, reference in zip(caches, references, strict=True):
        _assert_committed_cache_equal(cache, reference)


@pytest.mark.parametrize("path", [[], [1], [0, 0], [0, 2, 1], [0, 1, 2], [0, 3]])
def test_commit_rejects_non_paths_before_publishing(path):
    model = tiny()
    cache = _prefill(model, [1, 2, 3, 4, 5, 6])
    reference = copy.deepcopy(cache)
    logits, features, transaction = AttentionTreeExecutor.create().forward(
        model, [17, 19, 23], [-1, 0, 0], cache, (3,), cached=True)
    mx.eval(logits, features)
    with pytest.raises(ValueError, match="root-connected path"):
        transaction.commit_paths([path])
    assert not transaction.closed
    transaction.abort()
    _assert_committed_cache_equal(cache, reference)


def test_chain_deeper_than_sliding_window_matches_serial():
    model = tiny(window=3)
    cache = _prefill(model, [1, 2, 3, 4, 5])
    reference = copy.deepcopy(cache)
    tokens = [7, 3, 9, 11, 2, 5, 8, 13]
    logits, features, transaction = AttentionTreeExecutor.create().forward(
        model, tokens, list(range(-1, len(tokens) - 1)), cache, (3,), cached=True)
    mx.eval(logits, features)
    for row in range(len(tokens)):
        expected, _ = _serial_last_logits(model, reference, tokens[:row + 1])
        np.testing.assert_allclose(np.asarray(logits[0, row]), expected, atol=2e-4, rtol=2e-4)
    transaction.abort()


def test_cohort_result_assembly_failure_aborts_all_lanes(monkeypatch):
    model = tiny()
    caches = [_prefill(model, list(range(1, 7))), _prefill(model, list(range(7, 13)))]
    references = copy.deepcopy(caches)
    original_concat = mx.concatenate

    def fail_logits_join(arrays, axis=0, **kwargs):
        if axis == 0 and len(arrays) == 2 and all(value.shape == (1, 1, 64) for value in arrays):
            raise RuntimeError("injected cohort assembly failure")
        return original_concat(arrays, axis=axis, **kwargs)

    monkeypatch.setattr(mx, "concatenate", fail_logits_join)
    with pytest.raises(RuntimeError, match="injected cohort assembly failure"):
        AttentionTreeExecutor.create().forward_many(
            model, [[17], [19]], [[-1], [-1]], caches, (3,), cached=True, configured_limit=2)
    for cache, reference in zip(caches, references, strict=True):
        _assert_committed_cache_equal(cache, reference)


@pytest.mark.parametrize("shared_list", [False, True])
def test_cohort_refuses_aliased_owners_before_target_mutation(shared_list):
    model = tiny()
    cache = _prefill(model, list(range(1, 7)))
    reference = copy.deepcopy(cache)
    second = cache if shared_list else list(cache)
    with pytest.raises(ValueError, match="independent cache owners"):
        AttentionTreeExecutor.create().forward_many(
            model, [[17], [19]], [[-1], [-1]], [cache, second], (3,), cached=True, configured_limit=2)
    _assert_committed_cache_equal(cache, reference)

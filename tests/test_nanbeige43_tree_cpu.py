"""Synthetic only; no real checkpoint or qualification claim.

Source binding: base 89e6938ea58a0060026815c94e0ed4d33729d775 plus working
nanbeige43_tree.py/nanbeige43_attention.py identities recorded by the provider.
"""
import copy
import hashlib
import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters.nanbeige43_tree import NanbeigePackedTreeExecutor
from mlx2.runtime.models.standard_decoder import Model, ModelArgs


@pytest.fixture(autouse=True)
def source_receipt(record_property):
    root = Path(__file__).resolve().parents[1]
    record_property("source_revision", subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip())
    files = [root / "src/mlx2/adapters" / name for name in
             ("nanbeige43_tree.py", "nanbeige43_attention.py", "nanbeige43_pld.py")]
    record_property("working_source_sha256", hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest())
    fixture = Path(__file__).read_bytes()
    record_property("checkpoint", "synthetic-nanbeige43-tree-fixtures@sha256:"+hashlib.sha256(fixture).hexdigest())


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def tiny():
    mx.random.seed(47)
    model = Model(ModelArgs(model_type="llama", hidden_size=56, intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=7, num_key_value_heads=1, head_dim=8,
        vocab_size=64, rms_norm_eps=1e-5, max_position_embeddings=4096,
        rope_theta=120000000, tie_word_embeddings=False))
    mx.eval(model.parameters())
    return model


def prefill(model, tokens):
    cache = model.make_cache()
    if tokens:
        mx.eval(model(mx.array([tokens]), cache))
    return cache


def serial(model, cache, tokens):
    for token in tokens:
        logits = model(mx.array([[token]]), cache)
    return logits


def path(tree, row):
    out = []
    while row >= 0:
        out.append(row)
        row = tree[row]
    return out[::-1]


@pytest.mark.parametrize("prefixes", [(0, 0), (2, 5), (257, 3)])
def test_packed_one_traversal_all_rows_and_taps_match_paths(prefixes, monkeypatch):
    model = tiny()
    caches = [prefill(model, [i % 64 for i in range(length)]) for length in prefixes]
    reference = copy.deepcopy(caches)
    tokens = [[1, 3, 5, 7], [2, 4, 6, 8]]
    parents = [[-1, 0, 0, 2], [-1, 0, 1, 0]]
    original = model.forward_with_taps
    shapes = []
    def counted(inputs, *args, **kwargs):
        shapes.append(inputs.shape)
        return original(inputs, *args, **kwargs)
    monkeypatch.setattr(model, "forward_with_taps", counted)
    provider = NanbeigePackedTreeExecutor.create()
    logits, taps, txn = provider.forward_many(model, tokens, parents, caches, (0, 2), cached=True, configured_limit=2)
    mx.eval(logits, taps)
    assert shapes == [(2, 4)]
    assert taps.shape == (2, 4, 112)
    assert provider.dispatch_metrics(2, txn, cached=True).physical_packed
    for lane in range(2):
        for row in range(4):
            expected = serial(model, copy.deepcopy(reference[lane]), [tokens[lane][n] for n in path(parents[lane], row)])
            np.testing.assert_allclose(np.asarray(logits[lane, row]), np.asarray(expected[0, -1]), atol=3e-5, rtol=3e-5)
    txn.abort()
    for cache, ref in zip(caches, reference):
        for actual, want in zip(cache, ref):
            assert actual.offset == want.offset
            if want.keys is not None:
                np.testing.assert_array_equal(np.asarray(actual.keys_and_values()[0]), np.asarray(want.keys_and_values()[0]))


def test_different_acceptance_paths_commit_and_continue():
    model = tiny()
    caches = [prefill(model, [1, 2]), prefill(model, [3, 4, 5])]
    references = copy.deepcopy(caches)
    inputs = [[7, 8, 9, 10], [11, 12, 13, 14]]
    parents = [[-1, 0, 0, 2], [-1, 0, 1, 0]]
    logits, taps, txn = NanbeigePackedTreeExecutor.create().forward_many(model, inputs, parents, caches, (2,), cached=False, configured_limit=2)
    mx.eval(logits, taps)
    txn.commit_paths([[0, 2, 3], [0]])
    for lane, kept in enumerate([[0, 2, 3], [0]]):
        actual = model(mx.array([[21]]), caches[lane])
        expected = serial(model, references[lane], [inputs[lane][r] for r in kept]+[21])
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=3e-5, rtol=3e-5)


def test_exception_and_invalid_owner_fail_without_mutation(monkeypatch):
    model = tiny()
    caches = [prefill(model, [1, 2]), prefill(model, [3, 4, 5])]
    references = copy.deepcopy(caches)
    original = model.forward_with_taps
    def broken(*args, **kwargs):
        out = original(*args, **kwargs)
        mx.eval(*out)
        raise RuntimeError("injected")
    monkeypatch.setattr(model, "forward_with_taps", broken)
    provider = NanbeigePackedTreeExecutor.create()
    with pytest.raises(RuntimeError, match="injected"):
        provider.forward_many(model, [[1, 2]]*2, [[-1, 0]]*2, caches, (2,), cached=False, configured_limit=2)
    for cache, ref in zip(caches, references):
        for actual, want in zip(cache, ref):
            assert actual.offset == want.offset
            np.testing.assert_array_equal(np.asarray(actual.keys_and_values()[0]), np.asarray(want.keys_and_values()[0]))
    with pytest.raises(ValueError, match="independent"):
        provider.forward_many(model, [[1]]*2, [[-1]]*2, [caches[0], list(caches[0])], (2,), cached=False, configured_limit=2)
    assert all("rope" not in module.__dict__ for module in [layer.self_attn for layer in model.layers])


def test_implicit_gqa7_cpu_reference_and_neutral_padding():
    from mlx2.adapters.nanbeige43_attention import attend
    mx.random.seed(7)
    q = mx.random.normal((2, 56, 4, 128))
    k = mx.random.normal((2, 8, 11, 128))
    v = mx.random.normal((2, 8, 11, 128))
    ancestors = mx.array([[[True, False, False, False], [True, True, False, False], [True, False, True, False], [True, False, True, True]]]*2)
    prefixes = [7, 1]
    result = attend(q, k, v, prefixes=prefixes, ancestors=ancestors, scale=128**-0.5)
    mx.eval(result)
    for lane, prefix in enumerate(prefixes):
        logical_k = k[lane:lane+1, :, 7-prefix:]
        logical_v = v[lane:lane+1, :, 7-prefix:]
        mask = mx.concatenate([mx.ones((4, prefix), dtype=mx.bool_), ancestors[lane]], axis=1)
        want = mx.fast.scaled_dot_product_attention(q[lane:lane+1], logical_k, logical_v, scale=128**-0.5, mask=mask)
        np.testing.assert_allclose(np.asarray(result[lane:lane+1]), np.asarray(want), atol=3e-5, rtol=3e-5)


def test_implicit_refuses_wrong_geometry_before_cache_write():
    model = tiny()
    cache = prefill(model, [1, 2])
    with pytest.raises(ValueError, match="Q56"):
        NanbeigePackedTreeExecutor.create(implicit_attention=True).forward(model, [7], [-1], cache, (2,), cached=False)
    assert all(item.offset == 2 for item in cache)


def test_adaptive_pld_chain_branch_no_match_and_explicit_topology(monkeypatch):
    from mlx2.adapters.nanbeige43_pld import (
        NanbeigeAdaptivePromptLookup,
        cost_control_defaults,
    )
    from mlx2.runtime.cost_per_commit_handoff import CostPerCommitHandoffSettings
    draft = NanbeigeAdaptivePromptLookup(tiny())
    monkeypatch.setattr(draft, "copy_paths", lambda *args: ([(3, 4, 5, 6, 7, 8), (3, 4, 5, 9)], 6))
    result = draft.propose_tree([1], None, None, 6, processor_histories=[[1, 2]], topology="drafter")
    assert result == [([3, 4, 5, 6, 7, 8], [-1, 0, 1, 2, 3, 4])]
    monkeypatch.setattr(draft, "copy_paths", lambda *args: ([(3, 4, 5, 6), (7, 8, 9)], 2))
    result = draft.propose_tree([1], None, None, 6, processor_histories=[[1, 2]], topology="drafter")
    assert result == [([3, 4, 5, 7, 8, 9], [-1, 0, 1, -1, 3, 4])]
    assert draft.adaptive_stats["branches"] == 1
    monkeypatch.setattr(draft, "copy_paths", lambda *args: ([], 0))
    assert draft.propose_tree([1], None, None, 6, processor_histories=[[1, 2]]) == [([], [])]
    assert draft.adaptive_stats["no_match"] == 1
    # Existing common controller validates these bounds; this policy does not
    # price GPU work from host graph construction or discard startup costs.
    settings = CostPerCommitHandoffSettings(**cost_control_defaults())
    assert settings.settle_rounds == 0 and settings.reprobe_max_tokens == 256
    called = []
    from mlx2.runtime.drafters.prompt_lookup_draft import PromptLookupDraftModel
    monkeypatch.setattr(PromptLookupDraftModel, "propose_tree", lambda *a, **kw: called.append(kw["topology"]) or [([], [])])
    draft.propose_tree([1], None, None, 6, topology="3x2")
    assert called == ["3x2"]


def test_real_index_adaptive_pld_candidates_are_committed_history_copies():
    from mlx2.adapters.nanbeige43_pld import NanbeigeAdaptivePromptLookup
    draft = NanbeigeAdaptivePromptLookup(tiny())
    history = [1, 2, 3, 4, 5, 6, 1, 2, 3]
    result = draft.propose_tree([4], None, None, 6, processor_histories=[history], topology="drafter")
    tokens, parents = result[0]
    assert tokens and tokens[0] == 5
    # Every root branch comes from one previously observed continuation of
    # the query suffix, never from invented filler probabilities/tokens.
    copies, _ = draft.copy_paths(history, 4, 6, 4)
    for row in range(len(tokens)):
        chosen = tuple(tokens[i] for i in path(parents, row))
        assert any(tuple(c[:len(chosen)]) == chosen for c in copies)


def test_adaptive_pld_packed_executor_serving_generator_end_to_end():
    from mlx2.adapters.nanbeige43_pld import NanbeigeAdaptivePromptLookup
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
    model = tiny()
    prompts = [[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3], [6, 5, 4, 3, 6, 5, 4, 3, 6, 5]]
    expected = []
    for prompt in prompts:
        cache = prefill(model, prompt)
        out = []
        # Last prompt row was consumed; get its logits from an independent
        # prefill rather than feeding it twice.
        cache = model.make_cache()
        logits = model(mx.array([prompt]), cache)
        for _ in range(16):
            token = int(mx.argmax(logits[0, -1]).item())
            out.append(token)
            logits = model(mx.array([[token]]), cache)
        expected.append(out)
    batch = ExternalDraftBatchGenerator(model, draft_model=NanbeigeAdaptivePromptLookup(model),
        binding="synthetic-nanbeige43-tree-cpu", num_draft=15, completion_batch_size=2,
        dynamic_singleton_tree=True, dynamic_tree_max_width=4,
        tree_node_budget_by_lanes={1: 15, 2: 7, 3: 4, 4: 3},
        target_tree_executor_provider=NanbeigePackedTreeExecutor.create(),
        memory_headroom=lambda: 1 << 40)
    uids = batch.insert(prompts, max_tokens=[16, 16], sampling_configs=[{"sampling_temp": 0.0}]*2)
    outputs = {uid: [] for uid in uids}
    for _ in range(150):
        _, responses = batch.next()
        for response in responses:
            outputs[response.uid].append(response.token)
        if not batch.lanes:
            break
    assert [outputs[uid] for uid in uids] == expected
    assert batch.scheduler_stats["external_tree_rounds"] > 0


@pytest.mark.parametrize("mode", ["ordinary", "packed", "implicit"])
def test_adapter_selects_explicit_tree_provider_without_mutable_identity(mode, monkeypatch):
    from mlx2.adapters.nanbeige43 import Nanbeige43Adapter
    from mlx2.adapters.standard_decoder import StandardDecoderAdapter
    obj = object.__new__(Nanbeige43Adapter)
    obj.nanbeige_tuning = {"tree": mode}
    obj.nanbeige_fused = False
    monkeypatch.setattr(StandardDecoderAdapter, "create_external_batch", lambda self, **kw: kw)
    kwargs = obj.create_external_batch(marker="preserved")
    assert kwargs["marker"] == "preserved"
    if mode == "ordinary":
        assert "target_tree_executor_provider" not in kwargs
    else:
        provider = kwargs["target_tree_executor_provider"]
        assert provider.implicit_attention == (mode == "implicit")
        identity = provider.source_revision_token
        provider.dispatches += 17
        provider.validate_revision()
        assert provider.source_revision_token == identity
        first = obj.execution_numerics_contract()
        second = obj.execution_numerics_contract()
        assert first == second
        assert first["tree_contract"]["dispatches"] == 0


def test_implicit_context_growth_does_not_specialize_metal_source_per_prefix(monkeypatch):
    from functools import lru_cache

    from mlx2.adapters import nanbeige43_attention as attention
    calls = []
    def partial(**kwargs):
        calls.append(("partial", kwargs["template"]))
        return [mx.zeros(shape, dtype=dtype) for shape, dtype in zip(kwargs["output_shapes"], kwargs["output_dtypes"])]
    def combine(**kwargs):
        calls.append(("combine", kwargs["template"]))
        return [mx.zeros(shape, dtype=dtype) for shape, dtype in zip(kwargs["output_shapes"], kwargs["output_dtypes"])]
    @lru_cache(maxsize=8)
    def fake_factory(scale):
        return partial, combine
    # Dispatch-only fake: underlying default device remains CPU; no GPU
    # ownership or shader invocation is performed by this regression.
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    monkeypatch.setattr(attention, "_kernels", fake_factory)
    trace = []
    for prefix in (0, 257, 258, 259, 510, 513):
        q = mx.zeros((1, 56, 3, 128))
        k = mx.zeros((1, 8, prefix+3, 128))
        attention.attend(q, k, k, prefixes=[prefix], ancestors=mx.ones((1, 3, 3), dtype=mx.bool_),
                         scale=128**-0.5, dispatch_trace=trace)
    for stage in ("partial", "combine"):
        templates = [tuple((key, str(value)) for key, value in template)
                     for kind, template in calls if kind == stage]
        assert len(set(templates)) == 1
        assert all(key not in {"MAX_PREFIX", "SPLITS", "SPAN"} for template in templates for key, _ in template)
    assert len({row["splits"] for row in trace}) > 1
    assert fake_factory.cache_info().misses == 1


def test_shallow_copied_kv_descriptors_refused_but_cow_clones_are_independent():
    model = tiny()
    base = prefill(model, [1, 2, 3])
    before = copy.deepcopy(base)
    aliased = [base, [copy.copy(item) for item in base]]
    assert aliased[0][0].keys is aliased[1][0].keys
    provider = NanbeigePackedTreeExecutor.create()
    with pytest.raises(ValueError, match="independent mutable KV descriptors"):
        provider.forward_many(model, [[7, 8], [9, 10]], [[-1, 0]]*2, aliased, (2,), cached=False, configured_limit=2)
    assert provider.dispatches == 0
    for actual, reference in zip(base, before):
        assert actual.offset == reference.offset
        np.testing.assert_array_equal(np.asarray(actual.keys), np.asarray(reference.keys))
    # Storage sharing is legal; sharing a mutable array descriptor is not.
    proper = [[item.committed_descriptor_clone() for item in base] for _ in range(2)]
    assert proper[0][0].keys is not proper[1][0].keys
    reference = copy.deepcopy(proper)
    logits, taps, txn = provider.forward_many(model, [[7, 8], [9, 10]], [[-1, 0]]*2, proper, (2,), cached=False, configured_limit=2)
    mx.eval(logits, taps)
    txn.commit_paths([[0, 1], [0]])
    for lane, tokens in enumerate([[7, 8], [9]]):
        expected = serial(model, reference[lane], tokens+[17])
        actual = model(mx.array([[17]]), proper[lane])
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("implicit", [False, True])
def test_temporary_memory_estimate_is_conservative_and_giant_geometry_refused(implicit, monkeypatch):
    from mlx2.adapters.nanbeige43_tree import (
        DEFAULT_TEMPORARY_BUDGET_BYTES,
        estimate_temporary_memory,
    )
    from mlx2.runtime.models.cache import KVCache
    geometry = [(8, 128, 128, 2, 2)]*52
    estimate = estimate_temporary_memory(prefixes=[524288, 262144], width=32,
        layer_geometry=geometry, query_heads=56, implicit_attention=implicit)
    assert estimate["total_bytes"] > DEFAULT_TEMPORARY_BUDGET_BYTES
    assert estimate["padded_kv_bytes"] > 0
    # Broadcast shape metadata is huge without allocating/materializing it.
    model = tiny()
    caches = []
    for _ in range(2):
        lane = []
        for _layer in model.layers:
            cache = KVCache()
            cache.offset = 1_000_000_000
            cache.keys = mx.broadcast_to(mx.zeros((1, 1, 1, 8)), (1, 1, cache.offset, 8))
            cache.values = mx.broadcast_to(mx.zeros((1, 1, 1, 8)), (1, 1, cache.offset, 8))
            lane.append(cache)
        caches.append(lane)
    before = [(item, item.offset, item.keys, item.values) for lane in caches for item in lane]
    def forbidden(*args, **kwargs):
        pytest.fail("memory admission must refuse before target forward")
    monkeypatch.setattr(model, "forward_with_taps", forbidden)
    # Both candidate modes use the same guard. Implicit geometry is tested
    # by the pure estimator above; this tiny model intentionally has D8.
    provider = NanbeigePackedTreeExecutor.create()
    with pytest.raises(MemoryError, match="temporary estimate"):
        provider.forward_many(model, [[1, 2]]*2, [[-1, 0]]*2, caches, (2,), cached=False, configured_limit=2)
    assert provider.dispatches == 0
    assert provider.receipt()["temporary_budget_bytes"] == 2 << 30
    assert all(item.offset == offset and item.keys is keys and item.values is values
               for item, offset, keys, values in before)


def test_explicit_smaller_memory_budget_changes_identity_and_refuses_before_write():
    model = tiny()
    caches = [prefill(model, [1, 2]), prefill(model, [3, 4, 5])]
    before = copy.deepcopy(caches)
    provider = NanbeigePackedTreeExecutor.create(temporary_budget_bytes=1)
    assert provider.source_revision_token != NanbeigePackedTreeExecutor.create().source_revision_token
    with pytest.raises(MemoryError, match="budget 1 bytes"):
        provider.forward_many(model, [[1]]*2, [[-1]]*2, caches, (2,), cached=False, configured_limit=2)
    for cache, reference in zip(caches, before):
        for item, want in zip(cache, reference):
            assert item.offset == want.offset
            np.testing.assert_array_equal(np.asarray(item.keys), np.asarray(want.keys))
    with pytest.raises(ValueError, match="at most 2 GiB"):
        NanbeigePackedTreeExecutor.create(temporary_budget_bytes=(2 << 30)+1)


def test_stale_source_and_quantized_kv_refused_before_target(monkeypatch):
    from mlx2.adapters import nanbeige43_tree as tree
    from mlx2.runtime.models.cache import QuantizedKVCache
    model = tiny()
    cache = prefill(model, [1, 2])
    provider = NanbeigePackedTreeExecutor.create()
    monkeypatch.setattr(tree, "source_identity", lambda: "new-source")
    with pytest.raises(RuntimeError, match="source changed"):
        provider.forward(model, [7], [-1], cache, (2,), cached=False)
    assert all(item.offset == 2 for item in cache)
    monkeypatch.undo()
    cache = [QuantizedKVCache(group_size=64, bits=8) for _ in model.layers]
    with pytest.raises(ValueError, match="plain full KVCache"):
        NanbeigePackedTreeExecutor.create().forward(model, [7], [-1], cache, (2,), cached=False)
    assert all(item.offset == 0 and item.keys is None for item in cache)

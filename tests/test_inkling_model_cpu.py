"""Synthetic Inkling CPU contract checks; no checkpoint qualification claim."""

import copy

import mlx.core as mx
import pytest

from mlx2.adapters.inkling_output import InklingOutputParser
from mlx2.runtime.cache_operations import cache_operations
from mlx2.runtime.models.cache import CacheList
from mlx2.runtime.models.inkling import Model, ModelArgs

mx.set_default_device(mx.cpu)


def pairs(cache):
    return [CacheList(cache[i], cache[i + 1]) for i in range(0, len(cache), 2)]


def args():
    return ModelArgs(
        hidden_size=16,
        num_hidden_layers=2,
        vocab_size=32,
        unpadded_vocab_size=30,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        swa_num_attention_heads=2,
        swa_num_key_value_heads=1,
        swa_head_dim=8,
        moe_intermediate_size=8,
        intermediate_size=32,
        dense_intermediate_size=32,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=2,
        d_rel=2,
        rel_extent=8,
        sliding_window_size=4,
        local_layer_ids=[0],
        dense_mlp_idx=1,
    )


def model():
    mx.random.seed(11)
    value = Model(args())
    for layer in value.layers:
        layer.self_attn.rel_proj = (
            mx.random.normal(layer.self_attn.rel_proj.shape) * 0.1
        )
    value.eval()
    mx.eval(value.parameters())
    return value


def test_moe_alias_does_not_use_dense_width():
    a = args()
    assert a.intermediate_size == 8
    assert a.dense_intermediate_size == 32
    m = model()
    assert m.layers[1].mlp.switch_mlp.up_proj.weight.shape == (4, 8, 16)


def test_chunked_and_one_token_decode_match_and_cache_planes_are_distinct():
    m = model()
    whole, single = m.make_cache(), m.make_cache()
    tokens = mx.array([[1, 4, 3, 2, 5, 7]])
    a = m(tokens, cache=whole)
    b = mx.concatenate(
        [m(tokens[:, i : i + 1], cache=single) for i in range(6)], axis=1
    )
    mx.eval(a, b)
    assert bool(mx.allclose(a, b, atol=1e-5).item())
    for left, right in zip(pairs(whole), pairs(single)):
        assert isinstance(left, CacheList)
        assert left[0].offset == right[0].offset == 6
        assert (
            cache_operations(left[1], capability="test").state_revision
            == "mlx2.inkling-four-shortconv.v1"
        )
        for x, y in zip(left[1].cache, right[1].cache):
            assert bool(mx.allclose(x, y, atol=1e-5).item())


def test_rejected_verify_rows_restore_all_four_convolutions():
    m = model()
    cache = m.make_cache()
    m(mx.array([[1, 2]]), cache=cache)
    expected = copy.deepcopy(cache)
    for item in cache:
        item.start_speculation(16)
    m(mx.array([[3, 4, 5, 6]]), cache=cache)
    for item in cache:
        item.trim(2)
        item.stop_speculation()
    m(mx.array([[3, 4]]), cache=expected)
    a, b = m(mx.array([[7]]), cache=cache), m(mx.array([[7]]), cache=expected)
    mx.eval(a, b)
    assert bool(mx.allclose(a, b, atol=1e-5).item())
    for left, right in zip(pairs(cache), pairs(expected)):
        for x, y in zip(left[1].cache, right[1].cache):
            assert bool(mx.allclose(x, y, atol=1e-5).item())


def test_full_reference_ast_branch_commit_matches_selected_ancestry():
    from mlx2.runtime.hybrid_tree_executor import HybridTreeExecutor

    m = model()
    live = m.make_cache()
    m(mx.array([[1, 2]]), cache=live)
    expected = copy.deepcopy(live)
    executor = HybridTreeExecutor.create()
    logits, _, transaction = executor.forward(
        m, [3, 4, 5, 6], [-1, 0, 0, 2], live, [1], cached=False
    )
    transaction.commit_paths([[0, 2, 3]])
    ref = m(mx.array([[3, 5, 6]]), cache=expected)
    mx.eval(logits, ref)
    assert bool(mx.allclose(logits[:, 3], ref[:, -1], atol=1e-5).item())
    a, b = m(mx.array([[7]]), cache=live), m(mx.array([[7]]), cache=expected)
    assert bool(mx.allclose(a, b, atol=1e-5).item())


def test_channel_parser_handles_every_split():
    wire = "<|content_thinking|>Reason<|end_message|><|message_model|><|content_text|>Answer<|end_message|><|content_model_end_sampling|>"
    for split in range(len(wire) + 1):
        p = InklingOutputParser(chat=True)
        events = p.push(wire[:split]) + p.push(wire[split:], final=True)
        assert "".join(e.get("content", "") for e in events) == "Answer"
        assert "".join(e.get("reasoning_content", "") for e in events) == "Reason"
        assert p.stopped


def test_channel_parser_refuses_unsupported_tool_header():
    with pytest.raises(ValueError):
        InklingOutputParser(chat=True).push(
            "web<|content_invoke_tool_json|>{}", final=True
        )


def test_packed_ast_commits_all_four_histories_and_ignores_siblings():
    from mlx2.runtime.inkling_tree_executor import InklingTreeExecutor

    m = model()
    live = m.make_cache()
    m(mx.array([[1, 2, 3, 4, 5]]), cache=live)
    expected = copy.deepcopy(live)
    executor = InklingTreeExecutor.create()
    values, _, tx = executor.forward(
        m, [6, 7, 8, 9], [-1, 0, 0, 2], live, [1], cached=False
    )
    assert live[0].offset == 5
    with pytest.raises(ValueError):
        tx.commit_paths([[0, 1, 3]])
    # Invalid paths never close or mutate the transaction.
    tx.commit_paths([[0, 2, 3]])
    ref = m(mx.array([[6, 8, 9]]), cache=expected)
    a, b = m(mx.array([[7]]), cache=live), m(mx.array([[7]]), cache=expected)
    mx.eval(values, ref, a, b)
    assert bool(mx.allclose(values[:, 3], ref[:, -1], atol=1e-5).item())
    assert bool(mx.allclose(a, b, atol=1e-5).item())


def test_mixed_length_batch_keeps_relative_positions_and_four_histories():
    m = model()
    for layer in m.layers:
        layer.self_attn.rel_proj = (
            mx.random.normal(layer.self_attn.rel_proj.shape) * 0.1
        )
    caches = [m.make_cache(), m.make_cache()]
    m(mx.array([[1, 2]]), cache=caches[0])
    m(mx.array([[1, 3, 2, 5, 4, 6]]), cache=caches[1])
    references = copy.deepcopy(caches)
    merged = [type(caches[0][i]).merge([caches[0][i], caches[1][i]]) for i in range(4)]
    for step in range(5):
        tokens = mx.array([[7 + step], [8 + step]])
        together = m(tokens, cache=merged)
        separate = mx.concatenate(
            [m(tokens[i : i + 1], cache=references[i]) for i in range(2)], axis=0
        )
        assert bool(mx.allclose(together, separate, atol=1e-5).item())


def test_apcv2_warm_prefix_restores_nested_kv_and_all_four_histories(tmp_path):
    from mlx2.runtime.apc_v2 import APCKey, APCv2

    m = model()
    prefix = [1, 2, 3, 4, 5, 6]
    cache = m.make_cache()
    m(mx.array([prefix]), cache=cache)
    for item in cache:
        item.state_checkpoint([len(prefix)], force=True)
    mx.eval([item.state for item in cache])
    key = APCKey(
        "synthetic-inkling",
        revision="cpu-test",
        cache_layout_fingerprint=m.apc_v2_layout,
    )
    apc = APCv2(max_size=4, layout_name=m.apc_v2_layout)
    try:
        stored = apc.store(key, prefix, cache)
        assert stored.stored and stored.exact_prefix
        restored = apc.lookup(key, prefix + [7, 8])
        assert restored.hit and restored.cached_tokens == len(prefix)
        expected = m.make_cache()
        ref = m(mx.array([prefix + [7, 8]]), cache=expected)
        got = m(mx.array([[7, 8]]), cache=restored.cache)
        assert bool(mx.allclose(got[:, -1], ref[:, -1], atol=1e-5).item())
        assert restored.cache[1].apcv2_recurrent_kind == "inkling_shortconv"
        restored.cache.close()
    finally:
        apc.close()


def test_query_mask_tiles_preserve_relative_offset(monkeypatch):
    from mlx2.runtime.models.inkling import language

    m = model()
    tokens = mx.array([[1, 2, 3, 4, 5, 6]])
    expected = m(tokens, cache=m.make_cache())
    monkeypatch.setattr(language, "_MASK_BUDGET_BYTES", 32)
    tiled = m(tokens, cache=m.make_cache())
    assert bool(mx.allclose(expected, tiled, atol=1e-5).item())


def test_persistent_apcv2_restart_preserves_shortconv_revision(tmp_path):
    from mlx2.runtime.apc_v2 import APCKey, APCv2

    m = model()
    prefix = [1, 2, 3, 4, 5, 6]
    cache = m.make_cache()
    m(mx.array([prefix]), cache=cache)
    for item in cache:
        item.state_checkpoint([len(prefix)], force=True)
    key = APCKey(
        "synthetic-inkling",
        revision="cpu-test",
        adapter="inkling",
        tokenizer_fingerprint="synthetic",
        cache_layout_fingerprint=m.apc_v2_layout,
        semantic_fingerprint=("text-token-v1", "tenant", "test"),
    )

    def store():
        return APCv2(
            max_size=4,
            layout_name=m.apc_v2_layout,
            idle_disk_seconds=60,
            idle_disk_dir=str(tmp_path),
            persist_dir=str(tmp_path),
            persist_identity=key,
            persist_semantic_namespace="tenant",
        )

    apc = store()
    assert apc.store(key, prefix, cache, session_tag=("test", "inkling")).stored
    apc.park_session("test", "inkling", ttl_seconds=600)
    apc.close()
    apc = store()
    try:
        restored = apc.lookup(key, prefix + [7, 8], session_tag=("test", "inkling"))
        assert restored.hit
        assert restored.cache[1].apcv2_recurrent_kind == "inkling_shortconv"
        assert (
            cache_operations(restored.cache[1], capability="test").state_revision
            == "mlx2.inkling-four-shortconv.v1"
        )
        ref = m(mx.array([prefix + [7, 8]]), cache=m.make_cache())
        got = m(mx.array([[7, 8]]), cache=restored.cache)
        assert bool(mx.allclose(ref[:, -1], got[:, -1], atol=1e-5).item())
        restored.cache.close()
    finally:
        apc.close()


def test_single_node_packed_ast_uses_chronological_wrapped_local_ring():
    from mlx2.runtime.inkling_tree_executor import InklingTreeExecutor

    m = model()
    cache = m.make_cache()
    # Repeated singleton updates leave a physical ring with a nonzero index.
    for token in range(1, 10):
        m(mx.array([[token]]), cache=cache)
    expected = copy.deepcopy(cache)
    values, _, tx = InklingTreeExecutor.create().forward(
        m, [10], [-1], cache, [1], cached=False
    )
    reference = m(mx.array([[10]]), cache=expected)
    assert bool(mx.allclose(values, reference, atol=1e-5).item())
    tx.commit_paths([[0]])
    got = m(mx.array([[11]]), cache=cache)
    ref = m(mx.array([[11]]), cache=expected)
    assert bool(mx.allclose(got, ref, atol=1e-5).item())


def test_router_padding_never_enters_expert_or_shared_logits():
    m = model().layers[1].mlp
    x = mx.random.normal((3, 16))
    original = m.router_logits(x)
    assert original.shape == (3, 6)
    assert m.gate.weight.shape == (8, 16)
    m.gate.weight[-2:] = mx.full((2, 16), 1000000.0)
    changed = m.router_logits(x)
    assert bool(mx.array_equal(original, changed).item())
    idx, _, gamma = m._route(changed)
    assert int(mx.max(idx).item()) < 4
    assert gamma.shape == (3, 16)


@pytest.mark.parametrize("packed", [False, True])
def test_apcv2_warm_tree_forks_descriptors_and_commits_selected_path(packed):
    from mlx2.runtime.apc_v2 import APCKey, APCv2
    from mlx2.runtime.hybrid_tree_executor import HybridTreeExecutor
    from mlx2.runtime.inkling_tree_executor import InklingTreeExecutor

    m = model()
    prefix = [1, 2, 3, 4, 5, 6]
    cache = m.make_cache()
    m(mx.array([prefix]), cache=cache)
    for item in cache:
        item.state_checkpoint([len(prefix)], force=True)
    key = APCKey(
        "synthetic-inkling",
        revision="cpu-test",
        cache_layout_fingerprint=m.apc_v2_layout,
    )
    apc = APCv2(max_size=4, layout_name=m.apc_v2_layout)
    try:
        assert apc.store(key, prefix, cache).stored
        restored = apc.lookup(key, prefix + [7, 8])
        assert restored.hit
        live = restored.cache
        before = [int(item.offset) for item in live[::2]]
        executor = (InklingTreeExecutor if packed else HybridTreeExecutor).create()
        logits, _, tx = executor.forward(
            m, [7, 8, 9], [-1, 0, 0], live, [1], cached=True
        )
        mx.eval(logits)
        assert [int(item.offset) for item in live[::2]] == before
        tx.abort()
        _, _, tx = executor.forward(m, [7, 8, 9], [-1, 0, 0], live, [1], cached=True)
        tx.commit_paths([[0, 2]])
        assert [int(item.offset) for item in live[::2]] == [n + 2 for n in before]
        actual = m(mx.array([[10]]), cache=live)
        reference = m(mx.array([prefix + [7, 9, 10]]), cache=m.make_cache())
        assert bool(mx.allclose(actual[:, -1], reference[:, -1], atol=1e-5).item())
        # A second borrower of the same stored prefix stays unchanged.
        other = apc.lookup(key, prefix + [11])
        expected = m(mx.array([prefix + [11]]), cache=m.make_cache())
        got = m(mx.array([[11]]), cache=other.cache)
        assert bool(mx.allclose(got[:, -1], expected[:, -1], atol=1e-5).item())
        other.cache.close()
        live.close()
    finally:
        apc.close()


def test_equal_length_merged_batch_keeps_mask_free_conv_and_exact_rollback():
    from mlx2.runtime.models.inkling.language import _cache_padding_mask
    m = model()
    rows = [m.make_cache() for _ in range(2)]
    merged = [type(rows[0][i]).merge([r[i] for r in rows]) for i in range(4)]
    m(mx.array([[1, 2, 3], [4, 5, 6]]), cache=merged)
    assert merged[1].make_mask(1) is None
    assert _cache_padding_mask(merged[1], 1) is None
    for c in merged:
        c.start_speculation(16)
    m(mx.array([[7, 8, 9], [7, 8, 9]]), cache=merged)
    for c in merged:
        c.trim(2)

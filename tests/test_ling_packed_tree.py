"""CPU parent-prefix oracle for packed Ling math, including state commit."""

import copy
import hashlib
import subprocess

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx2.runtime.hybrid_tree_executor import tree_forward
from mlx2.runtime.ling_tree_executor import LingTreeExecutor
from mlx2.runtime.models.bailing_hybrid import Model, ModelArgs


@pytest.fixture(autouse=True)
def bind_synthetic_identity(request, monkeypatch):
    factory = tiny_model
    request.node.user_properties.append((
        "source_revision", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    ))

    def identified(**kwargs):
        model = factory(**kwargs)
        mx.eval(model.parameters())
        digest = hashlib.sha256()
        for name, value in tree_flatten(model.parameters()):
            digest.update((name + str(value.shape) + str(value.dtype)).encode())
            digest.update(np.asarray(value).tobytes())
        request.node.user_properties.append((
            "checkpoint", "synthetic-ling-parent-tree@sha256:" + digest.hexdigest(),
        ))
        request.node.user_properties.append(("device", "cpu"))
        return model

    monkeypatch.setattr(request.module, "tiny_model", identified)


def tiny_model(*, moe=False, lora=False):
    return Model(ModelArgs(
        model_type="bailing_hybrid", vocab_size=48, hidden_size=32,
        intermediate_size=64, moe_intermediate_size=32, num_hidden_layers=3,
        num_attention_heads=2, num_key_value_heads=2, num_experts=4 if moe else None,
        num_experts_per_tok=2, num_shared_experts=1 if moe else 0, n_group=1,
        topk_group=1, first_k_dense_replace=1, layer_group_size=2,
        max_position_embeddings=128, rms_norm_eps=1e-6, rope_theta=10000,
        routed_scaling_factor=1.0, head_dim=16, kv_lora_rank=16,
        qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=16,
        no_kda_lora=not lora, gated_attention_proj_granularity_type="head_wise",
        kda_lower_bound=-5.0,
    ))


def assert_close(left, right):
    np.testing.assert_allclose(np.asarray(left), np.asarray(right), rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("moe,lora", [(False, False), (True, True)])
@pytest.mark.parametrize("prefill", [[], [9, 8, 7]])
@pytest.mark.parametrize("merged_conv", [False, True])
def test_packed_tree_matches_serial_nodes_commit_and_continuation(moe, lora, prefill, merged_conv):
    mx.random.seed(71)
    model = tiny_model(moe=moe, lora=lora)
    model.eval()
    cache = model.make_cache()
    if prefill:
        mx.eval(model(mx.array([prefill]), cache))
    serial_cache, packed_cache = copy.deepcopy(cache), copy.deepcopy(cache)
    tokens, parents = [1, 2, 3, 4, 5], [-1, 0, 0, 2, 1]
    expected, expected_taps, reference = tree_forward(
        model, tokens, parents, serial_cache, [0, 1, 2], "oracle",
    )
    for layer in model.layers:
        if not layer.is_global:
            layer.attention.ling_merged_conv = merged_conv
    provider = LingTreeExecutor(optimizations={"merged_conv": merged_conv})
    actual, actual_taps, transaction = provider.forward(
        model, tokens, parents, packed_cache, [0, 1, 2], cached=False,
    )
    mx.eval(expected, actual, expected_taps, actual_taps)
    assert_close(actual, expected)
    assert_close(actual_taps, expected_taps)
    # No live state publication before commit.
    assert packed_cache[1].offset == len(prefill)
    reference.commit_paths([[0, 2, 3]])
    transaction.commit_paths([[0, 2, 3]])
    for left, right in zip(serial_cache, packed_cache, strict=True):
        if hasattr(left, "offset"):
            assert left.offset == right.offset
            assert_close(left.keys[..., :left.offset, :], right.keys[..., :right.offset, :])
            assert_close(left.values[..., :left.offset, :], right.values[..., :right.offset, :])
        else:
            for a, b in zip(left.cache, right.cache, strict=True):
                assert_close(a, b)
    assert_close(model(mx.array([[6]]), serial_cache), model(mx.array([[6]]), packed_cache))
    assert provider.receipt()["within_request_projection_packing"]
    assert not provider.receipt()["cross_request_physical_packing"]


def test_abort_and_invalid_commit_leave_live_state_unchanged():
    model = tiny_model()
    cache = model.make_cache()
    provider = LingTreeExecutor.create()
    logits, taps, transaction = provider.forward(model, [1, 2, 3], [-1, 0, 0], cache, [0], cached=False)
    mx.eval(logits, taps)
    with pytest.raises(ValueError, match="root path"):
        transaction.commit_paths([[0, 1, 2]])
    transaction.abort()
    assert cache[1].offset == 0
    assert all(value is None for value in cache[0].cache)


@pytest.mark.parametrize("lanes", [2, 4])
def test_request_cohort_preserves_independent_caches(lanes):
    model = tiny_model()
    caches = [model.make_cache() for _ in range(lanes)]
    provider = LingTreeExecutor.create()
    logits, taps, transaction = provider.forward_many(
        model, [[3 * i + 1, 3 * i + 2, 3 * i + 3] for i in range(lanes)],
        [[-1, 0, 0]] * lanes, caches, [0],
        cached=True, configured_limit=4,
    )
    mx.eval(logits, taps)
    assert logits.shape == (lanes, 3, 48)
    transaction.commit_paths([[0, 1 + i % 2] for i in range(lanes)])
    assert [cache[1].offset for cache in caches] == [2] * lanes
    assert not mx.array_equal(caches[0][0].cache[3], caches[1][0].cache[3]).item()


def test_32_rows_match_independent_ordinary_ancestor_prefixes():
    mx.random.seed(93)
    model = tiny_model(moe=True, lora=True)
    model.eval()
    cache = model.make_cache()
    mx.eval(model(mx.array([[9, 8]]), cache))
    tokens = list(range(1, 33))
    parents = [-1, *((i - 1) // 2 for i in range(1, 32))]
    logits, taps, transaction = LingTreeExecutor.create().forward(
        model, tokens, parents, copy.deepcopy(cache), [0, 1, 2], cached=False,
    )
    mx.eval(logits, taps)
    for row in range(32):
        ancestry = []
        node = row
        while node >= 0:
            ancestry.append(node)
            node = parents[node]
        reference = copy.deepcopy(cache)
        for node in reversed(ancestry):
            expected, expected_taps = model.forward_with_taps(
                mx.array([[tokens[node]]]), reference, [0, 1, 2],
            )
        assert_close(logits[:, row:row + 1], expected)
        assert_close(taps[:, row:row + 1], expected_taps)
    transaction.abort()


def test_cohort_commit_prepares_every_lane_before_publication(monkeypatch):
    model = tiny_model()
    caches = [model.make_cache(), model.make_cache()]
    provider = LingTreeExecutor.create()
    _, _, transaction = provider.forward_many(
        model, [[1, 2], [3, 4]], [[-1, 0]] * 2, caches, [0],
        cached=False, configured_limit=4,
    )

    def fail(_path):
        raise RuntimeError("injected commit preparation failure")

    monkeypatch.setattr(transaction.trees[1], "prepare_commit", fail)
    with pytest.raises(RuntimeError, match="injected"):
        transaction.commit_paths([[0, 1], [0, 1]])
    assert [cache[1].offset for cache in caches] == [0, 0]
    assert all(value is None for cache in caches for value in cache[0].cache)


@pytest.mark.parametrize('mask_kind', ['none', 'mask', 'lengths'])
@pytest.mark.parametrize('warm', [False, True])
@pytest.mark.parametrize('rows', [1, 5])
def test_merged_convolution_matches_reference_histories_and_masked_fallback(mask_kind, warm, rows):
    model = tiny_model()
    model.eval()
    module = model.layers[0].attention
    raw = tuple(mx.random.normal((2, rows, module.projection_dim)) for _ in range(3))
    states = tuple(mx.random.normal((2, 3, module.projection_dim)) if warm else None for _ in range(3))
    mask = mx.array([[i < 2 for i in range(rows)], [True]*rows]) if mask_kind == 'mask' else None
    lengths = mx.array([min(2, rows), rows]) if mask_kind == 'lengths' else None
    reference = module.convolve_qkv(raw, states, mask, lengths)
    module.ling_merged_conv = True
    actual = module.convolve_qkv(raw, states, mask, lengths)
    for lefts, rights in zip(reference, actual, strict=True):
        for left, right in zip(lefts, rights, strict=True):
            assert_close(left, right)
    assert module.ling_calls['merged_conv'] == (1 if mask_kind == 'none' and rows == 1 else 0)
    # A changed weight invalidates the cached concatenation.
    if mask_kind == 'none':
        module.q_conv1d.conv.weight = module.q_conv1d.conv.weight * 2
        changed = module.convolve_qkv(raw, states)
        module.ling_merged_conv = False
        expected = module.convolve_qkv(raw, states)
        for left, right in zip(changed[0], expected[0], strict=True):
            assert_close(left, right)


@pytest.mark.parametrize('mixed', ['raw', 'history', 'weight'])
def test_merged_convolution_preserves_independent_mixed_dtype_branches(mixed):
    model = tiny_model()
    module = model.layers[0].attention
    module.set_dtype(mx.bfloat16)
    module.eval()
    raw = [mx.random.normal((1, 1, module.projection_dim)).astype(mx.bfloat16) for _ in range(3)]
    states = [mx.zeros((1, 3, module.projection_dim), dtype=mx.bfloat16) for _ in range(3)]
    if mixed == 'raw':
        raw[1] = raw[1].astype(mx.float32)
    elif mixed == 'history':
        states[1] = states[1].astype(mx.float32)
    else:
        module.k_conv1d.conv.weight = module.k_conv1d.conv.weight.astype(mx.float32)
    expected = module.convolve_qkv(raw, states)
    module.ling_merged_conv = True
    actual = module.convolve_qkv(raw, states)
    for lefts, rights in zip(actual, expected, strict=True):
        for left, right in zip(lefts, rights, strict=True):
            assert left.dtype == right.dtype
            assert bool(mx.array_equal(left, right))
    assert module.ling_calls['merged_conv'] == 0


@pytest.mark.parametrize('policy,helpers', [
    ({}, set()),
    ({'merged_conv': True}, set()),
    ({'fused_norm': True}, {'ling_fused.py', 'qwen4_fused_gdn_prefill.py', 'served_exp.py'}),
    ({'fused_mla_decode': True}, {'kimi_linear_mla_fused.py', 'glm53_sparse_attention.py'}),
])
def test_tree_hashes_only_bound_helpers_and_detects_exact_byte_changes(policy, helpers, tmp_path):
    import os

    provider = LingTreeExecutor(optimizations=policy)
    core = {'ling_tree_executor.py', 'ling_tree.py', 'bailing_hybrid.py', 'hybrid_tree_executor.py'}
    assert {p.name for p in provider._sources} == core | helpers
    copies = []
    for source in provider._sources:
        copied = tmp_path / source.name
        copied.write_bytes(source.read_bytes())
        copies.append(copied)
    provider._sources = tuple(copies)
    provider.validate_revision()
    disabled = tmp_path / 'disabled_helper.py'
    disabled.write_text('unselected helper changes do not alter this execution law')
    provider.validate_revision()
    # Preserve size and mtime to ensure revision admission still hashes bytes.
    changed = copies[-1]
    before = changed.stat()
    raw = changed.read_bytes()
    changed.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    os.utime(changed, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(RuntimeError, match='source changed'):
        provider.validate_revision()


@pytest.mark.parametrize('flag', ['ling_merged_conv', 'ling_fused_norm', 'ling_fused_mla'])
def test_tree_refuses_optimization_drift_before_state_or_projection_work(flag):
    model = tiny_model()
    cache = model.make_cache()
    provider = LingTreeExecutor.create()
    layer = next(layer for layer in model.layers if hasattr(layer.attention, flag))
    setattr(layer.attention, flag, True)
    with pytest.raises(RuntimeError, match='optimization binding changed'):
        provider.forward(model, [1, 2], [-1, 0], cache, [0], cached=False)
    assert provider.dispatches == provider.projection_rows == 0
    assert all(item is None for item in cache[0])


def test_tree_reuses_ancestor_index_array_across_channels_and_layers(monkeypatch):
    from mlx2.runtime.models.ling_tree import LingTree

    model = tiny_model()
    tree = LingTree([-1, 0, 0, 2], model.make_cache())
    made = []
    array = mx.array

    def tracked(value, *args, **kwargs):
        made.append(value)
        return array(value, *args, **kwargs)

    monkeypatch.setattr(mx, 'array', tracked)
    for channels in (3, 5, 3, 5):
        raw = mx.arange(4*channels, dtype=mx.float32).reshape(1, 4, channels)
        windows = tree.convolution_windows(4, raw, None)
        source = np.concatenate([np.zeros((3, channels), np.float32), np.asarray(raw)[0]])
        expected = np.stack([source[[0, 1, 2, 3]], source[[1, 2, 3, 4]],
                             source[[1, 2, 3, 5]], source[[2, 3, 5, 6]]])
        np.testing.assert_array_equal(np.asarray(windows), expected)
    assert len(made) == 1
    assert tree._convolution_indices[4].shape == (4, 4)

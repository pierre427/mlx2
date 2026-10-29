"""One shared HC/GDN/decode tile keeps independent lane checkpoints."""
import copy

import mlx.core as mx
import numpy as np
import pytest
from test_apc_hits_hybrid_gdn_self_mtp import tiny_qwen4_mtp

from mlx2.adapters.flash_next_decode_tile import prepare_gdn_decode_tile


@pytest.mark.parametrize("width", [1, 2, 4])
def test_block_tile_matches_independent_decode_and_commits_only_selected_lanes(width):
    model, _ = tiny_qwen4_mtp()
    model.eval()
    layer = model.language_model.model.layers[0]
    assert layer.is_linear
    caches, inputs, outputs, expected_states = [], [], [], []
    for row in range(width):
        cache = model.make_cache()[0]
        prefix = mx.random.normal((1, 3 + row, layer.attn_hyper_connection.hc_count * layer.attn_hyper_connection.hidden_size))
        layer(prefix, mx.zeros((1, 3 + row), dtype=mx.int32), cache=cache)
        x = mx.random.normal((1, 1, prefix.shape[-1]))
        reference = copy.deepcopy(cache)
        outputs.append(layer(x, mx.zeros((1, 1), dtype=mx.int32), cache=reference))
        inputs.append(x); caches.append(cache); expected_states.append(reference)
    before = [tuple(c.cache) for c in caches]
    out, tx = prepare_gdn_decode_tile(layer, mx.concatenate(inputs, axis=0), caches,
        mx.zeros((width, 1), dtype=mx.int32), revision="fixture@source", cpu_reference=True)
    np.testing.assert_allclose(np.array(out), np.array(mx.concatenate(outputs, axis=0)), atol=5e-5, rtol=2e-5)
    for row, expected in enumerate(expected_states):
        for a, b in zip(tx.successors[row].cache, expected.cache):
            if a is not None:
                np.testing.assert_allclose(np.array(a), np.array(b), atol=5e-5, rtol=2e-5)
        assert all(a is b for a,b in zip(caches[row].cache, before[row]))
    committed = tx.commit([width - 1], revision="fixture@source")
    assert len(committed) == 1 and tx.closed
    assert tx.metrics["ordinary_model_replays"] == 0
    with pytest.raises(ValueError, match="closed"):
        tx.commit([], revision="fixture@source")


def test_decode_tile_rejects_aliased_lane_owners():
    model, _ = tiny_qwen4_mtp(); model.eval()
    layer = model.language_model.model.layers[0]; cache = model.make_cache()[0]
    h = layer.attn_hyper_connection.hidden_size
    with pytest.raises(ValueError, match="distinct"):
        prepare_gdn_decode_tile(layer, mx.zeros((2,1,4*h)), [cache,cache],
            mx.zeros((2,1),dtype=mx.int32), revision="fixture", cpu_reference=True)


@pytest.mark.parametrize("change", ["revision", "owner", "weight"])
def test_decode_tile_rejects_stale_commit_without_mutating_owners(change):
    model, _ = tiny_qwen4_mtp(); model.eval()
    layer = model.language_model.model.layers[0]; cache = model.make_cache()[0]
    h = layer.attn_hyper_connection.hidden_size
    x = mx.zeros((1, 1, layer.attn_hyper_connection.hc_count * h)); tokens = mx.zeros((1, 1), dtype=mx.int32)
    _, tx = prepare_gdn_decode_tile(layer, x, [cache], tokens,
        revision="fixture", cpu_reference=True)
    if change == "owner":
        layer(x, tokens, cache=cache)
    elif change == "weight":
        layer.linear_attn.in_proj_qkv.weight = layer.linear_attn.in_proj_qkv.weight + 0
    before = tuple(cache.cache)
    with pytest.raises(ValueError, match="changed"):
        tx.commit([0], revision="different" if change == "revision" else "fixture")
    assert all(a is b for a, b in zip(cache.cache, before))
    tx.abort()

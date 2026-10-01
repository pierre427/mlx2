"""Explicit per-layer draft context bounds preserve target laws and absolute RoPE."""

import copy
from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from test_lilicorr_serving_cpu import pair
from test_standard_xpress_serving_cpu import drain, generator, reference, tiny
from test_xpress_cpu import tiny_config

from mlx2.runtime.drafters.attention_windows import validate_attention_windows
from mlx2.runtime.drafters.xpress import XPressDraftModel


@pytest.mark.parametrize("value", [[], [1], [1, 0], [1, True], [1, -2], (1, 2), "1,2"])
def test_window_validation_is_explicit(value):
    with pytest.raises(ValueError, match="draft_attention_windows"):
        validate_attention_windows(value, 2)
    assert validate_attention_windows(None, 2) is None
    assert validate_attention_windows([None, 3], 2) == (None, 3)


def test_per_layer_windows_long_chunks_wrap_and_absolute_rope_oracle():
    args = replace(
        tiny_config(), num_hidden_layers=3, layer_types=["full_attention"] * 3
    )
    model = XPressDraftModel(args, draft_attention_windows=[3, 1, None])
    caches = model.make_cache()
    rng = np.random.default_rng(36)
    history = mx.array(rng.normal(size=(1, 18, 8)).astype(np.float32))
    end = 0
    for length in [11, 1, 1, 5]:
        model.append_context(history[:, end : end + length], caches)
        end += length
        assert [cache.offset for cache in caches] == [end, end, end]
        assert caches[0].keys.shape[2] <= 3 and caches[1].keys.shape[2] <= 1
        projected = model.hidden_norm(model.fc(history[:, :end]))
        for layer, entry, window in zip(model.layers, caches, [3, 1, None]):
            attn = layer.self_attn
            keys = attn.k_norm(
                attn.k_proj(projected).reshape(1, end, attn.n_kv_heads, attn.head_dim)
            ).transpose(0, 2, 1, 3)
            expected = model.rope(keys, offset=0)
            if window:
                actual = entry._temporal_order(entry.keys)
                expected = expected[:, :, -window:]
            else:
                actual = entry.state[0][:, :, : entry.offset]
            np.testing.assert_allclose(
                np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-6
            )
    assert model.config.layer_types == ["full_attention"] * 3
    assert model.receipt_settings["draft_attention_windows"] == [3, 1, None]
    assert [
        (layer.self_attn.is_sliding, layer.self_attn.sliding_window)
        for layer in model.layers
    ] == [(True, 4), (True, 2), (False, None)]


def test_single_step_and_long_context_backbone_updates_remain_bounded():
    from types import SimpleNamespace

    from mlx import nn

    model = XPressDraftModel(tiny_config(), draft_attention_windows=[2])
    model.bind(SimpleNamespace(model=SimpleNamespace(embed_tokens=nn.Embedding(9, 4))))
    cache = model.make_cache()
    hidden = mx.ones((1, 12, 8))
    tokens, laws = model.draft_distributions(
        [1], hidden, cache, 1, [object()], [0.0], processor_histories=[[2]]
    )
    assert cache[0].offset == 12 and cache[0].keys.shape[2] == 2
    tokens, laws = model.draft_distributions(
        [3], mx.zeros((1, 0, 8)), cache, 3, [object()], [1.0], processor_histories=[[1]]
    )
    assert cache[0].offset == 12 and cache[0].keys.shape[2] == 2
    assert len(tokens[0]) == 3 and all(q[t] == 1 for q, t in zip(laws[0], tokens[0]))
    model.append_context(mx.ones((1, 1, 8)), cache)
    assert cache[0].offset == 13 and cache[0].keys.shape[2] == 2


def model_pair(kind):
    target, draft = tiny() if kind == "xpress" else pair()
    draft = type(draft)(draft.config, draft_attention_windows=[2]).bind(target)
    return target, draft


@pytest.mark.parametrize("kind", ["xpress", "lilicorr"])
def test_window_route_target_greedy_apcv2_resume_and_revision_binding(
    kind, monkeypatch
):
    from mlx2.runtime.apc_v2 import APCKey, APCv2

    monkeypatch.setattr(mx, "clear_cache", lambda: None)
    model, draft = model_pair(kind)
    scheduler = generator(model, draft)
    prompt = [1, 2, 3, 4, 5, 6, 7]
    scheduler.insert([prompt], max_tokens=[6])
    outputs, finished = drain(scheduler)
    assert outputs[0] == reference(model, prompt, 6)
    end = finished[0]
    assert all(cache.keys.shape[2] <= 2 for cache in end.cache_sidecar.state[0])
    assert all(cache.offset == len(end.all_tokens) for cache in end.prompt_cache)
    assert end.speculative_receipt["draft_settings"]["draft_attention_windows"] == [2]
    key = APCKey(
        "bounded-draft",
        revision=kind + "-window2",
        cache_layout_fingerprint="tiny-window2",
    )
    apc = APCv2(max_size=2, layout_name="tiny-window2")
    apc.store(key, end.all_tokens, end.prompt_cache, sidecar=end.cache_sidecar)
    hit = apc.lookup(key, end.all_tokens + [end.token])
    assert hit.hit
    resumed = generator(model, draft)
    resumed.insert(
        [[end.token]],
        max_tokens=[4],
        caches=[hit.cache],
        all_tokens=[end.all_tokens],
        cache_states=[hit.sidecar],
    )
    actual, final = drain(resumed)
    assert actual[0] == reference(model, end.all_tokens + [end.token], 4)
    final[0].cache_sidecar.validate("tiny-xpress-v1", len(final[0].all_tokens))
    assert all(cache.keys.shape[2] <= 2 for cache in final[0].cache_sidecar.state[0])
    assert not apc.lookup(
        APCKey(
            "bounded-draft",
            revision=kind + "-window3",
            cache_layout_fingerprint="tiny-window2",
        ),
        end.all_tokens + [end.token],
    ).hit
    hit.cache.close()
    apc.clear(release_memory=False)


@pytest.mark.parametrize("kind", ["xpress", "lilicorr"])
def test_window_sampled_target_exact_pointmass_law(kind, monkeypatch):
    import mlx2.runtime.external_speculative as external
    from mlx2.runtime.sample_utils import LaneRNG
    from mlx2.runtime.speculative_sampling import RequestRNG, softmax, verify_proposals

    model, draft = model_pair(kind)
    scheduler = generator(model, draft)
    prompt = [1, 2, 3, 4, 5, 6, 7]
    expected = softmax(
        np.asarray(model(mx.array([prompt]), cache=model.make_cache())[0, -1]), 0.8
    )
    captured = []

    def checked(tokens, proposals, targets, rng, **kw):
        if tokens:
            assert all(
                q[token] == 1 and np.count_nonzero(q) == 1
                for token, q in zip(tokens, proposals)
            )
            captured.append(
                (list(tokens), copy.deepcopy(proposals), copy.deepcopy(targets))
            )
        return verify_proposals(tokens, proposals, targets, rng, **kw)

    monkeypatch.setattr(external, "verify_proposals", checked)
    scheduler.insert(
        [prompt],
        max_tokens=[6],
        lane_rngs=[LaneRNG(22)],
        sampling_configs=[{"sampling_temp": 0.8}],
    )
    output, _ = drain(scheduler)
    assert len(output[0]) == 6 and captured
    tokens, q, p = captured[0]
    np.testing.assert_allclose(p[0], expected, rtol=1e-6, atol=1e-7)
    counts = np.zeros(9)
    rng = RequestRNG(93)
    for _ in range(10000):
        counts[verify_proposals(tokens, q, p, rng).emitted[0]] += 1
    np.testing.assert_allclose(counts / counts.sum(), expected, atol=0.018)


@pytest.mark.parametrize("kind", ["xpress", "lilicorr"])
def test_window_cache_rollback_after_target_failure_is_exact(kind, monkeypatch):
    model, draft = model_pair(kind)
    scheduler = generator(model, draft)
    scheduler.insert([[1, 2, 3, 4, 5, 6, 7]], max_tokens=[5])
    lane = scheduler.lanes[0]
    while lane.remaining:
        scheduler._prefill(lane)
    before = copy.deepcopy(lane.__dict__)
    original = model.forward_with_taps

    def fail(*a, **kw):
        original(*a, **kw)
        raise RuntimeError("target failed after append")

    monkeypatch.setattr(model, "forward_with_taps", fail)
    with pytest.raises(RuntimeError, match="failed after append"):
        scheduler._round([lane])
    for now, old in zip(lane.draft_cache, before["draft_cache"]):
        assert now.offset == old.offset and now._idx == old._idx
        for actual, expected in zip(now.state, old.state):
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    np.testing.assert_array_equal(np.asarray(lane.tail), np.asarray(before["tail"]))
    assert lane.rng.snapshot() == before["rng"].snapshot()
    monkeypatch.setattr(model, "forward_with_taps", original)
    outputs, _ = drain(scheduler)
    assert outputs[0] == reference(model, [1, 2, 3, 4, 5, 6, 7], 5)

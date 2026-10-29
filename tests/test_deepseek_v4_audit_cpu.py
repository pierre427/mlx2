"""Synthetic CPU regressions for the 64K audit; no model qualification."""

import dataclasses

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from test_deepseek_v4_cache_cpu import advance, layer
from test_deepseek_v4_forward_cpu import SmallWeights, config

from mlx2.runtime.models.deepseek_v4 import DeepSeekV4Model
from mlx2.runtime.models.deepseek_v4_cache import DeepSeekV4LayerCache
from mlx2.runtime.models.deepseek_v4_packed_cache import DeepSeekV4PackedLayerCache


def test_packed_cache_refuses_incompatible_rotary_tail_before_weight_load():
    c = dataclasses.replace(
        config((4,)), head_dim=512, qk_rope_head_dim=32, simulate_quantization=True
    )
    with pytest.raises(ValueError, match="rotary64"):
        DeepSeekV4Model(c, None, cache_policy="fp8_mixed")


def test_visible_attention_refuses_incompatible_head_before_weight_load():
    with pytest.raises(ValueError, match="D512"):
        DeepSeekV4Model(config((4,)), None, attention_policy="visible_rows_fp32")


@pytest.mark.parametrize("count,k", [(8, 3), (16384, 512)])
def test_indexer_cutoff_ties_are_chronological_at_64k(count, k):
    from mlx2.runtime.models.deepseek_v4 import stable_pool_topk

    scores = np.zeros((1, 2, count), np.float32)
    scores[0, 0, -1] = 1
    scores[0, 1, k:] = -np.inf  # Future completed pools must stay masked.
    actual = np.array(stable_pool_topk(mx.array(scores), k))
    expected = [list(range(k - 1)) + [count - 1], list(range(k))]
    np.testing.assert_array_equal(actual[0], expected)


def test_indexer_exact_order_does_not_depend_on_partition_tie_order(monkeypatch):
    from mlx2.runtime.models.deepseek_v4 import stable_pool_topk

    original = mx.argpartition

    def alternate(a, kth, axis):
        # A valid partition with the opposite arbitrary ordering within ties.
        order = mx.argsort(a[..., ::-1], axis=axis)
        return a.shape[-1] - 1 - order

    monkeypatch.setattr(mx, "argpartition", alternate)
    scores = mx.array([[[0.0, 0.0, 2.0, 1.0, 0.0, 0.0]]])
    np.testing.assert_array_equal(
        np.array(stable_pool_topk(scores, 4)), [[[0, 1, 2, 3]]]
    )
    monkeypatch.setattr(mx, "argpartition", original)


@pytest.mark.parametrize("window,offset", [(2**31, 1), (4, 2**31 - 1)])
def test_visible_attention_refuses_int32_position_overflow(window, offset):
    from mlx2.runtime.models.deepseek_v4_visible_attention import validate_geometry

    with pytest.raises(ValueError, match="int32"):
        validate_geometry(
            np.zeros((1, 1, 1, 512)),
            np.zeros((1, 1, 1, 512)),
            None,
            None,
            np.zeros((1,)),
            offset,
            window,
            0,
        )


def test_disk_restore_refuses_incomplete_pool_frontier():
    cache = layer()
    advance(cache, 133)
    metadata = list(cache.meta_state)
    metadata[7] = "132"  # Same window count, different pending compressor frontier.
    arrays = list(cache.state)
    pool = list(arrays[1])
    pool[:2] = [x[:, :0] for x in pool[:2]]
    arrays[1] = tuple(pool)
    with pytest.raises(ValueError, match="frontier|incomplete"):
        type(cache).from_state(tuple(arrays), tuple(metadata))


def test_disk_restore_refuses_missing_completed_pool_rows():
    cache = layer()
    advance(cache, 133)
    arrays = list(cache.state)
    pool = list(arrays[1])
    pool[-1] = pool[-1][:, :-1]
    arrays[1] = tuple(pool)
    with pytest.raises(ValueError, match="count|frontier|incomplete"):
        type(cache).from_state(tuple(arrays), cache.meta_state)


@pytest.mark.parametrize("plane", ["local", "carry", "pool"])
def test_disk_restore_refuses_changed_physical_dtype(plane):
    cache = layer()
    advance(cache, 133)
    arrays = list(cache.state)
    if plane == "local":
        arrays[0] = arrays[0].astype(np.float16)
    else:
        pool = list(arrays[1])
        index = 0 if plane == "carry" else 4
        pool[index] = pool[index].astype(np.float16)
        arrays[1] = tuple(pool)
    with pytest.raises(ValueError, match="FP32|dtype"):
        type(cache).from_state(tuple(arrays), cache.meta_state)


def test_invalid_metadata_assignment_does_not_mutate_live_frontier():
    cache = layer()
    advance(cache, 133)
    before = cache.meta_state
    invalid = list(before)
    invalid[6] = "134"
    invalid[8] = "none"
    with pytest.raises(ValueError, match="topology"):
        cache.meta_state = invalid
    assert cache.meta_state == before


def test_closed_cache_cannot_be_repopulated_by_restore():
    cache = layer()
    advance(cache, 133)
    saved = cache.snapshot()
    cache.close()
    with pytest.raises(ValueError, match="closed"):
        cache.restore(saved)
    assert cache.nbytes == 0


def test_forward_rejects_aliased_layer_caches_before_mutation():
    c = config((0, 0))
    model = DeepSeekV4Model(c, SmallWeights(c), identity="audit@synthetic")
    cache = model.make_cache()[0]
    with pytest.raises(ValueError, match="independent|alias"):
        model(mx.array([[1]], dtype=mx.int32), [cache, cache])
    assert cache.offset == 0


@pytest.mark.parametrize("selected", ["fp32", "fp8_mixed"])
def test_forward_rejects_other_physical_layout_before_mutation(selected):
    c = dataclasses.replace(
        config((0,)),
        head_dim=512,
        qk_rope_head_dim=64,
        index_head_dim=128,
        simulate_quantization=True,
    )
    model = DeepSeekV4Model(
        c, SmallWeights(c), identity="audit@synthetic", cache_policy=selected
    )
    wrong = DeepSeekV4PackedLayerCache if selected == "fp32" else DeepSeekV4LayerCache
    cache = wrong(
        0,
        xp=mx,
        identity=model.identity,
        head_dim=512,
        index_head_dim=128,
        window_size=c.sliding_window,
    )
    with pytest.raises(ValueError, match="layout|identity"):
        model(mx.array([[1]], dtype=mx.int32), [cache])
    assert cache.offset == 0

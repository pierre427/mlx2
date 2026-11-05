"""Window-blocked sliding prefill: same keys per row as stock SDPA."""
import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mlx2.runtime.models.base import create_attention_mask
from mlx2.runtime.models.cache import RotatingKVCache
from mlx2.runtime.models.ricursive import window_blocked_attention


def rand(*shape, seed):
    return mx.array(np.random.default_rng(seed).normal(size=shape).astype(np.float32))


@pytest.mark.parametrize("prefix", [0, 5, 40])
@pytest.mark.parametrize("length", [9, 23])
def test_blocked_matches_stock_after_real_rotating_updates(prefix, length):
    window = 6
    cache = RotatingKVCache(window, keep=0)
    if prefix:
        cache.update_and_fetch(rand(1, 2, prefix, 8, seed=1), rand(1, 2, prefix, 8, seed=2))
    q = rand(1, 4, length, 8, seed=3)
    mask = create_attention_mask(mx.zeros((1, length, 8)), cache, window_size=window, return_array=True)
    k, v = cache.update_and_fetch(rand(1, 2, length, 8, seed=4), rand(1, 2, length, 8, seed=5))
    expected = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.35, mask=mask)
    actual = window_blocked_attention(q, k, v, scale=0.35, mask=mask, window=window, block=4)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=2e-5, atol=2e-5)


def test_refuses_string_masks_and_short_chunks():
    q, k = rand(1, 4, 3, 8, seed=1), rand(1, 2, 9, 8, seed=2)
    assert window_blocked_attention(q, k, k, scale=0.3, mask="causal", window=6, block=4) is None
    assert window_blocked_attention(q, k, k, scale=0.3, mask=mx.ones((3, 9), dtype=mx.bool_), window=6, block=4) is None


def test_model_greedy_parity_with_blocked_prefill():
    from test_ripple_pld_cpu import greedy, tiny

    model = tiny()
    prompt = [(7 * i) % 60 + 1 for i in range(300)]
    reference = greedy(model, prompt, 8)
    model.set_window_blocked_prefill(True)
    assert greedy(model, prompt, 8) == reference
    assert model.window_blocked_calls() > 0

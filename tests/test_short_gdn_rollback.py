"""A short GDN verify after a wider one must use its own rollback record.

The peer failure (llama.cpp#29454) restored a slot left unwritten by the
shorter verify. This exercises mlx2's actual recurrent cache on CPU, including
different accepted lengths in a batch and a cold ordinary-forward oracle.
"""

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models import qwen3_5
from mlx2.runtime.models.cache import ArraysCache


def _layer():
    args = qwen3_5.TextModelArgs(
        hidden_size=32,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
    )
    layer = qwen3_5.GatedDeltaNet(args)
    mx.eval(layer.parameters())
    return layer


def _assert_cache_equal(actual, expected):
    assert len(actual.cache) == len(expected.cache)
    for got, want in zip(actual.cache, expected.cache):
        assert got.shape == want.shape and got.dtype == want.dtype
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("array_accept", (False, True))
def test_short_verify_after_wide_verify_matches_cold_replay_per_row(
    monkeypatch, array_accept
):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        monkeypatch.setattr(qwen3_5, "_GDN_ARRAY_ACCEPT", array_accept)
        mx.random.seed(29454)
        layer = _layer()
        prefix = mx.random.normal((2, 4, 32))
        wide = mx.random.normal((2, 5, 32))
        short = mx.random.normal((2, 2, 32))
        continuation = mx.random.normal((2, 1, 32))

        cache = ArraysCache(2)
        layer(prefix, cache=cache)
        cache.start_speculation()
        layer(wide, cache=cache)
        assert cache._rollbacks[-1].span == 5
        assert (cache._rollbacks[-1].per_row_fn is not None) == array_accept
        cache.trim_ragged([3, 1])  # Retain 2 and 4 tokens from the wide verify.
        layer(short, cache=cache)
        assert cache._rollbacks[-1].span == 2
        assert (cache._rollbacks[-1].per_row_fn is not None) == array_accept
        cache.trim_ragged([1, 2])  # Retain 1 and 0 from the short verify.

        for row, (wide_keep, short_keep) in enumerate(((2, 1), (4, 0))):
            cold = ArraysCache(2)
            replay = mx.concatenate(
                [prefix[row : row + 1], wide[row : row + 1, :wide_keep],
                 short[row : row + 1, :short_keep]], axis=1
            )
            layer(replay, cache=cold)
            actual = cache.extract(row)
            _assert_cache_equal(actual, cold)
            got = layer(continuation[row : row + 1], cache=actual)
            want = layer(continuation[row : row + 1], cache=cold)
            np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=2e-5, atol=2e-6)
    finally:
        mx.set_default_device(previous)

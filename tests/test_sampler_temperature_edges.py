"""Temperature edge cases of the MLX sampler (CPU)."""

import mlx.core as mx
import pytest

from mlx2.runtime.sample_utils import make_sampler

mx.set_default_device(mx.cpu)


def _normalized(row):
    return row - mx.logsumexp(row, axis=-1, keepdims=True)


@pytest.mark.parametrize("temp", [1e-6, 1e-39, 1e-300])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16, mx.float16])
def test_make_sampler_vanishing_temperature_is_greedy(temp, dtype):
    # 1 / 1e-39 overflows; the sampler used to return token 0 here.
    row = mx.log(mx.array([[0.1, 0.449, 0.451]] * 64)).astype(dtype)
    assert set(make_sampler(temp=temp)(row).tolist()) == {2}


@pytest.mark.parametrize("temp", [1e-5, 1e-4])
def test_float16_small_temperature_does_not_collapse_to_token_zero(temp):
    flat = _normalized(mx.zeros((1, 151936))).astype(mx.float16)
    mx.random.seed(0)
    draws = make_sampler(temp=temp)(mx.repeat(flat, 16, axis=0)).tolist()
    assert len(set(draws)) > 1
    two = mx.log(mx.array([[0.5, 0.5, 0.0, 0.0]])).astype(mx.float16)
    mx.random.seed(0)
    assert set(make_sampler(temp=1e-5)(mx.repeat(two, 64, axis=0)).tolist()) == {0, 1}

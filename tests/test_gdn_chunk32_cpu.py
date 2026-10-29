"""Independent token recurrence, reset, tail and incoming-state gates."""
import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models.gdn_chunk32 import chunk32, refusal


@pytest.mark.parametrize("length", [1, 17, 31, 32, 33, 65])
@pytest.mark.parametrize("heads", [(1, 8), (16, 48)])
def test_chunk32_matches_fp64_token_oracle(length, heads):
    rng = np.random.default_rng(length)
    hk, hv = heads
    q = rng.normal(size=(1, length, hk, 128)).astype(np.float32) / 12
    k = rng.normal(size=q.shape).astype(np.float32) / 12
    v = rng.normal(size=(1, length, hv, 128)).astype(np.float32)
    g = rng.uniform(.5, 1, size=(1, length, hv)).astype(np.float32)
    g[:, length // 2, 0] = 0
    beta = rng.uniform(0, .8, size=g.shape).astype(np.float32)
    state = rng.normal(size=(1, hv, 128, 128)).astype(np.float32) / 10
    current = state.astype(np.float64)
    expected = []
    for t in range(length):
        qt, kt = (np.repeat(x[:, t], hv // hk, axis=1).astype(np.float64)
                  for x in (q, k))
        current *= g[:, t, :, None, None]
        delta = (v[:, t] - (current * kt[:, :, None, :]).sum(-1)) * beta[:, t, :, None]
        current += delta[..., None] * kt[:, :, None, :]
        expected.append((current * qt[:, :, None, :]).sum(-1))
    y, final = chunk32(*(mx.array(x) for x in (q, k, v, g, beta, state)))
    np.testing.assert_allclose(np.array(y), np.stack(expected, axis=1), atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(np.array(final), current, atol=2e-6, rtol=2e-5)


def test_state_storage_and_mask_are_not_silently_changed():
    q = mx.zeros((1, 1, 1, 128)); v = mx.zeros((1, 1, 8, 128))
    g = mx.ones((1, 1, 8)); s = mx.zeros((1, 8, 128, 128))
    assert refusal(q, q, v, g, g, s.astype(mx.float16)) == "fp32_state_required"
    assert refusal(q, q, v, g, g, s, mx.ones((1, 1))) == "masked"


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_half_activations_retain_fp32_state(dtype):
    mx.random.seed(74)
    q = (mx.random.normal((1, 33, 1, 128)) / 12).astype(dtype)
    k = (mx.random.normal(q.shape) / 12).astype(dtype)
    v = mx.random.normal((1, 33, 8, 128)).astype(dtype)
    g = mx.full((1, 33, 8), .8, dtype=mx.float32)
    beta = mx.full(g.shape, .4)
    state = mx.zeros((1, 8, 128, 128), dtype=mx.float32)
    output, final = chunk32(q, k, v, g, beta, state)
    control, carry = chunk32(q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32), g, beta, state)
    assert output.dtype == dtype and final.dtype == mx.float32
    np.testing.assert_array_equal(np.array(output.view(mx.uint16)), np.array(control.astype(dtype).view(mx.uint16)))
    np.testing.assert_allclose(np.array(final), np.array(carry), atol=1e-6)

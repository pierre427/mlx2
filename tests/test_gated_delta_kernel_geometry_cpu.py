"""CPU evidence for fixed-warp Metal admission; no native kernel runs."""

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models import gated_delta as gdn


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def inputs(dk, vector=False):
    rng = np.random.default_rng(49 + dk)
    q = rng.normal(size=(2, 3, 2, dk)).astype(np.float32) * 0.1
    k = rng.normal(size=q.shape).astype(np.float32) * 0.1
    v = rng.normal(size=(2, 3, 4, 5)).astype(np.float32) * 0.1
    g = rng.uniform(0.7, 0.95, size=(2, 3, 4, dk) if vector else (2, 3, 4)).astype(
        np.float32
    )
    beta = rng.uniform(0.1, 0.6, size=(2, 3, 4)).astype(np.float32)
    state = rng.normal(size=(2, 4, 5, dk)).astype(np.float32) * 0.1
    mask = np.array([[False, True, True], [True, False, True]])
    return [mx.array(a) for a in (q, k, v, g, beta, state, mask)]


def oracle(q, k, v, g, beta, state, mask):
    q, k, v, g, beta, state, mask = [
        np.asarray(a) for a in (q, k, v, g, beta, state, mask)
    ]
    state = state.copy()
    out = []
    for t in range(q.shape[1]):
        key = np.repeat(k[:, t], 2, axis=1)
        query = np.repeat(q[:, t], 2, axis=1)
        gate = g[:, t]
        decayed = state * (
            gate[..., None, :] if gate.ndim == 3 else gate[..., None, None]
        )
        delta = (v[:, t] - (decayed * key[..., None, :]).sum(axis=-1)) * beta[
            :, t, ..., None
        ]
        updated = decayed + key[..., None, :] * delta[..., None]
        readout = (updated * query[..., None, :]).sum(axis=-1)
        state = np.where(mask[:, t, None, None, None], updated, state)
        out.append(np.where(mask[:, t, None, None], readout, 0))
    return np.stack(out, axis=1), state


@pytest.mark.parametrize("dk", [8, 16, 31, 33, 40, 63])
@pytest.mark.parametrize("vector", [False, True])
def test_incomplete_simd_strips_skip_native_kernel_and_keep_masked_recurrence(
    monkeypatch, dk, vector
):
    def forbidden(**kwargs):
        raise AssertionError("unsupported fixed-warp native dispatch")

    for name in (
        "_gated_delta_kernel",
        "_gated_delta_kernel_masked",
        "_gated_delta_kernel_vec",
        "_gated_delta_kernel_vec_masked",
        "_gated_delta_kernel_packed",
    ):
        monkeypatch.setattr(gdn, name, forbidden)
    args = inputs(dk, vector)
    actual = gdn.gated_delta_kernel(*args)
    mx.eval(actual)
    expected = oracle(*args)
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(np.asarray(got), want, atol=2e-7, rtol=2e-6)


@pytest.mark.parametrize("dk", [32, 64, 96, 128])
def test_complete_simd_strips_preserve_generic_kernel_dispatch(monkeypatch, dk):
    calls = []

    def kernel(**kwargs):
        calls.append(kwargs)
        return [
            mx.full(shape, 0.25, dtype=dtype)
            for shape, dtype in zip(kwargs["output_shapes"], kwargs["output_dtypes"])
        ]

    monkeypatch.setattr(gdn, "_gated_delta_kernel_masked", kernel)
    args = inputs(dk)
    result = gdn.gated_delta_kernel(*args)
    assert len(calls) == 1 and dict(calls[0]["template"])["Dk"] == dk
    assert calls[0]["grid"] == (32, 5, 8)
    assert all(np.all(np.asarray(a) == 0.25) for a in result)


def test_small_keys_reached_from_update_dispatch_preserve_exact_state(monkeypatch):
    # Patch only the admission report. The underlying default device stays CPU.
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    q, k, v, _, _, state, mask = inputs(8)
    a = mx.full((2, 3, 4), -0.4)
    b = mx.full((2, 3, 4), 0.3)
    A = mx.zeros((4,))
    bias = mx.zeros((4,))

    def forbidden(**kwargs):
        raise AssertionError("small key native kernel")

    monkeypatch.setattr(gdn, "_gated_delta_kernel_masked", forbidden)
    actual = gdn.gated_delta_update(q, k, v, a, b, A, bias, state, mask)
    expected = gdn.gated_delta_update(
        q, k, v, a, b, A, bias, state, mask, use_kernel=False
    )
    mx.eval(actual, expected)
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))

"""Real-Metal gate: Source mHC ``mega_v1`` equals the served composition bit for bit.

Run under the GPU lock with ``MLX2_RUN_METAL_TESTS=1``. Synthetic tensors at
production DS4.1 geometry (HC4 x 5120, fn 24 x 20480); no checkpoint load.
The reference is the ordinary Source composition with the selected
``source_fused_fp32`` coefficient kernel.
"""
import os

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models import deepseek_v41_source_hc_mega as mega

pytestmark = pytest.mark.skipif(
    os.environ.get('MLX2_RUN_METAL_TESTS') != '1' or not mx.metal.is_available(),
    reason='real-Metal gate; set MLX2_RUN_METAL_TESTS=1 under the GPU lock')

OPTIONS = dict(iterations=20, hc_eps=1e-6, rms_eps=1e-20)


@pytest.fixture(autouse=True)
def gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def bits(value):
    return np.array(value).view(np.uint32)


def inputs(seed, magnitude, weight_dtype):
    rng = np.random.default_rng(seed)
    h = mx.array((rng.normal(size=(1, 1, 4, 5120)) * magnitude).astype(np.float32))
    h = h.astype(mx.bfloat16).astype(mx.float32)
    hc = (mx.array((rng.normal(size=(24, 20480)) * 0.01).astype(np.float32)),
          mx.array(rng.uniform(0.05, 3.0, size=3).astype(np.float32)),
          mx.array((rng.normal(size=24) * 2).astype(np.float32)))
    prev = rng.uniform(0.0, 1.5, size=(1, 1, 4)).astype(np.float32)
    if seed % 3 == 0:
        prev[0, 0, seed % 4] = 0.0
    weight = mx.array(rng.normal(size=5120).astype(np.float32)).astype(weight_dtype)
    branch = mx.array((rng.normal(size=(1, 1, 5120)) * magnitude).astype(np.float32))
    shared = mx.array((rng.normal(size=(1, 1, 5120)) * magnitude).astype(np.float32))
    return h, mx.array(prev), hc, weight, branch, shared


@pytest.mark.parametrize('seed', range(24))
def test_pre_and_store_bitwise_equal_served_composition(seed):
    magnitude = (1e-3, 0.05, 1.0, 7.0, 40.0, 300.0)[seed % 6]
    h, prev, hc, weight, branch, shared = inputs(seed, magnitude, mx.bfloat16 if seed % 2 else mx.float32)
    actual = mega.pre(h, prev, hc, weight, **OPTIONS)
    expected = mega.reference_pre(h, prev, hc, weight, **OPTIONS)
    pairs = list(zip(actual, expected))
    for extra in (None, shared):
        pairs.append((mega.store(branch, h, expected[1], expected[2], extra),
                      mega.reference_store(branch, h, expected[1], expected[2], extra)))
    mx.eval(pairs)
    for index, (a, b) in enumerate(pairs):
        assert a.shape == b.shape, index
        np.testing.assert_array_equal(bits(a), bits(b), err_msg=f'output {index}')


def test_first_use_gate_and_chained_layers():
    assert mega.self_check(**OPTIONS)
    h, prev, hc, weight, branch, shared = inputs(99, 1.0, mx.bfloat16)
    ref_h, ref_pre, new_h, new_pre = h, prev, h, prev
    for _ in range(6):
        p, q, c, x = mega.reference_pre(ref_h, ref_pre, hc, weight, **OPTIONS)
        ref_h, ref_pre = mega.reference_store(x * 0.5, ref_h, q, c, x * 0.25), p
        p, q, c, x = mega.pre(new_h, new_pre, hc, weight, **OPTIONS)
        new_h, new_pre = mega.store(x * 0.5, new_h, q, c, x * 0.25), p
    mx.eval(ref_h, new_h, ref_pre, new_pre)
    np.testing.assert_array_equal(bits(ref_h), bits(new_h))
    np.testing.assert_array_equal(bits(ref_pre), bits(new_pre))

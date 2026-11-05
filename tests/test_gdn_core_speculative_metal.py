"""Opt-in real Metal recurrence gate for GDN core/rollback composition.

Synthetic seeded production GDN geometry, no model checkpoint loaded. Run only
inside a validated GPU ownership window with MLX2_RUN_METAL_TESTS=1.
"""
import os

import mlx.core as mx
import pytest

from mlx2.runtime.models import gated_delta as gd

pytestmark = pytest.mark.skipif(
    os.environ.get("MLX2_RUN_METAL_TESTS") != "1" or not mx.metal.is_available(),
    reason="requires explicit Metal ownership and MLX2_RUN_METAL_TESTS=1",
)


@pytest.fixture(autouse=True)
def gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


@pytest.mark.parametrize("qwen4_law", [False, True])
def test_selected_core_keeps_speculative_prefixes_on_ordinary_recurrence(qwen4_law, monkeypatch):
    if gd._core_gated_delta_update is None:
        pytest.skip("MLX core scan is unavailable")
    mx.random.seed(11303)
    batch, steps, key_heads, value_heads, dim = 1, 20, 16, 48, 128
    q, k = [mx.random.normal((batch, steps, key_heads, dim)).astype(mx.bfloat16)
            for _ in range(2)]
    q, k = gd.normalize_gdn_qk(q, k)
    v = (mx.random.normal((batch, steps, value_heads, dim)) * 0.1).astype(mx.bfloat16)
    a, beta = [mx.random.normal((batch, steps, value_heads)).astype(mx.bfloat16)
               for _ in range(2)]
    a_log = mx.zeros((value_heads,), mx.float32)
    dt_bias = mx.zeros((value_heads,), mx.bfloat16)
    initial = mx.random.normal((batch, value_heads, dim, dim)) * 0.02
    law = dict(beta_input_dtype=qwen4_law, decay_input_float32=qwen4_law)
    monkeypatch.setattr(gd, "_ENABLE_GDN_CORE", True)
    before = gd.gdn_core_status()["calls"]
    ordinary = gd.gated_delta_update(q, k, v, a, beta, a_log, dt_bias, initial, **law)
    mx.eval(ordinary)
    assert gd.gdn_core_status()["calls"] == before + 1

    serial = initial
    checkpoints = {}
    for position in range(steps):
        _, serial = gd.gated_delta_update(
            *(value[:, position:position + 1] for value in (q, k, v, a, beta)),
            a_log, dt_bias, serial, allow_core=False, **law,
        )
        if position + 1 in (8, 18, 20):
            checkpoints[position + 1] = serial
    for accepted in (20, 18, 8):
        _, restored = gd.gated_delta_update(
            *(value[:, :accepted] for value in (q, k, v, a, beta)),
            a_log, dt_bias, initial, allow_core=False, **law,
        )
        mx.eval(restored, checkpoints[accepted])
        assert bool(mx.array_equal(restored, checkpoints[accepted])), accepted
    assert gd.gdn_core_status()["calls"] == before + 1

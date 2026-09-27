"""27B-style GDN rollback oracle for the later #3958 fused-verify gate.

This exercises mlx2's ordinary reference and local ArraysCache only. It does
not import an upstream kernel or qualify any fused candidate.
"""

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models import qwen3_5
from mlx2.runtime.models.cache import ArraysCache
from mlx2.runtime.models.gated_delta import gated_delta_update
from mlx2.runtime.models.qwen3_next import Qwen3NextRMSNormGated


@pytest.mark.parametrize("steps", (3, 5))
def test_ordinary_gdn_every_accepted_prefix_and_next_step_on_cpu(steps):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        mx.random.seed(3958)
        args = qwen3_5.TextModelArgs(
            hidden_size=32,
            linear_num_value_heads=4,
            linear_num_key_heads=2,
            linear_key_head_dim=16,
            linear_value_head_dim=8,
            linear_conv_kernel_dim=4,
        )
        layer = qwen3_5.GatedDeltaNet(args)
        layer.set_dtype(mx.bfloat16)
        mx.eval(layer.parameters())
        prefix = mx.random.normal((1, 4, 32)).astype(mx.bfloat16)
        verify = mx.random.normal((1, steps, 32)).astype(mx.bfloat16)
        next_row = mx.random.normal((1, 1, 32)).astype(mx.bfloat16)

        for accepted in range(steps + 1):
            speculative = ArraysCache(2)
            layer(prefix, cache=speculative)
            speculative.start_speculation()
            layer(verify, cache=speculative)
            if accepted < steps:
                assert speculative.trim(steps - accepted) == steps - accepted

            ordinary = ArraysCache(2)
            layer(prefix, cache=ordinary)
            if accepted:
                layer(verify[:, :accepted], cache=ordinary)
            for got, expected in zip(speculative.cache, ordinary.cache):
                np.testing.assert_array_equal(
                    np.asarray(got.astype(mx.float32)),
                    np.asarray(expected.astype(mx.float32)),
                )
            got = layer(next_row, cache=speculative)
            expected = layer(next_row, cache=ordinary)
            np.testing.assert_array_equal(
                np.asarray(got.astype(mx.float32)),
                np.asarray(expected.astype(mx.float32)),
            )
    finally:
        mx.set_default_device(previous)


def _equal_cache(actual, expected):
    assert len(actual.cache) == len(expected.cache) == 2
    for got, want in zip(actual.cache, expected.cache):
        if got is None or want is None:
            # A cold m=0 rollback materializes the convolution's zero
            # history, while the untouched reference keeps it lazy.
            if got is not None:
                np.testing.assert_array_equal(
                    np.asarray(got.astype(mx.float32)), 0
                )
            if want is not None:
                np.testing.assert_array_equal(
                    np.asarray(want.astype(mx.float32)), 0
                )
            continue
        if got is not None:
            np.testing.assert_array_equal(
                np.asarray(got.astype(mx.float32)),
                np.asarray(want.astype(mx.float32)),
            )


@pytest.mark.parametrize("warm", (False, True))
def test_27b_gdn_geometry_every_acceptance_and_second_verify_on_cpu(warm):
    """Exercise the real recurrent shape without constructing 27B weights.

    The small hidden size keeps the projections cheap; Hk/Hv/Dk/Dv and the
    three-row convolution history match the pinned 27B config. The first
    verify is followed by another partial verify and then an ordinary token.
    """
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        mx.random.seed(3958)
        args = qwen3_5.TextModelArgs(
            hidden_size=32,
            linear_num_value_heads=48,
            linear_num_key_heads=16,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
        )
        layer = qwen3_5.GatedDeltaNet(args)
        layer.set_dtype(mx.bfloat16)
        mx.eval(layer.parameters())
        prefix = mx.random.normal((1, 4, 32)).astype(mx.bfloat16)
        first = mx.random.normal((1, 3, 32)).astype(mx.bfloat16)
        second = mx.random.normal((1, 3, 32)).astype(mx.bfloat16)
        next_row = mx.random.normal((1, 1, 32)).astype(mx.bfloat16)

        for accepted in range(4):
            speculative = ArraysCache(2)
            ordinary = ArraysCache(2)
            if warm:
                layer(prefix, cache=speculative)
                layer(prefix, cache=ordinary)
            speculative.start_speculation()
            layer(first, cache=speculative)
            if accepted < 3:
                assert speculative.trim(3 - accepted) == 3 - accepted
            if accepted:
                layer(first[:, :accepted], cache=ordinary)
            _equal_cache(speculative, ordinary)
            assert speculative.rollback_marker()[1] == accepted

            layer(second, cache=speculative)
            assert speculative.trim(2) == 2
            layer(second[:, :1], cache=ordinary)
            _equal_cache(speculative, ordinary)
            assert speculative.rollback_marker()[1] == accepted + 1
            assert speculative.retire_rollbacks() >= 0
            assert len(speculative._rollbacks) <= 1

            got = layer(next_row, cache=speculative)
            want = layer(next_row, cache=ordinary)
            np.testing.assert_array_equal(
                np.asarray(got.astype(mx.float32)),
                np.asarray(want.astype(mx.float32)),
            )
            _equal_cache(speculative, ordinary)
            assert speculative.rollback_marker()[1] == accepted + 2
    finally:
        mx.set_default_device(previous)


def test_upstream_beta_rounding_and_sigmoid_only_gate_are_not_local_reference():
    """The pinned upstream math must be adapted before a fused candidate gate."""
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        b = mx.array([[[0.734375]]], dtype=mx.bfloat16)
        q = mx.array([[[[0.5, -0.25]]]], dtype=mx.bfloat16)
        k = mx.array([[[[0.25, 0.5]]]], dtype=mx.bfloat16)
        v = mx.array([[[[0.75]]]], dtype=mx.bfloat16)
        a = mx.array([[[-1.0]]], dtype=mx.bfloat16)
        a_log = mx.array([0.0], dtype=mx.bfloat16)
        dt_bias = mx.array([0.0], dtype=mx.bfloat16)
        local = gated_delta_update(
            q, k, v, a, b, a_log, dt_bias, use_kernel=False,
            beta_input_dtype=False,
        )
        upstream_beta = gated_delta_update(
            q, k, v, a, b, a_log, dt_bias, use_kernel=False,
            beta_input_dtype=True,
        )
        assert not np.array_equal(
            np.asarray(local[1].astype(mx.float32)),
            np.asarray(upstream_beta[1].astype(mx.float32)),
        )

        norm = Qwen3NextRMSNormGated(128)
        h = mx.ones((1, 1, 1, 128), dtype=mx.bfloat16)
        z = mx.full((1, 1, 1, 128), 2.0, dtype=mx.bfloat16)
        precise_silu = norm(h, z)
        sigmoid_only = mx.fast.rms_norm(h, norm.weight, norm.eps) * mx.sigmoid(z)
        assert not np.array_equal(
            np.asarray(precise_silu.astype(mx.float32)),
            np.asarray(sigmoid_only.astype(mx.float32)),
        )
    finally:
        mx.set_default_device(previous)

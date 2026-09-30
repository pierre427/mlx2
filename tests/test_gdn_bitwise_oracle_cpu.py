"""CPU falsifiers for the storage-bit/finite GDN Metal comparison helper."""

import mlx.core as mx
import pytest

from test_qwen4_fused_gdn_verify_metal import _eq


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
def test_equal_finite_arrays_pass(dtype):
    x = mx.array([0.0, 1.0, -2.0], dtype=dtype)
    assert _eq(x, mx.array(x))


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
def test_signed_zero_does_not_pass_bit_identity(dtype):
    assert not _eq(mx.array([0.0], dtype), mx.array([-0.0], dtype))


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_matching_nonfinite_payloads_are_refused(value):
    x = mx.array([value])
    assert not _eq(x, mx.array(x))


def test_dtype_or_shape_mismatch_is_refused():
    assert not _eq(mx.array([1.0]), mx.array([1.0], mx.bfloat16))
    assert not _eq(mx.array([1.0]), mx.array([[1.0]]))

"""Real-Metal parity gate for Qwen3.8 bounded fused GDN blocks.

Run under the GPU lock with ``MLX2_RUN_METAL_TESTS=1``. Every admitted width
is bit-exactly compared with the Qwen3.5/Qwen3.8 reference arithmetic for the
snapshot verify and always-committed catch-up forms.
"""
import os

import mlx.core as mx
import pytest
from mlx import nn

from mlx2.runtime.models import qwen4_fused_gdn_verify as fused
from mlx2.runtime.models.gated_delta import gated_delta_update, normalize_gdn_qk

pytestmark = pytest.mark.skipif(
    os.environ.get("MLX2_RUN_METAL_TESTS") != "1" or not mx.metal.is_available(),
    reason="set MLX2_RUN_METAL_TESTS=1 under the GPU lock",
)

HK, HV, DK, DV, K = 16, 48, 128, 128, 4
KD, VD, CD = HK * DK, HV * DV, 2 * HK * DK + HV * DV
WIDTHS = range(2, fused.MAX_VERIFY_WIDTH_PROVEN + 1)


def exact(left, right):
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    bits = mx.uint16 if left.dtype.size == 2 else mx.uint32
    same = mx.array_equal(left.view(bits), right.view(bits))
    finite = mx.all(mx.isfinite(left)) & mx.all(mx.isfinite(right))
    mx.eval(same, finite)
    return bool(same.item()) and bool(finite.item())


def weights():
    dtype = mx.bfloat16
    return (
        (mx.random.normal((CD, K, 1), key=mx.random.key(11)) * 0.02).astype(dtype),
        (mx.random.normal((HV,), key=mx.random.key(12)) * 0.2).astype(mx.float32),
        (mx.random.normal((HV,), key=mx.random.key(13)) * 0.2).astype(dtype),
        (mx.random.normal((DV,), key=mx.random.key(14)) * 0.05 + 1).astype(dtype),
    )


def inputs(steps, salt):
    def draw(shape, offset):
        return (mx.random.normal(shape, key=mx.random.key(1000 * steps + 10 * salt + offset))
                * 0.2).astype(mx.bfloat16)
    return (draw((1, steps, CD), 1), draw((1, steps, VD), 2),
            draw((1, steps, HV), 3), draw((1, steps, HV), 4))


def zeros():
    return (mx.zeros((1, K - 1, CD), mx.bfloat16),
            mx.zeros((1, HV, DV, DK), mx.float32))


def reference(qkv, z, b, a, conv, state, w):
    steps = qkv.shape[1]
    conv_input = mx.concatenate([conv, qkv], axis=1)
    next_conv = mx.contiguous(conv_input[:, -K + 1:, :])
    convolved = nn.silu(mx.conv1d(conv_input, w[0], groups=CD))
    q, k, v = [
        value.reshape(1, steps, heads, dim)
        for value, heads, dim in zip(
            mx.split(convolved, [KD, 2 * KD], axis=-1),
            (HK, HK, HV), (DK, DK, DV),
        )
    ]
    q, k = normalize_gdn_qk(q, k)

    def update(count):
        return gated_delta_update(
            q[:, :count], k[:, :count], v[:, :count], a[:, :count], b[:, :count],
            w[1], w[2], state, None, use_kernel=True,
        )

    out, next_state = update(steps)
    restore_states = [update(count)[1] for count in range(1, steps)]
    restore_convs = [mx.contiguous(conv_input[:, count:count + K - 1])
                     for count in range(1, steps)]
    normalized = mx.fast.rms_norm(out, w[3], 1.0e-6).astype(mx.float32)
    gate = nn.silu(z.reshape(1, steps, HV, DV).astype(mx.float32))
    output = (normalized * gate).astype(mx.bfloat16).reshape(1, steps, VD)
    return output, next_conv, next_state, restore_states, restore_convs


@pytest.fixture
def gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


@pytest.mark.parametrize("steps", WIDTHS)
def test_qwen38_snapshot_verify_matches_reference(gpu, steps):
    ty = fused.probe_qwen4_fused_gdn_verify(
        mx.bfloat16, steps, architecture="qwen38", num_value_heads=HV
    )
    assert ty is not None
    w = weights()
    stock_conv, stock_state = zeros()
    fused_conv, fused_state = zeros()
    for block in range(2):
        qkv, z, b, a = inputs(steps, block)
        stock = reference(qkv, z, b, a, stock_conv, stock_state, w)
        out = fused.qwen4_fused_gdn_verify(
            qkv, z, b, a, fused_conv, w[0], w[1], w[2], fused_state, w[3],
            1.0e-6, threadgroup_y=ty, architecture="qwen38", num_value_heads=HV,
        )
        mx.eval(*stock[:3], *stock[3], *stock[4], *out)
        assert all(exact(stock[i], out[i]) for i in range(3))
        for point in range(steps - 1):
            assert exact(stock[3][point], out[3][:, point])
            assert exact(stock[4][point], out[4][:, point])
        keep = 1 if block == 0 else 0
        if keep:
            stock_conv, stock_state = stock[4][0], stock[3][0]
            fused_conv, fused_state = out[4][:, 0], out[3][:, 0]
        else:
            stock_conv, stock_state = stock[1], stock[2]
            fused_conv, fused_state = out[1], out[2]


@pytest.mark.parametrize("steps", WIDTHS)
def test_qwen38_catchup_matches_reference(gpu, steps):
    ty = fused.probe_qwen4_fused_gdn_catchup(
        mx.bfloat16, steps, architecture="qwen38", num_value_heads=HV
    )
    assert ty is not None
    w = weights()
    stock_conv, stock_state = zeros()
    fused_conv, fused_state = zeros()
    for block in range(2):
        qkv, z, b, a = inputs(steps, block + 2)
        stock = reference(qkv, z, b, a, stock_conv, stock_state, w)
        out = fused.qwen4_fused_gdn_catchup(
            qkv, z, b, a, fused_conv, w[0], w[1], w[2], fused_state, w[3],
            1.0e-6, threadgroup_y=ty, architecture="qwen38", num_value_heads=HV,
        )
        mx.eval(*stock[:3], *out)
        assert all(exact(stock[i], out[i]) for i in range(3))
        stock_conv, stock_state = stock[1], stock[2]
        fused_conv, fused_state = out[1], out[2]

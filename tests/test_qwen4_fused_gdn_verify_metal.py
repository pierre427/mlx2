"""Real-Metal gate: fused Qwen4 GDN verify at every width 2..MAX_VERIFY_WIDTH_PROVEN.

Run under the GPU lock with ``MLX2_RUN_METAL_TESTS=1``.  Each width is checked
bit-exact against the stock Qwen4 verify block (conv1d + SiLU, direct L2 q/k,
``gated_delta_update`` with the Metal kernel, gated RMSNorm), over a chain of
blocks that alternately commit and roll back so later blocks start from a
restored state, for:

* the compact-replay verify (the served path) plus template reconstruction at
  every partial acceptance 1..S-1 and the dynamic (device-count) rebuild;
* the snapshot verify kernel (per-token restore points);
* the catch-up kernel (always commits).

The reference is adapted from mlx-lm-unified
``tests/test_qwen4_fused_gdn_verify_metal.py`` (MIT; see docs/PROVENANCE.md).
"""

import os

import mlx.core as mx
import pytest
from mlx import nn

from mlx2.runtime.models import qwen4_fused_gdn as fused_gdn
from mlx2.runtime.models import qwen4_fused_gdn_verify as fused_verify
from mlx2.runtime.models.gated_delta import gated_delta_update

pytestmark = pytest.mark.skipif(
    os.environ.get("MLX2_RUN_METAL_TESTS") != "1" or not mx.metal.is_available(),
    reason="set MLX2_RUN_METAL_TESTS=1 (under the GPU lock) for the real-Metal gate",
)

WIDTHS = list(range(2, fused_verify.MAX_VERIFY_WIDTH_PROVEN + 1))
BLOCKS = 5


def _stock_block(qkv, z, b, a, conv_state, conv_weight, A_log, dt_bias, norm_weight, state):
    steps = qkv.shape[1]
    keep = fused_gdn.CONV_KERNEL - 1
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    next_conv = mx.contiguous(conv_input[:, -keep:, :])
    convolved = nn.silu(mx.conv1d(conv_input, conv_weight, groups=fused_gdn.CONV_DIM))
    q, k, v = [
        t.reshape(1, steps, h, d)
        for t, h, d in zip(
            mx.split(convolved, [fused_gdn.KEY_DIM, 2 * fused_gdn.KEY_DIM], axis=-1),
            [fused_gdn.NUM_KEY_HEADS, fused_gdn.NUM_KEY_HEADS, fused_gdn.NUM_VALUE_HEADS],
            [fused_gdn.KEY_HEAD_DIM, fused_gdn.KEY_HEAD_DIM, fused_gdn.VALUE_HEAD_DIM],
        )
    ]
    q = q * mx.rsqrt(mx.sum(mx.square(q), axis=-1, keepdims=True) + 1.0e-6)
    k = k * mx.rsqrt(mx.sum(mx.square(k), axis=-1, keepdims=True) + 1.0e-6)
    q = q * (fused_gdn.KEY_HEAD_DIM ** -0.5)

    def update(m):
        return gated_delta_update(
            q[:, :m], k[:, :m], v[:, :m], a[:, :m], b[:, :m], A_log, dt_bias,
            state, None, use_kernel=True, beta_input_dtype=True,
        )

    out, next_state = update(steps)
    restore_states = [update(m)[1] for m in range(1, steps)]
    restore_convs = [mx.contiguous(conv_input[:, m : m + keep, :]) for m in range(1, steps)]
    gate = mx.sigmoid(z.reshape(1, steps, fused_gdn.NUM_VALUE_HEADS, -1).astype(mx.float32))
    output = (mx.fast.rms_norm(out, norm_weight, 1.0e-6).astype(mx.float32) * gate).astype(qkv.dtype)
    return (output.reshape(1, steps, fused_gdn.VALUE_DIM), next_conv, next_state,
            restore_states, restore_convs)


def _weights(dtype=mx.bfloat16):
    conv_weight = (mx.random.normal((fused_gdn.CONV_DIM, fused_gdn.CONV_KERNEL, 1),
                                    key=mx.random.key(1)) * 0.02).astype(dtype)
    A_log = (mx.random.normal((fused_gdn.NUM_VALUE_HEADS,), key=mx.random.key(2)) * 0.2).astype(mx.float32)
    dt_bias = (mx.random.normal((fused_gdn.NUM_VALUE_HEADS,), key=mx.random.key(3)) * 0.2).astype(dtype)
    norm_weight = (mx.random.normal((fused_gdn.VALUE_HEAD_DIM,), key=mx.random.key(4)) * 0.05 + 1).astype(dtype)
    return conv_weight, A_log, dt_bias, norm_weight


def _inputs(steps, block, salt, dtype=mx.bfloat16):
    seed = salt * 1000 * steps + 10 * block

    def draw(shape, offset):
        return (mx.random.normal(shape, key=mx.random.key(seed + offset)) * 0.2).astype(dtype)

    return (draw((1, steps, fused_gdn.CONV_DIM), 1), draw((1, steps, fused_gdn.VALUE_DIM), 2),
            draw((1, steps, fused_gdn.NUM_VALUE_HEADS), 3), draw((1, steps, fused_gdn.NUM_VALUE_HEADS), 4))


def _zeros():
    conv = mx.zeros((1, fused_gdn.CONV_KERNEL - 1, fused_gdn.CONV_DIM), dtype=mx.bfloat16)
    state = mx.zeros((1, fused_gdn.NUM_VALUE_HEADS, fused_gdn.VALUE_HEAD_DIM,
                      fused_gdn.KEY_HEAD_DIM), dtype=mx.float32)
    return conv, state


@pytest.fixture
def gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def _eq(x, y):
    return bool(mx.array_equal(x, y).item())


@pytest.mark.parametrize("steps", WIDTHS)
def test_compact_replay_verify_and_every_rollback_match_stock(gpu, steps):
    ty = fused_verify.probe_qwen4_fused_gdn_replay_verify(mx.bfloat16, steps)
    assert ty is not None, f"probe declined width {steps}"
    dyn_ty = fused_verify.probe_qwen4_fused_gdn_replay_verify(mx.bfloat16, steps, dynamic_accept=True)
    assert dyn_ty is not None
    w = _weights()
    (sc, ss), (fc, fs) = _zeros(), _zeros()
    for block in range(BLOCKS):
        qkv, z, b, a = _inputs(steps, block, 1)
        stock = _stock_block(qkv, z, b, a, sc, w[0], w[1], w[2], w[3], ss)
        out = fused_verify.qwen4_fused_gdn_replay_verify(
            qkv, z, b, a, fc, w[0], w[1], w[2], fs, w[3], 1.0e-6, threadgroup_y=ty)
        keys, corrections, decay = out[3:]
        mx.eval(*stock[:3], *stock[3], *stock[4], *out)
        assert _eq(stock[0], out[0]), ("output", steps, block)
        assert _eq(stock[1], out[1]), ("conv", steps, block)
        assert _eq(stock[2], out[2]), ("state", steps, block)
        for m in range(1, steps):
            rebuilt = fused_verify.qwen4_fused_gdn_reconstruct(
                fs, keys, corrections, decay, m, threadgroup_y=ty)
            assert _eq(stock[3][m - 1], rebuilt), ("reconstruct", steps, block, m)
        counts = mx.array([block % steps], dtype=mx.int32)
        dynamic = fused_verify.qwen4_fused_gdn_reconstruct(
            fs, keys, corrections, decay, counts, threadgroup_y=dyn_ty)
        expected = fs if block % steps == 0 else stock[3][block % steps - 1]
        assert _eq(expected, dynamic), ("dynamic", steps, block)
        keep = block % steps  # 0 commits; k > 0 rolls back to k tokens
        if keep == 0:
            sc, ss, fc, fs = stock[1], stock[2], out[1], out[2]
        else:
            restored_conv = mx.contiguous(mx.concatenate([fc, qkv], axis=1)[:, keep : keep + 3])
            assert _eq(stock[4][keep - 1], restored_conv)
            sc, ss = stock[4][keep - 1], stock[3][keep - 1]
            fc = restored_conv
            fs = fused_verify.qwen4_fused_gdn_reconstruct(
                fs, keys, corrections, decay, keep, threadgroup_y=ty)


def test_short_verify_after_wide_uses_only_its_own_replay_tape(gpu):
    """A width-2 verify after width-8 must not restore an old tape slot.

    Rebuild the committed prefix from zero after each rollback, then use the
    next verify as a continuation check on both logits input and GDN state.
    """
    weights = _weights()
    fused_conv, fused_state = _zeros()
    committed = []
    for block, (steps, keep) in enumerate(((8, 2), (2, 1), (3, 0))):
        ty = fused_verify.probe_qwen4_fused_gdn_replay_verify(mx.bfloat16, steps)
        assert ty is not None
        stock_conv, stock_state = _zeros()
        for old_qkv, old_z, old_b, old_a in committed:
            old = _stock_block(
                old_qkv, old_z, old_b, old_a, stock_conv,
                weights[0], weights[1], weights[2], weights[3], stock_state,
            )
            stock_conv, stock_state = old[1:3]
        assert _eq(fused_conv, stock_conv)
        assert _eq(fused_state, stock_state)

        qkv, z, b, a = _inputs(steps, block, 4)
        stock = _stock_block(
            qkv, z, b, a, stock_conv,
            weights[0], weights[1], weights[2], weights[3], stock_state,
        )
        out = fused_verify.qwen4_fused_gdn_replay_verify(
            qkv, z, b, a, fused_conv, weights[0], weights[1], weights[2],
            fused_state, weights[3], 1.0e-6, threadgroup_y=ty,
        )
        mx.eval(*stock[:3], *out)
        assert _eq(out[0], stock[0]), ("output", steps)
        assert _eq(out[1], stock[1]), ("conv", steps)
        assert _eq(out[2], stock[2]), ("state", steps)

        if keep:
            committed.append(tuple(value[:, :keep] for value in (qkv, z, b, a)))
            fused_conv = mx.contiguous(
                mx.concatenate([fused_conv, qkv], axis=1)[:, keep : keep + 3]
            )
            fused_state = fused_verify.qwen4_fused_gdn_reconstruct(
                fused_state, *out[3:], keep, threadgroup_y=ty,
            )


@pytest.mark.parametrize("steps", WIDTHS)
def test_snapshot_verify_matches_stock(gpu, steps):
    ty = fused_verify.probe_qwen4_fused_gdn_verify(mx.bfloat16, steps)
    assert ty is not None
    w = _weights()
    (sc, ss), (fc, fs) = _zeros(), _zeros()
    for block in range(BLOCKS):
        qkv, z, b, a = _inputs(steps, block, 2)
        stock = _stock_block(qkv, z, b, a, sc, w[0], w[1], w[2], w[3], ss)
        out = fused_verify.qwen4_fused_gdn_verify(
            qkv, z, b, a, fc, w[0], w[1], w[2], fs, w[3], 1.0e-6, threadgroup_y=ty)
        mx.eval(*stock[:3], *stock[3], *stock[4], *out)
        for i, name in enumerate(("output", "conv", "state")):
            assert _eq(stock[i], out[i]), (name, steps, block)
        for p in range(steps - 1):
            assert _eq(stock[3][p], out[3][:, p]), ("state point", steps, block, p)
            assert _eq(stock[4][p], out[4][:, p]), ("conv point", steps, block, p)
        keep = block % steps
        if keep == 0:
            sc, ss, fc, fs = stock[1], stock[2], out[1], out[2]
        else:
            sc, ss = stock[4][keep - 1], stock[3][keep - 1]
            fc, fs = out[4][:, keep - 1], out[3][:, keep - 1]


@pytest.mark.parametrize("steps", WIDTHS)
def test_catchup_matches_stock(gpu, steps):
    ty = fused_verify.probe_qwen4_fused_gdn_catchup(mx.bfloat16, steps)
    assert ty is not None
    w = _weights()
    (sc, ss), (fc, fs) = _zeros(), _zeros()
    for block in range(BLOCKS):
        qkv, z, b, a = _inputs(steps, block, 3)
        stock = _stock_block(qkv, z, b, a, sc, w[0], w[1], w[2], w[3], ss)
        out = fused_verify.qwen4_fused_gdn_catchup(
            qkv, z, b, a, fc, w[0], w[1], w[2], fs, w[3], 1.0e-6, threadgroup_y=ty)
        mx.eval(*stock[:3], *out)
        for i, name in enumerate(("output", "conv", "state")):
            assert _eq(stock[i], out[i]), (name, steps, block)
        sc, ss, fc, fs = stock[1], stock[2], out[1], out[2]

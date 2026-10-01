"""Sorted MoE gather padded up to MLX's streaming-kernel floor (mlx-serve#671).

CPU: the pad arithmetic, the pad-row layout, and that padded and unpadded
sorted gathers return the same rows in the same order (the CPU gather_qmm has
no kernel switch, so the comparison is bit-exact there).  Metal: the padded
gather is compared bit for bit against the unpadded one; it is NOT expected to
match (gather_qmm_rhs vs gather_qmv reduce differently), so the test records
the difference and checks the pad rows are discarded and the real rows stay
close.  Run with MLX2_RUN_METAL_TESTS=1 under the GPU queue.
"""

import os

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx2.runtime.models import switch_layers as SL
from mlx2.runtime.models.switch_layers import (
    QuantizedSwitchLinear,
    SwitchGLU,
    SwitchLinear,
    SwitchMLP,
)


@pytest.fixture
def pad_floor():
    saved = SL._RHS_PAD_MIN_ROWS_PER_EXPERT

    def set_floor(value):
        SL._RHS_PAD_MIN_ROWS_PER_EXPERT = value

    yield set_floor
    SL._RHS_PAD_MIN_ROWS_PER_EXPERT = saved


def test_pad_arithmetic(pad_floor):
    pad_floor(0)
    assert SL._rhs_stream_pad(300, 128) == 0  # off by default
    pad_floor(2)
    assert SL._rhs_stream_pad(255, 128) == 0  # under 2 rows/expert
    assert SL._rhs_stream_pad(256, 128) == 512 - 256
    assert SL._rhs_stream_pad(511, 128) == 1
    assert SL._rhs_stream_pad(512, 128) == 0  # MLX streams already
    assert SL._rhs_stream_pad(4096, 128) == 0
    assert SL._rhs_stream_pad(300, None) == 0
    pad_floor(1)
    assert SL._rhs_stream_pad(3, 2) == 16 - 3  # B >= 16 floor as well


def test_pad_experts_only_for_native_quantized():
    q = SwitchLinear(64, 32, 8, bias=False).to_quantized(group_size=64, bits=4)
    q_tail = SwitchLinear(96, 32, 8, bias=False).to_quantized(group_size=32, bits=4)
    dense = SwitchLinear(64, 32, 8, bias=False)
    assert SL._rhs_pad_experts(q, q) == 8
    assert SL._rhs_pad_experts(q, dense) is None  # gather_mm streams already
    assert SL._rhs_pad_experts(q, q_tail) is None  # K % 64: runs unsorted


def test_gather_sort_pad_rows_layout(pad_floor):
    pad_floor(1)
    T, k, E, D = 5, 3, 4, 8
    x = mx.arange(T * D, dtype=mx.float32).reshape(1, T, 1, 1, D)
    indices = mx.array([[[3, 0, 1], [2, 1, 0], [0, 3, 2], [1, 2, 3], [0, 1, 2]]], mx.uint32)
    xs, idx, inv = SL._gather_sort(x, indices, E)
    n, pad = T * k, 16 - T * k
    assert idx.size == n + pad and xs.shape[0] == n + pad
    host = idx.tolist()
    assert host == sorted(host)  # still sorted with the pad
    assert host[n:] == [host[n - 1]] * pad  # pad repeats the last sorted row
    assert inv.size == n and max(inv.tolist()) < n  # unsort never reads a pad row
    # Each real sorted row carries its own token's activation.
    flat = indices.reshape(-1).tolist()
    for a, pos in enumerate(inv.tolist()):
        assert host[pos] == flat[a]
        assert mx.array_equal(xs[pos], x.reshape(T, 1, D)[a // k]).item()


@pytest.mark.parametrize("cls", ["mlp", "glu"])
def test_padded_switch_matches_unpadded_on_cpu(pad_floor, cls):
    mx.random.seed(3)
    T, k, E, D, H = 7, 3, 16, 64, 128  # 42 sorted rows < 4 * 16
    if cls == "mlp":
        mod = SwitchMLP(D, H, E, activation=nn.ReLU2())
    else:
        mod = SwitchGLU(D, H, E)
    nn.quantize(mod, group_size=64, bits=4)
    assert isinstance(next(m for _, m in mod.named_modules()
                           if isinstance(m, QuantizedSwitchLinear)), QuantizedSwitchLinear)
    x = mx.random.normal((2, T, D)).astype(mx.bfloat16)
    indices = mx.argpartition(mx.random.normal((2, T, E)), kth=E - k, axis=-1)[..., E - k:]
    indices = indices.astype(mx.uint32)
    pad_floor(0)
    base = mod(x, indices)
    calls, rows = SL.rhs_pad_calls, SL.rhs_pad_rows
    pad_floor(1)
    padded = mod(x, indices)
    assert SL.rhs_pad_calls == calls + 1
    assert SL.rhs_pad_rows == rows + (4 * E - 2 * T * k)
    assert padded.shape == base.shape == (2, T, k, D)
    assert mx.array_equal(padded, base).item()


metal = pytest.mark.skipif(
    os.environ.get("MLX2_RUN_METAL_TESTS") != "1" or not mx.metal.is_available(),
    reason="set MLX2_RUN_METAL_TESTS=1 under the GPU queue",
)


@metal
def test_metal_padded_gather_bits(pad_floor):
    """Records whether the streaming kernel is bit-exact against gather_qmv."""
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        mx.random.seed(7)
        # 96 sorted rows < 4 * 32: MLX runs gather_qmv unless padded.
        T, k, E, D, H = 16, 6, 32, 256, 128
        mod = SwitchMLP(D, H, E, activation=nn.ReLU2())
        nn.quantize(mod, group_size=64, bits=8)
        x = mx.random.normal((1, T, D)).astype(mx.bfloat16)
        indices = mx.argpartition(mx.random.normal((1, T, E)), kth=E - k, axis=-1)[..., E - k:]
        indices = indices.astype(mx.uint32)
        pad_floor(0)
        base = mod(x, indices)
        pad_floor(1)
        calls = SL.rhs_pad_calls
        padded = mod(x, indices)
        mx.eval(base, padded)
        assert SL.rhs_pad_calls == calls + 1
        diff = mx.abs(padded.astype(mx.float32) - base.astype(mx.float32))
        scale = mx.abs(base.astype(mx.float32)).max().item()
        # Not bit-exact by construction (different reduction order); the
        # real rows must still be the same rows, within bf16 rounding.
        assert diff.max().item() <= 2e-2 * scale
        print("metal padded vs unpadded: equal=%s n_diff=%d max_abs=%.3e scale=%.3e" % (
            mx.array_equal(padded, base).item(), int((diff > 0).sum().item()),
            diff.max().item(), scale))
    finally:
        mx.set_default_device(previous)

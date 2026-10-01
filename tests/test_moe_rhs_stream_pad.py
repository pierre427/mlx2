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
from mlx2.runtime.models.qwen3_next import FusedGateUpSwitchGLU


@pytest.fixture
def pad_floor():
    saved = SL._RHS_PAD_MIN_ROWS_PER_EXPERT

    def set_floor(value):
        SL._RHS_PAD_MIN_ROWS_PER_EXPERT = value

    yield set_floor
    SL._RHS_PAD_MIN_ROWS_PER_EXPERT = saved


def test_pad_default_is_three_rows_per_expert():
    assert SL._rhs_pad_floor({}) == 3
    assert SL._rhs_pad_floor({"MLX2_MOE_RHS_PAD_MIN_ROWS": "0"}) == 0
    assert SL._rhs_pad_floor({"MLX2_MOE_RHS_PAD_MIN_ROWS": "2"}) == 2


def test_pad_arithmetic(pad_floor):
    pad_floor(0)
    assert SL._rhs_stream_pad(300, 128) == 0  # 0 turns it off
    pad_floor(2)
    assert SL._rhs_stream_pad(255, 128) == 0  # under 2 rows/expert
    assert SL._rhs_stream_pad(256, 128) == 512 - 256
    assert SL._rhs_stream_pad(511, 128) == 1
    assert SL._rhs_stream_pad(512, 128) == 0  # MLX streams already
    assert SL._rhs_stream_pad(4096, 128) == 0
    pad_floor(1)
    assert SL._rhs_stream_pad(3, 2) == 16 - 3  # B >= 16 floor as well


def test_pad_sorted_tail_layout():
    x = mx.arange(5 * 4, dtype=mx.float32).reshape(5, 1, 4)
    idx = mx.array([0, 0, 2, 3, 3], mx.uint32)
    xp, ip = SL._pad_sorted_tail(x, idx, 3)
    assert ip.tolist() == [0, 0, 2, 3, 3, 3, 3, 3]  # sorted, no new expert
    assert mx.array_equal(xp[:5], x).item()  # real rows untouched, in order
    assert all(mx.array_equal(xp[i], x[4]).item() for i in range(5, 8))


def _quantized_projection(D, N, E, bits=4, group_size=64):
    return SwitchLinear(D, N, E, bias=False).to_quantized(group_size=group_size, bits=bits)


def test_projection_pads_only_native_sorted(pad_floor):
    pad_floor(1)
    mx.random.seed(1)
    E, D, N, n = 8, 64, 32, 20  # 20 rows < 4 * 8
    proj = _quantized_projection(D, N, E)
    x = mx.random.normal((n, 1, D))
    idx = mx.sort(mx.random.randint(0, E, (n,)).astype(mx.uint32))
    calls, rows = SL.rhs_pad_calls, SL.rhs_pad_rows
    out = proj(x, idx, sorted_indices=True)
    assert out.shape == (n, 1, N)
    assert (SL.rhs_pad_calls, SL.rhs_pad_rows) == (calls + 1, rows + 4 * E - n)
    proj(x, idx, sorted_indices=False)  # unsorted: never padded
    tail = _quantized_projection(96, N, E, group_size=32)  # K % 64: runs unsorted
    proj_dense = SwitchLinear(D, N, E, bias=False)  # gather_mm streams already
    tail(mx.random.normal((n, 1, 96)), idx, sorted_indices=True)
    proj_dense(x, idx, sorted_indices=True)
    assert SL.rhs_pad_calls == calls + 1


@pytest.mark.parametrize("cls", ["mlp", "glu", "fused_glu"])
def test_padded_switch_matches_unpadded_on_cpu(pad_floor, cls):
    mx.random.seed(3)
    T, k, E, D, H = 7, 3, 16, 64, 128  # 42 sorted rows < 4 * 16
    if cls == "mlp":
        mod = SwitchMLP(D, H, E, activation=nn.ReLU2())
    elif cls == "glu":
        mod = SwitchGLU(D, H, E)
    else:
        mod = FusedGateUpSwitchGLU(D, H, E)
    nn.quantize(mod, group_size=64, bits=4)
    nproj = sum(isinstance(m, QuantizedSwitchLinear) for _, m in mod.named_modules())
    assert nproj == (2 if cls != "glu" else 3)
    x = mx.random.normal((2, T, D)).astype(mx.bfloat16)
    indices = mx.argpartition(mx.random.normal((2, T, E)), kth=E - k, axis=-1)[..., E - k:]
    indices = indices.astype(mx.uint32)
    pad_floor(0)
    base = mod(x, indices)
    calls, rows = SL.rhs_pad_calls, SL.rhs_pad_rows
    pad_floor(1)
    padded = mod(x, indices)
    # One padded gather_qmm per projection, each lifted to 4 rows/expert.
    assert SL.rhs_pad_calls == calls + nproj
    assert SL.rhs_pad_rows == rows + nproj * (4 * E - 2 * T * k)
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
        assert SL.rhs_pad_calls == calls + 2  # fc1 and fc2
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

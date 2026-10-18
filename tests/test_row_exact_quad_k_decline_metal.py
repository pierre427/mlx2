"""Real-Metal gate: no row-exact projection route differs from the one-row call
at K = 64/128 (the kernel declines there; the per-row form runs)."""

import mlx.core as mx
import pytest
from mlx import nn

from mlx2.runtime.models import row_exact_qmv as REQ
from mlx2.runtime.row_exact import ops
from mlx2.runtime.row_exact.canary import CANARIES

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")


@pytest.fixture(autouse=True)
def gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    CANARIES.reset()
    try:
        yield
    finally:
        CANARIES.reset()
        mx.set_default_device(previous)


def _linear(k, n=512, bits=8, group_size=64, seed=5):
    mx.random.seed(seed)
    dense = nn.Linear(k, n, bias=False)
    dense.weight = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
    linear = nn.QuantizedLinear.from_linear(dense, group_size=group_size, bits=bits)
    linear.scales = linear.scales.astype(mx.bfloat16)
    linear.biases = linear.biases.astype(mx.bfloat16)
    mx.eval(linear.parameters())
    return linear


def _strips(k, count=32, rows=16):
    return [
        (mx.random.normal((1, rows, k), key=mx.random.key(100 + i)) * 2).astype(mx.bfloat16)
        for i in range(count)
    ]


def _mismatches(have, want):
    return int(mx.sum(have.view(mx.uint16) != want.view(mx.uint16)).item())


@pytest.mark.parametrize("k", [64, 128])
def test_quad_k_projection_equals_its_one_row_calls(k):
    linear = _linear(k)
    for strip in _strips(k):
        y = ops.quantized_linear(linear, strip, stage="t")
        ref = REQ.per_row(linear, strip)
        assert _mismatches(y, ref) == 0

"""row_exact_qmv declines the two K values its one-row reference leaves.

On MLX 0.32.4 (M3 Pro, 2026-10-18) the stock one-row quantized call at
K = 64 and 128 (exactly) differs from the qmv order ``row_exact_qmv``
reproduces on about 1 in 10^4 outputs, for every bits and group size
tried; K = 32, 96, 160, 192 and up match.  A canary probing a few rows of
a narrow layer (for example 4 rows x 256 outputs) passed anyway.  The kernel now declines
those two K values, so the per-row form (the one-row call) runs there.
The Metal file pins the bits.
"""

import mlx.core as mx
import pytest
from mlx import nn

from mlx2.runtime.models import row_exact_qmv as REQ


@pytest.mark.parametrize("k, reason", [
    (32, None),
    (64, "quad_k_stock_traversal"),
    (96, None),
    (128, "quad_k_stock_traversal"),
    (192, None),
    (4096, None),
])
def test_small_k_layouts_decline_the_kernel(monkeypatch, k, reason):
    monkeypatch.setattr(REQ.mx, "default_device", lambda: REQ.mx.gpu)
    monkeypatch.setattr(REQ.mx.metal, "is_available", lambda: True)
    linear = nn.QuantizedLinear(k, 256, bias=False, group_size=32, bits=8)
    linear.set_dtype(mx.bfloat16)
    assert REQ.decline_reason(linear, mx.zeros((1, 3, k), mx.bfloat16)) == reason
    assert REQ.layout_decline_reason(linear) == reason

"""Metal: multi-row dense matvec with MLX's one-row gemv arithmetic per row.

Every row of ``gemv_rows`` (and of ``row_exact.ops.matvec_rows`` under
``MLX2_GEMV_ROWS_KERNEL=fused``) must equal the stock one-row ``x @ W.T``
bit for bit, at the Super VL MoE shapes (latent in/out, fp32 router) and at
a non-aligned edge shape.
"""

import mlx.core as mx
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="gemv rows kernel is Metal-only"
)

from mlx2.runtime.models import gemv_rows_exact as gemv  # noqa: E402
from mlx2.runtime.row_exact import ops  # noqa: E402

SHAPES = [
    (1024, 4096, mx.bfloat16),  # fc1_latent_proj
    (4096, 1024, mx.bfloat16),  # fc2_latent_proj
    (512, 4096, mx.float32),    # router in fp32
    (4096, 4096, mx.bfloat16),
    (1000, 4100, mx.bfloat16),  # tail rows and a K leftover block
]


@pytest.fixture(autouse=True)
def _gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    yield
    mx.set_default_device(previous)


def _bits(value):
    mx.eval(value)
    return np.asarray(value.view(mx.uint32 if value.dtype == mx.float32 else mx.uint16))


@pytest.mark.parametrize("n,k,dtype", SHAPES)
def test_every_row_matches_the_one_row_call(monkeypatch, n, k, dtype):
    key = mx.random.key(n * 7 + k)
    weight = (mx.random.normal((n, k), key=key) * 0.05).astype(dtype)
    x = mx.random.normal((1, 16, k), key=mx.random.split(key)[0]).astype(dtype)
    expected = _bits(mx.concatenate([x[:, r : r + 1] @ weight.T for r in range(16)], axis=1))
    for rows in range(2, 17):
        assert gemv.decline_reason(weight, x[:, :rows]) is None
        np.testing.assert_array_equal(
            _bits(gemv.gemv_rows(weight, x[:, :rows])), expected[:, :rows],
            err_msg=f"gemv_rows rows={rows}",
        )
    monkeypatch.setenv("MLX2_GEMV_ROWS_KERNEL", "fused")
    for rows in (2, 6, 8, 9, 16):
        np.testing.assert_array_equal(
            _bits(ops.matvec_rows(weight, x[:, :rows])), expected[:, :rows],
            err_msg=f"matvec_rows fused rows={rows}",
        )
    assert all(gemv.status()["verdicts"].values())


def test_unmirrored_tiles_and_modes_decline(monkeypatch):
    x = mx.zeros((1, 4, 64), dtype=mx.bfloat16)
    assert gemv.decline_reason(mx.zeros((256, 64), dtype=mx.bfloat16), x) == "tile"
    assert gemv.tile(128, 4096) is None  # K >= 16 N takes MLX's BN = 8 branch
    assert gemv.decline_reason(
        mx.zeros((1024, 4096), dtype=mx.bfloat16), mx.zeros((1, 1, 4096), dtype=mx.bfloat16)
    ) == "rows"
    monkeypatch.setenv("MLX2_GEMV_ROWS_KERNEL", "wide")
    with pytest.raises(ValueError):
        gemv.kernel_mode()

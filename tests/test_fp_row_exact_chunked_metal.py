"""Metal: exact FP8/NVFP4 verify rows, every mode, against the one-row call.

``fused_chunked`` runs 9..16-row strips as balanced fused launches of at
most 8 rows; every row must still equal the stock one-row
``quantized_matmul`` bit for bit, as ``fused`` (2..8) and ``loop`` do.
"""

import mlx.core as mx
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="fp row-exact kernels are Metal-only"
)

from mlx2.runtime.models import fp_row_exact_qmv as fq  # noqa: E402

# (bits, group size, quantize mode); shapes of the Super VL dense layers.
FORMATS = [(8, 32, "mxfp8"), (4, 16, "nvfp4")]
SHAPES = [(4096, 8192), (8192, 4096), (4096, 18560)]


@pytest.fixture(autouse=True)
def _gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    yield
    mx.set_default_device(previous)


def test_chunks_are_balanced_and_never_single_rows():
    for rows in range(9, 17):
        chunks = fq.fused_chunks(rows)
        assert sum(chunks) == rows and max(chunks) <= fq.FUSED_MAX_ROWS
        assert min(chunks) >= 2 and max(chunks) - min(chunks) <= 1
    for bad in (8, 17, 1):
        with pytest.raises(ValueError):
            fq.fused_chunks(bad)


@pytest.mark.parametrize("k,n", SHAPES)
@pytest.mark.parametrize("bits,group,mode", FORMATS)
def test_every_mode_matches_the_one_row_call(monkeypatch, bits, group, mode, k, n):
    key = mx.random.key(k + n + bits)
    weight = (mx.random.normal((n, k), key=key) * 0.05).astype(mx.bfloat16)
    wq, scales = mx.quantize(weight, group_size=group, bits=bits, mode=mode)[:2]
    x = mx.random.normal((1, 16, k), key=mx.random.split(key)[0]).astype(mx.bfloat16)
    reference = mx.concatenate([
        mx.quantized_matmul(x[:, row : row + 1], wq, scales=scales, biases=None,
                            transpose=True, group_size=group, bits=bits, mode=mode)
        for row in range(16)
    ], axis=1)
    mx.eval(reference)
    expected = np.asarray(reference.view(mx.uint16))
    for kernel_mode in fq._KERNEL_MODES:
        monkeypatch.setenv("MLX2_FP_ROW_EXACT_KERNEL", kernel_mode)
        for rows in range(2, 17):
            y = fq.rows_qmv(x[:, :rows], wq, scales, bits=bits, group_size=group)
            mx.eval(y)
            np.testing.assert_array_equal(
                np.asarray(y.view(mx.uint16)), expected[:, :rows],
                err_msg=f"{kernel_mode} rows={rows}",
            )

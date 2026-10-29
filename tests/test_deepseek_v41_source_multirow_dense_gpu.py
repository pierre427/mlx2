"""Synthetic GPU gate: R-row Source dense products equal R stock M=1 products.

Bitwise per row, at the DS4.1 Source projection geometries. No model loads.
"""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from mlx2.runtime.gguf import source_multirow_dense as multirow

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal GPU required")


@pytest.fixture(autouse=True)
def _gpu_default():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)

# (N, K): q_a, kv, q_b, wo_b, shared gate/up, shared down, compressor, ragged.
MXFP8 = [(1280, 5120), (512, 5120), (32768, 1280), (5120, 8192), (2304, 5120),
         (5120, 2304), (1024, 5120), (102, 1280), (96, 2336)]


def _bits(value):
    return np.array(value).view(np.uint32)


@pytest.mark.parametrize("n,k", MXFP8)
@pytest.mark.parametrize("rows", (1, 2, 3, 6))
def test_mxfp8_rows_match_stock_m1_bitwise(n, k, rows):
    with mx.stream(mx.gpu):
        q, scales = mx.quantize(mx.random.normal((n, k), key=mx.random.key(n + k)) * 0.05, mode="mxfp8")[:2]
        x = mx.random.normal((rows, k), key=mx.random.key(rows))
        reference = mx.concatenate([mx.quantized_matmul(x[r:r + 1][None], q, scales=scales, mode="mxfp8")
                                    .reshape(1, n) for r in range(rows)])
        actual = multirow.qmv(x, q, scales)
        mx.eval(reference, actual)
    assert np.array_equal(_bits(reference), _bits(actual))


@pytest.mark.parametrize("n,k,dtype,groups", [
    (129280, 5120, "float32", 1), (129280, 5120, "bfloat16", 1),
    (1024, 4096, "bfloat16", 8), (384, 5120, "float32", 1), (384, 5120, "bfloat16", 1)])
@pytest.mark.parametrize("rows", (1, 2, 5))
def test_gemv_rows_match_stock_m1_bitwise(n, k, dtype, groups, rows):
    with mx.stream(mx.gpu):
        weight = (mx.random.normal((groups * n, k), key=mx.random.key(7)) * 0.02).astype(getattr(mx, dtype))
        x = mx.random.normal((rows, groups, k), key=mx.random.key(rows + 11))
        full = weight.astype(mx.float32).reshape(groups, n, k)
        reference = mx.stack([mx.stack([x[r, g][None] @ full[g].T for g in range(groups)]).reshape(groups, n)
                              for r in range(rows)])
        actual = multirow.gemv(x, weight, groups=groups)
        mx.eval(reference, actual)
    assert actual.shape == (rows, groups, n)
    assert np.array_equal(_bits(reference), _bits(actual))


def test_gemv_matches_source_dense_head_law():
    from mlx2.runtime.gguf import source_dense
    if not source_dense.device_supported():
        pytest.skip("pinned Source dense helper requires its audited device")
    with mx.stream(mx.gpu):
        weight = (mx.random.normal((129280, 5120), key=mx.random.key(3)) * 0.02).astype(mx.bfloat16)
        x = mx.random.normal((3, 1, 5120), key=mx.random.key(4))
        reference = mx.concatenate([source_dense.project(x[r][None], weight, "head").reshape(1, -1)
                                    for r in range(3)])
        actual = multirow.gemv(x, weight).reshape(3, -1)
        mx.eval(reference, actual)
    assert np.array_equal(_bits(reference), _bits(actual))


def test_refuses_unadmitted_geometry():
    with pytest.raises(ValueError):
        multirow.qmv_law(4, 5120)
    with pytest.raises(ValueError):
        multirow.gemv_bm(24, 20480)  # stock BN=8 HC-mix law is not reimplemented
    with pytest.raises(ValueError):
        multirow.qmv(mx.zeros((9, 512)), mx.zeros((8, 128), mx.uint32), mx.zeros((8, 16), mx.uint8))

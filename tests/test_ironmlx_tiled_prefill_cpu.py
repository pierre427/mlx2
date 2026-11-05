"""Independent affine4/SwiGLU CPU oracles; no shader dispatch."""
import subprocess
import sys

import numpy as np
import pytest

from mlx2.experimental.ironmlx_tiled_prefill import (
    ENV,
    bf16,
    metal_source,
    project,
    reference,
    status,
)


def packed(codes, ntcol):
    n, k = codes.shape
    words = np.zeros((n, k // 8), dtype=np.uint32)
    for i in range(8):
        words |= codes[:, i::8].astype(np.uint32) << (4 * i)
    return words.reshape(n // ntcol, ntcol, k // 64, 8).transpose(0, 2, 1, 3).copy()


@pytest.mark.parametrize("ntcol", [32, 64])
@pytest.mark.parametrize("rows", [1, 31, 128, 129])
def test_tiled_weight_addressing_and_fused_output_against_dense_oracle(ntcol, rows):
    rng = np.random.default_rng(61)
    codes = rng.integers(0, 16, (128, 128), dtype=np.uint32)
    scales = np.full((128, 2), 1 / 128, dtype=np.float32)
    biases = np.full((128, 2), -1 / 16, dtype=np.float32)
    x = bf16(rng.normal(0, .05, (rows, 128)))
    # All dequantized operands are exactly representable: isolate packing
    # and epilogue behavior from a different CPU affine/FMA convention.
    weight = codes.astype(np.float32) / 128 - 1 / 16
    g, u = np.split(bf16(x @ weight.T), 2, axis=-1)
    expected = g / (1 + np.exp(-g)) * u
    actual = reference(x, packed(codes, ntcol), scales, biases, ntcol=ntcol)
    np.testing.assert_allclose(actual, expected, rtol=.03, atol=2e-6)
    assert actual.shape == (rows, 64) and np.isfinite(actual).all()


def test_bf16_rounding_ties_even_and_refuses_nonfinite():
    bits = np.array([0x3f808000, 0x3f818000], dtype=np.uint32)
    assert bf16(bits.view(np.float32)).view(np.uint32).tolist() == [0x3f800000, 0x3f820000]
    with pytest.raises(ValueError, match="finite"):
        bf16([float("nan")])


def test_sigmoid_positive_subtraction_rounds_after_fp32_not_before():
    # A constant affine bias yields an exactly known gate projection. This
    # BF16 boundary distinguishes rounding y from rounding the final 1-y.
    x = np.ones((1, 64), dtype=np.float32)
    codes = np.zeros((64, 64), dtype=np.uint32)
    scales = np.zeros((64, 1), dtype=np.float32)
    biases = np.full((64, 1), np.float32(1 / 64))
    biases[:32] = np.float32(0.00799560546875 / 64)
    out = reference(x, packed(codes, 32), scales, biases)
    gate = np.float32(0.00799560546875)
    expected = bf16(bf16(gate * bf16(np.float32(1) - np.float32(1) /
        (np.float32(1) + np.exp(gate)))) * np.float32(1))
    np.testing.assert_array_equal(out, np.full((1, 32), expected))
    assert out[0, 0] == np.float32(0.0040283203125)
    assert out[0, 0] != bf16(gate * bf16(np.float32(0.5)))


@pytest.mark.parametrize("ntcol", [0, 16, True])
def test_unknown_weight_layout_refused(ntcol):
    with pytest.raises(ValueError, match="layout"):
        metal_source(ntcol)


def test_disabled_route_refuses_before_device_import(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    with pytest.raises(RuntimeError, match="select"):
        project(None, None, None, None)
    assert status()["selected"] is False and status()["qualified"] is False
    monkeypatch.setenv(ENV, "yes")
    with pytest.raises(ValueError):
        status()


def test_shader_generation_and_cpu_reference_have_no_mlx_import():
    result = subprocess.run([sys.executable, "-c",
        ("import sys; from mlx2.experimental.ironmlx_tiled_prefill import metal_source; "
        "assert 'mlx.core' not in sys.modules; assert 'kernel void' in metal_source(); "
        "assert 'mlx.core' not in sys.modules")], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_invalid_metadata_fails_before_numpy_gemm():
    with pytest.raises(ValueError, match="geometry"):
        reference(np.ones((1, 64)), np.zeros(10, dtype=np.uint32),
                  np.ones((64, 1)), np.zeros((64, 1)))


@pytest.mark.parametrize("rows,n,k", [(0, 64, 64), (1, 0, 64), (1, 64, 0)])
def test_empty_geometry_refused(rows, n, k):
    with pytest.raises(ValueError, match="geometry"):
        reference(np.ones((rows, k)), np.zeros(n * k // 8, dtype=np.uint32),
                  np.ones((n, k // 64)), np.zeros((n, k // 64)))


def test_cpu_reference_refuses_array_protocol_before_conversion():
    class DeviceArray:
        def __array__(self, *args, **kwargs):
            raise AssertionError("device synchronization must not occur")
    with pytest.raises(ValueError, match="NumPy"):
        reference(DeviceArray(), np.zeros(512, dtype=np.uint32),
                  np.ones((64, 1)), np.zeros((64, 1)))

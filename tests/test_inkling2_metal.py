"""Inkling 2 Metal kernels (GPU; run under the GPU lock)."""

import mlx.core as mx
import pytest

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")


@pytest.mark.parametrize("rows,n,k", [(1, 4099, 256), (3, 1024, 2304), (17, 515, 128), (8, 201024, 2304)])
def test_head_gemv_matches_fp32_gemm(rows, n, k):
    from mlx2.runtime.models.inkling2.language import head_gemv

    mx.random.seed(rows)
    x = mx.random.normal((1, rows, k)).astype(mx.bfloat16)
    w = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
    with mx.stream(mx.gpu):
        got = head_gemv(x, w)
        ref = x.astype(mx.float32) @ w.astype(mx.float32).T
    assert got.dtype == mx.float32 and got.shape == (1, rows, n)
    scale = mx.abs(ref).max().item()
    assert mx.abs(got - ref).max().item() <= 2e-6 * max(scale, 1.0) * (k / 128)


def test_head_gemv_on_leading_row_view_matches_copy():
    """The kernel reads the cropped head as a leading-row view; per-row
    reductions make it bit-identical to a contiguous copy."""
    from mlx2.runtime.models.inkling2.language import head_gemv

    mx.random.seed(5)
    x = mx.random.normal((1, 2, 2304)).astype(mx.bfloat16)
    w = (mx.random.normal((1030, 2304)) * 0.02).astype(mx.bfloat16)
    with mx.stream(mx.gpu):
        view = head_gemv(x, w[:1000])
        copy = head_gemv(x, mx.array(w[:1000].tolist(), dtype=mx.bfloat16))
    assert mx.array_equal(view, copy)


@pytest.mark.parametrize("rows,keys", [(1, 37), (2, 1024), (1, 5000)])
def test_sdpa_sinks_matches_concatenated_zero_column(rows, keys):
    """The ``inkling2_sink_sdpa`` identity on the GPU at BF16."""
    mx.random.seed(keys)
    h, hk, d = 24, 8, 128
    q = mx.random.normal((rows, h, 1, d)).astype(mx.bfloat16)
    k = mx.random.normal((rows, hk, keys, d)).astype(mx.bfloat16)
    v = mx.random.normal((rows, hk, keys, d)).astype(mx.bfloat16)
    mask = (mx.random.normal((rows, h, 1, keys)) * 2).astype(mx.bfloat16)
    sink = (mx.random.normal((rows, h)) * 3).astype(mx.bfloat16)
    with mx.stream(mx.gpu):
        zeros = mx.zeros((rows, hk, 1, d), mx.bfloat16)
        ref = mx.fast.scaled_dot_product_attention(
            q, mx.concatenate([k, zeros], 2), mx.concatenate([v, zeros], 2), scale=d**-0.5,
            mask=mx.concatenate([mask, sink[:, :, None, None]], -1))
        got = mx.concatenate([
            mx.fast.scaled_dot_product_attention(
                q[i:i + 1], k[i:i + 1], v[i:i + 1], scale=d**-0.5, mask=mask[i:i + 1],
                sinks=sink[i])
            for i in range(rows)], axis=0)
    diff = mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max().item()
    assert diff <= 2 ** -7 * max(1.0, mx.abs(ref).max().item())  # BF16 output rounding

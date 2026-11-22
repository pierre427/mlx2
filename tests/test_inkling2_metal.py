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


def _reference_attention(q, k, v, rl, sink, *, offset, left_padding, window, scale, extent):
    # The composed path in Attention.__call__: dense mask + sink column + SDPA.
    b, h, length, d = q.shape
    mask = _mask_from_rows(rl, offset, k.shape[2], window, extent, left_padding)
    zeros = mx.zeros((b, k.shape[1], 1, d), k.dtype)
    kk = mx.concatenate([k, zeros], axis=2)
    vv = mx.concatenate([v, zeros], axis=2)
    mask = mx.concatenate([mask, sink.transpose(0, 2, 1)[..., None].astype(mask.dtype)], axis=-1)
    return mx.fast.scaled_dot_product_attention(q, kk, vv, scale=scale, mask=mask)


def _mask_from_rows(rl, offset, size, window, extent, left_padding):
    # Same law as banded_additive_mask, from precomputed bias rows rl [B, L, H, E].
    b, length, h, e = rl.shape
    dist = (mx.arange(length) + offset)[:, None] - mx.arange(size)[None, :]
    gidx = mx.broadcast_to(mx.clip(dist, 0, e - 1)[None, None], (b, h, length, size))
    pb = mx.take_along_axis(rl.transpose(0, 2, 1, 3), gidx, axis=-1)
    pb = mx.where((dist >= extent)[None, None], mx.array(0.0, rl.dtype), pb)
    masked = dist < 0
    if window:
        masked = masked | (dist >= window)
    mask = mx.where(masked[None, None], mx.array(-1e30, rl.dtype), pb)
    if left_padding is not None:
        valid = mx.arange(size)[None, :] >= left_padding[:, None]
        mask = mx.where(valid[:, None, None, :], mask, mx.array(-1e30, rl.dtype))
    return mask.astype(rl.dtype)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("case", [
    dict(b=1, length=70, size=300, window=0, extent=64, pad=None),     # global, offset
    dict(b=2, length=40, size=40, window=0, extent=16, pad=[0, 7]),    # prefill, left pad
    dict(b=1, length=96, size=150, window=32, extent=32, pad=None),    # local window
    dict(b=1, length=8, size=1033, window=1024, extent=1024, pad=None),  # verify rows
])
def test_flash_relative_attention_matches_composed_path(dtype, case):
    from mlx2.runtime.models.inkling2.language import flash_relative_attention

    mx.random.seed(case["size"])
    b, length, size = case["b"], case["length"], case["size"]
    h, hk, d = 24, 8, 128
    q = (mx.random.normal((b, length, h, d)) * 0.3).astype(dtype).transpose(0, 2, 1, 3)
    kv = (mx.random.normal((b, hk, size + 5, d)) * 0.3).astype(dtype)
    k = kv[:, :, 5:]  # a strided view, as the sliding-window slice produces
    v = (mx.random.normal((b, hk, size, d))).astype(dtype)
    rl = (mx.random.normal((b, length, h, case["extent"])) * 2).astype(dtype)
    sink = (mx.random.normal((b, length, h)) * 2).astype(dtype)
    pad = None if case["pad"] is None else mx.array(case["pad"], dtype=mx.int32)
    kw = dict(offset=size - length, left_padding=pad, window=case["window"], scale=d ** -0.5)
    with mx.stream(mx.gpu):
        got = flash_relative_attention(q, k, v, rl, sink, **kw)
        ref = _reference_attention(q, k, v, rl, sink, extent=case["extent"], **kw)
    diff = mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max().item()
    tol = 2e-5 if dtype == mx.float32 else 2 ** -7 * max(1.0, mx.abs(ref).max().item())
    assert got.shape == ref.shape and diff <= tol, diff


def test_flash_prefill_model_parity_on_gpu():
    """Tiny FP32 model: flash prefill vs the composed path, chunked prefill
    with a wrapped window (8) and log scaling engaged."""
    import test_inkling2_cpu as t

    with mx.stream(mx.gpu):
        model = t._tiny_model()
        tokens = mx.random.randint(0, 500, (1, 37))
        base = t._run(model, tokens, [(0, 20), (20, 37)])
        model.configure_kernels(flash_prefill=True)
        flash = t._run(model, tokens, [(0, 20), (20, 37)])
        assert model.kernel_receipt()["flash_prefill_calls"] == 48 * 2
    assert mx.abs(flash - base).max().item() < 1e-3

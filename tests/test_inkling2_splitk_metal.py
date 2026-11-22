"""``inkling2_verify_splitk``: GQA-packed split-K kernel vs the composed path (Metal).

Run under the GPU lock. The reference is ``splitk.reference_attention``: the
dense ``banded_additive_mask`` law plus a zero K/V sink column and one SDPA,
which is what ``Attention.__call__`` runs with the option off. The kernel
reduces in a different order and stages BF16 tiles in FP16, so the check is
max |diff| against a tolerance, as for the flash-prefill kernel.
"""

import mlx.core as mx
import pytest

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")

H, HK, D, EXTENT = 24, 8, 128, 1024  # Inkling-2-Mini geometry (both layer kinds)


def _inputs(b, length, size, dtype, layout, seed, pad=None):
    mx.random.seed(seed)
    q = (mx.random.normal((b, length, H, D)) * 0.3).astype(dtype).transpose(0, 2, 1, 3)
    if layout == "contiguous":
        k = (mx.random.normal((b, HK, size, D)) * 0.3).astype(dtype)
        v = mx.random.normal((b, HK, size, D)).astype(dtype)
    elif layout == "cache_view":
        # A KVCache fetch: a leading-key view of a buffer with spare capacity.
        k = (mx.random.normal((b, HK, size + 200, D)) * 0.3).astype(dtype)[:, :, :size]
        v = mx.random.normal((b, HK, size + 200, D)).astype(dtype)[:, :, :size]
    elif layout == "misaligned_base":
        # Row stride a multiple of 8 but the base 8 bytes off a 16-byte
        # boundary: the alignment guard alone selects the scalar loads.
        k = (mx.random.normal((b, HK, size, D + 8)) * 0.3).astype(dtype)[..., 4:D + 4]
        v = mx.random.normal((b, HK, size, D + 8)).astype(dtype)[..., 4:D + 4]
    else:  # "odd_stride": row stride not a multiple of 8 -> in-kernel scalar loads
        k = (mx.random.normal((b, HK, size, D + 2)) * 0.3).astype(dtype)[..., 1:D + 1]
        v = mx.random.normal((b, HK, size, D + 2)).astype(dtype)[..., :D]
    rel = (mx.random.normal((b, length, H, EXTENT)) * 2).astype(dtype)
    sinks = (mx.random.normal((b, length, H)) * 2).astype(dtype)
    left = None if pad is None else mx.array(pad, dtype=mx.int32)
    return q, k, v, rel, sinks, left


def _check(q, k, v, rel, sinks, left, window, dtype):
    from mlx2.runtime.models.inkling2 import splitk

    size, length = k.shape[2], q.shape[2]
    kw = dict(offset=size - length, left_padding=left, window=window, scale=D ** -0.5)
    with mx.stream(mx.gpu):
        got = splitk.gqa_splitk_attention(q, k, v, rel, sinks, **kw)
        ref = splitk.reference_attention(q, k, v, rel, sinks, **kw)
        mx.eval(got, ref)
    assert got.shape == ref.shape and got.dtype == q.dtype
    diff = mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max().item()
    scale = max(1.0, mx.abs(ref.astype(mx.float32)).max().item())
    tol = 2e-5 * scale if dtype == mx.float32 else 2 ** -7 * scale  # BF16 output rounding
    assert diff <= tol, (diff, tol)
    return diff


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("b", [1, 2])
@pytest.mark.parametrize("window", [0, 1024])
@pytest.mark.parametrize("size", [37, 520, 4096, 40960])
@pytest.mark.parametrize("length", [1, 2, 8, 16])
def test_splitk_matches_composed_path(length, size, window, b, dtype):
    """L in {1, 2, 8, 16} (direct L == 1 kernel past 2048 keys, packed split
    kernel otherwise; 16 rows x G = 3 is two query tiles), one split (37),
    split tails and an empty split (520), multi-split (4K, 40K); global and
    window 1024 over uncropped keys; B2 with left padding on row 1."""
    if size < length:
        pytest.skip("fewer keys than queries")
    pad = None if b == 1 else [0, min(37, size // 3)]
    args = _inputs(b, length, size, dtype, "cache_view", seed=size + length + b, pad=pad)
    _check(*args, window, dtype)


@pytest.mark.parametrize("layout", ["contiguous", "cache_view", "misaligned_base", "odd_stride"])
@pytest.mark.parametrize("length", [1, 8])
def test_splitk_kv_layouts(layout, length):
    """16-byte loads (contiguous, cache view) and the scalar path (odd row
    stride) give the same law."""
    args = _inputs(1, length, 5000, mx.bfloat16, layout, seed=7, pad=None)
    _check(*args, 0, mx.bfloat16)


@pytest.mark.parametrize("length,size", [(1, 300), (1, 3000), (8, 300), (8, 3000)])
def test_splitk_negative_and_full_left_padding(length, size):
    """Negative padding (a merged equal-length batch) means no padding; a
    row padded over every key attends only to the sink (output 0)."""
    args = _inputs(2, length, size, mx.bfloat16, "cache_view", seed=size, pad=[-2, size])
    _check(*args, 0, mx.bfloat16)


def test_splitk_cropped_window_matches_uncropped():
    """The model crops sliding K/V to window + L - 1 keys before attention;
    the kernel's window law gives the same result on either."""
    from mlx2.runtime.models.inkling2 import splitk

    length, size, window = 8, 6000, 1024
    q, k, v, rel, sinks, _ = _inputs(1, length, size, mx.bfloat16, "contiguous", seed=3)
    start = size - length - (window - 1)
    kw = dict(left_padding=None, window=window, scale=D ** -0.5)
    with mx.stream(mx.gpu):
        full = splitk.gqa_splitk_attention(q, k, v, rel, sinks, offset=size - length, **kw)
        crop = splitk.gqa_splitk_attention(q, k[:, :, start:], v[:, :, start:], rel, sinks,
                                           offset=size - start - length, **kw)
    diff = mx.abs(full.astype(mx.float32) - crop.astype(mx.float32)).max().item()
    assert diff <= 2 ** -7 * max(1.0, mx.abs(full.astype(mx.float32)).max().item())


@pytest.mark.parametrize("length", [1, 8])
def test_splitk_peaked_softmax_at_clip_norm(length):
    """Q and K at the L2 clip norm (22.6) times a tau of 1.3: large, peaked
    logits, the worst case for FP16 staging of Q/K and P."""
    q, k, v, rel, sinks, _ = _inputs(1, length, 8192, mx.bfloat16, "contiguous", seed=11)
    unit = lambda a: a / mx.sqrt(mx.sum(mx.square(a.astype(mx.float32)), -1, keepdims=True))
    q = (unit(q) * 22.63 * 1.3).astype(mx.bfloat16)
    k = (unit(k) * 22.63).astype(mx.bfloat16)
    _check(q, k, v, rel * 1.3, sinks, None, 0, mx.bfloat16)


def test_splitk_model_parity_on_gpu():
    """Tiny FP32 model (G = 2, D = 32): verify blocks (8 rows) and one-token
    decode through the kernel vs the composed path, with a wrapped ring
    (window 8) and log scaling; the 20-row prefill chunk engages too."""
    import test_inkling2_cpu as t

    plan = [(0, 20), (20, 28)] + [(i, i + 1) for i in range(28, 36)]
    with mx.stream(mx.gpu):
        model = t._tiny_model()
        tokens = mx.random.randint(0, 500, (1, 36))
        base = t._run(model, tokens, plan)
        model.configure_kernels(verify_splitk=True)
        got = t._run(model, tokens, plan)
        receipt = model.kernel_receipt()
        model.configure_kernels()
    assert receipt["verify_splitk_calls"] == 48 * len(plan)
    assert receipt["verify_splitk_declines"] == {}
    assert mx.abs(got - base).max().item() < 1e-3

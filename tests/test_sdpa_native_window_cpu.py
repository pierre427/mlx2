"""Opt-in native-window bridge (MLX2_SDPA_NATIVE_WINDOW): a pure causal-window
mask built by ``causal_window_mask`` reaches MLX as ``mask="causal",
window_size=W``; every other array keeps the array path."""
import mlx.core as mx
import pytest

from mlx2.runtime.models import base
from mlx2.runtime.models.cache import RotatingKVCache
from mlx2.runtime.technologies import window_attention as wa


@pytest.fixture
def native(monkeypatch):
    monkeypatch.setenv("MLX2_SDPA_NATIVE_WINDOW", "1")
    wa._NATIVE_WINDOW_MASKS.clear()
    yield
    wa._NATIVE_WINDOW_MASKS.clear()


def _qkv(qL, kL, D=64, Hq=4, Hkv=2, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((1, Hq, qL, D))
    k = mx.random.normal((1, Hkv, kL, D))
    v = mx.random.normal((1, Hkv, kL, D))
    return q, k, v


def _spy(monkeypatch):
    calls = []
    real = mx.fast.scaled_dot_product_attention

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", spy)
    return calls


def test_default_off_records_nothing(monkeypatch):
    monkeypatch.delenv("MLX2_SDPA_NATIVE_WINDOW", raising=False)
    wa._NATIVE_WINDOW_MASKS.clear()
    wa.causal_window_mask(16, 8, 4)
    assert not wa._NATIVE_WINDOW_MASKS


@pytest.mark.parametrize("qL,offset,window", [(16, 0, 4), (16, 7, 8), (12, 31, 5), (9, 0, 32)])
def test_bridge_matches_array_path(native, monkeypatch, qL, offset, window):
    q, k, v = _qkv(qL, offset + qL)
    mask = wa.causal_window_mask(qL, offset, window)
    assert wa.native_window_of(mask, qL, offset + qL) == window
    calls = _spy(monkeypatch)
    out = base.fast_sdpa(q, k, v, scale=0.125, mask=mask)
    assert calls[-1]["mask"] == "causal" and calls[-1]["window_size"] == window
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125, mask=mask)
    assert mx.allclose(out, ref, atol=1e-5).item()


def test_bridge_with_sinks(native):
    q, k, v = _qkv(16, 24, Hq=4)
    sinks = mx.array([0.5, -1.0, 2.0, 0.0])
    mask = wa.causal_window_mask(16, 8, 6)
    out = base.fast_sdpa(q, k, v, scale=0.125, mask=mask, sinks=sinks)
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125, mask=mask, sinks=sinks)
    assert mx.allclose(out, ref, atol=1e-5).item()


def test_unrecorded_or_mismatched_masks_keep_array(native, monkeypatch):
    calls = _spy(monkeypatch)
    q, k, v = _qkv(16, 24)
    mask = wa.causal_window_mask(16, 8, 6)
    # A derived array (tail slice, lift) is a different object: not bridged.
    base.fast_sdpa(q[:, :, -1:], k, v, scale=0.125, mask=mask[-1:])
    assert not isinstance(calls[-1]["mask"], str)
    base.fast_sdpa(q, k, v, scale=0.125, mask=mask[None, None])
    assert not isinstance(calls[-1]["mask"], str)
    # Same object, wrong key length (keys not ending at the newest query).
    q2, k2, v2 = _qkv(16, 30)
    assert wa.native_window_of(mask, 16, 30) is None
    # Padded masks and full causal masks are never recorded.
    padded = wa.causal_window_mask(16, 8, 6, left_padding=mx.array([0, 3]))
    full = wa.causal_window_mask(16, 8, None)
    assert wa.native_window_of(padded, 16, 24) is None
    assert wa.native_window_of(full, 16, 24) is None


def test_batched_rows_gqa_and_decode_row(native, monkeypatch):
    calls = _spy(monkeypatch)
    for qL, offset in ((16, 8), (1, 20)):
        mx.random.seed(3)
        q = mx.random.normal((3, 8, qL, 32))
        k = mx.random.normal((3, 2, offset + qL, 32))
        v = mx.random.normal((3, 2, offset + qL, 32))
        mask = wa.causal_window_mask(qL, offset, 5)
        out = base.fast_sdpa(q, k, v, scale=0.2, mask=mask)
        assert calls[-1]["mask"] == "causal"
        ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.2, mask=mask)
        assert mx.allclose(out, ref, atol=1e-5).item()


def test_nonpositive_window_is_not_recorded(native):
    mask = wa.causal_window_mask(8, 0, 0)
    assert wa.native_window_of(mask, 8, 8) is None


def test_flag_turned_off_clears_registry(native, monkeypatch):
    mask = wa.causal_window_mask(8, 4, 3)
    assert wa.native_window_of(mask, 8, 12) == 3
    monkeypatch.setenv("MLX2_SDPA_NATIVE_WINDOW", "0")
    assert wa.native_window_of(mask, 8, 12) is None
    assert not wa._NATIVE_WINDOW_MASKS


def test_registry_is_bounded(native):
    keep = [wa.causal_window_mask(4, i, 2) for i in range(3 * wa._NATIVE_WINDOW_LIMIT)]
    assert len(wa._NATIVE_WINDOW_MASKS) <= wa._NATIVE_WINDOW_LIMIT
    assert wa.native_window_of(keep[-1], 4, 4 + 3 * wa._NATIVE_WINDOW_LIMIT - 1) == 2


def test_refusals_fall_back_to_the_array(native, monkeypatch):
    q, k, v = _qkv(16, 24)
    mask = wa.causal_window_mask(16, 8, 6)
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125, mask=mask)
    real = mx.fast.scaled_dot_product_attention

    def refuse(error):
        def fake(*args, **kwargs):
            if "window_size" in kwargs:
                raise error("refused")
            return real(*args, **kwargs)
        return fake

    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", refuse(ValueError))
    assert mx.allclose(base.fast_sdpa(q, k, v, scale=0.125, mask=mask), ref).item()
    assert wa.native_window_requested()  # a geometry refusal keeps the bridge on
    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", refuse(TypeError))
    try:
        assert mx.allclose(base.fast_sdpa(q, k, v, scale=0.125, mask=mask), ref).item()
        assert not wa.native_window_requested() and not wa._NATIVE_WINDOW_MASKS
    finally:
        wa._NATIVE_WINDOW_SUPPORTED[0] = True


def test_rotating_cache_prefill_chunk_is_bridged(native, monkeypatch):
    """The RotatingKVCache chunk geometry (keys = window - 1 + N, temporal
    order) is what the bridge serves; outputs equal the array path."""
    W, N = 8, 12
    cache = RotatingKVCache(max_size=W, keep=0)
    calls = _spy(monkeypatch)
    for step in range(3):
        q, k, v = _qkv(N, N, seed=step)
        mask = cache.make_mask(N, window_size=W, return_array=step > 0)
        keys, values = cache.update_and_fetch(k, v)
        out = base.fast_sdpa(q, keys, values, scale=0.125, mask=mask)
        if isinstance(mask, mx.array):
            assert calls[-1]["mask"] == "causal" and calls[-1]["window_size"] == W
            ref = mx.fast.scaled_dot_product_attention(q, keys, values, scale=0.125, mask=mask)
            assert mx.allclose(out, ref, atol=1e-5).item()

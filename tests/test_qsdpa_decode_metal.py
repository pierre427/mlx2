"""GQA-aware quantized decode kernel: gating on CPU, equivalence on Metal.

The Metal cases run only with MLX2_RUN_GPU_TESTS=1 (while holding the GPU
lease); routine test runs are CPU-only and exercise the gating logic.
"""

import os

import mlx.core as mx
import pytest

import mlx2.runtime.models.qsdpa_decode_metal as qdm

GPU = os.environ.get("MLX2_RUN_GPU_TESTS") == "1"


def _quantized(Hq, Hkv, T, D=256, kb=8, vb=8, B=1, dtype=mx.bfloat16, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((B, Hq, 1, D)).astype(dtype)
    k = mx.random.normal((B, Hkv, T, D)).astype(dtype)
    v = mx.random.normal((B, Hkv, T, D)).astype(dtype)
    return q, mx.quantize(k, group_size=64, bits=kb), mx.quantize(v, group_size=64, bits=vb)


@pytest.fixture
def pretend_gpu(monkeypatch):
    monkeypatch.setattr(qdm.mx, "default_device", lambda: mx.gpu)


def test_config_policy():
    assert qdm.decode_kernel_config(131072, 8) == (4, 2, 256)
    assert qdm.decode_kernel_config(32768, 8) == (4, 2, 128)
    assert qdm.decode_kernel_config(131072, 6) == (2, 1, 64)
    assert qdm.decode_kernel_config(32768, 4) == (2, 1, 64)


def test_supported_shapes(pretend_gpu):
    q, qk, qv = _quantized(16, 2, 64)
    kw = dict(group_size=64, key_bits=8, value_bits=8)
    assert qdm.decode_attention_supported(q, qk, qv, **kw)
    assert not qdm.decode_attention_supported(q, qk, qv, mask=mx.ones((1, 1, 1, 64), dtype=mx.bool_), **kw)
    assert not qdm.decode_attention_supported(q, qk, qv, group_size=32, key_bits=8, value_bits=8)
    q3 = mx.zeros((1, 16, 3, 256), dtype=mx.bfloat16)
    assert not qdm.decode_attention_supported(q3, qk, qv, **kw)  # verify rows
    q12, qk12, qv12 = _quantized(24, 2, 64)
    assert not qdm.decode_attention_supported(q12, qk12, qv12, **kw)  # GQA 12 > 8
    q128, qk128, qv128 = _quantized(16, 2, 64, D=128, kb=4, vb=4)
    assert not qdm.decode_attention_supported(q128, qk128, qv128, group_size=64, key_bits=4, value_bits=4)
    q32, qk32, qv32 = _quantized(16, 2, 64, dtype=mx.float32)
    assert not qdm.decode_attention_supported(q32, qk32, qv32, **kw)


def test_supported_is_false_on_cpu():
    q, qk, qv = _quantized(16, 2, 64)
    assert not qdm.decode_attention_supported(q, qk, qv, group_size=64, key_bits=8, value_bits=8)


def test_threshold_and_kill_switch(pretend_gpu, monkeypatch):
    kw = dict(group_size=64, key_bits=8, value_bits=8)
    short = _quantized(16, 2, qdm.MIN_CONTEXT - 1)
    long = _quantized(16, 2, qdm.MIN_CONTEXT)
    assert not qdm.use_decode_kernel(*short, **kw)
    assert qdm.use_decode_kernel(*long, **kw)
    monkeypatch.setenv("MLX2_QSDPA_DECODE_KERNEL", "0")
    assert not qdm.use_decode_kernel(*long, **kw)


def test_composed_path_is_used_on_cpu():
    from mlx2.runtime.models.base import quantized_scaled_dot_product_attention

    q, qk, qv = _quantized(16, 2, qdm.MIN_CONTEXT, D=128)
    out = quantized_scaled_dot_product_attention(q, qk, qv, scale=0.1, mask=None, group_size=64, bits=8)
    assert out.shape == (1, 16, 1, 128)


# --- Metal ----------------------------------------------------------------

metal = pytest.mark.skipif(not GPU, reason="Metal kernel test; set MLX2_RUN_GPU_TESTS=1")


def _strided(Hkv, T, D, kb, vb, B, dtype=mx.bfloat16, seed=3):
    """Quantized caches as views of 256-row-step storage, like a live cache."""
    from mlx2.runtime.models.cache import KVCache

    mx.random.seed(seed)
    caches = []
    for _ in range(B):
        c = KVCache()
        for s in range(0, T, 2048):
            n = min(2048, T - s)
            c.update_and_fetch(mx.random.normal((1, Hkv, n, D)).astype(dtype),
                               mx.random.normal((1, Hkv, n, D)).astype(dtype))
        caches.append(c.to_quantized(group_size=64, key_bits=kb, value_bits=vb))
    if B == 1:
        return caches[0].keys_and_values()
    parts = [c.keys_and_values() for c in caches]
    return tuple(tuple(mx.concatenate([p[i][j] for p in parts], axis=0) for j in range(3)) for i in range(2))


@metal
@pytest.mark.parametrize("Hq,Hkv", [(16, 2), (24, 4), (16, 4), (8, 8)])
@pytest.mark.parametrize("bits", [(8, 8), (8, 4), (4, 4)])
@pytest.mark.parametrize("T", [16421, 70001])
def test_kernel_matches_composed(Hq, Hkv, bits, T):
    from mlx2.runtime.models import base

    mx.set_default_device(mx.gpu)
    try:
        qk, qv = _strided(Hkv, T, 256, *bits, B=1)
        q = mx.random.normal((1, Hq, 1, 256)).astype(mx.bfloat16)
        got = qdm.gqa_quantized_decode_attention(q, qk, qv, scale=0.0625, key_bits=bits[0], value_bits=bits[1])
        # the composed path, forced by the kill switch
        os.environ["MLX2_QSDPA_DECODE_KERNEL"] = "0"
        want = base.quantized_scaled_dot_product_attention(
            q, qk, qv, scale=0.0625, mask=None, group_size=64, key_bits=bits[0], value_bits=bits[1])
        del os.environ["MLX2_QSDPA_DECODE_KERNEL"]
        routed = base.quantized_scaled_dot_product_attention(
            q, qk, qv, scale=0.0625, mask=None, group_size=64, key_bits=bits[0], value_bits=bits[1])
        err = mx.abs(got.astype(mx.float32) - want.astype(mx.float32)).max().item()
        assert err < 5e-3
        assert mx.array_equal(routed, got).item()  # serving path took the kernel
    finally:
        os.environ.pop("MLX2_QSDPA_DECODE_KERNEL", None)
        mx.set_default_device(mx.cpu)


@metal
def test_kernel_batch_and_head_dim_128():
    mx.set_default_device(mx.gpu)
    try:
        from mlx2.runtime.models import base

        for D, B in ((256, 2), (128, 1), (128, 2)):
            qk, qv = _strided(2, 20000, D, 8, 8, B=B)
            q = mx.random.normal((B, 16, 1, D)).astype(mx.bfloat16)
            got = qdm.gqa_quantized_decode_attention(q, qk, qv, scale=0.1)
            os.environ["MLX2_QSDPA_DECODE_KERNEL"] = "0"
            want = base.quantized_scaled_dot_product_attention(q, qk, qv, scale=0.1, mask=None, group_size=64, bits=8)
            os.environ.pop("MLX2_QSDPA_DECODE_KERNEL")
            assert mx.abs(got.astype(mx.float32) - want.astype(mx.float32)).max().item() < 5e-3
    finally:
        os.environ.pop("MLX2_QSDPA_DECODE_KERNEL", None)
        mx.set_default_device(mx.cpu)


# --- unquantized (fp16/bf16) tiled kernel: opt-in ------------------------------


def test_fp_kernel_is_opt_in(pretend_gpu, monkeypatch):
    q = mx.zeros((1, 16, 1, 256), dtype=mx.bfloat16)
    k = mx.zeros((1, 2, qdm.FP_MIN_CONTEXT, 256), dtype=mx.bfloat16)
    monkeypatch.delenv("MLX2_FP_DECODE_KERNEL", raising=False)
    assert not qdm.use_fp_decode_kernel(q, k, k)
    monkeypatch.setenv("MLX2_FP_DECODE_KERNEL", "1")
    assert qdm.use_fp_decode_kernel(q, k, k)
    short = mx.zeros((1, 2, qdm.FP_MIN_CONTEXT - 1, 256), dtype=mx.bfloat16)
    assert not qdm.use_fp_decode_kernel(q, short, short)
    assert not qdm.use_fp_decode_kernel(q, k, k, mask=mx.ones((1, 1, 1, qdm.FP_MIN_CONTEXT), dtype=mx.bool_))
    assert not qdm.use_fp_decode_kernel(mx.zeros((1, 16, 2, 256), dtype=mx.bfloat16), k, k)


@metal
@pytest.mark.parametrize("Hq,Hkv", [(16, 2), (24, 4), (16, 4), (8, 8)])
def test_fp_kernel_matches_sdpa(Hq, Hkv, monkeypatch):
    from mlx2.runtime.models import base
    from mlx2.runtime.models.cache import KVCache

    mx.set_default_device(mx.gpu)
    try:
        mx.random.seed(5)
        c = KVCache()
        for s in range(0, qdm.FP_MIN_CONTEXT + 37, 4096):
            n = min(4096, qdm.FP_MIN_CONTEXT + 37 - s)
            c.update_and_fetch(mx.random.normal((1, Hkv, n, 256)).astype(mx.bfloat16),
                               mx.random.normal((1, Hkv, n, 256)).astype(mx.bfloat16))
        k, v = c.keys_and_values()
        q = mx.random.normal((1, Hq, 1, 256)).astype(mx.bfloat16)
        want = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.0625)
        got = qdm.gqa_decode_attention_fp(q, k, v, scale=0.0625)
        assert mx.abs(got.astype(mx.float32) - want.astype(mx.float32)).max().item() < 5e-3
        monkeypatch.setenv("MLX2_FP_DECODE_KERNEL", "1")
        routed = base.scaled_dot_product_attention(q, k, v, cache=c, scale=0.0625, mask=None)
        assert mx.array_equal(routed, got).item()
        monkeypatch.delenv("MLX2_FP_DECODE_KERNEL")
        assert mx.array_equal(base.scaled_dot_product_attention(q, k, v, cache=c, scale=0.0625, mask=None), want).item()
    finally:
        mx.set_default_device(mx.cpu)

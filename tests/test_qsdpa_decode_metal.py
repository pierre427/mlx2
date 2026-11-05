"""GQA-aware quantized decode kernel: gating on CPU, equivalence on Metal.

The Metal cases run only with MLX2_RUN_GPU_TESTS=1 (while holding the GPU
lease); routine test runs are CPU-only and exercise the gating logic.
"""

import os
from types import SimpleNamespace

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
    assert qdm.use_fp_decode_kernel(q, k, k, mask=mx.ones((1, 1, 1, qdm.FP_MIN_CONTEXT), dtype=mx.bool_))
    assert not qdm.use_fp_decode_kernel(q, k, k, mask=mx.zeros((1, 1, 1, qdm.FP_MIN_CONTEXT), dtype=mx.float32))
    assert not qdm.use_fp_decode_kernel(mx.zeros((1, 16, 2, 256), dtype=mx.bfloat16), k, k)


def _shape(shape, dtype=mx.bfloat16):
    return SimpleNamespace(shape=shape, ndim=len(shape), dtype=dtype)


def test_fp_muse_geometry_and_selection(pretend_gpu, monkeypatch):
    q = _shape((1, 32, 1, 128))
    k = _shape((1, 2, qdm.FP_MIN_CONTEXT, 128))
    assert qdm.fp_decode_attention_supported(q, k, k)
    monkeypatch.delenv("MLX2_FP_DECODE_KERNEL", raising=False)
    assert not qdm.use_fp_decode_kernel(q, k, k)
    monkeypatch.setenv("MLX2_FP_DECODE_KERNEL", "1")
    assert qdm.use_fp_decode_kernel(q, k, k)
    local = _shape((1, 2, 2048, 128))
    assert qdm.fp_decode_attention_supported(q, local, local)
    assert not qdm.use_fp_decode_kernel(q, local, local)
    assert qdm.fp_decode_kernel_config(65536, 16) == (4, 2, 256)
    assert qdm.fp_decode_kernel_config(2048, 16) == (4, 2, 128)
    assert qdm.fp_decode_kernel_config(65536, 8) == qdm.decode_kernel_config(65536, 8)


def test_fp_decode_law_binds_apcv2_namespace():
    from mlx2.runtime.apc_numerics import execution_numerics_identity, apc_execution_fingerprint

    ordinary = execution_numerics_identity({})
    assert ordinary is None
    assert execution_numerics_identity({"MLX2_FP_DECODE_KERNEL": "true"}) is None
    selected = execution_numerics_identity({"MLX2_FP_DECODE_KERNEL": "1"})
    assert selected["fp_decode_attention"]["version"] == qdm.FP_KERNEL_VERSION
    assert apc_execution_fingerprint("prefix", selected) != apc_execution_fingerprint("prefix", ordinary)


@pytest.mark.parametrize("q,k,v", [
    ((1, 32, 128), (1, 2, 64, 128), (1, 2, 64, 128)),
    ((1, 32, 1, 128), (1, 0, 64, 128), (1, 0, 64, 128)),
    ((1, 32, 1, 128), (1, 2, 0, 128), (1, 2, 0, 128)),
    ((2, 32, 1, 128), (1, 2, 64, 128), (1, 2, 64, 128)),
    ((1, 32, 1, 128), (1, 2, 64, 256), (1, 2, 64, 256)),
    ((1, 32, 1, 128), (1, 2, 64, 128), (1, 2, 63, 128)),
    ((1, 24, 1, 128), (1, 2, 64, 128), (1, 2, 64, 128)),
    ((1, 64, 1, 128), (1, 2, 64, 128), (1, 2, 64, 128)),
])
def test_fp_invalid_geometry_fails_closed(pretend_gpu, q, k, v):
    arrays = tuple(_shape(s) for s in (q, k, v))
    assert not qdm.fp_decode_attention_supported(*arrays)
    with pytest.raises(ValueError, match="geometry"):
        qdm.gqa_decode_attention_fp(*arrays, scale=0.1)


@pytest.mark.parametrize("launch", [
    {"blocks": 0}, {"blocks": -1}, {"simdgroups": True},
    {"simdgroups": 33}, {"head_split": 3}, {"head_split": 0},
    {"simdgroups": 32, "head_split": 1},
])
def test_fp_invalid_launch_refused_before_dispatch(pretend_gpu, launch):
    with pytest.raises(ValueError):
        qdm.gqa_decode_attention_fp(_shape((1, 32, 1, 128)),
            _shape((1, 2, 64, 128)), _shape((1, 2, 64, 128)), scale=0.1, **launch)


@metal
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("D,B,T", [(128, 1, 1), (128, 2, 2049),
                                  (128, 1, 65573), (256, 2, 65573)])
def test_fp_gqa16_strided_matches_sdpa(dtype, D, B, T, monkeypatch):
    from mlx2.runtime.models.base import scaled_dot_product_attention

    mx.set_default_device(mx.gpu)
    try:
        mx.random.seed(19)
        # Padding in batch/head/sequence/dimension axes exercises real cache
        # views and the non-unit inner strides the old FP kernel ignored.
        q = mx.random.normal((B + 1, 64, 1, D * 2)).astype(dtype)[1:, ::2, :, ::2]
        k = mx.random.normal((B + 1, 4, T + 7, D * 2)).astype(dtype)[1:, ::2, :T, ::2]
        v = mx.random.normal((B + 1, 4, T + 7, D * 2)).astype(dtype)[1:, ::2, :T, ::2]
        scale = D ** -0.5
        want = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        before = qdm.STATS["fp_decode_gqa16_calls"]
        got = qdm.gqa_decode_attention_fp(q, k, v, scale=scale)
        assert mx.all(mx.isfinite(got)).item()
        assert mx.abs(got.astype(mx.float32) - want.astype(mx.float32)).max().item() < 5e-3
        assert qdm.STATS["fp_decode_gqa16_calls"] == before + 1
        monkeypatch.setenv("MLX2_FP_DECODE_KERNEL", "1")
        routed = scaled_dot_product_attention(q, k, v, cache=None, scale=scale, mask=None)
        if T >= qdm.FP_MIN_CONTEXT:
            assert qdm.STATS["fp_decode_gqa16_calls"] == before + 2
            assert mx.array_equal(routed, got).item()
        else:
            assert qdm.STATS["fp_decode_gqa16_calls"] == before + 1
            assert mx.array_equal(routed, want).item()
    finally:
        mx.set_default_device(mx.cpu)


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


@metal
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("gqa", [8, 16])
def test_fp_boolean_padding_and_per_head_mask_matches_sdpa(dtype, gqa, monkeypatch):
    from mlx2.runtime.models.base import scaled_dot_product_attention

    mx.set_default_device(mx.gpu)
    try:
        mx.random.seed(29)
        n, heads = 65573, 2 * gqa
        q = mx.random.normal((2, heads, 1, 128)).astype(dtype)
        k = mx.random.normal((2, 2, n + 7, 128)).astype(dtype)[:, :, :n]
        v = mx.random.normal((2, 2, n + 7, 128)).astype(dtype)[:, :, :n]
        # A real serving padding mask broadcasts across query heads. Also
        # exercise individual heads and an entirely masked head (zero output).
        for mask in ((mx.arange(n)[None, None, None] >= mx.array([0, 1007])[:, None, None, None]),
                     (mx.arange(n)[None, None, None] >= mx.arange(heads)[None, :, None, None] * 17)):
            expected = mx.fast.scaled_dot_product_attention(q, k, v, scale=128 ** -.5, mask=mask)
            got = qdm.gqa_decode_attention_fp(q, k, v, scale=128 ** -.5, mask=mask)
            assert mx.all(mx.isfinite(got)).item()
            assert mx.max(mx.abs(got.astype(mx.float32) - expected.astype(mx.float32))).item() < .005
            monkeypatch.setenv("MLX2_FP_DECODE_KERNEL", "1")
            routed = scaled_dot_product_attention(q, k, v, cache=None, scale=128 ** -.5, mask=mask)
            assert mx.array_equal(routed, got).item()
        all_masked = mx.zeros((2, 1, 1, n), dtype=mx.bool_)
        got = qdm.gqa_decode_attention_fp(q, k, v, scale=128 ** -.5, mask=all_masked)
        assert mx.all(got == 0).item()
    finally:
        mx.set_default_device(mx.cpu)

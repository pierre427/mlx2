"""Head-dim 512 decode kernel (TK 2): gating on CPU, bitwise stock parity on Metal.

The Metal cases run only with MLX2_RUN_GPU_TESTS=1 (while holding the GPU
lease); routine test runs are CPU-only and exercise the gating logic.
"""

import os

import mlx.core as mx
import pytest

import mlx2.runtime.models.base as base
import mlx2.runtime.models.sdpa_vector_d512 as d512

GPU = os.environ.get("MLX2_RUN_GPU_TESTS") == "1"


def _qkv(Hq, Hkv, N, B=1, D=512, dtype=mx.bfloat16, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((B, Hq, 1, D)).astype(dtype)
    k = mx.random.normal((B, Hkv, N, D)).astype(dtype)
    v = mx.random.normal((B, Hkv, N, D)).astype(dtype)
    return q, k, v


@pytest.fixture
def pretend_gpu(monkeypatch):
    monkeypatch.setattr(d512.mx, "default_device", lambda: mx.gpu)
    monkeypatch.setattr(d512, "device_class", lambda: "s")
    # The gating cases model MLX 0.32.5 (no #4941); the guard has its own test.
    monkeypatch.setattr(d512, "upstream_tk2", lambda: False)


def test_declines_when_upstream_has_4941(pretend_gpu, monkeypatch):
    q, k, v = _qkv(32, 4, 2048)
    monkeypatch.setenv("MLX2_SDPA_D512_TK2", "1")
    assert d512.use_kernel(q, k, v)
    monkeypatch.setattr(d512, "upstream_tk2", lambda: True)
    assert d512.supported(q, k, v)
    assert not d512.use_kernel(q, k, v)
    status = d512.status()
    assert status["upstream_tk2_4941"] is True and "4941" in status["declined"]
    monkeypatch.setenv("MLX2_SDPA_D512_TK2", "0")
    assert "declined" not in d512.status()


def test_stock_blocks_copies_mlx_0324():
    assert d512.stock_blocks(4096, 8, "s") == 128
    assert d512.stock_blocks(8192, 8, "s") == 128
    assert d512.stock_blocks(16384, 8, "s") == 256
    assert d512.stock_blocks(32768, 8, "s") == 256
    assert d512.stock_blocks(65536, 8, "s") == 512
    assert d512.stock_blocks(131072, 8, "s") == 1024
    assert d512.stock_blocks(131072, 4, "s") == 64
    assert d512.stock_blocks(1024, 8, "s") == 64
    assert d512.stock_blocks(70000, 8, "d") == 1024
    assert d512.stock_blocks(20000, 2, "d") == 256
    assert d512.stock_blocks(20000, 8, "g") == 64


def test_stock_blocks_env_override(monkeypatch):
    monkeypatch.setenv("MLX_SDPA_BLOCKS", "100")
    assert d512.stock_blocks(131072, 8, "s") == 128


def test_two_pass_route_copies_mlx_0324():
    assert d512.stock_two_pass(1024, 1, "s") and not d512.stock_two_pass(1023, 8, "s")
    assert d512.stock_two_pass(4096, 8, "g") and not d512.stock_two_pass(4096, 1, "g")


def test_supported_geometry(pretend_gpu):
    q, k, v = _qkv(32, 4, 2048)
    assert d512.supported(q, k, v)
    assert d512.supported(q, k, v, mask="causal")
    assert not d512.supported(q, k, v, mask=mx.ones((1, 1, 1, 2048), dtype=mx.bool_))
    assert not d512.supported(q, k, v, sinks=mx.zeros((32,)))
    assert not d512.supported(q, k[..., :256], v[..., :256])
    assert not d512.supported(mx.concatenate([q, q], axis=2), k, v)
    assert not d512.supported(q, k[:, :, :512], v[:, :, :512])  # stock uses one pass
    assert not d512.supported(q.astype(mx.float16), k, v)
    # Stock MLX sends D=512 at any GQA but 8 to the unfused path.
    for hq, hkv in ((8, 8), (16, 4), (32, 2)):
        assert not d512.supported(*_qkv(hq, hkv, 2048))
    assert not d512.supported(q, k, v, mask="unknown")
    # Strided cache views (a preallocated buffer sliced to the offset) are taken.
    assert d512.supported(q, k[:, :, :1500], v[:, :, :1500])


def test_min_kl_env(pretend_gpu, monkeypatch):
    q, k, v = _qkv(32, 4, 2048)
    monkeypatch.setenv("MLX_SDPA_D512_MIN_KL", "4096")
    assert not d512.supported(q, k, v)
    monkeypatch.setenv("MLX_SDPA_D512_MIN_KL", "0")
    assert d512.supported(*_qkv(16, 4, 2048))


def test_off_by_default(pretend_gpu, monkeypatch):
    q, k, v = _qkv(32, 4, 2048)
    monkeypatch.delenv("MLX2_SDPA_D512_TK2", raising=False)
    assert not d512.use_kernel(q, k, v)
    monkeypatch.setenv("MLX2_SDPA_D512_TK2", "1")
    assert d512.use_kernel(q, k, v)
    assert d512.status()["selected"] is True


def test_base_dispatch_takes_kernel_when_selected(pretend_gpu, monkeypatch):
    q, k, v = _qkv(16, 2, 2048)
    monkeypatch.setenv("MLX2_SDPA_D512_TK2", "1")
    calls = []
    monkeypatch.setattr(d512, "attention", lambda *a, **kw: calls.append(kw) or "kernel")
    assert base.scaled_dot_product_attention(q, k, v, cache=None, scale=1.0, mask=None) == "kernel"
    assert calls == [{"scale": 1.0}]


gpu = pytest.mark.skipif(not GPU, reason="Metal case; set MLX2_RUN_GPU_TESTS=1 under the GPU lease")


@gpu
@pytest.mark.parametrize("Hq,Hkv", [(32, 4), (16, 2)])
@pytest.mark.parametrize("N", [1024, 5001, 16384, 40000, 70001])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_bitwise_stock_parity(Hq, Hkv, N, dtype):
    with mx.stream(mx.gpu):
        q, k, v = _qkv(Hq, Hkv, N, dtype=dtype)
        stock = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0)
        ours = d512.attention(q, k, v, scale=1.0)
        assert ours.shape == stock.shape and ours.dtype == stock.dtype
        assert bool(mx.array_equal(ours, stock).item())


@gpu
@pytest.mark.parametrize("B", [1, 2])
def test_bitwise_stock_parity_strided_views(B):
    """A preallocated cache sliced to its offset, as KVCache.update_and_fetch returns."""
    with mx.stream(mx.gpu):
        q, k, v = _qkv(32, 4, 24576, B=B, seed=3)
        k, v = k[:, :, :20000], v[:, :, :20000]
        qt = mx.random.normal((B, 1, 32, 512)).astype(mx.bfloat16).transpose(0, 2, 1, 3)
        for query in (q, qt):
            stock = mx.fast.scaled_dot_product_attention(query, k, v, scale=0.0625)
            ours = d512.attention(query, k, v, scale=0.0625)
            assert bool(mx.array_equal(ours, stock).item())


@gpu
def test_tk_does_not_change_result():
    with mx.stream(mx.gpu):
        q, k, v = _qkv(32, 4, 33333, seed=5)
        ref = d512.attention(q, k, v, scale=1.0, tk=4)
        for tk in (1, 2):
            assert bool(mx.array_equal(d512.attention(q, k, v, scale=1.0, tk=tk), ref).item())

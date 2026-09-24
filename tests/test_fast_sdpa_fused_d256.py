"""fast_sdpa widens MLX's fused head_dim-256 causal range to 256..1023 rows.

CPU: the eligibility rules, and a clean fallback where no fused kernel
exists. Metal (MLX2_RUN_GPU_TESTS=1): the forced kernel actually runs and
matches MLX's default path.
"""
import os

import mlx.core as mx
import pytest

from mlx2.runtime.models import base

GPU = os.environ.get("MLX2_RUN_GPU_TESTS") == "1"
metal = pytest.mark.skipif(not GPU, reason="Metal test; set MLX2_RUN_GPU_TESTS=1")


def _qkv(L, *, D=256, offset=64, hq=4, hkv=2, dtype=mx.bfloat16, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((1, hq, L, D)).astype(dtype)
    k = (mx.random.normal((1, hkv, offset + L, D)) * 0.5).astype(dtype)
    v = mx.random.normal((1, hkv, offset + L, D)).astype(dtype)
    return q, k, v


class _Spy:
    def __init__(self, monkeypatch):
        self.forced = []
        real = mx.fast.scaled_dot_product_attention

        def spy(*args, **kwargs):
            self.forced.append(bool(kwargs.get("force_fused")))
            return real(*args, **kwargs)

        monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", spy)


@pytest.mark.parametrize(
    "L,D,mask,dtype,sinks,expect",
    [
        (256, 256, "causal", mx.bfloat16, None, True),
        (1023, 256, "causal", mx.float16, None, True),
        (255, 256, "causal", mx.bfloat16, None, False),
        (1024, 256, "causal", mx.bfloat16, None, False),  # MLX already fuses
        (512, 128, "causal", mx.bfloat16, None, False),
        (512, 256, None, mx.bfloat16, None, False),
        (512, 256, "causal", mx.float32, None, False),
        (512, 256, "causal", mx.bfloat16, "sinks", False),
    ],
)
def test_eligibility(monkeypatch, L, D, mask, dtype, sinks, expect):
    monkeypatch.setattr(base, "_FUSED_D256_UNAVAILABLE", set())
    spy = _Spy(monkeypatch)
    q = mx.zeros((1, 4, L, D), dtype)
    k = mx.zeros((1, 2, L + 8, D), dtype)
    v = mx.zeros((1, 2, L + 8, D), dtype)
    s = mx.zeros((4,), dtype) if sinks else None
    base.fast_sdpa(q, k, v, scale=0.1, mask=mask, sinks=s)
    assert spy.forced[0] is expect


def test_env_zero_disables(monkeypatch):
    monkeypatch.setattr(base, "_FUSED_D256_MIN_L", 0)
    spy = _Spy(monkeypatch)
    q, k, v = _qkv(512)
    base.fast_sdpa(q, k, v, scale=0.0625, mask="causal")
    assert spy.forced == [False]


def test_cpu_falls_back_once_and_matches(monkeypatch):
    monkeypatch.setattr(base, "_FUSED_D256_UNAVAILABLE", set())
    with mx.stream(mx.cpu):
        q, k, v = _qkv(300)
        spy = _Spy(monkeypatch)
        got = base.fast_sdpa(q, k, v, scale=0.0625, mask="causal")
        want = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.0625, mask="causal")
        assert mx.array_equal(got, want).item()
        if spy.forced[0]:  # the CPU refused the fused kernel: remembered
            base.fast_sdpa(q, k, v, scale=0.0625, mask="causal")
            assert spy.forced[-1] is False


@metal
@pytest.mark.parametrize("L,offset", [(256, 0), (512, 4096), (768, 777)])
def test_metal_forced_kernel_runs_and_matches(monkeypatch, L, offset):
    monkeypatch.setattr(base, "_FUSED_D256_UNAVAILABLE", set())
    with mx.stream(mx.gpu):  # conftest makes the CPU the default device
        q, k, v = _qkv(L, offset=offset, hq=16, hkv=2)
        got = base.fast_sdpa(q, k, v, scale=0.0625, mask="causal")
        assert not base._FUSED_D256_UNAVAILABLE, "fused kernel was refused on Metal"
        want = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.0625, mask="causal")
        f32 = [x.astype(mx.float32) for x in (q, k, v)]
        ref = mx.fast.scaled_dot_product_attention(*f32, scale=0.0625, mask="causal")

        def err(x):
            return mx.max(mx.abs(x.astype(mx.float32) - ref)).item()

        # Both paths round to bf16; the forced kernel must be no less exact.
        assert err(got) <= 1.5 * err(want) + 1e-3, (err(got), err(want))

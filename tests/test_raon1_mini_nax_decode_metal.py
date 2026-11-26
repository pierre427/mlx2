"""Metal canary: head-group-generic absorbed NAX latent flash at Raon's 40 heads.

Opt in with MLX2_RUN_METAL_TESTS=1 on an Apple M5 (the NAX kernels' gate).
Kernel vs ``absorbed_reference`` (FP32 ops law) within Kimi's 2e-2 relative
bound, and Kimi's 32-head kernel unchanged on the same operands.
"""
import os

import mlx.core as mx
import pytest

from mlx2.runtime.models import kimi_linear_mla_flash as fl
from mlx2.runtime.models import raon1_mini as R

pytestmark = pytest.mark.skipif(
    os.environ.get("MLX2_RUN_METAL_TESTS") != "1" or not mx.metal.is_available(),
    reason="explicit Metal oracle (MLX2_RUN_METAL_TESTS=1)",
)


@pytest.fixture(autouse=True)
def gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    if not fl._device_is_m5():
        mx.set_default_device(previous)
        pytest.skip("NAX kernels need an M5")
    yield
    mx.set_default_device(previous)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_canary_40_heads_matches_reference(dtype):
    result = R.absorbed_flash_canary(40, dtype)
    assert result["ok"], result


@pytest.mark.parametrize("heads", [40, 48, 16])
def test_grouped_kernel_padded_heads_and_batch_lanes(heads):
    key = mx.random.key(7)
    k1, k2, k3, k4 = mx.random.split(key, 4)
    b, rows, width, cap = 2, 3, 3000, 3072
    q = (mx.random.normal((b, heads, rows, fl.LATENT), key=k1) * 0.05).astype(mx.bfloat16)
    qs = (mx.random.normal((b, heads, rows, fl.SHARED), key=k4) * 0.5).astype(mx.bfloat16)
    lat = mx.random.normal((b, 1, cap, fl.LATENT), key=k2).astype(mx.bfloat16)
    sh = mx.random.normal((b, 1, cap, fl.SHARED), key=k3).astype(mx.bfloat16)
    starts = mx.array([0, 777], mx.int32)
    scale = (fl.NOPE + fl.SHARED) ** -0.5
    got = fl.absorbed(q, qs, lat, sh, "causal", scale=scale, heads=heads, split=2048, width=width,
                      starts=starts)
    ref = fl.absorbed_reference(q, qs, lat, sh, "causal", scale=scale, heads=heads, width=width,
                                starts=starts)
    assert got.shape == (b, heads, rows, fl.LATENT)
    rel, finite = fl._rel(ref, got)
    assert finite and rel < 2e-2, rel
    # Each head's output depends only on its own query: the first 32 heads of
    # a 40-head call match a 32-head-only reference on the same operands.
    if heads == 40:
        sub = fl.absorbed_reference(q[:, :32], qs[:, :32], lat, sh, "causal", scale=scale, heads=32,
                                    width=width, starts=starts)
        rel, _ = fl._rel(sub, got[:, :32])
        assert rel < 2e-2


def test_kimi_32_head_kernel_default_heads_deterministic_and_within_bound():
    # Source/name/grid/template identity is pinned on CPU
    # (test_raon1_mini_nax_decode_cpu.py); this checks the GPU run.
    key = mx.random.key(11)
    k1, k2, k3, k4 = mx.random.split(key, 4)
    q = (mx.random.normal((1, 32, 1, fl.LATENT), key=k1) * 0.05).astype(mx.bfloat16)
    qs = (mx.random.normal((1, 32, 1, fl.SHARED), key=k4) * 0.5).astype(mx.bfloat16)
    lat = mx.random.normal((1, 1, 5000, fl.LATENT), key=k2).astype(mx.bfloat16)
    sh = mx.random.normal((1, 1, 5000, fl.SHARED), key=k3).astype(mx.bfloat16)
    scale = (fl.NOPE + fl.SHARED) ** -0.5
    a = fl.absorbed(q, qs, lat, sh, "causal", scale=scale)
    b = fl.absorbed(q, qs, lat, sh, "causal", scale=scale, heads=32)
    assert bool(mx.array_equal(a, b))
    rel, finite = fl._rel(fl.absorbed_reference(q, qs, lat, sh, "causal", scale=scale), a)
    assert finite and rel < 2e-2

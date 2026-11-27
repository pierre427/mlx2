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


# -- dense head-row packing (mla_decode nax_absorbed_packed, 2026-11-26) ----------

@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_packed_canary_matches_reference_and_grouped_bitwise(dtype):
    result = R.absorbed_flash_canary(40, dtype, packing="dense")
    assert result["ok"], result
    assert all(case["bitwise_vs_grouped"] for case in result["cases"]), result


@pytest.mark.parametrize("rows,width,split,starts", [
    (6, 2050, 2048, None),      # nd5 verify across a split boundary
    (6, 5000, None, None),      # auto split plan
    (16, 4097, 2048, None),     # 640 slots: 14 tiles over 5 threadgroups
    (1, 130, None, None),       # decode row: three partly filled tiles
    (3, 3000, 2048, (0, 777)),  # padded batch lanes
])
def test_packed_partials_are_bitwise_the_grouped_kernels(rows, width, split, starts):
    k1, k2, k3, k4 = mx.random.split(mx.random.key(rows * 7919 + width), 4)
    b = 1 if starts is None else len(starts)
    cap = width + 64
    q = (mx.random.normal((b, 40, rows, fl.LATENT), key=k1) * 0.05).astype(mx.bfloat16)
    qs = (mx.random.normal((b, 40, rows, fl.SHARED), key=k4) * 0.5).astype(mx.bfloat16)
    lat = mx.random.normal((b, 1, cap, fl.LATENT), key=k2).astype(mx.bfloat16)
    sh = mx.random.normal((b, 1, cap, fl.SHARED), key=k3).astype(mx.bfloat16)
    starts = None if starts is None else mx.array(starts, mx.int32)
    scale = (fl.NOPE + fl.SHARED) ** -0.5
    kwargs = dict(scale=scale, heads=40, split=split, width=width, starts=starts, stage="partial")
    grouped = fl.absorbed(q, qs, lat, sh, "causal", **kwargs)
    packed = fl.absorbed(q, qs, lat, sh, "causal", packing="dense", **kwargs)
    # Grouped partials are at 48 padded heads; compare the 40 real ones.
    for got, ref, width_ in zip(packed, grouped, (1, 1, fl.LATENT)):
        ref = ref.reshape(b, rows, -1, 48, width_)[:, :, :, :40]
        got = got.reshape(b, rows, -1, 40, width_)
        assert mx.array_equal(got, ref).item()


def test_packed_device_offset_reads_the_true_length():
    """The packed kernel at a device causal offset over a slack read equals
    the host-offset call at the exact width (DC-5 + mla_decode packed)."""
    k1, k2, k3, k4 = mx.random.split(mx.random.key(11), 4)
    rows, fill, slack = 6, 4093, 6
    cap = fill + rows + slack
    q = (mx.random.normal((1, 40, rows, fl.LATENT), key=k1) * 0.05).astype(mx.bfloat16)
    qs = (mx.random.normal((1, 40, rows, fl.SHARED), key=k4) * 0.5).astype(mx.bfloat16)
    lat = mx.random.normal((1, 1, cap, fl.LATENT), key=k2).astype(mx.bfloat16)
    sh = mx.random.normal((1, 1, cap, fl.SHARED), key=k3).astype(mx.bfloat16)
    scale = (fl.NOPE + fl.SHARED) ** -0.5
    exact = fl.absorbed(q, qs, lat[:, :, :fill + rows], sh[:, :, :fill + rows], "causal",
                        scale=scale, heads=40, split=2048, packing="dense")
    device = fl.absorbed(q, qs, lat, sh, "causal", scale=scale, heads=40, packing="dense",
                         offset_array=mx.array([fill], mx.int32))
    assert mx.array_equal(exact, device).item()

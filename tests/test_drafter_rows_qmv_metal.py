"""Metal check of the drafter multi-row 4-bit matvec against stock MLX.

Not row-exact by design (drafter proposals are target-verified): each output
must match the float32 stock law within float tolerance at the Super VL
LiLiCorr drafter's real projection shapes and its 2..16-row blocks.
"""

import mlx.core as mx
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="drafter rows qmv is Metal-only"
)

from mlx2.runtime.models import drafter_rows_qmv as drq  # noqa: E402

# (N, K) of the Super VL LiLiCorr drafter (hidden 4096, intermediate 12288,
# 8 query / 2 KV heads of 128 per 4096 width, 5 taps, head width 1024).
BACKBONE = [
    (4096, 20480),   # fc (5 target taps)
    (4096, 4096),    # q_proj, o_proj
    (1024, 4096),    # k_proj, v_proj
    (12288, 4096),   # gate_proj, up_proj
    (4096, 12288),   # down_proj
]
HEAD = [
    (1024, 4096),    # lilicorr token / context / pass-hidden projections
    (1024, 3072),    # factor_input_proj
    (1024, 1024),    # out / in / anchor heads, lattice out_proj
    (2048, 1024),    # lattice mlp up
    (1024, 2048),    # lattice mlp down
]
ROWS = [2, 7, 8, 12, 16]
MODES = [("affine", 64), ("affine", 32), ("nvfp4", 16)]


@pytest.fixture(autouse=True)
def _gpu():
    # tests/conftest.py makes the CPU the default device.
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    yield
    mx.set_default_device(previous)


def _quantized(n, k, mode, group_size, seed):
    mx.random.seed(seed)
    weight = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
    parts = mx.quantize(weight, group_size=group_size, bits=4, mode=mode)
    return parts[0], parts[1], (parts[2] if mode == "affine" else None)


def _reference(x, w, s, b, mode, group_size, *, bias=None, scale=None):
    y = mx.quantized_matmul(
        x.astype(mx.float32), w, s, None if b is None else b.astype(mx.float32),
        transpose=True, group_size=group_size, bits=4, mode=mode,
    )
    if scale is not None:
        y = y * scale
    if bias is not None:
        y = y + bias.astype(mx.float32)
    return y


def _check(got, want, tol=1.5e-2):
    got = np.asarray(got.astype(mx.float32))
    want = np.asarray(want)
    assert np.isfinite(got).all()
    scale = float(np.max(np.abs(want)))
    err = float(np.max(np.abs(got - want)))
    assert err <= tol * scale, (err, scale)


@pytest.mark.parametrize("mode,group_size", MODES)
@pytest.mark.parametrize("n,k", BACKBONE + HEAD)
def test_matches_stock_at_drafter_shapes(n, k, mode, group_size):
    w, s, b = _quantized(n, k, mode, group_size, seed=n + k)
    for rows in ROWS:
        x = mx.random.normal((1, rows, k)).astype(mx.bfloat16)
        got = drq.rows_qmv(x, w, s, b, mode=mode, group_size=group_size)
        assert got.shape == (1, rows, n) and got.dtype == mx.bfloat16
        _check(got, _reference(x, w, s, b, mode, group_size))


def test_owned_head_shape_and_fused_epilogue():
    n, k = 131072, 4096
    for mode, group_size in (("affine", 64), ("nvfp4", 16)):
        w, s, b = _quantized(n, k, mode, group_size, seed=7)
        bias = (mx.random.normal((n,)) * 0.1).astype(mx.bfloat16)
        scale = mx.array(0.37, dtype=mx.float32) if mode == "nvfp4" else None
        x = mx.random.normal((1, 7, k)).astype(mx.bfloat16)
        got = drq.rows_qmv(x, w, s, b, mode=mode, group_size=group_size,
                           bias=bias, global_scale=scale)
        _check(got, _reference(x, w, s, b, mode, group_size, bias=bias, scale=scale))
        del w, s, b


def test_float16_batch_rows_and_simdgroup_engine():
    n, k = 4096, 4096
    w, s, b = _quantized(n, k, "affine", 64, seed=11)
    x = mx.random.normal((2, 4, k)).astype(mx.float16)  # 2 x 4 = 8 rows
    for engine in ("nax", "simdgroup") if drq.nax_available() else ("simdgroup",):
        got = drq.rows_qmv(x, w, s, b, mode="affine", group_size=64, engine=engine)
        assert got.shape == (2, 4, n) and got.dtype == mx.float16
        _check(got, _reference(x, w, s, b, "affine", 64))
    w, s, _ = _quantized(n, k, "nvfp4", 16, seed=12)
    for rows in (2, 8, 16):
        x = mx.random.normal((1, rows, k)).astype(mx.bfloat16)
        got = drq.rows_qmv(x, w, s, None, mode="nvfp4", group_size=16, engine="simdgroup")
        _check(got, _reference(x, w, s, None, "nvfp4", 16))


def test_installed_modules_route_rows_and_fall_back_outside():
    from mlx import nn

    from mlx2.runtime.nvfp4_linear import GlobalScaledNVFP4Linear

    mx.random.seed(5)
    linear = nn.Linear(4096, 1024, bias=True)
    quantized = nn.QuantizedLinear.from_linear(linear, group_size=64, bits=4)
    stock = nn.QuantizedLinear.from_linear(linear, group_size=64, bits=4)
    nv = GlobalScaledNVFP4Linear(4096, 1024, bias=True)
    packed, scales = mx.quantize(linear.weight, group_size=16, bits=4, mode="nvfp4")
    nv.weight, nv.scales = packed, scales
    nv.global_scale = mx.array(0.75, dtype=mx.float32)
    nv.bias = linear.bias
    nv_stock = GlobalScaledNVFP4Linear(4096, 1024, bias=True)
    nv_stock.update(nv.parameters())
    holder = nn.Module()
    holder.a, holder.b = quantized, nv
    receipt = drq.install(holder)
    assert receipt["modules"] == ["a", "b"]
    for rows in (1, 2, 8, 16, 17):
        x = mx.random.normal((1, rows, 4096)).astype(mx.bfloat16)
        before = drq.status()
        for mine, ref in ((holder.a, stock), (holder.b, nv_stock)):
            got, want = mine(x), ref(x)
            if 2 <= rows <= 16:
                _check(got, want.astype(mx.float32), tol=2e-2)
            else:
                np.testing.assert_array_equal(
                    np.asarray(got.astype(mx.float32)), np.asarray(want.astype(mx.float32))
                )
        after = drq.status()
        routed = after["kernel_calls"] - before["kernel_calls"]
        assert routed == (2 if 2 <= rows <= 16 else 0)


@pytest.mark.parametrize("engine", ["nax", "simdgroup"])
def test_nvfp4_large_group_scales_and_tiny_global_scale_stay_finite(engine):
    """Real NVFP4 exports carry E4M3 group scales up to 448 with a tiny
    per-tensor scale; 448 * 2^14 overflows half if unfolded in half."""
    if engine == "nax" and not drq.nax_available():
        pytest.skip("no NAX tensor ops on this GPU")
    n, k = 4096, 4096
    weight = mx.random.normal((n, k), key=mx.random.key(7)) * 1000.0
    wq, scales = mx.quantize(weight.astype(mx.bfloat16), group_size=16, bits=4, mode="nvfp4")[:2]
    assert int(np.asarray(scales).astype(np.uint8).max() & 127) >= 120  # scales near 448
    gscale = mx.array(1e-4, dtype=mx.float32)
    x = mx.random.normal((1, 7, k), key=mx.random.key(8)).astype(mx.bfloat16)
    ref = (mx.quantized_matmul(x, wq, scales=scales, biases=None, transpose=True,
                               group_size=16, bits=4, mode="nvfp4").astype(mx.float32) * gscale)
    out = drq.rows_qmv(x, wq, scales, mode="nvfp4", group_size=16, global_scale=gscale,
                       engine=engine).astype(mx.float32)
    mx.eval(ref, out)
    out, ref = np.asarray(out), np.asarray(ref)
    assert np.isfinite(out).all()
    assert np.abs(out - ref).max() <= 0.015 * np.abs(ref).max()

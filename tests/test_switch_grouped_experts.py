"""Grouped-expert q4 SwitchGLU kernels (mlx-lm #2044 port): admission and tolerance law.

The kernels are a separate numerical law from the stock gathers (fp32
accumulation in another order), so the Metal cases compare under a relative
tolerance, against both an fp32 dequantized reference and the stock path.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx2.runtime.models import switch_grouped_experts as G
from mlx2.runtime.models import switch_layers as SL


def _glu(D, F, E, bits=4, seed=0):
    mx.random.seed(seed)
    glu = SL.SwitchGLU(D, F, E)
    nn.quantize(glu, group_size=64, bits=bits, class_predicate=lambda _, m: hasattr(m, "to_quantized"))
    mx.eval(glu.parameters())
    return glu


def _route(T, E, k, seed=1, skew=False):
    mx.random.seed(seed)
    u = mx.random.uniform(shape=(1, T, E))
    if skew:  # concentrate routing on the first min(16, E/2) experts
        u = u + (mx.arange(E) < min(16, E // 2)).astype(mx.float32)
    return mx.argsort(-u, axis=-1)[..., :k].astype(mx.uint32)


def _reference(glu, x, idx):
    w = [mx.dequantize(p["weight"], p["scales"], p["biases"], group_size=64, bits=4).astype(mx.float32)
         for p in (glu.gate_proj, glu.up_proj, glu.down_proj)]
    wg, wu, wd = (a[idx] for a in w)
    xx = x.astype(mx.float32)[..., None, :, None]
    g, u = (wg @ xx).squeeze(-1), (wu @ xx).squeeze(-1)
    return (wd @ (nn.silu(g) * u)[..., None]).squeeze(-1)


def _rel(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return (mx.linalg.norm((a - b).flatten()) / mx.linalg.norm(b.flatten())).item()


def test_default_off_and_stock_route_unchanged():
    assert G._env_mode({}) == "off"
    assert G._env_mode({"MLX2_MOE_GROUPED_EXPERTS": "1"}) == "on"
    with pytest.raises(ValueError):
        G._env_mode({"MLX2_MOE_GROUPED_EXPERTS": "lane"})
    assert G.MODE == "off"
    glu = _glu(128, 128, 8)
    x = mx.random.normal((1, 2, 128)).astype(mx.bfloat16)
    idx = _route(2, 8, 2)
    before = G.status()
    y = glu(x, idx)
    assert y.shape == (1, 2, 2, 128)
    assert G.status()["calls"] == before["calls"] == 0  # hook never consulted while off


def test_refusals_counted():
    G.status(reset=True)
    glu8 = _glu(128, 128, 8, bits=8)
    x = mx.random.normal((1, 2, 128)).astype(mx.bfloat16)
    idx = _route(2, 8, 2)
    assert G.switch_glu(glu8, x, idx) is None
    glu4 = _glu(128, 128, 8)
    assert G.switch_glu(glu4, x.astype(mx.float32), idx) is None
    many = _route(512, 8, 2)
    assert G.switch_glu(glu4, mx.zeros((1, 512, 128), mx.bfloat16), many) is None
    st = G.status(reset=True)
    assert st["calls"] == 0 and st["fallbacks"] == 3
    if not mx.metal.is_available():
        assert set(st["reasons"]) == {"metal unavailable"}
    else:
        assert set(st["reasons"]) == {"table not affine q4 g64", "activation dtype not bf16/fp16",
                                      "load above max_load"}


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")
@pytest.mark.parametrize("shape", [(2048, 512, 256, 8), (256, 384, 16, 4), (512, 192, 8, 2)])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("skew", [False, True])
def test_tolerance_vs_reference_and_stock(shape, dtype, skew):
    D, F, E, k = shape
    glu = _glu(D, F, E)
    for T in (1, 2, 3, 8, 17, 32):
        if T * k / E > G.max_load(D, F):
            continue
        mx.random.seed(100 + T)
        x = (mx.random.normal((1, T, D)) * 0.5).astype(dtype)
        idx = _route(T, E, k, seed=T, skew=skew)
        y = G.switch_glu(glu, x, idx, count=False)
        assert y is not None and y.shape == (1, T, k, D) and y.dtype == dtype
        stock = glu(x, idx)
        ref = _reference(glu, x, idx)
        e_ref, e_stock, e_base = _rel(y, ref), _rel(y, stock), _rel(stock, ref)
        # The stock bf16/fp16 path sits ~e_base from fp32; the grouped law must
        # sit no further than that plus a small margin, and close to stock.
        assert e_ref < max(2 * e_base, 5e-3), (shape, T, dtype, skew, e_ref, e_base)
        assert e_stock < 1e-2, (shape, T, dtype, skew, e_stock)

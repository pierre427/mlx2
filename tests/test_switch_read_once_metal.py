"""Real-Metal gate for the read-once small-row expert kernels (switch_read_once).

Run with ``MLX2_RUN_METAL_TESTS=1``.  Synthetic 4-bit banks (split and fused
gate/up, plain and clamped SwiGLU, heavy expert reuse past the chunk size)
and, when the checkpoint is on disk, one real Hy4-flash layer (layer 10,
~1.9 GB; ``MLX2_HY4_MODEL`` overrides the path).  The kernels sum in fp32 in
another order than ``gather_qmv``, so outputs are compared with the stock
SwitchGLU under a bound of 2 bf16 ulps of each call's largest output, and
both are compared with an fp32 reference of the same law: the read-once
output may not be further from it than the stock one by more than 25%.
"""

import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx2.runtime.models import hy_v4_flash as H
from mlx2.runtime.models import switch_layers as SL
from mlx2.runtime.models import switch_read_once as RO

pytestmark = pytest.mark.skipif(
    os.environ.get("MLX2_RUN_METAL_TESTS") != "1" or not mx.metal.is_available(),
    reason="set MLX2_RUN_METAL_TESTS=1 for the real-Metal gate",
)

HY4 = Path(os.environ.get("MLX2_HY4_MODEL", "~/mlx-models/Hy4-flash-MLX-4.6bit-mlx2"))


@pytest.fixture(autouse=True)
def gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    yield
    mx.set_default_device(previous)


def _glu(D, F, E, activation, seed=0):
    mx.random.seed(seed)
    glu = SL.SwitchGLU(D, F, E, activation=activation)
    for name in ("gate_proj", "up_proj", "down_proj"):
        setattr(glu, name, getattr(glu, name).to_quantized(64, 4))
    for p in (glu.gate_proj, glu.up_proj, glu.down_proj):
        p.scales, p.biases = p.scales.astype(mx.bfloat16), p.biases.astype(mx.bfloat16)
    mx.eval(glu.parameters())
    return glu


def _ulps(y, ref):
    a, b = np.array(y.astype(mx.float32)), np.array(ref.astype(mx.float32))
    ulp = 2.0 ** (np.floor(np.log2(np.abs(b).max())) - 7)
    return float(np.abs(a - b).max() / ulp)


def _reference(glu, x, idx, limit):
    """fp32 per-assignment outputs of the stock law (bf16 where it rounds)."""
    bf = lambda a: a.astype(mx.bfloat16).astype(mx.float32)
    deq = lambda p, e: mx.dequantize(p["weight"][e], p["scales"][e], p["biases"][e],
                                     group_size=64, bits=4).astype(mx.float32)
    idx = np.array(idx)
    out = np.zeros(idx.shape + (x.shape[-1],), np.float32)
    xf = x.astype(mx.float32)
    for e in np.unique(idx):
        rows, slots = np.nonzero(idx == e)
        xe = xf[mx.array(rows)]
        g, u = bf(xe @ deq(glu.gate_proj, int(e)).T), bf(xe @ deq(glu.up_proj, int(e)).T)
        gm = mx.minimum(g, limit)
        h = bf(bf(gm * bf(mx.sigmoid(gm))) * bf(mx.clip(u, -limit, limit)))
        out[rows, slots] = np.array(bf(h @ deq(glu.down_proj, int(e)).T))
    return out


def _check(glu, x, ind, limit):
    RO.status(reset=True)
    ref = glu(x, ind)
    y = RO.switch_glu(glu, x, ind)
    assert y is not None, RO.status()["reasons"]
    assert y.shape == ref.shape and y.dtype == ref.dtype
    assert RO.status()["calls"] == 1
    assert _ulps(y, ref) <= 2.0
    fp = _reference(glu, x, ind, limit)
    err = lambda a: float(np.linalg.norm(np.array(a.astype(mx.float32)) - fp) / np.linalg.norm(fp))
    # Within 1.25x stock's distance from the fp32 law, or within bf16
    # rounding of it (rel L2 2^-7).  The second arm covers small synthetic
    # tables where >= 4 assignments per expert send stock to MLX's streaming
    # rhs kernel, which lands closer to the reference (0.0028 vs 0.0055 on
    # 32 experts); on the real 512-expert layer read-once is the closer one
    # (hy4-readonce-exactness receipt: 0.0047 vs stock 0.0051).
    assert err(y) <= max(1.25 * err(ref), 2.0 ** -7)


def _route(rows, k, E, seed, pool=None):
    rng = np.random.default_rng(seed)
    ids = np.arange(E) if pool is None else rng.choice(E, pool, replace=False)
    return mx.array(np.stack([rng.choice(ids, k, replace=False) for _ in range(rows)]).astype(np.uint32))


@pytest.mark.parametrize("act,limit", [(SL.SwiGLU(), float("inf")), (H.ClampedSwiGLU(1.5), 1.5)])
@pytest.mark.parametrize("rows,pool", [(2, None), (7, None), (8, 16), (16, 8), (33, None), (64, 12)])
def test_synthetic_split_bank(act, limit, rows, pool):
    glu = _glu(512, 768, 32, act)
    x = (mx.random.normal((rows, 512)) * 2).astype(mx.bfloat16)
    # pool < rows * k / MCH: every expert serves several chunks of assignments.
    _check(glu, x, _route(rows, 8, 32, rows, pool), limit if limit != float("inf") else 3.0e38)


def test_synthetic_batched_shape_and_fused_bank_match_split():
    glu = _glu(512, 512, 16, H.ClampedSwiGLU(2.0), seed=4)
    x = (mx.random.normal((2, 3, 512))).astype(mx.bfloat16)
    ind = _route(6, 4, 16, 9).reshape(2, 3, 4)
    y = RO.switch_glu(glu, x, ind)
    assert y.shape == (2, 3, 4, 512)
    assert _ulps(y, glu(x, ind)) <= 2.0
    fused = SL.GateUpFusedSwitchGLU.from_switch_glu(glu)
    for part in ("weight", "scales", "biases"):
        setattr(fused.gate_up_proj, part, mx.concatenate([glu.gate_proj[part], glu.up_proj[part]], axis=1))
    mx.eval(fused.parameters())
    yf = RO.switch_glu(fused, x, ind)
    # Same words, same order: the fused bank reads the same rows bit for bit.
    assert np.array_equal(np.array(yf.astype(mx.float32)), np.array(y.astype(mx.float32)))


@pytest.mark.skipif(not (HY4 / "model-experts-L10.safetensors").exists(), reason="Hy4-flash checkpoint not on disk")
@pytest.mark.parametrize("rows,law", [(2, "uniform"), (7, "overlap"), (8, "uniform"), (16, "overlap"),
                                      (32, "overlap"), (64, "uniform")])
def test_real_hy4_layer10(rows, law):
    glu = _real_layer()
    rng = np.random.default_rng(rows)
    if law == "uniform":
        idx = np.stack([rng.choice(512, 16, replace=False) for _ in range(rows)])
    else:
        pool = rng.choice(512, max(16, int(round(rows * 16 / 1.7))), replace=False)
        idx = np.stack([rng.choice(pool, 16, replace=False) for _ in range(rows)])
    x = mx.random.normal((rows, 1536)).astype(mx.bfloat16)
    _check(glu, x, mx.array(idx.astype(np.uint32)), 10.0)


_REAL = {}


def _real_layer():
    if "glu" not in _REAL:
        w = mx.load(str(HY4 / "model-experts-L10.safetensors"))
        stem = "model.layers.10.mlp.switch_mlp."
        glu = SL.SwitchGLU(1536, 1536, 512, activation=H.ClampedSwiGLU(10.0))
        for name in ("gate_proj", "up_proj", "down_proj"):
            bank = SL.QuantizedSwitchLinear.__new__(SL.QuantizedSwitchLinear)
            nn.Module.__init__(bank)
            bank.group_size, bank.bits, bank.mode = 64, 4, "affine"
            bank.weight, bank.scales, bank.biases = (w[stem + name + "." + p] for p in ("weight", "scales", "biases"))
            bank.freeze()
            setattr(glu, name, bank)
        mx.eval(glu.parameters())
        _REAL["glu"] = glu
    return _REAL["glu"]


def test_synthetic_fp16_bank():
    glu = _glu(512, 768, 32, H.ClampedSwiGLU(4.0), seed=2)
    for p in (glu.gate_proj, glu.up_proj, glu.down_proj):
        p.scales, p.biases = p.scales.astype(mx.float16), p.biases.astype(mx.float16)
    x = mx.random.normal((8, 512)).astype(mx.float16)
    _check(glu, x, _route(8, 8, 32, 5, 12), 4.0)


def test_moe_block_serves_combines_and_counts():
    from test_hy_v4_flash_cpu import tiny_config

    mx.set_default_device(mx.gpu)  # that module pins the CPU device at import

    mx.random.seed(8)
    args = H.ModelArgs.from_hf_config(tiny_config(
        hidden_size=256, moe_intermediate_size=512, routed_expert_hidden_size=512,
        n_routed_experts=32, num_experts_per_tok=8, swiglu_limit=3.0))
    moe = H.MoE(args)
    moe.set_dtype(mx.bfloat16)
    nn.quantize(moe, group_size=64, bits=4,
                class_predicate=lambda p, m: hasattr(m, "to_quantized") and not p.endswith("gate"))
    mx.eval(moe.parameters())
    x = mx.random.normal((1, 7, 256)).astype(mx.bfloat16)
    ref = moe(x)
    moe.expert_read_once = True
    y = moe(x)
    assert moe.read_once_calls == 1 and moe.read_once_declines == {}
    assert _ulps(y, ref) <= 2.0
    moe(x[:, :1])  # one row keeps the stock route
    assert moe.read_once_calls == 1 and moe.read_once_declines == {"below_min_rows": 1}

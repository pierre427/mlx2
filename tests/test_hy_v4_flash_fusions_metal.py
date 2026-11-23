"""Real-Metal gate for the Hy4-flash fusion kernels (hy_v4_flash_fusions).

Run with ``MLX2_RUN_METAL_TESTS=1``.  Synthetic tensors at the real Hy4 widths
(hidden 3072, hc_mult 4, indexer 16 x 128) and the tiny random checkpoint:
no model checkpoint, no speed or qualification claim.
"""

import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx2.runtime.models import hy_v4_flash as H
from mlx2.runtime.models import hy_v4_flash_fusions as F

pytestmark = pytest.mark.skipif(
    os.environ.get("MLX2_RUN_METAL_TESTS") != "1" or not mx.metal.is_available(),
    reason="set MLX2_RUN_METAL_TESTS=1 for the real-Metal gate",
)


@pytest.fixture(autouse=True)
def gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    yield
    mx.set_default_device(previous)


def _seam_inputs(rows, D=3072, n=4, dtype=mx.bfloat16, seed=0, batch=1):
    rng = np.random.default_rng(seed)
    a = lambda *s, sc=1.0: mx.array((rng.standard_normal(s) * sc).astype(np.float32))
    return dict(res=a(batch, rows, n, D).astype(dtype), branch=a(batch, rows, D).astype(dtype),
                post=mx.abs(a(batch, rows, n)) + 0.5, hc_fn=a(2 * n, n * D, sc=0.05), base=a(2 * n, sc=0.5),
                scale=a(2, sc=0.5), w=(1 + 0.1 * a(D)).astype(dtype))


# (batch, rows): decode, verify, prefill, and B > 1 with row counts that fill
# neither the 8-row partial blocks nor the 256-thread tiles.
SEAM_SHAPES = [(1, 1), (1, 6), (1, 64), (1, 513), (3, 5), (2, 13), (4, 1)]


@pytest.mark.parametrize("batch,rows", SEAM_SHAPES)
@pytest.mark.parametrize("merge", [True, False])
def test_hc_seam_kernel_matches_reference(batch, rows, merge):
    t = _seam_inputs(rows, batch=batch)
    eps, hce, mag = 1e-5, 1e-6, 2.0
    streams = H._hc_merge(t["post"], t["branch"], t["res"]) if merge else t["res"]
    post, col = H._hc_pre_kernel(4, eps, hce, mag)(streams, t["hc_fn"], t["base"], t["scale"])
    normed = mx.fast.rms_norm(col, t["w"], eps)
    consts = mx.array([eps, hce, mag, eps], dtype=mx.float32)
    got_s, got_p, got_n = F.hc_seam_metal(t["res"], t["hc_fn"], t["base"], t["scale"], t["w"], consts,
                                          t["post"] if merge else None, t["branch"] if merge else None)
    assert got_s.shape == streams.shape and got_p.shape == post.shape and got_n.shape == normed.shape
    # The merge keeps the reference's rounding points: bitwise.
    assert mx.array_equal(got_s, streams).item()
    # Gates: fp32 split-K reassociation only.
    assert float(mx.abs(got_p - post).max()) < 1e-5
    # Normed input: rare bf16 rounding flips.  The collapse can cancel, so a
    # flip is bounded against the row scale (rms ~1), not the element:
    # measured max 2**-6 at 513 rows, 0.04% of elements.
    diff = mx.abs(got_n.astype(mx.float32) - normed.astype(mx.float32))
    bound = mx.maximum(mx.abs(normed.astype(mx.float32)), 1.0) * 2.0 ** -6
    assert bool(mx.all(diff <= bound).item())
    assert int((got_n != normed).sum().item()) <= max(2, normed.size // 1000)


def test_hc_seam_kernel_refuses_gradients():
    t = _seam_inputs(2)
    consts = mx.array([1e-5, 1e-6, 2.0, 1e-5], dtype=mx.float32)

    def loss(branch):
        _, post, normed = F.hc_seam_metal(t["res"], t["hc_fn"], t["base"], t["scale"], t["w"], consts,
                                          t["post"], branch)
        return normed.astype(mx.float32).sum() + post.sum()

    with pytest.raises(ValueError, match="without a gradient"):
        mx.eval(mx.grad(loss)(t["branch"].astype(mx.float32)))


# (B, heads, rows, G): B > 1, rows not a multiple of the 2-row block, G not a
# multiple of the 256-thread tile; own groups vary per row.
@pytest.mark.parametrize("B,rows,G", [(1, 1, 2048), (1, 6, 16384), (1, 64, 8192), (2, 5, 3001),
                                      (3, 7, 777), (2, 1, 300)])
def test_indexer_kernel_selects_reference_sets(B, rows, G):
    rng = np.random.default_rng(3)
    Hh, d = 16, 128
    k = min(511, G - 1)
    start = 4 * G - rows - 37
    q = mx.array(rng.standard_normal((B, Hh, rows, d)).astype(np.float32))
    gk = mx.array(rng.standard_normal((B, G, d)).astype(np.float32) * 0.3)
    w = mx.array(rng.standard_normal((B, rows, Hh)).astype(np.float32) * 0.02)
    own = (mx.arange(start, start + rows) // 4).astype(mx.int32)
    ref = F.indexer_logits_reference(q, gk, w, own)
    neg = F.indexer_logits_metal(q, gk, w, own, negate=True)
    assert neg.shape == ref.shape
    assert mx.array_equal(mx.isfinite(ref), mx.isfinite(neg)).item()
    fin = mx.isfinite(ref)
    scale = float(mx.max(mx.where(fin, mx.abs(ref), 0.0)).item())
    assert float(mx.max(mx.where(fin, mx.abs(ref + neg), 0.0)).item()) <= 2e-6 * scale
    want = np.sort(np.array(mx.argpartition(-ref, kth=k - 1, axis=-1)[..., :k]), axis=-1)
    got = np.sort(np.array(mx.argpartition(neg, kth=k - 1, axis=-1)[..., :k]), axis=-1)
    np.testing.assert_array_equal(got, want)


def _linears(widths, width, bits, bias, seed, dtype=mx.bfloat16):
    rng = np.random.default_rng(seed)
    mods = []
    for out in widths:
        lin = nn.Linear(width, out, bias=bias)
        lin.weight = mx.array((rng.standard_normal((out, width)) * 0.02).astype(np.float32)).astype(dtype)
        if bias:
            lin.bias = mx.array((rng.standard_normal((out,)) * 0.1).astype(np.float32)).astype(dtype)
        mods.append(nn.QuantizedLinear.from_linear(lin, group_size=64, bits=bits) if bits else lin)
    return mods


@pytest.mark.parametrize("bits", [0, 4, 8])
def test_fused_projection_bitwise_within_self_checked_bound(bits):
    mods = _linears((1024, 576, 128, 16, 8192), 3072, bits, False, 1)  # q_a, kv_a, tk, hw, gate
    fused = F.FusedLinear(mods, [str(i) for i in range(5)])
    measured = fused.self_check(mx.bfloat16)
    assert 1 <= measured <= fused.max_rows
    fused.set_max_rows(measured)
    rng = np.random.default_rng(5)
    for rows in (1, 6, measured, measured + 1, 200):
        x = mx.array(rng.standard_normal((1, rows, 3072)).astype(np.float32)).astype(mx.bfloat16)
        for part, mod in zip(fused.segments(x), mods):
            assert mx.array_equal(part, mod(x)).item(), rows


def test_quantized_engram_projection_at_real_widths():
    """The 4.6-bit pack's Engram banks: 1536 -> 5 x 3072, 8-bit affine g64 with bias."""
    args = H.ModelArgs(engram_layer_ids=[2], engram_layer_head_vocab_sizes={"2": [[7] * 8, [11] * 8]})
    layer = H.EngramLayer(args, 2)
    mods = _linears((3072,) * 5, 1536, 8, True, 2)
    layer.key_projs, layer.value_proj = mods[:4], mods[4]
    assert F.fuse_engram_inputs(layer) is None
    fused = layer.fused_proj
    assert fused.max_rows == F.FUSED_MAX_ROWS["quantized"]  # not the dense 2048
    fused.set_max_rows(fused.self_check(mx.bfloat16))
    rng = np.random.default_rng(9)
    for rows in (1, 12, 13, 16, 32, 64):
        x = mx.array(rng.standard_normal((1, rows, 1536)).astype(np.float32)).astype(mx.bfloat16)
        for part, mod in zip(fused.segments(x), mods):
            assert mx.array_equal(part, mod(x)).item(), rows


@pytest.mark.parametrize("dtype,atol", [(mx.bfloat16, 0.0), (mx.float32, 1e-3)])
def test_tiny_model_all_fusions_match_reference_elementwise(dtype, atol):
    """Layer outputs and logits against the unfused model: exact in bf16 (measured), and
    within fp32 reassociation noise in fp32; a wrong kernel is off by O(1)."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import test_hy_v4_flash_cpu as T

    mx.set_default_device(mx.gpu)  # that module selects the CPU at import
    config = T.tiny_config()
    W = T.checkpoint(config)
    models = []
    for fused in (False, True):
        m, _ = T.build(config, W)
        m.set_dtype(dtype)
        if fused:
            F.install(m, hc_seam=True, mla_inputs=True, engram_inputs=True, indexer_scores=True)
        outs = []
        m.layer_output_readers = {i: (lambda i, s, ids, prefix, outs=outs: outs.append(s)) for i in range(6)}
        cache = m.make_cache()
        logits = [m(mx.array([T.LONG[:30]]), cache)]
        for tok in T.LONG[30:34]:
            logits.append(m(mx.array([[tok]]), cache))
        models.append((m, outs, logits))
    (_, ref_outs, ref_logits), (model, outs, logits) = models
    assert len(outs) == len(ref_outs) == 6 * 5
    for got, want in zip(outs + logits, ref_outs + ref_logits):
        diff = float(mx.abs(got.astype(mx.float32) - want.astype(mx.float32)).max())
        assert diff <= atol, diff
    st = F.status(model)
    assert st["hc_seam_fused"]["calls"]["metal"] > 0
    assert st["indexer_fused_scores"]["observed_used"]
    assert st["mla_fused_input_proj"]["observed_used"] and st["engram_fused_proj"]["observed_used"]

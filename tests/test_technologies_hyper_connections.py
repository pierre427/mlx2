"""CPU parity: every migrated hyper-connection copy equals the reference.

Each ``_legacy_*`` function below is the model's former code, moved here
verbatim (only renamed, with module constants inlined as arguments) before
the model was switched to ``mlx2.runtime.technologies.hyper_connections``.
Comparisons are exact (``mx.array_equal`` plus dtype), never allclose.
"""
import math
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models import deepseek_v4 as ds4
from mlx2.runtime.models import deepseek_v41 as ds41
from mlx2.runtime.models import glm53_gguf as glm
from mlx2.runtime.models import maxwell_moe as maxwell
from mlx2.runtime.models import xing4_0 as xing
from mlx2.runtime.models.precise_ops import gate_sigmoid
from mlx2.runtime.technologies import hyper_connections as hc

DTYPES = (mx.float32, mx.bfloat16, mx.float16)
HC_EPS = 1e-6


@pytest.fixture(autouse=True)
def _cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _exact(a, b):
    assert a.dtype == b.dtype, (a.dtype, b.dtype)
    assert a.shape == b.shape, (a.shape, b.shape)
    assert mx.array_equal(a, b, equal_nan=True).item()


def _all_exact(xs, ys):
    xs = xs if isinstance(xs, tuple) else (xs,)
    ys = ys if isinstance(ys, tuple) else (ys,)
    assert len(xs) == len(ys)
    for x, y in zip(xs, ys):
        _exact(x, y)


# Mix cases: ordinary, saturating, exact ties, zeros, a huge negative row.
def _mix_cases(rng, lead, width):
    yield rng.normal(size=(*lead, width)).astype(np.float32) * 3
    yield rng.normal(size=(*lead, width)).astype(np.float32) * 60
    yield np.ones((*lead, width), np.float32) * 0.25
    yield np.zeros((*lead, width), np.float32)
    big = rng.normal(size=(*lead, width)).astype(np.float32)
    big[..., 2 * 4 :] = -1e4
    yield big


LEADS = ((1, 1), (1, 7), (2, 3))


# ---------------------------------------------------------------------------
# DeepSeek V4 (deepseek_v4.py at d46e7a3ea)
# ---------------------------------------------------------------------------


def _legacy_ds4_sinkhorn_mix(mixes, scale, base, count, iterations, eps):
    """Always compute hyper-connection coefficients in FP32, including loaded base."""
    return _legacy_ds4_sinkhorn_mix_fp32(
        mixes.astype(mx.float32),
        scale.astype(mx.float32),
        base.astype(mx.float32),
        count,
        iterations,
        eps,
    )


def _legacy_ds4_sinkhorn_mix_fp32(z, scale, base, count, iterations, eps):
    """Same equation order with already resident FP32 static HC parameters."""
    pre = mx.sigmoid(z[..., :count] * scale[0] + base[:count]) + eps
    post = 2 * mx.sigmoid(
        z[..., count : 2 * count] * scale[1] + base[count : 2 * count]
    )
    comb = (z[..., 2 * count :] * scale[2] + base[2 * count :]).reshape(
        *z.shape[:-1], count, count
    )
    comb = mx.softmax(comb, axis=-1, precise=True) + eps
    comb /= mx.sum(comb, axis=-2, keepdims=True) + eps
    for _ in range(iterations - 1):
        comb /= mx.sum(comb, axis=-1, keepdims=True) + eps
        comb /= mx.sum(comb, axis=-2, keepdims=True) + eps
    return pre, post, comb


def _legacy_ds4_hc_expand(branch, residual, post, comb):
    value = post[..., None] * branch[..., None, :].astype(mx.float32)
    value += mx.matmul(mx.swapaxes(comb, -1, -2), residual.astype(mx.float32))
    return value.astype(branch.dtype)


def _legacy_ds4_hyper_connection_call(self, value):
    """HyperConnection.__call__ with policy=reference, projection reference."""
    c = self.config
    value32 = value.astype(mx.float32)
    flat = value32.reshape(*value.shape[:-2], -1)
    normalized = mx.fast.rms_norm(flat, None, c.rms_norm_eps)
    mixes = normalized @ self.fn_fp32.T
    if self.head:
        pre = mx.sigmoid(mixes * self.scale_fp32 + self.base_fp32) + c.hc_eps
        return mx.sum(pre[..., None] * value32, axis=-2).astype(value.dtype)
    pre, post, comb = _legacy_ds4_sinkhorn_mix_fp32(
        mixes,
        self.scale_fp32,
        self.base_fp32,
        c.hc_mult,
        c.hc_sinkhorn_iters,
        c.hc_eps,
    )
    return mx.sum(pre[..., None] * value32, axis=-2).astype(value.dtype), post, comb


@pytest.mark.parametrize("iterations", (0, 1, 2, 20))
@pytest.mark.parametrize("lead", LEADS)
def test_ds4_sinkhorn_mix_exact(lead, iterations):
    rng = np.random.default_rng(41)
    scale = mx.array(rng.uniform(0.1, 2.0, 3).astype(np.float32))
    base = mx.array(rng.normal(size=24).astype(np.float32))
    for case in _mix_cases(rng, lead, 24):
        for dtype in DTYPES:
            z = mx.array(case).astype(dtype)
            _all_exact(
                ds4.sinkhorn_mix(z, scale.astype(dtype), base.astype(dtype), 4, iterations, 1e-6),
                _legacy_ds4_sinkhorn_mix(z, scale.astype(dtype), base.astype(dtype), 4, iterations, 1e-6),
            )
        z = mx.array(case)
        _all_exact(
            ds4._sinkhorn_mix_fp32(z, scale, base, 4, iterations, 1e-6),
            _legacy_ds4_sinkhorn_mix_fp32(z, scale, base, 4, iterations, 1e-6),
        )


@pytest.mark.parametrize("count", (2, 4))
def test_ds4_sinkhorn_other_stream_counts(count):
    rng = np.random.default_rng(count)
    width = (2 + count) * count
    z = mx.array(rng.normal(size=(1, 5, width)).astype(np.float32))
    scale = mx.array(rng.uniform(0.1, 2.0, 3).astype(np.float32))
    base = mx.array(rng.normal(size=width).astype(np.float32))
    _all_exact(
        ds4._sinkhorn_mix_fp32(z, scale, base, count, 7, 1e-5),
        _legacy_ds4_sinkhorn_mix_fp32(z, scale, base, count, 7, 1e-5),
    )


@pytest.mark.parametrize("lead", LEADS)
def test_ds4_expand_exact(lead):
    rng = np.random.default_rng(7)
    for dtype in DTYPES:
        branch = mx.array(rng.normal(size=(*lead, 32)).astype(np.float32)).astype(dtype)
        residual = mx.array(rng.normal(size=(*lead, 4, 32)).astype(np.float32))
        post = mx.array(rng.uniform(0, 2, (*lead, 4)).astype(np.float32))
        comb = mx.array(rng.uniform(0, 1, (*lead, 4, 4)).astype(np.float32))
        for res in (residual, residual.astype(dtype)):
            _exact(ds4.hc_expand(branch, res, post, comb), _legacy_ds4_hc_expand(branch, res, post, comb))


@pytest.mark.parametrize("head", (False, True))
@pytest.mark.parametrize("lead", LEADS)
def test_ds4_hyper_connection_exact(lead, head):
    rng = np.random.default_rng(11)
    config = SimpleNamespace(hidden_size=16, hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
                             rms_norm_eps=1e-20)
    width = 4 if head else 24
    hcm = object.__new__(ds4.HyperConnection)
    hcm.config, hcm.head, hcm.policy, hcm.projection_policy = config, head, "reference", "reference"
    hcm.fn_fp32 = mx.array(rng.normal(size=(width, 64)).astype(np.float32) * 0.2)
    hcm.base_fp32 = mx.array(rng.normal(size=width).astype(np.float32))
    hcm.scale_fp32 = mx.array(rng.uniform(0.1, 2.0, 1 if head else 3).astype(np.float32))
    hcm.hc_spec = ds4.config_hc_spec(config)
    for magnitude in (0.0, 1e-3, 1.0, 50.0):
        for dtype in DTYPES:
            value = mx.array(rng.normal(size=(*lead, 4, 16)).astype(np.float32) * magnitude).astype(dtype)
            _all_exact(hcm(value), _legacy_ds4_hyper_connection_call(hcm, value))


# ---------------------------------------------------------------------------
# DeepSeek V4.1 (deepseek_v41.py at d46e7a3ea): post-projection RMS mixes and
# the staggered collapse.
# ---------------------------------------------------------------------------


def _legacy_ds41_mixes(h, fn, rms_norm_eps):
    flat = h.reshape(*h.shape[:2], -1).astype(mx.float32)
    # V4.1 normalizes after projection; V4's pre-normalized GEMM differs.
    mixes = (flat @ fn.T) * mx.rsqrt(
        mx.mean(mx.square(flat), -1, keepdims=True) + rms_norm_eps
    )
    return mixes


def _legacy_ds41_collapse(h, pre):
    return mx.sum(h.astype(mx.float32) * pre[..., None], -2)


@pytest.mark.parametrize("length", (1, 6))
def test_ds41_mixes_and_collapse_exact(length):
    rng = np.random.default_rng(4100 + length)
    c = SimpleNamespace(hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6, rms_norm_eps=1e-6)
    fn = mx.array(rng.normal(size=(24, 4 * 32)).astype(np.float32) * 0.1)
    scale = mx.array(rng.uniform(0.1, 2.0, 3).astype(np.float32))
    base = mx.array(rng.normal(size=24).astype(np.float32))
    for magnitude in (0.0, 1.0, 40.0):
        for dtype in DTYPES:
            h = mx.array(rng.normal(size=(1, length, 4, 32)).astype(np.float32) * magnitude).astype(dtype)
            mixes = hc.hc_mixes(ds41._hc_spec(c), h.reshape(*h.shape[:2], -1), fn)
            _exact(mixes, _legacy_ds41_mixes(h, fn, c.rms_norm_eps))
            pre, post, comb = ds41._sinkhorn_mix_fp32(mixes, scale, base, 4, 20, 1e-6)
            _all_exact(
                (pre, post, comb), _legacy_ds4_sinkhorn_mix_fp32(mixes, scale, base, 4, 20, 1e-6)
            )
            for p in (pre, pre.astype(mx.bfloat16)):
                _exact(
                    hc.hc_collapse(ds41._hc_spec(c), h, p, out_dtype=mx.float32),
                    _legacy_ds41_collapse(h, p),
                )


def test_ds41_model_collapse_method_exact():
    rng = np.random.default_rng(3)
    model = object.__new__(ds41.DeepSeekV41Model)
    model.args = SimpleNamespace(hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6, rms_norm_eps=1e-6,
                                 activation_dtype="bfloat16")
    h = mx.array(rng.normal(size=(1, 3, 4, 16)).astype(np.float32))
    pre = mx.array(rng.uniform(0, 1, (1, 3, 4)).astype(np.float32))
    _exact(model.collapse(h, pre), model.round(_legacy_ds41_collapse(h, pre)))


# ---------------------------------------------------------------------------
# GLM-5.3 (glm53_gguf.py at d46e7a3ea)
# ---------------------------------------------------------------------------


def _legacy_glm_rms(x, weight, eps):
    return mx.fast.rms_norm(x.astype(mx.float32), weight, eps)


def _legacy_glm_hc_collapse(x, weights, prefix, eps):
    z = _legacy_glm_rms(x.reshape(*x.shape[:-2], -1), None, eps)
    mixes = weights.linear(prefix + "_fn")(z)
    scale, base = weights.array(prefix + "_scale"), weights.array(prefix + "_base")
    pre = gate_sigmoid(mixes[..., :4] * scale[0] + base[:4]) + 1e-6
    post = 2 * gate_sigmoid(mixes[..., 4:8] * scale[1] + base[4:8])
    comb = (
        mx.softmax(
            (mixes[..., 8:] * scale[2] + base[8:]).reshape(*mixes.shape[:-1], 4, 4),
            axis=-1,
            precise=True,
        )
        + 1e-6
    )
    comb = comb / (comb.sum(-2, keepdims=True) + 1e-6)
    for _ in range(19):
        comb = comb / (comb.sum(-1, keepdims=True) + 1e-6)
        comb = comb / (comb.sum(-2, keepdims=True) + 1e-6)
    return (x * pre[..., None]).sum(-2), post, comb


def _legacy_glm_hc_expand(branch, residual, post, comb):
    return branch[..., None, :] * post[..., None] + comb.swapaxes(-1, -2) @ residual


class _GLMWeights:
    def __init__(self, rng, dtype):
        self.fn = mx.array(rng.normal(size=(24, 4 * 32)).astype(np.float32) * 0.1).astype(dtype)
        self.scale = mx.array(rng.uniform(0.1, 2.0, 3).astype(np.float32)).astype(dtype)
        self.base = mx.array(rng.normal(size=24).astype(np.float32)).astype(dtype)

    def linear(self, name):
        assert name.endswith("_fn")
        return lambda z: z @ self.fn.T

    def array(self, name):
        return self.scale if name.endswith("_scale") else self.base


@pytest.mark.parametrize("lead", LEADS)
def test_glm_hc_collapse_and_expand_exact(lead):
    rng = np.random.default_rng(53)
    for dtype in DTYPES:
        w = _GLMWeights(rng, dtype)
        for magnitude in (0.0, 1.0, 30.0):
            x = mx.array(rng.normal(size=(*lead, 4, 32)).astype(np.float32) * magnitude).astype(dtype)
            for eps in (1e-5, 1e-6):
                got = glm.hc_collapse(x, w, "blk.0.hc_attn", eps, fused=False)
                want = _legacy_glm_hc_collapse(x, w, "blk.0.hc_attn", eps)
                _all_exact(got, want)
            _, post, comb = want
            branch = mx.array(rng.normal(size=(*lead, 32)).astype(np.float32)).astype(dtype)
            _exact(
                glm.hc_expand(branch, x, post, comb, fused=False),
                _legacy_glm_hc_expand(branch, x, post, comb),
            )


def test_glm_hc_collapse_exact_in_precise_span():
    from mlx2.runtime.models import precise_ops

    rng = np.random.default_rng(9)
    w = _GLMWeights(rng, mx.float32)
    x = mx.array(rng.normal(size=(1, 2, 4, 32)).astype(np.float32) * 8)
    token = precise_ops._IN_PRECISE_SPAN.set(True)
    try:
        got = glm.hc_collapse(x, w, "p", 1e-5, fused=False)
        want = _legacy_glm_hc_collapse(x, w, "p", 1e-5)
    finally:
        precise_ops._IN_PRECISE_SPAN.reset(token)
    _all_exact(got, want)


# ---------------------------------------------------------------------------
# Maxwell (maxwell_moe.py at d46e7a3ea), eager and under mx.compile
# ---------------------------------------------------------------------------


def _legacy_maxwell_hc_pre_body(x, w, alpha_pre, alpha_post, alpha_res, bias, *, n, iters):
    """mHC mappings + stream collapse.  Returns (aggregated, h_post, h_res)."""
    b, t, nc = x.shape
    c = nc // n
    xf = x.astype(mx.float32)
    norm = mx.sqrt(mx.sum(xf * xf, axis=-1, keepdims=True))
    r = 1.0 / (norm / math.sqrt(nc) + HC_EPS)
    proj = xf @ w.astype(mx.float32).T
    alpha = mx.concatenate([
        mx.broadcast_to(alpha_pre, (n,)), mx.broadcast_to(alpha_post, (n,)),
        mx.broadcast_to(alpha_res, (n * n,)),
    ]).astype(mx.float32)
    h = r * proj * alpha + bias.astype(mx.float32)
    h_pre = (mx.sigmoid(h[..., :n]) + HC_EPS).astype(x.dtype)
    h_post = (mx.sigmoid(h[..., n:2 * n]) * 2).astype(x.dtype)
    m = mx.softmax(h[..., 2 * n:].reshape(b, t, n, n), axis=-1)
    m = m + HC_EPS
    m = m / (mx.sum(m, axis=-2, keepdims=True) + HC_EPS)
    for _ in range(iters - 1):
        m = m / (mx.sum(m, axis=-1, keepdims=True) + HC_EPS)
        m = m / (mx.sum(m, axis=-2, keepdims=True) + HC_EPS)
    s = x.reshape(b, t, n, c).astype(mx.float32)
    agg = (s * h_pre.astype(mx.float32)[..., None]).sum(axis=2).astype(x.dtype)
    return agg, h_post, m.astype(x.dtype)


def _legacy_maxwell_hc_post_body(out, residual, h_post, h_res, *, n):
    b, t, c = out.shape
    res = residual.reshape(b, t, n, c).astype(mx.float32)
    mixed = mx.swapaxes(h_res.astype(mx.float32), -1, -2) @ res
    expanded = h_post.astype(mx.float32)[..., None] * out.astype(mx.float32)[:, :, None, :]
    return (expanded + mixed).astype(residual.dtype).reshape(b, t, n * c)


def _legacy_maxwell_hc_head_body(x, fn, base, scale, *, n):
    b, t, nc = x.shape
    c = nc // n
    xf = x.astype(mx.float32)
    rs = mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + HC_EPS)
    mixes = (xf @ fn.astype(mx.float32).T) * rs
    w = mx.sigmoid(mixes * scale.astype(mx.float32) + base.astype(mx.float32)) + HC_EPS
    s = x.reshape(b, t, n, c).astype(mx.float32)
    return (s * w[..., None]).sum(axis=2).astype(x.dtype)


def _both(fn, legacy, *args, compile_=False, **kw):
    if compile_:
        new = mx.compile(lambda *a: fn(*a, **kw))(*args)
        old = mx.compile(lambda *a: legacy(*a, **kw))(*args)
    else:
        new, old = fn(*args, **kw), legacy(*args, **kw)
    _all_exact(new, old)


@pytest.mark.parametrize("compile_", (False, True))
@pytest.mark.parametrize("lead", ((1, 1), (2, 5)))
def test_maxwell_mhc_bodies_exact(lead, compile_):
    rng = np.random.default_rng(220)
    n, c = 4, 24
    for dtype in DTYPES:
        for magnitude in (0.0, 1.0, 25.0):
            x = mx.array(rng.normal(size=(*lead, n * c)).astype(np.float32) * magnitude).astype(dtype)
            w = mx.array(rng.normal(size=(24, n * c)).astype(np.float32) * 0.3)
            alphas = [mx.array(rng.uniform(0.1, 2.0, 1).astype(np.float32)).astype(dtype) for _ in range(3)]
            bias = mx.array(rng.normal(size=24).astype(np.float32)).astype(dtype)
            for iters in (0, 1, 20):
                _both(maxwell._hc_pre_body, _legacy_maxwell_hc_pre_body, x, w, *alphas, bias,
                      n=n, iters=iters, compile_=compile_)
            agg, h_post, h_res = _legacy_maxwell_hc_pre_body(x, w, *alphas, bias, n=n, iters=20)
            out = mx.array(rng.normal(size=(*lead, c)).astype(np.float32)).astype(dtype)
            _both(maxwell._hc_post_body, _legacy_maxwell_hc_post_body, out, x, h_post, h_res,
                  n=n, compile_=compile_)
            fn = mx.array(rng.normal(size=(n, n * c)).astype(np.float32) * 0.3)
            base = mx.array(rng.normal(size=n).astype(np.float32))
            scale = mx.array(rng.uniform(0.1, 2.0, 1).astype(np.float32))
            _both(maxwell._hc_head_body, _legacy_maxwell_hc_head_body, x, fn, base, scale,
                  n=n, compile_=compile_)


# ---------------------------------------------------------------------------
# Xing4.0 (xing4_0.py at d46e7a3ea), eager and under shapeless mx.compile
# ---------------------------------------------------------------------------


def _legacy_xing_mhc_mix_logits(streams, hc_fn, norm_eps):
    """Unweighted fp32 RMSNorm over the flattened streams, then ``@ hc_fn.T``."""
    flat = streams.reshape(*streams.shape[:-2], -1).astype(mx.float32)
    flat = mx.fast.rms_norm(flat, None, norm_eps)
    return flat @ hc_fn.astype(mx.float32).T


def _legacy_xing_mhc_coeffs_body(streams, pre_w, post_w, comb_w, scale, pre_b, post_b, comb_b, iters, eps, lo, hi):
    """fp32 pre/post/comb + Sinkhorn + collapse. Every input shape-agnostic."""
    pre = mx.sigmoid(pre_w * scale[0] + pre_b)
    post = 2.0 * mx.sigmoid(post_w * scale[1] + post_b)
    comb = mx.clip(comb_w * scale[2] + comb_b, lo, hi)
    comb = mx.exp(comb - comb.max(axis=-1, keepdims=True))
    for _ in range(iters):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    collapsed = (mx.expand_dims(pre, -1) * streams.astype(mx.float32)).sum(axis=-2)
    return post, comb, collapsed.astype(streams.dtype)


def _legacy_xing_mhc_update_body(streams, out, post, comb):
    """``post[..., None] * out[..., None, :] + comb @ streams`` in fp32 (HF orientation)."""
    s32 = streams.astype(mx.float32)
    new = mx.expand_dims(post, -1) * mx.expand_dims(out.astype(mx.float32), -2) + comb @ s32
    return new.astype(streams.dtype)


def _xing_split(mix, hc_base, hc_scale, hc):
    base = hc_base.astype(mx.float32)
    return (
        mix[..., :hc],
        mix[..., hc : 2 * hc],
        mix[..., 2 * hc :].reshape(*mix.shape[:-1], hc, hc),
        hc_scale.astype(mx.float32),
        base[:hc],
        base[hc : 2 * hc],
        base[2 * hc :].reshape(hc, hc),
    )


@pytest.mark.parametrize("compile_", (False, True))
@pytest.mark.parametrize("lead", LEADS)
def test_xing_mhc_bodies_exact(lead, compile_):
    rng = np.random.default_rng(40)
    hc_n, hid = 4, 16
    for dtype in DTYPES:
        hc_fn = mx.array(rng.normal(size=(24, hc_n * hid)).astype(np.float32) * 0.3).astype(dtype)
        hc_base = mx.array(rng.normal(size=24).astype(np.float32))
        hc_scale = mx.array(rng.uniform(0.1, 2.0, 3).astype(np.float32))
        for magnitude in (0.0, 1.0, 400.0):
            streams = mx.array(
                rng.normal(size=(*lead, hc_n, hid)).astype(np.float32) * magnitude
            ).astype(dtype)
            _exact(xing._mhc_mix_logits(streams, hc_fn, 1e-6),
                   _legacy_xing_mhc_mix_logits(streams, hc_fn, 1e-6))
            mix = _legacy_xing_mhc_mix_logits(streams, hc_fn, 1e-6)
            parts = _xing_split(mix, hc_base, hc_scale, hc_n)
            for iters, eps, lo, hi in ((20, 1e-6, -30.0, 30.0), (1, 1e-5, -0.5, 0.5), (0, 1e-6, -30.0, 30.0)):
                if compile_:
                    new = mx.compile(lambda *a: xing._mhc_coeffs_body(*a, iters, eps, lo, hi),
                                     shapeless=True)(streams, *parts)
                    old = mx.compile(lambda *a: _legacy_xing_mhc_coeffs_body(*a, iters, eps, lo, hi),
                                     shapeless=True)(streams, *parts)
                else:
                    new = xing._mhc_coeffs_body(streams, *parts, iters, eps, lo, hi)
                    old = _legacy_xing_mhc_coeffs_body(streams, *parts, iters, eps, lo, hi)
                _all_exact(new, old)
            post, comb, _ = old
            out = mx.array(rng.normal(size=(*lead, hid)).astype(np.float32)).astype(dtype)
            if compile_:
                new = mx.compile(xing._mhc_update_body, shapeless=True)(streams, out, post, comb)
                old = mx.compile(_legacy_xing_mhc_update_body, shapeless=True)(streams, out, post, comb)
            else:
                new = xing._mhc_update_body(streams, out, post, comb)
                old = _legacy_xing_mhc_update_body(streams, out, post, comb)
            _exact(new, old)


def test_xing_hyper_connection_reference_exact():
    rng = np.random.default_rng(1)
    args = SimpleNamespace(hc_mult=4, hidden_size=16, hc_sinkhorn_iters=20, hc_eps=1e-6,
                           rms_norm_eps=1e-6, mhc_h_res_clamp_min=-30.0, mhc_h_res_clamp_max=30.0)
    layer = xing.HyperConnection(args)
    layer.hc_fn = mx.array(rng.normal(size=(24, 64)).astype(np.float32) * 0.3)
    layer.hc_base = mx.array(rng.normal(size=24).astype(np.float32))
    layer.hc_scale = mx.array(rng.uniform(0.1, 2.0, 3).astype(np.float32))
    streams = mx.array(rng.normal(size=(1, 3, 4, 16)).astype(np.float32)).astype(mx.bfloat16)
    parts = _xing_split(_legacy_xing_mhc_mix_logits(streams, layer.hc_fn, 1e-6),
                        layer.hc_base, layer.hc_scale, 4)
    _all_exact(layer.reference(streams),
               _legacy_xing_mhc_coeffs_body(streams, *parts, 20, 1e-6, -30.0, 30.0))


# ---------------------------------------------------------------------------
# spec / config derivation / dispatch
# ---------------------------------------------------------------------------


def test_from_config_derives_checkpoint_fields():
    ds = SimpleNamespace(hc_mult=4, hc_sinkhorn_iters=12, hc_eps=1e-5, rms_norm_eps=1e-20)
    spec = hc.HCSpec.from_config(ds)
    assert (spec.hc_mult, spec.sinkhorn_iters, spec.eps, spec.norm_eps, spec.comb_clip) == (
        4, 12, 1e-5, 1e-20, None)
    assert spec == ds4.config_hc_spec(ds)
    xa = {"hc_mult": 2, "hc_sinkhorn_iters": 9, "hc_eps": 1e-6, "rms_norm_eps": 1e-6,
          "mhc_h_res_clamp_min": -30, "mhc_h_res_clamp_max": 30}
    spec = hc.HCSpec.from_config(xa, **xing._HC_LAW)
    assert spec == xing._hc_spec(2, 9, 1e-6, -30.0, 30.0)
    mw = {"num_residual_streams": 4, "mhc_sinkhorn_iterations": 20, "rms_norm_eps": 1e-5}
    spec = hc.HCSpec.from_config(mw, norm_eps=1e-6, **maxwell._HC_LAW)
    assert spec == maxwell._hc_spec(4, 20)


def test_spec_rejects_unknown_law_values():
    with pytest.raises(ValueError):
        hc.HCSpec(orientation="sideways")
    with pytest.raises(ValueError):
        hc.HCSpec(sinkhorn_iters=-1)
    with pytest.raises(ValueError):
        hc.HCSpec(hc_mult=2.5)
    assert hc.HCSpec(hc_mult=np.int64(4), eps=np.float32(1e-6)).hc_mult == 4
    assert hc.HCSpec(sinkhorn_iters=0).sinkhorn_iters == 0  # legacy DS4/Maxwell accept 0


def test_dispatch_is_opt_in_and_fails_closed():
    spec = hc.HCSpec()
    fn, receipt = hc.select("coefficients", spec)
    assert fn is hc.hc_coefficients and receipt.selected == "reference"
    with pytest.raises(RuntimeError, match="refused"):
        # CPU tests: the Metal path refuses (no owned GPU window) and fails closed.
        hc.select("coefficients", spec, policy="ds4_split_sinkhorn_fp32")
    with pytest.raises(RuntimeError, match="refused"):
        hc.select("coefficients", xing._hc_spec(4), policy="ds4_split_sinkhorn_fp32")
    reason = hc.COEFFICIENTS._paths[0].admits(xing._hc_spec(4))
    assert reason is not None and reason.startswith("spec_law: ") and "pre_eps" in reason
    assert hc.COEFFICIENTS.technology == "hyper_connections"
    assert hc.COEFFICIENTS.op == "coefficients"
    with pytest.raises(RuntimeError, match="no single reference"):
        hc.select("hyper_connections.boundary", spec)


# ---------------------------------------------------------------------------
# DS4.1 AST batched dense exact mixes tile
# (adapters/deepseek_v41_ast_batched_dense_exact.py at d46e7a3ea)
# ---------------------------------------------------------------------------


def _legacy_ds41_dense_exact_mixes_tile(block, h, kind):
    from mlx2.adapters import deepseek_v41_ast_batched_dense_exact as exact
    from mlx2.runtime.models.deepseek_v4 import _sinkhorn_mix_fp32
    c, model = block.c, block.model
    rows = exact._rows(h, ndim=4, width=c.hidden_size)
    if h.shape[2] != c.hc_mult or kind not in ("attn", "ffn"):
        raise ValueError("Source AST M1 HC kind/hidden geometry mismatch")
    fn, scale, base = block.hc[kind]
    flat = h.reshape(*h.shape[:2], -1).astype(mx.float32)
    mixes = exact._float_batch_matmul(flat, fn)*mx.rsqrt(
        mx.mean(mx.square(flat), -1, keepdims=True)+c.rms_norm_eps)
    exact._record(model, "mixes", rows)
    return _sinkhorn_mix_fp32(mixes, scale, base, c.hc_mult, c.hc_sinkhorn_iters, c.hc_eps)


@pytest.mark.parametrize("rows", (1, 4, 16))
def test_ds41_dense_exact_mixes_tile_exact(rows):
    from mlx2.adapters import deepseek_v41_ast_batched_dense_exact as exact

    rng = np.random.default_rng(612 + rows)
    c = SimpleNamespace(hc_mult=4, hidden_size=8, hc_sinkhorn_iters=20, hc_eps=1e-6, rms_norm_eps=1e-6)
    hcw = tuple(mx.array(a) for a in (rng.normal(size=(24, 32)).astype(np.float32) * 0.2,
                                      rng.uniform(0.1, 2.0, 3).astype(np.float32),
                                      rng.normal(size=24).astype(np.float32)))
    for magnitude in (0.0, 0.2, 30.0):
        h = mx.array(rng.normal(0, 1, (1, rows, 4, 8)).astype(np.float32) * magnitude)
        for kind in ("attn", "ffn"):
            blocks = [SimpleNamespace(c=c, model=SimpleNamespace(hc_policy="reference"),
                                      hc={"attn": hcw, "ffn": hcw}) for _ in range(2)]
            _all_exact(exact.mixes_tile(blocks[0], h, kind),
                       _legacy_ds41_dense_exact_mixes_tile(blocks[1], h, kind))
            assert (blocks[0].model.source_ast_m1_batch_axis_dense_calls
                    == blocks[1].model.source_ast_m1_batch_axis_dense_calls)


# ---------------------------------------------------------------------------
# fast-path admits: each kernel admits only the exact law it implements
# ---------------------------------------------------------------------------

_DS41_SPEC = ds41._hc_spec(SimpleNamespace(hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6, rms_norm_eps=1e-6))
_MODEL_SPECS = {
    "ds4": ds4.hc_spec(4, 20, 1e-6, 1e-20),
    "ds41": _DS41_SPEC,
    "glm53": glm.HC_SPEC,
    "maxwell": maxwell._hc_spec(4, 20),
    "xing": xing._hc_spec(4, 20, 1e-6, -30.0, 30.0),
}
#: path -> the model specs whose law it implements (everything else refused).
_ADMITTED = {
    "ds4_split_sinkhorn_fp32": {"ds4", "ds41", "glm53"},
    "ds41_source_split_fp32": {"ds41"},
    "ds4_weighted_sum_fp32": {"ds4", "ds41", "glm53", "maxwell", "xing"},
    "ds4_expand_fp32": {"ds4", "ds41", "glm53", "maxwell"},
    "ds4_head_reduce_fp32": {"ds4", "ds41", "maxwell"},
    "ds4_projection_fixed_fp32": {"ds4", "xing"},
    "glm53_hc_mega_post": {"glm53"},
    "glm53_hc_mega_pre": {"glm53"},
    "glm53_hc_fused_boundary": {"glm53"},
    "ds41_source_hc_mega_pre": {"ds41"},
    "ds41_source_hc_mega_store": {"ds41"},
    "ds41_source_hc_expand_collapse": {"ds41"},
    "ds41_fused_expand": {"ds41"},
    "maxwell_mhc_pre": {"maxwell"},
    "maxwell_mhc_post": {"maxwell"},
    "xing4_0_mhc_pre": {"xing"},
    "xing4_0_mhc_update": {"xing"},
}


def test_fast_path_admits_only_their_exact_law():
    paths = {p.name: p for r in hc.REGISTRIES.values() for p in r._paths}
    assert set(paths) == set(_ADMITTED)
    for name, admitted in _ADMITTED.items():
        for model, spec in _MODEL_SPECS.items():
            reason = paths[name].admits(spec)
            # On CPU an admitted law still refuses for the device, never the law.
            if model in admitted:
                assert reason is not None and reason.startswith("device: "), (name, model, reason)
            else:
                assert reason is not None and reason.split(":")[0] in ("spec_law", "geometry"), (
                    name, model, reason)


def test_glm_mega_refuses_the_ds41_law():
    with pytest.raises(RuntimeError, match="spec_law"):
        hc.select("boundary", _DS41_SPEC, policy="glm53_hc_mega_pre")
    assert hc.COEFFICIENTS._paths[0].admits(hc.derive(_MODEL_SPECS["ds4"], sinkhorn_iters=0)) \
        .startswith("iterations: ")
    assert hc.MIXES._paths[0].admits(_MODEL_SPECS["ds4"], project=lambda x: x).startswith("policy_off: ")


def test_fast_path_runs_take_the_reference_signature(monkeypatch):
    from mlx2.runtime.models import deepseek_v4_hc, deepseek_v41_source_hc

    calls = []

    def fake(name, value):
        def run(*args, **kwargs):
            calls.append((name, args, kwargs))
            return value
        return run

    f32 = mx.zeros((1, 2, 4, 8), mx.float32)
    monkeypatch.setattr(deepseek_v4_hc, "split_sinkhorn", fake("split", (1, 2, 3)))
    monkeypatch.setattr(deepseek_v41_source_hc, "split", fake("source", (4, 5, 6)))
    monkeypatch.setattr(deepseek_v4_hc, "weighted_sum", fake("sum", f32[..., 0, :]))
    monkeypatch.setattr(deepseek_v4_hc, "expand", fake("expand", f32))
    monkeypatch.setattr(deepseek_v4_hc, "head_reduce", fake("head", f32[..., 0, :]))
    spec = hc.derive(_MODEL_SPECS["ds4"], sinkhorn_iters=7, eps=1e-5)
    paths = {p.name: p for r in hc.REGISTRIES.values() for p in r._paths}
    z = mx.zeros((1, 2, 24))
    assert paths["ds4_split_sinkhorn_fp32"].run(spec, z, z, z, stream_dtype=mx.bfloat16) == (1, 2, 3)
    assert calls[-1][2] == {"iterations": 7, "eps": 1e-5}
    assert paths["ds41_source_split_fp32"].run(spec, z, z, z) == (4, 5, 6)
    assert calls[-1][2] == {"iterations": 7, "eps": 1e-5}
    bf = mx.zeros((1, 2, 4, 8), mx.bfloat16)
    assert paths["ds4_weighted_sum_fp32"].run(spec, bf, z[..., :4]).dtype == mx.bfloat16
    assert paths["ds4_weighted_sum_fp32"].run(spec, bf, z[..., :4], out_dtype=mx.float32).dtype == mx.float32
    assert paths["ds4_expand_fp32"].run(spec, bf[..., 0, :], f32, z[..., :4], f32[..., :4]).dtype == mx.bfloat16
    assert paths["ds4_head_reduce_fp32"].run(spec, bf, z[..., :4], z[..., :1], z[0, 0, :4]).dtype == mx.bfloat16
    assert calls[-1][2] == {"eps": 1e-5}
    glm_spec = glm.HC_SPEC
    assert paths["ds4_expand_fp32"].run(glm_spec, bf[..., 0, :], bf, z[..., :4], f32[..., :4]).dtype == mx.float32


def test_fixed_projection_gates_the_effective_head_normalization(monkeypatch):
    monkeypatch.setattr(hc, "_gpu_refusal", lambda: None)
    spec = hc.derive(_MODEL_SPECS["ds4"], head_norm_kind="post_inv_l2")
    name = "ds4_projection_fixed_fp32"
    assert hc.select("mixes", spec, policy=name, head=False)[1].selected == name
    with pytest.raises(RuntimeError, match="pre_rms"):
        hc.select("mixes", spec, policy=name, head=True)

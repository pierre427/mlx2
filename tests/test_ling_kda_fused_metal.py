"""ling_kda_fused decode_v1 vs ``KimiDeltaAttention``'s eager chain, bit for bit.

Synthetic layer at Ling-3.2-air's KDA geometry (32 heads x 128, 4 conv taps,
``no_kda_lora``, lower bound -5, rms eps 1e-6, bf16 activations and conv/norm
weights, fp32 A_log / dt_bias / recurrent state); no checkpoint is loaded.
Several decode tokens are chained with all four cache planes carried, and
every step compares the layer output and each plane bitwise.

Real Metal only: run under the GPU lease (cpg_job --lease ->
scripts/run_with_gpu_locks.py) with MLX2_RUN_METAL_TESTS=1.
"""
import os
from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest
from test_kimi_linear import args

from mlx2.runtime.models import ling_kda_fused as lk
from mlx2.runtime.models.bailing_hybrid import KimiDeltaAttention
from mlx2.runtime.models.cache import ArraysCache

pytestmark = pytest.mark.skipif(
    os.environ.get("MLX2_RUN_METAL_TESTS") != "1" or not mx.metal.is_available(),
    reason="set MLX2_RUN_METAL_TESTS=1 (under the GPU lock) for the real-Metal gate",
)

HIDDEN = 256


@pytest.fixture(autouse=True)
def gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def bits(a):
    a = np.array(a.astype(mx.float32) if a.dtype == mx.bfloat16 else a)
    return a.view(np.uint32)


def same(a, b):
    mx.eval(a, b)
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(bits(a), bits(b))


def first_diff(a, b):
    x, y = bits(a).ravel(), bits(b).ravel()
    idx = np.flatnonzero(x != y)
    return (len(idx), idx[:4].tolist()) if len(idx) else None


def layer(seed=3242, scale=1.0, lower_bound=-5.0):
    mx.random.seed(seed)
    model = KimiDeltaAttention(replace(
        args(), hidden_size=HIDDEN, num_attention_heads=32, head_dim=128,
        short_conv_kernel_size=4, no_kda_lora=True, kda_lower_bound=lower_bound,
        rms_norm_eps=1e-6))
    model.eval()
    rng = np.random.default_rng(seed)
    params = {}
    for name in ("q_proj", "k_proj", "v_proj", "f_proj", "g_proj", "b_proj", "o_proj"):
        w = getattr(model, name).weight
        std = scale / np.sqrt(w.shape[-1]) * (2.0 if name in ("f_proj", "g_proj", "b_proj") else 1.0)
        params[name] = {"weight": mx.array(rng.normal(0, std, w.shape).astype(np.float32)).astype(mx.bfloat16)}
    for p in "qkv":
        params[f"{p}_conv1d"] = {"conv": {"weight": mx.array(
            rng.normal(0, 0.4, (lk.INNER, 4, 1)).astype(np.float32)).astype(mx.bfloat16)}}
    params["o_norm"] = {"weight": mx.array(rng.normal(1, 0.2, 128).astype(np.float32)).astype(mx.bfloat16)}
    params["A_log"] = mx.array(np.log(rng.uniform(1, 16, 32)).astype(np.float32))
    params["dt_bias"] = mx.array(rng.normal(-2, 0.5, lk.INNER).astype(np.float32))
    model.update(params)
    mx.eval(model.parameters())
    return model


def fused_step(model, x, planes):
    """The layer's decode step with the fused body in place of its chain."""
    raw = (model.q_proj(x), model.k_proj(x), model.v_proj(x))
    convs = tuple(planes[:3])
    weights = (model.q_conv1d.conv.weight, model.k_conv1d.conv.weight, model.v_conv1d.conv.weight)
    a = model.f_proj(x).reshape(1, 1, 32, 128)
    b = model.b_proj(x).reshape(1, 1, 32)
    gate = model.g_proj(x).reshape(1, 1, 32, 128)
    ops = dict(raw=raw, convs=convs, weights=weights, a=a, b=b, A_log=model.A_log,
               dt_bias=model.dt_bias, state=planes[3], gate=gate, norm=model.o_norm.weight)
    assert lk.ling_kda_fused_refusal(**ops) is None
    out, new_convs, state = lk.ling_kda_fused_decode(
        **ops, rms_eps=model.o_norm.eps, lower_bound=model.lower_bound)
    return model.o_proj(out), [*new_convs, state], out


def prefill(model, rows, rng, scale):
    cache = ArraysCache(4)
    if rows:
        x = mx.array(rng.normal(0, scale, (1, rows, HIDDEN)).astype(np.float32)).astype(mx.bfloat16)
        model(x, cache=cache)
    return cache


def planes_of(cache):
    if cache[0] is None:
        zeros = mx.zeros((1, 3, lk.INNER), mx.bfloat16)
        return [zeros, zeros, zeros, mx.zeros((1, 32, 128, 128), mx.float32)]
    return [cache[i] for i in range(4)]


def chain(model, *, steps=6, prefill_rows=5, seed=7, scale=1.0, merged_conv=False):
    rng = np.random.default_rng(seed)
    model.ling_merged_conv = merged_conv
    cache = prefill(model, prefill_rows, rng, scale)
    planes = planes_of(cache)
    for step in range(steps):
        x = mx.array(rng.normal(0, scale, (1, 1, HIDDEN)).astype(np.float32)).astype(mx.bfloat16)
        want = model(x, cache=cache)
        got, planes, _ = fused_step(model, x, planes)
        assert same(got, want), (step, "output", first_diff(got, want))
        for i, name in enumerate(("q_conv", "k_conv", "v_conv", "state")):
            assert same(planes[i], cache[i]), (step, name, first_diff(planes[i], cache[i]))


@pytest.mark.parametrize("seed", [7, 8, 9])
def test_decode_steps_chain_bitwise_with_all_planes_carried(seed):
    chain(layer(), seed=seed)


def test_merged_conv_reference_is_the_same_chain():
    chain(layer(), merged_conv=True)


def test_fresh_cache_first_token_and_long_chain():
    chain(layer(seed=11), prefill_rows=0, steps=12)


@pytest.mark.parametrize("scale", [0.05, 4.0, 16.0])
def test_activation_scales_stress_norms_and_gates(scale):
    chain(layer(seed=21, scale=scale), scale=scale, steps=4)


def test_contraction_is_moot_for_bf16_operands():
    """Every conv product and rms square multiplies two bf16 values, which is
    exact in fp32 (8 + 8 mantissa bits), so fused and separate multiply-adds
    round identically: all four contraction choices equal the reference.
    (Unlike the fp32 GLM port, where the pin is load-bearing.)"""
    model = layer(seed=5)
    rng = np.random.default_rng(5)
    cache = prefill(model, 4, rng, 1.0)
    planes = planes_of(cache)
    x = mx.array(rng.normal(0, 1, (1, 1, HIDDEN)).astype(np.float32)).astype(mx.bfloat16)
    raw = (model.q_proj(x), model.k_proj(x), model.v_proj(x))
    ops = dict(raw=raw, convs=tuple(planes[:3]),
               weights=(model.q_conv1d.conv.weight, model.k_conv1d.conv.weight, model.v_conv1d.conv.weight),
               a=model.f_proj(x).reshape(1, 1, 32, 128), b=model.b_proj(x).reshape(1, 1, 32),
               A_log=model.A_log, dt_bias=model.dt_bias, state=planes[3],
               gate=model.g_proj(x).reshape(1, 1, 32, 128), norm=model.o_norm.weight)
    want = model(x, cache=cache)
    results = {}
    for pin in ((True, True), (False, True), (True, False), (False, False)):
        out, convs, state = lk.ling_kda_fused_decode(
            **ops, rms_eps=1e-6, lower_bound=-5.0, _contraction=pin)
        results[pin] = same(model.o_proj(out), want) and same(state, cache[3])
    assert all(results.values()), results


def test_refusal_declines_batch_fp32_and_prefill():
    model = layer(seed=2)
    x = mx.zeros((1, 1, HIDDEN), mx.bfloat16)
    base = dict(
        raw=(model.q_proj(x),) * 3, convs=(mx.zeros((1, 3, lk.INNER), mx.bfloat16),) * 3,
        weights=(model.q_conv1d.conv.weight,) * 3, a=model.f_proj(x), b=model.b_proj(x),
        A_log=model.A_log, dt_bias=model.dt_bias, state=mx.zeros((1, 32, 128, 128)),
        gate=model.g_proj(x), norm=model.o_norm.weight)
    assert lk.ling_kda_fused_refusal(**base) is None
    two = mx.zeros((1, 2, lk.INNER), mx.bfloat16)
    assert lk.ling_kda_fused_refusal(**{**base, "raw": (two,) * 3})
    wide = mx.zeros((2, 1, lk.INNER), mx.bfloat16)
    assert lk.ling_kda_fused_refusal(**{**base, "raw": (wide,) * 3})
    assert lk.ling_kda_fused_refusal(**{**base, "gate": base["gate"].astype(mx.float32)})


@pytest.mark.parametrize("prefill_rows", [0, 5])
def test_layer_call_engages_and_chains_bitwise_against_the_eager_layer(prefill_rows):
    """``ling_kda_fused`` on the layer itself: engagement counters, and every
    decode output and cache plane equal to the same layer with it off."""
    model = layer(seed=31)
    rng = np.random.default_rng(31)
    xs = [mx.array(rng.normal(0, 1, (1, r, HIDDEN)).astype(np.float32)).astype(mx.bfloat16)
          for r in ([prefill_rows] if prefill_rows else []) + [1] * 8]
    results = {}
    for fused in (False, True):
        model.ling_kda_fused = fused
        model.ling_calls.clear()
        cache = ArraysCache(4)
        outs = [model(x, cache=cache) for x in xs]
        mx.eval(outs, *[cache[i] for i in range(4)])
        results[fused] = (outs, [cache[i] for i in range(4)], dict(model.ling_calls))
    (want, want_planes, _), (got, got_planes, calls) = results[False], results[True]
    assert calls.get("kda_fused") == 8, calls
    assert calls.get("kda_fused_declined", 0) == (1 if prefill_rows else 0), calls
    for step, (a, b) in enumerate(zip(got, want)):
        assert same(a, b), (step, first_diff(a, b))
    for i in range(4):
        assert same(got_planes[i], want_planes[i]), (i, first_diff(got_planes[i], want_planes[i]))


def test_layer_call_declines_speculation_on_metal():
    model = layer(seed=32)
    model.ling_kda_fused = True
    model.ling_calls.clear()
    cache = ArraysCache(4)
    model(mx.zeros((1, 3, HIDDEN), mx.bfloat16), cache=cache)
    cache.start_speculation()
    model(mx.ones((1, 1, HIDDEN), mx.bfloat16), cache=cache)
    assert model.ling_calls["kda_fused_declined:speculating"] == 1
    assert "kda_fused" not in model.ling_calls

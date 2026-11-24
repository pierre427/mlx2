"""ling_kda_fused (decode_v1): source, constants and admission off Metal.

The bitwise comparison with ``KimiDeltaAttention``'s eager chain is in
tests/test_ling_kda_fused_metal.py (MLX2_RUN_METAL_TESTS=1, GPU lease only).
"""
import re

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models import ling_fused
from mlx2.runtime.models import ling_kda_fused as lk

SPELL = ("metal::exp", "metal::precise::exp")


def _source(**kw):
    args = dict(rms_eps=1e-6, lower_bound=-5.0, scale_q=2.0**-7, scale_k=0.08837890625,
                gate_exp="metal::precise::exp", beta_exp="metal::precise::exp")
    args.update(kw)
    return lk.kernel_source(**args)


def test_source_substitutes_every_placeholder_and_pins_constants():
    source = _source()
    for name in lk._PLACEHOLDERS:
        assert not re.search(rf"\b{name}\d?\b", source), name
    assert lk._hexf(-5.0) in source and lk._hexf(1e-6) in source
    assert lk._hexf(2.0**-7) in source and lk._hexf(0.08837890625) in source
    # Four conv taps and eight rms squares, every one an explicit FMA.
    assert source.count("metal::fma(") == 4 + 4 + 4


def test_unfused_contraction_spells_separate_products():
    source = _source(conv_fma=False, rms_fma=False)
    assert "metal::fma(" not in source
    assert source.count("const float p_ =") == 12


@pytest.mark.parametrize("gate_exp", SPELL)
@pytest.mark.parametrize("beta_exp", SPELL)
def test_eager_sigmoids_copy_ling_fused_body_with_each_spelling(gate_exp, beta_exp):
    source = _source(gate_exp=gate_exp, beta_exp=beta_exp)
    body = ling_fused._SIGMOID.strip().replace("metal::precise::exp", gate_exp)
    assert body in source
    beta = ling_fused._SIGMOID.strip().replace("metal::precise::exp", beta_exp).replace("T ", "float ")
    assert beta in source


def test_scalar_constants_are_mlx_scalar_promotion():
    scale = 128.0 ** -0.5
    assert lk.scalar_in(mx.bfloat16, scale**2) == 2.0**-7
    expected = float(mx.array(scale, dtype=mx.float32).astype(mx.bfloat16).astype(mx.float32).item())
    assert lk.scalar_in(mx.bfloat16, scale) == expected


def _operands(**over):
    bf = mx.bfloat16
    ops = dict(
        raw=tuple(mx.zeros((1, 1, lk.INNER), bf) for _ in range(3)),
        convs=tuple(mx.zeros((1, 3, lk.INNER), bf) for _ in range(3)),
        weights=tuple(mx.zeros((lk.INNER, 4, 1), bf) for _ in range(3)),
        a=mx.zeros((1, 1, 32, 128), bf), b=mx.zeros((1, 1, 32), bf),
        A_log=mx.zeros((32,)), dt_bias=mx.zeros((lk.INNER,)),
        state=mx.zeros((1, 32, 128, 128)), gate=mx.zeros((1, 1, 32, 128), bf),
        norm=mx.ones((128,), bf),
    )
    ops.update(over)
    return ops


def test_refusal_off_metal_names_the_device(monkeypatch):
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    assert lk.ling_kda_fused_refusal(**_operands()) == "not on the Metal GPU"


@pytest.mark.parametrize("over, reason", [
    (dict(a=mx.zeros((1, 1, 32, 128))), "dtype is not bfloat16"),
    (dict(state=mx.zeros((1, 32, 128, 128), mx.bfloat16)), "recurrent state"),
    (dict(state=mx.zeros((2, 32, 128, 128))), "recurrent state"),
    (dict(raw=tuple(mx.zeros((2, 1, lk.INNER), mx.bfloat16) for _ in range(3))), "B1"),
    (dict(raw=tuple(mx.zeros((1, 2, lk.INNER), mx.bfloat16) for _ in range(3))), "B1"),
    (dict(convs=tuple(mx.zeros((1, 2, lk.INNER), mx.bfloat16) for _ in range(3))), "conv state"),
    (dict(A_log=mx.zeros((32,), mx.float16)), "A_log or dt_bias dtype"),
    (dict(norm=mx.ones((64,), mx.bfloat16)), "norm geometry"),
])
def test_refusal_names_each_geometry_and_dtype(monkeypatch, over, reason):
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    monkeypatch.setattr(lk, "served_spellings", lambda: SPELL + SPELL[:1])
    refusal = lk.ling_kda_fused_refusal(**_operands(**over))
    assert refusal is not None and reason in refusal


def test_refusal_accepts_ling_geometry_and_declines_unprobed_spelling(monkeypatch):
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    monkeypatch.setattr(lk, "served_spellings", lambda: SPELL + SPELL[:1])
    assert lk.ling_kda_fused_refusal(**_operands()) is None
    assert lk.ling_kda_fused_refusal(**_operands(A_log=mx.zeros((32,), mx.bfloat16))) is None
    monkeypatch.setattr(lk, "served_spellings", lambda: None)
    assert "sigmoid" in lk.ling_kda_fused_refusal(**_operands())


# ---- wiring into KimiDeltaAttention and the ling3_flash policy ------------

def _layer():
    from dataclasses import replace

    from test_kimi_linear import args

    from mlx2.runtime.models.bailing_hybrid import KimiDeltaAttention

    mx.random.seed(2311)
    layer = KimiDeltaAttention(replace(args(), head_dim=16, no_kda_lora=True,
                                       kda_lower_bound=-5.0))
    layer.eval()
    return layer


def _decode(layer, fused, steps=3, speculate=False):
    from mlx2.runtime.models.cache import ArraysCache

    layer.ling_kda_fused = fused
    layer.ling_calls.clear()
    mx.random.seed(5)
    cache = ArraysCache(4)
    outs = [layer(mx.random.normal((1, 4, 32)), cache=cache)]
    if speculate:
        cache.start_speculation()
    for _ in range(steps):
        outs.append(layer(mx.random.normal((1, 1, 32)), cache=cache))
    return outs, cache, dict(layer.ling_calls)


def test_layer_declines_off_metal_and_keeps_the_eager_chain_bitwise():
    layer = _layer()
    want, want_cache, _ = _decode(layer, False)
    got, got_cache, calls = _decode(layer, True)
    for a, b in zip(got, want):
        assert np.array_equal(np.array(a), np.array(b))
    for i in range(4):
        assert np.array_equal(np.array(got_cache[i]), np.array(want_cache[i]))
    assert calls["kda_fused_declined"] == 4 and "kda_fused" not in calls
    assert calls["kda_fused_declined:not B1 one-token decode"] == 1
    assert calls["kda_fused_declined:not on the Metal GPU"] == 3


def test_layer_declines_speculation_and_keeps_rollback_records():
    layer = _layer()
    _, cache, calls = _decode(layer, True, steps=2, speculate=True)
    assert calls["kda_fused_declined:speculating"] == 2
    assert len(cache._rollbacks) == 2


def test_layer_declines_unbounded_gate_law():
    layer = _layer()
    layer.lower_bound = None
    _, _, calls = _decode(layer, True, steps=1)
    assert calls["kda_fused_declined:unbounded gate law"] == 1


def test_policy_key_defaults_off_and_validates():
    from mlx2.adapters import ling3_flash

    assert ling3_flash.optimization_policy(None)["kda_fused"] is False
    assert ling3_flash.optimization_policy({"kda_fused": True})["kda_fused"] is True
    with pytest.raises(ValueError, match="Ling optimization"):
        ling3_flash.optimization_policy({"kda_fused": 1})


def test_layer_declines_under_the_differentiable_reference():
    from mlx2.runtime.models import differentiable

    layer = _layer()
    differentiable.reset_reached()
    with differentiable.reference_path():
        _, _, calls = _decode(layer, True, steps=1)
    assert calls["kda_fused_declined:differentiable reference"] == 2
    assert differentiable.reached().get("kda_fused") == 2
    assert "kda_fused" in differentiable.ROUTED_KERNELS

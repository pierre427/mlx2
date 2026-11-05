"""CPU algebra/routing only; no Metal VJP or checkpoint qualification claim."""
from types import SimpleNamespace

import mlx.core as mx
from mlx import nn
import numpy as np
import pytest

from mlx2.runtime.models.kimi_linear import MultiLatentAttention
from mlx2.experimental.kimi_linear_lora.training import TrainingLoRALinear


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(917)
    monkeypatch.delenv("MLX_SDPA_VJP_FALLBACK", raising=False)
    yield
    mx.set_default_device(old)


def attention():
    return MultiLatentAttention(SimpleNamespace(
        num_attention_heads=2, qk_nope_head_dim=128, qk_rope_head_dim=64,
        v_head_dim=128, kv_lora_rank=16, hidden_size=32))


def assert_close(a, b, atol=2e-5):
    np.testing.assert_allclose(np.array(a), np.array(b), atol=atol, rtol=2e-5)


def mock_fused_on_cpu(monkeypatch):
    original = mx.fast.scaled_dot_product_attention
    calls = []
    def call(q, k, v, *, scale, mask=None, force_fused=False, **kwargs):
        calls.append((q.shape, k.shape, v.shape, scale, mask, force_fused))
        # CPU cannot execute the fused Metal kernel. This preserves the CPU
        # reference algebra while recording the actual routing request.
        return original(q, k, v, scale=scale, mask=mask, **kwargs)
    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", call)
    return calls


@pytest.mark.parametrize("mask", [None, "causal"])
@pytest.mark.parametrize("length", [9, 17])
def test_zero_value_padding_forward_and_qkv_gradients(mask, length):
    q = mx.random.normal((1, 2, length, 192)) * .2
    k = mx.random.normal(q.shape) * .2
    v = mx.random.normal((1, 2, length, 128)) * .2
    cotangent = mx.random.normal(v.shape)
    scale = 192 ** -.5
    def ordinary(q, k, v):
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    def padded(q, k, v):
        pv = mx.pad(v, [(0, 0), (0, 0), (0, 0), (0, 64)])
        return mx.fast.scaled_dot_product_attention(q, k, pv, scale=scale, mask=mask)[..., :128]
    out, grads = mx.vjp(ordinary, [q, k, v], [cotangent])
    other, other_grads = mx.vjp(padded, [q, k, v], [cotangent])
    mx.eval(out, grads, other, other_grads)
    assert_close(out[0], other[0])
    for a, b in zip(grads, other_grads, strict=True):
        assert_close(a, b)


@pytest.mark.parametrize("mask", [None, "causal"])
def test_model_routing_preserves_scale_crop_and_default(monkeypatch, mask):
    module = attention()
    x = mx.random.normal((1, 11, 32))
    calls = mock_fused_on_cpu(monkeypatch)
    base = module(x, mask=mask)
    mx.eval(base)
    assert module.training_mla_policy == "expanded"
    assert calls[-1][-1] is False
    module.training_mla_policy = "padded_sdpa"
    actual = module(x, mask=mask)
    mx.eval(actual)
    assert calls[-1] == ((1, 2, 11, 192), (1, 2, 11, 192), (1, 2, 11, 192),
                         192 ** -.5, mask, True)
    assert actual.shape == (1, 11, 32)
    assert module.calls["training_padded_sdpa"] == 1
    assert_close(base, actual)


@pytest.mark.parametrize("mask", [None, "causal"])
def test_quantized_lora_toy_forward_and_factor_gradients(monkeypatch, mask):
    module = attention()
    for name in ("q_proj", "o_proj"):
        quantized = nn.QuantizedLinear.from_linear(module[name], group_size=32, bits=8)
        module[name] = TrainingLoRALinear(quantized, rank=4, scale=.5)
        module[name].lora_b = mx.random.normal(module[name].lora_b.shape) * .03
    calls = mock_fused_on_cpu(monkeypatch)
    x = mx.random.normal((1, 11, 32)) * .2
    cotangent = mx.random.normal((1, 11, 32))
    factors = [module.q_proj.lora_a, module.q_proj.lora_b,
               module.o_proj.lora_a, module.o_proj.lora_b]
    def fn(a, b, c, d):
        module.q_proj.lora_a, module.q_proj.lora_b = a, b
        module.o_proj.lora_a, module.o_proj.lora_b = c, d
        return module(x, mask=mask)
    out, grads = mx.vjp(fn, factors, [cotangent])
    mx.eval(out, grads)
    module.training_mla_policy = "padded_sdpa"
    other, other_grads = mx.vjp(fn, factors, [cotangent])
    mx.eval(other, other_grads)
    assert any(c[-1] for c in calls)
    assert_close(out[0], other[0])
    for a, b in zip(grads, other_grads, strict=True):
        assert np.isfinite(np.array(b)).all()
        assert np.linalg.norm(np.array(b)) > 0
        assert_close(a, b, atol=5e-5)


@pytest.mark.parametrize("change", ["unknown", "eval", "auto", "flash", "mask", "cache", "tree", "short", "geometry", "env"])
def test_explicit_unsupported_requests_refuse(monkeypatch, change):
    module = attention()
    module.training_mla_policy = "padded_sdpa"
    x = mx.zeros((1, 11, 32))
    kwargs = {"mask": "causal"}
    if change == "unknown": module.training_mla_policy = "other"
    elif change == "eval": module.eval()
    elif change == "auto": module.policy = "auto"
    elif change == "flash": module.flash = "all"
    elif change == "mask": kwargs["mask"] = mx.ones((11, 11), mx.bool_)
    elif change == "cache": kwargs["cache"] = object()
    elif change == "tree": kwargs["tree"] = object()
    elif change == "short": x = x[:, :8]
    elif change == "geometry": module.args.v_head_dim = 127
    elif change == "env": monkeypatch.setenv("MLX_SDPA_VJP_FALLBACK", "1")
    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported request reached fast SDPA")
    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", forbidden)
    with pytest.raises(ValueError): module(x, **kwargs)
    assert module.calls["training_padded_sdpa"] == 0


def test_fused_refusal_is_not_silently_retried(monkeypatch):
    module = attention()
    module.training_mla_policy = "padded_sdpa"
    def refuse(*args, **kwargs): raise RuntimeError("fused kernel unavailable")
    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", refuse)
    with pytest.raises(RuntimeError, match="unavailable"):
        module(mx.zeros((1, 11, 32)), mask="causal")
    assert module.calls["training_padded_sdpa"] == 0

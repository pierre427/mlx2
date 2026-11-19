"""Gradients through the real Qwen3.5/3.8 hybrid trunk with the GPU kernel selection live.

The amoeba bridge lab's bridgeT2 smoke (host Qwen3.5-9B-4bit, mlx2 bdb927e78)
died in ``mx.grad`` with ``[Primitive::vjp] Not implemented for CustomKernel``:
on the GPU, ``gated_delta_update`` picks the GatedDeltaNet Metal kernel, which
has no VJP.  CPU tests never saw it because off the GPU the selection already
returns the ops reference.  Here the selection is made to believe the GPU is
present and the kernel is replaced by a stand-in whose VJP raises exactly like
a ``mx.fast.metal_kernel`` launch, so the real model code path (tiny size,
quantized like the 4-bit host, fused-GDN switches configured through the
adapters' own ``configure``) reproduces the failure on CPU, and
``differentiable.reference_path()`` is shown to route around it.
"""

from __future__ import annotations

import re
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx2.experimental.weight_space import residual_ablation as ra
from mlx2.runtime.models import differentiable, gated_delta
from mlx2.runtime.models import base, qwen38_fused_gdn, qwen4_fused_gdn_verify
from mlx2.runtime.models.qwen38_27b import Model, ModelArgs

ROOT = Path(__file__).parents[1]
VJP_ERROR = "[Primitive::vjp] Not implemented for CustomKernel."


def _tiny_hybrid(seed=3):
    text = dict(
        model_type="qwen3_5", hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, vocab_size=128,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8,
        linear_value_head_dim=8, linear_conv_kernel_dim=3, full_attention_interval=4,
        partial_rotary_factor=0.5, rope_parameters=None, max_position_embeddings=128,
        mtp_num_hidden_layers=0,
    )
    mx.random.seed(seed)
    model = Model(ModelArgs.from_dict({"model_type": "qwen3_5", "text_config": text}))
    # The served host is 4-bit: gradients reach the writes through quantized matmuls.
    nn.quantize(model, group_size=32, bits=4,
                class_predicate=lambda _, m: isinstance(m, nn.Linear) and m.weight.shape[1] % 32 == 0)
    # Through the adapters' configure(): fused GDN decode on (the Qwen3.8 27B donor's default; the 9B leaves it
    # off), qwen35 contract, prefill off, so the fused shortcut is on the path too.
    previous = qwen4_fused_gdn_verify.MAX_VERIFY_STEPS  # configure() widens this process-wide; put it back
    try:
        qwen38_fused_gdn.configure(model, True, architecture="qwen35", prefill=False)
    finally:
        qwen4_fused_gdn_verify.set_verify_max_steps(previous)
    model.eval()
    mx.eval(model.parameters())
    return model


@pytest.fixture
def gpu_kernel_selection(monkeypatch):
    """The GDN selection sees a GPU; its Metal kernel is a stand-in without a VJP."""
    calls = {"kernel": 0}

    def kernel_without_vjp(q, k, v, g, beta, state, mask=None):
        calls["kernel"] += 1

        @mx.custom_function
        def launch(q, k, v, g, beta, state):
            return gated_delta.gated_delta_ops(q, k, v, g, beta, state, mask)

        @launch.vjp
        def launch_vjp(primals, cotangents, outputs):
            raise ValueError(VJP_ERROR)

        return launch(q, k, v, g, beta, state)

    monkeypatch.setattr(gated_delta.mx, "default_device", lambda: mx.gpu)
    monkeypatch.setattr(gated_delta.mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(gated_delta, "gated_delta_kernel", kernel_without_vjp)
    return calls


def _write_grad(model, tokens, layer):
    """d NLL / d (a residual write at ``layer``), the bridge lab's gradient shape."""
    layers = model.layers

    def loss(write):
        def hook(out):
            return out + write[None].astype(out.dtype)

        with ra._hooked([(layers[layer], hook)]):
            lg = model(mx.array([tokens]), cache=model.make_cache())[0].astype(mx.float32)
        lp = lg - mx.logsumexp(lg, -1, keepdims=True)
        return -mx.take_along_axis(lp[:-1], mx.array(tokens[1:])[:, None], -1).mean()

    return mx.grad(loss)(mx.zeros((len(tokens), 32)))


TOKENS = [5, 17, 3, 99, 42, 8, 61, 23, 7, 110, 4, 77, 31, 2, 90, 15, 66, 12, 50, 9]


def test_gradient_through_gdn_kernel_fails_without_the_reference_path(gpu_kernel_selection):
    model = _tiny_hybrid()
    with pytest.raises(ValueError, match=re.escape(VJP_ERROR)):
        mx.eval(_write_grad(model, TOKENS, layer=0))
    assert gpu_kernel_selection["kernel"] > 0  # the serving selection did pick the kernel


def test_reference_path_routes_around_the_kernel_and_matches_it(gpu_kernel_selection):
    model = _tiny_hybrid()
    # Forward parity: the kernel stand-in computes the ops arithmetic, so the two paths agree exactly here;
    # on the GPU the bridge lab measures the real kernel against the reference (run.json host_grad_path).
    kernel_logits = model(mx.array([TOKENS]), cache=model.make_cache())
    used = gpu_kernel_selection["kernel"]
    assert used > 0
    differentiable.reset_reached()
    with differentiable.reference_path():
        assert differentiable.active()
        ref_logits = model(mx.array([TOKENS]), cache=model.make_cache())
        grad = _write_grad(model, TOKENS, layer=0)
        mx.eval(grad)
    assert not differentiable.active()
    assert gpu_kernel_selection["kernel"] == used  # no kernel launch inside the reference path
    consulted = differentiable.reached()
    assert consulted.get("gated_delta", 0) > 0 and consulted.get("sdpa", 0) > 0, consulted
    assert mx.allclose(kernel_logits, ref_logits, atol=1e-5).item()
    assert bool(mx.isfinite(grad).all().item()) and float(mx.abs(grad).max().item()) > 0
    # The reference gradient is the training-mode (use_kernel=False) gradient, bit for bit.
    model.train()
    expected = _write_grad(model, TOKENS, layer=0)
    model.eval()
    assert mx.array_equal(grad, expected).item()


def test_fused_gdn_shortcuts_stand_down_inside_the_reference_path():
    model = _tiny_hybrid()
    gdn = next(m for _, m in model.named_modules() if isinstance(m, qwen38_fused_gdn.GatedDeltaNet))
    assert gdn.fused_gdn_enabled
    before = dict(gdn.fused_gdn_counters)
    x = mx.zeros((1, 1, 4 * 8 * 2 + 4 * 8))
    with differentiable.reference_path():
        assert gdn._try_fused_decode(x, None, None, None, None, None) is None
    assert gdn.fused_gdn_counters == before  # not a serving fallback: no counter moves


def test_bridge_lab_takes_every_host_gradient_on_the_reference_path():
    source = (ROOT / "scripts" / "amoeba_bridge_lab.py").read_text().splitlines()
    grads = [i for i, line in enumerate(source) if "mx.grad(" in line and "run(H" in line]
    assert len(grads) >= 2  # the load-time probe and the gradient-direction targets
    for i in grads:
        assert "dref.reference_path()" in source[i - 1], source[i]
    module_grads = [line for line in source if "nn.value_and_grad(mod" in line]
    assert module_grads and all("on_reference(" in line for line in module_grads)
    joined = "\n".join(source)
    assert "host_grad_path=hg" in joined  # the receipt names the gradient path and its forward parity


def test_metal_attention_kernels_stand_down_inside_the_reference_path(monkeypatch):
    from mlx2.runtime.models import fp_verify_tensor

    def no_vjp_kernel(*a, **kw):
        raise AssertionError("Metal FP verify attention selected inside the reference path")

    monkeypatch.setattr(fp_verify_tensor, "use_kernel", lambda *a, **kw: True)
    monkeypatch.setattr(fp_verify_tensor, "attention", no_vjp_kernel)
    q = mx.random.normal((1, 2, 4, 8))
    k = mx.random.normal((1, 2, 6, 8))
    with pytest.raises(AssertionError):  # the serving selection takes the (opt-in) kernel for 2..16 query rows
        base.scaled_dot_product_attention(q, k, k, cache=None, scale=0.35, mask=None)
    with differentiable.reference_path():
        out = base.scaled_dot_product_attention(q, k, k, cache=None, scale=0.35, mask=None)
    expected = mx.fast.scaled_dot_product_attention(q, k, k, scale=0.35, mask=None)
    assert mx.allclose(out, expected, atol=1e-6).item()


def test_switch_is_off_by_default_and_scoped():
    assert not differentiable.active() and not differentiable.routed("gated_delta")
    with differentiable.reference_path():
        with differentiable.reference_path():
            assert differentiable.active()
        assert differentiable.active()
    assert not differentiable.active()

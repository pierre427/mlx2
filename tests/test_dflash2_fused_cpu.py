# SPDX-License-Identifier: MIT
"""Synthetic CPU contracts; no model checkpoint is loaded or qualified."""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx2.runtime.drafters import dflash2_fused as fused
from mlx2.runtime.drafters.dflash2 import configure_draft_kernel, _grouped_dynamic_convolve
from mlx2.runtime.drafters.dflash2_kernel import draft_kernel_law
from test_dflash2_draft_kernel_cpu import _model, _vocab_artifacts
from test_qwen38_dflash2_adapter import _pins


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    # A CPU contract must not even create a Metal kernel object.
    monkeypatch.setattr(mx.fast, "metal_kernel", lambda **kw: pytest.fail("CPU built Metal"))
    try:
        yield
    finally:
        mx.set_default_device(previous)


def close(got, want, *, atol=fused.STAGE_ATOL, rtol=fused.STAGE_RTOL):
    assert got.shape == want.shape and got.dtype == want.dtype
    np.testing.assert_allclose(np.asarray(got.astype(mx.float32)), np.asarray(want.astype(mx.float32)), atol=atol, rtol=rtol)


def random(shape, seed, dtype=mx.bfloat16, scale=0.3):
    return (mx.random.normal(shape, key=mx.random.key(seed)) * scale).astype(dtype)


def mock_lane(monkeypatch, *, law="lane-matmul-v1+rows-le-128+grouped", covered=True):
    from mlx2.runtime import lane
    from mlx2.runtime.lane import installer

    monkeypatch.setattr(lane, "backend", lambda: "mpp")
    monkeypatch.setattr(lane, "available", lambda: True)
    calls = []

    def install(model, **kwargs):
        calls.append(kwargs)
        return {"law_id": law, "backend": "mpp", "covered": 30, "groups": 4}

    monkeypatch.setattr(installer, "install", install)
    monkeypatch.setattr(installer, "installed_lane", lambda _: object() if covered else None)
    monkeypatch.setattr(installer, "uninstall", lambda _: calls.append("uninstall"))
    return calls


def test_policy_and_fingerprint(tmp_path, monkeypatch):
    from mlx2.adapters.dflash2 import validate_draft_kernel
    from mlx2.adapters.qwen38_27b import inspect_external_policy

    mock_lane(monkeypatch)
    assert validate_draft_kernel("fused") == "fused"
    assert validate_draft_kernel("stock") is None
    with pytest.raises(ValueError, match="draft_kernel must be"):
        validate_draft_kernel("mystery")
    assert draft_kernel_law("fused") == draft_kernel_law("lane") + "+fused-layer-v2"
    target, draft = _vocab_artifacts(tmp_path)
    pins = _pins(target, draft)
    records = [inspect_external_policy({**pins, "draft_kernel": k}, target) for k in ("stock", "lane", "fused")]
    assert len({r["fingerprint"] for r in records}) == 3
    assert records[-1]["draft_kernel"] == "fused"


def test_receipt_and_per_instance_hooks(monkeypatch):
    calls = mock_lane(monkeypatch)
    model = _model()
    ordinary = _model()
    receipt = configure_draft_kernel(model, "fused")
    assert receipt["kernel"] == "fused"
    assert receipt["law_id"] == draft_kernel_law("fused")
    assert receipt["qualified"] is False
    assert receipt["fused_stages"] == list(fused.STAGES)
    assert receipt["fused_context_projections"] is True and receipt["lattice_topk"] == "block256"
    assert calls == [dict(min_rows=1, max_rows=128, chunk_above_max=False, unquantized=True)]
    assert model.layers[0].__dict__["_fused_layer"] is fused.layer_forward
    assert model.layers[0].self_attn.__dict__["_fused_forward"] is fused.attention
    assert "_fused_layer" not in ordinary.layers[0].__dict__
    object.__setattr__(model, "draft_kernel_receipt", receipt)
    assert model.receipt_settings["projection_kernel"] == {
        "kernel": "fused", "law_id": draft_kernel_law("fused"),
    }


@pytest.mark.parametrize("kernel", ["stock", "lane"])
def test_existing_paths_install_no_layer_hooks(monkeypatch, kernel):
    mock_lane(monkeypatch)
    model = _model()
    configure_draft_kernel(model, kernel)
    for layer in model.layers:
        assert "_fused_layer" not in layer.__dict__
        assert "_fused_forward" not in layer.self_attn.__dict__
    # The original call paths execute without entering either fused function.
    monkeypatch.setattr(fused, "layer_forward", lambda *a: pytest.fail("fused layer"))
    monkeypatch.setattr(fused, "attention", lambda *a: pytest.fail("fused attention"))
    x, ctx = random((1, 4, 16), 5, mx.float32), random((1, 2, 16), 6, mx.float32)
    mx.eval(model.layers[0](x, ctx, model.rope, model.make_cache()[0]))


def test_no_backend_fails_closed():
    with pytest.raises(ValueError, match="lane matmul backend"):
        configure_draft_kernel(_model(), "fused")


@pytest.mark.parametrize("case, error", [
    ("fold_model", "FOLD"), ("fold_layer", "FOLD"), ("sink", "sinks"),
    ("dim", "head_dim"), ("rope", "default nn.RoPE"), ("group", "conv unsupported"),
    ("base", "conv unsupported"), ("heads", "head counts"), ("norm", "RMSNorm"),
    ("post_norm", "finish norm"),
])
def test_unsupported_model_refused_before_lane_install(monkeypatch, case, error):
    calls = mock_lane(monkeypatch)
    model = _model()
    attn = model.layers[0].self_attn
    if case == "fold_model":
        object.__setattr__(model, "fold_policy", {})
    elif case == "fold_layer":
        object.__setattr__(attn, "fold_policy", {})
    elif case == "sink":
        attn.attention_sink_bias = mx.zeros((2,))
    elif case == "dim":
        attn.head_dim = 3
        model.rope.dims = 3
    elif case == "rope":
        model.rope.scale = 0.5
    elif case == "group":
        model.layers[0].attention_conv.group_size = 3
    elif case == "base":
        model.layers[0].mlp_conv.base_kernel = mx.zeros((2, 3, 16))
    elif case == "heads":
        attn.n_kv_heads = 3
    elif case == "norm":
        attn.k_norm.weight = mx.zeros((7,))
    elif case == "post_norm":
        model.layers[0].post_attention_layernorm.weight = mx.zeros((7,))
    with pytest.raises(ValueError, match=error):
        configure_draft_kernel(model, "fused")
    assert calls == []
    assert "_fused_layer" not in model.layers[0].__dict__


@pytest.mark.parametrize("failure", ["uncovered", "law"])
def test_failed_install_rolls_back_without_hooks(monkeypatch, failure):
    calls = mock_lane(monkeypatch, covered=failure != "uncovered", law="wrong" if failure == "law" else "lane-matmul-v1+rows-le-128+grouped")
    model = _model()
    with pytest.raises(ValueError, match="does not cover|differs from"):
        configure_draft_kernel(model, "fused")
    assert calls[-1] == "uninstall"
    assert "_fused_layer" not in model.layers[0].__dict__


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("batch,length,kernel,group", [(1, 1, 2, 4), (2, 16, 2, 4), (4, 64, 5, 8)])
@pytest.mark.parametrize("finish", [False, True])
def test_conv_reference_vs_unfused(dtype, batch, length, kernel, group, finish):
    x = random((batch, length, 32), 10, dtype)
    dynamic = random((batch, length, kernel, 32 // group), 11, dtype)
    base = random((kernel, 32), 12, dtype)
    residual = random(x.shape, 13, dtype) if finish else None
    got = fused.conv_reference(x, dynamic, base, group, residual)
    want = _grouped_dynamic_convolve(x, dynamic, base, group)
    if finish:
        want = residual + want
    close(got, want, **({"atol": 2e-5, "rtol": 2e-5} if dtype == mx.float32 else {}))


def test_conv_zero_fill_when_kernel_longer_than_block():
    x = mx.ones((2, 1, 8))
    got = fused.conv_reference(x, mx.zeros((2, 1, 5, 2)), mx.ones((5, 8)), 4)
    close(got, x, atol=0, rtol=0)


def unfused_qk(q, ck, pk, qw, kw, offsets, rope, qeps=1e-6, keps=1e-6):
    span = ck.shape[1]
    out = []
    for x, w, eps, delta in ((q, qw, qeps, span), (ck, kw, keps, 0), (pk, kw, keps, span)):
        y = mx.fast.rms_norm(x, w, eps).transpose(0, 2, 1, 3)
        # MLX CPU RoPE need not support an empty axis; no elements to rotate.
        out.append(y if not x.shape[1] else mx.concatenate([rope(y[i:i+1], offset=o + delta) for i, o in enumerate(offsets)], axis=0))
    return tuple(out)


@pytest.mark.parametrize("traditional", [False, True])
@pytest.mark.parametrize("length,span,offsets", [(1, 0, [0]), (16, 8, [100]), (4, 100, [0, 7, 83, 1024])])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_qk_reference_vs_unfused(traditional, length, span, offsets, dtype):
    b, d = len(offsets), 128
    q, ck, pk = random((b, length, 4, d), 20, dtype), random((b, span, 2, d), 21, dtype), random((b, length, 2, d), 22, dtype)
    qw, kw = random((d,), 23, dtype, 1), random((d,), 24, dtype, 1)
    rope = nn.RoPE(d, traditional=traditional, base=10000000)
    for got, want in zip(fused.qk_rope_reference(q, ck, pk, qw, kw, offsets, rope, 1e-6, 1e-6), unfused_qk(q, ck, pk, qw, kw, offsets, rope)):
        close(got, want, **({"atol": 2e-5, "rtol": 2e-5} if dtype == mx.float32 else {}))


def test_dispatch_and_runtime_shapes_fail_closed_on_cpu():
    x = mx.ones((1, 2, 8))
    with pytest.raises(ValueError, match="requires the GPU"):
        fused.conv(x, mx.zeros((1, 2, 2, 2)), mx.ones((2, 8)), 4)
    with pytest.raises(ValueError, match="dynamic shape"):
        fused.conv_reference(x, mx.zeros((1, 2, 2, 3)), mx.ones((2, 8)), 4)
    q, ck, pk = mx.ones((1, 2, 2, 8)), mx.ones((1, 0, 1, 8)), mx.ones((1, 2, 1, 8))
    with pytest.raises(ValueError, match="offset"):
        fused.qk_rope_reference(q, ck, pk, mx.ones((8,)), mx.ones((8,)), [-1], nn.RoPE(8), 1e-6, 1e-6)


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("span", [0, 2, 8])
@pytest.mark.parametrize("prefix", [False, True])
def test_fused_layer_reference_preserves_cache_and_group_inputs(monkeypatch, batched, span, prefix):
    model = _model()
    layer = model.layers[0]
    # Use nonzero learned static conv weights so the finish stages matter.
    for i, module in enumerate((layer.attention_conv, layer.mlp_conv)):
        module.base_kernel = random(module.base_kernel.shape, 90 + i, mx.float32)
    b = 4 if batched else 1
    x, ctx = random((b, 4, 16), 70, mx.float32), random((b, span, 16), 71, mx.float32)

    def caches():
        entries = []
        for row in range(b):
            c = model.make_cache()[0]
            if prefix:
                k = random((1, 1, row + 1, 8), 80 + row, mx.float32)
                c.update_and_fetch(k, k)
            entries.append(c)
        return SimpleNamespace(rows=entries) if batched else entries[0]

    plain, fast = caches(), caches()
    expected = layer(x, ctx, model.rope, plain)
    # Same hook path as Metal, changing only the leaf math implementation.
    monkeypatch.setattr(fused, "conv", fused.conv_reference)
    monkeypatch.setattr(fused, "conv_finish_norm", fused.conv_finish_norm_reference)
    monkeypatch.setattr(fused, "qk_rope", fused.qk_rope_reference)
    fused.install(model)
    got = layer(x, ctx, model.rope, fast)
    close(got, expected, atol=2e-5, rtol=2e-5)
    aa = plain.rows if batched else [plain]
    bb = fast.rows if batched else [fast]
    for a, b in zip(aa, bb):
        assert a.offset == b.offset and a._idx == b._idx
        for actual, want in zip(b.state, a.state):
            close(actual, want, atol=2e-5, rtol=2e-5)


def test_fused_layer_keeps_projection_input_identity(monkeypatch):
    from mlx2.runtime.lane import installer

    mock_lane(monkeypatch)
    model = _model()
    configure_draft_kernel(model, "fused")
    layer, seen = model.layers[0], {}
    attn = layer.self_attn
    marker = object()
    monkeypatch.setattr(installer, "_group", lambda _: marker)
    monkeypatch.setattr(fused, "conv", fused.conv_reference)
    monkeypatch.setattr(fused, "conv_finish_norm", fused.conv_finish_norm_reference)
    monkeypatch.setattr(fused, "qk_rope", fused.qk_rope_reference)
    for parent, names in ((attn, ("q_proj", "k_proj", "v_proj")),
                          (layer.mlp, ("gate_proj", "up_proj"))):
        for name in names:
            original = getattr(parent, name)

            def spy(x, name=name, original=original):
                seen[name] = x
                return original(x)

            object.__setattr__(parent, name, spy)
    mx.eval(layer(random((1, 4, 16), 100, mx.float32), random((1, 2, 16), 101, mx.float32), model.rope, model.make_cache()[0]))
    assert seen["q_proj"] is seen["k_proj"] is seen["v_proj"]
    assert seen["gate_proj"] is seen["up_proj"]


def test_runtime_lane_unavailable_fails_before_install(monkeypatch):
    from mlx2.runtime import lane

    calls = mock_lane(monkeypatch)
    monkeypatch.setattr(lane, "available", lambda: False)
    with pytest.raises(ValueError, match="fused.*lane matmul backend"):
        configure_draft_kernel(_model(), "fused")
    assert calls == []


def test_fused_refuses_target_binding_before_install(monkeypatch):
    calls = mock_lane(monkeypatch)
    model = _model()
    model.lm_head = nn.Linear(16, 64, bias=False)
    with pytest.raises(ValueError, match="before binding"):
        configure_draft_kernel(model, "fused")
    assert calls == []


@pytest.mark.parametrize("hidden_dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("dynamic_dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("base_dtype", [mx.bfloat16, mx.float32])
@pytest.mark.parametrize("residual_dtype", [None, mx.bfloat16, mx.float16, mx.float32])
def test_conv_mixed_dtype_promotion(hidden_dtype, dynamic_dtype, base_dtype, residual_dtype):
    x = random((2, 4, 16), 200, hidden_dtype)
    dynamic = random((2, 4, 2, 4), 201, dynamic_dtype)
    base = random((2, 16), 202, base_dtype)
    residual = None if residual_dtype is None else random(x.shape, 203, residual_dtype)
    want = _grouped_dynamic_convolve(x, dynamic, base, 4)
    if residual is not None:
        want = residual + want
    got = fused.conv_reference(x, dynamic, base, 4, residual)
    assert got.dtype == want.dtype == fused.conv_output_dtype(x, dynamic, residual)
    close(got, want)


@pytest.mark.parametrize("hidden_dtype,residual_dtype,weight_dtype", [
    (mx.bfloat16, mx.bfloat16, mx.bfloat16),
    (mx.bfloat16, mx.float32, mx.bfloat16),
    (mx.float32, mx.bfloat16, mx.bfloat16),
    (mx.float16, mx.bfloat16, mx.float32),
    (mx.bfloat16, mx.bfloat16, mx.float32),
    (mx.float32, mx.float32, mx.float32),
])
def test_finish_norm_reference_promotion_and_unrounded_norm(hidden_dtype, residual_dtype, weight_dtype):
    x = random((2, 4, 64), 210, hidden_dtype)
    dynamic = random((2, 4, 2, 4), 211, hidden_dtype)
    base = random((2, 64), 212, hidden_dtype)
    residual = random(x.shape, 213, residual_dtype)
    weight = random((64,), 214, weight_dtype, 1)
    out, normalized = fused.conv_finish_norm_reference(x, dynamic, base, 16, residual, weight, 1e-6)
    # The unrounded fp32 law can be expressed using fp32 inputs to conv_reference.
    raw = fused.conv_reference(x.astype(mx.float32), dynamic.astype(mx.float32),
                               base.astype(mx.float32), 16, residual.astype(mx.float32))
    close(out, raw.astype(mx.result_type(x.dtype, dynamic.dtype, residual.dtype)), atol=0, rtol=0)
    expected = mx.fast.rms_norm(raw, weight.astype(mx.float32), 1e-6).astype(normalized.dtype)
    close(normalized, expected, atol=2e-5, rtol=2e-5)
    assert normalized.dtype == mx.result_type(out.dtype, weight.dtype)
    ordinary = residual + _grouped_dynamic_convolve(x, dynamic, base, 16)
    close(out, ordinary)
    close(normalized, mx.fast.rms_norm(ordinary, weight, 1e-6))
    if out.dtype == mx.bfloat16 and weight.dtype == mx.bfloat16:
        # Removing the intermediate residual cast is an intentional new law.
        rounded = mx.fast.rms_norm(out, weight, 1e-6)
        assert np.any(np.asarray(normalized.astype(mx.float32)) != np.asarray(rounded.astype(mx.float32)))


@pytest.mark.parametrize("batch", [1, 4])
def test_quantized_layer_serves_bf16_residual_with_fp32_cached_kv(monkeypatch, batch):
    from mlx2.runtime.drafters.dflash2 import DFlash2DraftModel
    from mlx2.runtime.drafters.dflash2_config import DFlash2Config

    model = DFlash2DraftModel(DFlash2Config(
        hidden_size=64, intermediate_size=128, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=32,
        vocab_size=64, num_target_layers=1, target_layer_ids=[0],
        layer_types=["sliding_attention"], sliding_window=128,
        conv_kernel_size=2, conv_group_size=16, selector_rank=8,
        selector_top_k=4, mask_token_id=63,
    ))
    nn.quantize(model, group_size=64, bits=4, class_predicate=lambda _, m: isinstance(m, nn.Linear))
    model.set_dtype(mx.bfloat16)
    layer = model.layers[0]
    for idx, module in enumerate((layer.attention_conv, layer.mlp_conv)):
        module.base_kernel = random(module.base_kernel.shape, 220 + idx)
    x, ctx = random((batch, 4, 64), 222), random((batch, 8, 64), 223)

    def cache():
        entries = []
        for row in range(batch):
            entry = model.make_cache()[0]
            assert entry.keys.dtype == mx.float32  # serialized empty planes
            k = random((1, 1, 17 + row, 32), 224 + row)
            entry.update_and_fetch(k, k)  # same direct seed as the Metal test
            assert entry.keys.dtype == entry.values.dtype == mx.float32
            entries.append(entry)
        return SimpleNamespace(rows=entries) if batch > 1 else entries[0]

    # Diagnose the legal promotion without changing the whole-layer Metal test.
    prepared, dynamic = layer.attention_conv.prepare(layer.input_layernorm(x))
    attn_output = layer.self_attn(prepared, ctx, model.rope, cache())
    assert x.dtype == mx.bfloat16 and attn_output.dtype == mx.float32
    h = x + layer.attention_conv.finish(attn_output, dynamic)
    mlp_prepared, _ = layer.mlp_conv.prepare(layer.post_attention_layernorm(h))
    assert layer.mlp(mlp_prepared).dtype == mx.float32
    want = layer(x, ctx, model.rope, cache())
    monkeypatch.setattr(fused, "conv", fused.conv_reference)
    monkeypatch.setattr(fused, "conv_finish_norm", fused.conv_finish_norm_reference)
    monkeypatch.setattr(fused, "qk_rope", fused.qk_rope_reference)
    fused.install(model)
    got = layer(x, ctx, model.rope, cache())
    assert got.dtype == want.dtype == mx.float32
    close(got, want, atol=fused.LAYER_ATOL, rtol=fused.LAYER_RTOL)


def test_metadata_cache_reuses_equal_values_without_changing_them():
    a = fused._metadata(mx.uint32, [5120, 16, 81920])
    assert a is fused._metadata(mx.uint32, (5120, 16, 81920))
    assert a is not fused._metadata(mx.int32, [5120, 16, 81920])
    np.testing.assert_array_equal(np.asarray(a), [5120, 16, 81920])


@pytest.mark.parametrize("kernel", ["lane", "fused"])
def test_stage_instrumentation_matches_the_layer_path(monkeypatch, kernel):
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts/bench_dflash2_fused_layer.py"
    spec = importlib.util.spec_from_file_location("dflash_fused_bench", script)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    mock_lane(monkeypatch)
    model = _model()
    configure_draft_kernel(model, kernel)
    monkeypatch.setattr(fused, "conv", fused.conv_reference)
    monkeypatch.setattr(fused, "conv_finish_norm", fused.conv_finish_norm_reference)
    monkeypatch.setattr(fused, "qk_rope", fused.qk_rope_reference)
    layer = model.layers[0]
    for idx, module in enumerate((layer.attention_conv, layer.mlp_conv)):
        module.base_kernel = random(module.base_kernel.shape, 240 + idx, mx.float32)
    x, ctx = random((1, 4, 16), 242, mx.float32), random((1, 2, 16), 243, mx.float32)
    stages = []

    def measure(name, fn):
        stages.append(name)
        value = fn()
        mx.eval(value)
        return value

    got = bench.stage_forward(layer, x, ctx, model.rope, model.make_cache()[0],
                              fused_path=kernel == "fused", measure=measure)
    want = layer(x, ctx, model.rope, model.make_cache()[0])
    close(got, want, atol=2e-5, rtol=2e-5)
    assert {"input_norm", "attention_kernel_projection", "attention_conv_prepare", "qkv_projections",
            "qk_norm_rope", "sdpa_cache_append", "o_proj", "mlp_kernel_projection", "mlp_conv_prepare",
            "mlp_gate_up", "swiglu", "down_proj", "mlp_conv_finish_residual"} <= set(stages)
    if kernel == "fused":
        assert "attention_conv_finish_residual_norm" in stages
        assert "post_attention_norm" not in stages
    else:
        assert {"attention_conv_finish_residual", "post_attention_norm"} <= set(stages)


@pytest.mark.parametrize("reference", [False, True])
def test_finish_norm_validates_before_dispatch(reference):
    fn = fused.conv_finish_norm_reference if reference else fused.conv_finish_norm
    x = mx.ones((1, 2, 8), dtype=mx.bfloat16)
    dynamic, base = mx.zeros((1, 2, 2, 2)), mx.ones((2, 8))
    with pytest.raises(ValueError, match="requires a residual"):
        fn(x, dynamic, base, 4, None, mx.ones((8,)), 1e-6)
    with pytest.raises(ValueError, match="hidden-width"):
        fn(x, dynamic, base, 4, x, mx.ones((7,)), 1e-6)
    with pytest.raises(ValueError, match="epsilon"):
        fn(x, dynamic, base, 4, x, mx.ones((8,)), float("nan"))
    if not reference:
        with pytest.raises(ValueError, match="requires the GPU"):
            fn(x, dynamic, base, 4, x, mx.ones((8,)), 1e-6)

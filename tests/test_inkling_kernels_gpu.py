"""Owned Metal probes for synthetic Inkling components, never model qualification."""

import copy
import os

import mlx.core as mx
import pytest
from mlx import nn

from mlx2.runtime.inkling_tree_executor import InklingTreeExecutor
from mlx2.runtime.models.cache import ArraysCache
from mlx2.runtime.models.inkling import Model, ModelArgs
from mlx2.runtime.models.inkling.language import (
    InklingShortConvolution,
    InklingSparseMoE,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("MLX2_RUN_GPU_TESTS") != "1", reason="owned GPU window required"
)


@pytest.fixture(autouse=True)
def metal():
    mx.set_default_device(mx.gpu)
    yield
    mx.set_default_device(mx.cpu)
    mx.clear_cache()


@pytest.mark.parametrize("batch,length", [(1, 1), (2, 1), (1, 4), (2, 8)])
def test_fused_shortconv_preserves_strided_input_and_residual(batch, length):
    mx.random.seed(321)
    conv = InklingShortConvolution(128, 4, 0)
    conv.conv.weight = conv.conv.weight.astype(mx.bfloat16)
    raw = mx.random.normal((batch, length, 512)).astype(mx.bfloat16)
    x = raw[..., 128:256]
    residual = raw[..., 256:384]
    ordinary, fused = ArraysCache(4), ArraysCache(4)
    state = mx.random.normal((batch, 3, 128)).astype(mx.bfloat16).astype(mx.float32)
    ordinary[0] = state
    fused[0] = state
    a = conv(x, cache=ordinary, residual=residual)
    conv.fused = True
    b = conv(x, cache=fused, residual=residual)
    mx.eval(a, b, ordinary.cache, fused.cache)
    assert bool(
        mx.allclose(
            a.astype(mx.float32), b.astype(mx.float32), atol=0.016, rtol=0.01
        ).item()
    )
    assert bool(mx.array_equal(ordinary[0], fused[0]).item())
    assert conv.fused_calls == 1


def _moe(experts=8):
    mx.random.seed(44)
    args = ModelArgs(
        hidden_size=128,
        intermediate_size=2048,
        dense_intermediate_size=256,
        n_routed_experts=experts,
        num_experts_per_tok=6,
        n_shared_experts=2,
    )
    m = InklingSparseMoE(args)
    m.set_dtype(mx.bfloat16)
    m.e_score_correction_bias = mx.linspace(-0.5, 0.5, experts)
    nn.quantize(m.switch_mlp, group_size=128, bits=2, mode="affine")
    m.eval()
    mx.eval(m.parameters())
    return m


def test_q2_group128_down_combine_engages_and_matches_reference():
    m = _moe()
    for rows in (1, 4, 8):
        x = mx.random.normal((1, rows, 128)).astype(mx.bfloat16)
        a = m(x)
        m.down_combine = True
        b = m(x)
        mx.eval(a, b)
        error = mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()
        assert error <= 0.125
        assert bool(
            mx.allclose(
                a.astype(mx.float32), b.astype(mx.float32), atol=0.04, rtol=0.03
            ).item()
        )
        m.down_combine = False
    assert m.down_combine_calls == 3


def test_fused_router_selects_same_experts_and_weights():
    m = _moe()
    logits = mx.random.normal((7, 10)).astype(mx.bfloat16)
    a_idx, a_w, a_s = m._route(logits)
    m.fused = True
    b_idx, b_w, b_s = m._route(logits)
    mx.eval(a_idx, a_w, a_s, b_idx, b_w, b_s)
    assert bool(mx.array_equal(mx.sort(a_idx, axis=-1), mx.sort(b_idx, axis=-1)).item())
    a_order, b_order = mx.argsort(a_idx, axis=-1), mx.argsort(b_idx, axis=-1)
    assert bool(
        mx.allclose(
            mx.take_along_axis(a_w, a_order, -1),
            mx.take_along_axis(b_w, b_order, -1),
            atol=0.01,
            rtol=0.01,
        ).item()
    )
    assert bool(mx.allclose(a_s, b_s, atol=0.01, rtol=0.01).item())


@pytest.mark.parametrize("single_node", [False, True])
def test_packed_ast_under_installed_lane_law_matches_singletons(single_node):
    from mlx2.runtime.lane import install, installed

    mx.random.seed(71)
    args = ModelArgs(
        hidden_size=128,
        num_hidden_layers=2,
        vocab_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=32,
        intermediate_size=128,
        dense_intermediate_size=256,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=2,
        d_rel=8,
        rel_extent=16,
        sliding_window_size=4,
        local_layer_ids=[0],
        dense_mlp_idx=1,
    )
    model = Model(args)
    model.set_dtype(mx.bfloat16)
    for layer in model.layers:
        layer.self_attn.rel_proj = (
            mx.random.normal(layer.self_attn.rel_proj.shape).astype(mx.bfloat16) * 0.1
        )
    model.eval()
    mx.eval(model.parameters())
    receipt = install(model, min_rows=1, max_rows=16)
    assert receipt["covered"]
    assert all(
        installed(module)
        for _, module in model.named_modules()
        if isinstance(module, nn.Linear)
    )
    cache = model.make_cache()
    model(mx.array([[1, 2, 3, 4, 5]]), cache=cache)
    expected = copy.deepcopy(cache)
    proposed, parents, path, selected = (
        ([6], [-1], [0], [6])
        if single_node
        else ([6, 7, 8, 9], [-1, 0, 0, 2], [0, 2, 3], [6, 8, 9])
    )
    values, _, tx = InklingTreeExecutor.create().forward(
        model, proposed, parents, cache, [1], cached=False
    )
    tx.commit_paths([path])
    ref = []
    for token in selected:
        ref.append(model(mx.array([[token]]), cache=expected))
    a, b = model(mx.array([[10]]), cache=cache), model(mx.array([[10]]), cache=expected)
    mx.eval(values, ref, a, b)
    assert bool(mx.array_equal(values[:, path[-1]], ref[-1][:, -1]).item())
    assert bool(mx.array_equal(a, b).item())


@pytest.mark.parametrize("experts", [8, 256])
def test_split_q2_nax_swiglu_uses_original_tables_and_reports_engagement(experts):
    from mlx2.runtime.models import moe_nax_gather as nax

    m = _moe(experts)
    x = mx.random.normal((1, 128, 128)).astype(mx.bfloat16)
    old_mode = nax.set_mode("off")
    old_bits = nax.set_affine_bits((2, 4, 6, 8))
    try:
        a = m(x)
        mx.eval(a)
        nax.status(reset=True)
        nax.set_mode("fused")
        with nax.forward_scope(mx.zeros((1, 128), mx.uint32), None):
            b = m(x)
        mx.eval(b)
        receipt = nax.status()
        if not receipt["nax_host"]:
            pytest.skip("M5 NAX unavailable on this host")
        assert any(
            "/affine-2" in name and count
            for name, count in receipt["calls_by_format"].items()
        ), receipt
        assert receipt["calls"]["swiglu_split"] > 0, receipt
        assert bool(mx.array_equal(a, b).item()), receipt
    finally:
        nax.set_mode(old_mode)
        nax.set_affine_bits(old_bits)


def test_fused_relative_mask_query_tiles_match_materialized_reference(monkeypatch):
    from mlx2.runtime.models.inkling import language

    mx.random.seed(987)
    rel = mx.random.normal((2, 5, 4, 8)).astype(mx.bfloat16)
    proj = mx.random.normal((8, 16)).astype(mx.bfloat16)
    carrier = mx.zeros((2, 2, 20, 32), mx.bfloat16)
    left = mx.array([0, 3], mx.int32)
    # Runtime offset is intentionally not S-LQ (a non-final query tile).
    a = language.banded_additive_mask(rel, proj, 7, 20, 0, 16, left_padding=left)
    b = language.banded_additive_mask(
        rel, proj, 7, 20, 0, 16, left_padding=left, shape_ref=carrier, fused=True
    )
    mx.eval(a, b)
    assert bool(
        mx.allclose(
            a.astype(mx.float32), b.astype(mx.float32), atol=0.0625, rtol=0.02
        ).item()
    )

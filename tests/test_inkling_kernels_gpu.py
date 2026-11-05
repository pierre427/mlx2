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


@pytest.mark.parametrize(
    "batch,length", [(1, 1), (2, 1), (1, 4), (2, 8), (1, 128), (2, 257)]
)
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
    conv.prefill_fused = True
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


@pytest.mark.parametrize("rows", [2, 4, 8, 12, 16, 32, 64])
def test_tree_q2_decode_chunks_keep_singleton_reductions(rows):
    from mlx2.runtime.lane import install

    args = ModelArgs(
        hidden_size=512,
        intermediate_size=2048,
        n_routed_experts=8,
        num_experts_per_tok=6,
        n_shared_experts=2,
    )
    mx.random.seed(902 + rows)
    m = InklingSparseMoE(args)
    m.set_dtype(mx.bfloat16)
    nn.quantize(m.switch_mlp, group_size=128, bits=2, mode="affine")
    m.fused = m.down_combine = True
    install(m, min_rows=1, max_rows=64, chunk_above_max=False)
    mx.eval(m.parameters())
    x = mx.random.normal((1, rows, 512)).astype(mx.bfloat16)
    expected = mx.concatenate([m(x[:, i : i + 1]) for i in range(rows)], axis=1)
    before = m.down_combine_calls
    actual = m.tree(x)
    assert m.down_combine_calls - before == (rows + 7) // 8
    reordered = m.tree(x[:, ::-1])[:, ::-1]
    mx.eval(actual, expected, reordered)
    assert bool(mx.array_equal(actual, expected).item())
    assert bool(mx.array_equal(actual, reordered).item())


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


@pytest.mark.parametrize("backend", ["simd", "nax"])
@pytest.mark.parametrize(
    "batch,queries,keys,window,offset,padding",
    [
        (1, 1, 32768, 0, 32767, None),
        (2, 8, 1049, 512, 1041, [0, 37]),
        (2, 3, 37, 0, 0, [0, 2]),
        (1, 33, 1025, 0, 992, None),
        (1, 128, 512, 512, 384, None),
    ],
)
def test_inline_relative_attention_matches_independent_and_native_oracles(
    batch, queries, keys, window, offset, padding, backend
):
    from mlx2.runtime.models.inkling.attention import relative_attention
    from mlx2.runtime.models.inkling.language import banded_additive_mask

    if backend == "nax":
        from mlx2.runtime.models.inkling.attention_nax import relative_attention_nax

        relative_attention = relative_attention_nax
    mx.random.seed(97)
    # Actual checkpoint GQA/head/relative dimensions, deliberately strided.
    q = mx.random.normal((batch, 32, queries, 256)).astype(mx.bfloat16)[..., ::2]
    k = mx.random.normal((batch, 8, keys, 256)).astype(mx.bfloat16)[..., ::2]
    v = mx.random.normal((batch, 8, keys, 256)).astype(mx.bfloat16)[..., ::2]
    r = (mx.random.normal((batch, queries, 32, 32)) * 0.1).astype(mx.bfloat16)[..., ::2]
    proj = (mx.random.normal((16, 1024)) * 0.1).astype(mx.bfloat16)
    pad = None if padding is None else mx.array(padding, mx.int32)
    tau = mx.full((batch, queries), 1.046875, mx.bfloat16)
    q = q * tau[:, None, :, None]
    mask = banded_additive_mask(r, proj, offset, keys, window, 1024, left_padding=pad)
    mask = mx.where(mask > -1e29, mask * tau[:, None, :, None], mask)
    got = relative_attention(
        q,
        k,
        v,
        r,
        proj,
        offset=offset,
        scale=1 / 128,
        window=window,
        left_padding=pad,
        tau=tau,
    )
    native = mx.fast.scaled_dot_product_attention(q, k, v, scale=1 / 128, mask=mask)
    expanded_k = mx.repeat(k, 4, axis=1).astype(mx.float32)
    expanded_v = mx.repeat(v, 4, axis=1).astype(mx.float32)
    scores = (q.astype(mx.float32) @ expanded_k.swapaxes(-1, -2)) / 128
    independent = (
        mx.softmax(scores + mask.astype(mx.float32), axis=-1) @ expanded_v
    ).astype(q.dtype)
    mx.eval(got, native, independent)
    assert bool(mx.allclose(got, independent, atol=0.008, rtol=0.025).item())
    assert bool(mx.allclose(got, native, atol=0.008, rtol=0.025).item())


@pytest.mark.parametrize("chunk", [1, 2, 4])
def test_grouped_q2_private_rows_match_singletons_and_stock(chunk):
    from mlx2.runtime.models.inkling.experts import routed

    mx.random.seed(394)
    args = ModelArgs(
        hidden_size=512,
        intermediate_size=2048,
        n_routed_experts=8,
        num_experts_per_tok=6,
        n_shared_experts=2,
    )
    m = InklingSparseMoE(args)
    m.set_dtype(mx.bfloat16)
    nn.quantize(m.switch_mlp, group_size=128, bits=2, mode="affine")
    mx.eval(m.parameters())
    x = mx.random.normal((8, 512)).astype(mx.bfloat16)
    indices = mx.array(
        [[int((i + j) % 8) for j in range(6)] for i in range(8)], mx.uint32
    )
    a = routed(x, indices, m.switch_mlp, chunk=chunk)
    b = mx.concatenate(
        [routed(x[i : i + 1], indices[i : i + 1], m.switch_mlp) for i in range(8)]
    )
    reference = mx.concatenate(
        [m.switch_mlp(x[i : i + 1], indices[i : i + 1]) for i in range(8)]
    )
    mx.eval(a, b, reference)
    assert bool(mx.array_equal(a, b).item())
    assert bool(
        mx.allclose(
            a.astype(mx.float32), reference.astype(mx.float32), atol=0.02, rtol=0.03
        ).item()
    )
    reverse = routed(x[::-1], indices[::-1], m.switch_mlp, chunk=chunk)[::-1]
    mx.eval(reverse)
    assert bool(mx.array_equal(a, reverse).item())


def test_tree_convolution_kernel_preserves_each_ancestry_and_tail():
    from mlx2.runtime.models.inkling.ancestry import convolution

    mx.random.seed(12)
    conv = InklingShortConvolution(128, 4, 0)
    conv.conv.weight = conv.conv.weight.astype(mx.bfloat16)
    conv.fused = True
    raw = mx.random.normal((1, 5, 128)).astype(mx.bfloat16)
    state = mx.random.normal((1, 3, 128)).astype(mx.bfloat16).astype(mx.float32)
    paths = [[0], [0, 1], [0, 2], [0, 2, 3], [0, 1, 4]]
    values, tails = convolution(conv, raw, state, paths)
    for row, path in enumerate(paths):
        cache = ArraysCache(4)
        cache[0] = state
        out = None
        for node in path:
            out = conv(raw[:, node : node + 1], cache=cache)
        mx.eval(values, tails, out, cache.cache)
        assert bool(mx.array_equal(values[:, row : row + 1], out).item())
        assert bool(mx.array_equal(tails[row : row + 1], cache[0]).item())


@pytest.mark.parametrize(
    "prefix,window", [(127, 0), (512, 512), (16384, 0), (32768, 0)]
)
def test_mapped_tree_attention_matches_same_singleton_reductions(prefix, window):
    from mlx2.runtime.models.inkling.tree_attention import attend

    mx.random.seed(887)
    paths = [[0], [0, 1], [0, 2], [0, 2, 3]]
    q = (mx.random.normal((1, 32, 4, 128)) * 0.2).astype(mx.bfloat16)
    k = (mx.random.normal((1, 8, prefix + 4, 128)) * 0.2).astype(mx.bfloat16)
    v = mx.random.normal(k.shape).astype(mx.bfloat16)
    r = (mx.random.normal((1, 4, 32, 16)) * 0.1).astype(mx.bfloat16)
    proj = (mx.random.normal((16, 1024)) * 0.1).astype(mx.bfloat16)
    a = attend(
        q, k, v, r, proj, scale=128**-0.5, paths=paths, prefix=prefix, window=window
    )
    refs = []
    for row, path in enumerate(paths):
        ids = mx.concatenate([mx.arange(prefix), mx.array(path) + prefix])
        if window:
            ids = ids[-window:]
        refs.append(
            attend(
                q[:, :, row : row + 1],
                mx.take(k, ids, axis=2),
                mx.take(v, ids, axis=2),
                r[:, row : row + 1],
                proj,
                scale=128**-0.5,
                window=window,
            )
        )
    b = mx.concatenate(refs, axis=2)
    mx.eval(a, b)
    assert bool(mx.array_equal(a, b).item())


@pytest.mark.parametrize("cohort_width", [1, 2, 4])
@pytest.mark.parametrize("kernel_policy", ["mapped_grouped", "ordinary_decode"])
@pytest.mark.parametrize("root_only", [False, True])
def test_physical_cohort_preserves_independent_accepted_state(
    cohort_width, kernel_policy, root_only
):
    from mlx2.runtime.lane import install

    mx.random.seed(615)
    args = ModelArgs(
        hidden_size=512,
        num_hidden_layers=2,
        vocab_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=128,
        intermediate_size=2048,
        dense_intermediate_size=1024,
        n_routed_experts=8,
        num_experts_per_tok=6,
        n_shared_experts=2,
        d_rel=16,
        rel_extent=32,
        sliding_window_size=8,
        local_layer_ids=[0],
        dense_mlp_idx=0,
    )
    model = Model(args)
    model.set_dtype(mx.bfloat16)
    for layer in model.layers:
        layer.self_attn.rel_proj = (
            mx.random.normal(layer.self_attn.rel_proj.shape) * 0.1
        ).astype(mx.bfloat16)
        nn.quantize(layer.mlp.switch_mlp, group_size=128, bits=2, mode="affine")
    model.configure_kernels(
        fused=True,
        grouped=kernel_policy == "mapped_grouped",
        down_combine=kernel_policy == "ordinary_decode",
        tree_kernels=True,
        tree_cohort=True,
        attention="mapped" if kernel_policy == "mapped_grouped" else "decode",
    )
    model.eval()
    mx.eval(model.parameters())
    install(model, min_rows=1, max_rows=64, chunk_above_max=False)
    caches = [model.make_cache() for _ in range(cohort_width)]
    for i, cache in enumerate(caches):
        model(mx.array([[i + 1] * (15 + i)]), cache=cache)
    refs = copy.deepcopy(caches)
    executor = InklingTreeExecutor.create()
    inputs = [[10 + 4 * lane + i for i in range(4)] for lane in range(cohort_width)]
    parents = [
        [-1, 0, 0, 2] if lane % 2 == 0 else [-1, 0, 1, 0]
        for lane in range(cohort_width)
    ]
    paths = [[0, 2, 3] if lane % 2 == 0 else [0, 1, 2] for lane in range(cohort_width)]
    if root_only:
        inputs = [row[:1] for row in inputs]
        paths = [[0]] * cohort_width
        values, _, transaction = executor.forward_plain_anchors(
            model, mx.array(inputs), caches, [1]
        )
    else:
        values, _, transaction = executor.forward_many(
            model, inputs, parents, caches, [1], cached=False, configured_limit=4
        )
    transaction.commit_paths(paths)
    assert transaction.physical_packed == (cohort_width > 1 or not root_only)
    receipt = executor.receipt()
    assert receipt["cross_request_physical_packing"] == (cohort_width > 1)
    assert receipt["cross_request_packed_dispatches"] == int(cohort_width > 1)
    for lane, path in enumerate(paths):
        for node in path:
            expected = model(mx.array([[inputs[lane][node]]]), cache=refs[lane])
            mx.eval(expected, values)
            assert bool(
                mx.array_equal(
                    values[lane : lane + 1, node : node + 1], expected
                ).item()
            )
        a = model(mx.array([[30 + lane]]), cache=caches[lane])
        b = model(mx.array([[30 + lane]]), cache=refs[lane])
        mx.eval(a, b)
        assert bool(mx.array_equal(a, b).item())


@pytest.mark.parametrize("prefix", [16383, 32767])
def test_decode_mapped_tree_matches_ordinary_singleton_attention(prefix):
    from mlx2.runtime.models.inkling.attention import relative_attention
    from mlx2.runtime.models.inkling.tree_attention import attend

    mx.random.seed(888)
    paths = [[0], [0, 1], [0, 2], [0, 2, 3]]
    tau = mx.array([[1, 1.125, 1.25, 1.5]], mx.bfloat16)
    q = (mx.random.normal((1, 32, 4, 128)) * 0.2).astype(mx.bfloat16)
    q = q * tau[:, None, :, None]
    k = (mx.random.normal((1, 8, prefix + 4, 128)) * 0.2).astype(mx.bfloat16)
    v = mx.random.normal(k.shape).astype(mx.bfloat16)
    r = (mx.random.normal((1, 4, 32, 16)) * 0.1).astype(mx.bfloat16)
    proj = (mx.random.normal((16, 1024)) * 0.1).astype(mx.bfloat16)
    actual = attend(
        q, k, v, r, proj, scale=128**-0.5, paths=paths, prefix=prefix, tau=tau
    )
    expected = []
    for row, path in enumerate(paths):
        ids = mx.concatenate([mx.arange(prefix), mx.array(path) + prefix])
        expected.append(
            relative_attention(
                q[:, :, row : row + 1],
                mx.take(k, ids, axis=2),
                mx.take(v, ids, axis=2),
                r[:, row : row + 1],
                proj,
                scale=128**-0.5,
                offset=prefix + len(path) - 1,
                tau=tau[:, row : row + 1],
            )
        )
    expected = mx.concatenate(expected, axis=2)
    mx.eval(actual, expected)
    assert bool(mx.array_equal(actual, expected).item())

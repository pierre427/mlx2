"""Adapter load contracts and ordinary arithmetic for mixed-dtype AST GDN."""

import json
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx2.adapters.qwen36_35b import Qwen3635BA3BAdapter
from mlx2.adapters.tensorfold_dtype_gate import (
    artifact_tree_tile_dtype_issue,
    model_tree_tile_dtype_issue,
)
from mlx2.runtime.models import qwen38_tree_gdn as gdn


def test_header_allowance_requires_adapter_contract_and_bf16_loader(tmp_path):
    with mx.stream(mx.cpu):
        mx.save_safetensors(str(tmp_path / "model.safetensors"), {
            "language_model.model.layers.0.input_layernorm.weight": mx.ones((4,)),
            "language_model.model.layers.0.linear_attn.conv1d.weight": mx.ones((4, 4, 1)),
            "language_model.model.layers.0.linear_attn.dt_bias": mx.ones((4,)),
            "language_model.model.layers.0.linear_attn.norm.weight": mx.ones((4,)),
        })
    options = Qwen3635BA3BAdapter.ast_tree_tile_dtype_contract
    assert artifact_tree_tile_dtype_issue(tmp_path, **options)
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"dtype": "bfloat16"}}))
    assert artifact_tree_tile_dtype_issue(tmp_path, **options) is None
    assert artifact_tree_tile_dtype_issue(tmp_path)
    from mlx2.server import apply_ast_invocation
    args = SimpleNamespace(model=str(tmp_path), ast=True, ast_drafter=None,
                           draft_model=None, native_mtp=False, external_draft=False,
                           prompt_lookup=False)
    resolution = SimpleNamespace(adapter_type=Qwen3635BA3BAdapter,
                                 artifact={"has_mtp": True}, default_route="native_mtp")
    assert apply_ast_invocation(args, None, resolution)["draft_model"] == str(tmp_path)
    assert args.ast_route["tree_tiles"] is True and args.external_draft
    (tmp_path / "config.json").write_text('{"dtype":"float32"}')
    assert artifact_tree_tile_dtype_issue(tmp_path, **options)


def test_loaded_gate_requires_normalized_compute_weights_but_preserves_gdn_f32():
    class Layer(nn.Module):
        def __init__(self):
            self.input_layernorm = nn.RMSNorm(4)
            self.linear_attn = nn.Module()
            self.linear_attn.norm = nn.RMSNorm(4)
            self.linear_attn.dt_bias = mx.ones((4,))

    with mx.stream(mx.cpu):
        model = nn.Module()
        model.layers = [Layer()]
        assert model_tree_tile_dtype_issue(model, mixed_gdn_f32=True)
        model.layers[0].input_layernorm.weight = mx.ones((4,), dtype=mx.bfloat16)
        assert model_tree_tile_dtype_issue(model, mixed_gdn_f32=True) is None
        assert model_tree_tile_dtype_issue(model)
        model.layers[0].linear_attn.dt_bias = mx.ones((4,), dtype=mx.float16)
        assert model_tree_tile_dtype_issue(model, mixed_gdn_f32=True)


def test_mixed_decay_reuses_ordinary_f32_law_without_rounding_bias():
    from mlx2.runtime.models.gated_delta import compute_g

    with mx.stream(mx.cpu):
        rows, channels, nv = 2, 128, 2
        qkv = mx.zeros((1, rows, channels), dtype=mx.bfloat16)
        a = mx.array([[[0.125, -0.25], [0.5, 1.0]]], dtype=mx.bfloat16)
        bias = mx.array([0.01367891, -0.16789123], dtype=mx.float32)
        alog = mx.array([-0.87654321, 0.34567891])
        calls = []

        def kernel(**kwargs):
            calls.append(kwargs)
            return tuple(mx.zeros(shape, dtype=dtype) for shape, dtype in zip(
                kwargs["output_shapes"], kwargs["output_dtypes"]
            ))

        args = (kernel, qkv, [mx.zeros((1, 3, channels), dtype=mx.bfloat16)],
                mx.ones((channels, 4, 1), dtype=mx.bfloat16),
                mx.array([0] * (rows * 4), dtype=mx.int32), a, a,
                alog, bias, mx.zeros((rows,), dtype=mx.int32))
        out = gdn._launch(*args, nk=1, nv=nv, dk=32, dv=32, zs=nv, ao=0, bo=0)
        expected = compute_g(alog, a, bias)
        rounded = compute_g(alog, a, bias.astype(mx.bfloat16))
        assert mx.array_equal(out[3].view(mx.uint32), expected.view(mx.uint32)).item()
        assert not mx.array_equal(expected, rounded).item()
        assert args[8].dtype == mx.float32
        assert len(calls) == 1


@pytest.mark.parametrize("packed", [False, True])
def test_mixed_post_matches_ordinary_and_keeps_bf16_dispatch(packed):
    from mlx2.runtime.models.qwen3_next import Qwen3NextRMSNormGated

    calls = []
    sentinel = object()
    def original(*args):
        calls.append(args)
        return sentinel
    glue = SimpleNamespace(gdn_post=original)
    fuse = SimpleNamespace(gdn_post=original)
    gdn.install_mixed_post_hooks(glue, fuse)
    installed = glue.gdn_post
    gdn.install_mixed_post_hooks(glue, fuse)
    assert glue.gdn_post is installed
    with mx.stream(mx.cpu):
        y = mx.linspace(-1.7, 2.1, 64).reshape(1, 2, 2, 16).astype(mx.bfloat16)
        z = mx.linspace(-3, 3, 64).reshape(1, 2, 32).astype(mx.bfloat16)
        norm = Qwen3NextRMSNormGated(16)
        norm.weight = mx.linspace(0.897123, 1.271236, 16)
        weight_before = norm.weight.tolist()
        actual_z = mx.concatenate([z, mx.zeros((1, 2, 4), dtype=mx.bfloat16)], -1) if packed else z
        fn = fuse.gdn_post if packed else glue.gdn_post
        actual = fn(y, actual_z, norm.weight, norm.eps)
        expected = norm(y, z.reshape(y.shape)).reshape(1, 2, 32)
        assert mx.array_equal(actual.view(mx.uint16), expected.view(mx.uint16)).item()
        assert actual.dtype == mx.bfloat16 and norm.weight.dtype == mx.float32
        assert norm.weight.tolist() == weight_before
        assert fn(y, actual_z, norm.weight.astype(mx.bfloat16), norm.eps) is sentinel
        assert len(calls) == 1
        with pytest.raises(ValueError, match="gated norm"):
            fn(y, actual_z, norm.weight.astype(mx.float16), norm.eps)

"""--ast selects the tree-tile route and drafter, or the family's chain route."""

from types import SimpleNamespace

import pytest

from mlx2.server import apply_ast_invocation


def _args(**overrides):
    values = dict(model="/m/target", ast=True, ast_drafter=None, ast_topology=None,
                  ast_graft_pld=False, draft_model=None, external_draft=False,
                  native_mtp=False, prompt_lookup=False)
    values.update(overrides)
    return SimpleNamespace(**values)


def _resolution(*, tiles, mtp):
    adapter = type("Adapter", (), {"ast_tree_tiles": tiles})
    return SimpleNamespace(adapter_type=adapter, artifact={"has_mtp": mtp},
                           default_route="native_mtp" if mtp else "ordinary")


def test_tree_family_defaults_to_its_own_mtp_head():
    args = _args()
    policy = apply_ast_invocation(args, None, _resolution(tiles=True, mtp=True))
    assert policy == {"draft_model": "/m/target"} and args.external_draft


def test_tree_family_without_a_head_uses_prompt_lookup_tiles():
    args = _args(ast_topology="5x3", ast_graft_pld=False)
    policy = apply_ast_invocation(args, None, _resolution(tiles=True, mtp=False))
    assert policy == {"draft_model": "prompt_lookup", "ast_topology": "5x3"}


def test_tree_family_external_drafter_gets_tile_budgets_and_graft():
    args = _args(ast_drafter="/m/dflash", ast_graft_pld=True)
    policy = apply_ast_invocation(args, {}, _resolution(tiles=True, mtp=True))
    assert policy["draft_model"] == "/m/dflash"
    assert policy["batch_size_route"] == "tree15_b1_b4_chain_b5plus_v1"
    assert policy["ast_graft_prompt_lookup"] is True


@pytest.mark.parametrize("mtp, flag", [(True, "native_mtp"), (False, "prompt_lookup")])
def test_chain_family_takes_its_best_chain_route(mtp, flag):
    args = _args()
    assert apply_ast_invocation(args, None, _resolution(tiles=False, mtp=mtp)) == {}
    assert getattr(args, flag) and not args.external_draft


def test_ast_options_require_ast_and_refuse_conflicts():
    with pytest.raises(ValueError, match="require --ast"):
        apply_ast_invocation(_args(ast=False, ast_topology="3x5"), None, _resolution(tiles=True, mtp=True))
    with pytest.raises(ValueError, match="no native MTP head"):
        apply_ast_invocation(_args(ast_drafter="mtp"), None, _resolution(tiles=True, mtp=False))
    with pytest.raises(ValueError, match="AST topology"):
        apply_ast_invocation(_args(ast_topology="9x9"), None, _resolution(tiles=True, mtp=True))
    with pytest.raises(ValueError, match="--ast-drafter"):
        apply_ast_invocation(_args(), {"draft_model": "/x"}, _resolution(tiles=True, mtp=True))


def test_attention_family_auto_drafter_takes_the_pld_route_explicit_keeps_tiles():
    adapter = type("Adapter", (), {"ast_tree_tiles": True, "attention_tree_tiles": True})
    resolution = SimpleNamespace(adapter_type=adapter, artifact={"has_mtp": False},
                                 default_route="ordinary")
    args = _args()
    assert apply_ast_invocation(args, None, resolution) == {}
    assert args.prompt_lookup and not args.external_draft
    args = _args(ast_drafter="pld")
    assert apply_ast_invocation(args, None, resolution) == {"draft_model": "prompt_lookup"}
    assert args.external_draft


def _artifact(tmp_path, dtype):
    import mlx.core as mx

    tensors = {
        "language_model.model.layers.0.linear_attn.conv1d.weight": mx.zeros((4, 1, 4), dtype=dtype),
        "language_model.model.layers.0.input_layernorm.weight": mx.zeros((4,), dtype=dtype),
        "language_model.model.layers.0.mlp.down_proj.weight": mx.zeros((4, 4), dtype=mx.bfloat16),
        "language_model.mtp.pre_fc_norm_hidden.weight": mx.zeros((4,), dtype=mx.float32),
    }
    mx.save_safetensors(str(tmp_path / "model.safetensors"), tensors)
    return str(tmp_path)


def test_non_bf16_norm_or_conv_tensors_keep_the_chain_route(tmp_path):
    """GGUF conversions store norms/conv in F32 (OrcaSAQ); the bf16 tree
    kernels cannot run them, so --ast takes the chain route and says why."""
    import mlx.core as mx

    for dtype in (mx.float32, mx.float16):
        target = tmp_path / str(dtype).rsplit(".", 1)[-1]
        target.mkdir()
        path = _artifact(target, dtype)
        args = _args(model=path, ast_drafter="/m/dflash")
        policy = apply_ast_invocation(args, {}, _resolution(tiles=True, mtp=True))
        assert policy == {"draft_model": "/m/dflash"} and args.external_draft
        assert "batch_size_route" not in policy
        assert args.ast_route["tree_tiles"] is False
        assert "conv1d" in args.ast_route["tree_tiles_refused"] or "layernorm" in args.ast_route["tree_tiles_refused"]
        args = _args(model=path)
        assert apply_ast_invocation(args, None, _resolution(tiles=True, mtp=True)) == {}
        assert args.native_mtp and not args.external_draft
        with pytest.raises(ValueError, match="bf16"):
            apply_ast_invocation(_args(model=path, ast_tiles=2), None, _resolution(tiles=True, mtp=True))


def test_bf16_artifact_keeps_tree_tiles_and_mtp_f32_is_ignored(tmp_path):
    import mlx.core as mx

    args = _args(model=_artifact(tmp_path, mx.bfloat16))
    assert apply_ast_invocation(args, None, _resolution(tiles=True, mtp=True)) == {"draft_model": args.model}
    assert args.ast_route["tree_tiles"] is True and "tree_tiles_refused" not in args.ast_route


def test_model_gate_reads_live_parameter_dtypes():
    import mlx.core as mx
    import mlx.nn as nn

    from mlx2.adapters.tensorfold_dtype_gate import model_tree_tile_dtype_issue

    class Layer(nn.Module):
        def __init__(self, dtype):
            super().__init__()
            self.input_layernorm = nn.RMSNorm(4)
            self.input_layernorm.weight = self.input_layernorm.weight.astype(dtype)

    assert model_tree_tile_dtype_issue(Layer(mx.bfloat16)) is None
    assert "float32" in model_tree_tile_dtype_issue(Layer(mx.float32))

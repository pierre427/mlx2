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


def test_adapter_drafter_defaults_fill_only_unnamed_keys():
    seen = []

    def defaults(cls, drafter):
        seen.append(drafter)
        return {} if drafter in ("mtp", "pld") else {
            "draft_kernel": "lane", "draft_quantization": {"bits": 4, "group_size": 64},
        }

    adapter = type("Adapter", (), {
        "ast_tree_tiles": True, "ast_drafter_policy_defaults": classmethod(defaults),
    })
    resolution = SimpleNamespace(adapter_type=adapter, artifact={"has_mtp": True},
                                 default_route="native_mtp")
    args = _args(ast_drafter="/m/dflash")
    policy = apply_ast_invocation(args, {"draft_quantization": None}, resolution)
    # An operator key (null included) wins; the rest is filled and receipted.
    assert policy["draft_kernel"] == "lane" and policy["draft_quantization"] is None
    assert args.ast_route["drafter_policy_defaults"] == {"draft_kernel": "lane"}
    args = _args()
    assert apply_ast_invocation(args, None, resolution) == {"draft_model": "/m/target"}
    assert "drafter_policy_defaults" not in args.ast_route
    assert seen == ["/m/dflash", "mtp"]


def _head_resolution(tmp_path, *, hook=True):
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter

    head = tmp_path / "head"
    head.mkdir(exist_ok=True)
    (head / "config.json").write_text('{"model_type": "qwen3_5_mtp"}')
    seen = {}

    def resolve(resolution, path):
        seen["head"] = str(path)
        return SimpleNamespace(adapter_type=resolution.adapter_type, artifact={"has_mtp": True},
                               default_route="native_mtp")

    attributes = {"ast_tree_tiles": True,
                  "is_external_mtp_head": staticmethod(Qwen3827BAdapter.is_external_mtp_head)}
    if hook:
        attributes["external_mtp_head_resolution"] = staticmethod(resolve)
    adapter = type("Adapter", (), attributes)
    resolution = SimpleNamespace(adapter_type=adapter, artifact={"has_mtp": False},
                                 default_route="ordinary")
    return head, resolution, seen


def test_ast_drafter_naming_an_mtp_head_attaches_it_as_the_native_head(tmp_path):
    from mlx2.server import apply_external_mtp_head

    head, resolution, seen = _head_resolution(tmp_path)
    args = _args(ast_drafter=str(head), mtp_head=None)
    policy, resolution = apply_external_mtp_head(args, None, resolution)
    assert seen["head"] == str(head) and policy == {"mtp_head": str(head.resolve())}
    assert args.ast_drafter == "mtp" and resolution.artifact["has_mtp"]
    policy = apply_ast_invocation(args, policy, resolution)
    # The tile drafter is the target's (now attached) native head.
    assert policy == {"mtp_head": str(head.resolve()), "draft_model": "/m/target"}
    assert args.external_draft


def test_mtp_head_flag_without_a_hook_or_with_a_conflict_fails_closed(tmp_path):
    from mlx2.server import apply_external_mtp_head

    head, resolution, _ = _head_resolution(tmp_path, hook=False)
    with pytest.raises(ValueError, match="does not attach external MTP heads"):
        apply_external_mtp_head(_args(ast=False, mtp_head=str(head)), None, resolution)
    head, resolution, _ = _head_resolution(tmp_path)
    with pytest.raises(ValueError, match="differ"):
        apply_external_mtp_head(_args(ast=False, mtp_head=str(head)), {"mtp_head": "/elsewhere"}, resolution)
    # An ordinary DFlash2 path after --ast-drafter is not an MTP head.
    args = _args(ast_drafter=str(tmp_path), mtp_head=None)
    assert apply_external_mtp_head(args, None, resolution) == (None, resolution)
    assert args.ast_drafter == str(tmp_path)


def test_attached_head_on_the_27b_target_takes_no_artifact_drafter_defaults(tmp_path):
    # Qwen3.8-27B's --ast drafter defaults (q4 lane DFlash2) are measured for
    # an artifact drafter only.  A head named by --ast-drafter becomes the
    # native head ("mtp"), so it must get the same (empty) defaults as a
    # colocated head, not the DFlash2 lane arithmetic.
    from mlx2.adapters.qwen38_27b import QWEN38_27B, Qwen3827BAdapter
    from mlx2.server import apply_external_mtp_head

    head, resolution, _ = _head_resolution(tmp_path)
    seen = []
    real = Qwen3827BAdapter.__dict__["ast_drafter_policy_defaults"].__func__

    def defaults(cls, drafter):
        seen.append(drafter)
        return real(cls, drafter)

    adapter = resolution.adapter_type
    adapter.descriptor = QWEN38_27B
    adapter.ast_drafter_policy_defaults = classmethod(defaults)
    args = _args(ast_drafter=str(head), mtp_head=None)
    policy, resolution = apply_external_mtp_head(args, None, resolution)
    policy = apply_ast_invocation(args, policy, resolution)
    assert seen == ["mtp"]
    assert policy == {"mtp_head": str(head.resolve()), "draft_model": "/m/target"}
    assert "drafter_policy_defaults" not in args.ast_route
    assert args.ast_route["drafter"] == "mtp" and args.ast_route["tree_tiles"]

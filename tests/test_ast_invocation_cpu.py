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

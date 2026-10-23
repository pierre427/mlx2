"""Adapter-owned numerical geometry; no scheduler/model-name dispatch."""
from mlx2.adapters.qwen36_35b import Qwen3635BA3BAdapter
from mlx2.adapters.qwen38_27b import _native_mtp_tree_policy


def test_qwen36_ast_default_freezes_timing_sensitive_geometry():
    policy = _native_mtp_tree_policy({"draft_model": "target"}, numerical_defaults=Qwen3635BA3BAdapter.native_tree_numerical_defaults)
    assert policy["ast_adaptive_rows"] is False
    assert policy["cost_per_commit_handoff"] is False
    assert policy["batch_size_route"] == "tree15_b1_b4_chain_b5plus_v1"


def test_other_adapters_retain_prior_defaults_and_explicit_options_win():
    legacy = _native_mtp_tree_policy({"draft_model": "target"})
    assert "ast_adaptive_rows" not in legacy
    assert "cost_per_commit_handoff" not in legacy
    selected = _native_mtp_tree_policy({"draft_model": "target", "ast_adaptive_rows": True, "cost_per_commit_handoff": True}, numerical_defaults=Qwen3635BA3BAdapter.native_tree_numerical_defaults)
    assert selected["ast_adaptive_rows"] is True
    assert isinstance(selected["cost_per_commit_handoff"], dict)


def test_default_map_is_never_mutated():
    original = dict(Qwen3635BA3BAdapter.native_tree_numerical_defaults)
    result = _native_mtp_tree_policy({"draft_model": "target"}, numerical_defaults=original)
    result["ast_adaptive_rows"] = True
    assert original == Qwen3635BA3BAdapter.native_tree_numerical_defaults

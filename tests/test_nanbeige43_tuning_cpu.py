"""Candidate binding/scheduling contracts; no model qualification claim."""
import hashlib
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx2.adapters import nanbeige43_tuning as tuning


@pytest.fixture(autouse=True)
def evidence(record_property):
    record_property("checkpoint", "synthetic-nanbeige43-tuning@sha256:" + hashlib.sha256(b"tuning-v1").hexdigest())
    root = Path(__file__).resolve().parents[1]
    record_property("source_revision", subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip())
    record_property("working_source_sha256", hashlib.sha256(
        (root / "src/mlx2/adapters/nanbeige43.py").read_bytes()
        + (root / "src/mlx2/adapters/nanbeige43_tuning.py").read_bytes()).hexdigest())


def test_context_load_plan_respects_operator_and_active_decode():
    plan = tuning.PrefillPlan()
    assert plan(total_tokens=4096, active_decode_lanes=0, maximum=512) == 512
    assert plan(total_tokens=4096, active_decode_lanes=1, maximum=512) == 128
    assert plan(total_tokens=4096, active_decode_lanes=4, maximum=512) == 64
    assert plan(total_tokens=4096, active_decode_lanes=0, maximum=96) == 96
    assert plan.receipt()["calls"] == 4


@pytest.mark.parametrize("value", [{"tree":"implicit"}, {"kernels":{"q8_swiglu":True}},
                                    {"lean_greedy":1}, {"kv_reserve_tokens":8193},
                                    {"adaptive_pld":True}, {"unknown":True}])
def test_candidate_options_fail_closed_without_required_route(value):
    with pytest.raises((ValueError, TypeError)):
        tuning.split_tuning_policy({"nanbeige":value})


def test_trained_payload_binding_not_just_matching_dimensions(monkeypatch):
    from mlx2.adapters import dflash2
    args = SimpleNamespace(hidden_size=7168, vocab_size=166912, num_target_layers=52,
                           block_size=8, num_hidden_layers=3, head_dim=128,
                           num_attention_heads=28, num_key_value_heads=4,
                           rope_theta=120000000, rms_norm_eps=1e-5, target_layer_ids=[1,25,49])
    record = {"args":args, "config":{"dflash_config":{}},
              "weight_sha256":[tuning.PUBLISHED_DRAFT_WEIGHT_SHA256]}
    monkeypatch.setattr(dflash2,"inspect_drafter",lambda *_:record)
    assert tuning.inspect_learned_draft("unused","unused") is record
    record["weight_sha256"] = ["0"*64]
    with pytest.raises(ValueError, match="payload"):
        tuning.inspect_learned_draft("unused","unused")
    record["weight_sha256"] = [tuning.PUBLISHED_DRAFT_WEIGHT_SHA256]
    args.target_layer_ids = [2,26,50]
    with pytest.raises(ValueError, match="post-layer"):
        tuning.inspect_learned_draft("unused","unused")


def test_learned_loader_exposes_provider_role_for_standard_diagnostics(monkeypatch):
    from mlx2.adapters import dflash2
    draft = SimpleNamespace()
    monkeypatch.setattr(dflash2, "load_drafter", lambda *_, **__: draft)
    assert tuning.load_learned_draft({}, object()) is draft
    assert draft.receipt_kind == "external_dflash2"


def test_wall_cost_history_cannot_select_a_different_target_tree_law():
    from mlx2.adapters.nanbeige43 import split_policy

    _, policy = split_policy({"draft_model": "pld"})
    assert policy["ast_adaptive_rows"] is False
    with pytest.raises(ValueError, match="tree-size invariant"):
        split_policy({"draft_model": "pld", "ast_adaptive_rows": True})


def test_capability_declarations_refuse_incompatible_projection_and_cache_owners():
    from mlx2.adapters.nanbeige43 import Nanbeige43Adapter

    adapter = object.__new__(Nanbeige43Adapter)
    adapter.nanbeige_tuning = {"int8_prefill": True}
    assert adapter.int8_prefill_supported() == ("mlp", "all")
    adapter.nanbeige_tuning["kernels"] = {"compiled_blocks": True}
    assert adapter.int8_prefill_supported() == ()
    adapter.nanbeige_tuning = {"kv_reserve_tokens": 1024}
    assert adapter.approximate_kv_operations() == {}
    adapter.nanbeige_tuning = {}
    assert "kv_q8" in adapter.approximate_kv_operations()

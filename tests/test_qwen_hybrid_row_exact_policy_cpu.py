"""Adapter wiring of the Qwen hybrid row-exact (exact AST) external route.

Policy admission, refusals, batch/receipt/identity binding and the shipped
smoke policies, without loading any artifact weights.  Implemented and
unqualified; GPU evidence is pending.
"""

import json
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import pytest
from test_qwen38_dflash2_cpu import tiny_target
from test_qwen_progressive_external_portability_cpu import tiny_qwen36

from mlx2.adapters.external_draft_policy import normalize_progressive_external_policy
from mlx2.adapters.qwen36_35b import EXTERNAL_POLICY_KEYS as QWEN36_KEYS
from mlx2.adapters.qwen36_35b import (
    Qwen3635BA3BAdapter,
    configure_environment,
    partition_external_target_policy,
)
from mlx2.adapters.qwen38_27b import EXTERNAL_POLICY_KEYS as QWEN38_KEYS
from mlx2.adapters.qwen38_27b import (
    Qwen3827BAdapter,
    descriptor_for,
    validate_target_verify_row_exact,
)
from mlx2.runtime.models import qwen_hybrid_row_exact as QHR
from mlx2.runtime.progressive_external_verify import (
    PROGRESSIVE_FIDELITY_EXACT,
    PROGRESSIVE_TARGET_PROTOCOL_VERSION,
    progressive_target_protocol_binding,
)

POLICIES = Path(__file__).resolve().parents[1] / (
    "qualification/runs/ast-smokes-20261018/policies"
)
AST = {
    "draft_model": "/draft",
    "draft_revision": "d" * 64,
    "target_revision": "t" * 64,
    "num_draft": 7,
    "proposal_composition": False,
    "continuation_pool": {"sources": ["external"], "limit": 15},
    "target_verify_row_exact": True,
    "progressive_verification_tile": 3,
    "progressive_multilane_draft_cap": 4,
    "progressive_alternative_branches": 4,
    "progressive_alternative_strategy": "confidence_tree",
    "progressive_alternative_depth_policy": "confidence_tail",
    "progressive_cross_session_pack_rows": 8,
    "progressive_cross_session_pack_mode": "ordered_live_frontier",
}


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _normalized(policy=AST):
    return normalize_progressive_external_policy(
        policy, num_draft=policy["num_draft"], minimum_proposals=3
    )


def test_ast_policy_is_admitted_by_both_adapters():
    assert set(AST) <= QWEN38_KEYS
    assert set(AST) <= QWEN36_KEYS
    # Qwen3.6 has no approximate target protocol: fidelity stays refused.
    assert "progressive_verification_fidelity" not in QWEN36_KEYS
    normalized = _normalized()
    assert normalized["progressive_verification_fidelity"] == PROGRESSIVE_FIDELITY_EXACT
    assert normalized["proposal_composition"] is False
    assert validate_target_verify_row_exact(normalized) is True
    # Not selected and no exact tile: nothing to admit (default stays off).
    assert validate_target_verify_row_exact({"draft_model": "/draft"}) is False


def test_qwen36_attention_window_is_target_owned_and_requires_row_exact():
    with pytest.raises(ValueError, match="requires target_verify_row_exact"):
        partition_external_target_policy({"attention_window_row_exact": True})
    with pytest.raises(ValueError, match="must be boolean"):
        partition_external_target_policy(
            {
                "target_verify_row_exact": True,
                "attention_window_row_exact": "yes",
            }
        )
    external, target = partition_external_target_policy(
        {**AST, "attention_window_row_exact": True}
    )
    assert "attention_window_row_exact" not in external
    assert target == {"attention_window_row_exact": True}
    environment = configure_environment(target, "off")
    assert environment["MLX_QWEN36_ATTN_WINDOW_ROW_EXACT"] == "1"


@pytest.mark.parametrize(
    "change, message",
    [
        ({"target_verify_row_exact": "true"}, "must be a boolean"),
        ({"target_verify_row_exact": False}, "requires target_verify_row_exact"),
        ({"progressive_verification_tile": None}, "requires progressive_verification_tile"),
        (
            {"progressive_verification_fidelity": "near_tie_bounded"},
            "approximate progressive",
        ),
        ({"batch_size_route": "tree15_b1_chain_b2plus_v1"}, "tree15/TensorFold"),
        ({"tensorfold_cohort_limit": 1}, "tree15/TensorFold"),
        ({"pairwise_selection": "batched"}, "host pairwise"),
    ],
)
def test_unsupported_combinations_are_refused(change, message):
    policy = {**AST, **change}
    policy = {key: value for key, value in policy.items() if value is not None}
    with pytest.raises(ValueError, match=message):
        validate_target_verify_row_exact(policy)


def test_approximate_27b_candidate_keeps_its_separate_contract():
    policy = {
        "draft_model": "/draft",
        "num_draft": 3,
        "progressive_verification_tile": 2,
        "progressive_verification_fidelity": "near_tie_bounded",
    }
    assert validate_target_verify_row_exact(policy) is False


@pytest.mark.parametrize("cls", [Qwen3827BAdapter, Qwen3635BA3BAdapter])
def test_policy_refusals_happen_before_any_artifact_load(cls, tmp_path):
    exact_without_selection = {
        key: value for key, value in AST.items() if key != "target_verify_row_exact"
    }
    with pytest.raises(ValueError, match="requires target_verify_row_exact"):
        cls(str(tmp_path / "target"), execution_policy=exact_without_selection)
    # Qwen3.6 has no tree route at all (unknown key); Qwen3.8 refuses the pair.
    with pytest.raises(ValueError, match="tree15/TensorFold|unknown keys"):
        cls(
            str(tmp_path / "target"),
            execution_policy={**AST, "batch_size_route": "tree15_b1_chain_b2plus_v1"},
        )


def test_qwen36_expert_streaming_refuses_the_ast_route(tmp_path):
    from mlx2.runtime.streamed_load import WeightStreamRequest

    with pytest.raises(ValueError, match="expert streaming refuses"):
        Qwen3635BA3BAdapter(
            str(tmp_path / "target"),
            execution_policy=AST,
            weight_streaming=WeightStreamRequest(
                mode="moe_experts", budget_bytes=1 << 30, read_workers=2
            ),
        )


def _bound_adapter(cls, model, monkeypatch):
    model.configure_target_verify_row_exact(True)
    adapter = object.__new__(cls)
    adapter.model = model
    adapter.external_policy = _normalized()
    adapter.draft_model = type(
        "BoundContinuation", (), {"policy": object(), "last_continuation_selections": ()}
    )()
    adapter.identity = {"fingerprint": "test"}
    adapter._progressive_external_binding = progressive_target_protocol_binding(
        model, PROGRESSIVE_FIDELITY_EXACT
    )
    calls = []
    monkeypatch.setattr(
        "mlx2.runtime.external_speculative.ExternalDraftBatchGenerator",
        lambda *args, **kwargs: calls.append(kwargs) or object(),
    )
    return adapter, calls


@pytest.mark.parametrize(
    "cls, factory",
    [(Qwen3827BAdapter, tiny_target), (Qwen3635BA3BAdapter, tiny_qwen36)],
)
def test_ast_selection_reaches_batch_and_execution_config(cls, factory, monkeypatch):
    adapter, calls = _bound_adapter(cls, factory(), monkeypatch)
    adapter.create_external_batch()
    kwargs = calls[-1]
    assert kwargs["continuation_pool_ast_bridge"] is True
    assert kwargs["continuation_pool"] is adapter.draft_model.policy
    assert kwargs["progressive_verification_tile"] == 3
    assert kwargs["multilane_draft_cap"] == 4
    assert kwargs["progressive_alternative_branches"] == 4
    assert kwargs["progressive_alternative_strategy"] == "confidence_tree"
    assert kwargs["progressive_alternative_depth_policy"] == "confidence_tail"
    assert kwargs["progressive_ast_rootless_fallback"] is True
    assert kwargs["progressive_cross_session_pack_rows"] == 8
    assert kwargs["progressive_cross_session_pack_mode"] == "ordered_live_frontier"
    assert kwargs["progressive_verification_fidelity"] == PROGRESSIVE_FIDELITY_EXACT
    assert kwargs["progressive_target_protocol"] == PROGRESSIVE_TARGET_PROTOCOL_VERSION
    config = adapter.execution_config(max_lanes=4, prefill_step=16)
    assert config["target_verify_row_exact"] is True
    assert config["continuation_pool_ast_bridge"] is True
    assert config["progressive_verification_fidelity"] == PROGRESSIVE_FIDELITY_EXACT
    assert config["progressive_ast_rootless_fallback"] is True
    assert config["progressive_target_protocol"] == PROGRESSIVE_TARGET_PROTOCOL_VERSION

    adapter.external_policy["progressive_ast_rootless_fallback"] = False
    adapter.create_external_batch()
    assert calls[-1]["progressive_ast_rootless_fallback"] is False
    assert adapter.execution_config(max_lanes=4, prefill_step=16)[
        "progressive_ast_rootless_fallback"
    ] is False

    # Without a pool the bridge stays off.
    adapter.external_policy = {
        key: value
        for key, value in adapter.external_policy.items()
        if key != "continuation_pool"
    }
    adapter.draft_model = object()
    adapter.create_external_batch()
    assert "continuation_pool_ast_bridge" not in calls[-1]
    assert "continuation_pool_ast_bridge" not in adapter.execution_config(
        max_lanes=4, prefill_step=16
    )


def test_route_identity_and_metadata_bind_the_selection():
    model = tiny_target()
    adapter = object.__new__(Qwen3827BAdapter)
    adapter.model = model
    adapter.external_policy = _normalized()
    with pytest.raises(ValueError, match="not installed"):
        adapter._external_execution_numerics()
    adapter.model.configure_target_verify_row_exact(True)
    contract = adapter._external_execution_numerics()["target_verify_row_exact"]
    assert contract["algorithm"] == QHR.ALGORITHM
    assert set(contract["source_hashes"]) == set(QHR.SOURCE_PATHS)
    assert contract["static_refusals"] == {}
    adapter.external_policy = {"draft_model": "/draft"}
    assert "target_verify_row_exact" not in adapter._external_execution_numerics()

    adapter.target_verify_row_exact = QHR.installed(model)
    adapter.descriptor = descriptor_for(has_mtp=False)
    adapter._describe_target_verify_row_exact()
    metadata = adapter.descriptor.metadata["target_verify_row_exact"]
    assert metadata == {
        "algorithm": QHR.ALGORITHM,
        "implemented": True,
        "qualified": False,
        "selected": True,
        "gpu_evidence": QHR.GPU_EVIDENCE,
        "performance_claim": False,
    }


def test_qwen36_numerics_contract_binds_only_a_selected_route():
    from mlx2.runtime.models.qwen36_35b import GatedDeltaNet

    model = tiny_qwen36()
    adapter = object.__new__(Qwen3635BA3BAdapter)
    adapter.model = model
    adapter._kernels = {}
    adapter.moe_nax_gather = "off"
    adapter.external_policy = {"draft_model": "/draft"}
    plain = adapter.execution_numerics_contract()["qwen36_target"]
    assert "target_verify_row_exact" not in plain
    adapter.external_policy = _normalized()
    model.configure_target_verify_row_exact(True)
    QHR.installed(model).set_attention_window_row_exact(True)
    selected = adapter.execution_numerics_contract()["qwen36_target"]
    assert selected["target_verify_row_exact"]["algorithm"] == QHR.ALGORITHM
    assert selected["target_verify_row_exact"]["attention_window_row_exact"] is True
    assert {k: v for k, v in selected.items() if k != "target_verify_row_exact"} == plain
    assert any(isinstance(m, GatedDeltaNet) for _, m in model.named_modules())


def test_qwen36_adapter_sets_the_installed_attention_window_selection():
    model = tiny_qwen36()
    adapter = object.__new__(Qwen3635BA3BAdapter)
    adapter.model = model
    adapter.external_policy = _normalized()
    adapter._kernels = {"attention_window_row_exact": True}
    adapter._select_target_verify_row_exact()
    route = QHR.installed(model)
    assert adapter.target_verify_row_exact is route
    assert route.attention_window_row_exact is True
    assert route.status()["attention_window_row_exact"] is True
    assert route.receipt()["target_verify_row_exact"][
        "attention_window_row_exact"
    ] is True


def test_selected_route_refuses_dynamic_lora():
    from mlx2.serving import row_exact_target_mutation_guard

    model = tiny_target()
    adapter = type("Adapter", (), {})()
    adapter.model = model
    row_exact_target_mutation_guard(adapter, lora=True)  # not selected: no-op
    model.configure_target_verify_row_exact(True)
    with pytest.raises(ValueError, match="dynamic LoRA"):
        row_exact_target_mutation_guard(adapter, lora=True)


@pytest.mark.parametrize(
    "name, keys",
    [
        ("qwen36-35b-ast.json", QWEN36_KEYS),
        ("qwen36-35b-heretic-4bit-ast.json", QWEN36_KEYS),
        ("qwen36-35b-ast-q8-moe.json", QWEN36_KEYS),
        ("qwen36-35b-ast-q8-moe-gdn.json", QWEN36_KEYS),
        ("qwen36-35b-ast-q8-moe-attn.json", QWEN36_KEYS),
        ("qwen38-27b-ast.json", QWEN38_KEYS),
    ],
)
def test_shipped_smoke_policies_parse(name, keys):
    policy = json.loads((POLICIES / name).read_text())
    assert set(policy) <= keys
    for key in ("draft_revision", "target_revision"):
        assert len(policy[key]) == 64
    assert policy["num_draft"] == 7  # DFlash2 block 8
    assert policy["progressive_verification_tile"] < policy["num_draft"]
    normalized = _normalized(policy)
    assert validate_target_verify_row_exact(normalized) is True
    from mlx2.runtime.proposal_providers import ContinuationPoolPolicy

    ContinuationPoolPolicy.from_value(normalized["continuation_pool"])
    reference = {
        key: value
        for key, value in AST.items()
        if key not in ("draft_model", "draft_revision", "target_revision")
    }
    assert {key: policy[key] for key in reference} == reference
    if name in {
        "qwen36-35b-ast-q8-moe.json",
        "qwen36-35b-ast-q8-moe-gdn.json",
        "qwen36-35b-ast-q8-moe-attn.json",
    }:
        assert policy["moe_window_row_exact"] is True
    if name == "qwen36-35b-ast-q8-moe-gdn.json":
        assert policy["fused_gdn_batch_verify"] is True
    if name == "qwen36-35b-ast-q8-moe-attn.json":
        assert policy["attention_window_row_exact"] is True


@pytest.mark.parametrize(
    "name, num_draft, tile, rootless_fallback",
    [
        ("qwen38-27b-ast-rootless-guard.json", 7, 3, False),
        ("qwen38-27b-ast-k3-primary.json", 3, 1, None),
    ],
)
def test_qwen38_guarded_ast_candidate_policies_parse(
    name, num_draft, tile, rootless_fallback
):
    policy = json.loads((POLICIES / name).read_text())
    assert set(policy) <= QWEN38_KEYS
    assert len(policy["draft_revision"]) == 64
    assert len(policy["target_revision"]) == 64
    assert policy["num_draft"] == num_draft
    assert policy["progressive_verification_tile"] == tile
    normalized = _normalized(policy)
    assert validate_target_verify_row_exact(normalized) is True
    if rootless_fallback is None:
        assert "progressive_ast_rootless_fallback" not in policy
        assert "continuation_pool" not in policy
        assert "progressive_alternative_branches" not in policy
    else:
        assert policy["progressive_ast_rootless_fallback"] is rootless_fallback
        from mlx2.runtime.proposal_providers import ContinuationPoolPolicy

        ContinuationPoolPolicy.from_value(normalized["continuation_pool"])


def test_descriptor_replace_keeps_existing_metadata():
    # The metadata helper extends, never replaces, the bound descriptor's.
    adapter = object.__new__(Qwen3827BAdapter)
    adapter.target_verify_row_exact = object()
    adapter.descriptor = replace(
        descriptor_for(has_mtp=False), metadata={"progressive_verification": {"x": 1}}
    )
    adapter._describe_target_verify_row_exact()
    assert adapter.descriptor.metadata["progressive_verification"] == {"x": 1}
    assert adapter.descriptor.metadata["target_verify_row_exact"]["selected"] is True

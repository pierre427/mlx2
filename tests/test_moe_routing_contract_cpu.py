"""Model-owned routed-MoE admission contracts (CPU/source only)."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import mlx.core as mx
import pytest

from mlx2.runtime.models import qwen4_moe_window as window
from mlx2.runtime.models import qwen4_routed_decode as routed
from mlx2.runtime.models.moe_routing_contract import (
    QWEN4_E512_K10_ROUTING,
    QWEN36_E256_K8_ROUTING,
)


def test_contracts_pin_both_proven_router_laws():
    flash = QWEN4_E512_K10_ROUTING
    qwen36 = QWEN36_E256_K8_ROUTING
    assert (flash.expert_count, flash.top_k) == (512, 10)
    assert (qwen36.expert_count, qwen36.top_k) == (256, 8)
    assert flash.selection_law == qwen36.selection_law == (
        "precise_softmax_argpartition"
    )
    assert flash.tie_order == qwen36.tie_order == "stable_ascending_expert"
    assert flash.slot_order == qwen36.slot_order == (
        "ascending_selected_probability"
    )
    assert flash.projection_layouts == frozenset(
        {"fused_gate_up", "split_gate_up"}
    )
    assert qwen36.projection_layouts == frozenset({"split_gate_up"})
    with pytest.raises(FrozenInstanceError):
        flash.top_k = 8


@pytest.mark.parametrize(
    "contract,experts,top_k",
    [
        (QWEN4_E512_K10_ROUTING, 512, 10),
        (QWEN36_E256_K8_ROUTING, 256, 8),
    ],
)
def test_router_admits_only_the_declared_family(monkeypatch, contract, experts, top_k):
    monkeypatch.setattr(routed, "runtime_supported", lambda: True)
    logits = mx.zeros((2, experts), mx.bfloat16)
    assert window.admit_router_topk(
        logits,
        top_k,
        True,
        contract=contract,
        consumer="verify",
    ) is None
    other = 256 if experts == 512 else 512
    refusal = window.admit_router_topk(
        mx.zeros((1, other), mx.bfloat16),
        top_k,
        True,
        contract=contract,
    )
    assert refusal is not None and "declared" in refusal


def test_unknown_and_semantically_mismatched_contracts_fail_closed(monkeypatch):
    monkeypatch.setattr(routed, "runtime_supported", lambda: True)
    logits = mx.zeros((1, 512), mx.bfloat16)
    assert "missing typed" in window.admit_router_topk(
        logits, 10, True, contract=None
    )
    assert "normalization" in window.admit_router_topk(
        logits, 10, False, contract=QWEN4_E512_K10_ROUTING
    )
    wrong_ties = replace(QWEN4_E512_K10_ROUTING, tie_order="not-the-stock-law")
    assert "tie order" in window.admit_router_topk(
        logits, 10, True, contract=wrong_ties
    )


def test_row_consumer_and_projection_layout_are_contract_owned(monkeypatch):
    monkeypatch.setattr(routed, "runtime_supported", lambda: True)
    decode_only = replace(
        QWEN4_E512_K10_ROUTING,
        allowed_row_consumers=frozenset({"batch_decode"}),
    )
    refusal = window.admit_router_topk(
        mx.zeros((2, 512), mx.bfloat16),
        10,
        True,
        contract=decode_only,
        consumer="verify",
    )
    assert refusal is not None and "consumer" in refusal
    assert QWEN36_E256_K8_ROUTING.routing_refusal(
        expert_count=256,
        top_k=8,
        normalize_topk=True,
        activation_dtype="bfloat16",
        projection_layout="fused_gate_up",
    ) == (
        "projection layout 'fused_gate_up' is not allowed by "
        "qwen3.6-35b-a3b-e256-k8-v1"
    )


def test_shared_admission_sources_have_no_model_name_or_geometry_pair_branch():
    window_source = Path(window.__file__).read_text()
    routed_source = Path(routed.__file__).read_text()
    assert "(256, 8)" not in window_source
    assert "qwen36 =" not in routed_source
    assert "model_type" not in window_source

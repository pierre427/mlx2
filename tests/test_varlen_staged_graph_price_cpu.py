"""Fail-closed CPU contract for the paired research B2 ready-step receipt."""

import copy
import json
import sys
from pathlib import Path

import pytest

from mlx2.runtime.paged_pack_price import (
    load_price,
    load_research_graph_b2_price,
)
from mlx2.runtime.paged_pack_scheduler import ReservedRows, choose_paged_pack

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/research"))
import varlen_staged_graph_price as graph_price


def proof(arm):
    value = {"arm": arm, "request_inserted": True, "admitted": True,
             "cache_transaction_published": True, "response_emitted": True,
             "sampled_output": True, "synchronized": True,
             "request_removed": True, "request_state_released": True,
             "ordered_context_tokens": [63, 65],
             "output_token_ids": [[2, 3], [4, 5]], "output_token_id": 5,
             "model_layers": 28, "peak_resident_bytes": 4096,
             "ready_pack_step_ms": 1.0,
             "scheduler_step": copy.deepcopy(graph_price.STEP),
             "sampler_config": copy.deepcopy(graph_price.SAMPLER)}
    if arm == "paged":
        first = {"research_executed": True, "native_read_calls": 56,
                 "terminal_successes": 56}
        second = {"route": "native_qwen3_paged_graph_research",
                  "research_executed": True, "selected": False,
                  "observed_used": False, "packed_lanes": 2,
                  "native_span_counts": [2] * 28,
                  "native_read_delta": 28, "terminal_success_delta": 28,
                  "native_read_calls": 84, "terminal_successes": 84}
        value.update(route_receipt={**second, "serving_selected": False},
                     response_receipts=[[first, {**second,
                                                "published_layer_offsets": [64] * 28}],
                                        [first, {**second,
                                                 "published_layer_offsets": [66] * 28}]],
                     response_execution_widths=[[1, 2], [1, 2]],
                     ordinary_model_forward_calls=0, paged_read_calls=84,
                     prefill_read_calls=56, decode_read_calls=28,
                     terminal_successes=84, pending_native_epochs=0,
                     retained_pages=0)
    else:
        value.update(route_receipt={"route": "ordinary"},
                     ordinary_model_forward_calls=2)
    return value


def test_research_b2_dry_run_and_exact_proofs():
    planned = graph_price.dry_run()
    assert planned["context_tokens"] == [63, 65]
    assert planned["ready_step"]["active_lane_count"] == 2
    assert planned["gpu_executed"] is False and planned["price_usable"] is False
    assert graph_price.validate_proof(proof("paged"), "paged", (63, 65))
    assert graph_price.validate_proof(proof("ordinary"), "ordinary", (63, 65))
    pair = {"order": ["ordinary", "paged"], "ordinary": {"proof": proof("ordinary")},
            "paged": {"proof": proof("paged")}}
    mismatch = graph_price.PairParityFailure(pair, 0)
    assert mismatch.pair["ordinary"]["proof"]["output_token_ids"] == [[2, 3], [4, 5]]
    assert mismatch.case_index == 0
    compact_failure = RuntimeError("paired B2 generator failed in 2 lane(s)")
    compact_failure.physical_counters = {
        "grouped_q1_writes": 0, "native_write_dispatches": 448,
        "pending_native_epochs": 0}
    receipt = graph_price.failure_receipt(compact_failure, {}, "fake:create")
    assert receipt["failure_physical_counters"] == compact_failure.physical_counters
    assert "reads" not in receipt["error"]
    failed = graph_price.failure_receipt(mismatch, {"diagnostic_mode": True}, "fake:create")
    assert failed["status"] == "parity_failed" and failed["gpu_executed"] is True
    assert failed["price_usable"] is False and failed["failed_pair"] == pair
    assert graph_price.failure_receipt(RuntimeError("preflight"), {}, "fake:create")[
        "gpu_executed"] is None


@pytest.mark.parametrize("field,value", [
    ("cache_transaction_published", False), ("request_state_released", False),
    ("ordered_context_tokens", [65, 63]), ("ready_pack_step_ms", 0.0),
    ("ordinary_model_forward_calls", 1), ("decode_read_calls", 56),
    ("paged_read_calls", 56), ("terminal_successes", 83),
    ("pending_native_epochs", 1), ("retained_pages", 1),
])
def test_research_b2_refuses_missing_or_changed_native_evidence(field, value):
    bad = proof("paged")
    bad[field] = value
    with pytest.raises(RuntimeError):
        graph_price.validate_proof(bad, "paged", (63, 65))


def test_research_b2_refuses_serial_width_one_and_price_admission(tmp_path):
    bad = proof("paged")
    bad["response_receipts"][0][1]["packed_lanes"] = 1
    with pytest.raises(RuntimeError, match="two-span"):
        graph_price.validate_proof(bad, "paged", (63, 65))

    identity = {"host": "test", "hardware": "M5", "artifact_sha256": "a" * 64,
                "source_commit": "b" * 40, "source_tree_sha256": "c" * 64,
                "mlx_wheel_version": "test", "mlx_wheel_sha256": "d" * 64,
                "kernel_sha256": "e" * 64}
    data = {"schema": graph_price.SCHEMA, "status": "screened",
            "measurement_scope": graph_price.SCOPE, "gpu_executed": True,
            "identity": identity, "context_tokens": [63, 65],
            "scheduler_step": graph_price.STEP,
            "gpuq_owner": {"session": "test", "lease_id": "test-1", "pid": 1},
            "profile_id": "research-b2", "qualified": False, "selected": False,
            "serving_selected": False, "research_executed": True,
            "research_price_candidate": True,
            "paired_cases": [
                {"order": ["ordinary", "paged"] if i != 1 else ["paged", "ordinary"],
                 "ordinary": {"complete_request_ms": 2.0,
                              "proof": proof("ordinary")},
                 "paged": {"complete_request_ms": 3.0,
                           "proof": proof("paged")}}
                for i in range(3)], "price_usable": False}
    body = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    import hashlib
    data["research_request_evidence_sha256"] = hashlib.sha256(body).hexdigest()
    receipt = tmp_path / "research-b2.json"
    receipt.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_price(receipt, live_identity=identity, context_tokens=(63, 65))
    price = load_research_graph_b2_price(
        receipt, live_identity=identity, context_tokens=(63, 65))
    assert price.validated and price.upper_bound_ms == 1.0
    rows = (ReservedRows(1, "decode", 1, 0, 10.0),
            ReservedRows(2, "decode", 1, 0, 10.0))
    assert choose_paged_pack(rows, (), price=price, row_capacity=2,
                             free_pages=2, permit_candidate=True).reason == \
           "research_graph_price_disabled"
    assert choose_paged_pack(rows, (), price=price, row_capacity=2,
                             free_pages=2, permit_candidate=True,
                             permit_research_graph=True).estimated_ms == 1.0
    edited = copy.deepcopy(data)
    edited["paired_cases"][0]["paged"]["proof"]["response_execution_widths"] = [[1, 1], [1, 1]]
    receipt.write_text(json.dumps(edited))
    with pytest.raises(ValueError, match="digest"):
        load_research_graph_b2_price(receipt, live_identity=identity,
                                     context_tokens=(63, 65))

def test_aligned_b2_control_keeps_exact_context_and_publication_proof():
    aligned = proof("paged")
    aligned["ordered_context_tokens"] = [63, 63]
    aligned["scheduler_step"] = graph_price.step_for((63, 63))
    for receipts in aligned["response_receipts"]:
        receipts[1]["published_layer_offsets"] = [64] * 28
    assert graph_price.contexts_for({"context_tokens": [63, 63]}) == (63, 63)
    assert graph_price.validate_proof(aligned, "paged", (63, 63)) is aligned
    with pytest.raises(RuntimeError):
        graph_price.validate_proof(aligned, "paged", (63, 65))
    for invalid in ([64, 65], [64, 64], [64, 128], [64], [True, 64]):
        with pytest.raises(RuntimeError):
            graph_price.contexts_for({"context_tokens": invalid})

"""Fail-closed CPU checks for the live q=1 request price contract."""

import json
import sys
from pathlib import Path

import pytest

from mlx2.runtime.paged_pack_price import evidence_sha256, load_price

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/research"))
import varlen_live_request_price as live


def proof(arm):
    value = {"arm": arm, "request_inserted": True, "admitted": True,
             "cache_transaction_published": True, "response_emitted": True,
             "sampled_output": True, "synchronized": True,
             "request_removed": True, "request_state_released": True,
             "output_token_id": 42, "output_token_ids": [17, 42], "model_layers": 28,
             "peak_resident_bytes": 4096}
    if arm == "paged":
        value.update(route_receipt={"route": "native_qwen3_paged",
                                    "research_executed": True,
                                    "serving_selected": False,
                                    "observed_used": False, "selected": False},
                     ordinary_model_forward_calls=0, paged_read_calls=56,
                     decode_read_calls=28,
                     prefill_read_calls=28,
                     first_response_receipt={"research_executed": True,
                                             "native_read_calls": 28},
                     second_response_receipt={"research_executed": True,
                                              "native_read_calls": 56,
                                              "terminal_successes": 56},
                     terminal_successes=56, pending_native_epochs=0,
                     retained_pages=0)
    else:
        value.update(route_receipt={"route": "ordinary"},
                     ordinary_model_forward_calls=2)
    value["q1_step_ms"] = 1.0
    return value


def test_first_live_cell_is_exact_q1_and_bounded():
    plan = live.dry_run()
    assert plan["context_tokens"] == [63]
    assert plan["request_shape"]["decode_rows_after_prefill"] == 1
    assert plan["paired_repeats"] == 3
    assert plan["hard_seconds"] == 180
    assert plan["max_resident_bytes"] == 24 * 1024**3
    assert plan["gpu_executed"] is False


@pytest.mark.parametrize("field,value", [
    ("request_inserted", False), ("admitted", False),
    ("cache_transaction_published", False), ("response_emitted", False),
    ("sampled_output", False), ("synchronized", False),
    ("request_removed", False), ("request_state_released", False),
    ("model_layers", 27), ("peak_resident_bytes", 24 * 1024**3 + 1),
    ("ordinary_model_forward_calls", 1), ("paged_read_calls", 27),
    ("decode_read_calls", 27), ("q1_step_ms", 0),
    ("terminal_successes", 55), ("pending_native_epochs", 1),
    ("retained_pages", 1),
])
def test_live_native_proof_rejects_partial_path(field, value):
    bad = proof("paged")
    bad[field] = value
    with pytest.raises(RuntimeError):
        live.validate_proof(bad, "paged")


def test_unobserved_and_ordinary_reference_proofs_fail_closed():
    bad = proof("paged")
    bad["route_receipt"]["research_executed"] = False
    with pytest.raises(RuntimeError, match="native route"):
        live.validate_proof(bad, "paged")
    bad = proof("ordinary")
    bad["ordinary_model_forward_calls"] = 0
    with pytest.raises(RuntimeError, match="ordinary live"):
        live.validate_proof(bad, "ordinary")


def test_research_q1_receipt_cannot_load_as_serving_price(tmp_path):
    identity = {"host": "test", "hardware": "M5", "artifact_sha256": "a" * 64,
                "source_commit": "b" * 40, "source_tree_sha256": "c" * 64,
                "mlx_wheel_version": "test", "mlx_wheel_sha256": "d" * 64,
                "kernel_sha256": "e" * 64}
    data = {"schema": live.SCHEMA, "status": "calibrated",
            "measurement_scope": live.SCOPE, "gpu_executed": True,
            "gpuq_owner": {"session": "test", "lease_id": "test-1", "pid": 1},
            "measured_route": "native_qwen3_paged",
            "profile_id": "qwen3-q1", "identity": identity, "context_tokens": [63],
            "cases": [{"reserved": [["decode", 1]], "prefill_rows": 0,
                       "forward_ms": [1.0, 2.0, 3.0],
                       "ordinary_request_ms": [2.0, 3.0, 4.0],
                       "request_proofs": [{"ordinary": proof("ordinary"),
                                           "paged": proof("paged")} for _ in range(3)],
                       "kernel_engagement": {"paged_read_calls": 168,
                                             "terminal_successes": 168}}]}
    data["complete_request_evidence_sha256"] = evidence_sha256(data)
    path = tmp_path / "live-price.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="measured price receipt"):
        load_price(path, live_identity=identity, context_tokens=(63,))

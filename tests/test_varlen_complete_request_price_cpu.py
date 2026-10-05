"""CPU-only gates for the offline Qwen3 paired request cell."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from mlx2.runtime.paged_pack_price import load_price


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/research"
sys.path.insert(0, str(SCRIPTS))
import varlen_complete_request_price as cell
import varlen_qwen3_request_driver as qwen3


def proof(arm="paged"):
    result = {"arm": arm, "admitted": True,
              "cache_transaction_published": True, "sampled_output": True,
              "synchronized": True, "request_state_released": True,
              "queue_state_released": True, "model_layers": 28,
              "peak_resident_bytes": 1024, "output_token_ids": [123]}
    if arm == "paged":
        result.update(paged_read_calls=56, terminal_successes=56,
                      pending_native_epochs=0, retained_pages=0)
    return result


def test_dry_run_is_offline_and_bounded():
    plan = cell.dry_run()
    assert plan["gpu_executed"] is False
    assert plan["measurement_scope"] == "offline_request_probe"
    assert plan["planned_paired_requests"] == 48
    assert plan["hard_seconds"] == 180
    assert plan["max_resident_bytes"] == 24 * 1024**3


@pytest.mark.parametrize("field,value", [
    ("admitted", False), ("cache_transaction_published", False),
    ("sampled_output", False), ("synchronized", False),
    ("request_state_released", False), ("queue_state_released", False),
    ("model_layers", 27), ("peak_resident_bytes", 24 * 1024**3 + 1),
    ("output_token_ids", []), ("paged_read_calls", 27),
    ("terminal_successes", 55), ("pending_native_epochs", 1),
    ("retained_pages", 1),
])
def test_incomplete_paged_request_proof_fails_closed(field, value):
    bad = proof()
    bad[field] = value
    with pytest.raises(RuntimeError):
        cell.validate_proof(bad, arm="paged", expected_layers=28)


def test_exact_context_and_shape_required_before_model_work():
    request = qwen3.Qwen3RequestDriver._requests
    prompts, suffixes = request((("decode", 1),), 17, (63, 65, 129))
    assert tuple(map(len, prompts)) == (63, 65, 129)
    assert tuple(map(len, suffixes)) == (1, 17)
    with pytest.raises(ValueError, match="context"):
        request((("decode", 1),), 17, (63, 65, 130))
    with pytest.raises(ValueError, match="shape"):
        request((("decode", 2),), 17, (63, 65, 129))


def test_gpu_execution_preflight_fails_before_mlx_import(monkeypatch):
    def deny(_manifest):
        raise RuntimeError("source bytes differ")
    monkeypatch.setattr(cell, "preflight_cpu", deny)
    with pytest.raises(RuntimeError, match="source bytes differ"):
        cell.execute({}, "varlen_qwen3_request_driver:make_driver")


def test_offline_probe_receipt_cannot_load_as_serving_price(tmp_path):
    identity = {"host": "test", "hardware": "M5", "artifact_sha256": "a" * 64,
                "source_commit": "b" * 40, "source_tree_sha256": "c" * 64,
                "mlx_wheel_version": "test", "mlx_wheel_sha256": "d" * 64,
                "kernel_sha256": "e" * 64}
    receipt = {"schema": cell.SCHEMA, "status": "measured",
               "measurement_scope": cell.SCOPE, "profile_id": "offline",
               "identity": identity, "context_tokens": [63, 65, 129],
               "offline_request_evidence_sha256": "f" * 64,
               "cases": [{"reserved": [["decode", 1]], "prefill_rows": 0,
                          "forward_ms": [1.0, 2.0, 3.0],
                          "ordinary_request_ms": [1.0, 2.0, 3.0],
                          "kernel_engagement": {"paged_read_calls": 84,
                                                "terminal_successes": 84}}]}
    path = tmp_path / "offline.json"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="complete-request"):
        load_price(path, live_identity=identity, context_tokens=(63, 65, 129))

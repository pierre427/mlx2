"""CPU host accounting for terminal native cleanup in the live driver."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/research"))
from varlen_live_qwen3_request_driver import retired_native_state
from varlen_live_request_price import validate_proof


def writer(epochs=(), ledger=0, allocated=0):
    return SimpleNamespace(
        pending_epochs=tuple(epochs),
        ledger=SimpleNamespace(pending_count=ledger),
        pool=SimpleNamespace(allocated_count=allocated),
    )


def test_matched_terminal_and_release_proves_clean_state():
    pending, retained, released = retired_native_state(writer())
    first = {"route": "native_qwen3_paged", "research_executed": True,
             "selected": False, "observed_used": False,
             "native_read_calls": 28, "terminal_successes": 28}
    second = {**first, "native_read_calls": 56, "terminal_successes": 56,
              "serving_selected": False}
    proof = {
        "arm": "paged", "request_inserted": True, "admitted": True,
        "cache_transaction_published": True, "response_emitted": True,
        "sampled_output": True, "synchronized": True,
        "request_removed": True, "request_state_released": released,
        "output_token_ids": [17, 42], "output_token_id": 42,
        "model_layers": 28, "peak_resident_bytes": 1024,
        "q1_step_ms": 1.0, "route_receipt": second,
        "first_response_receipt": first, "second_response_receipt": second,
        "ordinary_model_forward_calls": 0, "prefill_read_calls": 28,
        "paged_read_calls": 56, "decode_read_calls": 28,
        "terminal_successes": 56,
        "pending_native_epochs": pending, "retained_pages": retained,
    }
    assert validate_proof(proof, "paged") is proof


@pytest.mark.parametrize("state,expected", [
    (writer((1,)), (1, 0, False)),
    (writer(ledger=1), (1, 0, False)),
    (writer(allocated=1), (0, 1, False)),
])
def test_pending_or_retained_native_state_refuses_cleanup_proof(state, expected):
    assert retired_native_state(state) == expected

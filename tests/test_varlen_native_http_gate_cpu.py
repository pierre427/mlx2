"""Pure CPU validation of actual-native HTTP gate receipt checks."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "research"))
from varlen_native_http_gate import (
    _assert_native,
    _assert_refusal,
    _prompt_for_63_tokens,
)


def test_prompt_search_uses_real_tokenizer_and_exact_context():
    class Tokenizer:
        def decode(self, tokens):
            return "x " * len(tokens)

    class Adapter:
        tokenizer = Tokenizer()

        def prompt_tokens(self, request):
            return request["prompt"].split()

    prompt, tokens = _prompt_for_63_tokens(Adapter())
    assert len(tokens) == 63
    assert prompt.split() == tokens


def test_native_receipt_requires_physical_read_terminal_and_honest_state():
    receipt = {
        "route": "native_qwen3_paged", "implemented": True,
        "qualified": False, "selected": True, "observed_used": True,
        "price_provenance": "research_calibrated", "price_evidence_sha256": "a" * 64,
        "apcv2": "native_checkpoint_unavailable", "ordinary_forward_calls": 0,
        "native_reader_lease": "held_through_token_step",
        "native_read_calls": 56, "terminal_successes": 56,
    }
    body = {"mlx2": {"route": "native_qwen3_paged", "cached_tokens": 0,
                     "route_receipt": receipt}}
    assert _assert_native(body, SimpleNamespace(evidence_sha256="a" * 64)) == receipt
    for key, bad in (("qualified", True), ("observed_used", False),
                     ("native_read_calls", 0), ("terminal_successes", 55)):
        altered = {**receipt, key: bad}
        with pytest.raises(RuntimeError):
            _assert_native({"mlx2": {**body["mlx2"], "route_receipt": altered}},
                           SimpleNamespace(evidence_sha256="a" * 64))


def test_refusal_requires_http_400_and_reason():
    _assert_refusal(400, {"error": {"message": "cold no-APCv2 route"}},
                    "cold no-APCv2")
    with pytest.raises(RuntimeError):
        _assert_refusal(200, {"error": {"message": "cold no-APCv2 route"}},
                        "cold no-APCv2")

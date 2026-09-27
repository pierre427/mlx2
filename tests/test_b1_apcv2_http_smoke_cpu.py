"""CPU-only contract checks for the bounded APCv2 HTTP smoke harness."""

import importlib.util
import io
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/smoke_b1_apcv2_http.py"
spec = importlib.util.spec_from_file_location("smoke_b1_apcv2_http", SCRIPT)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def _row(cached):
    return {
        "text": "APCv2 retains exact state.", "token_ids": [12, 13],
        "token_arrival_s": [0.3, 0.4], "first_token_s": 0.3,
        "receipt": {"cache": "apcv2", "cached_tokens": cached,
                    "route": "ordinary", "ordinary_compute_width": 1,
                    "route_receipt": "candidate_validation", "prompt_tokens": 800},
    }


def test_validate_requires_cold_zero_warm_hit_and_parity():
    assert smoke.validate(_row(0), _row(799))["passed"]
    assert not smoke.validate(_row(1), _row(799))["passed"]
    assert not smoke.validate(_row(0), _row(0))["passed"]
    changed = _row(799)
    changed["token_ids"] = [12, 14]
    assert "cold/warm token IDs differ" in smoke.validate(_row(0), changed)["failures"]
    changed = _row(799)
    changed["receipt"]["ordinary_compute_width"] = 2
    assert not smoke.validate(_row(0), changed)["passed"]


def test_stream_one_records_token_ids_receipt_and_interarrival():
    payload = b"".join([
        b'data: {"choices":[{"delta":{},"logprobs":{"content":[{"id":12}]}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"A"}}]}\n\n',
        b'data: {"choices":[{"delta":{},"logprobs":{"content":[{"id":13}]}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"B"}}]}\n\n',
        b'data: {"choices":[{"delta":{},"finish_reason":"length"}],"mlx2":{"cache":"apcv2"}}\n\n',
        b'data: [DONE]\n\n',
    ])

    class Response(io.BytesIO):
        status = 200

    with patch.object(smoke.urllib.request, "urlopen", return_value=Response(payload)):
        result = smoke.stream_one("http://127.0.0.1:8393", {"stream": True}, timeout=5)
    assert result["token_ids"] == [12, 13]
    assert result["text"] == "AB"
    assert result["receipt"]["cache"] == "apcv2"
    assert len(result["inter_token_s"]) == 1
    assert result["first_token_s"] is not None

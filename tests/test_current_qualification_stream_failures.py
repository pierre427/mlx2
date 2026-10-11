"""CPU contract for bounded public qualification-ladder request evidence."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "qualification/runs/qualify-1010-correctness"

@pytest.fixture
def ladder(monkeypatch):
    old_root = os.environ.get("MLX2_CAMPAIGN_ROOT")
    monkeypatch.setenv("MLX2_CAMPAIGN_ROOT", str(ROOT))
    monkeypatch.syspath_prepend(str(RUN))
    spec = importlib.util.spec_from_file_location("_test_public_stream_ladder", RUN / "thermal_ladder.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)
    if old_root is None: os.environ.pop("MLX2_CAMPAIGN_ROOT", None)
    else: os.environ["MLX2_CAMPAIGN_ROOT"] = old_root

def response(monkeypatch, events):
    class Response:
        status = 200
        def __enter__(self): return iter([("data: " + json.dumps(e)).encode() if not isinstance(e, bytes) else e for e in events])
        def __exit__(self, *_): return False
    monkeypatch.setattr("urllib.request.urlopen", lambda *_a, **_k: Response())

def test_transport_failure_retains_prior_event_and_is_not_functional(ladder, monkeypatch):
    client = ladder.Stream("http://example.invalid", 1, "model")
    class Response:
        status = 200
        def __enter__(self):
            def stream():
                yield b'data: {"choices":[{"delta":{"content":"needle"}}]}'
                raise OSError("stream reset")
            return iter(stream())
        def __exit__(self, *_): return False
    monkeypatch.setattr("urllib.request.urlopen", lambda *_a, **_k: Response())
    row = client.request("prompt", 2)
    assert row["content"] == "needle"
    assert row["failure_class"] == "transport"
    assert row["raw_sse_diagnostics"]["event_count"] == 1
    assert ladder.summarize_run({"rows":[row]}, {"rows":[row]}, ["needle"])["functional_failures"] == []

def test_server_error_and_strict_token_counts_are_separate(ladder, monkeypatch):
    client = ladder.Stream("http://example.invalid", 1, "model")
    response(monkeypatch, [{"error":{"message":"interrupted"},"choices":[]}, b"data: [DONE]"])
    row = client.request("prompt", 2)
    assert "server" in row["failure_classes"]
    assert row["server_errors"]
    response(monkeypatch, [{"usage":{"prompt_tokens":True,"completion_tokens":2},"mlx2":{"cached_tokens":0},"choices":[{"finish_reason":"length","delta":{}}]}, b"data: [DONE]"])
    row = client.request("prompt", 2)
    assert "server_protocol" in row["failure_classes"]
    assert any("strict integer" in x for x in row["protocol_errors"])
    assert row["prompt_tokens"] == 0

def test_batch_keeps_sibling_lanes_when_local_request_raises(ladder, monkeypatch):
    client = ladder.Stream("http://example.invalid", 1, "model")
    monkeypatch.setattr(client, "_guarded", lambda f: f())
    def request(prompt, _tokens):
        if prompt == "bad": raise ValueError("formatter defect")
        return {"done": True, "content": prompt, "completion_tokens": 1}
    monkeypatch.setattr(client, "request", request)
    result = client.batch(["bad", "good"], 2)
    assert [r["lane_index"] for r in result["rows"]] == [0, 1]
    assert result["rows"][0]["failure_class"] == "harness_error"
    assert result["rows"][1]["content"] == "good"

def test_pending_attempt_saved_with_prompt_and_refusal_phase_snapshots(ladder):
    cell, snapshots = {}, []
    pending = ladder._persist_pending_attempt(cell, tag="run0", measured=True,
        effective_target_tokens=10, width=1, prompt_recipes=[{"prompt_sha256":"a"}],
        needles=["needle"], save=lambda: snapshots.append(json.loads(json.dumps(cell))))
    phases = {"cold":[], "warm":[]}
    ladder._record_phase_refusal(pending, phases, "cold", 1, [{"http_status":429}], {"busy":True})
    assert snapshots[0]["pending_attempt"]["requests"] == {}
    assert pending["refusal_attempts"]["cold"][0]["rows"][0]["http_status"] == 429
    assert pending["refusal_attempts"]["warm"] == []

def test_sse_and_terminal_diagnostics_stay_bounded(ladder, monkeypatch):
    client = ladder.Stream("http://example.invalid", 1, "model")
    payload = {"error": {"message": "x" * 5000}, "mlx2": {"profile": "p" * 5000}, "choices": []}
    events = [payload] * 100 + [b"data: [DONE]"]
    response(monkeypatch, events)
    row = client.request("prompt", 2)
    diagnostics = row["raw_sse_diagnostics"]
    assert diagnostics["event_count"] == 101
    assert row["diagnostic_chars"] <= ladder.MAX_ROW_DIAGNOSTIC_CHARS
    assert len(diagnostics["samples"]) <= ladder.MAX_SSE_DIAGNOSTIC_EVENTS
    assert all(len(item["payload"]["text"]) <= ladder.MAX_SSE_DIAGNOSTIC_EVENT_CHARS for item in diagnostics["samples"])
    assert len(row["server_errors"]) <= ladder.MAX_SERVER_ERROR_ITEMS
    assert len(row["terminal_receipt_diagnostics"]["excerpt"]) <= ladder.MAX_RECEIPT_DIAGNOSTIC_CHARS

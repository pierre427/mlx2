"""CPU-only guards for the source-bound MTPLX real-child research runner."""

from __future__ import annotations

import argparse
import io
import json

import pytest

from scripts.research import mtplx520_real_child_gate as gate


def test_qualification_preflight_refuses_stale_source_without_importing_mlx(tmp_path):
    source = tmp_path / "mlx2"
    source.mkdir()
    (source / "a.py").write_text("old = 1\n")
    old_hash = gate.source_sha256(source)
    record = tmp_path / "qualification.json"
    record.write_text(json.dumps({"runtime": {"source_sha256": old_hash}, "passed": True}))
    (source / "a.py").write_text("new = 2\n")
    result = gate.qualification_preflight(tmp_path / "model", record, source)
    assert result["record_passed"] is True
    assert result["source_match"] is False
    assert result["qualified_admission"] == "refused_stale_source"


def test_source_match_still_requires_full_live_identity(tmp_path):
    source = tmp_path / "mlx2"
    source.mkdir()
    (source / "a.py").write_text("x = 1\n")
    record = tmp_path / "qualification.json"
    record.write_text(json.dumps({"runtime": {"source_sha256": gate.source_sha256(source)},
                                  "passed": True}))
    result = gate.qualification_preflight(tmp_path / "model", record, source)
    assert result["source_match"] is True
    assert result["qualified_admission"] == "needs_live_full_identity_check"


class _Connection:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _Response:
    status = 200

    def __init__(self, frames):
        self.stream = io.BytesIO(frames)

    def getheader(self, key, default=""):
        assert key == "Content-Type"
        return "text/event-stream"

    def readline(self):
        return self.stream.readline()


def _event(model, *, terminal=False, receipt="candidate_validation"):
    payload = {"model": model, "choices": [{"delta": {"content": "x"}}]}
    if terminal:
        payload["mlx2"] = {"qualification": "candidate", "route_receipt": receipt,
                           "route": "ordinary", "request_id": "request-1"}
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


def test_candidate_sse_requires_exact_terminal_receipt(monkeypatch):
    connection = _Connection()
    frames = _event("a") + _event("a", terminal=True) + b"data: [DONE]\n\n"
    monkeypatch.setattr(gate, "_request", lambda *args, **kwargs: (connection, _Response(frames)))
    result = gate._stream(1, "secret", "a", "/v1/chat/completions")
    assert result["done"] and result["terminal"]["route_receipt"] == "candidate_validation"
    assert connection.closed


@pytest.mark.parametrize("frames", [
    _event("a") + b"data: [DONE]\n\n",
    _event("a", terminal=True, receipt="wrong") + b"data: [DONE]\n\n",
    _event("b") + _event("b", terminal=True) + b"data: [DONE]\n\n",
])
def test_candidate_sse_fails_closed_on_missing_wrong_or_other_model(monkeypatch, frames):
    connection = _Connection()
    monkeypatch.setattr(gate, "_request", lambda *args, **kwargs: (connection, _Response(frames)))
    with pytest.raises(RuntimeError):
        gate._stream(1, "secret", "a", "/v1/completions")
    assert connection.closed


def test_candidate_launch_requires_owned_gpu_before_model_access(tmp_path):
    args = argparse.Namespace(i_own_gpu=False)
    with pytest.raises(ValueError, match="shared GPU lease"):
        gate.candidate_probe(args, tmp_path)

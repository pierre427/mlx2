"""The opt-in serving-loop trace: inert by default, complete when enabled."""

import json

from mlx2.runtime import loop_trace
from mlx2.runtime.adaptive_policy import DecodeTimeFairness


def test_trace_is_inert_without_the_environment_variable(monkeypatch):
    monkeypatch.setattr(loop_trace, "_PATH", None)
    monkeypatch.setattr(loop_trace, "_events", [])
    loop_trace.event("prefill", 1.0, 2.0, kind="mtp")
    loop_trace.flush({"t": 0.0})
    assert loop_trace._events == []


def test_fairness_hooks_write_one_line_per_iteration(monkeypatch, tmp_path):
    path = tmp_path / "trace.jsonl"
    monkeypatch.setattr(loop_trace, "_PATH", str(path))
    monkeypatch.setattr(loop_trace, "_events", [])
    monkeypatch.setattr(loop_trace, "_handle", None)
    fairness = DecodeTimeFairness(enabled=True)
    fairness.observe_prefill(
        256, 0.2, contended=True, rows=1, depth=4096, kind="mtp", decode_rows=1
    )
    fairness.cap(512, contended=True, depth=4352, kind="mtp")
    fairness.observe_decode(0.05)
    loop_trace.flush(None)  # the first loop iteration has no record yet
    loop_trace.flush({"t": 1.0, "next0": 1.0, "next1": 1.3})
    loop_trace._handle.flush()
    lines = path.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    kinds = [event["k"] for event in record["events"]]
    assert kinds == ["prefill", "cap", "decode"]
    prefill = record["events"][0]
    assert prefill["kind"] == "mtp" and prefill["tokens"] == 256
    assert abs(prefill["ms"] - 200.0) < 1e-6

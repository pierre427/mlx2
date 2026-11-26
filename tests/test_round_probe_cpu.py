"""CPU tests for the opt-in Phase 0 round probe and its harness math.

Design: docs/research/2026-11-25-gpu-resident-spec-rounds-design.md (Phase 0).
CPU device only; no Metal work.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from mlx2.runtime import round_probe  # noqa: E402
import spec_round_phase0 as harness  # noqa: E402


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.setenv(round_probe.SITES_ENV, "4")
    round_probe.uninstall()
    round_probe.reset()
    assert round_probe.install()
    try:
        yield round_probe
    finally:
        round_probe.uninstall()
        round_probe.reset()


def test_off_by_default_patches_nothing():
    assert not round_probe.enabled_by_env({})
    assert not round_probe.enabled_by_env({round_probe.ENV: "0"})
    assert round_probe.enabled_by_env({round_probe.ENV: "1"})
    if not round_probe.ACTIVE:
        assert not hasattr(mx.eval, "_round_probe_original")
        assert not hasattr(mx.array.item, "_round_probe_original")
        assert round_probe.status() == {"enabled": False}
        assert round_probe.scope("x") is round_probe._NULL


def test_counts_explicit_and_implicit_syncs_inside_a_round(probe):
    x = mx.array([1.0, 2.0, 3.0])
    with probe.scope("external:Fake:chain") as round_:
        y = x * 2
        mx.eval(y)
        mx.async_eval(y + 1)
        float(y.sum().item())
        y.tolist()
        np.asarray(y + 3)
        int(mx.array(4))
        bool(mx.array(True))
        mx.synchronize()
        round_.tokens(3)
    routes = probe.status()["routes"]
    row = routes["external:Fake:chain"]
    assert row["rounds"] == 1 and row["tokens"] == 3
    assert set(row["explicit_syncs"]) == {"eval", "async_eval", "synchronize"}
    implicit = row["implicit_syncs"]
    for kind in ("item", "tolist", "__buffer__", "__int__", "__bool__"):
        assert implicit[kind]["calls"] == 1, (kind, implicit)
    # float(item()) calls item once; float() of a Python float is not an array read.
    assert "__float__" not in implicit
    # async_eval submits without waiting: counted by kind, not as a host sync.
    assert row["syncs"] == 7 and row["explicit_syncs"]["async_eval"]["calls"] == 1
    assert 0 <= row["blocking_syncs"] <= row["syncs"]
    assert row["wall_ns"] >= row["encode_submit_ns"] + row["wait_ns"]
    assert row["host_python_ns"] == row["wall_ns"] - row["encode_submit_ns"] - row["wait_ns"]
    assert row["per_round"]["syncs"] == 7
    # Call sites point at this file.
    assert any("test_round_probe_cpu.py" in site for site in row["sites"]["item"])


def test_blocking_count_separates_waits_from_reads_of_evaluated_arrays(probe):
    with probe.scope("b"):
        probe._account("eval", 5_000, probe.BLOCKING_NS + 1, None)  # waited on the device
        probe._account("tolist", 0, 1_000, None)  # array already evaluated
        probe._account("async_eval", 3_000, 0, None)
    row = probe.status()["routes"]["b"]
    assert row["syncs"] == 2 and row["blocking_syncs"] == 1
    assert row["explicit_syncs"]["eval"]["blocking"] == 1
    assert row["implicit_syncs"]["tolist"]["blocking"] == 0


def test_reads_outside_a_round_are_not_counted(probe):
    value = mx.array([5]) + 1
    mx.eval(value)
    value.item()
    assert probe.status()["routes"] == {}


def test_results_are_unchanged(probe):
    x = mx.arange(6).reshape(2, 3)
    with probe.scope("r"):
        assert (x + 1).tolist() == [[1, 2, 3], [4, 5, 6]]
        assert (x.sum()).item() == 15
        assert np.asarray(x).sum() == 15
        mx.eval(x, [x + 1, {"k": x}])  # trees still accepted
    assert probe.status()["routes"]["r"]["explicit_syncs"]["eval"]["calls"] == 1


def test_retag_moves_the_whole_round(probe):
    with probe.scope("external:D:chain"):
        mx.array([1]).item()
        probe.retag("external:D:ordinary")
        mx.array([2]).item()
    routes = probe.status()["routes"]
    assert list(routes) == ["external:D:ordinary"]
    assert routes["external:D:ordinary"]["implicit_syncs"]["item"]["calls"] == 2


def test_nested_scope_is_owned_by_the_outer_round(probe):
    with probe.scope("outer"):
        with probe.scope("inner"):
            mx.array([1]).item()
    routes = probe.status()["routes"]
    assert list(routes) == ["outer"] and routes["outer"]["rounds"] == 1


def test_site_table_is_bounded(probe):
    with probe.scope("s"):
        for i in range(6):
            exec(f"value.item()  # {i}", {"value": mx.array([i])})  # distinct <string> sites
        for _ in range(3):
            mx.array([1]).item()
    sites = probe.status()["routes"]["s"]["sites"]["item"]
    assert len(sites) <= 5  # limit 4 plus "<other>"
    assert sum(sites.values()) == 9


def test_failing_round_still_closes(probe):
    with pytest.raises(RuntimeError):
        with probe.scope("boom"):
            mx.array([1]).item()
            raise RuntimeError("x")
    assert probe.status()["routes"]["boom"]["rounds"] == 1
    # The thread is no longer inside a round.
    mx.array([1]).item()
    assert probe.status()["routes"]["boom"]["implicit_syncs"]["item"]["calls"] == 1


def test_uninstall_restores_originals(probe):
    probe.uninstall()
    assert not hasattr(mx.eval, "_round_probe_original")
    assert not hasattr(mx.array.item, "_round_probe_original")
    assert probe.status() == {"enabled": False}


def test_status_echoes_command_buffer_env(probe, monkeypatch):
    monkeypatch.setenv("MLX_MAX_OPS_PER_BUFFER", "200")
    block = probe.status()
    assert block["command_buffer_env"]["MLX_MAX_OPS_PER_BUFFER"] == "200"
    assert block["performance_claim"] is False


def test_external_run_round_labels_branch_and_tokens(probe):
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    class Drafter:
        pass

    engine = object.__new__(ExternalDraftBatchGenerator)
    engine.draft = Drafter()
    lane = SimpleNamespace(ready=[])

    def impl(cohort, *, forced_depth=None):
        engine._probe_branch("ordinary")
        mx.array([1]).item()
        lane.ready.extend([1, 2])

    engine._run_round_impl = impl
    engine._run_round([lane])
    row = probe.status()["routes"]["external:Drafter:ordinary"]
    assert row["rounds"] == 1 and row["tokens"] == 2


def test_external_run_round_is_a_passthrough_when_off():
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    assert not round_probe.ACTIVE
    engine = object.__new__(ExternalDraftBatchGenerator)
    calls = []
    engine._run_round_impl = lambda cohort, *, forced_depth=None: calls.append(forced_depth) or "r"
    assert engine._run_round([], forced_depth=0) == "r"
    assert calls == [0]


def test_generate_probed_next(probe):
    from mlx2.runtime.generate import BatchGenerator

    class FakeBatch:
        def next(self):
            mx.array([1]).item()
            return ["a", "b", "c"]

    owner = object.__new__(BatchGenerator)
    assert BatchGenerator._probed_next(owner, FakeBatch()) == ["a", "b", "c"]
    row = probe.status()["routes"]["generate:FakeBatch"]
    assert row["rounds"] == 1 and row["tokens"] == 3


# --------------------------------------------------------------- harness --

def _block(rounds, wall, encode, wait, item_calls):
    return {"enabled": True, "routes": {"external:D:chain": {
        "rounds": rounds, "tokens": rounds * 4, "wall_ns": wall, "encode_submit_ns": encode,
        "wait_ns": wait, "host_python_ns": wall - encode - wait, "syncs": item_calls,
        "explicit_syncs": {}, "implicit_syncs": {"item": {"calls": item_calls,
                                                          "encode_submit_ns": encode, "wait_ns": wait}}}}}


def test_probe_delta_and_per_round():
    before = _block(10, 100_000_000, 10_000_000, 60_000_000, 20)
    after = _block(60, 600_000_000, 60_000_000, 360_000_000, 120)
    delta = harness.probe_delta(after, before)
    row = delta["external:D:chain"]
    assert row["rounds"] == 50 and row["implicit_syncs"]["item"]["calls"] == 100
    pr = harness.per_round(row)
    assert pr["wall_ms"] == 10.0 and pr["wait_ms"] == 6.0 and pr["host_python_ms"] == 3.0
    assert pr["syncs"] == 2.0 and pr["tokens"] == 4.0
    total = harness.add_deltas({}, delta)
    harness.add_deltas(total, delta)
    assert total["external:D:chain"]["rounds"] == 100
    assert harness.probe_delta(before, None)["external:D:chain"]["rounds"] == 10


def _cell(condition, tok_s, wall, encode, python, gpu=None, tokens=4.0):
    return {"size": 1024, "condition": condition, "route_label": "external:D:chain",
            "decode_tok_s_median": tok_s, "rounds_per_s": tok_s / tokens,
            "per_round": {"tokens": tokens, "wall_ms": wall, "encode_submit_ms": encode,
                          "wait_ms": wall - encode - python, "host_python_ms": python, "syncs": 3},
            "load": {"load1_mean": 70 if condition == "contended" else 3},
            "metal_trace": {"gpu_ms_per_round": gpu} if gpu is not None else None}


def test_summary_go_on_contended_slowdown():
    doc = {"checkpoint": "Raon-1-mini-mlx@sha256:x", "arm": "dflash2",
           "cells": [_cell("quiet", 108.7, 10.0, 0.5, 0.5), _cell("contended", 65.3, 16.0, 2.0, 4.0)]}
    row = harness.summarize(doc)["rows"][0]
    assert row["contended_slowdown"] == pytest.approx(1 - 65.3 / 108.7)
    assert row["phase1_go"] and "contended" in row["go_reasons"][0]
    assert row["host_share_basis"].startswith("(encode")
    assert row["quiet_in_round_host_share"] == pytest.approx(0.1)


def test_summary_host_share_prefers_gpu_trace():
    doc = {"checkpoint": "c@f", "arm": "a",
           "cells": [_cell("quiet", 100, 10.0, 0.1, 0.1, gpu=8.0), _cell("contended", 95, 10.5, 0.2, 0.2)]}
    row = harness.summarize(doc)["rows"][0]
    assert row["quiet_host_share"] == pytest.approx(0.2)
    assert row["phase1_go"] and row["go_reasons"] == ["quiet host share 20.0% >= 15%"]


def test_summary_counts_time_between_rounds():
    quiet = _cell("quiet", 100, 10.0, 0.2, 0.3, gpu=9.5)
    quiet["per_round"].update(period_ms=12.5, outside_ms=2.5)
    doc = {"checkpoint": "c@f", "arm": "a", "cells": [quiet, _cell("contended", 95, 10.5, 0.2, 0.2)]}
    row = harness.summarize(doc)["rows"][0]
    # The serving loop's 2.5 ms between rounds is host time the GPU idles through.
    assert row["quiet_host_share"] == pytest.approx(1 - 9.5 / 12.5)
    assert row["host_share_basis"] == "1 - gpu_ms/period_ms" and row["phase1_go"]


def test_summary_no_go():
    doc = {"checkpoint": "c@f", "arm": "a",
           "cells": [_cell("quiet", 100, 10.0, 0.2, 0.3, gpu=9.5), _cell("contended", 95, 10.5, 0.2, 0.2)]}
    row = harness.summarize(doc)["rows"][0]
    assert not row["phase1_go"] and row["go_reasons"] == []


GPU_XML = """<?xml version="1.0"?><trace-query-result><node><schema name="metal-gpu-intervals">
<col><mnemonic>start</mnemonic></col><col><mnemonic>duration</mnemonic></col>
<col><mnemonic>process</mnemonic></col></schema>
<row><start-time id="1">100</start-time><duration id="2">50</duration><sentinel/></row>
<row><start-time id="3">200</start-time><duration id="4">100</duration><process id="5" fmt="python (42)"><pid id="6">42</pid></process></row>
<row><start-time id="7">250</start-time><duration ref="2"/><process ref="5"/></row>
<row><start-time id="8">400</start-time><duration ref="2"/><process id="9" fmt="loginwindow (7)"/></row>
</node></trace-query-result>"""


def test_gpu_interval_parse_and_union():
    rows = harness.parse_gpu_intervals(GPU_XML)
    assert len(rows) == 4 and rows[2]["process"][1] == "python (42)"
    busy = harness.gpu_busy(rows, 42)
    # [200,300) and [250,300) overlap: union 100 ns over a 100 ns span.
    assert busy == {"intervals": 2, "pid_matched_intervals": 2, "busy_ns": 100,
                    "span_ns": 100, "filter": "pid"}
    every = harness.gpu_busy(rows, None)
    assert every["filter"] == "all_rows" and every["busy_ns"] == 200 and every["span_ns"] == 350


def test_load_generator_stops_its_workers():
    generator = harness.LoadGenerator(2, 30).start()
    try:
        assert all(proc.is_alive() for proc in generator.procs)
    finally:
        procs = list(generator.procs)
        generator.stop()
    assert not any(proc.is_alive() for proc in procs)
    sample = harness.load_sample()
    assert {"load1", "runnable"} <= set(sample)


def test_harness_run_end_to_end_against_a_fake_server(tmp_path):
    """Quiet cells before contended ones, probe deltas per request, summary row."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    state = {"rounds": 0, "order": []}
    lock = threading.Lock()

    def block():
        r = state["rounds"]
        return {"enabled": True, "routes": {"external:DFlash2Drafter:chain": {
            "rounds": r, "tokens": 4 * r, "wall_ns": 10_000_000 * r,
            "encode_submit_ns": 1_000_000 * r, "wait_ns": 8_000_000 * r,
            "host_python_ns": 1_000_000 * r, "syncs": 3 * r,
            "explicit_syncs": {"eval": {"calls": 2 * r, "encode_submit_ns": r, "wait_ns": r}},
            "implicit_syncs": {"item": {"calls": r, "encode_submit_ns": r, "wait_ns": r}}}}}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, payload):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send({"model": "Fake", "artifact": "sha256:f", "round_probe": block()})

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path.endswith("/apc/clear"):
                return self._send({"cleared": True})
            with lock:
                state["rounds"] += 0 if payload["max_tokens"] == 8 else 64
                state["order"].append(payload["max_tokens"])
                probe_block = block()
            self._send({"usage": {"prompt_tokens": 100, "completion_tokens": payload["max_tokens"]},
                        "choices": [{"finish_reason": "length", "message": {"content": "x"}}],
                        "mlx2": {"route": "external_draft", "timing": {"ttft_seconds": 0.0001},
                                 "round_probe": probe_block}})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    token = tmp_path / "token"
    token.write_text("t")
    out = tmp_path / "out"
    try:
        harness.main(["run", "--base-url", f"http://127.0.0.1:{server.server_port}",
                      "--admin-token", str(token), "--checkpoint", "Fake@sha256:f", "--arm", "dflash2",
                      "--source-revision", "abc", "--sizes", "256,512", "--rounds", "100",
                      "--load-workers", "1", "--load-ramp-s", "0", "--cooldown-s", "0",
                      "--max-tokens", "32", "--xctrace", "--server-pid", "1", "--trace-max-size", "128",
                      "--output", str(out)])
    finally:
        server.shutdown()
    document = json.loads((out / "phase0.json").read_text())
    cells = document["cells"]
    assert [(c["size"], c["condition"]) for c in cells] == [
        (256, "quiet"), (512, "quiet"), (256, "contended"), (512, "contended")]
    for cell in cells:
        assert cell["rounds"] == 128 and len(cell["requests"]) == 2  # 64 rounds per request
        assert cell["per_round"]["wall_ms"] == 10.0 and cell["per_round"]["tokens"] == 4.0
        assert cell["per_round"]["by_kind"] == {"eval": 2.0, "item": 1.0}
        assert cell["load"]["samples"]
        if cell["condition"] == "quiet":  # beyond --trace-max-size: never recorded
            assert cell["metal_trace"]["status"] == "skipped" and "trace-max-size" in cell["metal_trace"]["reason"]
        else:
            assert cell["metal_trace"] is None  # contended cells are never traced
    assert document["load_nice"] == 0 and document["cooldown_s"] == 0
    with pytest.raises(SystemExit):  # never overwrite evidence
        harness.main(["run", "--base-url", "http://127.0.0.1:9", "--admin-token", str(token),
                      "--checkpoint", "c@f", "--arm", "a", "--source-revision", "x", "--output", str(out)])
    rows = harness.summarize(document)["rows"]
    assert rows[0]["contended_slowdown"] == pytest.approx(0.0, abs=1.0)
    assert rows[0]["host_share_basis"].endswith("/period_ms")
    assert rows[0]["quiet_in_round_host_share"] == pytest.approx(0.2)  # (1 + 1) / 10 ms
    assert rows[0]["phase1_go"]
    for cell in cells:
        assert cell["prompt_tokens"] == 100 and cell["per_round"]["period_ms"] > 0


def test_mixed_round_labels_share_the_measured_decode_period(monkeypatch, tmp_path):
    block = {"enabled": True, "routes": {
        "chain": {"rounds": 3, "tokens": 9, "wall_ns": 6_000_000},
        "ordinary": {"rounds": 1, "tokens": 1, "wall_ns": 2_000_000}}}
    request = {"decode_tok_s": 100.0, "decode_s": 0.02, "prompt_tokens": 32}
    monkeypatch.setattr(harness, "request", lambda *args: (request, block))
    args = SimpleNamespace(max_requests=1, max_tokens=10, rounds=4, xctrace=False)
    cell, _ = harness.measure_cell(None, "fake", "", args, "quiet", tmp_path, 32, None, None)
    assert cell["rounds"] == 4
    assert cell["per_round"]["period_ms"] == 5
    assert cell["per_round"]["tokens"] == 2.5
    assert cell["per_round"]["wall_ms"] == 2


def test_overrun_does_not_enter_contended_measurement(monkeypatch, tmp_path):
    # Stop the cell's load window while the HTTP request is outstanding.
    import time
    now = time.monotonic()

    def request(*args):
        monkeypatch.setattr(harness.time, "monotonic", lambda: now + 10)
        return {"completion_tokens": 3}, {"enabled": True, "routes": {"a": {"rounds": 1}}}

    args = SimpleNamespace(max_requests=1, max_tokens=3, rounds=1, xctrace=False)
    monkeypatch.setattr(harness, "request", request)
    cell, _ = harness.measure_cell(None, "fake", "", args, "contended", tmp_path, 1, None, None,
                                   load_deadline=now + 5)
    assert cell["rounds"] == 0 and cell["decode_tok_s_median"] is None
    assert cell["request_past_load_deadline"] == {"completion_tokens": 3}


@pytest.mark.parametrize("keep", [False, True])
def test_failed_traced_request_stops_recorder_and_respects_keep(monkeypatch, tmp_path, keep):
    bundle = tmp_path / "metal-1-quiet.trace"
    stopped = []

    class Recorder:
        def start(self):
            self.proc = object()
            bundle.mkdir()
            (bundle / "partial").write_bytes(b"data")
            return True

        def stop(self):
            stopped.append(True)
            return 0

    def failed_request(*args):
        raise TimeoutError("request timed out")

    monkeypatch.setattr(harness, "MetalTrace", lambda *args: Recorder())
    monkeypatch.setattr(harness, "request", failed_request)
    args = SimpleNamespace(max_requests=0, xctrace=True, trace_max_size=16, trace_min_free_gib=0,
                           trace_limit_s=60, trace_max_tokens=64, keep_traces=keep)
    with pytest.raises(TimeoutError, match="request timed out"):
        harness.measure_cell(None, "fake", "", args, "quiet", tmp_path, 1, 123, None)
    assert stopped == [True]
    assert bundle.exists() == keep


def test_preexisting_trace_is_preserved_on_refusal(tmp_path):
    bundle = tmp_path / "metal-1-quiet.trace"
    bundle.mkdir()
    marker = bundle / "existing"
    marker.write_bytes(b"evidence")
    args = SimpleNamespace(max_requests=0, xctrace=True, trace_max_size=16, trace_min_free_gib=0,
                           trace_limit_s=60, trace_max_tokens=64, keep_traces=False)
    with pytest.raises(FileExistsError, match="refuse to overwrite"):
        harness.measure_cell(None, "fake", "", args, "quiet", tmp_path, 1, 123, None)
    assert marker.read_bytes() == b"evidence"


def test_missing_contention_measurement_leaves_negative_verdict_open():
    doc = {"cells": [_cell("quiet", 100, 10.0, 0.2, 0.3)]}
    assert harness.summarize(doc)["rows"][0]["phase1_go"] is None


@pytest.mark.parametrize(("contended_tps", "tokens", "rounds", "verdict"), [
    (100 / 1.2, 4, 200, True),
    (50, 2, 200, None),
    (50, 4, 199, None),
])
def test_contention_decision_uses_round_period_and_adequate_comparable_cells(
        contended_tps, tokens, rounds, verdict):
    quiet = {**_cell("quiet", 100, 10, 0.1, 0.1), "rounds": 200}
    contended = {**_cell("contended", contended_tps, 10, 0.1, 0.1, tokens=tokens), "rounds": rounds}
    result = harness.summarize({"rounds_target": 200, "cells": [quiet, contended]})
    assert result["rows"][0]["phase1_go"] is verdict

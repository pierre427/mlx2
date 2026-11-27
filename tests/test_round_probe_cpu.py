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


# ------------------------------------------------ 2026-11-26 rerun options --

def test_size_max_tokens_parse_and_lookup():
    assert harness.parse_size_max_tokens("") == {}
    assert harness.parse_size_max_tokens("65536=32, 16384=128") == {65536: 32, 16384: 128}
    for bad in ("65536", "65536=0", "x=3"):
        with pytest.raises(ValueError):
            harness.parse_size_max_tokens(bad)
    args = SimpleNamespace(max_tokens=256, rounds=200, size_max_tokens={65536: 32}, contended_rounds=60)
    assert harness.max_tokens_for(args, 65536) == 32 and harness.max_tokens_for(args, 1024) == 256
    assert harness.rounds_target(args, "contended") == 60 and harness.rounds_target(args, "quiet") == 200
    plain = SimpleNamespace(max_tokens=256, rounds=200)  # Phase 0 defaults are unchanged
    assert harness.max_tokens_for(plain, 65536) == 256 and harness.rounds_target(plain, "contended") == 200


def test_mtp_transition_summarises_fallback_and_handoff():
    fallback = {"mtp": {"route": "ordinary_after_mtp_fallback", "num_draft": 2,
                        "observed_compute_widths": [1], "ordinary_compute_widths": [],
                        "mtp_plain_fallback": {"transition": "fallback", "reason": "admission_plain",
                                               "committed_tokens_before_fallback": 0}}}
    got = harness.mtp_transition(fallback)
    assert got["transition"] == "fallback" and got["reason"] == "admission_plain"
    assert got["committed_tokens_before"] == 0
    handoff = {"mtp": {"route": "ordinary_after_mtp_handoff", "mtp_ordinary_handoff": {
        "reason": "static_width_threshold", "committed_tokens_before_handoff": 5, "projected_width": 2}}}
    got = harness.mtp_transition(handoff)
    assert got["transition"] == "handoff" and got["projected_width"] == 2
    assert harness.mtp_transition({"mtp": {"route": "native_mtp"}}) == {
        "route": "native_mtp", "num_draft": None, "observed_compute_widths": None,
        "ordinary_compute_widths": None}
    assert harness.mtp_transition({"mtp": None}) is None and harness.mtp_transition(None) is None


def test_counter_delta_keeps_changed_numeric_leaves():
    before = harness.counter_snapshot({"scheduler": {"a": 1, "b": 2, "flag": True},
                                       "apcv2": {"hits": {"disk": 0}}, "other": {"x": 1}})
    after = harness.counter_snapshot({"scheduler": {"a": 1, "b": 5, "c": 1, "flag": False},
                                      "apcv2": {"hits": {"disk": 1}}, "other": {"x": 9}})
    assert harness.counter_delta(after, before) == {
        "apcv2.hits.disk": 1, "scheduler.b": 3, "scheduler.c": 1}


def test_mismatched_request_lengths_leave_the_pair_incomplete():
    quiet = {**_cell("quiet", 100, 10, 0.1, 0.1), "rounds": 200, "max_tokens": 256}
    contended = {**_cell("contended", 50, 10, 0.1, 0.1), "rounds": 200, "max_tokens": 32}
    result = harness.summarize({"rounds_target": 200, "cells": [quiet, contended]})
    assert result["rows"][0]["contended_period_increase"] == pytest.approx(1.0)
    assert not any("contended" in reason for reason in result["rows"][0]["go_reasons"])


def test_rerun_options_end_to_end_against_a_fake_server(tmp_path):
    """Per-size max_tokens in both conditions, a lower contended round target,
    receipts and counter deltas per request, and a complete 64K-style pair."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    state = {"rounds": 0, "order": [], "fallbacks": 0, "collections": 0}
    lock = threading.Lock()

    def block():
        r = state["rounds"]
        return {"enabled": True, "routes": {"generate:MTPGenerationBatch": {
            "rounds": r, "tokens": 2 * r, "wall_ns": 10_000_000 * r,
            "encode_submit_ns": 1_000_000 * r, "wait_ns": 8_000_000 * r,
            "host_python_ns": 1_000_000 * r, "syncs": 3 * r}}}

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
            with lock:
                state["collections"] += 1  # each read sees a newer worker snapshot
                self._send({"model": "Fake", "round_probe": block(),
                            "host_snapshot_timing": {"collections": state["collections"]},
                            "scheduler": {"mtp_plain_fallback_events": state["fallbacks"]}})

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path.endswith("/apc/clear"):
                return self._send({"cleared": True})
            tokens = payload["max_tokens"]
            with lock:
                state["order"].append(tokens)
                state["rounds"] += 0 if tokens == 8 else tokens // 2
                fallback = tokens != 8 and len(state["order"]) == 3  # second 1K quiet request
                state["fallbacks"] += fallback
                probe_block = block()
            mtp = ({"route": "ordinary_after_mtp_fallback", "num_draft": 2,
                    "mtp_plain_fallback": {"transition": "fallback", "reason": "admission_plain"}}
                   if fallback else {"route": "native_mtp"})
            self._send({"usage": {"prompt_tokens": 100, "completion_tokens": tokens},
                        "choices": [{"finish_reason": "length", "message": {"content": "x"}}],
                        "mlx2": {"route": "native_mtp", "timing": {"ttft_seconds": 0.0001},
                                 "mtp": mtp, "round_probe": probe_block}})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    token = tmp_path / "token"
    token.write_text("t")
    out = tmp_path / "out"
    try:
        harness.main(["run", "--base-url", f"http://127.0.0.1:{server.server_port}",
                      "--admin-token", str(token), "--checkpoint", "Fake@sha256:f", "--arm", "mtp",
                      "--source-revision", "abc", "--sizes", "256,512", "--rounds", "64",
                      "--contended-rounds", "16", "--size-max-tokens", "512=16",
                      "--load-workers", "1", "--load-ramp-s", "0", "--cooldown-s", "0",
                      "--max-tokens", "64", "--scheduler-deltas", "--output", str(out)])
    finally:
        server.shutdown()
    document = json.loads((out / "phase0.json").read_text())
    assert document["size_max_tokens"] == {"512": 16} and document["rounds_target_contended"] == 16
    measured = [t for t in state["order"] if t != 8]
    # quiet 256: 2x64 tok (32 rounds each); quiet 512: 8x16 (8 rounds each);
    # contended 256: 1x64; contended 512: 2x16.
    assert measured == [64, 64] + [16] * 8 + [64] + [16, 16]
    by = {(c["size"], c["condition"]): c for c in document["cells"]}
    assert by[(512, "quiet")]["max_tokens"] == 16 and by[(256, "quiet")]["max_tokens"] == 64
    assert by[(256, "contended")]["rounds_target"] == 16 and by[(256, "quiet")]["rounds_target"] == 64
    first = by[(256, "quiet")]
    assert first["route_transitions"] == [{"request": 1, "route": "ordinary_after_mtp_fallback",
                                           "num_draft": 2, "observed_compute_widths": None,
                                           "ordinary_compute_widths": None, "transition": "fallback",
                                           "reason": "admission_plain", "committed_tokens_before": None,
                                           "projected_width": None, "decision": None}]
    deltas = [{k: v for k, v in r["counter_deltas"].items()} for r in first["requests"]]
    assert deltas == [{}, {"scheduler.mtp_plain_fallback_events": 1}]
    assert all(r["counter_read"]["fresh"] for r in first["requests"])
    assert "mlx2_receipt" not in first["requests"][0] and "mlx2_receipt" not in first["warmup"]
    lines = (out / "receipts-256-quiet.jsonl").read_text().splitlines()
    assert [json.loads(line)["request"] for line in lines] == [0, 1]
    assert "round_probe" not in json.loads(lines[0])["receipt"]
    rows = {r["size"]: r for r in harness.summarize(document)["rows"]}
    assert rows[512]["quiet"]["route_transitions"] == 0
    assert rows[256]["quiet"]["route_transitions"] == 1
    assert rows[512]["phase1_go"] is not None  # complete pair at matched length
    assert first["cold_requests"] == 0  # no cached_tokens reported: unknown, never counted cold


def test_cold_requests_count_only_measured_misses(monkeypatch, tmp_path):
    rows = iter([({"prompt_tokens": 100, "cached_tokens": 99, "decode_s": 1.0, "decode_tok_s": 9.0},
                  {"enabled": True, "routes": {"a": {"rounds": 5}}}),
                 ({"prompt_tokens": 100, "cached_tokens": 0, "decode_s": 1.0, "decode_tok_s": 9.0},
                  {"enabled": True, "routes": {"a": {"rounds": 10}}})])
    monkeypatch.setattr(harness, "request", lambda *args: next(rows))
    args = SimpleNamespace(max_requests=2, max_tokens=10, rounds=100, xctrace=False)
    cell, _ = harness.measure_cell(None, "fake", "", args, "quiet", tmp_path, 100, None, None)
    assert cell["cold_requests"] == 1 and cell["rounds"] == 10


def test_probe_route_change_flags_target_only_ordinary_requests(monkeypatch, tmp_path):
    """No mtp receipt, but the probe shows the request left the MTP batch."""
    blocks = iter([
        {"enabled": True, "routes": {"generate:MTPGenerationBatch": {"rounds": 5, "tokens": 9}}},
        {"enabled": True, "routes": {"generate:MTPGenerationBatch": {"rounds": 5, "tokens": 9},
                                     "generate:GenerationBatch": {"rounds": 9, "tokens": 9}}}])
    row = lambda: {"prompt_tokens": 100, "cached_tokens": 99, "decode_s": 1.0, "decode_tok_s": 9.0,  # noqa: E731
                   "mtp": None}
    monkeypatch.setattr(harness, "request", lambda *args: (row(), next(blocks)))
    args = SimpleNamespace(max_requests=2, max_tokens=10, rounds=100, xctrace=False)
    cell, _ = harness.measure_cell(None, "fake", "", args, "quiet", tmp_path, 100, None, None)
    assert cell["route_transitions"] == []
    assert cell["probe_route_changes"] == [{"request": 1, "routes": ["generate:GenerationBatch"],
                                            "first_routes": ["generate:MTPGenerationBatch"]}]
    row_summary = harness.summarize({"rounds_target": 100, "cells": [cell]})["rows"][0]
    assert row_summary["quiet"]["route_transitions"] == 1
    old = {k: v for k, v in cell.items() if k not in ("route_transitions", "probe_route_changes")}
    assert harness.summarize({"rounds_target": 100, "cells": [old]})["rows"][0]["quiet"]["route_transitions"] is None


def test_fresh_status_waits_for_new_snapshots_and_never_raises(monkeypatch):
    monkeypatch.setattr(harness.time, "sleep", lambda s: None)

    class Client:
        def __init__(self, values):
            self.values = iter(values)

        def call(self, path, timeout=None):
            value = next(self.values)
            if isinstance(value, Exception):
                raise value
            return value

    snap = lambda n: {"host_snapshot_timing": {"collections": n}, "scheduler": {"x": n}}  # noqa: E731
    status, info = harness.fresh_status(Client([snap(4), snap(4), snap(5), snap(6)]))
    assert status["scheduler"]["x"] == 6 and info["fresh"] and info["collections"] == 6
    status, info = harness.fresh_status(Client([TimeoutError("slow")]))
    assert status == {} and "TimeoutError" in info["error"]
    status, info = harness.fresh_status(Client(["not json"]))
    assert status == {} and "TypeError" in info["error"]


def test_past_deadline_request_receipt_is_marked(monkeypatch, tmp_path):
    import json
    import time
    now = time.monotonic()

    def request(*args):
        monkeypatch.setattr(harness.time, "monotonic", lambda: now + 10)
        return ({"completion_tokens": 3, "mlx2_receipt": {"route": "native_mtp"}},
                {"enabled": True, "routes": {"a": {"rounds": 1}}})

    monkeypatch.setattr(harness, "request", request)
    args = SimpleNamespace(max_requests=1, max_tokens=3, rounds=1, xctrace=False)
    cell, _ = harness.measure_cell(None, "fake", "", args, "contended", tmp_path, 1, None, None,
                                   load_deadline=now + 5)
    line = json.loads((tmp_path / "receipts-1-contended.jsonl").read_text())
    assert line["past_deadline"] is True and cell["rounds"] == 0


def test_rerun_options_are_refused_before_evidence_is_written(tmp_path):
    token = tmp_path / "token"
    token.write_text("t")
    base = ["run", "--base-url", "http://127.0.0.1:9", "--admin-token", str(token), "--checkpoint", "c@f",
            "--arm", "a", "--source-revision", "x", "--sizes", "65536"]
    for extra, message in ((["--size-max-tokens", "65535=32"], "not in --sizes"),
                           (["--size-max-tokens", "65536=16"], "cannot book 200 rounds"),
                           (["--size-max-tokens", "65536"], "SIZE=TOKENS")):
        out = tmp_path / extra[1].replace("=", "_")
        with pytest.raises(SystemExit, match=message):
            harness.main(base + extra + ["--output", str(out)])
        assert not out.exists()
    args = SimpleNamespace(max_requests=8, max_tokens=256, rounds=136, contended_rounds=60,
                           size_max_tokens={65536: 64})
    harness.validate_rerun_options(args, [65536], ["quiet", "contended"])  # the rerun recipe fits


def test_request_extra_fields_and_output_digest(monkeypatch):
    sent = []

    class Client:
        def call(self, path, payload=None, **kwargs):
            sent.append(payload)
            return {"usage": {"completion_tokens": 3, "prompt_tokens": 9},
                    "choices": [{"message": {"content": "abc"}, "finish_reason": "length"}],
                    "mlx2": {"timing": {"ttft_seconds": 0.001}, "round_probe": {"enabled": True}}}

    monkeypatch.setattr(harness, "REQUEST_EXTRA", {"presence_penalty": 0})
    row, block = harness.request(Client(), "m", "hello", 4)
    assert sent[0]["presence_penalty"] == 0 and sent[0]["temperature"] == 0
    assert row["text_sha256"] == __import__("hashlib").sha256(b"abc").hexdigest()
    assert block == {"enabled": True}


class _FakeTimeline:
    """Stands in for the native step_replay timeline (host seconds)."""

    CAPABILITIES = ["timeline"]
    MLX_VERSION = "test"

    def __init__(self):
        self.now = 100.0
        self.entries = []
        self.pending = 0
        self.recordings = 0

    def host_now(self):
        self.now += 0.001
        return self.now

    def timeline_drain(self):
        taken, self.entries = self.entries, []
        return taken

    def timeline_pending(self):
        return self.pending

    def timeline_dropped(self):
        return 0

    def set_timeline(self, on):
        pass

    def gpu_identity(self):
        return "fake"


def test_gpu_timeline_attribution_splits_idle_inside_and_outside_syncs(probe, monkeypatch):
    fake = _FakeTimeline()
    monkeypatch.setattr(round_probe, "_native", fake)
    round_probe._gpu.update(enabled=True)
    # One round, driven by hand: [t0, t1] = [10.0, 10.010]; one sync window
    # [10.004, 10.009]; GPU busy [10.001, 10.003] (during host Python) and
    # [10.005, 10.008] (inside the sync); 10.008 -> 10.009 is completion tail.
    with round_probe._lock:
        round_probe._pending.append(("r", 10.0, 10.010, ((10.004, 10.009),)))
    fake.entries = [(10.0005, 10.001, 10.003, 4), (10.0045, 10.005, 10.008, 4),
                    (10.0046, 10.0, 10.0, 5)]  # an errored buffer is skipped
    status = round_probe.status()
    row = status["routes"]["r"]
    assert row["gpu_rounds"] == 1
    assert row["gpu_wall_ns"] == pytest.approx(10_000_000, abs=2)
    assert row["gpu_busy_ns"] == pytest.approx(5_000_000, abs=2)
    assert row["gpu_eval_ns"] == pytest.approx(5_000_000, abs=2)
    assert row["gpu_eval_busy_ns"] == pytest.approx(3_000_000, abs=2)
    assert row["gpu_eval_tail_ns"] == pytest.approx(1_000_000, abs=2)
    assert row["gpu_command_buffers"] == 2
    shares = harness.gpu_per_round(row)
    assert shares["idle_share"] == pytest.approx(0.5)
    assert shares["eval_idle_less_tail_share"] == pytest.approx(0.1)  # (5-3-1)/10
    assert shares["outside_eval_idle_share"] == pytest.approx(0.3)   # (5-2)/10
    assert status["gpu_timeline"]["errored"] == 1


def test_gpu_timeline_waits_for_unsettled_rounds(probe, monkeypatch):
    fake = _FakeTimeline()
    monkeypatch.setattr(round_probe, "_native", fake)
    round_probe._gpu.update(enabled=True)
    fake.pending = 1  # a command buffer has not completed yet
    with round_probe._lock:
        round_probe._pending.append(("r", fake.now, fake.now + 0.001, ()))
    assert "gpu_rounds" not in round_probe.status()["routes"].get("r", {}) or \
        round_probe.status()["routes"]["r"]["gpu_rounds"] == 0
    fake.pending = 0
    assert round_probe.status()["routes"]["r"]["gpu_rounds"] == 1


def test_round_scope_records_windows_with_native(probe, monkeypatch):
    fake = _FakeTimeline()
    monkeypatch.setattr(round_probe, "_native", fake)
    with round_probe.scope("r") as scope:
        mx.eval(mx.array([1.0]) + 1)
        scope.tokens(1)
    (route, t0, t1, windows), = list(round_probe._pending)
    assert route == "r" and t0 < t1 and len(windows) == 1
    assert t0 <= windows[0][0] < windows[0][1] <= t1


def test_probe_delta_carries_gpu_keys():
    before = {"enabled": True, "routes": {"r": {"rounds": 1, "gpu_rounds": 1, "gpu_wall_ns": 5}}}
    after = {"enabled": True, "routes": {"r": {"rounds": 3, "gpu_rounds": 2, "gpu_wall_ns": 9}}}
    delta = harness.probe_delta(after, before)
    assert delta["r"]["gpu_rounds"] == 1 and delta["r"]["gpu_wall_ns"] == 4
    plain = harness.probe_delta({"enabled": True, "routes": {"r": {"rounds": 2}}}, {})
    assert "gpu_rounds" not in plain["r"]


def test_clean_dispatches_excludes_reads_of_later_overwritten_buffers():
    # Kernels: "w" writes slot 1 and reads slot 0; "r" reads slot 0 only.
    reflection = {"w": {0: False, 1: True}, "r": {0: False}}
    kernels = ["w", "r"]
    table = [
        {"pipeline": 0, "slots": [(0, "buffer", 10, 0), (1, "buffer", 11, 0)]},  # 10 -> 11
        {"pipeline": 1, "slots": [(0, "buffer", 11, 0)]},                        # reads 11
        {"pipeline": 0, "slots": [(0, "buffer", 12, 0), (1, "buffer", 11, 0)]},  # 12 -> 11 again
        {"pipeline": 1, "slots": [(0, "buffer", 12, 0)]},                        # reads 12
    ]
    # 0 wrote 11 but 2 rewrites it; 1 read 11 before 2 rewrote it; 2 and 3 are clean.
    assert round_probe.clean_dispatches(table, kernels, reflection) == [2, 3]
    # Unreflected kernels are never clean.
    assert round_probe.clean_dispatches(table, ["x", "r"], reflection) == [3]


def test_flag_ab_instance_prefers_clean_and_skips_unsafe_writers():
    kernels = ["gemv", "scatter_x", "custom_kernel_y"]
    table = [{"pipeline": 0, "slots": []}, {"pipeline": 0, "slots": []},
             {"pipeline": 1, "slots": []}, {"pipeline": 2, "slots": []},
             {"pipeline": 2, "slots": []}]
    chosen = round_probe.flag_ab_instance(table, kernels, clean=[1, 4])
    assert chosen == {"gemv": (1, True), "custom_kernel_y": (4, True)}  # scatter: no clean one


def test_bootstrap_ratio_brackets_the_median():
    stats = round_probe._bootstrap_ratio([10.0] * 20, [10.0 + 0.1 * (i % 3) for i in range(20)])
    assert stats["pairs"] == 20 and stats["ci95"][0] <= stats["median"] <= stats["ci95"][1]


def test_clean_dispatches_ignores_leftover_bindings():
    # "r" uses only slot 0; slot 5 is a leftover from an earlier kernel in the same encoder.
    reflection = {"w": {0: False, 1: True}, "r": {0: False}}
    kernels = ["w", "r"]
    table = [
        {"pipeline": 1, "slots": [(0, "buffer", 10, 0), (5, "buffer", 11, 0)]},  # leftover 11
        {"pipeline": 0, "slots": [(0, "buffer", 12, 0), (1, "buffer", 11, 0)]},  # writes 11 later
    ]
    # Dispatch 0 does not read 11, so the later write does not make it unclean.
    assert round_probe.clean_dispatches(table, kernels, reflection) == [0, 1]


def test_recording_reflection_follows_the_recordings_kernel_order(monkeypatch, tmp_path):
    """``pipelines`` in a recording follows ``summary["kernels"]``, not the native table's
    hash-map order (2026-11-27: indexing the old order by ``dispatch["pipeline"]`` swapped
    kernel names between rounds)."""
    import json
    import types

    class Handle:
        def summary(self):
            return {"kernels": ["copy", "sort", "gemv"]}

        def dispatch_table(self):
            return [{"pipeline": 1, "slots": []}, {"pipeline": 0, "slots": []}]

    native = types.SimpleNamespace(
        pipelines=lambda: [{"function": "gemv"}, {"function": "unused"}, {"function": "sort"},
                           {"function": "copy"}],
        compare_recordings=lambda a, b: "same", gpu_identity=lambda: "fake", MLX_VERSION="0",
        CAPABILITIES=())
    monkeypatch.setattr(round_probe, "_native", native)
    monkeypatch.setitem(round_probe._record, "dir", str(tmp_path))
    monkeypatch.setitem(round_probe._record, "written", 0)
    monkeypatch.setitem(round_probe._record, "errors", [])
    monkeypatch.setitem(round_probe._record, "previous", None)
    monkeypatch.setattr(round_probe, "_flag_ab_indices", lambda: ())
    round_probe._write_recording("r", Handle(), 0.0)
    assert round_probe._record["errors"] == []
    doc = json.loads(next(tmp_path.glob("round-*.json")).read_text())
    assert [p["function"] for p in doc["pipelines"]] == doc["summary"]["kernels"]


def test_mixed_route_cells_give_no_verdict():
    """Maxwell 16K at a 2 GiB cap: one request decoded as a plain fallback, so
    the quiet averages mixed two routes and a host-share "go" described neither."""
    quiet = _cell("quiet", 100, 10.0, 0.2, 0.3, gpu=7.0)
    contended = _cell("contended", 95, 10.5, 0.2, 0.2)
    base = {"checkpoint": "c@f", "arm": "a", "cells": [quiet, contended]}
    assert harness.summarize(base)["rows"][0]["phase1_go"] is True
    for cell in (quiet, contended):
        cell["route_transitions"], cell["probe_route_changes"] = [], []
    clean = harness.summarize(base)["rows"][0]
    assert clean["phase1_go"] is True and clean["mixed_route_cells"] == []
    quiet["probe_route_changes"] = [{"request": 1, "routes": ["generate:GenerationBatch"],
                                     "first_routes": ["generate:MTPGenerationBatch"]}]
    row = harness.summarize(base)["rows"][0]
    assert row["mixed_route_cells"] == ["quiet"]
    assert row["phase1_go"] is None and row["go_reasons"] == []


def test_phase0_gsm8k_comparisons_require_complete_token_evidence(tmp_path):
    import json

    def write_arm(name, rows):
        folder = tmp_path / name / "gsm8k"
        folder.mkdir(parents=True, exist_ok=True)
        receipt = {"dataset": {"questions": 2, "offset": 0, "revision": "fixture"},
                   "checkpoint": "fixture@sha256:1234", "driver_revision": {"head": "test"},
                   "arms": [{"rows": rows, "summary": {"questions": len(rows)}}]}
        (folder / "receipt.json").write_text(json.dumps(receipt))

    rows = [{"index": i, "test_index": i, "text": "42", "completion_tokens": 1,
             "predicted": "42", "finish_reason": "stop", "route": "external_draft",
             "accept_hist": {"0": 1}, "depth_hist": {"1": 1}, "token_ids": [42]}
            for i in range(2)]
    write_arm("off", rows)

    def compare(other):
        write_arm("on", other)
        return harness.compare_gsm8k_arms(tmp_path, ["off", "on"], "off", {})

    assert compare(rows)["arms"]["on"]["exact"] is True
    mismatch = compare([rows[0], {**rows[1], "token_ids": [17]}])["arms"]["on"]
    assert mismatch["token_exact"] is False and mismatch["token_mismatched"] == [1]
    for other in (rows[:1], [rows[0], rows[0]], [rows[0], {**rows[1], "test_index": 2}],
                  [rows[0], {k: v for k, v in rows[1].items() if k != "token_ids"}]):
        result = compare(other)
        assert "on" in result["incomplete"]
        assert result["arms"]["on"]["token_exact"] is None
    write_arm("off", rows[:1])
    assert compare(rows[:1])["arms"]["on"]["token_exact"] is None

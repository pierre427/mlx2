"""CPU tests for scripts/inkling2_context_ladder.py (no model, no GPU).

The end-to-end test drives every phase against a stdlib stub that stands in for
``python -m mlx2.server`` on a loopback port, so the launcher, APC clear, cold/warm,
evidence, comparison and process-group stop paths all run.
"""
from __future__ import annotations

import json
import socket
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import inkling2_context_ladder as L  # noqa: E402


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ev(ids, windows=None, text="ABC-12345-0F ok"):
    return {"token_ids": ids, "windows": windows or [None] * len(ids), "text": text}


def test_parse_stream_reads_ids_windows_and_receipt():
    lines = [
        'data: {"choices":[{"delta":{"content":"A"},"logprobs":{"content":[{"id":5,"logprob":-0.1,'
        '"top_logprobs":[{"id":5,"logprob":-0.1},{"id":7,"logprob":-0.3}]}]}}]}',
        'data: {"choices":[{"delta":{"content":"B"},"logprobs":{"content":[{"id":6,"logprob":-0.2}]},'
        '"finish_reason":"stop"}]}',
        'data: {"choices":[],"mlx2":{"route":"ordinary","cached_tokens":0}}',
        "data: [DONE]",
    ]
    ticks = iter([1.0, 2.0, 3.0])
    out = L.parse_stream(lines, 0.0, clock=lambda: next(ticks))
    assert out["token_ids"] == [5, 6] and out["text"] == "AB"
    assert out["windows"][0] == {5: -0.1, 7: -0.3} and out["windows"][1] is None
    assert out["ttft_s"] == 1.0 and out["decode_tok_s"] == 1.0
    assert out["receipt"]["route"] == "ordinary"


def test_parse_stream_refuses_a_stream_without_receipt():
    with pytest.raises(ValueError):
        L.parse_stream(["data: [DONE]"], 0.0)


def test_classify_identical_near_tie_failed_and_unbound():
    code = "ABC-12345-0F"
    ref = {"token_ids": [1, 2, 3], "text": code}
    assert L.classify(ref, {"token_ids": [1, 2, 3], "text": code}, None, None, code)["verdict"] == "identical"
    arm = {"token_ids": [1, 2, 4], "text": code}
    tie_ref = _ev([1, 2, 3], [None, None, {3: -0.60, 4: -0.80}])
    tie_arm = _ev([1, 2, 4], [None, None, {4: -0.65, 3: -0.70}])
    row = L.classify(ref, arm, tie_ref, tie_arm, code)
    assert row["verdict"] == "near_tie_equivalent" and row["first_divergence"] == 2
    wide = _ev([1, 2, 4], [None, None, {4: -0.01, 3: -4.0}])
    assert L.classify(ref, arm, tie_ref, wide, code)["verdict"] == "high_margin_divergence"
    unbound = _ev([1, 9, 4], [None, None, {4: -0.65, 3: -0.70}])
    assert "evidence" in L.classify(ref, arm, tie_ref, unbound, code)["why"]
    assert L.classify(ref, {"token_ids": [1, 2, 4], "text": "nope"}, tie_ref, tie_arm, code)["verdict"] == "failed"
    assert L.classify(ref, {"token_ids": [1, 2], "text": code}, None, None, code)["why"] == "length-only difference"


def test_aggregate_and_cold_warm_rules():
    results = [{"first_s": 1.0, "last_s": 3.0, "token_ids": [1, 2, 3]},
               {"first_s": 2.0, "last_s": 5.0, "token_ids": [1, 2]}]
    agg = L.aggregate(results, [100, 100])
    assert agg["ttft_max_s"] == 2.0 and agg["prefill_tok_s_aggregate"] == 100.0
    assert agg["decode_tok_s_aggregate"] == 0.75  # 3 decode tokens over 4 s
    cell = {"cold": [{"token_ids": [1, 2]}], "warm": [{"token_ids": [1, 3], "cached_tokens": 9}],
            "lanes": [{"code": "ABC-12345-0F"}]}
    assert L.cold_warm(cell, 1)["exact_required"] and not L.cold_warm(cell, 1)["all_equal"]
    assert not L.cold_warm(cell, 4)["exact_required"]


def test_server_command_routes_and_lane_width():
    a = type("A", (), {"server_python": Path("py"), "port": 8396})()
    by = {p["name"]: p for p in L.PHASES}
    ordinary = L.server_command(a, by["ordinary_b1"], Path("m"), Path("t"), Path("p"), 5000)
    dspark = L.server_command(a, by["dspark_b4_refusal"], Path("m"), Path("t"), Path("p"), 5000)
    sink = L.server_command(a, by["sink_sdpa_b1"], Path("m"), Path("t"), Path("p"), 5000)
    assert "--ordinary" in sink and sink[sink.index("--execution-policy") + 1] == "p"
    assert "--ordinary" in ordinary and "--external-draft" not in ordinary
    assert dspark[dspark.index("--max-lanes") + 1] == "4" and "--external-draft" in dspark
    assert dspark[dspark.index("--execution-policy") + 1] == "p"
    assert "8282" not in ordinary + dspark


def test_main_refuses_8282_and_a_nonempty_directory(tmp_path):
    with pytest.raises(SystemExit):
        L.main(["--model", str(tmp_path), "--out-dir", str(tmp_path / "o"), "--port", "8282"])
    (tmp_path / "o2").mkdir()
    (tmp_path / "o2" / "x").write_text("x")
    with pytest.raises(SystemExit):
        L.main(["--model", str(tmp_path), "--out-dir", str(tmp_path / "o2"), "--port", str(_free_port())])


STUB = textwrap.dedent('''\
    #!{python}
    """Stand-in for python -m mlx2.server: toy greedy decode, fake APC, thinking channel, knobs, refusals."""
    import hashlib, json, sys
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    argv = sys.argv[1:]
    if "--ast" in argv:
        print("server.py: error: --native-mtp requested, but inkling2 artifact has no implemented native MTP route",
              file=sys.stderr)
        raise SystemExit(2)
    port = int(argv[argv.index("--port") + 1])
    lanes = int(argv[argv.index("--max-lanes") + 1])
    spec = "--external-draft" in argv
    policy = json.load(open(argv[argv.index("--execution-policy") + 1])) if "--execution-policy" in argv else {{}}
    if spec and lanes != 1:
        print("ValueError: Inkling 2 DSpark serving is B1 only: start with --max-lanes 1", file=sys.stderr)
        raise SystemExit(1)
    hard = policy.get("inkling2_hard_no_thinking", False)
    # Adapter defaults since 2026-11-22: the composed set and split-K are on
    # unless the policy says otherwise.
    defaults = {{"inkling2_sink_sdpa": True, "inkling2_sink_slot": True, "inkling2_verify_splitk": True,
                "inkling2_fuse_controllers": True, "inkling2_fuse_residual": True,
                "inkling2_fuse_sconv": True, "inkling2_fuse_attention": True}}
    names = {{"inkling2_hard_no_thinking": "hard_no_thinking", "inkling2_sink_sdpa": "sink_sdpa",
             "inkling2_prefill_mask_mb": "prefill_mask_mb", "inkling2_sink_slot": "sink_slot",
             "inkling2_fuse_controllers": "fuse_controllers", "inkling2_fuse_residual": "fuse_residual",
             "inkling2_fuse_sconv": "fuse_sconv", "inkling2_fuse_attention": "fuse_attention",
             "inkling2_verify_splitk": "verify_splitk"}}
    effective = {{**defaults, **policy}}
    kernel_policy = {{names[k]: effective.get(k, False if k != "inkling2_prefill_mask_mb" else None)
                     for k in names}}
    sink = kernel_policy["sink_sdpa"]
    seen, calls = set(), [0]

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self._json({{"status": "ok"}})
            return self._json({{"execution": {{"checkpoint": "Inkling-2-Mini@stubfp",
                                              "kernel_policy": kernel_policy,
                                              "kernel_counters": {{
                                                  "sink_sdpa_calls": calls[0] if sink and not kernel_policy["verify_splitk"] else 0,
                                                  "sink_slot_calls": calls[0] if kernel_policy["sink_slot"] and not kernel_policy["verify_splitk"] else 0,
                                                  "verify_splitk_calls": calls[0] if kernel_policy["verify_splitk"] else 0}},
                                              "dspark": {{"prefill_reclaim": spec}}}},
                               "metal_peak_bytes": 2**31, "process_physical_footprint_bytes": 2**32}})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/v1/admin/apc/clear":
                assert self.headers["Authorization"].startswith("Bearer ")
                seen.clear()
                return self._json({{"cleared": True}})
            text = body["messages"][0]["content"]
            code = text.split("vault is ")[1].split(".")[0]
            key = hashlib.sha256(text.encode()).hexdigest()
            hit = key in seen
            seen.add(key)
            calls[0] += 1
            # token plan: (id, alternative, selected logprob, alternative logprob, channel, piece)
            plan = [[100, 900, -0.1, -2.0, "content" if hard else "reasoning", code],
                    [101, 444, -0.6, -0.8, "content", " summary"],
                    [102, 555, -0.6, -0.8, "content", " text"]]
            if lanes == 4:  # batched: position 2 flips at a near tie (B4 vs B1 lane 0)
                plan[2][:2] = [555, 102]
                if hit and int(key[0], 16) % 2 == 0:  # some warm replays flip position 1 at a near tie
                    plan[1][:2] = [444, 101]
            if sink:  # a high-margin flip at position 2
                plan[2][:4] = [666, 102, -0.05, -3.0]
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
            top = body.get("top_logprobs", 0)
            for tok, alt, lp, alp, channel, piece in plan[:body["max_tokens"]]:
                item = {{"id": tok, "logprob": lp}}
                if top:
                    item["top_logprobs"] = [{{"id": tok, "logprob": lp}}, {{"id": alt, "logprob": alp}}]
                delta = {{"reasoning_content" if channel == "reasoning" else "content": piece}}
                ev = {{"choices": [{{"delta": delta, "logprobs": {{"content": [item]}}}}]}}
                self.wfile.write(("data: " + json.dumps(ev) + "\\n\\n").encode())
            receipt = {{"route": "external_draft" if spec else "ordinary", "prompt_tokens": 50,
                        "cached_tokens": 10 if hit else 0,
                        "request_controls": {{"thinking_control": {{
                            "mode": "hard_text_prefill" if hard else "template_effort"}}}}}}
            if spec:
                receipt["speculation"] = {{"verify_accept_hist": {{"2": 1}}, "verify_depth_hist": {{"7": 1}},
                                          "proposed": 7, "verify_full_accept_rounds": 0}}
            self.wfile.write(("data: " + json.dumps({{"choices": [], "mlx2": receipt}}) + "\\n\\n").encode())
            self.wfile.write(b"data: [DONE]\\n\\n")

    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
''')


class _Tok:
    def encode(self, text, add_special_tokens=False):
        return [0] * (len(text) // 4)


def _run(tmp_path, monkeypatch, *extra):
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: _Tok())
    stub = tmp_path / "stub_server"
    stub.write_text(STUB.format(python=sys.executable))
    stub.chmod(0o755)
    model = tmp_path / "Inkling-2-Mini"
    model.mkdir(exist_ok=True)
    out = tmp_path / "runs"
    rc = L.main(["--model", str(model), "--out-dir", str(out), "--port", str(_free_port()),
                 "--cells", "256,512", "--b4-cells", "256,512", "--ab-cells", "256", "--max-tokens", "8",
                 "--server-python", str(stub), "--probe-python", str(stub), "--load-timeout", "30",
                 "--swap-stop-mb", "1e9", *extra])  # host swap moves under other jobs
    return rc, json.loads((out / "ladder.json").read_text()), out


def test_oracle_scores_both_channels():
    code = "ABC-12345-0F"
    assert L.oracle_channels({"text": "", "reasoning_text": f"it is {code}"}, code) == {"pass": True, "channel": "reasoning"}
    assert L.oracle_channels({"text": "ABC–12345–0F", "reasoning_text": code}, code)["channel"] == "both"
    assert L.oracle_channels({"text": "x", "reasoning_text": ""}, code) == {"pass": False, "channel": None}


def test_high_margin_and_missing_token_rules():
    code = "ABC-12345-0F"
    ref, arm = {"token_ids": [1, 2], "text": code}, {"token_ids": [1, 3], "text": code}
    far = _ev([1, 3], [None, {3: -0.01, 7: -0.9, 8: -1.0, 9: -1.1, 10: -1.2}])  # 2 absent, window spans > 0.5
    near = _ev([1, 2], [None, {2: -0.6, 3: -0.8}])
    assert L.classify(ref, arm, near, far, code)["verdict"] == "high_margin_divergence"
    short = _ev([1, 3], [None, {3: -0.01, 7: -0.2}])  # 2 absent, window does not bound it
    assert L.classify(ref, arm, near, short, code)["verdict"] == "unclassified"
    # the first bound record wins; an unbound one is skipped
    assert L.classify(ref, arm, [_ev([9, 9], [None, {9: 0.0}]), near], _ev([1, 3], [None, {3: -0.7, 2: -0.75}]),
                      code)["verdict"] == "near_tie_equivalent"


def test_estimate_is_an_upper_bound_shape():
    a = type("A", (), {"cells": [1024, 131072], "b4_cells": [1024], "ab_cells": [1024], "max_tokens": 1024})()
    est = L.estimate_plan(L.PHASES, a)
    assert est["phases"]["ordinary_b1"]["cells_s"][131072] > 600  # 548 s TTFT + 2 x 1024 tokens at 15 tok/s
    assert est["phases"]["dspark_b4_refusal"]["phase_s"] == L.REFUSAL_S
    assert L.estimate_cell_s(1024, 4, 1024) > L.estimate_cell_s(1024, 1, 1024)


def test_end_to_end_against_a_stub_server(tmp_path, monkeypatch):
    rc, report, out = _run(tmp_path, monkeypatch)
    assert report["ast_probe"]["verdict"] == "fails_closed"
    phases = {p["name"]: p for p in report["phases"]}
    assert list(phases) == [p["name"] for p in L.PHASES]
    refusal = phases["dspark_b4_refusal"]
    assert refusal["verdict"] == "fails_closed" and "B1 only" in refusal["message"]
    assert refusal["server_stop"] == "stopped" and not refusal["cells"]
    assert [c["cell"] for c in phases["ordinary_b1"]["cells"]] == [256, 512]
    assert [c["cell"] for c in phases["sink_sdpa_b1"]["cells"]] == [256]
    for p in phases.values():
        assert p["status"] == "completed" and p["server_stop"] == "stopped", (p["name"], p.get("status"))
        assert not (out / p["name"] / "admin-token").exists()
        if p.get("kind"):
            continue
        assert p["checkpoint"] == "Inkling-2-Mini@stubfp"
        v = p["verdict"]
        assert not (v["oracle_failures"] or v["warm_misses"] or v["route_mismatches"] or v["empty_content"]
                    or v["spec_cells_without_verify_rounds"] or v["width_one_cold_warm_exact_failures"]
                    or v["thinking_mode_mismatches"] or v["knob_not_engaged"] or v["batched_cold_warm_failures"])
        for cell in p["cells"]:
            assert cell["evidence_unmet"] == [], (p["name"], cell["cell"], cell["evidence_unmet"])
    # the code is scored in whichever channel it appeared in
    assert phases["ordinary_b1"]["verdict"]["oracle_channels"] == {"reasoning": 4}
    assert phases["hard_no_thinking_b1"]["verdict"]["oracle_channels"] == {"content": 2}
    assert phases["hard_no_thinking_b1"]["cells"][0]["cold"][0]["thinking_mode"] == "hard_text_prefill"
    # width one: the warm timed request carries the windows; no extra request was needed
    b1 = phases["ordinary_b1"]["cells"][0]
    assert b1["warm"][0]["evidence_bearing"] and b1["evidence"][0][0]["shape"] == "warm_timed"
    assert b1["evidence_attempts"] == []
    assert phases["sink_sdpa_b1"]["cells"][0]["knob_engaged"] is True
    assert phases["mask512_b1"]["cells"][0]["knob_engaged"] is True
    # width four: cold/warm flips get windows from both a warm- and a cold-shaped attempt
    b4 = phases["ordinary_b4"]
    flips = [lane for c in b4["cells"] for lane in c["cold_warm"]["lanes"] if not lane["equal"]]
    assert flips, "the stub flips some warm replays"
    assert all(lane["classification"]["verdict"] == "near_tie_equivalent" for lane in flips)
    shapes = {e["shape"] for c in b4["cells"] for lane in c["evidence"] for e in lane}
    assert shapes == {"warm", "cold"}
    assert all(e["max_tokens"] <= 3 for c in b4["cells"] for lane in c["evidence"] for e in lane)
    assert b4["verdict"]["batched_cold_warm_divergences"] == {"near_tie_equivalent": len(flips)}
    counts = {c["arm"]: c["counts"] for c in report["comparisons"]}
    expected = {"dspark_b1": {"identical": 2}, "ordinary_b4": {"near_tie_equivalent": 2},
                "sink_sdpa_b1": {"high_margin_divergence": 1}, "mask512_b1": {"identical": 1},
                "dspark_reclaim_b1": {"identical": 2}}
    assert {k: counts[k] for k in expected} == expected
    # Every armed phase is compared; the stub flips position 2 whenever sink
    # SDPA is effective (it is part of the new defaults, so default_b1 too).
    assert set(counts) == {p["name"] for p in L.PHASES if p.get("reference")}
    assert counts["default_b1"] == {"high_margin_divergence": 2}
    sink_row = next(c for c in report["comparisons"] if c["arm"] == "sink_sdpa_b1")["rows"][0]
    assert sink_row["first_divergence"] == 2 and sink_row["ref_window"] and sink_row["arm_window"]
    assert rc == 0, report["status"]


def test_budget_skips_cells_and_phases(tmp_path, monkeypatch):
    rc, report, _ = _run(tmp_path, monkeypatch, "--budget-min", "0.05", "--phases", "ordinary_b1,sink_sdpa_b1")
    phases = {p["name"]: p for p in report["phases"]}
    assert rc == 1 and report["status"].startswith("partial")
    skipped = [c for p in phases.values() for c in p["cells"] if c.get("skipped")]
    assert skipped and all("budget" in c["skipped"] for c in skipped)


def test_dry_run_reports_commands_and_estimate(tmp_path, monkeypatch):
    rc, report, _ = _run(tmp_path, monkeypatch, "--dry-run")
    assert rc == 0 and report["status"] == "dry_run"
    assert set(report["dry_run"]) == {p["name"] for p in L.PHASES}
    assert report["budget"]["estimate"]["total_s"] > 0

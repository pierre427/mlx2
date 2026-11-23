"""CPU tests for the junction-checkpoint GPU check scripts.

``scripts/research/junction_neartie_classify.py``: first-divergence near-tie
classification (every AGENTS.md clause, missing-token handling, the row gates
and the GSM8K functional oracle).  ``scripts/research/junction_restore_decode_ab.py``:
the interleaving scheduler, the verdict, the layout probes, and the whole
in-process plant/restore/decode flow on tiny random models.
"""
import importlib.util
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/research" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nt = _load("junction_neartie_classify")
ab = _load("junction_restore_decode_ab")


def w(selected, *pairs):
    """A window: selected id and ``(id, logprob)`` entries."""
    top = sorted(([int(t), float(lp)] for t, lp in pairs), key=lambda x: (-x[1], x[0]))
    return {"selected": selected, "selected_logprob": dict(pairs).get(selected), "top": top}


# ------------------------------------------------------------- contract --

def test_two_way_clause():
    a = w(1, (1, -0.60), (2, -0.90), (3, -3.0))
    b = w(2, (2, -0.55), (1, -0.95), (3, -3.0))
    result = nt.contract(a, b)
    assert result["verdict"] == "contract_pass" and result["clause"] == "two_way"
    assert result["margins"]["a"] == pytest.approx(0.30)


def test_two_way_needs_both_margins_within_the_ceiling():
    # Same top two, but A's margin is 0.7 nat: B's token is 0.7 below A's top.
    a = w(1, (1, -0.10), (2, -0.80), (3, -4.0))
    b = w(2, (2, -0.30), (1, -0.50), (3, -4.0))
    result = nt.contract(a, b)
    assert result["verdict"] == "high_margin"
    assert result["cross"]["b_selected_in_a"]["status"] == "far"


def test_exact_tie_clause_when_other_top_two_differs():
    # A's top is an exact tie (1, 2); B selected the tied token 2 with margin
    # 0.3 over token 3, and A's token 1 sits 1.2 nat down in B: only the
    # exact-tie clause can accept it.
    a = w(1, (1, -0.69), (2, -0.69), (3, -2.5))
    b = w(2, (2, -0.40), (3, -0.70), (1, -1.60))
    result = nt.contract(a, b)
    assert result["verdict"] == "contract_pass" and result["clause"] == "exact_tie"


def test_exact_tie_needs_the_other_margin_within_the_ceiling():
    a = w(1, (1, -0.69), (2, -0.69), (3, -2.5))
    b = w(2, (2, -0.20), (3, -0.90), (1, -1.60))
    assert nt.contract(a, b)["verdict"] == "high_margin"


def test_exact_tie_needs_a_tied_token_selected():
    a = w(1, (1, -0.69), (2, -0.69), (3, -0.75))
    b = w(3, (3, -0.40), (2, -0.60), (1, -1.6))   # 3 is not tied at A's top
    result = nt.contract(a, b)
    assert result["clause"] != "exact_tie"
    assert result["verdict"] == "high_margin"     # A's token 1 is 1.2 below B's top


def test_multi_way_clause_with_different_top_two():
    a = w(1, (1, -1.00), (2, -1.05), (3, -1.20))
    b = w(2, (2, -1.00), (3, -1.02), (1, -1.30))
    result = nt.contract(a, b)
    assert {t for t, _ in a["top"][:2]} != {t for t, _ in b["top"][:2]}
    assert result["verdict"] == "contract_pass" and result["clause"] == "multi_way"


def test_missing_cross_token_not_proven_far_is_unclassified():
    # B's window ends 0.3 below its top and lacks A's token 1: it could be
    # anywhere below -1.3, inside or outside the ceiling.
    a = w(1, (1, -1.00), (2, -1.05), (3, -1.20))
    b = w(2, (2, -1.00), (3, -1.10), (4, -1.30))
    result = nt.contract(a, b)
    assert result["verdict"] == "unclassified"
    assert result["cross"]["a_selected_in_b"]["status"] == "missing"


def test_missing_cross_token_proven_far_fails():
    # B's K-th entry is already 0.8 below its top: an absent token is further.
    a = w(1, (1, -1.00), (2, -1.05), (3, -1.20))
    b = w(2, (2, -0.20), (3, -0.40), (4, -1.00))
    result = nt.contract(a, b)
    assert result["verdict"] == "high_margin"
    assert result["cross"]["a_selected_in_b"]["status"] == "far"


def test_unreadable_windows_are_unclassified():
    single = w(1, (1, -0.1))
    assert nt.contract(single, w(2, (2, -0.1), (1, -0.2)))["verdict"] == "unclassified"
    not_top = w(2, (1, -0.1), (2, -0.4))           # greedy selected a non-top token
    assert nt.contract(w(1, (1, -0.1), (2, -0.2)), not_top)["verdict"] == "unclassified"
    assert nt.contract(None, not_top)["verdict"] == "unclassified"


def test_contract_agrees_with_the_shared_near_tie_rule():
    from mlx2.near_tie import near_tie_equivalent

    cases = [
        (w(1, (1, -0.60), (2, -0.90), (3, -3.0)), w(2, (2, -0.55), (1, -0.95), (3, -3.0)), True),
        (w(1, (1, -0.69), (2, -0.69), (3, -2.5)), w(2, (2, -0.40), (3, -0.70), (1, -1.60)), True),
        (w(1, (1, -1.00), (2, -1.05), (3, -1.20)), w(2, (2, -1.00), (3, -1.02), (1, -1.30)), True),
        (w(1, (1, -0.10), (2, -0.80), (3, -4.0)), w(2, (2, -0.30), (1, -0.50), (3, -4.0)), False),
    ]
    for a, b, expected in cases:
        ours = nt.contract(a, b)["verdict"] == "contract_pass"
        shared = near_tie_equivalent(a["selected"], dict(map(tuple, a["top"])),
                                     b["selected"], dict(map(tuple, b["top"])))
        assert ours == shared == expected


def test_window_from_the_servers_own_serializer():
    from mlx2.logprobs import token_logprob

    class Tok:
        def convert_ids_to_tokens(self, ids):
            return [f"t{i}" for i in ids]

    row = np.log(np.array([0.05, 0.40, 0.35, 0.15, 0.05], dtype=np.float32))
    entry = token_logprob(row, 1, Tok(), top_n=3, array_module=np)
    window = nt.window_from_entry(entry)
    assert window["selected"] == 1
    assert [t for t, _ in window["top"]] == [1, 2, 3]
    assert nt.margin(window) == pytest.approx(float(row[1] - row[2]))
    assert nt.window_from_entry({"token": "x"}) is None


# ------------------------------------------------------------ row gates --

def _oracle(passed=True):
    return {"passed": passed}


def _evidence(trace_a, trace_b, wa, wb):
    return {"a": {"window": wa, "trace": trace_a}, "b": {"window": wb, "trace": trace_b}}


def test_row_near_tie_and_oracle_gate():
    ta, tb = [5, 6, 1, 9], [5, 6, 2, 7]
    wa = w(1, (1, -0.60), (2, -0.90))
    wb = w(2, (2, -0.55), (1, -0.95))
    row = nt.classify_row(ta, tb, _evidence(ta[:3], tb[:3], wa, wb), _oracle(True))
    assert row["verdict"] == "near_tie_equivalent" and row["first_divergence"] == 2
    assert row["clause"] == "two_way" and row["tokens"] == {"a": 1, "b": 2}
    row = nt.classify_row(ta, tb, _evidence(ta[:3], tb[:3], wa, wb), _oracle(False))
    assert row["verdict"] == "oracle_failed"


def test_row_gates():
    wa, wb = w(1, (1, -0.6), (2, -0.9)), w(2, (2, -0.55), (1, -0.95))
    assert nt.classify_row([1, 2], [1, 2], None, _oracle())["verdict"] == "no_divergence_on_replay"
    length = nt.classify_row([1, 2], [1, 2, 3], None, _oracle())
    assert length["verdict"] == "unclassified" and "length-only" in length["reason"]
    assert nt.classify_row([1], [2], None, _oracle())["verdict"] == "unclassified"
    # The window replay emitted a different prefix: not reproduced.
    row = nt.classify_row([4, 1], [4, 2], _evidence([3, 1], [4, 2], wa, wb), _oracle())
    assert row["verdict"] == "unclassified" and "reproduce" in row["reason"]
    # The window selected a different token than the trace phase did.
    row = nt.classify_row([4, 1], [4, 2], _evidence([4, 1], [4, 2], wb, wb), _oracle())
    assert row["verdict"] == "unclassified"


def test_functional_oracle():
    ok = nt.functional_oracle("so 18\n#### 18", "#### 18", "18")
    assert ok["passed"] and ok["a_correct"] and ok["b_correct"]
    same_wrong = nt.functional_oracle("#### 17", "the answer is 17", "18")
    assert same_wrong["passed"] and not same_wrong["a_correct"] and same_wrong["same_answer"]
    split = nt.functional_oracle("#### 18", "#### 17", "18")
    assert not split["passed"]
    assert not nt.functional_oracle("no number", "none", "18")["passed"]


def test_selection_order_and_counts():
    a = [{"test_index": 0, "text": "x", "completion_tokens": 3},
         {"test_index": 1, "text": "y", "completion_tokens": 4},
         {"test_index": 2, "text": "z", "completion_tokens": 5}]
    b = [{"test_index": 0, "text": "x", "completion_tokens": 3},
         {"test_index": 1, "text": "y", "completion_tokens": 9},   # reasoning-only divergence
         {"test_index": 2, "text": "q", "completion_tokens": 5}]
    assert nt.select_questions(a, b, "text") == [2]
    assert nt.select_questions(a, b, "any") == [1, 2]
    assert nt.first_divergence([1, 2, 3], [1, 2, 4]) == 2
    assert nt.first_divergence([1, 2], [1, 2, 3]) == 2
    assert nt.first_divergence([1], [1]) is None
    assert nt.replay_sequence([0, 1, 2, 3, 4], [1, 3], [0, 1], "original") == [0, 1, 2, 3]
    assert nt.replay_sequence([0, 1, 2, 3, 4], [1, 3], [0, 1], "selected") == [0, 1, 3]
    counts = nt.count_verdicts([{"verdict": "near_tie_equivalent", "clause": "multi_way"},
                                {"verdict": "high_margin"}, {"verdict": "unclassified"}])
    assert counts["near_tie_equivalent"] == 1 and counts["failures"] == 1
    assert counts["near_tie_clauses"] == {"multi_way": 1} and counts["questions"] == 3


def test_classifier_refuses_reserved_port_and_nonempty_output(tmp_path):
    common = ["--gsm8k-dir", str(tmp_path), "--arm-a", "a", "--arm-b", "b", "--model", str(tmp_path),
              "--cpg-lock", str(tmp_path)]
    with pytest.raises(SystemExit):
        nt.main(common + ["--output", str(tmp_path / "out"), "--port", "8282"])
    full = tmp_path / "full"
    full.mkdir()
    (full / "x").write_text("1")
    with pytest.raises(SystemExit, match="nonempty"):
        nt.main(common + ["--output", str(full)])


# ------------------------------------------------------- decode A/B logic --

def test_interleave_schedule_alternates_ab_ba():
    assert ab.interleave_schedule(3) == [(0, "fresh"), (0, "restored"), (1, "restored"), (1, "fresh"),
                                         (2, "fresh"), (2, "restored")]
    with pytest.raises(ValueError):
        ab.interleave_schedule(0)


def _blocks(fresh_tps, restored_tps, rounds=None):
    blocks = []
    for pair, (f, r) in enumerate(zip(fresh_tps, restored_tps)):
        for arm, tps in (("fresh", f), ("restored", r)):
            block = {"pair": pair, "arm": arm, "tokens": 32, "seconds": 32 / tps}
            if rounds is not None:
                block["rounds"] = rounds[arm]
            blocks.append(block)
    return blocks


def test_verdict_restored_slower_faster_and_flat():
    slow = ab.decode_verdict(_blocks([35, 34, 36, 35], [28, 27, 29, 28]), tokens_identical=True, speculative=False)
    assert slow["verdict"] == "restored_slower" and slow["basis"] == "tokens"
    assert slow["median_pair_ratio"] == pytest.approx(0.8, abs=0.02)
    assert slow["pairs_restored_slower"] == 4
    fast = ab.decode_verdict(_blocks([28, 27, 29, 28], [35, 34, 36, 35]), tokens_identical=True, speculative=False)
    assert fast["verdict"] == "restored_faster"
    flat = ab.decode_verdict(_blocks([35, 34, 36, 35], [35.5, 33.8, 36.2, 34.9]),
                             tokens_identical=True, speculative=False)
    assert flat["verdict"] == "no_material_difference"
    # One slow outlier pair is not a consistent slowdown.
    mixed = ab.decode_verdict(_blocks([35, 35, 35, 35], [20, 36, 35, 34]), tokens_identical=True, speculative=False)
    assert mixed["verdict"] == "no_material_difference"
    few = ab.decode_verdict(_blocks([35, 35], [20, 20]), tokens_identical=True, speculative=False)
    assert few["verdict"] == "inconclusive"


def test_verdict_uses_round_rate_when_tokens_differ():
    # Restored emits fewer tokens per block-second only because its tokens
    # differ (lower acceptance); per verify round it is as fast.
    blocks = []
    for pair in range(4):
        blocks.append({"pair": pair, "arm": "fresh", "tokens": 32, "seconds": 1.0, "rounds": 10})
        blocks.append({"pair": pair, "arm": "restored", "tokens": 24, "seconds": 1.0, "rounds": 10})
    result = ab.decode_verdict(blocks, tokens_identical=False, speculative=True)
    assert result["basis"] == "rounds" and result["verdict"] == "no_material_difference"
    # Ordinary decode costs one step per token whatever the token: token rate.
    ordinary = ab.decode_verdict(blocks, tokens_identical=False, speculative=False)
    assert ordinary["basis"] == "tokens" and ordinary["verdict"] == "restored_slower"
    identical = ab.decode_verdict(blocks, tokens_identical=True, speculative=True)
    assert identical["basis"] == "tokens"


def test_verdict_excludes_short_blocks():
    blocks = _blocks([35, 34, 36, 35], [28, 27, 29, 28])
    for block in blocks:
        if block["arm"] == "restored" and block["pair"] >= 1:
            block["short"] = True
    result = ab.decode_verdict(blocks, tokens_identical=True, speculative=False)
    assert result["short_blocks_excluded"] == 3 and result["verdict"] == "inconclusive"


def test_restore_engagement_checks():
    rows = {"fresh": [{"x": 1}], "restored": [{"x": 1}]}
    ok = {"fresh": {"paired_cache_resumes": 0, "prefill_rounds": 2},
          "restored": {"paired_cache_resumes": 1, "prefill_rounds": 1}}
    assert ab.restore_engagement("dspark", ok, rows) is None
    assert ab.restore_engagement("ordinary", ok, rows) is None
    cold = {"fresh": {"prefill_rounds": 2}, "restored": {"prefill_rounds": 2}}
    assert "paired_cache_resumes" in ab.restore_engagement("dspark", cold, rows)
    assert "prefill" in ab.restore_engagement("ordinary", cold, rows)
    assert "located" in ab.restore_engagement("ordinary", ok, {"fresh": [], "restored": [{"x": 1}]})


def test_overall_verdict_pools_questions():
    records = [{"route": "ordinary", "tokens_identical": True,
                "blocks": _blocks([35, 34], [28, 27])},
               {"route": "ordinary", "tokens_identical": True,
                "blocks": _blocks([36, 35], [29, 28])}]
    overall = ab.overall_verdict(records, 0.05)
    assert overall["ordinary"]["pairs"] == 4 and overall["ordinary"]["verdict"] == "restored_slower"


def test_buffer_facts_detect_views_and_aliases():
    base = mx.zeros((1, 2, 8, 4), dtype=mx.bfloat16)
    mx.eval(base)
    whole = ab.buffer_facts(base)
    assert whole["c_contiguous"] and whole["span"] == whole["nbytes"] == 128
    view = base[..., :5, :]
    facts = ab.buffer_facts(view)
    assert not facts["c_contiguous"] and facts["shape"] == [1, 2, 5, 4]
    ranges = [(whole["start"], whole["start"] + whole["span"])]
    assert ab.overlaps(facts, ranges)
    assert ab.overlaps(ab.buffer_facts(mx.stop_gradient(base)), ranges)
    copy = mx.array(base) + 0
    mx.eval(copy)
    assert not ab.overlaps(ab.buffer_facts(copy), ranges)
    rows = ab.layout_facts({"k": [base, view]}, ranges)
    assert ab.layout_summary(rows) == {"arrays": 2, "unreadable": 0, "non_contiguous": 1, "aliases_store": 2,
                                       "bytes": 208, "aliased_bytes": 208}


def test_op_counter_counts_and_restores():
    original = mx.concatenate
    with ab.OpCounter() as counter:
        mx.concatenate([mx.zeros((2,)), mx.ones((2,))])
        mx.contiguous(mx.zeros((2, 2)).T)
    assert counter.counts == {"contiguous": 1, "concatenate": 1}
    assert mx.concatenate is original


def test_tiny_in_process_flow_on_cpu(tmp_path):
    """Plant a junction through the generator, store and restore it through
    APCv2, and decode both arms interleaved, on both routes (CPU, tiny)."""
    out = tmp_path / "ab"
    code = ab.main(["--tiny", "--output", str(out), "--block-tokens", "3", "--pairs", "3",
                    "--prefill-step", "16", "--measure-indices", "2"])
    receipt = json.loads((out / "receipt.json").read_text())
    assert code == 0 and receipt["errors"] == []
    assert receipt["junction"]["position"] == 40
    for route in ("ordinary", "dspark"):
        assert receipt["plant"][route]["purpose"] == "JUNCTION"
    assert receipt["plant"]["dspark"]["has_sidecar"]
    for record in receipt["questions"]:
        assert record["hit"]["cached_tokens"] == 40
        control = record["layout_at_restore"]["summary"]
        # The alias detector's positive control: a restored branch shares
        # every stored buffer before a lane writes to it.
        assert control["arrays"] > 0 and control["aliases_store"] == control["arrays"]
        assert len(record["blocks"]) == 6 and all(b["tokens"] == 3 for b in record["blocks"])
        assert record["tokens_identical"]
        assert record["prefill_stats"]["restored"]["prefill_rounds"] < record["prefill_stats"]["fresh"]["prefill_rounds"]
        assert record["verdict"]["verdict"] in ("restored_slower", "restored_faster", "no_material_difference")
        assert set(record["op_probe"]) == {"fresh", "restored"}
    dspark = next(r for r in receipt["questions"] if r["route"] == "dspark")
    assert all(b["rounds"] > 0 for b in dspark["blocks"])
    assert set(receipt["overall"]) == {"ordinary", "dspark"}


def test_classifier_end_to_end_against_stub_servers(tmp_path, monkeypatch):
    """Both phases against stub servers: the trace phase finds the divergence,
    the window phase reads only that position, and the row is classified."""
    import os as _os
    import sys as _sys
    import threading
    import types
    from http.server import BaseHTTPRequestHandler, HTTPServer

    driver = nt.driver
    train = [{"question": f"T{i}?", "answer": f"x\n#### {i}"} for i in range(5)]
    test = [{"question": "Same?", "answer": "#### 4"}, {"question": "How many?", "answer": "so\n#### 18"}]
    (tmp_path / "a").write_bytes(b"a")
    (tmp_path / "b").write_bytes(b"b")
    monkeypatch.setattr(driver, "dataset_files", lambda: ("rev", {"train": tmp_path / "a", "test": tmp_path / "b"}))
    monkeypatch.setattr(driver, "load_split", lambda path: train if path.name == "a" else test)
    state = {"arm": None, "bodies": []}
    traces = {"a": [5, 6, 1, 9], "b": [5, 6, 2, 7]}
    windows = {"a": {"id": 1, "logprob": -0.60, "top_logprobs": [
                   {"id": 1, "logprob": -0.60}, {"id": 2, "logprob": -0.90}, {"id": 3, "logprob": -3.0}]},
               "b": {"id": 2, "logprob": -0.55, "top_logprobs": [
                   {"id": 2, "logprob": -0.55}, {"id": 1, "logprob": -0.95}, {"id": 3, "logprob": -3.0}]}}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, document):
            body = json.dumps(document).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send({"healthy": True, "model": "M", "route_receipt": {"route": state["arm"]},
                        "apcv2": {"junction_hits": 1}})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["bodies"].append((state["arm"], body))
            arm = state["arm"]
            diverging = "How many" in json.dumps(body["messages"][-1])
            ids = traces[arm] if diverging else [5, 6, 7]
            text = ("#### 18" if arm == "a" else "so #### 18") if diverging else "#### 4"
            mlx2 = {"route": "x"}
            if body.get("return_token_trace"):
                mlx2["token_trace"] = {"token_ids": ids, "execution_widths": [1] * len(ids), "truncated": False}
            choice = {"message": {"content": text}, "finish_reason": "stop"}
            if body.get("logprobs"):
                assert body["logprobs_start"] == body["logprobs_end"] == 3
                assert body["max_tokens"] == 64          # verify shapes unchanged
                choice["logprobs"] = {"content": [windows[arm]]}
            self._send({"choices": [choice], "mlx2": mlx2,
                        "usage": {"completion_tokens": len(ids),
                                  "prompt_tokens_details": {"cached_tokens": 30 if arm == "b" else 0}}})

    stub = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=stub.serve_forever, daemon=True).start()

    class FakeServer:
        pid = 999_999

        def __init__(self):
            self.done = None

        def poll(self):
            return self.done

        def wait(self, timeout=None):
            self.done = 0
            return 0

    real_popen = nt.subprocess.Popen

    def popen(command, *args, **kwargs):
        if "mlx2.server" not in command:
            return real_popen(command, *args, **kwargs)
        state["arm"] = "a" if "--ordinary" in command else "b"
        return FakeServer()

    monkeypatch.setattr(nt.subprocess, "Popen", popen)
    monkeypatch.setattr(nt.ArmServer, "port_busy", lambda self: False)   # the stub owns the port
    monkeypatch.setattr(_os, "killpg", lambda *a: None)
    ownership = types.ModuleType("qualification_gpu_ownership")
    ownership.require_qualification_lease = lambda **kwargs: {"stub": True}
    monkeypatch.setitem(_sys.modules, "qualification_gpu_ownership", ownership)
    from mlx2.adapters import registry

    monkeypatch.setattr(registry, "inspect_model", lambda path: types.SimpleNamespace(artifact_fingerprint="0" * 64))
    lock = tmp_path / "lock"
    lock.mkdir()
    (lock / "owner.json").write_text(json.dumps({"cpg_task": "t", "cpg_generation": 1}))
    model = tmp_path / "M"
    model.mkdir()
    run = tmp_path / "run"
    for name in ("a", "b"):
        (run / name).mkdir(parents=True)
        (run / name / "rows.json").write_text(json.dumps([
            {"test_index": 0, "text": "#### 4", "completion_tokens": 3, "cached_tokens": 0},
            {"test_index": 1, "text": "#### 18" if name == "a" else "so #### 18", "completion_tokens": 4,
             "cached_tokens": 0 if name == "a" else 30}]))
    (run / "receipt.json").write_text(json.dumps({
        "checkpoint": "M@sha256:" + "0" * 64,
        "prompt": {"format": "chat-fewshot", "mode": "chat 5-shot", "max_tokens": 64, "temperature": 0.0},
        "arms": [{"name": "a", "tree": str(ROOT), "args": ["--ordinary"], "request": {"reasoning_effort": "low"}},
                 {"name": "b", "tree": str(ROOT), "args": ["--external-draft"],
                  "request": {"reasoning_effort": "low"}}]}))
    out = tmp_path / "out"
    try:
        code = nt.main(["--gsm8k-dir", str(run), "--arm-a", "a", "--arm-b", "b", "--model", str(model),
                        "--cpg-lock", str(lock), "--output", str(out), "--port", str(stub.server_port)])
    finally:
        stub.shutdown()
    receipt = json.loads((out / "receipt.json").read_text())
    assert code == 0
    assert receipt["checkpoint"] == "M@sha256:" + "0" * 64 and receipt["source_revision"]["head"]
    assert receipt["selection"]["selected"] == [1]
    assert receipt["replay"]["sequence"] == [0, 1]           # original order, junction planted first
    (row,) = receipt["rows"]
    assert row["verdict"] == "near_tie_equivalent" and row["clause"] == "two_way"
    assert row["first_divergence"] == 2 and row["oracle"]["passed"]
    assert row["cached_tokens"]["b"] == row["cached_tokens"]["b_original"] == 30
    assert row["replay_matches_original"] == {"a": True, "b": True}
    assert set(receipt["phases"]) == {"a-trace", "b-trace", "a-window", "b-window"}
    assert receipt["width_scope"]["lane_width_one"] and row["lane_width_one"]
    assert all(p["server_stopped"] == "stopped" for p in receipt["phases"].values())
    # Evidence fields only on the selected question; the arm's request fields on every body.
    for _arm, body in state["bodies"]:
        assert body["reasoning_effort"] == "low" and body["temperature"] == 0
        if "Same?" in json.dumps(body["messages"][-1]):
            assert "logprobs" not in body and "return_token_trace" not in body
    assert not (out / "admin-token").exists()


def test_classifier_edges():
    # A greedy pick a float hair below its window's top is still the top.
    a = w(1, (1, -0.6000001), (2, -0.6), (3, -3.0))
    b = w(2, (2, -0.55), (1, -0.95), (3, -3.0))
    assert nt.contract(a, b)["verdict"] == "contract_pass"
    # Exact tie among three: any tied token counts, not only the first two.
    a = w(1, (1, -1.0), (2, -1.0), (3, -1.0), (4, -4.0))
    b = w(3, (3, -0.5), (4, -0.8), (1, -2.0))
    result = nt.contract(a, b)
    assert result["verdict"] == "contract_pass" and result["clause"] == "exact_tie"
    counts = nt.count_verdicts([{"verdict": "near_tie_equivalent", "clause": "two_way"},
                                {"verdict": "no_divergence_on_replay"}])
    assert not counts["all_near_tie"] and counts["not_reproduced"] == 1
    assert nt.count_verdicts([{"verdict": "near_tie_equivalent", "clause": "two_way"}])["all_near_tie"]
    assert nt.replay_sequence([0, 1, 2], [0], [0, 1], "selected") == [0, 1]


def test_classifier_refuses_one_entry_windows(tmp_path):
    with pytest.raises(SystemExit):
        nt.main(["--gsm8k-dir", str(tmp_path), "--arm-a", "a", "--arm-b", "b", "--model", str(tmp_path),
                 "--cpg-lock", str(tmp_path), "--output", str(tmp_path / "o"), "--top-logprobs", "1"])


def test_arm_server_refuses_a_busy_port(tmp_path):
    import socket

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    try:
        args = types_namespace(port=listener.getsockname()[1], python="python", model=tmp_path, max_context=8)
        server = nt.ArmServer(args, {"tree": str(ROOT), "args": []}, tmp_path / "s", tmp_path / "tok")
        with pytest.raises(RuntimeError, match="already serving"):
            server.__enter__()
        assert not (tmp_path / "s").exists()
    finally:
        listener.close()


def types_namespace(**kwargs):
    import types

    return types.SimpleNamespace(**kwargs)

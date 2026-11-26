"""GSM8K acceptance harness: answer extraction, acceptance math, base-url mode."""
import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("spec_acceptance_gsm8k", ROOT / "scripts/spec_acceptance_gsm8k.py")
gsm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gsm)


def test_answer_extraction():
    assert gsm.extract_answer("She sells 9 eggs, 9 * 2 = 18.\n#### 18") == "18"
    assert gsm.extract_answer("The total is $1,250.00 dollars") == "1250"
    assert gsm.extract_answer("#### -3.50") == "-3.5"
    assert gsm.extract_answer("no digits here") is None
    assert gsm.reference_answer("Natalia sold 48/2 = <<48/2=24>>24 clips.\n#### 72") == "72"


def test_few_shot_prefix_strips_calculator_annotations():
    prefix = gsm.few_shot_prefix([{"question": "Q1?", "answer": "2+2 = <<2+2=4>>4\n#### 4"}], 5)
    assert prefix == "Question: Q1?\nAnswer: 2+2 = 4\n#### 4\n\n"


def test_acceptance_matches_vllm_definitions():
    receipts = [{"verify_accept_hist": {"0": 2, "3": 1}, "proposed": 21},
                {"verify_accept_hist": {"7": 1}, "proposed": 7}, None, {}]
    got = gsm.acceptance_summary(receipts)
    assert got["verify_rounds"] == 4 and got["accepted_tokens"] == 10
    assert got["mean_acceptance_length"] == pytest.approx(1 + 10 / 4)
    assert got["per_position"]["p1"] == pytest.approx(2 / 4)
    assert got["per_position"]["p3"] == pytest.approx(2 / 4)
    assert got["per_position"]["p4"] == pytest.approx(1 / 4)
    assert got["per_position"]["p7"] == pytest.approx(1 / 4)
    assert got["draft_acceptance_rate"] == pytest.approx(10 / 28)
    assert got["requests_without_rounds"] == 2
    assert gsm.acceptance_summary([None]) == {"verify_rounds": 0, "requests_without_rounds": 1}


def test_per_position_divides_by_rounds_that_drafted_the_position():
    # Four rounds: two drafted 4 tokens (accepted 4 and 1), two were shallow
    # rounds that drafted 1 token (accepted 1 and 0).  Position 3 was drafted
    # in 2 rounds and accepted in 1: p3 = 1/2, not 1/4.
    receipts = [{"verify_accept_hist": {"4": 1, "1": 2, "0": 1},
                 "verify_depth_hist": {"4": 2, "1": 2}, "proposed": 10}]
    got = gsm.acceptance_summary(receipts)
    assert got["per_position_denominator"] == "rounds_drafting_position"
    assert got["per_position_basis"] == "deepest_drafted"
    assert got["per_position"]["p1"] == pytest.approx(3 / 4)
    assert got["per_position"]["p2"] == pytest.approx(1 / 2)
    assert got["per_position"]["p3"] == pytest.approx(1 / 2)
    assert got["per_position"]["p4"] == pytest.approx(1 / 2)
    # The old figures stay available and show the under-report.
    assert got["per_position_per_round"]["p3"] == pytest.approx(1 / 4)
    assert got["drafted_at_position"] == {"p1": 4, "p2": 2, "p3": 2, "p4": 2}
    # A position no round drafted has no rate, rather than a 0.
    deep = gsm.acceptance_summary(receipts, draft_depth=6)
    assert deep["per_position"]["p5"] is None and deep["per_position_per_round"]["p5"] == 0


def test_per_position_falls_back_when_any_receipt_lacks_depths():
    receipts = [{"verify_accept_hist": {"2": 1}, "verify_depth_hist": {"2": 1}},
                {"verify_accept_hist": {"0": 1}}]
    got = gsm.acceptance_summary(receipts)
    assert got["per_position_denominator"].startswith("verify_rounds")
    assert got["per_position"] == got["per_position_per_round"]
    assert got["requests_without_depth_hist"] == 1


def test_per_position_refuses_more_accepts_than_drafts():
    with pytest.raises(ValueError, match="more rounds accepted"):
        gsm.acceptance_summary([{"verify_accept_hist": {"3": 1}, "verify_depth_hist": {"2": 1}}])


def test_list_histograms_from_hy4_dfly2_are_read_by_index():
    # Hy4-flash DFly2 writes index-is-value lists: 3 rounds drafted 3 tokens
    # (accepted 3, 3, 0) and 1 tail round drafted 1 token (accepted 1).
    got = gsm.acceptance_summary([{"verify_accept_hist": [1, 1, 0, 2], "verify_depth_hist": [0, 1, 0, 3]}])
    assert got["per_position_denominator"] == "rounds_drafting_position"
    assert got["per_position"]["p1"] == pytest.approx(3 / 4)
    assert got["per_position"]["p2"] == pytest.approx(2 / 3)
    assert got["per_position_per_round"]["p2"] == pytest.approx(2 / 4)


def test_native_mtp_receipt_carries_depths():
    extra = {"mtp": {"stats": {"verify_accept_hist": {"1": 1}, "verify_depth_hist": {"2": 1},
                               "draft_proposed": 2}}}
    assert gsm.speculation_receipt(extra)["verify_depth_hist"] == {"2": 1}


def test_vendored_dataset_matches_its_source_record():
    revision, files = gsm.dataset_files()
    test, train = gsm.load_split(files["test"]), gsm.load_split(files["train"])
    assert revision and len(test) == 1319 and len(train) == 8
    assert all(gsm.reference_answer(item["answer"]) for item in test + train)


def test_base_url_mode_against_a_stub_server(tmp_path, monkeypatch):
    train = [{"question": f"T{i}?", "answer": f"x\n#### {i}"} for i in range(5)]
    test = [{"question": "How many?", "answer": "so\n#### 18"}, {"question": "And?", "answer": "#### 4"}]
    monkeypatch.setattr(gsm, "dataset_files", lambda: ("rev", {"train": tmp_path / "a", "test": tmp_path / "b"}))
    (tmp_path / "a").write_bytes(b"a")
    (tmp_path / "b").write_bytes(b"b")
    monkeypatch.setattr(gsm, "load_split", lambda path: train if path.name == "a" else test)
    seen = []

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
            self._send({"healthy": True, "model": "served-alias", "artifact": "0" * 64, "route_receipt": {"route": "external_draft"}})

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(payload)
            answer = "18" if "How many" in payload["prompt"] else "5"
            self._send({"choices": [{"text": f" so\n#### {answer}", "finish_reason": "stop"}],
                        "usage": {"completion_tokens": 9, "prompt_tokens": 40},
                        "mlx2": {"ttft_seconds": 0.001, "route": "external_draft",
                                 "speculation": {"verify_accept_hist": {"2": 3}, "proposed": 21}}})

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        out = tmp_path / "out"
        assert gsm.main(["--output", str(out), "--base-url", f"http://127.0.0.1:{server.server_port}",
                         "--checkpoint", "M@sha256:" + "0" * 64, "--num-questions", "2",
                         "--format", "completion"]) == 0
    finally:
        server.shutdown()
    receipt = json.loads((out / "receipt.json").read_text())
    summary = receipt["arms"][0]["summary"]
    assert [r["test_index"] for r in receipt["arms"][0]["rows"]] == [0, 1]
    assert summary["accuracy"] == 0.5 and summary["questions"] == 2
    assert summary["acceptance"]["mean_acceptance_length"] == pytest.approx(3.0)
    assert seen[0]["model"] == "served-alias"
    assert seen[0]["temperature"] == 0 and seen[0]["stop"] == gsm.STOPS
    assert seen[0]["prompt"].startswith("Question: T0?\nAnswer: x\n#### 0\n\n")
    assert seen[0]["prompt"].endswith("Question: How many?\nAnswer:")
    assert receipt["checkpoint"].startswith("M@sha256:") and receipt["dataset"]["revision"] == "rev"
    with pytest.raises(RuntimeError, match="nonempty"):
        gsm.main(["--output", str(out), "--base-url", "http://x", "--checkpoint", "M@sha256:0"])


def test_few_shot_turns_mirror_the_completion_examples():
    turns = gsm.few_shot_turns([{"question": "Q1?", "answer": "2+2 = <<2+2=4>>4\n#### 4"},
                                {"question": "Q2?", "answer": "#### 7"}], 5)
    assert turns == [{"role": "user", "content": "Q1?" + gsm.CHAT_INSTRUCTION},
                     {"role": "assistant", "content": "2+2 = 4\n#### 4"},
                     {"role": "user", "content": "Q2?" + gsm.CHAT_INSTRUCTION},
                     {"role": "assistant", "content": "#### 7"}]


def test_default_format_is_chat_fewshot(tmp_path, monkeypatch):
    train = [{"question": f"T{i}?", "answer": f"x\n#### {i}"} for i in range(5)]
    test = [{"question": "How many?", "answer": "so\n#### 18"}]
    monkeypatch.setattr(gsm, "dataset_files", lambda: ("rev", {"train": tmp_path / "a", "test": tmp_path / "b"}))
    (tmp_path / "a").write_bytes(b"a")
    (tmp_path / "b").write_bytes(b"b")
    monkeypatch.setattr(gsm, "load_split", lambda path: train if path.name == "a" else test)
    seen = []

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
            self._send({"healthy": True})

        def do_POST(self):
            seen.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            self._send({"choices": [{"message": {"content": "so\n#### 18"}, "finish_reason": "stop"}],
                        "usage": {"completion_tokens": 9, "prompt_tokens": 40}, "mlx2": {"ttft_seconds": 0.001}})

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        out = tmp_path / "out"
        assert gsm.main(["--output", str(out), "--base-url", f"http://127.0.0.1:{server.server_port}",
                         "--checkpoint", "M@sha256:" + "0" * 64, "--num-questions", "1",
                         "--request", '{"enable_thinking": false}']) == 0
    finally:
        server.shutdown()
    path, payload = seen[0]
    assert path == "/v1/chat/completions" and payload["max_tokens"] == 2048 and "stop" not in payload
    assert payload["enable_thinking"] is False
    assert len(payload["messages"]) == 11 and payload["messages"][1] == {"role": "assistant", "content": "x\n#### 0"}
    assert payload["messages"][-1]["content"] == "How many?" + gsm.CHAT_INSTRUCTION
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["prompt"]["format"] == "chat-fewshot" and receipt["prompt"]["mode"] == "chat 5-shot"
    assert receipt["arms"][0]["summary"]["accuracy"] == 1.0
    row = receipt["arms"][0]["rows"][0]
    assert row["text"] == "so\n#### 18" and row["finish_reason"] == "stop" and row["test_index"] == 0
    with pytest.raises(SystemExit):
        gsm.main(["--output", str(tmp_path / "o2"), "--base-url", "http://x", "--checkpoint", "M@sha256:0",
                  "--chat", "--format", "completion"])


def test_sampling_fields_are_greedy_by_default_and_seeded_per_question():
    assert gsm.sampling_fields(None, 3) == {}
    sampling = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": 100}
    assert gsm.sampling_fields(sampling, 0) == {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": 100}
    assert gsm.sampling_fields(sampling, 7)["seed"] == 107
    assert gsm.sampling_fields({"temperature": 1.0, "top_p": None, "top_k": None, "seed": None}, 2) == {
        "temperature": 1.0}


def test_filters_without_temperature_are_refused(tmp_path):
    with pytest.raises(SystemExit):
        gsm.main(["--output", str(tmp_path / "o"), "--base-url", "http://x",
                  "--checkpoint", "M@sha256:" + "0" * 64, "--top-k", "20"])


def test_full_accept_fraction_counts_rounds_that_hit_the_draft_ceiling():
    receipts = [{"verify_accept_hist": {"4": 1, "1": 2, "0": 1},
                 "verify_depth_hist": {"4": 2, "1": 2, "0": 0},
                 "verify_full_accept_rounds": 3, "proposed": 10}]
    got = gsm.acceptance_summary(receipts)
    # Rounds drafting anything: 4; full: the 4-of-4 round and both 1-of-1.
    assert got["full_accept_rounds"] == 3
    assert got["full_accept_fraction"] == pytest.approx(3 / 4)
    old = [{"verify_accept_hist": {"1": 1}, "verify_depth_hist": {"1": 1}}]
    assert gsm.acceptance_summary(old + receipts)["full_accept_fraction"] is None
    mtp = gsm.speculation_receipt({"mtp": {"stats": {
        "verify_accept_hist": {"2": 1}, "verify_depth_hist": {"2": 1},
        "verify_full_accept_rounds": 1, "draft_proposed": 2}}})
    assert mtp["verify_full_accept_rounds"] == 1


def test_annotated_format_keeps_markers_and_records_calculator_counters(tmp_path, monkeypatch):
    train = [{"question": f"T{i}?", "answer": f"2*{i} = <<2*{i}={2 * i}>>{2 * i}\n#### {2 * i}"} for i in range(5)]
    test = [{"question": "How many?", "answer": "so\n#### 18"}]
    monkeypatch.setattr(gsm, "dataset_files", lambda: ("rev", {"train": tmp_path / "a", "test": tmp_path / "b"}))
    (tmp_path / "a").write_bytes(b"a")
    (tmp_path / "b").write_bytes(b"b")
    monkeypatch.setattr(gsm, "load_split", lambda path: train if path.name == "a" else test)
    seen = []
    calc = {"plans": 1, "marker_open": 1, "tokens_written": 1}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, document):
            body = json.dumps(document).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send({"healthy": True})

        def do_POST(self):
            seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self._send({"choices": [{"message": {"content": "9*2 = <<9*2=18>>18\n#### 18", "reasoning_content": "<<1+1="},
                                     "finish_reason": "stop"}],
                        "usage": {"completion_tokens": 9}, "mlx2": {"ttft_seconds": 0.001,
                        "route": {"amoeba_det_tiles_request": {"calculator": calc}}}})

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        out = tmp_path / "out"
        assert gsm.main(["--output", str(out), "--base-url", f"http://127.0.0.1:{server.server_port}",
                         "--checkpoint", "M@sha256:" + "0" * 64, "--num-questions", "1",
                         "--format", "chat-fewshot-annotated"]) == 0
    finally:
        server.shutdown()
    messages = seen[0]["messages"]
    assert messages[1]["content"] == "2*0 = <<2*0=0>>0\n#### 0"
    assert all(m["content"].endswith(gsm.ANNOTATED_INSTRUCTION) for m in messages if m["role"] == "user")
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["prompt"]["format"] == "chat-fewshot-annotated"
    row = receipt["arms"][0]["rows"][0]
    assert row["markers"] == 1 and row["reasoning_markers"] == 1 and row["det_tiles"] == calc


def test_stop_at_is_the_next_occurrence_and_stops_new_questions():
    import datetime as dt
    base = dt.datetime(2026, 11, 20, 23, 30).timestamp()
    assert gsm.stop_time("06:30", base) == dt.datetime(2026, 11, 21, 6, 30).timestamp()
    assert gsm.stop_time("23:45", base) == dt.datetime(2026, 11, 20, 23, 45).timestamp()
    assert gsm.stop_time(None) is None
    late = dt.datetime(2026, 11, 21, 6, 45).timestamp()     # just after the stop: already passed
    assert gsm.stop_time("06:30", late) == late
    assert gsm.stop_time("06:30", dt.datetime(2026, 11, 20, 21, 45).timestamp()) == dt.datetime(2026, 11, 21, 6, 30).timestamp()

    class Never:
        def call(self, *a, **k):
            raise AssertionError("no question may start after the deadline")

    assert gsm.run_questions(Never(), "m", "", [{"question": "q", "answer": "#### 1"}], chat=True,
                             max_tokens=8, deadline=0.0) == []


def test_model_mode_runs_every_arm_through_every_question(tmp_path, monkeypatch):
    """Regression (2026-11-20): the per-arm readiness bound overwrote the
    --stop-at deadline with a monotonic value, so every arm asked 0 questions
    and the next arm was skipped as ``stop_at``.  Regression (2026-11-24): the
    top-level --request was ignored in --model mode, so enable_thinking=false
    never reached the Ling-3.2-air night arms."""
    import sys as _sys
    import types

    train = [{"question": f"T{i}?", "answer": f"x\n#### {i}"} for i in range(5)]
    test = [{"question": "How many?", "answer": "so\n#### 18"}, {"question": "And?", "answer": "#### 4"}]
    monkeypatch.setattr(gsm, "dataset_files", lambda: ("rev", {"train": tmp_path / "a", "test": tmp_path / "b"}))
    (tmp_path / "a").write_bytes(b"a")
    (tmp_path / "b").write_bytes(b"b")
    monkeypatch.setattr(gsm, "load_split", lambda path: train if path.name == "a" else test)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, document, text=False):
            body = (document if text else json.dumps(document)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain" if text else "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/metrics":
                self._send("# none\n", text=True)
            else:
                self._send({"healthy": True, "route_receipt": {"route": "ordinary"}})

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(payload)
            prompt = json.dumps(payload)
            answer = "18" if "How many" in prompt else "4"
            self._send({"choices": [{"message": {"content": f"so\n#### {answer}"}, "finish_reason": "stop"}],
                        "usage": {"completion_tokens": 9, "prompt_tokens": 40},
                        "mlx2": {"ttft_seconds": 0.001, "route": "ordinary"}})

    seen = []
    stub = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=stub.serve_forever, daemon=True).start()

    class FakeServer:
        pid = 999_999

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    ownership = types.ModuleType("qualification_gpu_ownership")
    ownership.require_qualification_lease = lambda **kwargs: {"stub": True}
    monkeypatch.setitem(_sys.modules, "qualification_gpu_ownership", ownership)
    import mlx2.adapters.registry as registry

    monkeypatch.setattr(registry, "inspect_model", lambda path: types.SimpleNamespace(
        artifact_fingerprint="0" * 64, adapter_type=type("Stub", (), {})))
    real_popen = gsm.subprocess.Popen
    monkeypatch.setattr(gsm.subprocess, "Popen", lambda command, *a, **k: (
        FakeServer() if "mlx2.server" in command else real_popen(command, *a, **k)))
    monkeypatch.setattr(gsm.os, "killpg", lambda *a: None)
    lock = tmp_path / "lock"
    lock.mkdir()
    (lock / "owner.json").write_text(json.dumps({"cpg_task": "t", "cpg_generation": 1}))
    arms = tmp_path / "arms.json"
    arms.write_text(json.dumps([{"name": "a", "tree": str(ROOT)},
                                {"name": "b", "tree": str(ROOT), "request": {"enable_thinking": True, "x": 1}}]))
    model = tmp_path / "M"
    model.mkdir()
    out = tmp_path / "out"
    try:
        assert gsm.main(["--output", str(out), "--model", str(model), "--arms", str(arms),
                         "--cpg-lock", str(lock), "--port", str(stub.server_port),
                         "--num-questions", "2", "--request", '{"enable_thinking": false}']) == 0
    finally:
        stub.shutdown()
    receipt = json.loads((out / "receipt.json").read_text())
    assert [arm["name"] for arm in receipt["arms"]] == ["a", "b"]
    for arm in receipt["arms"]:
        assert "skipped" not in arm and arm["summary"]["questions"] == 2
        assert arm["summary"]["accuracy"] == 1.0 and not arm["stopped_at_deadline"]
    assert receipt["arms"][0]["request"] == {"enable_thinking": False}
    assert receipt["arms"][1]["request"] == {"enable_thinking": True, "x": 1}
    chats = [payload for payload in seen if "messages" in payload]
    assert [payload.get("enable_thinking") for payload in chats] == [False, False, True, True]
    assert [payload.get("x") for payload in chats] == [None, None, 1, 1]



@pytest.mark.parametrize("name", ["unrelated", "M"])
def test_advertised_artifact_must_match_even_with_same_model_name(name):
    with pytest.raises(ValueError, match="requested checkpoint"):
        gsm.served_model_name({"model": name, "target_artifact": "1" * 64}, "M@sha256:" + "0" * 64)


@pytest.mark.parametrize("selection", [["--indices=-1"], ["--indices=1"], ["--offset", "-1"],
                                       ["--offset", "1"], ["--num-questions", "0"]])
def test_invalid_question_selection_fails_before_contacting_server(tmp_path, monkeypatch, selection):
    monkeypatch.setattr(gsm, "dataset_files", lambda: ("fixture", {"train": "train", "test": "test"}))
    monkeypatch.setattr(gsm, "load_split", lambda path: [{"question": "Q", "answer": "#### 1"}])
    with pytest.raises(SystemExit):
        gsm.main(["--output", str(tmp_path), "--base-url", "http://unused", "--checkpoint", "M@fixture",
                  *selection])


def test_http_status_exposes_the_inspected_target():
    from mlx2.server import handler_for
    from test_serving_contract import FakeEngine

    target = "1" * 64
    engine = FakeEngine()
    with HTTPServer(("127.0.0.1", 0), handler_for(engine, target_artifact=target)) as server:
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()
        status = gsm.Client(f"http://127.0.0.1:{server.server_port}").call("/v1/status")
        thread.join(timeout=5)
    assert status["target_artifact"] == target
    assert gsm.served_model_name(status, "target-path@sha256:" + target) == "fixture"


def test_external_route_distinguishes_target_from_composite_digest():
    target, composite = "1" * 64, "2" * 64
    checkpoint = "M@sha256:" + target
    status = {"model": "M", "artifact": composite, "base_artifact": composite,
              "execution": {"route": "external_draft", "artifact": {"checkpoint": "M@sha256:" + target[:16]}}}
    assert gsm.served_model_name(status, checkpoint) == "M"
    status["execution"]["checkpoint"] = "M@" + composite
    assert gsm.served_model_name(status, checkpoint) == "M"  # legacy composite only
    status.update(model="alias", target_artifact=target)
    assert gsm.served_model_name(status, checkpoint) == "alias"
    assert gsm.served_model_name(status, "M@sha256:" + composite) == "alias"
    with pytest.raises(ValueError):
        gsm.served_model_name(status, "M@sha256:" + "3" * 64)


def test_native_mtp_zero_round_histograms_are_evidence():
    receipt = gsm.speculation_receipt({"mtp": {"stats": {"verify_accept_hist": {}, "verify_depth_hist": {}}}})
    assert receipt["verify_accept_hist"] == receipt["verify_depth_hist"] == {}

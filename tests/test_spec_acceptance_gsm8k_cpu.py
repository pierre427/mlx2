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
            self._send({"healthy": True, "route_receipt": {"route": "external_draft"}})

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
                         "--checkpoint", "M@sha256:" + "0" * 64, "--num-questions", "2"]) == 0
    finally:
        server.shutdown()
    receipt = json.loads((out / "receipt.json").read_text())
    summary = receipt["arms"][0]["summary"]
    assert summary["accuracy"] == 0.5 and summary["questions"] == 2
    assert summary["acceptance"]["mean_acceptance_length"] == pytest.approx(3.0)
    assert seen[0]["temperature"] == 0 and seen[0]["stop"] == gsm.STOPS
    assert seen[0]["prompt"].startswith("Question: T0?\nAnswer: x\n#### 0\n\n")
    assert seen[0]["prompt"].endswith("Question: How many?\nAnswer:")
    assert receipt["checkpoint"].startswith("M@sha256:") and receipt["dataset"]["revision"] == "rev"
    with pytest.raises(RuntimeError, match="nonempty"):
        gsm.main(["--output", str(out), "--base-url", "http://x", "--checkpoint", "M@sha256:0"])


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

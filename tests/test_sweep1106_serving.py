"""Bug sweep 2026-11-06, serving lane: CPU regressions."""
import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "spec_acceptance_gsm8k_sweep1106", ROOT / "scripts/spec_acceptance_gsm8k.py"
)
gsm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gsm)


def _native_mtp_receipt():
    # The terminal route receipt of a native self-MTP request
    # (serving.py: "mtp": response.mtp_receipt, "speculation": None;
    # generate.py _complete_responses: mtp_receipt["stats"] = vars(lane.stats)).
    return {"route": "continuous_batched_self_mtp", "speculation": None,
            "mtp": {"route": "continuous_batched_self_mtp", "num_draft": 3,
                    "stats": {"verify_accept_hist": {"0": 1, "3": 3},
                              "draft_proposed": 12, "draft_accepted": 9}},
            "ttft_seconds": 0.001}


def test_gsm8k_native_mtp_receipt_is_scored():
    """SERVING-1: native MTP reported zero verify rounds."""
    spec = gsm.speculation_receipt(_native_mtp_receipt())
    summary = gsm.acceptance_summary([spec])
    assert summary["verify_rounds"] == 4 and summary["accepted_tokens"] == 9
    assert summary["mean_acceptance_length"] == pytest.approx(1 + 9 / 4)
    assert summary["draft_acceptance_rate"] == pytest.approx(9 / 12)
    assert summary["requests_without_rounds"] == 0
    # External-draft receipts keep precedence and pass through unchanged.
    external = {"verify_accept_hist": {"2": 1}, "proposed": 7}
    assert gsm.speculation_receipt({"speculation": external, "mtp": None}) is external
    assert gsm.speculation_receipt({"speculation": None, "mtp": None}) is None


def test_gsm8k_base_url_mode_scores_native_mtp(tmp_path, monkeypatch):
    test = [{"question": "How many?", "answer": "so\n#### 18"}]
    monkeypatch.setattr(gsm, "dataset_files", lambda: ("rev", {"train": tmp_path / "a", "test": tmp_path / "b"}))
    monkeypatch.setattr(gsm, "load_split", lambda path: [] if path.name == "a" else test)
    (tmp_path / "a").write_bytes(b"a")
    (tmp_path / "b").write_bytes(b"b")

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
            self.rfile.read(int(self.headers["Content-Length"]))
            self._send({"choices": [{"text": " #### 18", "finish_reason": "stop"}],
                        "usage": {"completion_tokens": 13, "prompt_tokens": 20},
                        "mlx2": _native_mtp_receipt()})

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        out = tmp_path / "out"
        assert gsm.main(["--output", str(out), "--base-url", f"http://127.0.0.1:{server.server_port}",
                         "--checkpoint", "M@sha256:" + "0" * 64, "--num-questions", "1",
                         "--shots", "0"]) == 0
    finally:
        server.shutdown()
    acceptance = json.loads((out / "receipt.json").read_text())["arms"][0]["summary"]["acceptance"]
    assert acceptance["verify_rounds"] == 4
    assert acceptance["mean_acceptance_length"] == pytest.approx(3.25)


def test_gsm8k_leading_dot_decimal_is_not_its_integer_digits():
    """SERVING-2: ".5" was read as 5 and scored correct against 5."""
    assert gsm.extract_answer("#### .5") == "0.5"
    assert gsm.extract_answer("so the answer is -.25") == "-0.25"
    assert gsm.extract_answer("#### 5") == "5"
    assert gsm.extract_answer("#### 1,250.00") == "1250"
    # Review follow-up: zero forms must not vanish after trailing-zero strip.
    assert gsm.extract_answer("#### .00") == "0"
    assert gsm.extract_answer("#### -.0") == "0"
    assert gsm.extract_answer("#### 0") == "0"


def test_gsm8k_per_position_runs_to_the_draft_depth():
    """SERVING-3: depths past the deepest accepted one were missing, not 0."""
    receipts = [{"verify_accept_hist": {"0": 2, "1": 1, "2": 1}, "proposed": 28}]
    plain = gsm.acceptance_summary(receipts)
    assert list(plain["per_position"]) == ["p1", "p2"]
    assert plain["per_position_basis"] == "deepest_accepted"
    deep = gsm.acceptance_summary(receipts, draft_depth=7)
    assert list(deep["per_position"]) == [f"p{k}" for k in range(1, 8)]
    assert deep["per_position"]["p1"] == 0.5 and deep["per_position"]["p3"] == 0.0
    assert deep["per_position_basis"] == "draft_depth"
    with pytest.raises(ValueError, match="below an accepted depth"):
        gsm.acceptance_summary(receipts, draft_depth=1)

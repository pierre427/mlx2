"""Regression tests for the feature-smoke harness failures seen on Gemma 4 31B.

1. ``responses_stream`` sent no output limit, so a base checkpoint generated to
   the server default (the remaining context, ~32.7K tokens), outlived the
   1800 s check timeout, and kept generating after the check was abandoned.
2. ``apc_admin_quiesce_suspend_resume`` gave up waiting for ``suspended`` and
   returned without resuming, so the server stayed closed to every later check.
"""

import importlib.util
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]
HARNESSES = (
    ROOT / "qualification/runs/series-20260924/feature_smoke.py",
    ROOT / "qualification/runs/quality-campaign-20260919/feature_smoke.py",
)
GENERATION_PATHS = {"/v1/chat/completions", "/v1/completions", "/v1/responses", "/v1/messages"}
OUTPUT_BOUNDS = {"max_tokens", "max_completion_tokens", "max_output_tokens"}


def load(path):
    name = f"feature_smoke_{path.parent.name.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class RecordingHTTP:
    """Answers every check just enough to reach its requests."""

    def __init__(self, admin_state="suspended"):
        self.posts = []
        self.admin_state = admin_state

    def request(self, method, path, body=None, *, stream=False):
        if method == "POST":
            self.posts.append((path, body))
        if path == "/v1/admin/state":
            return {"status": 200, "body": {"state": self.admin_state}}
        if path == "/v1/admin/resume":
            self.admin_state = "serving"
        if path.startswith("/v1/apc/sessions/") and method == "GET":
            return {"status": 200, "body": {"state": "resident"}}
        if path == "/v1/status":
            return {"status": 200, "body": {"max_context": 32768}}
        return {"status": 200, "body": {}, "events": []}

    def get(self, path):
        return self.request("GET", path)

    def post(self, path, body, *, stream=False):
        return self.request("POST", path, body, stream=stream)


class RunAll:
    def __init__(self, http, capabilities):
        self.http = http
        self.args = SimpleNamespace(
            capabilities=set(capabilities),
            admin_token="token",
            check_timeout=1800,
            speculative=False,
            fly=False,
        )

    def check(self, name, function=None, *, applies=True, reason=""):
        if applies and function is not None:
            try:
                function()
            except Exception:  # noqa: BLE001 - only the requests matter here
                pass


@pytest.mark.parametrize("path", HARNESSES, ids=lambda p: p.parent.name)
def test_every_core_generation_request_carries_an_output_bound(path, monkeypatch):
    feature = load(path)
    clock = iter(range(0, 10**9))
    # Polling loops only need their deadlines to pass, not wall time.
    monkeypatch.setattr(feature.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(feature.time, "monotonic", lambda: float(next(clock)))
    http = RecordingHTTP()
    caps = {"text", "tools", "reasoning", "thinking-deferral", "grammar", "apc", "vision", "audio"}
    feature.core_checks(RunAll(http, caps))
    generation = [(p, body) for p, body in http.posts if p in GENERATION_PATHS]
    assert any(p == "/v1/responses" and body.get("stream") for p, body in generation)
    unbounded = [(p, body) for p, body in generation if not OUTPUT_BOUNDS & set(body)]
    assert unbounded == []


class HangingStream(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_GET(self):
        encoded = json.dumps({"status": "ok", "error": None}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(b'data: {"type": "response.created"}\n\n')
        self.wfile.flush()
        # A generation that outlives the check: stream until the client leaves.
        deadline = time.monotonic() + 10
        try:
            while time.monotonic() < deadline:
                self.wfile.write(b'data: {"type": "response.output_text.delta"}\n\n')
                self.wfile.flush()
                time.sleep(0.02)
        except OSError:
            self.server.client_gone.set()


@pytest.mark.parametrize("path", HARNESSES, ids=lambda p: p.parent.name)
def test_timed_out_check_closes_its_abandoned_stream(path, tmp_path):
    feature = load(path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), HangingStream)
    server.client_gone = threading.Event()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        http = feature.HTTP(f"http://127.0.0.1:{server.server_port}", 30)
        matrix = feature.Matrix(SimpleNamespace(raw_dir=tmp_path, check_timeout=0.5), http)
        matrix.check("hang", lambda: http.post("/v1/responses", {"stream": True}, stream=True))
        assert matrix.rows[-1]["status"] == "FAIL"
        assert "timeout" in matrix.rows[-1]["reason"]
        # The server sees the client go away instead of generating for nobody.
        assert server.client_gone.wait(3)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


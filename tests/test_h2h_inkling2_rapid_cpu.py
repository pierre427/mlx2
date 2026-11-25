"""CPU checks for ``scripts/research/h2h_inkling2_rapid.py`` server handling (no model, no GPU)."""

import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import h2h_inkling2_rapid as H


def _group(cmd):
    return subprocess.Popen(cmd, start_new_session=True)


def test_stop_group_after_the_leader_already_exited_does_not_raise():
    # The leader exits on its own and stays an unreaped zombie: macOS answers
    # killpg with EPERM there, which used to escape and lose the arm's receipt.
    proc = _group([sys.executable, "-c", "pass"])
    time.sleep(0.5)
    t0 = time.monotonic()
    H.stop_group(proc.pid, proc)
    assert proc.returncode is not None and time.monotonic() - t0 < 5


def test_stop_group_stops_a_running_group():
    proc = _group([sys.executable, "-c", "import time; time.sleep(60)"])
    time.sleep(0.3)
    H.stop_group(proc.pid, proc)
    assert proc.poll() is not None


def _serve(statuses):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            seen.append(self.path)
            code = statuses.pop(0) if len(statuses) > 1 else statuses[0]
            self.send_response(code)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *a):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, seen


def test_healthy_waits_for_200_and_accepts_a_server_without_health():
    httpd, seen = _serve([503, 200])
    port = httpd.server_address[1]
    try:
        assert H._healthy(port) is False  # mlx2 still loading
        assert H._healthy(port) is True
        assert seen == ["/health", "/health"]
    finally:
        httpd.shutdown()
    httpd, _ = _serve([404])  # a server with no /health route
    try:
        assert H._healthy(httpd.server_address[1]) is True
    finally:
        httpd.shutdown()

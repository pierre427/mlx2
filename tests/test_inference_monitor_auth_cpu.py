"""Synthetic telemetry only; no model checkpoint, inference or GPU work."""

import dataclasses
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import psutil
import pytest

from mlx2 import inference_monitor as monitor


def test_local_key_requires_current_same_user_listener(monkeypatch, tmp_path):
    endpoint = monitor.ServerEndpoint("http://127.0.0.1:18285", pid=123, discovered=True)
    key = tmp_path / "private key"
    key.write_text("synthetic-secret\n")
    key.chmod(0o600)
    process = SimpleNamespace(
        uids=lambda: SimpleNamespace(real=os.getuid()),
        cmdline=lambda: ["python", "-m", "mlx2.server", "--api-key-file=private key"],
        cwd=lambda: str(tmp_path),
        environ=lambda: {"TEST_KEY": "synthetic-env-secret"},
    )
    monkeypatch.setattr(psutil, "Process", lambda pid: process)
    monkeypatch.setattr(monitor, "discover_servers", lambda: (endpoint,))
    assert monitor.local_server_api_key(endpoint) == "synthetic-secret"
    key.chmod(0o644)
    with pytest.raises(ValueError, match="0600"):
        monitor.local_server_api_key(endpoint)
    process.cmdline = lambda: ["python", "-m", "mlx2.server", "--api-key-env", "TEST_KEY"]
    assert monitor.local_server_api_key(endpoint) == "synthetic-env-secret"
    process.uids = lambda: SimpleNamespace(real=os.getuid() + 1)
    assert monitor.local_server_api_key(endpoint) is None
    process.uids = lambda: SimpleNamespace(real=os.getuid())
    monkeypatch.setattr(monitor, "discover_servers", lambda: ())
    assert monitor.local_server_api_key(endpoint) is None


@pytest.mark.parametrize("url,discovered", [
    ("http://remote.example:18285", True),
    ("http://127.0.0.1:18285", False),
    ("http://127.0.0.1:18285/other", True),
    ("http://user@127.0.0.1:18285", True),
])
def test_untrusted_endpoint_never_reads_local_credentials(monkeypatch, url, discovered):
    def unexpected():
        pytest.fail("must not even discover for an untrusted URL")
    monkeypatch.setattr(monitor, "discover_servers", unexpected)
    assert monitor.local_server_api_key(monitor.ServerEndpoint(url, 123, discovered=discovered)) is None


def test_collector_authenticates_all_telemetry_without_exporting_key(monkeypatch):
    requests = []
    redirect = False

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            requests.append((self.path, self.headers.get("Authorization")))
            if redirect:
                self.send_response(302)
                self.send_header("Location", "/must-not-follow")
                self.end_headers()
                return
            if self.headers.get("Authorization") != "Bearer synthetic-secret":
                self.send_response(401)
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()
            if self.path == "/metrics":
                self.wfile.write(b'mlx2_build_info{model="synthetic"} 1\nmlx2_requests_total{status="completed"} 2\n')
            else:
                self.wfile.write(b'{}')

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = monitor.ServerEndpoint(f"http://127.0.0.1:{server.server_port}", 123, discovered=True)
    resolutions = []
    monkeypatch.setattr(monitor, "discover_servers", lambda: (endpoint,))
    monkeypatch.setattr(monitor, "local_server_api_key", lambda e: resolutions.append(e) or "synthetic-secret")
    collector = monitor.InferenceCollector()
    try:
        reading, = collector.sample()
        assert reading.reachable
        assert "synthetic-secret" not in repr(dataclasses.asdict(reading))
        assert requests == [
            ("/metrics", None), ("/metrics", "Bearer synthetic-secret"),
            ("/v1/status/batching", "Bearer synthetic-secret"),
            ("/v1/status", "Bearer synthetic-secret"),
        ]
        assert resolutions == [endpoint]
        collector.api_key = "explicit-wrong-key"
        reading, = collector.sample()
        assert not reading.reachable and reading.error == "HTTP 401"
        assert resolutions == [endpoint]  # explicit configuration takes precedence
        redirect = True
        collector.api_key = "synthetic-secret"
        reading, = collector.sample()
        assert not reading.reachable and reading.error == "HTTP 302"
        assert all(path != "/must-not-follow" for path, _ in requests)
    finally:
        collector.close()
        server.shutdown()
        server.server_close()


def test_sticky_sessions_are_scoped_to_server_and_credentials(monkeypatch):
    import json

    endpoint = monitor.ServerEndpoint("http://127.0.0.1:18285")
    uptime = [10]
    revision = ["fixture-a"]
    batching = {"active_requests": [], "completed_requests": [{
        "request_id": "done", "tenant_id": "alice", "session_id": "chat",
        "status": "completed", "prompt_tokens": 100, "completion_tokens": 10,
        "context_tokens": 110, "prefill_tokens_per_second": 20.0,
        "decode_tokens_per_second": 5.0,
    }]}

    def get(_url, path, _key, _timeout):
        if path == "/metrics":
            return (f'mlx2_build_info{{revision="{revision[0]}"}} 1\n'
                    f'mlx2_process_uptime_seconds {uptime[0]}\n').encode()
        return json.dumps(batching if path.endswith("batching") else {}).encode()

    monkeypatch.setattr(monitor, "_get", get)
    collector = monitor.InferenceCollector(discover=False, urls=(endpoint.url,))
    try:
        first, = collector.sample()
        assert first.observed_sessions[0].context_tokens == 110
        batching.clear()
        idle, = collector.sample()
        assert idle.observed_sessions == first.observed_sessions
        # A changed credential must never inherit another tenant's idle rows.
        collector.api_key = "synthetic-other-tenant"
        other, = collector.sample()
        assert other.observed_sessions == ()
        batching.update({"completed_requests": [{"request_id": "bob-done", "tenant_id": "bob"}]})
        assert collector.sample()[0].observed_sessions
        batching.clear()
        uptime[0] = 1
        restarted, = collector.sample()
        assert restarted.observed_sessions == ()
        batching.update({"completed_requests": [{"request_id": "again", "tenant_id": "bob"}]})
        assert collector.sample()[0].observed_sessions
        batching.clear()
        revision[0] = "fixture-b"
        reloaded, = collector.sample()
        assert reloaded.observed_sessions == ()
    finally:
        collector.close()

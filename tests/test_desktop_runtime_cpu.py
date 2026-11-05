"""Synthetic owned-engine controls; no checkpoint, GPU, or live server used."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from mlx2.desktop_runtime import DesktopRuntime, RuntimeRefusal


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.setattr("mlx2.desktop_runtime._child_exited", lambda child: False)
    engine = SimpleNamespace(
        child=SimpleNamespace(pid=47, poll=lambda: None), lifecycle_lock=threading.RLock(),
        worker={"url": "http://127.0.0.1:18285", "state": "Ready", "execution_kind": "autoregressive",
                "checkpoint": "synthetic@fixture", "admin_token_file": "/synthetic/token"},
        chat_target=lambda url, pid: {"url": url},
    )
    service = DesktopRuntime(SimpleNamespace(engines={"default": engine}))
    yield service, engine, {"profile_id": "default", "url": engine.worker["url"], "pid": 47}
    service.close()


def wait(service, identity):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        value = service.dispatch({"command": "runtime_operation", "operation_id": identity})
        if not value["pending"]:
            return value
        time.sleep(.005)
    raise AssertionError("Synthetic operation did not complete")


@pytest.mark.parametrize("changes", [{"pid": 48}, {"pid": True}, {"url": "http://external.example"}, {"profile_id": "missing"}])
def test_runtime_refuses_stale_or_unowned_target(runtime, changes):
    service, _, request = runtime
    with pytest.raises(ValueError):
        service.status({**request, **changes})


def test_runtime_pending_operation_is_submit_once_and_receipts_are_copied(runtime, monkeypatch):
    service, _, request = runtime
    gate = threading.Event()
    calls = []
    def transport(*args):
        calls.append(args)
        assert gate.wait(2)
        return {"id": "attempt-1", "pending": True}
    monkeypatch.setattr(service, "_request", transport)
    try:
        first = service.begin({**request, "action": "weight_space", "body": {"tool": "trial_candidate", "arguments": {"attempt_id": "attempt-1"}}})
        with pytest.raises(ValueError, match="already pending"):
            service.begin({**request, "action": "weight_space", "body": {}})
        first["pending"] = False
        assert service.operations[first["id"]]["pending"]
    finally:
        gate.set()
    result = wait(service, first["id"])
    assert result["state"] == "submitted" and result["receipt"]["pending"]
    assert len(calls) == 1


@pytest.mark.parametrize("failure, state", [
    (TimeoutError("synthetic timeout"), "unknown"),
    (RuntimeRefusal(500, {"error": {"message": "apply failed"}, "mlx2": {"state": "failed"}}), "failed"),
])
def test_runtime_never_calls_transport_failures_success(runtime, monkeypatch, failure, state):
    service, _, request = runtime
    def transport(*args):
        raise failure
    monkeypatch.setattr(service, "_request", transport)
    first = service.begin({**request, "action": "reload_kernels", "body": {"modules": ["synthetic.module"]}})
    result = wait(service, first["id"])
    assert result["state"] == state
    if state == "unknown":
        assert "before retrying" in result["guidance"]
    else:
        assert result["receipt"]["mlx2"]["state"] == "failed"


def test_runtime_route_allowlist_and_json_bound(runtime):
    service, _, request = runtime
    for action, body in [("/v1/admin/arbitrary", {}), ("load_lora", []),
                         ("reload_kernels", {"modules": ["x" * 65536]}),
                         ("weight_space", {"strength": float("nan")})]:
        with pytest.raises(ValueError):
            service.begin({**request, "action": action, "body": body})
    assert not service.operations


def test_runtime_disabled_weight_space_is_distinct_from_refusal(runtime, monkeypatch):
    service, _, request = runtime
    status = {"model": "synthetic", "multi_lora": {"enabled": False}}
    def transport(owner, method, path):
        if path.endswith("capabilities"):
            raise RuntimeRefusal(404, {"error": {"message": "disabled"}})
        return status if path == "/v1/status" else {"enabled": False}
    monkeypatch.setattr(service, "_request", transport)
    result = service.status(request)
    assert result["status"] == status and result["weight_space"]["enabled"] is False
    def denied(*args):
        raise RuntimeRefusal(403, {"error": {"message": "not authorized"}})
    monkeypatch.setattr(service, "_request", denied)
    with pytest.raises(RuntimeRefusal, match="not authorized"):
        service.status(request)


def test_runtime_transport_revalidates_owner_and_refuses_redirects(runtime, monkeypatch):
    service, engine, request = runtime
    observations = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass
        def do_GET(self):
            observations.append((self.path, self.headers.get("Authorization")))
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "/must-not-follow")
                self.end_headers()
                return
            if self.path == "/replaced":
                engine.child = SimpleNamespace(pid=48, poll=lambda: None)
            data = json.dumps({"ok": True}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        engine.worker["url"] = f"http://127.0.0.1:{server.server_port}"
        request["url"] = engine.worker["url"]
        monkeypatch.setattr("mlx2.http_security.load_secret_file", lambda *a, **k: "synthetic-admin")
        monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
        monkeypatch.setenv("no_proxy", "")
        owner = service._owner(request)
        assert service._request(owner, "GET", "/ok") == {"ok": True}
        with pytest.raises(RuntimeRefusal):
            service._request(owner, "GET", "/redirect")
        with pytest.raises(ValueError, match="changed during"):
            service._request(owner, "GET", "/replaced")
        assert observations == [(path, "Bearer synthetic-admin") for path in ("/ok", "/redirect", "/replaced")]
    finally:
        server.shutdown()
        server.server_close()

"""CPU checks that the external-route serving smoke measures its own server."""

import importlib.util
import json
from pathlib import Path
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "smoke_external_route_serving.py"


def _module():
    spec = importlib.util.spec_from_file_location("smoke_external_route_serving", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_server(model, port=0):
    """Answers health, models and chat; records every chat request."""
    chats = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, payload):
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/v1/models":
                self._send({"object": "list", "data": [{"id": model}]})
            else:
                self._send({"status": "ok", "healthy": True, "model": model})

        def do_POST(self):
            chats.append(self.path)
            self.rfile.read(int(self.headers.get("content-length") or 0))
            self._send({"choices": [{"message": {"content": "x"}}], "mlx2": {}})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.chats = chats
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class _Launched:
    pid = -1

    def __init__(self, server=None):
        self.server = server
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        return 0


def _free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _run(module, monkeypatch, tmp_path, port, launch):
    launched = []
    real = module.subprocess.Popen

    def popen(command, *args, **kwargs):
        if "mlx2.server" in command:
            launched.append(command)
            return launch()
        return real(command, *args, **kwargs)

    monkeypatch.setattr(module.subprocess, "Popen", popen)
    monkeypatch.setattr(module.os, "killpg", lambda *args: None)
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.setattr(module.tempfile, "mkdtemp", lambda **kwargs: str(workdir))
    out = tmp_path / "smoke.json"
    rc = module.main([
        "--family", "north", "--model", "/models/INTENDED-MODEL", "--draft", "/models/d",
        "--port", str(port), "--startup-timeout", "10", "--max-tokens", "4",
        "--i-own-the-gpu", "--out", str(out),
    ])
    return rc, launched, (json.loads(out.read_text()) if out.exists() else None)


def test_smoke_refuses_a_port_another_server_already_holds(tmp_path, monkeypatch):
    # Regression: the first /health poll could be answered by another
    # session's server while the launched one was still loading (or dying
    # on EADDRINUSE), and the smoke then measured that server.
    module = _module()
    foreign = _fake_server("SOME-OTHER-SESSIONS-MODEL")
    try:
        rc, launched, _ = _run(module, monkeypatch, tmp_path, foreign.server_port, _Launched)
    finally:
        foreign.shutdown()
        foreign.server_close()
    assert rc == 2 and launched == []
    assert foreign.chats == []


@pytest.mark.parametrize("served", ["SOME-OTHER-MODEL", "INTENDED-MODEL"])
def test_smoke_measures_only_the_model_it_launched(tmp_path, monkeypatch, served):
    module = _module()
    port = _free_port()
    fakes = []

    def launch():
        fakes.append(_fake_server(served, port))
        return _Launched(fakes[-1])

    rc, launched, result = _run(module, monkeypatch, tmp_path, port, launch)
    assert len(launched) == 1 and rc == 1
    chats = fakes[0].chats
    if served == "INTENDED-MODEL":
        assert chats, "the launched model must be measured"
    else:
        assert chats == []
        assert any("SOME-OTHER-MODEL" in failure for failure in result["failures"])

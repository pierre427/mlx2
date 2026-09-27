"""Loopback fake-child wire tests; no MLX or model load."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from scripts.research.mtplx520_http_preflight import forward_stream
from scripts.research.mtplx520_supervisor_preflight import ModelSpec, Refused, SupervisorPreflight


def _spec(name):
    return ModelSpec(
        model_id=name, artifact=f"artifact-{name}", runtime={"source_sha256": f"sha-{name}"},
        settings={"route": "self_mtp", "mtp": True}, route_receipt=f"receipt-{name}",
        resident_bytes=60, apc_dir=f"/private/{name}",
        capabilities=frozenset({"text", "mtp", "apc_v2"}),
    )


def _event(model, *, terminal=False, mismatch=False):
    data = {"model": model, "delta": "x"}
    if terminal:
        data["choices"] = [{"index": 0, "finish_reason": "stop", "delta": {}}]
        data["mlx2"] = {
            "route_receipt": "wrong" if mismatch else f"receipt-{model}",
            "route": "self_mtp", "qualification": "qualified",
        }
    return b"data: " + json.dumps(data).encode() + b"\n\n"


@pytest.fixture
def rig():
    seen = []
    mode = ["normal"]

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def do_POST(self):
            size = int(self.headers["Content-Length"])
            seen.append((self.path, dict(self.headers), self.rfile.read(size)))
            model = json.loads(seen[-1][2])["model"]
            first = _event(model)
            second = _event(model, terminal=True, mismatch=mode[0] == "mismatch")
            if mode[0] == "death":
                payload = first
            else:
                payload = first + second + b"data: [DONE]\n\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()
            if mode[0] == "death":
                self.close_connection = True

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()

    class Child:
        def __init__(self, spec):
            self.spec = spec
            self.endpoint = server.server_address

        def status(self):
            spec = self.spec
            return {
                "healthy": True, "model": spec.model_id, "artifact": spec.artifact,
                "runtime": dict(spec.runtime), "settings": dict(spec.settings),
                "route_receipt": spec.route_receipt, "qualification": "qualified",
                "qualified_capabilities": sorted(spec.capabilities),
                "selected_capabilities": sorted(spec.capabilities),
            }

        def close(self):
            pass

    supervisor = SupervisorPreflight(
        {name: _spec(name) for name in ("a", "b")}, 100,
        Child, lambda: 0.0, "secret", evict_to_fit=True,
    )
    try:
        yield supervisor, seen, mode
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def _forward(supervisor, name="a", **kwargs):
    return forward_stream(
        supervisor, name, credential=kwargs.pop("credential", "secret"),
        body=kwargs.pop("body", json.dumps({"model": name}).encode()),
        headers=kwargs.pop("headers", {}), **kwargs,
    )


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/completions"])
def test_exact_child_stream_and_receipt(path, rig):
    supervisor, seen, _ = rig
    stream = _forward(supervisor, path=path,
                      headers={"Authorization": "Bearer attacker", "X-Tenant-ID": "victim"})
    assert len(list(stream.events())) == 3
    assert stream.terminal_seen and stream.done_seen and stream.closed
    assert supervisor._children["a"].pins == 0
    assert seen[0][0] == path
    assert seen[0][1]["Authorization"] == "Bearer secret"
    assert seen[0][1]["X-Tenant-ID"] == "default"


def test_auth_size_and_endpoint_fail_before_child_or_body_read(rig):
    supervisor, seen, _ = rig
    for kwargs, code in (
        ({"credential": "wrong"}, "unauthorized"),
        ({"body": b"x" * 9, "max_body_bytes": 8}, "request_too_large"),
        ({"path": "/v1/responses"}, "unsupported_endpoint"),
    ):
        with pytest.raises(Refused, match=code):
            _forward(supervisor, **kwargs)
    assert seen == [] and supervisor._children == {}


def test_cancel_releases_exact_lease_and_drain_blocks_new_request(rig):
    supervisor, seen, _ = rig
    stream = _forward(supervisor)
    iterator = stream.events()
    assert _event("a") == next(iterator)
    assert not supervisor.drain("a")
    with pytest.raises(Refused, match="child_draining_or_failed"):
        _forward(supervisor)
    stream.close()
    iterator.close()
    assert supervisor._children["a"].pins == 0
    assert len(seen) == 1
    supervisor.unload("a")


@pytest.mark.parametrize("mode,code", [
    ("death", "child_stream_missing_terminal_receipt"),
    ("mismatch", "child_receipt_mismatch"),
])
def test_child_failure_after_headers_never_becomes_success(mode, code, rig):
    supervisor, seen, setting = rig
    setting[0] = mode
    stream = _forward(supervisor)
    with pytest.raises(Refused, match=code):
        list(stream.events())
    assert stream.closed and supervisor._children["a"].pins == 0
    assert len(seen) == 1


def test_pinned_child_blocks_second_model_until_cancel(rig):
    supervisor, _, _ = rig
    stream = _forward(supervisor)
    with pytest.raises(Refused, match="all_eviction_candidates_pinned"):
        _forward(supervisor, "b")
    stream.close()
    second = _forward(supervisor, "b")
    assert len(list(second.events())) == 3

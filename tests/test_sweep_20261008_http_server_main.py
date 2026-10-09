"""Sweep 2026-10-08 (server-http): the HTTP server ``main()`` builds (CPU).

``main()`` built ``BoundedHTTPServer`` with a fixed 32-connection cap whatever
``--max-inflight`` said, and a full pool closed new sockets with no HTTP
response: an idle keep-alive connection holds its slot for the 30 s idle
timeout, so a client arriving after a burst was reset (ECONNRESET) instead of
being told to retry.
"""

import http.client
import io
import json
import select
import socket
import sys
import threading
import time
import types

import pytest
from test_serving_contract import FakeEngine

from mlx2.server import BoundedHTTPServer, handler_for


class _EngineReached(BaseException):
    pass


def run_main_until_engine(monkeypatch, argv):
    """Run the real ``main()`` through its bind; stop where it loads weights."""
    from mlx2 import exit_trace, server
    from mlx2.adapters import registry

    built = {}

    class RecordingServer(server.BoundedHTTPServer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            built["server"] = self

    monkeypatch.setattr(registry, "inspect_model", lambda path: types.SimpleNamespace())
    monkeypatch.setattr(
        server,
        "resolve_route_selection",
        lambda args, policy, resolution: types.SimpleNamespace(
            native_mtp=False, route="ordinary", source="test"
        ),
    )
    monkeypatch.setattr(
        server, "resolve_execution_policy_defaults", lambda policy, *a, **k: policy
    )
    monkeypatch.setattr(exit_trace.ExitTrace, "install", lambda self, fault_log=None: self)
    monkeypatch.setattr(server, "BoundedHTTPServer", RecordingServer)

    def engine(*args, **kwargs):
        raise _EngineReached

    monkeypatch.setattr(server, "ServingEngine", engine)
    monkeypatch.setattr(sys, "argv", ["mlx2.server", "--model", "unused", *argv])
    with pytest.raises(_EngineReached):
        server.main()
    # main() closes the server when engine construction fails.
    return built


@pytest.mark.parametrize("inflight", [8, 32, 96])
def test_main_never_caps_connections_below_max_inflight(monkeypatch, inflight):
    built = run_main_until_engine(
        monkeypatch,
        ["--host", "127.0.0.1", "--port", "0", "--max-inflight", str(inflight)],
    )
    slots = built["server"].connections
    capacity = 0
    while capacity < 1000 and slots.acquire(blocking=False):
        capacity += 1
    # Each in-flight request holds one connection; the cap cannot be lower,
    # and it is never below the previous fixed 32.
    assert capacity >= max(inflight, 32), capacity


@pytest.fixture
def bounded_server():
    server = BoundedHTTPServer(
        ("127.0.0.1", 0), handler_for(FakeEngine()), max_connections=2
    )
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    clients = []
    yield server, clients
    for client in clients:
        client.close()
    server.shutdown()
    server.server_close()
    thread.join(5)


def _idle_keepalive(port):
    client = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    client.request("GET", "/health")
    response = client.getresponse()
    response.read()
    assert response.status == 200
    assert (response.getheader("Connection") or "").lower() != "close"
    return client


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_full_connection_pool_answers_503_instead_of_resetting(bounded_server, method):
    server, clients = bounded_server
    port = server.server_port
    # Every slot is held by a keep-alive connection with nothing in flight.
    clients.extend(_idle_keepalive(port) for _ in range(2))
    late = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    clients.append(late)
    try:
        if method == "GET":
            late.request("GET", "/health")
        else:
            # A body the refusal never reads must not turn the reply into a RST.
            late.request(
                "POST", "/v1/chat/completions",
                body=json.dumps({"messages": [{"role": "user", "content": "x" * 50000}]}),
                headers={"Content-Type": "application/json"},
            )
        response = late.getresponse()
        payload = json.loads(response.read())
    except (ConnectionError, http.client.RemoteDisconnected) as error:
        pytest.fail(f"connection dropped without an HTTP response: {error!r}")
    assert response.status == 503
    assert response.getheader("Retry-After") == "1"
    assert (response.getheader("Connection") or "").lower() == "close"
    assert payload["error"]["type"] == "server_error"
    # The held connections are untouched and the refusal returned its slot.
    clients[0].request("GET", "/health")
    reply = clients[0].getresponse()
    reply.read()
    assert reply.status == 200


def _wait_until(predicate, seconds=5.0):
    deadline = time.monotonic() + seconds
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.01)


def _late_request(port):
    """Send the request only once the reply is readable, head and body apart.

    A server that closed the socket after answering resets these writes.
    """
    sock = socket.create_connection(("127.0.0.1", port), timeout=5)
    try:
        readable, _, _ = select.select([sock], [], [], 5)
        assert readable, "no reply before the request"
        sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                     b"Content-Type: application/json\r\nContent-Length: 2\r\n\r\n")
        time.sleep(0.1)
        sock.sendall(b"{}")
        raw = b""
        while chunk := sock.recv(4096):
            raw += chunk
    finally:
        sock.close()
    response = http.client.HTTPResponse(_Replay(raw))
    response.begin()
    return response, response.read()


class _Replay:
    def __init__(self, raw):
        self.raw = raw

    def makefile(self, *_args, **_kwargs):
        return io.BytesIO(self.raw)


@pytest.mark.parametrize("client", ["GET", "POST", "late-request"])
def test_saturated_refusal_workers_still_answer_503(client):
    # Review round 1: with every refusal worker busy (each waits up to
    # refusal_seconds for a request head) the next connection was closed
    # unanswered.  It must get the same 503 at once, from the accept thread,
    # including when its request bytes arrive after the reply.
    class OneRefusal(BoundedHTTPServer):
        max_refusals = 1
        refusal_seconds = 10.0

    server = OneRefusal(("127.0.0.1", 0), handler_for(FakeEngine()), max_connections=1)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    port = server.server_port
    held = _idle_keepalive(port)
    stalled = socket.create_connection(("127.0.0.1", port), timeout=15)
    late = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        # The only refusal worker waits for the rest of this request head.
        stalled.sendall(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n")
        _wait_until(lambda: server.refusals._value == 0)
        started = time.monotonic()
        try:
            if client == "late-request":
                response, body = _late_request(port)
            else:
                if client == "GET":
                    late.request("GET", "/health")
                else:
                    late.request(
                        "POST", "/v1/chat/completions",
                        body=json.dumps({"messages": [{"role": "user", "content": "hi"}]}),
                        headers={"Content-Type": "application/json"},
                    )
                response = late.getresponse()
                body = response.read()
        except (ConnectionError, http.client.HTTPException) as error:
            pytest.fail(f"connection dropped without an HTTP response: {error!r}")
        # Answered without waiting for the busy worker's 10 s budget.
        assert time.monotonic() - started < 5
        assert response.status == 503
        assert response.getheader("Retry-After") == "1"
        assert (response.getheader("Connection") or "").lower() == "close"
        assert json.loads(body)["error"]["type"] == "server_error"
        # The accept thread was not blocked: the stalled refusal completes,
        # and the held connection is still served.
        stalled.sendall(b"\r\n")
        assert stalled.recv(64).startswith(b"HTTP/1.1 503 ")
        held.request("GET", "/health")
        reply = held.getresponse()
        reply.read()
        assert reply.status == 200
        # Answered sockets are closed once their client has closed.
        late.close()
        _wait_until(lambda: not server._lingering)
    finally:
        for connection in (held, stalled, late):
            connection.close()
        server.shutdown()
        server.server_close()
        thread.join(5)


# --host ::1 / [::1] / :: are documented binds (SERVING.md "HTTP security",
# http_security.LOOPBACK_HOSTS / WILDCARD_BINDS); an AF_INET-only server died
# on them with an unhandled socket.gaierror.


def _host_has_ipv6_loopback():
    if not socket.has_ipv6:
        return False
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as probe:
            probe.bind(("::1", 0))
    except OSError:
        return False
    return True


needs_ipv6 = pytest.mark.skipif(
    not _host_has_ipv6_loopback(), reason="host has no IPv6 loopback"
)


@needs_ipv6
@pytest.mark.parametrize("host", ["::1", "[::1]"])
def test_main_binds_documented_ipv6_loopback_hosts(monkeypatch, host):
    built = run_main_until_engine(monkeypatch, ["--host", host, "--port", "0"])
    server = built["server"]
    assert server.address_family == socket.AF_INET6
    assert server.server_address[0] == "::1"


@needs_ipv6
def test_ipv6_loopback_server_answers_and_counts_as_loopback():
    from mlx2.server import authorize_admin, is_loopback_address

    server = BoundedHTTPServer(("[::1]", 0), handler_for(FakeEngine()))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = http.client.HTTPConnection("::1", server.server_port, timeout=5)
        client.request("GET", "/health", headers={"Host": "[::1]"})
        assert client.getresponse().status == 200
        client.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert authorize_admin("::1", None) is None
    # The bracketed bind form is loopback too (no non-loopback warning).
    assert is_loopback_address("[::1]")


def test_wildcard_ipv6_bind_uses_an_ipv6_socket():
    # Not bound here: a wildcard listener in a test would open a public port.
    server = BoundedHTTPServer(
        ("::", 0), handler_for(FakeEngine()), bind_and_activate=False
    )
    try:
        assert server.address_family == server.socket.family == socket.AF_INET6
    finally:
        server.server_close()


def test_ipv4_binds_unchanged():
    for host in ("127.0.0.1", "localhost"):
        server = BoundedHTTPServer((host, 0), handler_for(FakeEngine()))
        try:
            assert server.socket.family == socket.AF_INET
            assert server.server_address[0] == "127.0.0.1"
        finally:
            server.server_close()

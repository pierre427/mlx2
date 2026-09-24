"""Host-only regressions for request ownership on persistent HTTP connections."""

import json
import socket
import threading
import time
from contextlib import contextmanager
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from test_server_quiesce import AdminEngine

from mlx2.batch_metrics import HttpRuntimeMetrics
from mlx2.http_security import HTTPSecurityPolicy
from mlx2.server import handler_for


@contextmanager
def _server(engine=None, *, deadline=None, **kwargs):
    engine = engine or AdminEngine()
    handler = handler_for(engine, **kwargs)
    if deadline is not None:
        handler.REQUEST_BODY_DEADLINE_SECONDS = deadline
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield engine, server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_keepalive_has_fresh_metrics_traces_and_auth_receipts():
    engine = AdminEngine()
    engine.http_metrics = HttpRuntimeMetrics()
    traces = []

    class Tracer:
        def start(self, name, headers, attributes):
            statuses = []
            traces.append((name, statuses))
            return SimpleNamespace(finish=statuses.append)

    class Authenticator:
        def authenticate(self, headers, **kwargs):
            return SimpleNamespace(tenant="tenant-a", method="api_key")

    with _server(
        engine,
        http_security=HTTPSecurityPolicy(allowed_hosts=frozenset({"localhost"})),
        tenant_authenticator=Authenticator(),
        request_tracer=Tracer(),
    ) as (_, port):
        connection = HTTPConnection("127.0.0.1", port, timeout=2)
        try:
            connection.request("GET", "/v1/models", headers={"Host": "localhost"})
            first = connection.getresponse()
            assert first.status == 200
            assert first.getheader("X-MLX2-Tenant-Auth") == "api_key"
            first.read()
            connection.request("GET", "/health", headers={"Host": "untrusted"})
            second = connection.getresponse()
            assert second.status == 421
            assert second.getheader("X-MLX2-Tenant-Auth") is None
            second.read()
        finally:
            connection.close()
    assert traces == [("GET models", [200]), ("GET health", [421])]
    assert engine.http_metrics.prometheus_snapshot()["requests"] == {
        ("GET", "models", "2xx"): 1,
        ("GET", "health", "4xx"): 1,
    }


def test_body_deadline_expires_while_bytes_keep_arriving():
    with (
        _server(deadline=0.3) as (_, port),
        socket.create_connection(("127.0.0.1", port), timeout=3) as client,
    ):
        client.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
            b"Content-Length: 4000\r\n\r\n"
        )
        stopped = threading.Event()

        def trickle():
            for _ in range(40):
                try:
                    client.sendall(b" ")
                except OSError:
                    return
                if stopped.wait(0.04):
                    return

        sender = threading.Thread(target=trickle)
        started = time.monotonic()
        sender.start()
        try:
            response = client.recv(4096)
            elapsed = time.monotonic() - started
            assert b" 408 " in response.split(b"\r\n", 1)[0]
            assert elapsed < 1.0
        finally:
            stopped.set()
            sender.join()


def test_expect_continue_keeps_connection_open_until_body_arrives():
    with (
        _server() as (_, port),
        socket.create_connection(("127.0.0.1", port), timeout=2) as client,
    ):
        client.sendall(
            b"POST /v1/admin/resume HTTP/1.1\r\nHost: localhost\r\n"
            b"Content-Length: 2\r\nExpect: 100-continue\r\n\r\n"
        )
        stream = client.makefile("rb")
        assert b" 100 " in stream.readline()
        assert stream.readline() == b"\r\n"
        client.sendall(b"{}")
        assert b" 202 " in stream.readline()
        headers = {}
        while (line := stream.readline()) != b"\r\n":
            assert line
            name, value = line.decode().split(":", 1)
            headers[name.lower()] = value.strip()
        assert headers.get("connection") != "close"
        stream.read(int(headers["content-length"]))
        client.sendall(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        assert b" 200 " in stream.readline()


@pytest.mark.parametrize(
    "framing",
    [
        b"Content-Length: 2\r\nContent-Length: 2\r\n",
        b"Content-Length: 2\r\nContent-Length: 40\r\n",
        b"Transfer-Encoding: chunked\r\nContent-Length: 2\r\n",
        b"Content-Length: +2\r\n",
    ],
)
def test_ambiguous_or_invalid_framing_is_rejected_and_closed(framing):
    with (
        _server() as (_, port),
        socket.create_connection(("127.0.0.1", port), timeout=2) as client,
    ):
        client.sendall(
            b"POST /v1/admin/resume HTTP/1.1\r\nHost: localhost\r\n"
            + framing + b"\r\n{}"
        )
        stream = client.makefile("rb")
        status = stream.readline()
        assert b" 400 " in status
        headers = []
        while (line := stream.readline()) != b"\r\n":
            assert line
            headers.append(line.lower())
        assert b"connection: close\r\n" in headers


def test_early_rejection_does_not_interpret_unread_body_as_next_request():
    with (
        _server() as (_, port),
        socket.create_connection(("127.0.0.1", port), timeout=2) as client,
    ):
        nested = b"GET /v1/admin/state HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
        client.sendall(
            b"POST /unknown HTTP/1.1\r\nHost: localhost\r\n"
            + f"Content-Length: {len(nested)}\r\n\r\n".encode() + nested
        )
        chunks = []
        while chunk := client.recv(4096):
            chunks.append(chunk)
        response = b"".join(chunks)
        assert response.count(b"HTTP/1.1 ") == 1
        assert response.startswith(b"HTTP/1.1 404 ")


@pytest.mark.parametrize("body", [b"[]", b"null", b"1", b'{"messages":' + b"[" * 1500 + b"]" * 1500 + b"}"])
def test_count_tokens_rejects_nonobject_or_excessively_nested_json(body):
    with _server() as (_, port):
        connection = HTTPConnection("127.0.0.1", port, timeout=2)
        try:
            connection.request("POST", "/v1/messages/count_tokens", body=body)
            response = connection.getresponse()
            assert response.status == 400
            assert json.loads(response.read())["type"] == "error"
        finally:
            connection.close()

from contextlib import contextmanager
from email.message import Message
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import json
import logging
import os
import threading

import pytest

from mlx2.http_security import (
    DEFAULT_API_KEY_ENV,
    GateRejection,
    HTTPSecurityPolicy,
    authorize,
    check_api_key,
    check_authority,
    load_api_key,
    normalize_host,
    parse_authority,
    policy_for_bind,
)
from mlx2.server import build_parser, handler_for, load_admin_token
from test_server_quiesce import AdminEngine


KEY = "sk-local-test-key"


def _headers(**values):
    message = Message()
    for name, value in values.items():
        for item in value if isinstance(value, list) else [value]:
            message[name.replace("_", "-")] = item
    return message


@contextmanager
def _server(policy, *, admin_token=None):
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        handler_for(AdminEngine(), http_security=policy, admin_token=admin_token),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _request(port, method, path, *, headers=None, body=None):
    connection = HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        raw = response.read()
        return response.status, dict(response.getheaders()), (
            json.loads(raw) if raw else None
        )
    finally:
        connection.close()


def test_parse_authority_and_normalize_host():
    assert parse_authority("localhost:8285") == ("localhost", 8285)
    assert parse_authority("[::1]:8285") == ("::1", 8285)
    assert parse_authority("Example.COM.") == ("example.com", None)
    for bad in ("", "a b", "user@host", "host/path", "host:notaport", "[::1"):
        with pytest.raises(ValueError):
            parse_authority(bad)
    assert normalize_host("[::1]") == "::1"
    assert normalize_host("Box.Local.") == "box.local"
    with pytest.raises(ValueError, match="port"):
        normalize_host("box.local:8285")


def test_policy_for_bind_defaults(caplog):
    loopback = policy_for_bind("127.0.0.1")
    assert loopback.allowed_hosts == {"localhost", "127.0.0.1", "::1"}
    assert loopback.api_key is None and loopback.check_origin
    assert "box.lan" in policy_for_bind("localhost", allowed_hosts=["Box.LAN"]).allowed_hosts
    assert "::1" in policy_for_bind("::1").allowed_hosts
    with caplog.at_level(logging.WARNING):
        assert policy_for_bind("0.0.0.0").allowed_hosts is None
        assert policy_for_bind("<private-git-host>").allowed_hosts is None
    assert "Host header check is disabled" in caplog.text
    wildcard = policy_for_bind("0.0.0.0", allowed_hosts=["box.lan"])
    assert wildcard.allowed_hosts == {"localhost", "127.0.0.1", "::1", "box.lan"}
    assert "<private-git-host>" in policy_for_bind(
        "<private-git-host>", allowed_hosts=["box.lan"]
    ).allowed_hosts


def test_authority_checks():
    allowed = frozenset({"localhost", "127.0.0.1", "::1"})
    for host in ("127.0.0.1:8285", "localhost:8285", "[::1]:8285", "localhost"):
        check_authority(_headers(Host=host), allowed)
    check_authority(
        _headers(Host="localhost:8285", Origin="http://localhost:8285"), allowed
    )
    check_authority(_headers(Host="localhost", Origin="http://localhost:80"), allowed)
    # The socket's own address is trusted even when not listed.
    check_authority(_headers(Host="<private-git-host>:8285"), allowed, local_address="<private-git-host>")
    cases = [
        (_headers(Host="evil.example:8285"), 421),
        (_headers(), 421),
        (_headers(Host=["localhost", "evil.example"]), 400),
        (_headers(Host="local host"), 400),
        (_headers(Host="localhost:8285", Origin="http://evil.example"), 403),
        (_headers(Host="localhost:8285", Origin="http://localhost:9999"), 403),
        (_headers(Host="localhost:8285", Origin="null"), 403),
        (_headers(Host="localhost:8285", Origin="http://localhost:8285/x"), 403),
    ]
    for headers, status in cases:
        with pytest.raises(GateRejection) as caught:
            check_authority(headers, allowed)
        assert caught.value.status == status
    # Host check disabled (non-loopback bind): Origin must still match Host.
    check_authority(_headers(Host="anything:1"), None)
    with pytest.raises(GateRejection):
        check_authority(_headers(Host="box:1", Origin="http://evil.example"), None)


def test_api_key_header_forms():
    check_api_key(_headers(), None)
    check_api_key(_headers(Authorization=f"Bearer {KEY}"), KEY)
    check_api_key(_headers(Authorization=f"bearer {KEY}"), KEY)
    check_api_key(_headers(x_api_key=KEY), KEY)
    # Claude Code with both variables set sends both; either may match.
    check_api_key(_headers(Authorization="Bearer other", x_api_key=KEY), KEY)
    for headers in (
        _headers(),
        _headers(Authorization=KEY),
        _headers(Authorization=f"Basic {KEY}"),
        _headers(Authorization="Bearer wrong"),
        _headers(x_api_key="wrong"),
        _headers(x_api_key=[KEY, KEY]),
        _headers(Authorization=[f"Bearer {KEY}", f"Bearer {KEY}"]),
        _headers(x_api_key="sk-lócal"),
    ):
        with pytest.raises(GateRejection) as caught:
            check_api_key(headers, KEY)
        assert caught.value.status == 401
        assert caught.value.headers == {"WWW-Authenticate": "Bearer"}


def test_authorize_exemptions():
    policy = HTTPSecurityPolicy(allowed_hosts=frozenset({"localhost"}), api_key=KEY)
    host = _headers(Host="localhost:1")
    authorize(policy, "GET", "/health", host)
    with pytest.raises(GateRejection):
        authorize(policy, "GET", "/metrics", host)
    with pytest.raises(GateRejection):
        authorize(policy, "POST", "/health", host)
    # Admin token configured: that token governs the admin surface.
    authorize(policy, "GET", "/v1/admin/state", host, admin_token_configured=True)
    with pytest.raises(GateRejection):
        authorize(policy, "GET", "/v1/admin/state", host)
    # Health is exempt from the key, never from Host.
    with pytest.raises(GateRejection):
        authorize(policy, "GET", "/health", _headers(Host="evil.example"))
    authorize(None, "POST", "/v1/chat/completions", _headers())


def test_load_api_key_sources(tmp_path):
    assert load_api_key() is None
    path = tmp_path / "api.key"
    path.write_text(KEY + "\n")
    os.chmod(path, 0o600)
    assert load_api_key(key_file=path) == KEY
    os.chmod(path, 0o644)
    with pytest.raises(ValueError, match="--api-key-file permissions"):
        load_api_key(key_file=path)
    os.chmod(path, 0o600)
    path.write_text("sk with space\n")
    with pytest.raises(ValueError, match="visible ASCII"):
        load_api_key(key_file=path)
    env = {DEFAULT_API_KEY_ENV: KEY}
    assert load_api_key(key_env=DEFAULT_API_KEY_ENV, environ=env) == KEY
    with pytest.raises(ValueError, match="unset or empty"):
        load_api_key(key_env="MISSING", environ=env)
    assert load_api_key(key_env="MISSING", environ=env, required=False) is None
    with pytest.raises(ValueError, match="mutually exclusive"):
        load_api_key(key_file=path, key_env="X")
    # Admin token loader keeps its flag-specific messages.
    path.write_text("secret\n")
    os.chmod(path, 0o640)
    with pytest.raises(ValueError, match="--admin-token-file permissions"):
        load_admin_token(path)


def test_cli_flags_parse():
    args = build_parser().parse_args(
        ["--model", "m", "--allowed-host", "a", "--allowed-host", "b",
         "--api-key-env", "MLX2_API_KEY"]
    )
    assert args.allowed_host == ["a", "b"]
    assert args.api_key_env == "MLX2_API_KEY" and args.api_key_file is None
    defaults = build_parser().parse_args(["--model", "m"])
    assert defaults.allowed_host == [] and defaults.api_key_env is None
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["--model", "m", "--api-key-env", "X", "--api-key-file", "f"]
        )


def test_default_loopback_gate_over_http():
    with _server(policy_for_bind("127.0.0.1")) as port:
        # urllib/httpx/OpenAI SDK style: Host = dialled address, no Origin.
        assert _request(port, "GET", "/v1/models")[0] == 200
        assert _request(port, "GET", "/v1/models", headers={"Host": f"localhost:{port}"})[0] == 200
        assert _request(port, "GET", "/v1/models", headers={"Host": f"[::1]:{port}"})[0] == 200
        same_origin = {"Host": f"localhost:{port}", "Origin": f"http://localhost:{port}"}
        assert _request(port, "GET", "/v1/models", headers=same_origin)[0] == 200
        # DNS rebinding: attacker name resolving to loopback.
        status, _, body = _request(
            port, "GET", "/v1/models", headers={"Host": f"evil.example:{port}"}
        )
        assert status == 421 and body["error"]["code"] == "untrusted_host"
        status, _, body = _request(
            port,
            "POST",
            "/v1/chat/completions",
            headers={"Origin": "http://evil.example", "Content-Type": "application/json"},
            body=b"{}",
        )
        assert status == 403 and body["error"]["type"] == "permission_error"
        status, _, body = _request(
            port, "POST", "/v1/messages",
            headers={"Origin": "https://evil.example"}, body=b"{}",
        )
        assert status == 403 and body["type"] == "error"
        assert body["error"]["type"] == "permission_error"


def test_api_key_gate_over_http():
    with _server(policy_for_bind("127.0.0.1", api_key=KEY)) as port:
        assert _request(port, "GET", "/health")[0] == 200
        assert _request(port, "GET", "/health?probe=1")[0] == 401
        status, headers, body = _request(port, "GET", "/v1/models")
        assert status == 401 and headers["WWW-Authenticate"] == "Bearer"
        assert body["error"]["code"] == "invalid_api_key"
        assert _request(port, "GET", "/v1/models", headers={"Authorization": f"Bearer {KEY}"})[0] == 200
        assert _request(port, "GET", "/v1/models", headers={"x-api-key": KEY})[0] == 200
        assert _request(port, "GET", "/metrics")[0] == 401
        # The gate runs before the body is read; the connection is closed.
        status, headers, body = _request(
            port, "POST", "/v1/messages",
            headers={"x-api-key": "wrong", "Content-Type": "application/json"},
            body=b'{"model": "m"}',
        )
        assert status == 401 and body["error"]["type"] == "authentication_error"
        assert headers.get("Connection") == "close"
        assert _request(port, "DELETE", "/v1/files/x")[0] == 401
        assert _request(port, "DELETE", "/v1/files/x", headers={"x-api-key": KEY})[0] == 404
        # Admin without an admin token falls back to the API key.
        assert _request(port, "GET", "/v1/admin/state")[0] == 401
        assert _request(port, "GET", "/v1/admin/state", headers={"x-api-key": KEY})[0] == 200


def test_admin_token_keeps_its_own_bearer_with_api_key():
    with _server(policy_for_bind("127.0.0.1", api_key=KEY), admin_token="admin") as port:
        assert _request(port, "GET", "/v1/admin/state", headers={"x-api-key": KEY})[0] == 401
        assert _request(
            port, "GET", "/v1/admin/state", headers={"Authorization": "Bearer admin"}
        )[0] == 200
        assert _request(
            port, "GET", "/v1/admin/state",
            headers={"Authorization": "Bearer admin", "Host": "evil.example"},
        )[0] == 421


def test_embedding_without_policy_is_unchanged():
    with _server(None) as port:
        assert _request(port, "GET", "/v1/models", headers={"Host": "x"})[0] == 200
        assert _request(
            port, "GET", "/v1/models", headers={"Origin": "http://evil.example"}
        )[0] == 200


def test_rejected_body_is_not_parsed_as_a_pipelined_request():
    import socket

    with _server(policy_for_bind("127.0.0.1", api_key=KEY)) as port:
        inner = (
            f"GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
            f"x-api-key: {KEY}\r\n\r\n"
        ).encode()
        outer = (
            f"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
            f"Content-Length: {len(inner)}\r\n\r\n"
        ).encode() + inner
        with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
            sock.sendall(outer)
            received = b""
            while chunk := sock.recv(65536):
                received += chunk
        assert received.count(b"HTTP/1.") == 1
        assert received.startswith(b"HTTP/1.0 401") or received.startswith(b"HTTP/1.1 401")


def test_main_rejects_bad_key_config_before_loading(monkeypatch, capsys):
    import sys

    from mlx2 import server

    monkeypatch.delenv("MLX2_TEST_MISSING_KEY", raising=False)
    monkeypatch.setattr(
        sys, "argv",
        ["mlx2.server", "--model", "unused", "--api-key-env", "MLX2_TEST_MISSING_KEY"],
    )
    with pytest.raises(SystemExit) as caught:
        server.main()
    assert caught.value.code == 2
    assert "unset or empty" in capsys.readouterr().err
    monkeypatch.setattr(
        sys, "argv",
        ["mlx2.server", "--model", "unused", "--allowed-host", "box:8285"],
    )
    with pytest.raises(SystemExit):
        server.main()
    assert "must not include a port" in capsys.readouterr().err

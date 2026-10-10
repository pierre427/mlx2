"""Host-only MCP transport and executor identity regressions."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from mlx2.tool_backend import (
    ConfiguredToolBackend,
    HostedToolError,
    _HTTPMCPClient,
    _rpc_payload,
)


@pytest.mark.parametrize("mcp_first", [False, True])
def test_mcp_and_caller_function_names_must_not_collide(mcp_first):
    backend = ConfiguredToolBackend({"servers": {"docs": {"server_url": "http://127.0.0.1:9999/mcp"}}})
    client = backend.clients["docs"]
    client.list_tools = lambda: [{"name": "search"}]
    mcp = {"type": "mcp", "server_label": "docs", "server_url": client.url, "require_approval": "never"}
    function = {"type": "function", "name": "mcp__docs__search", "parameters": {"type": "object"}}
    with pytest.raises(ValueError, match="collid"):
        backend.prepare([mcp, function] if mcp_first else [function, mcp])


def test_sse_response_matches_request_id_and_joins_data_lines():
    raw = (
        b'data:{"jsonrpc":"2.0","id":7,\n'
        b'data: "result":{"tools":[]}}\n\n'
        b'data: {"jsonrpc":"2.0","method":"notifications/tools/list_changed"}\n\n'
    )
    assert _rpc_payload(raw, expected_id=7)["result"] == {"tools": []}


@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\x85"])
def test_sse_keeps_unicode_line_separators_inside_json_text(separator):
    response = {"jsonrpc": "2.0", "id": 7, "result": {"text": "a" + separator + "b"}}
    raw = ("data: " + json.dumps(response, ensure_ascii=False) + "\n\n").encode()
    assert _rpc_payload(raw, expected_id=7) == response


@pytest.mark.parametrize("reply", [
    {"jsonrpc": "2.0", "id": 8, "result": {}},
    {"jsonrpc": "2.0", "method": "notifications/tools/list_changed"},
    {"jsonrpc": "2.0", "id": 7},
])
def test_rpc_rejects_missing_or_unrelated_response(reply):
    with pytest.raises(RuntimeError, match="response"):
        _rpc_payload(json.dumps(reply).encode(), expected_id=7)


def test_list_tools_follows_pagination_and_rejects_cursor_cycles():
    client = _HTTPMCPClient("http://127.0.0.1:9999/mcp", {})
    client.initialized = True
    calls = []

    def rpc(method, params=None):
        calls.append((method, params))
        return ({"tools": [{"name": "first"}], "nextCursor": "page2"}
                if params is None else {"tools": [{"name": "last"}]})

    client._call = rpc
    assert [tool["name"] for tool in client.list_tools()] == ["first", "last"]
    assert calls == [("tools/list", None), ("tools/list", {"cursor": "page2"})]
    client._call = lambda *args: {"tools": [], "nextCursor": "same"}
    with pytest.raises(RuntimeError, match="cursor"):
        client.list_tools()


def test_list_tools_bounds_even_unique_continuation_cursors():
    client = _HTTPMCPClient("http://127.0.0.1:9999/mcp", {})
    client.initialized = True
    calls = []

    def rpc(*_args):
        calls.append(None)
        return {"tools": [], "nextCursor": str(len(calls))}

    client._call = rpc
    with pytest.raises(RuntimeError, match="page bound"):
        client.list_tools()
    assert len(calls) == 128


def _sse(*events):
    return "".join(
        "event: message\ndata: " + json.dumps(event) + "\n\n" for event in events
    ).encode()


@pytest.mark.parametrize("ping_first", [True, False])
def test_sse_server_request_sharing_the_id_is_not_the_response(ping_first):
    # MCP Streamable HTTP lets the server send its own requests (ping needs no
    # capability) on the POST's SSE stream; JSON-RPC ids are per sender, so a
    # server request may carry the same integer as the client's request.
    ping = {"jsonrpc": "2.0", "id": 2, "method": "ping"}
    reply = {"jsonrpc": "2.0", "id": 2, "result": {"content": []}}
    raw = _sse(ping, reply) if ping_first else _sse(reply, ping)
    assert _rpc_payload(raw, expected_id=2) == reply


@pytest.fixture
def restarting_mcp_server():
    """A Streamable HTTP MCP server whose sessions can be invalidated.

    Per the MCP Streamable HTTP transport, a request carrying an unknown or
    terminated Mcp-Session-Id is answered 404, and the client must start a
    new session with an InitializeRequest that carries no session id.
    """
    state = {"generation": 1, "init_session_headers": [], "calls": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def _json(self, payload, session=None):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            if session:
                self.send_header("Mcp-Session-Id", session)
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            session = self.headers.get("Mcp-Session-Id")
            current = f"s{state['generation']}"
            if body.get("method") == "initialize":
                state["init_session_headers"].append(session)
                self._json(
                    {"jsonrpc": "2.0", "id": body["id"], "result": {
                        "protocolVersion": "2025-03-26", "capabilities": {},
                        "serverInfo": {"name": "t", "version": "1"},
                    }},
                    session=current,
                )
                return
            if session != current:
                self.send_response(404)
                self.end_headers()
                return
            if "id" not in body:
                self.send_response(202)
                self.end_headers()
                return
            if body["method"] == "tools/list":
                result = {"tools": [{"name": "echo", "inputSchema": {"type": "object"}}]}
            else:
                state["calls"] += 1
                result = {"content": [{"type": "text", "text": "ok"}]}
            self._json({"jsonrpc": "2.0", "id": body["id"], "result": result})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/mcp", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_tool_call_reinitializes_after_mcp_session_expires(restarting_mcp_server):
    url, state = restarting_mcp_server
    backend = ConfiguredToolBackend({"servers": {"local": {"server_url": url}}})
    client = backend.clients["local"]
    assert backend.execute((client, "echo"), {})["content"][0]["text"] == "ok"

    state["generation"] += 1  # server restart: session s1 is gone
    assert backend.execute((client, "echo"), {})["content"][0]["text"] == "ok"
    assert client.session_id == "s2"
    # The fresh InitializeRequest must not carry the dead session id, and the
    # rejected call never ran, so the tool ran once per execute.
    assert state["init_session_headers"] == [None, None]
    assert state["calls"] == 2


def test_tool_discovery_reinitializes_after_mcp_session_expires(restarting_mcp_server):
    url, state = restarting_mcp_server
    backend = ConfiguredToolBackend({"servers": {"local": {"server_url": url}}})
    mcp = {"type": "mcp", "server_label": "local", "server_url": url, "require_approval": "never"}
    _tools, executors = backend.prepare([mcp])
    assert list(executors) == ["mcp__local__echo"]

    state["generation"] += 1
    _tools, executors = backend.prepare([mcp])
    assert list(executors) == ["mcp__local__echo"]
    assert state["init_session_headers"] == [None, None]


def test_mcp_session_reinitialization_is_bounded(restarting_mcp_server):
    """A server that ends every new session at once still fails the call."""
    url, state = restarting_mcp_server
    backend = ConfiguredToolBackend({"servers": {"local": {"server_url": url}}})
    client = backend.clients["local"]
    backend.execute((client, "echo"), {})
    original = client._call

    def always_stale(method, params=None, *, notification=False):
        if method == "tools/call":
            state["generation"] += 1  # the session dies before the call lands
        return original(method, params, notification=notification)

    client._call = always_stale
    with pytest.raises(HostedToolError, match="404"):
        backend.execute((client, "echo"), {})
    assert state["init_session_headers"] == [None, None]


@pytest.mark.parametrize(
    "url",
    [
        # Sweep 2026-10-09 review: the prefix check admitted userinfo tricks.
        "http://127.0.0.1:80@evil.example/mcp",
        "https://user:pw@docs.example/mcp",
        "http://127.0.0.1/mcp",  # loopback HTTP needs an explicit port
        "http://127.0.0.1:x/mcp",
        "http://localhost:9999/mcp",
        "http://127.0.0.2:9999/mcp",
        "https:///mcp",
        "ftp://docs.example/mcp",
        42,
    ],
)
def test_server_url_allowlist_is_structural(url):
    with pytest.raises(ValueError, match="allowlisted"):
        ConfiguredToolBackend({"servers": {"docs": {"server_url": url}}})


@pytest.mark.parametrize(
    "url", ["https://docs.example/mcp", "https://docs.example:8443/", "http://127.0.0.1:9999/mcp"]
)
def test_server_url_allowlist_accepts_https_and_loopback_http(url):
    assert ConfiguredToolBackend({"servers": {"docs": {"server_url": url}}}).clients["docs"].url == url


@pytest.mark.parametrize("code", [301, 302, 307, 308])
def test_mcp_client_refuses_redirects(code):
    # Review 2026-10-09: urllib follows a 3xx anywhere and copies the
    # configured headers to the new host; the allowlist covers one origin.
    hits = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            hits.append((self.path, self.headers.get("X-Api-Key")))
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.path == "/mcp":
                self.send_response(code)
                self.send_header("Location", f"http://127.0.0.1:{port}/elsewhere")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"jsonrpc":"2.0","id":1,"result":{}}')

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        backend = ConfiguredToolBackend(
            {"servers": {"local": {"server_url": f"http://127.0.0.1:{port}/mcp",
                                   "headers": {"X-Api-Key": "secret"}}}}
        )
        with pytest.raises(HostedToolError, match=str(code)):
            backend.clients["local"]._call("initialize", {})
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert hits == [("/mcp", "secret")]  # the redirect target was never contacted

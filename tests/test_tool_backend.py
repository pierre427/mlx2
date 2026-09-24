"""Host-only MCP transport and executor identity regressions."""
import json

import pytest

from mlx2.tool_backend import ConfiguredToolBackend, _HTTPMCPClient, _rpc_payload


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

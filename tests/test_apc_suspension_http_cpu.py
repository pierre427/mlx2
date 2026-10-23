"""Authenticated cache metadata API probes against a fake loopback engine."""
import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from test_server_quiesce import AdminEngine

from mlx2.server import handler_for


@pytest.fixture
def suspension_http():
    engine = AdminEngine()
    calls = []
    engine.cache_branches = lambda limit: calls.append(("branches", limit)) or {"entries": [], "revision": 7, "clock_frozen": False}
    engine.export_suspension = lambda identity: calls.append(("export", identity)) or {"suspension_id": identity, "clock_frozen": True}
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine, admin_token="private-admin"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield "http://127.0.0.1:" + str(server.server_port), calls
    server.shutdown()
    server.server_close()
    thread.join(2)


def test_branch_metadata_and_suspend_export_require_admin_bearer(suspension_http):
    base, calls = suspension_http
    for path, body in (("/v1/admin/apc/branches", None), ("/v1/admin/apc/suspend-export", b'{"suspension_id":"one"}')):
        with pytest.raises(HTTPError) as caught:
            urlopen(Request(base + path, data=body, headers={"Content-Type": "application/json"}))
        assert caught.value.code == 401
    assert calls == []
    with urlopen(Request(base + "/v1/admin/apc/branches?limit=3", headers={"Authorization": "Bearer private-admin"})) as response:
        assert json.load(response)["revision"] == 7
    with urlopen(Request(base + "/v1/admin/apc/suspend-export", data=b'{"suspension_id":"one"}',
                         headers={"Authorization": "Bearer private-admin", "Content-Type": "application/json"})) as response:
        assert json.load(response)["clock_frozen"]
    assert calls == [("branches", 3), ("export", "one")]


def test_export_has_no_arbitrary_path_authority_and_diagnostic_query_is_exact(suspension_http):
    base, calls = suspension_http
    for body in ({}, {"suspension_id": "one", "directory": "/anywhere"}, []):
        with pytest.raises(HTTPError) as caught:
            urlopen(Request(base + "/v1/admin/apc/suspend-export", data=json.dumps(body).encode(),
                            headers={"Authorization": "Bearer private-admin", "Content-Type": "application/json"}))
        assert caught.value.code == 400
    for query in ("limit=x", "limit=1&limit=2", "arbitrary=true"):
        with pytest.raises(HTTPError) as caught:
            urlopen(Request(base + "/v1/admin/apc/branches?" + query, headers={"Authorization": "Bearer private-admin"}))
        assert caught.value.code == 400
    assert calls == []

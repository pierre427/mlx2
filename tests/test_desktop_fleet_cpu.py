"""Synthetic loopback community attachment; never touch a live daemon/model."""
import http.client
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

from mlx2 import desktop_fleet as fleet
from mlx2.fabric.community import create_community
from mlx2.fabric.node import FabricNode


@pytest.fixture
def daemon(tmp_path, monkeypatch):
    home = tmp_path / "community"
    community = create_community(home, name="synthetic attachment")
    selected = []
    class FakePeer(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def reply(self, payload, content_type="application/json"):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        def do_GET(self):
            query = parse_qs(urlsplit(self.path).query)
            selected.append(("GET", query["model"][0], query["path"][0]))
            self.reply(b"mlx2_requests_total 1\n" if query["path"] == ["/metrics"] else b'{"model":"synthetic-peer-model"}')
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            selected.append(("POST", body["model"], self.path))
            if self.path == "/fabric/tokenize":
                self.reply(b'{"count":12}')
            else:
                self.reply(b'{"choices":[{"message":{"content":"synthetic peer reply"}}]}')
    peer_server = ThreadingHTTPServer(("127.0.0.1", 0), FakePeer)
    peer_thread = threading.Thread(target=peer_server.serve_forever, daemon=True)
    peer_thread.start()
    node = FabricNode(home, front_port=0, use_zeroconf=False)
    node.community = community
    remote = SimpleNamespace(state="alive")
    node._peers["synthetic-peer"] = remote
    monkeypatch.setattr(node, "_peer_view", lambda p: SimpleNamespace(models={"synthetic-peer-model"}))
    nodes = [{"nid": community.nid, "state": "alive", "hostname": "owner", "models": []},
             {"nid": "synthetic-peer", "state": "alive", "hostname": "second machine", "models": ["synthetic-peer-model"]}]
    monkeypatch.setattr(node, "_nodes", lambda: nodes)
    monkeypatch.setattr(node, "_connect", lambda peer, **kw: http.client.HTTPConnection("127.0.0.1", peer_server.server_port, timeout=2))
    front = ThreadingHTTPServer(("127.0.0.1", 0), node._front_handler())
    node._front_server = front
    front_thread = threading.Thread(target=front.serve_forever, daemon=True)
    front_thread.start()
    node._server_threads[id(front)] = front_thread
    node._write_desktop_descriptor()
    (home / "node.pid").write_text(str(os.getpid()))
    preferences = {"community_home": str(home), "community_front_port": front.server_port,
                   "community_fabric_port": 8291, "default_share": False, "shared_servers": []}
    control = SimpleNamespace(home=tmp_path, preferences=preferences, engines={}, profile=lambda name: None)
    monkeypatch.setattr(FabricNode, "start", lambda self: (_ for _ in ()).throw(AssertionError("must not take over an existing node")))
    class Collector:
        def __init__(self, urls, **kw):
            self.url = urls[0]
        def sample(self):
            return [{"endpoint": {"url": self.url, "pid": None}}]
        def close(self):
            pass
    monkeypatch.setattr(fleet, "InferenceCollector", Collector)
    desktop = fleet.DesktopFleet(control)
    yield SimpleNamespace(home=home, node=node, front=front, desktop=desktop, controller=control, selected=selected, nodes=nodes)
    desktop.close()
    node.close()
    peer_server.shutdown(); peer_server.server_close()
    front_thread.join(2); peer_thread.join(2)


def request(port, path, *, method="GET", body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    if body is not None:
        body = json.dumps(body)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    data = response.read()
    connection.close()
    return response.status, data


def test_attach_exact_telemetry_chat_tokenize_and_detach_preserves_owner(daemon):
    env = daemon
    env.desktop.start()
    assert not env.desktop.owns_node and env.desktop.node is None
    snapshot = env.desktop.snapshot()
    assert snapshot["ownership"] == "attached" and len(snapshot["nodes"]) == 2
    assert snapshot["servers"][0]["endpoint"]["nid"] == "synthetic-peer"
    target = env.desktop.chat_target("synthetic-peer", "synthetic-peer-model")
    path = urlsplit(target["url"]).path
    headers = {"Authorization": "Bearer " + target["api_key"]}
    status, metrics = request(env.desktop.proxy.server_port, path + "/metrics", headers=headers)
    assert status == 200 and b"mlx2_requests_total" in metrics
    status, reply = request(env.desktop.proxy.server_port, path + "/v1/chat/completions", method="POST", body={"model": "forged-other-model", "messages": []}, headers=headers)
    assert status == 200 and b"synthetic peer reply" in reply
    status, count = request(env.desktop.proxy.server_port, path + "/tokenize", method="POST", body={"messages": []}, headers=headers)
    assert status == 200 and json.loads(count)["count"] == 12
    assert all(row[1] == "synthetic-peer-model" for row in env.selected)
    proxy = env.desktop.proxy
    env.desktop.close()
    env.desktop.proxy = None
    assert proxy.fileno() == -1
    assert (env.home / "node.pid").read_text() == str(os.getpid())
    assert request(env.front.server_port, "/v1/fabric/nodes")[0] == 200


def test_front_private_token_host_origin_and_exact_model_gate(daemon):
    env = daemon
    port = env.front.server_port
    assert request(port, "/v1/fabric/desktop/state")[0] == 403
    headers = {"Authorization": "Bearer " + env.node._desktop_token}
    assert request(port, "/v1/fabric/desktop/state", headers=headers)[0] == 200
    assert request(port, "/v1/fabric/desktop/state", headers={**headers, "Host": "evil.example"})[0] == 421
    assert request(port, "/v1/fabric/desktop/state", headers={**headers, "Origin": f"http://127.0.0.1:{port}"})[0] == 403
    body = {"nid": "synthetic-peer", "model": "absent", "method": "POST", "path": "/v1/chat/completions", "body": {"messages": []}}
    assert request(port, "/v1/fabric/desktop/request", method="POST", body=body, headers=headers)[0] == 400
    assert not env.selected


def test_private_descriptor_stale_identity_and_old_daemon_refuse_no_takeover(daemon):
    env = daemon
    descriptor = env.home / "desktop-control.json"
    assert descriptor.stat().st_mode & 0o777 == 0o600
    data = json.loads(descriptor.read_text())
    data["created"] += 1
    descriptor.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="stale"):
        env.desktop.start()
    descriptor.unlink()
    with pytest.raises(ValueError, match="predates"):
        env.desktop.start()


def test_ui_share_overlay_is_opt_in_expiring_and_preserves_owner_entries(daemon, monkeypatch):
    env = daemon
    owner = {"url": "http://127.0.0.1:9998", "pid": os.getpid(), "api_key_file": None}
    env.node.extra_local_servers = (owner,)
    env.desktop.start()
    env.desktop.sync()
    assert not env.node._desktop_shares
    worker = {"url": "http://127.0.0.1:9999", "api_key_file": None}
    env.controller.engines["default"] = SimpleNamespace(child=SimpleNamespace(pid=os.getpid()), worker=worker)
    env.controller.preferences["default_share"] = True
    env.desktop.sync()
    assert len(env.node._desktop_shares) == 1
    assert {row["url"] for row in env.node._servers()} == {owner["url"], worker["url"]}
    env.node._desktop_shares[env.desktop.client_id]["expires"] = 0
    assert [row["url"] for row in env.node._servers()] == [owner["url"]]
    env.desktop.sync()
    assert env.node._desktop_shares
    env.desktop.close(); env.desktop.proxy = None
    assert not env.node._desktop_shares
    assert env.node.extra_local_servers == (owner,)


def test_descriptor_can_attach_when_legacy_pid_file_was_removed(daemon):
    env = daemon
    (env.home / "node.pid").unlink()
    env.desktop.start()
    assert env.desktop.snapshot()["ownership"] == "attached"


def test_node_owner_lock_refuses_second_identity_owner_before_bind(tmp_path):
    import fcntl

    from mlx2.fabric import FabricError
    home = tmp_path / "community"
    create_community(home, name="synthetic lock")
    with (home / "node-owner.lock").open("a") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(FabricError, match="live node owner"):
            FabricNode(home, use_zeroconf=False).start()

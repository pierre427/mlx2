"""Community-authenticated telemetry and exact-node chat for the native app."""
from __future__ import annotations

import base64
import http.client
import json
import os
import threading
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .fabric import FabricError
from .fabric.community import load_community
from .fabric.node import FabricNode
from .inference_monitor import InferenceCollector


class DesktopFleet:
    def __init__(self, controller):
        self.controller = controller
        self.node = None
        self.proxy = None
        self.key = os.urandom(32).hex()
        self.collectors = {}
        self.error = None
        self.community = None
        self.control = None
        self.client_id = uuid.uuid4().hex
        self.owns_node = False

    def start(self):
        settings = self.controller.preferences
        home = Path(settings["community_home"]).expanduser()
        self.community = load_community(home)  # Admission is required; never create implicit trust.
        from .fabric.registry import pid_alive
        pid_path = home / "node.pid"
        pid = int(pid_path.read_text()) if pid_path.exists() else 0
        descriptor = home / "desktop-control.json"
        if (not pid or not pid_alive(pid)) and descriptor.exists():
            from .http_security import load_secret_file
            candidate = json.loads(load_secret_file(descriptor, flag="Community desktop control"))
            if type(candidate.get("pid")) is int and pid_alive(candidate["pid"]):
                pid = candidate["pid"]
        if pid and pid_alive(pid):
            self._attach(home, pid)
        else:
            node = FabricNode(home, front_port=settings["community_front_port"],
                              fabric_port=settings["community_fabric_port"])
            node.start()
            self.node = node
            self.owns_node = True
            self.pid_path = pid_path
            pid_path.write_text(str(os.getpid()))
        fleet = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            def log_message(self, *_args):
                pass
            def handle_request(self):
                from .http_security import GateRejection, authorize, policy_for_bind
                begun = False
                try:
                    authorize(policy_for_bind("127.0.0.1", api_key=fleet.key), self.command, self.path,
                              self.headers, local_address=self.connection.getsockname()[0])
                    if self.headers.get("Origin"):
                        self.send_error(403)
                        return
                    parts = urllib.parse.urlsplit(self.path).path.split("/")
                    if len(parts) < 5 or parts[1] != "peer":
                        raise ValueError("Invalid peer route")
                    nid, encoded = parts[2:4]
                    model = base64.b64decode(encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True).decode()
                    path = "/" + "/".join(parts[4:])
                    body = None
                    if self.command == "POST":
                        length = int(self.headers.get("Content-Length", "0"))
                        if not 0 < length <= 64 * 1024 * 1024:
                            raise ValueError("Chat request exceeds the size bound")
                        body = json.loads(self.rfile.read(length))
                    connection, response = fleet.open_request(nid, model, self.command, path, body)
                    begun = True
                    FabricNode._desktop_stream(self, connection, response)
                except GateRejection as error:
                    self.send_error(error.status)
                    self.close_connection = True
                except (OSError, ValueError, TypeError, KeyError, FabricError, http.client.HTTPException):
                    self.close_connection = True
                    if not begun:
                        self.send_error(400, "Community request refused")
            do_GET = handle_request
            do_POST = handle_request
        self.proxy = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.proxy.daemon_threads = True
        threading.Thread(target=self.proxy.serve_forever, daemon=True, name="mlx2-community-ui").start()

    def _attach(self, home, pid):
        import psutil

        from .http_security import load_secret_file
        path = home / "desktop-control.json"
        if not path.exists():
            raise ValueError("This running community node predates UI attachment. Its owner must upgrade and restart it; the UI will not take it over.")
        value = json.loads(load_secret_file(path, flag="Community desktop control"))
        parsed = urllib.parse.urlsplit(value.get("url", ""))
        if (value.get("version") != 1 or value.get("pid") != pid or value.get("created") != psutil.Process(pid).create_time()
                or value.get("nid") != self.community.nid or value.get("community_id") != self.community.community_id
                or parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"} or not parsed.port
                or parsed.username or parsed.password or parsed.path not in {"", "/"} or parsed.query or parsed.fragment
                or not isinstance(value.get("token"), str) or len(value["token"]) != 64 or any(c not in "0123456789abcdef" for c in value["token"])):
            raise ValueError("The running community node's private descriptor is stale or mismatched")
        self.control = value
        self._nodes()  # Verify authenticated identity before declaring attachment.

    def _control_request(self, method, path, body=None):
        parsed = urllib.parse.urlsplit(self.control["url"])
        connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=600 if path.endswith("/request") else 5)
        headers = {"Authorization": "Bearer " + self.control["token"]}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        try:
            connection.request(method, path, body=data, headers=headers)
            return connection, connection.getresponse()
        except BaseException:
            connection.close()
            raise

    def _control_json(self, method, path, body=None):
        connection, response = self._control_request(method, path, body)
        try:
            data = response.read(2 * 1024 * 1024 + 1)
            if response.status != 200 or len(data) > 2 * 1024 * 1024:
                raise ValueError("The existing community node refused the private desktop request")
            return json.loads(data)
        finally:
            connection.close()

    def _nodes(self):
        if self.node is not None:
            return self.node._nodes()
        if self.control is None:
            return []
        state = self._control_json("GET", "/v1/fabric/desktop/state")
        if state.get("nid") != self.community.nid or state.get("community_id") != self.community.community_id:
            raise ValueError("The attached community node identity changed")
        return state["nodes"]

    def open_request(self, nid, model, method, path, body=None):
        if self.node is not None:
            return self.node._desktop_open(nid, model, method, path, body)
        if self.control is None:
            raise ValueError("Community is disconnected")
        return self._control_request("POST", "/v1/fabric/desktop/request",
            {"nid": nid, "model": model, "method": method, "path": path, "body": body})

    def sync(self):
        if self.node is None and self.control is None:
            return
        entries = []
        for identity, engine in self.controller.engines.items():
            profile = self.controller.profile(identity) or {}
            share = self.controller.preferences["default_share"] if identity == "default" else profile.get("share")
            if share and engine.child is not None and engine.worker:
                entries.append({"url": engine.worker["url"], "pid": engine.child.pid,
                                "api_key_file": engine.worker["api_key_file"]})
        from .inference_monitor import discover_servers, local_server_api_key
        shared = {(s["url"], s["pid"]) for s in self.controller.preferences["shared_servers"]}
        for endpoint in discover_servers() if shared else ():
            if (endpoint.url, endpoint.pid) not in shared:
                continue
            key = local_server_api_key(endpoint)
            key_path = None
            if key:
                folder = self.controller.home / "community-keys"
                folder.mkdir(mode=0o700, exist_ok=True)
                import hashlib
                key_path = folder / hashlib.sha256(f"{endpoint.url}|{endpoint.pid}".encode()).hexdigest()
                fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
                with os.fdopen(fd, "w") as stream:
                    os.fchmod(stream.fileno(), 0o600)
                    stream.write(key)
            entries.append({"url": endpoint.url, "pid": endpoint.pid,
                            "api_key_file": str(key_path) if key_path else None})
        if self.node is not None:
            self.node._desktop_set_shares(self.client_id, entries)
        else:
            self._control_json("POST", "/v1/fabric/desktop/shares", {"client_id": self.client_id, "entries": entries})

    def base_url(self, nid, model):
        encoded = base64.urlsafe_b64encode(model.encode()).decode().rstrip("=")
        return f"http://127.0.0.1:{self.proxy.server_port}/peer/{nid}/{encoded}"

    def snapshot(self):
        if self.node is None and self.control is None:
            return {"connected": False, "nodes": [], "servers": []}
        servers, live = [], set()
        nodes = self._nodes()
        from .desktop import json_safe
        for view in nodes:
            if view["state"] != "alive" or view["nid"] == self.community.nid:
                continue
            nid = view["nid"]
            for model in view["models"]:
                identity = (nid, model)
                live.add(identity)
                if identity not in self.collectors:
                    self.collectors[identity] = InferenceCollector(urls=(self.base_url(nid, model),),
                        discover=False, api_key=self.key, timeout_seconds=2)
                reading, = self.collectors[identity].sample()
                item = json_safe(reading)
                item["endpoint"].update(remote=True, nid=nid, remote_model=model,
                                          hostname=view.get("hostname"), model_path=model)
                item["model"] = f"{model} · {view.get('hostname') or nid}"
                servers.append(item)
        for identity in set(self.collectors) - live:
            self.collectors.pop(identity).close()
        front = self.control["url"] if self.control else f"http://127.0.0.1:{self.node.actual_front_port}"
        return {"connected": True, "ownership": "managed" if self.owns_node else "attached",
                "nodes": nodes, "servers": servers, "front_url": front + "/v1",
                "community_id": self.community.community_id}

    def chat_target(self, nid, model):
        view = next((row for row in self._nodes() if row["nid"] == nid), None)
        if not view or view["state"] != "alive" or model not in view["models"]:
            raise ValueError("Peer or model is no longer available")
        return {"url": self.base_url(nid, model), "api_key": self.key, "model": model}

    def close(self):
        for collector in self.collectors.values():
            collector.close()
        if self.proxy:
            self.proxy.shutdown()
            self.proxy.server_close()
        if self.control is not None:
            try:
                self._control_json("POST", "/v1/fabric/desktop/shares", {"client_id": self.client_id, "entries": []})
            except (OSError, ValueError, http.client.HTTPException):
                pass  # The owned overlay expires after 30 seconds if its owner vanishes.
            self.control = None
        if self.node and self.owns_node:
            self.node.close()
        if getattr(self, "pid_path", None) and self.pid_path.exists() and self.pid_path.read_text().strip() == str(os.getpid()):
            self.pid_path.unlink()


def membership_command(controller, request):
    from .fabric.community import create_community, make_invite
    home = Path(controller.preferences["community_home"]).expanduser()
    command = request["command"]
    if command == "community_create":
        community = create_community(home, name=request.get("name") or "MLX2")
        return {"invite": make_invite(community), "nid": community.nid}
    if command == "community_invite":
        return {"invite": make_invite(load_community(home))}
    if command == "community_join":
        from types import SimpleNamespace

        from .fabric.cli import _join
        _join(SimpleNamespace(home=home, invite=request["invite"], via=request.get("via"), timeout=15))
        return {"joined": True}
    raise ValueError("Unknown community operation")

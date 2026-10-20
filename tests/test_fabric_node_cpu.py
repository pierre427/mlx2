from __future__ import annotations

import hashlib
import http.client
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from mlx2.fabric.community import (
    client_context,
    create_community,
    install_join_bundle,
    join_client_context,
    join_proof,
    make_csr,
    revoke,
)
from mlx2.fabric.discovery import Candidate
from mlx2.fabric.link import LinkReceipt
from mlx2.fabric.node import FabricNode, _Peer
from mlx2.fabric.registry import register_local_server


def _port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Fake(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"data": [{"id": "fake-b", "loaded": True}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        request_body = self.rfile.read(length)
        if b'"slow"' in request_body:
            time.sleep(1)
        body = json.dumps(
            {"id": "fixed", "choices": [{"message": {"content": "ok"}}]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format, *_args):
        pass


def _wait(url, predicate, timeout=12):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                value = json.load(response)
            if predicate(value):
                return value
        except OSError:
            pass
        time.sleep(0.1)
    raise AssertionError(f"timeout waiting for {url}")


def test_probe_reuses_exactly_one_connection(tmp_path) -> None:
    class Response:
        status = 200

        def __init__(self, length):
            self.length = length

        def read(self):
            return bytes(self.length)

    class Connection:
        def __init__(self):
            self.length = 0
            self.closed = False

        def request(self, _method, _path, *, body, headers):
            assert int(headers["Content-Length"]) == len(body)
            self.length = int(headers["X-Mlx2-Download-Len"])

        def getresponse(self):
            return Response(self.length)

        def close(self):
            self.closed = True

    node = FabricNode(tmp_path, use_zeroconf=False)
    peer = _Peer(Candidate("0123456789abcdef", ("10.0.0.1",), 8291, "", 0, "", "static"))
    connection = Connection()
    calls = 0

    def connect(_peer, *, operation_timeout_s):
        nonlocal calls
        calls += 1
        assert operation_timeout_s == 3.0
        return connection

    node._connect = connect
    receipt = node._probe(peer)
    assert calls == 1
    assert connection.closed is True
    assert receipt.rtt_ms_median < 5


def test_local_poll_is_independent_of_peer_heartbeat(tmp_path, monkeypatch) -> None:
    node = FabricNode(
        tmp_path, use_zeroconf=False, heartbeat_s=1, local_poll_s=5
    )

    class StopAfterFour:
        calls = 0

        def wait(self, _timeout):
            self.calls += 1
            return self.calls > 4

    peer = _Peer(
        Candidate("0123456789abcdef", ("10.0.0.1",), 8291, "", 0, "", "static")
    )
    node._stop = StopAfterFour()
    node._peers[peer.candidate.nid] = peer
    node._last_local_poll = 0
    refreshes = []
    heartbeats = []
    monkeypatch.setattr(
        "mlx2.fabric.node.time.monotonic", iter((1.0, 2.0, 4.0, 6.0)).__next__
    )
    node._refresh_advert = lambda: refreshes.append(True)
    node._heartbeat = lambda current: heartbeats.append(current.candidate.nid)

    node._maintain()

    assert len(refreshes) == 1
    assert len(heartbeats) == 4


def test_link_probe_cadence_and_bounded_history(tmp_path) -> None:
    node = FabricNode(tmp_path, use_zeroconf=False)
    peer = _Peer(
        Candidate("0123456789abcdef", ("10.0.0.1",), 8291, "", 0, "", "static")
    )
    probes = []

    def probe(_peer, *, bandwidth):
        probes.append(bandwidth)
        value = float(len(probes))
        return LinkReceipt(
            peer.candidate.nid,
            "tls1.3/tcp",
            value,
            value,
            20,
            1.0 if bandwidth else None,
            1.0 if bandwidth else None,
            (8 << 20) if bandwidth else 0,
            value,
        )

    node._probe = probe
    node._probe_link_if_due(peer, 1)
    node._probe_link_if_due(peer, 30)
    node._probe_link_if_due(peer, 31)
    node._probe_link_if_due(peer, 301)
    assert probes == [True, False, True]

    for index in range(25):
        receipt = LinkReceipt(
            peer.candidate.nid,
            "tls1.3/tcp",
            index,
            index,
            20,
            None,
            None,
            0,
            index,
        )
        node._record_link(peer, receipt)
    assert len(peer.link_history) == 20
    assert peer.link_history[0].measured_at == 5
    assert peer.link is peer.link_history[-1]


def test_two_node_join_route_receipts_refusal_and_access_controls(tmp_path, monkeypatch):
    # Loopback-only unit test: pin the listener address instead of
    # enumerating interfaces, which needs ifaddr from ~/.mlx2-fabric-site
    # (outside the venv; see qualification/runs/fabric-m5-m3-20261018).
    monkeypatch.setenv("MLX2_FABRIC_ADDRESSES", "127.0.0.1")
    home_a, home_b = tmp_path / "a", tmp_path / "b"
    a = create_community(home_a, hosts=("127.0.0.1",))
    key_b, csr_b = make_csr(home_b)
    from mlx2.fabric.community import node_id_for_key

    nid_b = node_id_for_key(key_b)
    port_a, port_b, front_a, front_b = _port(), _port(), _port(), _port()
    (home_a / "peers.json").write_text(
        json.dumps([{"host": "127.0.0.1", "port": port_b, "nid": nid_b}])
    )
    node_a = FabricNode(
        home_a,
        front_port=front_a,
        fabric_port=port_a,
        include_loopback=True,
        use_zeroconf=False,
        heartbeat_s=0.2,
        request_timeout_s=2,
    ).start()
    fake = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    fake_thread = threading.Thread(target=fake.serve_forever, daemon=True)
    fake_thread.start()
    node_b = None
    try:
        conn = http.client.HTTPSConnection(
            "127.0.0.1", port_a, context=join_client_context(), timeout=5
        )
        conn.connect()
        digest = hashlib.sha256(conn.sock.getpeercert(binary_form=True)).hexdigest()
        proof = join_proof(a.secret, csr_b, digest)
        conn.request(
            "POST",
            "/fabric/join",
            json.dumps({"csr_pem": csr_b, "proof": proof}),
            {"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        joined = json.loads(response.read())
        conn.close()
        assert response.status == 200
        b = install_join_bundle(home_b, joined["bundle"], key_b)
        (home_b / "peers.json").write_text(
            json.dumps([{"host": "127.0.0.1", "port": port_a, "nid": a.nid}])
        )
        register_local_server(home_b, port=fake.server_port, pid=os.getpid())
        node_b = FabricNode(
            home_b,
            front_port=front_b,
            fabric_port=port_b,
            include_loopback=True,
            use_zeroconf=False,
            heartbeat_s=0.2,
            request_timeout_s=2,
        ).start()
        _wait(
            f"http://127.0.0.1:{front_a}/v1/fabric/nodes",
            lambda value: any(
                item["nid"] == b.nid and item["state"] == "alive" and item["link"]
                for item in value["data"]
            ),
        )

        peer = node_a._peers[b.nid]
        assert peer.link.rtt_ms_median < 5

        links = _wait(
            f"http://127.0.0.1:{front_a}/v1/fabric/links",
            lambda value: bool(value["data"]),
        )["data"]
        link_entry = next(item for item in links if item["peer_nid"] == b.nid)
        assert link_entry["latest"] == link_entry["history"][-1]
        assert len(link_entry["history"]) <= 20

        keepalive = http.client.HTTPSConnection(
            "127.0.0.1", port_a, context=client_context(b), timeout=5
        )
        keepalive.request("GET", "/fabric/advert")
        current = keepalive.getresponse()
        etag = current.getheader("ETag")
        current.read()
        assert current.version == 11
        assert current.getheader("Content-Length") is not None
        keepalive.request("GET", "/fabric/advert", headers={"If-None-Match": etag})
        unchanged = keepalive.getresponse()
        unchanged.read()
        keepalive.close()
        assert unchanged.status == 304
        assert unchanged.version == 11
        assert unchanged.getheader("Content-Length") == "0"

        front = http.client.HTTPConnection("127.0.0.1", front_a, timeout=5)
        front.request("GET", "/v1/fabric/nodes")
        front_response = front.getresponse()
        front_response.read()
        front.close()
        assert front_response.version == 11
        assert front_response.getheader("Content-Length") is not None

        request = urllib.request.Request(
            f"http://127.0.0.1:{front_a}/v1/chat/completions",
            data=json.dumps({"model": "fake-b", "messages": []}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            value = json.load(response)
            route_header = response.headers["X-Mlx2-Fabric-Route"]
        assert route_header and value["mlx2_fabric"]["served_by"]["nid"] == b.nid
        assert value["mlx2_fabric"]["states"]["observed_used"] is True
        assert (
            json.loads(
                (home_a / "receipts" / "routes.jsonl").read_text().splitlines()[-1]
            )["states"]["observed_used"]
            is True
        )

        streaming = urllib.request.Request(
            f"http://127.0.0.1:{front_a}/v1/chat/completions",
            data=json.dumps(
                {"model": "fake-b", "messages": [], "stream": True}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(streaming, timeout=5) as response:
            assert response.headers["X-Mlx2-Fabric-Route"]
            assert json.load(response)["id"] == "fixed"

        slow = urllib.request.Request(
            f"http://127.0.0.1:{front_a}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "fake-b",
                    "messages": [{"role": "user", "content": "slow"}],
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        closer = threading.Timer(0.2, node_b.close)
        closer.start()
        try:
            urllib.request.urlopen(slow, timeout=5)
            raise AssertionError("lost peer must not return a late success")
        except urllib.error.HTTPError as exc:
            assert exc.code == 502
            aborted = json.load(exc)["mlx2_fabric"]
            assert aborted["outcome"] == "aborted_peer_lost"
            assert aborted["states"]["observed_used"] is False
        finally:
            closer.join(timeout=2)
        node_b = None

        missing = urllib.request.Request(
            f"http://127.0.0.1:{front_a}/v1/completions",
            data=json.dumps({"model": "missing", "prompt": "x"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(missing, timeout=5)
        except urllib.error.HTTPError as exc:
            assert (
                exc.code == 404 and json.load(exc)["mlx2_fabric"]["kind"] == "refused"
            )

        revoke(a, b.nid, reason="test")
        _wait(
            f"http://127.0.0.1:{front_a}/v1/fabric/nodes",
            lambda value: any(
                item["nid"] == b.nid and item["state"] == "revoked"
                for item in value["data"]
            ),
        )
        try:
            urllib.request.urlopen(request, timeout=5)
        except urllib.error.HTTPError as exc:
            assert exc.code == 404

        anonymous = http.client.HTTPSConnection(
            "127.0.0.1", port_a, context=join_client_context(), timeout=5
        )
        anonymous.request("GET", "/fabric/advert")
        denied = anonymous.getresponse()
        denied.read()
        anonymous.close()
        assert denied.status == 403
        wrong = http.client.HTTPSConnection(
            "127.0.0.1", port_a, context=join_client_context(), timeout=5
        )
        wrong.request(
            "POST",
            "/fabric/join",
            json.dumps({"csr_pem": csr_b, "proof": "0" * 64}),
            {"Content-Type": "application/json"},
        )
        denied = wrong.getresponse()
        denied.read()
        wrong.close()
        assert denied.status == 403
    finally:
        if node_b:
            node_b.close()
        node_a.close()
        fake.shutdown()
        fake.server_close()
        fake_thread.join(timeout=2)


def test_rescan_rebinds_listeners_advertiser_and_browser_when_addresses_change(monkeypatch, tmp_path):
    """Found live 2026-10-18: reseating cables renumbered link-local addresses
    and nodes kept their startup bindings, so peers vanished from each other."""
    import threading as _threading
    from types import SimpleNamespace

    import mlx2.fabric.node as node_module

    class FakeServer:
        def __init__(self, address):
            self.server_address = (address, 9999)
            self.server_port = 9999
            self._stop = _threading.Event()
            self.closed = False

        def serve_forever(self, poll_interval=0.05):
            self._stop.wait()

        def shutdown(self):
            self._stop.set()

        def server_close(self):
            self.closed = True

    made = {"advertisers": [], "browsers": []}

    class Fake:
        def __init__(self, kind, **kwargs):
            self.kwargs, self.closed = kwargs, False
            made[kind].append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(node_module, "Advertiser", lambda **kw: Fake("advertisers", **kw))
    monkeypatch.setattr(node_module, "Browser", lambda **kw: Fake("browsers", **kw))
    node = node_module.FabricNode(tmp_path, use_zeroconf=True)
    node.community = SimpleNamespace(secret=b"s" * 32, epoch=1, nid="0" * 16)
    node._stable_advert_digest = "a" * 64
    monkeypatch.setattr(node, "_bind_fabric", lambda address: FakeServer(address))
    monkeypatch.setattr(node, "_refresh_advert", lambda: None)
    node._fabric_servers = [FakeServer("10.0.0.1"), FakeServer("169.254.55.226")]
    for server in node._fabric_servers:
        node._serve(server)
    node._addresses = ["10.0.0.1", "169.254.55.226"]
    node._advertiser = Fake("advertisers")
    node._browser = Fake("browsers")
    old_usb = node._fabric_servers[1]

    current = ["10.0.0.1", "169.254.210.91"]
    monkeypatch.setattr(node_module, "allowed_addresses", lambda _loopback: list(current))
    assert node._rescan_addresses() is True
    assert node._addresses == ["10.0.0.1", "169.254.210.91"]
    assert old_usb.closed
    assert made["advertisers"][0].closed and made["browsers"][0].closed
    assert made["advertisers"][-1].kwargs["addresses"] == ["10.0.0.1", "169.254.210.91"]
    assert made["browsers"][-1].kwargs["addresses"] == ["10.0.0.1", "169.254.210.91"]
    assert node._rescan_addresses() is False  # unchanged set: nothing rebuilt
    current[:] = []
    assert node._rescan_addresses() is False  # no address at all: keep what is bound
    for server in node._fabric_servers:
        server.shutdown()

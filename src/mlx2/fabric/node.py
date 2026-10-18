from __future__ import annotations

import base64
import hashlib
import http.client
import json
import os
import socket
import ssl
import tempfile
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from . import FabricError
from .capability import (
    advert_digest,
    build_advert,
    host_facts,
    local_models,
    sign_advert,
    stable_advert_digest,
    verify_advert,
)
from .community import (
    admission_entries,
    admit,
    canonical_json,
    check_peer,
    client_context,
    community_tag,
    ensure_fresh_certificate,
    join_ack_proof,
    join_proof,
    load_community,
    merge_admission_log,
    merge_revocations,
    revocation_entries,
    revoked_nids,
    server_context,
    sha256_hex,
    sign,
    verify,
)
from .discovery import (
    Advertiser,
    Browser,
    Candidate,
    allowed_addresses,
    connect_https,
    load_static_peers,
)
from .link import LinkBudget, LinkReceipt, echo_response, probe_link
from .placement import PeerView, Placement, place, prefill_candidate
from .registry import list_local_servers


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode().rstrip("=")


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical_json(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(raw, path)
    except BaseException:
        Path(raw).unlink(missing_ok=True)
        raise


def _api_key(path: str | None) -> str | None:
    if path is None:
        return None
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _response_reason(status: int, body: bytes) -> str:
    try:
        value = json.loads(body)
        error = value.get("error") if isinstance(value, dict) else None
        if isinstance(error, dict):
            return str(error.get("reason") or error.get("message") or f"HTTP {status}")
        if isinstance(error, str):
            return error
        if isinstance(value, dict) and isinstance(value.get("reason"), str):
            return value["reason"]
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass
    return f"HTTP {status}"


def _handoff_observed(body: bytes) -> bool:
    values = []
    try:
        values.append(json.loads(body))
    except (UnicodeDecodeError, json.JSONDecodeError):
        for line in body.splitlines():
            if not line.startswith(b"data:"):
                continue
            payload = line.removeprefix(b"data:").strip()
            if payload == b"[DONE]":
                continue
            try:
                values.append(json.loads(payload))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
    for value in values:
        if not isinstance(value, dict):
            continue
        used = (value.get("mlx2") or {}).get("prefill_handoff") or value.get(
            "mlx2.prefill_handoff"
        )
        if isinstance(used, dict) and used.get("observed_used") is True:
            return True
    return False


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


@dataclass
class _Peer:
    candidate: Candidate
    address: str | None = None
    state: str = "suspect"
    advert: dict[str, Any] | None = None
    last_digest: str | None = None
    last_seen: float | None = None
    failures: int = 0
    first_failure: float | None = None
    reason: str = "not contacted"
    link: LinkReceipt | None = None
    link_history: deque[LinkReceipt] = field(default_factory=lambda: deque(maxlen=20))
    last_log: float = 0.0
    last_rtt_probe: float = 0.0
    last_full_probe: float = 0.0


class FabricNode:
    def __init__(
        self,
        home: Path,
        front_host: str = "127.0.0.1",
        front_port: int = 8290,
        fabric_port: int = 8291,
        include_loopback: bool = False,
        extra_local_servers=(),
        peers_file: Path | None = None,
        use_zeroconf: bool = True,
        heartbeat_s: float = 1.0,
        local_poll_s: float = 5.0,
        address_rescan_s: float = 5.0,
        link_budget_ms: float = 5.0,
        request_timeout_s: float = 600,
    ) -> None:
        if front_host not in {"127.0.0.1", "::1", "localhost"}:
            raise FabricError("fabric front door must bind loopback")
        if heartbeat_s <= 0 or local_poll_s <= 0:
            raise FabricError("fabric polling intervals must be positive")
        self.home = Path(home)
        self.front_host = front_host
        self.front_port = front_port
        self.fabric_port = fabric_port
        self.include_loopback = include_loopback
        self.extra_local_servers = tuple(extra_local_servers)
        self.peers_file = Path(peers_file) if peers_file is not None else None
        self.use_zeroconf = use_zeroconf
        self.heartbeat_s = heartbeat_s
        self.local_poll_s = local_poll_s
        # Interfaces come and go (a cable reseated, Thunderbolt renegotiating,
        # link-local renumbering); bindings follow them instead of going stale.
        self.address_rescan_s = address_rescan_s
        self._last_address_scan = 0.0
        self._server_threads: dict[int, threading.Thread] = {}
        self.link_budget = LinkBudget(link_budget_ms)
        self.request_timeout_s = request_timeout_s
        self.community = None
        self._addresses: list[str] = []
        self._fabric_servers: list[_Server] = []
        self._front_server: _Server | None = None
        self._threads: list[threading.Thread] = []
        self._maintenance: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._advertiser = None
        self._browser = None
        self._signed_advert: dict[str, Any] = {}
        self._advert_digest = ""
        self._stable_advert_digest = ""
        self._last_advert_signed = 0.0
        self._last_local_poll = 0.0
        self._seq = 0
        self._runtime: dict | None = None
        self._runtime_tried = False
        self._local: list[dict] = []
        self._peers: dict[str, _Peer] = {}
        self._join_failures: dict[str, list[float]] = {}

    @property
    def actual_front_port(self) -> int:
        return self._front_server.server_port if self._front_server else self.front_port

    @property
    def actual_fabric_port(self) -> int:
        return (
            self._fabric_servers[0].server_port
            if self._fabric_servers
            else self.fabric_port
        )

    def start(self) -> FabricNode:
        if self.community is not None:
            return self
        self.community = load_community(self.home)
        self._addresses = allowed_addresses(self.include_loopback)
        if not self._addresses:
            self.community = None
            raise FabricError("no allowed private address for fabric listener")
        ensure_fresh_certificate(
            self.community, hosts=(*self._addresses, socket.gethostname())
        )
        self.community = load_community(self.home)
        for address in self._addresses:
            self._fabric_servers.append(self._bind_fabric(address))
        self._front_server = _Server(
            (self.front_host, self.front_port), self._front_handler()
        )
        self.front_port = self._front_server.server_port
        self._refresh_advert()
        tag = community_tag(self.community.secret, self.community.epoch)
        if self.use_zeroconf:
            self._advertiser = Advertiser(
                nid=self.community.nid,
                port=self.actual_fabric_port,
                tag=tag,
                epoch=self.community.epoch,
                adv=self._stable_advert_digest[:12],
                addresses=self._addresses,
            )
        peer_path = self.peers_file or self.home / "peers.json"
        if peer_path.exists():
            self._set_candidates(load_static_peers(peer_path))
        elif self.use_zeroconf:
            self._browser = Browser(tag=tag, addresses=self._addresses)
        for server in [*self._fabric_servers, self._front_server]:
            self._serve(server)
        self._last_address_scan = time.monotonic()
        self._maintenance = threading.Thread(target=self._maintain, daemon=True)
        self._maintenance.start()
        return self

    def _bind_fabric(self, address: str) -> _Server:
        assert self.community is not None
        server = _Server((address, self.fabric_port), self._fabric_handler())
        server.socket = server_context(self.community).wrap_socket(
            server.socket, server_side=True
        )
        if self.fabric_port == 0:
            self.fabric_port = server.server_port
        return server

    def _serve(self, server: _Server) -> None:
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )
        thread.start()
        self._threads.append(thread)
        self._server_threads[id(server)] = thread

    def _rescan_addresses(self) -> bool:
        """Rebind listeners, advertiser and browser when the allowed set changes."""
        assert self.community is not None
        current = allowed_addresses(self.include_loopback)
        if not current or set(current) == set(self._addresses):
            return False
        bound = {server.server_address[0]: server for server in self._fabric_servers}
        for address in sorted(set(bound) - set(current)):
            server = bound.pop(address)
            server.shutdown()
            server.server_close()
            self._fabric_servers.remove(server)
            thread = self._server_threads.pop(id(server), None)
            if thread is not None:
                thread.join(timeout=2)
        for address in sorted(set(current) - set(bound)):
            try:
                server = self._bind_fabric(address)
            except OSError:
                continue  # the address is not usable yet; the next scan retries
            self._fabric_servers.append(server)
            self._serve(server)
        self._addresses = sorted(server.server_address[0] for server in self._fabric_servers)
        if self.use_zeroconf:
            tag = community_tag(self.community.secret, self.community.epoch)
            if self._advertiser is not None:
                self._advertiser.close()
            self._advertiser = Advertiser(
                nid=self.community.nid,
                port=self.actual_fabric_port,
                tag=tag,
                epoch=self.community.epoch,
                adv=self._stable_advert_digest[:12],
                addresses=self._addresses,
            )
            if self._browser is not None:
                self._browser.close()
                self._browser = Browser(tag=tag, addresses=self._addresses)
        self._refresh_advert()
        return True

    def close(self) -> None:
        self._stop.set()
        if self._browser is not None:
            self._browser.close()
            self._browser = None
        if self._advertiser is not None:
            self._advertiser.close()
            self._advertiser = None
        for server in [*self._fabric_servers, self._front_server]:
            if server is not None:
                server.shutdown()
                server.server_close()
        if (
            self._maintenance is not None
            and self._maintenance is not threading.current_thread()
        ):
            self._maintenance.join(timeout=2)
        for thread in self._threads:
            thread.join(timeout=2)
        self._fabric_servers.clear()
        self._front_server = None
        self.community = None

    def _servers(self) -> list[dict]:
        servers = list_local_servers(self.home)
        for value in self.extra_local_servers:
            if isinstance(value, str):
                servers.append(
                    {
                        "url": value,
                        "pid": os.getpid(),
                        "api_key_file": None,
                        "admin_token_file": None,
                    }
                )
            else:
                servers.append(dict(value))
        return servers

    def _refresh_advert(self) -> None:
        assert self.community is not None
        servers = self._servers()
        query = [
            {"url": item["url"], "api_key": _api_key(item.get("api_key_file"))}
            for item in servers
        ]
        models = local_models(query, timeout_s=min(2.0, self.heartbeat_s * 2 + 0.1))
        self._last_local_poll = time.monotonic()
        if not self._runtime_tried:
            self._runtime_tried = True
            try:
                from mlx2.serving import runtime_identity

                self._runtime = runtime_identity()
            except Exception:  # noqa: BLE001 - runtime identity is optional telemetry.
                self._runtime = None
        runtime = self._runtime
        if runtime is not None and "digest" not in runtime:
            runtime = {**runtime, "digest": sha256_hex(canonical_json(runtime))}
            self._runtime = runtime
        doc = build_advert(
            nid=self.community.nid,
            community_id=self.community.community_id,
            epoch=self.community.epoch,
            roles=self.community.roles,
            port=self.actual_fabric_port,
            addresses=self._addresses,
            models=models,
            host=host_facts(),
            runtime=runtime,
            seq=self._seq,
        )
        stable_digest = stable_advert_digest(doc)
        now = time.monotonic()
        with self._lock:
            self._local = list(models)
            stable_changed = stable_digest != self._stable_advert_digest
            if (
                self._signed_advert
                and not stable_changed
                and now - self._last_advert_signed < 10.0
            ):
                return
            self._seq += 1
            doc["seq"] = self._seq
            digest = advert_digest(doc)
            self._signed_advert = sign_advert(
                doc,
                key_path=self.community.node_key,
                cert_path=self.community.node_cert,
            )
            self._advert_digest = digest
            self._stable_advert_digest = stable_digest
            self._last_advert_signed = now
        if stable_changed and self._advertiser is not None:
            self._advertiser.update(adv=stable_digest[:12])

    def _set_candidates(self, candidates: list[Candidate]) -> None:
        assert self.community is not None
        with self._lock:
            for candidate in candidates:
                if candidate.nid != self.community.nid:
                    old = self._peers.get(candidate.nid)
                    if old is None:
                        self._peers[candidate.nid] = _Peer(candidate)
                    else:
                        old.candidate = candidate

    def _maintain(self) -> None:
        while not self._stop.wait(self.heartbeat_s):
            try:
                now = time.monotonic()
                if (
                    self._fabric_servers
                    and now - self._last_address_scan >= self.address_rescan_s
                ):
                    self._last_address_scan = now
                    self._rescan_addresses()
                if now - self._last_local_poll >= self.local_poll_s:
                    self._refresh_advert()
                if self._browser is not None:
                    self._set_candidates(self._browser.candidates())
                for peer in list(self._peers.values()):
                    self._heartbeat(peer)
            except Exception:  # noqa: BLE001, S112 - maintenance retries on its next tick.
                continue

    def _connect(
        self, peer: _Peer, *, operation_timeout_s: float
    ) -> http.client.HTTPSConnection:
        assert self.community is not None
        conn, address = connect_https(
            peer.candidate.addresses,
            peer.candidate.port,
            context=client_context(self.community),
            local_addresses=self._addresses,
            preferred_address=peer.address,
        )
        verdict = check_peer(
            self.community,
            conn.sock.getpeercert(binary_form=True),
            expected_nid=peer.candidate.nid,
        )
        if not verdict.ok:
            conn.close()
            raise FabricError(verdict.reason)
        peer.address = address
        conn.timeout = operation_timeout_s
        if conn.sock is not None:
            conn.sock.settimeout(operation_timeout_s)
        return conn

    def _maintenance_timeout(self) -> float:
        return max(2.0, 3 * self.heartbeat_s)

    def _heartbeat(self, peer: _Peer) -> None:
        assert self.community is not None
        if peer.candidate.nid in revoked_nids(self.community):
            peer.state, peer.reason = "revoked", "revoked nid"
            return
        try:
            conn = self._connect(peer, operation_timeout_s=self._maintenance_timeout())
            headers = {"If-None-Match": peer.last_digest} if peer.last_digest else {}
            conn.request("GET", "/fabric/advert", headers=headers)
            response = conn.getresponse()
            if response.status == 304:
                response.read()
            elif response.status == 200:
                signed = json.loads(response.read())
                doc = verify_advert(
                    signed,
                    ca_cert=self.community.ca_cert,
                    expected_nid=peer.candidate.nid,
                )
                peer.advert = doc
                peer.last_digest = advert_digest(doc)
            else:
                raise FabricError(f"advert HTTP {response.status}")
            conn.close()
            now = time.monotonic()
            peer.state, peer.reason, peer.failures, peer.last_seen = (
                "alive",
                "ok",
                0,
                now,
            )
            peer.first_failure = None
            if not peer.last_log or now - peer.last_log >= 30:
                self._pull_log(peer)
                peer.last_log = now
            self._probe_link_if_due(peer, now)
        except Exception as exc:  # noqa: BLE001 - all transport failures update liveness.
            peer.failures += 1
            peer.reason = str(exc)
            now = time.monotonic()
            if peer.first_failure is None:
                peer.first_failure = now
            lease_start = (
                peer.last_seen if peer.last_seen is not None else peer.first_failure
            )
            if now - lease_start >= 5:
                peer.state = "dead"
            elif peer.failures >= 3:
                peer.state = "suspect"

    def _pull_log(self, peer: _Peer) -> None:
        assert self.community is not None
        conn = self._connect(peer, operation_timeout_s=self._maintenance_timeout())
        try:
            conn.request("GET", "/fabric/log")
            response = conn.getresponse()
            data = json.loads(response.read())
            if response.status != 200:
                raise FabricError(f"log HTTP {response.status}")
            merge_admission_log(self.community, data.get("admission", []))
            merge_revocations(self.community, data.get("revocations", []))
        finally:
            conn.close()

    def _record_link(self, peer: _Peer, receipt: LinkReceipt) -> None:
        with self._lock:
            peer.link = receipt
            peer.link_history.append(receipt)

    def _probe_link_if_due(self, peer: _Peer, now: float) -> None:
        if not peer.last_full_probe or now - peer.last_full_probe >= 300:
            receipt = self._probe(peer, bandwidth=True)
            peer.last_full_probe = now
            peer.last_rtt_probe = now
            self._record_link(peer, receipt)
        elif not peer.last_rtt_probe or now - peer.last_rtt_probe >= 30:
            receipt = self._probe(peer, bandwidth=False)
            peer.last_rtt_probe = now
            self._record_link(peer, receipt)

    def _probe(self, peer: _Peer, *, bandwidth: bool = True) -> LinkReceipt:
        conn = self._connect(peer, operation_timeout_s=self._maintenance_timeout())

        def roundtrip(upload: bytes, download_len: int) -> bytes:
            conn.request(
                "POST",
                "/fabric/echo",
                body=upload,
                headers={
                    "X-Mlx2-Download-Len": str(download_len),
                    "Content-Length": str(len(upload)),
                },
            )
            response = conn.getresponse()
            body = response.read()
            if response.status != 200:
                raise FabricError(f"echo HTTP {response.status}")
            return body

        try:
            receipt = probe_link(
                roundtrip,
                peer_nid=peer.candidate.nid,
                payload_bytes=(8 << 20) if bandwidth else 0,
            )
        finally:
            conn.close()
        _atomic_json(
            self.home / "receipts" / f"link-{peer.candidate.nid}-{time.time_ns()}.json",
            receipt.as_dict(),
        )
        return receipt

    def _peer_view(self, peer: _Peer) -> PeerView:
        advert = peer.advert or {}
        host = advert.get("host") or {}
        runtime = advert.get("runtime") or {}
        return PeerView(
            peer.candidate.nid,
            peer.state,
            {
                item["id"]
                for item in advert.get("models", [])
                if item.get("loaded") and isinstance(item.get("id"), str)
            },
            advert.get("queue_depth"),
            host.get("free_bytes"),
            host.get("device_class"),
            runtime.get("digest"),
            peer.link,
            host.get("hostname"),
            tuple(peer.link_history),
            tuple(item for item in advert.get("models", []) if isinstance(item, dict)),
        )

    def _local_map(self) -> dict[str, dict]:
        with self._lock:
            return {item["id"]: item for item in self._local if item.get("loaded")}

    def _forward_local(
        self,
        path: str,
        body: bytes,
        model: str,
        *,
        admin: bool = False,
        extra_headers: dict[str, str] | None = None,
        content_type: str = "application/json",
    ) -> tuple[int, dict[str, str], bytes]:
        conn, response = self._open_local(
            path,
            body,
            model,
            admin=admin,
            extra_headers=extra_headers,
            content_type=content_type,
        )
        try:
            return response.status, dict(response.headers), response.read()
        finally:
            conn.close()

    def _open_local(
        self,
        path: str,
        body: bytes,
        model: str,
        *,
        admin: bool = False,
        extra_headers: dict[str, str] | None = None,
        content_type: str = "application/json",
    ) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
        item = self._local_map().get(model)
        if item is None:
            raise KeyError(model)
        parsed = urlsplit(item["server_url"])
        connection_type = (
            http.client.HTTPSConnection
            if parsed.scheme == "https"
            else http.client.HTTPConnection
        )
        conn = connection_type(
            parsed.hostname, parsed.port, timeout=self.request_timeout_s
        )
        server = next(
            (
                entry
                for entry in self._servers()
                if entry["url"].rstrip("/") == item["server_url"].rstrip("/")
            ),
            None,
        )
        token_file = "admin_token_file" if admin else "api_key_file"
        key = _api_key(server.get(token_file)) if server else None
        if admin and not key:
            conn.close()
            raise FabricError("local_admin_token_unavailable")
        headers = {"Content-Type": content_type, "Content-Length": str(len(body))}
        headers.update(extra_headers or {})
        if key:
            headers["Authorization"] = f"Bearer {key}"
        conn.request("POST", path, body=body, headers=headers)
        return conn, conn.getresponse()

    def _serve_receipt(
        self,
        *,
        request_id: str,
        origin_nid: str,
        model: str,
        server_url: str,
        status: int,
        elapsed_ms: float,
        blob_sha256: str | None = None,
    ) -> str:
        assert self.community is not None
        receipt = {
            "schema": "mlx2.fabric.serve.v1",
            "request_id": request_id,
            "origin_nid": origin_nid,
            "served_by": self.community.nid,
            "model": model,
            "local_server_url": server_url,
            "runtime_digest": (self._runtime or {}).get("digest"),
            "status": status,
            "elapsed_ms": elapsed_ms,
        }
        if blob_sha256 is not None:
            receipt["blob_sha256"] = blob_sha256
        signed = {
            "receipt": receipt,
            "signature": sign(self.community.node_key, canonical_json(receipt)),
            "cert_pem": self.community.node_cert.read_text(encoding="utf-8"),
        }
        return _b64(canonical_json(signed))

    def _verify_serve(self, value: str, *, peer: _Peer, request_id: str) -> dict:
        assert self.community is not None
        signed = json.loads(_unb64(value))
        receipt = signed["receipt"]
        if (
            receipt.get("served_by") != peer.candidate.nid
            or receipt.get("request_id") != request_id
        ):
            raise FabricError("serve receipt identity mismatch")
        verdict_nid = check_peer(
            self.community,
            ssl.PEM_cert_to_DER_cert(signed["cert_pem"]),
            expected_nid=peer.candidate.nid,
        )
        if not verdict_nid.ok or not verify(
            signed["cert_pem"],
            canonical_json(receipt),
            signed["signature"],
            ca_cert=self.community.ca_cert,
        ):
            raise FabricError("invalid serve receipt signature")
        return receipt

    @staticmethod
    def _request_opt_in(body: bytes, header_value: str | None) -> tuple[bytes, bool]:
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return body, header_value == "prefill-handoff-approx-v1"
        if not isinstance(payload, dict):
            return body, header_value == "prefill-handoff-approx-v1"
        fabric = payload.pop("mlx2_fabric", None)
        body_opt_in = (
            isinstance(fabric, dict) and fabric.get("allow_approximate_prefill") is True
        )
        return canonical_json(payload), bool(
            body_opt_in or header_value == "prefill-handoff-approx-v1"
        )

    def _local_view(self) -> PeerView:
        assert self.community is not None
        own = self._signed_advert.get("advert", {})
        host = own.get("host") or {}
        runtime = own.get("runtime") or {}
        return PeerView(
            self.community.nid,
            "alive",
            set(self._local_map()),
            0,
            host.get("free_bytes"),
            host.get("device_class"),
            runtime.get("digest"),
            None,
            host.get("hostname"),
            (),
            tuple(self._local_map().values()),
        )

    def _plan_handoff(
        self, body: bytes, model: str, *, producer_runtime_digest: str
    ) -> dict[str, Any]:
        try:
            request = json.loads(body)
        except json.JSONDecodeError as exc:
            raise FabricError("malformed_request") from exc
        if not isinstance(request, dict):
            raise FabricError("malformed_request")
        request["producer_runtime_digest"] = producer_runtime_digest
        status, _, output = self._forward_local(
            "/v1/admin/prefill-handoff/plan",
            canonical_json(request),
            model,
            admin=True,
        )
        if status != 200:
            raise FabricError(_response_reason(status, output))
        try:
            value = json.loads(output)
        except json.JSONDecodeError as exc:
            raise FabricError("malformed_plan_response") from exc
        if not isinstance(value, dict) or not isinstance(value.get("plan"), dict):
            raise FabricError("malformed_plan_response")
        return value

    def _prepare_handoff(
        self,
        *,
        requested: bool,
        model: str,
        body: bytes,
        peers: list[PeerView],
    ) -> tuple[dict[str, Any], PeerView | None, dict | None, dict | None, str | None]:
        local_model = self._local_map().get(model)
        if not requested:
            candidate, producer, record = prefill_candidate(
                requested=False,
                model=model,
                local_model=local_model,
                peers=peers,
                link_budget=self.link_budget,
                plan_response=None,
            )
            return candidate, producer, record, None, None
        plan_response = None
        plan_failure = None
        local_cap = (
            local_model.get("prefill_handoff")
            if isinstance(local_model, dict)
            else None
        )
        has_possible_exporter = any(
            peer.state == "alive"
            and any(
                item.get("id") == model
                and isinstance(item.get("prefill_handoff"), dict)
                and item["prefill_handoff"].get("export") is True
                for item in peer.model_entries
            )
            for peer in peers
        )
        if (
            requested
            and isinstance(local_cap, dict)
            and local_cap.get("import") is True
            and has_possible_exporter
        ):
            try:
                _, producer, _ = prefill_candidate(
                    requested=True,
                    model=model,
                    local_model=local_model,
                    peers=peers,
                    link_budget=self.link_budget,
                    plan_response=None,
                )
                if producer is not None:
                    entry = next(
                        item
                        for item in producer.model_entries
                        if item.get("id") == model and item.get("loaded") is True
                    )
                    producer_digest = entry["prefill_handoff"]["runtime_digest"]
                    plan_response = self._plan_handoff(
                        body,
                        model,
                        producer_runtime_digest=producer_digest,
                    )
            except Exception as exc:  # noqa: BLE001 - becomes an explicit fallback.
                plan_failure = str(exc) or type(exc).__name__
        candidate, producer, record = prefill_candidate(
            requested=requested,
            model=model,
            local_model=local_model,
            peers=peers,
            link_budget=self.link_budget,
            plan_response=plan_response,
        )
        if plan_failure is not None and candidate.get("reason") == "plan_unavailable":
            candidate["reason"] = f"plan_failed: {plan_failure}"
        return candidate, producer, record, plan_response, plan_failure

    def _handoff_placement(
        self,
        *,
        request_id: str,
        model: str,
        producer: PeerView,
        record: dict,
        candidate: dict[str, Any],
        plan_response: dict,
    ) -> Placement:
        assert self.community is not None
        return Placement(
            request_id=request_id,
            model=model,
            kind="approximate_prefill_offload",
            origin_nid=self.community.nid,
            served_by=self._local_view(),
            reason=candidate["reason"],
            candidates=[candidate],
            selected=True,
            exactness="approximate_cross_host_prefill",
            approximate=True,
            qualified=record.get("record_state") == "final",
            prefill_handoff={
                "producer": producer.nid,
                "record_id": record.get("record_id"),
                "record_state": record.get("record_state"),
                "plan_digest": plan_response.get("plan_digest"),
                "bytes": None,
                "export_ms": None,
                "import_ms": None,
            },
        )

    def _export_from_peer(
        self,
        *,
        peer: _Peer,
        request_id: str,
        model: str,
        plan_response: dict,
    ) -> bytes:
        payload = {
            "model": model,
            "token_ids": plan_response["token_ids"],
            "plan": plan_response["plan"],
            "plan_digest": plan_response["plan_digest"],
            "expected": {
                key: plan_response[key]
                for key in (
                    "artifact_fingerprint",
                    "tokenizer_digest",
                    "layout_fingerprint",
                )
            },
        }
        conn = self._connect(peer, operation_timeout_s=self.request_timeout_s)
        try:
            encoded = canonical_json(payload)
            conn.request(
                "POST",
                "/fabric/v1/prefill-handoff/export",
                body=encoded,
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(len(encoded)),
                    "X-Mlx2-Request-Id": request_id,
                    "X-Mlx2-Fabric-Hop": "1",
                },
            )
            response = conn.getresponse()
            output = response.read()
            if response.status != 200:
                raise FabricError(_response_reason(response.status, output))
            serve = response.getheader("X-Mlx2-Serve-Receipt")
            if not serve:
                raise FabricError("missing_serve_receipt")
            receipt = self._verify_serve(serve, peer=peer, request_id=request_id)
            if receipt.get("blob_sha256") != sha256_hex(output):
                raise FabricError("blob_digest_mismatch")
            return output
        finally:
            conn.close()

    def _import_local(
        self, *, blob: bytes, model: str, plan_response: dict
    ) -> dict[str, Any]:
        expected = {
            key: plan_response[key]
            for key in (
                "artifact_fingerprint",
                "tokenizer_digest",
                "layout_fingerprint",
                "token_digest",
                "plan_digest",
            )
        }
        status, _, output = self._forward_local(
            "/v1/admin/prefill-handoff/import",
            blob,
            model,
            admin=True,
            content_type="application/octet-stream",
            extra_headers={"X-Mlx2-Handoff-Expected": _b64(canonical_json(expected))},
        )
        if status != 200:
            raise FabricError(_response_reason(status, output))
        try:
            value = json.loads(output)
        except json.JSONDecodeError as exc:
            raise FabricError("malformed_import_response") from exc
        if not isinstance(value, dict) or not isinstance(value.get("handle"), str):
            raise FabricError("malformed_import_response")
        return value

    def _forward_handoff_export(self, body: bytes) -> tuple[str, bytes, dict[str, str]]:
        try:
            payload = json.loads(body)
            model = payload.pop("model")
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise FabricError("malformed") from exc
        if not isinstance(model, str):
            raise FabricError("malformed")
        local_model = self._local_map().get(model)
        capability = (
            local_model.get("prefill_handoff")
            if isinstance(local_model, dict)
            else None
        )
        if local_model is None:
            raise FabricError("model_not_resident_locally")
        if not isinstance(capability, dict) or capability.get("export") is not True:
            raise FabricError("local_export_unavailable")
        status, headers, output = self._forward_local(
            "/v1/admin/prefill-handoff/export",
            canonical_json(payload),
            model,
            admin=True,
        )
        if status != 200:
            raise FabricError(_response_reason(status, output))
        return model, output, headers

    def _route(
        self, path: str, body: bytes, *, allow_approximate_prefill: bool = False
    ) -> tuple[int, dict[str, str], bytes, Placement]:
        assert self.community is not None
        request_id = uuid.uuid4().hex
        try:
            payload = json.loads(body)
            model = payload["model"]
            if not isinstance(model, str):
                raise TypeError
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            model = ""
        with self._lock:
            peers = [self._peer_view(peer) for peer in self._peers.values()]
        candidate, producer, record, plan_response, plan_failure = (
            self._prepare_handoff(
                requested=allow_approximate_prefill,
                model=model,
                body=body,
                peers=peers,
            )
        )

        if producer is not None and record is not None and plan_response is not None:
            placement = self._handoff_placement(
                request_id=request_id,
                model=model,
                producer=producer,
                record=record,
                candidate=candidate,
                plan_response=plan_response,
            )
            failed_step = None
            try:
                started = time.perf_counter()
                blob = self._export_from_peer(
                    peer=self._peers[producer.nid],
                    request_id=request_id,
                    model=model,
                    plan_response=plan_response,
                )
                placement.prefill_handoff["bytes"] = len(blob)
                placement.prefill_handoff["export_ms"] = (
                    time.perf_counter() - started
                ) * 1000
                failed_step = "import"
                started = time.perf_counter()
                imported = self._import_local(
                    blob=blob, model=model, plan_response=plan_response
                )
                placement.prefill_handoff["import_ms"] = (
                    time.perf_counter() - started
                ) * 1000
            except Exception as exc:  # noqa: BLE001 - pre-decode failure falls back.
                if failed_step is None:
                    failed_step = "export"
                reason = str(exc) or type(exc).__name__
                candidate["verdict"] = "fallback"
                candidate["reason"] = reason
                placement = Placement(
                    request_id=request_id,
                    model=model,
                    kind="local",
                    origin_nid=self.community.nid,
                    served_by=self._local_view(),
                    reason=reason,
                    candidates=[candidate],
                    selected=False,
                    observed_used=False,
                    approximate=False,
                    qualified=False,
                    fallback="ordinary_local",
                    failed_step=failed_step,
                )
                try:
                    status, headers, response_body = self._forward_local(
                        path, body, model
                    )
                except Exception as local_exc:  # noqa: BLE001 - explicit local failure.
                    placement.outcome = "ordinary_local_failed"
                    placement.reason = f"{placement.reason}; local: {local_exc}"
                    return (
                        502,
                        {"Content-Type": "application/json"},
                        canonical_json(
                            {
                                "error": {
                                    "message": placement.reason,
                                    "type": "fabric_local_failure",
                                },
                                "mlx2_fabric": placement.receipt(),
                            }
                        ),
                        placement,
                    )
                if status < 400:
                    placement.selected = True
                    placement.observed_used = True
                else:
                    placement.outcome = "ordinary_local_failed"
                return self._attach_receipt(status, headers, response_body, placement)

            # Decode is the committed route. Any failure here is returned as-is;
            # retrying would duplicate a request after the single-use handle.
            failed_step = "decode"
            chat = json.loads(body)
            chat["prefill_handoff"] = imported["handle"]
            try:
                status, headers, response_body = self._forward_local(
                    path, canonical_json(chat), model, admin=True
                )
            except Exception as exc:  # noqa: BLE001 - no retry after decode starts.
                placement.failed_step = failed_step
                placement.reason = str(exc) or type(exc).__name__
                placement.outcome = "decode_failed"
                return (
                    502,
                    {"Content-Type": "application/json"},
                    canonical_json(
                        {
                            "error": {
                                "message": placement.reason,
                                "type": "fabric_decode_failure",
                            },
                            "mlx2_fabric": placement.receipt(),
                        }
                    ),
                    placement,
                )
            placement.observed_used = _handoff_observed(response_body)
            if status >= 400:
                placement.failed_step = "decode"
                placement.outcome = "decode_failed"
                placement.reason = _response_reason(status, response_body)
            return self._attach_receipt(status, headers, response_body, placement)

        placement = place(
            request_id=request_id,
            model=model,
            local_nid=self.community.nid,
            local_models=set(self._local_map()),
            peers=peers,
            link_budget=self.link_budget,
            approximate_candidate=candidate,
        )
        if placement.kind == "local" and placement.served_by is not None:
            placement.served_by = self._local_view()
            if plan_failure is not None and candidate.get("reason", "").startswith(
                "plan_failed:"
            ):
                candidate["verdict"] = "fallback"
                placement.fallback = "ordinary_local"
                placement.failed_step = "plan"
                placement.reason = candidate["reason"]
                placement.selected = False
                placement.observed_used = False
        if placement.kind == "refused":
            return (
                404,
                {"Content-Type": "application/json"},
                canonical_json(
                    {
                        "error": {
                            "message": placement.reason,
                            "type": "fabric_refusal",
                        },
                        "mlx2_fabric": placement.receipt(),
                    }
                ),
                placement,
            )
        try:
            if placement.kind == "local":
                status, headers, response_body = self._forward_local(path, body, model)
                if status < 400:
                    placement.selected = True
                    placement.observed_used = True
                elif placement.fallback is not None:
                    placement.outcome = "ordinary_local_failed"
            else:
                peer = self._peers[placement.served_by.nid]
                conn = self._connect(peer, operation_timeout_s=self.request_timeout_s)
                try:
                    conn.request(
                        "POST",
                        "/fabric" + path,
                        body=body,
                        headers={
                            "Content-Type": "application/json",
                            "Content-Length": str(len(body)),
                            "X-Mlx2-Request-Id": request_id,
                        },
                    )
                    response = conn.getresponse()
                    status, headers, response_body = (
                        response.status,
                        dict(response.headers),
                        response.read(),
                    )
                    serve = response.getheader("X-Mlx2-Serve-Receipt")
                    if not serve:
                        raise FabricError("missing serve receipt")
                    self._verify_serve(serve, peer=peer, request_id=request_id)
                    placement.observed_used = True
                finally:
                    conn.close()
        except Exception as exc:  # noqa: BLE001 - a routed request must fail closed.
            placement.outcome = "aborted_peer_lost"
            placement.reason = f"aborted_peer_lost: {exc}"
            return (
                502,
                {"Content-Type": "application/json"},
                canonical_json(
                    {
                        "error": {
                            "message": placement.reason,
                            "type": "fabric_peer_lost",
                        },
                        "mlx2_fabric": placement.receipt(),
                    }
                ),
                placement,
            )
        return self._attach_receipt(status, headers, response_body, placement)

    @staticmethod
    def _attach_receipt(
        status: int,
        headers: dict[str, str],
        response_body: bytes,
        placement: Placement,
    ) -> tuple[int, dict[str, str], bytes, Placement]:
        content_type = next(
            (value for key, value in headers.items() if key.lower() == "content-type"),
            "",
        )
        if content_type.startswith("application/json"):
            try:
                value = json.loads(response_body)
                if isinstance(value, dict):
                    value["mlx2_fabric"] = placement.receipt()
                    response_body = canonical_json(value)
            except json.JSONDecodeError:
                pass
        return status, headers, response_body, placement

    def _append_route(self, receipt: dict) -> None:
        path = self.home / "receipts" / "routes.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, canonical_json(receipt) + b"\n")
        finally:
            os.close(fd)

    @staticmethod
    def _write_chunk(handler: BaseHTTPRequestHandler, body: bytes) -> None:
        if body:
            handler.wfile.write(f"{len(body):X}\r\n".encode() + body + b"\r\n")
            handler.wfile.flush()

    def _stream_route(
        self, handler: BaseHTTPRequestHandler, path: str, body: bytes
    ) -> None:
        assert self.community is not None
        request_id = uuid.uuid4().hex
        try:
            model = json.loads(body)["model"]
            if not isinstance(model, str):
                raise TypeError
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            model = ""
        with self._lock:
            peers = [self._peer_view(peer) for peer in self._peers.values()]
        placement = place(
            request_id=request_id,
            model=model,
            local_nid=self.community.nid,
            local_models=set(self._local_map()),
            peers=peers,
            link_budget=self.link_budget,
            approximate_candidate={
                "option": "approximate_prefill_offload",
                "verdict": "not_requested",
                "reason": "not_requested",
            },
        )
        if placement.kind == "refused":
            receipt = placement.receipt()
            self._append_route(receipt)
            self._json_response(
                handler,
                404,
                {
                    "error": {"message": placement.reason, "type": "fabric_refusal"},
                    "mlx2_fabric": receipt,
                },
                {"X-Mlx2-Fabric-Route": _b64(canonical_json(receipt))},
            )
            return
        conn = None
        try:
            if placement.kind == "local":
                conn, response = self._open_local(path, body, model)
                placement.observed_used = True
            else:
                peer = self._peers[placement.served_by.nid]
                conn = self._connect(peer, operation_timeout_s=self.request_timeout_s)
                conn.request(
                    "POST",
                    "/fabric" + path,
                    body=body,
                    headers={
                        "Content-Type": "application/json",
                        "Content-Length": str(len(body)),
                        "X-Mlx2-Request-Id": request_id,
                    },
                )
                response = conn.getresponse()
                serve = response.getheader("X-Mlx2-Serve-Receipt")
                if not serve:
                    raise FabricError("missing serve receipt")
                self._verify_serve(serve, peer=peer, request_id=request_id)
                placement.observed_used = True
        except Exception as exc:  # noqa: BLE001 - no alternate route after selection.
            if conn is not None:
                conn.close()
            placement.outcome = "aborted_peer_lost"
            placement.reason = f"aborted_peer_lost: {exc}"
            receipt = placement.receipt()
            self._append_route(receipt)
            self._json_response(
                handler,
                502,
                {
                    "error": {"message": placement.reason, "type": "fabric_peer_lost"},
                    "mlx2_fabric": receipt,
                },
                {"X-Mlx2-Fabric-Route": _b64(canonical_json(receipt))},
            )
            return
        receipt = placement.receipt()
        self._append_route(receipt)
        handler.send_response(response.status)
        handler.send_header(
            "Content-Type", response.getheader("Content-Type", "text/event-stream")
        )
        handler.send_header("Transfer-Encoding", "chunked")
        handler.send_header("X-Mlx2-Fabric-Route", _b64(canonical_json(receipt)))
        handler.end_headers()
        try:
            while chunk := response.read(64 << 10):
                self._write_chunk(handler, chunk)
            handler.wfile.write(b"0\r\n\r\n")
            handler.wfile.flush()
        finally:
            conn.close()

    def _json_response(
        self,
        handler: BaseHTTPRequestHandler,
        status: int,
        value: Any,
        headers: dict[str, str] | None = None,
    ) -> None:
        body = canonical_json(value)
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        for key, item in (headers or {}).items():
            handler.send_header(key, item)
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    def _authorize(self, handler: BaseHTTPRequestHandler):
        assert self.community is not None
        cert = handler.connection.getpeercert(binary_form=True)
        return check_peer(self.community, cert)

    def _fabric_handler(self):
        node = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _peer_verdict(self):
                verdict = getattr(self, "_cached_peer_verdict", None)
                if verdict is None:
                    verdict = node._authorize(self)
                    self._cached_peer_verdict = verdict
                return verdict

            def do_GET(self):
                verdict = self._peer_verdict()
                if not verdict.ok:
                    node._json_response(self, 403, {"error": verdict.reason})
                    return
                if self.path == "/fabric/advert":
                    with node._lock:
                        digest, signed = node._advert_digest, node._signed_advert
                    if self.headers.get("If-None-Match") == digest:
                        self.send_response(304)
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    node._json_response(self, 200, signed, {"ETag": digest})
                elif self.path == "/fabric/log":
                    node._json_response(
                        self,
                        200,
                        {
                            "admission": admission_entries(node.community),
                            "revocations": revocation_entries(node.community),
                        },
                    )
                else:
                    node._json_response(self, 404, {"error": "not found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                if self.path == "/fabric/join":
                    node._join(self, body)
                    return
                verdict = self._peer_verdict()
                if not verdict.ok:
                    node._json_response(self, 403, {"error": verdict.reason})
                    return
                if self.path == "/fabric/echo":
                    try:
                        output = echo_response(
                            body, int(self.headers.get("X-Mlx2-Download-Len", "0"))
                        )
                    except (ValueError, FabricError) as exc:
                        node._json_response(self, 400, {"error": str(exc)})
                        return
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(output)))
                    self.end_headers()
                    self.wfile.write(output)
                    return
                if self.path == "/fabric/v1/prefill-handoff/export":
                    model = ""
                    started = time.perf_counter()
                    try:
                        if self.headers.get("X-Mlx2-Fabric-Hop") != "1":
                            raise FabricError("invalid_hop_limit")
                        model, output, headers = node._forward_handoff_export(body)
                        status = 200
                        server_url = node._local_map()[model]["server_url"]
                        receipt = node._serve_receipt(
                            request_id=self.headers.get("X-Mlx2-Request-Id", ""),
                            origin_nid=verdict.nid or "",
                            model=model,
                            server_url=server_url,
                            status=status,
                            elapsed_ms=(time.perf_counter() - started) * 1000,
                            blob_sha256=sha256_hex(output),
                        )
                    except (
                        FabricError,
                        KeyError,
                        TypeError,
                        json.JSONDecodeError,
                    ) as exc:
                        node._json_response(
                            self,
                            409,
                            {
                                "error": {
                                    "reason": str(exc) or type(exc).__name__,
                                    "type": "prefill_handoff_refusal",
                                }
                            },
                        )
                        return
                    self.send_response(200)
                    self.send_header(
                        "Content-Type",
                        headers.get("Content-Type", "application/octet-stream"),
                    )
                    self.send_header("X-Mlx2-Serve-Receipt", receipt)
                    self.send_header("Content-Length", str(len(output)))
                    self.end_headers()
                    self.wfile.write(output)
                    return
                if self.path in {
                    "/fabric/v1/chat/completions",
                    "/fabric/v1/completions",
                }:
                    model = ""
                    try:
                        payload = json.loads(body)
                        model = payload["model"]
                        started = time.perf_counter()
                        if payload.get("stream") is True:
                            local_conn, local_response = node._open_local(
                                self.path.removeprefix("/fabric"), body, model
                            )
                            if node._stop.is_set():
                                local_conn.close()
                                self.close_connection = True
                                return
                            server_url = node._local_map()[model]["server_url"]
                            receipt = node._serve_receipt(
                                request_id=self.headers.get("X-Mlx2-Request-Id", ""),
                                origin_nid=verdict.nid or "",
                                model=model,
                                server_url=server_url,
                                status=local_response.status,
                                elapsed_ms=(time.perf_counter() - started) * 1000,
                            )
                            self.send_response(local_response.status)
                            self.send_header(
                                "Content-Type",
                                local_response.getheader(
                                    "Content-Type", "text/event-stream"
                                ),
                            )
                            self.send_header("X-Mlx2-Serve-Receipt", receipt)
                            self.send_header("Transfer-Encoding", "chunked")
                            self.end_headers()
                            try:
                                while chunk := local_response.read(64 << 10):
                                    node._write_chunk(self, chunk)
                                self.wfile.write(b"0\r\n\r\n")
                                self.wfile.flush()
                            finally:
                                local_conn.close()
                            return
                        status, headers, output = node._forward_local(
                            self.path.removeprefix("/fabric"), body, model
                        )
                        if node._stop.is_set():
                            self.close_connection = True
                            try:
                                self.connection.shutdown(socket.SHUT_RDWR)
                            except OSError:
                                pass
                            self.connection.close()
                            return
                        server_url = node._local_map()[model]["server_url"]
                        receipt = node._serve_receipt(
                            request_id=self.headers.get("X-Mlx2-Request-Id", ""),
                            origin_nid=verdict.nid or "",
                            model=model,
                            server_url=server_url,
                            status=status,
                            elapsed_ms=(time.perf_counter() - started) * 1000,
                        )
                    except (KeyError, ValueError, json.JSONDecodeError):
                        refusal = place(
                            request_id=self.headers.get("X-Mlx2-Request-Id", ""),
                            model=model if isinstance(model, str) else "",
                            local_nid=node.community.nid,
                            local_models=set(node._local_map()),
                            peers=[],
                            link_budget=node.link_budget,
                        )
                        refusal.reason = "model not resident locally"
                        node._json_response(
                            self,
                            404,
                            {
                                "error": "model not resident",
                                "mlx2_fabric": refusal.receipt(),
                            },
                        )
                        return
                    self.send_response(status)
                    self.send_header(
                        "Content-Type", headers.get("Content-Type", "application/json")
                    )
                    self.send_header("X-Mlx2-Serve-Receipt", receipt)
                    self.send_header("Content-Length", str(len(output)))
                    self.end_headers()
                    self.wfile.write(output)
                    return
                node._json_response(self, 404, {"error": "not found"})

            def log_message(self, _format, *_args):
                pass

        return Handler

    def _join(self, handler: BaseHTTPRequestHandler, body: bytes) -> None:
        assert self.community is not None
        remote = handler.client_address[0]
        now = time.monotonic()
        failures = [
            stamp for stamp in self._join_failures.get(remote, []) if now - stamp < 60
        ]
        self._join_failures[remote] = failures
        if len(failures) >= 5:
            self._json_response(handler, 429, {"error": "too many failed join proofs"})
            return
        try:
            value = json.loads(body)
            csr, proof = value["csr_pem"], value["proof"]
            der = ssl.PEM_cert_to_DER_cert(self.community.node_cert.read_text())
            digest = hashlib.sha256(der).hexdigest()
            import hmac

            if not hmac.compare_digest(
                proof, join_proof(self.community.secret, csr, digest)
            ):
                raise FabricError("invalid join proof")
            bundle = admit(self.community, csr, hosts=())
            self._json_response(
                handler,
                200,
                {
                    "bundle": bundle,
                    "ack": join_ack_proof(self.community.secret, csr, digest),
                },
            )
        except Exception as exc:  # noqa: BLE001 - malformed CSRs/proofs are join refusals.
            failures.append(now)
            self._json_response(handler, 403, {"error": str(exc)})

    def _front_handler(self):
        node = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):
                if self.path == "/v1/models":
                    entries = [
                        {
                            "id": model,
                            "object": "model",
                            "mlx2_fabric": {
                                "nid": node.community.nid,
                                "hostname": socket.gethostname(),
                                "where": "local",
                            },
                        }
                        for model in sorted(node._local_map())
                    ]
                    with node._lock:
                        for peer in node._peers.values():
                            if peer.state == "alive":
                                view = node._peer_view(peer)
                                entries.extend(
                                    {
                                        "id": model,
                                        "object": "model",
                                        "mlx2_fabric": {
                                            "nid": view.nid,
                                            "hostname": view.hostname,
                                            "where": "remote",
                                        },
                                    }
                                    for model in sorted(view.models)
                                )
                    node._json_response(self, 200, {"object": "list", "data": entries})
                    return
                if self.path == "/v1/fabric/nodes":
                    node._json_response(self, 200, {"data": node._nodes()})
                    return
                if self.path == "/v1/fabric/links":
                    with node._lock:
                        links = [
                            {
                                "peer_nid": peer.candidate.nid,
                                "latest": peer.link.as_dict(),
                                "history": [
                                    item.as_dict() for item in peer.link_history
                                ],
                            }
                            for peer in node._peers.values()
                            if peer.link
                        ]
                    node._json_response(self, 200, {"data": links})
                    return
                node._json_response(self, 404, {"error": "not found"})

            def do_POST(self):
                if self.path not in {"/v1/chat/completions", "/v1/completions"}:
                    node._json_response(self, 404, {"error": "not found"})
                    return
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                body, allow_approximate = node._request_opt_in(
                    body,
                    self.headers.get("X-Mlx2-Allow-Approximate-Prefill"),
                )
                try:
                    streaming = json.loads(body).get("stream") is True
                except (AttributeError, json.JSONDecodeError):
                    streaming = False
                if streaming and not allow_approximate:
                    node._stream_route(self, self.path, body)
                    return
                status, headers, output, placement = node._route(
                    self.path,
                    body,
                    allow_approximate_prefill=allow_approximate,
                )
                receipt = placement.receipt()
                node._append_route(receipt)
                self.send_response(status)
                self.send_header(
                    "Content-Type", headers.get("Content-Type", "application/json")
                )
                self.send_header("X-Mlx2-Fabric-Route", _b64(canonical_json(receipt)))
                self.send_header("Content-Length", str(len(output)))
                self.end_headers()
                self.wfile.write(output)

            def log_message(self, _format, *_args):
                pass

        return Handler

    def _nodes(self) -> list[dict]:
        assert self.community is not None
        own = self._signed_advert.get("advert", {})
        host = own.get("host") or {}
        result = [
            {
                "nid": self.community.nid,
                "state": "alive",
                "hostname": host.get("hostname"),
                "device_class": host.get("device_class"),
                "memory_bytes": host.get("memory_bytes"),
                "free_bytes": host.get("free_bytes"),
                "swapouts": host.get("swapouts"),
                "runtime_digest": (own.get("runtime") or {}).get("digest"),
                "models": sorted(self._local_map()),
                "last_seen_age_s": 0.0,
                "link": None,
                "verdict": "self",
            }
        ]
        with self._lock:
            for peer in self._peers.values():
                view = self._peer_view(peer)
                host = (peer.advert or {}).get("host") or {}
                result.append(
                    {
                        "nid": view.nid,
                        "state": view.state,
                        "hostname": view.hostname,
                        "device_class": view.device_class,
                        "memory_bytes": host.get("memory_bytes"),
                        "free_bytes": view.free_bytes,
                        "swapouts": host.get("swapouts"),
                        "runtime_digest": view.runtime_digest,
                        "models": sorted(view.models),
                        "last_seen_age_s": time.monotonic() - peer.last_seen
                        if peer.last_seen is not None
                        else None,
                        "link": view.link.as_dict() if view.link else None,
                        "verdict": peer.reason,
                    }
                )
        return result

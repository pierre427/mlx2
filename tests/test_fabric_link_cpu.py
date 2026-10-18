from __future__ import annotations

import copy
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from mlx2.fabric import FabricError
from mlx2.fabric.capability import (
    ADVERT_SCHEMA,
    advert_digest,
    build_advert,
    host_facts,
    local_models,
    sign_advert,
    stable_advert_digest,
    verify_advert,
)
from mlx2.fabric.community import create_community
from mlx2.fabric.link import LinkBudget, LinkReceipt, echo_response, probe_link


def test_host_facts_has_documented_keys_and_never_raises(monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        raise OSError("no host tools")

    monkeypatch.setattr("mlx2.fabric.capability.subprocess.run", fail)
    facts = host_facts()
    assert set(facts) == {
        "chip",
        "memory_bytes",
        "free_bytes",
        "swapouts",
        "hostname",
        "device_class",
    }
    assert facts["chip"] is None
    assert facts["memory_bytes"] is None


class _ModelsHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path != "/v1/models":
            self.send_error(404)
            return
        body = json.dumps(
            {
                "data": [
                    {
                        "id": "resident-model",
                        "loaded": True,
                        "qualification": "unqualified",
                        "capabilities": {"route": True},
                    },
                    {"id": "not-loaded", "loaded": False},
                ]
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args) -> None:
        pass


def test_local_models_refuses_remote_and_reads_loopback_server() -> None:
    with pytest.raises(FabricError, match="loopback"):
        local_models([{"url": "http://192.168.1.2:8285", "api_key": None}])
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ModelsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        assert local_models([{"url": url, "api_key": "local-secret"}]) == [
            {
                "id": "resident-model",
                "server_url": url,
                "loaded": True,
                "qualification": "unqualified",
                "capabilities": {"route": True},
            }
        ]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _advert(nid: str = "0123456789abcdef") -> dict:
    return build_advert(
        nid=nid,
        community_id="community-id",
        epoch=1,
        roles=("serve",),
        port=8443,
        addresses=("127.0.0.1",),
        models=[
            {
                "id": "model-a",
                "server_url": "http://127.0.0.1:8285",
                "loaded": True,
                "qualification": "unqualified",
                "capabilities": {"route": True},
                "tenant": "must-not-leak",
            }
        ],
        host={
            "chip": "Apple M5 Max",
            "memory_bytes": 128,
            "free_bytes": 64,
            "swapouts": 0,
            "hostname": "node",
            "device_class": "Apple_M5_Max",
            "secret": "must-not-leak",
        },
        runtime={"identity": "runtime", "tenant_id": "must-not-leak"},
        seq=4,
        queue_depth=2,
    )


def test_advert_is_public_and_digest_is_canonical() -> None:
    advert = _advert()
    assert advert["schema"] == ADVERT_SCHEMA
    encoded = json.dumps(advert).lower()
    assert "secret" not in encoded
    assert "tenant" not in encoded
    assert advert_digest(advert) == advert_digest(dict(reversed(list(advert.items()))))


def test_stable_advert_digest_ignores_live_fields_but_tracks_capabilities() -> None:
    advert = _advert()
    baseline = stable_advert_digest(advert)
    live = copy.deepcopy(advert)
    live["seq"] += 1
    live["queue_depth"] += 1
    live["host"]["free_bytes"] -= 1
    live["host"]["swapouts"] += 1
    assert stable_advert_digest(live) == baseline

    changed = copy.deepcopy(advert)
    changed["models"].append({"id": "model-b", "loaded": True})
    assert stable_advert_digest(changed) != baseline


def test_signed_advert_round_trip_and_tamper_refusals(tmp_path) -> None:
    community = create_community(tmp_path / "community")
    advert = _advert(community.nid)
    signed = sign_advert(
        advert, key_path=community.node_key, cert_path=community.node_cert
    )
    assert verify_advert(
        signed, ca_cert=community.ca_cert, expected_nid=community.nid
    ) == advert

    tampered = copy.deepcopy(signed)
    tampered["advert"]["seq"] += 1
    with pytest.raises(FabricError, match="signature"):
        verify_advert(tampered, ca_cert=community.ca_cert, expected_nid=community.nid)
    with pytest.raises(FabricError, match="expected peer"):
        verify_advert(
            signed, ca_cert=community.ca_cert, expected_nid="fedcba9876543210"
        )
    wrong_nid = copy.deepcopy(signed)
    wrong_nid["advert"]["nid"] = "fedcba9876543210"
    with pytest.raises(FabricError, match="certificate"):
        verify_advert(wrong_nid, ca_cert=community.ca_cert, expected_nid=None)


def test_probe_link_stats_bandwidth_and_length_validation(monkeypatch) -> None:
    durations = iter([0.010, 0.010, 0.001, 0.002, 0.003, 0.004, 0.008])
    clock = 0.0
    pending = None

    def perf_counter() -> float:
        nonlocal clock, pending
        if pending is None:
            pending = next(durations)
            return clock
        clock += pending
        pending = None
        return clock

    monkeypatch.setattr("mlx2.fabric.link.time.perf_counter", perf_counter)

    def roundtrip(_upload: bytes, download_len: int) -> bytes:
        return bytes(download_len)

    receipt = probe_link(
        roundtrip, peer_nid="peer", rtt_samples=3, payload_bytes=1024
    )
    assert receipt.rtt_ms_median == pytest.approx(2.0)
    assert receipt.rtt_ms_p90 == pytest.approx(3.0)
    assert receipt.rtt_samples == 3
    assert receipt.upload_mbps == pytest.approx(2.048)
    assert receipt.download_mbps == pytest.approx(1.024)
    assert receipt.payload_bytes == 1024
    assert receipt.as_dict()["peer_nid"] == "peer"

    monkeypatch.undo()
    with pytest.raises(FabricError, match="length mismatch"):
        probe_link(lambda _body, length: bytes(length - 1), peer_nid="bad", rtt_samples=1)


def test_rtt_only_probe_skips_bandwidth_payloads() -> None:
    calls = []

    def roundtrip(upload: bytes, download_len: int) -> bytes:
        calls.append((len(upload), download_len))
        return bytes(download_len)

    receipt = probe_link(
        roundtrip, peer_nid="peer", rtt_samples=20, payload_bytes=0
    )
    assert calls == [(64, 64)] * 22
    assert receipt.upload_mbps is None
    assert receipt.download_mbps is None
    assert receipt.payload_bytes == 0


def test_link_budget_and_echo_response() -> None:
    receipt = LinkReceipt(
        peer_nid="peer",
        transport="tls1.3/tcp",
        rtt_ms_median=6.25,
        rtt_ms_p90=7.0,
        rtt_samples=20,
        upload_mbps=100.0,
        download_mbps=100.0,
        payload_bytes=1024,
        measured_at=1.0,
    )
    budget = LinkBudget(per_round_budget_ms=5.0)
    for kind in LinkBudget.PER_ROUND_KINDS:
        assert budget.admit(kind, None) == (False, "link_unmeasured")
        admitted, reason = budget.admit(kind, receipt)
        assert admitted is False
        assert reason == "worst RTT p90(last 1) 7 ms > budget 5 ms"
    admitted, reason = budget.admit("route", receipt)
    assert admitted is True
    assert "latest RTT p90 7 ms" in reason
    admitted, reason = budget.admit("route", None)
    assert admitted is True
    assert "link_unmeasured" in reason

    assert echo_response(b"ab", 5) == b"ababa"
    assert echo_response(b"x", 100, max_len=7) == b"xxxxxxx"
    assert echo_response(b"", 3) == bytes(3)


def test_link_budget_uses_worst_p90_from_last_three_receipts() -> None:
    history = tuple(
        LinkReceipt("peer", "tls", 1, p90, 20, None, None, 0, measured)
        for measured, p90 in enumerate((99.0, 2.0, 8.0, 3.0), start=1)
    )
    admitted, reason = LinkBudget(5).admit(
        "pipeline", history[-1], history=history
    )
    assert admitted is False
    assert reason == "worst RTT p90(last 3) 8 ms > budget 5 ms"

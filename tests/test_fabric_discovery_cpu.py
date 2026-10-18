from __future__ import annotations

import json
import ssl
import threading
import time
from types import SimpleNamespace

import pytest

from mlx2.fabric.discovery import (
    Advertiser,
    Browser,
    Candidate,
    allowed_addresses,
    connect_https,
    discover,
    load_static_peers,
    order_addresses,
    parse_txt,
    txt_record,
)


def test_txt_record_is_minimal_and_parser_fails_closed() -> None:
    record = txt_record(nid="0123456789abcdef", tag="opaque", epoch=3, adv="abc123")
    assert record == {
        "v": "1",
        "nid": "0123456789abcdef",
        "cm": "opaque",
        "ep": "3",
        "adv": "abc123",
    }
    assert not ({"model", "memory", "tenant"} & set(record))
    assert parse_txt({key.encode(): value.encode() for key, value in record.items()}) == {
        "v": "1",
        "nid": "0123456789abcdef",
        "cm": "opaque",
        "ep": 3,
        "adv": "abc123",
    }
    assert parse_txt({**record, "v": "2"}) is None
    assert parse_txt({key: value for key, value in record.items() if key != "adv"}) is None
    assert parse_txt({**record, "nid": "not-hex---------"}) is None
    assert parse_txt({**record, "model": "private"}) is None


def test_allowed_addresses_excludes_public_wifi_link_local_and_vpn(monkeypatch) -> None:
    addresses = [
        "10.1.2.3",
        "172.16.4.5",
        "192.168.1.9",
        "127.0.0.1",
        "8.8.8.8",
        "100.64.0.1",
        "fd00::1",
    ]
    adapter = SimpleNamespace(
        name="en0", ips=[SimpleNamespace(ip=address) for address in addresses]
    )
    wifi_link_local = SimpleNamespace(name="en0", ips=[SimpleNamespace(ip="169.254.1.2")])
    monkeypatch.setattr("mlx2.fabric.discovery._WIFI_DEVICES", frozenset({"en0"}))
    monkeypatch.delenv("MLX2_FABRIC_ADDRESSES", raising=False)
    monkeypatch.setattr(
        "mlx2.fabric.discovery._ifaddr_module",
        lambda: SimpleNamespace(get_adapters=lambda: [adapter, wifi_link_local]),
    )
    assert allowed_addresses() == ["10.1.2.3", "172.16.4.5", "192.168.1.9"]
    assert allowed_addresses(include_loopback=True) == [
        "10.1.2.3",
        "127.0.0.1",
        "172.16.4.5",
        "192.168.1.9",
    ]


def test_address_order_prefers_the_shared_interface_and_connects_in_order(
    monkeypatch,
) -> None:
    adapter = SimpleNamespace(
        ips=[
            SimpleNamespace(ip="192.168.10.93", network_prefix=24),
            SimpleNamespace(ip="10.0.0.2", network_prefix=24),
        ]
    )
    monkeypatch.setattr(
        "mlx2.fabric.discovery._ifaddr_module",
        lambda: SimpleNamespace(get_adapters=lambda: [adapter]),
    )
    remote = ["192.168.20.21", "10.0.0.1"]
    local = ["192.168.10.93", "10.0.0.2"]
    assert order_addresses(remote, local) == ["10.0.0.1", "192.168.20.21"]

    attempts = []

    class Connection:
        def __init__(self, host, port, *, context, timeout):
            attempts.append((host, port, context, timeout))
            self.host = host

        def connect(self):
            if self.host == "10.0.0.1":
                raise TimeoutError("unreachable")

        def close(self):
            pass

    monkeypatch.setattr(
        "mlx2.fabric.discovery.http.client.HTTPSConnection", Connection
    )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    connection, address = connect_https(
        remote,
        8291,
        context=context,
        local_addresses=local,
        connect_timeout_s=0.25,
    )
    assert address == "192.168.20.21"
    assert connection.host == address
    assert [attempt[0] for attempt in attempts] == ["10.0.0.1", "192.168.20.21"]
    assert all(attempt[3] == 0.25 for attempt in attempts)


def test_address_order_falls_back_to_24_when_interface_metadata_is_missing(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "mlx2.fabric.discovery._ifaddr_module",
        lambda: SimpleNamespace(get_adapters=list),
    )
    assert order_addresses(
        ["192.168.20.21", "10.0.0.1"], ["192.168.10.93", "10.0.0.2"]
    ) == ["10.0.0.1", "192.168.20.21"]


def test_advertiser_update_is_noop_for_same_digest_and_uses_update_service() -> None:
    calls = []
    advertiser = Advertiser.__new__(Advertiser)
    advertiser._closed = False
    advertiser._adv = "stable"
    advertiser._info = object()
    advertiser._zc = SimpleNamespace(update_service=calls.append)
    advertiser._make_info = lambda: "replacement"

    advertiser.update(adv="stable")
    assert calls == []
    advertiser.update(adv="changed")
    assert calls == ["replacement"]


def test_static_peers_parse_and_win_over_zeroconf(tmp_path, monkeypatch) -> None:
    peers_path = tmp_path / "peers.json"
    peers_path.write_text(
        json.dumps(
            [
                {
                    "host": "192.168.1.20",
                    "port": 8443,
                    "nid": "0123456789abcdef",
                }
            ]
        ),
        encoding="utf-8",
    )
    expected = [
        Candidate(
            nid="0123456789abcdef",
            addresses=("192.168.1.20",),
            port=8443,
            tag="",
            epoch=0,
            adv="",
            source="static",
        )
    ]
    assert load_static_peers(peers_path) == expected

    def forbidden_browser(**_kwargs):
        raise AssertionError("zeroconf must not start when peers.json exists")

    monkeypatch.setattr("mlx2.fabric.discovery.Browser", forbidden_browser)
    assert discover(tag="ignored", home=tmp_path) == expected


def _wait_for_candidate(browser: Browser, nid: str, timeout: float = 5.0) -> Candidate | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for candidate in browser.candidates():
            if candidate.nid == nid:
                return candidate
        time.sleep(0.05)
    return None


def test_zeroconf_loopback_round_trip_and_tag_filter() -> None:
    pytest.importorskip("zeroconf", reason="zeroconf is not installed in the fabric site")
    tag = "community-one"
    found = threading.Event()
    browser = advertiser = foreign = None
    nid = "0123456789abcdef"
    try:
        browser = Browser(
            tag=tag,
            addresses=["127.0.0.1"],
            on_change=found.set,
        )
        advertiser = Advertiser(
            nid=nid,
            port=18765,
            tag=tag,
            epoch=7,
            adv="aaa111",
            addresses=["127.0.0.1"],
            instance_name=f"lane-b-{time.time_ns()}",
        )
        foreign = Advertiser(
            nid="fedcba9876543210",
            port=18766,
            tag="community-two",
            epoch=7,
            adv="bbb222",
            addresses=["127.0.0.1"],
            instance_name=f"lane-b-foreign-{time.time_ns()}",
        )
        found.wait(5)
        candidate = _wait_for_candidate(browser, nid)
        if candidate is None:
            pytest.skip("multicast DNS is unavailable on loopback in this sandbox")
        assert candidate.addresses == ("127.0.0.1",)
        assert candidate.port == 18765
        assert candidate.tag == tag
        assert candidate.epoch == 7
        assert candidate.adv == "aaa111"
        assert candidate.source == "zeroconf"
        assert all(item.nid != "fedcba9876543210" for item in browser.candidates())
    except OSError as exc:
        pytest.skip(f"multicast DNS is unavailable in this sandbox: {exc}")
    finally:
        if foreign is not None:
            foreign.close()
        if advertiser is not None:
            advertiser.close()
        if browser is not None:
            browser.close()


def test_direct_cable_link_local_is_allowed_and_ranked_after_shared_private(monkeypatch) -> None:
    """A Mac-to-Mac USB/Thunderbolt cable comes up 169.254.x until someone
    assigns addresses (the M5<->M2 USB link, 2026-10-18)."""
    usb = SimpleNamespace(name="en10", ips=[SimpleNamespace(ip="169.254.20.182")])
    reserved = SimpleNamespace(name="en11", ips=[SimpleNamespace(ip="169.254.0.7")])
    wifi = SimpleNamespace(name="en0", ips=[SimpleNamespace(ip="169.254.9.9")])
    lan = SimpleNamespace(name="bridge0", ips=[SimpleNamespace(ip="10.0.0.1")])
    monkeypatch.delenv("MLX2_FABRIC_ADDRESSES", raising=False)
    monkeypatch.setattr("mlx2.fabric.discovery._WIFI_DEVICES", frozenset({"en0"}))
    monkeypatch.setattr(
        "mlx2.fabric.discovery._ifaddr_module",
        lambda: SimpleNamespace(get_adapters=lambda: [usb, reserved, wifi, lan]),
    )
    from mlx2.fabric.discovery import order_addresses

    assert allowed_addresses() == ["10.0.0.1", "169.254.20.182"]
    monkeypatch.setenv("MLX2_FABRIC_ADDRESSES", "169.254.10.18,169.254.255.3")
    assert allowed_addresses() == ["169.254.10.18"]
    monkeypatch.setattr("mlx2.fabric.discovery._local_networks", lambda _local: (
        __import__("ipaddress").ip_network("10.0.0.0/24"),
        __import__("ipaddress").ip_network("169.254.0.0/16"),
    ))
    assert order_addresses(
        ["192.168.10.197", "169.254.10.18", "10.0.0.3"], ["10.0.0.1", "169.254.20.182"]
    ) == ["10.0.0.3", "169.254.10.18", "192.168.10.197"]


def test_dnssd_output_parsers() -> None:
    from mlx2.fabric.discovery import (
        parse_dnssd_addresses,
        parse_dnssd_browse_line,
        parse_dnssd_resolve,
    )

    line = "16:20:19.038  Add        3  27 local.               _mlx2-node._tcp.     0123456789abcdef"
    assert parse_dnssd_browse_line(line) == ("Add", "0123456789abcdef")
    assert parse_dnssd_browse_line(line.replace("Add", "Rmv")) == ("Rmv", "0123456789abcdef")
    assert parse_dnssd_browse_line(line.replace("_mlx2-node._tcp.", "_ssh._tcp.")) is None
    assert parse_dnssd_browse_line("Timestamp     A/R    Flags  if Domain") is None
    resolved = parse_dnssd_resolve(
        "Lookup 0123456789abcdef._mlx2-node._tcp.local.\r\n"
        "16:20:23.048  0123456789abcdef._mlx2-node._tcp.local. can be reached at "
        "0123456789abcdef.local.:8291 (interface 27)\r\n"
        " v=1 nid=0123456789abcdef cm=fedcba9876543210 ep=1 adv=a1b2c3d4e5f6\r\n"
    )
    assert resolved == (
        "0123456789abcdef.local.",
        8291,
        {"v": "1", "nid": "0123456789abcdef", "cm": "fedcba9876543210", "ep": "1", "adv": "a1b2c3d4e5f6"},
    )
    assert parse_dnssd_resolve("Lookup x\r\n...STARTING...\r\n") is None
    addresses = parse_dnssd_addresses(
        "Timestamp     A/R  Flags         IF  Hostname                  Address    TTL\r\n"
        "16:21:00.1  Add  40000002      27  0123456789abcdef.local.  169.254.10.18  120\r\n"
        "16:21:00.1  Add  40000002      12  0123456789abcdef.local.  192.168.10.197  120\r\n"
        "16:21:00.1  Add  40000002      12  0123456789abcdef.local.  192.168.10.197  120\r\n"
    )
    assert addresses == ("169.254.10.18", "192.168.10.197")

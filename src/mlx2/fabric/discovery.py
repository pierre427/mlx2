from __future__ import annotations

import http.client
import ipaddress
import json
import os
import socket
import ssl
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlx2.fabric import FabricError, FabricUnavailable

SERVICE_TYPE = "_mlx2-node._tcp.local."
_TXT_KEYS = frozenset({"v", "nid", "cm", "ep", "adv"})
_PRIVATE_V4 = tuple(
    ipaddress.ip_network(network) for network in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
)


@dataclass(frozen=True)
class Candidate:
    nid: str
    addresses: tuple[str, ...]
    port: int
    tag: str
    epoch: int
    adv: str
    source: str


def txt_record(*, nid: str, tag: str, epoch: int, adv: str) -> dict[str, str]:
    return {"v": "1", "nid": nid, "cm": tag, "ep": str(epoch), "adv": adv}


def _text(value: bytes | str | None) -> str | None:
    if value is None:
        return None
    try:
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)
    except UnicodeDecodeError:
        return None


def parse_txt(props: Mapping[bytes | str, bytes | str | None]) -> dict[str, Any] | None:
    decoded: dict[str, str | None] = {}
    for raw_key, raw_value in props.items():
        key = _text(raw_key)
        if key is None or key in decoded:
            return None
        decoded[key] = _text(raw_value)
    if set(decoded) != _TXT_KEYS or any(value is None for value in decoded.values()):
        return None
    if decoded["v"] != "1":
        return None
    nid = decoded["nid"]
    assert nid is not None
    if len(nid) != 16 or any(char not in "0123456789abcdef" for char in nid):
        return None
    try:
        epoch = int(decoded["ep"] or "")
    except ValueError:
        return None
    if epoch < 0:
        return None
    return {
        "v": "1",
        "nid": nid,
        "cm": decoded["cm"],
        "ep": epoch,
        "adv": decoded["adv"],
    }


# RFC 3927 reserves the first and last /24 of 169.254.0.0/16.
_LINK_LOCAL_USABLE = ipaddress.ip_network("169.254.0.0/16")
_LINK_LOCAL_RESERVED = (
    ipaddress.ip_network("169.254.0.0/24"),
    ipaddress.ip_network("169.254.255.0/24"),
)
_WIFI_DEVICES: frozenset[str] | None = None


def _wifi_devices() -> frozenset[str]:
    """BSD names of Wi-Fi hardware ports (empty when unknown)."""
    global _WIFI_DEVICES
    if _WIFI_DEVICES is None:
        devices: set[str] = set()
        try:
            listing = subprocess.run(
                ["/usr/sbin/networksetup", "-listallhardwareports"],
                capture_output=True, text=True, timeout=5, check=False,
            ).stdout
            port = None
            for line in listing.splitlines():
                if line.startswith("Hardware Port:"):
                    port = line.split(":", 1)[1].strip()
                elif line.startswith("Device:") and port in {"Wi-Fi", "AirPort"}:
                    devices.add(line.split(":", 1)[1].strip())
        except (OSError, subprocess.SubprocessError):
            pass
        _WIFI_DEVICES = frozenset(devices)
    return _WIFI_DEVICES


def _usable_link_local(parsed: ipaddress.IPv4Address, interface: str | None) -> bool:
    """A self-assigned address on a direct cable (USB or Thunderbolt Mac-to-Mac).

    Direct links come up link-local until someone assigns addresses, so they
    count as a direct link.  Wi-Fi is excluded: a self-assigned address there
    means a failed DHCP lease on someone else's network.  ``interface`` None
    means the operator listed the address explicitly (MLX2_FABRIC_ADDRESSES).
    """
    if parsed not in _LINK_LOCAL_USABLE or any(parsed in net for net in _LINK_LOCAL_RESERVED):
        return False
    return interface is None or interface not in _wifi_devices()


def _allowed_address(
    address: str, *, include_loopback: bool, interface: str | None = None
) -> bool:
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return False
    if not isinstance(parsed, ipaddress.IPv4Address):
        return False
    if parsed.is_link_local:
        return _usable_link_local(parsed, interface)
    if parsed.is_loopback:
        return include_loopback and parsed == ipaddress.ip_address("127.0.0.1")
    return any(parsed in network for network in _PRIVATE_V4)


def _ifaddr_module():
    try:
        import ifaddr
    except ImportError as exc:
        raise FabricUnavailable(
            "fabric discovery requires ifaddr from ~/.mlx2-fabric-site"
        ) from exc
    return ifaddr


def allowed_addresses(include_loopback: bool = False) -> list[str]:
    override = os.environ.get("MLX2_FABRIC_ADDRESSES")
    if override is not None:
        candidates = [(part.strip(), None) for part in override.split(",") if part.strip()]
    else:
        candidates = []
        for adapter in _ifaddr_module().get_adapters():
            name = getattr(adapter, "name", "") or ""
            for ip in adapter.ips:
                address = ip.ip[0] if isinstance(ip.ip, tuple) else ip.ip
                candidates.append((address, name))
    return sorted(
        {
            address
            for address, interface in candidates
            if isinstance(address, str)
            and _allowed_address(address, include_loopback=include_loopback, interface=interface)
        }
    )


def _local_networks(local_addresses: Sequence[str]) -> tuple[ipaddress.IPv4Network, ...]:
    prefixes: dict[str, int] = {}
    try:
        for adapter in _ifaddr_module().get_adapters():
            for item in adapter.ips:
                address = item.ip[0] if isinstance(item.ip, tuple) else item.ip
                prefix = getattr(item, "network_prefix", None)
                if isinstance(address, str) and isinstance(prefix, int):
                    prefixes[address] = prefix
    except FabricUnavailable:
        pass
    networks: list[ipaddress.IPv4Network] = []
    for address in local_addresses:
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError:
            continue
        if isinstance(parsed, ipaddress.IPv4Address):
            prefix = prefixes.get(address, 24)
            networks.append(ipaddress.ip_network(f"{address}/{prefix}", strict=False))
    return tuple(dict.fromkeys(networks))


def order_addresses(
    addresses: Sequence[str], local_addresses: Sequence[str]
) -> list[str]:
    """Put peer addresses on a directly shared local network first."""
    networks = _local_networks(local_addresses)

    def shared_network(address: str) -> bool:
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError:
            return False
        return isinstance(parsed, ipaddress.IPv4Address) and any(
            parsed in network for network in networks
        )

    def link_local(address: str) -> bool:
        try:
            return ipaddress.ip_address(address).is_link_local
        except ValueError:
            return False

    # Shared routable private networks first, then shared direct-cable
    # link-local links, then everything else.
    unique = list(dict.fromkeys(addresses))
    return sorted(unique, key=lambda address: (not shared_network(address), link_local(address)))


def connect_https(
    addresses: Sequence[str],
    port: int,
    *,
    context: ssl.SSLContext,
    local_addresses: Sequence[str],
    preferred_address: str | None = None,
    connect_timeout_s: float = 2.0,
) -> tuple[http.client.HTTPSConnection, str]:
    """Return the first HTTPS connection whose TCP and TLS handshake completes."""
    ordered = order_addresses(addresses, local_addresses)
    if preferred_address in ordered:
        ordered.remove(preferred_address)
        ordered.insert(0, preferred_address)
    failures: list[str] = []
    for address in ordered:
        conn = http.client.HTTPSConnection(
            address, port, context=context, timeout=connect_timeout_s
        )
        try:
            conn.connect()
            return conn, address
        except (OSError, ssl.SSLError) as exc:
            conn.close()
            failures.append(f"{address}: {exc}")
    detail = "; ".join(failures) if failures else "no candidate addresses"
    raise FabricError(f"could not connect to fabric peer: {detail}")


def _zeroconf_types():
    try:
        from zeroconf import ServiceBrowser, ServiceInfo, Zeroconf
    except ImportError as exc:
        raise FabricUnavailable(
            "fabric discovery requires zeroconf from ~/.mlx2-fabric-site"
        ) from exc
    return ServiceBrowser, ServiceInfo, Zeroconf


def _validate_addresses(addresses: Sequence[str]) -> tuple[str, ...]:
    result = tuple(dict.fromkeys(addresses))
    if not result:
        raise FabricError("fabric discovery requires at least one interface address")
    for address in result:
        if not _allowed_address(address, include_loopback=True):
            raise FabricError(f"fabric discovery address is not private or loopback: {address}")
    return result


class Advertiser:
    def __init__(
        self,
        *,
        nid: str,
        port: int,
        tag: str,
        epoch: int,
        adv: str,
        addresses: Sequence[str],
        instance_name: str | None = None,
    ) -> None:
        _, service_info, zeroconf = _zeroconf_types()
        self._service_info = service_info
        self._addresses = _validate_addresses(addresses)
        self._nid = nid
        self._port = port
        self._tag = tag
        self._epoch = epoch
        self._adv = adv
        label = instance_name or nid
        self._name = label if label.endswith(SERVICE_TYPE) else f"{label}.{SERVICE_TYPE}"
        self._zc = zeroconf(interfaces=list(self._addresses))
        self._info = self._make_info()
        try:
            self._zc.register_service(self._info)
        except Exception:
            self._zc.close()
            raise
        self._closed = False

    def _make_info(self):
        return self._service_info(
            SERVICE_TYPE,
            self._name,
            addresses=[socket.inet_aton(address) for address in self._addresses],
            port=self._port,
            properties=txt_record(
                nid=self._nid, tag=self._tag, epoch=self._epoch, adv=self._adv
            ),
            server=f"{self._nid}.local.",
        )

    def update(self, *, adv: str) -> None:
        if self._closed:
            raise FabricError("advertiser is closed")
        if adv == self._adv:
            return
        self._adv = adv
        self._info = self._make_info()
        self._zc.update_service(self._info)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._zc.unregister_service(self._info)
        finally:
            self._zc.close()


_DNS_SD = "/usr/bin/dns-sd"


def parse_dnssd_browse_line(line: str) -> tuple[str, str] | None:
    """('Add'|'Rmv', instance name) from a ``dns-sd -B`` line, else None."""
    fields = line.split()
    if len(fields) < 7 or fields[1] not in {"Add", "Rmv"}:
        return None
    if not fields[5].startswith("_mlx2-node._tcp"):
        return None
    return fields[1], " ".join(fields[6:])


def parse_dnssd_resolve(text: str) -> tuple[str, int, dict[str, str]] | None:
    """(host, port, TXT) from ``dns-sd -L`` output, else None."""
    host = port = None
    txt: dict[str, str] = {}
    for line in text.splitlines():
        if "can be reached at" in line and host is None:
            target = line.split("can be reached at", 1)[1].split()[0]
            name, _, raw_port = target.rpartition(":")
            try:
                host, port = name, int(raw_port)
            except ValueError:
                return None
        elif host is not None and "=" in line and not txt:
            for item in line.split():
                key, sep, value = item.partition("=")
                if sep:
                    txt[key] = value
    if host is None or port is None or not txt:
        return None
    return host, port, txt


def parse_dnssd_addresses(text: str) -> tuple[str, ...]:
    """IPv4 addresses from ``dns-sd -G v4`` output."""
    found: list[str] = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 6 and fields[1] == "Add":
            for field in fields[4:]:
                try:
                    parsed = ipaddress.ip_address(field)
                except ValueError:
                    continue
                if isinstance(parsed, ipaddress.IPv4Address) and field not in found:
                    found.append(field)
    return tuple(found)


def _dnssd_spawn(args: Sequence[str]) -> tuple[subprocess.Popen, int] | None:
    """Start dns-sd on a pseudo-terminal: piped, it block-buffers its output
    and a killed process loses whatever it had not flushed."""
    import pty

    try:
        master, slave = pty.openpty()
    except OSError:
        return None
    try:
        process = subprocess.Popen(
            [_DNS_SD, *args], stdout=slave, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL, close_fds=True,
        )
    except OSError:
        os.close(master)
        return None
    finally:
        os.close(slave)
    return process, master


def _dnssd_read(master: int, seconds: float) -> str:
    import select

    chunks: list[bytes] = []
    deadline = time.monotonic() + seconds
    while (remaining := deadline - time.monotonic()) > 0:
        ready, _, _ = select.select([master], [], [], remaining)
        if not ready:
            break
        try:
            data = os.read(master, 65536)
        except OSError:
            break
        if not data:
            break
        chunks.append(data)
    return b"".join(chunks).decode("utf-8", "replace")


def _dnssd_run(args: Sequence[str], seconds: float) -> str:
    """Run a dns-sd query for a bounded time (it never exits on its own)."""
    spawned = _dnssd_spawn(args)
    if spawned is None:
        return ""
    process, master = spawned
    try:
        return _dnssd_read(master, seconds)
    finally:
        process.kill()
        process.wait()
        os.close(master)


class _SystemResolver:
    """macOS mDNSResponder as a second discovery source.

    python-zeroconf missed a peer on a USB link when the host had two
    link-local interfaces in 169.254/16, while the OS daemon resolved it
    (M5<->M2, 2026-10-18).  Candidates from here are merged with zeroconf's
    and filtered by the same community tag.
    """

    RESOLVE_EVERY_S = 30.0

    def __init__(self, tag: str) -> None:
        self._tag = tag
        self._lock = threading.Lock()
        self._names: dict[str, float] = {}
        self._by_name: dict[str, Candidate] = {}
        self._stop = threading.Event()
        self._process: subprocess.Popen | None = None
        self._threads = [
            threading.Thread(target=self._browse, daemon=True),
            threading.Thread(target=self._resolve_loop, daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def _browse(self) -> None:
        spawned = _dnssd_spawn(["-B", "_mlx2-node._tcp", "local."])
        if spawned is None:
            return
        self._process, master = spawned
        pending = ""
        try:
            while not self._stop.is_set():
                pending += _dnssd_read(master, 1.0)
                *lines, pending = pending.split("\n")
                for line in lines:
                    parsed = parse_dnssd_browse_line(line.strip())
                    if parsed is None:
                        continue
                    action, name = parsed
                    with self._lock:
                        if action == "Add":
                            self._names.setdefault(name, 0.0)
                        else:
                            self._names.pop(name, None)
                            self._by_name.pop(name, None)
                if self._process.poll() is not None:
                    break
        finally:
            os.close(master)

    def _resolve_loop(self) -> None:
        while not self._stop.wait(0.5):
            now = time.monotonic()
            with self._lock:
                due = [n for n, last in self._names.items() if now - last >= self.RESOLVE_EVERY_S]
                for name in due:
                    self._names[name] = now
            for name in due:
                candidate = self._resolve(name)
                with self._lock:
                    if name not in self._names:
                        continue
                    if candidate is None:
                        self._by_name.pop(name, None)
                    else:
                        self._by_name[name] = candidate

    def _resolve(self, name: str) -> Candidate | None:
        resolved = parse_dnssd_resolve(_dnssd_run(["-L", name, "_mlx2-node._tcp", "local."], 2.0))
        if resolved is None:
            return None
        host, port, txt = resolved
        parsed = parse_txt(txt)
        if parsed is None or parsed["cm"] != self._tag:
            return None
        addresses = parse_dnssd_addresses(_dnssd_run(["-G", "v4", host], 2.0))
        if not addresses:
            return None
        return Candidate(
            nid=parsed["nid"], addresses=addresses, port=port, tag=parsed["cm"],
            epoch=parsed["ep"], adv=parsed["adv"], source="mdnsresponder",
        )

    def candidates(self) -> list[Candidate]:
        with self._lock:
            return list(self._by_name.values())

    def close(self) -> None:
        self._stop.set()
        if self._process is not None:
            self._process.kill()


class Browser:
    def __init__(
        self,
        *,
        tag: str,
        addresses: Sequence[str] | None = None,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        service_browser, _, zeroconf = _zeroconf_types()
        chosen = _validate_addresses(
            addresses if addresses is not None else allowed_addresses(include_loopback=True)
        )
        self._tag = tag
        self._on_change = on_change
        self._lock = threading.Lock()
        self._by_nid: dict[str, Candidate] = {}
        self._name_to_nid: dict[str, str] = {}
        self._zc = zeroconf(interfaces=list(chosen))
        self._browser = service_browser(self._zc, SERVICE_TYPE, self)
        self._system = _SystemResolver(tag) if os.path.exists(_DNS_SD) else None
        self._closed = False

    def add_service(self, zc, service_type: str, name: str) -> None:
        self._resolve(zc, service_type, name)

    def update_service(self, zc, service_type: str, name: str) -> None:
        self._resolve(zc, service_type, name)

    def remove_service(self, _zc, _service_type: str, name: str) -> None:
        changed = False
        with self._lock:
            nid = self._name_to_nid.pop(name, None)
            if nid is not None and self._by_nid.get(nid) is not None:
                del self._by_nid[nid]
                changed = True
        if changed and self._on_change is not None:
            self._on_change()

    def _resolve(self, zc, service_type: str, name: str) -> None:
        info = zc.get_service_info(service_type, name)
        if info is None:
            return
        parsed = parse_txt(info.properties)
        if parsed is None or parsed["cm"] != self._tag:
            self._drop_name(name)
            return
        addresses = tuple(info.parsed_addresses())
        candidate = Candidate(
            nid=parsed["nid"],
            addresses=addresses,
            port=info.port,
            tag=parsed["cm"],
            epoch=parsed["ep"],
            adv=parsed["adv"],
            source="zeroconf",
        )
        with self._lock:
            old_nid = self._name_to_nid.get(name)
            if old_nid is not None and old_nid != candidate.nid:
                self._by_nid.pop(old_nid, None)
            changed = self._by_nid.get(candidate.nid) != candidate
            self._name_to_nid[name] = candidate.nid
            self._by_nid[candidate.nid] = candidate
        if changed and self._on_change is not None:
            self._on_change()

    def _drop_name(self, name: str) -> None:
        changed = False
        with self._lock:
            nid = self._name_to_nid.pop(name, None)
            if nid is not None and nid in self._by_nid:
                del self._by_nid[nid]
                changed = True
        if changed and self._on_change is not None:
            self._on_change()

    def candidates(self) -> list[Candidate]:
        with self._lock:
            merged = dict(self._by_nid)
        for extra in self._system.candidates() if self._system is not None else ():
            known = merged.get(extra.nid)
            if known is None:
                merged[extra.nid] = extra
            else:
                addresses = tuple(dict.fromkeys((*known.addresses, *extra.addresses)))
                merged[extra.nid] = Candidate(
                    nid=known.nid, addresses=addresses, port=known.port, tag=known.tag,
                    epoch=known.epoch, adv=known.adv, source=known.source,
                )
        return sorted(merged.values(), key=lambda item: item.nid)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._system is not None:
            self._system.close()
        self._browser.cancel()
        self._zc.close()


def load_static_peers(path: Path) -> list[Candidate]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FabricError(f"cannot load static peers from {path}: {exc}") from exc
    if not isinstance(raw, list):
        raise FabricError("static peers file must contain a JSON list")
    peers: list[Candidate] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or set(item) != {"host", "port", "nid"}:
            raise FabricError(f"invalid static peer at index {index}")
        host, port, nid = item["host"], item["port"], item["nid"]
        if not isinstance(host, str) or not isinstance(port, int) or not isinstance(nid, str):
            raise FabricError(f"invalid static peer at index {index}")
        if not (0 < port <= 65535) or len(nid) != 16 or any(
            char not in "0123456789abcdef" for char in nid
        ):
            raise FabricError(f"invalid static peer at index {index}")
        peers.append(Candidate(nid, (host,), port, "", 0, "", "static"))
    return peers


def discover(
    *,
    tag: str,
    home: Path | None = None,
    addresses: Sequence[str] | None = None,
    on_change: Callable[[], None] | None = None,
) -> list[Candidate] | Browser:
    peers_path = (home or Path.home() / ".mlx2" / "fabric") / "peers.json"
    if peers_path.exists():
        return load_static_peers(peers_path)
    return Browser(tag=tag, addresses=addresses, on_change=on_change)

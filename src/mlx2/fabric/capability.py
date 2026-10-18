from __future__ import annotations

import json
import re
import socket
import subprocess
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from mlx2.fabric import FabricError
from mlx2.fabric.community import canonical_json, cert_nid, sha256_hex, sign, verify

ADVERT_SCHEMA = "mlx2.fabric.advert.v1"
_PUBLIC_MODEL_KEYS = (
    "id",
    "server_url",
    "loaded",
    "qualification",
    "capabilities",
    "prefill_handoff",
)
_PUBLIC_HOST_KEYS = (
    "chip",
    "memory_bytes",
    "free_bytes",
    "swapouts",
    "hostname",
    "device_class",
)


class _LocalModels(list[dict[str, Any]]):
    def __init__(
        self, values: list[dict[str, Any]], errors: list[dict[str, str]]
    ) -> None:
        super().__init__(values)
        self.errors = errors


def _command(args: list[str]) -> str | None:
    try:
        return subprocess.run(
            args, check=True, capture_output=True, text=True, timeout=2
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _integer(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _vm_facts(output: str | None) -> tuple[int | None, int | None]:
    if not output:
        return None, None
    page_match = re.search(r"page size of (\d+) bytes", output)
    page_size = int(page_match.group(1)) if page_match else None
    counts: dict[str, int] = {}
    for line in output.splitlines():
        match = re.match(r"([^:]+):\s*(\d+)\.?$", line.strip())
        if match:
            counts[match.group(1)] = int(match.group(2))
    free_bytes = None
    if page_size is not None:
        free_bytes = page_size * sum(
            counts.get(name, 0)
            for name in ("Pages free", "Pages inactive", "Pages speculative")
        )
    return free_bytes, counts.get("Swapouts")


def _device_class(chip: str | None) -> str | None:
    if not chip:
        return None
    return re.sub(r"[^A-Za-z0-9]+", "_", chip).strip("_") or None


def host_facts() -> dict[str, Any]:
    try:
        chip = _command(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"])
        memory = _integer(_command(["/usr/sbin/sysctl", "-n", "hw.memsize"]))
        free_bytes, swapouts = _vm_facts(_command(["/usr/bin/vm_stat"]))
        try:
            hostname = socket.gethostname()
        except OSError:
            hostname = None
        return {
            "chip": chip,
            "memory_bytes": memory,
            "free_bytes": free_bytes,
            "swapouts": swapouts,
            "hostname": hostname,
            "device_class": _device_class(chip),
        }
    except Exception:  # noqa: BLE001 -- telemetry must never prevent advertising.
        return {key: None for key in _PUBLIC_HOST_KEYS}


def _loopback_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
            return False
        if parsed.hostname is None or parsed.query or parsed.fragment:
            return False
        return parsed.hostname == "localhost" or socket.gethostbyname(
            parsed.hostname
        ).startswith("127.")
    except (OSError, ValueError):
        return False


def local_models(servers: Sequence[dict], *, timeout_s: float = 2.0) -> list[dict]:
    models: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for server in servers:
        url = server.get("url")
        if not isinstance(url, str) or not _loopback_url(url):
            raise FabricError(f"local model server must use a loopback URL: {url!r}")
        request = urllib.request.Request(url.rstrip("/") + "/v1/models")
        api_key = server.get("api_key")
        if api_key:
            request.add_header("Authorization", f"Bearer {api_key}")
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                payload = json.load(response)
            entries = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(entries, list):
                raise TypeError("response data is not a list")
            for item in entries:
                if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                    continue
                loaded = item.get("loaded", True)
                if loaded is not True:
                    continue
                model = {
                    "id": item["id"],
                    "server_url": url.rstrip("/"),
                    "loaded": True,
                    "qualification": item.get("qualification", "unqualified"),
                    "capabilities": item.get("capabilities", {}),
                }
                if isinstance(item.get("prefill_handoff"), dict):
                    model["prefill_handoff"] = item["prefill_handoff"]
                models.append(model)
        except (OSError, TypeError, ValueError) as exc:
            errors.append({"server_url": url.rstrip("/"), "error": str(exc)})
    return _LocalModels(models, errors)


def _public(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _public(item)
            for key, item in value.items()
            if isinstance(key, str) and not _private_key(key)
        }
    if isinstance(value, (list, tuple)):
        return [_public(item) for item in value]
    return value


def _private_key(key: str) -> bool:
    lowered = key.lower()
    return (
        "secret" in lowered
        or lowered.startswith("tenant")
        or lowered in {"api_key", "authorization"}
    )


def build_advert(
    *,
    nid: str,
    community_id: str,
    epoch: int,
    roles,
    port: int,
    addresses,
    models,
    host,
    runtime: dict | None,
    seq: int,
    queue_depth: int | None = None,
) -> dict:
    public_models = [
        {key: _public(model.get(key)) for key in _PUBLIC_MODEL_KEYS if key in model}
        for model in models
        if isinstance(model, dict)
    ]
    doc = {
        "schema": ADVERT_SCHEMA,
        "nid": nid,
        "community_id": community_id,
        "epoch": epoch,
        "roles": list(roles),
        "endpoint": {"port": port, "addresses": list(addresses)},
        "models": public_models,
        "host": {key: host.get(key) for key in _PUBLIC_HOST_KEYS},
        "runtime": _public(runtime),
        "seq": seq,
        "queue_depth": queue_depth,
    }
    errors = getattr(models, "errors", None)
    if errors:
        doc["errors"] = _public(errors)
    return doc


def advert_digest(doc: dict) -> str:
    return sha256_hex(canonical_json(doc))


def stable_advert_digest(doc: dict) -> str:
    host = doc.get("host") or {}
    stable = {
        "nid": doc.get("nid"),
        "community_id": doc.get("community_id"),
        "epoch": doc.get("epoch"),
        "roles": doc.get("roles"),
        "endpoint": doc.get("endpoint"),
        "models": doc.get("models"),
        "runtime": doc.get("runtime"),
        "host": {
            key: host.get(key)
            for key in ("device_class", "chip", "memory_bytes", "hostname")
        },
    }
    return sha256_hex(canonical_json(stable))


def sign_advert(doc: dict, *, key_path: Path, cert_path: Path) -> dict:
    return {
        "advert": doc,
        "signature": sign(key_path, canonical_json(doc)),
        "cert_pem": cert_path.read_text(encoding="utf-8"),
    }


def verify_advert(signed: dict, *, ca_cert: Path, expected_nid: str | None) -> dict:
    try:
        doc = signed["advert"]
        signature = signed["signature"]
        cert_pem = signed["cert_pem"]
        nid = doc["nid"]
    except (KeyError, TypeError) as exc:
        raise FabricError("malformed signed capability advert") from exc
    if not isinstance(doc, dict) or doc.get("schema") != ADVERT_SCHEMA:
        raise FabricError("invalid capability advert schema")
    if not isinstance(nid, str) or nid != cert_nid(cert_pem):
        raise FabricError("capability advert nid does not match certificate")
    if expected_nid is not None and nid != expected_nid:
        raise FabricError("capability advert nid does not match expected peer")
    if not verify(cert_pem, canonical_json(doc), signature, ca_cert=ca_cert):
        raise FabricError("invalid capability advert signature")
    return doc

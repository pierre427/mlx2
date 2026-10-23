from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import ssl
import subprocess
import tempfile
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from . import FabricError

FABRIC_SCHEMA = "mlx2.fabric.v1"
_OPENSSL = "/usr/bin/openssl"
_MAX_CERT_LIFETIME_S = 7 * 86400
_HEX_16 = re.compile(r"^[0-9a-f]{16}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


def canonical_json(obj) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def default_home() -> Path:
    configured = os.environ.get("MLX2_FABRIC_HOME")
    return Path(configured).expanduser() if configured else Path.home() / ".mlx2" / "fabric"


@dataclass(frozen=True)
class Community:
    home: Path
    community_id: str
    epoch: int
    secret: bytes
    ca_key: Path
    ca_cert: Path
    node_key: Path
    node_cert: Path
    nid: str
    roles: tuple[str, ...]


@dataclass(frozen=True)
class PeerVerdict:
    ok: bool
    nid: str | None
    reason: str


def _run(args: Sequence[str], *, input_data: bytes | None = None) -> bytes:
    try:
        result = subprocess.run(
            [_OPENSSL, *args],
            input=input_data,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise FabricError(f"openssl unavailable: {exc}") from exc
    if result.returncode:
        detail = result.stderr.decode(errors="replace").strip()
        raise FabricError(f"openssl failed: {detail or 'unknown error'}")
    return result.stdout


def _write(path: Path, data: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, mode)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        Path(temporary).unlink(missing_ok=True)
        raise


def _append_json(path: Path, entry: dict) -> None:
    encoded = canonical_json(entry) + b"\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        remaining = memoryview(encoded)
        while remaining:
            remaining = remaining[os.write(fd, remaining) :]
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    entries = []
    try:
        for line in path.read_bytes().splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    entries.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise FabricError(f"invalid {path.name}: {exc}") from exc
    return entries


def _generate_key(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        generated = Path(temporary.name)
    try:
        _run(["ecparam", "-genkey", "-name", "prime256v1", "-noout", "-out", str(generated)])
        os.chmod(generated, 0o600)
        os.replace(generated, path)
    finally:
        generated.unlink(missing_ok=True)


def _public_der_from_key(path: Path) -> bytes:
    return _run(["pkey", "-in", str(path), "-pubout", "-outform", "DER"])


def _public_der_from_cert(path: Path) -> bytes:
    public_pem = _run(["x509", "-in", str(path), "-pubkey", "-noout"])
    return _run(["pkey", "-pubin", "-outform", "DER"], input_data=public_pem)


def node_id_for_key(key_path: Path) -> str:
    return sha256_hex(_public_der_from_key(Path(key_path)))[:16]


def _cert_der(cert_pem: str | Path) -> bytes:
    if isinstance(cert_pem, Path):
        return _run(["x509", "-in", str(cert_pem), "-outform", "DER"])
    return _run(["x509", "-outform", "DER"], input_data=cert_pem.encode())


def cert_nid(cert_pem: str) -> str:
    public_pem = _run(["x509", "-pubkey", "-noout"], input_data=cert_pem.encode())
    public_der = _run(["pkey", "-pubin", "-outform", "DER"], input_data=public_pem)
    return sha256_hex(public_der)[:16]


def _ca_config(path: Path) -> str:
    return "\n".join(
        [
            "[req]",
            "distinguished_name = dn",
            "x509_extensions = ca_ext",
            "prompt = no",
            "[dn]",
            "CN = mlx2 community CA",
            "[ca_ext]",
            "basicConstraints = critical,CA:TRUE",
            "keyUsage = critical,keyCertSign,cRLSign",
            "subjectKeyIdentifier = hash",
            "authorityKeyIdentifier = keyid:always",
            "[ca]",
            "default_ca = local_ca",
            "[local_ca]",
            f"database = {path / 'index.txt'}",
            f"new_certs_dir = {path / 'newcerts'}",
            f"serial = {path / 'serial'}",
            "default_md = sha256",
            "default_days = 1",
            "policy = policy_any",
            "unique_subject = no",
            "copy_extensions = none",
            "[policy_any]",
            "commonName = supplied",
            "organizationalUnitName = optional",
            "[node_ext]",
            "basicConstraints = critical,CA:FALSE",
            "keyUsage = critical,digitalSignature,keyAgreement",
            "extendedKeyUsage = serverAuth,clientAuth",
            "subjectKeyIdentifier = hash",
            "authorityKeyIdentifier = keyid,issuer",
        ]
    ) + "\n"


def _new_ca(key: Path, cert: Path) -> None:
    _generate_key(key)
    with tempfile.TemporaryDirectory() as raw:
        config = Path(raw) / "openssl.cnf"
        config.write_text(_ca_config(Path(raw)))
        _run(
            [
                "req",
                "-new",
                "-x509",
                "-sha256",
                "-days",
                "3650",
                "-key",
                str(key),
                "-out",
                str(cert),
                "-config",
                str(config),
                "-extensions",
                "ca_ext",
            ]
        )
    os.chmod(key, 0o600)
    os.chmod(cert, 0o644)


def _make_csr_for_key(key: Path, nid: str) -> str:
    return _run(["req", "-new", "-sha256", "-key", str(key), "-subj", f"/CN={nid}"]).decode()


def _csr_nid(csr_pem: str) -> str:
    _run(["req", "-verify", "-noout"], input_data=csr_pem.encode())
    public_pem = _run(["req", "-pubkey", "-noout"], input_data=csr_pem.encode())
    public_der = _run(["pkey", "-pubin", "-outform", "DER"], input_data=public_pem)
    return sha256_hex(public_der)[:16]


def _clean_roles(roles: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(roles, (list, tuple)):
        raise FabricError("invalid roles")
    result = tuple(dict.fromkeys(roles))
    if not result or any(not isinstance(role, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", role) for role in result):
        raise FabricError("invalid roles")
    return result


def _san_lines(hosts: Sequence[str]) -> list[str]:
    import ipaddress

    values = []
    for index, host in enumerate(dict.fromkeys(hosts), 1):
        try:
            ipaddress.ip_address(host)
            values.append(f"IP.{index} = {host}")
        except ValueError:
            if not host or any(char in host for char in "\n\r=,/"):
                raise FabricError("invalid certificate host")
            values.append(f"DNS.{index} = {host}")
    return values


def _issue_certificate(
    ca_key: Path,
    ca_cert: Path,
    csr_pem: str,
    *,
    nid: str,
    epoch: int,
    roles: Sequence[str],
    hosts: Sequence[str],
    lifetime_s: int,
) -> str:
    if not isinstance(lifetime_s, int) or lifetime_s <= 0:
        raise FabricError("certificate lifetime must be positive")
    if lifetime_s > _MAX_CERT_LIFETIME_S:
        raise FabricError("certificate lifetime exceeds 7 days")
    clean_roles = _clean_roles(roles)
    now = datetime.now(UTC) - timedelta(seconds=1)
    end = now + timedelta(seconds=lifetime_s)
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        (root / "newcerts").mkdir()
        (root / "index.txt").write_text("")
        (root / "serial").write_text(secrets.token_hex(16).upper() + "\n")
        csr = root / "node.csr"
        cert = root / "node.pem"
        config = root / "openssl.cnf"
        csr.write_text(csr_pem)
        text = _ca_config(root)
        sans = _san_lines(hosts)
        if sans:
            text += "subjectAltName = @alt_names\n[alt_names]\n" + "\n".join(sans) + "\n"
        config.write_text(text)
        subject = f"/CN={nid}/OU=epoch={epoch};roles={','.join(clean_roles)}"
        _run(
            [
                "ca",
                "-batch",
                "-config",
                str(config),
                "-keyfile",
                str(ca_key),
                "-cert",
                str(ca_cert),
                "-in",
                str(csr),
                "-out",
                str(cert),
                "-notext",
                "-extensions",
                "node_ext",
                "-startdate",
                now.strftime("%y%m%d%H%M%SZ"),
                "-enddate",
                end.strftime("%y%m%d%H%M%SZ"),
                "-subj",
                subject,
            ]
        )
        return cert.read_text()


def _metadata(c: Community, *, name: str | None = None, genesis_digest: str) -> dict:
    return {
        "schema": FABRIC_SCHEMA,
        "community_id": c.community_id,
        "epoch": c.epoch,
        "secret_hex": c.secret.hex(),
        "roles": list(c.roles),
        "nid": c.nid,
        "name": name,
        "genesis_digest": genesis_digest,
    }


def _entry_digest(entry: dict) -> str:
    return sha256_hex(canonical_json(entry))


def _unsigned(entry: dict) -> dict:
    return {key: value for key, value in entry.items() if key != "signature"}


def _signed_entry(entry: dict, key: Path) -> dict:
    result = dict(entry)
    result["signature"] = sign(key, canonical_json(entry))
    return result


def create_community(home: Path, *, name: str | None = None, hosts: Sequence[str] = ()) -> Community:
    home = Path(home)
    if home.exists() and (community_exists(home) or any(home.iterdir())):
        raise FabricError(f"community home is not empty: {home}")
    home.mkdir(parents=True, exist_ok=True)
    ca_key, ca_cert = home / "ca.key", home / "ca.pem"
    node_key, node_cert = home / "node.key", home / "node.pem"
    _new_ca(ca_key, ca_cert)
    _generate_key(node_key)
    nid = node_id_for_key(node_key)
    community_id = secrets.token_hex(16)
    secret = secrets.token_bytes(32)
    roles = ("serve",)
    csr = _make_csr_for_key(node_key, nid)
    cert_pem = _issue_certificate(
        ca_key, ca_cert, csr, nid=nid, epoch=1, roles=roles, hosts=hosts, lifetime_s=86400
    )
    _write(node_cert, cert_pem.encode(), 0o644)
    community = Community(home, community_id, 1, secret, ca_key, ca_cert, node_key, node_cert, nid, roles)
    entry = _signed_entry(
        {
            "schema": FABRIC_SCHEMA,
            "community_id": community_id,
            "epoch": 1,
            "kind": "admit",
            "admitted_nid": nid,
            "admitted_cert_sha256": sha256_hex(_cert_der(cert_pem)),
            "roles": list(roles),
            "admitted_by": nid,
            "admitter_cert_pem": cert_pem,
            "issued_at": time.time(),
            "prev": None,
        },
        node_key,
    )
    _append_json(home / "admission.jsonl", entry)
    _write(home / "community.json", canonical_json(_metadata(community, name=name, genesis_digest=_entry_digest(entry))), 0o600)
    _write(home / "revocations.jsonl", b"", 0o600)
    return community


def load_community(home: Path | None = None) -> Community:
    root = Path(home) if home is not None else default_home()
    try:
        meta = json.loads((root / "community.json").read_text())
        if meta.get("schema") != FABRIC_SCHEMA:
            raise ValueError("wrong schema")
        community_id = meta["community_id"]
        epoch = meta["epoch"]
        secret = bytes.fromhex(meta["secret_hex"])
        roles = tuple(meta["roles"])
        recorded_nid = meta["nid"]
        if (
            not re.fullmatch(r"[0-9a-f]{32}", community_id)
            or not _HEX_64.fullmatch(meta["secret_hex"])
            or not isinstance(epoch, int)
            or isinstance(epoch, bool)
            or epoch < 1
        ):
            raise ValueError("invalid metadata")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise FabricError(f"not joined: {root}") from exc
    ca_key, ca_cert = root / "ca.key", root / "ca.pem"
    node_key, node_cert = root / "node.key", root / "node.pem"
    if not all(path.is_file() for path in (ca_key, ca_cert, node_key, node_cert)):
        raise FabricError(f"not joined: {root}")
    nid = node_id_for_key(node_key)
    if nid != recorded_nid or cert_nid(node_cert.read_text()) != nid:
        raise FabricError("node identity does not match community metadata")
    return Community(root, community_id, epoch, secret, ca_key, ca_cert, node_key, node_cert, nid, roles)


def community_exists(home: Path | None = None) -> bool:
    root = Path(home) if home is not None else default_home()
    return (root / "community.json").is_file()


def make_invite(c: Community) -> str:
    payload = canonical_json({"community_id": c.community_id, "epoch": c.epoch, "secret_hex": c.secret.hex()})
    return "mlx2inv1." + base64.urlsafe_b64encode(payload).decode().rstrip("=")


def parse_invite(token: str) -> dict:
    try:
        if not isinstance(token, str) or not token.startswith("mlx2inv1."):
            raise ValueError
        encoded = token.removeprefix("mlx2inv1.")
        if not encoded or not re.fullmatch(r"[A-Za-z0-9_-]+", encoded):
            raise ValueError
        decoded = base64.b64decode(encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True)
        value = json.loads(decoded)
        if set(value) != {"community_id", "epoch", "secret_hex"}:
            raise ValueError
        if not _HEX_64.fullmatch(value["secret_hex"]):
            raise ValueError
        if not re.fullmatch(r"[0-9a-f]{32}", value["community_id"]):
            raise ValueError
        if not isinstance(value["epoch"], int) or isinstance(value["epoch"], bool) or value["epoch"] < 1:
            raise ValueError
        return value
    except (UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise FabricError("invalid invite") from exc


def community_tag(secret: bytes, epoch: int) -> str:
    if not isinstance(epoch, int) or epoch < 0:
        raise FabricError("invalid epoch")
    return hmac.new(secret, b"mlx2-cm" + epoch.to_bytes(8, "big"), hashlib.sha256).hexdigest()[:16]


def make_csr(c_home: Path) -> tuple[Path, str]:
    home = Path(c_home)
    home.mkdir(parents=True, exist_ok=True)
    key = home / "node.key"
    if not key.exists():
        _generate_key(key)
    os.chmod(key, 0o600)
    return key, _make_csr_for_key(key, node_id_for_key(key))


def _join_hmac(label: bytes, secret: bytes, csr_pem: str, server_cert_sha256: str) -> str:
    if not _HEX_64.fullmatch(server_cert_sha256):
        raise FabricError("invalid server certificate digest")
    body = label + b"\0" + csr_pem.encode() + b"\0" + server_cert_sha256.encode()
    return hmac.new(secret, body, hashlib.sha256).hexdigest()


def join_proof(secret: bytes, csr_pem: str, server_cert_sha256: str) -> str:
    return _join_hmac(b"mlx2-join", secret, csr_pem, server_cert_sha256)


def join_ack_proof(secret: bytes, csr_pem: str, server_cert_sha256: str) -> str:
    return _join_hmac(b"mlx2-join-ok", secret, csr_pem, server_cert_sha256)


def sign(key_path: Path, data: bytes) -> str:
    signature = _run(["dgst", "-sha256", "-sign", str(key_path)], input_data=data)
    return base64.b64encode(signature).decode()


def verify(cert_pem: str, data: bytes, signature_b64: str, *, ca_cert: Path) -> bool:
    return _verify_at(cert_pem, data, signature_b64, ca_cert=ca_cert, attime=None)


def _verify_at(
    cert_pem: str,
    data: bytes,
    signature_b64: str,
    *,
    ca_cert: Path,
    attime: float | None,
) -> bool:
    try:
        if not isinstance(signature_b64, str):
            return False
        signature = base64.b64decode(signature_b64, validate=True)
        # Strict decoding still permits nonzero pad bits. Log digests
        # include this text, so reject byte-equivalent
        # aliases instead of giving one signed entry multiple identities.
        if base64.b64encode(signature).decode("ascii") != signature_b64:
            return False
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            cert = root / "cert.pem"
            public = root / "public.pem"
            sig = root / "signature.der"
            body = root / "body"
            cert.write_text(cert_pem)
            sig.write_bytes(signature)
            body.write_bytes(data)
            verify_args = [_OPENSSL, "verify", "-CAfile", str(ca_cert)]
            if attime is not None:
                verify_args.extend(["-attime", str(int(attime))])
            checked = subprocess.run([*verify_args, str(cert)], capture_output=True, check=False)
            if checked.returncode:
                return False
            public.write_bytes(_run(["x509", "-in", str(cert), "-pubkey", "-noout"]))
            checked = subprocess.run(
                [_OPENSSL, "dgst", "-sha256", "-verify", str(public), "-signature", str(sig), str(body)],
                capture_output=True,
                check=False,
            )
            return checked.returncode == 0
    except (FabricError, OSError, ValueError):
        return False


def _meta(c: Community) -> dict:
    try:
        return json.loads((c.home / "community.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise FabricError("invalid community metadata") from exc


def _admission_reason(c: Community, entry: dict, admitted: set[str], valid_digests: set[str]) -> str | None:
    required = {
        "schema", "community_id", "epoch", "kind", "admitted_nid", "admitted_cert_sha256",
        "roles", "admitted_by", "admitter_cert_pem", "issued_at", "prev", "signature",
    }
    if set(entry) != required or entry.get("schema") != FABRIC_SCHEMA or entry.get("kind") != "admit":
        return "invalid admission shape"
    if entry.get("community_id") != c.community_id or entry.get("epoch") != c.epoch:
        return "foreign community or epoch"
    if not _HEX_16.fullmatch(entry.get("admitted_nid", "")) or not _HEX_16.fullmatch(entry.get("admitted_by", "")):
        return "invalid nid"
    if not _HEX_64.fullmatch(entry.get("admitted_cert_sha256", "")):
        return "invalid certificate digest"
    if (
        not isinstance(entry.get("issued_at"), int | float)
        or isinstance(entry["issued_at"], bool)
        or not math.isfinite(entry["issued_at"])
    ):
        return "invalid issued_at"
    try:
        _clean_roles(entry["roles"])
        signer_nid = cert_nid(entry["admitter_cert_pem"])
    except (FabricError, TypeError):
        return "invalid signer certificate"
    if signer_nid != entry["admitted_by"]:
        return "signer nid mismatch"
    if not _verify_at(
        entry["admitter_cert_pem"],
        canonical_json(_unsigned(entry)),
        entry["signature"],
        ca_cert=c.ca_cert,
        attime=entry["issued_at"],
    ):
        return "invalid admission signature"
    digest = _entry_digest(entry)
    genesis = _meta(c).get("genesis_digest")
    is_genesis = digest == genesis
    if is_genesis:
        if entry["admitted_by"] != entry["admitted_nid"] or entry["prev"] is not None:
            return "invalid genesis"
    else:
        if entry["admitted_by"] not in admitted:
            return "admitter not admitted"
        if entry["prev"] is not None and entry["prev"] not in valid_digests:
            return "unknown admission predecessor"
    return None


def _valid_admissions(c: Community, entries: Iterable[dict]) -> tuple[list[dict], list[dict]]:
    valid, rejected = [], []
    admitted: set[str] = set()
    digests: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            rejected.append({"reason": "invalid admission shape"})
            continue
        try:
            digest = _entry_digest(entry)
            reason = _admission_reason(c, entry, admitted, digests)
        except (TypeError, ValueError):
            rejected.append({"reason": "invalid admission encoding"})
            continue
        if reason:
            rejected.append({"digest": digest, "reason": reason})
            continue
        valid.append(entry)
        digests.add(digest)
        admitted.add(entry["admitted_nid"])
    return valid, rejected


# Validation runs openssl once per entry, and check_peer consults both logs on
# every handshake, so validated views are memoised on the files' stat identity.
_VALIDATED: dict[tuple, list[dict]] = {}


def _stat_key(kind: str, c: Community, *names: str) -> tuple | None:
    key: list = [kind, str(c.home), c.community_id, c.epoch]
    for name in (*names, "community.json", "ca.pem"):
        try:
            stat = (c.home / name).stat()
        except OSError:
            return None
        key.append((name, stat.st_mtime_ns, stat.st_size, stat.st_ino))
    return tuple(key)


def _memoised(key: tuple | None, compute) -> list[dict]:
    if key is not None and key in _VALIDATED:
        return [dict(entry) for entry in _VALIDATED[key]]
    value = compute()
    if key is not None:
        _VALIDATED[key] = [dict(entry) for entry in value]
    return value


def admission_entries(c: Community) -> list[dict]:
    def compute() -> list[dict]:
        valid, _ = _valid_admissions(c, _read_jsonl(c.home / "admission.jsonl"))
        return valid

    return _memoised(_stat_key("admission", c, "admission.jsonl"), compute)


def merge_admission_log(c: Community, entries: Iterable[dict]) -> dict:
    current = admission_entries(c)
    known = {_entry_digest(entry) for entry in current}
    admitted = {entry["admitted_nid"] for entry in current}
    added = 0
    rejected = []
    for entry in entries:
        if not isinstance(entry, dict):
            rejected.append({"reason": "invalid admission shape"})
            continue
        try:
            digest = _entry_digest(entry)
        except (TypeError, ValueError):
            rejected.append({"reason": "invalid admission encoding"})
            continue
        if digest in known:
            continue
        try:
            reason = _admission_reason(c, entry, admitted, known)
        except (TypeError, ValueError):
            rejected.append({"digest": digest, "reason": "invalid admission encoding"})
            continue
        if reason:
            rejected.append({"digest": digest, "reason": reason})
            continue
        _append_json(c.home / "admission.jsonl", entry)
        known.add(digest)
        admitted.add(entry["admitted_nid"])
        added += 1
    return {"added": added, "rejected": rejected}


def admitted_nids(c: Community) -> set[str]:
    return {entry["admitted_nid"] for entry in admission_entries(c)}


def _last_admission_digest(c: Community) -> str | None:
    entries = admission_entries(c)
    return _entry_digest(entries[-1]) if entries else None


def admit(
    c: Community,
    csr_pem: str,
    *,
    roles=("serve",),
    hosts=(),
    lifetime_s=86400,
) -> dict:
    nid = _csr_nid(csr_pem)
    clean_roles = _clean_roles(roles)
    cert_pem = _issue_certificate(
        c.ca_key, c.ca_cert, csr_pem, nid=nid, epoch=c.epoch, roles=clean_roles, hosts=hosts, lifetime_s=lifetime_s
    )
    entry = _signed_entry(
        {
            "schema": FABRIC_SCHEMA,
            "community_id": c.community_id,
            "epoch": c.epoch,
            "kind": "admit",
            "admitted_nid": nid,
            "admitted_cert_sha256": sha256_hex(_cert_der(cert_pem)),
            "roles": list(clean_roles),
            "admitted_by": c.nid,
            "admitter_cert_pem": c.node_cert.read_text(),
            "issued_at": time.time(),
            "prev": _last_admission_digest(c),
        },
        c.node_key,
    )
    result = merge_admission_log(c, [entry])
    if result["added"] != 1:
        raise FabricError(f"locally created admission was rejected: {result['rejected']}")
    return {
        "community_id": c.community_id,
        "epoch": c.epoch,
        "secret_hex": c.secret.hex(),
        "ca_cert_pem": c.ca_cert.read_text(),
        "ca_key_pem": c.ca_key.read_text(),
        "node_cert_pem": cert_pem,
        "admission_log": admission_entries(c),
        "revocations": revocation_entries(c),
    }


def install_join_bundle(home: Path, bundle: dict, key_path: Path) -> Community:
    required = {
        "community_id", "epoch", "secret_hex", "ca_cert_pem", "ca_key_pem", "node_cert_pem",
        "admission_log", "revocations",
    }
    if not isinstance(bundle, dict) or set(bundle) != required:
        raise FabricError("invalid join bundle")
    try:
        secret = bytes.fromhex(bundle["secret_hex"])
        nid = cert_nid(bundle["node_cert_pem"])
        if node_id_for_key(Path(key_path)) != nid or len(secret) != 32:
            raise ValueError
    except (FabricError, TypeError, ValueError) as exc:
        raise FabricError("join bundle does not match node key") from exc
    home = Path(home)
    with tempfile.TemporaryDirectory(dir=home.parent if home.parent.exists() else None) as raw:
        stage = Path(raw) / "community"
        stage.mkdir()
        _write(stage / "ca.key", bundle["ca_key_pem"].encode(), 0o600)
        _write(stage / "ca.pem", bundle["ca_cert_pem"].encode(), 0o644)
        _write(stage / "node.key", Path(key_path).read_bytes(), 0o600)
        _write(stage / "node.pem", bundle["node_cert_pem"].encode(), 0o644)
        if _public_der_from_key(stage / "ca.key") != _public_der_from_cert(stage / "ca.pem"):
            raise FabricError("join bundle CA key does not match CA certificate")
        logs = bundle["admission_log"]
        if not isinstance(logs, list) or not logs:
            raise FabricError("join bundle has no admission genesis")
        meta = {
            "schema": FABRIC_SCHEMA,
            "community_id": bundle["community_id"],
            "epoch": bundle["epoch"],
            "secret_hex": bundle["secret_hex"],
            "roles": next((entry["roles"] for entry in logs if entry.get("admitted_nid") == nid), ["serve"]),
            "nid": nid,
            "name": None,
            "genesis_digest": _entry_digest(logs[0]),
        }
        _write(stage / "community.json", canonical_json(meta), 0o600)
        _write(stage / "admission.jsonl", b"", 0o600)
        _write(stage / "revocations.jsonl", b"", 0o600)
        candidate = load_community(stage)
        merged = merge_admission_log(candidate, logs)
        if merged["rejected"] or nid not in admitted_nids(candidate):
            raise FabricError(f"invalid admission log in join bundle: {merged['rejected']}")
        node_digest = sha256_hex(_cert_der(bundle["node_cert_pem"]))
        if not any(
            entry["admitted_nid"] == nid and entry["admitted_cert_sha256"] == node_digest
            for entry in admission_entries(candidate)
        ):
            raise FabricError("join certificate is not bound to its admission entry")
        revoked = merge_revocations(candidate, bundle["revocations"])
        if revoked["rejected"]:
            raise FabricError(f"invalid revocations in join bundle: {revoked['rejected']}")
        home.mkdir(parents=True, exist_ok=True)
        for name in ("ca.key", "ca.pem", "node.key", "node.pem", "community.json", "admission.jsonl", "revocations.jsonl"):
            source = stage / name
            _write(home / name, source.read_bytes(), source.stat().st_mode & 0o777)
    return load_community(home)


def _revocation_reason(c: Community, entry: dict, admitted: set[str], revoked: set[str]) -> str | None:
    required = {
        "schema", "community_id", "kind", "nid", "reason", "revoked_by", "revoker_cert_pem", "issued_at", "signature",
    }
    if set(entry) != required or entry.get("schema") != FABRIC_SCHEMA or entry.get("kind") != "revoke":
        return "invalid revocation shape"
    if entry.get("community_id") != c.community_id:
        return "foreign community"
    if not _HEX_16.fullmatch(entry.get("nid", "")) or not _HEX_16.fullmatch(entry.get("revoked_by", "")):
        return "invalid nid"
    if not isinstance(entry.get("reason"), str) or not entry["reason"]:
        return "invalid revocation reason"
    if (
        not isinstance(entry.get("issued_at"), int | float)
        or isinstance(entry["issued_at"], bool)
        or not math.isfinite(entry["issued_at"])
    ):
        return "invalid issued_at"
    try:
        signer_nid = cert_nid(entry["revoker_cert_pem"])
    except (FabricError, TypeError):
        return "invalid signer certificate"
    if signer_nid != entry["revoked_by"]:
        return "signer nid mismatch"
    if entry["revoked_by"] not in admitted:
        return "revoker not admitted"
    if entry["revoked_by"] in revoked:
        return "revoker is revoked"
    if not _verify_at(
        entry["revoker_cert_pem"],
        canonical_json(_unsigned(entry)),
        entry["signature"],
        ca_cert=c.ca_cert,
        attime=entry["issued_at"],
    ):
        return "invalid revocation signature"
    return None


def revocation_entries(c: Community) -> list[dict]:
    def compute() -> list[dict]:
        admitted = admitted_nids(c)
        revoked: set[str] = set()
        valid = []
        for entry in _read_jsonl(c.home / "revocations.jsonl"):
            if _revocation_reason(c, entry, admitted, revoked) is None:
                valid.append(entry)
                revoked.add(entry["nid"])
        return valid

    key = _stat_key("revocation", c, "revocations.jsonl", "admission.jsonl")
    return _memoised(key, compute)


def merge_revocations(c: Community, entries: Iterable[dict]) -> dict:
    current = revocation_entries(c)
    known = {_entry_digest(entry) for entry in current}
    revoked = {entry["nid"] for entry in current}
    admitted = admitted_nids(c)
    added = 0
    rejected = []
    for entry in entries:
        if not isinstance(entry, dict):
            rejected.append({"reason": "invalid revocation shape"})
            continue
        try:
            digest = _entry_digest(entry)
        except (TypeError, ValueError):
            rejected.append({"reason": "invalid revocation encoding"})
            continue
        if digest in known:
            continue
        try:
            reason = _revocation_reason(c, entry, admitted, revoked)
        except (TypeError, ValueError):
            rejected.append({"digest": digest, "reason": "invalid revocation encoding"})
            continue
        if reason:
            rejected.append({"digest": digest, "reason": reason})
            continue
        _append_json(c.home / "revocations.jsonl", entry)
        known.add(digest)
        revoked.add(entry["nid"])
        added += 1
    return {"added": added, "rejected": rejected}


def revoke(c: Community, nid: str, *, reason: str) -> dict:
    if not _HEX_16.fullmatch(nid) or not reason:
        raise FabricError("invalid revocation")
    entry = _signed_entry(
        {
            "schema": FABRIC_SCHEMA,
            "community_id": c.community_id,
            "kind": "revoke",
            "nid": nid,
            "reason": reason,
            "revoked_by": c.nid,
            "revoker_cert_pem": c.node_cert.read_text(),
            "issued_at": time.time(),
        },
        c.node_key,
    )
    result = merge_revocations(c, [entry])
    if result["added"] != 1:
        raise FabricError(f"locally created revocation was rejected: {result['rejected']}")
    return entry


def revoked_nids(c: Community) -> set[str]:
    return {entry["nid"] for entry in revocation_entries(c)}


def _decode_der(cert_der: bytes) -> tuple[str, dict]:
    pem = ssl.DER_cert_to_PEM_cert(cert_der)
    with tempfile.NamedTemporaryFile(suffix=".pem") as temporary:
        temporary.write(pem.encode())
        temporary.flush()
        decoded = ssl._ssl._test_decode_cert(temporary.name)  # type: ignore[attr-defined]
    return pem, decoded


def _subject_values(decoded: dict, key: str) -> list[str]:
    return [value for rdn in decoded.get("subject", ()) for name, value in rdn if name == key]


def check_peer(c: Community, peer_cert_der: bytes | None, *, expected_nid: str | None = None) -> PeerVerdict:
    if not peer_cert_der:
        return PeerVerdict(False, None, "no client certificate")
    try:
        pem, decoded = _decode_der(peer_cert_der)
        nid = cert_nid(pem)
    except (FabricError, OSError, ssl.SSLError, ValueError):
        return PeerVerdict(False, None, "invalid peer certificate")
    common_names = _subject_values(decoded, "commonName")
    if common_names != [nid]:
        return PeerVerdict(False, nid, "certificate CN does not match key-derived nid")
    units = _subject_values(decoded, "organizationalUnitName")
    marker = f"epoch={c.epoch};"
    if len(units) != 1 or not units[0].startswith(marker):
        return PeerVerdict(False, nid, "foreign epoch")
    try:
        not_before = ssl.cert_time_to_seconds(decoded["notBefore"])
        not_after = ssl.cert_time_to_seconds(decoded["notAfter"])
    except (KeyError, ValueError):
        return PeerVerdict(False, nid, "invalid certificate validity")
    if not_after - not_before > _MAX_CERT_LIFETIME_S + 2:
        return PeerVerdict(False, nid, "certificate lifetime exceeds 7 days")
    now = time.time()
    if now > not_after:
        return PeerVerdict(False, nid, "certificate expired")
    if now < not_before:
        return PeerVerdict(False, nid, "certificate not yet valid")
    with tempfile.NamedTemporaryFile(suffix=".pem") as cert_file:
        cert_file.write(pem.encode())
        cert_file.flush()
        checked = subprocess.run(
            [_OPENSSL, "verify", "-CAfile", str(c.ca_cert), cert_file.name], capture_output=True, check=False
        )
    if checked.returncode:
        return PeerVerdict(False, nid, "certificate not signed by community CA")
    if nid in revoked_nids(c):
        return PeerVerdict(False, nid, "revoked nid")
    if nid not in admitted_nids(c):
        return PeerVerdict(False, nid, "nid not in admission log")
    if expected_nid is not None and nid != expected_nid:
        return PeerVerdict(False, nid, "expected nid mismatch")
    return PeerVerdict(True, nid, "ok")


def _certificate_times(path: Path) -> tuple[float, float]:
    decoded = ssl._ssl._test_decode_cert(str(path))  # type: ignore[attr-defined]
    return ssl.cert_time_to_seconds(decoded["notBefore"]), ssl.cert_time_to_seconds(decoded["notAfter"])


def ensure_fresh_certificate(c: Community, *, hosts: Sequence[str] = (), lifetime_s: int = 86400) -> Path:
    if lifetime_s > _MAX_CERT_LIFETIME_S or lifetime_s <= 0:
        raise FabricError("certificate lifetime must be between 1 second and 7 days")
    fresh = False
    if c.node_cert.exists():
        try:
            start, end = _certificate_times(c.node_cert)
            fresh = time.time() < end and end - time.time() >= (end - start) / 3
        except (OSError, KeyError, ValueError, ssl.SSLError):
            pass
    if not fresh:
        csr = _make_csr_for_key(c.node_key, c.nid)
        cert = _issue_certificate(
            c.ca_key,
            c.ca_cert,
            csr,
            nid=c.nid,
            epoch=c.epoch,
            roles=c.roles,
            hosts=hosts,
            lifetime_s=lifetime_s,
        )
        _write(c.node_cert, cert.encode(), 0o644)
    return c.node_cert


def server_context(c: Community) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.load_cert_chain(c.node_cert, c.node_key)
    context.load_verify_locations(cafile=c.ca_cert)
    context.verify_mode = ssl.CERT_OPTIONAL
    return context


def client_context(c: Community) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.check_hostname = False
    context.verify_mode = ssl.CERT_REQUIRED
    context.load_verify_locations(cafile=c.ca_cert)
    context.load_cert_chain(c.node_cert, c.node_key)
    return context


def join_client_context(server_cert_check=None) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    if server_cert_check is not None and not callable(server_cert_check):
        raise FabricError("server_cert_check must be callable")
    return context

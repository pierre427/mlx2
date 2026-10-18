from __future__ import annotations

import copy
import hashlib
import socket
import ssl
import threading
import time
from pathlib import Path

import pytest

import mlx2.fabric.community as community_module
from mlx2.fabric import FabricError
from mlx2.fabric.community import (
    FABRIC_SCHEMA,
    admission_entries,
    admit,
    admitted_nids,
    canonical_json,
    cert_nid,
    check_peer,
    client_context,
    create_community,
    ensure_fresh_certificate,
    install_join_bundle,
    join_ack_proof,
    join_client_context,
    join_proof,
    load_community,
    make_csr,
    make_invite,
    merge_admission_log,
    merge_revocations,
    node_id_for_key,
    parse_invite,
    revocation_entries,
    revoke,
    server_context,
    sign,
)


def _der(path: Path) -> bytes:
    return ssl.PEM_cert_to_DER_cert(path.read_text())


def _join(member, home: Path, *, lifetime_s: int = 86400):
    key, csr = make_csr(home)
    bundle = admit(member, csr, hosts=("127.0.0.1",), lifetime_s=lifetime_s)
    return install_join_bundle(home, bundle, key), csr


def _mint(c, home: Path, *, epoch: int | None = None, lifetime_s: int = 86400):
    key, csr = make_csr(home)
    nid = node_id_for_key(key)
    cert = community_module._issue_certificate(
        c.ca_key,
        c.ca_cert,
        csr,
        nid=nid,
        epoch=c.epoch if epoch is None else epoch,
        roles=("serve",),
        hosts=("127.0.0.1",),
        lifetime_s=lifetime_s,
    )
    path = home / "node.pem"
    path.write_text(cert)
    return key, path, nid, csr


def _tls_pair(server, client):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    result = {}

    def serve():
        try:
            raw, _ = listener.accept()
            with raw, server.wrap_socket(raw, server_side=True) as stream:
                result["server_peer"] = stream.getpeercert(binary_form=True)
                stream.sendall(b"ok")
        except OSError as exc:  # TLS errors are part of the test result.
            result["server_error"] = exc
        finally:
            listener.close()

    thread = threading.Thread(target=serve)
    thread.start()
    try:
        with (
            socket.create_connection(listener.getsockname(), timeout=5) as raw,
            client.wrap_socket(raw, server_hostname="localhost") as stream,
        ):
            result["client_peer"] = stream.getpeercert(binary_form=True)
            result["payload"] = stream.recv(2)
    except OSError as exc:
        result["client_error"] = exc
    thread.join(timeout=5)
    assert not thread.is_alive()
    return result


def test_create_invite_admit_install_and_merge_round_trip(tmp_path):
    first = create_community(tmp_path / "first", name="cpu", hosts=("127.0.0.1",))
    assert parse_invite(make_invite(first)) == {
        "community_id": first.community_id,
        "epoch": first.epoch,
        "secret_hex": first.secret.hex(),
    }

    second, _ = _join(first, tmp_path / "second")
    assert load_community(first.home).nid == first.nid
    assert load_community(second.home).nid == second.nid
    assert first.nid != second.nid

    merge_admission_log(first, admission_entries(second))
    merge_admission_log(second, admission_entries(first))
    expected = {first.nid, second.nid}
    assert admitted_nids(first) == expected
    assert admitted_nids(second) == expected
    assert first.ca_key.stat().st_mode & 0o777 == 0o600
    assert second.node_key.stat().st_mode & 0o777 == 0o600
    assert second.node_cert.stat().st_mode & 0o777 == 0o644


def test_real_mutual_tls_and_peer_checks(tmp_path):
    first = create_community(tmp_path / "first", hosts=("127.0.0.1",))
    second, _ = _join(first, tmp_path / "second")
    result = _tls_pair(server_context(first), client_context(second))

    assert result["payload"] == b"ok"
    assert check_peer(first, result["server_peer"]) == community_module.PeerVerdict(True, second.nid, "ok")
    assert check_peer(second, result["client_peer"]) == community_module.PeerVerdict(True, first.nid, "ok")


def test_tls_rejects_foreign_ca_and_optional_server_reports_no_client_cert(tmp_path):
    first = create_community(tmp_path / "first", hosts=("127.0.0.1",))
    foreign = create_community(tmp_path / "foreign", hosts=("127.0.0.1",))
    rejected = _tls_pair(server_context(first), client_context(foreign))
    assert isinstance(rejected.get("server_error") or rejected.get("client_error"), ssl.SSLError)

    anonymous = _tls_pair(server_context(first), join_client_context())
    assert anonymous["payload"] == b"ok"
    verdict = check_peer(first, anonymous["server_peer"])
    assert not verdict.ok and verdict.reason == "no client certificate"


def test_peer_refusals_for_revocation_unadmitted_epoch_and_expected_nid(tmp_path):
    first = create_community(tmp_path / "first")
    second, _ = _join(first, tmp_path / "second")

    revocation = revoke(first, second.nid, reason="test")
    assert merge_revocations(second, [revocation]) == {"added": 1, "rejected": []}
    assert check_peer(second, _der(second.node_cert)).reason == "revoked nid"

    _, unadmitted_cert, _, _ = _mint(first, tmp_path / "unadmitted")
    assert check_peer(first, _der(unadmitted_cert)).reason == "nid not in admission log"

    _, wrong_epoch_cert, _, _ = _mint(first, tmp_path / "epoch", epoch=first.epoch + 1)
    assert check_peer(first, _der(wrong_epoch_cert)).reason == "foreign epoch"
    assert check_peer(first, _der(first.node_cert), expected_nid=second.nid).reason == "expected nid mismatch"


def test_peer_refuses_expired_and_overlong_certificates(tmp_path, monkeypatch):
    first = create_community(tmp_path / "first")
    _, expired, _, _ = _mint(first, tmp_path / "expired", lifetime_s=1)
    time.sleep(1.1)
    assert check_peer(first, _der(expired)).reason == "certificate expired"

    monkeypatch.setattr(community_module, "_MAX_CERT_LIFETIME_S", 9 * 86400)
    _, overlong, _, _ = _mint(first, tmp_path / "overlong", lifetime_s=8 * 86400)
    monkeypatch.setattr(community_module, "_MAX_CERT_LIFETIME_S", 7 * 86400)
    assert check_peer(first, _der(overlong)).reason == "certificate lifetime exceeds 7 days"


def test_forged_admissions_are_rejected(tmp_path):
    first = create_community(tmp_path / "first")
    second, _ = _join(first, tmp_path / "second")
    _, csr = make_csr(tmp_path / "third")
    admit(first, csr)
    valid = admission_entries(first)[-1]

    tampered = copy.deepcopy(valid)
    raw = bytearray(tampered["signature"].encode())
    raw[-2] = ord("A") if raw[-2] != ord("A") else ord("B")
    tampered["signature"] = raw.decode()
    result = merge_admission_log(second, [tampered])
    assert result["added"] == 0
    assert result["rejected"][0]["reason"] == "invalid admission signature"

    rogue_key, rogue_cert, rogue_nid, _ = _mint(first, tmp_path / "rogue")
    target_key, _ = make_csr(tmp_path / "target")
    unsigned = {
        "schema": FABRIC_SCHEMA,
        "community_id": first.community_id,
        "epoch": first.epoch,
        "kind": "admit",
        "admitted_nid": node_id_for_key(target_key),
        "admitted_cert_sha256": hashlib.sha256(b"target").hexdigest(),
        "roles": ["serve"],
        "admitted_by": rogue_nid,
        "admitter_cert_pem": rogue_cert.read_text(),
        "issued_at": time.time(),
        "prev": community_module._entry_digest(admission_entries(second)[-1]),
    }
    forged = {**unsigned, "signature": sign(rogue_key, canonical_json(unsigned))}
    result = merge_admission_log(second, [forged])
    assert result["added"] == 0
    assert result["rejected"][0]["reason"] == "admitter not admitted"


def test_revocation_by_unadmitted_signer_is_rejected(tmp_path):
    first = create_community(tmp_path / "first")
    second, _ = _join(first, tmp_path / "second")
    rogue_key, rogue_cert, rogue_nid, _ = _mint(first, tmp_path / "rogue")
    unsigned = {
        "schema": FABRIC_SCHEMA,
        "community_id": first.community_id,
        "kind": "revoke",
        "nid": second.nid,
        "reason": "forged",
        "revoked_by": rogue_nid,
        "revoker_cert_pem": rogue_cert.read_text(),
        "issued_at": time.time(),
    }
    forged = {**unsigned, "signature": sign(rogue_key, canonical_json(unsigned))}
    result = merge_revocations(second, [forged])
    assert result["added"] == 0
    assert result["rejected"][0]["reason"] == "revoker not admitted"
    assert revocation_entries(second) == []


def test_join_proofs_bind_label_csr_and_server_certificate(tmp_path):
    community = create_community(tmp_path / "community")
    _, csr = make_csr(tmp_path / "joiner")
    _, other_csr = make_csr(tmp_path / "other")
    digest = hashlib.sha256(_der(community.node_cert)).hexdigest()
    other_digest = hashlib.sha256(b"other server certificate").hexdigest()

    proof = join_proof(community.secret, csr, digest)
    assert proof != join_ack_proof(community.secret, csr, digest)
    assert proof != join_proof(community.secret, other_csr, digest)
    assert proof != join_proof(community.secret, csr, other_digest)
    for garbage in ("", "mlx2inv1.", "garbage", "mlx2inv1.%%%%", "mlx2inv1.e30"):
        with pytest.raises(FabricError, match="invalid invite"):
            parse_invite(garbage)


def test_ensure_fresh_certificate_renews_below_one_third_with_same_nid(tmp_path, monkeypatch):
    community = create_community(tmp_path / "community")
    csr = community_module._make_csr_for_key(community.node_key, community.nid)
    short_cert = community_module._issue_certificate(
        community.ca_key,
        community.ca_cert,
        csr,
        nid=community.nid,
        epoch=community.epoch,
        roles=community.roles,
        hosts=(),
        lifetime_s=120,
    )
    community.node_cert.write_text(short_cert)
    before = community.node_cert.read_bytes()
    start, end = community_module._certificate_times(community.node_cert)
    monkeypatch.setattr(community_module.time, "time", lambda: end - (end - start) / 4)

    ensure_fresh_certificate(community, lifetime_s=86400)

    assert community.node_cert.read_bytes() != before
    assert cert_nid(community.node_cert.read_text()) == community.nid

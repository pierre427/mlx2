from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
import urllib.request
from pathlib import Path

from . import FabricError
from .community import (
    community_tag,
    create_community,
    default_home,
    install_join_bundle,
    join_ack_proof,
    join_client_context,
    join_proof,
    load_community,
    make_csr,
    make_invite,
    parse_invite,
    revoke,
)
from .discovery import Browser, allowed_addresses, connect_https
from .node import FabricNode
from .registry import pid_alive


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlx2-node")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "invite", "join", "run", "status", "revoke", "link"):
        item = sub.add_parser(name)
        item.add_argument("--home", type=Path, default=default_home())
        if name == "init":
            item.add_argument("--name")
        elif name == "join":
            item.add_argument("invite")
            item.add_argument("--via")
            item.add_argument("--timeout", type=float, default=15)
        elif name == "run":
            item.add_argument("--front-port", type=int, default=8290)
            item.add_argument("--fabric-port", type=int, default=8291)
            item.add_argument("--include-loopback", action="store_true")
            item.add_argument("--local-server", action="append", default=[])
            item.add_argument("--no-zeroconf", action="store_true")
            item.add_argument("--peers-file", type=Path)
            item.add_argument("--link-budget-ms", type=float, default=5)
            item.add_argument("--local-poll-s", type=float, default=5)
        elif name == "status":
            item.add_argument("--front-port", type=int, default=8290)
        elif name == "revoke":
            item.add_argument("nid")
            item.add_argument("--reason", required=True)
        elif name == "link":
            item.add_argument("nid")
    return parser


def _member(invite: dict, via: str | None, timeout: float) -> tuple[tuple[str, ...], int]:
    if via:
        host, raw_port = via.rsplit(":", 1)
        return (host,), int(raw_port)
    browser = Browser(
        tag=community_tag(bytes.fromhex(invite["secret_hex"]), invite["epoch"])
    )
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            candidates = browser.candidates()
            if candidates:
                return candidates[0].addresses, candidates[0].port
            time.sleep(0.1)
    finally:
        browser.close()
    raise FabricError("no community member discovered")


def _join(args) -> None:
    invite = parse_invite(args.invite)
    key, csr = make_csr(args.home)
    addresses, port = _member(invite, args.via, args.timeout)
    conn, _ = connect_https(
        addresses,
        port,
        context=join_client_context(),
        local_addresses=allowed_addresses(include_loopback=True),
        connect_timeout_s=min(2.0, args.timeout),
    )
    conn.timeout = args.timeout
    if conn.sock is not None:
        conn.sock.settimeout(args.timeout)
    cert_digest = hashlib.sha256(conn.sock.getpeercert(binary_form=True)).hexdigest()
    proof = join_proof(bytes.fromhex(invite["secret_hex"]), csr, cert_digest)
    conn.request(
        "POST",
        "/fabric/join",
        body=json.dumps({"csr_pem": csr, "proof": proof}),
        headers={"Content-Type": "application/json"},
    )
    response = conn.getresponse()
    value = json.loads(response.read())
    conn.close()
    if response.status != 200:
        raise FabricError(value.get("error", f"join HTTP {response.status}"))
    expected = join_ack_proof(bytes.fromhex(invite["secret_hex"]), csr, cert_digest)
    import hmac

    if not hmac.compare_digest(value.get("ack", ""), expected):
        raise FabricError("invalid join acknowledgement")
    community = install_join_bundle(args.home, value["bundle"], key)
    print(community.nid)


def _live_pid(path: Path) -> bool:
    try:
        pid = int(path.read_text())
    except (OSError, ValueError):
        return False
    return pid_alive(pid)


def _run(args) -> None:
    pid_path = args.home / "node.pid"
    if _live_pid(pid_path):
        raise FabricError(
            f"fabric node already running: pid {pid_path.read_text().strip()}"
        )
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()))
    node = FabricNode(
        args.home,
        front_port=args.front_port,
        fabric_port=args.fabric_port,
        include_loopback=args.include_loopback,
        extra_local_servers=args.local_server,
        peers_file=args.peers_file,
        use_zeroconf=not args.no_zeroconf,
        link_budget_ms=args.link_budget_ms,
        local_poll_s=args.local_poll_s,
    )
    stopped = False

    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True

    old_int = signal.signal(signal.SIGINT, stop)
    old_term = signal.signal(signal.SIGTERM, stop)
    try:
        node.start()
        while not stopped:
            time.sleep(0.2)
    finally:
        node.close()
        pid_path.unlink(missing_ok=True)
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "init":
            community = create_community(args.home, name=args.name)
            print(make_invite(community))
            print(community.nid)
        elif args.command == "invite":
            print(make_invite(load_community(args.home)))
        elif args.command == "join":
            _join(args)
        elif args.command == "run":
            _run(args)
        elif args.command == "status":
            with urllib.request.urlopen(
                f"http://127.0.0.1:{args.front_port}/v1/fabric/nodes", timeout=5
            ) as response:
                data = json.load(response)["data"]
            for item in data:
                print(
                    f"{item['nid']}\t{item['state']}\t{item.get('hostname') or '-'}\t{','.join(item.get('models', [])) or '-'}"
                )
        elif args.command == "revoke":
            revoke(load_community(args.home), args.nid, reason=args.reason)
        elif args.command == "link":
            paths = sorted((args.home / "receipts").glob(f"link-{args.nid}-*.json"))
            if not paths:
                raise FabricError("no stored link receipt")
            print(paths[-1].read_text())
        return 0
    except (FabricError, OSError, ValueError) as exc:
        print(f"mlx2-node: {exc}", file=sys.stderr)
        return 1

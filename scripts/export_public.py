#!/usr/bin/env python3
"""Export a scrubbed snapshot of a git ref and optionally sync it to a public repo.

The public mirror carries no history from the private tree: each sync is one
commit ("Sync from mlx2 <ref> @ <sha>") on top of the mirror's own history.
Only tracked files are exported (``git archive``), so ignored and untracked
files never leave the machine.

Scrubbing is derived from the local environment at run time, so this script
itself names no user, host or path:

* the home directory becomes ``~``;
* per-user macOS temp directories (``/var/folders/.../T``) become ``$TMPDIR``;
* the machine's host names become ``<host>``;
* ``com.<user>.`` launchd labels become ``com.example.``;
* the private git remote's host becomes ``<private-git-host>``;
* any remaining bare login name becomes ``user``.

Hidden paths (other than ``.gitignore``) and ``*.pid`` files are dropped. After
scrubbing, the export is scanned for every scrubbed token plus common
credential shapes; any hit aborts before anything is pushed.

Usage:
    python scripts/export_public.py --out /tmp/mlx2-export            # export + scan only
    python scripts/export_public.py --push git@github.com:OWNER/mlx2.git
"""

from __future__ import annotations

import argparse
import getpass
import io
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

KEEP_HIDDEN = {".gitignore"}
DROP_SUFFIXES = {".pid"}
CREDENTIAL_PATTERNS = [
    r"ghp_[A-Za-z0-9]{20,}",
    r"github_pat_[A-Za-z0-9_]{20,}",
    r"hf_[A-Za-z0-9]{30,}",
    r"sk-(ant-|proj-|or-)?[A-Za-z0-9_-]{30,}",
    r"AKIA[0-9A-Z]{16}",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    r"xox[bpas]-[A-Za-z0-9-]{10,}",
    r"AIza[0-9A-Za-z_-]{35}",
]


def run(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def optional(*args: str) -> str:
    try:
        return run(*args)
    except (OSError, subprocess.CalledProcessError):
        return ""


def environment_tokens(repo: Path) -> tuple[list[tuple[re.Pattern[str], str]], list[str]]:
    """Return (ordered substitutions, literal tokens that must not survive)."""
    home = str(Path.home())
    user = getpass.getuser()
    hosts = {socket.gethostname(), optional("scutil", "--get", "LocalHostName"),
             optional("scutil", "--get", "ComputerName"), optional("hostname")}
    hosts = {h for host in hosts if host for h in (host, host.removesuffix(".local"))}
    remote_hosts = set()
    for url in optional("git", "-C", str(repo), "remote", "-v").split():
        match = re.match(r"^(?:ssh://)?(?:[^@/]+@)?(\d+\.\d+\.\d+\.\d+|[^:/]+\.(?:lan|local|internal))[:/]", url)
        if match:
            remote_hosts.add(match.group(1))

    subs: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"/(?:private/)?var/folders/[^/\s\"']+/[^/\s\"']+/T\b"), "$TMPDIR"),
        (re.compile(re.escape("/private" + home)), "~"),
        (re.compile(re.escape(home) + r"(?![A-Za-z0-9_-])"), "~"),
    ]
    for host in sorted(hosts, key=len, reverse=True):
        subs.append((re.compile(re.escape(host), re.IGNORECASE), "<host>"))
    for host in remote_hosts:
        subs.append((re.compile(re.escape(host)), "<private-git-host>"))
    subs.append((re.compile(r"\bcom\." + re.escape(user) + r"\."), "com.example."))
    subs.append((re.compile(r"\b" + re.escape(user) + r"\b"), "user"))
    forbidden = [home, user, *hosts, *remote_hosts]
    return subs, forbidden


def export(repo: Path, ref: str, out: Path) -> tuple[str, int]:
    sha = run("git", "-C", str(repo), "rev-parse", ref)
    archive = subprocess.run(
        ["git", "-C", str(repo), "archive", "--format=tar", sha], check=True, capture_output=True
    ).stdout
    subs, _ = environment_tokens(repo)
    scrubbed = 0
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            if any(p.startswith(".") and p not in KEEP_HIDDEN for p in parts):
                continue
            if Path(member.name).suffix in DROP_SUFFIXES:
                continue
            data = tar.extractfile(member).read()
            if b"\0" not in data:
                try:
                    text = data.decode("utf-8")
                except UnicodeDecodeError:
                    text = None
                if text is not None:
                    new = text
                    for pattern, replacement in subs:
                        new = pattern.sub(replacement, new)
                    if new != text:
                        scrubbed += 1
                        data = new.encode("utf-8")
            target = out / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            target.chmod(member.mode & 0o777)
    return sha, scrubbed


def scan(repo: Path, out: Path) -> list[str]:
    _, forbidden = environment_tokens(repo)
    literal = [re.compile(re.escape(t).encode(), re.IGNORECASE) for t in forbidden if t]
    creds = [re.compile(p.encode()) for p in CREDENTIAL_PATTERNS]
    hits = []
    for path in sorted(out.rglob("*")):
        if not path.is_file():
            continue
        data = path.read_bytes()
        for pattern in literal + creds:
            match = pattern.search(data)
            if match:
                hits.append(f"{path.relative_to(out)}: {match.group(0)[:12].decode(errors='replace')}...")
                break
    return hits


def push(repo: Path, out: Path, remote: str, sha: str, ref: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mirror = Path(tmp) / "mirror"
        cloned = subprocess.run(["git", "clone", "-q", remote, str(mirror)], check=False, capture_output=True, text=True)
        if cloned.returncode != 0:
            raise SystemExit(f"clone failed: {cloned.stderr.strip()}")
        run("git", "checkout", "-q", "-B", "main", cwd=mirror)
        for child in mirror.iterdir():
            if child.name != ".git":
                shutil.rmtree(child) if child.is_dir() else child.unlink()
        shutil.copytree(out, mirror, dirs_exist_ok=True)
        run("git", "add", "-A", cwd=mirror)
        if not run("git", "status", "--porcelain", cwd=mirror):
            print("mirror already matches; nothing to push")
            return
        login = run("gh", "api", "user", "--jq", ".login")
        uid = run("gh", "api", "user", "--jq", ".id")
        name = optional("git", "-C", str(repo), "config", "user.name") or login
        email = f"{uid}+{login}@users.noreply.github.com"
        run(
            "git", "-c", f"user.name={name}", "-c", f"user.email={email}",
            "commit", "-q", "-m", f"Sync from mlx2 {ref} @ {sha[:12]}",
            cwd=mirror,
        )
        run("git", "push", "-q", "origin", "main", cwd=mirror)
        print(f"pushed {sha[:12]} to {remote}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--ref", default="main")
    parser.add_argument("--out", type=Path, help="export directory (default: a temporary directory)")
    parser.add_argument("--push", metavar="REMOTE", help="public repo to sync the export to")
    args = parser.parse_args()

    tmp = None
    out = args.out
    if out is None:
        tmp = tempfile.TemporaryDirectory()
        out = Path(tmp.name) / "export"
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)

    sha, scrubbed = export(args.repo, args.ref, out)
    files = sum(1 for p in out.rglob("*") if p.is_file())
    print(f"exported {args.ref} @ {sha[:12]}: {files} files, {scrubbed} scrubbed -> {out}")
    hits = scan(args.repo, out)
    if hits:
        print(f"scan found {len(hits)} leak(s); refusing to publish:", file=sys.stderr)
        for hit in hits[:50]:
            print(f"  {hit}", file=sys.stderr)
        raise SystemExit(1)
    print("scan clean")
    if args.push:
        push(args.repo, out, args.push, sha, args.ref)
    if tmp is not None:
        tmp.cleanup()


if __name__ == "__main__":
    main()

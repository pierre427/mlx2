#!/usr/bin/env python3
"""Prepare an explicitly allowlisted, code-only snapshot for local review.

This tool never publishes. Start from the current public repository head in an
isolated checkout, apply only reviewed exported paths, and inspect the staged
diff before a separate publication. Private qualification data, logs, model
assets and history are not publication inputs.

Usage:
    python scripts/export_public.py --out /tmp/mlx2-export --include src/mlx2/cli.py

Each --include names one exact tracked source path; directories and globs are
refused. Environment scrubbing and credential scanning are defense in depth,
not proof that arbitrary private content is safe to publish.
"""

from __future__ import annotations

import argparse
import getpass
import io
import re
import socket
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

CODE_ROOTS = {"src", "scripts", "tests"}
ROOT_FILES = {".gitignore", "LICENSE", "NOTICE", "README.md", "pyproject.toml"}
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


def _selected_paths(repo: Path, sha: str, include) -> tuple[str, ...]:
    paths = tuple(dict.fromkeys(include or ()))
    if not paths:
        raise ValueError("an explicit nonempty --include code-path allowlist is required")
    for name in paths:
        if not isinstance(name, str):
            raise ValueError("export paths must be strings")
        path = Path(name)
        if (
            path.is_absolute()
            or path.as_posix() != name or ".." in path.parts
            or any(c in name for c in "\n\r\0*?[")
            or not (name in ROOT_FILES or (
                len(path.parts) > 1 and path.parts[0] in CODE_ROOTS
                and path.suffix == ".py"
                and all(not part.startswith(".") for part in path.parts)
            ))
        ):
            raise ValueError(f"not an approved code export path: {name!r}")
        record = run("git", "-C", str(repo), "ls-tree", sha, "--", name)
        fields = record.split("\t", 1)
        if len(fields) != 2 or fields[1] != name or fields[0].split()[:2] not in (
            ["100644", "blob"], ["100755", "blob"]
        ):
            raise ValueError(f"export path must be one tracked regular file: {name!r}")
    return paths


def export(repo: Path, ref: str, out: Path, *, include=()) -> tuple[str, int]:
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("export directory must be empty")
    sha = run("git", "-C", str(repo), "rev-parse", "--verify", f"{ref}^{{commit}}")
    selected = _selected_paths(repo, sha, include)
    archive = subprocess.run(
        ["git", "-C", str(repo), "archive", "--format=tar", sha, "--", *selected], check=True, capture_output=True
    ).stdout
    subs, _ = environment_tokens(repo)
    scrubbed = 0
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            if member.name not in selected:
                raise ValueError(f"archive returned an unselected path: {member.name!r}")
            data = tar.extractfile(member).read()
            if b"\0" in data:
                raise ValueError(f"binary payload is not a code export: {member.name}")
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(f"non-UTF-8 payload is not a code export: {member.name}") from error
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
    raise RuntimeError(
        "automatic public mirroring is disabled; review an explicit code diff "
        "against the current public head in an isolated checkout"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--ref", default="main")
    parser.add_argument("--out", type=Path, help="export directory (default: a temporary directory)")
    parser.add_argument("--include", action="append", default=[], metavar="PATH",
                        help="exact tracked code file to export (repeatable; required)")
    parser.add_argument("--push", metavar="REMOTE", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.push:
        parser.error("--push is retired; prepare and review an allowlisted local export")
    if not args.include:
        parser.error("at least one explicit --include code path is required")

    tmp = None
    out = args.out
    if out is None:
        tmp = tempfile.TemporaryDirectory()
        out = Path(tmp.name) / "export"
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)

    sha, scrubbed = export(args.repo, args.ref, out, include=args.include)
    files = sum(1 for p in out.rglob("*") if p.is_file())
    print(f"exported {args.ref} @ {sha[:12]}: {files} files, {scrubbed} scrubbed -> {out}")
    hits = scan(args.repo, out)
    if hits:
        print(f"scan found {len(hits)} leak(s); refusing to publish:", file=sys.stderr)
        for hit in hits[:50]:
            print(f"  {hit}", file=sys.stderr)
        raise SystemExit(1)
    print("scan clean")
    if tmp is not None:
        tmp.cleanup()


if __name__ == "__main__":
    main()

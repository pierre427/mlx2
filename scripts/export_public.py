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
import os
import re
import socket
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

CODE_ROOTS = {"src", "scripts", "tests"}
ROOT_FILES = {".gitignore", "LICENSE", "NOTICE", "README.md", "pyproject.toml"}
# Never published, whatever the allowlist says (Pierre, 2026-10-08): MCP
# servers and their tests stay private.  A path is private when any of its
# components names an MCP (``example_mcp/``, ``test_example_mcp.py``, ...).
PRIVATE_PATH_RE = re.compile(
    r"(?:^|/)[^/]*mcp[^/]*(?:/|$)"
    # Weight-space / residual-ablation MECHANISM (the reusable library under
    # src/mlx2) is public. What stays private is our probe harness and the
    # data used to perform ablation: the scripts/ runner holds refusal
    # markers, judge prompts, dataset wiring and our orchestration (Pierre,
    # 2026-11-06). Prompt datasets, run evidence, decision rows and any
    # ablated outputs live under docs/qualification/provenance and are not
    # exported at all.
    r"|(?:^|/)scripts/[^/]*/?[^/]*residual_ablation[^/]*"
    r"|(?:^|/)scripts/[^/]*/?heretic_domain_bench\.py$"
    # The desktop-UI ablation test embeds the harmful prompt dataset id and the
    # compliance-probe prompt as assertions, so it is part of the private probe
    # surface; the engine/overlay mechanism tests stay public.
    r"|(?:^|/)tests/test_desktop_residual_ablation[^/]*",
    re.IGNORECASE,
)
# pyproject.toml is published; lines that declare an MCP extra or console
# entry point are removed from the public copy so it never points at an
# excluded module.
PRIVATE_PYPROJECT_LINE_RE = re.compile(r"^\s*[\"']?[\w.-]*mcp[\w.-]*[\"']?\s*=", re.IGNORECASE)


_MCP_PATH_RE = re.compile(r"(?:^|/)[^/]*mcp[^/]*(?:/|$)", re.IGNORECASE)
# The withhold list lives next to this script in the private checkout and is
# not exported (not a .py file). MLX2_PUBLIC_EXPORT_WITHHELD may point at a
# copy when this script runs from elsewhere.
WITHHELD_NAME = "public_export_withheld.txt"


def _withheld_path() -> Path | None:
    for candidate in (os.environ.get("MLX2_PUBLIC_EXPORT_WITHHELD"), Path(__file__).with_name(WITHHELD_NAME)):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


def parse_withheld(text: str) -> dict[str, dict[str, set[str]]]:
    """{kind: {"exact": paths, "prefix": dir prefixes}} for kinds never and frozen."""
    out = {kind: {"exact": set(), "prefix": set()} for kind in ("never", "frozen")}
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 2 or fields[0] not in out:
            raise ValueError(f"{WITHHELD_NAME}:{number}: expected 'never|frozen <path>'")
        kind, name = fields
        if name.startswith("/") or ".." in Path(name).parts:
            raise ValueError(f"{WITHHELD_NAME}:{number}: path must be repository-relative")
        out[kind]["prefix" if name.endswith("/") else "exact"].add(name)
    return out


_WITHHELD_FILE = _withheld_path()
WITHHELD = parse_withheld(_WITHHELD_FILE.read_text()) if _WITHHELD_FILE else None


def withheld_kind(name: str) -> str | None:
    """'never' or 'frozen' when the withhold list names this path, else None."""
    for kind in ("never", "frozen"):
        rules = (WITHHELD or {}).get(kind)
        if rules and (name in rules["exact"] or any(name.startswith(p) for p in rules["prefix"])):
            return kind
    return None


def is_private_path(name: str) -> bool:
    """True for paths that are never publication inputs (PRIVATE_PATH_RE or the withhold list)."""
    return bool(PRIVATE_PATH_RE.search(name)) or withheld_kind(name) is not None


def never_public(name: str) -> bool:
    """True for paths that must not exist anywhere in the public history."""
    return bool(_MCP_PATH_RE.search(name)) or withheld_kind(name) == "never"


def is_public_code_path(name: str) -> bool:
    """The publication allowlist minus the private exclusions."""
    path = Path(name)
    allowed = name in ROOT_FILES or (
        len(path.parts) > 1 and path.parts[0] in CODE_ROOTS
        and path.suffix == ".py"
        and all(not part.startswith(".") for part in path.parts)
    )
    return allowed and not is_private_path(name)


def public_text(name: str, text: str) -> str:
    """The public form of an exported file: MCP declarations dropped from pyproject."""
    if name != "pyproject.toml":
        return text
    return "".join(
        line for line in text.splitlines(keepends=True)
        if not PRIVATE_PYPROJECT_LINE_RE.match(line)
    )
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
            or not is_public_code_path(name)
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
    if WITHHELD is None:
        raise ValueError(f"withhold list {WITHHELD_NAME} not found next to export_public.py "
                         "(or via MLX2_PUBLIC_EXPORT_WITHHELD); refusing to export")
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
            new = public_text(member.name, text)
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


ZERO = "0" * 40


def check_tree(clone: Path, rev: str = "HEAD") -> list[str]:
    """Paths in a public checkout's tree at rev that must never be public."""
    names = run("git", "-C", str(clone), "ls-tree", "-r", "--name-only", rev).splitlines()
    return [name for name in names if never_public(name)]


def check_push(clone: Path, updates: list[tuple[str, str, str, str]]) -> list[str]:
    """Problems with a push: never-public paths in any pushed tip or in any new commit."""
    problems = []
    for local_ref, local_sha, remote_ref, remote_sha in updates:
        if local_sha == ZERO:
            continue  # branch deletion
        for name in check_tree(clone, local_sha):
            problems.append(f"{remote_ref}: tip {local_sha[:10]} contains {name}")
        span = [local_sha, "--not", "--remotes"] if remote_sha == ZERO else [local_sha, "--not", remote_sha]
        for commit in run("git", "-C", str(clone), "rev-list", *span).split():
            added = run("git", "-C", str(clone), "diff-tree", "--root", "--no-commit-id", "-r",
                        "--name-only", "--diff-filter=AMR", commit).splitlines()
            for name in added:
                if never_public(name):
                    problems.append(f"{remote_ref}: commit {commit[:10]} adds {name}")
    return problems


def install_push_guard(clone: Path) -> None:
    """Pre-push hook and info/exclude entries in a local public clone (never pushed)."""
    if WITHHELD is None:
        raise ValueError(f"withhold list {WITHHELD_NAME} not found; cannot install the guard")
    git_dir = Path(run("git", "-C", str(clone), "rev-parse", "--absolute-git-dir"))
    hook = git_dir / "hooks" / "pre-push"
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text("#!/bin/sh\n# Installed by scripts/export_public.py --install-push-guard\n"
                    f'exec "{sys.executable}" "{Path(__file__).resolve()}" --pre-push "$1" "$2"\n')
    hook.chmod(0o755)
    exclude = git_dir / "info" / "exclude"
    exclude.parent.mkdir(parents=True, exist_ok=True)
    marker = "# mlx2 never-public paths (export_public.py --install-push-guard)"
    kept = exclude.read_text().split(marker)[0].rstrip("\n") if exclude.is_file() else ""
    entries = sorted(WITHHELD["never"]["exact"] | WITHHELD["never"]["prefix"])
    exclude.write_text((kept + "\n" if kept else "") + marker + "\n"
                       + "".join(f"/{name}\n" for name in entries))


PUBLIC_REPO_URL_RE = r"github\.com[:/]pierre427/mlx2(\.git)?/?$"


def install_private_guard(repo: Path) -> None:
    """Block pushes from a private checkout (and its worktrees) to the public repository.

    Private history carries home paths, private evidence and withheld code;
    the public mirror is built only by projecting exports into a public clone.
    The hook refuses any push whose URL is the public repository, and every
    remote that points at it becomes fetch-only.
    """
    common = Path(run("git", "-C", str(repo), "rev-parse", "--path-format=absolute", "--git-common-dir"))
    hooks = run("git", "-C", str(repo), "config", "--get", "core.hooksPath") if optional(
        "git", "-C", str(repo), "config", "--get", "core.hooksPath") else ""
    hook = (Path(hooks) if hooks else common / "hooks") / "pre-push"
    marker = "# mlx2 private checkout: never push to the public repository"
    body = hook.read_text() if hook.is_file() else "#!/bin/sh\n"
    if marker not in body:
        guard = (f"{marker} (export_public.py --install-private-guard)\n"
                 f'if printf "%s" "$2" | grep -Eq \'{PUBLIC_REPO_URL_RE}\'; then\n'
                 '  echo "refusing: this is a private mlx2 checkout; publish through a public clone" >&2\n'
                 "  exit 1\nfi\n")
        lines = body.splitlines(keepends=True)
        shebang, rest = (lines[0], "".join(lines[1:])) if lines and lines[0].startswith("#!") else ("#!/bin/sh\n", body)
        hook.parent.mkdir(parents=True, exist_ok=True)
        hook.write_text(shebang + guard + rest)
        hook.chmod(0o755)
    for line in optional("git", "-C", str(repo), "remote", "-v").splitlines():
        name, url, kind = line.split()
        if kind == "(fetch)" and re.search(PUBLIC_REPO_URL_RE, url):
            run("git", "-C", str(repo), "config", f"remote.{name}.pushurl", "DISABLED-public-repo-is-fetch-only")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--ref", default="main")
    parser.add_argument("--out", type=Path, help="export directory (default: a temporary directory)")
    parser.add_argument("--include", action="append", default=[], metavar="PATH",
                        help="exact tracked code file to export (repeatable; required)")
    parser.add_argument("--push", metavar="REMOTE", help=argparse.SUPPRESS)
    parser.add_argument("--check-tree", type=Path, metavar="CLONE",
                        help="refuse if a public checkout's HEAD contains a never-public path")
    parser.add_argument("--install-push-guard", type=Path, metavar="CLONE",
                        help="install the pre-push guard and info/exclude entries in a public clone")
    parser.add_argument("--install-private-guard", type=Path, metavar="REPO",
                        help="block pushes from a private checkout to the public repository")
    parser.add_argument("--pre-push", nargs=2, metavar=("REMOTE", "URL"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.install_private_guard:
        install_private_guard(args.install_private_guard)
        print(f"private push guard installed for {args.install_private_guard}")
        return
    if args.install_push_guard:
        install_push_guard(args.install_push_guard)
        print(f"push guard installed in {args.install_push_guard}")
        return
    if args.check_tree or args.pre_push:
        if WITHHELD is None:
            raise SystemExit(f"withhold list {WITHHELD_NAME} not found; refusing")
        if args.check_tree:
            problems = [f"HEAD contains {name}" for name in check_tree(args.check_tree)]
        else:
            updates = [tuple(line.split()) for line in sys.stdin.read().splitlines() if line.strip()]
            problems = check_push(Path.cwd(), updates)
        if problems:
            print("never-public paths; refusing:", file=sys.stderr)
            for problem in problems[:50]:
                print(f"  {problem}", file=sys.stderr)
            raise SystemExit(1)
        print("public tree guard: clean")
        return
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

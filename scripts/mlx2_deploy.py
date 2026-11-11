#!/usr/bin/env python3
"""mlx2 production deploys: immutable releases, a locked venv, launchd services.

Production never runs from a development checkout or its venv.

* ``release REF`` builds ``<prod>/releases/<sha>``: a ``git archive`` of one
  commit's runtime paths plus ``RELEASE.json``, import-smoked, made read-only.
* ``venv --lock FILE`` builds ``<prod>/venvs/<lock hash>`` from an exact,
  dependency-free package list and checks its freeze against the lock.
* ``promote NAME --release SHA --venv HASH`` writes and starts one launchd
  service from ``ops/services/NAME.json``. It stops and disables every other
  service on the same port (one enabled owner per port, also at login), waits
  for ``/health``, and restores the previous plist and port owner on failure.
  Without ``--yes`` it prints the plan and changes nothing.
* ``install NAME --release SHA --venv HASH`` writes an inactive service's plist
  and keeps it disabled, so it never starts at login beside the port's owner.
* ``status`` shows definitions, deployments, launchd state and port owners.
* ``gc`` removes releases and venvs no deployment or plist references
  (``--yes`` to delete; default is a dry run).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import plistlib
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

if sys.version_info < (3, 12):
    raise SystemExit("mlx2_deploy.py needs Python 3.12+ (use the dev or production venv python)")

REPO = Path(__file__).resolve().parents[1]
PROD = Path(os.environ.get("MLX2_PROD_ROOT", "~/Library/Application Support/mlx2/prod")).expanduser()
LAUNCH_AGENTS = Path(os.environ.get("MLX2_LAUNCH_AGENTS", "~/Library/LaunchAgents")).expanduser()
SERVICES = REPO / "ops" / "services"
DOMAIN = f"gui/{os.getuid()}"
# Runtime paths a release carries (the server reads only src/; scripts and
# data travel for tools run against the release). Evidence trees stay out.
RELEASE_PATHS = ("src", "scripts", "third_party", "benchmarks", "configs",
                 # read by mlx2.runtime.prefill_handoff.default_records_dir()
                 "qualification/approximate",
                 "pyproject.toml", "LICENSE", "NOTICE", "README.md")
SERVICE_KEYS = {"label", "port", "model", "args", "env", "keepalive", "health_timeout_s", "note"}
HEALTH_TIMEOUT_S = 900


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run(*args, cwd=None, check=True, env=None, capture=True):
    return subprocess.run(list(args), cwd=cwd, check=check, env=env, text=True,
                          capture_output=capture)


def git(*args, cwd=None) -> str:
    return run("git", *args, cwd=cwd or REPO).stdout.strip()


def write_json(path: Path, value) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def make_writable(root: Path) -> None:
    for path in [root, *root.rglob("*")]:
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | 0o200)


def make_read_only(root: Path) -> None:
    for path in [*root.rglob("*"), root]:
        if not path.is_symlink():
            path.chmod(path.stat().st_mode & ~0o222)


# ---------------------------------------------------------------- releases

def release_dir(sha: str) -> Path:
    return PROD / "releases" / sha


def build_release(ref: str, *, venv: Path | None = None, repo: Path | None = None) -> Path:
    repo = repo or REPO
    sha = git("rev-parse", "--verify", f"{ref}^{{commit}}", cwd=repo)
    target = release_dir(sha)
    if (target / "RELEASE.json").is_file():
        return target
    present = [p for p in RELEASE_PATHS if git("ls-tree", sha, "--", p, cwd=repo)]
    if "src" not in present:
        raise SystemExit(f"{sha[:12]} has no src/; not a release")
    archive = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", sha, "--", *present],
                             check=True, capture_output=True).stdout
    work = target.with_name(f".tmp-{sha}")
    if work.exists():
        make_writable(work)
        shutil.rmtree(work)
    work.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(work, filter="data")
    info = {"sha": sha, "ref": ref, "created": now(), "paths": present,
            "archive_sha256": hashlib.sha256(archive).hexdigest(),
            "subject": git("log", "-1", "--format=%s", sha, cwd=repo)}
    write_json(work / "RELEASE.json", info)
    if venv is not None:
        smoke = run(str(venv / "bin" / "python"), "-c",
                    "import json, pathlib, mlx2, mlx2.server; "
                    "root = pathlib.Path(mlx2.__file__).resolve().parents[2]; "
                    "assert json.loads((root / 'RELEASE.json').read_text())['sha'], 'RELEASE.json not seen'",
                    cwd=work, check=False,
                    env={**os.environ, "PYTHONPATH": str(work / "src"),
                         "MLX2_PREFLIGHT_CPU_ONLY": "1"})
        if smoke.returncode:
            make_writable(work)
            shutil.rmtree(work)
            raise SystemExit(f"release import smoke failed:\n{smoke.stderr[-3000:]}")
        # The tree is sealed read-only; compile now so starts need not.
        run(str(venv / "bin" / "python"), "-m", "compileall", "-q", str(work / "src"), check=False)
    make_read_only(work)
    os.replace(work, target)
    return target


# ---------------------------------------------------------------- venvs

def lock_text(path: Path) -> str:
    lines = [l.strip() for l in path.read_text().splitlines()]
    return "\n".join(sorted(l for l in lines if l and not l.startswith("#"))) + "\n"


def lock_hash(path: Path, python: str = "") -> str:
    return hashlib.sha256((f"python={python}\n" + lock_text(path)).encode()).hexdigest()[:16]


def venv_dir(digest: str) -> Path:
    return PROD / "venvs" / digest


def parse_lock(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        m = re.match(r"([A-Za-z0-9_.-]+)\s*(?:==\s*(\S+)|@\s*(\S+))", line)
        if m:
            out[m.group(1).lower().replace("_", "-")] = m.group(2) or m.group(3)
    return out


def build_venv(lock: Path, python: str) -> Path:
    """Build the venv in place (venvs hold absolute paths); VENV.json marks it complete."""
    digest = lock_hash(lock, python)
    target = venv_dir(digest)
    if (target / "VENV.json").is_file():
        return target
    uv = shutil.which("uv")
    if not uv:
        raise SystemExit("uv is required to build the production venv")
    if target.exists():
        shutil.rmtree(target)  # an earlier build never completed
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        run(uv, "venv", "--python", python, str(target), capture=False)
        req = target / "lock.txt"
        req.write_text(lock_text(lock))
        run(uv, "pip", "install", "--python", str(target / "bin" / "python"), "--no-deps",
            "-r", str(req), capture=False)
        freeze = run(uv, "pip", "freeze", "--python", str(target / "bin" / "python")).stdout
        want, have = parse_lock(lock_text(lock)), parse_lock(freeze)
        drift = {k: (v, have.get(k)) for k, v in want.items() if have.get(k) != v}
        extra = sorted(set(have) - set(want))
        if drift or extra:
            raise SystemExit(f"venv freeze differs from the lock: {drift} extra={extra}")
        version = run(str(target / "bin" / "python"), "-c",
                      "import sys; print(sys.version.split()[0])").stdout.strip()
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise
    write_json(target / "VENV.json", {"lock_sha256_16": digest, "lock": str(lock), "python": python,
                                      "python_version": version, "created": now(), "freeze": freeze})
    return target


# ---------------------------------------------------------------- services

# Flags and environment the deploy owns; a definition may not set them.
OWNED_ARGS = ("--port", "--host", "--model", "--allow-production-port")
OWNED_ENV = {"PYTHONPATH", "MLX2_ALLOW_PRODUCTION_PORT", "MLX2_RELEASE"}
# launchd keys a definition may pass through (the heretic service sets both).
LAUNCHD_KEYS = {"ProcessType", "ThrottleInterval"}
SERVICE_KEYS = SERVICE_KEYS | {"launchd"}


def load_service(name: str) -> dict:
    path = SERVICES / f"{name}.json"
    spec = json.loads(path.read_text())
    unknown = set(spec) - SERVICE_KEYS
    if unknown:
        raise SystemExit(f"{path}: unknown keys {sorted(unknown)}")
    for key in ("label", "port", "model"):
        if key not in spec:
            raise SystemExit(f"{path}: missing {key}")
    bad = [a for a in spec.get("args", []) if any(a == o or a.startswith(o + "=") for o in OWNED_ARGS)]
    if bad:
        raise SystemExit(f"{path}: {bad} are set by the deploy (service fields), not args")
    if OWNED_ENV & set(spec.get("env", {})):
        raise SystemExit(f"{path}: env may not set {sorted(OWNED_ENV & set(spec['env']))}")
    if set(spec.get("launchd", {})) - LAUNCHD_KEYS:
        raise SystemExit(f"{path}: launchd keys limited to {sorted(LAUNCHD_KEYS)}")
    spec["name"] = name
    return spec


def committed_definition(name: str) -> str:
    """Blob sha of the committed definition; refuses an uncommitted or edited one."""
    rel = (SERVICES / f"{name}.json").relative_to(REPO)
    if run("git", "diff", "--quiet", "HEAD", "--", str(rel), cwd=REPO, check=False).returncode:
        raise SystemExit(f"{rel} differs from HEAD; commit the service definition before deploying it")
    blob = run("git", "rev-parse", f"HEAD:{rel}", cwd=REPO, check=False)
    if blob.returncode:
        raise SystemExit(f"{rel} is not committed; commit the service definition before deploying it")
    return blob.stdout.strip()


def all_services() -> list[dict]:
    specs = [load_service(p.stem) for p in sorted(SERVICES.glob("*.json"))]
    labels = [s["label"] for s in specs]
    if len(labels) != len(set(labels)):
        raise SystemExit(f"duplicate labels in {SERVICES}: {sorted(l for l in labels if labels.count(l) > 1)}")
    return specs


def service_root(name: str) -> Path:
    return PROD / "services" / name


def plist_for(spec: dict, release: Path, venv: Path) -> dict:
    logs = service_root(spec["name"]) / "logs"
    env = {"PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           **spec.get("env", {}), "PYTHONPATH": str(release / "src"), "MLX2_RELEASE": release.name}
    return {
        "Label": spec["label"],
        "ProgramArguments": [str(venv / "bin" / "python"), "-u", "-m", "mlx2.server",
                             "--model", spec["model"], "--host", "127.0.0.1",
                             "--port", str(spec["port"]), *spec.get("args", [])],
        "WorkingDirectory": str(release),
        "EnvironmentVariables": env,
        "RunAtLoad": True,
        "KeepAlive": spec.get("keepalive", True),
        "StandardOutPath": str(logs / "stdout.log"),
        "StandardErrorPath": str(logs / "stderr.log"),
        **spec.get("launchd", {}),
    }


_SCRIPT_PORT = re.compile(r'--port[= ]+"?(\d{2,5})\b|\bPORT:-(\d{2,5})\b|\bPORT="?(\d{2,5})\b')
SHELLS = {"sh", "bash", "zsh", "dash"}


def plist_port(data: dict) -> int | None:
    """The TCP port a launchd job serves on: --port N, --port=N, env PORT, or its launcher script."""
    args = [str(a) for a in data.get("ProgramArguments") or []]
    for i, arg in enumerate(args):
        if arg == "--port" and i + 1 < len(args) and args[i + 1].isdigit():
            return int(args[i + 1])
        if arg.startswith("--port=") and arg[7:].isdigit():
            return int(arg[7:])
    env_port = str((data.get("EnvironmentVariables") or {}).get("PORT", ""))
    if env_port.isdigit():
        return int(env_port)
    for arg in args[:3]:
        match = _SCRIPT_PORT.search(arg)  # sh -c "... --port 8282"
        if match:
            return int(next(g for g in match.groups() if g))
        path = Path(arg)
        if path.name in SHELLS or not path.is_file():
            continue
        try:
            if path.stat().st_size > 1 << 20:
                continue
            text = path.read_text(errors="replace")
        except OSError:
            continue
        if text.startswith("#!"):
            match = _SCRIPT_PORT.search(text)
            if match:
                return int(next(g for g in match.groups() if g))
    return None


def read_plist(path: Path) -> dict | None:
    try:
        with path.open("rb") as fh:
            return plistlib.load(fh)
    except Exception:
        return None


class Launchd:
    """launchctl wrapper (replaced in tests)."""

    def info(self, label: str) -> dict | None:
        out = run("launchctl", "print", f"{DOMAIN}/{label}", check=False)
        if out.returncode:
            return None
        fields = {}
        for key in ("pid", "runs", "last exit code", "state"):
            match = re.search(rf"^\s*{key} = (.+)$", out.stdout, re.M)
            if match:
                fields[key] = match.group(1).strip()
        return fields

    def loaded(self, label: str) -> bool:
        return self.info(label) is not None

    def disabled(self) -> set[str]:
        out = run("launchctl", "print-disabled", DOMAIN).stdout
        return {m.group(1) for m in re.finditer(r'"([^"]+)" => (?:disabled|true)', out)}

    def bootout(self, label: str) -> None:
        run("launchctl", "bootout", f"{DOMAIN}/{label}", check=False)

    def bootstrap(self, plist: Path) -> None:
        run("launchctl", "bootstrap", DOMAIN, str(plist))

    def enable(self, label: str) -> None:
        run("launchctl", "enable", f"{DOMAIN}/{label}")

    def disable(self, label: str) -> None:
        run("launchctl", "disable", f"{DOMAIN}/{label}")


def port_listeners(port: int) -> set[int]:
    out = run("lsof", "-nP", "-t", f"-iTCP:{port}", "-sTCP:LISTEN", check=False).stdout
    return {int(pid) for pid in out.split() if pid.isdigit()}


def port_owners(port: int, launchd: Launchd, skip: str | None = None) -> list[dict]:
    """Every LaunchAgent on this port that is loaded or enabled (so may start at login)."""
    disabled = launchd.disabled()
    defined = {}
    if SERVICES.is_dir():
        for spec in all_services():
            defined[spec["label"]] = spec["port"]
    owners = []
    for path in sorted(LAUNCH_AGENTS.glob("*.plist")):
        data = read_plist(path)
        if not data:
            continue
        label = data.get("Label") or path.stem
        if label == skip or (defined.get(label, plist_port(data)) != port):
            continue
        loaded = launchd.loaded(label)
        at_login = label not in disabled
        if loaded or at_login:
            owners.append({"label": label, "plist": str(path), "loaded": loaded, "at_login": at_login})
    return owners


def wait_stopped(label: str, port: int, launchd: Launchd, listeners=port_listeners, timeout: float = 60) -> None:
    """bootout returns before teardown ends; wait until the job is gone and the port is free."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not launchd.loaded(label) and not listeners(port):
            return
        time.sleep(0.5)
    raise RuntimeError(f"{label} did not stop (or port {port} stayed in use) within {timeout:.0f} s")


def healthy(port: int, timeout: float, *, label: str, launchd: Launchd, listeners=port_listeners) -> bool:
    """/health ok from the new job's own process; gives up early when the job crash-loops."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        info = launchd.info(label) or {}
        try:
            if int(info.get("runs", "1")) > 3:
                return False
        except ValueError:
            pass
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as resp:
                ok = resp.status == 200 and json.load(resp).get("status") == "ok"
            pid = info.get("pid", "")
            if ok and pid.isdigit() and int(pid) in listeners(port):
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def preflight(spec: dict, release: Path, venv: Path) -> None:
    """Checks that need no GPU and no port, run before anything live is stopped."""
    if not (Path(spec["model"]) / "config.json").is_file():
        raise SystemExit(f"model artifact missing: {spec['model']}/config.json")
    smoke = run(str(venv / "bin" / "python"), "-c", "import mlx2.server", cwd=release, check=False,
                env={**os.environ, "PYTHONPATH": str(release / "src"), "MLX2_PREFLIGHT_CPU_ONLY": "1"})
    if smoke.returncode:
        raise SystemExit(f"release does not import under the production venv:\n{smoke.stderr[-2000:]}")


def _lock():
    import fcntl
    PROD.mkdir(parents=True, exist_ok=True)
    handle = open(PROD / ".deploy.lock", "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another mlx2_deploy promote/install is running")
    return handle


def _write_plist(path: Path, data: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("wb") as fh:
        plistlib.dump(data, fh)
    os.replace(tmp, path)


def _resolve(name: str, sha: str, venv_digest: str):
    spec = load_service(name)
    all_services()  # duplicate-label check
    release = release_dir(git("rev-parse", "--verify", f"{sha}^{{commit}}"))
    venv = venv_dir(venv_digest)
    for needed in (release / "RELEASE.json", venv / "VENV.json"):
        if not needed.is_file():
            raise SystemExit(f"missing {needed}; build it first")
    return spec, release, venv


def promote(name: str, sha: str, venv_digest: str, *, yes: bool, launchd: Launchd | None = None,
            health=healthy, listeners=port_listeners, check_model: bool = True) -> dict:
    launchd = launchd or Launchd()
    spec, release, venv = _resolve(name, sha, venv_digest)
    plist_path = LAUNCH_AGENTS / f"{spec['label']}.plist"
    new = plist_for(spec, release, venv)
    owners = port_owners(spec["port"], launchd, skip=spec["label"])
    plan = {"service": name, "label": spec["label"], "port": spec["port"], "release": release.name,
            "venv": venv_digest, "plist": str(plist_path), "stop_and_disable": owners}
    if not yes:
        return {**plan, "dry_run": True}
    plan["definition_blob"] = committed_definition(name)
    if check_model:
        preflight(spec, release, venv)
    lock = _lock()
    root = service_root(name)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    backup = root / "previous.plist"
    had_previous = plist_path.is_file()
    if had_previous:
        shutil.copy2(plist_path, backup)
    was_loaded = launchd.loaded(spec["label"])
    disabled_before = launchd.disabled()
    record = {**plan, "at": now(), "ok": False}
    try:
        for owner in owners:
            launchd.bootout(owner["label"])
            launchd.disable(owner["label"])
            wait_stopped(owner["label"], spec["port"], launchd, listeners)
        launchd.bootout(spec["label"])
        wait_stopped(spec["label"], spec["port"], launchd, listeners)
        _write_plist(plist_path, new)
        launchd.enable(spec["label"])
        launchd.bootstrap(plist_path)
        record["ok"] = bool(health(spec["port"], spec.get("health_timeout_s", HEALTH_TIMEOUT_S),
                                   label=spec["label"], launchd=launchd, listeners=listeners))
    except BaseException as error:  # interrupt or launchctl failure: roll back, then re-raise
        record["error"] = repr(error)[:500]
        _rollback(spec, plist_path, backup, had_previous, was_loaded, owners, disabled_before, launchd, record)
        _history(root, record)
        lock.close()
        raise
    if record["ok"]:
        write_json(root / "deployed.json", {**record, "active": True, "plist_content": new})
        for other in all_services():
            marker = service_root(other["name"]) / "deployed.json"
            if other["name"] != name and other["port"] == spec["port"] and marker.is_file():
                info = json.loads(marker.read_text())
                if info.get("active"):
                    write_json(marker, {**info, "active": False})
    else:
        _rollback(spec, plist_path, backup, had_previous, was_loaded, owners, disabled_before, launchd, record)
    _history(root, record)
    lock.close()
    return record


def _rollback(spec, plist_path, backup, had_previous, was_loaded, owners, disabled_before, launchd, record):
    """Best effort, every step attempted: restore the plist, enable state and the port's previous owners."""
    errors = []

    def attempt(step, *args):
        try:
            step(*args)
        except BaseException as error:  # keep going; report what failed
            errors.append(f"{getattr(step, '__name__', step)}{args}: {error!r}"[:300])

    attempt(launchd.bootout, spec["label"])
    if had_previous:
        attempt(shutil.copy2, backup, plist_path)
    else:
        attempt(plist_path.unlink, True)
    labels = [spec["label"], *(o["label"] for o in owners)]
    for label in labels:
        attempt(launchd.disable if label in disabled_before else launchd.enable, label)
    if had_previous and was_loaded:
        attempt(launchd.bootstrap, plist_path)
    for owner in owners:
        if owner["loaded"]:
            attempt(launchd.bootstrap, Path(owner["plist"]))
    record["rolled_back"] = True
    if errors:
        record["rollback_errors"] = errors


def _history(root: Path, record: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "history.jsonl").open("a") as fh:
        fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def install(name: str, sha: str, venv_digest: str, *, yes: bool, launchd: Launchd | None = None,
            force: bool = False) -> dict:
    """Write an inactive service's plist against a release and keep it disabled (not started).

    Inactive services must not start at login next to the port's active owner;
    ``promote`` is the only way a service starts. The port's live or enabled
    owner is refused (it may be paused, not inactive): replacing it is a promotion.
    """
    launchd = launchd or Launchd()
    spec, release, venv = _resolve(name, sha, venv_digest)
    if launchd.loaded(spec["label"]):
        raise SystemExit(f"{spec['label']} is running; use promote to replace a live service")
    marker = service_root(name) / "deployed.json"
    active = marker.is_file() and json.loads(marker.read_text()).get("active")
    enabled = spec["label"] not in launchd.disabled()
    if (active or enabled) and not force:
        raise SystemExit(f"{spec['label']} is {'the active service' if active else 'enabled at login'} "
                         "(it may only be paused); use promote, or --force if it is really retired")
    plist_path = LAUNCH_AGENTS / f"{spec['label']}.plist"
    new = plist_for(spec, release, venv)
    plan = {"service": name, "label": spec["label"], "port": spec["port"], "release": release.name,
            "venv": venv_digest, "plist": str(plist_path), "active": False}
    if not yes:
        return {**plan, "dry_run": True}
    plan["definition_blob"] = committed_definition(name)
    lock = _lock()
    root = service_root(name)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    if plist_path.is_file():
        shutil.copy2(plist_path, root / "previous.plist")
    launchd.disable(spec["label"])
    _write_plist(plist_path, new)
    record = {**plan, "at": now(), "ok": True}
    write_json(root / "deployed.json", {**record, "plist_content": new})
    _history(root, record)
    lock.close()
    return record


# ---------------------------------------------------------------- status / gc

def referenced() -> tuple[set[str], set[str]]:
    releases, venvs = set(), set()
    texts = []
    for path in LAUNCH_AGENTS.glob("*.plist"):
        data = read_plist(path)
        if data:
            texts.append(json.dumps(data, default=str))
    for path in (PROD / "services").glob("*/deployed.json"):
        texts.append(path.read_text())
    blob = "\n".join(texts)
    for path in (PROD / "releases").glob("*"):
        if str(path) in blob or path.name in blob:
            releases.add(path.name)
    for path in (PROD / "venvs").glob("*"):
        if str(path) in blob:
            venvs.add(path.name)
    return releases, venvs


def gc(keep: int, yes: bool) -> dict:
    used_releases, used_venvs = referenced()
    out = {"remove": [], "keep": []}
    for kind, used in (("releases", used_releases), ("venvs", used_venvs)):
        entries = sorted((p for p in (PROD / kind).glob("*") if not p.name.startswith(".")),
                         key=lambda p: p.stat().st_mtime, reverse=True)
        for index, path in enumerate(entries):
            if path.name in used or index < keep:
                out["keep"].append(str(path))
                continue
            out["remove"].append(str(path))
            if yes:
                make_writable(path)
                shutil.rmtree(path)
    return out


def status(launchd: Launchd | None = None) -> list[dict]:
    launchd = launchd or Launchd()
    rows = []
    for spec in all_services():
        deployed = service_root(spec["name"]) / "deployed.json"
        info = json.loads(deployed.read_text()) if deployed.is_file() else {}
        rows.append({"service": spec["name"], "label": spec["label"], "port": spec["port"],
                     "loaded": launchd.loaded(spec["label"]), "active": info.get("active"),
                     "disabled": spec["label"] in launchd.disabled(),
                     "release": info.get("release"), "venv": info.get("venv"), "deployed_at": info.get("at"),
                     "other_port_owners": [o["label"] for o in port_owners(spec["port"], launchd,
                                                                            skip=spec["label"])]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("release"); p.add_argument("ref")
    p.add_argument("--venv", required=True, help="production venv hash used for the import smoke")
    p = sub.add_parser("venv"); p.add_argument("--lock", type=Path, default=REPO / "ops" / "prod-lock.txt")
    p.add_argument("--python", default="3.12.13")
    for command in ("promote", "install"):
        p = sub.add_parser(command); p.add_argument("name"); p.add_argument("--release", required=True)
        p.add_argument("--venv", required=True); p.add_argument("--yes", action="store_true")
        if command == "install":
            p.add_argument("--force", action="store_true", help="install over an enabled (not running) label")
    sub.add_parser("status")
    p = sub.add_parser("gc"); p.add_argument("--keep", type=int, default=3); p.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    if args.command == "release":
        print(build_release(args.ref, venv=venv_dir(args.venv)))
    elif args.command == "venv":
        print(build_venv(args.lock, args.python))
    elif args.command == "promote":
        result = promote(args.name, args.release, args.venv, yes=args.yes)
        print(json.dumps(result, indent=2))
        if args.yes and not result.get("ok"):
            raise SystemExit(1)
    elif args.command == "install":
        print(json.dumps(install(args.name, args.release, args.venv, yes=args.yes, force=args.force), indent=2))
    elif args.command == "status":
        print(json.dumps(status(), indent=2))
    else:
        print(json.dumps(gc(args.keep, args.yes), indent=2))


if __name__ == "__main__":
    main()

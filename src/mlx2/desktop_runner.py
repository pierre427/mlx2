"""CPU-only preparation of source-bound, per-artifact desktop virtualenvs.

Environments are published only after dependency installation and metadata
checks. Preparation never imports MLX, loads a checkpoint, or reserves a GPU.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path


class PreparationCancelled(RuntimeError):
    pass


def source_files(root: Path):
    # Installed companions also own the test harness. Keep the inputs needed
    # for a bound preflight, but never package user campaign output/logs.
    for directory in ("src", "provenance", "third_party", "scripts", "tests", "native", "qualification", "docs/experiments"):
        for path in sorted((root / directory).rglob("*")):
            if (
                "__pycache__" in path.parts
                or any(part.endswith(".egg-info") for part in path.parts)
                or path.suffix == ".pyc"
                or path.is_relative_to(root / "qualification/runs")
                or not path.is_file()
            ):
                continue
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"Runner source link leaves the checkout: {path}")
            yield path
    from .desktop_campaign import QUALIFICATION_SOURCE_FILES
    for name in QUALIFICATION_SOURCE_FILES:
        path = root / name
        if path.is_file():
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"Qualification source link leaves the checkout: {path}")
            yield path
    if (root / "src/mlx2/desktop_preflight_inputs.py").is_file():
        from .desktop_preflight_inputs import preflight_inputs
        yield from preflight_inputs(root)
    for name in ("pyproject.toml", "README.md", "LICENSE", "NOTICE", "docs/PROVENANCE.md", "pytest.ini", "conftest.py"):
        if (root / name).is_file():
            yield root / name


def source_manifest(root: Path) -> dict:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_files(root)
    }


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class RunnerManager:
    def __init__(self, home: Path, source: Path | None = None):
        self.home = home
        self.source = source or Path(__file__).resolve().parents[2]

    def ensure(
        self,
        model: Path,
        fingerprint: str,
        *,
        multimodal: bool,
        cancel: threading.Event,
        progress,
        log_path: Path,
        purpose="model",
        extras_override=None,
    ) -> dict:
        from .desktop import atomic_json

        def check_cancel():
            if cancel.is_set():
                raise PreparationCancelled("Runner preparation cancelled")

        check_cancel()
        progress("Checking MLX2 source and dependency versions")
        files = source_manifest(self.source)
        source_hash = digest(files)
        packages = {
            d.metadata["Name"]: d.version
            for d in importlib.metadata.distributions()
            if d.metadata.get("Name") and d.metadata["Name"].lower() != "mlx2"
        }
        extras = extras_override or (
            "serve,observability,multimodal" if multimodal else "serve,observability"
        )
        if purpose == "desktop-companion":
            extras += ",desktop-web"
        binding = {
            "purpose": purpose,
            "model_path": str(model) if purpose == "model" else None,
            "checkpoint": f"{model.name}@{fingerprint}" if purpose == "model" else None,
            "source_sha256": source_hash,
            "seed_packages_sha256": digest(packages),
            "extras": extras,
            "python_version": sys.version,
        }
        identity = digest(binding)
        root = self.home / "desktop-runners" / identity
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        ready = root / "ready.json"
        if ready.exists():
            record = json.loads(ready.read_text())
            python = Path(record["python"])
            snapshot = Path(record["source"])
            if (
                record.get("binding") == binding
                and python.is_file()
                and python.is_relative_to(root)
                and not root.is_symlink()
                and snapshot.is_relative_to(self.home / "desktop-sources")
                and digest(source_manifest(snapshot)) == source_hash
            ):
                progress("Reusing the model's prepared MLX2 runner")
                return record
            raise RuntimeError(
                "Saved runner failed its identity check; inspect or remove its runner record before retrying"
            )
        attempt = root / f"attempt-{uuid.uuid4().hex}"
        attempt.mkdir(mode=0o700)
        snapshot = self.home / "desktop-sources" / source_hash
        if not snapshot.exists():
            progress("Saving the MLX2 runtime snapshot")
            staging = snapshot.with_name(f"{source_hash}.building-{uuid.uuid4().hex}")
            staging.mkdir(parents=True, mode=0o700)
            for relative in files:
                check_cancel()
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.source / relative, target)
            if source_manifest(staging) != files:
                raise RuntimeError(
                    "Source changed while preparing the runner; retry with a stable checkout"
                )
            try:
                revision = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=self.source,
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                ).strip()
                commit = subprocess.check_output(
                    ["git", "cat-file", "commit", revision], cwd=self.source,
                    stderr=subprocess.DEVNULL, timeout=5,
                )
                (staging / "desktop-source-commit.bin").write_bytes(commit)
            except (OSError, subprocess.SubprocessError):
                previous = self.source / "desktop-source.json"
                revision = (
                    json.loads(previous.read_text()).get(
                        "source_revision", "unavailable"
                    )
                    if previous.exists()
                    else "unavailable"
                )
                previous_commit = self.source / "desktop-source-commit.bin"
                if previous_commit.is_file():
                    shutil.copy2(previous_commit, staging / previous_commit.name)
            (staging / "desktop-source.json").write_text(
                json.dumps({"source_revision": revision, "source_sha256": source_hash})
            )
            staging.rename(snapshot)
        elif digest(source_manifest(snapshot)) != source_hash:
            raise RuntimeError(
                "Runtime snapshot bytes differ from the recorded source identity"
            )
        check_cancel()
        uv = next(
            (
                path
                for path in (
                    shutil.which("uv"),
                    "/opt/homebrew/bin/uv",
                    "/usr/local/bin/uv",
                )
                if path and os.access(path, os.X_OK)
            ),
            None,
        )
        if uv is None:
            raise RuntimeError(
                "Runner setup needs uv; install uv and retry Start Model"
            )
        environment = attempt / "venv"
        python = environment / "bin/python"
        constraints = attempt / "constraints.txt"
        constraints.write_text(
            "".join(
                f"{name}=={version}\n" for name, version in sorted(packages.items())
            )
        )
        env = {**os.environ, "PYTHONPATH": str(snapshot / "src"), "UV_NO_PROGRESS": "1"}

        def run(args, *, allow_failure=False):
            check_cancel()
            with log_path.open("ab") as log:
                child = subprocess.Popen(
                    args,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    env=env,
                    start_new_session=True,
                )
                deadline = time.monotonic() + 1800
                try:
                    while child.poll() is None:
                        check_cancel()
                        if time.monotonic() > deadline:
                            raise RuntimeError(
                                "Runner dependency installation timed out; see preparation log"
                            )
                        time.sleep(0.1)
                finally:
                    if child.poll() is None:
                        from .desktop import stop_group

                        stop_group(child)
                if child.returncode and not allow_failure:
                    raise RuntimeError(
                        f"Runner setup failed ({child.returncode}); see {log_path}"
                    )
                return child.returncode

        progress("Creating a virtual environment for this model")
        run([uv, "venv", "--python", sys.executable, str(environment)])
        install = [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "--link-mode",
            "clone",
            "--constraint",
            constraints.as_uri(),
            f"mlx2[{extras}] @ {snapshot.as_uri()}",
        ]
        progress("Installing pinned serving dependencies from the local cache")
        if run([*install, "--offline"], allow_failure=True):
            check_cancel()
            progress("Downloading missing pinned dependencies")
            run(install)
        progress("Checking the prepared runner")
        run([uv, "pip", "check", "--python", str(python)])
        # Metadata/spec inspection only. Importing server or MLX here would
        # turn environment preparation into unrequested model/GPU work.
        validation = (
            "import importlib.util,sys; assert importlib.util.find_spec('mlx2.media.server'); "
            "assert 'mlx.core' not in sys.modules"
            if purpose == "canonical-media" else
            "import importlib.util,importlib.metadata,sys; "
            "assert importlib.util.find_spec('mlx2.server'); "
            "assert importlib.metadata.version('mlx') == " + repr(packages.get("mlx")) + "; "
            "assert 'mlx.core' not in sys.modules"
        )
        run(
            [
                str(python),
                "-c",
                validation,
            ]
        )
        check_cancel()
        record = {
            "schema": "mlx2.desktop.runner.v1",
            "binding": binding,
            "python": str(python),
            "source": str(snapshot),
            "log_path": str(log_path),
            "qualification": "unqualified" if purpose == "model" else "not_applicable",
            "prepared_at": time.time(),
        }
        atomic_json(ready, record)
        return record

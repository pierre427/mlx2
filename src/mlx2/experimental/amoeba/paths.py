"""Where the research code, the shared inputs and the helper interpreters live."""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

ARTIFACTS = Path.home() / "Library/Application Support/mlx2/research-artifacts/amoeba"
GPU_LOCK = Path("/Users/Shared/mlxuag/gpu.lock")

# Inputs pinned by the FN box stage (coordinator, 2026-11-16): probe corpus v5.1 and the manifest at 65ee4e19a.
DEFAULT_PROBE_CORPUS = ARTIFACTS / "topic-probes-e8c22f31a/corpus.jsonl"
DEFAULT_PROBE_MANIFEST_REV = "65ee4e19a"
DEFAULT_CORE_CORPORA = (ARTIFACTS / "graded-cores-72f0ac6ed/tier-L1.jsonl",
                        ARTIFACTS / "graded-cores-72f0ac6ed/tier-L2.jsonl")


def repo_root() -> Path:
    """The checkout (or release tree) holding ``research/amoeba/carve`` and ``scripts/``."""
    env = os.environ.get("MLX2_AMOEBA_RESEARCH_ROOT")
    cands = [Path(env)] if env else []
    cands.append(Path(__file__).resolve().parents[4])
    for c in cands:
        if (c / "research/amoeba/carve/__main__.py").exists() and (c / "scripts/amoeba_onpolicy_gen.py").exists():
            return c
    raise SystemExit("mlx2 amoeba: the research code (research/amoeba/carve, scripts/amoeba_*) is not next to this "
                     "package; run from an mlx2 checkout or release tree, or set MLX2_AMOEBA_RESEARCH_ROOT")


def research_imports() -> None:
    """Make research.amoeba.* and scripts/amoeba_* importable in this (parent) process, as children get them."""
    r = repo_root()
    for d in (r / "scripts", r):
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))


def default_taxonomy() -> Path:
    return repo_root() / "docs/research/amoeba-topics/taxonomy.v1.json"


def child_env(extra: dict | None = None) -> dict:
    """Environment for every child: this tree's src first (AGENTS.md: an editable install may point elsewhere)."""
    r = repo_root()
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(r / "src"), str(r), str(r / "scripts")] +
                                        ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.setdefault("HF_HUB_OFFLINE", "1")
    env["PYTHONUNBUFFERED"] = "1"
    env.update(extra or {})
    return env


def python() -> str:
    return sys.executable


def arrow_python() -> str | None:
    """An interpreter with pyarrow (the prompt-set Parquet writer)."""
    cands = [os.environ.get("MLX2_AMOEBA_ARROW_PYTHON"), sys.executable]
    for c in cands:
        if c and Path(c).exists():
            r = subprocess.run([c, "-c", "import pyarrow"], capture_output=True)
            if r.returncode == 0:
                return c
    return None


def git_rev(root: Path | None = None) -> dict:
    root = root or repo_root()
    try:
        rev = subprocess.check_output(["git", "rev-parse", "--short=9", "HEAD"], cwd=root, text=True,
                                      stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                                             text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": None}
    return {"commit": rev, "dirty": dirty}


def sha256_file(path, n: int | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    d = h.hexdigest()
    return d[:n] if n else d


def git_show(rev: str, path: str, dest: Path) -> Path:
    data = subprocess.check_output(["git", "show", f"{rev}:{path}"], cwd=repo_root())
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return dest

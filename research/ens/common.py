"""Shared A0 plumbing: batching, the published bus codebook, checkpoints and receipts."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import socket
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from research.ens import families as F
from research.ens import vocab as V
from research.ens.model import Core, CoreConfig

ROOT = Path(__file__).resolve().parents[2]

# ENS runs on the M3. ENS_DEVICE=cpu keeps checks on other hosts (the M5) off their GPU.
if os.environ.get("ENS_DEVICE", "gpu") == "cpu":
    mx.set_default_device(mx.cpu)
CODEBOOK_SEED = "ens.a0.codebook.v0"


# ------------------------------------------------------------------ codebook
def codebook(width: int) -> mx.array:
    """The published Neural ABI bus convention: e(z) = M[z], unit-RMS rows, (64, width)."""
    seed = int(hashlib.sha256(f"{CODEBOOK_SEED}:{width}".encode()).hexdigest()[:8], 16)
    M = mx.random.normal((V.N_VALUES, width), key=mx.random.key(seed))
    return M / mx.sqrt(mx.mean(M * M, axis=-1, keepdims=True))


def codebook_digest(width: int) -> str:
    import numpy as np
    return hashlib.sha256(np.array(codebook(width)).astype("<f4").tobytes()).hexdigest()


def jitter(msg: mx.array, rng: random.Random, *, scale=(0.7, 1.3), noise=0.1, drop=0.1) -> mx.array:
    """ABI-bounded provider variation: norm scale, Gaussian noise and channel dropout."""
    B, w = msg.shape
    key = mx.random.key(rng.randrange(1 << 30))
    k1, k2, k3 = mx.random.split(key, 3)
    s = mx.random.uniform(scale[0], scale[1], (B, 1), key=k1)
    n = mx.random.normal((B, w), key=k2) * noise
    keep = (mx.random.uniform(shape=(B, w), key=k3) >= drop).astype(msg.dtype)
    return (msg * s + n) * keep


# ------------------------------------------------------------------ batches
def batch(rows, mode: str):
    """Stack rows laid out in `mode` into arrays: tokens, query_idx, field_idx, ans_idx, answers, zs."""
    toks, q, f, a, ans, zs = [], [], [], [], [], []
    for r in rows:
        ids, qi, fi, ai = F.layout(r, mode)
        toks.append(ids); q.append(qi); f.append(fi); a.append(ai); ans.append(r.answer); zs.append(r.z)
    return (mx.array(toks), mx.array(q), mx.array(f), mx.array(a), mx.array(ans), mx.array(zs))


# ------------------------------------------------------------------ provenance
def source_revision() -> dict:
    """mlx2 commit of research/ens. On the M3 (no mlx2 history) read the sync stamp and re-hash the tree."""
    stamp = ROOT / ".ens-source.json"
    if stamp.exists():
        from research.ens.m3 import tree_sha256
        s = json.loads(stamp.read_text())
        return {"source_revision": s["source_revision"], "source_branch": s.get("branch"),
                "research_ens_dirty": tree_sha256(ROOT) != s["tree_sha256"], "source_via": "m3-sync-stamp"}

    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()
    try:
        sha = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain", "--", "research/ens"))
    except Exception:          # noqa: BLE001 - a missing git must not hide the run
        sha, dirty = "unknown", True
    return {"source_revision": sha, "research_ens_dirty": dirty}


def host_record() -> dict:
    return {"host": socket.gethostname(), "machine": platform.machine(), "mlx": mx.__version__,
            "device": str(mx.default_device()), "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def fresh_out(path: str | Path) -> Path:
    p = Path(path)
    if p.exists() and any(p.iterdir()):
        raise SystemExit(f"refusing nonempty evidence directory: {p}")
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True))
    tmp.replace(path)


# ------------------------------------------------------------------ checkpoints
def save_core(core: Core, out: Path, name: str) -> str:
    wpath = out / "core.safetensors"
    mx.save_safetensors(str(wpath), dict(tree_flatten(core.parameters())))
    write_json(out / "core_config.json", core.cfg.to_json())
    sha = hashlib.sha256(wpath.read_bytes()).hexdigest()
    return f"{name}@{sha[:12]}"


def load_checkpoint_weights(module, weights: Path, checkpoint: str) -> None:
    with weights.open("rb") as stream:
        fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()[:12]
    if checkpoint.rpartition("@")[2] != fingerprint:
        raise ValueError(f"{weights}: weights differ from the training receipt")
    module.load_weights(str(weights))


def load_core(ckpt_dir: str | Path) -> tuple[Core, str]:
    d = Path(ckpt_dir)
    cfg = CoreConfig.from_json(json.loads((d / "core_config.json").read_text()))
    weights = d / "core.safetensors"
    receipt = json.loads((d / "receipt.json").read_text())
    with weights.open("rb") as stream:
        fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()[:12]
    if receipt["checkpoint"].rpartition("@")[2] != fingerprint or cfg != CoreConfig.from_json(receipt["config"]):
        raise ValueError(f"{d}: core weights or configuration differ from the training receipt")
    core = Core(cfg)
    core.load_weights(str(weights))
    return core, receipt["checkpoint"]


def n_params(module: nn.Module, trainable=False) -> int:
    tree = module.trainable_parameters() if trainable else module.parameters()
    return sum(v.size for _, v in tree_flatten(tree))

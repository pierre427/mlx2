#!/usr/bin/env python3
"""Shared pieces for the amoeba topic map (docs/research/amoeba-topics/DESIGN.md).

Backends (one interface, so the same capture/label/stub code runs on any dense SwiGLU model):
  TorchBackend   HF transformers on the CPU (Qwen3-0.6B and other small models). The MLP activation is
                 the input of down_proj (silu(gate(x)) * up(x)), read and masked by a forward pre-hook.
  Mlx27bBackend  amoeba27b_lib.Host (MLX, GPU, Qwen3.8-27B q8), the same activation via Ctl.stats, masked
                 through Ctl.core (zero ablation).
Both expose: L, N, checkpoint, encode(text), set_mask(mask[L,N] float or None), and
  stats(ids, mu)   per-layer [B, N] mean over positions >= 1 of |act - mu|  (mu: [L, N] or None = 0)
  nll(ids)         per-sequence mean next-token NLL over targets at positions >= 2 (inputs >= 1)
The statistics never use position 0 (the prompt-format / attention-sink artifact).

Statistics helpers (numpy): Cohen's d, AUC, permutation null, random twins with per-layer counts.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
TAXO = ROOT / "docs/research/amoeba-topics/taxonomy.v1.json"


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short=9", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        # an rsync copy without .git (the M3) records the revision it was synced from
        f = Path(ROOT) / "SYNCED_HEAD"
        return f"{f.read_text().strip()} (synced copy)" if f.exists() else "unknown"


def sha_file(p, n=None) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    d = h.hexdigest()
    return d[:n] if n else d


def stable_seed(s: str) -> int:
    """Process-independent seed from a string (Python's hash() is salted per process)."""
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def make_log(out: Path):
    f = open(out / "run.log", "a")

    def log(m):
        line = f"{time.strftime('%H:%M:%S')} {m}"
        print(line, flush=True)
        f.write(line + "\n")
        f.flush()
    return log


def fresh_dir(p) -> Path:
    p = Path(p)
    if p.exists() and any(p.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {p}")
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(p, obj):
    tmp = Path(str(p) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")
    tmp.replace(p)


# ----------------------------------------------------------------------------- taxonomy and corpus

class Taxonomy:
    def __init__(self, path=TAXO):
        d = json.loads(Path(path).read_text())
        self.version = d["version"]
        self.nodes = {n["id"]: n for n in d["nodes"]}
        self.children = {}
        for n in d["nodes"]:
            if n.get("parent"):
                self.children.setdefault(n["parent"], []).append(n["id"])
        self.roots = [n["id"] for n in d["nodes"] if not n.get("parent")]
        self.legacy = d.get("legacy_task_labels", {})

    def ancestors(self, t):
        out = []
        while t:
            out.append(t)
            t = self.nodes[t].get("parent")
        return out


def load_items(path):
    return [json.loads(l) for l in open(path)]


def load_resp(cap, l, N=None):
    """resp_L<l>.npy as float32 [S, N]. Ragged carve captures store each layer at its own width;
    columns past it are zero padding (meta "widths"), so readers see one uniform [S, N] layout."""
    R = np.asarray(np.load(Path(cap) / f"resp_L{l:02d}.npy", mmap_mode="r"), dtype=np.float32)
    if N is not None and R.shape[1] < N:
        R = np.concatenate([R, np.zeros((R.shape[0], N - R.shape[1]), np.float32)], axis=1)
    return R


def tokenize_items(encode, items, seq_len=64, min_len=48):
    """Host-tokenised windows: first seq_len tokens; items shorter than min_len are dropped."""
    out = []
    for it in items:
        ids = encode(it["text"])[:seq_len]
        if len(ids) >= min_len:
            out.append((it, ids))
    return out


def batches_by_len(seqs, bs):
    by = {}
    for j, s in enumerate(seqs):
        by.setdefault(len(s), []).append(j)
    for ln, js in sorted(by.items()):
        for b in range(0, len(js), bs):
            yield js[b:b + bs]


def topic_membership(tax: Taxonomy, items):
    """item index -> set of topic ids including all ancestors of every label."""
    mem = []
    for it in items:
        s = set()
        for t in it["topics"]:
            s.update(tax.ancestors(t))
        mem.append(s)
    return mem


# ----------------------------------------------------------------------------- backends

class TorchBackend:
    def __init__(self, model_id, log=print, dtype="float32", threads=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if threads:
            torch.set_num_threads(threads)
        self.torch = torch
        t0 = time.time()
        path = Path(model_id)
        if not path.exists():
            from huggingface_hub import snapshot_download
            path = Path(snapshot_download(model_id, local_files_only=True))
        self.path = path
        self.tok = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=getattr(torch, dtype))
        self.model.eval()
        cfg = self.model.config
        self.L, self.N = cfg.num_hidden_layers, cfg.intermediate_size
        rev = path.name if path.parent.name == "snapshots" else sha_file(next(path.glob("*.safetensors")), 8)
        name = model_id if not Path(model_id).exists() else path.name
        self.checkpoint = f"{name}@{rev[:8]}"
        self.fingerprint = rev if path.parent.name == "snapshots" else sha_file(next(path.glob("*.safetensors")))
        self._mask = None
        self._cap = None
        self._mu = None
        self._hooks = []
        layers = self.model.model.layers
        for l, layer in enumerate(layers):
            self._hooks.append(layer.mlp.down_proj.register_forward_pre_hook(self._hook(l)))
        log(f"loaded {self.checkpoint} ({self.L} x {self.N}) in {time.time() - t0:.0f}s")

    def _hook(self, l):
        def f(mod, args):
            act = args[0]
            if self._cap is not None:
                mu = 0.0 if self._mu is None else self._mu[l]
                self._cap[l] = (act[:, 1:].float() - mu).abs().mean(1).cpu().numpy()
            if self._mask is not None:
                return (act * self._mask[l].to(act.dtype),)
            return None
        return f

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def set_mask(self, mask):
        self._mask = None if mask is None else self.torch.as_tensor(np.asarray(mask, dtype=np.float32))

    def stats(self, ids, mu=None):
        torch = self.torch
        self._cap = [None] * self.L
        self._mu = None if mu is None else torch.as_tensor(np.asarray(mu, dtype=np.float32))
        with torch.no_grad():
            self.model.model(torch.tensor(ids))
        out, self._cap = self._cap, None
        return out

    def mean_act(self, seqs, bs=8):
        torch = self.torch
        acc = np.zeros((self.L, self.N), dtype=np.float64)
        cnt = 0
        hooks = []

        def mk(l):
            def f(mod, args):
                acc[l] += args[0][:, 1:].float().sum((0, 1)).cpu().numpy()
            return f
        for l, layer in enumerate(self.model.model.layers):
            hooks.append(layer.mlp.down_proj.register_forward_pre_hook(mk(l)))
        try:
            with torch.no_grad():
                for g in batches_by_len(seqs, bs):
                    ids = torch.tensor([seqs[j] for j in g])
                    self.model.model(ids)
                    cnt += ids.shape[0] * (ids.shape[1] - 1)
        finally:
            for h in hooks:
                h.remove()
        return (acc / cnt).astype(np.float32)

    def nll(self, ids):
        torch = self.torch
        with torch.no_grad():
            x = torch.tensor(ids)
            lg = self.model(x).logits[:, 1:-1].float()
            tgt = x[:, 2:]
            ce = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction="none")
            return ce.reshape(tgt.shape).mean(1).numpy()


class Mlx27bBackend:
    def __init__(self, model_path, log=print, policy=None):
        import amoeba27b_lib as A
        self.A = A
        self.host = A.Host(model_path, log=log)
        self.L, self.N = self.host.L, self.host.N
        self.checkpoint = self.host.checkpoint
        self.fingerprint = self.host.fingerprint
        self._mask = None

    def encode(self, text):
        return self.host.encode(text)

    def set_mask(self, mask):
        import mlx.core as mx
        self._mask = None if mask is None else mx.array(np.asarray(mask, dtype=np.float32))

    def _cfg(self):
        self.host.ctl.reset()
        self.host.ctl.core = self._mask

    def stats(self, ids, mu=None):
        import mlx.core as mx
        L = self.L
        cap = [None] * L
        mu_mx = None if mu is None else (mu if isinstance(mu, mx.array) else mx.array(np.asarray(mu, dtype=np.float32)))

        def st(l, act, x):
            a = act[:, 1:].astype(mx.float32)
            cap[l] = mx.abs(a - mu_mx[l]).mean(1) if mu_mx is not None else mx.abs(a).mean(1)
        self._cfg()
        self.host.ctl.stats = st
        h = self.host.hidden(mx.array(ids))
        mx.eval(h, *cap)
        self.host.ctl.reset()
        out = [np.array(c) for c in cap]
        mx.clear_cache()
        return out

    def nll(self, ids):
        import mlx.core as mx
        A = self.A
        x = mx.array(ids)
        self._cfg()
        lg = self.host.logits_at(x, list(range(1, x.shape[1] - 1)))
        ce = A.ce_rows(lg, x[:, 2:]).mean(1)
        mx.eval(ce)
        self.host.ctl.reset()
        out = np.array(ce)
        mx.clear_cache()
        return out


def make_backend(kind, model, log=print, threads=None):
    if kind == "torch":
        return TorchBackend(model, log=log, threads=threads)
    if kind == "mlx27b":
        return Mlx27bBackend(model, log=log)
    raise SystemExit(f"unknown backend {kind}")


# ----------------------------------------------------------------------------- statistics

def cohen_d(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / max(s, 1e-12))


def cohen_d_rows(X, pos, neg):
    """d per column of X [S, G] for row index sets pos, neg (vectorised)."""
    A, B = X[pos], X[neg]
    na, nb = len(pos), len(neg)
    s = np.sqrt(((na - 1) * A.var(0, ddof=1) + (nb - 1) * B.var(0, ddof=1)) / max(na + nb - 2, 1))
    return (A.mean(0) - B.mean(0)) / np.maximum(s, 1e-12)


def auc(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if not len(a) or not len(b):
        return float("nan")
    allv = np.concatenate([a, b])
    ranks = allv.argsort().argsort().astype(np.float64) + 1
    ra = ranks[:len(a)].sum()
    return float((ra - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def perm_null_d(x, pos, neg, n_perm, rng):
    """Null distribution of d(x[pos], x[neg]) under label shuffling within pos+neg (sizes kept)."""
    idx = np.concatenate([pos, neg])
    v = x[idx]
    k = len(pos)
    P = np.stack([rng.permutation(len(idx)) for _ in range(n_perm)])          # [n_perm, n]
    A, B = v[P[:, :k]], v[P[:, k:]]
    s = np.sqrt(((k - 1) * A.var(1, ddof=1) + (len(neg) - 1) * B.var(1, ddof=1)) / max(len(idx) - 2, 1))
    return (A.mean(1) - B.mean(1)) / np.maximum(s, 1e-12)


def twin_sets(members_by_layer, pool_by_layer, n, rng):
    """n random neuron sets with the same per-layer counts, drawn from pool_by_layer (dict l -> array)."""
    out = []
    for _ in range(n):
        tw = {}
        for l, cols in members_by_layer.items():
            pool = pool_by_layer[l]
            tw[l] = np.sort(rng.choice(pool, size=min(len(cols), len(pool)), replace=False))
        out.append(tw)
    return out


def by_layer(pairs):
    """[n, 2] (layer, neuron) -> {layer: sorted neuron array}."""
    pairs = np.asarray(pairs)
    return {int(l): np.sort(pairs[pairs[:, 0] == l, 1]) for l in np.unique(pairs[:, 0])}


def mask_from(by_l, L, N, value=1.0):
    m = np.zeros((L, N), dtype=np.float32)
    for l, cols in by_l.items():
        m[l, cols] = value
    return m

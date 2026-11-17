#!/usr/bin/env python3
"""Amoeba 27B: neuron atlas and unnamed co-firing clusters from a streamed capture (CPU, no model).

Input: the atlas/ directory written by amoeba27b_core.py --phases atlas (per-layer fp16
[S, N] per-sequence peaks of |act - mean| at positions >= 1, domain profiles, top tokens) and the
export core (export_core.npz). Nothing loads the 27B.

1. Atlas (address book; not a measured claim):
   - stable id "<artifact>:L<l>:N<n>"; virtual experts per layer = spherical k-means of neurons
     on relative, centred sequence profiles (the 0.6B unsupervised-box normalisation),
     --ve-per-layer each, id "L<l>:VE<k>";
   - tags: relative domain profile, most selective source, top tokens (decoded);
   - membership: in the export core or in the knowledge pool.
2. Unnamed clusters: the same clustering over the whole knowledge pool (every neuron outside the
   export core, all layers together), --k clusters. No labels are used to form them. Each is
   described by source profile, layer histogram, top sequences and top tokens; functional
   classification (add to core / remove from full, with random twins) runs on the GPU
   (amoeba27b_boxes.py --phases clusters).
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np


def spherical_kmeans(X, k, iters, seed, log):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(X.shape[0], k, replace=False)].copy()
    lab = None
    for it in range(iters):
        sims = X @ C.T
        new = sims.argmax(1)
        if lab is not None and (new == lab).mean() > 0.999:
            lab = new
            break
        lab = new
        for j in range(k):
            m = lab == j
            if m.any():
                c = X[m].sum(0)
                C[j] = c / max(np.linalg.norm(c), 1e-8)
            else:
                C[j] = X[rng.integers(X.shape[0])]
    return lab, C


PROJ = None   # optional [S, D] Gaussian projection (Johnson-Lindenstrauss) of the sequence axis


def profile_rows(F):
    """F: [S, n] peaks -> rows [n, S or D]: relative to each neuron's mean, centred, L2-normalised,
    then (if PROJ is set) projected to D dims and renormalised; cosine is approximately preserved."""
    F = F.astype(np.float32)
    Z = F / np.maximum(F.mean(0, keepdims=True), 1e-8)
    Z = Z - Z.mean(0, keepdims=True)
    X = Z.T
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    if PROJ is not None:
        X = X @ PROJ
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="amoeba27b_core.py evidence dir (atlas/, export_core.npz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=str(Path.home() / "mlx-models/Qwen3.8-27B-BF16-q8-mlx2"))
    ap.add_argument("--ve-per-layer", type=int, default=32)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--atlas-dir", default="", help="capture dir if not <run>/atlas")
    ap.add_argument("--proj-dim", type=int, default=0, help="project the sequence axis to this many dims (0 = none)")
    ap.add_argument("--tags-dir", default="", help="where neuron_tags.jsonl.gz goes (too large for git)")
    args = ap.parse_args()
    run, out = Path(args.run), Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty dir {out}")
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "run.log", "a")

    def log(m):
        line = f"{time.strftime('%H:%M:%S')} {m}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    adir = Path(args.atlas_dir) if args.atlas_dir else run / "atlas"
    meta = json.loads((adir / "meta.json").read_text())
    st = np.load(adir / "atlas_stats.npz", allow_pickle=False)
    core = np.load(run / "export_core.npz")["core"] > 0
    L, N, S = meta["layers"], meta["neurons"], meta["sequences"]
    global PROJ
    if args.proj_dim and args.proj_dim < S:
        PROJ = (np.random.default_rng(args.seed + 31).standard_normal((S, args.proj_dim)) / np.sqrt(args.proj_dim)).astype(np.float32)
    D = PROJ.shape[1] if PROJ is not None else S
    ckpt = meta["checkpoint"]
    names = [str(x) for x in st["domain_names"]]
    prof = st["domain_profile"]                       # [D, L, N]
    top_toks = st["top_toks"]                         # [L, k, N]
    seq_src = np.array([str(x) for x in st["seq_source"]])
    rel = prof / np.maximum(prof.mean(0, keepdims=True), 1e-8)        # relative domain profile
    log(f"{ckpt}: {L} x {N} neurons, {S} sequences, sources {names}; export core keeps {core.mean():.4f}")

    # ------------------------------------------------------------- 1. per-layer virtual experts
    ve = np.zeros((L, N), dtype=np.int16)
    ve_rows = []
    t0 = time.time()
    for l in range(L):
        F = np.load(adir / f"peak_L{l:02d}.npy", mmap_mode="r")
        X = profile_rows(np.asarray(F))
        lab, _ = spherical_kmeans(X, args.ve_per_layer, args.iters, args.seed + l, log)
        ve[l] = lab
        for k in range(args.ve_per_layer):
            m = lab == k
            if not m.any():
                continue
            dom = rel[:, l, m].mean(1)
            toks = top_toks[l, 0, m]
            uniq, cnt = np.unique(toks, return_counts=True)
            ve_rows.append({"id": f"L{l}:VE{k}", "layer": l, "size": int(m.sum()),
                            "core_frac": float(core[l, m].mean()),
                            "domain_rel": {n: round(float(v), 3) for n, v in zip(names, dom)},
                            "top_tokens": [tok.decode([int(t)]) for t in uniq[np.argsort(-cnt)][:8]]})
        if l % 8 == 7:
            log(f"  virtual experts through L{l} ({time.time() - t0:.0f}s)")
    np.savez_compressed(out / "atlas_ve.npz", ve=ve, core=core)
    (out / "virtual_experts.json").write_text(json.dumps(ve_rows, indent=0))

    tags_dir = Path(args.tags_dir) if args.tags_dir else out
    tags_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(tags_dir / "neuron_tags.jsonl.gz", "wt") as f:
        for l in range(L):
            sel = rel[:, l, :].argmax(0)
            for n in range(N):
                f.write(json.dumps({"id": f"{ckpt}:L{l}:N{n}", "ve": f"L{l}:VE{int(ve[l, n])}",
                                    "core": bool(core[l, n]), "selective_for": names[int(sel[n])],
                                    "top_tokens": [int(t) for t in top_toks[l, :4, n]]}) + "\n")
    log(f"atlas: {len(ve_rows)} virtual experts; tags at {tags_dir / 'neuron_tags.jsonl.gz'}")

    # ------------------------------------------------------------- 2. unnamed pool clusters
    pool = np.argwhere(~core)                          # [P, 2]
    P = pool.shape[0]
    log(f"knowledge pool: {P} neurons ({P / (L * N):.4f} of all)")
    X = np.zeros((P, D), dtype=np.float32)
    off = 0
    for l in range(L):
        cols = pool[pool[:, 0] == l, 1]
        if len(cols) == 0:
            continue
        F = np.asarray(np.load(adir / f"peak_L{l:02d}.npy", mmap_mode="r")[:, cols])
        X[off:off + len(cols)] = profile_rows(F)
        off += len(cols)
    lab, C = spherical_kmeans(X, args.k, args.iters, args.seed, log)
    sizes = np.bincount(lab, minlength=args.k)
    log(f"pool clusters: sizes min {sizes.min()} median {int(np.median(sizes))} max {sizes.max()}")
    srcs = sorted(set(seq_src))
    # mean relative peak per source, per cluster (from the profiles' un-normalised means)
    clusters = []
    for j in range(args.k):
        m = lab == j
        members = pool[m]
        # source profile: average over member neurons of (mean peak on source / mean peak overall)
        acc = np.zeros(len(srcs))
        n_used = 0
        for l in np.unique(members[:, 0]):
            cols = members[members[:, 0] == l, 1]
            F = np.asarray(np.load(adir / f"peak_L{l:02d}.npy", mmap_mode="r")[:, cols]).astype(np.float32)
            base = np.maximum(F.mean(0), 1e-8)
            for si, s in enumerate(srcs):
                acc[si] += (F[seq_src == s].mean(0) / base).sum()
            n_used += len(cols)
        src_rel = acc / max(1, n_used)
        hist = np.bincount(members[:, 0], minlength=L)
        toks = top_toks[members[:, 0], 0, members[:, 1]]
        uniq, cnt = np.unique(toks, return_counts=True)
        cs = C[j].astype(np.float32) @ PROJ.T if PROJ is not None else C[j]   # back to sequence space (approx.)
        top_seq = np.argsort(-cs)[:5]
        clusters.append({"id": f"pool:C{j}", "size": int(m.sum()),
                         "layers": {"min": int(members[:, 0].min()), "max": int(members[:, 0].max()),
                                    "median": int(np.median(members[:, 0])), "hist": hist.tolist()},
                         "source_rel": {s: round(float(v), 3) for s, v in zip(srcs, src_rel)},
                         "most_selective_source": srcs[int(np.argmax(src_rel))],
                         "selectivity": round(float(src_rel.max() / max(np.median(src_rel), 1e-8)), 3),
                         "top_tokens": [tok.decode([int(t)]) for t in uniq[np.argsort(-cnt)][:10]],
                         "top_sequence_ids": [int(i) for i in top_seq],
                         "top_sequence_sources": [str(seq_src[i]) for i in top_seq]})
    np.savez_compressed(out / "pool_clusters.npz", pool=pool.astype(np.int32), labels=lab.astype(np.int16),
                        centroids=C.astype(np.float16))
    (out / "pool_clusters.json").write_text(json.dumps({"checkpoint": ckpt, "k": args.k, "pool_neurons": int(P),
                                                       "normalisation": "relative to neuron mean, centred, L2",
                                                       "labels_used_to_form": "none",
                                                       "clusters": clusters}, indent=1))
    log(f"wrote {out}")


if __name__ == "__main__":
    main()

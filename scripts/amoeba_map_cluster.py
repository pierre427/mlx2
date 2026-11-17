#!/usr/bin/env python3
"""aMoEba map, step 2: unsupervised clusters over every unit of a topic capture (CPU, no model).

Input: a topic capture (scripts/amoeba_topics_capture.py: resp_L*.npy [S, N] fp16, items.json,
meta.json with optional per-layer "widths" for ragged carve captures). No labels are used to form
the clusters; only the fit split's texts enter the profiles.

Two levels, so models with millions of units fit in memory:
  1. virtual experts: per layer, spherical k-means of that layer's units on their fit-split topic
     profile (relative to the unit's mean, centred over texts, L2-normalised: the normalisation of
     scripts/amoeba27b_atlas.py and scripts/amoeba_topics_label.py). A layer of routed experts is
     clustered the same way, its units being experts.
  2. global clusters: weighted spherical k-means of all virtual-expert centroids (weight = size).
     Each unit takes its virtual expert's global cluster.

Writes OUT/clusters.npz (pool [P, 2] = (layer, unit), labels [P] int16: the --clusters input of
amoeba_topics_label.py), OUT/virtual_experts.npz (ve [L, N] int16, -1 = padding), and
OUT/clusters.json (sizes, layer spans, widths, revision, checkpoint).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np


def spherical_kmeans(X, k, iters=30, seed=0, w=None):
    """Rows of X are unit vectors. Weighted (w per row) Lloyd iterations on cosine similarity."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    k = min(k, n)
    p = None if w is None else w / w.sum()
    C = X[rng.choice(n, k, replace=False, p=p)].copy()
    lab = None
    for _ in range(iters):
        new = (X @ C.T).argmax(1)
        if lab is not None and (new == lab).mean() > 0.999:
            lab = new
            break
        lab = new
        for j in range(k):
            m = lab == j
            if m.any():
                c = (X[m] * (w[m][:, None] if w is not None else 1.0)).sum(0)
                C[j] = c / max(np.linalg.norm(c), 1e-8)
            else:
                C[j] = X[rng.integers(n)]
    return lab, C


def profiles(F):
    """F [S, n] responses -> [n, S] relative, centred, unit-norm profiles."""
    Z = F / np.maximum(F.mean(0, keepdims=True), 1e-8)
    Z = Z - Z.mean(0, keepdims=True)
    X = np.ascontiguousarray(Z.T)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short=9", "HEAD"], text=True,
                                       cwd=Path(__file__).resolve().parent).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=64, help="global clusters")
    ap.add_argument("--ve-per-layer", type=int, default=64, help="cap; a layer of n units gets min(cap, n // 4)")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", default="", help="NPZ[:key] [L, N] mask (>0 = excluded, e.g. the English core): those "
                    "units are left out of the clusters (mapping-workflow s5.1.2: cluster the non-core units)")
    args = ap.parse_args()
    cap, out = Path(args.capture), Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty dir {out}")
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((cap / "meta.json").read_text())
    items = json.loads((cap / "items.json").read_text())
    L, N, S = meta["layers"], meta["neurons"], meta["sequences"]
    widths = meta.get("widths") or [N] * L
    fit = np.array([it["split"] == "fit" for it in items])
    t0 = time.time()

    def log(m):
        print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)

    log(f"{meta['checkpoint']}: {L} layers, {sum(widths)} units, {S} texts ({int(fit.sum())} fit)")
    excl = np.zeros((L, N), bool)
    if args.exclude:
        pth, _, key = args.exclude.partition(":")
        z = np.load(pth)
        e = z[key or z.files[0]] > 0
        excl[:, :min(N, e.shape[1])] = e[:L, :N]
        log(f"excluding {int(excl.sum())} units ({args.exclude})")
    ve = np.full((L, N), -1, dtype=np.int16)
    cents, sizes, owner = [], [], []
    dead = 0
    for l in range(L):
        n = int(widths[l])
        if n == 0:
            continue
        keep = np.nonzero(~excl[l, :n])[0]
        if not len(keep):
            continue
        F = np.asarray(np.load(cap / f"resp_L{l:02d}.npy", mmap_mode="r")[:, :n], dtype=np.float32)[fit][:, keep]
        live = F.max(0) > 0
        dead += int((~live).sum())
        X = profiles(F)
        k = max(1, min(args.ve_per_layer, len(keep) // 4))
        lab, C = spherical_kmeans(X, k, args.iters, args.seed + l)
        ve[l, keep] = lab
        for j in range(C.shape[0]):
            m = int((lab == j).sum())
            if m:
                cents.append(C[j])
                sizes.append(m)
                owner.append((l, j))
        if l % 8 == 7 or l == L - 1:
            log(f"  virtual experts through L{l} ({time.time() - t0:.0f}s)")
    Cv = np.stack(cents).astype(np.float32)
    w = np.asarray(sizes, np.float64)
    glab, G = spherical_kmeans(Cv, args.k, args.iters, args.seed, w=w)
    vmap = {o: int(g) for o, g in zip(owner, glab)}
    pool, labels = [], []
    for l in range(L):
        n = int(widths[l])
        for u in range(n):
            if ve[l, u] < 0:
                continue                      # excluded (core) unit
            pool.append((l, u))
            labels.append(vmap[(l, int(ve[l, u]))])
    pool = np.asarray(pool, np.int32)
    labels = np.asarray(labels, np.int16)
    K = int(labels.max()) + 1
    np.savez_compressed(out / "clusters.npz", pool=pool, labels=labels)
    np.savez_compressed(out / "virtual_experts.npz", ve=ve, widths=np.asarray(widths, np.int32),
                        global_of_ve=np.asarray([[l, j, vmap[(l, j)]] for (l, j) in owner], np.int32))
    rows = []
    for j in range(K):
        m = labels == j
        ls = pool[m, 0]
        rows.append({"id": f"pool:C{j}", "units": int(m.sum()), "layers": {"min": int(ls.min()), "max": int(ls.max()),
                     "median": int(np.median(ls)), "hist": np.bincount(ls, minlength=L).tolist()}})
    rec = {"schema": "mlx2.amoeba-map-clusters.v1", "checkpoint": meta["checkpoint"],
           "capture": str(cap), "capture_revision": meta.get("mlx2_source_revision"),
           "corpus_sha256": meta.get("corpus_sha256"), "mlx2_source_revision": git_rev(),
           "method": "per-layer spherical k-means virtual experts on fit-split topic profiles, then weighted "
                     "spherical k-means of their centroids; labels_used_to_form: none",
           "args": vars(args), "excluded_units": int(excl.sum()), "layers": L, "widths": [int(x) for x in widths], "units": int(len(pool)),
           "units_never_active_on_fit": dead, "virtual_experts": int(len(owner)), "clusters": rows,
           "seconds": round(time.time() - t0, 1)}
    (out / "clusters.json").write_text(json.dumps(rec, indent=1))
    log(f"{len(owner)} virtual experts -> {K} clusters; sizes min {min(r['units'] for r in rows)} "
        f"max {max(r['units'] for r in rows)}; wrote {out}")


if __name__ == "__main__":
    main()

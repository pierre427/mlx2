"""Fisher carving: carve masks from saved per-sequence mask gradients, with no model loaded.

Capture (GPU, model loaded). For each sequence, labels y_t ~ p_full(. | x_<t) are sampled at
positions >= 1 (seeded), and U = d log p(y | x) / d m is taken at the full model (m = 1), with the
serving ablation (mean for dense neurons, drop rule for experts) and position 0 exact. U is a
vector per sequence and layer: for a dense neuron i, U_i = sum_t (W_down^T g_t)_i (act_ti - mu_i).

Second-order estimate. Dropping the units with (1 - m) changes the mean per-position KL by about

    cost(m) = 1/(2 (T-1)) * E_seq[ ((1 - m) . U_seq)^2 ]                                  (*)

(the Fisher information of the output distribution; label sampling makes cross-sequence and
cross-position terms vanish in expectation, so one vector per sequence is unbiased). (*) is
block diagonal across layers: interactions between layers are ignored, and so is everything past
second order. That is what the e2e refinement (carve.py rank, warm-started) is for.

With J vectors and N units per layer, the J-sample estimate has rank <= J and a mask can cancel
its sum on those samples; the carve therefore reports cost on held-out vectors, and blends in the
diagonal (no-cancellation) term ``alpha / (2 (T-1)) * sum_i (1 - m_i) mean_j U_ji^2`` (``--fisher-alpha``,
a fixed setting: choose it by comparing held-out cost across runs; no automatic search).

Status: implemented, unmeasured. Whether it ranks as well as the e2e carve is the A/B in
docs/research/amoeba-carve/README.md; until then the e2e carve stays the default.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np


def capture(host, seqs, out_dir: Path, mean=None, seed=0, log=print, every=32):
    """Write U vectors: out_dir/U_L{l:02d}.npy float32 [J, n_l], J = len(seqs). Written into
    out_dir.partial and renamed only when every vector is in, so a killed capture never looks done."""
    import shutil
    import mlx.core as mx
    from .carving import ce_rows
    final = out_dir
    if final.exists():
        raise SystemExit(f"fisher capture: {final} exists")
    out_dir = final.with_name(final.name + ".partial")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    J = len(seqs)
    mm = [np.lib.format.open_memmap(out_dir / f"U_L{l:02d}.npy", mode="w+", dtype=np.float32, shape=(J, n))
          for l, n in enumerate(host.widths)]
    key = mx.random.key(seed)
    ones = host.ones()
    t0 = time.time()
    for j, q in enumerate(seqs):
        ids = mx.array([q])
        pos = list(range(1, ids.shape[1]))
        host.ctl.reset()
        lc = host.logits_at(ids, pos)
        key, sub = mx.random.split(key)
        y = mx.stop_gradient(mx.random.categorical(lc, key=sub))               # 1, T-1
        mx.eval(y)

        def hl(hid, y=y, pos=pos):
            return -ce_rows(host.head(hid[:, mx.array(pos)]), y).sum()           # log p(y)
        host.train_mode(True)
        _, g = host.value_and_grad(ids, ones, hl, mean=mean)
        host.train_mode(False)
        g = g["core"] if isinstance(g, dict) else g
        for l in range(host.L):
            mm[l][j] = np.array(g[l], dtype=np.float32)
        mx.clear_cache()
        if j % every == 0 or j == J - 1:
            log(f"  fisher capture {j + 1}/{J} ({(time.time() - t0) / (j + 1):.2f} s/seq)")
    for m in mm:
        m.flush()
    del mm
    out_dir.rename(final)
    return {"vectors": J, "seq_len": len(seqs[0]), "seed": seed, "s": round(time.time() - t0, 1)}


def load(out_dir: Path, L: int):
    return [np.load(out_dir / f"U_L{l:02d}.npy", mmap_mode="r") for l in range(L)]


def cost(U, drop, T1):
    """(*) for one layer: U [J, n], drop [n] in {0,1}."""
    r = np.asarray(U, dtype=np.float64) @ np.asarray(drop, dtype=np.float64)
    return float(0.5 * (r ** 2).mean() / T1)


def carve(Us, fixed, pool, lam, T1, steps=300, lr=0.1, alpha=0.0, val_frac=0.25, seed=0, log=print, z0=3.0):
    """Straight-through carve of every layer from U (the e2e recipe with (*) as the loss).

    fixed: per-layer 0/1 (always kept, e.g. a lower tier); pool: per-layer 0/1 (units allowed to be
    dropped; None = all). Penalty lam * kept fraction of the free units, normalised over all layers
    as in train_mask. Returns (Z per layer, report). Layers are independent under (*)."""
    import mlx.core as mx
    import mlx.optimizers as optim
    rng = np.random.default_rng(seed)
    J = Us[0].shape[0]
    perm = rng.permutation(J)
    nval = int(round(val_frac * J))
    val, fit = perm[:nval], perm[nval:]
    frees = []
    for l, U in enumerate(Us):
        f = 1.0 - np.asarray(fixed[l], dtype=np.float32)
        if pool is not None:
            f = f * np.asarray(pool[l], dtype=np.float32)
        frees.append(f)
    nfree = max(1.0, float(sum(f.sum() for f in frees)))
    Zs, rep = [], {"layers": [], "fit_vectors": int(len(fit)), "val_vectors": int(nval), "alpha": alpha, "lam": lam}
    t0 = time.time()
    for l, U in enumerate(Us):
        Uf = mx.array(np.asarray(U)[fit])
        diag = mx.array((np.asarray(U)[fit].astype(np.float64) ** 2).mean(0).astype(np.float32))
        free = mx.array(frees[l])
        fx = mx.array(np.asarray(fixed[l], dtype=np.float32))
        params = {"Z": mx.full(free.shape, z0)}
        opt = optim.Adam(learning_rate=lr)
        for _ in range(steps):
            Z = params["Z"]
            m = fx + free * (Z > 0).astype(mx.float32)
            d = 1.0 - m
            r = Uf @ d                                                    # J_fit
            # d cost / d m = -(1/T1) mean_j r_j U_j - alpha/(2 T1) diag
            gm = -(Uf.T @ r) / (T1 * Uf.shape[0]) * (1 - alpha) - alpha * diag / (2 * T1)
            s = mx.sigmoid(Z)
            gZ = free * s * (1 - s) * (gm + lam / nfree)
            params = opt.apply_gradients({"Z": gZ}, params)
            mx.eval(params, opt.state)
        Z = np.array(params["Z"])
        Z[(frees[l] == 0) & (np.asarray(fixed[l]) == 0)] = -1e9      # outside the pool: dropped, never kept
        drop = 1.0 - (np.asarray(fixed[l]) + frees[l] * (Z > 0))
        Uv = np.asarray(U)[val] if nval else np.asarray(U)[fit]
        row = {"layer": l, "kept": float(1 - drop.mean()), "cost_fit": cost(np.asarray(U)[fit], drop, T1),
               "cost_val": cost(Uv, drop, T1)}
        rep["layers"].append(row)
        Zs.append(Z)
    rep["cost_fit"] = sum(r["cost_fit"] for r in rep["layers"])
    rep["cost_val"] = sum(r["cost_val"] for r in rep["layers"])
    rep["kept"] = float(np.mean(np.concatenate([1 - (1.0 - (np.asarray(fixed[l]) + frees[l] * (Zs[l] > 0)))
                                                for l in range(len(Us))])))
    rep["s"] = round(time.time() - t0, 1)
    log(f"  fisher carve lam={lam} alpha={alpha}: kept {rep['kept']:.4f} predicted dKL fit {rep['cost_fit']:.4f} "
        f"val {rep['cost_val']:.4f} ({rep['s']} s)")
    return Zs, rep

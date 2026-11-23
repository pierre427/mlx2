"""H14: a tile serving incompatible sub-tasks splits where its per-row gradients cluster.

One MLP tile is trained on a mixed pool of two dev laws on different structures (A, B). Its tap sees only the
key, and the two laws want different z for the same key, so one tile cannot serve both. Then:

  1. per-row gradients of the loss on the tile's output layer (dL/dW3 = dL/do x a2), randomly projected
  2. k-means (k = 2) on those features -> cluster labels; ARI against the true family labels
  3. children: copies of the parent, each fine-tuned on its cluster's rows only
  4. a router (linear on the core's SLOT + key state) trained on *cluster* labels, never family labels
  5. eval on the mixed held pool

Arms: parent (one tile), split (router -> child), split_oracle_route (true family -> its child), separate
(tiles trained on each true family from scratch, oracle choice), random_split (random labels, same steps).

  python -m research.ens.split --core <c6 dir> --family-a graph.4-2.h1 --family-b fst.3-1.h1 --out <dir>
"""
from __future__ import annotations

import argparse
import copy
import json
import random

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from research.ens import common as C
from research.ens import families as F
from research.ens.contrib import Module, check_core_families, core_site, unit_rms
from research.ens.claim import _pick
from research.ens.model import BusCall, value_logits
from research.ens.route import site_hidden


def _train(core, site, mod, rows, steps, *, lr=1e-3, batch=64, seed=0):
    opt = optim.Adam(learning_rate=lr)
    rng = random.Random(seed)

    def lf(m, tokens, q, f, a, ans):
        return nn.losses.cross_entropy(value_logits(core(tokens, BusCall(site, q, f, m)), a), ans, reduction="mean")
    g = nn.value_and_grad(mod, lf)
    for _ in range(steps):
        b = [rows[rng.randrange(len(rows))] for _ in range(batch)]
        tokens, q, f, a, ans, _ = C.batch(b, "bus")
        _, gr = g(mod, tokens, q, f, a, ans)
        opt.update(mod, gr)
        mx.eval(mod.parameters(), opt.state)
    return mod


def grad_features(core, site, mod, rows, dim=256, seed=0):
    """Per-row features: dL/dW3 = outer(dL/do, a2) randomly projected; dL/do alone; and the state baseline."""
    tokens, q, f, a, ans, _ = C.batch(rows, "bus")
    h = site_hidden(core, tokens, site)
    tap = core.ports[site].tap(h[mx.arange(len(rows)), q])
    a2 = nn.silu(mod.l2(nn.silu(mod.l1(tap))))
    o = mod.l3(a2)

    def per_row_loss_sum(o_):
        lg = value_logits(core(tokens, BusCall(site, q, f, unit_rms(o_))), a)
        return nn.losses.cross_entropy(lg, ans, reduction="sum")
    go = mx.grad(per_row_loss_sum)(o)                         # (B, w): rows are independent
    outer = (go[:, :, None] * a2[:, None, :]).reshape(len(rows), -1)
    R = mx.random.normal((outer.shape[1], dim), key=mx.random.key(seed)) / dim ** 0.5
    keys = np.array([r.x for r in rows])
    go_np = np.array(go)
    go_centred = go_np - np.stack([go_np[keys == k].mean(0) if np.any(keys == k) else 0 * go_np[0] for k in keys])
    state = np.array(_pick(h, q, f, "both"))           # no gradients: what the core state alone separates
    return {"outer": np.array(outer @ R), "go": go_np, "go_key_centred": go_centred, "state": state}


def kmeans2(X, iters=50, seed=0):
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(8):
        c = X[rng.choice(len(X), 2, replace=False)]
        for _ in range(iters):
            lab = np.argmax(X @ c.T, axis=1)
            c = np.stack([X[lab == k].mean(0) if np.any(lab == k) else c[k] for k in (0, 1)])
        score = sum(float((X[lab == k] @ c[k]).sum()) for k in (0, 1))
        if best is None or score > best[0]:
            best = (score, lab)
    return best[1]


def ari(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    ct = np.zeros((a.max() + 1, b.max() + 1))
    for x, y in zip(a, b):
        ct[x, y] += 1
    comb = lambda v: v * (v - 1) / 2
    sij, sa, sb = comb(ct).sum(), comb(ct.sum(1)).sum(), comb(ct.sum(0)).sum()
    exp = sa * sb / comb(n)
    return float((sij - exp) / (0.5 * (sa + sb) - exp)) if (0.5 * (sa + sb) - exp) else 0.0


def train_router(core, site, rows, labels, steps=1500, seed=0):
    router = nn.Linear(core.cfg.d * 2, 2)
    opt = optim.Adam(learning_rate=1e-3)
    rng = random.Random(seed)

    def lf(r, h, y):
        return nn.losses.cross_entropy(r(h), y, reduction="mean")
    g = nn.value_and_grad(router, lf)
    for _ in range(steps):
        idx = [rng.randrange(len(rows)) for _ in range(128)]
        tokens, q, f, *_ = C.batch([rows[i] for i in idx], "bus")
        _, gr = g(router, _pick(site_hidden(core, tokens, site), q, f, "both"), mx.array([labels[i] for i in idx]))
        opt.update(router, gr)
        mx.eval(router.parameters(), opt.state)
    return router


def probe(core, site, fa, fb, *, steps=6000, child_steps=3000, n_feat=1024, seed=0) -> dict:
    w = core.cfg.bus_width
    ta, tb = F.family_rows(fa, "train"), F.family_rows(fb, "train")
    train = ta + tb
    fam_lab = [0] * len(ta) + [1] * len(tb)
    ha, hb = F.family_rows(fa, "held"), F.family_rows(fb, "held")
    held = ha + hb
    held_lab = mx.array([0] * len(ha) + [1] * len(hb))
    mx.random.seed(seed)
    parent = _train(core, site, Module(w), train, steps, seed=seed)
    rng = random.Random(seed)
    sample = rng.sample(range(len(train)), n_feat)
    feats = grad_features(core, site, parent, [train[i] for i in sample], seed=seed)
    truth = [fam_lab[i] for i in sample]
    clus = kmeans2(feats["outer"], seed=seed)
    res = {"families": [fa.fid, fb.fid], "n_feat": n_feat,
           "ari": ari(clus, truth),
           # review controls: the families differ in structure, so any feature may separate them
           "ari_go_only": ari(kmeans2(feats["go"], seed=seed), truth),
           "ari_go_key_centred": ari(kmeans2(feats["go_key_centred"], seed=seed), truth),
           "ari_state_no_gradient": ari(kmeans2(feats["state"], seed=seed), truth),
           "cluster_sizes": [int((clus == k).sum()) for k in (0, 1)]}
    # assign every training row a cluster: nearest centroid in feature space would need features for all rows;
    # use the router trained on the sampled rows' cluster labels instead (it only sees core state, not labels)
    rs = [train[i] for i in sample]
    router = train_router(core, site, rs, [int(c) for c in clus], seed=seed)

    def route(rows):
        out = []
        for i in range(0, len(rows), 512):
            tokens, q, f, *_ = C.batch(rows[i:i + 512], "bus")
            out += mx.argmax(router(_pick(site_hidden(core, tokens, site), q, f, "both")), -1).tolist()
        return out
    tr_clus = route(train)
    children, res["child_rows"], res["child_fallback"] = [], [], []
    for k in (0, 1):
        rows_k = [r for r, c in zip(train, tr_clus) if c == k]
        res["child_rows"].append(len(rows_k)); res["child_fallback"].append(not rows_k)
        children.append(_train(core, site, copy.deepcopy(parent), rows_k or train, child_steps, seed=seed + 1 + k))

    tokens, q, f, a, ans, zs = C.batch(held, "bus")

    def acc_with(choice, mods):
        from research.ens.evaluate import module_messages
        msgs = mx.stack([module_messages(core, site, m, tokens, q, f, zs) for m in mods], 1)
        pred = mx.argmax(value_logits(core(tokens, BusCall(site, q, f, msgs[mx.arange(len(held)), choice])), a), -1)
        hit = pred == ans
        return {"all": float(mx.mean(hit).item()),
                fa.fid: float((mx.sum(hit & (held_lab == 0)) / mx.sum(held_lab == 0)).item()),
                fb.fid: float((mx.sum(hit & (held_lab == 1)) / mx.sum(held_lab == 1)).item())}
    zero = mx.zeros((len(held),), dtype=mx.int32)
    res["parent"] = acc_with(zero, [parent])
    res["split"] = acc_with(mx.array(route(held)), children)
    # which child serves which family: the child whose cluster holds most of that family's rows
    maj = [int(np.bincount([c for c, l in zip(tr_clus, fam_lab) if l == k], minlength=2).argmax()) for k in (0, 1)]
    res["child_for_family"] = maj
    res["split_oracle_route"] = acc_with(mx.array([maj[l] for l in held_lab.tolist()]), children)
    # budget-matched: each child had the parent's steps plus its own (review F3)
    sep = [_train(core, site, Module(w), t, steps + child_steps, seed=seed + 10 + i) for i, t in enumerate((ta, tb))]
    res["separate_steps"] = steps + child_steps
    res["separate_oracle"] = acc_with(held_lab, sep)
    # random split control: same procedure with random cluster labels
    rlab = [rng.randrange(2) for _ in sample]
    rrouter = train_router(core, site, rs, rlab, seed=seed + 5)
    rtr = []
    for i in range(0, len(train), 512):
        tk, qq, ff, *_ = C.batch(train[i:i + 512], "bus")
        rtr += mx.argmax(rrouter(_pick(site_hidden(core, tk, site), qq, ff, "both")), -1).tolist()
    res["random_split_ari_vs_family"] = ari(rtr, fam_lab)
    res["random_child_rows"] = [sum(1 for c in rtr if c == k) for k in (0, 1)]
    rchildren = [_train(core, site, copy.deepcopy(parent), [r for r, c in zip(train, rtr) if c == k] or train,
                        child_steps, seed=seed + 20 + k) for k in (0, 1)]
    tk, qq, ff, *_ = C.batch(held, "bus")
    rchoice = mx.argmax(rrouter(_pick(site_hidden(core, tk, site), qq, ff, "both")), -1)
    res["random_split"] = acc_with(rchoice, rchildren)
    per = lambda d: min(d[fa.fid], d[fb.fid])
    res["kill"] = {"ari_le_0.2": res["ari"] <= 0.2,
                   "split_below_separate_by_0.05": per(res["split"]) < per(res["separate_oracle"]) - 0.05,
                   "random_as_good": per(res["random_split"]) >= per(res["split"]) - 0.02,
                   "gradients_add_nothing_over_state": res["ari_state_no_gradient"] >= res["ari"] - 0.05}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--core", required=True)
    ap.add_argument("--family-a", required=True)
    ap.add_argument("--family-b", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--child-steps", type=int, default=3000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = C.fresh_out(args.out)
    core, ckpt = C.load_core(args.core)
    core.freeze()
    check_core_families(args.core)
    site = core_site(args.core)
    fa, fb = F.BY_ID[args.family_a], F.BY_ID[args.family_b]
    for fam in (fa, fb):
        if fam.split in F.SEALED or fam.split == "train":
            raise SystemExit(f"{fam.fid}: not a dev contributor family")
    res = probe(core, site, fa, fb, steps=args.steps, child_steps=args.child_steps)
    C.write_json(out / "receipt.json", {"schema": "ens.a0.split_receipt.v0", "hypothesis": "H14", "core_checkpoint": ckpt,
                                       "site": site, "results": res, **C.source_revision(), **C.host_record()})
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

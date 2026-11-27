"""Donor as suggestion, not fact: downweighted / capped donor evidence in the L28 code table.

Key: k-means-4096 code of the donor's L28 state on PCA-128 (the best arm in donor_grid2.py).
All arms back off to the same donor-free base (KN, 2-token context), so the donor enters
only through the code table:

  TEXT                code table from hard counts only (what the text says, keyed by code)
  SOFT a=X            hard + X * donor top-k probs (X = 1 reproduces the earlier best arm)
  PRIOR rho=X         donor as a capped Dirichlet prior: per code, donor pseudo-count
                      = X * (hard count of that code), spread by the donor's average
                      distribution for that code. With rho <= 1 the donor never holds more
                      evidence than the text for that code.
  ... GATE            each arm blended with KN 4-token counts by log2 code-count bucket

Override metrics (vs TEXT, same key and backoff):
  override_2x         share of held tokens whose probability the arm cut by more than 2x
  override_2x_seen    same, restricted to (code, token) pairs the text showed >= 3 times
"""

import argparse
import json
import time

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from donor_grid2 import Table, context_ids, nll  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--k", type=int, default=4096)
    args = ap.parse_args()
    t0 = time.time()
    log = lambda *m: print(f"[{time.time() - t0:6.1f}s]", *m, flush=True)
    rng = np.random.default_rng(0)

    S = {s: dict(np.load(f"{args.prefix}.{s}.npz")) for s in ("fit", "val", "held")}
    V0 = int(max(S[s][k].max() for s in S for k in ("ids", "top_ids"))) + 1
    BOS, W = V0, V0 + 1
    ctx = context_ids(S, BOS, 4)
    valid, nxt = {}, {}
    for s, d in S.items():
        n = len(d["ids"])
        last = np.zeros(n, bool)
        last[np.concatenate([d["doc_start"][1:] - 1, [n - 1]])] = True
        valid[s] = ~last & np.isfinite(d["lp_next"])
        nxt[s] = np.concatenate([d["ids"][1:], [0]]).astype(np.int64)
    M = valid
    w = {s: nxt[s][M[s]] for s in S}
    nf = int(M["fit"].sum())
    Kt = S["fit"]["top_ids"].shape[1]
    soft_w = S["fit"]["top_ids"][M["fit"]].reshape(-1).astype(np.int64)
    soft_p = np.exp(S["fit"]["top_lp"][M["fit"]].astype(np.float32)).reshape(-1).astype(np.float64)
    rep = lambda x: np.repeat(x, Kt)

    # donor-free KN chain (base for backoff, and the memory side of the gate)
    kn = {c: Table(ctx[c]["fit"][M["fit"]], w["fit"], np.ones(nf), W) for c in range(1, 5)}
    cont = np.bincount(kn[1].k % W, minlength=W) + 0.5
    p1 = cont / cont.sum()

    def kn_chain(s, upto, D=0.75):
        p = p1[w[s]]
        for c in range(1, upto + 1):
            cx = ctx[c][s][M[s]]
            tot, n1 = kn[c].tot(cx)
            with np.errstate(divide="ignore", invalid="ignore"):
                p = np.where(tot > 0, (np.maximum(kn[c].pair(cx, w[s]) - D, 0) + D * n1 * p) / tot, p)
        return p
    base = {s: kn_chain(s, 2) for s in ("val", "held")}
    kn4 = {s: kn_chain(s, 4) for s in ("val", "held")}

    # codes (same recipe as donor_grid2)
    H = {s: np.load(f"{args.prefix}.{s}.h{args.layer}.npy", mmap_mode="r")[:len(S[s]["ids"])] for s in S}
    samp = np.sort(rng.choice(np.where(M["fit"])[0], 150000, replace=False))
    X = np.asarray(H["fit"][samp], np.float32)
    mu = X.mean(0)
    P = np.linalg.svd(X - mu, full_matrices=False)[2][:128].T
    Z = {}
    for s in S:
        idx = np.where(M[s])[0]
        Z[s] = np.concatenate([(np.asarray(H[s][idx[i:i + 100000]], np.float32) - mu) @ P
                               for i in range(0, len(idx), 100000)])

    def assign(Xs, C):
        cn = (C ** 2).sum(1)
        return np.concatenate([np.argmin(cn - 2 * Xs[i:i + 50000] @ C.T, 1) for i in range(0, len(Xs), 50000)])
    sub = Z["fit"][rng.choice(len(Z["fit"]), 120000, replace=False)]
    C = sub[rng.choice(len(sub), args.k, replace=False)].copy()
    for _ in range(15):
        a = assign(sub, C)
        sums = np.zeros_like(C)
        np.add.at(sums, a, sub)
        n = np.bincount(a, minlength=args.k)
        C = np.where(n[:, None] > 0, sums / np.maximum(n, 1)[:, None], sub[rng.choice(len(sub), args.k)])
    code = {s: assign(Z[s], C) for s in S}
    log("codes done")

    hard = Table(code["fit"], w["fit"], np.ones(nf), W)
    soft = Table(rep(code["fit"]), soft_w, soft_p, W)        # donor-only evidence per code

    sl_all = np.ones(int(M["held"].sum()), bool)
    seen3 = hard.pair(code["held"], w["held"]) >= 3
    unseen_bi = kn[1].pair(ctx[1]["held"][M["held"]], w["held"]) == 0
    slices = {"all": sl_all, "unseen_bigram": unseen_bi, "text_seen_3x": seen3}
    held_lp = S["held"]["lp_next"][M["held"]]
    res = {"prefix": args.prefix, "layer": args.layer, "k": args.k,
           "slice_sizes": {k: int(v.sum()) for k, v in slices.items()},
           "donor": {k: float(-np.mean(held_lp[m])) for k, m in slices.items()}, "arms": {}}
    score = lambda p: {k: nll(p[m]) for k, m in slices.items()}

    def table_prob(s, a, rho, beta):
        cs, ws = code[s], w[s]
        hp = hard.pair(cs, ws)
        ht, _ = hard.tot(cs)
        sp = soft.pair(cs, ws)
        st, _ = soft.tot(cs)
        if rho is None:              # soft counts, weight a
            num, den = hp + a * sp, ht + a * st
        else:                        # capped prior: donor mass = rho * text mass for this code
            kap = rho * ht
            q = np.where(st > 0, sp / np.maximum(st, 1e-12), 0)
            num, den = hp + kap * q, ht + kap
        return (num + beta * base[s]) / (den + beta), den

    def fit_arm(a, rho):
        betas = (1, 3, 10, 30, 100, 300, 1000, 3000)
        b = min(betas, key=lambda bb: nll(table_prob("val", a, rho, bb)[0]))
        pv, dv = table_prob("val", a, rho, b)
        ph, dh = table_prob("held", a, rho, b)
        bv = np.minimum(np.floor(np.log2(dv + 1)), 14).astype(int)
        bh = np.minimum(np.floor(np.log2(dh + 1)), 14).astype(int)
        pg = np.empty_like(ph)
        for k in range(15):
            mv, mh = bv == k, bh == k
            lam = 0.3 if not mv.any() else min((x / 20 for x in range(21)),
                                               key=lambda l: nll(l * kn4["val"][mv] + (1 - l) * pv[mv]))
            pg[mh] = lam * kn4["held"][mh] + (1 - lam) * ph[mh]
        return b, ph, pg

    _, p_text, g_text = fit_arm(0.0, None)
    arms = [("TEXT", 0.0, None)] + [(f"SOFT a={a}", a, None) for a in (0.03, 0.1, 0.3, 1.0, 3.0)] + \
           [(f"PRIOR rho={r}", None, r) for r in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)]
    for name, a, rho in arms:
        b, ph, pg = fit_arm(a if a is not None else 0.0, rho)
        cut = ph < 0.5 * p_text
        lr = np.log(ph / p_text)
        row = {"beta": b, **score(ph), "gate": score(pg),
               "override_2x": float(cut.mean()), "override_2x_seen": float(cut[seen3].mean()),
               "override_1.25x": float((lr < -np.log(1.25)).mean()),
               "override_1.25x_seen": float((lr[seen3] < -np.log(1.25)).mean()),
               "override_1.1x_seen": float((lr[seen3] < -np.log(1.1)).mean()),
               "worst_cut_seen_x": float(np.exp(-lr[seen3].min())),
               "boost_2x": float((ph > 2 * p_text).mean())}
        res["arms"][name] = row
        log(name, {k: round(v, 3) for k, v in score(ph).items()}, "| gate",
            {k: round(v, 3) for k, v in score(pg).items()},
            f"| cut>1.25x {row['override_1.25x']:.3f} seen {row['override_1.25x_seen']:.3f} cut>1.1x seen {row['override_1.1x_seen']:.3f} worst seen {row['worst_cut_seen_x']:.2f}x | boost>2x {row['boost_2x']:.3f}")
    json.dump(res, open(args.out, "w"), indent=1)
    log("wrote", args.out, "donor", res["donor"], "slices", res["slice_sizes"])


if __name__ == "__main__":
    main()

"""Two tests on the Qwen3-0.6B dump (CPU only).

A. Does a counted tile ADD to the donor?  P = lam * P_donor + (1 - lam) * P_tile, lam per
   (tile evidence bucket x donor confidence bucket), tuned on val. P_donor(true token) is exact
   (lp_next). Tiles: code-keyed tables at L14 / L28 (k-means 4096 on PCA-128) with text counts
   and the capped donor prior (rho), a pure token table (KN 4-token), and the merged-state table
   from part B. Bar: held NLL below the donor by >= 0.05 nats.

B. State merging: grow the grid's own keys, no donor anywhere.
   s0_t = token. Round r: key_t = (s_{r-1} at t-1, token t); keys seen >= MIN times get a row of
   next-token counts (top 4096 next tokens + other) -> PPMI -> SVD-64 -> cosine k-means into K
   merged states. Rare / unseen keys fall back to the state of their token's pooled row.
   After R rounds the state is a recurrent summary of the history the text itself found
   predictive. Table: P(next | s_R) from text counts, backed off to KN 2-token.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from donor_grid2 import Table, context_ids, nll  # noqa: E402


def kmeans_cos(X, k, rng, iters=20):
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    C = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        a = np.concatenate([np.argmax(X[i:i + 50000] @ C.T, 1) for i in range(0, len(X), 50000)])
        sums = np.zeros_like(C)
        np.add.at(sums, a, X)
        n = np.bincount(a, minlength=k)
        C = np.where(n[:, None] > 0, sums, X[rng.choice(len(X), k)])
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--min-count", type=int, default=8)
    args = ap.parse_args()
    t0 = time.time()
    log = lambda *m: print(f"[{time.time() - t0:6.1f}s]", *m, flush=True)
    rng = np.random.default_rng(0)

    S = {s: dict(np.load(f"{args.prefix}.{s}.npz")) for s in ("fit", "val", "held")}
    V0 = int(max(S[s][k].max() for s in S for k in ("ids", "top_ids"))) + 1
    BOS, W = V0, V0 + 1
    ctx = context_ids(S, BOS, 4)
    M, nxt, isstart = {}, {}, {}
    for s, d in S.items():
        n = len(d["ids"])
        last = np.zeros(n, bool)
        last[np.concatenate([d["doc_start"][1:] - 1, [n - 1]])] = True
        M[s] = ~last & np.isfinite(d["lp_next"])
        nxt[s] = np.concatenate([d["ids"][1:], [0]]).astype(np.int64)
        st = np.zeros(n, bool)
        st[d["doc_start"]] = True
        isstart[s] = st
    w = {s: nxt[s][M[s]] for s in S}
    nf = int(M["fit"].sum())
    Kt = S["fit"]["top_ids"].shape[1]
    soft_w = S["fit"]["top_ids"][M["fit"]].reshape(-1).astype(np.int64)
    soft_p = np.exp(S["fit"]["top_lp"][M["fit"]].astype(np.float32)).reshape(-1).astype(np.float64)
    rep = lambda x: np.repeat(x, Kt)
    p_donor = {s: np.exp(S[s]["lp_next"][M[s]].astype(np.float64)) for s in ("val", "held")}
    d_conf = {s: np.exp(S[s]["top_lp"][M[s]].astype(np.float32).max(1)) for s in ("val", "held")}

    # donor-free KN chains
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
        return p, tot
    base = {s: kn_chain(s, 2)[0] for s in ("val", "held")}
    kn4 = {s: kn_chain(s, 4) for s in ("val", "held")}

    seen3_tok = kn[1].pair(ctx[1]["held"][M["held"]], w["held"]) >= 3
    slices = {"all": np.ones(int(M["held"].sum()), bool),
              "unseen_bigram": kn[1].pair(ctx[1]["held"][M["held"]], w["held"]) == 0,
              "bigram_seen_3x": seen3_tok}
    score = lambda p: {k: nll(p[m]) for k, m in slices.items()}
    donor_score = score(p_donor["held"])
    res = {"prefix": args.prefix, "slice_sizes": {k: int(v.sum()) for k, v in slices.items()},
           "donor": donor_score, "tiles": {}, "state_merge": {}}
    log("donor", donor_score)

    def blend(name, pv, tv, ph, th):
        """lam per (log2 tile-evidence bucket x donor-confidence bucket), tuned on val."""
        eb = lambda t: np.minimum(np.floor(np.log2(t + 1)), 14).astype(int)
        cb = lambda c: np.digitize(c, [0.25, 0.5, 0.8])
        gv, gh = eb(tv) * 4 + cb(d_conf["val"]), eb(th) * 4 + cb(d_conf["held"])
        grid = [x / 40 for x in range(41)]
        lam_g = min(grid, key=lambda l: nll(l * p_donor["val"] + (1 - l) * pv))
        out = np.empty_like(ph)
        lams = {}
        for g in range(60):
            mv, mh = gv == g, gh == g
            if not mh.any():
                continue
            lam = lam_g if mv.sum() < 200 else min(grid, key=lambda l: nll(l * p_donor["val"][mv] + (1 - l) * pv[mv]))
            lams[g] = lam
            out[mh] = lam * p_donor["held"][mh] + (1 - lam) * ph[mh]
        glob = lam_g * p_donor["held"] + (1 - lam_g) * ph
        sc, sg = score(out), score(glob)
        row = {"tile_alone": score(ph), "blend_global": {"lambda_donor": lam_g, **sg},
               "blend_bucketed": sc, "delta_vs_donor": {k: sc[k] - donor_score[k] for k in sc}}
        res["tiles"][name] = row
        log(f"{name:28s} alone {score(ph)['all']:.3f} | +donor global(l={lam_g}) {sg['all']:.4f}"
            f" | bucketed {sc['all']:.4f}  delta all {sc['all'] - donor_score['all']:+.4f}"
            f" unseen_bi {sc['unseen_bigram'] - donor_score['unseen_bigram']:+.4f}"
            f" seen3 {sc['bigram_seen_3x'] - donor_score['bigram_seen_3x']:+.4f}")
        return row

    def code_table(code, rho):
        hard = Table(code["fit"], w["fit"], np.ones(nf), W)
        soft = Table(rep(code["fit"]), soft_w, soft_p, W)

        def prob(s, beta):
            cs, ws = code[s], w[s]
            ht, _ = hard.tot(cs)
            st, _ = soft.tot(cs)
            q = np.where(st > 0, soft.pair(cs, ws) / np.maximum(st, 1e-12), 0)
            kap = rho * ht
            return (hard.pair(cs, ws) + kap * q + beta * base[s]) / (ht + kap + beta), ht
        b = min((1, 3, 10, 30, 100, 300, 1000, 3000), key=lambda bb: nll(prob("val", bb)[0]))
        (pv, tv), (ph, th) = prob("val", b), prob("held", b)
        return pv, tv, ph, th

    blend("KN 4-token (text only)", kn4["val"][0], kn4["val"][1], kn4["held"][0], kn4["held"][1])

    # ---- A: code-keyed tiles
    for L in (14, 28):
        H = {s: np.load(f"{args.prefix}.{s}.h{L}.npy", mmap_mode="r")[:len(S[s]["ids"])] for s in S}
        samp = np.sort(rng.choice(np.where(M["fit"])[0], 150000, replace=False))
        X = np.asarray(H["fit"][samp], np.float32)
        mu = X.mean(0)
        P = np.linalg.svd(X - mu, full_matrices=False)[2][:128].T
        Z = {}
        for s in S:
            idx = np.where(M[s])[0]
            Z[s] = np.concatenate([(np.asarray(H[s][idx[i:i + 100000]], np.float32) - mu) @ P
                                   for i in range(0, len(idx), 100000)])
        sub = Z["fit"][rng.choice(len(Z["fit"]), 120000, replace=False)]
        C = sub[rng.choice(len(sub), 4096, replace=False)].copy()

        def assign(Xs, C):
            cn = (C ** 2).sum(1)
            return np.concatenate([np.argmin(cn - 2 * Xs[i:i + 50000] @ C.T, 1) for i in range(0, len(Xs), 50000)])
        for _ in range(15):
            a = assign(sub, C)
            sums = np.zeros_like(C)
            np.add.at(sums, a, sub)
            n = np.bincount(a, minlength=4096)
            C = np.where(n[:, None] > 0, sums / np.maximum(n, 1)[:, None], sub[rng.choice(len(sub), 4096)])
        code = {s: assign(Z[s], C) for s in S}
        for rho in (0.0, 0.25, 0.5, 1.0):
            blend(f"code L{L} rho={rho}", *code_table(code, rho))
        del H, Z

    # ---- B: state merging (no donor)
    TOPN = 4096
    top_next = np.argsort(-np.bincount(w["fit"], minlength=W))[:TOPN]
    col = np.full(W, TOPN, np.int64)
    col[top_next] = np.arange(TOPN)
    BOS_STATE = args.K
    # positions arrays over ALL tokens (not only valid) so recurrence can run along the text
    tok = {s: S[s]["ids"].astype(np.int64) for s in S}
    state = {s: tok[s].copy() for s in S}          # round 0: the token itself (id space W)
    prev_space = W + 1
    for r in range(1, args.rounds + 1):
        prev = {}
        for s in S:
            p = np.concatenate([[0], state[s][:-1]])
            p[isstart[s]] = prev_space - 1        # BOS
            prev[s] = p
        key = {s: prev[s] * W + tok[s] for s in S}
        fk, fw_ = key["fit"][M["fit"]], w["fit"]
        uk, inv, cnt = np.unique(fk, return_inverse=True, return_counts=True)
        keep = cnt >= args.min_count
        kid = np.full(len(uk), -1)
        kid[keep] = np.arange(keep.sum())
        rows = np.zeros((keep.sum(), TOPN + 1), np.float32)
        sel = kid[inv] >= 0
        np.add.at(rows, (kid[inv][sel], col[fw_[sel]]), 1)
        # token-pooled rows for fallback
        trow = np.zeros((W, TOPN + 1), np.float32) if W * (TOPN + 1) * 4 < 4e9 else None
        tvocab = np.unique(tok["fit"][M["fit"]])
        tix = np.full(W, -1)
        tix[tvocab] = np.arange(len(tvocab))
        trow = np.zeros((len(tvocab), TOPN + 1), np.float32)
        np.add.at(trow, (tix[tok["fit"][M["fit"]]], col[fw_]), 1)
        colm = rows.sum(0) + trow.sum(0) + 1

        def ppmi(R):
            tot = colm.sum()
            rs = R.sum(1, keepdims=True) + 1e-9
            with np.errstate(divide="ignore"):
                x = np.log(R * tot / (rs * colm[None, :]))
            x[~np.isfinite(x)] = 0
            return np.maximum(x, 0)
        Pr = ppmi(rows)
        _, _, Vt = np.linalg.svd(Pr[rng.choice(len(Pr), min(len(Pr), 60000), replace=False)], full_matrices=False)
        B = Vt[:64].T
        E = Pr @ B
        Cc = kmeans_cos(E, min(args.K, len(E)), rng)
        assign_c = lambda X: np.concatenate([np.argmax((X[i:i + 50000] / (np.linalg.norm(X[i:i + 50000], axis=1, keepdims=True) + 1e-9)) @ Cc.T, 1)
                                             for i in range(0, len(X), 50000)])
        key_state = assign_c(E)
        tok_state = np.full(W, 0)
        tok_state[tvocab] = assign_c(ppmi(trow) @ B)
        sorted_keys = uk[keep]
        new = {}
        for s in S:
            i = np.minimum(np.searchsorted(sorted_keys, key[s]), len(sorted_keys) - 1)
            hit = sorted_keys[i] == key[s]
            new[s] = np.where(hit, key_state[i], tok_state[tok[s]])
        state = new
        prev_space = args.K + 1
        # evaluate this round's state as a key
        sk = {s: state[s][M[s]] for s in S}
        hard = Table(sk["fit"], w["fit"], np.ones(nf), W)

        def prob(s, beta):
            ht, _ = hard.tot(sk[s])
            return (hard.pair(sk[s], w[s]) + beta * base[s]) / (ht + beta), ht
        b = min((1, 3, 10, 30, 100, 300, 1000, 3000), key=lambda bb: nll(prob("val", bb)[0]))
        (pv, tv), (ph, th) = prob("val", b), prob("held", b)
        # state + last token (longer effective key)
        sk2 = {s: sk[s] * W + ctx[1][s][M[s]] for s in S}
        hard2 = Table(sk2["fit"], w["fit"], np.ones(nf), W)

        def prob2(s, beta, inner):
            ht, _ = hard2.tot(sk2[s])
            return (hard2.pair(sk2[s], w[s]) + beta * inner) / (ht + beta), ht
        b2 = min((1, 3, 10, 30, 100, 300, 1000, 3000), key=lambda bb: nll(prob2("val", bb, pv)[0]))
        (pv2, tv2), (ph2, th2) = prob2("val", b2, pv), prob2("held", b2, ph)
        cover = float(np.mean(np.isin(key["held"][M["held"]], sorted_keys)))
        res["state_merge"][f"round {r}"] = {"keys_kept": int(keep.sum()), "held_key_coverage": cover,
                                            "beta": b, "state": score(ph), "state+tok": score(ph2),
                                            "stored": hard.size() + hard2.size()}
        log(f"merge round {r}: keys kept {keep.sum()} held coverage {cover:.2f} | state {score(ph)['all']:.3f}"
            f" | state+tok {score(ph2)['all']:.3f} (unseen_bi {score(ph2)['unseen_bigram']:.3f}) stored {hard.size() + hard2.size()}")
        if r == args.rounds:
            blend(f"merged state R{r} +tok", pv2, tv + tv2, ph2, th + th2)
        json.dump(res, open(args.out, "w"), indent=1)
    log("wrote", args.out)


if __name__ == "__main__":
    main()

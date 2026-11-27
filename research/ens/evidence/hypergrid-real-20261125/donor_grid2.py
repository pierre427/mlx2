"""Longer token keys (3/4-token contexts) and bus-query keys (quantized donor hidden state).

Input: donor_hidden.py outputs <prefix>.{fit,val,held}.npz + .h<L>.npy.
Scores: held NLL in nats per donor token, all arms on the same positions.

Token-keyed chains (context = last c tokens, c = 2, 3, 4; backoff down to unigram):
  KN_c      hard counts, interpolated absolute discounting (D=0.75), KN continuation unigram
  SOFT_c    hard + alpha * donor top-k probs, additive interpolation, beta per order tuned on val
Bus-query keys (code of the donor hidden state at the position that predicts the token):
  CODE[L,K]       P(w | code) from soft counts, backed off to the best SOFT chain
  CODE+tok[L,K]   key = (code, last token), backed off to CODE[L,K]
  codes: k-means on PCA-128 (K = 1024, 4096) or product key PQ 256x256 (two k-means of 64 dims)
  GATE            best code arm blended with KN_4 by per-bucket weight of log2 code count
"""

import argparse
import json
import time

import numpy as np


def nll(p):
    return float(-np.mean(np.log(np.maximum(p, 1e-12))))


class Table:
    """Sparse (ctx, w) -> weight, with ctx totals and distinct-w counts."""

    def __init__(self, ctx, w, wt, W):
        self.W = W
        key = ctx * W + w
        self.k, inv = np.unique(key, return_inverse=True)
        self.c = np.bincount(inv, weights=wt)
        kc = self.k // W
        self.ck, inv2 = np.unique(kc, return_inverse=True)
        self.ctot = np.bincount(inv2, weights=self.c)
        self.cn1 = np.bincount(inv2)

    @staticmethod
    def _look(keys, vals, q):
        i = np.minimum(np.searchsorted(keys, q), len(keys) - 1)
        return np.where(keys[i] == q, vals[i], 0)

    def pair(self, ctx, w):
        return self._look(self.k, self.c, ctx * self.W + w)

    def tot(self, ctx):
        return self._look(self.ck, self.ctot, ctx).astype(float), self._look(self.ck, self.cn1, ctx)

    def size(self):
        return len(self.k)


def context_ids(splits, BOS, max_c):
    """ctx[c][split] = integer id of the last c tokens at each position (shared id space)."""
    W = BOS + 1
    toks = {}
    for name, d in splits.items():
        ids = d["ids"].astype(np.int64)
        starts = np.zeros(len(ids), bool)
        starts[d["doc_start"]] = True
        lag = [ids]
        for j in range(1, max_c):
            prev = np.concatenate([[BOS] * j, ids[:-j]])
            # anything that crosses a doc start becomes BOS
            pos = np.arange(len(ids))
            ds = d["doc_start"][np.searchsorted(d["doc_start"], pos, side="right") - 1]
            prev = np.where(pos - j < ds, BOS, prev)
            lag.append(prev)
        toks[name] = lag
    ctx = {1: {n: toks[n][0] for n in splits}}
    for c in range(2, max_c + 1):
        names = list(splits)
        key = np.concatenate([ctx[c - 1][n] * W + toks[n][c - 1] for n in names])
        _, inv = np.unique(key, return_inverse=True)
        out, o = {}, 0
        for n in names:
            m = len(toks[n][0])
            out[n] = inv[o:o + m]
            o += m
        ctx[c] = out
    return ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="14,28")
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()
    t0 = time.time()
    log = lambda *m: print(f"[{time.time() - t0:6.1f}s]", *m, flush=True)
    rng = np.random.default_rng(0)

    S = {s: dict(np.load(f"{args.prefix}.{s}.npz")) for s in ("fit", "val", "held")}
    V0 = int(max(S[s][k].max() for s in S for k in ("ids", "top_ids"))) + 1
    BOS = V0
    W = V0 + 1
    MAXC = 4
    ctx = context_ids(S, BOS, MAXC)
    nxt, valid = {}, {}
    for s, d in S.items():
        n = len(d["ids"])
        last = np.zeros(n, bool)
        last[np.concatenate([d["doc_start"][1:] - 1, [n - 1]])] = True
        nxt[s] = np.concatenate([d["ids"][1:], [0]]).astype(np.int64)
        valid[s] = ~last & np.isfinite(d["lp_next"])
    fm, vm, hm = valid["fit"], valid["val"], valid["held"]
    log(f"W={W} fit={fm.sum()} val={vm.sum()} held={hm.sum()}")
    fw, vw, hw = nxt["fit"][fm], nxt["val"][vm], nxt["held"][hm]

    # soft rows for fit
    K = S["fit"]["top_ids"].shape[1]
    soft_w = S["fit"]["top_ids"][fm].reshape(-1).astype(np.int64)
    soft_p = args.alpha * np.exp(S["fit"]["top_lp"][fm].astype(np.float32)).reshape(-1)
    rep = lambda x: np.repeat(x, K)

    # ---- unigram bases
    hard_uni = np.bincount(fw, minlength=W).astype(float)
    cont = Table(ctx[1]["fit"][fm], fw, np.ones(fm.sum()), W)   # (prev token, w) distinct
    kn_p1 = (np.bincount(cont.k % W, minlength=W) + 0.5)
    kn_p1 /= kn_p1.sum()
    soft_uni = hard_uni + np.bincount(soft_w, weights=soft_p, minlength=W)
    soft_p1 = (soft_uni + 1) / (soft_uni.sum() + W)

    res = {"prefix": args.prefix, "W": W, "alpha": args.alpha, "arms": {}}
    held_lp = S["held"]["lp_next"][hm]
    # slices from hard fit counts
    bi_hard = Table(ctx[1]["fit"][fm], fw, np.ones(fm.sum()), W)
    tri_hard = Table(ctx[2]["fit"][fm], fw, np.ones(fm.sum()), W)
    q4_hard = Table(ctx[3]["fit"][fm], fw, np.ones(fm.sum()), W)
    hb, h2, h3 = ctx[1]["held"][hm], ctx[2]["held"][hm], ctx[3]["held"][hm]
    sl = {"all": np.ones(hm.sum(), bool),
          "unseen_bigram": bi_hard.pair(hb, hw) == 0,
          "unseen_4gram_seen_trigram": (tri_hard.pair(h2, hw) > 0) & (q4_hard.pair(h3, hw) == 0)}
    score = lambda p: {k: nll(p[m]) for k, m in sl.items()}
    res["slice_sizes"] = {k: int(m.sum()) for k, m in sl.items()}
    res["arms"]["donor"] = {k: float(-np.mean(held_lp[m])) for k, m in sl.items()}
    log("donor", res["arms"]["donor"])

    # ---- token-keyed chains. order n = context of c = n-1 tokens; ctx[c] keys c tokens
    kn_tabs = {c: Table(ctx[c]["fit"][fm], fw, np.ones(fm.sum()), W) for c in range(1, MAXC + 1)}
    soft_tabs = {c: Table(np.concatenate([ctx[c]["fit"][fm], rep(ctx[c]["fit"][fm])]),
                          np.concatenate([fw, soft_w]),
                          np.concatenate([np.ones(fm.sum()), soft_p]), W) for c in range(1, MAXC + 1)}
    log("tables built", {c: (kn_tabs[c].size(), soft_tabs[c].size()) for c in kn_tabs})

    def kn_chain(split, m, w, upto, D=0.75):
        p = kn_p1[w]
        for c in range(1, upto + 1):
            t = kn_tabs[c]
            cx = ctx[c][split][m]
            tot, n1 = t.tot(cx)
            with np.errstate(divide="ignore", invalid="ignore"):
                p = np.where(tot > 0, (np.maximum(t.pair(cx, w) - D, 0) + D * n1 * p) / tot, p)
        return p

    betas = {}

    def soft_chain(split, m, w, upto):
        p = soft_p1[w]
        for c in range(1, upto + 1):
            t = soft_tabs[c]
            cx = ctx[c][split][m]
            tot, _ = t.tot(cx)
            p = (t.pair(cx, w) + betas[c] * p) / (tot + betas[c])
        return p

    for c in range(1, MAXC + 1):     # greedy beta per order on val
        best = None
        for b in (0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000):
            betas[c] = b
            s_ = nll(soft_chain("val", vm, vw, c))
            if best is None or s_ < best[0]:
                best = (s_, b)
        betas[c] = best[1]
    log("soft betas", betas)

    pk = {}
    for c in range(2, MAXC + 1):
        pk[c] = kn_chain("held", hm, hw, c)
        stored = sum(kn_tabs[j].size() for j in range(1, c + 1))
        res["arms"][f"KN ctx{c}"] = {"stored": stored, **score(pk[c])}
        ps = soft_chain("held", hm, hw, c)
        stored_s = sum(soft_tabs[j].size() for j in range(1, c + 1))
        res["arms"][f"SOFT ctx{c}"] = {"stored": stored_s, **score(ps)}
        log(f"ctx{c} KN", {k: round(v, 3) for k, v in score(pk[c]).items()},
            "| SOFT", {k: round(v, 3) for k, v in score(ps).items()}, f"stored {stored}/{stored_s}")
    best_c = min(range(2, MAXC + 1), key=lambda c: nll(soft_chain("val", vm, vw, c)))
    base_val = soft_chain("val", vm, vw, best_c)
    base_held = soft_chain("held", hm, hw, best_c)
    kn_val = kn_chain("val", vm, vw, MAXC)

    # ---- bus-query codes
    for L in [int(x) for x in args.layers.split(",")]:
        H = {s: np.load(f"{args.prefix}.{s}.h{L}.npy", mmap_mode="r")[:len(S[s]["ids"])] for s in S}
        samp = rng.choice(np.where(fm)[0], 150000, replace=False)
        X = np.asarray(H["fit"][np.sort(samp)], np.float32)
        mu = X.mean(0)
        _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
        P = Vt[:128].T

        def proj(s, m):
            idx = np.where(m)[0]
            out = np.empty((len(idx), 128), np.float32)
            for i in range(0, len(idx), 100000):
                out[i:i + 100000] = (np.asarray(H[s][idx[i:i + 100000]], np.float32) - mu) @ P
            return out
        Z = {"fit": proj("fit", fm), "val": proj("val", vm), "held": proj("held", hm)}
        log(f"L{L} PCA done")

        def kmeans(Xs, k, iters=15):
            C = Xs[rng.choice(len(Xs), k, replace=False)].copy()
            for _ in range(iters):
                a = assign(Xs, C)
                sums = np.zeros_like(C)
                np.add.at(sums, a, Xs)
                n = np.bincount(a, minlength=k)
                C = np.where(n[:, None] > 0, sums / np.maximum(n, 1)[:, None],
                             Xs[rng.choice(len(Xs), k)])
            return C

        def assign(Xs, C):
            cn = (C ** 2).sum(1)
            out = np.empty(len(Xs), np.int64)
            for i in range(0, len(Xs), 50000):
                out[i:i + 50000] = np.argmin(cn - 2 * Xs[i:i + 50000] @ C.T, 1)
            return out

        sub = Z["fit"][rng.choice(len(Z["fit"]), 120000, replace=False)]
        schemes = {}
        for k in (1024, 4096):
            C = kmeans(sub, k)
            schemes[f"km{k}"] = {s: assign(Z[s], C) for s in Z}
        C1 = kmeans(sub[:, :64], 256)
        C2 = kmeans(sub[:, 64:], 256)
        schemes["pq256x256"] = {s: assign(Z[s][:, :64], C1) * 256 + assign(Z[s][:, 64:], C2) for s in Z}
        log(f"L{L} codes done")

        for name, code in schemes.items():
            cf = code["fit"]
            tab = Table(np.concatenate([cf, rep(cf)]), np.concatenate([fw, soft_w]),
                        np.concatenate([np.ones(len(cf)), soft_p]), W)
            def pcode(split, w, base, b):
                tot, _ = tab.tot(code[split])
                return (tab.pair(code[split], w) + b * base) / (tot + b), tot
            bb = min((0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000), key=lambda b: nll(pcode("val", vw, base_val, b)[0]))
            pv, totv = pcode("val", vw, base_val, bb)
            ph, toth = pcode("held", hw, base_held, bb)
            res["arms"][f"CODE L{L} {name}"] = {"stored": tab.size(), "beta": bb, **score(ph)}
            # code + last token
            ck = {s: code[s] * W + ctx[1][s][{"fit": fm, "val": vm, "held": hm}[s]] for s in code}
            tab2 = Table(np.concatenate([ck["fit"], rep(ck["fit"])]), np.concatenate([fw, soft_w]),
                         np.concatenate([np.ones(len(cf)), soft_p]), W)
            def pck(split, w, base, b):
                tot, _ = tab2.tot(ck[split])
                return (tab2.pair(ck[split], w) + b * base) / (tot + b)
            b2 = min((0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000), key=lambda b: nll(pck("val", vw, pv, b)))
            ph2 = pck("held", hw, ph, b2)
            pv2 = pck("val", vw, pv, b2)
            res["arms"][f"CODE+tok L{L} {name}"] = {"stored": tab.size() + tab2.size(), "beta": b2, **score(ph2)}
            # gate against KN ctx4 by log2 code count
            bv = np.minimum(np.floor(np.log2(totv + 1)), 12).astype(int)
            bh = np.minimum(np.floor(np.log2(toth + 1)), 12).astype(int)
            pg = np.empty(len(ph2))
            lams = {}
            for kb in range(13):
                mv, mh = bv == kb, bh == kb
                lam = 0.3 if not mv.any() else min((x / 20 for x in range(21)),
                                                   key=lambda l: nll(l * kn_val[mv] + (1 - l) * pv2[mv]))
                lams[kb] = lam
                pg[mh] = lam * pk[MAXC][mh] + (1 - lam) * ph2[mh]
            res["arms"][f"GATE L{L} {name}"] = {"lambda_kn_by_log2_code_count": lams, **score(pg)}
            log(f"L{L} {name}: CODE", {k: round(v, 3) for k, v in score(ph).items()},
                "| +tok", {k: round(v, 3) for k, v in score(ph2).items()},
                "| GATE", {k: round(v, 3) for k, v in score(pg).items()},
                f"stored {tab.size()}+{tab2.size()}")
        json.dump(res, open(args.out, "w"), indent=1)
    log("wrote", args.out)


if __name__ == "__main__":
    main()

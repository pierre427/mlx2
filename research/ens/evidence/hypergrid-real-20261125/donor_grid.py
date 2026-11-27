"""Donor-steered count grid on the donor's own tokens (CPU, numpy).

Inputs: donor_dump.py outputs <prefix>.{fit,val,held}.npz.
Every arm is scored on the same held positions: tokens 2..n-1 of each doc that have a
recorded donor log-prob, so the donor itself can be scored as a reference.

Arms:
  donor        the donor's own NLL (the ceiling the grid is chasing)
  A_kn         raw trigram counts, KN-style (the hypergrid; same as the word-level run)
  A_add        raw counts, additive interpolated estimator (fractional-count capable)
  S_soft(a)    counts + a * donor top-k probabilities at every fit position ("soft counts")
  S_promote    soft counts only for alternatives with donor p >= 0.1 that the text did not
               contain there (promoter: grow missing edges, leave the rest hard)
  G_gate       A_kn and S_soft blended with a weight per context-count bucket (tuned on val)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from hypergrid_real import Counts, KNTrigram, nll  # noqa: E402


def load(prefix, split):
    return dict(np.load(f"{prefix}.{split}.npz"))


def contexts(d, BOS):
    """Per position t: (a, b, w) = (ids[t-1], ids[t], ids[t+1]) with BOS padding at doc start."""
    ids = d["ids"].astype(np.int64)
    n = len(ids)
    starts = np.zeros(n, bool)
    starts[d["doc_start"]] = True
    prev = np.concatenate([[BOS], ids[:-1]])
    prev[starts] = BOS
    nxt = np.concatenate([ids[1:], [0]])
    last = np.zeros(n, bool)
    last[np.concatenate([d["doc_start"][1:] - 1, [n - 1]])] = True
    valid = ~last & np.isfinite(d["lp_next"])
    return prev, ids, nxt, valid


class AddInterp:
    """P3 = (c3 + b3*P2)/(c3. + b3), P2 = (c2 + b2*P1)/(c2. + b2), P1 = (c1 + 1)/(N + V).
    Counts may be fractional (soft)."""

    def __init__(self, a, b, w, wt, V):
        self.V = V
        self.k3 = Counts(np.zeros(1, np.int64))
        self.tab = {}
        for name, key, ctx in (("3", (a * V + b) * V + w, a * V + b), ("2", b * V + w, b)):
            uk, inv = np.unique(key, return_inverse=True)
            self.tab[name] = (uk, np.bincount(inv, weights=wt))
            uc, invc = np.unique(ctx, return_inverse=True)
            self.tab[name + "c"] = (uc, np.bincount(invc, weights=wt))
        self.c1 = np.bincount(w, weights=wt, minlength=V)
        self.p1 = (self.c1 + 1) / (self.c1.sum() + V)
        self.b2, self.b3 = 1.0, 1.0
        self.n_stored = len(self.tab["3"][0]) + len(self.tab["2"][0]) + V

    def _get(self, name, q):
        k, c = self.tab[name]
        i = np.minimum(np.searchsorted(k, q), len(k) - 1)
        return np.where(k[i] == q, c[i], 0.0)

    def prob(self, a, b, w):
        V = self.V
        p2 = (self._get("2", b * V + w) + self.b2 * self.p1[w]) / (self._get("2c", b) + self.b2)
        return (self._get("3", (a * V + b) * V + w) + self.b3 * p2) / (self._get("3c", a * V + b) + self.b3)

    def tune(self, a, b, w):
        best = None
        for b2 in (0.1, 0.3, 1, 3, 10):
            for b3 in (0.1, 0.3, 1, 3, 10):
                self.b2, self.b3 = b2, b3
                s = nll(self.prob(a, b, w))
                if best is None or s < best[0]:
                    best = (s, b2, b3)
        _, self.b2, self.b3 = best
        return best[0]


def soft_rows(prev, cur, nxt, valid, top_ids, top_lp, alpha, mode):
    """Soft count rows (a, b, w', weight) from the donor's top-k at each fit position."""
    m = valid
    a = np.repeat(prev[m], top_ids.shape[1])
    b = np.repeat(cur[m], top_ids.shape[1])
    w = top_ids[m].reshape(-1).astype(np.int64)
    p = np.exp(top_lp[m].astype(np.float32)).reshape(-1)
    if mode == "promote":
        actual = np.repeat(nxt[m], top_ids.shape[1])
        keep = (p >= 0.1) & (w != actual)
        a, b, w, p = a[keep], b[keep], w[keep], p[keep]
    return a, b, w, alpha * p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    log = lambda *m: print(f"[{time.time() - t0:6.1f}s]", *m, flush=True)

    fit, val, held = (load(args.prefix, s) for s in ("fit", "val", "held"))
    V0 = int(max(d[k].max() for d in (fit, val, held) for k in ("ids", "top_ids"))) + 1
    BOS = V0
    V = V0 + 1
    fa, fb, fw, fv = contexts(fit, BOS)
    va, vb, vw, vv = contexts(val, BOS)
    ha, hb, hw, hv = contexts(held, BOS)
    # hard training events: every position with a next token in the doc
    hard_m = np.ones(len(fa), bool)
    hard_m[np.concatenate([fit["doc_start"][1:] - 1, [len(fa) - 1]])] = False
    ev = np.stack([fa[hard_m], fb[hard_m], fw[hard_m]], 1)
    log(f"V={V} fit_events={len(ev)} val={vv.sum()} held={hv.sum()}")

    va, vb, vw = va[vv], vb[vv], vw[vv]
    held_lp = held["lp_next"][hv]
    ha, hb, hw = ha[hv], hb[hv], hw[hv]
    bi = Counts(ev[:, 1] * V + ev[:, 2])
    tri = Counts((ev[:, 0] * V + ev[:, 1]) * V + ev[:, 2])
    ctx3 = Counts(ev[:, 0] * V + ev[:, 1])
    unseen_bi = bi.get(hb * V + hw) == 0
    unseen_tri = ~unseen_bi & (tri.get((ha * V + hb) * V + hw) == 0)
    slices = {"all": np.ones(len(ha), bool), "unseen_bigram": unseen_bi,
              "unseen_trigram_seen_bigram": unseen_tri}
    score = lambda p: {k: nll(p[m]) for k, m in slices.items()}
    res = {"prefix": args.prefix, "vocab": V, "fit_events": int(len(ev)),
           "slice_sizes": {k: int(m.sum()) for k, m in slices.items()}, "arms": {}}
    res["arms"]["donor"] = {k: float(-np.mean(held_lp[m])) for k, m in slices.items()}
    log("donor", res["arms"]["donor"])

    kn = KNTrigram(ev, V)
    pk = kn.prob(np.stack([ha, hb, hw], 1))
    pk_val = kn.prob(np.stack([va, vb, vw], 1))
    res["arms"]["A_kn"] = {"stored": kn.n_stored, **score(pk)}
    log("A_kn", res["arms"]["A_kn"])

    ones = np.ones(len(ev))
    add = AddInterp(ev[:, 0], ev[:, 1], ev[:, 2], ones, V)
    add.tune(va, vb, vw)
    res["arms"]["A_add"] = {"stored": add.n_stored, "b2": add.b2, "b3": add.b3, **score(add.prob(ha, hb, hw))}
    log("A_add", res["arms"]["A_add"])

    best_soft = None
    for mode, alphas in (("soft", (0.25, 1.0, 4.0)), ("promote", (0.25, 1.0, 4.0))):
        for alpha in alphas:
            sa, sb, sw, sp = soft_rows(fa, fb, fw, fv, fit["top_ids"], fit["top_lp"], alpha, mode)
            m = AddInterp(np.concatenate([ev[:, 0], sa]), np.concatenate([ev[:, 1], sb]),
                          np.concatenate([ev[:, 2], sw]), np.concatenate([ones, sp]), V)
            dv = m.tune(va, vb, vw)
            ph = m.prob(ha, hb, hw)
            name = f"S_{mode} a={alpha}"
            res["arms"][name] = {"stored": m.n_stored, "val": dv, "b2": m.b2, "b3": m.b3, **score(ph)}
            log(name, {k: round(v, 3) for k, v in score(ph).items()}, f"stored={m.n_stored}")
            if mode == "soft" and (best_soft is None or dv < best_soft[0]):
                best_soft = (dv, name, ph, m.prob(va, vb, vw))

    # gate: per context-count bucket weight between raw KN memory and the best soft grid
    _, sname, ps, ps_val = best_soft
    bucket = lambda a, b: np.minimum(np.floor(np.log2(ctx3.get(a * V + b) + 1)), 8).astype(int)
    bv, bh = bucket(va, vb), bucket(ha, hb)
    lams = {}
    pg = np.empty(len(ha))
    for k in range(9):
        mv = bv == k
        lam = 0.5 if not mv.any() else min((x / 20 for x in range(0, 21)),
                                           key=lambda l: nll(l * pk_val[mv] + (1 - l) * ps_val[mv]))
        lams[k] = lam
        mh = bh == k
        pg[mh] = lam * pk[mh] + (1 - lam) * ps[mh]
    res["arms"]["G_gate"] = {"soft_arm": sname, "lambda_memory_by_log2_ctx_count": lams, **score(pg)}
    log("G_gate", res["arms"]["G_gate"])
    json.dump(res, open(args.out, "w"), indent=1)
    log("wrote", args.out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extend the AB cross-token ridge lambda grid (1000 was the edge of LAMS). FIT/VAL = train prompts only.
Reports VAL recall@8 (mean over layers 1-39) per lambda; writes heads_lambda.json. CPU only."""
import json

import numpy as np

from common import CAPTURE, LAYERS, OUT, atomic_json, load_gate, load_trace, provenance
from heads import HIDDEN, X, recall_k, ridge_eig, router, topn

LAMS = (300.0, 1000.0, 3000.0, 10000.0, 30000.0)


def main():
    z = load_trace(); ids = z["ids"].astype(np.int64); T = len(ids)
    req, dec, split = z["tok_req"], z["tok_decode"].astype(bool), z["tok_split"]
    replies = [json.loads(l) for l in open(f"{CAPTURE}/replies.jsonl")]
    train_reqs = [i for i, r in enumerate(replies) if r["split"] == "train"]
    val_reqs = set(train_reqs[::4])
    t = np.arange(T - 1)
    pp = t[dec[t + 1] & (req[t] == req[t + 1]) & (req[t] >= 0) & (split[t + 1] == 0)]
    isval = np.array([req[x] in val_reqs for x in pp])
    fitp, valp = pp[~isval], pp[isval]
    x39 = np.asarray(X(39), np.float32); x0 = np.asarray(X(0), np.float32)
    Ff = np.concatenate([x39[fitp], x0[fitp + 1]], 1); Fv = np.concatenate([x39[valp], x0[valp + 1]], 1)
    mu, s, V = ridge_eig(Ff); Xc = Ff - mu
    score = {lam: [] for lam in LAMS}
    for L in range(1, LAYERS):
        g, b = load_gate(L)
        xl = np.asarray(X(L), np.float32)
        S_f = router(g, xl[fitp + 1]); base_f = router(g, Ff[:, HIDDEN:]); base_v = router(g, Fv[:, HIDDEN:])
        R = S_f - base_f; mr = R.mean(0)
        Q = V.T @ (Xc.T @ (R - mr)).astype(np.float64)
        for lam in LAMS:
            W = (V @ (Q / (s + lam)[:, None])).astype(np.float32)
            sc = base_v + (Fv - mu) @ W + mr + b
            score[lam].append(recall_k(topn(sc, 8), ids[valp + 1, L], 8))
    res = provenance(lams=LAMS, split="FIT/VAL inside train prompts", metric="VAL recall@8 mean L1-39")
    res["val_recall8"] = {str(k): float(np.mean(v)) for k, v in score.items()}
    atomic_json(f"{OUT}/heads_lambda.json", res)
    print(res["val_recall8"])


if __name__ == "__main__":
    main()

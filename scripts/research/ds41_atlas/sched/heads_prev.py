#!/usr/bin/env python3
"""Cross-token head + previous-token routing prior: score_L(t+1) = AB_ridge_L + beta * onehot(ids_L(t)).

beta chosen on the validation subset of TRAIN prompts (recall@8), scored on HELDOUT.  Adds pred/cal
'xtok_ABp' to preds.npz and 'xtok_ABp_h1' to heads_eval.json.  CPU only.
"""
import json

import numpy as np

from common import CAPTURE, HIDDEN, LAYERS, OUT, atomic_json, atomic_npz, load_gate, load_trace
from heads import TOPN, X, rank_hit_prob, recall_k, router, topn

BETAS = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8)


def main():
    z = load_trace(); ids = z["ids"].astype(np.int64); T = len(ids)
    req, dec, split = z["tok_req"], z["tok_decode"].astype(bool), z["tok_split"]
    hz = np.load(f"{OUT}/xtok_heads.npz")
    W = hz["W_AB_h1"]; mu = hz["mu_AB_h1"]; mr = hz["mr_AB_h1"]
    t = np.arange(T - 1)
    pp = t[dec[t + 1] & (req[t] == req[t + 1]) & (req[t] >= 0)]
    replies = [json.loads(l) for l in open(f"{CAPTURE}/replies.jsonl")]
    train_reqs = [i for i, r in enumerate(replies) if r["split"] == "train"]
    val = np.array([req[x] in set(train_reqs[::4]) for x in pp])
    x39 = np.asarray(X(39), np.float32); x0 = np.asarray(X(0), np.float32)
    F = np.concatenate([x39[pp], x0[pp + 1]], 1) - mu
    sp = split[pp + 1]
    P = np.full((T, LAYERS, TOPN), -1, np.int16)
    cal = np.zeros((LAYERS, TOPN))
    score_b = {b: [] for b in BETAS}
    rows = []
    gates = [load_gate(L) for L in range(LAYERS)]
    scs = []
    for L in range(LAYERS):
        g, bias = gates[L]
        sc = router(g, x0[pp + 1]) + F @ W[:, L * 384:(L + 1) * 384].astype(np.float32) + mr[L * 384:(L + 1) * 384] + bias
        scs.append(sc)
        prev = np.zeros_like(sc); np.put_along_axis(prev, ids[pp, L], 1.0, 1)
        scale = sc.std(1, keepdims=True)
        for b in BETAS:
            pv = topn((sc + b * scale * prev)[val & (sp == 0)], 8)
            score_b[b].append(recall_k(pv, ids[pp[val & (sp == 0)] + 1, L], 8))
    beta = max(BETAS, key=lambda b: np.mean(score_b[b]))
    for L in range(LAYERS):
        sc = scs[L]
        prev = np.zeros_like(sc); np.put_along_axis(prev, ids[pp, L], 1.0, 1)
        full = sc + beta * sc.std(1, keepdims=True) * prev
        P[pp + 1, L] = topn(full, TOPN)
        ho = sp == 1
        ph = P[pp[ho] + 1, L].astype(np.int64); th = ids[pp[ho] + 1, L]
        rows.append({"layer": L, "r6": recall_k(ph, th, 6), "r8": recall_k(ph, th, 8), "r12": recall_k(ph, th, 12),
                     "r24": recall_k(ph, th, 24)})
        cal[L] = rank_hit_prob(P[pp[sp == 0] + 1, L].astype(np.int64), ids[pp[sp == 0] + 1, L])
    ev = json.load(open(f"{OUT}/heads_eval.json"))
    o = {"variant": "AB+prev", "beta_by_val_recall8": {str(b): float(np.mean(v)) for b, v in score_b.items()},
         "beta": beta, "rows": rows}
    for k in (6, 8, 12, 24):
        o[f"mean_r{k}"] = float(np.mean([r[f"r{k}"] for r in rows]))
    ev["evals"]["xtok_ABp_h1"] = o
    atomic_json(f"{OUT}/heads_eval.json", ev)
    arr = dict(np.load(f"{OUT}/preds.npz"))
    arr["pred_xtok_ABp"] = P; arr["cal_xtok_ABp"] = cal.astype(np.float32)
    atomic_npz(f"{OUT}/preds.npz", **arr)
    print("beta", beta, {k: round(o[f"mean_r{k}"], 4) for k in (6, 8, 12, 24)})


if __name__ == "__main__":
    main()

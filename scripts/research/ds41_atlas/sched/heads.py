#!/usr/bin/env python3
"""Expert-routing prediction heads for the DS4.1 Source prefetch scheduler study (CPU only).

a. existing d1/d2 within-token lookahead heads (heads-d12-lam100.npz): x_{L-d}(t) -> routing_L(t)
b. cross-token "EAGLE-style" rung-1 ridge heads: features of token t / t+1 -> router scores of token
   t+1 at every layer.  Variants: AB = concat(x_39(t), x_0(t+1)); A = x_39(t) only (available one
   token early); B = x_0(t+1) only.  Residual on base = router_L(x_0(t+1)) (B, AB) or router_L(x_39(t)) (A).
   Horizon heads h=2,3: concat(x_39(t), x_0(t+h)) -> routing(t+h); x_0(t+h) stands in for a DSpark draft
   token, so these are UPPER BOUNDS (real drafts can be wrong).
Lambda chosen on a validation subset of TRAIN prompts (every 4th train prompt), refit on all train,
scored on HELDOUT prompts only.  Outputs: heads_eval.json, preds.npz (top-24 predictions per token per
layer for every head, plus per-rank hit probabilities calibrated on train rows), xtok_heads.npz.
"""
import json
import time

import numpy as np

from common import (CAPTURE, HIDDEN, LAYERS, OUT, TOPK, atomic_json, atomic_npz, load_gate, load_trace,
                    provenance, softplus)

TOPN = 24
LAMS = (10.0, 100.0, 1000.0)
XDIR = f"{CAPTURE}/x_by_layer"


def X(L):
    return np.load(f"{XDIR}/L{L:02d}.npy", mmap_mode="r")


def router(gate, x):
    return np.sqrt(softplus(x @ gate.T))


def topn(s, n):
    idx = np.argpartition(-s, n - 1, axis=1)[:, :n]
    order = np.argsort(-np.take_along_axis(s, idx, 1), axis=1)
    return np.take_along_axis(idx, order, 1)


def recall_k(pred_rank, true_ids, k):
    """pred_rank [n, >=k] ranked ids; true_ids [n, 6]. fraction of true routed experts in top-k."""
    p = pred_rank[:, :k]
    hit = (p[:, :, None] == true_ids[:, None, :]).any(1)
    return float(hit.mean())


def rank_hit_prob(pred_rank, true_ids):
    """P(rank-r prediction is actually routed), per rank."""
    return (pred_rank[:, :, None] == true_ids[:, None, :]).any(2).mean(0)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ridge_eig(Xf):
    mu = Xf.mean(0)
    G = np.zeros((Xf.shape[1], Xf.shape[1]), np.float64)
    for i in range(0, len(Xf), 2048):
        c = (Xf[i:i + 2048] - mu).astype(np.float64)
        G += c.T @ c
    s, V = np.linalg.eigh(G)
    del G
    return mu, s, V


def main():
    t_start = time.time()
    z = load_trace()
    ids = z["ids"].astype(np.int64)
    T = ids.shape[0]
    req, dec, split = z["tok_req"], z["tok_decode"].astype(bool), z["tok_split"]
    gates = [load_gate(L) for L in range(LAYERS)]
    bias = np.stack([g[1] for g in gates])                               # [40,384]
    log("gates loaded")
    # true router scores of every token at every layer (float16 store), sanity vs captured ids
    S = np.zeros((T, LAYERS, 384), np.float16)
    agree = []
    for L in range(LAYERS):
        xl = np.asarray(X(L), np.float32)
        s = router(gates[L][0], xl)
        S[:, L] = s
        sel = topn(s + bias[L], TOPK)
        agree.append(float((np.sort(sel, 1) == np.sort(ids[:, L], 1)).all(1).mean()))
    log("true scores; captured-id agreement min", min(agree))
    res = provenance(topn=TOPN, lams=LAMS, base="router_L(x_0(t+1)) for B/AB, router_L(x_39(t)) for A")
    res["captured_id_agreement_by_layer"] = agree

    # ---- pairs (t, t+h): t+h decode token of the same request ----
    def pairs(h):
        t = np.arange(T - h)
        ok = dec[t + h] & (req[t] == req[t + h]) & (req[t] >= 0)
        # all intermediate rows same request (rows are request-contiguous)
        return t[ok]
    replies = [json.loads(l) for l in open(f"{CAPTURE}/replies.jsonl")]
    train_reqs = [i for i, r in enumerate(replies) if r["split"] == "train"]
    val_reqs = set(train_reqs[::4])

    preds, calib, evals = {}, {}, {}
    x39 = np.asarray(X(39), np.float32)
    x0 = np.asarray(X(0), np.float32)

    # ---- baselines on the h=1 heldout pairs ----
    p1 = pairs(1)
    sp = split[p1 + 1]
    ho = p1[sp == 1]
    base_rows = {}
    for name in ("prev_token", "prev2_union", "router_x0_untrained", "router_x39_untrained"):
        base_rows[name] = []
    for L in range(LAYERS):
        true = ids[ho + 1, L]
        prev = ids[ho, L]
        u2 = np.concatenate([ids[ho, L], ids[np.maximum(ho - 1, 0), L]], 1)
        base_rows["prev_token"].append({"layer": L, "r6": recall_k(prev, true, 6)})
        # union of last two tokens: up to 12 distinct; recall at its size
        hit = (u2[:, :, None] == true[:, None, :]).any(1).mean()
        base_rows["prev2_union"].append({"layer": L, "r12": float(hit)})
        for nm, xx in (("router_x0_untrained", x0[ho + 1]), ("router_x39_untrained", x39[ho])):
            pr = topn(router(gates[L][0], xx) + bias[L], 12)
            base_rows[nm].append({"layer": L, "r6": recall_k(pr, true, 6), "r8": recall_k(pr, true, 8),
                                  "r12": recall_k(pr, true, 12)})
    evals["baselines"] = base_rows
    log("baselines done")

    # ---- existing d1/d2 heads (heldout decode tokens) + preds for the simulator ----
    hz = np.load(f"{CAPTURE}/heads-d12-lam100.npz")
    for d in (1, 2):
        P = np.full((T, LAYERS, TOPN), -1, np.int16)
        rows = []
        cal = np.zeros((LAYERS, TOPN))
        for L in range(d, LAYERS):
            hid = np.asarray(X(L - d), np.float32)
            w = hz[f"w_{L}_{d}"].astype(np.float32); b = hz[f"b_{L}_{d}"].astype(np.float32)
            sc = router(gates[L][0], hid) + hid @ w.T + b + bias[L]
            P[:, L] = topn(sc, TOPN)
            hod = np.where(dec & (split == 1))[0]; trd = np.where(dec & (split == 0))[0]
            rows.append({"layer": L, "r6": recall_k(P[hod, L], ids[hod, L], 6), "r8": recall_k(P[hod, L], ids[hod, L], 8),
                         "r12": recall_k(P[hod, L], ids[hod, L], 12)})
            cal[L] = rank_hit_prob(P[trd, L].astype(np.int64), ids[trd, L])
        preds[f"d{d}"] = P; calib[f"d{d}"] = cal; evals[f"d{d}"] = rows
        log(f"d{d} preds done mean r6", np.mean([r["r6"] for r in rows]))

    # ---- cross-token ridge heads ----
    heads_out = {}

    def feats(variant, t, h):
        if variant == "A":
            return x39[t]
        if variant == "B":
            return x0[t + h]
        return np.concatenate([x39[t], x0[t + h]], 1)

    def base_of(variant, F, L):
        if variant == "A":
            return router(gates[L][0], F)
        xb = F if variant == "B" else F[:, HIDDEN:]
        return router(gates[L][0], xb)

    CH = 4  # layers per output chunk

    def fit_eval(variant, h, lams, save=False):
        pp = pairs(h)
        sp_ = split[pp + h]
        trp = pp[sp_ == 0]; hop = pp[sp_ == 1]
        fit_mask = np.array([req[t] not in val_reqs for t in trp])
        out = {"variant": variant, "h": h, "n_train": int(len(trp)), "n_heldout": int(len(hop))}
        # lambda selection on FIT -> VAL (train prompts only)
        if len(lams) > 1:
            fitp, valp = trp[fit_mask], trp[~fit_mask]
            Ff = feats(variant, fitp, h); Fv = feats(variant, valp, h)
            mu, s, V = ridge_eig(Ff)
            Xc = Ff - mu
            score = {lam: [] for lam in lams}
            for L0 in range(0, LAYERS, CH):
                Ls = range(L0, min(LAYERS, L0 + CH))
                R = np.concatenate([S[fitp + h, L].astype(np.float32) - base_of(variant, Ff, L) for L in Ls], 1)
                mr = R.mean(0)
                Q = V.T @ (Xc.T @ (R - mr)).astype(np.float64)
                for lam in lams:
                    W = (V @ (Q / (s + lam)[:, None])).astype(np.float32)
                    pv = (Fv - mu) @ W + mr
                    for j, L in enumerate(Ls):
                        sc = base_of(variant, Fv, L) + pv[:, j * 384:(j + 1) * 384] + bias[L]
                        score[lam].append(recall_k(topn(sc, 8), ids[valp + h, L], 8))
            del V
            out["val_recall8_by_lambda"] = {str(k): float(np.mean(v)) for k, v in score.items()}
            lam = max(lams, key=lambda k: np.mean(score[k]))
        else:
            lam = lams[0]
        out["lambda"] = lam
        log(variant, h, "lambda", lam)
        Ft = feats(variant, trp, h)
        mu, s, V = ridge_eig(Ft)
        Xc = Ft - mu
        rows, cal = [], np.zeros((LAYERS, TOPN))
        P = np.full((T, LAYERS, TOPN), -1, np.int16)
        Wall = np.zeros((Ft.shape[1], LAYERS * 384), np.float16) if save else None
        mrall = np.zeros(LAYERS * 384, np.float32)
        allp = pp  # predictions for every valid pair row (train + heldout)
        Fa = feats(variant, allp, h)
        for L0 in range(0, LAYERS, CH):
            Ls = range(L0, min(LAYERS, L0 + CH))
            R = np.concatenate([S[trp + h, L].astype(np.float32) - base_of(variant, Ft, L) for L in Ls], 1)
            mr = R.mean(0)
            Q = V.T @ (Xc.T @ (R - mr)).astype(np.float64)
            W = (V @ (Q / (s + lam)[:, None])).astype(np.float32)
            if save:
                Wall[:, L0 * 384:(L0 + len(Ls)) * 384] = W
            mrall[L0 * 384:(L0 + len(Ls)) * 384] = mr
            pa = (Fa - mu) @ W + mr
            for j, L in enumerate(Ls):
                sc = base_of(variant, Fa, L) + pa[:, j * 384:(j + 1) * 384] + bias[L]
                P[allp + h, L] = topn(sc, TOPN)
                hrow = allp + h
                is_ho = split[hrow] == 1
                ph = P[hrow[is_ho], L].astype(np.int64); th = ids[hrow[is_ho], L]
                rows.append({"layer": L, "r6": recall_k(ph, th, 6), "r8": recall_k(ph, th, 8),
                             "r12": recall_k(ph, th, 12), "r24": recall_k(ph, th, 24)})
                cal[L] = rank_hit_prob(P[hrow[~is_ho], L].astype(np.int64), ids[hrow[~is_ho], L])
        out["rows"] = rows
        for k in (6, 8, 12, 24):
            out[f"mean_r{k}"] = float(np.mean([r[f"r{k}"] for r in rows]))
        log(variant, h, {k: round(out[f"mean_r{k}"], 4) for k in (6, 8, 12)})
        if save:
            heads_out[f"{variant}_h{h}"] = (Wall, mu.astype(np.float32), mrall, lam)
        return out, P, cal

    for variant in ("B", "A", "AB"):
        o, P, cal = fit_eval(variant, 1, LAMS, save=(variant in ("AB", "A")))
        evals[f"xtok_{variant}_h1"] = o; preds[f"xtok_{variant}"] = P; calib[f"xtok_{variant}"] = cal
        res["evals"] = evals
        atomic_json(f"{OUT}/heads_eval.json", res)
    lam_ab = evals["xtok_AB_h1"]["lambda"]
    for h in (2, 3):
        o, P, cal = fit_eval("AB", h, (lam_ab,), save=False)
        evals[f"xtok_AB_h{h}"] = o; preds[f"xtok_AB_h{h}"] = P; calib[f"xtok_AB_h{h}"] = cal
        res["evals"] = evals
        atomic_json(f"{OUT}/heads_eval.json", res)

    arrays = {f"pred_{k}": v for k, v in preds.items()}
    arrays.update({f"cal_{k}": v.astype(np.float32) for k, v in calib.items()})
    atomic_npz(f"{OUT}/preds.npz", **arrays)
    harr = {}
    for k, (W, mu, mr, lam) in heads_out.items():
        harr[f"W_{k}"] = W; harr[f"mu_{k}"] = mu; harr[f"mr_{k}"] = mr; harr[f"lam_{k}"] = np.array(lam)
    harr["manifest"] = np.frombuffer(json.dumps({"binding": res["checkpoint"], "schema": "ds41-xtok-ridge-v1",
                                                 "variants": sorted(heads_out), "base": res["params"]["base"],
                                                 "train_split_only": True}).encode(), np.uint8)
    atomic_npz(f"{OUT}/xtok_heads.npz", **harr)
    res["evals"] = evals
    res["elapsed_s"] = round(time.time() - t_start, 1)
    atomic_json(f"{OUT}/heads_eval.json", res)
    log("done", res["elapsed_s"])


if __name__ == "__main__":
    main()

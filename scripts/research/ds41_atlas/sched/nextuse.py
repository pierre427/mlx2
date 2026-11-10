#!/usr/bin/env python3
"""Learned next-use eviction head (Parrot/LRB-style learned Belady, logistic rung).

Replays an LRU (75 GB) over the serving-order access stream.  At every 8th decode step a random layer
position is chosen and 64 resident keys are sampled; features (same definitions as eventsim.Sim):
  log1p(age in steps), EMA_fast (alpha 1/8 per step), EMA_slow (alpha 1/128), log1p(train-split frequency),
  layer cycle distance d/40, pn = xtok-AB hit probability for the current token (0 if not predicted)
label = key is accessed again within H steps (H = 16/64/256).  Fit on TRAIN-request samples, AUC on
HELDOUT-request samples.  Writes nextuse.json with raw-scale weights for each H (with and without pn);
'model' = chosen default (picked later by the train-split event sim in sweep.py).
"""
import math
from collections import OrderedDict

import numpy as np

from common import EXPERTS, LAYERS, OUT, atomic_json, provenance
from eventsim import NK, next_use_steps, preds, trace

AF, AS = 1 / 8, 1 / 128
HS = (16, 64, 256)


def auc(score, y):
    o = np.argsort(score); r = np.empty(len(o)); r[o] = np.arange(1, len(o) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0))


def fit_logistic(X, y, l2=1e-3, iters=30):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (p - y) + l2 * np.r_[0, w[1:]]
        H = (Z * (p * (1 - p))[:, None]).T @ Z + l2 * np.eye(len(w))
        w -= np.linalg.solve(H, g)
    raw = np.r_[w[0] - (w[1:] * mu / sd).sum(), w[1:] / sd]
    return raw


def main():
    tr = trace()
    nxt, first, _, _ = next_use_steps(tr)
    P = preds(); pab, cab = P["pred_xtok_AB"], P["cal_xtok_AB"]
    cap = int(75e9 // 18_800_640)
    lru = OrderedDict()
    last = np.full(NK, -(1 << 30), np.int64); ef = np.zeros(NK); efa = np.zeros(NK, np.int64)
    es = np.zeros(NK); esa = np.zeros(NK, np.int64)
    nextstep = np.array(first, np.int64)
    logf = np.log1p(tr["freq_train"].astype(np.float64))
    rng = np.random.default_rng(0)
    rows_X, rows_nu, rows_split = [], [], []
    i = 0
    dec_count = 0
    split = tr["tok_split"]
    for si, stp in enumerate(tr["steps"]):
        if stp[0] == 1:
            r = stp[1]; dec_count += 1
            sample_layer = int(rng.integers(LAYERS)) if dec_count % 8 == 0 else -1
            row = tr["ids"][r].tolist()
            layer_lists = row
        else:
            r = stp[1]; sample_layer = -1
            layer_lists = stp[2]
        for l in range(LAYERS):
            if l == sample_layer and len(lru) >= cap:
                keys = np.array(list(lru.keys()), np.int64)
                cand = keys[rng.integers(len(keys), size=64)]
                age = si - last[cand]
                f_ef = ef[cand] * (1 - AF) ** (si - efa[cand])
                f_es = es[cand] * (1 - AS) ** (si - esa[cand])
                d = (((cand // EXPERTS) - l - 1) % LAYERS + 1) / LAYERS
                pn = np.zeros(len(cand))
                pr = pab[r]
                if pr[0, 0] >= 0 or pr[1, 0] >= 0:
                    for j, q in enumerate(cand.tolist()):
                        L, e = divmod(q, EXPERTS)
                        hit = np.where(pr[L] == e)[0]
                        if len(hit):
                            pn[j] = cab[L, hit[0]]
                X = np.c_[np.log1p(age), f_ef, f_es, logf[cand], d, pn]
                rows_X.append(X); rows_nu.append(nextstep[cand] - si); rows_split.append(np.full(64, split[r]))
            for e in layer_lists[l]:
                q = l * EXPERTS + e
                nextstep[q] = nxt[i]; i += 1
                last[q] = si
                if efa[q] != si:
                    ef[q] = ef[q] * (1 - AF) ** (si - efa[q]) + AF; efa[q] = si
                    es[q] = es[q] * (1 - AS) ** (si - esa[q]) + AS; esa[q] = si
                if q in lru:
                    lru.move_to_end(q)
                else:
                    lru[q] = None
                    if len(lru) > cap:
                        lru.popitem(last=False)
    X = np.concatenate(rows_X); NU = np.concatenate(rows_nu); SP = np.concatenate(rows_split)
    tr_m, ho_m = SP == 0, SP == 1
    out = provenance(alpha_fast=AF, alpha_slow=AS, horizons=HS, samples=int(len(X)),
                     train_samples=int(tr_m.sum()), heldout_samples=int(ho_m.sum()), replay="LRU 75 GB")
    models = {}
    for H in HS:
        y = (NU <= H).astype(np.float64)
        for use_pn in (False, True):
            Xu = X if use_pn else X[:, :5]
            w = fit_logistic(Xu[tr_m], y[tr_m])
            z = np.c_[np.ones(ho_m.sum()), Xu[ho_m]] @ w
            # baselines on the same heldout samples: LRU (recency) and EMA-fast
            m = {"w": (w.tolist() + ([0.0] if not use_pn else [])), "alpha_fast": AF, "alpha_slow": AS, "H": H,
                 "use_pn": use_pn, "heldout_auc": auc(z, y[ho_m]),
                 "heldout_auc_recency": auc(-X[ho_m, 0], y[ho_m]), "heldout_auc_ema_fast": auc(X[ho_m, 1], y[ho_m]),
                 "heldout_pos_rate": float(y[ho_m].mean())}
            models[f"H{H}{'_pn' if use_pn else ''}"] = m
            print(f"H={H} pn={use_pn} auc {m['heldout_auc']:.4f} recency {m['heldout_auc_recency']:.4f} "
                  f"ema_fast {m['heldout_auc_ema_fast']:.4f} pos {m['heldout_pos_rate']:.3f} w {np.round(w, 3).tolist()}",
                  flush=True)
    out["models"] = models
    out["feature_order"] = ["bias", "log1p_age_steps", "ema_fast", "ema_slow", "log1p_train_freq", "cycle_dist_frac",
                            "xtokAB_prob_current_token"]
    out["model"] = models["H64"]
    atomic_json(f"{OUT}/nextuse.json", out)


if __name__ == "__main__":
    main()

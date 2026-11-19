#!/usr/bin/env python3
"""Untangling diagnostic for shared FN neurons (Pierre 2026-11-18: "take some shared neuron. Is it at all possible to
un-entangle the dims from shared neurons where maybe the number of shared connections for an arbitrary neuron are
because of some small % of the dims? And in those cases can we migrate the dims out of that neuron into another box's
neurons or a new neuron in another box?"). CPU only (MLX on the CPU stream for dequantization); reads the x moments of
scripts/amoeba_fn_profile_capture.py, an scripts/amoeba_fn_partition.py run and the model's switch-MLP weights.

Neuron math: a = silu(g . x) (u . x), output a * w_down (times the routing weight), x the expert input.
For a neuron claimed by boxes A and B (pair tile) and topic means mu_A, mu_B of x over the tokens routed to its expert:
  contrast   c_d = g_d (mu_A - mu_B)_d, sum = g . (mu_A - mu_B) (the pre-activation difference); the same for u.
             concentration: L1 share of the top-k |c_d| (k = --topk), dims for 90% of the L1 mass, participation
             ratio (effective dims), and the share carried along the top principal directions of the pooled A+B
             covariance (Nystrom sketch + diagonal; scripts/amoeba_fn_profile_lib.py).
  drives     with a baseline x group (--baseline, English): g . (mu_A - mu_0) and g . (mu_B - mu_0), their top-k dim
             sets and the Jaccard of the two sets (do the two topics drive the neuron through different dims?).
  split      copy the neuron into box B with g, u restricted to a dim set S (g_S = g * 1_S), remove that part from the
             original (g_O = g - g_S); S = the top-k dims by |g_d delta_d| / sum + |u_d delta_d| / sum, delta the A-B
             contrast ("contrast") or the moved topic's drive ("drive"). Under a Gaussian model of x per topic (mean,
             low-rank + diagonal covariance from the capture), with common random draws:
               keep = E_A[(a - a_O)^2] / E_A[a^2]   (the remainder serves A alone: box A)
               copy = E_B[(a - a_S)^2] / E_B[a^2]   (the copy serves B alone: box B)
               sum  = E_B[(a - a_O - a_S)^2] / E_B[a^2] (both open on B: remainder in the core, copy in box B)
               leak = E_B[a_O^2] / E_B[a^2]          (how much of the remainder still fires on B)
             both directions (B moved out, A moved out). A pair split is viable at k when max(keep, copy) <= --tau;
             candidates rank by the smallest viable k, then the error.
For a 3+-shared (core) neuron: each claimed box b against the pooled other claimed boxes, the core scenario
(remainder stays in the core, copy in box b): viable when max(keep on the others, sum on b) <= --tau.
Noise: the per-dim standard error of mu_A - mu_B (from n, mean, E[x^2]; tokens taken as independent, so optimistic)
gives each contrast a noise null (the top-k L1 share a pure-noise contrast shows; well above k / D, how far depends
on how heavy-tailed w * se is) and a signal-to-noise |w . delta| / se (se: the larger of the diagonal estimate and the
moment model's w^T Sigma w / n); a split is ranked only at SNR >= --min-snr. Model check: the Gaussian model's E[a^2] per neuron and topic against the exact E[h^2] the capture keeps
(the rank-8 sketch under-states variance along data-aligned directions); a split is ranked only when every topic is
within --max-model-ratio (viable_raw keeps the ungated answer), plus the Tool A check of predicted mean |a| * w.
Expert topic mixing (the router alternative: duplicate an expert and split its routing): per (layer, expert) the
distribution of its routed tokens over the profile groups (each group's rate, so group sizes cancel) and over the box
groups only; effective group count exp(H); the share of its neurons that are pair / 3+-shared; the rank correlation of
the two, and the experts where both are high (duplication candidates).

Writes OUT/report.json and OUT/neurons.jsonl (one row per analysed neuron).
Usage: amoeba_fn_untangle_analyse.py --capture CAP --partition PART --model MODEL_DIR --out DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import amoeba_fn_profile_lib as PL  # noqa: E402


def topk_dims(g, u, delta, k):
    sg = np.abs(g * delta)
    su = np.abs(u * delta)
    s = sg / max(sg.sum(), 1e-30) + su / max(su.sum(), 1e-30)
    return np.argsort(-s)[:k]


def spearman(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if x.size < 3:
        return None
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    c = np.corrcoef(rx, ry)[0, 1]
    return None if not np.isfinite(c) else float(c)


def entropy_eff(p, axis=0):
    p = p / np.maximum(p.sum(axis, keepdims=True), 1e-30)
    h = -(np.where(p > 0, p * np.log(np.maximum(p, 1e-300)), 0.0)).sum(axis)
    return np.exp(h)


class XCache:
    """Per-layer x moments of the needed x groups (one layer resident at a time)."""

    def __init__(self, cap, names):
        self.st = {x: cap.xstats(x) for x in names}
        self.li = None
        self.data = {}

    def get(self, xg, li, e):
        if self.li != li:
            self.data, self.li = {}, li
        if xg not in self.data:
            self.data[xg] = self.st[xg].layer(li)
        mu, m2, y = self.data[xg]
        n = self.st[xg].n[li, e]
        return n, mu[e].astype(np.float64), m2[e].astype(np.float64), None if y is None else y[e].astype(np.float64)

    def h(self, xg, li, e):
        if self.li != li:
            self.data, self.li = {}, li
        key = ("h", xg)
        if key not in self.data:
            self.data[key] = self.st[xg].h_moments(li)
        h1, h2 = self.data[key]
        return h1[e], h2[e]


def moments(stat, omega):
    n, mu, m2, y = stat
    B = PL.nystrom_factor(y, omega) if y is not None and omega is not None else None
    return PL.Moments(mu, m2, B)


def pcs(mom, k):
    """Top-k principal directions of the model covariance (span of the Nystrom factor and the mean)."""
    A = np.concatenate([mom.B, mom.mu[:, None]], axis=1)
    Q, _ = np.linalg.qr(A)
    S = mom.cov(Q)
    w, V = np.linalg.eigh(0.5 * (S + S.T))
    o = np.argsort(-w)[:k]
    return Q @ V[:, o], w[o]


def split_eval(g, u, mom_keep, mom_move, S, z):
    mask = np.zeros(g.size)
    mask[S] = 1.0
    rk = PL.gaussian_split(mom_keep, g, u, mask, z)
    rm = PL.gaussian_split(mom_move, g, u, mask, z)
    d_k, d_m = max(rk["a2"], 1e-30), max(rm["a2"], 1e-30)
    return {"keep": rk["keep"] / d_k, "copy": rm["copy"] / d_m, "sum": rm["sum"] / d_m, "leak": rm["leak"] / d_m}


def snr_of(w, delta, se, momA, nA, momB, nB):
    """|w . delta| over its standard error, the larger of two variance estimates: the moment model's w^T Sigma w / n
    per topic (diagonal + sketch: correlated dims) and the diagonal one (review 2026-11-18). Tokens are counted as
    independent, so the SNR is still optimistic."""
    vm = float(momA.cov(w[:, None])[0, 0]) / max(nA, 1.0) + float(momB.cov(w[:, None])[0, 0]) / max(nB, 1.0)
    vd = float(((w * se) ** 2).sum())
    return abs(float(w @ delta)) / max(np.sqrt(max(vm, vd, 0.0)), 1e-30)


def rnd(d, sig=4):
    """Four significant figures (FN's E[h^2] is ~1e-4..1e-2: decimal rounding would print zeros)."""
    if isinstance(d, dict):
        return {k: rnd(v, sig) for k, v in d.items()}
    if isinstance(d, list):
        return [rnd(v, sig) for v in d]
    if isinstance(d, float):
        return float(f"{d:.{sig}g}")
    return d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--partition", required=True)
    ap.add_argument("--model", required=True, help="model dir (safetensors) the capture ran")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", type=int, default=12, help="largest pair tiles analysed")
    ap.add_argument("--per-tile", type=int, default=300)
    ap.add_argument("--shared3", type=int, default=1500, help="3+-shared neurons sampled")
    ap.add_argument("--topk", default="8,32,128")
    ap.add_argument("--pcs", type=int, default=8)
    ap.add_argument("--mc", type=int, default=512)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--baseline", default="english", help="x group of the maps' baseline ('' for none)")
    ap.add_argument("--min-snr", type=float, default=3.0,
                    help="rank a split only when |w . delta| / its standard error >= this (g or u)")
    ap.add_argument("--max-model-ratio", type=float, default=2.0,
                    help="rank a split only when the model's E[a^2] is within this factor of the exact E[h^2] on "
                         "every topic")
    ap.add_argument("--min-tokens", type=float, default=32, help="skip a (group, expert) with fewer routed tokens")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {out}")
    out.mkdir(parents=True, exist_ok=True)
    log = lambda *m: print(*m, flush=True)    # noqa: E731
    ks = [int(x) for x in a.topk.split(",") if x]
    cap = PL.Capture(a.capture)
    part = PL.partition_categories(a.partition, log=log)
    if part["layers"] != cap.layers or part["n"] != cap.n:
        raise SystemExit("capture and partition layouts differ")
    if part["partition"].get("fingerprint") and part["partition"]["fingerprint"] != cap.rep["fingerprint"]:
        raise SystemExit("capture and partition are of different models")
    cat, claims, names = part["cat"], part["claims"], part["names"]
    E, F, L = cap.E, cap.F, len(cap.layers)
    xgs = set(cap.xgroups())
    x_of = {g["id"]: g["x_group"] for g in cap.rep["groups"]}
    box_x = {}
    for b in names:
        gid = cap.rep.get("x_box_alias", {}).get(b)
        xg = x_of.get(gid) if gid else None
        box_x[b] = xg if xg in xgs and xg != "rest" else None
    base = a.baseline if a.baseline in xgs else None
    rng = np.random.default_rng(a.seed)
    z = rng.standard_normal((a.mc, 4))
    bit = {nm: 1 << i for i, nm in enumerate(names)}

    # ------------------------------------------------------------------ neuron selection
    jobs = []
    pairs = [k for k in part["partition"]["pairs"] if all(box_x.get(b) for b in k.split("__"))][: a.pairs]
    for key in pairs:
        A_, B_ = key.split("__")
        m = (cat == PL.CATS["pair"]) & (claims == (bit[A_] | bit[B_]))
        idx = np.argwhere(m)
        if len(idx) > a.per_tile:
            idx = idx[rng.choice(len(idx), a.per_tile, replace=False)]
        jobs += [("pair", int(li), int(u), [A_, B_]) for li, u in idx]
    s3 = np.argwhere(cat == PL.CATS["shared3"])
    rng.shuffle(s3)
    took = 0
    for li, u in s3:
        bs = [b for b in PL.members(claims[li, u], names) if box_x.get(b)]
        if len(bs) >= 2:
            jobs.append(("shared3", int(li), int(u), bs))
            took += 1
            if took >= a.shared3:
                break
    jobs.sort(key=lambda j: (j[1], j[2]))
    need = sorted({box_x[b] for j in jobs for b in j[3]} | ({base} if base else set()))
    log(f"{len(jobs)} neurons ({len(pairs)} pair tiles, {took} shared3); x groups {need}")
    W = PL.ExpertWeights(a.model, cap.D, F)
    xc = XCache(cap, need)
    omega = None
    if need:
        omega = xc.st[need[0]].omega
    wsum = cap.wsum()
    gi_of = {g: i for i, g in enumerate(cap.groups)}
    alias = cap.rep.get("x_box_alias", {})

    rows, gauss, gauss_a2 = [], [], []
    for jn, (kind, li, unit, boxes) in enumerate(jobs):
        e, f = divmod(unit, F)
        g, u, d = W.rows(cap.layers[li], e)
        g, u, wd = g[f].astype(np.float64), u[f].astype(np.float64), d[:, f].astype(np.float64)
        st = {b: xc.get(box_x[b], li, e) for b in boxes}
        if any(st[b][0] < a.min_tokens for b in boxes):
            continue
        st0 = xc.get(base, li, e) if base else None
        row = {"kind": kind, "layer": cap.layers[li], "expert": e, "neuron": f, "unit": unit, "boxes": boxes,
               "routed": {b: float(st[b][0]) for b in boxes}, "w_down_norm": float(np.linalg.norm(wd))}
        moms = {b: moments(st[b], omega) for b in boxes}
        row["model"] = {}
        for b in boxes:          # the Gaussian model of x against this neuron's exact E[h], E[h^2] (and Tool A's mean)
            r = PL.gaussian_split(moms[b], g, u, np.zeros(g.size), z)
            h1, h2 = xc.h(box_x[b], li, e)
            lr = float((moms[b].B ** 2).sum())
            row["model"][b] = {"a2_model": r["a2"], "a2_exact": float(h2[f]), "ea_model": r["ea"],
                               "ea_exact": float(h1[f]), "ratio": r["a2"] / max(float(h2[f]), 1e-30),
                               "sketch_share": lr / max(float(st[b][2].sum()), 1e-30)}
            gauss_a2.append((r["a2"], float(h2[f])))
            gi = gi_of.get(alias.get(b))
            if gi is not None:
                pred = r["eabs"] * float(wsum[gi, li, e]) / max(cap.tokens[gi], 1.0)
                gauss.append((pred, cap.profile_value("mean", gi, li, unit)))
        if kind == "pair":
            A_, B_ = boxes
            delta = st[A_][1] - st[B_][1]
            row["contrast"] = {"g": PL.concentration(g * delta, ks), "u": PL.concentration(u * delta, ks)}
            se = PL.mean_diff_se(st[A_], st[B_])
            for nm, w in (("g", g), ("u", u)):
                null, _ = PL.noise_null(w, se, ks, rng)
                row["contrast"][nm]["null_l1_share"] = null
                row["contrast"][nm]["snr"] = snr_of(w, delta, se, moms[A_], st[A_][0], moms[B_], st[B_][0])
            row["snr_max"] = max(row["contrast"]["g"]["snr"], row["contrast"]["u"]["snr"])
            npool, mu_p, m2_p, y_p = PL.Moments.pooled([st[A_], st[B_]])
            mp = moments((npool, mu_p, m2_p, y_p), omega)
            V, lam = pcs(mp, a.pcs)
            gd = float(g @ delta)
            row["pc"] = {"carried_g": float(g @ V @ (V.T @ delta)) / gd if gd else None,
                         "delta_energy": float(((V.T @ delta) ** 2).sum() / max((delta ** 2).sum(), 1e-30)),
                         "eigvals": [float(x) for x in lam]}
            drives = {}
            if st0 is not None and st0[0] >= a.min_tokens:
                for b in boxes:
                    dv = st[b][1] - st0[1]
                    drives[b] = dv
                    row.setdefault("drive", {})[b] = {"g_dot": float(g @ dv), "u_dot": float(u @ dv),
                                                       "g": PL.concentration(g * dv, ks)}
                row["drive_jaccard"] = {}
                for k in ks:
                    sa = set(topk_dims(g, u, drives[A_], k).tolist())
                    sb = set(topk_dims(g, u, drives[B_], k).tolist())
                    row["drive_jaccard"][k] = len(sa & sb) / max(1, len(sa | sb))
            splits = []
            for keep, move in ((A_, B_), (B_, A_)):
                modes = {"contrast": delta}
                if move in drives:
                    modes["drive"] = drives[move]
                for mode, dv in modes.items():
                    for k in ks:
                        r = split_eval(g, u, moms[keep], moms[move], topk_dims(g, u, dv, k), z)
                        splits.append({"move": move, "mode": mode, "k": k, **r})
            row["splits"] = splits
            ok = [s for s in splits if max(s["keep"], s["copy"]) <= a.tau]
            row["viable"] = min(ok, key=lambda s: (s["k"], max(s["keep"], s["copy"]))) if ok else None
            row["best"] = min(splits, key=lambda s: max(s["keep"], s["copy"]))
        else:
            per_b = []
            for b in boxes:
                others = [st[o] for o in boxes if o != b]
                npool, mu_o, m2_o, y_o = PL.Moments.pooled(others)
                mo = moments((npool, mu_o, m2_o, y_o), omega)
                delta = st[b][1] - mu_o
                se = PL.mean_diff_se(st[b], (npool, mu_o, m2_o))
                snr = max(snr_of(w, delta, se, moms[b], st[b][0], mo, npool) for w in (g, u))
                best = None
                for k in ks:
                    r = split_eval(g, u, mo, moms[b], topk_dims(g, u, delta, k), z)
                    r = {"box": b, "k": k, "keep": r["keep"], "sum": r["sum"], "leak": r["leak"]}
                    if best is None or max(r["keep"], r["sum"]) < max(best["keep"], best["sum"]):
                        best = r
                per_b.append({"box": b, "contrast_g": PL.concentration(g * delta, ks), "snr": snr, "best": best})
            row["per_box"] = per_b
            row["snr_max"] = max(p["snr"] for p in per_b)
            def pick(c):
                return min(c, key=lambda s: (s["k"], max(s["keep"], s["sum"]))) if c else None
            row["viable_raw"] = pick([p["best"] for p in per_b if max(p["best"]["keep"], p["best"]["sum"]) <= a.tau])
            row["viable"] = pick([p["best"] for p in per_b
                                  if max(p["best"]["keep"], p["best"]["sum"]) <= a.tau and p["snr"] >= a.min_snr])
        # gates: a split chosen on a noise-level contrast, or estimated by a moment model that misses this neuron's
        # exact E[h^2] by more than --max-model-ratio on a topic, is reported (viable_raw) but not ranked (viable)
        snr_ok = row["snr_max"] >= a.min_snr
        fid_ok = all(1 / a.max_model_ratio <= m["ratio"] <= a.max_model_ratio for m in row["model"].values())
        if kind == "pair":
            row["viable_raw"] = row["viable"]
        row["gates"] = {"snr": snr_ok, "model": fid_ok}
        if not (snr_ok and fid_ok):
            row["viable"] = None
        rows.append(rnd(row))
        if jn % 200 == 0:
            log(f"{jn}/{len(jobs)} neurons")
    with open(out / "neurons.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")

    # ------------------------------------------------------------------ summaries
    summ = {"pair": {}, "shared3": {}}
    for key in pairs:
        rs = [r for r in rows if r["kind"] == "pair" and "__".join(r["boxes"]) == key]
        if not rs:
            continue
        v = [r["viable"] for r in rs if r["viable"]]
        summ["pair"][key] = {
            "neurons": len(rs),
            "contrast_g_l1_share": {k: float(np.median([r["contrast"]["g"]["l1_share"][k] for r in rs])) for k in ks},
            "contrast_u_l1_share": {k: float(np.median([r["contrast"]["u"]["l1_share"][k] for r in rs])) for k in ks},
            "noise_null_g_l1_share": {k: float(np.median([r["contrast"]["g"]["null_l1_share"][k] for r in rs]))
                                      for k in ks},
            "excess_g_l1_share_median": {k: float(np.median([r["contrast"]["g"]["l1_share"][k]
                                                             - r["contrast"]["g"]["null_l1_share"][k] for r in rs]))
                                         for k in ks},
            "snr_max_median": float(np.median([r["snr_max"] for r in rs])),
            "snr_ok_share": float(np.mean([r["gates"]["snr"] for r in rs])),
            "model_ok_share": float(np.mean([r["gates"]["model"] for r in rs])),
            "viable_raw_share": sum(1 for r in rs if r["viable_raw"]) / len(rs),
            "contrast_g_dims90_median": float(np.median([r["contrast"]["g"]["dims90"] for r in rs])),
            "contrast_g_pr_median": float(np.median([r["contrast"]["g"]["pr"] for r in rs])),
            "pc_delta_energy_median": float(np.median([r["pc"]["delta_energy"] for r in rs])),
            "drive_jaccard_median": ({k: float(np.median([r["drive_jaccard"][k] for r in rs if "drive_jaccard" in r]))
                                      for k in ks} if any("drive_jaccard" in r for r in rs) else None),
            "viable_share": len(v) / len(rs),
            "viable_by_k": {k: sum(1 for x in v if x["k"] == k) for k in ks},
            "best_err_median": float(np.median([max(r["best"]["keep"], r["best"]["copy"]) for r in rs]))}
    rs3 = [r for r in rows if r["kind"] == "shared3"]
    if rs3:
        v3 = [r["viable"] for r in rs3 if r["viable"]]
        summ["shared3"] = {"neurons": len(rs3), "viable_share": len(v3) / len(rs3),
                           "viable_raw_share": sum(1 for r in rs3 if r["viable_raw"]) / len(rs3),
                           "snr_ok_share": float(np.mean([r["gates"]["snr"] for r in rs3])),
                           "model_ok_share": float(np.mean([r["gates"]["model"] for r in rs3])),
                           "viable_by_k": {k: sum(1 for x in v3 if x["k"] == k) for k in ks},
                           "by_claims": {str(n): sum(1 for r in rs3 if len(r["boxes"]) == n)
                                         for n in sorted({len(r["boxes"]) for r in rs3})}}
    cand = sorted([r for r in rows if r["viable"]],
                  key=lambda r: (r["viable"]["k"], max(r["viable"]["keep"], r["viable"].get("copy", r["viable"].get("sum")))))
    gk = np.array(gauss) if gauss else np.zeros((0, 2))
    gcheck = {"neurons_x_topics": int(gk.shape[0]),
              "spearman_pred_vs_captured": spearman(gk[:, 0], gk[:, 1]) if gk.size else None,
              "median_ratio_pred_over_captured": float(np.median(gk[:, 0] / np.maximum(gk[:, 1], 1e-30)))
              if gk.size else None,
              "note": "predicted mean |a| x mean routing weight x routed share vs the captured mean |h| w per token; "
                      "assumes |a| and w independent per token"}
    ga = np.array(gauss_a2) if gauss_a2 else np.zeros((0, 2))
    if ga.size:
        rat = ga[:, 0] / np.maximum(ga[:, 1], 1e-30)
        gcheck["a2_exact"] = {"spearman": spearman(ga[:, 0], ga[:, 1]), "median_ratio": float(np.median(rat)),
                              "share_within_max_ratio": float(np.mean((rat >= 1 / a.max_model_ratio)
                                                                      & (rat <= a.max_model_ratio))),
                              "quantiles_ratio": [float(x) for x in np.quantile(rat, [0.1, 0.25, 0.5, 0.75, 0.9])],
                              "note": "the Gaussian model's E[a^2] vs the exact E[h^2] over the same routed tokens "
                                      "(per neuron and topic): the normaliser of every split error"}

    # ------------------------------------------------------------------ expert topic mixing
    routed = np.asarray(cap.routed(), np.float64)                     # [G, L, E]
    rate = routed / np.maximum(cap.tokens, 1.0)[:, None, None]
    eff_all = entropy_eff(rate, 0)                                    # [L, E]
    bg = sorted({gi_of[alias[b]] for b in names if alias.get(b) in gi_of}
                | {gi_of[g] for g, x in x_of.items() if base and x == base})      # + the baseline's profile groups
    eff_box = entropy_eff(rate[bg], 0) if bg else np.zeros_like(eff_all)
    cr = cat.reshape(L, E, F)
    pair_share = (cr == PL.CATS["pair"]).mean(2)
    s3_share = (cr == PL.CATS["shared3"]).mean(2)
    breadth = np.zeros((L, E), np.int64)
    cl3 = claims.reshape(L, E, F)
    orr = np.bitwise_or.reduce(cl3, axis=2)
    for i in range(len(names)):
        breadth += (orr >> i) & 1
    shared = pair_share + s3_share
    used = routed.sum(0) > 0
    mix = {"groups": len(cap.groups), "box_groups": len(bg),
           "eff_groups_all_median": float(np.median(eff_all[used])) if used.any() else None,
           "eff_box_groups_median": float(np.median(eff_box[used])) if used.any() else None,
           "spearman_eff_box_vs_shared_share": spearman(eff_box[used], shared[used]),
           "spearman_eff_box_vs_claim_breadth": spearman(eff_box[used], breadth[used]),
           "by_layer": {str(cap.layers[l]): {"eff_box_median": float(np.median(eff_box[l][used[l]])) if used[l].any()
                                             else None, "shared_share_mean": float(shared[l].mean())}
                        for l in range(L)}}
    score = np.where(used, eff_box / max(1, len(bg)) * shared, -1)
    top = np.dstack(np.unravel_index(np.argsort(-score, axis=None)[:40], score.shape))[0]
    mix["duplication_candidates"] = []
    for l, e in top:
        if score[l, e] <= 0:
            break
        p = rate[bg, l, e] / max(rate[bg, l, e].sum(), 1e-30)
        o = np.argsort(-p)[:5]
        mix["duplication_candidates"].append(
            {"layer": cap.layers[l], "expert": int(e), "eff_box_groups": round(float(eff_box[l, e]), 2),
             "pair_share": round(float(pair_share[l, e]), 3), "shared3_share": round(float(s3_share[l, e]), 3),
             "claim_breadth": int(breadth[l, e]),
             "top_box_groups": {cap.groups[bg[i]]: round(float(p[i]), 3) for i in o}})
    rep = {"schema": "mlx2.amoeba-fn-untangle.v1", "checkpoint": cap.rep["checkpoint"], "capture": str(cap.d),
           "capture_revision": cap.rep["mlx2_source_revision"], "partition": str(a.partition), "model": str(a.model),
           "params": vars(a), "baseline_x_group": base, "box_x_groups": box_x, "pairs_analysed": pairs,
           "summary": summ, "gaussian_check": gcheck, "expert_mixing": mix,
           "top_candidates": [{k: r[k] for k in ("kind", "layer", "expert", "neuron", "boxes", "viable")}
                              for r in cand[:50]],
           "model_note": "split errors are estimates under a Gaussian model of x per topic (mean + rank-r Nystrom + "
                         "diagonal covariance); they rank candidates, a carve-side A/B on held text decides"}
    PL.write_json(out / "report.json", rep)
    log(json.dumps({"summary": summ, "gaussian_check": gcheck}, default=str)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())

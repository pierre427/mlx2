#!/usr/bin/env python3
"""Firing map from two captured corpora (scripts/amoeba_firing_capture.py sums): which units fire more on the target
corpus than on the baseline, per document (Pierre 2026-11-16). CPU only.

Per unit: Welch t on the per-document mean contribution, Cohen's d (pooled sd), two-sided normal p, Benjamini-Hochberg
over every unit of every site. A unit is a target unit at q <= --q and d >= --min-d. Checks: split halves (target half
h vs baseline half h, every source pooled), and per target source (each source vs the whole baseline; a target unit
must hold in every source, so a register is not read as the topic). The two captures must be of the same model
(fingerprint) and capture revision.

Writes OUT/map.npz (per site: target, every_source, d, q, mean_target, mean_baseline; routed shares for MoE sites) and
OUT/report.json.

Usage: amoeba_firing_compare.py --target CAPTURE/NAME --baseline CAPTURE/NAME --out DIR [--base T2.npz]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def bh(p):
    n = p.size
    o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1.0)
    return out


def erfc_p(t):
    """Two-sided normal p (Numerical Recipes erfcc, relative error < 1.2e-7)."""
    z = np.abs(t) / math.sqrt(2)
    tt = 1.0 / (1.0 + 0.5 * z)
    y = tt * np.exp(-z * z - 1.26551223 + tt * (1.00002368 + tt * (0.37409196 + tt * (0.09678418 + tt * (
        -0.18628806 + tt * (0.27886807 + tt * (-1.13520398 + tt * (1.48851587 + tt * (-0.82215223 + tt * 0.17087277)))))))))
    return np.minimum(1.0, y)


def load(d):
    d = Path(d)
    rep = json.loads((d / "report.json").read_text())
    z = np.load(d / "sums.npz")
    G, S = len(rep["groups"]), len(rep["sites"])
    s1 = [[z[f"S1_{g}_{s}"].astype(np.float64) for s in range(S)] for g in range(G)]
    s2 = [[z[f"S2_{g}_{s}"].astype(np.float64) for s in range(S)] for g in range(G)]
    r = [[z[f"R_{g}_{s}"].astype(np.float64) for s in range(S)] for g in range(G)]
    return rep, s1, s2, r, [int(x) for x in z["n"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="", help="base tier NPZ (key core; [L, E] expert units are expanded) for overlap")
    ap.add_argument("--q", type=float, default=0.01)
    ap.add_argument("--min-d", type=float, default=0.5)
    a = ap.parse_args()
    tr, t1, t2, trr, tn = load(a.target)
    br, b1, b2, brr, bn = load(a.baseline)
    if tr["fingerprint"] != br["fingerprint"]:
        raise SystemExit(f"captures of different models: {tr['fingerprint']} vs {br['fingerprint']}")
    if tr["identity"]["capture_rev"] != br["identity"]["capture_rev"] or tr["sites"] != br["sites"]:
        raise SystemExit("captures from different capture revisions or site layouts")
    for k in ("tokens", "split", "seed", "source_rev"):
        if tr["identity"].get(k) != br["identity"].get(k):
            raise SystemExit(f"target and baseline captures differ in {k}: {tr['identity'].get(k)} vs {br['identity'].get(k)}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    sites = tr["sites"]
    S = len(sites)
    tg, bg = tr["groups"], br["groups"]

    def mom(s1, s2, n, gs):
        N = sum(n[g] for g in gs)
        m = [sum(s1[g][s] for g in gs) / N for s in range(S)]
        v = [np.maximum(sum(s2[g][s] for g in gs) / N - m[s] ** 2, 0.0) * N / max(1, N - 1) for s in range(S)]
        return N, m, v

    def test(tgs, bgs):
        n1, m1, v1 = mom(t1, t2, tn, tgs)
        n0, m0, v0 = mom(b1, b2, bn, bgs)
        t, d = [], []
        for s in range(S):
            se = np.sqrt(v1[s] / n1 + v0[s] / n0)
            sd = np.sqrt(((n1 - 1) * v1[s] + (n0 - 1) * v0[s]) / max(1, n1 + n0 - 2))
            with np.errstate(divide="ignore", invalid="ignore"):
                t.append(np.where(se > 0, (m1[s] - m0[s]) / se, 0.0))
                d.append(np.where(sd > 0, (m1[s] - m0[s]) / sd, 0.0))
        flat = np.concatenate(t)
        q = bh(erfc_p(flat))
        qs, o = [], 0
        for s in range(S):
            qs.append(q[o:o + t[s].size])
            o += t[s].size
        tm = [(qs[s] <= a.q) & (d[s] >= a.min_d) for s in range(S)]
        return {"tm": tm, "d": d, "q": qs, "m1": m1, "m0": m0, "n1": n1, "n0": n0}
    allt = list(range(len(tg)))
    allb = list(range(len(bg)))
    full = test(allt, allb)
    halves = [test([i for i, g in enumerate(tg) if g.endswith(f":{h}")], [i for i, g in enumerate(bg) if g.endswith(f":{h}")])
              for h in (0, 1)]
    tsrc = sorted({g.rsplit(":", 1)[0] for g in tg})
    by_src = {s: test([i for i, g in enumerate(tg) if g.rsplit(":", 1)[0] == s], allb) for s in tsrc}
    every = [np.logical_and.reduce([by_src[x]["tm"][s] for x in tsrc]) for s in range(S)]
    cnt = lambda ms: int(sum(int(m.sum()) for m in ms))                                   # noqa: E731
    inter = sum(int((halves[0]["tm"][s] & halves[1]["tm"][s]).sum()) for s in range(S))
    union = sum(int((halves[0]["tm"][s] | halves[1]["tm"][s]).sum()) for s in range(S))
    dc = np.corrcoef(np.concatenate(halves[0]["d"]), np.concatenate(halves[1]["d"]))[0, 1]
    per_site = []
    for s, st in enumerate(sites):
        row = {"layer": st["layer"], "class": st["class"], "units": st["n"], "target": int(full["tm"][s].sum()),
               "target_every_source": int((full["tm"][s] & every[s]).sum())}
        if st["class"] == "moe":
            ex = full["tm"][s].reshape(st["E"], st["F"]).sum(1)
            row.update({"experts_any": int((ex > 0).sum()), "experts_90pct_target": int((ex >= 0.9 * st["F"]).sum())})
        per_site.append(row)
    rep = {"schema": "mlx2.amoeba-firing-compare.v1", "checkpoint": tr["checkpoint"], "fingerprint": tr["fingerprint"],
           "quant": tr.get("quant"), "target": {"dir": a.target, "corpus": tr["corpus"], "docs": tn, "groups": tg},
           "baseline": {"dir": a.baseline, "corpus": br["corpus"], "docs": bn, "groups": bg},
           "capture_rev": tr["identity"]["capture_rev"], "mlx2_source_revision": tr.get("mlx2_source_revision"),
           "test": {"q": a.q, "min_d": a.min_d, "fdr": "Benjamini-Hochberg over every unit", "p": "normal on Welch t"},
           "units_total": int(sum(st["n"] for st in sites)), "target_units": cnt(full["tm"]),
           "target_in_every_source": int(sum(int((full["tm"][s] & every[s]).sum()) for s in range(S))),
           "split_half": {"h0": cnt(halves[0]["tm"]), "h1": cnt(halves[1]["tm"]), "jaccard": inter / max(1, union),
                          "d_corr": float(dc)},
           "by_target_source": {x: cnt(by_src[x]["tm"]) for x in tsrc}, "per_site": per_site}
    if a.base:     # a padded [L, max width] tier core, in expert units ([L, max E]) or this capture's units
        bm = np.load(a.base)["core"] > 0
        moe_e = max([st["E"] for st in sites if st["class"] == "moe"] or [0])
        expert_units = moe_e and bm.shape[1] == moe_e and all(st["n"] <= moe_e for st in sites if st["class"] == "dense")
        inb = 0
        for s, st in enumerate(sites):
            row = bm[st["layer"]]
            if st["class"] == "moe" and expert_units:
                row = np.repeat(row[:st["E"]], st["F"])
            if row.size < st["n"]:
                raise SystemExit(f"--base {a.base}: layer {st['layer']} row has {row.size} units, site has {st['n']} "
                                 f"(a base tier of another model or unit mode?)")
            inb += int((full["tm"][s] & row[:st["n"]]).sum())
        rep["base"] = {"path": a.base, "target_in_base": inb, "base_units": "expert" if expert_units else "capture"}
    arr = {}
    for s in range(S):
        arr.update({f"target_{s}": full["tm"][s], f"every_source_{s}": every[s], f"d_{s}": full["d"][s].astype(np.float32),
                    f"q_{s}": full["q"][s].astype(np.float32), f"mean_target_{s}": full["m1"][s].astype(np.float32),
                    f"mean_baseline_{s}": full["m0"][s].astype(np.float32)})
        if sites[s]["class"] == "moe":
            arr[f"routed_target_{s}"] = (sum(trr[g][s] for g in allt) / max(1, sum(tn))).astype(np.float32)
            arr[f"routed_baseline_{s}"] = (sum(brr[g][s] for g in allb) / max(1, sum(bn))).astype(np.float32)
    np.savez_compressed(out / "map.npz", **arr)
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "per_site"}, indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Per-document statistics for the aMoEba battery and the two P0 experiments (CPU only, no model).

docs/research/2026-11-20-amoeba-next-experiments.md 7.1: the battery used to write one mean dCE per held set, so no
paired standard error or group bootstrap could be computed. The writers (scripts/amoeba_box_battery.py --perdoc,
scripts/amoeba_fn_suff.py, scripts/amoeba_fn_pareto.py) now append per-document rows to a sidecar:

  PERDOC.jsonl   one row per (key, arm, set): {"key", "arm", "set", "ce": [...], "clean_ce": [...], "n": [...]}
                 (per-document CE sums of the arm and of the unmasked model, and the scored-target counts)
  DOCS.json      per held set: {"spec", "ids", "groups", "strata"} in document order

A document's dCE is (ce - clean_ce) / n. A paired difference between two arms on the same set is taken document by
document; its SE is sd / sqrt(n_docs); the 95% CI is a percentile bootstrap over source groups (2,000 resamples,
groups resampled with replacement within each stratum; a set whose groups are all distinct is a per-document
bootstrap, stratified by source). This module also holds the pre-registered decision rules of the two P0
experiments (1.5 and 4.5), applied by the experiment scripts at the end of their window and re-runnable here:

  amoeba_battery_stats.py paired --dir OUT --a KEY/ARM --b KEY/ARM --set S       one paired difference
  amoeba_battery_stats.py suff --dir OUT [--reference-battery B/battery.jsonl]    1.5 rules -> OUT/summary.json
  amoeba_battery_stats.py pareto --dir OUT                                        4.5 rules -> OUT/summary.json
  amoeba_battery_stats.py npz --dir OUT                                           PERDOC.jsonl -> battery.perdoc.npz
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

N_BOOT = 2000
LANG10 = ("cmn", "fra", "deu", "spa", "jpn", "rus", "arb", "hin", "swh", "vie")


# ----------------------------------------------------------------------------- sidecar I/O

def append_line(path, row):
    """Append one JSON row; a partial last line (a killed writer) is closed first, so this row stays readable."""
    p = Path(path)
    lead = ""
    if p.exists() and p.stat().st_size:
        with p.open("rb") as fh:
            fh.seek(-1, 2)
            lead = "" if fh.read(1) == b"\n" else "\n"
    with p.open("a") as fh:
        fh.write(lead + json.dumps(row) + "\n")
    return row


def append_perdoc(path, key, arm, set_name, rec):
    """One per-document row (rec = evals.english_cost perdoc entry: ce, clean_ce, n)."""
    row = {"key": key, "arm": arm, "set": set_name, "ce": [float(x) for x in rec["ce"]],
           "clean_ce": [float(x) for x in rec["clean_ce"]], "n": [int(x) for x in rec["n"]]}
    return append_line(path, row)


def load_perdoc(path):
    """{(key, arm, set): row}; a later row of the same triple replaces an earlier one (a resumed run)."""
    out = {}
    p = Path(path)
    if not p.exists():
        return out
    for x in p.read_text().split("\n"):
        try:
            r = json.loads(x)
        except ValueError:
            continue
        out[(r["key"], r["arm"], r["set"])] = r
    return out


def doc_dce(row):
    """Per-document dCE of a sidecar row (nats per scored target)."""
    ce, cl, n = (np.asarray(row[k], np.float64) for k in ("ce", "clean_ce", "n"))
    return (ce - cl) / np.maximum(n, 1)


def to_npz(perdoc_jsonl, out_npz):
    """The sidecar as battery.perdoc.npz: per (key|arm|set) the per-document dCE (float64) and counts."""
    rows = load_perdoc(perdoc_jsonl)
    arr = {}
    for (k, a, s), r in rows.items():
        arr[f"{k}|{a}|{s}|dce"] = doc_dce(r)
        arr[f"{k}|{a}|{s}|n"] = np.asarray(r["n"], np.int64)
    tmp = Path(str(out_npz) + ".tmp.npz")
    np.savez(tmp, **arr)
    tmp.replace(out_npz)
    return len(rows)


def doc_meta(rows, spec):
    """DOCS.json entry for a held set's loaded rows (in document order): ids, source groups and strata.

    Group: the row's "group" (category / subject / language corpora carry one; GSM8K has none, so each document is
    its own group). Stratum: "register" when present, else the source before its first ':' (each language corpus is
    about half web and half encyclopedia rows), else one stratum."""
    ids, groups, strata = [], [], []
    for i, r in enumerate(rows):
        ids.append(str(r.get("id", i)))
        groups.append(str(r.get("group") or f"doc:{r.get('id', i)}"))
        src = r.get("register") or str(r.get("source") or "").split(":")[0] or "all"
        strata.append(str(src))
    return {"spec": spec, "ids": ids, "groups": groups, "strata": strata}


# ----------------------------------------------------------------------------- statistics

def mean_se(v):
    v = np.asarray(v, np.float64)
    if v.size == 0:
        return float("nan"), float("nan")
    se = float(v.std(ddof=1) / math.sqrt(v.size)) if v.size > 1 else float("nan")
    return float(v.mean()), se


def _plan(groups, strata):
    """Per stratum, the list of document-index arrays of its groups."""
    groups = list(groups)
    strata = list(strata) if strata is not None else ["all"] * len(groups)
    by = {}
    for i, (g, s) in enumerate(zip(groups, strata)):
        by.setdefault(s, {}).setdefault(g, []).append(i)
    return [[np.asarray(ix, np.int64) for ix in gs.values()] for _, gs in sorted(by.items())]


def boot_indices(groups, strata=None, n_boot=N_BOOT, seed=0):
    """n_boot resampled document-index arrays: groups drawn with replacement within each stratum."""
    plan = _plan(groups, strata)
    g = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        parts = []
        for gs in plan:
            pick = g.integers(0, len(gs), len(gs))
            parts.extend(gs[j] for j in pick)
        out.append(np.concatenate(parts))
    return out


def group_bootstrap(values, groups, strata=None, n_boot=N_BOOT, seed=0, stat=np.mean):
    """Percentile 95% CI of stat(values) under the source-group bootstrap; values [docs] or [docs, ...]."""
    v = np.asarray(values, np.float64)
    if v.shape[0] != len(groups):
        raise ValueError(f"{v.shape[0]} values for {len(groups)} documents")
    bs = np.array([stat(v[ix]) for ix in boot_indices(groups, strata, n_boot, seed)])
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def paired(a, b, groups, strata=None, n_boot=N_BOOT, seed=0):
    """a - b per document: mean, SE, bootstrap 95% CI, n."""
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    m, se = mean_se(d)
    lo, hi = group_bootstrap(d, groups, strata, n_boot, seed)
    return {"mean": m, "se": se, "ci95": [lo, hi], "n": int(d.size)}


def summary_of(v, groups, strata=None, n_boot=N_BOOT, seed=0):
    m, se = mean_se(v)
    lo, hi = group_bootstrap(v, groups, strata, n_boot, seed)
    return {"mean": m, "se": se, "ci95": [lo, hi], "n": int(len(v))}


# ----------------------------------------------------------------------------- 1.5: core-without-box rules

SUFF_RULES = {
    "min_gain": 0.01, "rand_floor": 0.005, "rand_x": 2.0, "target": 0.05, "necessity_high": 0.01,
    "min_fill": 0.9, "floor_x": 2.0,
    "source": "docs/research/2026-11-20-amoeba-next-experiments.md 1.5 (pre-registered)",
    "useful": "suff_gain >= min_gain and its group-bootstrap 95% lower bound > 0 and suff_gain >= rand_x x "
              "max(rand_gain, rand_gain_inexp, rand_floor)",
    "served": "no if core_plus_own > target; vacuous if core_alone_own <= target; yes otherwise (in that order)",
    "four_way": "necessity high = necessity_net >= necessity_high (the battery's 14-of-99 threshold; not in the "
                "1.5 text, set here); sufficiency high = useful; high/high real box, low/high redundant (OR), "
                "high/low needed but core_plus fails elsewhere, low/low cosmetic",
    "size_matched": "a random reopening enters the ratio only when it reopened >= min_fill of the box's own-tile "
                    "units (core_plus_rand_inexp is drawn inside the box's experts among the units core_plus "
                    "closes, which can be few: it is then reported, marked not size-matched, and left out)",
    "floor": "7.6: a useful box whose suff_gain is under floor_x x the measured document-resampling floor is "
             "flagged below_floor (no finding)",
}


def four_way(nec_high, suff_high):
    if nec_high is None or suff_high is None:
        return None
    return {(True, True): "real_box", (False, True): "redundant_or", (True, False): "needed_core_plus_fails",
            (False, False): "cosmetic"}[(bool(nec_high), bool(suff_high))]


def suff_box(own, rules=SUFF_RULES, necessity_net=None, groups=None, strata=None, n_boot=N_BOOT, seed=0,
             mean_own=None, mean_necessity_net=None, fill=None):
    """The 1.5 verdict of one box. own: {arm: per-document dCE on the box's own held set, net of the path offset};
    arms core_plus, core_alone, core_plus_rand, core_plus_rand_inexp. necessity_net: without_own - random_own (a
    float). mean_own: the same arms under mean ablation (core_plus, core_alone) and mean_necessity_net. fill:
    {arm: reopened units / own-tile units} of the random reopenings (None = matched)."""
    cp, ca = np.asarray(own["core_plus"]), np.asarray(own["core_alone"])
    g = groups if groups is not None else [str(i) for i in range(cp.size)]
    sg = paired(ca, cp, g, strata, n_boot, seed)
    rg = {k: paired(ca, own[a], g, strata, n_boot, seed) for k, a in
          (("rand_gain", "core_plus_rand"), ("rand_gain_inexp", "core_plus_rand_inexp")) if a in own}
    cpm, cam = float(cp.mean()), float(ca.mean())
    arm_of = {"rand_gain": "core_plus_rand", "rand_gain_inexp": "core_plus_rand_inexp"}
    for k in rg:
        f = (fill or {}).get(arm_of[k])
        rg[k]["fill"] = f
        rg[k]["size_matched"] = bool(f is None or f >= rules["min_fill"])
    rmax = max([rg[k]["mean"] for k in rg if rg[k]["size_matched"]] + [rules["rand_floor"]])
    useful = bool(sg["mean"] >= rules["min_gain"] and sg["ci95"][0] > 0 and sg["mean"] >= rules["rand_x"] * rmax)
    # NaN-safe: a non-finite mean (a non-finite CE on any held document) is never "yes"
    served = ("no" if not (cpm <= rules["target"]) or cam != cam else
              "vacuous" if cam <= rules["target"] else "yes")
    nec_high = None if necessity_net is None else bool(necessity_net >= rules["necessity_high"])
    row = {"core_plus_own": cpm, "core_alone_own": cam, "suff_gain": sg, **rg,
           "recovery": (sg["mean"] / cam) if cam > 0 else None, "necessity_net": necessity_net,
           "useful": useful, "served": served, "four_way": four_way(nec_high, useful),
           "rand_gain_used": rmax}
    if mean_own is not None and {"core_plus", "core_alone"} <= set(mean_own):
        sm = paired(mean_own["core_alone"], mean_own["core_plus"], g, strata, n_boot, seed)
        rm = rmax
        useful_m = bool(sm["mean"] >= rules["min_gain"] and sm["ci95"][0] > 0 and sm["mean"] >= rules["rand_x"] * rm)
        nn = mean_necessity_net if mean_necessity_net is not None else necessity_net
        cls_m = four_way(None if nn is None else nn >= rules["necessity_high"], useful_m)
        row["mean_ablation"] = {"suff_gain": sm, "useful": useful_m, "necessity_net": mean_necessity_net,
                                "four_way": cls_m, "rand_gain_used": rm,
                                "note": "random reopenings are zero-ablation arms (used as the ratio floor for both)"}
        row["ablation_sensitive"] = bool(cls_m != row["four_way"]) if cls_m and row["four_way"] else None
    return row


def expected_false_positives(n_boxes, floor_sd, threshold=SUFF_RULES["min_gain"]):
    """Expected count of boxes over `threshold` by noise alone if each suff_gain were N(0, floor_sd) (1.5
    'Counting'): n x P(Z > threshold / floor_sd)."""
    if not floor_sd or floor_sd <= 0 or not math.isfinite(floor_sd):
        return None
    return float(n_boxes * 0.5 * math.erfc(threshold / floor_sd / math.sqrt(2)))


# ----------------------------------------------------------------------------- 4.5: Pareto rules

PARETO_RULES = {
    "bar": 0.05, "h4_x": 2.0, "grain_points": 0.05, "comp_min": 0.01, "cliff_points": 0.10,
    "source": "docs/research/2026-11-20-amoeba-next-experiments.md 4.5 (pre-registered)",
    "cf": "CF@bar(order) = the largest closed fraction whose English dCE upper 95% bound is <= bar, linearly "
          "interpolated between that cell and the next cell above the bar; a cell flagged by the cliff guard "
          "counts as above the bar; censored (cf_censored) when every cell is under the bar",
    "band": "bootstrap band: the 2.5-97.5 percentiles of CF on the mean curve over the same source-group "
            "resamples (documents are shared by every cell, so resamples are joint); O-rand's band is the union "
            "over its seeds",
    "h4": "CF(O-fire) inside the O-rand band and CF(best causal of O-causal0, O-fisher) >= h4_x x CF(O-fire) > 0; "
          "the band test compares like with like: O-fire's CF on its mean curve (cf_mean) against the O-rand band "
          "of mean-curve CFs; the 2x clause uses the upper-bound CF",
    "grain_orders": "neuron grain: O-fire, O-causal0, O-fisher, O-reap-n (O-part is tile grain, O-rand the "
                    "control); expert grain: O-reap-e, O-freq-e",
    "grain": "neuron beats expert: CF(O-reap-n) - CF(O-reap-e) >= grain_points and CF(best neuron order) - "
             "CF(best expert order of O-reap-e, O-freq-e) >= grain_points",
    "comp": "mean compensation is a candidate feature: at the 20% cell of the best order, English dCE(zero) - "
            "dCE(mean) >= comp_min with its group-bootstrap lower bound > 0, and GSM8K dCE and capitals accuracy "
            "no worse (the 95% CI of mean - zero does not lie wholly on the harmful side)",
    "cliff": "a cell whose capitals accuracy drops by more than cliff_points vs the full model while English dCE "
             "<= bar is flagged and its fraction is excluded from CF",
    "floor": "7.4 / 7.6: the repeat cells (O-rand and O-causal0 at 20% on disjoint English val rows) give the "
             "document-sampling floor |repeat - cell|; an English effect under floor_x x floor is no finding",
    "floor_x": 2.0,
}


def cf_at(fracs, ub, bar, flagged=None):
    """(CF, censored) from per-cell (fraction, upper bound) points; fraction 0 is implicit (0 cost)."""
    pts = sorted(zip([0.0] + list(fracs), [0.0] + list(ub), [False] + list(flagged or [False] * len(fracs))))
    # censored only when every cell is under the bar (a flagged cell counts as above it): a curve that crossed the
    # bar and came back under it at the last cell (a non-monotone upper bound) is not censored
    best, censored = 0.0, all(u <= bar and not fl for _, u, fl in pts)
    for i, (f, u, fl) in enumerate(pts):
        ok = (u <= bar) and not fl
        if not ok:
            continue
        nxt = pts[i + 1] if i + 1 < len(pts) else None
        if nxt is None:
            best = f
            continue
        nf, nu, nfl = nxt
        if nu <= bar and not nfl:
            best = nf
            continue
        if nfl:
            best = max(best, f)
        else:
            best = max(best, f + (nf - f) * (bar - u) / (nu - u) if nu != u else f)
    return float(best), censored


def pareto_cf(cells, groups, strata, rules=PARETO_RULES, n_boot=N_BOOT, seed=0):
    """cells: [{"fraction", "english": per-doc dCE, "flagged": bool}] of one order. Returns CF on the upper bound,
    the bootstrap band of CF on the mean curve, and the per-cell summaries."""
    cells = sorted(cells, key=lambda c: c["fraction"])
    if not cells:
        return None
    E = np.stack([np.asarray(c["english"], np.float64) for c in cells])          # cells x docs
    fr = [c["fraction"] for c in cells]
    fl = [bool(c.get("flagged")) for c in cells]
    idx = boot_indices(groups, strata, n_boot, seed)
    bs = np.stack([E[:, ix].mean(1) for ix in idx])                               # boot x cells
    ub = np.percentile(bs, 97.5, axis=0)
    cf, cens = cf_at(fr, ub, rules["bar"], fl)
    cfm, _ = cf_at(fr, E.mean(1), rules["bar"], fl)
    cfb = [cf_at(fr, b, rules["bar"], fl)[0] for b in bs]
    return {"cf": cf, "cf_censored": cens, "cf_mean": cfm,
            "band": [float(np.percentile(cfb, 2.5)), float(np.percentile(cfb, 97.5))],
            "cells": [{"fraction": f, "english_mean": float(E[i].mean()), "english_ub95": float(ub[i]),
                       "flagged": fl[i]} for i, f in enumerate(fr)]}


# ----------------------------------------------------------------------------- CLI

def _docs(d):
    return json.loads((Path(d) / "perdoc.docs.json").read_text())


def _split(spec):
    k, _, a = spec.partition("/")
    return k, a


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("paired", "suff", "pareto", "npz"))
    ap.add_argument("--dir", required=True)
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--set")
    ap.add_argument("--reference-battery", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args(argv)
    d = Path(a.dir)
    if a.cmd == "npz":
        n = to_npz(d / "perdoc.jsonl", d / "battery.perdoc.npz")
        print(f"{n} rows -> {d / 'battery.perdoc.npz'}")
        return 0
    if a.cmd == "paired":
        rows, docs = load_perdoc(d / "perdoc.jsonl"), _docs(d)
        ka, aa = _split(a.a)
        kb, ab = _split(a.b)
        meta = docs[a.set]
        r = paired(doc_dce(rows[(ka, aa, a.set)]), doc_dce(rows[(kb, ab, a.set)]), meta["groups"], meta["strata"],
                   a.n_boot, a.seed)
        print(json.dumps(r, indent=1))
        return 0
    if a.cmd == "suff":
        import amoeba_fn_suff as S
        print(json.dumps(S.summarize(d, a.reference_battery or None), indent=1)[:4000])
        return 0
    import amoeba_fn_pareto as PA
    print(json.dumps(PA.summarize(d), indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

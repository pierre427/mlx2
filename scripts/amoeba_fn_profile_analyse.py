#!/usr/bin/env python3
"""FN neuron topic profiles: which groups the unclaimed / 3+-shared / pair neurons fire for, clusters of them by
profile, and the rare or dead neurons (Pierre 2026-11-18; aMoEba). CPU only; reads scripts/amoeba_fn_profile_capture.py
output(s) and an scripts/amoeba_fn_partition.py run (partition.json + tiles.npz + the maps / T2 it names).

Categories (recomputed from the partition's inputs with its rule and checked against its counts and box tiles):
t2, eng (English map outside T2), shared3 (selected by >= 3 boxes), pair, box, unassigned (the "unclaimed" core
remainder). Per unit and profile group g: mean_g (mean |h| * w per token) and strong_g (fraction of tokens over the
layer threshold). Family (level-1 prefix: prog, sci, lang, sess, english, ...) profiles average a family's groups.
Enrichment e_g = mean_g / median over FAMILIES of the family means (each family counts once, so the ~210 language
groups do not set the baseline); a unit fires for g when e_g >= --enrich and strong_g >= --min-strong; family
enrichment uses the same baseline.
Clusters: spherical k-means on the Hellinger profile sqrt(mean_g / sum mean_g) of a seeded sample, then every unit of
the category assigned to its nearest centroid; a cluster is labelled with its centroid's most enriched groups and
families. Rare / dead (from the exact fp32 totals): never routed (its expert got no token), dead (no strong token at
all and total mean < --dead-rel x the layer's median total mean), rare (not dead, max over groups of strong_g <
--rare).

Writes OUT/report.json, OUT/units.npz (cat, rare, dead, never_routed [L, n]; per category: top1 group, top1
enrichment, enriched-group count, top1 family, cluster id, all [L, n] with -1 / 0 outside the category) and
OUT/clusters_<category>.json.

Usage: amoeba_fn_profile_analyse.py --capture CAP [--capture CAP2] --partition PART --out DIR
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

FOCUS = ("unassigned", "shared3", "pair")


class Profiles:
    """One or more captures of the same model / layout / thresholds, concatenated along groups."""

    def __init__(self, dirs):
        self.caps = [PL.Capture(d) for d in dirs]
        c0 = self.caps[0]
        for c in self.caps[1:]:
            if (c.rep["fingerprint"], c.layers, c.E, c.F, c.rep["thresholds"]["sha12"]) != \
                    (c0.rep["fingerprint"], c0.layers, c0.E, c0.F, c0.rep["thresholds"]["sha12"]):
                raise SystemExit(f"{c.d}: another model, layout or threshold set than {c0.d} (pass --thresholds)")
        self.layers, self.E, self.F, self.n = c0.layers, c0.E, c0.F, c0.n
        ids, seen = [], set()
        for k, c in enumerate(self.caps):
            for g in c.groups:
                gid = g if g not in seen else f"{g}@{k}"
                seen.add(gid)
                ids.append(gid)
        self.groups = ids
        self.tokens = np.concatenate([c.tokens for c in self.caps])
        self.family = [g.split(".")[0].split("@")[0] for g in ids]
        self.fams = sorted(set(self.family))
        self.fam_idx = np.array([self.fams.index(f) for f in self.family])

    def layer(self, li, cols=None):
        """(mean [G, k], strong [G, k]) float32 for layer index li (columns cols, or all)."""
        m = np.concatenate([c.profile_layer("mean", li, cols) for c in self.caps])
        s = np.concatenate([c.profile_layer("strong", li, cols) for c in self.caps])
        return m, s

    def totals(self):
        s = sum(c.totals()["sum"].astype(np.float64) for c in self.caps)
        st = sum(c.totals()["strong"].astype(np.float64) for c in self.caps)
        tok = sum(float(c.totals()["tokens"]) for c in self.caps)
        rout = sum(np.asarray(c.routed(), np.float64).sum(0) for c in self.caps)
        return s / max(tok, 1.0), st / max(tok, 1.0), rout


def family_profile(m, fam_idx, nfam):
    out = np.zeros((nfam, m.shape[1]), np.float32)
    for i in range(nfam):
        sel = fam_idx == i
        if sel.any():
            out[i] = m[sel].mean(0)
    return out


def enrichment(m, fam_idx, nfam, eps=1e-12):
    """(group enrichment [G, k], family enrichment [F, k]) against the median FAMILY mean: each family (lang, prog,
    sci, sess, english, ...) counts once, so the ~210 language groups do not set the baseline (a median over groups
    would be a language value and make every general-English neuron look enriched in every English group)."""
    fp = family_profile(m, fam_idx, nfam)
    base = np.median(fp, axis=0)[None, :] + eps
    return m / base, fp / base


def hellinger(m):
    tot = m.sum(0, keepdims=True)
    p = np.where(tot > 0, m / np.maximum(tot, 1e-30), 0.0)
    f = np.sqrt(p).T
    nrm = np.linalg.norm(f, axis=1, keepdims=True)
    return np.where(nrm > 0, f / np.maximum(nrm, 1e-30), 0.0).astype(np.float32)


def spherical_kmeans(X, k, seed, iters=30):
    """k-means on unit vectors by cosine (k-means++ init on the sample)."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    k = min(k, n)
    C = np.empty((k, X.shape[1]), np.float32)
    C[0] = X[rng.integers(n)]
    d = 1 - X @ C[0]
    for i in range(1, k):
        p = np.maximum(d, 0)
        p = p / p.sum() if p.sum() > 0 else None
        C[i] = X[rng.choice(n, p=p)]
        d = np.minimum(d, 1 - X @ C[i])
    lab = np.zeros(n, np.int64)
    for _ in range(iters):
        lab_new = np.argmax(X @ C.T, axis=1)
        if _ and np.array_equal(lab_new, lab):
            break
        lab = lab_new
        for i in range(k):
            sel = X[lab == i]
            if len(sel):
                v = sel.sum(0)
                C[i] = v / max(np.linalg.norm(v), 1e-30)
            else:
                C[i] = X[rng.integers(n)]
    return C, lab


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="append", required=True)
    ap.add_argument("--partition", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--enrich", type=float, default=3.0)
    ap.add_argument("--min-strong", type=float, default=1e-4)
    ap.add_argument("--rare", type=float, default=1e-5)
    ap.add_argument("--dead-rel", type=float, default=1e-3)
    ap.add_argument("--clusters", type=int, default=64, help="k for the unassigned neurons")
    ap.add_argument("--clusters-shared", type=int, default=24, help="k for the shared3 and pair neurons")
    ap.add_argument("--cluster-sample", type=int, default=300_000)
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {out}")
    out.mkdir(parents=True, exist_ok=True)
    log = lambda *m: print(*m, flush=True)    # noqa: E731
    P = Profiles(a.capture)
    part = PL.partition_categories(a.partition, log=log)
    if part["layers"] != P.layers or part["n"] != P.n:
        raise SystemExit("capture and partition layouts differ")
    if part["partition"].get("fingerprint") and part["partition"]["fingerprint"] != P.caps[0].rep["fingerprint"]:
        raise SystemExit("capture and partition are of different models")
    cat, claims, names = part["cat"], part["claims"], part["names"]
    L, n, G = len(P.layers), P.n, len(P.groups)
    gidx = {g: i for i, g in enumerate(P.groups)}
    alias = {}
    for b in names:
        alias[b] = gidx.get(b, gidx.get("lang." + b))
    bmat = np.zeros((len(names), G), np.float32)                 # box -> its profile group
    for i, b in enumerate(names):
        if alias[b] is not None:
            bmat[i, alias[b]] = 1

    # ------------------------------------------------------------------ rare / dead
    tmean, tstr, rout = P.totals()
    never = np.repeat(rout <= 0, P.F, axis=1)
    dead = np.zeros((L, n), bool)
    for li in range(L):
        ok = ~never[li]
        med = np.median(tmean[li][ok]) if ok.any() else 0.0
        dead[li] = ok & (tstr[li] == 0) & (tmean[li] < a.dead_rel * med)
    maxstrong = np.zeros((L, n), np.float32)
    per = {c: {k: np.full((L, n), -1, np.int16) for k in ("top1", "fam1", "cluster")} for c in FOCUS}
    for c in FOCUS:
        per[c]["e1"] = np.zeros((L, n), np.float16)
        per[c]["n_enriched"] = np.zeros((L, n), np.int16)
    group_counts = {c: np.zeros(G, np.int64) for c in FOCUS}
    fam_counts = {c: np.zeros(len(P.fams), np.int64) for c in FOCUS}
    claimed_share, top1_claimed = {c: [] for c in ("shared3", "pair")}, {c: [] for c in ("shared3", "pair")}
    pair_prof = {}
    shared3_by_k = {}
    rng = np.random.default_rng(a.seed)
    tot_focus = {c: int((cat == PL.CATS[c]).sum()) for c in FOCUS}
    samples = {c: [] for c in FOCUS}
    for li in range(L):
        m, s = P.layer(li)
        maxstrong[li] = s.max(0)
        for c in FOCUS:
            cols = np.nonzero((cat[li] == PL.CATS[c]) & ~never[li] & ~dead[li])[0]
            if not cols.size:
                continue
            mc, sc = m[:, cols], s[:, cols]
            e, fe = enrichment(mc, P.fam_idx, len(P.fams))
            fired = (e >= a.enrich) & (sc >= a.min_strong)
            t1 = np.argmax(e, axis=0)
            per[c]["top1"][li, cols] = t1
            per[c]["e1"][li, cols] = np.minimum(e[t1, np.arange(cols.size)], 6e4)
            per[c]["n_enriched"][li, cols] = fired.sum(0)
            hit = fired[t1, np.arange(cols.size)]
            np.add.at(group_counts[c], t1[hit], 1)
            f1 = np.argmax(fe, axis=0)
            per[c]["fam1"][li, cols] = f1
            np.add.at(fam_counts[c], f1[fe[f1, np.arange(cols.size)] >= a.enrich], 1)
            take = rng.random(cols.size) < min(1.0, a.cluster_sample / max(1, tot_focus[c]))
            if take.any():
                samples[c].append(hellinger(mc[:, take]))
            if c in ("shared3", "pair"):
                p = (mc / np.maximum(mc.sum(0), 1e-30)).T                          # [k, G]
                cl = claims[li, cols]
                bits = ((cl[:, None] >> np.arange(len(names))[None, :]) & 1).astype(bool)   # [k, boxes]
                ind = (bits.astype(np.float32) @ bmat) > 0                            # claimed groups [k, G]
                has = ind.any(1)
                mass = (p * ind).sum(1)
                in1 = ind[np.arange(cols.size), t1]
                claimed_share[c].append(np.stack([mass[has], ind.sum(1)[has]], 1))
                top1_claimed[c].append(in1[has])
                if c == "pair":
                    for v in np.unique(cl[has]):
                        sel = has & (cl == v)
                        bb = [names[i] for i in range(len(names)) if (int(v) >> i) & 1]
                        acc = pair_prof.setdefault("__".join(bb), [np.zeros(G), 0, 0])
                        acc[0] += p[sel].sum(0)
                        acc[1] += int(sel.sum())
                        acc[2] += int(in1[sel].sum())
                else:
                    kk = np.minimum(bits.sum(1), 6)
                    nf = fired.sum(0)
                    for v in np.unique(kk[has]):
                        sel = has & (kk == v)
                        acc = shared3_by_k.setdefault(int(v), [0, 0, 0.0, 0])
                        acc[0] += int(sel.sum())
                        acc[1] += int(in1[sel].sum())
                        acc[2] += float(mass[sel].sum())
                        acc[3] += int(nf[sel].sum())
        log(f"layer {P.layers[li]} profiled")
    rare = ~never & ~dead & (maxstrong < a.rare)

    # ------------------------------------------------------------------ clusters
    cents = {}
    for c in FOCUS:
        if not samples[c]:
            continue
        X = np.concatenate(samples[c])
        k = a.clusters if c == "unassigned" else a.clusters_shared
        C, _ = spherical_kmeans(X, k, a.seed)
        cents[c] = C
    sizes = {c: np.zeros(len(cents[c]), np.int64) for c in cents}
    cl_mean = {c: np.zeros((len(cents[c]), G)) for c in cents}
    cl_layers = {c: np.zeros((len(cents[c]), L), np.int64) for c in cents}
    cl_experts = {c: np.zeros((len(cents[c]), L * P.E), bool) for c in cents}
    cl_rare = {c: np.zeros(len(cents[c]), np.int64) for c in cents}
    for li in range(L):
        m, _ = P.layer(li)
        for c in cents:
            cols = np.nonzero((cat[li] == PL.CATS[c]) & ~never[li] & ~dead[li])[0]
            if not cols.size:
                continue
            H = hellinger(m[:, cols])
            lab = np.argmax(H @ cents[c].T, axis=1)
            per[c]["cluster"][li, cols] = lab
            np.add.at(sizes[c], lab, 1)
            np.add.at(cl_layers[c][:, li], lab, 1)
            np.add.at(cl_rare[c], lab[rare[li, cols]], 1)
            mc = m[:, cols]
            for i in np.unique(lab):
                sel = lab == i
                cl_mean[c][i] += mc[:, sel].sum(1)
                cl_experts[c][i, li * P.E + cols[sel] // P.F] = True

    def label(prof):
        """Top groups and families of a mean profile, as enrichment over the median family (as per unit)."""
        fp = family_profile(prof[:, None], P.fam_idx, len(P.fams))[:, 0]
        base = np.median(fp) + 1e-30
        o = np.argsort(-prof)[: a.top]
        fo = np.argsort(-fp)[:3]
        return ({P.groups[i]: round(float(prof[i] / base), 2) for i in o},
                {P.fams[i]: round(float(fp[i] / base), 2) for i in fo})

    for c in cents:
        rows = []
        for i in np.argsort(-sizes[c]):
            if not sizes[c][i]:
                continue
            gl, fl = label(cl_mean[c][i] / sizes[c][i])
            rows.append({"cluster": int(i), "units": int(sizes[c][i]), "top_groups_enrichment": gl,
                         "top_families_enrichment": fl, "experts": int(cl_experts[c][i].sum()),
                         "rare_units": int(cl_rare[c][i]),
                         "layers": {str(P.layers[l]): int(v) for l, v in enumerate(cl_layers[c][i]) if v}})
        PL.write_json(out / f"clusters_{c}.json", {"category": c, "k": len(cents[c]), "clusters": rows})

    # ------------------------------------------------------------------ report
    cnames = {v: k for k, v in PL.CATS.items()}
    dr = {}
    for cv, cn in cnames.items():
        sel = cat == cv
        dr[cn] = {"units": int(sel.sum()), "never_routed": int((never & sel).sum()), "dead": int((dead & sel).sum()),
                  "rare": int((rare & sel).sum())}
    rep = {"schema": "mlx2.amoeba-fn-profile-analysis.v1", "captures": [str(c.d) for c in P.caps],
           "checkpoint": P.caps[0].rep["checkpoint"], "capture_revs": [c.rep["mlx2_source_revision"] for c in P.caps],
           "partition": str(a.partition), "params": vars(a), "groups": len(P.groups), "families": P.fams,
           "box_groups": {b: (P.groups[alias[b]] if alias[b] is not None else None) for b in names},
           "rare_dead_by_category": dr,
           "rare_dead_by_layer": {str(P.layers[l]): {"dead": int(dead[l].sum()), "rare": int(rare[l].sum()),
                                                      "never_routed": int(never[l].sum())} for l in range(L)},
           "fires_for_top1": {c: {P.groups[i]: int(v) for i in np.argsort(-group_counts[c])[:40]
                                  if (v := group_counts[c][i])} for c in FOCUS},
           "family_top1": {c: {P.fams[i]: int(v) for i, v in enumerate(fam_counts[c]) if v} for c in FOCUS},
           "enriched_group_count_hist": {c: np.bincount(np.minimum(per[c]["n_enriched"][cat == PL.CATS[c]], 20),
                                                        minlength=21).tolist() for c in FOCUS},
           "claimed": {},
           "shared3_by_claims": {str(k): {"units": v[0], "top1_in_claimed": v[1] / max(1, v[0]),
                                          "mean_claimed_mass": v[2] / max(1, v[0]),
                                          "mean_enriched_groups": v[3] / max(1, v[0])}
                                 for k, v in sorted(shared3_by_k.items())},
           "pairs": {}}
    for c in ("shared3", "pair"):
        cs = np.concatenate(claimed_share[c]) if claimed_share[c] else np.zeros((0, 2))
        t1c = np.concatenate(top1_claimed[c]) if top1_claimed[c] else np.zeros(0, bool)
        rep["claimed"][c] = {"units_with_box_groups": int(cs.shape[0]),
                             "top1_in_claimed": float(t1c.mean()) if t1c.size else None,
                             "median_claimed_mass": float(np.median(cs[:, 0])) if cs.size else None,
                             "median_uniform_mass": float(np.median(cs[:, 1] / G)) if cs.size else None,
                             "note": "claimed mass = share of the profile (mean_g / sum) on the claiming boxes' "
                                     "groups; uniform = the same share for a flat profile"}
    for key, (s, cnt, hit) in sorted(pair_prof.items(), key=lambda kv: -kv[1][1])[:60]:
        gl, fl = label(s / max(1, cnt))
        rep["pairs"][key] = {"units_profiled": cnt, "top1_in_pair": hit / max(1, cnt), "top_groups": gl,
                             "top_families": fl}
    arrays = {"cat": cat, "rare": rare, "dead": dead, "never_routed": never, "layers": np.array(P.layers),
              "groups": np.array(P.groups)}
    for c in FOCUS:
        for k, v in per[c].items():
            arrays[f"{c}_{k}"] = v
    np.savez_compressed(out / "units.npz", **arrays)
    PL.write_json(out / "report.json", rep)
    log(json.dumps({"rare_dead": dr, "clusters": {c: len(cents[c]) for c in cents}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

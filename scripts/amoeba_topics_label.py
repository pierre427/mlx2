#!/usr/bin/env python3
"""Label knowledge clusters with taxonomy topics, against controls; sub-cluster them (CPU, no model).

DESIGN: docs/research/amoeba-topics/DESIGN.md. Inputs: a topic capture (amoeba_topics_capture.py), the
host's clusters (pool_clusters.npz: pool [P,2] (layer, neuron), labels [P]) or --form-clusters K (spherical
k-means on fit-split profiles of the knowledge pool, for models without an atlas), and the core mask.

Per text s and neuron n: rel[s,n] = r[s,n] / mean_fit r[.,n]. A group's score on s = mean rel over its
neurons. For a taxonomy node X: positives = texts labelled X or a descendant; contrast = the texts of X's
siblings (other children of X's parent), or every other text for a root (or when siblings are too few).

Gates for "cluster c carries X" (choice on fit, confirmation on held):
  fit:  d >= 0.8;  d >= twin95 + 0.5 (twins: --twins random sets, same per-layer counts, from the pool);
        shuffled-label z-test, Bonferroni over every (group, node) test, p < 0.01
  held: d >= 0.8 and d > twin95 (same twins), same sign
The hierarchical label descends from the roots into the best passing child; the label is the deepest
passing node. Up to 3 passing roots are kept (multi-topic clusters).

Subclusters: spherical k-means of a cluster's neurons on their fit topic profile (mean rel per tested
leaf), k in 2..6, the largest k whose held profiles agree >= 0.8 with the fit assignment; each subcluster
is labelled by the same test with twins drawn inside its parent cluster.

Per-neuron topic maps (for core tags and stubs): d per neuron per tested node, fit and held, saved to
OUT_ARRAYS/neuron_d_<split>.npy float16 [nodes, L, N].
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import amoeba_topics_lib as T

MIN_SPLIT = 16


def spherical_kmeans(X, k, iters=40, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(X.shape[0], k, replace=False)].copy()
    lab = None
    for _ in range(iters):
        new = (X @ C.T).argmax(1)
        if lab is not None and (new == lab).mean() > 0.999:
            lab = new
            break
        lab = new
        for j in range(k):
            m = lab == j
            if m.any():
                c = X[m].sum(0)
                C[j] = c / max(np.linalg.norm(c), 1e-8)
            else:
                C[j] = X[rng.integers(X.shape[0])]
    return lab, C


def unit_rows(X):
    X = X - X.mean(1, keepdims=True)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)


def norm_p(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


class Nodes:
    """Tested taxonomy nodes with their positive / contrast text sets per split.
    contrast "own": siblings for a non-root (all other texts when siblings are too few), all other
    texts for a root. Every non-root is also tested against "global" (all texts not in the node)."""

    SPLITS = ("fit", "held", "fitA", "fitB")

    def __init__(self, tax, items, groups=None):
        self.tax = tax
        mem = T.topic_membership(tax, items)
        split = np.array([it["split"] for it in items])
        # groups (source file / bundle / family): the effective sample unit of the null, and the key that
        # halves the fit split (fitA / fitB) for fit-internal model selection (subcluster k)
        self.group = np.array(groups if groups is not None else [it["id"] for it in items])
        half = np.array([T.stable_seed("half:" + g) % 2 for g in self.group])
        self.split_of = {"fit": split == "fit", "held": split == "held",
                         "fitA": (split == "fit") & (half == 0), "fitB": (split == "fit") & (half == 1)}
        self.S = len(items)
        self.split = split
        nodes = sorted({t for s in mem for t in s})
        self.pos, self.neg, self.contrast = {}, {}, {}
        for x in nodes:
            par = tax.nodes[x].get("parent")
            inx = np.array([x in m for m in mem])
            if any((inx & (split == sp)).sum() < MIN_SPLIT for sp in ("fit", "held")):
                continue
            glob = ~inx
            own, kind = glob, "global"
            if par:
                inp = np.array([par in m for m in mem])
                sib = inp & ~inx
                if all((sib & (split == sp)).sum() >= MIN_SPLIT for sp in ("fit", "held")):
                    own, kind = sib, "siblings"
            self.contrast[x] = kind
            for sp in self.SPLITS:
                sm = self.split_of[sp]
                self.pos[(x, sp)] = np.where(inx & sm)[0]
                self.neg[(x, sp, "own")] = np.where(own & sm)[0]
                self.neg[(x, sp, "global")] = np.where(glob & sm)[0]
                self.neg[(x, sp)] = self.neg[(x, sp, "own")]
        self.ids = sorted(self.contrast)

    def n_groups(self, idx):
        return max(1, len(set(self.group[idx].tolist())))

    def children(self, x):
        return [c for c in self.tax.children.get(x, []) if c in self.contrast]

    def is_root(self, x):
        return not self.tax.nodes[x].get("parent")


def group_scores(cap, L, groups, fit_mask, log):
    """groups: list of {layer: cols}. Returns scores [S, G] (mean rel over each group's neurons)."""
    S = None
    acc, cnt = None, np.zeros(len(groups))
    by_l = {}
    for g, gm in enumerate(groups):
        for l, cols in gm.items():
            by_l.setdefault(l, []).append((g, cols))
    t0 = time.time()
    for l in range(L):
        if l not in by_l:
            continue
        R = T.load_resp(cap, l)          # groups index real columns only
        if acc is None:
            S = R.shape[0]
            acc = np.zeros((S, len(groups)), dtype=np.float64)
        R /= np.maximum(R[fit_mask].mean(0, keepdims=True), 1e-6)
        RT = np.ascontiguousarray(R.T)
        del R
        for g, cols in by_l[l]:
            acc[:, g] += RT[cols].sum(0)
            cnt[g] += len(cols)
        if l % 16 == 15:
            log(f"    group scores through L{l} ({time.time() - t0:.0f}s)")
    return acc / np.maximum(cnt, 1)[None, :]


def neuron_node_d(cap, L, N, nodes, fit_mask, out_dir, log):
    """d per neuron per node, per split -> float16 [nodes, L, N] memmaps.
    Exact and fast: texts with the same (split, node-membership) pattern form an "atom"; per layer the
    per-atom sums and sums of squares are taken once (O(S x N)), and every node's positive and contrast
    statistics are small combinations of atoms (K x atoms @ atoms x N)."""
    K = len(nodes.ids)
    outs = {sp: np.lib.format.open_memmap(out_dir / f"neuron_d_{sp}.npy", mode="w+", dtype=np.float16, shape=(K, L, N))
            for sp in nodes.SPLITS}
    S = nodes.S
    # per split: atom id of each text in that split, and the K x atoms count matrices of pos / neg
    plans = {}
    for sp in nodes.SPLITS:
        idx_pos = {x: set(nodes.pos[(x, sp)].tolist()) for x in nodes.ids}
        idx_neg = {x: set(nodes.neg[(x, sp)].tolist()) for x in nodes.ids}
        texts = np.where(nodes.split_of[sp])[0]
        sig = {}
        atom_of = np.full(S, -1, np.int64)
        for i in texts:
            key = (tuple(k for k, x in enumerate(nodes.ids) if i in idx_pos[x]),
                   tuple(k for k, x in enumerate(nodes.ids) if i in idx_neg[x]))
            atom_of[i] = sig.setdefault(key, len(sig))
        nA = len(sig)
        Pm = np.zeros((K, nA), np.float32)
        Qm = np.zeros((K, nA), np.float32)
        for key, aid in sig.items():
            Pm[list(key[0]), aid] = 1
            Qm[list(key[1]), aid] = 1
        order = texts[np.argsort(atom_of[texts], kind="stable")]
        bounds = np.searchsorted(atom_of[order], np.arange(nA))
        cnt = np.bincount(atom_of[texts], minlength=nA).astype(np.float32)
        na, nb = Pm @ cnt, Qm @ cnt
        plans[sp] = (order, bounds, Pm, Qm, na, nb)
        log(f"    {sp}: {len(texts)} texts in {nA} atoms")
    t0 = time.time()
    for l in range(L):
        R = T.load_resp(cap, l, N)
        R /= np.maximum(R[fit_mask].mean(0, keepdims=True), 1e-6)
        for sp, (order, bounds, Pm, Qm, na, nb) in plans.items():
            Ro = R[order]
            As = np.add.reduceat(Ro, bounds, axis=0)                       # [atoms, N]
            Aq = np.add.reduceat(Ro * Ro, bounds, axis=0)
            sa, sq_a = Pm @ As, Pm @ Aq
            sb, sq_b = Qm @ As, Qm @ Aq
            ma, mb = sa / np.maximum(na, 1)[:, None], sb / np.maximum(nb, 1)[:, None]
            va = np.maximum(sq_a - na[:, None] * ma * ma, 0) / np.maximum(na - 1, 1)[:, None]
            vb = np.maximum(sq_b - nb[:, None] * mb * mb, 0) / np.maximum(nb - 1, 1)[:, None]
            s_ = np.sqrt(((na - 1)[:, None] * va + (nb - 1)[:, None] * vb) / np.maximum(na + nb - 2, 1)[:, None])
            outs[sp][:, l, :] = np.clip((ma - mb) / np.maximum(s_, 1e-6), -60000, 60000).astype(np.float16)
        if l % 8 == 7:
            log(f"    neuron d through L{l} ({time.time() - t0:.0f}s)")
    for o in outs.values():
        o.flush()
    return outs


def gate(nodes, scores, twin_scores, x, which, n_tests):
    r = {}
    for sp in ("fit", "held"):
        pos, neg = nodes.pos[(x, sp)], nodes.neg[(x, sp, which)]
        d = T.cohen_d(scores[pos], scores[neg])
        td = T.cohen_d_rows(twin_scores, pos, neg) if twin_scores is not None and twin_scores.shape[1] else np.array([0.0])
        r[f"d_{sp}"] = round(d, 3)
        r[f"auc_{sp}"] = round(T.auc(scores[pos], scores[neg]), 3)
        r[f"twin95_{sp}"] = round(float(np.percentile(td, 95)), 3)
        r[f"n_{sp}"] = [int(len(pos)), int(len(neg))]
        if sp == "fit":
            # label-shuffle null of d ~ N(0, se), se from the number of distinct groups (texts from one
            # file are correlated, so counting texts would understate the null spread)
            ga, gb = nodes.n_groups(pos), nodes.n_groups(neg)
            z = d / math.sqrt(1.0 / ga + 1.0 / gb)
            r["groups_fit"] = [ga, gb]
            r["perm_z"] = round(z, 2)
            r["perm_p_bonf"] = min(1.0, norm_p(z) * n_tests)
    fit_ok = r["d_fit"] >= 0.8 and r["d_fit"] >= r["twin95_fit"] + 0.5 and r["perm_p_bonf"] < 0.01
    held_ok = r["d_held"] >= 0.8 and r["d_held"] > r["twin95_held"]
    r["fit_ok"] = bool(fit_ok)
    r["pass"] = bool(fit_ok and held_ok)
    if not fit_ok:
        r["fail"] = [w for w, bad in (("d_fit<0.8", r["d_fit"] < 0.8), ("not above twins", r["d_fit"] < r["twin95_fit"] + 0.5),
                                      ("shuffled-label p", r["perm_p_bonf"] >= 0.01)) if bad]
    elif not held_ok:
        r["fail"] = ["held split did not confirm"]
    return r


def node_profile(nodes, scores, twin_scores, x, n_tests, rng, n_perm):
    """Gates for one group at node x: its own contrast (siblings / all for a root) and, for a non-root,
    the global contrast too. Passing nodes also get an empirical label-shuffle p (n_perm)."""
    r = {"node": x, "contrast": nodes.contrast[x], "own": gate(nodes, scores, twin_scores, x, "own", n_tests)}
    if not nodes.is_root(x):
        r["global"] = gate(nodes, scores, twin_scores, x, "global", n_tests)
    r["pass"] = r["own"]["pass"] and r.get("global", {"pass": True})["pass"]
    for k in ("d_fit", "d_held", "twin95_fit", "twin95_held", "auc_held", "perm_z", "perm_p_bonf"):
        r[k] = r["own"][k]
    if r["pass"] and n_perm:
        pos, neg = nodes.pos[(x, "fit")], nodes.neg[(x, "fit", "own")]
        null = T.perm_null_d(scores, pos, neg, n_perm, rng)
        r["perm_p_emp_textlevel"] = float((np.sum(null >= r["own"]["d_fit"]) + 1) / (n_perm + 1))
    if not r["pass"]:
        r["fail"] = (["own: " + f for f in r["own"].get("fail", [])] +
                     ["global: " + f for f in r.get("global", {}).get("fail", [])])
    return r


def hier_label(nodes, scores, twin_scores, n_tests, rng, n_perm, max_labels=3):
    """Every tested node is gated; the labels are the finest passing nodes (a passing node with a
    passing descendant is replaced by that descendant), ranked by held d, at most max_labels."""
    tested = {x: node_profile(nodes, scores, twin_scores, x, n_tests, rng, n_perm) for x in nodes.ids}
    passing = {x for x, r in tested.items() if r["pass"]}
    finest = [x for x in passing if not any(y != x and y.startswith(x + ".") for y in passing)]
    finest.sort(key=lambda x: -tested[x]["d_fit"])           # chosen on fit; held only confirms
    labels = []
    for x in finest[:max_labels]:
        path = [a for a in reversed(nodes.tax.ancestors(x))]
        labels.append({"topic": x, "path": path, "path_passing": [a for a in path if a in passing], "gates": tested[x]})
    best_fail = None
    if not labels:
        best_fail = max(tested.values(), key=lambda r: min(r["d_fit"], r.get("global", r["own"])["d_fit"]))
        best_fail = {k: v for k, v in best_fail.items()}
    return labels, best_fail, tested


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--taxonomy", default=str(T.TAXO))
    ap.add_argument("--out", required=True, help="small JSON results (git)")
    ap.add_argument("--arrays", required=True, help="large arrays (research-artifacts)")
    ap.add_argument("--clusters", default="", help="pool_clusters.npz (pool, labels)")
    ap.add_argument("--core", default="", help="npz with core [L,N] (>0 kept) or box-registry core json")
    ap.add_argument("--form-clusters", type=int, default=0)
    ap.add_argument("--no-core", action="store_true", help="map mode: no English core, the pool is every unit")
    ap.add_argument("--twins", type=int, default=32)
    ap.add_argument("--perm", type=int, default=2000)
    ap.add_argument("--sub-kmax", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = T.fresh_dir(args.out)
    arrays = T.fresh_dir(args.arrays)
    log = T.make_log(out)
    rng = np.random.default_rng(args.seed)
    cap = Path(args.capture)
    meta = json.loads((cap / "meta.json").read_text())
    L, N, S = meta["layers"], meta["neurons"], meta["sequences"]
    tax = T.Taxonomy(args.taxonomy)
    capitems = json.loads((cap / "items.json").read_text())
    gmap = {}
    if Path(args.corpus).exists() and Path(args.corpus).stat().st_size:
        gmap = {it["id"]: it["group"] for it in T.load_items(args.corpus)}
    nodes = Nodes(tax, capitems, [gmap.get(it["id"], it["id"]) for it in capitems])
    fit_mask = nodes.split == "fit"
    log(f"{meta['checkpoint']}: {S} texts, {len(nodes.ids)} tested nodes "
        f"({sum(1 for x in nodes.ids if not x.startswith('lang.'))} non-lang)")

    core = None
    if args.core.endswith(".npz"):
        core = np.load(args.core)["core"] > 0
    elif args.core.endswith(".json"):
        rec = json.loads(Path(args.core).read_text())
        core = np.zeros((L, N), bool)
        for u in rec["origin"]["units"]:
            core[u["layer"], u["neurons"]] = True
    if core is None and args.no_core:
        core = np.zeros((L, N), bool)          # map mode: every unit is in the pool
    if core is None:
        raise SystemExit("--core is required (the English core the pool is defined against), or --no-core")
    # ragged captures (carve backend): zero-padded columns past a layer's width are not units; they
    # are excluded exactly as core units are, and reported separately
    padding = np.zeros((L, N), bool)
    if meta.get("widths"):
        padding = np.arange(N)[None, :] >= np.asarray(meta["widths"])[:, None]
        core = core | padding
    pool = np.argwhere(~core)
    pool_by_l = T.by_layer(pool)

    # ------------------------------------------------------------------ per-neuron node d (fit, held)
    log("per-neuron d per node")
    neuron_node_d(cap, L, N, nodes, fit_mask, arrays, log)
    T.write_json(arrays / "neuron_d_nodes.json", nodes.ids)

    # ------------------------------------------------------------------ clusters
    if args.clusters:
        z = np.load(args.clusters)
        cpool, clab = z["pool"], z["labels"].astype(int)
        cl_source = args.clusters
    else:
        log(f"forming {args.form_clusters} clusters on fit-split profiles of the knowledge pool")
        if len(pool) * int(fit_mask.sum()) > 1.5e9:
            raise SystemExit("--form-clusters: pool x fit texts too large for an in-memory k-means (use an atlas)")
        X = np.zeros((len(pool), int(fit_mask.sum())), np.float32)
        off = 0
        for l in range(L):
            cols = pool_by_l.get(l, np.array([], int))
            if not len(cols):
                continue
            F = np.asarray(np.load(cap / f"resp_L{l:02d}.npy", mmap_mode="r")[:, cols], dtype=np.float32)[fit_mask]
            Z = F / np.maximum(F.mean(0, keepdims=True), 1e-8)
            Z = Z - Z.mean(0, keepdims=True)
            Xl = Z.T
            X[off:off + len(cols)] = Xl / np.maximum(np.linalg.norm(Xl, axis=1, keepdims=True), 1e-8)
            off += len(cols)
        cpool = np.concatenate([np.stack([np.full(len(pool_by_l[l]), l), pool_by_l[l]], 1) for l in sorted(pool_by_l)])
        clab, _ = spherical_kmeans(X, args.form_clusters, seed=args.seed)
        cl_source = f"formed here: spherical k-means k={args.form_clusters} on fit-split topic-probe profiles"
        np.savez_compressed(out / "formed_clusters.npz", pool=cpool.astype(np.int32), labels=clab.astype(np.int16))
        del X
    inside = core[cpool[:, 0], cpool[:, 1]]
    if inside.any():
        raise SystemExit(f"{int(inside.sum())} cluster neurons lie inside --core: clusters and core disagree")
    K = int(clab.max()) + 1
    clusters = [T.by_layer(cpool[clab == j]) for j in range(K)]
    groups, gmeta = [], []
    for j, cm in enumerate(clusters):
        groups.append(cm)
        gmeta.append(("cluster", j, -1))
        for t, tw in enumerate(T.twin_sets(cm, pool_by_l, args.twins, rng)):
            groups.append(tw)
            gmeta.append(("twin", j, t))
    log(f"{K} clusters, {len(groups)} groups with twins; scoring")
    G = group_scores(cap, L, groups, fit_mask, log)
    np.save(arrays / "cluster_group_scores.npy", G.astype(np.float32))
    n_tests = K * sum(1 if nodes.is_root(x) else 2 for x in nodes.ids)
    results = []
    for j in range(K):
        gi = [i for i, m in enumerate(gmeta) if m[1] == j]
        sc = G[:, gi[0]]
        tw = G[:, gi[1:]]
        labels, best_fail, tested = hier_label(nodes, sc, tw, n_tests, rng, args.perm)
        size = int(sum(len(c) for c in clusters[j].values()))
        layers = sorted(clusters[j])
        results.append({"id": f"pool:C{j}", "neurons": size,
                        "layers": {"min": layers[0], "max": layers[-1],
                                   "median": int(np.median(np.concatenate([[l] * len(c) for l, c in clusters[j].items()])))},
                        "labels": labels, "status": "labelled" if labels else "unlabelled",
                        "best_failed_candidate": best_fail,
                        "top_nodes_fit": sorted(({"node": k, "d_fit": v["d_fit"], "d_held": v["d_held"],
                                                  "twin95_fit": v["twin95_fit"], "pass": v["pass"],
                                                  "fit_ok": v.get("fit_ok"), "perm_p_bonf": v.get("perm_p_bonf")}
                                                 for k, v in tested.items()), key=lambda r: -r["d_fit"])[:8]})
        lab = ",".join(l["topic"] for l in labels) or "-"
        log(f"  C{j} ({size}): {lab}")

    # ------------------------------------------------------------------ subclusters
    log("subclusters")
    leafish = [x for x in nodes.ids if not nodes.children(x)
               and all(len(nodes.pos[(x, h)]) >= MIN_SPLIT // 2 for h in ("fitA", "fitB"))]
    nd_fit = np.load(arrays / "neuron_d_fit.npy", mmap_mode="r")
    nd_fa = np.load(arrays / "neuron_d_fitA.npy", mmap_mode="r")
    nd_fb = np.load(arrays / "neuron_d_fitB.npy", mmap_mode="r")
    li = np.array([nodes.ids.index(x) for x in leafish])
    # per-cluster topic profiles, gathered one layer at a time (memmap reads stay contiguous)
    prof_f = {j: [] for j in range(K)}
    prof_a = {j: [] for j in range(K)}
    prof_b = {j: [] for j in range(K)}
    for l in range(L):
        bf = np.asarray(nd_fit[:, l, :], np.float32)[li]          # [leaves, N]
        ba = np.asarray(nd_fa[:, l, :], np.float32)[li]
        bb = np.asarray(nd_fb[:, l, :], np.float32)[li]
        for j, cm in enumerate(clusters):
            if l in cm:
                prof_f[j].append(bf[:, cm[l]].T)
                prof_a[j].append(ba[:, cm[l]].T)
                prof_b[j].append(bb[:, cm[l]].T)
    sub_groups, sub_meta = [], []
    for j, cm in enumerate(clusters):
        pairs = np.concatenate([np.stack([np.full(len(c), l), c], 1) for l, c in sorted(cm.items())])
        Xf = unit_rows(np.concatenate(prof_f.pop(j)))               # [n, leaves], same order as pairs
        Xa = unit_rows(np.concatenate(prof_a.pop(j)))
        Xb = unit_rows(np.concatenate(prof_b.pop(j)))
        best = None
        for k in range(2, args.sub_kmax + 1):          # k chosen inside fit: cluster on fitA, check on fitB
            if len(pairs) < 64 * k:
                break
            labA, CA = spherical_kmeans(Xa, k, seed=args.seed + j)
            agree = float(((Xb @ CA.T).argmax(1) == labA).mean())
            if agree < 0.8 or np.bincount(labA, minlength=k).min() < 32:
                continue
            lab, C = spherical_kmeans(Xf, k, seed=args.seed + j)
            if np.bincount(lab, minlength=k).min() >= 32:
                best = (k, lab, agree)
        if best is None:
            results[j]["subclusters"] = {"k": 1, "note": "no k in 2..%d with fitA/fitB agreement >= 0.8" % args.sub_kmax}
            continue
        k, lab, agree = best
        results[j]["subclusters"] = {"k": k, "agreement_fitA_fitB": round(agree, 3), "items": []}
        parent_by_l = cm
        for s in range(k):
            sm = T.by_layer(pairs[lab == s])
            sub_groups.append(sm)
            sub_meta.append(("sub", j, s, -1))
            for t, tw in enumerate(T.twin_sets(sm, parent_by_l, args.twins, rng)):
                sub_groups.append(tw)
                sub_meta.append(("subtwin", j, s, t))
    if sub_groups:
        GS = group_scores(cap, L, sub_groups, fit_mask, log)
        np.save(arrays / "subcluster_group_scores.npy", GS.astype(np.float32))
        n_sub_tests = sum(1 for m in sub_meta if m[0] == "sub") * sum(1 if nodes.is_root(x) else 2 for x in nodes.ids)
        for j in range(K):
            sc = results[j].get("subclusters", {})
            if sc.get("k", 1) < 2:
                continue
            for s in range(sc["k"]):
                gi = [i for i, m in enumerate(sub_meta) if m[1] == j and m[2] == s]
                labels, best_fail, tested = hier_label(nodes, GS[:, gi[0]], GS[:, gi[1:]], n_sub_tests, rng, args.perm)
                n = int(sum(len(c) for c in sub_groups[gi[0]].values()))
                sc["items"].append({"id": f"pool:C{j}.s{s}", "neurons": n, "labels": labels,
                                    "status": "labelled" if labels else "unlabelled",
                                    "twin": "random subsets of the parent cluster, same per-layer counts",
                                    "best_failed_candidate": best_fail})
                log(f"  C{j}.s{s} ({n}): {','.join(l['topic'] for l in labels) or '-'}")
        # save subcluster membership
        np.savez_compressed(out / "subclusters.npz", **{
            f"C{m[1]}.s{m[2]}": np.concatenate([np.stack([np.full(len(c), l), c], 1) for l, c in sorted(g.items())]).astype(np.int32)
            for g, m in zip(sub_groups, sub_meta) if m[0] == "sub"})

    rep = {"schema": "mlx2.amoeba-topic-labels.v1", "checkpoint": meta["checkpoint"],
           "capture": str(cap), "capture_revision": meta["mlx2_source_revision"], "corpus_sha256": meta["corpus_sha256"],
           "mlx2_source_revision": T.git_rev(), "taxonomy_version": tax.version, "clusters_source": cl_source,
           "core_source": args.core or ("none (--no-core)" if args.no_core else ""),
           "core_fraction": float((core & ~padding).sum() / max(1, (~padding).sum())),
           "padding_units_excluded": int(padding.sum()), "tested_nodes": nodes.ids,
           "contrast": nodes.contrast,
           "gates": {"fit": "d>=0.8, d>=twin95+0.5, label-shuffle null as a z-test with se from distinct groups, Bonferroni p<0.01",
                     "held": "d>=0.8 and d>twin95", "twins": args.twins, "perm": args.perm,
                     "bonferroni_tests": n_tests},
           "args": vars(args), "clusters": results,
           "summary": {"labelled": sum(r["status"] == "labelled" for r in results), "clusters": K,
                       "subclustered": sum(r.get("subclusters", {}).get("k", 1) > 1 for r in results)}}
    T.write_json(out / "labels.json", rep)
    log(f"done: {rep['summary']}")


if __name__ == "__main__":
    main()

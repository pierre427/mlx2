#!/usr/bin/env python3
"""Plan the functional check of a topic labelling run (CPU, no model).

Reads labels.json (amoeba_topics_label.py) and its per-neuron d arrays, and writes OUT/plan.json +
OUT/units.npz: every unit to test functionally, its topics, and its twin compartment.

Units:
  cluster      every pool cluster: its labels (deepest topic per label) or, when unlabelled, its best
               failed node, so the 53-of-64 "unclassified" clusters still get a topic-text test
  subcluster   every labelled subcluster
  core_map     for every tested node not carried by a labelled cluster/subcluster: the core neurons with
               per-neuron d >= --core-d on fit AND held (the core-resident map; tested as a removal from
               the full model against a twin drawn from the core, same per-layer counts)
A unit with a labelled topic also gets per-layer-band removal arms (the per-layer strength a stub needs).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import amoeba_topics_lib as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="dir with labels.json (+ formed_clusters.npz / subclusters.npz)")
    ap.add_argument("--arrays", required=True, help="dir with neuron_d_{fit,held}.npy")
    ap.add_argument("--clusters", default="", help="pool_clusters.npz if the labels used an atlas")
    ap.add_argument("--core", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--core-d", type=float, default=1.0)
    ap.add_argument("--min-map", type=int, default=8, help="smallest core map tested")
    ap.add_argument("--max-map", type=int, default=4096, help="cap: strongest neurons by held d")
    ap.add_argument("--bands", type=int, default=8)
    ap.add_argument("--rollup-size", type=int, default=1024,
                    help="child units under this many neurons are also tested merged into one parent-level unit")
    ap.add_argument("--rollup-max", type=int, default=8)
    ap.add_argument("--extra", default="", help="puzzle dir: its plan_extra.json units and units.npz are appended")
    args = ap.parse_args()
    out = T.fresh_dir(args.out)
    lab = json.loads((Path(args.labels) / "labels.json").read_text())
    core = np.load(args.core)["core"] > 0
    L, N = core.shape
    cz = np.load(args.clusters) if args.clusters else np.load(Path(args.labels) / "formed_clusters.npz")
    cpool, clab = cz["pool"], cz["labels"].astype(int)
    subs = np.load(Path(args.labels) / "subclusters.npz") if (Path(args.labels) / "subclusters.npz").exists() else {}
    node_ids = json.loads((Path(args.arrays) / "neuron_d_nodes.json").read_text())
    nd_fit = np.load(Path(args.arrays) / "neuron_d_fit.npy", mmap_mode="r")
    nd_held = np.load(Path(args.arrays) / "neuron_d_held.npy", mmap_mode="r")
    units, arrays = [], {}
    carried = set()
    for c in lab["clusters"]:
        j = int(c["id"].split("C")[1])
        key = f"C{j}"
        arrays[key] = cpool[clab == j].astype(np.int32)
        topics = [l["topic"] for l in c["labels"]]
        cand = None
        if not topics and c.get("best_failed_candidate"):
            bf = c["best_failed_candidate"]
            cand = bf.get("node") if isinstance(bf, dict) else None
        units.append({"unit": c["id"], "kind": "cluster", "key": key, "labelled_topics": topics,
                      "candidate_topic": cand, "tests": topics or ([cand] if cand else []),
                      "add_arm": True, "bands": bool(topics), "twin": "pool"})
        carried.update(topics)
        for s in c.get("subclusters", {}).get("items", []):
            st = [l["topic"] for l in s["labels"]]
            if not st:
                continue
            k = s["id"].split(":")[1]
            arrays[k] = np.asarray(subs[k], np.int32)
            units.append({"unit": s["id"], "kind": "subcluster", "key": k, "labelled_topics": st, "candidate_topic": None,
                          "tests": st, "add_arm": True, "bands": True, "twin": "parent", "parent_key": key})
            carried.update(st)
    for k, x in enumerate(node_ids):          # carried topics get a core map too: used only if no unit passes
        df = np.asarray(nd_fit[k], np.float32)
        dh = np.asarray(nd_held[k], np.float32)
        # selected on FIT only (held texts are what the functional check measures); the count of core
        # neurons as strongly selective in the opposite direction estimates the chance count
        sel = core & (df >= args.core_d)
        n = int(sel.sum())
        n_chance = int((core & (df <= -args.core_d)).sum())
        held_conf = round(float((dh[sel] >= args.core_d).mean()), 3) if n else None
        if n < max(args.min_map, 2 * n_chance):
            units.append({"unit": f"core:{x}", "kind": "core_map", "key": None, "labelled_topics": [], "topic": x,
                          "tests": [], "n_selective": n, "n_chance": n_chance, "held_confirm_frac": held_conf,
                          "note": f"selective core neurons ({n}) under max({args.min_map}, 2 x chance {n_chance})"})
            continue
        pairs = np.argwhere(sel)
        if n > args.max_map:
            pairs = pairs[np.argsort(-df[sel], kind="stable")[:args.max_map]]
        key = f"core_{x}"
        arrays[key] = pairs.astype(np.int32)
        units.append({"unit": f"core:{x}", "kind": "core_map", "key": key, "labelled_topics": [], "topic": x,
                      "tests": [x], "n_selective": n, "n_chance": n_chance, "held_confirm_frac": held_conf,
                      "n_tested": int(len(pairs)), "add_arm": False, "bands": True, "twin": "core",
                      "selection": f"core neurons with fit d >= {args.core_d}, strongest {args.max_map} by fit d"})
    # roll-up (Pierre 2026-11-14): small fine-topic units under one parent are also tested as ONE merged,
    # uncarved union unit; the registry keeps a member in it only if each member holds its own score
    # inside the merged unit against its standalone unit, and the off-topic control does not get worse
    tax = T.Taxonomy()
    small = {}
    for u in units:
        if u["kind"] in ("cluster", "subcluster") and u.get("labelled_topics") and len(arrays[u["key"]]) < args.rollup_size:
            for t in u["labelled_topics"]:
                par = tax.nodes[t].get("parent")
                if par:
                    small.setdefault(par, []).append((t, u))
    for par, mem in sorted(small.items()):
        if len({t for t, _ in mem}) < 2:
            continue
        if len({t for t, _ in mem}) > args.rollup_max:
            continue                    # too many members to test in one window (and unlikely to merge cleanly)
        key = "rollup_" + par
        arrays[key] = np.unique(np.concatenate([arrays[u["key"]] for _, u in mem]), axis=0).astype(np.int32)
        members = sorted({t for t, _ in mem})
        units.append({"unit": f"rollup:{par}", "kind": "rollup", "key": key, "labelled_topics": [],
                      "rollup_members": members, "member_units": sorted({u["unit"] for _, u in mem}),
                      "tests": members, "add_arm": True, "bands": False, "twin": "pool",
                      "note": "uncarved union of small member units (no carving or training today, Pierre 2026-11-14)"})
    if args.extra:
        ex = json.loads((Path(args.extra) / "plan_extra.json").read_text())
        xa = np.load(Path(args.extra) / "units.npz")
        for u in ex["units"]:
            arrays[u["key"]] = np.asarray(xa[u["key"]], np.int32)
            units.append(u)
    np.savez_compressed(out / "units.npz", **arrays)
    T.write_json(out / "plan.json", {"schema": "mlx2.amoeba-topic-plan.v1", "labels": str(args.labels),
                                     "checkpoint": lab["checkpoint"], "core": args.core,
                                     "clusters": args.clusters or "formed", "core_d": args.core_d,
                                     "bands": args.bands, "mlx2_source_revision": T.git_rev(), "units": units,
                                     "counts": {k: sum(u["kind"] == k for u in units) for k in ("cluster", "subcluster", "core_map", "puzzle", "rollup")}})
    print(f"plan: {len(units)} units -> {out}")


if __name__ == "__main__":
    main()

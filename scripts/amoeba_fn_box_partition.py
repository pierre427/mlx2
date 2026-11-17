#!/usr/bin/env python3
"""FN expert-box partition (docs/research/amoeba-27b/FN_BOXES_PLAN.md, section 4), CPU only, no model.

Inputs: the T2 core and the box run dirs (each export_core.npz = its fixed tier + its box; box = export core minus
that run's own --fixed tier, which is T2 for a plain box and T2 + code_common for a language on top of code_common).
1. Picking: a box passes when its export core's held dCE <= 0.5 x T2-alone held dCE and <= 0.5 x the random
   same-count box's held dCE (both on the box's own held text), its gate passed, and it has >= --min-experts
   experts. Failed or gate-refused boxes are dropped before anything else (each expert stays in one place).
2. Merges: boxes whose pairwise Jaccard >= --merge-jaccard join transitively (connected components); a
   component becomes one box (union of experts, opens on every member's topic).
3. Distinctness: a box with < --min-own of its experts in no other box is folded into the box it overlaps most.
4. Partition: each pool expert (outside T2) with membership set S (the boxes holding it) goes to the tile of
   exactly S; S empty -> the remainder tile (pool.unclaimed; part 2 splits it into atlas-cluster remainders).
   Shared tiles smaller than --min-tile merge into one cross-box residue tile opening on all their boxes.
   Check: T2 + tiles cover every expert exactly once.
Writes partition.npz (core, <tile>__mask as [L, E] uint8) and partition.json (boxes, verdicts, merges, Jaccard,
own share, tiles, coverage).

Usage: amoeba_fn_box_partition.py --t2 T2.npz --boxes-root DIR --out DIR [--model-slug S]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t2", required=True)
    ap.add_argument("--boxes-root", required=True, help="dir holding box-<name>/ run dirs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-slug", default="qwen38-flash-next")
    ap.add_argument("--min-experts", type=int, default=4)
    ap.add_argument("--merge-jaccard", type=float, default=0.5)
    ap.add_argument("--min-own", type=float, default=0.5)
    ap.add_argument("--min-tile", type=int, default=4)
    ap.add_argument("--fold-mode", default="union", choices=["union", "pairwise", "none"],
                    help="union: fold a box whose own share against the union of all other boxes is < --min-own "
                         "(the plan's literal rule; cascades when many large boxes overlap a little each); pairwise: "
                         "fold only when >= 1 - --min-own of its experts lie in ONE other box (containment), once, "
                         "no cascade; none: shared tiles only")
    ap.add_argument("--gap-frac", type=float, default=0.5)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    t2 = np.load(a.t2)["core"] > 0
    pool = ~t2
    verdicts, boxes = {}, {}
    for d in sorted(Path(a.boxes_root).glob("box-*")):
        name = d.name[len("box-"):]
        try:
            r = json.loads((d / "carve.json").read_text())
        except FileNotFoundError:
            verdicts[name] = {"pass": False, "why": "no carve.json"}
            continue
        g = (r.get("inspect") or {}).get("paths", {}).get("layerwise_grad", {})
        if not (isinstance(g, dict) and g.get("ok")) or not (d / "export_core.npz").exists():
            verdicts[name] = {"pass": False, "why": "gate-refused or no export core"}
            continue
        mc = np.load(d / "export_core.npz")["core"] > 0
        if mc.shape != t2.shape:
            raise SystemExit(f"{d}: export core {mc.shape} vs T2 {t2.shape}: a neuron-unit box set needs T2 in the "
                             f"same layout (pass the expanded T2); the partition never mixes layouts")
        m = mc & pool
        fx = ((r.get("args") or {}).get("fixed") or "")      # a box carved on top of another tier (code hierarchy):
        fpath, _, fkey = fx.partition(":")                    # the box is its export core minus its own fixed tier
        if fpath and Path(fpath).exists():
            fx_ = np.load(fpath)[fkey or "core"] > 0
            if fx_.shape != m.shape:              # an expert-unit fixed tier of a neuron-unit box (--fixed-expert-units)
                fx_ = np.repeat(fx_, m.shape[1] // fx_.shape[1], axis=1)
            m &= ~fx_
        held = (r.get("export_core") or {}).get("held", {}).get("mean_d_ce")
        t2h = (r.get("fixed_only_held") or {}).get("mean_d_ce")
        rnd = ((r.get("export_core") or {}).get("random_same_count_held") or      # at the export count (box batch)
               (r.get("bisect") or {}).get("zero", {}).get("0.05", {}).get("random_same_counts_held") or {}).get("mean_d_ce")
        v = {"experts": int(m.sum()), "export_held": held, "t2_alone_held": t2h, "random_same_count_held": rnd}
        why = []
        if held is None or t2h is None or held > a.gap_frac * t2h:
            why.append(f"does not close >= {1 - a.gap_frac:.0%} of the T2 gap")
        if held is None or rnd is None or held > a.gap_frac * rnd:
            why.append(f"not <= {a.gap_frac} x random same-count")
        if m.sum() < a.min_experts:
            why.append(f"< {a.min_experts} experts")
        v.update({"pass": not why, "why": why})
        verdicts[name] = v
        if not why:
            boxes[name] = m
    names = sorted(boxes)
    jac = {}
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            j = float((boxes[x] & boxes[y]).sum() / max(1, (boxes[x] | boxes[y]).sum()))
            jac[f"{x}|{y}"] = j
            if j >= a.merge_jaccard:
                parent[find(x)] = find(y)
    comps = {}
    for n in names:
        comps.setdefault(find(n), []).append(n)
    merged = {}
    for members in comps.values():
        key = "+".join(sorted(members)) if len(members) > 1 else members[0]
        m = np.zeros_like(t2)
        for n in members:
            m |= boxes[n]
        merged[key] = {"mask": m, "members": sorted(members)}
    folds = []
    if a.fold_mode == "pairwise":                     # containment in one other box, decided on the unfolded boxes
        snap = {k: v["mask"].copy() for k, v in merged.items()}
        for k in sorted(snap, key=lambda k: snap[k].sum()):
            if k not in merged:
                continue
            m = snap[k]
            best = max((o for o in merged if o != k), key=lambda o: int((m & snap[o]).sum()), default=None)
            if best is None:
                continue
            cont = float((m & snap[best]).sum() / max(1, m.sum()))
            if cont >= 1 - a.min_own:
                merged[best]["mask"] = merged[best]["mask"] | m
                merged[best]["members"] = sorted(set(merged[best]["members"]) | set(merged[k]["members"]))
                folds.append({"box": k, "contained_share": cont, "into": best})
                del merged[k]
    changed = a.fold_mode == "union"
    while changed:                                    # fold boxes that are mostly someone else's
        changed = False
        for k in sorted(merged, key=lambda k: merged[k]["mask"].sum()):
            others = [o for o in merged if o != k]
            if not others:
                break
            union = np.zeros_like(t2)
            for o in others:
                union |= merged[o]["mask"]
            m = merged[k]["mask"]
            own = float((m & ~union).sum() / max(1, m.sum()))
            if own < a.min_own:
                tgt = max(others, key=lambda o: int((m & merged[o]["mask"]).sum()))
                merged[tgt]["mask"] = merged[tgt]["mask"] | m
                merged[tgt]["members"] = sorted(set(merged[tgt]["members"]) | set(merged[k]["members"]))
                folds.append({"box": k, "own_share": own, "into": tgt})
                del merged[k]
                changed = True
                break
    bnames = sorted(merged)
    code = np.zeros(t2.shape, np.int64)
    for i, n in enumerate(bnames):
        code |= merged[n]["mask"].astype(np.int64) << i
    tiles, rows = {}, []
    for c in sorted(set(np.unique(code[pool]).tolist())):
        m = pool & (code == c)
        members = [bnames[i] for i in range(len(bnames)) if c >> i & 1]
        tid = "pool.unclaimed" if not members else ("box." + members[0] if len(members) == 1 else "shared." + "&".join(members))
        tiles[tid] = m
        rows.append({"tile": tid, "kind": "remainder" if not members else ("box" if len(members) == 1 else "shared_cross_box"),
                     "opens_on": [t for n in members for t in merged[n]["members"]], "experts": int(m.sum())})
    small = [r for r in rows if r["kind"] == "shared_cross_box" and r["experts"] < a.min_tile]
    if small:
        m = np.zeros_like(t2)
        for r in small:
            m |= tiles.pop(r["tile"])
            rows.remove(r)
        rows.append({"tile": "shared.cross_box_residue", "kind": "shared_cross_box_residue",
                     "opens_on": sorted({x for r in small for x in r["opens_on"]}), "experts": int(m.sum()),
                     "merged_from": [r["tile"] for r in small]})
        tiles["shared.cross_box_residue"] = m
    total = t2.astype(np.int64) + sum(m.astype(np.int64) for m in tiles.values())
    cov = {"experts": int(t2.size), "covered_once": int((total == 1).sum()), "uncovered": int((total == 0).sum()),
           "covered_more_than_once": int((total > 1).sum())}
    cov["partition"] = cov["covered_once"] == cov["experts"]
    own = {}
    for n in bnames:
        union = np.zeros_like(t2)
        for o in bnames:
            if o != n:
                union |= merged[o]["mask"]
        mm = merged[n]["mask"]
        own[n] = float((mm & ~union).sum() / max(1, mm.sum()))
    for r in rows:
        r["tile_id"] = f"{a.model_slug}.amoeba.v1.tile.{r['tile']}"
    np.savez_compressed(out / "partition.npz", core=t2.astype(np.uint8), **{f"{t}__mask": m.astype(np.uint8) for t, m in tiles.items()})
    rep = {"schema": "mlx2.amoeba-fn-box-partition.v1", "t2": a.t2, "boxes_root": a.boxes_root, "params": vars(a),
           "verdicts": verdicts, "passing": names, "merges": {k: v["members"] for k, v in merged.items() if len(v["members"]) > 1},
           "folds": folds, "distinct_boxes": len(bnames), "own_share": own, "jaccard": jac, "tiles": rows, "coverage": cov,
           "t2_frac": float(t2.mean())}
    (out / "partition.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({"passing": len(names), "distinct": len(bnames), "tiles": len(rows), "coverage": cov}))
    if not cov["partition"]:
        raise SystemExit("not a partition")


if __name__ == "__main__":
    main()

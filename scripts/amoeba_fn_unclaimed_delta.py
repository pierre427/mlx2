#!/usr/bin/env python3
"""Where the unclaimed core went (Pierre 2026-11-18: "re-probe the ones we had missed"). CPU only.

Inputs: --old DIR, a partition (scripts/amoeba_fn_partition.py --emit-unassigned) of the earlier box set; --new DIR, a
partition of a larger box set over the same model, layers, T2 and English map (also --emit-unassigned, with tiles).
The unclaimed units of a partition are core units outside T2 and the English map that no box selected. Adding boxes
can only claim some of them, so the new unclaimed set should lie inside the old one (checked and reported).

Per old unclaimed unit, where it is in the new partition: still unclaimed, a box tile (per box), a pair tile (per pair,
summed), a family tile (per group), or the core because >= 3 boxes outside one kept family claim it (or a small pair
spilled). Writes OUT.json and OUT.md (shares of the whole unit count and of the old unclaimed set).

Usage: amoeba_fn_unclaimed_delta.py --old OLD_PARTITION --new NEW_PARTITION --out OUT_PREFIX
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", required=True, help="output prefix: writes PREFIX.json and PREFIX.md")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--skipped", default="", help="heldfit-check.json: record the boxes the battery skips")
    a = ap.parse_args()
    od, nd = Path(a.old), Path(a.new)
    oj, nj = json.loads((od / "partition.json").read_text()), json.loads((nd / "partition.json").read_text())
    for k in ("fingerprint",):
        if oj[k] != nj[k]:
            raise SystemExit(f"partitions differ in {k}")
    if oj["inputs"]["t2_sha256_12"] != nj["inputs"]["t2_sha256_12"] or oj["units_total"] != nj["units_total"]:
        raise SystemExit("partitions of another T2 or unit count")
    ou, nu = np.load(od / "unassigned.npz"), np.load(nd / "unassigned.npz")
    if [int(x) for x in ou["layers"]] != [int(x) for x in nu["layers"]]:
        raise SystemExit("layer lists differ")
    OU, NU = ou["unassigned"].astype(bool), nu["unassigned"].astype(bool)
    tz = np.load(nd / "tiles.npz")
    tot = int(oj["units_total"])
    n_old = int(OU.sum())
    where = {"still_unclaimed": int((OU & NU).sum())}
    boxes, pairs, groups, joints = {}, {}, {}, {}
    for k in tz.files:
        if k in ("layers", "core"):
            continue
        c = int((tz[k] & OU).sum())
        if not c:
            continue
        if k.startswith("box:"):
            boxes[k[4:]] = c
        elif k.startswith("pair:"):
            pairs[k[5:]] = c
        elif k.startswith("group:"):
            groups[k[6:]] = c
        elif k.startswith("joint:"):
            joints[k[6:]] = c
    core_claimed = int((tz["core"] & OU & ~NU).sum())
    where.update(box_tiles=sum(boxes.values()), pair_tiles=sum(pairs.values()), family_tiles=sum(groups.values()), joint_tiles=sum(joints.values()),
                 core_3plus_or_spilled=core_claimed)
    if sum(where.values()) != n_old:
        raise SystemExit(f"accounting error: {sum(where.values())} != {n_old}")
    outside = int((NU & ~OU).sum())
    claimed = n_old - where["still_unclaimed"]
    new_boxes = sorted(set(nj["boxes"]) - set(oj["boxes"]))
    rep = {"schema": "mlx2.amoeba-fn-unclaimed-delta.v1", "checkpoint": nj.get("checkpoint"), "old": str(od),
           "new": str(nd), "units_total": tot, "old_boxes": len(oj["boxes"]), "new_boxes": len(nj["boxes"]),
           "boxes_added": new_boxes,
           "old_unclaimed": n_old, "old_unclaimed_share": n_old / tot,
           "new_unclaimed": int(NU.sum()), "new_unclaimed_share": float(NU.mean()),
           "new_unclaimed_outside_old": outside,
           "old_unclaimed_now_claimed": claimed, "old_unclaimed_now_claimed_share_of_old": claimed / max(n_old, 1),
           "where": where, "where_share_of_old": {k: v / max(n_old, 1) for k, v in where.items()},
           "by_box": dict(sorted(boxes.items(), key=lambda x: -x[1])),
           "by_group": dict(sorted(groups.items(), key=lambda x: -x[1])),
           "by_pair_top": dict(sorted(pairs.items(), key=lambda x: -x[1])[:a.top]), "pairs_with_units": len(pairs),
           "core_share": {"old": oj["core"]["share"], "new": nj["core"]["share"]}}
    skipped = {}
    if a.skipped:
        skipped = json.loads(Path(a.skipped).read_text()).get("skipped_boxes") or {}
        rep["battery_skipped"] = skipped
    Path(a.out + ".json").write_text(json.dumps(rep, indent=1))
    pct = lambda x: f"{100 * x:.1f}%"                                    # noqa: E731
    md = [f"# Unclaimed core: {len(oj['boxes'])} boxes -> {len(nj['boxes'])} boxes\n",
          f"- unclaimed: {n_old:,} units ({pct(n_old / tot)} of {tot:,}) -> {int(NU.sum()):,} "
          f"({pct(float(NU.mean()))}); new unclaimed outside the old set: {outside:,}",
          f"- of the old unclaimed, now claimed: {claimed:,} ({pct(claimed / max(n_old, 1))})",
          f"- core share: {pct(oj['core']['share'])} -> {pct(nj['core']['share'])}\n",
          "| old unclaimed now in | units | share of old unclaimed |", "|---|---|---|"]
    md += [f"| {k} | {v:,} | {pct(v / max(n_old, 1))} |" for k, v in where.items()]
    md += ["", f"Top boxes (of {len(boxes)} with any):", "", "| box | units | new box |", "|---|---|---|"]
    md += [f"| {k} | {v:,} | {'yes' if k in new_boxes else ''} |" for k, v in list(rep["by_box"].items())[:a.top]]
    if groups:
        md += ["", "| family tile | units |", "|---|---|"] + [f"| {k} | {v:,} |" for k, v in rep["by_group"].items()]
    if skipped:
        md += ["", "Boxes in this partition (fit maps) but skipped by the battery:", ""] + \
            [f"- {k}: {v}" for k, v in sorted(skipped.items())]
    Path(a.out + ".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

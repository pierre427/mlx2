#!/usr/bin/env python3
"""Full partition of the 27B MLP into core + named-box tiles + bucket tiles (CPU, no model).

FULL_PARTITION_DESIGN.md. Every pool neuron has a membership (S, b): S = the named boxes holding it, b = its
bucket. S empty -> bucket tile b; S non-empty -> the named tile for exactly S (one tile per distinct S), so a
neuron shared by several boxes lives once (Pierre, 2026-11-15). A tile opens on the OR of its members: the
boxes in S and every bucket it holds neurons from. Writes partition.npz ({tile}__mask, [L, N] uint8) and
partition.json (tiles, members, sizes, opening rule) and fails unless core + tiles covers every neuron once.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", required=True, help="NPZ[:key] core mask")
    ap.add_argument("--boxes", required=True, help="boxes.npz with <name>__mask entries")
    ap.add_argument("--buckets", required=True, help="buckets dir (buckets.npz + buckets.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-tile", type=int, default=0, help="merge cross-box (shared) tiles smaller than this into one "
                    "'named.shared_cross_box' residue tile that opens on the OR of all their members (0 = never). Each "
                    "neuron still lives in one place; small tiles only cost per-matmul overhead")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    cp, _, ck = a.core.partition(":")
    core = np.load(cp)[ck or "core"] > 0
    L, N = core.shape
    bz = np.load(a.boxes)
    boxes = {k[:-len("__mask")]: bz[k] > 0 for k in bz.files if k.endswith("__mask")}
    bd = Path(a.buckets)
    bnpz = np.load(bd / "buckets.npz")
    binfo = json.loads((bd / "buckets.json").read_text())
    buckets = {k[:-len("__mask")]: bnpz[k] > 0 for k in bnpz.files if k.endswith("__mask")}
    for n, m in {**boxes, **buckets}.items():
        assert m.shape == (L, N), (n, m.shape)
    # a neuron in the core and in a box stays in the core
    box_in_core = {n: int((m & core).sum()) for n, m in boxes.items()}
    names = sorted(boxes)
    code = np.zeros((L, N), np.int64)                     # bit i = named box i
    for i, n in enumerate(names):
        code |= (boxes[n] & ~core).astype(np.int64) << i
    bucket_of = np.full((L, N), -1, np.int64)
    bnames = sorted(buckets)
    for j, n in enumerate(bnames):
        sel = buckets[n] & ~core
        assert not (bucket_of[sel] >= 0).any(), f"bucket {n} overlaps another bucket"
        bucket_of[sel] = j
    pool = ~core
    if (bucket_of[pool] < 0).any():
        raise SystemExit(f"{int((bucket_of[pool] < 0).sum())} pool neurons are in no bucket")
    tiles, rows = {}, []
    for j, n in enumerate(bnames):                         # bucket remainders (S empty)
        m = pool & (code == 0) & (bucket_of == j)
        if m.any():
            tid = binfo["buckets"].get(n, {}).get("tile_id") or f"bucket.{n}"
            tiles[tid] = m
            rows.append({"tile": tid, "kind": "bucket", "bucket": n, "boxes": [], "opens_on_buckets": [n],
                         "neurons": int(m.sum())})
    for c in sorted(set(np.unique(code[pool]).tolist()) - {0}):   # named tiles, one per distinct box set
        m = pool & (code == c)
        members = [names[i] for i in range(len(names)) if c >> i & 1]
        bk = sorted({bnames[j] for j in np.unique(bucket_of[m]).tolist()})
        tid = "named." + "+".join(members)
        tiles[tid] = m
        rows.append({"tile": tid, "kind": "named" if len(members) == 1 else "shared_cross_box", "boxes": members,
                     "opens_on_buckets": bk, "neurons": int(m.sum()),
                     "per_bucket": {bnames[j]: int((m & (bucket_of == j)).sum()) for j in np.unique(bucket_of[m]).tolist()}})
    if a.min_tile > 0:
        small = [r for r in rows if r["kind"] == "shared_cross_box" and r["neurons"] < a.min_tile]
        if small:
            m = np.zeros((L, N), bool)
            for r in small:
                m |= tiles.pop(r["tile"])
                rows.remove(r)
            rows.append({"tile": "named.shared_cross_box", "kind": "shared_cross_box_residue",
                         "boxes": sorted({x for r in small for x in r["boxes"]}),
                         "opens_on_buckets": sorted({x for r in small for x in r["opens_on_buckets"]}),
                         "neurons": int(m.sum()), "merged_from": [r["tile"] for r in small]})
            tiles["named.shared_cross_box"] = m
    total = core.astype(np.int64) + sum(m.astype(np.int64) for m in tiles.values())
    cov = {"neurons": int(L * N), "covered_once": int((total == 1).sum()), "uncovered": int((total == 0).sum()),
           "covered_more_than_once": int((total > 1).sum())}
    cov["partition"] = cov["covered_once"] == cov["neurons"]
    for n, m in boxes.items():                             # every named box is fully reachable through its tiles
        reach = np.zeros_like(m)
        for r in rows:
            if n in r["boxes"]:
                reach |= tiles[r["tile"]]
        assert not (m & ~core & ~reach).any(), f"box {n} not fully reachable through its tiles"
    np.savez_compressed(out / "partition.npz", core=core.astype(np.uint8),
                        **{f"{t}__mask": m.astype(np.uint8) for t, m in tiles.items()})
    summary = {"schema": "mlx2.amoeba-partition.v1", "core": a.core, "boxes": a.boxes, "buckets": a.buckets,
               "rule": "one place per neuron: core; else the named tile of its exact box set; else its bucket tile. "
                       "A tile opens on the OR of its boxes and of the buckets it holds neurons from.",
               "box_neurons_in_core_moved_to_core": box_in_core, "coverage": cov,
               "core_frac": float(core.mean()), "tiles": rows,
               "named_box_sizes": {n: int(m.sum()) for n, m in boxes.items()},
               "box_overlap_neurons": {f"{x}&{y}": int((boxes[x] & boxes[y]).sum())
                                       for i, x in enumerate(names) for y in names[i + 1:] if (boxes[x] & boxes[y]).any()}}
    (out / "partition.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"coverage": cov, "tiles": len(rows),
                      "named": [(r["tile"], r["neurons"]) for r in rows if r["kind"] != "bucket"]}, indent=1))
    if not cov["partition"]:
        raise SystemExit("not a partition")


if __name__ == "__main__":
    main()

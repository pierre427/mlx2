#!/usr/bin/env python3
"""FN neuron partition from firing maps (Pierre 2026-11-16/17: a box takes the neurons only its topic uses; shared
neurons go to the core or to shared tiles; "3+ shared -> core", two boxes -> a cross-box tile). CPU only.

Inputs: --box NAME=COMPARE_DIR (scripts/amoeba_firing_compare.py output: map.npz with per-site target / mean_target /
mean_baseline; one per box: languages, subjects, inference), --core-map COMPARE_DIR (the English map, English vs the
10-language mix: its target units belong to the core), --t2 T2.npz (the English core, expert units [L, E], expanded
to neurons: unit = expert * F + neuron).
Per box, the selection at exclusivity --ratio r: target units whose baseline mean is <= r x their target mean (the
cnbox rule; r = 0.25 measured on Mandarin: cmn +1.22 without it, English +0.002). Then per unit:
  in T2, in the English map, or selected by >= 3 boxes  -> core
  selected by exactly 2 boxes                           -> the pair tile <a>__<b>
  selected by exactly 1 box                             -> that box
  selected by none (and not core)                       -> core, counted as "unassigned" (a remainder for a later
                                                           cluster pass; kept in the core so the tiled model is never
                                                           worse than the base)
Exact cover is checked (every unit in exactly one tile). Writes OUT/tiles.npz (per tile a [L, E*F] bool mask; keys
core, box:<name>, pair:<a>__<b>) and OUT/partition.json (counts, expert equivalents, whole experts per tile, the
largest pair overlaps, inputs with their revisions).

Usage: amoeba_fn_partition.py --box cmn=DIR --box fra=DIR ... --core-map DIR --t2 T2.npz --out DIR [--ratio 0.25]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def load_map(d, key="target"):
    d = Path(d)
    rep = json.loads((d / "report.json").read_text())
    z = np.load(d / "map.npz")
    S = len(rep["per_site"])
    return rep, z, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", action="append", required=True, help="NAME=compare dir")
    ap.add_argument("--core-map", required=True)
    ap.add_argument("--t2", required=True)
    ap.add_argument("--ratio", type=float, default=0.25)
    ap.add_argument("--min-pair", type=int, default=0, help="pair tiles smaller than this go to the core (0: keep all)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    crep, cz, S = load_map(a.core_map)
    sites = crep["per_site"]
    if any(s["class"] != "moe" for s in sites):
        raise SystemExit("this partition is for MoE neuron units (every site MoE)")
    E, F = None, None
    L = len(sites)
    n = sites[0]["units"]
    if any(s["units"] != n for s in sites):
        raise SystemExit("uneven site widths")
    core_map = np.stack([cz[f"target_{s}"] for s in range(S)])                      # [L, n]
    t2 = np.load(a.t2)["core"] > 0
    layers = [s["layer"] for s in sites]
    t2 = t2[layers]
    if t2.shape[1] != n:
        E = t2.shape[1]
        F = n // E
        if E * F != n:
            raise SystemExit(f"T2 width {t2.shape[1]} does not divide the site width {n}")
        t2 = np.repeat(t2, F, axis=1)
    sel, info = {}, {}
    for spec in a.box:
        name, _, d = spec.partition("=")
        rep, z, S2 = load_map(d)
        if S2 != S or rep["fingerprint"] != crep["fingerprint"] or \
                [x["layer"] for x in rep["per_site"]] != layers or [x["units"] for x in rep["per_site"]] != [n] * S:
            raise SystemExit(f"{name}: map of another model or layout")
        tm = np.stack([z[f"target_{s}"] for s in range(S)])
        mt = np.stack([z[f"mean_target_{s}"] for s in range(S)]).astype(np.float64)
        mb = np.stack([z[f"mean_baseline_{s}"] for s in range(S)]).astype(np.float64)
        sel[name] = tm & (mb <= a.ratio * mt)
        info[name] = {"map": str(d), "target_units": int(tm.sum()), "selected": int(sel[name].sum()),
                      "baseline_corpus": rep["baseline"]["corpus"], "target_corpus": rep["target"]["corpus"],
                      "capture_rev": rep.get("capture_rev")}
    names = sorted(sel)
    claims = np.zeros((L, n), np.int16)
    for nm in names:
        claims += sel[nm]
    core = t2 | core_map | (claims >= 3)
    tiles = {"core": core.copy()}
    for nm in names:
        tiles[f"box:{nm}"] = sel[nm] & (claims == 1) & ~core
    pairs = Counter()
    two = (claims == 2) & ~core
    idx = {nm: i for i, nm in enumerate(names)}
    if two.any():
        stack = np.stack([sel[nm] for nm in names])                                   # [B, L, n]
        for i, x in enumerate(names):
            for y in names[i + 1:]:
                m = two & stack[idx[x]] & stack[idx[y]]
                c = int(m.sum())
                if c:
                    if c < a.min_pair:
                        tiles["core"] |= m
                    else:
                        tiles[f"pair:{x}__{y}"] = m
                        pairs[f"{x}__{y}"] = c
    unassigned = ~(tiles["core"] | np.logical_or.reduce([v for k, v in tiles.items() if k != "core"] or [np.zeros_like(core)]))
    tiles["core"] |= unassigned
    cover = sum(v.astype(np.int16) for v in tiles.values())
    if not (cover == 1).all():
        raise SystemExit(f"partition is not an exact cover: {int((cover != 1).sum())} units off")
    np.savez_compressed(out / "tiles.npz", layers=np.array(layers), **{k: v for k, v in tiles.items()})
    tot = L * n
    FF = F or crep["per_site"][0].get("F") or 640      # neurons per expert (T2 in expert units gives it directly)
    rep = {"schema": "mlx2.amoeba-fn-partition.v1", "checkpoint": crep["checkpoint"], "fingerprint": crep["fingerprint"],
           "rule": f"exclusivity r={a.ratio}; core = T2 | English map | >=3 boxes | unassigned; 2 boxes -> pair tile; "
                   "1 box -> box", "units_total": tot, "ratio": a.ratio,
           "core": {"units": int(tiles["core"].sum()), "share": float(tiles["core"].mean()),
                    "from_t2": int(t2.sum()), "from_english_map_outside_t2": int((core_map & ~t2).sum()),
                    "from_3plus_boxes": int(((claims >= 3) & ~t2 & ~core_map).sum()),
                    "unassigned": int(unassigned.sum()),
                    "small_pairs_spilled": int(tiles["core"].sum() - (t2 | core_map | (claims >= 3)).sum()
                                               - unassigned.sum())},
           "boxes": {nm: {**info[nm], "units": int(tiles[f"box:{nm}"].sum()),
                          "share": float(tiles[f"box:{nm}"].mean()),
                          "whole_experts": int((tiles[f"box:{nm}"].reshape(L, -1, FF).sum(2) == FF).sum())}
                     for nm in names},
           "pairs": dict(pairs.most_common()),
           "inputs": {"core_map": a.core_map, "t2": a.t2,
                      "t2_sha256_12": hashlib.sha256(Path(a.t2).read_bytes()).hexdigest()[:12]}}
    (out / "partition.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({"core": rep["core"], "boxes": {k: v["units"] for k, v in rep["boxes"].items()},
                      "pairs_top": dict(pairs.most_common(10)), "pair_tiles": len(pairs)}, indent=1))


if __name__ == "__main__":
    main()

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

Family tiles (opt-in, Pierre 2026-11-18: "were any set of expert boxes where the shared neurons were shared between 3
or more experts sufficiently isolated to similar experts? ... too strict?"): with --groups FILE.json (name -> member
boxes) and/or --groups-auto, a unit selected by >= 3 boxes (outside T2 and the English map) whose claim set lies
inside one kept group, and so is claimed by no box outside it, goes to the GROUP tile group:<name> instead of the
core; the smallest kept group that contains the claim set takes it. Groups must be laminar (any two nested or
disjoint): a declared conflict is refused, an automatic one is skipped and recorded. A group tile is the master of its
direct sub-tiles: its direct kept subgroups and its member boxes outside them that have a nonempty box tile. Pierre
2026-11-16: "never one master + one sub-tile", so a group with fewer than 2 direct sub-tiles is dropped (pair tiles
are cross tiles, not sub-tiles: a group whose only children are one box and one pair is dropped too), as is a group
of fewer than 3 members (two boxes are a pair tile) or one below --group-min-units. Groups are decided smallest
first, so a dropped subgroup's units pass to the parent before the parent is judged; a dropped group's units go to
its kept parent or the core. Every dropped group is recorded with its reason. --groups-auto proposes groups from the
claim-set statistics: every exact claim set of >= --auto-min-members boxes held by >= --auto-min-units units is a
candidate, accepted greedily by closed size (units whose claim set lies inside it) when it is laminar with every group
already accepted (declared groups first, even one later dropped). Closed size favours the larger sets, so an auto
group may join a family with one unrelated box (on the FN maps: life.games + the ml trio); nothing here measures
isolation, which is what the battery's group arm is for. Box and pair tiles are unchanged by groups (groups take only >= 3-claimed
units). --emit-unassigned writes OUT/unassigned.npz (the unclaimed units, a subset of the core); --check-against DIR
refuses unless this partition equals DIR's with the group tiles folded back into the core (same boxes, pairs,
unassigned count and T2). Without these flags the output is unchanged.

Usage: amoeba_fn_partition.py --box cmn=DIR --box fra=DIR ... --core-map DIR --t2 T2.npz --out DIR [--ratio 0.25]
       [--groups G.json] [--groups-auto [--auto-min-units N] [--auto-min-members 3]] [--group-min-units N]
       [--emit-unassigned] [--check-against PARTITION_DIR] [--no-tiles]
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


def load_groups(path, names):
    """Declared groups: {"name": [boxes]} or {"groups": {...}}; keys starting with "_" are notes. Unknown boxes are
    refused."""
    g = json.loads(Path(path).read_text())
    g = g.get("groups", g)
    out = {}
    for k, mem in g.items():
        if k.startswith("_"):
            continue
        bad = [m for m in mem if m not in names]
        if bad:
            raise SystemExit(f"group {k}: unknown boxes {bad}")
        if len(set(mem)) != len(mem):
            raise SystemExit(f"group {k}: repeated members")
        out[str(k)] = sorted(mem)
    return out


def laminar(a, b):
    a, b = set(a), set(b)
    return not (a & b) or a <= b or b <= a


def propose_groups(claim_sets, counts, names, min_units, min_members, accepted):
    """Auto groups from the exact claim sets of the >= 3-claimed units (claim_sets: box bitmasks; counts: units each).
    Candidates: sets of >= min_members boxes with >= min_units units. Accepted by closed size, laminar with `accepted`
    and with each other. Returns (groups, log of every candidate)."""
    bits = lambda v: [names[i] for i in range(len(names)) if (int(v) >> i) & 1]           # noqa: E731
    cands, log = [], []
    sets = np.asarray(claim_sets, np.uint64)
    cnt = np.asarray(counts, np.int64)
    for v, c in zip(sets, cnt):
        mem = bits(v)
        if len(mem) >= min_members and c >= min_units:
            closed = int(cnt[(sets & ~v) == 0].sum())
            cands.append((closed, int(c), mem))
    cands.sort(key=lambda x: (-x[0], len(x[2]), x[2]))
    out = {}
    for closed, exact, mem in cands:
        name = "+".join(mem)
        row = {"group": name, "members": mem, "exact_units": exact, "closed_units": closed}
        same = next((k for k, m in accepted.items() if sorted(m) == mem), None)
        if same is not None:
            log.append({**row, "dropped": f"same members as declared group {same}"})
            continue
        clash = next((k for k, m in list(accepted.items()) + list(out.items()) if not laminar(m, mem)), None)
        if clash is not None:
            log.append({**row, "dropped": f"not laminar with group {clash} (overlaps without nesting)"})
            continue
        out[name] = mem
        log.append({**row, "proposed": True})
    return out, log


def decide_groups(groups, three, cmask, idx, box_units, pair_units, min_units):
    """Assign the >= 3-claimed units to the smallest containing kept group, deciding groups smallest first.
    groups: name -> members (laminar); three: [L, n] bool; cmask: [L, n] uint64 claim bitmask; box_units: box -> units
    of its box tile; pair_units: 'a__b' -> units. Returns (assign: name -> mask, kept info, dropped info)."""
    order = sorted(groups, key=lambda k: (len(groups[k]), k))
    taken = np.zeros(three.shape, bool)
    assign, kept, dropped = {}, {}, {}
    for g in order:
        mem = groups[g]
        gm = np.uint64(0)
        for m in mem:
            gm |= np.uint64(1) << np.uint64(idx[m])
        subs = [k for k in kept if set(kept[k]["members"]) < set(mem)]
        direct = [k for k in subs if not any(set(kept[k]["members"]) < set(kept[j]["members"]) for j in subs)]
        covered = set().union(*[set(kept[k]["members"]) for k in direct]) if direct else set()
        dboxes = [m for m in mem if m not in covered and box_units.get(m, 0) > 0]
        dpairs = sorted(p for p, c in pair_units.items() if c and set(p.split("__")) <= set(mem)
                        and not any(set(p.split("__")) <= set(kept[k]["members"]) for k in direct))
        m = three & ~taken & ((cmask & ~gm) == 0)
        u = int(m.sum())
        why = None
        if len(mem) < 3:
            why = "fewer than 3 members (two boxes are a pair tile)"
        elif u == 0:
            why = "no units"
        elif u < min_units:
            why = f"{u} units < --group-min-units {min_units}"
        elif len(direct) + len(dboxes) < 2:
            why = ("only one box and one pair below it (Pierre: never one master + one sub-tile)" if
                   len(dboxes) == 1 and not direct and dpairs else
                   f"{len(direct) + len(dboxes)} direct sub-tile(s) (Pierre: never one master + one sub-tile)")
        info = {"members": mem, "units": u, "subgroups": direct, "boxes": dboxes, "pairs_inside": dpairs}
        if why:
            dropped[g] = {**info, "reason": why}
            continue
        taken |= m
        assign[g] = m
        kept[g] = info
    for g in kept:
        sup = [k for k in kept if set(kept[g]["members"]) < set(kept[k]["members"])]
        kept[g]["parent"] = min(sup, key=lambda k: (len(kept[k]["members"]), k)) if sup else None
    return assign, kept, dropped


def check_against(ref, tiles, unassigned, rep):
    """Refuse unless this partition equals the reference one with the group tiles folded back into the core."""
    rz = np.load(ref / "tiles.npz")
    rj = json.loads((ref / "partition.json").read_text())
    if rj.get("groups") or any(k.startswith("group:") for k in rz.files):
        raise SystemExit("check-against: the reference has group tiles; give a partition made without --groups")
    if rj["fingerprint"] != rep["fingerprint"] or rj["inputs"]["t2_sha256_12"] != rep["inputs"]["t2_sha256_12"]:
        raise SystemExit("check-against: another model or another T2")
    if rj.get("ratio") != rep["ratio"]:
        raise SystemExit(f"check-against: ratio {rj.get('ratio')} != {rep['ratio']}")
    mine = {k for k in tiles if not k.startswith("group:")}
    theirs = set(rz.files) - {"layers"}
    if theirs != mine:
        raise SystemExit(f"check-against: tile keys differ ({sorted(theirs ^ mine)[:6]} ...; a different "
                         "--min-pair, ratio or box set?)")
    if [int(x) for x in rz["layers"]] != rep["_layers"]:
        raise SystemExit("check-against: layer lists differ")
    folded = tiles["core"] | np.logical_or.reduce([v for k, v in tiles.items() if k.startswith("group:")] or
                                                   [np.zeros_like(tiles["core"])])
    for k in sorted(mine):
        if not np.array_equal(rz[k], folded if k == "core" else tiles[k]):
            raise SystemExit(f"check-against: tile {k} differs from {ref}")
    if int(unassigned.sum()) != rj["core"]["unassigned"]:
        raise SystemExit("check-against: unassigned count differs")
    return {"reference": str(ref), "equal_with_groups_folded": True, "tiles_checked": len(mine)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", action="append", required=True, help="NAME=compare dir")
    ap.add_argument("--core-map", required=True)
    ap.add_argument("--t2", required=True)
    ap.add_argument("--ratio", type=float, default=0.25)
    ap.add_argument("--min-pair", type=int, default=0, help="pair tiles smaller than this go to the core (0: keep all)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", default="", help="JSON name -> member boxes: family (group) tiles (opt-in)")
    ap.add_argument("--groups-auto", action="store_true", help="propose groups from the claim-set statistics")
    ap.add_argument("--auto-min-units", type=int, default=20000)
    ap.add_argument("--auto-min-members", type=int, default=3)
    ap.add_argument("--group-min-units", type=int, default=1)
    ap.add_argument("--emit-unassigned", action="store_true", help="also write OUT/unassigned.npz")
    ap.add_argument("--check-against", default="", help="partition dir this one must equal with groups folded in")
    ap.add_argument("--no-tiles", action="store_true", help="dry run: write partition.json only (no tiles.npz)")
    a = ap.parse_args()
    grouping = bool(a.groups or a.groups_auto)
    if a.auto_min_members < 3:
        raise SystemExit("--auto-min-members must be >= 3 (two boxes are a pair tile)")
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
    grep = None
    if grouping:
        if len(names) > 64:
            raise SystemExit("group tiles need <= 64 boxes (uint64 claim masks)")
        cmask = np.zeros((L, n), np.uint64)
        for nm in names:
            cmask |= sel[nm].astype(np.uint64) << np.uint64(idx[nm])
        three = (claims >= 3) & ~t2 & ~core_map
        declared = load_groups(a.groups, names) if a.groups else {}
        for x in declared:
            for y in declared:
                if x < y and not laminar(declared[x], declared[y]):
                    raise SystemExit(f"declared groups {x} and {y} overlap without nesting (groups must be laminar)")
        auto_log = []
        groups = dict(declared)
        if a.groups_auto:
            vals, cnts = np.unique(cmask[three], return_counts=True)
            prop, auto_log = propose_groups(vals, cnts, names, a.auto_min_units, a.auto_min_members, declared)
            groups.update(prop)
        bu = {nm: int(tiles[f"box:{nm}"].sum()) for nm in names}
        assign, kept, dropped = decide_groups(groups, three, cmask, idx, bu, dict(pairs), a.group_min_units)
        del cmask
        for g, m in assign.items():
            tiles[f"group:{g}"] = m
            tiles["core"] &= ~m
        FFg = F or crep["per_site"][0].get("F") or 640
        for g in kept:
            kept[g].update(share=float(assign[g].mean()), expert_equivalents=round(kept[g]["units"] / FFg, 1),
                           whole_experts=int((assign[g].reshape(L, -1, FFg).sum(2) == FFg).sum()),
                           source="declared" if g in declared else "auto")
        for g in dropped:
            dropped[g]["source"] = "declared" if g in declared else "auto"
        gsum = int(sum(v["units"] for v in kept.values()))
        grep = {"criterion": {
                    "assign": "a unit selected by >= 3 boxes, outside T2 and the English map, whose claim set lies "
                              "inside a kept group goes to the smallest such group (groups laminar)",
                    "keep": f">= 3 members, >= {a.group_min_units} units, >= 2 direct sub-tiles (kept subgroups + "
                            "member boxes outside them with a nonempty box tile; pairs are not sub-tiles); decided "
                            "smallest first",
                    "auto": (f"exact claim sets of >= {a.auto_min_members} boxes with >= {a.auto_min_units} units are "
                             "candidates, accepted by closed size (units whose claim set lies inside) when laminar "
                             "with every group already accepted; declared groups first") if a.groups_auto else None,
                    "declared_file": a.groups or None,
                    "declared_sha256_12": hashlib.sha256(Path(a.groups).read_bytes()).hexdigest()[:12]
                    if a.groups else None},
                "kept": kept, "dropped": dropped, "auto_candidates": auto_log, "units_in_group_tiles": gsum,
                "three_plus_outside_base": int(three.sum()),
                "core_share_without_groups": float((int(tiles["core"].sum()) + gsum) / (L * n))}
    cover = sum(v.astype(np.int16) for v in tiles.values())
    if not (cover == 1).all():
        raise SystemExit(f"partition is not an exact cover: {int((cover != 1).sum())} units off")
    if not a.no_tiles:
        np.savez_compressed(out / "tiles.npz", layers=np.array(layers), **{k: v for k, v in tiles.items()})
    if a.emit_unassigned:
        np.savez_compressed(out / "unassigned.npz", layers=np.array(layers), unassigned=unassigned)
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
    if grep is not None:
        rep["schema"] = "mlx2.amoeba-fn-partition.v2"
        rep["rule"] += "; >=3 boxes inside one kept group -> group tile (family tiles, see groups.criterion)"
        rep["core"]["to_group_tiles"] = grep["units_in_group_tiles"]
        rep["core"]["from_3plus_boxes_kept"] = rep["core"]["from_3plus_boxes"] - grep["units_in_group_tiles"]
        rep["min_pair"] = a.min_pair
        rep["core"]["small_pairs_spilled"] += grep["units_in_group_tiles"]       # the formula above counts them out
        rep["groups"] = grep
    if a.emit_unassigned:
        rep["unassigned_file"] = "unassigned.npz"
    if a.check_against:
        rep["check_against"] = check_against(Path(a.check_against), tiles, unassigned, {**rep, "_layers": layers})
    (out / "partition.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({"core": rep["core"], "boxes": {k: v["units"] for k, v in rep["boxes"].items()},
                      "pairs_top": dict(pairs.most_common(10)), "pair_tiles": len(pairs)}, indent=1))
    if grep is not None:
        print(json.dumps({"core_share": rep["core"]["share"], "core_share_without_groups":
                          grep["core_share_without_groups"],
                          "kept": {g: {k: v[k] for k in ("members", "units", "share", "parent", "subgroups", "boxes")}
                                   for g, v in grep["kept"].items()},
                          "dropped": {g: v["reason"] for g, v in grep["dropped"].items()}}, indent=1))


if __name__ == "__main__":
    main()

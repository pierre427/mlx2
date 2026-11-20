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

Source control (opt-in, 2026-11-18): --require-every-source makes a box select a unit only where the map's
every_source flag holds too (amoeba_firing_compare.py: the unit is a target unit against the whole baseline in each
of the target corpus's sources). Without it a unit driven by one source alone is read as the topic. On the FN maps
(fn-boxes2-dd1ef1d7d) the units claimed by >= 10 boxes were mostly forum-register experts that only the StackExchange
share of 15 subject corpora drives: they passed every_source 0-21% of the time, against 61-92% for each box's
exclusive units. The English core map is not filtered (its units go to the core, which only keeps more).

Nested boxes (opt-in, 2026-11-18, the category builds: a doc counts for its node and every ancestor, so a parent box's
corpus contains its subtopics'): with --nested, dotted box names are a taxonomy (sci > sci.physics >
sci.physics.quantum) and an ancestor's claim on a unit is dropped wherever a descendant box claims it too, before the
claims are counted. A unit claimed by sci.physics.quantum, sci.physics and sci is then quantum's alone (its box tile),
not a 3-box core unit; a unit claimed by sci alone stays sci's. Then (Pierre 2026-11-18: "Give it to the parent";
"a would hold it in that scenario, and parent tiles are always gated in for their subtiles"): a unit whose remaining
claimants are >= 2 boxes (siblings, at any depth) all inside one box's subtree goes to their lowest common ancestor
box P alone, whether or not P selects the unit itself; e.g. quantum + optics -> sci.physics, quantum + sci.biology ->
sci, a.b.c + a.b.d + a.e -> a. Only claimants with no common ancestor box keep the rule below: two make a pair, three
or more the core or the smallest kept family tile. partition.json records per box received_from_subtopics, split
into received_lca_selected (P's own candidate mask, before the drop, selects it) and received_lca_unselected, and
received_from_subtopics_outside_base (not T2 or English-map units). --not-nested a,b keeps those boxes (and every
box below them) out of their dotted-name ancestors' subtrees (e.g. sci.medicine: the category build drops
medical docs from sci, so sci's corpus does not contain medicine's).
partition.json records each box's selected_before_nesting and descendants, and nested.ancestors / nested.descendants
(the battery serves a nested box with its ancestors' box tiles and, for a parent, its whole subtree).

Joint tiles (opt-in, Pierre 2026-11-18: "Yes, do the joint tile variant"): with --joint-floor N (or auto), each exact
claim set of >= 3 boxes still in the core after nesting and family tiles (so with no common ancestor box, and not
inside any declared group, kept or dropped) holding >= N units becomes its own tile joint:<a>+<b>+..., gated by the
OR of its members' gates (no classifier of its own). --joint-sweep 5000,10000,... records, per floor, the tiles, the
units moved and the core share after; --joint-floor auto takes the swept floor with the largest core reduction per
tile (units moved / tiles; ties -> the lower floor) and records why. partition.json "joint" lists every tile.

Weak-claimant detach (opt-in, Pierre 2026-11-19: "If a box's claim on a neuron is tiny next to the other
claimants', drop that claim before deciding pair, family or core"): with --detach-share s > 0, a box's claim on a
unit is dropped when its drive there is below s x the strongest claimant's drive. Drive = the map's mean_target (the
mean per-token contribution |h| * w on the box's own target corpus, recorded by the capture; the same quantity the
selection's exclusivity test uses). Applied after --require-every-source and, with --nested, after the descendant
absorption and before the lowest-common-ancestor roll-up, so pair, family, LCA, joint and core decisions all see the
detached claim sets. The strongest claimant always keeps its claim, so a selected unit never becomes unclaimed; a
detached box loses only tiles it would have shared (the unit goes to the remaining claimants' tile or the core), and
the battery, run on this partition, measures any loss in that box's served arm. --detach-sweep 0.01,0.05,... records
per threshold the claim-share distribution and the units by claim count (box / pair / 3+) outside T2 and the English
map; partition.json "detach" records the counts per box.

English box (opt-in, Pierre 2026-11-19: "English gets its own box on FN; the core holds what's left"): with
--english-box NAME (taxonomy id lang.eng), the English-map units outside T2 leave the core for the tile box:NAME; the
core is then T2 + 3+-claimed + unassigned (+ small pairs spilled). Topic boxes' claims do not change it: an English-
map unit stays English's even where topic boxes select it (as it stayed core before; partition.json records how many),
so a unit never moves out of a served path that held it. Its gate is "the text is English": open when serving every
box except --english-closed-for (the language boxes, whose text is not English), which the battery reads from
partition.json "english_box". NAME is not a claimant, so it takes part in no pair, family, LCA or joint rule.
English-shared arm (opt-in, --english-shared with --english-box; Pierre 2026-11-19: "make sure to also try and build
shared expert boxes with the english box"): English becomes a claimant like any box. A box claims its target units
whether or not they pass the exclusivity ratio against English; where one fails it (English drives the unit at more
than r x the box's drive), English claims the unit too. English also claims the English-map units outside T2 (no
longer forced out of the shared rules). English's drive (for the detach) is the English baseline mean of the box maps
where it shares, the English map's mean_target on its own units (the larger where both). Then the ordinary rules: a
unit English alone claims is box:NAME, English + one box a pair tile, English + 2+ boxes whose other claimants lie
inside one kept family the family tile group:<family>+NAME (the smallest such family; gate: the family's or English),
a cross-family English set of >= 3 a joint tile at the joint floor, else the core. English is open for all English
text, so an English-shared tile is closed only when serving a closed_for (language) box that it does not name;
partition.json english_box.shared records the units per tile kind and that rule.

Any number of boxes: claim masks are [L, n, W] uint64 words (box i = bit i % 64 of word i // 64), pair tiles are kept
as unit positions and written one mask at a time, so 100 boxes and thousands of pair tiles fit in memory.

Usage: amoeba_fn_partition.py --box cmn=DIR --box fra=DIR ... --core-map DIR --t2 T2.npz --out DIR [--ratio 0.25]
       [--require-every-source]
       [--groups G.json] [--groups-auto [--auto-min-units N] [--auto-min-members 3]] [--group-min-units N]
       [--emit-unassigned] [--check-against PARTITION_DIR] [--no-tiles] [--nested]
       [--detach-share S [--detach-sweep S1,S2,...]] [--english-box lang.eng --english-closed-for cmn,fra,...]
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
    """Auto groups from the exact claim sets of the >= 3-claimed units (claim_sets: [K, W] uint64 box bitmasks, box i
    = bit i % 64 of word i // 64; counts: units each).
    Candidates: sets of >= min_members boxes with >= min_units units. Accepted by closed size, laminar with `accepted`
    and with each other. Returns (groups, log of every candidate)."""
    bits = lambda v: [names[i] for i in range(len(names)) if (int(v[i // 64]) >> (i % 64)) & 1]   # noqa: E731
    cands, log = [], []
    sets = np.asarray(claim_sets, np.uint64)
    if sets.ndim == 1:
        sets = sets[:, None]
    cnt = np.asarray(counts, np.int64)
    for v, c in zip(sets, cnt):
        mem = bits(v)
        if len(mem) >= min_members and c >= min_units:
            closed = int(cnt[((sets & ~v) == 0).all(1)].sum())
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
    groups: name -> members (laminar); three: [L, n] bool; cmask: [L, n, W] uint64 claim bitmask (box i = bit i % 64 of
    word i // 64); box_units: box -> units
    of its box tile; pair_units: 'a__b' -> units. Returns (assign: name -> mask, kept info, dropped info)."""
    order = sorted(groups, key=lambda k: (len(groups[k]), k))
    taken = np.zeros(three.shape, bool)
    assign, kept, dropped = {}, {}, {}
    for g in order:
        mem = groups[g]
        gm = np.zeros(cmask.shape[-1], np.uint64)
        for m in mem:
            gm[idx[m] // 64] |= np.uint64(1) << np.uint64(idx[m] % 64)
        subs = [k for k in kept if set(kept[k]["members"]) < set(mem)]
        direct = [k for k in subs if not any(set(kept[k]["members"]) < set(kept[j]["members"]) for j in subs)]
        covered = set().union(*[set(kept[k]["members"]) for k in direct]) if direct else set()
        dboxes = [m for m in mem if m not in covered and box_units.get(m, 0) > 0]
        dpairs = sorted(p for p, c in pair_units.items() if c and set(p.split("__")) <= set(mem)
                        and not any(set(p.split("__")) <= set(kept[k]["members"]) for k in direct))
        m = three & ~taken & ((cmask & ~gm) == 0).all(-1)
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


def check_against(ref, keys, get, unassigned, rep):
    """Refuse unless this partition equals the reference one with the group tiles folded back into the core."""
    rz = np.load(ref / "tiles.npz")
    rj = json.loads((ref / "partition.json").read_text())
    if rj.get("groups") or any(k.startswith("group:") for k in rz.files):
        raise SystemExit("check-against: the reference has group tiles; give a partition made without --groups")
    if rj["fingerprint"] != rep["fingerprint"] or rj["inputs"]["t2_sha256_12"] != rep["inputs"]["t2_sha256_12"]:
        raise SystemExit("check-against: another model or another T2")
    if rj.get("ratio") != rep["ratio"]:
        raise SystemExit(f"check-against: ratio {rj.get('ratio')} != {rep['ratio']}")
    if bool(rj.get("require_every_source")) != bool(rep.get("require_every_source")):
        raise SystemExit("check-against: one partition requires every_source and the other does not")
    mine = {k for k in keys if not k.startswith(("group:", "joint:"))}
    theirs = set(rz.files) - {"layers"}
    if theirs != mine:
        raise SystemExit(f"check-against: tile keys differ ({sorted(theirs ^ mine)[:6]} ...; a different "
                         "--min-pair, ratio or box set?)")
    if [int(x) for x in rz["layers"]] != rep["_layers"]:
        raise SystemExit("check-against: layer lists differ")
    folded = get("core").copy()
    for k in keys:
        if k.startswith(("group:", "joint:")):
            folded |= get(k)
    for k in sorted(mine):
        if not np.array_equal(rz[k], folded if k == "core" else get(k)):
            raise SystemExit(f"check-against: tile {k} differs from {ref}")
    if int(unassigned.sum()) != rj["core"]["unassigned"]:
        raise SystemExit("check-against: unassigned count differs")
    return {"reference": str(ref), "equal_with_groups_folded": True, "tiles_checked": len(mine)}


def claim_shares(sel, names, drive, outside):
    """Per claim (box, unit) of the units claimed by >= 2 boxes inside `outside`: drive / the strongest claimant's.
    drive: box -> (sorted flat positions of its original selection, drive there). Returns (max drive per unit, the
    concatenated shares)."""
    shape = next(iter(sel.values())).shape
    mx = np.zeros(int(np.prod(shape)), np.float32)
    cnt = np.zeros(mx.size, np.int16)
    looked = {}
    for nm in names:
        p = np.flatnonzero(sel[nm])
        pos, val = drive[nm]
        i = np.searchsorted(pos, p)
        if p.size and ((i >= pos.size).any() or (pos[np.minimum(i, pos.size - 1)] != p).any()):
            raise SystemExit(f"detach: box {nm} claims a unit outside its own selection (internal error)")
        v = val[i]
        looked[nm] = (p, v)
        mx[p] = np.maximum(mx[p], v)
        cnt[p] += 1
    multi = (cnt >= 2) & outside.ravel()
    sh = [v[multi[p]] / np.maximum(mx[p][multi[p]], 1e-30) for p, v in looked.values()]
    return mx, looked, cnt, (np.concatenate(sh) if sh else np.zeros(0, np.float32))


def detach_weak(sel, names, drive, share, outside, sweep=()):
    """Drop each claim whose drive is < share x the unit's strongest claimant's (Pierre 2026-11-19). Returns (new sel,
    report). `outside`: [L, n] bool, the units the report counts (not T2 or the English map)."""
    mx, looked, cnt, sh = claim_shares(sel, names, drive, outside)
    out = outside.ravel()
    qs = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9]
    rep = {"rule": "a claim is dropped where the box's drive (map mean_target) < share x the strongest claimant's; "
                   "applied before the pair, family, LCA, joint and core decisions",
           "share": share, "drive": "mean_target (per-token mean contribution on the box's target corpus)",
           "claims_on_multi_claimed_units": int(sh.size),
           "share_of_strongest_quantiles": dict(zip(map(str, qs), np.quantile(sh, qs).round(4).tolist()))
           if sh.size else {},
           "claims_below": {str(t): int((sh < t).sum()) for t in sorted({0.01, 0.025, 0.05, 0.1, 0.2, share} - {0})}}

    def counts(th):
        c = np.zeros(mx.size, np.int16)
        for p, v in looked.values():
            c[p[v >= th * mx[p]]] += 1
        return {"box": int(((c == 1) & out).sum()), "pair": int(((c == 2) & out).sum()),
                "three_plus": int(((c >= 3) & out).sum()), "unclaimed": int(((c == 0) & out).sum())}
    rep["claim_counts_outside_base"] = {str(t): counts(t) for t in sorted({0.0, share, *sweep})}
    rep["claim_counts_note"] = "claims at the detach (with --nested: after the descendant absorption, before the LCA"\
                               " roll-up)"
    new, per = {}, {}
    for nm in names:
        p, v = looked[nm]
        weak = v < share * mx[p]
        if weak.any():
            m = sel[nm].copy()                                  # never in place (nested code keeps views of maps)
            m.ravel()[p[weak]] = False
            new[nm] = m
        else:
            new[nm] = sel[nm]
        per[nm] = {"claims_detached": int(weak.sum()), "claims_detached_outside_base": int((weak & out[p]).sum())}
    rep["claims_detached"] = int(sum(x["claims_detached"] for x in per.values()))
    rep["claims_detached_outside_base"] = int(sum(x["claims_detached_outside_base"] for x in per.values()))
    rep["per_box"] = per
    return new, rep


def named_l(k, box, grep):
    """Tile key k names box (a pair, joint or kept family tile)."""
    if k.startswith("pair:"):
        return box in k[5:].split("__")
    if k.startswith("joint:"):
        return box in k[6:].split("+")
    return bool(grep) and box in grep["kept"].get(k[6:], {}).get("members", [])


def save_npz(path, items, get):
    """np.savez_compressed, one array at a time (pair masks are built as they are written, never all at once)."""
    import zipfile
    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for k, v in items:
            with zf.open(f"{k}.npy", "w", force_zip64=True) as fh:
                np.lib.format.write_array(fh, np.asanyarray(get(k) if v is None else v), allow_pickle=False)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", action="append", required=True, help="NAME=compare dir")
    ap.add_argument("--core-map", required=True)
    ap.add_argument("--t2", required=True)
    ap.add_argument("--ratio", type=float, default=0.25)
    ap.add_argument("--require-every-source", action="store_true",
                    help="a box selects a unit only where the map's every_source flag holds (source control)")
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
    ap.add_argument("--nested", action="store_true",
                    help="dotted box names are a taxonomy: an ancestor box's claim is dropped where a descendant box "
                         "claims the unit too (descendant absorption)")
    ap.add_argument("--not-nested", default="", help="comma list: boxes kept out of their ancestors' subtrees")
    ap.add_argument("--joint-floor", default="", help="N or auto: 3+ claim sets with >= N units -> joint tiles")
    ap.add_argument("--joint-sweep", default="", help="comma list of floors to report (auto picks among them)")
    ap.add_argument("--detach-share", type=float, default=0.0,
                    help="drop a claim whose drive is < this x the strongest claimant's (0: off; Pierre 2026-11-19)")
    ap.add_argument("--detach-sweep", default="", help="comma list of shares to report (with or without detaching)")
    ap.add_argument("--emit-detached", action="store_true", help="with --detach-share: write OUT/detached.npz (per "
                    "box the flat [L, n] positions of its dropped claims; amoeba_fn_interactions.py --detached)")
    ap.add_argument("--english-box", default="", help="tile name (lang.eng): English-map units outside T2 leave the "
                    "core for box:<name> (Pierre 2026-11-19)")
    ap.add_argument("--english-shared", action="store_true", help="English is a claimant: shared English tiles "
                    "(pair / family / joint) instead of exclusivity against English (Pierre 2026-11-19)")
    ap.add_argument("--english-closed-for", default="", help="comma list: boxes whose text is not English (the "
                    "English box's gate is closed when serving them; recorded for the battery)")
    a = ap.parse_args()
    if not 0.0 <= a.detach_share < 1.0:
        raise SystemExit("--detach-share: 0 (off) <= s < 1")
    try:
        dsweep = sorted({float(x) for x in a.detach_sweep.split(",") if x})
    except ValueError:
        raise SystemExit(f"--detach-sweep: a comma list of numbers, not {a.detach_sweep!r}") from None
    if any(not 0.0 <= x < 1.0 for x in dsweep):
        raise SystemExit("--detach-sweep: each share 0 <= s < 1")
    detaching = a.detach_share > 0 or bool(dsweep)
    if a.emit_detached and not a.detach_share > 0:
        raise SystemExit("--emit-detached needs --detach-share > 0")
    if a.english_closed_for and not a.english_box:
        raise SystemExit("--english-closed-for needs --english-box")
    if a.english_shared and not a.english_box:
        raise SystemExit("--english-shared needs --english-box")
    if a.english_shared and a.groups_auto:
        raise SystemExit("--english-shared with --groups-auto is not supported (auto families would absorb English)")
    if a.english_box and a.english_box in a.english_closed_for.split(","):
        raise SystemExit("--english-closed-for must not name the English box")
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
    sel, info, drive = {}, {}, {}
    if a.english_shared:
        en_sel, en_drive = np.zeros((L, n), bool), np.zeros((L, n), np.float32)
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
        before = int(sel[name].sum())
        if a.require_every_source:
            if any(f"every_source_{s}" not in z.files for s in range(S)):
                raise SystemExit(f"{name}: map has no every_source arrays (--require-every-source)")
            sel[name] &= np.stack([z[f"every_source_{s}"] for s in range(S)])
        if a.english_shared:                           # target units English drives too: shared with English
            ne = tm & ~(mb <= a.ratio * mt)
            if a.require_every_source:
                ne &= np.stack([z[f"every_source_{s}"] for s in range(S)])
            nonexcl = int(ne.sum())
            sel[name] = sel[name] | ne
            en_sel |= ne
            np.maximum(en_drive, np.where(ne, mb, 0.0).astype(np.float32), out=en_drive)
            del ne
        if detaching:                                  # drive at the box's own selection (sorted flat positions)
            p = np.flatnonzero(sel[name])
            drive[name] = (p, mt.ravel()[p].astype(np.float32))
        info[name] = {"map": str(d), "target_units": int(tm.sum()), "selected": int(sel[name].sum()),
                      "baseline_corpus": rep["baseline"]["corpus"], "target_corpus": rep["target"]["corpus"],
                      "capture_rev": rep.get("capture_rev")}
        if a.require_every_source:
            info[name]["selected_before_every_source"] = before
        if a.english_shared:
            info[name]["shared_with_english"] = nonexcl
    if a.english_box and (a.english_box in sel or a.english_box == "core"):
        raise SystemExit(f"--english-box {a.english_box}: also a box name")
    cm_core = core_map                                 # the English-map units the core (or English box) takes
    if a.english_shared:                               # English is a claimant: its map units join the shared rules
        own = core_map & ~t2
        en_sel |= own
        cmt = np.stack([cz[f"mean_target_{s}"] for s in range(S)]).astype(np.float32)
        np.maximum(en_drive, np.where(own, cmt, 0.0), out=en_drive)
        del cmt
        sel[a.english_box] = en_sel
        info[a.english_box] = {"map": a.core_map, "english_box": True, "english_map_units": int(own.sum()),
                               "shared_units": int((en_sel & ~own).sum()), "selected": int(en_sel.sum()),
                               "target_corpus": (crep.get("target") or {}).get("corpus")}
        if detaching:
            p = np.flatnonzero(en_sel)
            drive[a.english_box] = (p, en_drive.ravel()[p])
        cm_core = np.zeros_like(core_map)
        del en_drive, own
    names = sorted(sel)
    closed_for = [x for x in a.english_closed_for.split(",") if x]
    if any(x not in names for x in closed_for):
        raise SystemExit(f"--english-closed-for: unknown boxes {[x for x in closed_for if x not in names]}")
    drep = None
    detached = {}                                      # --emit-detached: box -> flat positions of dropped claims

    def run_detach():
        nonlocal sel, drep
        if not detaching:
            return
        new, drep = detach_weak(sel, names, drive, a.detach_share, ~t2 & ~cm_core, dsweep)
        if a.detach_share > 0 and a.emit_detached:
            for nm in names:
                if new[nm] is not sel[nm]:
                    dropped = np.flatnonzero(sel[nm] & ~new[nm])
                    detached[nm] = np.union1d(detached[nm], dropped) if nm in detached else dropped
        if a.detach_share > 0:
            sel = new
            for nm in names:
                info[nm]["selected_before_detach"] = info[nm]["selected"]
                info[nm]["selected"] = int(sel[nm].sum())
                info[nm]["claims_detached"] = drep["per_box"][nm]["claims_detached"]
        else:
            drep["share"] = None
            drep["rule"] += " (sweep only: nothing detached)"
            for x in drep["per_box"].values():
                x["claims_detached"] = x["claims_detached_outside_base"] = 0
            drep["claims_detached"] = drep["claims_detached_outside_base"] = 0
    nested = None
    if a.not_nested and not a.nested:
        raise SystemExit("--not-nested needs --nested")
    if a.nested:
        outside = [x for x in a.not_nested.split(",") if x]
        if any(x not in names for x in outside):
            raise SystemExit(f"--not-nested: unknown boxes {[x for x in outside if x not in names]}")
        orig = {nm: sel[nm] for nm in names}
        nested = {"rule": "a box's claim on a unit is dropped where any descendant box (dotted-name prefix) also "
                          "claims it, so a unit only a subtopic and its ancestors claim counts as the subtopic's",
                  "ancestors": {}, "descendants": {}, "not_nested": outside}
        below = lambda x, o: x == o or x.startswith(o + ".")                     # noqa: E731
        # x is cut from P's subtree when it is (or lies under) a --not-nested box strictly below P
        cut = lambda P, x: any(below(x, o) and o.startswith(P + ".") for o in outside)   # noqa: E731
        for nm in names:
            desc = [x for x in names if x.startswith(nm + ".") and not cut(nm, x)]
            anc = [y for y in names if nm.startswith(y + ".") and not cut(y, nm)]
            if anc:
                nested["ancestors"][nm] = anc
            if desc:
                nested["descendants"][nm] = desc
            if desc:
                any_desc = np.zeros_like(orig[nm])
                for x in desc:
                    any_desc |= orig[x]
                sel[nm] = orig[nm] & ~any_desc
            info[nm]["selected_before_nesting"] = int(orig[nm].sum())
            info[nm]["selected"] = int(sel[nm].sum())
            info[nm]["descendants"] = desc
        if detaching:                                  # an absorbed claim keeps the ancestor's drive (review: a
            for nm in names:                           # strong parent's unit must not be detached as the child's)
                anc = nested["ancestors"].get(nm, [])
                if not anc or nm not in drive:
                    continue
                pos, val = drive[nm]
                val = val.copy()
                for y in anc:
                    yd = np.zeros(L * n, np.float32)
                    yd[drive[y][0]] = drive[y][1]
                    np.maximum(val, yd[pos], out=val)
                    del yd
                drive[nm] = (pos, val)
        run_detach()                                   # after the descendant absorption, before the LCA roll-up
        # Pierre 2026-11-18 ("Give it to the parent"; "a would hold it ... parent tiles are always gated in for
        # their subtiles"): a unit whose remaining claimants (after the drop above) are >= 2 boxes all inside one box
        # P's subtree goes to P alone, selecting or not: the lowest common ancestor box (parents decided deepest
        # first, so a rolled-up unit is P's single claim when P's ancestors are judged); claimants with no common
        # ancestor box stay as they are.
        total = np.zeros((L, n), np.int16)
        for nm in names:
            total += sel[nm]
        depth = lambda x: x.count(".")                                          # noqa: E731
        for P in sorted(nested["descendants"], key=lambda x: (-depth(x), x)):
            desc = nested["descendants"][P]
            inside = np.zeros((L, n), np.int16)
            for x in desc:
                inside += sel[x]
            m = (inside >= 2) & (inside == total)
            got = int(m.sum())
            info[P]["received_from_subtopics"] = got
            info[P]["received_lca_selected"] = int((m & orig[P]).sum())
            info[P]["received_lca_unselected"] = got - info[P]["received_lca_selected"]
            info[P]["received_from_subtopics_outside_base"] = int((m & ~t2 & ~cm_core).sum())   # not core anyway
            if got:
                for x in desc:
                    if (sel[x] & m).any():
                        sel[x] &= ~m           # in place (a leaf's array is orig's; orig is read only for parents)
                sel[P] |= m                    # sel[P] is a fresh array (made by the drop above), never orig[P]
                total[m] = 1
        for nm in names:
            info[nm]["selected"] = int(sel[nm].sum())
        nested["rule"] += ("; then a unit claimed by >= 2 boxes all inside one box's subtree goes to their lowest "
                           "common ancestor box, selecting or not (Pierre 2026-11-18: give it to the parent; parent "
                           "tiles are always gated in for their subtiles)")
        del orig, total
    else:
        run_detach()
    drive.clear()
    claims = np.zeros((L, n), np.int16)
    for nm in names:
        claims += sel[nm]
    core = t2 | cm_core | (claims >= 3)
    tiles = {"core": core.copy()}
    for nm in names:
        tiles[f"box:{nm}"] = sel[nm] & (claims == 1) & ~core
    pairs = Counter()
    two = (claims == 2) & ~core
    idx = {nm: i for i, nm in enumerate(names)}
    B = len(names)
    pair_pos = {}                          # pair key -> flat unit positions (masks are built only when written)
    if two.any():
        P = np.flatnonzero(two)
        sub = np.stack([sel[nm].ravel()[P] for nm in names])                         # [B, K]: exactly 2 per column
        lo = sub.argmax(0).astype(np.int64)
        hi = (B - 1 - sub[::-1].argmax(0)).astype(np.int64)
        del sub
        pid = lo * B + hi
        order = np.argsort(pid, kind="stable")
        vals, start, cnt = np.unique(pid[order], return_index=True, return_counts=True)   # (lo, hi) ascending
        for v, s0, c in zip(vals, start, cnt):
            x, y = names[int(v) // B], names[int(v) % B]
            pos = P[order[s0:s0 + c]]
            if c < a.min_pair:
                tiles["core"].ravel()[pos] = True
            else:
                pair_pos[f"pair:{x}__{y}"] = pos
                pairs[f"{x}__{y}"] = int(c)
        del P, pid, order
    pair_any = np.zeros((L, n), bool)
    for pos in pair_pos.values():
        pair_any.ravel()[pos] = True

    def get(k):
        if k in joint_pos:
            m = np.zeros((L, n), bool)
            m.ravel()[joint_pos[k]] = True
            return m
        if k in pair_pos:
            m = np.zeros((L, n), bool)
            m.ravel()[pair_pos[k]] = True
            return m
        return tiles[k]

    unassigned = ~(tiles["core"] | pair_any |
                   np.logical_or.reduce([v for k, v in tiles.items() if k != "core"] or [np.zeros_like(core)]))
    tiles["core"] |= unassigned
    grep = None
    jointing = bool(a.joint_floor or a.joint_sweep)
    if a.joint_floor == "auto" and not a.joint_sweep:
        raise SystemExit("--joint-floor auto needs --joint-sweep")
    declared = load_groups(a.groups, names) if a.groups else {}
    if grouping or jointing:
        cmask = np.zeros((L, n, (B + 63) // 64), np.uint64)                          # box i: bit i % 64 of word i // 64
        for nm in names:
            cmask[..., idx[nm] // 64] |= sel[nm].astype(np.uint64) << np.uint64(idx[nm] % 64)
        three = (claims >= 3) & ~t2 & ~cm_core
    if grouping:
        for x in declared:
            for y in declared:
                if x < y and not laminar(declared[x], declared[y]):
                    raise SystemExit(f"declared groups {x} and {y} overlap without nesting (groups must be laminar)")
        auto_log = []
        groups = dict(declared)
        if a.groups_auto:
            vals, cnts = np.unique(cmask[three], axis=0, return_counts=True)
            prop, auto_log = propose_groups(vals, cnts, names, a.auto_min_units, a.auto_min_members, declared)
            groups.update(prop)
        bu = {nm: int(tiles[f"box:{nm}"].sum()) for nm in names}
        assign, kept, dropped = decide_groups(groups, three, cmask, idx, bu, dict(pairs), a.group_min_units)
        if nested is not None:                         # LCA rule: a family whose root box exists never gets units
            for g, d in dropped.items():
                root = [r for r in d["members"] if set(d["members"]) - {r} <= set(nested["descendants"].get(r, []))]
                if d["reason"] == "no units" and root:
                    d["reason"] = (f"no units: every claim set inside lies in box {root[0]}'s subtree, so the LCA "
                                   "rule gives it to that box (--nested)")
        for g, m in assign.items():
            tiles[f"group:{g}"] = m
            tiles["core"] &= ~m
        if a.english_shared and kept:                  # English + 2+ boxes of one kept family -> group:<family>+EN
            EB = a.english_box
            ebit = np.zeros(cmask.shape[-1], np.uint64)
            ebit[idx[EB] // 64] |= np.uint64(1) << np.uint64(idx[EB] % 64)
            has_en = ((cmask & ebit) != 0).any(-1)
            for g in sorted(list(kept), key=lambda k: (len(kept[k]["members"]), k)):
                gm = ebit.copy()
                for mbr in kept[g]["members"]:
                    gm[idx[mbr] // 64] |= np.uint64(1) << np.uint64(idx[mbr] % 64)
                m = three & has_en & tiles["core"] & ((cmask & ~gm) == 0).all(-1)
                if not m.any():
                    continue
                gk = f"{g}+{EB}"
                tiles[f"group:{gk}"] = m
                tiles["core"] &= ~m
                assign[gk] = m
                kept[gk] = {"members": sorted(kept[g]["members"] + [EB]), "units": int(m.sum()), "subgroups": [],
                            "boxes": [], "pairs_inside": [], "parent": g, "english_of": g}
            del has_en
        FFg = F or crep["per_site"][0].get("F") or 640
        for g in kept:
            kept[g].update(share=float(assign[g].mean()), expert_equivalents=round(kept[g]["units"] / FFg, 1),
                           whole_experts=int((assign[g].reshape(L, -1, FFg).sum(2) == FFg).sum()),
                           source="declared" if g in declared else "english" if kept[g].get("english_of") else "auto")
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
    joint_pos, jrep = {}, None
    if jointing:
        rest = three & tiles["core"]                   # 3+ units the families and nesting left in the core
        P = np.flatnonzero(rest)
        vals, inv, cnts = np.unique(cmask.reshape(-1, cmask.shape[-1])[P], axis=0, return_inverse=True,
                                    return_counts=True)
        inv = inv.ravel()
        bits = lambda v: [names[i] for i in range(B) if (int(v[i // 64]) >> (i % 64)) & 1]   # noqa: E731
        mem = [bits(v) for v in vals]
        fam = dict(declared)                            # declared (kept or dropped) and kept auto groups
        if grep is not None:
            fam.update({g: v["members"] for g, v in grep["kept"].items()})
        eligible = np.array([len(m) >= 3 and not any(set(m) <= set(g) for g in fam.values()) for m in mem], bool)
        tot, core_units = L * n, int(tiles["core"].sum())
        sweep = {}
        for f in sorted({int(x) for x in a.joint_sweep.split(",") if x}):
            ok_ = eligible & (cnts >= f)
            k, moved = int(ok_.sum()), int(cnts[ok_].sum())
            sweep[str(f)] = {"tiles": k, "units_moved": moved, "core_share_after": (core_units - moved) / tot,
                             "core_share_points_per_tile": round(100 * moved / tot / k, 4) if k else 0.0}
        why = None
        if a.joint_floor == "auto":
            best = max(sweep, key=lambda f: (sweep[f]["units_moved"] / max(sweep[f]["tiles"], 1), -int(f)))
            floor = int(best)
            why = (f"auto: floor {best} has the largest core reduction per tile among {list(sweep)} "
                   f"({sweep[best]['core_share_points_per_tile']} core-share points per tile)")
        else:
            try:
                floor = int(a.joint_floor) if a.joint_floor else None
            except ValueError:
                raise SystemExit(f"--joint-floor: an integer or auto, not {a.joint_floor!r}") from None
        jt = {}
        if floor is not None:
            order = np.argsort(inv, kind="stable")
            starts = np.concatenate([[0], np.cumsum(np.bincount(inv, minlength=len(vals)))])
            for j in np.flatnonzero(eligible & (cnts >= floor)):
                key = "joint:" + "+".join(mem[j])
                pos = P[order[starts[j]:starts[j + 1]]]
                joint_pos[key] = pos
                tiles["core"].ravel()[pos] = False
                jt[key[6:]] = {"members": mem[j], "units": int(cnts[j]), "share": float(cnts[j] / tot)}
        jrep = {"rule": "an exact claim set of >= 3 boxes still in the core (" +
                        ("no common ancestor box, " if nested is not None else "") +
                        "not inside any declared or kept group) with >= floor units -> joint:<members>; gate = OR of "
                        "the members' gates" + ("" if floor is not None else " (sweep only: no tiles written)"),
                "floor": floor, "floor_choice": why, "sweep": sweep,
                "core_units_before": core_units, "units_moved": int(sum(v["units"] for v in jt.values())),
                "tiles": dict(sorted(jt.items(), key=lambda x: -x[1]["units"])),
                "eligible_claim_sets": int(eligible.sum()), "eligible_units": int(cnts[eligible].sum())}
    if grouping or jointing:
        del cmask
    erep = None
    if a.english_box and not a.english_shared:         # Pierre 2026-11-19: English gets its own box on FN
        em = core_map & ~t2
        if (em & ~tiles["core"]).any():
            raise SystemExit("english box: an English-map unit outside T2 is not in the core (internal error)")
        ek = f"box:{a.english_box}"
        tiles["core"] &= ~em
        tiles[ek] = em
        ec = int(em.sum())
        erep = {"name": a.english_box, "tile": ek, "units": ec, "share": float(em.mean()),
                "rule": "the English-map units outside T2 (English vs the 10-language mix, compare map --core-map) "
                        "leave the core for this box; topic boxes' claims do not change that (as when it was core)",
                "gate": "the text is English: open when serving every box except closed_for",
                "closed_for": closed_for,
                "claimed_by_topic_boxes": {"1": int((em & (claims == 1)).sum()), "2": int((em & (claims == 2)).sum()),
                                           "3+": int((em & (claims >= 3)).sum())},
                "map": a.core_map}
    if a.english_shared:
        EB = a.english_box
        named = lambda k: EB in (k[5:].split("__") if k.startswith("pair:") else   # noqa: E731
                                 k[6:].split("+") if k.startswith("joint:") else
                                 (grep["kept"][k[6:]]["members"] if grep and k[6:] in grep["kept"] else []))
        units = lambda k: int(len(pair_pos[k]) if k in pair_pos else len(joint_pos[k]) if k in joint_pos   # noqa: E731
                              else tiles[k].sum())
        ek = [k for k in list(pair_pos) + list(joint_pos) + [k for k in tiles if k.startswith("group:")] if named(k)]
        by = {"box": int(tiles[f"box:{EB}"].sum()), "pair": 0, "family": 0, "joint": 0}
        for k in ek:
            by["pair" if k.startswith("pair:") else "joint" if k.startswith("joint:") else "family"] += units(k)
        closed = {}
        for L_ in closed_for:                       # serving a language box: English-naming tiles that stay closed
            closed[L_] = by["box"] + sum(units(k) for k in ek if not named_l(k, L_, grep))
        erep = {"name": EB, "tile": f"box:{EB}", "units": by["box"], "share": float(tiles[f"box:{EB}"].mean()),
                "rule": "English is a claimant (--english-shared): its shared units go to pair / family / joint "
                        "tiles like any box's", "gate": "the text is English: open when serving every box except "
                        "closed_for", "closed_for": closed_for, "map": a.core_map,
                "shared": {"tiles": len(ek), "units_by_kind": by, "units_naming_english": sum(by.values()),
                           "closed_when": "an English-naming tile is open for all English text (its gate is the OR "
                                          "of its members' and English is one); it is closed only while serving a "
                                          "closed_for box it does not name",
                           "units_closed_serving": closed, "tiles_list": sorted(ek)[:200]}}
    cover = np.zeros((L, n), np.int16)
    for v in tiles.values():
        cover += v
    for pos in list(pair_pos.values()) + list(joint_pos.values()):
        cover.ravel()[pos] += 1
    if not (cover == 1).all():
        raise SystemExit(f"partition is not an exact cover: {int((cover != 1).sum())} units off")
    del cover
    keys = ["core"] + [k for k in tiles if k.startswith("box:")] + list(pair_pos) + \
        [k for k in tiles if k.startswith("group:")] + list(joint_pos)
    if not a.no_tiles:
        save_npz(out / "tiles.npz", [("layers", np.array(layers))] + [(k, None) for k in keys], get)
    if a.emit_unassigned:
        np.savez_compressed(out / "unassigned.npz", layers=np.array(layers), unassigned=unassigned)
    if a.emit_detached:
        np.savez_compressed(out / "detached.npz", layers=np.array(layers), **detached)
    tot = L * n
    FF = F or crep["per_site"][0].get("F") or 640      # neurons per expert (T2 in expert units gives it directly)
    rep = {"schema": "mlx2.amoeba-fn-partition.v1", "checkpoint": crep["checkpoint"], "fingerprint": crep["fingerprint"],
           "rule": f"exclusivity r={a.ratio}; core = T2 | English map | >=3 boxes | unassigned; 2 boxes -> pair tile; "
                   "1 box -> box", "units_total": tot, "ratio": a.ratio,
           "core": {"units": int(tiles["core"].sum()), "share": float(tiles["core"].mean()),
                    "from_t2": int(t2.sum()), "from_english_map_outside_t2": int((cm_core & ~t2).sum()),
                    "from_3plus_boxes": int(((claims >= 3) & ~t2 & ~cm_core).sum()),
                    "unassigned": int(unassigned.sum()),
                    "small_pairs_spilled": int(tiles["core"].sum() - (t2 | cm_core | (claims >= 3)).sum()
                                               - unassigned.sum())},
           "boxes": {nm: {**info[nm], "units": int(tiles[f"box:{nm}"].sum()),
                          "share": float(tiles[f"box:{nm}"].mean()),
                          "whole_experts": int((tiles[f"box:{nm}"].reshape(L, -1, FF).sum(2) == FF).sum())}
                     for nm in names},
           "pairs": dict(pairs.most_common()),
           "inputs": {"core_map": a.core_map, "t2": a.t2,
                      "t2_sha256_12": hashlib.sha256(Path(a.t2).read_bytes()).hexdigest()[:12]}}
    if a.require_every_source:
        rep["require_every_source"] = True
        rep["rule"] += "; a box selects a unit only where its target holds in every source (every_source)"
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
    if jrep is not None:
        rep["joint"] = jrep
        rep["core"]["to_joint_tiles"] = jrep["units_moved"]
        rep["core"]["small_pairs_spilled"] += jrep["units_moved"]               # the formula above counts them out
        if jrep["floor"] is not None:
            rep["rule"] += f"; joint tiles: 3+ claim sets left in the core with >= {jrep['floor']} units"
        if "from_3plus_boxes_kept" in rep["core"]:
            rep["core"]["from_3plus_boxes_kept"] -= jrep["units_moved"]
    if drep is not None:
        rep["detach"] = drep
        if a.detach_share > 0:
            rep["rule"] += (f"; weak-claimant detach: a claim < {a.detach_share} x the strongest claimant's drive is "
                            "dropped first")
    if erep is not None and a.english_shared:
        rep["english_box"] = erep
        rep["rule"] += f"; English shared: {a.english_box} is a claimant (no exclusivity against English)"
    elif erep is not None:
        rep["english_box"] = erep
        rep["boxes"][a.english_box] = {"map": a.core_map, "units": erep["units"], "share": erep["share"],
                                       "english_box": True, "target_corpus": (crep.get("target") or {}).get("corpus"),
                                       "baseline_corpus": (crep.get("baseline") or {}).get("corpus"),
                                       "whole_experts": int((em.reshape(L, -1, FF).sum(2) == FF).sum())}
        rep["core"]["to_english_box"] = erep["units"]
        rep["core"]["from_english_map_outside_t2"] = 0                       # moved to the English box
        rep["core"]["small_pairs_spilled"] += erep["units"]                    # the formula above counts them out
        rep["rule"] = rep["rule"].replace("core = T2 | English map |", f"core = T2 | (English map -> box {a.english_box}) |")
    if nested is not None:
        rep["nested"] = nested
        rep["rule"] += "; nested: an ancestor's claim is dropped where a descendant box claims the unit"
    if a.check_against:
        rep["check_against"] = check_against(Path(a.check_against), keys, get, unassigned, {**rep, "_layers": layers})
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

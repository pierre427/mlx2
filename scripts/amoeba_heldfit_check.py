#!/usr/bin/env python3
"""Held / fit disjointness check between two per-box file sets (scripts/amoeba_category_box_files.py outputs). CPU.

Why (2026-11-18 night): category builds 98055e0a6 and fbf39cd15 have a split-skew bug (selections reused the split
hash, amoeba-corpus af51149; fixed in mlx2 8ec12cb93), so their held / val rows are not split-balanced. Captures read
only the fit rows of those builds; the battery's held and val rows come from a corrected build. The split rule is
sha256(group) % 8 in every build, so the two stay group-disjoint; this proves it on the files actually used.

--fit-files DIR: the box files whose fit rows were captured (every box, fit split only).
--held-files DIR: the box files the battery reads (the --splits rows, default held,val, of every box).
Refuses (exit 4) if any group or any text (the row's sha256, else the sha256 of its text) of a held-side row occurs in
any fit-side row of any box. Balance (exit 3): the skew shows as a register or source whose groups never reach held /
val (DOAB fit-only, the sec web register fit-only), so per box and per register and per source (prefix before ':') of
the held files (all their splits), with G distinct groups, the held-side groups must be at least --min-share (0.25) of
the 2/8 x G a sha256(group) % 8 split expects, wherever G >= --min-groups (24, i.e. >= 6 expected), and each held
split alone at least --min-share of its G/8 wherever G/8 >= --min-split-expected (10). This checks the property on
the files the battery reads, whatever revision built them. OUT.json records the counts either way, and
skipped_boxes: per unbalanced box "skipped: held split unbalanced (<cells>)" (Pierre 2026-11-18: "skip the box with a
note"; the caller drops those boxes from the battery). Exit 4 (leakage) takes precedence over 3 (skew); 2 is
argparse's usage error.

--groups-from RAW_DIR takes each cell's group total G from those files (the held files before a purge,
scripts/amoeba_held_purge.py, which removes whole groups) and the held-side counts from --held-files.

Usage: amoeba_heldfit_check.py --fit-files DIR --held-files DIR --out OUT.json [--splits held,val] [--groups-from RAW]
       [--min-groups 24] [--min-share 0.25]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def rows(path):
    with open(path) as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def norm_group(g):
    """The record's group without a routing suffix (agent_sources: a conversation is g in the tool boxes and g#defs in
    agent.tool_use.definitions, and the salted selection hash moves records between them across builds)."""
    return None if g is None else str(g).split("#", 1)[0]


def text_key(r):
    return r.get("sha256") or hashlib.sha256(r["text"].encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-files", required=True)
    ap.add_argument("--held-files", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="held,val")
    ap.add_argument("--min-groups", type=int, default=24)
    ap.add_argument("--min-share", type=float, default=0.25)
    ap.add_argument("--min-split-expected", type=float, default=10.0)
    ap.add_argument("--groups-from", default="", help="raw held files: the group totals G (after a purge)")
    a = ap.parse_args()
    fd, hd = Path(a.fit_files), Path(a.held_files)
    fm, hm = (json.loads((d / "manifest.json").read_text()) for d in (fd, hd))
    splits = {x for x in a.splits.split(",") if x}
    fit_groups, fit_texts, fit_n = set(), set(), {}
    for b in fm["boxes"]:
        n = 0
        for r in rows(fd / f"{b}.jsonl"):
            if r.get("split") == "fit":
                fit_groups.add(norm_group(r.get("group")))
                fit_texts.add(text_key(r))
                n += 1
        fit_n[b] = n
    fit_groups.discard(None)
    per_box, bad_groups, bad_texts = {}, {}, {}
    balance, skewed, skip_cells = {}, [], {}
    for b in hm["boxes"]:
        n = g = t = 0
        cells = {}                                     # (kind, value) -> [all groups, {split: groups}]
        for r in rows(hd / f"{b}.jsonl"):
            grp = r.get("group")
            for key in (("box", "*"), ("register", str(r.get("register"))),
                        ("source", str(r.get("source") or "").split(":", 1)[0])):
                c = cells.setdefault(key, [set(), {}])
                c[0].add(grp)
                if r.get("split") in splits:
                    c[1].setdefault(r["split"], set()).add(grp)
            if r.get("split") not in splits:
                continue
            n += 1
            if r.get("group") is None:
                raise SystemExit(f"{b}: held-side row {r.get('id')} has no group")
            if norm_group(r["group"]) in fit_groups:          # g and g#defs are one conversation
                g += 1
                bad_groups.setdefault(b, []).append(r["group"])
            if text_key(r) in fit_texts:
                t += 1
                bad_texts.setdefault(b, []).append(r.get("id"))
        per_box[b] = {"held_side_rows": n, "group_in_fit": g, "text_in_fit": t}
        if a.groups_from:                              # G from the raw files: a purged group still counts as expected
            for c in cells.values():
                c[0] = set()
            for r in rows(Path(a.groups_from) / f"{b}.jsonl"):
                for key in (("box", "*"), ("register", str(r.get("register"))),
                            ("source", str(r.get("source") or "").split(":", 1)[0])):
                    cells.setdefault(key, [set(), {}])[0].add(r.get("group"))
        balance[b] = {}
        for (kind, val), (allg, by) in sorted(cells.items()):
            G = len(allg)
            H = len(set().union(*by.values())) if by else 0
            exp = G * len(splits) / 8
            row = {"groups": G, "held_side_groups": H, "expected": round(exp, 1),
                   "per_split": {sp: len(by.get(sp, ())) for sp in sorted(splits)}}
            why = []
            if G >= a.min_groups and H < a.min_share * exp:
                why.append(f"{H} held-side groups of {G} (expected ~{exp:.0f})")
            for sp in sorted(splits):                  # one split skipped (e.g. val only): >= 10 expected per split
                e1 = G / 8
                if e1 >= a.min_split_expected and len(by.get(sp, ())) < a.min_share * e1:
                    why.append(f"{len(by.get(sp, ()))} {sp} groups of {G} (expected ~{e1:.0f})")
            if why:
                row["skewed"] = True
                skewed.append(f"{b} {kind}={val}: " + "; ".join(why))
                skip_cells.setdefault(b, []).append(f"{kind}={val}: " + "; ".join(why))
            balance[b][f"{kind}={val}"] = row
    ok = not bad_groups and not bad_texts
    rep = {"schema": "mlx2.amoeba-heldfit-check.v1", "fit_files": str(fd), "held_files": str(hd),
           "fit_builds": fm.get("builds") or [{"build": fm.get("build"), "corpus_sha256": fm.get("corpus_sha256")}],
           "held_builds": hm.get("builds") or [{"build": hm.get("build"), "corpus_sha256": hm.get("corpus_sha256")}],
           "held_splits": sorted(splits), "fit_rows": sum(fit_n.values()), "fit_groups": len(fit_groups),
           "disjoint": ok, "balanced": not skewed, "skewed": skewed,
           "skipped_boxes": {b: "skipped: held split unbalanced (" + "; ".join(c) + ")" for b, c in skip_cells.items()},
           "groups_from": a.groups_from or None,
           "balance_rule": {"min_groups": a.min_groups, "min_share": a.min_share, "splits_per_8": len(splits),
                            "min_split_expected": a.min_split_expected},
           "per_box": per_box, "balance": balance,
           "overlap_examples": {b: v[:5] for b, v in list(bad_groups.items())[:10]},
           "text_overlap_examples": {b: v[:5] for b, v in list(bad_texts.items())[:10]}}
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(f"held/fit check: {sum(v['held_side_rows'] for v in per_box.values())} held-side rows of {len(per_box)} boxes "
          f"vs {rep['fit_rows']} fit rows ({len(fit_groups)} groups): " + ("disjoint" if ok else
          f"OVERLAP groups {sum(len(v) for v in bad_groups.values())}, texts {sum(len(v) for v in bad_texts.values())}"))
    for x in skewed[:20]:
        print("SKEWED", x)
    if not ok:
        return 4
    return 0 if not skewed else 3


if __name__ == "__main__":
    sys.exit(main())

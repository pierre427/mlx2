#!/usr/bin/env python3
"""Purge held / val rows that leak into captured fit rows across two builds (CPU).

Why (2026-11-18 night, the catboxes preview): the captures read the fit rows of builds 98055e0a6 / fbf39cd15, the
battery reads held / val of the split-fixed build 8ec12cb93. The split rule is sha256(group) % 8 in both, and the
agent sources' group key (conv_key, a function of the record id) is unchanged, but two things put the same document
on both sides: (a) 622 held / val rows carry the text of an old fit row under another record's group (duplicate
texts; the dedup kept another duplicate this time; glaive, nemotron_agentic, toucan, one re-sourced arXiv abstract);
(b) the salted selection hash moves a conversation between agent.tool_use.definitions (group g#defs) and the other
tool / harness boxes (group g), and g and g#defs get independent splits. So texts are compared exactly and groups
with the routing suffix stripped.

This copies --held-files to --out, dropping from the --splits rows (held,val): every row whose text (row sha256, else
the sha256 of its text) or group (routing suffix "#..." stripped: g and g#defs are one conversation) occurs in any
fit row of --fit-files, and every held / val row, in any box, of a
group that had such a row (decided over all boxes first). Other splits are copied unchanged. A purged group leaves
the file entirely (a group has one split), so run the split-balance check with --groups-from the raw held files.
Writes OUT.tmp, then renames; manifest.json gains rows (recounted) and purge (per box: rows and groups dropped).

Usage: amoeba_held_purge.py --fit-files BOXFILES --held-files HELDFILES --out DIR [--splits held,val]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path


def rows(path):
    with open(path) as fh:
        for line in fh:
            if line.strip():
                yield line, json.loads(line)


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
    a = ap.parse_args()
    fd, hd, out = Path(a.fit_files), Path(a.held_files), Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing existing {out}")
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    splits = {x for x in a.splits.split(",") if x}
    fm, hm = (json.loads((d / "manifest.json").read_text()) for d in (fd, hd))
    fit_texts, fit_groups = set(), set()
    for b in fm["boxes"]:
        for _, r in rows(fd / f"{b}.jsonl"):
            if r.get("split") == "fit":
                fit_texts.add(text_key(r))
                fit_groups.add(norm_group(r.get("group")))
    tmp.mkdir(parents=True)
    purge, counts, regs = {}, defaultdict(Counter), defaultdict(Counter)
    bad_groups, why = set(), defaultdict(Counter)
    for b in hm["boxes"]:                                    # pass 1, every box: groups that touch the fit rows
        for _, r in rows(hd / f"{b}.jsonl"):
            if r.get("split") in splits:
                if text_key(r) in fit_texts:
                    bad_groups.add(norm_group(r.get("group")))
                    why[b]["text in fit"] += 1
                elif norm_group(r.get("group")) in fit_groups:
                    bad_groups.add(norm_group(r.get("group")))
                    why[b]["group in fit"] += 1
    for b in hm["boxes"]:
        dropped, gone = 0, set()
        with open(tmp / f"{b}.jsonl", "w") as fo:            # pass 2: copy, dropping those groups' held-side rows
            for line, r in rows(hd / f"{b}.jsonl"):
                if r.get("split") in splits and norm_group(r.get("group")) in bad_groups:
                    dropped += 1
                    gone.add(norm_group(r.get("group")))
                    continue
                fo.write(line if line.endswith("\n") else line + "\n")
                counts[b][r["split"]] += 1
                regs[b][f"{r['split']}:{r.get('register')}"] += 1
        if dropped:
            purge[b] = {"rows_dropped": dropped, "groups_dropped": len(gone), "matched": dict(why[b])}
    man = dict(hm)
    man.update(rows={b: dict(counts[b]) for b in hm["boxes"]}, registers={b: dict(regs[b]) for b in hm["boxes"]},
               purge={"rule": "held / val rows whose text or group occurs in a fit row of the fit files, and every "
                              "held / val row of their groups, dropped", "fit_files": str(fd), "held_files": str(hd),
                      "splits": sorted(splits), "boxes": purge, "groups_dropped": len(bad_groups),
                      "rows_dropped": sum(v["rows_dropped"] for v in purge.values())})
    (tmp / "manifest.json").write_text(json.dumps(man, indent=1))
    os.replace(tmp, out)
    print(f"purged {man['purge']['rows_dropped']} held-side rows in {len(purge)} boxes -> {out}")


if __name__ == "__main__":
    sys.exit(main())

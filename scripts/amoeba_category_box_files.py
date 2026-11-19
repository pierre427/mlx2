#!/usr/bin/env python3
"""Per-box corpus files from an aMoEba category build (amoeba-corpus builds/category-boxes-<rev>: one
corpus.jsonl.gz, one row per (doc, box) with the doc repeated under every ancestor box). CPU only.

The firing capture (scripts/amoeba_firing_capture.py) and the box battery (scripts/amoeba_box_battery.py) read a plain
JSONL whole and filter it by split, so one 700k-row gzip read once per box would cost minutes and gigabytes per box.
This writes OUT/<box>.jsonl with every row of that box (all splits, rows unchanged, build order) for the chosen boxes,
and OUT/manifest.json (the build, its corpus sha256 checked against the build manifest, per-box row counts by split
and register, the boxes left out and why, this script's revision).

Boxes: the build's meets_spec list (--boxes ALL_AT_SPEC, the default) or a comma list; --exclude-partition P.json
leaves out every box of that partition (boxes already captured by an earlier run) and --exclude a,b more.

--overlay BUILD:PREFIX[,PREFIX] takes the at-spec boxes at or under those prefixes from a second build (a rebuild of
one subtree, e.g. sec.*) and drops the primary's boxes there; the manifest then lists builds and box_build.

Usage: amoeba_category_box_files.py --build ~/Desktop/amoeba-corpus/builds/category-boxes-98055e0a6 --out DIR
       [--boxes ALL_AT_SPEC|a,b,...] [--exclude-partition PARTITION/partition.json] [--exclude a,b]
       [--overlay BUILD:sec]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_build(b):
    b = Path(b).expanduser()
    man = json.loads((b / "manifest.json").read_text())
    corpus = b / man.get("corpus_path", man.get("corpus", "corpus.jsonl.gz"))
    got = sha256(corpus)
    if got != man["corpus_sha256"]:
        raise SystemExit(f"{corpus}: sha256 {got} != manifest {man['corpus_sha256']}")
    return b, man, corpus, got


def extract(corpus, man, boxes, tmp, counts, regs):
    fhs = {x: open(tmp / f"{x}.jsonl", "w") for x in boxes}              # noqa: SIM115
    try:
        with gzip.open(corpus, "rt") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                x = r["box"]
                if x in fhs:
                    fhs[x].write(line if line.endswith("\n") else line + "\n")
                    counts[x][r["split"]] += 1
                    regs[x][f"{r['split']}:{r.get('register')}"] += 1
    finally:
        for f in fhs.values():
            f.close()
    for x in boxes:
        n = sum(counts[x].values())
        if n != man["boxes"][x]["docs"]:
            raise SystemExit(f"{x}: {n} rows != build manifest docs {man['boxes'][x]['docs']}")


def under(x, prefixes):
    return any(x == p or x.startswith(p + ".") for p in prefixes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boxes", default="ALL_AT_SPEC")
    ap.add_argument("--exclude-partition", default="")
    ap.add_argument("--exclude", default="")
    ap.add_argument("--overlay", default="", help="BUILD_DIR:PREFIX[,PREFIX]: the at-spec boxes at or under those "
                    "prefixes come from this build instead (e.g. a sec.* rebuild); the primary's boxes there are dropped")
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    if out.exists():
        raise SystemExit(f"refusing existing {out}")
    b, man, corpus, got = load_build(a.build)
    want = list(man["meets_spec"]) if a.boxes == "ALL_AT_SPEC" else [x for x in a.boxes.split(",") if x]
    unknown = [x for x in want if x not in man["boxes"]]
    if unknown:
        raise SystemExit(f"boxes not in the build: {unknown}")
    excluded = {}
    if a.exclude_partition:
        for x in json.loads(Path(a.exclude_partition).read_text())["boxes"]:
            excluded[x] = f"a box of {a.exclude_partition}"
    for x in [y for y in a.exclude.split(",") if y]:
        excluded.setdefault(x, "--exclude")
    ov = None
    if a.overlay:
        ob, _, pre = a.overlay.rpartition(":")
        prefixes = [x for x in pre.split(",") if x]
        if not ob or not prefixes:
            raise SystemExit("--overlay BUILD_DIR:PREFIX[,PREFIX]")
        ob, oman, ocorpus, ogot = load_build(ob)
        obox = [x for x in oman["meets_spec"] if under(x, prefixes) and x not in excluded]
        if not obox:
            raise SystemExit(f"overlay {ob}: no at-spec boxes under {prefixes}")
        for x in want:
            if under(x, prefixes):
                if x not in obox and x not in excluded:
                    print(f"note: {x} is at spec in {b} but not in the overlay build {ob}: left out")
                excluded.setdefault(x, f"replaced by the overlay build {ob}" if x in obox else
                                    f"under the overlay prefix; not at spec in the overlay build {ob}")
        ov = (ob, oman, ocorpus, ogot, prefixes, obox)
    boxes = [x for x in want if x not in excluded]
    if ov:
        boxes += [x for x in ov[5] if x not in boxes]
    if not boxes:
        raise SystemExit("no boxes left")
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        raise SystemExit(f"refusing existing {tmp} (an interrupted run: remove it)")
    tmp.mkdir(parents=True)
    counts = defaultdict(Counter)
    regs = defaultdict(Counter)
    t0 = time.time()
    prim = [x for x in boxes if not (ov and x in ov[5])]
    extract(corpus, man, prim, tmp, counts, regs)
    box_build = {x: str(b) for x in prim}
    builds = [{"build": str(b), "corpus_sha256": got, "build_revision": man.get("mlx2_source_revision"),
               "boxes": len(prim)}]
    if ov:
        ob, oman, ocorpus, ogot, prefixes, obox = ov
        extract(ocorpus, oman, obox, tmp, counts, regs)
        box_build.update({x: str(ob) for x in obox})
        builds.append({"build": str(ob), "corpus_sha256": ogot, "build_revision": oman.get("mlx2_source_revision"),
                       "boxes": len(obox), "overlay_prefixes": prefixes, "split": oman.get("split")})
    rev = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=9", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True).stdout.strip())
    rep = {"schema": "mlx2.amoeba-category-box-files.v1", "build": str(b), "build_schema": man.get("schema"),
           "build_revision": man.get("mlx2_source_revision"), "corpus_sha256": got, "split": man.get("split"),
           "boxes": boxes, "excluded": excluded,
           "rows": {x: dict(counts[x]) for x in boxes}, "registers": {x: dict(regs[x]) for x in boxes},
           "writer": "scripts/amoeba_category_box_files.py", "mlx2_source_revision": rev, "tree_dirty": dirty,
           "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "wall_s": round(time.time() - t0, 1)}
    if ov:
        rep.update(builds=builds, box_build=box_build)
    (tmp / "manifest.json").write_text(json.dumps(rep, indent=1))
    os.replace(tmp, out)
    print(f"{len(boxes)} boxes -> {out} ({len(excluded)} excluded) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    sys.exit(main())

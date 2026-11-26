#!/usr/bin/env python3
"""P6b of docs/research/2026-11-26-metal-replay-feasibility-probes.md: how many distinct
captures a route needs across its context range.

CPU only. Reads round recordings from a context-ladder run (the Phase 0 unit with
``SIZES=1024,...,65536``, ``CONDITIONS=quiet``, ``REPLAY_RECORD=N``, ``REPLAY_RECORD_MAX=N*sizes``):
with ``--cells PHASE0_DIR`` each recording belongs to the first cell whose first load sample
is later than the recording (its warm-up precedes the measured requests); without it,
recording ``i`` belongs to ``sizes[i // N]``, which is wrong whenever a cell's gap did not arm
(2026-11-26: the 1K warm-up followed the smoke too closely and recorded nothing).

A capture is identified by its *layout*: the kernel sequence plus, per dispatch, the bound
slot indices and kinds, the inline-bytes lengths and the threadgroup-memory lengths. Grid
sizes, buffer identities and inline-byte values are excluded: P6a patches grids on the GPU,
P5 owns buffers and P4 moves the bytes into constants. Two rounds with one layout differ only
in patchable fields, so one capture serves both.

    scripts/replay_p6b_buckets.py RECORDING_DIR --sizes 1024,...,65536 --per-size N --output p6b.json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path


def layout(doc):
    kernels = doc["summary"]["kernels"]
    parts = []
    for e in doc["dispatches"]:
        slots = tuple((s[0], s[1], s[3] if s[1] in ("bytes", "threadgroup") else None) for s in e["slots"])
        parts.append((kernels[e["pipeline"]], e.get("form"), slots))
    return tuple(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dir", type=Path)
    ap.add_argument("--sizes", required=True)
    ap.add_argument("--per-size", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cells", type=Path, help="phase0 output dir (cell-<size>-quiet.json) for time-based mapping")
    args = ap.parse_args(argv)
    sizes = [int(v) for v in args.sizes.split(",")]
    paths = sorted(args.dir.glob("round-*.json"))
    starts = []
    if args.cells:
        for size in sizes:
            cell = args.cells / f"cell-{size}-quiet.json"
            if cell.exists():
                samples = json.loads(cell.read_text())["load"]["samples"]
                if samples:
                    starts.append((samples[0]["t"], size))
        starts.sort()
    rows = []
    layouts = {}
    for i, path in enumerate(paths):
        doc = json.loads(path.read_text())
        key = layout(doc)
        digest = hashlib.sha256(repr(key).encode()).hexdigest()[:12]
        layouts.setdefault(digest, key)
        grids = [tuple(e["grid"]) + tuple(e["group"]) for e in doc["dispatches"]]
        if starts:
            size = next((sz for t, sz in starts if t > doc.get("wall_time", 0)), None)
        else:
            size = sizes[i // args.per_size] if i // args.per_size < len(sizes) else None
        rows.append({"file": path.name, "size": size, "layout": digest, "dispatches": len(doc["dispatches"]),
                     "_grids": grids, "_kernels": [k[0] for k in key]})
    by_size = collections.defaultdict(list)
    for r in rows:
        by_size[r["size"]].append(r)
    report = {"schema": "mlx2.replay-p6b.v1", "probe": "P6b", "recordings": len(rows),
              "distinct_layouts": len(layouts), "per_size": {}, "layout_sizes": {}, "transitions": [],
              "grid_only_differences": []}
    for digest in layouts:
        report["layout_sizes"][digest] = sorted({r["size"] for r in rows if r["layout"] == digest})
    for size in sizes:
        rs = by_size.get(size, [])
        report["per_size"][str(size)] = {"recordings": len(rs), "layouts": sorted({r["layout"] for r in rs}),
                                         "dispatches": [r["dispatches"] for r in rs]}
    # Which kernels appear only below / above each size boundary.
    for a, b in zip(sizes, sizes[1:]):
        ka = set().union(*[set(r["_kernels"]) for r in by_size.get(a, [])]) if by_size.get(a) else set()
        kb = set().union(*[set(r["_kernels"]) for r in by_size.get(b, [])]) if by_size.get(b) else set()
        shared = set(report["per_size"][str(a)]["layouts"]) & set(report["per_size"][str(b)]["layouts"])
        report["transitions"].append({"from": a, "to": b, "shared_layouts": len(shared),
                                      "kernels_gone": sorted(ka - kb)[:20], "kernels_new": sorted(kb - ka)[:20]})
    # Within one layout, how many dispatch grids differ (patchable on the GPU, P6a).
    by_layout = collections.defaultdict(list)
    for r in rows:
        by_layout[r["layout"]].append(r)
    for digest, rs in by_layout.items():
        if len(rs) < 2:
            continue
        base = rs[0]
        diffs = [sum(x != y for x, y in zip(base["_grids"], r["_grids"])) for r in rs[1:]]
        changed = collections.Counter()
        for r in rs[1:]:
            for k, x, y in zip(base["_kernels"], base["_grids"], r["_grids"]):
                if x != y:
                    changed[k] += 1
        report["grid_only_differences"].append({"layout": digest, "rounds": len(rs),
                                                "sizes": report["layout_sizes"][digest],
                                                "max_dispatches_with_grid_change": max(diffs),
                                                "kernels": dict(changed.most_common(8))})
    report["rows"] = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("recordings", "distinct_layouts", "per_size", "transitions")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

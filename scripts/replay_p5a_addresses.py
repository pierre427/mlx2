#!/usr/bin/env python3
"""P5a of docs/research/2026-11-26-metal-replay-feasibility-probes.md: does MLX already
reuse buffer addresses across rounds?

CPU only. Reads round recordings written with ``MLX2_ROUND_PROBE_RECORD_ADDRESSES=1``
(each has ``buffers``: [index, gpuAddress, length, storage, heap]). Rounds are grouped by
their kernel sequence (the verify width changes MLX's kernel choice, so only rounds of one
shape have comparable dispatch tables). Within a group, every (dispatch, buffer slot) binding
is stable when its gpuAddress + offset is identical in every round of the group.

Each binding is classified by what its buffer is to the round (reflected access):
  read_only     never written in the round (weights, embeddings, constants)
  state         read before its first write in the round (KV cache, recurrent state)
  intermediate  written, then read later in the round
  output        written and not read again in the round (consumed after it)

    scripts/replay_p5a_addresses.py RECORDING_DIR [...] --output p5a.json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path


def roles(doc):
    kernels = doc["summary"]["kernels"]
    refl = {p["function"]: {s[0]: s[2] for s in p["slots"] if s[1] == "buffer"} for p in doc["pipelines"]}
    first, written, read_after_write = {}, set(), set()
    for entry in doc["dispatches"]:
        slots = refl.get(kernels[entry["pipeline"]], {})
        reads = [s[2] for s in entry["slots"] if s[1] == "buffer" and not slots.get(s[0], True)]
        writes = [s[2] for s in entry["slots"] if s[1] == "buffer" and slots.get(s[0], True)]
        for b in reads:
            first.setdefault(b, "read")
            if b in written:
                read_after_write.add(b)
        for b in writes:
            first.setdefault(b, "write")
            written.add(b)
    out = {}
    for b, f in first.items():
        if b not in written:
            out[b] = "read_only"
        elif f == "read":
            out[b] = "state"
        elif b in read_after_write:
            out[b] = "intermediate"
        else:
            out[b] = "output"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    docs = []
    for directory in args.dirs:
        for path in sorted(directory.glob("round-*.json")):
            doc = json.loads(path.read_text())
            if "buffers" not in doc:
                continue
            doc["_file"] = path.name
            docs.append(doc)
    if not docs:
        raise SystemExit("no recordings with buffer addresses")
    groups = collections.defaultdict(list)
    for doc in docs:
        kernels = doc["summary"]["kernels"]
        groups[tuple(kernels[e["pipeline"]] for e in doc["dispatches"])].append(doc)
    report = {"schema": "mlx2.replay-p5a.v1", "probe": "P5a", "recordings": len(docs),
              "groups": [], "note": "stable = identical gpuAddress + offset in every round of the group"}
    totals = collections.defaultdict(lambda: [0, 0])
    buffer_totals = collections.defaultdict(lambda: [0, 0])
    for sequence, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(members) < 2:
            report["groups"].append({"rounds": 1, "dispatches": len(sequence), "files": [m["_file"] for m in members]})
            continue
        role = roles(members[0])
        addr = [{b[0]: b[1] for b in m["buffers"]} for m in members]
        counts = collections.defaultdict(lambda: [0, 0])
        bufs = collections.defaultdict(lambda: [0, 0])
        seen_buffers = {}
        for d_index, entry in enumerate(members[0]["dispatches"]):
            for s_index, slot in enumerate(entry["slots"]):
                if slot[1] != "buffer":
                    continue
                values = set()
                ok = True
                for m, a in zip(members, addr):
                    other = m["dispatches"][d_index]["slots"]
                    match = next((x for x in other if x[0] == slot[0] and x[1] == "buffer"), None)
                    if match is None:
                        ok = False
                        break
                    values.add((a.get(match[2]), match[3]))
                stable = ok and len(values) == 1
                r = role.get(slot[2], "unknown")
                counts[r][0] += stable
                counts[r][1] += 1
                key = slot[2]
                seen_buffers[key] = seen_buffers.get(key, True) and stable
        for b, stable in seen_buffers.items():
            r = role.get(b, "unknown")
            bufs[r][0] += stable
            bufs[r][1] += 1
        for r, (s, n) in counts.items():
            totals[r][0] += s
            totals[r][1] += n
        for r, (s, n) in bufs.items():
            buffer_totals[r][0] += s
            buffer_totals[r][1] += n
        sizes = collections.defaultdict(int)
        lengths = {b[0]: b[2] for b in members[0]["buffers"]}
        for b, stable in seen_buffers.items():
            if not stable:
                sizes[role.get(b, "unknown")] += lengths.get(b, 0)
        report["groups"].append({
            "rounds": len(members), "dispatches": len(sequence), "files": [m["_file"] for m in members],
            "bindings": {r: {"stable": s, "total": n, "stable_pct": round(100 * s / n, 2)} for r, (s, n) in sorted(counts.items())},
            "buffers": {r: {"stable": s, "total": n, "stable_pct": round(100 * s / n, 2)} for r, (s, n) in sorted(bufs.items())},
            "unstable_bytes": dict(sizes),
            "heap_backed_buffers": sum(1 for b in members[0]["buffers"] if b[4]),
            "storage_modes": dict(collections.Counter(b[3] for b in members[0]["buffers"])),
        })
    report["all_groups"] = {
        "bindings": {r: {"stable": s, "total": n, "stable_pct": round(100 * s / n, 2)} for r, (s, n) in sorted(totals.items())},
        "buffers": {r: {"stable": s, "total": n, "stable_pct": round(100 * s / n, 2)} for r, (s, n) in sorted(buffer_totals.items())},
    }
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"recordings": len(docs), "groups": [(g["rounds"], g["dispatches"]) for g in report["groups"]],
                      "all_groups": report["all_groups"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

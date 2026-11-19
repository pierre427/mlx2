#!/usr/bin/env python3
"""Coverage report for a category-boxes build + its probe set: every in-scope taxonomy node is `at spec`,
`short` (with the reason the builder recorded) or `no source` (no assigned docs at all). Writes REPORT.md and
coverage.tsv next to the box manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCOPE = ("math", "sci", "eng", "hum", "soc", "life", "prog", "swe", "cs", "sec", "agent", "dialog")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--taxonomy", default=str(ROOT / "docs/research/amoeba-topics/taxonomy.v1.json"))
    a = ap.parse_args()
    bm = json.loads((Path(a.boxes) / "manifest.json").read_text())
    pm = json.loads((Path(a.probes) / "manifest.json").read_text())
    tax = [n for n in json.load(open(a.taxonomy))["nodes"] if n["id"].split(".")[0] in SCOPE]
    rows, counts = [], {"at spec": 0, "short": 0, "no source": 0}
    for n in tax:
        b = bm["boxes"].get(n["id"])
        p = pm["topics"].get(n["id"], {})
        if b is None:
            st, why, regs = "no source", "no assigned docs", ""
        elif b["meets_spec"]:
            st, why = "at spec", ""
            regs = ", ".join(f"{k} {v['taken']}" for k, v in b["registers"].items())
        else:
            st, why = "short", b.get("short_reason", "")
            regs = ", ".join(f"{k} {v['taken']} (fit {v.get('fit', 0)})" for k, v in b["registers"].items())
        counts[st] += 1
        rows.append((n["id"], n["kind"], st, b["docs"] if b else 0, regs, why,
                     f"{p.get('fit', 0)}/{p.get('held', 0)}", "yes" if p.get("tested") else "no"))
    with open(Path(a.boxes) / "coverage.tsv", "w") as f:
        f.write("node\tkind\tstatus\tdocs\tregisters\treason\tprobes_fit/held\tprobe_tested\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")
    lines = [f"# Category coverage: {Path(a.boxes).name}", "",
             f"Spec: {bm['spec']}.", f"Split: {bm['split']}. Date cut: {bm['date_cut']}.", "",
             f"In scope: {len(rows)} nodes under {', '.join(SCOPE)}. **At spec {counts['at spec']}**, "
             f"short {counts['short']}, no source {counts['no source']}. Probe set {Path(a.probes).name}: "
             f"{pm['items']} windows, {pm['summary']['tested']} topics tested (>= 16 fit and 16 held docs).", "",
             "| node | status | docs | registers (taken) | why short | probes fit/held |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| `{r[0]}` | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} |")
    (Path(a.boxes) / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(counts))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare old-tree and new-tree arms of a spec_acceptance_gsm8k receipt.

Used by scripts/technology_parity_unit_20261125.sh to check that the
technology consolidation (one reference per technology, mlx2 2026-11-25)
leaves served outputs unchanged. Arms are paired by name: ``old-<route>`` with
``new-<route>``. At temperature 0 a pair passes only when every question's
text, completion-token count, finish reason and (for speculative routes) the
verify acceptance and depth histograms are identical. A text difference is
reported with the first differing character offset so the first divergent
token can be localised before any near-tie classification; this script never
classifies near ties itself.

Usage: technology_parity_compare.py RECEIPT_DIR [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FIELDS = ("text", "completion_tokens", "finish_reason", "accept_hist", "depth_hist")
TECH = re.compile(r'^mlx2_technology_(dispatch|refusals)_total\{([^}]*)\}\s+(\d+)', re.M)


def first_difference(a: str, b: str) -> int:
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return index
    return min(len(a), len(b))


def technology_counters(arm_dir: Path) -> list[str]:
    prom = arm_dir / "metrics.prom"
    if not prom.exists():
        return []
    return [f"{kind}{{{labels}}} {count}" for kind, labels, count in TECH.findall(prom.read_text())]


def compare(receipt_dir: Path) -> dict:
    receipt = json.loads((receipt_dir / "receipt.json").read_text())
    arms = {arm["name"]: arm for arm in receipt.get("arms", [])}
    pairs = []
    for name, old in sorted(arms.items()):
        if not name.startswith("old-"):
            continue
        route = name[len("old-"):]
        new = arms.get("new-" + route)
        pair = {"route": route, "old_revision": old.get("revision"),
                "new_revision": (new or {}).get("revision")}
        if new is None or not old.get("passed") or not new.get("passed"):
            pair.update(verdict="incomplete", old_passed=old.get("passed"),
                        new_passed=(new or {}).get("passed"),
                        old_error=old.get("error"), new_error=(new or {}).get("error"))
            pairs.append(pair)
            continue
        old_rows = {row["index"]: row for row in old.get("rows", [])}
        new_rows = {row["index"]: row for row in new.get("rows", [])}
        common = sorted(set(old_rows) & set(new_rows))
        mismatches = []
        for index in common:
            a, b = old_rows[index], new_rows[index]
            differing = [field for field in FIELDS if a.get(field) != b.get(field)]
            if differing:
                entry = {"index": index, "test_index": a.get("test_index"), "fields": differing,
                         "old_correct": a.get("correct"), "new_correct": b.get("correct")}
                if "text" in differing:
                    offset = first_difference(a.get("text") or "", b.get("text") or "")
                    entry["first_text_difference"] = offset
                    entry["old_context"] = (a.get("text") or "")[max(0, offset - 40): offset + 40]
                    entry["new_context"] = (b.get("text") or "")[max(0, offset - 40): offset + 40]
                mismatches.append(entry)
        routes = {"old": sorted({str(r.get("route")) for r in old.get("rows", [])}),
                  "new": sorted({str(r.get("route")) for r in new.get("rows", [])})}
        pair.update(
            verdict="exact" if not mismatches and len(common) == len(old_rows) == len(new_rows) and common
            else ("empty" if not common else "mismatch"),
            questions=len(common), mismatches=mismatches,
            accuracy={"old": (old.get("summary") or {}).get("accuracy"),
                      "new": (new.get("summary") or {}).get("accuracy")},
            mean_acceptance_length={"old": (old.get("summary") or {}).get("mean_acceptance_length"),
                                    "new": (new.get("summary") or {}).get("mean_acceptance_length")},
            routes=routes,
            new_technology_counters=technology_counters(receipt_dir / ("new-" + route)),
        )
        pairs.append(pair)
    verdicts = {pair["verdict"] for pair in pairs}
    return {"schema": "mlx2.technology-parity.v1", "checkpoint": receipt.get("checkpoint"),
            "pairs": pairs,
            "verdict": "exact" if pairs and verdicts == {"exact"} else
            ("mismatch" if "mismatch" in verdicts else "incomplete")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt_dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = compare(args.receipt_dir)
    text = json.dumps(result, indent=2, default=str) + "\n"
    if args.out:
        tmp = args.out.with_suffix(".tmp")
        tmp.write_text(text)
        tmp.replace(args.out)
    print(json.dumps({"checkpoint": result["checkpoint"], "verdict": result["verdict"],
                      "pairs": {p["route"]: p["verdict"] for p in result["pairs"]}}))
    return 0 if result["verdict"] == "exact" else 1


if __name__ == "__main__":
    raise SystemExit(main())

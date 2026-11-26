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
from collections import Counter
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


def histogram_rounds(histogram):
    if isinstance(histogram, dict):
        if not all(str(k).isdigit() for k in histogram):
            return None
        counts = histogram.values()
    elif isinstance(histogram, list):
        counts = histogram
    else:
        return None
    return sum(counts) if all(type(n) is int and n >= 0 for n in counts) else None


def indexed_rows(arm, expected, arm_dir):
    """The producer numbers rows locally and records their GSM8K test indices."""
    rows = arm.get("rows") or []
    if arm.get("stopped_at_deadline") or not expected or len(rows) != len(expected):
        return None
    # Older compact rows omit the route. Their arm summary lists JSON-encoded
    # route names, and the full rows retain each request's route in mixed arms.
    routes = (arm.get("summary") or {}).get("routes") or []
    ordinary_arm = routes in (['"ordinary"'], ["ordinary"])
    full_routes = {}
    if len(routes) > 1 and any("route" not in row for row in rows):
        full_rows = arm_dir / "rows.json"
        if full_rows.is_file():
            full_routes = {(r["index"], r["test_index"]): r.get("route") for r in json.loads(full_rows.read_text())}
    by_index = {}
    for row in rows:
        i = row.get("index")
        if (type(i) is not int or not 0 <= i < len(expected) or i in by_index
                or row.get("test_index") != expected[i]
                or not isinstance(row.get("text"), str)
                or type(row.get("completion_tokens")) is not int or row["completion_tokens"] < 0
                or not isinstance(row.get("finish_reason"), str) or not row["finish_reason"]
                or "accept_hist" not in row or "depth_hist" not in row):
            return None
        route = row.get("route") or full_routes.get((i, expected[i]))
        ordinary = route == "ordinary" if route is not None else ordinary_arm
        if not ordinary or row["accept_hist"] is not None or row["depth_hist"] is not None:
            accepted = histogram_rounds(row["accept_hist"])
            if accepted is None or accepted != histogram_rounds(row["depth_hist"]):
                return None
        by_index[i] = row
    return by_index


def compare(receipt_dir: Path, required_routes=()) -> dict:
    receipt = json.loads((receipt_dir / "receipt.json").read_text())
    dataset = receipt.get("dataset") or {}
    count = dataset.get("questions", 0)
    start = dataset.get("offset", 0)
    expected = dataset.get("indices", range(start, start + count))
    if len(expected) != count:
        expected = []
    entries = receipt.get("arms", [])
    names = Counter(arm["name"] for arm in entries)
    arms = {arm["name"]: arm for arm in entries}
    routes = set(required_routes)
    routes.update(name.split("-", 1)[1] for name in arms if name.startswith(("old-", "new-")))
    checkpoint_name, _, checkpoint_fingerprint = (receipt.get("checkpoint") or "").rpartition("@")
    pairs = []
    for route in sorted(routes):
        old_name, new_name = "old-" + route, "new-" + route
        old, new = arms.get(old_name, {}), arms.get(new_name, {})
        pair = {"route": route, "old_revision": old.get("revision"),
                "new_revision": new.get("revision")}
        pairs.append(pair)
        if (names[old_name] != 1 or names[new_name] != 1
                or not all(arm.get("passed") and (arm.get("revision") or {}).get("head")
                           for arm in (old, new))
                or not checkpoint_name or not checkpoint_fingerprint.removeprefix("sha256:")):
            pair.update(verdict="incomplete", old_passed=old.get("passed"),
                        new_passed=new.get("passed"), old_error=old.get("error"),
                        new_error=new.get("error"))
            continue
        old_rows = indexed_rows(old, expected, receipt_dir / old_name)
        new_rows = indexed_rows(new, expected, receipt_dir / new_name)
        if old_rows is None or new_rows is None:
            pair.update(verdict="incomplete", expected_questions=count,
                        old_questions=len(old.get("rows") or []), new_questions=len(new.get("rows") or []))
            continue
        mismatches = []
        for index in sorted(old_rows):
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
        old_summary, new_summary = old.get("summary") or {}, new.get("summary") or {}
        pair.update(
            verdict="mismatch" if mismatches else "exact",
            questions=len(old_rows), mismatches=mismatches,
            accuracy={"old": old_summary.get("accuracy"),
                      "new": new_summary.get("accuracy")},
            mean_acceptance_length={"old": (old_summary.get("acceptance") or {}).get("mean_acceptance_length"),
                                    "new": (new_summary.get("acceptance") or {}).get("mean_acceptance_length")},
            routes={"old": old_summary.get("routes", []), "new": new_summary.get("routes", [])},
            new_technology_counters=technology_counters(receipt_dir / ("new-" + route)),
        )
    verdicts = {pair["verdict"] for pair in pairs}
    return {"schema": "mlx2.technology-parity.v1", "checkpoint": receipt.get("checkpoint"),
            "pairs": pairs,
            "verdict": "exact" if pairs and verdicts == {"exact"} else
            ("mismatch" if "mismatch" in verdicts else "incomplete")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt_dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--require-route", action="append", default=[], help="route pair required for exact parity")
    args = parser.parse_args()
    result = compare(args.receipt_dir, args.require_route)
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

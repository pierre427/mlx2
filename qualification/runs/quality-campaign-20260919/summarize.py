#!/usr/bin/env python3
"""Write one Markdown summary table per campaign phase."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RUN = Path(__file__).resolve().parent


def load(path):
    try: return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError): return None


def smoke_row(name, entry, directory):
    reports = [load(directory / f"feature-{group}.json") for group in ("core", "opt-in", "persist-seed", "persist-rescan")]
    reports = [report for report in reports if report]
    counts = {key: sum(report["summary"].get(key, 0) for report in reports) for key in ("PASS", "FAIL", "SKIP")}
    steps = entry.get("steps", {})
    qualifier = steps.get("base:qualifier", {}).get("returncode", "—")
    sdk = steps.get("base:official-sdks", {}).get("returncode", "—")
    prefetch = steps.get("apc-prefetch", {}).get("returncode", "—")
    return [name, entry.get("state", "missing"), f"{counts['PASS']}/{counts['FAIL']}/{counts['SKIP']}", qualifier, sdk, prefetch]


def sanity_row(name, entry, directory):
    report = load(directory / "sanity.json") or {}
    return [name, entry.get("state", "missing"), f"{report.get('graded_correct', '—')}/{report.get('requests', '—')}", report.get("http_errors", "—"), json.dumps(report.get("issue_totals", {}), sort_keys=True), report.get("peak_observed_width", "—")]


def ladder_row(name, entry, directory):
    report = load(directory / "ladder.json") or {}
    rows = report.get("rows", [])
    needle = sum(row.get("needle_correct", 0) for row in rows)
    requests = sum(row.get("width", 0) for row in rows)
    warm = [row for row in rows if row.get("temperature") == "warm"]
    hits = sum(row.get("apc_hits", 0) for row in warm)
    warm_requests = sum(row.get("width", 0) for row in warm)
    peak = (report.get("peak_memory") or {}).get("bytes")
    return [name, entry.get("state", "missing"), len(rows), f"{needle}/{requests}", f"{hits}/{warm_requests}", peak if peak is not None else "—"]


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |" for row in rows]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, default=RUN / "status.json")
    parser.add_argument("--output", type=Path, default=RUN / "SUMMARY.md")
    args = parser.parse_args(argv)
    status = load(args.status)
    if not status: parser.error(f"cannot read {args.status}")
    sections = ["# GPU quality campaign summary", "", f"Campaign state: **{status.get('phase')}**, updated `{status.get('updated')}`."]
    definitions = {
        "smoke": (["Stage", "State", "Checks P/F/S", "Qualifier rc", "SDK rc", "APC prefetch rc"], smoke_row),
        "sanity": (["Stage", "State", "Correct", "HTTP errors", "Issues", "Peak width"], sanity_row),
        "ladder": (["Stage", "State", "Cells", "Needle correct", "Warm APC hits", "Peak memory bytes"], ladder_row),
    }
    for phase, (headers, builder) in definitions.items():
        rows = []
        for name in status.get("order", []):
            if not name.startswith(phase + "-"): continue
            entry = status.get("stages", {}).get(name, {})
            rows.append(builder(name, entry, RUN / "results" / name))
        sections += ["", f"## {phase.title()}", "", table(headers, rows) if rows else "No stages recorded."]
    args.output.write_text("\n".join(sections) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()

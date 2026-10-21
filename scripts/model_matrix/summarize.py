#!/usr/bin/env python3
"""Summarize model-matrix cell receipts into a per-checkpoint Markdown table.

A rerun of a route (evidence suffix) supersedes the earlier cell; the table
shows the latest verdict per (checkpoint, route) and the host it ran on.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys as _sys

_sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_matrix import effective_verdict  # noqa: E402

SHORT = {"pass": "P", "pass_with_warnings": "P*", "fail": "F", "fail_closed_expected": "FC",
         "skipped": "S", "aborted_swap": "A", "artifact_offline": "OFF"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    latest: dict[tuple, dict] = {}
    for path in sorted(args.evidence_root.resolve().glob("*/*/cell.json")):
        cell = json.loads(path.read_text())
        cell["verdict"] = effective_verdict(cell)
        key = (cell["checkpoint"], cell["route"])
        if key not in latest or (cell.get("finished_at") or "") >= (latest[key].get("finished_at") or ""):
            latest[key] = {**cell, "_dir": path.parent.name}
    by_checkpoint: dict[str, list] = defaultdict(list)
    for (checkpoint, _), cell in latest.items():
        by_checkpoint[checkpoint].append(cell)
    lines = ["| checkpoint | adapter | host | routes (P pass, P* pass w/ warnings, F fail, FC fail-closed, S skipped) |",
             "|---|---|---|---|"]
    for checkpoint in sorted(by_checkpoint):
        cells = sorted(by_checkpoint[checkpoint], key=lambda c: c["route"])
        name, _, fingerprint = checkpoint.partition("@sha256:")
        routes = ", ".join(f"{c['route']}={SHORT.get(c.get('verdict'), c.get('verdict'))}" for c in cells)
        hosts = ",".join(sorted({c.get("host") or "?" for c in cells}))
        lines.append(f"| {name}@{fingerprint[:12]} | {cells[0].get('adapter')} | {hosts} | {routes} |")
    text = "\n".join(lines) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

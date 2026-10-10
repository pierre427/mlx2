#!/usr/bin/env python3
"""Separate performance assessment for two matching context ladders."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

METRICS = ("prefill_tokens_per_second", "decode_tokens_per_second")


def _identity(ladder):
    initial = ladder.get("initial") or {}
    return {
        "model": ladder.get("model"),
        "route": ladder.get("route"),
        "max_context": ladder.get("max_context"),
        "host": ladder.get("host"),
        "artifact": initial.get("artifact"),
        "runtime": initial.get("runtime"),
        "settings": initial.get("settings"),
    }


def _cells(ladder):
    rows = ladder.get("cells")
    if not isinstance(rows, list):
        return {}
    return {
        (c.get("requested_tokens"), c.get("width")): c
        for c in rows
        if isinstance(c, dict)
    }


def _coverage_valid(ladder, cells):
    plan = ladder.get("expected_cells")
    if not isinstance(plan, list) or not plan:
        return False
    expected = set()
    for item in plan:
        if not isinstance(item, dict):
            return False
        tokens, width = item.get("requested_tokens"), item.get("width")
        if (
            not isinstance(tokens, int)
            or isinstance(tokens, bool)
            or not isinstance(width, int)
            or isinstance(width, bool)
            or width < 1
        ):
            return False
        expected.add((tokens, width))
    return len(expected) == len(plan) and set(cells) == expected


def _runs_clean(cell):
    rows = list(cell.get("attempts") or []) + list(cell.get("runs") or [])
    if len(cell.get("runs") or []) != 3 or cell.get("error"):
        return False
    for row in rows:
        if row.get("contaminated") or row.get("functional_failures"):
            return False
        if row.get("swapouts", {}).get("delta", 0) != 0 or row.get("foreign_activity"):
            return False
        pre = row.get("thermal_pre") or {}
        post = row.get("thermal_post") or {}
        if row.get("measured") and pre.get("stable") is not True:
            return False
        if row.get("measured"):
            samples = pre.get("samples") or []
            if not samples or any(
                int(sample.get("thermal_state", 3)) != 0 for sample in samples
            ):
                return False
        if post.get("breached") is True:
            return False
        for sample in post.get("samples") or []:
            if (
                int(sample.get("thermal_state", 0)) >= 2
                or sample.get("pmset_no_thermal_warning") is False
                or sample.get("pmset_no_performance_warning") is False
                or sample.get("pmset_no_cpu_power_warning") is False
                or int(sample.get("cpu_speed_limit", 100)) < 100
            ):
                return False
    return True


def assess(current, reference, tolerance=0.30):
    problems = []
    if current.get("runs_per_cell") != 3 or reference.get("runs_per_cell") != 3:
        problems.append("both ladders must have exactly three runs per cell")
    now_id, ref_id = _identity(current), _identity(reference)
    if any(
        now_id.get(key) in (None, "", {})
        for key in (
            "model",
            "route",
            "host",
            "max_context",
            "artifact",
            "runtime",
            "settings",
        )
    ):
        problems.append(
            "performance identity is missing model, route, host, context, artifact, runtime, or settings"
        )
    if now_id != ref_id:
        problems.append(
            f"model/route/artifact/runtime/settings/host/context mismatch: {now_id} vs {ref_id}"
        )
    now_cells, ref_cells = _cells(current), _cells(reference)
    comparisons = []
    if not _coverage_valid(current, now_cells) or not _coverage_valid(
        reference, ref_cells
    ):
        problems.append(
            "one or both ladders do not match their declared context/width plan"
        )
    if not now_cells or set(now_cells) != set(ref_cells):
        problems.append("context/width cell sets differ or are empty")
    for key in sorted(set(now_cells) & set(ref_cells)):
        now, ref = now_cells[key], ref_cells[key]
        if not _runs_clean(now) or not _runs_clean(ref):
            problems.append(f"{key}: thermal/environmental controls failed")
            continue
        metrics = {}
        for metric in METRICS:
            vn = [r.get("summary", {}).get(metric) for r in now.get("runs", [])]
            vr = [r.get("summary", {}).get(metric) for r in ref.get("runs", [])]
            if any(not isinstance(v, (int, float)) or v <= 0 for v in vn + vr):
                problems.append(f"{key}: missing positive {metric}")
                continue
            ratio = statistics.median(vn) / statistics.median(vr)
            metrics[metric] = {
                "current_median": statistics.median(vn),
                "reference_median": statistics.median(vr),
                "ratio": ratio,
            }
            if ratio < 1 - tolerance:
                problems.append(f"{key}: {metric} is {ratio:.0%} of reference")
        comparisons.append({"cell": key, "metrics": metrics})
    return {
        "schema": "mlx2.context-performance-assessment.v1",
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "identity": now_id,
        "tolerance": tolerance,
        "quoteable": not problems,
        "failures": problems,
        "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ladder", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.30)
    args = parser.parse_args()
    raw, ref_raw = args.ladder.read_bytes(), args.reference.read_bytes()
    result = assess(json.loads(raw), json.loads(ref_raw), args.tolerance)
    result["inputs"] = {
        "ladder": {"path": str(args.ladder), "sha256": hashlib.sha256(raw).hexdigest()},
        "reference": {
            "path": str(args.reference),
            "sha256": hashlib.sha256(ref_raw).hexdigest(),
        },
    }
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)
    return 0 if result["quoteable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

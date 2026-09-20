#!/usr/bin/env python3
"""Best CONSTANT self-MTP depth from measured cycle times and acceptance (CPU).

Inputs are both measurements, never modelled betas: a ``profile`` cycle table
(median seconds per cycle at each (lanes, depth)) and an acceptance log's
conditional per-position acceptance.  Expected committed tokens per cycle at
depth d is ``1 + s1 + s1 s2 + ...``; goodput is that over the measured cycle
time.  Prints the argmax depth per width.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlx2.runtime.mtp_confidence import load_acceptance_log  # noqa: E402


def conditional_acceptance(rows, positions: int) -> list[float]:
    out = []
    for k in range(positions):
        labels = [r.labels()[k] for r in rows if k < len(r.labels()) and r.labels()[k] is not None]
        out.append(sum(labels) / len(labels) if labels else 0.0)
    return out


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True, help="cost profile JSON with a measured cycle_table")
    parser.add_argument("--log", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    profile = json.loads(args.profile.read_text())
    schema = profile.get("schema")
    if schema not in ("mlx2.mtp_verify_cost.v1", "mlx2.mtp_verify_cost.v2"):
        parser.error(f"unsupported cost profile schema {schema!r}")
    if schema == "mlx2.mtp_verify_cost.v1" and "costs" in profile:
        # v1 files carry an `m_points`/`costs` pair that was a hardcoded cold
        # NAX curve, not a measurement, written beside the measured
        # `cycle_table`.  This tool never reads it; say so, because the file
        # does not.
        print(
            f"note: {args.profile} is schema v1; its 'm_points'/'costs' arrays "
            "are a hardcoded literal, not a measurement, and are ignored here. "
            "Only 'cycle_table' is measured.",
            file=sys.stderr,
        )
    table = profile["cycle_table"]
    rows = [r for path in args.log for r in load_acceptance_log(path) if r.verify_depth > 0 and r.features]
    if not rows:
        parser.error("no drafted rows in the acceptance log(s)")
    depths = max(len(row) for row in table.values())
    survival, running = [], 1.0
    for rate in conditional_acceptance(rows, depths - 1):
        running *= rate
        survival.append(running)
    report = {"profile": str(args.profile), "rows": len(rows),
              "conditional_acceptance": conditional_acceptance(rows, depths - 1),
              "survival": survival, "widths": {}}
    for lanes, seconds in sorted(table.items(), key=lambda kv: int(kv[0])):
        tokens = [1.0 + sum(survival[:d]) for d in range(len(seconds))]
        goodput = [t / s for t, s in zip(tokens, seconds)]
        best = max(range(len(goodput)), key=goodput.__getitem__)
        report["widths"][lanes] = {
            "tokens_per_cycle": tokens, "seconds_per_cycle": seconds,
            "relative_goodput": [g / goodput[best] for g in goodput],
            "best_depth": best,
            "best_over_max_depth": goodput[best] / goodput[-1] - 1.0,
        }
        print(f"lanes={lanes} best_depth={best} relative_goodput="
              f"{[round(g / goodput[best], 3) for g in goodput]}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()

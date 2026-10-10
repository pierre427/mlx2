#!/usr/bin/env python3
"""Offline oracle for the draft loop at cohort widths above one.

Self-MTP verifies a cohort with padded rows: every lane gets as many rows as
the deepest lane, so one lane's extension is paid by the whole cohort.  This
script asks which cohort rule would pay at each width, from:

``--log``   an acceptance log with lookahead (labelled exactly by
            ``dloop_oracle.py``'s stream rebuild);
``--cost``  ``mtp_confidence_gpu.py profile`` raw cycles per (lanes, depth).

Cohorts are drawn with replacement from the labelled cycles (lanes are
treated as independent), and each rule maps a cohort to one depth for every
lane.  Rules, for base depth ``b`` and end depth ``E``:

``fixed_d``        every lane at ``d``.
``any_g_E``        extend to ``E`` when at least one lane's first stage scores
                   >= g; every lane then drafts to ``E`` (the padded rows are
                   paid anyway, and a batched draft step costs about the same
                   for one lane or all).
``all_g_E``        extend only when every lane passes.
``half_g_E``       extend when at least half the lanes pass.
``oracle_E``       extend exactly when the extension lowers this cycle's
                   ms/token (needs the future; the ceiling for any rule).

Reported per width: ms/token = E[cycle] / E[tokens] and the speedup over
fixed base depth at that width.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("dloop_oracle", HERE / "dloop_oracle.py")
oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oracle)


def load_costs(path):
    data = json.loads(Path(path).read_text())
    cells = defaultdict(list)
    for item in data["raw"]:
        cells[(int(item["lanes"]), int(item["depth"]))].append(float(item["seconds_per_cycle"]))
    return {key: statistics.median(values) for key, values in cells.items()}


def cohort_tokens(cohort, depth):
    return sum(min(s["match"], depth) + 1 for s in cohort)


def stage_score(sample, base):
    return sum(sample["logq"][:base])


def evaluate_width(samples, costs, width, base, ends, thresholds, draws, sync, rng):
    cohorts = [[rng.choice(samples) for _ in range(width)] for _ in range(draws)]
    depths = sorted(d for (w, d) in costs if w == width)

    def summary(depth_of, decisions_of=lambda cohort: 0):
        tokens = seconds = 0.0
        extended = 0
        for cohort in cohorts:
            depth = depth_of(cohort)
            extended += depth != base
            tokens += cohort_tokens(cohort, depth)
            seconds += costs[(width, depth)] + sync * decisions_of(cohort)
        return {"ms_per_token": 1e3 * seconds / tokens,
                "tokens_per_cycle": tokens / len(cohorts),
                "extend_rate": extended / len(cohorts)}

    out = {}
    for depth in depths:
        if depth >= 1:
            out[f"fixed_{depth}"] = summary(lambda c, d=depth: d)
    for end in ends:
        if (width, end) not in costs or end <= base:
            continue
        for g in thresholds:
            passes = lambda c, g=g: sum(stage_score(s, base) >= g for s in c)
            out[f"any_{g:g}_{end}"] = summary(
                lambda c, e=end, p=passes: e if p(c) >= 1 else base, lambda c: 1)
            out[f"all_{g:g}_{end}"] = summary(
                lambda c, e=end, p=passes: e if p(c) == len(c) else base, lambda c: 1)
            out[f"half_{g:g}_{end}"] = summary(
                lambda c, e=end, p=passes: e if p(c) >= math.ceil(len(c) / 2) else base,
                lambda c: 1)
        base_cost, end_cost = costs[(width, base)], costs[(width, end)]
        out[f"oracle_{end}"] = summary(
            lambda c, e=end: e if cohort_tokens(c, e) / end_cost > cohort_tokens(c, base) / base_cost
            else base)
    reference = out[f"fixed_{base}"]["ms_per_token"]
    for value in out.values():
        value["speedup_vs_fixed_base"] = reference / value["ms_per_token"]
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--log", nargs="+", required=True, type=Path)
    parser.add_argument("--cost", required=True, type=Path)
    parser.add_argument("--widths", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--base", type=int, default=3)
    parser.add_argument("--ends", type=int, nargs="+", default=[4, 5, 6, 7, 9])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[-0.1, -0.25, -0.4, -0.75])
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--gate-sync-ms", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top", type=int, default=6)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    rows = oracle.split_requests(oracle.load_rows(args.log))
    streams, ends, _ = oracle.build_streams(rows)
    depth = max(args.ends + [args.base])
    samples, dropped = oracle.label_cycles(rows, streams, ends, depth)
    costs = load_costs(args.cost)
    rng = random.Random(args.seed)
    report = {"samples": len(samples), "dropped": dropped, "draws": args.draws,
              "cycle_ms": {f"{w}x{d}": 1e3 * s for (w, d), s in sorted(costs.items())},
              "widths": {}}
    for width in args.widths:
        if (width, args.base) not in costs:
            continue
        result = evaluate_width(samples, costs, width, args.base, args.ends, args.thresholds,
                                args.draws, args.gate_sync_ms / 1e3, rng)
        report["widths"][str(width)] = result
        best = sorted(result.items(), key=lambda kv: kv[1]["ms_per_token"])[: args.top]
        print(f"== width {width}: fixed_{args.base} "
              f"{result[f'fixed_{args.base}']['ms_per_token']:.1f} ms/token")
        for name, value in best:
            print(f"   {name:16} {value['ms_per_token']:6.1f} ms/tok  x{value['speedup_vs_fixed_base']:.3f}"
                  f"  tok/cyc {value['tokens_per_cycle']:.2f}  extend {value['extend_rate']:.2f}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    sys.exit(main())

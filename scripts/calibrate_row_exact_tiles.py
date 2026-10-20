"""Measure the row grouping table for the row-exact projection kernel.

``row_exact_qmv`` gives every verify row its own accumulator and K order, so
the number of rows that share a threadgroup changes only speed.  This script
times every admissible grouping for each (bits, N, K, rows) shape on this
chip and MLX build, checks each candidate's output bit for bit against
one-row ``quantized_matmul`` calls, and writes the fastest grouping to
``src/mlx2/runtime/row_exact/tables/<device>__<mlx build>.json``, which
``row_exact.planner`` reads.  Entries that do not beat the heuristic by
``--min-gain`` keep the heuristic (they are recorded, not written).

The default shapes are Nemotron-3.5-Super's 8-bit projections; pass
``--shape bits:n:k`` (repeatable) for other models.

  MLX_ENABLE_TF32=0 PYTHONPATH=src .venv/bin/python \\
      scripts/calibrate_row_exact_tiles.py --i-own-the-gpu --out calib.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

NEMOTRON_8BIT = [
    (8, 18560, 4096),  # Mamba in_proj
    (8, 4096, 8192),  # Mamba out_proj
    (8, 4096, 4096),  # attention q/o
    (8, 256, 4096),  # attention k/v
    (8, 5376, 4096),  # shared expert up
    (8, 4096, 5376),  # shared expert down
    (8, 131072, 4096),  # lm_head
]


def swapouts() -> int:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=False).stdout
    line = next(l for l in out.splitlines() if l.startswith("Swapouts"))
    return int(line.split(":")[1].strip(" ."))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-own-the-gpu", action="store_true", required=True)
    ap.add_argument("--shape", action="append", default=[], help="bits:n:k")
    ap.add_argument("--rows", nargs="+", type=int, default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument("--calls", type=int, default=32, help="calls per timed eval")
    ap.add_argument("--passes", type=int, default=11)
    ap.add_argument("--min-gain", type=float, default=0.03)
    ap.add_argument(
        "--min-win-rate",
        type=float,
        default=0.8,
        help="share of paired passes the winner must beat the heuristic in",
    )
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-write-table", action="store_true")
    args = ap.parse_args()

    import mlx.core as mx
    import mlx.nn as nn

    from mlx2.runtime.mlx_build import installed_mlx_build
    from mlx2.runtime.models import row_exact_qmv as REQ
    from mlx2.runtime.row_exact import canary, planner

    shapes = [tuple(int(v) for v in s.split(":")) for s in args.shape] or NEMOTRON_8BIT
    device, build = planner.device_name(), installed_mlx_build()
    swap0 = swapouts()
    forced = {"g": None}
    original = REQ._row_groups

    def row_groups(bits, fast, n, rows, k=0):
        return forced["g"] if forced["g"] is not None else original(bits, fast, n, rows, k)

    REQ._row_groups = row_groups
    cells, table = [], {}
    try:
        for bits, n, k in shapes:
            base = nn.Linear(k, n, bias=False)
            base.set_dtype(mx.bfloat16)
            linear = nn.QuantizedLinear.from_linear(base, group_size=args.group_size, bits=bits)
            mx.eval(linear.parameters())
            fast = REQ.qmv_fast_layout(k, n, bits)
            limit = REQ.row_group_limit(bits, fast)
            tiles = (n + 2 * REQ._RPS - 1) // (2 * REQ._RPS)
            for rows in args.rows:
                x = canary.probe_input(rows, k, mx.bfloat16, seed=rows * 31 + n)
                reference = REQ.per_row(linear, x)
                heuristic = planner.heuristic_rows_per_group(
                    rows=rows, limit=limit, tiles=tiles, narrow_tiles=REQ._SHARED_ROWS_MIN_TILES
                )
                candidates = list(range(1, min(rows, limit) + 1))
                timings = {g: [] for g in candidates}
                exact = {}
                for g in candidates:
                    forced["g"] = g
                    linear.__dict__.pop("_mlx2_row_exact_plans", None)
                    y, route = REQ.quantized_linear(linear, x)
                    mx.eval(y, reference)
                    exact[g] = route != "per_row" and canary.bit_mismatches(y, reference) == 0
                for p in range(args.passes):
                    order = candidates if p % 2 == 0 else candidates[::-1]
                    for g in order:
                        forced["g"] = g
                        linear.__dict__.pop("_mlx2_row_exact_plans", None)
                        mx.eval(REQ.quantized_linear(linear, x)[0])  # warm plan
                        t0 = time.perf_counter()
                        mx.eval([REQ.quantized_linear(linear, x)[0] for _ in range(args.calls)])
                        timings[g].append((time.perf_counter() - t0) * 1e6 / args.calls)
                forced["g"] = None
                us = {g: statistics.median(v) for g, v in timings.items()}
                valid = [g for g in candidates if exact[g]]
                best = min(valid, key=us.__getitem__) if valid else None
                gain = (us[heuristic] - us[best]) / us[heuristic] if best else 0.0
                # Small projections are launch-bound and jittery: a median win
                # must also hold pass by pass, or it is noise, not a setting.
                wins = (
                    sum(b < h for b, h in zip(timings[best], timings[heuristic]))
                    if best is not None
                    else 0
                )
                win_rate = wins / args.passes
                key = planner.table_key(bits, n, k, rows)
                written = (
                    best is not None
                    and best != heuristic
                    and gain >= args.min_gain
                    and win_rate >= args.min_win_rate
                )
                if written:
                    table[key] = best
                cells.append(
                    {
                        "key": key,
                        "heuristic": heuristic,
                        "best": best,
                        "gain_vs_heuristic": round(gain, 4),
                        "paired_win_rate": round(win_rate, 3),
                        "written": written,
                        "us_per_call": {str(g): round(v, 2) for g, v in us.items()},
                        "bit_exact": {str(g): v for g, v in exact.items()},
                    }
                )
                print(
                    f"{key:28s} heur={heuristic} best={best} gain={gain:+.1%} "
                    f"wins={win_rate:.0%} {'W' if written else '-'} "
                    + " ".join(f"g{g}={us[g]:.1f}{'' if exact[g] else '!'}" for g in candidates),
                    flush=True,
                )
    finally:
        REQ._row_groups = original

    swap1 = swapouts()
    inexact = [c["key"] for c in cells if not all(c["bit_exact"].values())]
    receipt = {
        "device": device,
        "mlx_build": build,
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "calls_per_eval": args.calls,
        "passes": args.passes,
        "min_gain": args.min_gain,
        "min_win_rate": args.min_win_rate,
        "swapouts_delta": swap1 - swap0,
        "inexact_cells": inexact,
        "cells": cells,
        "rows_per_group": table,
    }
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    if inexact:
        print(f"REFUSING to write a table: candidates not bit-exact at {inexact}")
        return 2
    if swap1 != swap0:
        print(f"REFUSING to write a table: swap moved by {swap1 - swap0} pages")
        return 3
    if not args.no_write_table:
        planner.TABLE_DIR.mkdir(exist_ok=True)
        path = planner.TABLE_DIR / planner.table_name(device, build)
        path.write_text(
            json.dumps(
                {
                    "device": device,
                    "mlx_build": build,
                    "source": "scripts/calibrate_row_exact_tiles.py",
                    "measured_at": receipt["measured_at"],
                    "rows_per_group": table,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(f"wrote {path} ({len(table)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

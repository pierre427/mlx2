#!/usr/bin/env python3
"""Verify-row cost staircase of one model: median ms of an R-token forward.

Uses ``runtime.verify_topology.probe_row_costs`` (fresh cache per forward,
interleaved row counts) with an optional lane matmul install, so a
multi-lane verify's total row count can be read against tile edges.
Refuses to touch Metal without ``--i-own-the-gpu``.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from mtp_confidence_gpu import _install_lane  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--i-own-the-gpu", action="store_true")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-rows", type=int, default=40)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--lane-matmul", choices=("off", "auto", "crossover", "exact"), default="off")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.i_own_the_gpu:
        raise SystemExit("refusing to run on Metal without --i-own-the-gpu")
    sys.path.insert(0, str(ROOT / "src"))
    from mlx2.adapters.registry import resolve_adapter

    adapter = resolve_adapter(args.model, mtp=True)(args.model)
    lane = _install_lane(args, adapter)
    from mlx2.runtime.verify_topology import probe_row_costs, tile_edges

    text = getattr(adapter.model.args, "text_config", None)
    vocab = int(text["vocab_size"]) if isinstance(text, dict) else int(adapter.model.args.vocab_size)
    costs = probe_row_costs(adapter.model, vocab, args.max_rows, reps=args.reps)
    report = {
        "schema": "mlx2.verify_row_staircase.v1",
        "model": args.model,
        "lane_matmul": lane,
        "reps": args.reps,
        "row_cost_ms": {str(r): 1e3 * s for r, s in costs.items()},
        "tile_edges": tile_edges(costs),
    }
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("tile_edges",)}))
    print(" ".join(f"{r}:{1e3 * s:.1f}" for r, s in costs.items()))


if __name__ == "__main__":
    main()

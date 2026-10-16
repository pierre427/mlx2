#!/usr/bin/env python3
"""Compare streamed-expert residency policies from one exact access trace."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlx2.runtime.expert_atlas import load_trace_sequence
from mlx2.runtime.expert_residency_probe import (
    replay_hybrid,
    replay_per_projection_lru,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--floor-entries", type=int, default=8)
    parser.add_argument("--budget-bytes", type=int)
    parser.add_argument("--pressure-budget-gib", type=float, default=4.0)
    parser.add_argument("--output")
    args = parser.parse_args()

    metadata = json.loads(Path(args.metadata).read_text())
    layers, units, sequence, trace = load_trace_sequence(args.trace)
    projections = metadata["projections"]
    capacities = {int(row["trace_layer"]): int(row["capacity"]) for row in projections}
    row_bytes = {int(row["trace_layer"]): int(row["expert_bytes"]) for row in projections}
    expected = set(range(layers))
    if set(capacities) != expected or set(row_bytes) != expected:
        raise ValueError("capture metadata does not describe every trace projection")
    budget = int(
        args.budget_bytes
        if args.budget_bytes is not None
        else metadata["stream_plan"]["ceiling_bytes"]
    )
    decode_start = int(metadata["decode_start_call"])
    terminal_start = int(metadata["terminal_prefill_start_call"])
    serving_start = int(metadata.get(
        "serving_start_call",
        int(metadata["prefill_call_cuts"][0]) - len(projections),
    ))
    common = {
        "decode_start": decode_start,
        "terminal_start": terminal_start,
        "serving_start": serving_start,
    }
    pressure_budget = int(args.pressure_budget_gib * (1 << 30))

    results = [
        replay_per_projection_lru(sequence, capacities=capacities, **common),
        replay_hybrid(
            sequence, row_bytes=row_bytes, total_budget_bytes=budget,
            floor_entries=args.floor_entries, **common,
        ),
        replay_hybrid(
            sequence, row_bytes=row_bytes, total_budget_bytes=budget,
            floor_entries=args.floor_entries, scan_resistant=True, **common,
        ),
        replay_hybrid(
            sequence, row_bytes=row_bytes, total_budget_bytes=pressure_budget,
            floor_entries=args.floor_entries, scan_resistant=True, **common,
        ),
        replay_hybrid(
            sequence, row_bytes=row_bytes, total_budget_bytes=budget,
            floor_entries=args.floor_entries, scan_resistant=True,
            budget_schedule=((decode_start, pressure_budget),), **common,
        ),
    ]
    report = {
        "schema": "mlx2.expert-residency-policy-replay.v1",
        "trace": {**trace, "layers": layers, "units": units},
        "capture": str(Path(args.metadata).resolve()),
        "budget_bytes": budget,
        "floor_entries": args.floor_entries,
        "decode_start_call": decode_start,
        "terminal_prefill_start_call": terminal_start,
        "serving_start_call": serving_start,
        "pressure_budget_bytes": pressure_budget,
        "results": results,
        "state": "experimental_host_only_replay",
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        if output.exists() and output.stat().st_size:
            raise FileExistsError(f"refusing to overwrite nonempty {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        if temporary.exists():
            raise FileExistsError(f"refusing to replace temporary {temporary}")
        temporary.write_text(payload)
        temporary.replace(output)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()

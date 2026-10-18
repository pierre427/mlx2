#!/usr/bin/env python3
"""Bounded CPU conversion of the pinned mixed-native Nemotron 3.5 Bolt checkpoint."""
import argparse
import json
from pathlib import Path

from mlx2.conversion.nemotron35_bolt import convert_checkpoint
from mlx2.conversion.nemotron35_bolt_plan import make_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path, nargs="?")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--reserve-bytes", type=int, default=50_000_000_000)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.plan_only:
        plan = make_plan(json.loads((args.source / "config.json").read_text()))
        print(json.dumps({k: v for k, v in plan.items() if k not in {"schema", "modules"}}, indent=2))
        return
    if args.destination is None:
        parser.error("destination is required unless --plan-only is used")
    receipt = convert_checkpoint(args.source, args.destination, reserve_bytes=args.reserve_bytes,
                                 workers=args.workers,
                                 progress=lambda value: print(json.dumps(value), flush=True))
    print(json.dumps({k: v for k, v in receipt.items()
                      if k not in {"errors", "relative_rms_vs_bf16_dequant", "source_shards"}}, indent=2))


if __name__ == "__main__":
    main()

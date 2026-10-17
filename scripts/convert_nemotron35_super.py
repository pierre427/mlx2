#!/usr/bin/env python3
"""Bounded CPU conversion; requires a complete pinned, verified acquisition."""
import argparse
import json
from pathlib import Path

from mlx2.conversion.nemotron35 import convert_checkpoint
from mlx2.conversion.nemotron35_plan import make_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path, nargs="?")
    parser.add_argument("--expert-bits", type=int, choices=(4, 5, 6, 8), default=4)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--reserve-bytes", type=int, default=50_000_000_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--importance", type=Path, help="Source-bound activation moments; requires 1024 image-text samples and falls back to stored-BF16 weight-MSE Q4 below 32 observations")
    args = parser.parse_args()
    if args.plan_only:
        plan = make_plan(json.loads((args.source / "config.json").read_text()), expert_bits=args.expert_bits)
        print(json.dumps({key: value for key, value in plan.items() if key not in {"schema", "modules"}}, indent=2))
        return
    if args.destination is None:
        parser.error("destination is required unless --plan-only is used")
    receipt = convert_checkpoint(args.source, args.destination, expert_bits=args.expert_bits,
                                 reserve_bytes=args.reserve_bytes,
                                 workers=args.workers,
                                 importance_path=args.importance,
                                 progress=lambda value: print(json.dumps(value), flush=True))
    print(json.dumps({key: value for key, value in receipt.items() if key != "errors"}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create a separately fingerprinted standard-compatible SC-MXFP4 artifact.

CPU-only offline conversion; no model load, GPU use, runtime installation or
qualification. Source artifacts and existing nonempty destinations are refused.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from mlx2.runtime.microscaling.conversion import convert_artifact


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=("experts", "mlp", "all"), default="mlp")
    parser.add_argument("--rows-per-chunk", type=int, default=128)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shard-bytes", type=int, default=2 << 30)
    parser.add_argument("--allow-requantization", action="store_true",
                        help="diagnostic conversion from an already-quantized source; adds error")
    parser.add_argument("--normalize-only", action="store_true",
                        help="group numbered experts and preserve all source weight bytes; no quantization")
    args = parser.parse_args(argv)
    receipt = convert_artifact(args.source, args.output, scope=args.scope,
        rows_per_chunk=args.rows_per_chunk, workers=args.workers, shard_bytes=args.shard_bytes,
        allow_requantization=args.allow_requantization, normalize_only=args.normalize_only,
        progress=lambda value: print(json.dumps(value), flush=True))
    print(json.dumps({"status": receipt["status"], "checkpoint": receipt["checkpoint"],
                      "converted_modules": receipt["converted_modules"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Import a complete ModelOpt W4A16 checkpoint without loading the model."""
import argparse
import json

from mlx2.conversion.nvfp4 import convert_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--mode", choices=("bf16", "native"), default="bf16")
    parser.add_argument("--reserve-bytes", type=int, default=50_000_000_000)
    args = parser.parse_args()
    print(json.dumps(convert_checkpoint(args.source, args.destination,
                     mode=args.mode, reserve_bytes=args.reserve_bytes), indent=2))


if __name__ == "__main__":
    main()

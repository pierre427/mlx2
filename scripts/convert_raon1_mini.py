#!/usr/bin/env python3
"""Convert KRAFTON/Raon-1-mini (BF16 + MXFP8 experts) to the exact mlx2 mxfp8e artifact.

    PYTHONPATH=src .venv/bin/python -I scripts/convert_raon1_mini.py \
        --source ~/mlx-models/Raon-1-mini --dest ~/mlx-models/Raon-1-mini-mlx2-mxfp8e

CPU device only (a byte repack; no GPU lease).  Refuses a nonempty
destination and an incomplete download.  See src/mlx2/conversion/raon1_mini.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--recipe", choices=("mxfp8e", "affine"), default="mxfp8e",
                        help="mxfp8e: exact expert repack; affine: a4g64 + 8-bit head")
    args = parser.parse_args(argv)
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    from mlx2.conversion.raon1_mini import convert, convert_affine

    run = convert_affine if args.recipe == "affine" else convert
    print(json.dumps(run(args.source, args.dest), indent=2))


if __name__ == "__main__":
    main()

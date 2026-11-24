#!/usr/bin/env python3
"""Convert the inclusionAI/Ling-3.2-air block-FP8 checkpoint to the mlx2 e4r8 artifact.

    PYTHONPATH=src .venv/bin/python -I scripts/convert_ling32_air.py \
        --source ~/mlx-models/Ling-3.2-air --dest ~/mlx-models/Ling-3.2-air-mlx2-e4r8

Runs on the CPU device by default (``--device cpu``): conversion is
I/O-bound, stays out of the GPU lease, and its memory high-water mark is one
layer (about 6 GiB BF16 expert stack plus its quantized copy). Refuses a
nonempty destination and an incomplete download. See
src/mlx2/conversion/ling32_air.py and provenance/ling32-air.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args(argv)

    import mlx.core as mx

    mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)
    from mlx2.conversion.ling32_air import convert

    receipt = convert(args.source.expanduser(), args.dest.expanduser(),
                      log=lambda message: print(message, flush=True))
    print(json.dumps({key: receipt[key] for key in ("recipe", "seconds", "total_bytes")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

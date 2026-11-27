#!/usr/bin/env python3
"""Calibrated mixed-precision affine Raon-1-mini artifact (recipe raon1-mini-affine-mix-v3; --scorer hessian gives v2).

    PYTHONPATH=src python -s scripts/convert_raon1_mini_hq.py \
        --source ~/mlx-models/Raon-1-mini-mlx2-mxfp8e --dest ~/mlx-models/Raon-1-mini-mlx2-amix \
        [--tokens 98304] [--budget-bpw 5.0] [--dense-bits 8]

GPU device: calibration forwards and GPTQ.  Run only inside a GPU lease (the
unit script scripts/raon1_mini_hq_unit_20261125.sh).  See
src/mlx2/conversion/raon1_mini_hq.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=98304)
    parser.add_argument("--budget-bpw", type=float, default=5.0)
    parser.add_argument("--dense-bits", type=int, choices=(4, 8), default=8)
    parser.add_argument("--scorer", choices=("kl", "hessian"), default="kl",
                        help="layer cost for the knapsack: kl (recipe v3, default) or hessian (v2)")
    parser.add_argument("--kl-rows", type=int, default=4)
    parser.add_argument("--kl-skip", type=int, default=1024, help="positions skipped per probe row")
    args = parser.parse_args(argv)
    from mlx2.conversion.raon1_mini_hq import HQSettings, convert_hq

    settings = HQSettings(tokens=args.tokens, budget_bpw=args.budget_bpw, dense_bits=args.dense_bits,
                          scorer=args.scorer, kl_rows=args.kl_rows, kl_skip=args.kl_skip)
    receipt = convert_hq(args.source, args.dest, settings, log=lambda m: print(m, flush=True))
    print(json.dumps({k: receipt[k] for k in ("recipe", "schedule", "moe_mean_bpw", "seconds")}, indent=1))


if __name__ == "__main__":
    main()

"""Micro-timing: drafter rows qmv (kernel alone) vs stock quantized_matmul.

Throughput of back-to-back launches, microseconds per call, at the Super VL
LiLiCorr drafter's main projection shapes.  Random weights; not a served
benchmark.  Takes well under a minute on an M5 Max.

    PYTHONPATH=src python scripts/bench_drafter_rows_qmv.py
"""

import json
import sys
import time

import mlx.core as mx

from mlx2.runtime.models import drafter_rows_qmv as drq

SHAPES = [(12288, 4096, "gate/up"), (4096, 12288, "down"), (4096, 4096, "q/o"),
          (131072, 4096, "lm_head")]
MODES = [("affine", 64), ("nvfp4", 16)]
ROWS = [1, 2, 3, 4, 6, 7, 8, 10, 12, 14, 16]


def bench(fn, n=30):
    for _ in range(3):
        mx.eval(fn())
    start = time.perf_counter()
    outs = [fn() for _ in range(n)]
    mx.eval(outs)
    return (time.perf_counter() - start) / n * 1e6


def main():
    results = []
    for n, k, label in SHAPES:
        weight = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
        for mode, gs in MODES:
            parts = mx.quantize(weight, group_size=gs, bits=4, mode=mode)
            w, s = parts[0], parts[1]
            b = parts[2] if mode == "affine" else None
            row = {"shape": f"{n}x{k}", "proj": label, "mode": f"{mode}_g{gs}",
                   "stock_us": {}, "kernel_us": {}}
            for r in ROWS:
                x = mx.random.normal((1, r, k)).astype(mx.bfloat16)
                row["stock_us"][r] = round(bench(lambda: mx.quantized_matmul(
                    x, w, s, b, transpose=True, group_size=gs, bits=4, mode=mode)), 1)
                if r >= drq.MIN_ROWS:
                    row["kernel_us"][r] = round(bench(lambda: drq.rows_qmv(
                        x, w, s, b, mode=mode, group_size=gs)), 1)
            results.append(row)
            print(json.dumps(row), flush=True)
            del parts, w, s, b
        del weight
    print(json.dumps({"device": mx.device_info().get("device_name"),
                      "engine": drq._engine(4096)}), file=sys.stderr)


if __name__ == "__main__":
    main()

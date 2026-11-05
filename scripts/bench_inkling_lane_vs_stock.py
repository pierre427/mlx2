#!/usr/bin/env python3
"""Lane (row-invariant) vs stock BF16 matmul at Inkling's dense shapes (diagnostic).

AST runs every dense BF16 projection under the exact lane law, including the
one-row steps it takes while parked.  This times both kernels at 1/4/8 rows
for each projection shape (40 issued per eval) and reports effective GB/s.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mlx.core as mx  # noqa: E402
from mlx import nn  # noqa: E402

SHAPES = {  # name: (out, in, count per token)
    "qkvr": (6656, 4096, 42), "o_proj": (4096, 4096, 42),
    "shared_gate_up": (8192, 4096, 40), "shared_down": (4096, 4096, 40),
    "dense_gate_up": (32768, 4096, 2), "dense_down": (4096, 16384, 2),
    "lm_head": (201024, 4096, 1),
}
ISSUES = 40


def timed(fn):
    mx.eval(fn())
    samples = []
    for _ in range(9):
        mx.synchronize()
        tic = time.perf_counter()
        mx.eval([fn() for _ in range(ISSUES)])
        samples.append((time.perf_counter() - tic) / ISSUES * 1e6)
    return statistics.median(samples)


def main():
    from mlx2.runtime.lane import lane_matmul, prepare

    mx.set_default_device(mx.gpu)
    mx.random.seed(5)
    out = {"cells": []}
    per_token = {"stock": 0.0, "lane": 0.0}
    for name, (n, k, count) in SHAPES.items():
        layer = nn.Linear(k, n, bias=False)
        layer.weight = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
        mx.eval(layer.parameters())
        lw = prepare(layer)
        nbytes = n * k * 2
        for rows in (1, 4, 8):
            x = mx.random.normal((1, rows, k)).astype(mx.bfloat16)
            stock = timed(lambda: layer(x))
            lane = timed(lambda: lane_matmul(x, lw))
            cell = {"shape": name, "rows": rows, "stock_us": round(stock, 1), "lane_us": round(lane, 1),
                    "stock_gbs": round(nbytes / stock / 1e3, 1), "lane_gbs": round(nbytes / lane / 1e3, 1)}
            out["cells"].append(cell)
            print(json.dumps(cell), flush=True)
            if rows == 1:
                per_token["stock"] += stock * count / 1000
                per_token["lane"] += lane * count / 1000
    out["one_row_ms_per_token"] = per_token
    print(json.dumps({"one_row_ms_per_token": per_token}), flush=True)


if __name__ == "__main__":
    main()

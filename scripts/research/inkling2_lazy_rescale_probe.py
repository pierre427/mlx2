#!/usr/bin/env python3
"""Operator A/B for ``inkling2_prefill_lazy_rescale`` (prefill_attention.py).

Runs the Inkling 2 tiled prefill kernel with the O-rescale skip off and on
over Inkling-2-Mini global-layer geometry (HQ 24, HKV 8, D 128, E 1024, BF16),
one prefill chunk at several key depths. Checks the outputs are bit-identical
and times both arms interleaved (ABBA). Synthetic inputs (RMS-normed q/k, as
the model's q_norm/k_norm leave them); operator time, not tok/s.
Writes one JSON receipt to --out.
"""
import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import mlx.core as mx

from mlx2.runtime.models.inkling2 import prefill_attention as pa


def inputs(seed, chunk, depth):
    mx.random.seed(seed)
    size = depth + chunk
    norm = lambda x: x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-6)
    q = norm(mx.random.normal((1, 24, chunk, 128))).astype(mx.bfloat16)
    k = norm(mx.random.normal((1, 8, size, 128))).astype(mx.bfloat16)
    v = mx.random.normal((1, 8, size, 128)).astype(mx.bfloat16)
    rel = (0.5 * mx.random.normal((1, chunk, 24, 1024))).astype(mx.bfloat16)
    sinks = mx.random.normal((1, chunk, 24)).astype(mx.float32)
    mx.eval(q, k, v, rel, sinks)
    return q, k, v, rel, sinks, depth


def run(lazy, args):
    q, k, v, rel, sinks, depth = args
    pa.set_lazy_rescale(lazy)
    out = pa.tiled_prefill_attention(q, k, v, rel, sinks, offset=depth, left_padding=None,
                                     window=0, scale=128 ** -0.5)
    mx.eval(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--depths", default="0,14336,30720,63488")
    ap.add_argument("--reps", type=int, default=8)
    a = ap.parse_args()
    cells = []
    for depth in [int(x) for x in a.depths.split(",")]:
        args = inputs(7, a.chunk, depth)
        base, lazy = run(False, args), run(True, args)
        exact = bool(mx.array_equal(base.view(mx.uint16), lazy.view(mx.uint16)).item())
        times = {False: [], True: []}
        for _ in range(2):  # warm-up both pipelines
            run(False, args); run(True, args)
        for rep in range(a.reps):
            for arm in ((False, True) if rep % 2 == 0 else (True, False)):
                t = time.perf_counter(); run(arm, args); times[arm].append(time.perf_counter() - t)
        med = lambda xs: sorted(xs)[len(xs) // 2] * 1e3
        cells.append({"depth": depth, "chunk": a.chunk, "bitwise_equal": exact,
                      "eager_ms_median": med(times[False]), "lazy_ms_median": med(times[True]),
                      "eager_ms": [x * 1e3 for x in times[False]],
                      "lazy_ms": [x * 1e3 for x in times[True]]})
        print(json.dumps({k: cells[-1][k] for k in ("depth", "bitwise_equal", "eager_ms_median",
                                                     "lazy_ms_median")}), flush=True)
        del args, base, lazy
    pa.set_lazy_rescale(False)
    root = Path(__file__).resolve().parents[2]
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=root).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "src"], capture_output=True,
                                text=True, cwd=root).stdout.strip())
    receipt = {"schema": "mlx2.inkling2-lazy-rescale-probe.v1", "mlx2_commit": rev,
               "src_dirty": dirty, "mlx": mx.__version__, "machine": platform.node(),
               "checkpoint": "synthetic (Inkling-2-Mini global-layer geometry; no weights)",
               "cells": cells}
    with open(a.out, "w") as f:
        json.dump(receipt, f, indent=1)


if __name__ == "__main__":
    main()

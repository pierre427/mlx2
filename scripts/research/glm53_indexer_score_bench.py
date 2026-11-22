#!/usr/bin/env python3
"""GLM-5.3 indexer score-law microbench (glm53_indexer_score; mlx-lm#2046 fusion).

Synthetic, no checkpoint: one 128-query prompt chunk (QUERY_CHUNK) against P
pools (P = context / 4) through ``blocked_select`` (SELECT_BLOCK pools per
block, the prompt-phase path) under each score law, interleaved. Records
median ms per chunk, MLX peak bytes, the score-law counters, and whether the
ids and the raw scores of every law equal the reference bit for bit. GPU only:
pass ``--i-own-the-gpu`` and run under the GPU lease. Exploratory timing, not
publishable statistics (no thermal gate).
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new or empty evidence directory")
    parser.add_argument("--contexts", default="16384,65536,131072")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--i-own-the-gpu", action="store_true")
    args = parser.parse_args()
    if not args.i_own_the_gpu:
        raise SystemExit("GPU only: pass --i-own-the-gpu under the GPU lease")
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"refusing nonempty evidence directory {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx
    import numpy as np

    from mlx2.runtime.models import glm53_gguf_indexer as ix

    mx.set_default_device(mx.gpu)
    gate = {"score_kernel_ready": ix.score_kernel_ready(), "gate_error": ix._GATE["score_kernel_error"]}
    rng = np.random.default_rng(5430)
    length, block = ix.QUERY_CHUNK, 4096
    cells = []
    for context in (int(c) for c in args.contexts.split(",")):
        pools = context // ix.KPOOL
        q = mx.array(rng.normal(size=(1, length, 32, 128)).astype(np.float32))
        w = mx.array((rng.normal(size=(1, length, 32)) * 32**-0.5).astype(np.float32))
        pooled = mx.array(rng.normal(size=(1, pools, 128)).astype(np.float32))
        positions = mx.arange(context - length, context, dtype=mx.int32)
        mx.eval(q, w, pooled, positions)
        times = {law: [] for law in ix.SCORE_LAWS}
        peaks, ids = {}, {}
        for rep in range(args.reps + 1):  # rep 0 warms up (compile, gate, JIT)
            order = list(ix.SCORE_LAWS)[rep % 3:] + list(ix.SCORE_LAWS)[: rep % 3]
            for law in order:
                mx.reset_peak_memory()
                start = time.perf_counter()
                got = ix.blocked_select(q, w, pooled, positions, 512, block=block, score=law)
                mx.eval(got)
                elapsed = (time.perf_counter() - start) * 1e3
                if rep:
                    times[law].append(elapsed)
                    peaks[law] = max(peaks.get(law, 0), mx.get_peak_memory())
                ids[law] = np.array(got[0])
        sample = pooled[:, : min(pools, block)]
        ref_scores = np.array(ix.pool_scores(q, w, sample)).view(np.uint32)
        cell = {"context": context, "pools": pools, "query_rows": length, "block": block,
                "median_ms": {law: statistics.median(v) for law, v in times.items()},
                "peak_bytes": peaks,
                "ids_equal_reference": {law: bool(np.array_equal(ids[law], ids["reference"]))
                                        for law in ix.SCORE_LAWS},
                "scores_equal_reference_first_block": {
                    law: bool(np.array_equal(np.array(ix.pool_scores(q, w, sample, law=law)).view(np.uint32),
                                             ref_scores)) for law in ix.SCORE_LAWS}}
        cells.append(cell)
        print(json.dumps(cell), flush=True)
    receipt = {
        "tool": "scripts/research/glm53_indexer_score_bench.py",
        "source_revision": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                          capture_output=True, text=True).stdout.strip(),
        "checkpoint": "none (synthetic operands)",
        "mlx": mx.__version__, "device": str(mx.default_device()),
        "gate": gate, "counters": dict(ix._SCHEDULED), "cells": cells,
        "verdict": "pass" if all(all(c["ids_equal_reference"].values())
                                 and all(c["scores_equal_reference_first_block"].values())
                                 for c in cells) else "fail",
    }
    tmp = args.out / "receipt.json.tmp"
    tmp.write_text(json.dumps(receipt, indent=1))
    tmp.replace(args.out / "receipt.json")
    print(receipt["verdict"])
    return 0 if receipt["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Operator A/B for the QSA NAX v2 lazy O rescale (qwen4_qsa_nax.set_lazy_rescale).

Synthetic BF16 q/k/v with the Flash-Next attention geometry (D 256, 24 heads,
2 KV heads) and the bench's top-512 block selections
(scripts/bench_qsa_nax_kernel_ab.make_case), one 2048-row prefill chunk at
several contexts. Law v2 (the Flash-Next default) eager vs lazy: bitwise
check, then interleaved ABBA timing. Operator time, not tok/s.

Usage: python scripts/research/qsa_nax_lazy_rescale_probe.py --out probe.json
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import mlx.core as mx  # noqa: E402

import bench_qsa_nax_kernel_ab as bench  # noqa: E402
from mlx2.runtime.models import qwen4_qsa_nax as nax  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--contexts", default="16384,32768,131072")
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--laws", default="v2,v2_hilo")
    a = ap.parse_args()
    cells = []
    for context in [int(x) for x in a.contexts.split(",")]:
        case = bench.make_case("fn", context, a.rows)
        ids, counts, n_sel, u_width, q_pos, left_pad, total = bench.compact(case)
        mx.eval(ids, counts, n_sel, q_pos, left_pad)
        kw = dict(scale=bench.D ** -0.5, u_width=u_width, total=total, n_kv_heads=case.kvh)
        for law in a.laws.split(","):
            def run(lazy):
                nax.set_lazy_rescale(lazy)
                out = nax.nax_qsa_attention(case.q, case.k, case.v, ids, counts, n_sel, q_pos,
                                            left_pad, law=law, **kw)
                mx.eval(out)
                return out
            eager, lazy = run(False), run(True)
            exact = bool(mx.array_equal(eager.view(mx.uint32), lazy.view(mx.uint32)).item())
            for _ in range(2):
                run(False); run(True)
            times = {False: [], True: []}
            for rep in range(a.reps):
                for arm in ((False, True) if rep % 2 == 0 else (True, False)):
                    t = time.perf_counter(); run(arm); times[arm].append((time.perf_counter() - t) * 1e3)
            med = lambda xs: sorted(xs)[len(xs) // 2]  # noqa: E731
            cell = {"context": context, "rows": a.rows, "law": law, "bitwise_equal": exact,
                    "eager_ms_median": med(times[False]), "lazy_ms_median": med(times[True]),
                    "eager_ms": times[False], "lazy_ms": times[True]}
            cells.append(cell)
            print(json.dumps({k: cell[k] for k in ("context", "law", "bitwise_equal",
                                                   "eager_ms_median", "lazy_ms_median")}), flush=True)
        del case
    nax.set_lazy_rescale(False)
    git = lambda *x: subprocess.run(["git", *x], capture_output=True, text=True, cwd=ROOT).stdout.strip()  # noqa: E731
    receipt = {"schema": "mlx2.qsa-nax-lazy-rescale-probe.v1", "mlx2_commit": git("rev-parse", "HEAD"),
               "src_dirty": bool(git("status", "--porcelain", "src")), "mlx": mx.__version__,
               "machine": platform.node(),
               "checkpoint": "synthetic (" + bench.GEOMETRIES["fn"]["geometry_of"] + " geometry; no weights)",
               "cells": cells}
    Path(a.out).write_text(json.dumps(receipt, indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""A/B/A/B Metal timing of the GDN prework kernels in both SiLU spellings.

The 2026-10-17 fix compiles the served ``metal::precise::exp`` instead of the
old fixed ``metal::exp`` in the Qwen3.8 tree/cohort pre kernel, the fused
prefill prework (Flash-Next and Qwen3.8 numerics) and the Qwen3.5 swish norm
gate.  This times both compiled spellings of each kernel in one process,
alternating arms block by block (A B A B ...), so drift and thermals hit both
arms alike.  Each block is ``--iters`` launches of one kernel on fixed
inputs, each evaluated; the reported figure per arm is the median block mean
in microseconds per launch.  Run under the GPU lock with ``--i-own-the-gpu``.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ARMS = ("metal::exp", "metal::precise::exp")


def _cases(mx, rows_tree, rows_prefill):
    from mlx2.runtime.models import qwen38_tree_gdn as tree
    from mlx2.runtime.models import qwen4_fused_gdn_prefill as prefill

    nk, nv, dk, dv, taps = 16, 48, 128, 128, 4
    c = 2 * nk * dk + nv * dv
    key = mx.random.key(1)
    ks = mx.random.split(key, 8)
    cases = {}

    qkv_t = mx.random.normal((1, rows_tree, c), key=ks[0]).astype(mx.bfloat16)
    state_t = mx.random.normal((1, taps - 1, c), key=ks[1]).astype(mx.bfloat16)
    weight = (mx.random.normal((c, taps, 1), key=ks[2]) * 0.5).astype(mx.bfloat16)
    a = mx.random.normal((1, rows_tree, nv), key=ks[3]).astype(mx.bfloat16)
    b = mx.random.normal((1, rows_tree, nv), key=ks[4]).astype(mx.bfloat16)
    a_log = mx.linspace(-2.0, 1.0, nv).astype(mx.bfloat16)
    dt_bias = mx.zeros((nv,), dtype=mx.bfloat16)
    windows = tree.ints([t + r for r in range(rows_tree) for t in range(taps)])
    row_stream = mx.zeros((rows_tree,), dtype=mx.int32)

    def tree_pre(spelling):
        kernel = tree._build(spelling)
        return lambda: tree._launch(kernel, qkv_t, [state_t], weight, windows, a, b, a_log,
                                    dt_bias, row_stream, nk=nk, nv=nv, dk=dk, dv=dv,
                                    zs=nv, ao=0, bo=0)

    cases[f"tree_pre_r{rows_tree}"] = tree_pre

    qkv_p = mx.random.normal((1, rows_prefill, c), key=ks[5]).astype(mx.bfloat16)

    def prework(numerics):
        gate = "sigmoid" if numerics == "qwen4" else "swish"

        def make(spelling):
            kernel = prefill._build(numerics, gate, spelling, spelling)[0]
            return lambda: prefill.qwen4_gdn_prefill_prework(
                qkv_p, state_t, weight, numerics=numerics, _kernel=kernel)

        return make

    for numerics in prefill.PREWORK_NUMERICS:
        cases[f"prefill_prework_{numerics}_s{rows_prefill}"] = prework(numerics)

    y = mx.random.normal((1, rows_prefill, nv, dv), key=ks[6]).astype(mx.bfloat16)
    z = mx.random.normal((1, rows_prefill, nv * dv), key=ks[7]).astype(mx.bfloat16)
    w = mx.ones((dv,), dtype=mx.bfloat16)

    def swish(spelling):
        kernel = prefill._build("qwen35", "swish", spelling, spelling)[1]
        return lambda: prefill.qwen4_gdn_prefill_norm_gate(y, z, w, 1e-6, gate="swish",
                                                           _kernel=kernel)

    cases[f"norm_gate_swish_s{rows_prefill}"] = swish
    return cases


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=6, help="A/B pairs per kernel")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--rows-tree", type=int, default=16)
    ap.add_argument("--rows-prefill", type=int, default=2048)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    args = ap.parse_args(argv)
    if not args.i_own_the_gpu:
        ap.error("run under scripts/run_with_gpu_locks.py and pass --i-own-the-gpu")
    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    report = {"git_head": head, "mlx": mx.__version__, "rounds": args.rounds,
              "iters": args.iters, "order": "A B A B ... (A = metal::exp, B = metal::precise::exp)",
              "kernels": {}}
    for name, make in _cases(mx, args.rows_tree, args.rows_prefill).items():
        runs = {arm: make(arm) for arm in ARMS}
        for arm in ARMS:  # compile + warm both arms first
            for _ in range(20):
                mx.eval(*runs[arm]())
        blocks = {arm: [] for arm in ARMS}
        for _ in range(args.rounds):
            for arm in ARMS:
                mx.synchronize()
                t0 = time.perf_counter()
                for _ in range(args.iters):
                    mx.eval(*runs[arm]())
                mx.synchronize()
                blocks[arm].append((time.perf_counter() - t0) / args.iters * 1e6)
        med = {arm: statistics.median(v) for arm, v in blocks.items()}
        report["kernels"][name] = {
            "us_per_launch_blocks": {arm: [round(x, 2) for x in v] for arm, v in blocks.items()},
            "median_us": {arm: round(v, 2) for arm, v in med.items()},
            "precise_over_fast": round(med[ARMS[1]] / med[ARMS[0]], 4),
            # Each round's B block over the A block just before it: robust to
            # slow drift that a median of unpaired blocks is not.
            "paired_ratio_median": round(statistics.median(
                b / a for a, b in zip(blocks[ARMS[0]], blocks[ARMS[1]])), 4),
        }
        print(name, json.dumps(report["kernels"][name]["median_us"]),
              report["kernels"][name]["precise_over_fast"],
              report["kernels"][name]["paired_ratio_median"], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

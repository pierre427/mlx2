#!/usr/bin/env python3
"""Microbenchmark MLX quantized matmul at M=1..8 columns (llama.cpp#29110 probe).

llama.cpp#29110 reports a Metal multi-column matvec for Q4/Q8 at 2-8 columns
running 1.3-1.9x faster per op than column-by-column matvec, which is +37% end to
end on a dense 27B with MTP, and +4% on 35B-A3B (M3 Ultra only). A verify window
of k+1 tokens is exactly an M=k+1 qmm. This probe asks whether MLX's qmm on
this machine already amortizes the weight read across M columns. It prints
time(M) / time(1). A value near 1.0 for M<=4 means the multi-column path is
already effectively free and the idea has no headroom here.

Metal only. Refuses to run without ``--i-own-the-gpu``. ``--dry-run`` prints the
plan.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import mlx.core as mx

# (label, out_features, in_features)
SHAPES = {
    # Dense 27B-class projections (hidden 5120).
    "dense27b_qkv": (8192, 5120),
    "dense27b_up": (17408, 5120),
    "dense27b_down": (5120, 17408),
    # 35B-A3B-class shared projections (hidden 2048).
    "a3b_attn": (4096, 2048),
    "a3b_expert_up": (512, 2048),
    "a3b_lm_head": (151936, 2048),
}


def _bench(m: int, out_f: int, in_f: int, bits: int, group: int, iters: int, warmup: int) -> float:
    w = mx.random.normal((out_f, in_f)).astype(mx.float16)
    wq, scales, biases = mx.quantize(w, group_size=group, bits=bits)
    x = mx.random.normal((m, in_f)).astype(mx.float16)
    mx.eval(wq, scales, biases, x)

    def step():
        return mx.quantized_matmul(x, wq, scales, biases, transpose=True, group_size=group, bits=bits)

    for _ in range(warmup):
        mx.eval(step())
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        # Chain 20 ops per sample so launch overhead does not dominate.
        mx.eval([step() for _ in range(20)])
        samples.append((time.perf_counter() - t0) / 20)
    return statistics.median(samples)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--i-own-the-gpu", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--bits", type=int, nargs="+", default=[4, 8])
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--max-m", type=int, default=8)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--shapes", nargs="+", default=list(SHAPES))
    args = p.parse_args()

    plan = {
        "device": "gpu",
        "bits": args.bits,
        "group_size": args.group_size,
        "m": list(range(1, args.max_m + 1)),
        "shapes": {k: SHAPES[k] for k in args.shapes},
        "will_execute": bool(args.i_own_the_gpu and not args.dry_run),
    }
    print(json.dumps({"plan": plan}, indent=2))
    if args.dry_run:
        return
    if not args.i_own_the_gpu:
        p.error("refusing Metal execution without --i-own-the-gpu")

    mx.set_default_device(mx.gpu)
    rows = []
    for name in args.shapes:
        out_f, in_f = SHAPES[name]
        for bits in args.bits:
            base = None
            for m in range(1, args.max_m + 1):
                t = _bench(m, out_f, in_f, bits, args.group_size, args.iters, args.warmup)
                base = base or t
                row = {
                    "shape": name,
                    "bits": bits,
                    "m": m,
                    "us": round(t * 1e6, 2),
                    "ratio_vs_m1": round(t / base, 3),
                    "per_column_speedup_vs_m_matvecs": round(m * base / t, 3),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
    print(json.dumps({"device_info": mx.metal.device_info() if hasattr(mx, "metal") else None}, default=str))


if __name__ == "__main__":
    main()

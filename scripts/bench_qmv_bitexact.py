#!/usr/bin/env python3
"""Cache-cold cost of the mlx bit-exact (batch-invariant) qmv mode.

GPU only. Needs an mlx build with ``mx.metal.set_qmv_bitexact``; the fork
branch is ``claude/rm09-bitexact-verify-20260919``. Point PYTHONPATH at an
overlay directory holding that build.

For each 27B verify shape (4- and 5-bit, gs 64) and each M in
{1, 2, 3, 4, 6, 8, 12, 16}, the script times the default routing (qmv_wide,
NAX or split-K) against the bit-exact routing. The arms are interleaved per
repetition. The timing is cache-cold: each timed op rotates through about
1 GiB of distinct weight copies, so the weights stream from DRAM. (A warm
cache faked 2x headroom before; see
qualification/runs/qmm-verify-flatten-tile8-20260919.)

Gates, all hard failures:
- A bit-exact arm whose ``qmv_bitexact_dispatches`` delta is 0 is refused.
- A default arm whose delta is non-zero is refused.
- Each bit-exact output row must equal the M = 1 result bitwise.

Output is JSONL: one row per (bits, shape, M, mode) with the median
microseconds per op, the ratio to the same mode at M = 1, and the ratio to
default at the same M.

  python scripts/bench_qmv_bitexact.py --i-own-the-gpu --out bench.jsonl
  python scripts/bench_qmv_bitexact.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

# (N, K) as (out, in): Qwen3.8-27B-oQ4e-mtp projections (safetensors headers;
# 4-bit with 5-bit overrides on some q/k/z/down/in_proj layers, gs 64).
SHAPES = {
    "27b_q": (12288, 5120),
    "27b_kv": (1024, 5120),
    "27b_o": (5120, 6144),
    "27b_gdn_qkv": (10240, 5120),
    "27b_gdn_z": (6144, 5120),
    "27b_gdn_out": (5120, 7680),
    "27b_gdn_ab": (48, 5120),
    "27b_gate_up": (17408, 5120),
    "27b_down": (5120, 17408),
    "27b_lm_head": (248320, 5120),
}
MS = (1, 2, 3, 4, 6, 8, 12, 16)
TARGET_BYTES = 1 << 30


def plan(args):
    rows = []
    for bits in args.bits:
        for name in args.shapes:
            for m in MS:
                for mode in ("default", "bitexact"):
                    rows.append({"bits": bits, "shape": name, "m": m, "mode": mode})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--i-own-the-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out")
    parser.add_argument("--bits", type=int, nargs="+", default=[4, 5])
    parser.add_argument("--shapes", nargs="+", default=list(SHAPES))
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--iters", type=int, default=6)
    args = parser.parse_args()
    unknown = set(args.shapes) - set(SHAPES)
    if unknown:
        parser.error(f"unknown shapes {sorted(unknown)}")
    if args.dry_run:
        for row in plan(args):
            print(json.dumps(row))
        print(f"# {len(plan(args))} arms; cache-cold rotation ~{TARGET_BYTES >> 20} MiB")
        return 0
    if not args.i_own_the_gpu:
        parser.error("refusing to run Metal kernels without --i-own-the-gpu")
    if not args.out:
        parser.error("--out is required")

    import mlx.core as mx

    for name in ("set_qmv_bitexact", "qmv_bitexact_dispatches"):
        if not hasattr(mx.metal, name):
            sys.exit(f"mlx {mx.__version__} has no mx.metal.{name}; use the rm09 fork build")
    mx.set_default_device(mx.gpu)
    previous = mx.metal.set_qmv_bitexact(False)
    out = open(args.out, "a")
    gs = args.group_size
    try:
        for bits in args.bits:
            for name in args.shapes:
                N, K = SHAPES[name]
                per = N * K * bits // 8
                copies = max(2, min(64, TARGET_BYTES // per))
                mats = []
                for _ in range(copies):
                    w = (mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16)
                    mats.append(mx.quantize(w, group_size=gs, bits=bits))
                    mx.eval(mats[-1])
                    del w
                xs = mx.random.normal((max(MS), K)).astype(mx.bfloat16)
                mx.eval(xs)
                # Correctness gate on one copy: bit-exact rows == M = 1 rows.
                mx.metal.set_qmv_bitexact(True)
                q = mats[0]
                singles = mx.concatenate(
                    [
                        mx.quantized_matmul(xs[i : i + 1], *q, transpose=True, group_size=gs, bits=bits)
                        for i in range(max(MS))
                    ]
                )
                for m in MS:
                    y = mx.quantized_matmul(xs[:m], *q, transpose=True, group_size=gs, bits=bits)
                    if not mx.array_equal(y.view(mx.uint16), singles[:m].view(mx.uint16)).item():
                        sys.exit(f"bit-exact mismatch: bits={bits} shape={name} M={m}")
                mx.metal.set_qmv_bitexact(False)

                times = {}
                for rep in range(args.reps):
                    order = ("default", "bitexact") if rep % 2 == 0 else ("bitexact", "default")
                    for m in MS:
                        x = xs[:m]
                        for mode in order:
                            mx.metal.set_qmv_bitexact(mode == "bitexact")

                            def run():
                                return [
                                    mx.quantized_matmul(x, *qq, transpose=True, group_size=gs, bits=bits)
                                    for qq in mats
                                ]

                            mx.eval(run())  # warm the pipeline, not the cache
                            before = mx.metal.qmv_bitexact_dispatches()
                            samples = []
                            for _ in range(args.iters):
                                t0 = time.perf_counter()
                                mx.eval(run())
                                samples.append((time.perf_counter() - t0) / copies)
                            delta = mx.metal.qmv_bitexact_dispatches() - before
                            if mode == "bitexact" and delta <= 0:
                                sys.exit(f"refusing arm: bit-exact route never engaged ({name} M={m})")
                            if mode == "default" and delta != 0:
                                sys.exit(f"refusing arm: default arm routed bit-exact ({name} M={m})")
                            times.setdefault((m, mode), []).extend(samples)
                        mx.metal.set_qmv_bitexact(False)
                base = {mode: statistics.median(times[(1, mode)]) for mode in ("default", "bitexact")}
                for m in MS:
                    d = statistics.median(times[(m, "default")])
                    for mode in ("default", "bitexact"):
                        t = statistics.median(times[(m, mode)])
                        row = {
                            "bits": bits,
                            "shape": name,
                            "N": N,
                            "K": K,
                            "m": m,
                            "mode": mode,
                            "us": round(t * 1e6, 1),
                            "rel_m1": round(t / base[mode], 3),
                            "rel_default": round(t / d, 3),
                            "copies": copies,
                        }
                        print(json.dumps(row), flush=True)
                        out.write(json.dumps(row) + "\n")
                        out.flush()
                mats.clear()
                mx.clear_cache()
    finally:
        mx.metal.set_qmv_bitexact(previous)
        out.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

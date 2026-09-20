#!/usr/bin/env python3
"""Multi-LoRA per-row delta cost model (CPU) and kernel microbench (GPU-gated).

For one Linear of shape (in, out) and a batch of B rows spread over N adapters
of rank R, time the batched LoRA delta variants against the base matmul:

* ``gather``  -- ``gather_mm(gather_mm(x, A, rhs=ids), B, rhs=ids)`` (the
  shipped path in ``mlx2.runtime.multi_lora``);
* ``sorted``  -- rows permuted so ids are sorted, ``sorted_indices=True``, then
  un-permuted (the segment/SGMV-style variant);
* ``take``    -- ``take`` the per-row A/B then batched ``matmul`` (materializes
  B copies of the adapter weights).

Every arm verifies its output against the ``gather`` arm and the delta is
checked to be nonzero; an arm that did not run the mechanism is refused.

``--device gpu`` refuses to run without ``--i-own-the-gpu``.  With
``--cold-rotate-gib`` the base weight is rotated through that many GiB of
copies so small-M numbers are cache-cold.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    p.add_argument("--i-own-the-gpu", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--batch", default="1,4,16,32")
    p.add_argument("--adapters", default="1,2,4")
    p.add_argument("--ranks", default="8,16,64")
    p.add_argument("--shapes", default="2048x2048,4096x4096")
    p.add_argument("--seq", type=int, default=1, help="tokens per row (1 = decode)")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--quantized-base", action="store_true", help="4-bit base weight (GPU realism)")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--cold-rotate-gib", type=float, default=0.0)
    p.add_argument("--out")
    return p.parse_args()


def ints(text):
    return [int(v) for v in text.split(",") if v]


def main():
    args = parse()
    shapes = [tuple(int(v) for v in s.split("x")) for s in args.shapes.split(",")]
    grid = [
        (b, n, r, shape)
        for shape in shapes
        for r in ints(args.ranks)
        for n in ints(args.adapters)
        for b in ints(args.batch)
    ]
    if args.dry_run:
        print(json.dumps({"device": args.device, "cells": len(grid), "grid": grid[:8]}, indent=1))
        return 0
    if args.device == "gpu" and not args.i_own_the_gpu:
        print("refusing to touch Metal without --i-own-the-gpu", file=sys.stderr)
        return 2
    import mlx.core as mx

    mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)
    dtype = getattr(mx, args.dtype)
    results = []
    for batch, adapters, rank, (din, dout) in grid:
        mx.random.seed(0)
        slots = adapters + 1
        weight = mx.random.normal((dout, din)).astype(dtype) * 0.02
        copies = 1
        if args.cold_rotate_gib > 0:
            copies = max(1, int(args.cold_rotate_gib * (1 << 30) // weight.nbytes))
        if args.quantized_base:
            base_weights = [mx.quantize(weight + i * 1e-3, group_size=64, bits=4) for i in range(copies)]
        else:
            base_weights = [weight + i * 1e-3 for i in range(copies)]
        mx.eval(base_weights)
        A = (mx.random.normal((slots, din, rank)) * 0.02).astype(dtype)
        Bm = (mx.random.normal((slots, rank, dout)) * 0.02).astype(dtype)
        A = mx.concatenate([mx.zeros_like(A[:1]), A[1:]])
        Bm = mx.concatenate([mx.zeros_like(Bm[:1]), Bm[1:]])
        # Rows round-robin over adapters 1..N (fully mixed batch).
        ids_list = [1 + (i % adapters) for i in range(batch)]
        ids = mx.array(ids_list, dtype=mx.uint32)
        perm = mx.argsort(ids)
        inv = mx.argsort(perm)
        sorted_ids = ids[perm]
        x = mx.random.normal((batch, args.seq, din)).astype(dtype)
        mx.eval(A, Bm, ids, perm, inv, sorted_ids, x)

        def base(i):
            w = base_weights[i % copies]
            if args.quantized_base:
                wq, s, bq = w
                return mx.quantized_matmul(x, wq, s, bq, transpose=True, group_size=64, bits=4)
            return x @ w.T

        def gather(_i):
            return mx.gather_mm(mx.gather_mm(x, A, None, ids), Bm, None, ids)

        def sorted_arm(_i):
            xs = x[perm]
            h = mx.gather_mm(xs, A, None, sorted_ids, sorted_indices=True)
            return mx.gather_mm(h, Bm, None, sorted_ids, sorted_indices=True)[inv]

        def take(_i):
            return (x @ mx.take(A, ids, axis=0)) @ mx.take(Bm, ids, axis=0)

        reference = gather(0)
        mx.eval(reference)
        if not bool(mx.any(reference != 0).item()):
            raise SystemExit("mechanism gate: gather delta is identically zero")
        row = {"batch": batch, "adapters": adapters, "rank": rank, "in": din, "out": dout, "seq": args.seq,
               "dtype": args.dtype, "quantized_base": args.quantized_base, "cold_copies": copies,
               "device": args.device}
        for name, fn in (("base", base), ("gather", gather), ("sorted", sorted_arm), ("take", take)):
            if name != "base":
                out = fn(0)
                tol = 1e-3 if dtype == mx.float32 else 5e-2
                if not mx.allclose(out, reference, atol=tol, rtol=tol).item():
                    raise SystemExit(f"mechanism gate: arm {name} disagrees with gather")
            for i in range(args.warmup):
                mx.eval(fn(i))
            times = []
            for i in range(args.iters):
                start = time.perf_counter()
                mx.eval(fn(i + args.warmup))
                times.append((time.perf_counter() - start) * 1e6)
            row[f"{name}_us"] = statistics.median(times)
        for name in ("gather", "sorted", "take"):
            row[f"{name}_over_base"] = row[f"{name}_us"] / row["base_us"]
        results.append(row)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    import mlx.core as mx2

    payload = {"schema": "mlx2.multi-lora-bench.v1", "mlx": mx2.__version__, "rows": results}
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(payload, handle, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

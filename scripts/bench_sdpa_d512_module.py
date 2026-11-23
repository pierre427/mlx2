#!/usr/bin/env python3
"""Time ``sdpa_vector_d512.attention`` (TK 2 and TK 4, stock block count) against stock MLX.

Original mlx2 script.  Gemma 4 global-attention shapes at head_dim 512 with
separate K and V buffers (the served layout), one query row, bf16.  Also
checks bitwise equality with ``mx.fast.scaled_dot_product_attention``.
GPU work: run it under the GPU lock.  Refuses to overwrite ``--out``.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import mlx.core as mx

from mlx2.runtime.models import sdpa_vector_d512 as d512


def timed(fn, iters, warmup):
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--contexts", default="4096,16384,32768,65536,131072")
    ap.add_argument("--shapes", default="31b:32:4,26b:16:2")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=5)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    mx.set_default_device(mx.gpu)
    rows = []
    for shape in a.shapes.split(","):
        name, hq, hkv = shape.split(":")
        hq, hkv = int(hq), int(hkv)
        for n in (int(x) for x in a.contexts.split(",")):
            mx.random.seed(n)
            q = mx.random.normal((1, hq, 1, 512)).astype(mx.bfloat16)
            k = mx.random.normal((1, hkv, n, 512)).astype(mx.bfloat16)
            v = mx.random.normal((1, hkv, n, 512)).astype(mx.bfloat16)
            mx.eval(q, k, v)
            stock = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0)
            row = {"shape": name, "q_heads": hq, "kv_heads": hkv, "context": n,
                   "stock_blocks": d512.stock_blocks(n, hq // hkv, d512.device_class()),
                   "kv_bytes": int(k.nbytes + v.nbytes),
                   "stock_ms": timed(lambda: mx.fast.scaled_dot_product_attention(
                       q, k, v, scale=1.0), a.iters, a.warmup)}
            for tk in (2, 4):
                res = d512.attention(q, k, v, scale=1.0, tk=tk)
                row[f"tk{tk}_bitwise_stock"] = bool(mx.array_equal(res, stock).item())
                row[f"tk{tk}_ms"] = timed(lambda: d512.attention(q, k, v, scale=1.0, tk=tk),
                                          a.iters, a.warmup)
            row["tk2_speedup_vs_stock"] = row["stock_ms"] / row["tk2_ms"]
            row["tk2_GBps"] = row["kv_bytes"] / (row["tk2_ms"] * 1e-3) / 1e9
            rows.append(row)
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                              for k, v in row.items()}), flush=True)
            del q, k, v, stock
    receipt = {"script": "scripts/bench_sdpa_d512_module.py", "mlx_version": mx.__version__,
               "device": mx.device_info(), "kernel_version": d512.VERSION,
               "mlx2_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                             text=True).stdout.strip(), "rows": rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(receipt, indent=1, default=str))
    tmp.rename(out)
    return 0 if all(r["tk2_bitwise_stock"] and r["tk4_bitwise_stock"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

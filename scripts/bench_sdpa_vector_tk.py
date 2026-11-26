#!/usr/bin/env python3
"""Microbenchmark: MLX ``sdpa_vector_2pass_1`` key-unroll ``TK`` at head dim 512.

ml-explore/mlx #4941 (open, read only) reports that the two-pass vector SDPA
kernel spills registers for head dims above 256 with the default ``TK = 4``
key prefetch, and uses ``TK = 2`` there.  mlx 0.32.4 (the venv) still compiles
the D=512 variant with ``TK = 4`` (``sdpa_vector.h:180``).  Gemma 4's
full-attention layers have ``global_head_dim`` 512 (31B: 32 q / 4 kv heads,
26B-A4B: 16 q / 2 kv heads, K == V), so their decode hits it.

This script rebuilds pass 1 of that kernel with ``mx.fast.metal_kernel`` at
TK 1, 2 and 4 (the body is the 0.32.4 kernel, unmasked, no sinks, one query
row; MIT, Copyright Apple Inc., see docs/PROVENANCE.md) and times it on the
same buffers.  The key visit order is the same for every TK, so the outputs
must be bitwise identical; the script checks that, and checks the combined
result against ``mx.fast.scaled_dot_product_attention``.  The stock fused call
(pass 1 + pass 2) is timed as an anchor.

Writes one JSON receipt.  GPU work: run it under the GPU lock.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import mlx.core as mx

SOURCE = r"""
    constexpr int BD = 32;
    constexpr int qk_per_thread = D / BD;
    constexpr int v_per_thread = D / BD;
    typedef float U;

    const int N = params[0];
    const int blocks = params[1];
    const float scale = as_type<float>(params[2]);

    const int kv_head_idx = threadgroup_position_in_grid.x;
    const int block_idx = threadgroup_position_in_grid.z;
    const int gqa_factor = threads_per_threadgroup.y;
    const int q_head_idx = gqa_factor * kv_head_idx + thread_position_in_threadgroup.y;
    const int simd_lid = thread_position_in_threadgroup.x;

    thread U q[qk_per_thread];
    thread U o[v_per_thread] = {0};

    const device T* qp = queries + q_head_idx * D + simd_lid * qk_per_thread;
    const size_t k_head_stride = size_t(N) * D;
    const device T* kp0 = keys + kv_head_idx * k_head_stride + block_idx * D + simd_lid * qk_per_thread;
    const device T* vp0 = values + kv_head_idx * k_head_stride + block_idx * D + simd_lid * v_per_thread;

    for (int i = 0; i < qk_per_thread; i++) {
        q[i] = static_cast<U>(scale) * qp[i];
    }
    U max_score = Limits<U>::finite_min;
    U sum_exp_score = 0;
    const int stride_tok = blocks * D;

    int i = block_idx;
    for (; i + (TK - 1) * blocks < N; i += blocks * TK) {
        T k_reg[TK][qk_per_thread];
        T v_reg[TK][v_per_thread];
        for (int t = 0; t < TK; t++) {
            const device T* kp = kp0 + t * stride_tok;
            for (int j = 0; j < qk_per_thread; j++) { k_reg[t][j] = kp[j]; }
        }
        for (int t = 0; t < TK; t++) {
            const device T* vp = vp0 + t * stride_tok;
            for (int j = 0; j < v_per_thread; j++) { v_reg[t][j] = vp[j]; }
        }
        for (int t = 0; t < TK; t++) {
            U score = 0;
            for (int j = 0; j < qk_per_thread; j++) {
                score += q[j] * static_cast<U>(k_reg[t][j]);
            }
            score = simd_sum(score);
            U new_max = max(max_score, score);
            U factor = fast::exp(max_score - new_max);
            U exp_score = fast::exp(score - new_max);
            max_score = new_max;
            sum_exp_score = sum_exp_score * factor + exp_score;
            for (int j = 0; j < v_per_thread; j++) {
                o[j] = o[j] * factor + exp_score * static_cast<U>(v_reg[t][j]);
            }
        }
        kp0 += TK * stride_tok;
        vp0 += TK * stride_tok;
    }
    for (; i < N; i += blocks) {
        U score = 0;
        for (int j = 0; j < qk_per_thread; j++) { score += q[j] * kp0[j]; }
        score = simd_sum(score);
        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;
        for (int j = 0; j < v_per_thread; j++) {
            o[j] = o[j] * factor + exp_score * vp0[j];
        }
        kp0 += stride_tok;
        vp0 += stride_tok;
    }
    const int o_offset = q_head_idx;
    if (simd_lid == 0) {
        sums[o_offset * blocks + block_idx] = sum_exp_score;
        maxs[o_offset * blocks + block_idx] = max_score;
    }
    device T* op = out + (o_offset * blocks + block_idx) * D + simd_lid * v_per_thread;
    for (int i = 0; i < v_per_thread; i++) { op[i] = static_cast<T>(o[i]); }
"""

KERNEL = mx.fast.metal_kernel(
    name="bench_sdpa_vector_2pass_1_tk",
    input_names=["queries", "keys", "values", "params"],
    output_names=["out", "sums", "maxs"],
    source=SOURCE,
)


def pass1(q, k, v, *, tk: int, blocks: int):
    hq, d = q.shape[1], q.shape[-1]
    hkv, n = k.shape[1], k.shape[2]
    gqa = hq // hkv
    scale_bits = mx.array([d ** -0.5], dtype=mx.float32).view(mx.int32)
    params = mx.concatenate([mx.array([n, blocks], dtype=mx.int32), scale_bits])
    return KERNEL(
        inputs=[q, k, v, params],
        template=[("T", q.dtype), ("D", d), ("TK", tk)],
        grid=(32 * hkv, gqa, blocks),
        threadgroup=(32, gqa, 1),
        output_shapes=[(hq, blocks, d), (hq, blocks), (hq, blocks)],
        output_dtypes=[q.dtype, mx.float32, mx.float32],
    )


def combine(out, sums, maxs):
    m = mx.max(maxs, axis=1, keepdims=True)
    f = mx.exp(maxs - m)
    den = mx.sum(sums * f, axis=1)
    num = mx.sum(out.astype(mx.float32) * f[..., None], axis=1)
    return (num / den[:, None])


def timed(fn, iters: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    return samples


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--contexts", default="4096,16384,32768,61440,131072")
    ap.add_argument("--shapes", default="31b:32:4,26b:16:2",
                    help="name:q_heads:kv_heads, all at D=512")
    ap.add_argument("--blocks", default="32,64,128,256")
    ap.add_argument("--tks", default="1,2,4")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    out_path = Path(args.out)
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite {out_path}")

    dtype = getattr(mx, args.dtype)
    d = 512
    tks = [int(x) for x in args.tks.split(",")]
    rows = []
    mx.random.seed(0)
    for shape in args.shapes.split(","):
        name, hq, hkv = shape.split(":")
        hq, hkv = int(hq), int(hkv)
        for n in [int(x) for x in args.contexts.split(",")]:
            q = mx.random.normal((1, hq, 1, d)).astype(dtype)
            k = mx.random.normal((1, hkv, n, d)).astype(dtype)
            v = k  # Gemma 4 global layers: attention_k_eq_v
            mx.eval(q, k)
            ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=d ** -0.5)
            mx.eval(ref)
            ref32 = ref.reshape(hq, d).astype(mx.float32)
            stock = timed(lambda: mx.fast.scaled_dot_product_attention(
                q, k, v, scale=d ** -0.5), args.iters, args.warmup)
            row = {"shape": name, "q_heads": hq, "kv_heads": hkv, "head_dim": d,
                   "context": n, "kv_bytes": int(k.nbytes),
                   "stock_fused_ms_median": statistics.median(stock),
                   "cells": []}
            for blocks in [int(x) for x in args.blocks.split(",")]:
                outs = {}
                for tk in tks:
                    res = pass1(q, k, v, tk=tk, blocks=blocks)
                    mx.eval(res)
                    outs[tk] = res
                    s = timed(lambda: pass1(q, k, v, tk=tk, blocks=blocks),
                              args.iters, args.warmup)
                    comb = combine(*res)
                    err = float(mx.max(mx.abs(comb - ref32)).item())
                    row["cells"].append({
                        "blocks": blocks, "tk": tk,
                        "pass1_ms_median": statistics.median(s),
                        "pass1_ms_p10": sorted(s)[len(s) // 10],
                        "effective_GBps": k.nbytes / (statistics.median(s) * 1e-3) / 1e9,
                        "max_abs_err_vs_stock_fused": err,
                    })
                base = outs[tks[-1]]
                for tk in tks[:-1]:
                    same = all(bool(mx.array_equal(a, b).item())
                               for a, b in zip(outs[tk], base))
                    row["cells"].append({"blocks": blocks, "bitwise_equal":
                                         f"tk{tk}==tk{tks[-1]}", "equal": same})
            rows.append(row)
            best = {}
            for c in row["cells"]:
                if "tk" in c:
                    best.setdefault(c["tk"], min(
                        (x["pass1_ms_median"] for x in row["cells"]
                         if x.get("tk") == c["tk"])))
            print(name, n, "stock", round(row["stock_fused_ms_median"], 3),
                  {f"tk{t}": round(v, 3) for t, v in best.items()}, flush=True)
            q = k = v = ref = None  # release (captured by the timed lambdas above)
    receipt = {
        "script": "scripts/bench_sdpa_vector_tk.py",
        "mlx_version": mx.__version__,
        "device": mx.device_info() if hasattr(mx, "device_info") else None,
        "machine": platform.machine(),
        "mlx2_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                      text=True).stdout.strip(),
        "dtype": args.dtype, "iters": args.iters, "warmup": args.warmup,
        "rows": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(receipt, indent=1, default=str))
    tmp.rename(out_path)
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

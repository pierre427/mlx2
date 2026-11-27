#!/usr/bin/env python3
"""Kernel-level checks for the MLX upgrade past 0.32.5 (#4941, #4802).

1. ``d512``: ``models.sdpa_vector_d512.attention`` (mlx2's TK-2 copy of
   ``sdpa_vector_2pass``) against ``mx.fast.scaled_dot_product_attention`` on
   the installed MLX, bitwise, for every one-row head-dim-512 geometry stock
   MLX sends to the two-pass vector kernel (GQA 8, no array mask), contiguous
   and strided (a preallocated cache sliced to its offset), bf16 and fp16.
   On a build with ml-explore/mlx #4941 stock already runs TK 2, so the two
   must stay bitwise equal and the copy is redundant.  Also times both (median
   of a few calls; sanity, not publishable statistics).
2. ``int8_qmm``: what ``MLX_ENABLE_INT8_QMM`` (#4802) does to an affine
   ``quantized_matmul`` by row count: whether the output changes (the route
   engaged), its error against an fp32 product of the dequantized weights
   beside the flag-off error, and the time of both.

Usage: mlx_upgrade_kernel_probe.py --out probe.json --i-own-the-gpu
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


def _time(fn, reps=7):
    import mlx.core as mx

    mx.eval(fn())
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        mx.eval(fn())
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def d512_cases():
    import mlx.core as mx

    from mlx2.runtime.models import sdpa_vector_d512 as d512

    rows = []
    for hq, hkv in ((32, 4), (16, 2)):
        for n in (1024, 4096, 16384, 65536):
            for dtype in (mx.bfloat16, mx.float16):
                for strided in (False, True):
                    mx.random.seed(n + hq)
                    cap = n + 256 if strided else n
                    q = mx.random.normal((1, hq, 1, 512)).astype(dtype)
                    k = mx.random.normal((1, hkv, cap, 512)).astype(dtype)
                    v = mx.random.normal((1, hkv, cap, 512)).astype(dtype)
                    if strided:
                        k, v = k[:, :, :n], v[:, :, :n]
                    mx.eval(q, k, v)
                    scale = 512 ** -0.5
                    if not d512.supported(q, k, v):
                        rows.append({"hq": hq, "hkv": hkv, "n": n, "dtype": str(dtype),
                                     "strided": strided, "supported": False})
                        continue
                    ours = d512.attention(q, k, v, scale=scale)
                    stock = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
                    mx.eval(ours, stock)
                    bitwise = bool(mx.array_equal(ours.view(mx.uint16), stock.view(mx.uint16)).item())
                    diff = mx.abs(ours.astype(mx.float32) - stock.astype(mx.float32)).max().item()
                    rows.append({
                        "hq": hq, "hkv": hkv, "n": n, "dtype": str(dtype), "strided": strided,
                        "supported": True, "bitwise_equal": bitwise, "max_abs_diff": diff,
                        "ours_ms": 1e3 * _time(lambda: d512.attention(q, k, v, scale=scale)),
                        "stock_ms": 1e3 * _time(
                            lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)),
                    })
                    print("d512", rows[-1], flush=True)
    return rows


def int8_qmm_cases():
    import mlx.core as mx

    rows = []
    has_config = hasattr(mx, "config")
    for bits in (4, 8):
        for (k_dim, n_dim) in ((2560, 9216), (9216, 2560)):
            mx.random.seed(bits * 7 + k_dim)
            w = (0.02 * mx.random.normal((n_dim, k_dim))).astype(mx.bfloat16)
            wq, sc, bi = mx.quantize(w, group_size=64, bits=bits)
            w_hat = mx.dequantize(wq, sc, bi, group_size=64, bits=bits, dtype=mx.float32)
            for m in (8, 16, 31, 32, 64, 128, 512, 2048):
                x = mx.random.normal((m, k_dim)).astype(mx.bfloat16)
                ref = x.astype(mx.float32) @ w_hat.T
                out = {}
                timing = {}
                for flag in (0, 1):
                    if has_config:
                        mx.config.update("MLX_ENABLE_INT8_QMM", flag)

                    def run():
                        return mx.quantized_matmul(x, wq, sc, bi, transpose=True,
                                                   group_size=64, bits=bits)

                    out[flag] = run()
                    mx.eval(out[flag])
                    timing[flag] = 1e3 * _time(run)
                if has_config:
                    mx.config.update("MLX_ENABLE_INT8_QMM", 0)
                rms_ref = mx.sqrt(mx.mean(ref * ref)).item()

                def rel(y):
                    e = y.astype(mx.float32) - ref
                    return mx.sqrt(mx.mean(e * e)).item() / rms_ref

                row = {"bits": bits, "m": m, "k": k_dim, "n": n_dim,
                       "engaged": not bool(mx.array_equal(out[0], out[1]).item()),
                       "rel_rms_err_flag0": rel(out[0]), "rel_rms_err_flag1": rel(out[1]),
                       "ms_flag0": timing[0], "ms_flag1": timing[1]}
                rows.append(row)
                print("int8_qmm", row, flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    ap.add_argument("--skip", nargs="*", default=[])
    args = ap.parse_args()
    if not args.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")
    import mlx.core as mx

    from mlx2.runtime import mlx_upstream

    mx.set_default_device(mx.gpu)
    receipt = {"mlx": mx.__version__, "device": mx.device_info(),
               "upstream": mlx_upstream.status()}
    if "d512" not in args.skip:
        receipt["d512"] = d512_cases()
    if "int8_qmm" not in args.skip:
        receipt["int8_qmm"] = int8_qmm_cases()
    Path(args.out).write_text(json.dumps(receipt, indent=1, default=str))
    d512_rows = [r for r in receipt.get("d512", []) if r.get("supported")]
    print(json.dumps({"d512_cases": len(d512_rows),
                      "d512_all_bitwise": all(r["bitwise_equal"] for r in d512_rows)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Metal gate for Ripple's metal_glue kernels and window-blocked prefill.

Random tensors at Ripple's geometry; no model weights. Exactness of each
kernel against the eager law it replaces (QK path compared with the default
metal_rmsnorm norm + eager rotation, which shares its reduction), per-token
timing of one decode token's glue (80 layers: Q and K norm+RoPE on 64 local
layers, norm-only on 16 global, 80 blends, 80 SwiGLUs) eager vs kernels,
and sliding-layer prefill attention per chunk (512 / 2048 rows) stock
masked vs window-blocked. Requires the caller's GPU ownership.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx2.runtime.models import ricursive, ricursive_glue, ricursive_rmsnorm
from mlx2.runtime.models.base import create_causal_mask

EPS = 1e-6


def compare(a, b):
    a, b = np.asarray(a.astype(mx.float32)), np.asarray(b.astype(mx.float32))
    diff = np.abs(a - b)
    return {"bit_equal": bool(np.array_equal(a, b)), "mismatch_fraction": float(np.mean(a != b)),
            "max_rel": float(diff.max() / max(np.abs(b).max(), 1e-30))}


def timed(fn, repeats=9):
    fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return sorted(samples)[repeats // 2] * 1e3


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if mx.default_device() != mx.gpu:
        raise RuntimeError("Metal gate")
    rng = np.random.default_rng(1105)

    def a(shape, scale=1.0, dtype=mx.bfloat16):
        return mx.array((rng.normal(size=shape) * scale).astype(np.float32)).astype(dtype)

    exact = {}
    for rows in (1, 8, 512):
        cos, sin = ricursive.rotary_tables(rows, 3000, 128, 10000.0, mx.bfloat16)
        for name, heads in (("q", 80), ("k", 8)):
            raw, w = a((1, rows, heads, 128), 4.0), a((128,), 0.3) + 1
            mx.eval(raw, w, cos, sin)
            normed = ricursive_rmsnorm.rms_norm(raw, w, EPS).transpose(0, 2, 1, 3)
            exact[f"{name}_rope_rows{rows}"] = compare(
                ricursive_glue.qk_norm_rope(raw, w, EPS, cos, sin), ricursive.apply_rotary(normed, cos, sin))
            exact[f"{name}_nope_rows{rows}"] = compare(ricursive_glue.qk_norm_rope(raw, w, EPS), normed)
        g, u = a((1, rows, 40960), 9.0), a((1, rows, 40960), 9.0)
        exact[f"swiglu_rows{rows}"] = compare(ricursive_glue.swiglu(g, u, 20.0), ricursive._swiglu(g, u, 20.0))
        seed, x = a((1, rows, 5120), 30.0, mx.float32), a((1, rows, 5120))
        value = mx.sigmoid(mx.array(1.1)).astype(mx.bfloat16).astype(mx.float32)
        exact[f"blend_rows{rows}"] = compare(ricursive_glue.blend(seed, x, value), ricursive._blend(seed, x, value))
    # Batched per-lane tables ([B, 1, L, half]).
    raw, w = a((3, 4, 8, 128), 4.0), a((128,), 0.3) + 1
    cos, sin = ricursive.rotary_tables(4, mx.array([10, 500, 2000]), 128, 10000.0, mx.bfloat16)
    normed = ricursive_rmsnorm.rms_norm(raw, w, EPS).transpose(0, 2, 1, 3)
    exact["k_rope_batched_lanes"] = compare(ricursive_glue.qk_norm_rope(raw, w, EPS, cos, sin),
                                            ricursive.apply_rotary(normed, cos, sin))

    # One decode token's glue.
    cos1, sin1 = ricursive.rotary_tables(1, 3000, 128, 10000.0, mx.bfloat16)
    qraw, kraw, wq = a((1, 1, 80, 128)), a((1, 1, 8, 128)), a((128,)) + 1
    g1, u1 = a((1, 1, 40960), 9.0), a((1, 1, 40960), 9.0)
    s1, x1 = a((1, 1, 5120), 30.0, mx.float32), a((1, 1, 5120))
    blend_value = mx.array(0.7, dtype=mx.float32)
    mx.eval(cos1, sin1, qraw, kraw, wq, g1, u1, s1, x1)

    def token(kernels):
        outs = []
        for layer in range(80):
            local = layer % 5 != 4
            if kernels:
                tables = (cos1, sin1) if local else (None, None)
                outs += [ricursive_glue.qk_norm_rope(qraw, wq, EPS, *tables),
                         ricursive_glue.qk_norm_rope(kraw, wq, EPS, *tables),
                         ricursive_glue.blend(s1, x1, blend_value), ricursive_glue.swiglu(g1, u1, 20.0)]
            else:
                q = ricursive_rmsnorm.rms_norm(qraw, wq, EPS).transpose(0, 2, 1, 3)
                k = ricursive_rmsnorm.rms_norm(kraw, wq, EPS).transpose(0, 2, 1, 3)
                if local:
                    q, k = ricursive.apply_rotary(q, cos1, sin1), ricursive.apply_rotary(k, cos1, sin1)
                outs += [q, k, ricursive._blend(s1, x1, blend_value), ricursive._swiglu(g1, u1, 20.0)]
        mx.eval(outs)

    timing = {"eager_with_metal_rmsnorm": timed(lambda: token(False)), "metal_glue": timed(lambda: token(True))}

    prefill = {}
    for rows in (512, 2048):
        n = rows + 1023
        q, k, v = a((1, 80, rows, 128)), a((1, 8, n, 128)), a((1, 8, n, 128))
        mask = create_causal_mask(rows, offset=n - rows, window_size=1024)
        mx.eval(q, k, v, mask)
        stock = lambda: mx.eval(mx.fast.scaled_dot_product_attention(q, k, v, scale=128 ** -0.5, mask=mask))
        blocked = lambda: mx.eval(ricursive.window_blocked_attention(q, k, v, scale=128 ** -0.5, mask=mask,
                                                                     window=1024))
        prefill[f"sliding_rows{rows}"] = {
            "stock_ms": timed(stock), "blocked_ms": timed(blocked),
            "blocked_vs_stock": compare(ricursive.window_blocked_attention(q, k, v, scale=128 ** -0.5, mask=mask,
                                                                           window=1024),
                                        mx.fast.scaled_dot_product_attention(q, k, v, scale=128 ** -0.5, mask=mask))}
    receipt = {"schema": "mlx2.ripple-glue-gate.v1", "mlx": mx.__version__,
               "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "checkpoint": "none (random tensors)", "exactness": exact, "token_glue_ms": timing,
               "sliding_prefill_attention": prefill}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"token_glue_ms": timing,
                      "worst_mismatch": max(v["mismatch_fraction"] for v in exact.values()),
                      "worst_max_rel": max(v["max_rel"] for v in exact.values()),
                      "prefill": {k: (round(v["stock_ms"], 2), round(v["blocked_ms"], 2)) for k, v in prefill.items()}}))


if __name__ == "__main__":
    main()

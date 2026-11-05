#!/usr/bin/env python3
"""Metal gate for the one-launch Ripple RMSNorm kernel (no model weights).

At Ripple's shapes (block norms 5120 wide, per-head Q/K norms 80x128 and
8x128, decode and 512-row chunks), compare the kernel with the eager source
law (``ricursive._rms``), plain and with the post-norm residual: mismatch
fraction, max |diff| and max ULP-scale gap. Time one decode token's worth of
norms (4 block + Q/K per layer x 80, plus the final norm) eager vs kernel,
issued as one evaluation. Requires the caller's GPU ownership.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx2.runtime.models import ricursive, ricursive_rmsnorm

EPS = 1e-6


def compare(a, b):
    a, b = np.asarray(a.astype(mx.float32)), np.asarray(b.astype(mx.float32))
    diff = np.abs(a - b)
    # Relative to the case's largest magnitude: a per-element ratio explodes
    # where the expected value is near zero (cancelling residuals).
    return {"bit_equal": bool(np.array_equal(a, b)), "mismatch_fraction": float(np.mean(a != b)),
            "max_abs": float(diff.max()), "max_rel": float(diff.max() / max(np.abs(b).max(), 1e-30))}


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
    rng = np.random.default_rng(2026)

    def a(shape, scale=1.0):
        return mx.array((rng.normal(size=shape) * scale).astype(np.float32)).astype(mx.bfloat16)

    exact = {}
    for rows in (1, 8, 512):
        for name, shape in (("block", (1, rows, 5120)), ("q_heads", (1, rows, 80, 128)), ("k_heads", (1, rows, 8, 128))):
            x, w = a(shape, 3.0), a(shape[-1:], 0.5) + 1
            res = a(shape, 10.0)
            mx.eval(x, w, res)
            exact[f"{name}_rows{rows}"] = compare(ricursive_rmsnorm.rms_norm(x, w, EPS), ricursive._rms(x, w, EPS))
            if name == "block":
                exact[f"{name}_residual_rows{rows}"] = compare(
                    ricursive_rmsnorm.rms_norm(x, w, EPS, residual=res), res + ricursive._rms(x, w, EPS))
    # One decode token: per layer 4 block norms (2 with residual) + Q and K head norms; 80 layers + final.
    xs = [a((1, 1, 5120)) for _ in range(4)]
    ws = a((5120,)) + 1
    qh, kh, wh = a((1, 1, 80, 128)), a((1, 1, 8, 128)), a((128,)) + 1
    mx.eval(xs, ws, qh, kh, wh)

    def token(norm, residual):
        outs = []
        for _ in range(80):
            outs += [norm(xs[0], ws), residual(xs[1], xs[2], ws), norm(xs[3], ws), residual(xs[2], xs[0], ws),
                     norm(qh, wh), norm(kh, wh)]
        outs.append(norm(xs[0], ws))
        mx.eval(outs)

    eager = lambda: token(lambda x, w: ricursive._rms(x, w, EPS), lambda r, x, w: r + ricursive._rms(x, w, EPS))
    kernel = lambda: token(lambda x, w: ricursive_rmsnorm.rms_norm(x, w, EPS),
                           lambda r, x, w: ricursive_rmsnorm.rms_norm(x, w, EPS, residual=r))
    fused = lambda: token(lambda x, w: ricursive.fused_norm(x, w, EPS),
                          lambda r, x, w: ricursive.fused_residual_norm(r, x, w, EPS))
    receipt = {"schema": "mlx2.ripple-rmsnorm-gate.v1", "mlx": mx.__version__,
               "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "checkpoint": "none (random tensors)", "exactness": exact,
               "token_norms_ms": {"eager": timed(eager), "compiled_regions": timed(fused), "metal_kernel": timed(kernel)},
               "norms_per_token": 80 * 6 + 1}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"token_norms_ms": receipt["token_norms_ms"],
                      "worst_mismatch": max(v["mismatch_fraction"] for v in exact.values()),
                      "worst_max_rel": max(v["max_rel"] for v in exact.values())}))


if __name__ == "__main__":
    main()

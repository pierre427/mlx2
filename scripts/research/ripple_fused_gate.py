#!/usr/bin/env python3
"""Metal gate for Ripple's fused elementwise regions, plus a qmv bandwidth probe.

1. Each fused region in mlx2.runtime.models.ricursive is compared bit for bit
   with its eager expression on the default (Metal) device at Ripple's real
   widths, for one decode row and one 512-row prefill chunk, in BF16.
2. B1 affine quantized matmul (group 64) at Ripple's MLP and attention shapes
   for 4/5/6/8 bits: median time and effective weight bandwidth, to see
   whether the oQ4e artifact's 5/6-bit modules decode slower per byte.

Random tensors only; no model weights are loaded. Requires the caller's GPU
ownership (CPG lease and lock). Writes one JSON receipt.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx2.runtime.models import ricursive as r


def bits_equal(a, b):
    a, b = np.asarray(a.astype(mx.float32)), np.asarray(b.astype(mx.float32))
    return {"equal": bool(np.array_equal(a, b)), "mismatch_fraction": float(np.mean(a != b)),
            "max_abs": float(np.max(np.abs(a - b))) if a.size else 0.0}


def regions(rows, rng):
    dt, eps, hidden, heads, kv, d, ffn = mx.bfloat16, 1e-6, 5120, 80, 8, 128, 40960

    def a(shape, scale=1.0, dtype=dt):
        return mx.array((rng.normal(size=shape) * scale).astype(np.float32)).astype(dtype)

    out = {}
    x, w, y = a((1, rows, hidden)), a((hidden,)), a((1, rows, hidden))
    out["norm"] = bits_equal(r.fused_norm(x, w, eps), r._rms(x, w, eps))
    out["residual_norm"] = bits_equal(r.fused_residual_norm(x, y, w, eps), x + r._rms(y, w, eps))
    cos, sin = r.rotary_tables(rows, 1000, d, 10000.0, dt)
    for name, n in (("q", heads), ("k", kv)):
        raw, wq = a((1, rows, n, d)), a((d,))
        out[f"{name}_norm_rope"] = bits_equal(
            r.fused_qk(raw, wq, eps, cos, sin), r.apply_rotary(r._rms(raw, wq, eps).transpose(0, 2, 1, 3), cos, sin))
        out[f"{name}_norm_nope"] = bits_equal(r.fused_qk(raw, wq, eps), r._rms(raw, wq, eps).transpose(0, 2, 1, 3))
    seed32 = a((1, rows, hidden), 30.0, mx.float32)
    blend = mx.sigmoid(mx.array(1.1)).astype(dt).astype(mx.float32)
    out["value_blend"] = bits_equal(r.fused_blend(seed32, x, blend), r._blend(seed32, x, blend))
    attn, g = a((1, heads, rows, d)), a((1, rows, heads * d), 3.0)
    out["gate"] = bits_equal(r.fused_gate(attn, g, True), r._gate(attn, g, True))
    gate, up = a((1, rows, ffn), 8.0), a((1, rows, ffn), 8.0)
    out["swiglu"] = bits_equal(r.fused_swiglu(gate, up, 20.0), r._swiglu(gate, up, 20.0))
    return out


def qmv_probe(rng, repeats=50):
    shapes = {"mlp_gate_up": (40960, 5120), "mlp_down": (5120, 40960), "q_or_g": (10240, 5120), "o": (5120, 10240)}
    rows = []
    for bits in (4, 5, 6, 8):
        for name, (n, k) in shapes.items():
            weight = mx.array(rng.normal(size=(n, k)).astype(np.float32)).astype(mx.bfloat16)
            wq, scales, biases = mx.quantize(weight, group_size=64, bits=bits)
            x = mx.array(rng.normal(size=(1, k)).astype(np.float32)).astype(mx.bfloat16)
            mx.eval(wq, scales, biases, x)
            del weight
            for _ in range(5):
                mx.eval(mx.quantized_matmul(x, wq, scales, biases, transpose=True, group_size=64, bits=bits))
            times = []
            for _ in range(repeats):
                start = time.perf_counter()
                outs = [mx.quantized_matmul(x, wq, scales, biases, transpose=True, group_size=64, bits=bits)
                        for _ in range(8)]
                mx.eval(outs)
                times.append((time.perf_counter() - start) / 8)
            median = sorted(times)[len(times) // 2]
            nbytes = wq.nbytes + scales.nbytes + biases.nbytes
            rows.append({"bits": bits, "shape": name, "n": n, "k": k, "bytes": nbytes,
                         "median_us": median * 1e6, "gb_per_s": nbytes / median / 1e9})
            del wq, scales, biases
            mx.clear_cache()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise RuntimeError("the gate is meaningful only on Metal")
    rng = np.random.default_rng(20261104)
    receipt = {"schema": "mlx2.ripple-fused-gate.v1",
               "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "mlx": mx.__version__, "device": mx.device_info(), "checkpoint": "none (random tensors)",
               "regions": {str(rows): regions(rows, rng) for rows in (1, 512)}}
    receipt["regions_all_equal"] = all(v["equal"] for per in receipt["regions"].values() for v in per.values())
    receipt["qmv"] = qmv_probe(rng)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"regions_all_equal": receipt["regions_all_equal"],
                      "failed": [f"{rows}:{k}" for rows, per in receipt["regions"].items()
                                 for k, v in per.items() if not v["equal"]],
                      "qmv": [(q["bits"], q["shape"], round(q["gb_per_s"])) for q in receipt["qmv"]]}))


if __name__ == "__main__":
    main()

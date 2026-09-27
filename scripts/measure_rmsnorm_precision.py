#!/usr/bin/env python3
"""How much precision does GroupRMSNorm actually need, and what does int8 cost?

The hyper-connection profile put ~13.7% of a Flash-Next prefill in
``GroupRMSNorm`` over the 10240-wide residual stream, and the roofline says it
is running at ~528 GB/s -- i.e. purely memory-bandwidth bound, moving 6.71 GB
per call at T=16384 because the eager path materialises an fp32 copy of a
335 MB bf16 tensor at every step.

The obvious question is whether the fp32 is necessary, or whether the traffic
could be cut by dropping precision: stay in bf16, use fp16, or quantize to
int8.  This measures it instead of arguing about it.

The distinction that matters is **accumulator dtype versus storage dtype**.
``GroupRMSNorm(hc_hidden, hidden_size, eps)`` normalises over groups of
``hidden_size`` = 2560 elements, so the mean-of-squares is a 2560-term
reduction.  Production already stores and returns bf16; the fp32 exists only
inside the reduction.  So there are two independent axes:

  * what the reduction accumulates in -- this is a correctness question;
  * what is materialised in HBM between ops -- this is a traffic question, and
    a fused kernel can keep bf16 in memory while accumulating in fp32 registers.

Arms, all compared against the production fp32-accumulate reference:
  bf16 accum      every step in bf16
  fp16 accum      every step in fp16
  int8 per-group  symmetric int8 quantize/dequantize with a per-group scale
  int8 per-tensor symmetric int8 with one scale for the whole tensor
  fused ideal     bf16 in, fp32 accumulate, bf16 out -- what a fused kernel does

Also reported: the relative error of the sum-of-squares reduction on its own,
which is where the precision is actually lost, and an fp16 overflow check on
both the elementwise square and the 2560-term accumulator.

Runs on CPU by default; no GPU is needed because the answer is a property of the
dtypes, not of the device.  --i-own-the-gpu moves it to Metal.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SCHEMA = "mlx2.rmsnorm-precision.v1"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",
                   default="~/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP",
                   help="artifact to read config.json from (metadata only)")
    p.add_argument("--tokens", type=int, default=1024)
    p.add_argument("--scale", type=float, default=1.7,
                   help="activation std; sets the fp16 overflow margin")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", required=True)
    p.add_argument("--i-own-the-gpu", action="store_true")
    a = p.parse_args()

    import mlx.core as mx

    if not a.i_own_the_gpu:
        mx.set_default_device(mx.cpu)

    config = json.loads((Path(a.model) / "config.json").read_text())
    tc = config.get("text_config", config)
    W = tc["hc_count"] * tc["hidden_size"]
    G = tc["hidden_size"]
    EPS = tc["rms_norm_eps"]
    T = a.tokens

    mx.random.seed(a.seed)
    x = (mx.random.normal((1, T, W)) * a.scale).astype(mx.bfloat16)
    weight = (1.0 + 0.05 * mx.random.normal((W,))).astype(mx.bfloat16)
    xf32 = x.astype(mx.float32)
    mx.eval(x, weight, xf32)

    def norm(xc, out_dtype):
        xr = xc.reshape(*xc.shape[:-1], -1, G)
        ss = mx.mean(xr * xr, axis=-1, keepdims=True)
        out = (xr * mx.rsqrt(ss + EPS)).reshape(*xc.shape)
        return (out * weight.astype(mx.float32)).astype(out_dtype)

    def q8(xc, per_group):
        """Symmetric int8 quantize then dequantize, in fp32."""
        v = xc.astype(mx.float32)
        if per_group:
            v = v.reshape(*v.shape[:-1], -1, G)
            scale = mx.max(mx.abs(v), axis=-1, keepdims=True) / 127.0
        else:
            scale = mx.max(mx.abs(v)) / 127.0
        q = mx.round(v / scale).astype(mx.int8)
        return (q.astype(mx.float32) * scale).reshape(*xc.shape)

    ref = norm(xf32, mx.bfloat16).astype(mx.float32)
    mx.eval(ref)

    arms = {
        "fp32_accum_production": norm(xf32, mx.bfloat16),
        "bf16_accum": norm(x, mx.bfloat16),
        "fp16_accum": norm(x.astype(mx.float16), mx.float16),
        "int8_per_group": norm(q8(x, True), mx.bfloat16),
        "int8_per_tensor": norm(q8(x, False), mx.bfloat16),
        "fused_bf16io_fp32accum": norm(xf32, mx.bfloat16),
    }

    results = {}
    for name, val in arms.items():
        v = val.astype(mx.float32)
        mx.eval(v)
        d = mx.abs(v - ref)
        rel = d / mx.maximum(mx.abs(ref), 1e-6)
        mx.eval(d, rel)
        results[name] = {
            "max_abs_delta": float(mx.max(d).item()),
            "rms_rel_delta": float(mx.sqrt(mx.mean(rel * rel)).item()),
            "bf16_ulps_at_unit_magnitude": float(mx.max(d).item()) / 2**-8,
            "pct_elements_differing": 100.0 * float(mx.mean((d > 0).astype(mx.float32)).item()),
            "bit_exact": float(mx.max(d).item()) == 0.0,
        }

    # the reduction on its own -- this is where precision is lost
    xr32 = xf32.reshape(*xf32.shape[:-1], -1, G)
    ss32 = mx.mean(xr32 * xr32, axis=-1, keepdims=True).astype(mx.float32)
    mx.eval(ss32)
    reduction = {}
    for label, dt in (("bf16", mx.bfloat16), ("fp16", mx.float16), ("fp32", mx.float32)):
        xr = xf32.astype(dt).reshape(*xf32.shape[:-1], -1, G)
        ss = mx.mean(xr * xr, axis=-1, keepdims=True).astype(mx.float32)
        mx.eval(ss)
        e = mx.abs(ss - ss32) / ss32
        mx.eval(e)
        reduction[label] = {"max_rel_err": float(mx.max(e).item()),
                            "mean_rel_err": float(mx.mean(e).item())}

    sq = xf32 * xf32
    mx.eval(sq)
    overflow = {
        "max_elementwise_square": float(mx.max(sq).item()),
        "fp16_max": 65504.0,
        "elementwise_square_overflows_fp16": bool(float(mx.max(sq).item()) > 65504.0),
        "worst_case_2560_term_accumulator": float(mx.max(sq).item()) * G,
        "accumulator_overflows_fp16": bool(float(mx.max(sq).item()) * G > 65504.0),
        "note": ("mx.mean reduces over group_size terms; the running sum can "
                 "exceed fp16 range long before the mean does"),
    }

    report = {
        "schema": SCHEMA,
        "model_config_only": a.model,
        "device": str(mx.default_device()),
        "mlx_version": mx.__version__,
        "geometry": {"stream_width": W, "group_size": G, "eps": EPS, "T": T},
        "activation_scale": a.scale,
        "seed": a.seed,
        "reference": "fp32 accumulate, bf16 output (what production returns)",
        "arms": results,
        "sum_of_squares_reduction_error": reduction,
        "fp16_overflow": overflow,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2))

    print(f"group_size {G} (reduction over {G} terms), stream width {W}, T={T}\n")
    hdr = f"{'arm':28}{'max|d|':>11}{'rms rel':>11}{'ULPs':>8}{'% diff':>9}  exact"
    print(hdr)
    print("-" * len(hdr))
    for name, d in results.items():
        print(f"{name:28}{d['max_abs_delta']:>11.3e}{d['rms_rel_delta']:>11.3e}"
              f"{d['bf16_ulps_at_unit_magnitude']:>8.2f}{d['pct_elements_differing']:>8.2f}%"
              f"  {d['bit_exact']}")
    print("\nsum-of-squares reduction error vs fp32:")
    for label, d in reduction.items():
        print(f"  {label:5} max {d['max_rel_err']:.3e}   mean {d['mean_rel_err']:.3e}")
    print(f"\nfp16 overflow: max x^2 {overflow['max_elementwise_square']:.1f} "
          f"(fp16 max 65504); worst-case {G}-term accumulator "
          f"{overflow['worst_case_2560_term_accumulator']:.0f} -> "
          f"overflows={overflow['accumulator_overflows_fp16']}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

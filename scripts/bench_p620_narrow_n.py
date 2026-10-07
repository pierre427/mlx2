"""Synthetic microbench for the p620 intake (2026-10-07): narrow-output projections
and hyper-connection kernels at decode/verify row counts.  Loads no model.

Arms per (shape, bits, rows):
  stock  mx.quantized_matmul (MLX qmv / qmm)
  lane   mlx2 lane matmul called directly (split-K; bypasses the >= 8 row policy)
Plus, at Flash-Next's HC shape, the full fused HC decode launch pair, and the
Xing4.0 mHC ``mhc_pre`` kernel against its compiled fallback.

Each timing is the median over reps of (one eval of CALLS independent calls) / CALLS.
"""

import argparse
import json
import statistics
import time
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn

from mlx2.runtime.lane import matmul as lane

SHAPES = [  # (name, K, N)
    ("fn_hc_down", 10240, 320),
    ("fn_router", 2560, 512),
    ("q38_27b_ba", 5120, 48),
    ("xing_lowrank", 3584, 576),
    ("ref_square", 5120, 5120),
]


def timed(fn, calls, reps):
    mx.eval(fn())
    samples = []
    for _ in range(reps):
        tic = time.perf_counter()
        mx.eval([fn() for _ in range(calls)])
        samples.append((time.perf_counter() - tic) / calls * 1e6)
    return round(statistics.median(samples), 2), round(min(samples), 2)


def linear(k, n, bits):
    layer = nn.Linear(k, n, bias=False)
    layer.weight = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
    q = nn.QuantizedLinear.from_linear(layer, group_size=64, bits=bits)
    mx.eval(q.parameters())
    return q


def bench_projections(a, out):
    for name, k, n in SHAPES:
        for bits in (4, 8):
            q = linear(k, n, bits)
            try:
                lw = lane.prepare(q)
            except lane.LaneUnsupported as exc:
                lw, why = None, str(exc)
            for rows in a.rows:
                x = (mx.random.normal((rows, k)) * 0.5).astype(mx.bfloat16)
                rec = {"shape": name, "K": k, "N": n, "bits": bits, "rows": rows,
                       "weight_mb": round((n * k * bits / 8 + n * k / 64 * 4) / 2**20, 3)}
                rec["stock_us"] = timed(lambda: q(x), a.calls, a.reps)
                if lw is not None:
                    rec["split_k"] = lw.split_k
                    rec["lane_us"] = timed(lambda: lane.lane_matmul(x, lw), a.calls, a.reps)
                    ref = q(x).astype(mx.float32)
                    got = lane.lane_matmul(x, lw).astype(mx.float32)
                    rec["lane_max_abs_diff"] = float(mx.max(mx.abs(ref - got)))
                else:
                    rec["lane"] = why
                out.append(rec)
                print(json.dumps(rec), flush=True)
            mx.clear_cache()


def bench_flash_next_hc(a, out):
    from mlx2.runtime.models import qwen4_exp as Q
    from mlx2.runtime.models import qwen4_hc_decode as HCD

    HCD.set_hc_multi_row_mode("on")
    args = SimpleNamespace(hc_count=4, hidden_size=2560, hc_lowrank=320, rms_norm_eps=1e-6)
    module = Q.GatedResidual(args, use_combine=True)
    for path in ("input_mix_weight_down", "input_mix_weight_up", "block_inject_weight"):
        lin = getattr(module, path, None)
        if lin is not None:
            lin.weight = (mx.random.normal(lin.weight.shape) * 0.02).astype(mx.bfloat16)
    module.set_dtype(mx.bfloat16)
    nn.quantize(module, group_size=64, bits=4)
    module.eval()
    mx.eval(module.parameters())
    for rows in a.rows:
        flat = (mx.random.normal((rows, 4 * 2560)) * 0.5).astype(mx.bfloat16)
        rec = {"shape": "fn_hc_full_launch_pair", "rows": rows}
        try:
            rec["hc_launch_us"] = timed(lambda: HCD.hc_decode_launch(module, flat), a.calls, a.reps)
        except Exception as exc:  # admission refusals are data here
            rec["hc_launch"] = f"{type(exc).__name__}: {exc}"
        down = module.input_mix_weight_down
        rec["stock_down_only_us"] = timed(lambda: down(flat), a.calls, a.reps)
        out.append(rec)
        print(json.dumps(rec), flush=True)


def bench_xing_mhc(a, out):
    from mlx2.runtime.models import xing4_0 as X

    args = SimpleNamespace(hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6, rms_norm_eps=1e-6,
                           mhc_h_res_clamp_min=-30.0, mhc_h_res_clamp_max=30.0, hidden_size=3584)
    module = X.HyperConnection(args)
    module.hc_fn = mx.random.normal(module.hc_fn.shape) * 0.01
    module.hc_base = mx.random.normal(module.hc_base.shape) * 0.1
    mx.eval(module.parameters())
    saved = X._MHC_KERNEL
    try:
        for rows in a.rows:
            streams = (mx.random.normal((1, rows, 4, 3584)) * 0.5).astype(mx.bfloat16)
            rec = {"shape": "xing_mhc_pre", "rows": rows, "hc_fn_mb": round(24 * 4 * 3584 * 4 / 2**20, 3)}
            X._MHC_KERNEL = True
            rec["kernel_us"] = timed(lambda: module(streams), a.calls, a.reps)
            X._MHC_KERNEL = False
            rec["compiled_us"] = timed(lambda: module(streams), a.calls, a.reps)
            out.append(rec)
            print(json.dumps(rec), flush=True)
    finally:
        X._MHC_KERNEL = saved


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--rows", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    p.add_argument("--calls", type=int, default=48)
    p.add_argument("--reps", type=int, default=15)
    a = p.parse_args()
    assert mx.default_device() == mx.gpu and mx.metal.is_available()
    mx.set_cache_limit(2 << 30)
    mx.random.seed(0)
    out = []
    info = {"device": mx.metal.device_info(), "lane_backend": lane.backend(),
            "mlx": getattr(mx, "__version__", "?")}
    print(json.dumps(info, default=str), flush=True)
    for part in (bench_projections, bench_flash_next_hc, bench_xing_mhc):
        try:
            part(a, out)
        except Exception as exc:
            out.append({"part": part.__name__, "error": f"{type(exc).__name__}: {exc}"})
            print(json.dumps(out[-1]), flush=True)
    with open(a.out, "w") as fh:
        json.dump({"info": info, "results": out}, fh, indent=1, default=str)


if __name__ == "__main__":
    main()

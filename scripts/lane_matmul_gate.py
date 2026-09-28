#!/usr/bin/env python3
"""GPU gate for the lane matmul: row invariance, accuracy and cost, all formats.

For each weight format (affine 2/3/4/5/6/8-bit at group sizes 32/64/128, and
bf16/fp16 unquantized) and each projection shape:

* invariance: every row of a 1..128-row call is bitwise equal to the same row
  computed alone;
* accuracy: max |lane - fp32 reference| is within 2x of max |MLX - fp32 ref|
  (plus one bf16 ulp of the output scale), where the reference dequantizes
  with MLX and multiplies in fp32;
* cost: median ms at 1, 8 and 16 rows for lane and for MLX's own kernel.

Needs an M5 GPU.  Writes one JSON receipt and exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mlx2.runtime.lane import matmul as lane

SHAPES = {"small": [(2048, 1024), (1024, 4096)],
          "model": [(5120, 6144), (17408, 5120), (5120, 17408), (5120, 248320)]}
ROWS = (1, 2, 3, 4, 7, 8, 13, 16, 17, 24, 32, 48, 64, 128)


def timed(fn, reps=7, batch=20):
    """Median ms per call, ``batch`` calls per sync so launch/sync jitter amortizes."""
    for _ in range(2):
        mx.eval([fn() for _ in range(batch)])
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        mx.eval([fn() for _ in range(batch)])
        out.append((time.perf_counter() - t) * 1000 / batch)
    return round(statistics.median(out), 4)


def case(bits, gs, k, n, dtype, key):
    kw, kx = mx.random.split(key)
    w = (mx.random.normal((n, k), key=kw) * 0.02).astype(dtype)
    x = mx.random.normal((max(ROWS), k), key=kx).astype(dtype)
    if bits == 16:
        from mlx import nn
        module = nn.Linear(k, n, bias=False)
        module.weight = w
        stock = lambda rows: rows @ w.T
        ref_w = w.astype(mx.float32)
    else:
        wq, scales, biases = mx.quantize(w, group_size=gs, bits=bits)
        from mlx import nn
        module = nn.QuantizedLinear(k, n, bias=False, group_size=gs, bits=bits)
        module.weight, module.scales, module.biases = wq, scales, biases
        stock = lambda rows: mx.quantized_matmul(
            rows, wq, scales, biases, transpose=True, group_size=gs, bits=bits)
        ref_w = mx.dequantize(wq, scales, biases, group_size=gs, bits=bits).astype(mx.float32)
    lw = lane.prepare(module)
    rec = {"bits": bits, "group_size": gs, "k": k, "n": n, "dtype": str(dtype),
           "split_k": lw.split_k}
    alone = mx.concatenate([lane.lane_matmul(x[i:i + 1], lw) for i in range(max(ROWS))], axis=0)
    mx.eval(alone)
    bad = []
    for r in ROWS:
        together = lane.lane_matmul(x[:r], lw)
        if not bool(mx.array_equal(together, alone[:r]).item()):
            bad.append(r)
    rec["invariant_fail_rows"] = bad
    ref = x[:16].astype(mx.float32) @ ref_w.T
    lane_err = float(mx.max(mx.abs(alone[:16].astype(mx.float32) - ref)).item())
    stock_err = float(mx.max(mx.abs(stock(x[:16]).astype(mx.float32) - ref)).item())
    scale = float(mx.max(mx.abs(ref)).item())
    ulp = scale * 2.0 ** -7
    rec.update(lane_max_err=lane_err, stock_max_err=stock_err, ref_max_abs=scale,
               accurate=lane_err <= 2 * stock_err + ulp)
    rec["ms"] = {f"{kind}_{r}": timed(
        (lambda r=r: lane.lane_matmul(x[:r], lw)) if kind == "lane" else (lambda r=r: stock(x[:r])))
        for kind in ("lane", "stock") for r in (1, 8, 16)}
    rec["passed"] = not bad and rec["accurate"]
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--shapes", choices=tuple(SHAPES), default="small")
    p.add_argument("--bits", default="2,3,4,5,6,8,16")
    p.add_argument("--group-sizes", default="32,64,128")
    p.add_argument("--dtypes", default="bfloat16")
    args = p.parse_args()
    if not lane.available():
        raise SystemExit("lane matmul needs an M5 GPU as the default device")
    key = mx.random.key(0)
    records = []
    for dname in args.dtypes.split(","):
        dtype = getattr(mx, dname)
        for bits in (int(b) for b in args.bits.split(",")):
            for gs in ((64,) if bits == 16 else tuple(int(g) for g in args.group_sizes.split(","))):
                for k, n in SHAPES[args.shapes]:
                    key, sub = mx.random.split(key)
                    try:
                        rec = case(bits, gs, k, n, dtype, sub)
                    except Exception as exc:  # noqa: BLE001 - record and keep going
                        rec = {"bits": bits, "group_size": gs, "k": k, "n": n,
                               "dtype": dname, "passed": False, "error": repr(exc)[:400]}
                    records.append(rec)
                    print(json.dumps({key2: rec.get(key2) for key2 in (
                        "bits", "group_size", "k", "n", "dtype", "passed", "invariant_fail_rows",
                        "lane_max_err", "stock_max_err", "ms", "error")}), flush=True)
    summary = {"cases": len(records), "passed": sum(r["passed"] for r in records)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema": "mlx2.lane-matmul-gate.v1",
                                       "mlx": mx.__version__, "summary": summary,
                                       "cases": records}, indent=1) + "\n")
    print(json.dumps(summary), flush=True)
    if summary["passed"] != summary["cases"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

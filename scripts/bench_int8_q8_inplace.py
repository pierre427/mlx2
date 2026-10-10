"""Microbenchmark: Q8 in-place W8A8 prefill vs stock and the requant path.

Synthesizes packed 8-bit GS64 affine weights with the projection shapes of a
model config (default: Qwen3.8-27B-MLX-8bit) -- the model itself is never
loaded -- and times, per shape and row count M:

  stock      mx.quantized_matmul (bf16 activations)
  rq_cached  int8_prefill requant path, int8 weight copy resident
  rq_none    int8_prefill requant path, weight requantized per call
  q8_row     q8 in place, per-row activation scales (Stage A + GEMM)
  q8_g64     q8 in place, GS64 activation scales (Stage A + GEMM)

and Stage A alone per (K, mode), so shared-input projections can be counted
once.  Median of N timed calls after warmup, mx.eval + mx.synchronize per call.
Run under the GPU lock: gpuq.sh <label> python scripts/bench_int8_q8_inplace.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import mlx.core as mx

from mlx2.runtime import int8_prefill as ip


def projections(config_path: Path):
    """[(name, n, k, calls_per_forward, stage_a_group)] for a qwen3_5 text config."""
    cfg = json.loads(config_path.read_text())
    t = cfg.get("text_config", cfg)
    h, inter = t["hidden_size"], t["intermediate_size"]
    layers = t["num_hidden_layers"]
    interval = t["full_attention_interval"]
    n_full = layers // interval
    n_lin = layers - n_full
    kd = t["linear_num_key_heads"] * t["linear_key_head_dim"]
    vd = t["linear_num_value_heads"] * t["linear_value_head_dim"]
    hd = t["head_dim"]
    q_out = t["num_attention_heads"] * hd * (2 if t.get("attn_output_gate") else 1)
    kv_out = t["num_key_value_heads"] * hd
    o_in = t["num_attention_heads"] * hd
    # stage_a_group: projections with the same group share one Stage A.
    return [
        ("mlp.gate_proj", inter, h, layers, "mlp_in"),
        ("mlp.up_proj", inter, h, layers, "mlp_in"),
        ("mlp.down_proj", h, inter, layers, "mlp_down"),
        ("linear_attn.in_proj_qkv", 2 * kd + vd, h, n_lin, "gdn_in"),
        ("linear_attn.in_proj_z", vd, h, n_lin, "gdn_in"),
        ("linear_attn.out_proj", h, vd, n_lin, "gdn_out"),
        ("self_attn.q_proj", q_out, h, n_full, "attn_in"),
        ("self_attn.k_proj", kv_out, h, n_full, "attn_in"),
        ("self_attn.v_proj", kv_out, h, n_full, "attn_in"),
        ("self_attn.o_proj", h, o_in, n_full, "attn_out"),
    ]


def timed(fn, warmup, reps):
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        mx.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1e3


def swapouts():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "Swapouts" in line:
            return int(line.split()[-1].rstrip("."))
    return -1


def rel_l2(a, ref):
    a = a.astype(mx.float32)
    return (mx.linalg.norm(a - ref) / mx.linalg.norm(ref)).item()


def bench_cell(x, w, args, stage_a, *, errors, tiles):
    """Timings (ms) for one activation ``x`` against one weight set ``w``."""
    wq, s, b, st, bt, rq_w, rq_s = w
    m, k = x.shape
    n = wq.shape[0]
    qx = {a: ip.q8_stage_a(x, a) for a in ip.Q8_ACT_SCALES}
    mx.eval(qx)
    cell = {
        "stock": timed(
            lambda: mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=8),
            args.warmup, args.reps),
        "rq_cached": timed(
            lambda: ip._int8_gemm(*ip._quantize_rows(x), rq_w, rq_s),
            args.warmup, args.reps),
        "rq_none": timed(
            lambda: ip._int8_gemm(
                *ip._quantize_rows(x), *ip._requant_packed(wq, s, b, bits=8, group_size=64)),
            args.warmup, args.reps),
    }
    for a, label in (("per_row", "q8_row"), ("group64", "q8_g64")):
        cell[label] = timed(
            lambda a=a: ip.q8_gemm(*ip.q8_stage_a(x, a), wq, st, bt, act_scale=a),
            args.warmup, args.reps)
        cell[label + "_gemm"] = timed(
            lambda a=a: ip.q8_gemm(*qx[a], wq, st, bt, act_scale=a),
            args.warmup, args.reps)
        key = f"{k}:{m}:{a}"
        if key not in stage_a:
            stage_a[key] = timed(lambda a=a: ip.q8_stage_a(x, a), args.warmup, args.reps)
    if errors:
        # Error vs the stock bf16 kernel and vs an fp32 reference.
        ref32 = x.astype(mx.float32) @ mx.dequantize(
            wq, s.astype(mx.float32), b.astype(mx.float32), group_size=64, bits=8).T
        stock_y = mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=8)
        arms = {
            "stock": stock_y,
            "rq": ip._int8_gemm(*ip._quantize_rows(x), rq_w, rq_s),
            "q8_row": ip.q8_gemm(*qx["per_row"], wq, st, bt, act_scale="per_row"),
            "q8_g64": ip.q8_gemm(*qx["group64"], wq, st, bt, act_scale="group64"),
        }
        cell["rel_l2_vs_fp32"] = {kk: rel_l2(v, ref32) for kk, v in arms.items()}
        cell["rel_l2_vs_stock"] = {
            kk: rel_l2(v, stock_y.astype(mx.float32)) for kk, v in arms.items() if kk != "stock"
        }
    if tiles:
        cell["tiles_g64_gemm"] = {
            f"{t[0]}x{t[1]}": timed(
                lambda t=t: ip.q8_gemm(*qx["group64"], wq, st, bt, act_scale="group64", tile=t),
                args.warmup, args.reps)
            for t in ip.Q8_TILES if n % (32 * t[1]) == 0
        }
    return cell


def bench_shape(name, n, k, rows, args, stage_a):
    wq, s, b = mx.quantize(mx.random.normal((n, k)) * 0.02, group_size=64, bits=8)
    s, b = s.astype(mx.bfloat16), b.astype(mx.bfloat16)
    st, bt = ip.q8_metadata(s, b)
    rq_w, rq_s = ip._requant_packed(wq, s, b, bits=8, group_size=64)
    weights = (wq, s, b, st, bt, rq_w, rq_s)
    mx.eval(*weights)
    out = {}
    for m in rows:
        x = (mx.random.normal((m, k)) * 0.5).astype(mx.bfloat16)
        mx.eval(x)
        cell = bench_cell(
            x, weights, args, stage_a,
            errors=m == rows[0], tiles=args.tiles and m == 2048,
        )
        out[str(m)] = cell
        print(name, m, {kk: (round(v, 3) if isinstance(v, float) else v)
                        for kk, v in cell.items()}, flush=True)
        mx.clear_cache()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path.home() / "mlx-models/Qwen3.8-27B-MLX-8bit/config.json"),
    )
    parser.add_argument("--rows", default="512,2048,8192")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=15)
    parser.add_argument("--tiles", action="store_true", help="also sweep q8 tiles")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    mx.set_cache_limit(4 << 30)
    ok, reason = ip.device_support()
    if not ok:
        raise SystemExit(f"unsupported device: {reason}")
    rows = [int(r) for r in args.rows.split(",")]
    projs = projections(Path(args.config))
    swap0 = swapouts()
    results = {
        "device": reason,
        "mlx": mx.__version__,
        "q8_kernel_revision": ip.Q8_KERNEL_REVISION,
        "kernel_revision": ip.KERNEL_REVISION,
        "warmup": args.warmup,
        "reps": args.reps,
        "rows": rows,
        "projections": [],
        "stage_a": {},
    }
    mx.random.seed(0)

    for name, n, k, calls, group in projs:
        if n % 128:
            print(f"skip {name} n={n} (ineligible)")
            continue
        entry = {"name": name, "n": n, "k": k, "calls": calls, "group": group}
        entry["ms"] = bench_shape(name, n, k, rows, args, results["stage_a"])
        results["projections"].append(entry)
        mx.clear_cache()
    results["swapouts_delta"] = swapouts() - swap0
    print("swapouts_delta", results["swapouts_delta"])
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()

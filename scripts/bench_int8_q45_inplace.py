"""Microbenchmark: Q4/Q5 in-place W4A8/W5A8 prefill vs stock quantized_matmul.

Synthesizes packed 4-bit and 5-bit GS64 affine weights with the projection
shapes of a model config (default: Qwen3.8-27B-oQ4e-mtp) -- the model itself
is never loaded -- and times, per shape, bit width and row count M:

  stock      mx.quantized_matmul (bf16 activations)
  q45_row    in place, per-row activation scales (Stage A v8 + GEMM)
  q45_g64    in place, GS64 activation scales (Stage A v8 + GEMM)

plus the GEMM alone per mode and Stage A alone per (K, mode), so
shared-input projections can be counted once.  Each shape is benched at the
bit widths the checkpoint actually uses for it (the config's per-module
overrides), and the per-chunk projection weights calls by those counts.
Median of N timed calls after warmup, mx.eval + mx.synchronize per call.
Run under the GPU lock: gpuq.sh <label> python scripts/bench_int8_q45_inplace.py
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import subprocess
import time
from pathlib import Path

import mlx.core as mx

from mlx2.runtime import int8_prefill as ip


def projections(config_path: Path):
    """[(name, n, k, {bits: layers}, stage_a_group)] for a qwen3_5 text config
    with mixed-bit quantization overrides."""
    cfg = json.loads(config_path.read_text())
    t = cfg.get("text_config", cfg)
    q = cfg.get("quantization") or cfg.get("quantization_config") or {}
    default_bits = int(q.get("bits", 4))
    h, inter = t["hidden_size"], t["intermediate_size"]
    layers = t["num_hidden_layers"]
    interval = t["full_attention_interval"]
    full = [i for i in range(layers) if (i + 1) % interval == 0]
    lin = [i for i in range(layers) if (i + 1) % interval != 0]
    kd = t["linear_num_key_heads"] * t["linear_key_head_dim"]
    vd = t["linear_num_value_heads"] * t["linear_value_head_dim"]
    hd = t["head_dim"]
    q_out = t["num_attention_heads"] * hd * (2 if t.get("attn_output_gate", True) else 1)
    kv_out = t["num_key_value_heads"] * hd
    o_in = t["num_attention_heads"] * hd
    overrides = {}
    for key, value in q.items():
        if not isinstance(value, dict):
            continue
        match = re.search(r"layers\.(\d+)\.(.+)$", key)
        if match and "mtp" not in key:
            overrides[(int(match.group(1)), match.group(2))] = int(value.get("bits", default_bits))

    def census(name, layer_ids):
        counts = collections.Counter(overrides.get((i, name), default_bits) for i in layer_ids)
        return dict(sorted(counts.items()))

    every = list(range(layers))
    return [
        ("mlp.gate_proj", inter, h, census("mlp.gate_proj", every), "mlp_in"),
        ("mlp.up_proj", inter, h, census("mlp.up_proj", every), "mlp_in"),
        ("mlp.down_proj", h, inter, census("mlp.down_proj", every), "mlp_down"),
        ("linear_attn.in_proj_qkv", 2 * kd + vd, h, census("linear_attn.in_proj_qkv", lin), "gdn_in"),
        ("linear_attn.in_proj_z", vd, h, census("linear_attn.in_proj_z", lin), "gdn_in"),
        ("linear_attn.out_proj", h, vd, census("linear_attn.out_proj", lin), "gdn_out"),
        ("self_attn.q_proj", q_out, h, census("self_attn.q_proj", full), "attn_in"),
        ("self_attn.k_proj", kv_out, h, census("self_attn.k_proj", full), "attn_in"),
        ("self_attn.v_proj", kv_out, h, census("self_attn.v_proj", full), "attn_in"),
        ("self_attn.o_proj", h, o_in, census("self_attn.o_proj", full), "attn_out"),
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


def bench_cell(x, w, bits, args, stage_a, *, errors, tiles):
    wq, s, b, st, bt = w
    m, k = x.shape
    n = wq.shape[0]
    qx = {a: ip.q45_stage_a(x, a) for a in ip.Q8_ACT_SCALES}
    mx.eval(qx)
    cell = {
        "stock": timed(
            lambda: mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=bits),
            args.warmup, args.reps),
    }
    for a, label in (("per_row", "q45_row"), ("group64", "q45_g64")):
        cell[label] = timed(
            lambda a=a: ip.q45_gemm(*ip.q45_stage_a(x, a), wq, st, bt, bits=bits, act_scale=a),
            args.warmup, args.reps)
        cell[label + "_gemm"] = timed(
            lambda a=a: ip.q45_gemm(*qx[a], wq, st, bt, bits=bits, act_scale=a),
            args.warmup, args.reps)
        key = f"{k}:{m}:{a}"
        if key not in stage_a:
            stage_a[key] = timed(lambda a=a: ip.q45_stage_a(x, a), args.warmup, args.reps)
    if errors:
        ref32 = x.astype(mx.float32) @ mx.dequantize(
            wq, s.astype(mx.float32), b.astype(mx.float32), group_size=64, bits=bits).T
        stock_y = mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=bits)
        arms = {
            "stock": stock_y,
            "q45_row": ip.q45_gemm(*qx["per_row"], wq, st, bt, bits=bits, act_scale="per_row"),
            "q45_g64": ip.q45_gemm(*qx["group64"], wq, st, bt, bits=bits, act_scale="group64"),
        }
        cell["rel_l2_vs_fp32"] = {kk: rel_l2(v, ref32) for kk, v in arms.items()}
        cell["rel_l2_vs_stock"] = {
            kk: rel_l2(v, stock_y.astype(mx.float32)) for kk, v in arms.items() if kk != "stock"
        }
    if tiles:
        cell["tiles_g64_gemm"] = {
            f"{t[0]}x{t[1]}": timed(
                lambda t=t: ip.q45_gemm(*qx["group64"], wq, st, bt, bits=bits,
                                        act_scale="group64", tile=t),
                args.warmup, args.reps)
            for t in ip.Q45_TILES if n % (32 * t[1]) == 0
        }
    return cell


def bench_shape(name, n, k, bits, rows, args, stage_a):
    wq, s, b = mx.quantize(mx.random.normal((n, k)) * 0.02, group_size=64, bits=bits)
    s, b = s.astype(mx.bfloat16), b.astype(mx.bfloat16)
    st, bt = ip.q8_metadata(s, b)
    weights = (wq, s, b, st, bt)
    mx.eval(*weights)
    out = {}
    for m in rows:
        x = (mx.random.normal((m, k)) * 0.5).astype(mx.bfloat16)
        mx.eval(x)
        cell = bench_cell(
            x, weights, bits, args, stage_a,
            errors=m == rows[0], tiles=args.tiles and m == 2048,
        )
        out[str(m)] = cell
        print(name, f"q{bits}", m, {kk: (round(v, 3) if isinstance(v, float) else v)
                                    for kk, v in cell.items()}, flush=True)
        mx.clear_cache()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path.home() / "mlx-models/Qwen3.8-27B-oQ4e-mtp/config.json"),
    )
    parser.add_argument("--rows", default="512,2048,8192")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=15)
    parser.add_argument("--tiles", action="store_true", help="also sweep tiles at M=2048")
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
        "q45_kernel_revision": ip.Q45_KERNEL_REVISION,
        "default_tiles": {f"q{k}": v for k, v in ip.Q45_DEFAULT_TILES.items()},
        "warmup": args.warmup,
        "reps": args.reps,
        "rows": rows,
        "projections": [],
        "stage_a": {},
    }
    mx.random.seed(0)
    for name, n, k, by_bits, group in projs:
        if n % 128:
            print(f"skip {name} n={n} (ineligible)")
            continue
        for bits, calls in by_bits.items():
            if bits not in ip.Q45_BITS:
                print(f"skip {name} q{bits} (not 4/5-bit)")
                continue
            entry = {"name": name, "n": n, "k": k, "bits": bits, "calls": calls, "group": group}
            entry["ms"] = bench_shape(name, n, k, bits, rows, args, results["stage_a"])
            results["projections"].append(entry)
            mx.clear_cache()
            if swapouts() > swap0 + 1000:
                raise SystemExit("ABORT: host started swapping")
    results["swapouts_delta"] = swapouts() - swap0
    print("swapouts_delta", results["swapouts_delta"])
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Op-level probe: mlx-lm #2044 grouped-expert q4 kernels vs mlx2 MoE paths.

Exploratory, coordinator-owned (refuses Metal without --i-own-the-gpu).
Builds a Qwen3.6 sparse block as the 35B profile serves it (FusedDownSwitchGLU,
separate gate/up, stock expert mode), loads one real MoE layer's expert tables
from the checkpoint (no full model load), and times the routed-expert op
``switch_mlp(x, inds)`` -> per-slot outputs at T tokens, interleaved arms:

  stock            mlx2 default (pad floor 3, policy floor)
  stock_floor1     MLX2_MOE_RHS_PAD_MIN_ROWS=1 (pad to rhs from 1 row/expert)
  stock_always     MLX2_MOE_RHS_PAD_POLICY=always
  grouped_qmv      mlx2 moe_grouped_rows=qmv (stock-exact grouping, T>=2)
  grouped_lane     mlx2 moe_grouped_rows=lane (tensor-unit law, T>=2)
  pr2044           switch_grouped_experts (q4 layers only)

Also reports each arm's relative error vs an fp32 dequantized reference.
Routing: uniform random top-k (real routing concentrates more, which favours
grouping) and a skewed variant.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--layers", default="5,0", help="layer ids (q4 and q6 on the 35B)")
    ap.add_argument("--tokens", default="1,2,4,8,16,32")
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--calls", type=int, default=20, help="calls queued per timed sample")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")
    import mlx.core as mx
    import mlx.nn as nn

    from mlx2.runtime.models import qwen3_next as QN
    from mlx2.runtime.models import qwen36_moe_decode as M
    from mlx2.runtime.models import switch_grouped_experts as PR
    from mlx2.runtime.models import switch_layers as SL

    mx.set_default_device(mx.gpu)
    QN._MOE_FUSED_GATE_UP = False
    cfg = json.loads((a.model / "config.json").read_text())
    tc = cfg.get("text_config", cfg)
    E, H, I, K = tc["num_experts"], tc["hidden_size"], tc["moe_intermediate_size"], tc["num_experts_per_tok"]
    q = cfg["quantization"]
    wm = json.loads((a.model / "model.safetensors.index.json").read_text())["weight_map"]
    shards = {}

    def tensor(key):
        f = wm[key]
        if f not in shards:
            shards[f] = mx.load(str(a.model / f))
        return shards[f][key]

    record = {"schema": "mlx2.probe_switch_grouped_experts.v1", "mlx": mx.__version__,
              "device": mx.device_info().get("device_name"), "E": E, "H": H, "I": I, "K": K,
              "rounds": a.rounds, "calls_per_sample": a.calls, "layers": []}
    tokens = [int(t) for t in a.tokens.split(",")]
    for layer in [int(x) for x in a.layers.split(",")]:
        prefix = f"language_model.model.layers.{layer}.mlp.switch_mlp"
        bits = q[f"{prefix}.gate_proj"]["bits"]
        gsz = q[f"{prefix}.gate_proj"]["group_size"]
        args = SimpleNamespace(hidden_size=H, moe_intermediate_size=I, shared_expert_intermediate_size=I,
                               norm_topk_prob=True, num_experts=E, num_experts_per_tok=K)
        block = M.Qwen36SparseMoeBlock(args)
        nn.quantize(block, group_size=gsz, bits=bits,
                    class_predicate=lambda p, m: hasattr(m, "to_quantized") and "switch_mlp" in p)
        block.set_dtype(mx.bfloat16)
        block.set_fused_expert_kernel_mode("stock")
        block.eval()
        sw = block.switch_mlp
        for proj in ("gate_proj", "up_proj", "down_proj"):
            p = getattr(sw, proj)
            for name in ("weight", "scales", "biases"):
                arr = tensor(f"{prefix}.{proj}.{name}")
                assert arr.shape == p[name].shape, (proj, name, arr.shape, p[name].shape)
                p[name] = arr
        mx.eval(sw.parameters())
        tables = [(getattr(sw, n)["weight"], getattr(sw, n)["scales"], getattr(sw, n)["biases"])
                  for n in ("gate_proj", "up_proj", "down_proj")]

        def reference(x, idx):
            """fp32 per-slot outputs on dequantized tables, 16 assignments at a time."""
            T = idx.shape[-2]
            flat = idx.reshape(-1)
            xs = mx.repeat(x.reshape(T, H).astype(mx.float32), K, axis=0)
            outs = []
            for s in range(0, flat.size, 16):
                e = flat[s:s + 16]
                wg, wu, wd = (mx.dequantize(w[e], sc[e], b[e], group_size=gsz, bits=bits).astype(mx.float32)
                              for (w, sc, b) in tables)
                xx = xs[s:s + 16, :, None]
                g, u = (wg @ xx).squeeze(-1), (wu @ xx).squeeze(-1)
                out = (wd @ (nn.silu(g) * u)[..., None]).squeeze(-1)
                mx.eval(out)
                outs.append(out)
            return mx.concatenate(outs).reshape(1, T, K, H)

        def stock_with(floor, policy):
            def run(x, idx):
                old_floor, old_policy = SL._RHS_PAD_MIN_ROWS_PER_EXPERT, SL._RHS_PAD_POLICY
                SL._RHS_PAD_MIN_ROWS_PER_EXPERT, SL._RHS_PAD_POLICY = floor, policy
                try:
                    return sw(x, idx)
                finally:
                    SL._RHS_PAD_MIN_ROWS_PER_EXPERT, SL._RHS_PAD_POLICY = old_floor, old_policy
            return run

        def grouped(law):
            def run(x, idx):
                block.set_grouped_rows(law)
                try:
                    return sw(x, idx)
                finally:
                    block.set_grouped_rows("off")
            return run

        arms = {"stock": stock_with(3, "floor"), "stock_floor1": stock_with(1, "floor"),
                "stock_always": stock_with(3, "always"), "grouped_qmv": grouped("qmv"),
                "grouped_lane": grouped("lane"),
                "pr2044": lambda x, idx: PR.switch_glu(sw, x, idx, count=False)}
        lrec = {"layer": layer, "bits": bits, "group_size": gsz, "cells": []}
        for T in tokens:
            for routing in ("uniform", "skew"):
                mx.random.seed(1000 * layer + T)
                x = (mx.random.normal((1, T, H)) * 0.5).astype(mx.bfloat16)
                u = mx.random.uniform(shape=(1, T, E))
                if routing == "skew":
                    u = u + 0.6 * (mx.arange(E) < 32).astype(mx.float32)
                idx = mx.argsort(-u, axis=-1)[..., :K].astype(mx.uint32)
                mx.eval(x, idx)
                ref = reference(x, idx)
                mx.eval(ref)
                distinct = len(set(idx.flatten().tolist()))
                cell = {"T": T, "routing": routing, "distinct_experts": distinct,
                        "load_TkE": T * K / E, "arms": {}}
                live = {}
                for name, fn in arms.items():
                    try:
                        y = fn(x, idx)
                    except Exception as exc:  # report, keep going
                        cell["arms"][name] = {"skipped": f"{type(exc).__name__}: {exc}"[:200]}
                        continue
                    if y is None:
                        cell["arms"][name] = {"skipped": "declined"}
                        continue
                    mx.eval(y)
                    err = (mx.linalg.norm((y.astype(mx.float32) - ref).flatten())
                           / mx.linalg.norm(ref.flatten())).item()
                    cell["arms"][name] = {"rel_err_vs_fp32": err, "samples_ms": []}
                    live[name] = fn
                stock_y = arms["stock"](x, idx)
                for name in live:
                    y = live[name](x, idx)
                    cell["arms"][name]["bitwise_eq_stock"] = bool(mx.array_equal(y, stock_y).item())
                    cell["arms"][name]["max_abs_vs_stock"] = mx.max(mx.abs(
                        y.astype(mx.float32) - stock_y.astype(mx.float32))).item()
                for name, fn in live.items():  # warm
                    for _ in range(3):
                        mx.eval([fn(x, idx) for _ in range(a.calls)])
                for _ in range(a.rounds):
                    for name, fn in live.items():
                        mx.synchronize()
                        t0 = time.perf_counter()
                        mx.eval([fn(x, idx) for _ in range(a.calls)])
                        mx.synchronize()
                        cell["arms"][name]["samples_ms"].append((time.perf_counter() - t0) * 1e3 / a.calls)
                base = statistics.median(cell["arms"]["stock"]["samples_ms"])
                for name in live:
                    s = cell["arms"][name]["samples_ms"]
                    cell["arms"][name]["median_ms"] = statistics.median(s)
                    cell["arms"][name]["speedup_vs_stock"] = base / statistics.median(s)
                lrec["cells"].append(cell)
                print(f"L{layer} q{bits} T={T:>2} {routing:7s} distinct={distinct:3d} " + "  ".join(
                    f"{n}={c['median_ms']:.3f}ms({c['speedup_vs_stock']:.2f}x)" for n, c in cell["arms"].items()
                    if "median_ms" in c), flush=True)
        record["layers"].append(lrec)
        del block, sw, tables
        shards.clear()
        mx.clear_cache()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()

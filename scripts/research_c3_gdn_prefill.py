#!/usr/bin/env python3
"""C3: GDN prefill recurrence -- sequential kernel vs chunked core scan.

Microbenchmarks the GDN recurrence at Qwen3.8-27B's linear-attention shapes
(read from the artifact's config.json: Hk=16, Hv=48, Dk=Dv=128) with the
real A_log / dt_bias of one GDN layer, for T in (512, 2048, 8192) and B in
(1, 4):

* ``seq_packed``  -- ``gated_delta_kernel`` (the served default: the packed
  sequential per-token kernel, MLX_GDN_PACKED=1);
* ``seq_generic`` -- the unpacked sequential kernel (MLX_GDN_PACKED=0);
* ``chunk8`` / ``chunk16`` -- ``gated_delta._chunked_prefill`` at the policy's
  default segment (2048 rows), i.e. what ``gdn_prefill_chunk`` 8/16 installs
  (``mx.fast.gated_delta_update`` from the MLX fork).

Numerics: every arm against ``seq_packed`` (max abs / mean abs of the bf16
readout and of the fp32 final state), and, at T=512 B=1, every arm against
the fp32 ops reference (``gated_delta_ops``) as a reorder noise floor.  A
downstream proxy projects the readout [T, 6144] through one fixed random
[6144, 5120] matrix and counts rows whose argmax differs from seq_packed's.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MODEL = Path.home() / "mlx-models/Qwen3.8-27B-oQ4e-mtp"


def swapouts() -> int:
    for line in subprocess.check_output(["vm_stat"], text=True).splitlines():
        if line.startswith("Swapouts"):
            return int(line.split(":")[1].strip().rstrip("."))
    return -1


def therm() -> str:
    return subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--layer", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx

    from mlx2.runtime.models import gated_delta as gd

    mx.set_cache_limit(4 << 30)
    swap0, therm0 = swapouts(), therm()
    cfg = json.loads((MODEL / "config.json").read_text())
    tc = cfg.get("text_config", cfg)
    Hk, Hv = tc["linear_num_key_heads"], tc["linear_num_value_heads"]
    Dk, Dv = tc["linear_key_head_dim"], tc["linear_value_head_dim"]
    hidden = tc["hidden_size"]
    n_layers = tc["num_hidden_layers"]
    n_gdn = sum(1 for i in range(n_layers) if (i + 1) % tc["full_attention_interval"] != 0)
    index = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"language_model.model.layers.{a.layer}.linear_attn."
    params = {}
    import struct

    import numpy as np

    for name in ("A_log", "dt_bias"):
        # Read just this tensor from the shard (numpy has no bfloat16).
        with open(MODEL / index[prefix + name], "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            meta = json.loads(f.read(n))[prefix + name]
            lo, hi = meta["data_offsets"]
            f.seek(8 + n + lo)
            raw = f.read(hi - lo)
        if meta["dtype"] == "BF16":
            arr = mx.array(np.frombuffer(raw, dtype=np.uint16)).view(mx.bfloat16)
        else:
            arr = mx.array(np.frombuffer(raw, dtype={"F32": np.float32, "F16": np.float16}[meta["dtype"]]))
        params[name] = arr.reshape(meta["shape"])
    A_log, dt_bias = params["A_log"], params["dt_bias"]

    def inputs(B, T, seed):
        k0 = mx.random.key(seed)
        ks = mx.random.split(k0, 5)
        q = mx.random.normal((B, T, Hk, Dk), key=ks[0]).astype(mx.bfloat16)
        k = mx.random.normal((B, T, Hk, Dk), key=ks[1]).astype(mx.bfloat16)
        q, k = gd.normalize_gdn_qk(q, k)
        v = mx.random.normal((B, T, Hv, Dv), key=ks[2]).astype(mx.bfloat16)
        av = mx.random.normal((B, T, Hv), key=ks[3]).astype(mx.bfloat16)
        bv = mx.random.normal((B, T, Hv), key=ks[4]).astype(mx.bfloat16)
        beta = gd.gate_sigmoid(bv.astype(mx.float32))
        g = gd.compute_g(A_log, av, dt_bias)
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
        mx.eval(q, k, v, g, beta, state)
        return q, k, v, g, beta, state

    def arms_for(q, k, v, g, beta, state):
        def chunk(c):
            def f():
                stats = {}
                from collections import Counter
                stats = Counter()
                r = gd._chunked_prefill(q, k, v, g, beta, state, None, c, stats, 2048)
                if r is None:
                    raise RuntimeError(f"chunk{c} declined: {dict(stats)}")
                return r
            return f
        return {
            "seq_packed": lambda: gd.gated_delta_kernel(q, k, v, g, beta, state),
            "seq_generic": lambda: gd._gated_delta_kernel_impl(q, k, v, g, beta, state, allow_packed=False),
            "chunk8": chunk(8),
            "chunk16": chunk(16),
        }

    W = (mx.random.normal((Hv * Dv, hidden), key=mx.random.key(77)) / (Hv * Dv) ** 0.5).astype(mx.bfloat16)
    mx.eval(W)

    def diff(y, s, ry, rs):
        dy = mx.abs(y.astype(mx.float32) - ry.astype(mx.float32))
        ds = mx.abs(s - rs)
        B, T = y.shape[:2]
        py = (y.reshape(B, T, -1) @ W).argmax(-1)
        pr = (ry.reshape(B, T, -1) @ W).argmax(-1)
        flips = (py != pr).sum()
        mx.eval(dy, ds, flips)
        return {
            "y_max_abs": float(dy.max()), "y_mean_abs": float(dy.mean()),
            "y_ref_max_abs": float(mx.abs(ry.astype(mx.float32)).max()),
            "state_max_abs": float(ds.max()), "state_mean_abs": float(ds.mean()),
            "state_ref_max_abs": float(mx.abs(rs).max()),
            "proxy_argmax_flips": int(flips), "proxy_rows": B * T,
            "y_bit_identical": bool(mx.array_equal(y, ry)),
        }

    rng = random.Random(6)
    cells = []
    for T in (512, 2048, 8192):
        for B in (1, 4):
            q, k, v, g, beta, state = inputs(B, T, 1000 + T + B)
            arms = arms_for(q, k, v, g, beta, state)
            outs = {}
            for name, fn in arms.items():
                r = fn()
                mx.eval(r)
                outs[name] = r
                mx.eval(fn())
            times = {n: [] for n in arms}
            for _ in range(a.reps):
                order = list(arms)
                rng.shuffle(order)
                for name in order:
                    t = time.perf_counter()
                    mx.eval(arms[name]())
                    times[name].append((time.perf_counter() - t) * 1e3)
            ref_y, ref_s = outs["seq_packed"]
            numerics = {n: diff(*outs[n], ref_y, ref_s) for n in arms if n != "seq_packed"}
            if T == 512 and B == 1:
                oy, os_ = gd.gated_delta_ops(q, k, v, g, beta, state)
                mx.eval(oy, os_)
                numerics["vs_fp32_ops_reference"] = {n: diff(*outs[n], oy, os_) for n in arms}
            med = {n: statistics.median(v) for n, v in times.items()}
            cell = {
                "T": T, "B": B,
                "median_ms": med,
                "min_max_ms": {n: [min(v), max(v)] for n, v in times.items()},
                "speedup_vs_seq_packed": {n: med["seq_packed"] / med[n] for n in med},
                "per_model_ms_all_gdn_layers": {n: med[n] * n_gdn for n in med},
                "numerics_vs_seq_packed": numerics,
            }
            cells.append(cell)
            print(json.dumps({"T": T, "B": B, "median_ms": med}), flush=True)
            del q, k, v, g, beta, state, outs, arms
            mx.clear_cache()
            if swapouts() > swap0:
                print(json.dumps({"abort": "swapouts rose"}), flush=True)
                break
    swap1 = swapouts()
    report = {
        "schema": "mlx2.research.c3-gdn-prefill.v1",
        "source_sha": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "model_config": str(MODEL / "config.json"),
        "shapes": {"Hk": Hk, "Hv": Hv, "Dk": Dk, "Dv": Dv, "hidden": hidden, "gdn_layers": n_gdn,
                   "dtype_qkv": "bfloat16", "g_beta_state": "float32"},
        "real_params_layer": a.layer,
        "segment_rows": 2048,
        "reps": a.reps,
        "cells": cells,
        "swapouts_before": swap0, "swapouts_after": swap1, "swap_rose": swap1 > swap0,
        "therm_before": therm0, "therm_after": therm(),
        "mlx_version": mx.__version__,
        "device": mx.device_info().get("device_name"),
    }
    (out / "gdn.json").write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()

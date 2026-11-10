#!/usr/bin/env python3
"""GPU parity of the DS4.1 Source EXL3 expert projections on a real bank expert.

Loads one expert through the runtime bank loader (deepseek_v41_source_exl3:
manifest -> sealed read -> upload) and compares, per projection and row count,
the runtime ``project`` (math reference and kernel) against the float64 NumPy
oracle ``x @ exl3_reference.effective_weight``. Inputs are fp8-QAT rows from a
fixed seed (activation-scale Gaussian; down inputs are SwiGLU-scale). Also times
both laws. No model load.

usage: adapter_parity.py --bank DIR --source SHA --layer L --expert E [--out JSON]
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import mlx.core as mx  # noqa: E402

from mlx2.adapters import deepseek_v41_source_exl3 as ex  # noqa: E402
from mlx2.runtime.models.deepseek_v41_quant import quantize  # noqa: E402
from mlx2.runtime.models.exl3_reference import effective_weight  # noqa: E402


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--expert", type=int, required=True)
    ap.add_argument("--rows", default="1,6,64,512")
    ap.add_argument("--out")
    ap.add_argument("--cpu", action="store_true", help="CPU device (kernel law then runs the reference law)")
    a = ap.parse_args()
    if a.cpu:
        mx.set_default_device(mx.cpu)
    root, records, fp, words, meta = ex.load_manifest(a.bank, a.source)
    key = (a.layer, a.expert)
    raw = ex.exl3_read(root, records[key], key, words)
    dense = {k: effective_weight(raw[k + ".trellis"][0], raw[k + ".suh"][0], raw[k + ".svh"][0], "mcg")
             for k in ex.PARTS}
    experts = {law: ex.upload(raw, words, law) for law in ex.MATH_LAWS}
    rng = np.random.default_rng(0)
    rows_list = [int(r) for r in a.rows.split(",")]
    out = {"bank_fingerprint": fp, "words": words, "layer": a.layer, "expert": a.expert,
           "device": str(mx.default_device()), "cells": []}
    for kind, (i, o) in ex.GEOMETRY.items():
        for rows in rows_list:
            scale = 1.0 if kind != "down" else 0.3
            x = np.array(quantize(mx.array((rng.standard_normal((rows, i)) * scale).astype(np.float32)), "fp8"))
            ref = x.astype(np.float64) @ dense[kind]
            cell = {"kind": kind, "rows": rows}
            for law, expert in experts.items():
                xa = mx.array(x)
                y = ex.project(expert, kind, xa); mx.eval(y)
                t0 = time.perf_counter()
                for _ in range(5):
                    y = ex.project(expert, kind, xa); mx.eval(y)
                dt = (time.perf_counter() - t0) / 5
                y = np.array(y, np.float64)
                cell[law] = {"rel_err": rel(y, ref), "max_abs": float(np.abs(y - ref).max()),
                             "ref_max": float(np.abs(ref).max()), "ms": round(dt * 1e3, 3),
                             "finite": bool(np.isfinite(y).all())}
            out["cells"].append(cell)
            print(json.dumps(cell), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

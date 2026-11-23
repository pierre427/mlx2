#!/usr/bin/env python3
"""Read-once expert kernels vs the stock SwitchGLU on one real Hy4-flash layer.

For each routing cell (rows x law) runs the stock ``SwitchGLU`` (gather_qmm)
and ``switch_read_once`` on the same inputs and an fp32 reference of the same
law (each touched expert dequantized once; gate/up and the down output
rounded to bf16 where the stock path rounds them, everything else fp32).
Reports, per cell: bit-exactness and max abs / relative / ulp differences of
read-once vs stock, and of each vs the fp32 reference, so a reduction-order
difference can be told from an error.  Also runs the full Hy4 combine
(``(y * router_weights).sum(-2)``) on both outputs with random router
weights and reports the same numbers on the combined row.

Single process on the GPU, one layer (~1.9 GB), no server; needs the lease.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import hy4_expert_kernel_bench as B  # noqa: E402
from mlx2.runtime.models import switch_read_once as RO  # noqa: E402


def bf16(a):
    return a.astype(mx.bfloat16).astype(mx.float32)


def reference(glu, x, idx, limit):
    """fp32 per-assignment outputs [R, k, D] of the stock law."""
    R, k = idx.shape
    D = x.shape[-1]
    out = np.zeros((R, k, D), dtype=np.float32)
    xf = x.astype(mx.float32)

    def deq(p, e):
        return mx.dequantize(p["weight"][e], p["scales"][e], p["biases"][e], group_size=64, bits=4).astype(mx.float32)

    for e in np.unique(idx):
        rows, slots = np.nonzero(idx == e)
        xe = xf[mx.array(rows)]
        g = bf16(xe @ deq(glu.gate_proj, int(e)).T)
        u = bf16(xe @ deq(glu.up_proj, int(e)).T)
        gm = mx.minimum(g, limit)
        h = bf16(bf16(gm * bf16(mx.sigmoid(gm))) * bf16(mx.clip(u, -limit, limit)))
        y = bf16(h @ deq(glu.down_proj, int(e)).T)
        out[rows, slots] = np.array(y)
    return out


def diff(a, b) -> dict:
    d = np.abs(a - b)
    ref_max = float(np.abs(b).max())
    ulp = 2.0 ** (np.floor(np.log2(max(ref_max, 1e-30))) - 7)
    return {"bit_exact": bool(np.array_equal(a, b)), "max_abs": float(d.max()),
            "max_abs_ulps_of_ref_max": float(d.max() / ulp), "ref_max_abs": ref_max,
            "rel_l2": float(np.linalg.norm(d) / max(np.linalg.norm(b), 1e-30)),
            "mismatch_frac": float((d > 0).mean())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--rows", default="1,2,4,6,7,8,12,16,24,28,32,48,64")
    ap.add_argument("--laws", default="uniform,overlap")
    ap.add_argument("--swiglu-limit", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; refusing to overwrite")
    mx.set_default_device(mx.gpu)
    RO.MIN_ROWS, RO.MAX_ASSIGNMENTS = 1, 1 << 30
    glu, _ = B.load_layer(a.model, a.layer, B.make_activation("clamped", a.swiglu_limit))
    rng = np.random.default_rng(a.seed)
    cells = []
    for law in a.laws.split(","):
        for r in [int(v) for v in a.rows.split(",")]:
            idx = B.routing(r, law, rng, 1.7)
            # Activations at the scale the routed-expert latent sees (rms ~1).
            x = mx.random.normal((r, 1536)).astype(mx.bfloat16)
            ind = mx.array(idx.astype(np.uint32))
            stock = np.array(glu(x, ind).astype(mx.float32))
            ro = np.array(RO.switch_glu(glu, x, ind).astype(mx.float32))
            ref = reference(glu, x, idx, a.swiglu_limit)
            w = rng.random((r, 16)).astype(np.float32)
            w = w / w.sum(-1, keepdims=True) * 3.998
            comb = lambda y: np.array((mx.array(y).astype(mx.bfloat16) * mx.array(w)[..., None])
                                      .sum(axis=-2).astype(mx.bfloat16).astype(mx.float32))
            cell = {"law": law, "rows": r, "unique_experts": int(len(np.unique(idx))),
                    "read_once_vs_stock": diff(ro, stock), "stock_vs_fp32_ref": diff(stock, ref),
                    "read_once_vs_fp32_ref": diff(ro, ref),
                    "combined_read_once_vs_stock": diff(comb(ro), comb(stock))}
            cells.append(cell)
            print(json.dumps({"law": law, "rows": r,
                              "ro_vs_stock_max_abs": cell["read_once_vs_stock"]["max_abs"],
                              "ro_vs_stock_ulps": cell["read_once_vs_stock"]["max_abs_ulps_of_ref_max"],
                              "ro_vs_stock_rel_l2": cell["read_once_vs_stock"]["rel_l2"],
                              "stock_vs_ref_rel_l2": cell["stock_vs_fp32_ref"]["rel_l2"],
                              "ro_vs_ref_rel_l2": cell["read_once_vs_fp32_ref"]["rel_l2"],
                              "combined_max_abs": cell["combined_read_once_vs_stock"]["max_abs"]}), flush=True)
    receipt = {"schema": "mlx2.hy4-read-once-exactness.v1", "qualification": "unqualified",
               "model": str(a.model), "layer": a.layer, "activation": f"clamped_swiglu({a.swiglu_limit})",
               "mlx2_commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                             capture_output=True, text=True).stdout.strip(),
               "mlx2_dirty": bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                                                 capture_output=True, text=True).stdout.strip()),
               "mlx": mx.__version__, "device": mx.device_info().get("device_name"),
               "read_once_config": RO.status()["config"], "cells": cells}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(receipt, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

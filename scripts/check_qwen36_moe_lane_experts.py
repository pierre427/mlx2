#!/usr/bin/env python3
# ruff: noqa: B023 -- every closure here is called within its own loop iteration.
"""Metal gate for the tensor-unit (lane law) Qwen3.6 expert tiles.

Coordinator-owned; refuses Metal without --i-own-the-gpu.  On synthetic
expert tables (affine g64 at 4/6/8 bits) it requires every (row, slot)
output of ``lane_grouped_rows`` to equal, bit for bit, the dense lane matmul
of that row through its expert's own projections (the lane law the tiles
claim), and reports the deviation from the stock gathers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--widths", default="2,3,8,16")
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")
    import mlx.core as mx
    import mlx.nn as nn

    from mlx2.runtime.lane import matmul as L
    from mlx2.runtime.models import qwen36_moe_grouped as G
    from mlx2.runtime.models.activations import swiglu
    from mlx2.runtime.models.switch_layers import SwitchGLU

    mx.set_default_device(mx.gpu)
    E, H, I, K = 16, 2048, 512, 8
    checks = []
    for bits in (4, 6, 8):
        mx.random.seed(bits)
        sw = SwitchGLU(H, I, E)
        for proj, shape in (("gate_proj", (E, I, H)), ("up_proj", (E, I, H)), ("down_proj", (E, H, I))):
            getattr(sw, proj).weight = mx.random.normal(shape) * 0.05
        nn.quantize(sw, group_size=64, bits=bits)
        sw.set_dtype(mx.bfloat16)

        def dense(layer, e):
            ql = nn.QuantizedLinear(64, 8, bias=False, group_size=64, bits=bits)
            ql.weight, ql.scales, ql.biases = layer["weight"][e], layer["scales"][e], layer["biases"][e]
            return L.prepare(ql)

        lanes = {e: [dense(sw.gate_proj, e), dense(sw.up_proj, e), dense(sw.down_proj, e)] for e in range(E)}
        for rows in [int(w) for w in a.widths.split(",")]:
            x = (mx.random.normal((1, rows, H)) * 0.5).astype(mx.bfloat16)
            inds = mx.argpartition(mx.random.uniform(shape=(1, rows, E)), kth=-K, axis=-1)[..., -K:]
            inds = inds.astype(mx.uint32)
            got = G.lane_grouped_rows(x, inds, sw.gate_proj, sw.up_proj, sw.down_proj)
            stock = sw(x, inds)
            ref = []
            for r in range(rows):
                for s in range(K):
                    e = int(inds[0, r, s].item())
                    g_, u_, d_ = lanes[e]
                    xr = x[0, r:r + 1]
                    h = swiglu(L.lane_matmul(xr, g_), L.lane_matmul(xr, u_))
                    ref.append(L.lane_matmul(h, d_))
            ref = mx.concatenate(ref, axis=0).reshape(got.shape)
            mx.eval(got, stock, ref)
            diff = mx.abs(got.astype(mx.float32) - ref.astype(mx.float32))
            sdiff = mx.abs(got.astype(mx.float32) - stock.astype(mx.float32))
            checks.append({
                "bits": bits, "rows": rows,
                "lane_law_equal": bool(mx.array_equal(got, ref).item()),
                "lane_law_max_abs_diff": float(mx.max(diff).item()),
                "stock_max_abs_diff": float(mx.max(sdiff).item()),
                "stock_rel_rms": float((mx.sqrt(mx.mean(sdiff * sdiff))
                                        / mx.sqrt(mx.mean(stock.astype(mx.float32) ** 2))).item()),
                "ref_vs_stock_rel_rms": float((mx.sqrt(mx.mean((ref.astype(mx.float32) - stock.astype(mx.float32)) ** 2))
                                               / mx.sqrt(mx.mean(stock.astype(mx.float32) ** 2))).item()),
            })
            print(checks[-1], flush=True)
    ok = all(c["lane_law_equal"] for c in checks)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"schema": "mlx2.qwen36_moe_lane_experts_check.v1", "mlx": mx.__version__,
                                 "passed": ok, "checks": checks}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

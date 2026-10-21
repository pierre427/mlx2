#!/usr/bin/env python3
"""Metal gate for the expert-grouped Qwen3.6 rows on synthetic tables.

Coordinator-owned; refuses Metal without --i-own-the-gpu.  For each format
(affine g64 at 4/6/8 bits) and window width it builds a Qwen3.6 sparse block
with random expert tables, routes natural and skewed (many rows sharing
experts) windows, and requires the grouped per-slot expert outputs and block
outputs to equal the stock module bit for bit, and every row to equal its
value in a differently composed tile (row invariance).  Real-weight checks
live in ``probe_qwen36_tree_moe_rows.py``.  Exit 0 only when every check holds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--widths", default="2,3,4,8,16,24,32")
    ap.add_argument("--geometry", default="{}", help="JSON merged into the grouped DEFAULT_GEOMETRY")
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")
    import mlx.core as mx
    import mlx.nn as nn

    from mlx2.runtime.models import qwen3_next as QN
    from mlx2.runtime.models import qwen36_moe_decode as M
    from mlx2.runtime.models import qwen36_moe_grouped as G

    G.DEFAULT_GEOMETRY.update(json.loads(a.geometry))

    mx.set_default_device(mx.gpu)
    QN._MOE_FUSED_GATE_UP = False
    E, H, I = 256, 2048, 512
    widths = [int(w) for w in a.widths.split(",")]
    checks = []
    for bits in (4, 6, 8):
        mx.random.seed(bits)
        args = SimpleNamespace(
            hidden_size=H, moe_intermediate_size=I, shared_expert_intermediate_size=I,
            norm_topk_prob=True, num_experts=E, num_experts_per_tok=8,
        )
        block = M.Qwen36SparseMoeBlock(args)
        for proj, shape in (("gate_proj", (E, I, H)), ("up_proj", (E, I, H)), ("down_proj", (E, H, I))):
            getattr(block.switch_mlp, proj).weight = mx.random.normal(shape) * 0.05
        nn.quantize(block, group_size=64, bits=bits,
                    class_predicate=lambda p, m: hasattr(m, "to_quantized") and "switch_mlp" in p)
        block.set_dtype(mx.bfloat16)
        block.set_fused_expert_kernel_mode("stock")  # the Qwen3.6 profile pins stock
        block.eval()
        sw = block.switch_mlp
        for rows in widths:
            for routing in ("natural", "skewed"):
                x = (mx.random.normal((1, rows, H)) * 0.5).astype(mx.bfloat16)
                if routing == "natural":
                    inds = mx.argpartition(mx.random.uniform(shape=(1, rows, E)), kth=-8, axis=-1)[..., -8:]
                else:   # every row from a pool of 12 experts: groups of up to `rows` members
                    perm = mx.argpartition(mx.random.uniform(shape=(1, rows, 12)), kth=-8, axis=-1)[..., -8:]
                    inds = perm
                inds = inds.astype(mx.uint32)
                block.set_grouped_rows(False)
                stock = sw(x, inds)
                stock_block = block(x)
                block.set_grouped_rows(True)
                refusal = G.admission_refusal(x, inds, sw.gate_proj, sw.up_proj, sw.down_proj)
                grouped = sw(x, inds)
                grouped_block = block(x)
                perm_rows = mx.array(list(range(rows))[::-1])
                rev = sw(x[:, perm_rows], inds[:, perm_rows])[:, perm_rows]
                block.set_grouped_rows(False)
                mx.eval(stock, stock_block, grouped, grouped_block, rev)
                checks.append({
                    "bits": bits, "rows": rows, "routing": routing, "admission": refusal,
                    "unique_experts": len(set(inds.reshape(-1).tolist())),
                    "slots_equal": bool(mx.array_equal(stock, grouped).item()),
                    "block_equal": bool(mx.array_equal(stock_block, grouped_block).item()),
                    "reversed_tile_equal": bool(mx.array_equal(rev, grouped).item()),
                    "max_abs_diff": float(mx.max(mx.abs(stock.astype(mx.float32) - grouped.astype(mx.float32))).item()),
                })
        stats = block.grouped_rows_stats()
        checks.append({"bits": bits, "grouped_stats": stats})
    rows_ok = [c for c in checks if "slots_equal" in c]
    ok = all(c["admission"] is None and c["slots_equal"] and c["block_equal"] and c["reversed_tile_equal"]
             for c in rows_ok)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"schema": "mlx2.qwen36_moe_grouped_rows_check.v1", "mlx": mx.__version__,
                                 "passed": ok, "checks": checks}, indent=1))
    print(json.dumps({"passed": ok, "summary": [
        (c["bits"], c["rows"], c["routing"], c["slots_equal"], c["block_equal"], c["reversed_tile_equal"],
         c["admission"]) for c in rows_ok]}, indent=0))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

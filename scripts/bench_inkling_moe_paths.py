#!/usr/bin/env python3
"""Inkling routed-MoE decode/verify paths at the real layer geometry (diagnostic).

Synthetic q2/g128 weights with the checkpoint's shapes (hidden 4096, 256 experts,
top-6, expert width 2048, two shared experts).  Times, per row count: the
unsorted M=1 gate/up gathers alone, the chunked exact routed path (gathers +
fused down-combine), the grouped exact routed path (experts.py), and the whole
MoE ``tree`` call for each; checks grouped == chunked bit for bit.  Not a model
throughput result.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mlx.core as mx  # noqa: E402
from mlx import nn  # noqa: E402


LAYERS = 40  # one model's worth of MoE layers per eval amortizes submit/sync


def timed(fn, repeats):
    out = fn()
    mx.eval(out)
    samples = []
    for _ in range(repeats):
        mx.synchronize()
        tic = time.perf_counter()
        mx.eval([fn() for _ in range(LAYERS)])
        samples.append((time.perf_counter() - tic) * 1000 / LAYERS)
    return out, statistics.median(samples)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--repeats", type=int, default=15)
    args = ap.parse_args()
    from mlx2.adapters.inkling import inspect_artifact
    from mlx2.runtime.models.inkling import ModelArgs
    from mlx2.runtime.models.inkling import language as L

    artifact = inspect_artifact(args.model)
    config = ModelArgs.from_dict(artifact["text_config"])
    mx.set_default_device(mx.gpu)
    mx.random.seed(17)
    m = L.InklingSparseMoE(config)
    m.set_dtype(mx.bfloat16)
    nn.quantize(m.switch_mlp, group_size=128, bits=2, mode="affine")
    m.fused = m.down_combine = True
    mx.eval(m.parameters())
    sm = m.switch_mlp
    gate_up_bytes_per_row = 2 * config.num_experts_per_tok * sm.gate_proj.weight[0].nbytes
    report = {
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "geometry": {"hidden": config.hidden_size, "experts": config.n_routed_experts,
                     "top_k": config.num_experts_per_tok,
                     "expert_width": config.moe_intermediate_size},
        "down_combine_variant": L._DOWN_COMBINE_VARIANT,
        "cells": [],
        "scope": "synthetic q2 weights at checkpoint geometry; median ms per layer (40 issued per eval); diagnostic only",
    }
    for rows in args.rows:
        x = mx.random.normal((1, rows, config.hidden_size)).astype(mx.bfloat16)
        xf = x.reshape(-1, config.hidden_size)
        idx, weights, _ = m._route(m.router_logits(xf))
        xe = mx.expand_dims(xf, (-2, -3))
        _, gate_up_ms = timed(lambda: (sm.up_proj(xe, idx), sm.gate_proj(xe, idx)), args.repeats)
        m.grouped = False
        chunked, chunked_ms = timed(lambda: m.tree(x), args.repeats)
        m.grouped = True
        grouped, grouped_ms = timed(lambda: m.tree(x), args.repeats)
        m.grouped = False
        cell = {
            "rows": rows,
            "gate_up_gather_ms": gate_up_ms,
            "gate_up_gb_per_s": gate_up_bytes_per_row * rows / gate_up_ms / 1e6,
            "tree_chunked_ms": chunked_ms,
            "tree_grouped_ms": grouped_ms,
            "grouped_equals_chunked": bool(mx.array_equal(chunked, grouped).item()),
        }
        report["cells"].append(cell)
        print(json.dumps(cell), flush=True)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

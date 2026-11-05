#!/usr/bin/env python3
"""Attribute Inkling packed-tree verify cost by ablation (diagnostic, not exact).

On a real prefilled prefix, times chain trees of width W through the packed
executor: as shipped; with the routed MoE swapped to the ordinary sorted path;
with attention (and its cache writes) stubbed out; and tree construction
(private cache descriptors) alone.  Ablated arms are numerically different by
design -- they only attribute time.
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
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import mlx.core as mx  # noqa: E402
from probe_inkling import atomic, context_cases, host, install_serving_lane, ownership  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--contexts", type=int, nargs="+", default=[16384, 32768])
    ap.add_argument("--widths", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--lane-matmul", default="exact", choices=["exact", "auto", "off"],
                    help="lane law to install, as `mlx2 serve --lane-matmul` (packed AST serves exact)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    from mlx2.adapters.inkling import InklingAdapter
    from mlx2.runtime.inkling_tree_executor import InklingTreeExecutor
    from mlx2.runtime.models.inkling import tree as tree_module

    policy = json.loads(args.policy.read_text())
    report = {"source_revision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "ownership": ownership(), "before": host(), "policy": policy, "cells": [],
        "scope": "ablation attribution; ablated arms are not numerically valid"}
    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(1 << 30)
    adapter = InklingAdapter(str(args.model), execution_policy=policy)
    report["checkpoint"] = args.model.name + "@" + adapter.identity["fingerprint"]
    lane = install_serving_lane(adapter, args.model, args.lane_matmul)
    report["lane_matmul"] = {"mode": args.lane_matmul, "law_id": (lane or {}).get("law_id", "stock")}
    model = adapter.model
    model.configure_kernels(
        fused=policy.get("inkling_fused", False), down_combine=policy.get("inkling_q2_down_combine", False),
        attention=policy.get("inkling_attention", "reference"), grouped=False,
        tree_kernels=True, tree_cohort=True,
        shortconv_prefill=policy.get("inkling_shortconv_prefill", False))
    executor = InklingTreeExecutor.create()
    sub = argparse.Namespace(model=args.model, contexts=args.contexts, context_tokens=0,
                             generation_tokens=256)
    cases = {k: v for k, v in context_cases(sub).items() if k.startswith("long_generation")}
    original_attention = tree_module.InklingTree.attention
    moe_layers = [layer.mlp for layer in model.layers if hasattr(layer.mlp, "switch_mlp")]
    original_tree = {id(m): m.tree for m in moe_layers}

    def stub_attention(self, module, x, item, *, packed=None, project_output=True):
        return module.o_proj(x) if project_output else x

    arms = {
        "shipped": lambda: None,
        "moe_sorted": lambda: [setattr(m, "tree", m.__call__) for m in moe_layers],
        "no_attention": lambda: setattr(tree_module.InklingTree, "attention", stub_attention),
    }

    def restore():
        tree_module.InklingTree.attention = original_attention
        for m in moe_layers:
            m.tree = original_tree[id(m)]

    try:
        for name, value in cases.items():
            prompt = adapter.prompt_tokens({"messages": [{"role": "user", "content": value[0]}]})
            cache = model.make_cache()
            for begin in range(0, len(prompt), 2048):
                model(mx.array([prompt[begin:begin + 2048]], mx.uint32), cache=cache)
                mx.eval([c.state for c in cache])
            vocab = len(adapter.tokenizer)
            samples = {}
            for repeat in range(args.repeats + 1):
                for arm, apply in (arms.items() if repeat % 2 == 0 else reversed(list(arms.items()))):
                    for w in args.widths:
                        apply()
                        proposed = [(prompt[-1] + 7 * i + 3) % vocab for i in range(w)]
                        mx.synchronize()
                        tic = time.perf_counter()
                        values, _, tx = executor.forward(
                            model, proposed, [-1] + list(range(w - 1)), cache, [1], cached=False)
                        mx.eval(values)
                        elapsed = (time.perf_counter() - tic) * 1000
                        restore()
                        del values, tx
                        if repeat:
                            samples.setdefault(f"{arm}:{w}", []).append(elapsed)
                for w in args.widths:
                    tic = time.perf_counter()
                    tree = tree_module.InklingTree([-1] + list(range(w - 1)), cache)
                    mx.eval([c.state for pair in tree.private for c in pair.caches])
                    elapsed = (time.perf_counter() - tic) * 1000
                    del tree
                    if repeat:
                        samples.setdefault(f"construct:{w}", []).append(elapsed)
            cell = {"case": name, "prompt_tokens": len(prompt),
                    "median_ms": {k: statistics.median(v) for k, v in samples.items()}}
            report["cells"].append(cell)
            atomic(args.out / "report.json", report)
            print(json.dumps(cell), flush=True)
            del cache
            mx.clear_cache()
    finally:
        restore()
        adapter.close()
        report["after"] = host()
        atomic(args.out / "report.json", report)


if __name__ == "__main__":
    main()

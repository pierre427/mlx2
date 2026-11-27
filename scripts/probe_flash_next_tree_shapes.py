"""Flash-Next AST tile shapes: cost and emitted tokens per shape (diagnostic).

For native-MTP packed slates of several shapes (one chain of 2..5 drafts up
to the 16-row 3x5 / 5x3 trees), at a few decode positions of a few prompts,
runs the exact tree law and the fastest row-invariant stack, and records per
shape: warm tile time (eval-fenced median), emitted tokens per tile (greedy
walk of the law's own argmax), distinct routed experts per layer, and the
projected tokens per second next to ordinary one-token decode. Bounded smoke,
not publishable statistics. GPU lease only.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

PROMPTS = (
    ("prose", "Explain how a database index speeds up queries. Use one short paragraph."),
    ("code", "Write a Python function that merges two sorted lists into one sorted list, with a docstring."),
    ("math", "A train travels 180 km in 2.5 hours, then 120 km in 1.5 hours. What is its average speed? Show the steps."),
)
POSITIONS = (8, 24, 40, 56)
SHAPES = {
    "1x2": (2,), "1x3": (3,), "1x4": (4,), "1x5": (5,),
    "2x2": (2, 2), "2x3": (3, 3), "3x2": (2, 2, 2), "2x4": (4, 4), "2x5": (5, 5),
    "3x3": (3, 3, 3), "3x4": (4, 4, 4), "3x5": (5, 5, 5), "5x3": (3, 3, 3, 3, 3),
}
REPEATS = 3


def accepted(tokens, parents, argmax):
    children = {}
    for row, parent in enumerate(parents):
        children.setdefault(parent, []).append(row)
    row, out = 0, []
    while True:
        want = argmax[row]
        out.append(want)
        nxt = [c for c in children.get(row, []) if tokens[c] == want]
        if not nxt:
            return out
        row = nxt[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.root, text=True).strip()
    if head != args.source:
        raise RuntimeError(f"source HEAD {head} != {args.source}")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=args.root, text=True).strip():
        raise RuntimeError("tracked source dirty")
    if args.out.exists():
        raise RuntimeError("refusing an existing evidence directory")
    args.out.mkdir(parents=True)
    report = {"schema": "mlx2.flash-next-tree-shapes.v1", "source_revision": head,
              "checkpoint": f"{args.model.name}@sha256:{args.fingerprint}", "status": "running",
              "qualified": False, "defaulted": False, "timing": "bounded smoke, eval-fenced median",
              "loadavg_before": subprocess.check_output(["sysctl", "-n", "vm.loadavg"], text=True).strip(),
              "samples": []}
    save = lambda: (args.out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    adapter = None
    try:
        from mlx2.adapters.registry import inspect_model
        if inspect_model(args.model).artifact["fingerprint"] != args.fingerprint:
            raise RuntimeError("checkpoint fingerprint changed")
        import mlx.core as mx
        import numpy as np
        mx.set_memory_limit(80 << 30)
        mx.set_cache_limit(256 << 20)
        from mlx2.adapters.flash_next import FlashNextAdapter
        policy = {"draft_model": str(args.model), "num_draft": 5, "row_exact_verify": False,
                  "progressive_verification_tile": 2, "progressive_alternative_branches": 3,
                  "progressive_alternative_strategy": "confidence_tree", "cost_per_commit_handoff": False,
                  "continuation_pool": {"sources": ["external"], "limit": 3, "ast_planner": True,
                                        "packed_pattern": [5, 5, 5]},
                  "ast_target_tree_oracle": str(args.out / "unused-oracle.json")}
        adapter = FlashNextAdapter(str(args.model), execution_policy=policy)
        model = adapter.model
        from mlx2.adapters import flash_next_tree as T
        from mlx2.runtime.models import flash_tree_mma_law as L, qwen3_next as M
        from mlx2.runtime.drafters.native_mtp import NativeMTPDraftModel
        if os.environ.get("MLX_QWEN4_PLE_NVME") != str(args.model / "ple_rows.bin"):
            raise RuntimeError("PLE offload is not active")
        L.FUSED_GLUE = L.CACHED_PADDING = True
        trunk = model.language_model.model
        layers = (len(trunk.layers) - 1,)
        laws = {
            "exact_scalar": T.FlashNextTargetTreeExecutor(model),
            "fast_stack": T.FlashNextTargetTreeExecutor(
                model, tile_law="row_invariant_mma",
                mma_parts=("projections", "experts_device", "qsa_chains", "router_gates", "router_kernel",
                           "hc_matrix")),
        }
        draft = NativeMTPDraftModel(model, num_draft=5, revision=adapter.identity["fingerprint"])

        def run(provider, tokens, parents, base):
            logits, features, transaction = provider.forward(model, tokens, parents, base, layers)
            mx.eval(logits, features)
            transaction.abort()
            return logits[0] if logits.ndim == 3 else logits

        captured = []
        stock = M._stock_routing

        def recording(block_, gates):
            inds, scores = stock(block_, gates)
            captured.append(inds)
            return inds, scores

        ordinary_ms = []
        for name, text in PROMPTS:
            prompt = list(adapter.prompt_tokens({"messages": [{"role": "user", "content": text}],
                                                 "enable_thinking": False}))
            cache = model.make_cache()
            logits = model(mx.array([prompt]), cache=cache)
            greedy = [int(mx.argmax(logits[0, -1]).item())]
            while len(greedy) < max(POSITIONS) + 8:
                mx.synchronize(); t0 = time.perf_counter()
                logits = model(mx.array([[greedy[-1]]]), cache=cache)
                greedy.append(int(mx.argmax(logits[0, -1]).item()))
                if len(greedy) > 8:
                    ordinary_ms.append((time.perf_counter() - t0) * 1000)
            del cache
            for position in POSITIONS:
                context, anchor = prompt + greedy[:position], greedy[position]
                base = model.make_cache()
                features = model.prefill_body(mx.array([context]), base, layers)
                mx.eval(features)
                for shape, pattern in SHAPES.items():
                    slate = draft.draft_packed_paths([anchor], features, draft.make_cache(), pattern,
                                                     context_tokens=[context[1:] + [anchor]])[0]
                    tokens, parents = [anchor], [-1]
                    for path in slate:
                        parent = 0
                        for token in path.tokens:
                            parents.append(parent)
                            parent = len(tokens)
                            tokens.append(token)
                    sample = {"prompt": name, "position": position, "shape": shape, "rows": len(tokens),
                              "laws": {}}
                    for law, provider in laws.items():
                        captured.clear()
                        M._stock_routing = recording
                        try:
                            full = run(provider, tokens, parents, base)
                        finally:
                            M._stock_routing = stock
                        distinct = None
                        if captured:
                            mx.eval(captured)
                            distinct = statistics.mean(len(set(np.asarray(c).reshape(-1).tolist())) for c in captured)
                        argmax = [int(v) for v in np.asarray(mx.argmax(full, axis=-1))]
                        samples = []
                        for _ in range(REPEATS):
                            mx.synchronize(); t0 = time.perf_counter()
                            run(provider, tokens, parents, base)
                            mx.synchronize()
                            samples.append((time.perf_counter() - t0) * 1000)
                        sample["laws"][law] = {"tile_ms": statistics.median(samples),
                                               "emitted": len(accepted(tokens, parents, argmax)),
                                               "distinct_experts_per_layer": distinct}
                    report["samples"].append(sample)
                    print(name, position, shape, {k: (round(v["tile_ms"], 1), v["emitted"]) for k, v in sample["laws"].items()},
                          flush=True)
                save()
                del base, features
                mx.clear_cache()
        ordinary = statistics.median(ordinary_ms)
        summary = {"ordinary_step_ms_median": ordinary, "ordinary_tok_s": 1000 / ordinary, "shapes": {}}
        for shape in SHAPES:
            rows = [s for s in report["samples"] if s["shape"] == shape]
            entry = {"rows": rows[0]["rows"]}
            for law in laws:
                tile = statistics.median(s["laws"][law]["tile_ms"] for s in rows)
                emitted = statistics.mean(s["laws"][law]["emitted"] for s in rows)
                distinct = [s["laws"][law]["distinct_experts_per_layer"] for s in rows
                            if s["laws"][law]["distinct_experts_per_layer"] is not None]
                entry[law] = {"tile_ms_median": tile, "tile_in_ordinary_steps": tile / ordinary,
                              "mean_emitted": emitted, "projected_tok_s": emitted / tile * 1000,
                              "speedup_vs_ordinary": (emitted / tile) / (1 / ordinary),
                              "distinct_experts_per_layer": statistics.mean(distinct) if distinct else None,
                              "experts_per_emitted_token": (statistics.mean(distinct) / emitted) if distinct else None}
            summary["shapes"][shape] = entry
        report["summary"] = summary
        report["status"] = "passed"
        print(json.dumps(summary, indent=1), flush=True)
    except BaseException as exc:
        report["status"], report["error"] = "failed", repr(exc)
        raise
    finally:
        if adapter is not None:
            adapter.close()
        report["loadavg_after"] = subprocess.check_output(["sysctl", "-n", "vm.loadavg"], text=True).strip()
        save()


if __name__ == "__main__":
    main()

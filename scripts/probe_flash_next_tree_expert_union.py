"""Distinct routed experts touched by real Flash-Next 16-row AST tree tiles.

Diagnostic only, never a serving, performance or qualification result. It
loads one Flash-Next artifact, builds real native-MTP packed slates (3x5 and
5x3) at a few decode positions of a few prompts, runs the existing
FlashNextTargetTreeExecutor forward on each tile, and records the routed
expert indices each MoE layer chose for the tile's rows. It reports the
distinct-expert union per layer, the rows-per-expert occupancy histogram and
the share of (row, slot) pairs whose expert has >= 3 rows in the tile (the
row count at which a matrix tile pays, cf.
docs/experiments/QWEN36-MOE-GROUPED-ROWS-2026-10-21.md). No timing is taken.
Run only through the GPU lease (cpg_job -> run_with_gpu_locks).
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import subprocess
import time

PROMPTS = (
    ("prose", "Explain how a database index speeds up queries. Use one short paragraph."),
    ("code", "Write a Python function that merges two sorted lists into one sorted list, with a docstring."),
    ("math", "A train travels 180 km in 2.5 hours, then 120 km in 1.5 hours. What is its average speed? Show the steps."),
    ("chat", "I just moved to a new city and feel a bit lonely. Any practical ideas for meeting people?"),
)
PATTERNS = {"3x5": (5, 5, 5), "5x3": (3, 3, 3, 3, 3)}
POSITIONS = (0, 16, 40)


def tile_stats(rows_by_layer, top_k, experts, path_rows):
    """rows_by_layer: per layer, a list of per-row expert-index lists."""
    per_layer = []
    for rows in rows_by_layer:
        occupancy = Counter(e for row in rows for e in row)
        slots = sum(occupancy.values())
        path = {e for r in path_rows for e in rows[r]}
        per_layer.append({
            "distinct": len(occupancy),
            "distinct_primary_path": len(path),
            "experts_with_1_row": sum(1 for c in occupancy.values() if c == 1),
            "experts_with_2_rows": sum(1 for c in occupancy.values() if c == 2),
            "experts_with_3plus_rows": sum(1 for c in occupancy.values() if c >= 3),
            "slots_on_3plus_row_experts": sum(c for c in occupancy.values() if c >= 3) / slots,
            "max_rows_on_one_expert": max(occupancy.values()),
        })
    rows = len(rows_by_layer[0])
    mean = lambda key: sum(layer[key] for layer in per_layer) / len(per_layer)
    return {
        "rows": rows, "layers": len(per_layer),
        "slot_fetches_per_layer": rows * top_k,
        "independent_routing_bound": experts * (1 - (1 - top_k / experts) ** rows),
        "mean_distinct": mean("distinct"),
        "max_distinct": max(layer["distinct"] for layer in per_layer),
        "min_distinct": min(layer["distinct"] for layer in per_layer),
        "mean_distinct_primary_path": mean("distinct_primary_path"),
        "primary_path_rows": len(path_rows),
        "mean_experts_with_3plus_rows": mean("experts_with_3plus_rows"),
        "mean_slots_on_3plus_row_experts": mean("slots_on_3plus_row_experts"),
        "mean_max_rows_on_one_expert": mean("max_rows_on_one_expert"),
        "per_layer_distinct": [layer["distinct"] for layer in per_layer],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--greedy-tokens", type=int, default=48)
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
    report = {"schema": "mlx2.flash-next-tree-expert-union.v1", "source_revision": head,
              "checkpoint": f"{args.model.name}@sha256:{args.fingerprint}", "status": "running",
              "qualified": False, "defaulted": False, "timing": "none", "started_at": time.time(),
              "swap_before": subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True).strip(),
              "tiles": []}

    def save():
        tmp = args.out / "report.json.tmp"
        tmp.write_text(json.dumps(report, indent=1) + "\n")
        tmp.replace(args.out / "report.json")

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
        # Model modules only after the adapter pinned its import-time environment.
        from mlx2.adapters import flash_next_tree as T
        from mlx2.runtime.models import qwen3_next as M
        from mlx2.runtime.drafters.native_mtp import NativeMTPDraftModel
        report["ple_nvme"] = os.environ.get("MLX_QWEN4_PLE_NVME")
        if report["ple_nvme"] != str(args.model / "ple_rows.bin"):
            raise RuntimeError("PLE offload is not active")
        trunk = model.language_model.model
        block = next(layer.mlp for layer in trunk.layers if hasattr(layer.mlp, "switch_mlp"))
        top_k, experts = int(block.top_k), int(block.num_experts)
        report["routing"] = {"top_k": top_k, "experts": experts, "layers": len(trunk.layers)}
        layers = (len(trunk.layers) - 1,)
        provider = T.FlashNextTargetTreeExecutor(model)
        draft = NativeMTPDraftModel(model, num_draft=5, revision=adapter.identity["fingerprint"])

        captured = []
        stock = M._stock_routing

        def recording(block_, gates):
            inds, scores = stock(block_, gates)
            captured.append(np.asarray(inds.reshape(-1, inds.shape[-1])).tolist())
            return inds, scores

        for name, text in PROMPTS:
            prompt = list(adapter.prompt_tokens({"messages": [{"role": "user", "content": text}],
                                                 "enable_thinking": False}))
            cache = model.make_cache()
            logits = model(mx.array([prompt]), cache=cache)
            greedy = [int(mx.argmax(logits[0, -1]).item())]
            while len(greedy) < args.greedy_tokens:
                logits = model(mx.array([[greedy[-1]]]), cache=cache)
                greedy.append(int(mx.argmax(logits[0, -1]).item()))
            del cache
            for position in POSITIONS:
                context, anchor = prompt + greedy[:position], greedy[position]
                for label, pattern in PATTERNS.items():
                    base = model.make_cache()
                    features = model.prefill_body(mx.array([context]), base, layers)
                    mx.eval(features)
                    slate = draft.draft_packed_paths([anchor], features, draft.make_cache(), pattern,
                                                     context_tokens=[context[1:] + [anchor]])[0]
                    tokens, parents = [anchor], [-1]
                    for path in slate:
                        parent = 0
                        for token in path.tokens:
                            parents.append(parent)
                            parent = len(tokens)
                            tokens.append(token)
                    if len(tokens) != 16:
                        raise RuntimeError(f"{name}/{position}/{label}: {len(tokens)} rows, not 16")
                    captured.clear()
                    M._stock_routing = recording
                    try:
                        out_logits, out_features, transaction = provider.forward(model, tokens, parents, base, layers)
                        mx.eval(out_logits, out_features)
                    finally:
                        M._stock_routing = stock
                    transaction.abort()
                    if len(captured) != len(trunk.layers) or any(len(rows) != 16 for rows in captured):
                        raise RuntimeError("routing capture does not cover every layer and row")
                    primary = [0] + list(range(1, 1 + pattern[0]))
                    stats = tile_stats(captured, top_k, experts, primary)
                    report["tiles"].append({"prompt": name, "position": position, "pattern": label,
                                            "tokens": tokens, "parents": parents, **stats})
                    print(f"{name} pos{position} {label}: mean distinct {stats['mean_distinct']:.1f} "
                          f"(max {stats['max_distinct']}), path {stats['mean_distinct_primary_path']:.1f}, "
                          f"3+row slots {stats['mean_slots_on_3plus_row_experts']:.2f}", flush=True)
                    del base, features, transaction, out_logits, out_features
                    gc.collect()
                    mx.clear_cache()
                    save()
        by = lambda key, value: [t["mean_distinct"] for t in report["tiles"] if t[key] == value]
        report["summary"] = {
            "tiles": len(report["tiles"]),
            "one_row_distinct": top_k,
            "mean_distinct_all": sum(t["mean_distinct"] for t in report["tiles"]) / len(report["tiles"]),
            "mean_distinct_by_pattern": {p: sum(by("pattern", p)) / len(by("pattern", p)) for p in PATTERNS},
            "mean_distinct_by_prompt": {n: sum(by("prompt", n)) / len(by("prompt", n)) for n, _ in PROMPTS},
            "mean_slots_on_3plus_row_experts": sum(t["mean_slots_on_3plus_row_experts"] for t in report["tiles"]) / len(report["tiles"]),
            "independent_routing_bound_16_rows": report["tiles"][0]["independent_routing_bound"],
            "peak_mlx_bytes": mx.get_peak_memory(),
        }
        report["status"] = "passed"
    except BaseException as exc:
        report["status"], report["error"] = "failed", repr(exc)
        raise
    finally:
        if adapter is not None:
            adapter.close()
        report["swap_after"] = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True).strip()
        report["finished_at"] = time.time()
        save()


if __name__ == "__main__":
    main()

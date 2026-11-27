"""Native-MTP drafting cost for 1..5-draft chains on Flash-Next (diagnostic).

Times ``NativeMTPDraftModel.draft_packed_paths`` for single chains of 1..5
drafts and the 3x5 tree at a few decode positions, so a target-only AST tile
time can be turned into an end-to-end projection. Bounded smoke. GPU lease only.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

PROMPT = "A train travels 180 km in 2.5 hours, then 120 km in 1.5 hours. What is its average speed? Show the steps."
PATTERNS = {"1x1": (1,), "1x2": (2,), "1x3": (3,), "1x4": (4,), "1x5": (5,), "3x5": (5, 5, 5)}


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
    if args.out.exists():
        raise RuntimeError("refusing an existing evidence directory")
    args.out.mkdir(parents=True)
    report = {"schema": "mlx2.flash-next-mtp-draft-cost.v1", "source_revision": head,
              "checkpoint": f"{args.model.name}@sha256:{args.fingerprint}", "status": "running",
              "loadavg_before": subprocess.check_output(["sysctl", "-n", "vm.loadavg"], text=True).strip(),
              "draft_ms": {}}
    save = lambda: (args.out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    adapter = None
    try:
        from mlx2.adapters.registry import inspect_model
        if inspect_model(args.model).artifact["fingerprint"] != args.fingerprint:
            raise RuntimeError("checkpoint fingerprint changed")
        import mlx.core as mx
        mx.set_memory_limit(80 << 30)
        from mlx2.adapters.flash_next import FlashNextAdapter
        policy = {"draft_model": str(args.model), "num_draft": 5, "row_exact_verify": False,
                  "progressive_verification_tile": 2, "progressive_alternative_branches": 3,
                  "progressive_alternative_strategy": "confidence_tree", "cost_per_commit_handoff": False,
                  "continuation_pool": {"sources": ["external"], "limit": 3, "ast_planner": True,
                                        "packed_pattern": [5, 5, 5]},
                  "ast_target_tree_oracle": str(args.out / "unused-oracle.json")}
        adapter = FlashNextAdapter(str(args.model), execution_policy=policy)
        model = adapter.model
        from mlx2.runtime.drafters.native_mtp import NativeMTPDraftModel
        if os.environ.get("MLX_QWEN4_PLE_NVME") != str(args.model / "ple_rows.bin"):
            raise RuntimeError("PLE offload is not active")
        trunk = model.language_model.model
        layers = (len(trunk.layers) - 1,)
        draft = NativeMTPDraftModel(model, num_draft=5, revision=adapter.identity["fingerprint"])
        prompt = list(adapter.prompt_tokens({"messages": [{"role": "user", "content": PROMPT}],
                                             "enable_thinking": False}))
        cache = model.make_cache()
        logits = model(mx.array([prompt]), cache=cache)
        greedy = [int(mx.argmax(logits[0, -1]).item())]
        while len(greedy) < 48:
            logits = model(mx.array([[greedy[-1]]]), cache=cache)
            greedy.append(int(mx.argmax(logits[0, -1]).item()))
        samples = {k: [] for k in PATTERNS}
        for position in (8, 24, 40):
            context, anchor = prompt + greedy[:position], greedy[position]
            base = model.make_cache()
            features = model.prefill_body(mx.array([context]), base, layers)
            mx.eval(features)
            for name, pattern in PATTERNS.items():
                for rep in range(4):
                    mx.synchronize(); t0 = time.perf_counter()
                    slate = draft.draft_packed_paths([anchor], features, draft.make_cache(), pattern,
                                                     context_tokens=[context[1:] + [anchor]])[0]
                    mx.eval([mx.array(p.tokens) for p in slate])
                    mx.synchronize()
                    if rep:
                        samples[name].append((time.perf_counter() - t0) * 1000)
        for name, values in samples.items():
            report["draft_ms"][name] = round(statistics.median(values), 2)
        report["status"] = "passed"
        print(json.dumps(report["draft_ms"]), flush=True)
    except BaseException as exc:
        report["status"], report["error"] = "failed", repr(exc)
        raise
    finally:
        if adapter is not None:
            adapter.close()
        save()


if __name__ == "__main__":
    main()

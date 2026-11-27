"""GPU busy time vs wall time per Flash-Next AST tile (Metal System Trace).

Diagnostic only. For each arm (ordinary one-row decode, the exact tree law,
the full row-invariant law) it records one bounded ``xctrace`` Metal System
Trace attached to this process around N warm, eval-fenced repetitions and
reports the union of this pid's ``metal-gpu-intervals`` per repetition next
to the wall time. busy << wall means the host (graph build + encode) sets
the pace; busy ~ wall means the GPU does. Tracing adds overhead; compare arms
within the run. Run only through the GPU lease.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spec_round_phase0 import MetalTrace, export_gpu_intervals, gpu_busy  # noqa: E402

PROMPT = "Explain how a database index speeds up queries. Use one short paragraph."
REPEATS = 12


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
    report = {"schema": "mlx2.flash-next-tree-gpu-busy.v1", "source_revision": head,
              "checkpoint": f"{args.model.name}@sha256:{args.fingerprint}", "status": "running",
              "qualified": False, "repeats": REPEATS,
              "loadavg_before": subprocess.check_output(["sysctl", "-n", "vm.loadavg"], text=True).strip(),
              "arms": {}}

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
        while len(greedy) < 20:
            logits = model(mx.array([[greedy[-1]]]), cache=cache)
            greedy.append(int(mx.argmax(logits[0, -1]).item()))
        context, anchor = prompt + greedy[:16], greedy[16]
        base = model.make_cache()
        features = model.prefill_body(mx.array([context]), base, layers)
        mx.eval(features)
        slate = draft.draft_packed_paths([anchor], features, draft.make_cache(), (5, 5, 5),
                                         context_tokens=[context[1:] + [anchor]])[0]
        tokens, parents = [anchor], [-1]
        for path in slate:
            parent = 0
            for token in path.tokens:
                parents.append(parent)
                parent = len(tokens)
                tokens.append(token)
        exact = T.FlashNextTargetTreeExecutor(model)
        best = T.FlashNextTargetTreeExecutor(model, tile_law="row_invariant_mma",
                                             mma_parts=("projections", "experts_device", "qsa_chains",
                                                        "router_gates"))

        def tile(provider):
            def step():
                out = provider.forward(model, tokens, parents, base, layers)
                mx.eval(out[0], out[1])
                out[2].abort()
            return step

        state = {"token": greedy[-1]}

        def ordinary():
            logits = model(mx.array([[state["token"]]]), cache=cache)
            state["token"] = int(mx.argmax(logits[0, -1]).item())

        arms = {"ordinary_step": ordinary, "exact_scalar_tile": tile(exact), "row_invariant_tile": tile(best)}
        for name, step in arms.items():
            for _ in range(3):
                step()
            mx.synchronize()
            trace = MetalTrace(os.getpid(), args.out / f"{name}.trace", 60)
            if not trace.start():
                raise RuntimeError(f"xctrace did not start for {name}")
            time.sleep(0.5)
            walls = []
            start = time.perf_counter_ns()
            for _ in range(REPEATS):
                t0 = time.perf_counter_ns()
                step()
                mx.synchronize()
                walls.append((time.perf_counter_ns() - t0) / 1e6)
            window = (time.perf_counter_ns() - start) / 1e6
            time.sleep(0.5)
            code = trace.stop()
            report["arms"][name] = {"wall_ms_median": statistics.median(walls), "wall_ms": walls,
                                    "window_ms": window, "xctrace_returncode": code}
            save()
        for name in arms:
            rows = export_gpu_intervals(args.out / f"{name}.trace", args.out / f"{name}-gpu-intervals.xml")
            busy = gpu_busy(rows, os.getpid())
            arm = report["arms"][name]
            arm.update(busy, gpu_busy_ms_per_rep=busy["busy_ns"] / 1e6 / REPEATS)
            arm["gpu_busy_share_of_wall"] = arm["gpu_busy_ms_per_rep"] / (arm["window_ms"] / REPEATS)
            print(name, json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in arm.items()
                                    if k not in ("wall_ms",)}), flush=True)
            save()
        report["status"] = "passed"
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

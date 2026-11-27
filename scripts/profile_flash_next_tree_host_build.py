"""Host graph-build profile of one Flash-Next AST tree tile (diagnostic only).

Splits a warm tile into graph build (``provider.forward`` returning lazy
arrays) and GPU evaluation, and records a cProfile of the build, for the
exact tree law and the full row-invariant law. No qualification or
performance claim. Run only through the GPU lease.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
from pathlib import Path
import pstats
import statistics
import subprocess
import time

PROMPT = "Explain how a database index speeds up queries. Use one short paragraph."
REPEATS = 7


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
    report = {"schema": "mlx2.flash-next-tree-host-build.v1", "source_revision": head,
              "checkpoint": f"{args.model.name}@sha256:{args.fingerprint}", "status": "running",
              "qualified": False, "loadavg_before": subprocess.check_output(["sysctl", "-n", "vm.loadavg"], text=True).strip(),
              "laws": {}}

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
        laws = {"exact_scalar": T.FlashNextTargetTreeExecutor(model),
                "mma_device_qsa": T.FlashNextTargetTreeExecutor(
                    model, tile_law="row_invariant_mma",
                    mma_parts=("projections", "experts_device", "qsa_chains")),
                "mma_device_qsa_fused": T.FlashNextTargetTreeExecutor(
                    model, tile_law="row_invariant_mma",
                    mma_parts=("projections", "experts_device", "qsa_chains")),
                "mma_device_qsa_gates_fused": T.FlashNextTargetTreeExecutor(
                    model, tile_law="row_invariant_mma",
                    mma_parts=("projections", "experts_device", "qsa_chains", "router_gates")),
                "mma_device_qsa_gates_fused_padcache": T.FlashNextTargetTreeExecutor(
                    model, tile_law="row_invariant_mma",
                    mma_parts=("projections", "experts_device", "qsa_chains", "router_gates")),
                "mma_all_router_kernel_padcache": T.FlashNextTargetTreeExecutor(
                    model, tile_law="row_invariant_mma",
                    mma_parts=("projections", "experts_device", "qsa_chains", "router_gates", "router_kernel")),
                "mma_all_router_kernel_padcache_bm32": T.FlashNextTargetTreeExecutor(
                    model, tile_law="row_invariant_mma",
                    mma_parts=("projections", "experts_device", "qsa_chains", "router_gates", "router_kernel"))}
        flags = {"mma_device_qsa": (False, False), "mma_device_qsa_fused": (True, False),
                 "mma_device_qsa_gates_fused": (True, False),
                 "mma_device_qsa_gates_fused_padcache": (True, True),
                 "mma_all_router_kernel_padcache": (True, True),
                 "mma_all_router_kernel_padcache_bm32": (True, True)}
        from mlx2.runtime.models import flash_tree_mma_law as L
        draft = NativeMTPDraftModel(model, num_draft=5, revision=adapter.identity["fingerprint"])
        import re
        from collections import Counter

        def graph_ops(*arrays):
            path = args.out / "graph.dot"
            mx.export_to_dot(str(path), *arrays)
            labels = re.findall(r'label ="([^"]+)", shape=rectangle', path.read_text())
            path.unlink()
            return Counter(labels)

        prompt = list(adapter.prompt_tokens({"messages": [{"role": "user", "content": PROMPT}],
                                             "enable_thinking": False}))
        cache = model.make_cache()
        logits = model(mx.array([prompt]), cache=cache)
        greedy = [int(mx.argmax(logits[0, -1]).item())]
        ordinary = []
        while len(greedy) < 24:
            mx.synchronize()
            start = time.perf_counter()
            logits = model(mx.array([[greedy[-1]]]), cache=cache)
            greedy.append(int(mx.argmax(logits[0, -1]).item()))
            ordinary.append((time.perf_counter() - start) * 1000)
        report["ordinary_step_ms_median"] = statistics.median(ordinary[8:])
        from mlx2.runtime.models import qwen4_exp as QX
        # The served step async-evaluates every few layers (eager dispatch), so
        # its graph is counted with eager and early-PLE dispatch off.
        saved = (QX._EAGER_DISPATCH, QX._PLE_EARLY_DISPATCH)
        QX._EAGER_DISPATCH = QX._PLE_EARLY_DISPATCH = False
        try:
            step = model(mx.array([[greedy[-1]]]), cache=cache)
            report["ordinary_step_graph_ops"] = dict(graph_ops(step).most_common())
        finally:
            QX._EAGER_DISPATCH, QX._PLE_EARLY_DISPATCH = saved
        report["ordinary_step_eager_dispatch"] = {"eager": saved[0], "ple_early": saved[1],
                                                 "stride": QX._EAGER_DISPATCH_STRIDE}
        report["ordinary_step_graph_ops_total"] = sum(report["ordinary_step_graph_ops"].values())
        mx.eval(step)
        del cache
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
        def attribute(provider):
            """Graph nodes added by each stage of one tile build (node ids from
            mx.export_to_dot; a stage owns the nodes first reachable from its
            outputs). Slow, attribution only."""
            seen, per_stage, kinds = set(), Counter(), {}
            def nodes(arrays):
                path = args.out / "stage.dot"
                flat = [a for a in arrays if isinstance(a, mx.array)]
                if not flat:
                    return {}
                mx.export_to_dot(str(path), *flat)
                found = dict(re.findall(r'\{ (\d+) \[label ="([^"]+)", shape=rectangle', path.read_text()))
                path.unlink()
                return found
            def flatten(value):
                if isinstance(value, mx.array):
                    return [value]
                if isinstance(value, (tuple, list)):
                    return [a for v in value for a in flatten(v)]
                return []
            def wrap(module, attr, stage):
                original = getattr(module, attr)
                def wrapped(*a, **k):
                    out = original(*a, **k)
                    found = nodes(flatten(out))
                    new = {i: op for i, op in found.items() if i not in seen}
                    seen.update(new)
                    per_stage[stage] += len(new)
                    kinds.setdefault(stage, Counter()).update(new.values())
                    return out
                setattr(module, attr, wrapped)
                return original
            from mlx2.runtime.models import qwen4_exp as Q, qwen4_tree_gdn as G
            targets = [(T, "_ple", "ple"), (T, "_hc", "hc"), (T, "_project_law", "projections"),
                       (T, "_moe", "moe"), (Q, "_apply_inject", "inject"), (G, "tree", "gdn_tree"),
                       (T.FlashNextTargetTreeExecutor, "_qsa_chains", "qsa_chains")]
            saved = [(m, a, wrap(m, a, st)) for m, a, st in targets]
            try:
                out = provider.forward(model, tokens, parents, base, layers)
                total = nodes(flatten(out[:2]))
                per_stage["unattributed"] = len(set(total) - seen)
                mx.eval(out[0], out[1]); out[2].abort()
            finally:
                for m, a, original in saved:
                    setattr(m, a, original)
            return dict(per_stage.most_common()), {k: dict(v.most_common(8)) for k, v in kinds.items()}

        def fenced(provider):
            """Exclusive GPU wall per stage: every wrapped stage evaluates its
            outputs and synchronizes (fencing inflates totals; relative only).
            Unwrapped glue between stages is charged to the next stage."""
            times, stack = Counter(), []
            def flatten(value):
                if isinstance(value, mx.array):
                    return [value]
                if isinstance(value, (tuple, list)):
                    return [a for v in value for a in flatten(v)]
                return []
            def wrap(module, attr, stage):
                original = getattr(module, attr)
                def wrapped(*a, **k):
                    stack.append(0.0)
                    t0 = time.perf_counter()
                    out = original(*a, **k)
                    arrays = flatten(out)
                    if arrays:
                        mx.eval(arrays)
                    mx.synchronize()
                    total = time.perf_counter() - t0
                    child = stack.pop()
                    times[stage] += (total - child) * 1000
                    if stack:
                        stack[-1] += total
                    return out
                setattr(module, attr, wrapped)
                return original
            from mlx2.runtime.models import qwen4_exp as Q, qwen4_tree_gdn as G
            targets = [(T, "_ple", "ple"), (T, "_hc", "hc"), (T, "_project_law", "projections"),
                       (T, "_moe", "moe"), (Q, "_apply_inject", "inject"), (G, "tree", "gdn_tree"),
                       (T.FlashNextTargetTreeExecutor, "_qsa_chains", "qsa_chains"),
                       (Q.Attention, "__call__", "qsa_attention")]
            saved = [(m, a, wrap(m, a, st)) for m, a, st in targets]
            results = []
            try:
                for _ in range(3):
                    times.clear()
                    mx.synchronize()
                    t0 = time.perf_counter()
                    out = provider.forward(model, tokens, parents, base, layers)
                    mx.eval(out[0], out[1]); mx.synchronize(); out[2].abort()
                    results.append((dict(times), (time.perf_counter() - t0) * 1000))
            finally:
                for m, a, original in saved:
                    setattr(m, a, original)
            stage_ms, total = results[-1]
            return {k: round(v, 2) for k, v in sorted(stage_ms.items(), key=lambda kv: -kv[1])}, round(total, 1)

        def ablate(provider, reps=5):
            """Marginal unfenced cost per stage: record every stage's outputs
            once, then time the tile with one stage replaced by its recorded
            (already evaluated) outputs. baseline - ablated = that stage's
            cost, GPU and host together. Numerics of the ablated run are
            irrelevant (timing only)."""
            from mlx2.runtime.models import qwen4_exp as Q, qwen4_tree_gdn as G
            targets = {"ple": (T, "_ple"), "hc": (T, "_hc"), "projections": (T, "_project_law"),
                       "moe": (T, "_moe"), "inject": (Q, "_apply_inject"), "gdn_tree": (G, "tree"),
                       "qsa_chains": (T.FlashNextTargetTreeExecutor, "_qsa_chains"),
                       "qsa_attention": (Q.Attention, "__call__")}
            def flatten(value):
                if isinstance(value, mx.array):
                    return [value]
                if isinstance(value, (tuple, list)):
                    return [a for v in value for a in flatten(v)]
                return []
            recorded = {stage: [] for stage in targets}
            originals = {stage: getattr(m, attr) for stage, (m, attr) in targets.items()}
            def recorder(stage):
                def wrapped(*a, **k):
                    out = originals[stage](*a, **k)
                    recorded[stage].append(out)
                    return out
                return wrapped
            for stage, (m, attr) in targets.items():
                setattr(m, attr, recorder(stage))
            try:
                out = provider.forward(model, tokens, parents, base, layers)
                mx.eval([a for outs in recorded.values() for o in outs for a in flatten(o)] + [out[0], out[1]])
                out[2].abort()
            finally:
                for stage, (m, attr) in targets.items():
                    setattr(m, attr, originals[stage])
            def timed():
                samples = []
                for _ in range(reps):
                    mx.synchronize(); t0 = time.perf_counter()
                    o = provider.forward(model, tokens, parents, base, layers)
                    mx.eval(o[0], o[1]); mx.synchronize()
                    samples.append((time.perf_counter() - t0) * 1000)
                    o[2].abort()
                return statistics.median(samples)
            baseline = timed()
            result = {"baseline_ms": round(baseline, 1), "calls": {k: len(v) for k, v in recorded.items()}}
            for stage, (m, attr) in targets.items():
                if not recorded[stage]:
                    continue
                queue = list(recorded[stage])
                position = {"i": 0}
                def replay(*a, **k):
                    item = queue[position["i"] % len(queue)]
                    position["i"] += 1
                    return item
                setattr(m, attr, replay)
                try:
                    position["i"] = 0
                    ablated = timed()
                finally:
                    setattr(m, attr, originals[stage])
                result[stage] = round(baseline - ablated, 1)
            return result

        for name, provider in laws.items():
            L.FUSED_GLUE, L.CACHED_PADDING = flags.get(name, (False, False))
            L.EXPERT_PLAN = "bm32" if name.endswith("_bm32") else None
            try:
                out = provider.forward(model, tokens, parents, base, layers)
                mx.eval(out[0], out[1]); out[2].abort()
            except ValueError as exc:
                report["laws"][name] = {"declined": repr(exc)}
                print(name, "declined", exc, flush=True)
                save()
                continue
            for _ in range(2):  # warm kernels and caches
                out = provider.forward(model, tokens, parents, base, layers)
                mx.eval(out[0], out[1]); out[2].abort()
            build, evaluate, total = [], [], []
            for _ in range(REPEATS):
                mx.synchronize()
                t0 = time.perf_counter()
                out = provider.forward(model, tokens, parents, base, layers)
                t1 = time.perf_counter()
                mx.eval(out[0], out[1])
                mx.synchronize()
                t2 = time.perf_counter()
                out[2].abort()
                build.append((t1 - t0) * 1000); evaluate.append((t2 - t1) * 1000); total.append((t2 - t0) * 1000)
            out = provider.forward(model, tokens, parents, base, layers)
            report["laws"].setdefault(name, {})
            ops = graph_ops(out[0], out[1])
            mx.eval(out[0], out[1]); out[2].abort()
            profiler = cProfile.Profile()
            mx.synchronize()
            profiler.enable()
            out = provider.forward(model, tokens, parents, base, layers)
            profiler.disable()
            mx.eval(out[0], out[1]); out[2].abort()
            text = {}
            for key in ("cumulative", "tottime"):
                buf = io.StringIO()
                pstats.Stats(profiler, stream=buf).sort_stats(key).print_stats(45)
                text[key] = buf.getvalue()
            (args.out / f"profile-{name}.txt").write_text(text["cumulative"] + "\n\n" + text["tottime"])
            stage_ops = stage_kinds = None
            if name != "exact_scalar":
                stage_ops, stage_kinds = attribute(provider)
            fenced_stage_ms, fenced_total_ms = fenced(provider)
            ablation_ms = ablate(provider)
            print(name, "ablation", ablation_ms, flush=True)
            report["laws"][name] = {"ablation_ms": ablation_ms, "fenced_stage_ms": fenced_stage_ms, "fenced_total_ms": fenced_total_ms,
                                    "stage_graph_ops": stage_ops, "stage_graph_op_kinds": stage_kinds,
                                    "graph_ops_total": sum(ops.values()), "graph_ops": dict(ops.most_common()),
                                    "build_ms_median": statistics.median(build),
                                    "eval_ms_median": statistics.median(evaluate),
                                    "tile_ms_median": statistics.median(total),
                                    "build_ms": build, "eval_ms": evaluate}
            print(name, json.dumps({k: round(v, 1) for k, v in report["laws"][name].items()
                                    if k.endswith("median")}), flush=True)
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

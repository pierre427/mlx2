"""Row-invariant matrix-unit tile law vs the exact scalar tree law on Flash-Next.

Diagnostic prototype, never a serving, performance or qualification result.
For real native-MTP packed 16-row tiles it runs the default exact-scalar tree
law and the opt-in ``row_invariant_mma`` law (runtime/models/flash_tree_mma_law.py)
and records:

* row invariance of each law: a row must be bitwise identical whether the
  forward holds only the primary path, the first two paths, or the whole
  16-row tile (the tree kernels admit at most 16 rows) - the property that makes a route exact with
  itself across batch widths (AGENTS.md fast-route contract);
* agreement with the exact law: per-row argmax, max |logit| difference, and the
  greedy-accepted path of each law;
* warm tile wall time of each law next to one ordinary decode step (median of a
  few repeats, eval-fenced; bounded smoke timing, not publishable statistics).
Run only through the GPU lease (cpg_job -> run_with_gpu_locks).
"""
from __future__ import annotations

import argparse
import gc
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
PATTERNS = {"3x5": (5, 5, 5), "5x3": (3, 3, 3, 3, 3)}
POSITIONS = (8, 40)
REPEATS = 5


def tree(anchor, slates):
    tokens, parents = [anchor], [-1]
    for slate in slates:
        for path in slate:
            parent = 0
            for token in path.tokens:
                parents.append(parent)
                parent = len(tokens)
                tokens.append(token)
    return tokens, parents


def accepted(tokens, parents, argmax):
    """Greedy walk: emitted tokens (accepted drafts + the bonus) from row 0."""
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
    parser.add_argument("--greedy-tokens", type=int, default=48)
    parser.add_argument("--group-rows", type=int, nargs="*", default=[])
    parser.add_argument("--attribution", action="store_true",
                        help="also run projections-only, experts-only and host-sync-only arms")
    parser.add_argument("--prompts", type=int, default=len(PROMPTS))
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
    report = {"schema": "mlx2.flash-next-tree-mma-law.v1", "source_revision": head,
              "checkpoint": f"{args.model.name}@sha256:{args.fingerprint}", "status": "running",
              "qualified": False, "defaulted": False, "timing": "bounded smoke, eval-fenced median",
              "started_at": time.time(),
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
        from mlx2.runtime.models import flash_tree_mma_law as L
        from mlx2.runtime.drafters.native_mtp import NativeMTPDraftModel
        if os.environ.get("MLX_QWEN4_PLE_NVME") != str(args.model / "ple_rows.bin"):
            raise RuntimeError("PLE offload is not active")
        trunk = model.language_model.model
        layers = (len(trunk.layers) - 1,)
        laws = {"exact_scalar": T.FlashNextTargetTreeExecutor(model)}
        mma = lambda g, parts: T.FlashNextTargetTreeExecutor(model, tile_law="row_invariant_mma",
                                                             group_rows=g, mma_parts=parts)
        for g in args.group_rows:
            laws[f"mma_g{g}"] = mma(g, ("projections", "experts"))
        laws["mma_device"] = mma(16, ("projections", "experts_device"))
        laws["qsa_chains_only"] = mma(16, ("qsa_chains",))
        laws["mma_device_qsa"] = mma(16, ("projections", "experts_device", "qsa_chains"))
        laws["mma_device_qsa_fused"] = mma(16, ("projections", "experts_device", "qsa_chains"))
        laws["mma_device_qsa_fused_gates"] = mma(16, ("projections", "experts_device", "qsa_chains",
                                                       "router_gates"))
        laws["mma_device_qsa_fused_gates_padcache"] = mma(16, ("projections", "experts_device", "qsa_chains",
                                                                "router_gates"))
        laws["mma_all_router_kernel_padcache"] = mma(16, ("projections", "experts_device", "qsa_chains",
                                                           "router_gates", "router_kernel"))
        laws["mma_all_router_kernel_padcache_bm32"] = mma(16, ("projections", "experts_device", "qsa_chains",
                                                                "router_gates", "router_kernel"))
        laws["exact_scalar_gdn_chains"] = T.FlashNextTargetTreeExecutor(model, gdn_chains=True)
        laws["mma_all_gdn_chains"] = T.FlashNextTargetTreeExecutor(
            model, tile_law="row_invariant_mma", gdn_chains=True,
            mma_parts=("projections", "experts_device", "qsa_chains", "router_gates", "router_kernel"))
        laws["mma_all_hc"] = T.FlashNextTargetTreeExecutor(
            model, tile_law="row_invariant_mma",
            mma_parts=("projections", "experts_device", "qsa_chains", "router_gates", "router_kernel", "hc_rows"))
        laws["mma_all_hc_matrix"] = T.FlashNextTargetTreeExecutor(
            model, tile_law="row_invariant_mma",
            mma_parts=("projections", "experts_device", "qsa_chains", "router_gates", "router_kernel", "hc_matrix"))
        laws["exact_scalar_hc_multirow"] = T.FlashNextTargetTreeExecutor(model, hc_multirow=True)
        laws["mma_all_hc_multirow"] = T.FlashNextTargetTreeExecutor(
            model, tile_law="row_invariant_mma", hc_multirow=True,
            mma_parts=("projections", "experts_device", "qsa_chains", "router_gates", "router_kernel"))
        fused_glue = {"mma_device_qsa_fused", "mma_device_qsa_fused_gates", "mma_device_qsa_fused_gates_padcache",
                      "mma_all_router_kernel_padcache", "mma_all_router_kernel_padcache_bm32",
                      "mma_all_gdn_chains", "mma_all_hc", "mma_all_hc_multirow",
                      "mma_all_hc_matrix"}
        if args.attribution:
            laws["projections_only"] = mma(16, ("projections",))
            laws["sync_only"] = mma(16, ("sync_probe",))
            for g in args.group_rows:
                laws[f"experts_only_g{g}"] = mma(g, ("experts",))
        draft = NativeMTPDraftModel(model, num_draft=5, revision=adapter.identity["fingerprint"])

        def run(provider, tokens, parents, base):
            logits, features, transaction = provider.forward(model, tokens, parents, base, layers)
            mx.eval(logits, features)
            transaction.abort()
            return logits[0] if logits.ndim == 3 else logits

        def timed(provider, tokens, parents, base):
            run(provider, tokens, parents, base)  # warm
            samples = []
            for _ in range(REPEATS):
                mx.synchronize()
                start = time.perf_counter()
                run(provider, tokens, parents, base)
                mx.synchronize()
                samples.append((time.perf_counter() - start) * 1000)
            return statistics.median(samples), samples

        def commit_check(provider, tokens, parents, base, primary):
            """State handoff sanity, not exactness: commit the primary path, then
            feed the same next token to the prepared cache and to an ordinary
            replay of the path; report whether the next argmax agrees."""
            path = T.ancestry(parents)[primary - 1]
            logits, features, transaction = provider.forward(model, tokens, parents, base, layers)
            prepared = transaction.prepared_cache(path)
            transaction.abort()
            replay = [T._clone(item) for item in base]
            ref = model(mx.array([[tokens[row] for row in path]]), cache=replay)
            bonus = int(mx.argmax(ref[0, -1]).item())
            a = int(mx.argmax(model(mx.array([[bonus]]), cache=prepared)[0, -1]).item())
            b = int(mx.argmax(model(mx.array([[bonus]]), cache=replay)[0, -1]).item())
            return a == b

        declined = {}
        ordinary_ms = []
        for name, text in PROMPTS[:args.prompts]:
            prompt = list(adapter.prompt_tokens({"messages": [{"role": "user", "content": text}],
                                                 "enable_thinking": False}))
            cache = model.make_cache()
            logits = model(mx.array([prompt]), cache=cache)
            greedy = [int(mx.argmax(logits[0, -1]).item())]
            while len(greedy) < args.greedy_tokens:
                mx.synchronize()
                start = time.perf_counter()
                logits = model(mx.array([[greedy[-1]]]), cache=cache)
                greedy.append(int(mx.argmax(logits[0, -1]).item()))
                if len(greedy) > 8:
                    ordinary_ms.append((time.perf_counter() - start) * 1000)
            del cache
            for position in POSITIONS:
                context, anchor = prompt + greedy[:position], greedy[position]
                base = model.make_cache()
                features = model.prefill_body(mx.array([context]), base, layers)
                mx.eval(features)
                slates = {label: draft.draft_packed_paths([anchor], features, draft.make_cache(), pattern,
                                                          context_tokens=[context[1:] + [anchor]])[0]
                          for label, pattern in PATTERNS.items()}
                for label in PATTERNS:
                    tokens, parents = tree(anchor, [slates[label]])
                    if len(tokens) != 16:
                        raise RuntimeError(f"{name}/{position}/{label}: {len(tokens)} rows")
                    primary = 1 + PATTERNS[label][0]
                    two_paths = 1 + sum(PATTERNS[label][:2])
                    entry = {"prompt": name, "position": position, "pattern": label,
                             "tokens": tokens, "parents": parents, "laws": {}}
                    reference = None
                    for law, provider in laws.items():
                        L.FUSED_GLUE = law in fused_glue
                        L.CACHED_PADDING = (law.endswith("_padcache") or law.endswith("_bm32")
                                            or law in ("mma_all_gdn_chains", "mma_all_hc", "mma_all_hc_multirow",
                                                       "mma_all_hc_matrix"))
                        L.EXPERT_PLAN = "bm32" if law.endswith("_bm32") else None
                        L.status(reset=True)
                        if law in declined:
                            continue
                        try:
                            full = run(provider, tokens, parents, base)
                        except ValueError as exc:
                            declined[law] = repr(exc)
                            print(law, "declined", exc, flush=True)
                            continue
                        commit = commit_check(provider, tokens, parents, base, primary)
                        sub = run(provider, tokens[:primary], parents[:primary], base)
                        mid = run(provider, tokens[:two_paths], parents[:two_paths], base)
                        stats = L.status(reset=True)["counts"]
                        argmax = [int(v) for v in np.asarray(mx.argmax(full, axis=-1))]
                        median, samples = timed(provider, tokens, parents, base)
                        result = {
                            "rows_invariant_primary_vs_full": bool(mx.array_equal(sub, full[:primary]).item()),
                            "rows_invariant_two_paths_vs_full": bool(mx.array_equal(mid, full[:two_paths]).item()),
                            "rows_invariant_primary_vs_two_paths": bool(mx.array_equal(sub, mid[:primary]).item()),
                            "argmax": argmax,
                            "emitted": accepted(tokens, parents, argmax),
                            "tile_ms_median": median, "tile_ms_samples": samples,
                            "law_counts": stats,
                            "commit_state_matches_ordinary_replay": commit,
                        }
                        if reference is None:
                            reference = (full, argmax, result["emitted"])
                        else:
                            result["argmax_rows_differing_from_exact"] = sum(
                                a != b for a, b in zip(argmax, reference[1]))
                            result["max_abs_logit_diff_vs_exact"] = float(
                                mx.max(mx.abs(full.astype(mx.float32) - reference[0].astype(mx.float32))).item())
                            result["emitted_same_as_exact"] = result["emitted"] == reference[2]
                        entry["laws"][law] = result
                        print(f"{name} pos{position} {label} {law}: {median:.1f} ms, "
                              f"emitted {len(result['emitted'])}, invariant "
                              f"{result['rows_invariant_primary_vs_full']}/{result['rows_invariant_two_paths_vs_full']}",
                              flush=True)
                    report["tiles"].append(entry)
                    save()
                del base, features
                gc.collect()
                mx.clear_cache()
        ordinary = statistics.median(ordinary_ms)
        summary = {"ordinary_step_ms_median": ordinary, "laws": {}}
        summary["declined"] = declined
        for law in laws:
            if law in declined:
                continue
            rows = [t["laws"][law] for t in report["tiles"]]
            tile = statistics.median(r["tile_ms_median"] for r in rows)
            emitted = statistics.mean(len(r["emitted"]) for r in rows)
            summary["laws"][law] = {
                "tile_ms_median": tile, "tile_in_ordinary_steps": tile / ordinary,
                "mean_emitted_per_tile": emitted,
                "projected_tok_s": emitted / tile * 1000, "ordinary_tok_s": 1000 / ordinary,
                "all_rows_invariant": all(r["rows_invariant_primary_vs_full"] and r["rows_invariant_two_paths_vs_full"]
                                          and r["rows_invariant_primary_vs_two_paths"] for r in rows),
                "tiles_emitting_same_as_exact": sum(r.get("emitted_same_as_exact", True) for r in rows),
                "tiles": len(rows),
            }
        summary["peak_mlx_bytes"] = mx.get_peak_memory()
        report["summary"] = summary
        report["status"] = "passed"
        print(json.dumps(summary, indent=1), flush=True)
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

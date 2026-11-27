"""HC decode (two-launch, law 0) cost vs row count on real Flash-Next modules.

Diagnostic only. Times the 97 HC calls of one tree tile (every layer's attn
and mlp hyper-connections plus the final mixer) at 1, 2, 4, 8 and 16 rows,
chained in one eval as in a tile, and the composed row-invariant ``hc_rows``
at 16 rows. No qualification or performance claim. GPU lease only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time


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
    report = {"schema": "mlx2.flash-tree-hc-rows-bench.v1", "source_revision": head,
              "checkpoint": f"{args.model.name}@sha256:{args.fingerprint}", "status": "running",
              "loadavg_before": subprocess.check_output(["sysctl", "-n", "vm.loadavg"], text=True).strip(),
              "rows": {}}
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
        from mlx2.adapters import flash_next_tree as T
        from mlx2.runtime.models import flash_tree_mma_law as L
        trunk = model.language_model.model
        modules = []
        for layer in trunk.layers:
            modules += [layer.attn_hyper_connection, layer.mlp_hyper_connection]
        modules.append(trunk.hyper_connection_mixer)
        width = trunk.args.hc_count * trunk.args.hidden_size
        report["hc_calls"] = len(modules)

        def timed(fn, reps=7):
            samples = []
            for _ in range(reps + 1):
                mx.synchronize(); t0 = time.perf_counter()
                mx.eval(fn()); mx.synchronize()
                samples.append((time.perf_counter() - t0) * 1000)
            return statistics.median(samples[1:])

        for rows in (1, 2, 4, 8, 16):
            x = (mx.random.normal((1, rows, width)) * 0.5).astype(mx.bfloat16)
            mx.eval(x)
            def law0():
                outs = []
                for m in modules:
                    out = T._hc(m, x)
                    outs += list(out) if isinstance(out, tuple) else [out]
                return [o for o in outs if isinstance(o, mx.array) and o is not x]
            report["rows"][str(rows)] = {"law0_ms": timed(law0)}
            print(rows, report["rows"][str(rows)], flush=True)
            save()
        for rows in (2, 4, 8, 16):
            x = (mx.random.normal((1, rows, width)) * 0.5).astype(mx.bfloat16)
            mx.eval(x)
            def multirow():
                outs = []
                for m in modules:
                    out = T._hc(m, x, multirow=True)
                    outs += list(out) if isinstance(out, tuple) else [out]
                return [o for o in outs if isinstance(o, mx.array) and o is not x]
            def law0():
                outs = []
                for m in modules:
                    out = T._hc(m, x)
                    outs += list(out) if isinstance(out, tuple) else [out]
                return [o for o in outs if isinstance(o, mx.array) and o is not x]
            a_out, b_out = multirow(), law0()
            mx.eval(a_out, b_out)
            equal = all(bool(mx.array_equal(p, q).item()) for p, q in zip(a_out, b_out))
            report["rows"][str(rows)].update(multirow_ms=timed(multirow), multirow_bitwise_equal_law0=equal)
            print(rows, report["rows"][str(rows)], flush=True)
            save()
        for rows in (2, 4, 8, 16):
            x = (mx.random.normal((1, rows, width)) * 0.5).astype(mx.bfloat16)
            mx.eval(x)
            law = frozenset({"hc_matrix"})
            def matrix():
                outs = []
                for m in modules:
                    out = T._hc(m, x, law=law)
                    outs += list(out) if isinstance(out, tuple) else [out]
                return [o for o in outs if isinstance(o, mx.array) and o is not x]
            def law0():
                outs = []
                for m in modules:
                    out = T._hc(m, x)
                    outs += list(out) if isinstance(out, tuple) else [out]
                return [o for o in outs if isinstance(o, mx.array) and o is not x]
            a_out, b_out = matrix(), law0()
            mx.eval(a_out, b_out)
            diff = max(float(mx.max(mx.abs(p.astype(mx.float32) - q.astype(mx.float32))).item())
                       for p, q in zip(a_out, b_out))
            sub = (x[:, :1] if rows > 2 else None)
            report["rows"][str(rows)].update(matrix_ms=timed(matrix), matrix_max_abs_diff_vs_law0=diff)
            print(rows, report["rows"][str(rows)], flush=True)
            save()
        x = (mx.random.normal((1, 16, width)) * 0.5).astype(mx.bfloat16)
        mx.eval(x)
        L.FUSED_GLUE = L.CACHED_PADDING = True
        def composed():
            L.begin_forward()
            outs = []
            for m in modules:
                mixed, inject = L.hc_rows(m, x)
                outs += [mixed] + ([inject] if inject is not None else [])
            return outs
        report["hc_rows_16_ms"] = timed(composed)
        report["status"] = "passed"
        print(json.dumps(report, indent=1), flush=True)
    except BaseException as exc:
        report["status"], report["error"] = "failed", repr(exc)
        raise
    finally:
        if adapter is not None:
            adapter.close()
        save()


if __name__ == "__main__":
    main()

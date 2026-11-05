"""Go/no-go probe for compiling Inkling layer MLP halves under ``mx.compile``.

Research only; nothing in serving changes. On the real checkpoint with the
serving lane law and kernel policy, for every decoder layer and each row count,
builds the MLP half (``post_attention_layernorm`` then ``mlp.tree`` for packed
tree rows, or ``mlp`` for the ordinary one-row step) eagerly and under
``mx.compile``, and records:

- bit equality of the compiled output against eager (``mx.array_equal``),
- host graph-build time per call (eager build vs compiled call, no eval),
- device time per call (build + eval, synchronized).

The inputs are embeddings of random token ids, so magnitudes are realistic
without needing a prefilled cache (the MLP half reads no cache).
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import mlx.core as mx  # noqa: E402
from probe_inkling import host, install_serving_lane, ownership  # noqa: E402


def _median_ms(fn, repeats):
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1e3)
    return statistics.median(samples)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4, 6, 8, 9, 12, 16])
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="layer indices (default: all)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    from mlx2.adapters.inkling import InklingAdapter

    policy = json.loads(args.policy.read_text())
    report = {
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "ownership": ownership(), "before": host(), "policy": policy,
        "scope": "research probe: compiled vs eager MLP halves; no serving change",
        "cells": [],
    }
    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(1 << 30)
    adapter = InklingAdapter(str(args.model), execution_policy=policy)
    report["checkpoint"] = args.model.name + "@" + adapter.identity["fingerprint"]
    lane = install_serving_lane(adapter, args.model, "exact")
    report["lane_matmul"] = {"mode": "exact", "law_id": (lane or {}).get("law_id", "stock")}
    model = adapter.model
    model.configure_kernels(
        fused=policy.get("inkling_fused", False),
        down_combine=policy.get("inkling_q2_down_combine", False),
        attention=policy.get("inkling_attention", "reference"), grouped=False,
        tree_kernels=True, tree_cohort=True,
        shortconv_prefill=policy.get("inkling_shortconv_prefill", False))
    layers = list(range(len(model.layers))) if args.layers is None else args.layers
    vocab = int(model.config.vocab_size) if hasattr(model, "config") else 32000
    mx.random.seed(0)
    for rows in args.rows:
        tokens = mx.random.randint(0, min(vocab, 150000), (1, rows)).astype(mx.uint32)
        x = model.model.embed(tokens)
        mx.eval(x)
        for index in layers:
            layer = model.layers[index]
            sparse = hasattr(layer.mlp, "switch_mlp")
            kinds = ["tree", "call"] if sparse else ["call"]
            if rows > 8:
                kinds = [k for k in kinds if k == "tree"] or kinds
            for kind in kinds:
                if kind == "tree":
                    def eager(value, layer=layer):
                        return layer.mlp.tree(layer.post_attention_layernorm(value))
                else:
                    def eager(value, layer=layer):
                        return layer.mlp(layer.post_attention_layernorm(value))
                compiled = mx.compile(eager)
                reference = eager(x)
                mx.eval(reference)
                try:
                    candidate = compiled(x)
                    mx.eval(candidate)
                except Exception as error:  # record and continue
                    report["cells"].append({"layer": index, "rows": rows, "kind": kind,
                                            "error": repr(error)[:400]})
                    continue
                equal = bool(mx.array_equal(reference, candidate).item())
                diff = float(mx.max(mx.abs(reference.astype(mx.float32)
                                           - candidate.astype(mx.float32))).item())
                build_eager = _median_ms(lambda: eager(x), args.repeats)
                build_compiled = _median_ms(lambda: compiled(x), args.repeats)
                run_eager = _median_ms(lambda: mx.eval(eager(x)), args.repeats)
                run_compiled = _median_ms(lambda: mx.eval(compiled(x)), args.repeats)
                cell = {"layer": index, "rows": rows, "kind": kind, "sparse": sparse,
                        "bit_equal": equal, "max_abs_diff": diff,
                        "build_ms_eager": build_eager, "build_ms_compiled": build_compiled,
                        "run_ms_eager": run_eager, "run_ms_compiled": run_compiled}
                report["cells"].append(cell)
                print(json.dumps(cell), flush=True)
    cells = [c for c in report["cells"] if "error" not in c]
    summary = {}
    for kind in ("tree", "call"):
        for rows in args.rows:
            group = [c for c in cells if c["kind"] == kind and c["rows"] == rows]
            if not group:
                continue
            summary[f"{kind}_r{rows}"] = {
                "layers": len(group),
                "bit_equal_layers": sum(c["bit_equal"] for c in group),
                "build_ms_eager_sum": round(sum(c["build_ms_eager"] for c in group), 3),
                "build_ms_compiled_sum": round(sum(c["build_ms_compiled"] for c in group), 3),
                "run_ms_eager_sum": round(sum(c["run_ms_eager"] for c in group), 3),
                "run_ms_compiled_sum": round(sum(c["run_ms_compiled"] for c in group), 3),
            }
    report["summary"] = summary
    report["errors"] = [c for c in report["cells"] if "error" in c]
    report["after"] = host()
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({"summary": summary, "errors": len(report["errors"])}, indent=1))


if __name__ == "__main__":
    main()

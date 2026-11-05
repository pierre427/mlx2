"""Where the Inkling packed-tree forward spends host graph-build time.

Research only; nothing in serving changes. On a real prefilled prefix, runs
packed tree forwards through the serving executor and accumulates the host
(Python) time of each section while the graph is built lazily, then times the
evaluation separately. Sections are inclusive wall times of the wrapped calls;
nested sections are reported on their own lines, so they are not additive.
"""

from __future__ import annotations

import argparse
import functools
import json
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import mlx.core as mx  # noqa: E402
from probe_inkling import context_cases, host, install_serving_lane, ownership  # noqa: E402

TIMES = defaultdict(float)
CALLS = defaultdict(int)


def timed(name, fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        started = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            TIMES[name] += time.perf_counter() - started
            CALLS[name] += 1
    return wrapper


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--context", type=int, default=16384)
    ap.add_argument("--widths", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--repeats", type=int, default=12)
    ap.add_argument("--compare-gather", action="store_true",
                    help="also check chain-slice logits equal the per-node gather, bit for bit")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    from mlx2.adapters.inkling import InklingAdapter
    from mlx2.runtime.inkling_tree_executor import InklingTreeExecutor
    from mlx2.runtime.models.inkling import tree as tree_module
    from mlx2.runtime.models.inkling import tree_attention, ancestry

    policy = json.loads(args.policy.read_text())
    report = {"source_revision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "ownership": ownership(), "before": host(), "policy": policy,
        "scope": "research: host build attribution of the packed tree forward", "cells": []}
    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(1 << 30)
    adapter = InklingAdapter(str(args.model), execution_policy=policy)
    report["checkpoint"] = args.model.name + "@" + adapter.identity["fingerprint"]
    install_serving_lane(adapter, args.model, "exact")
    model = adapter.model
    model.configure_kernels(
        fused=policy.get("inkling_fused", False),
        down_combine=policy.get("inkling_q2_down_combine", False),
        attention=policy.get("inkling_attention", "reference"), grouped=False,
        tree_kernels=True, tree_cohort=True,
        shortconv_prefill=policy.get("inkling_shortconv_prefill", False))
    executor = InklingTreeExecutor.create()

    Tree = tree_module.InklingTree
    patches = [
        (Tree, "__init__", "tree_construct"),
        (Tree, "attention", "tree_attention_total"),
        (Tree, "convolution", "tree_convolution"),
        (Tree, "lend_global_kv", "tree_lend_global_kv"),
        (Tree, "forward_cohort", "forward_cohort_total"),
        (tree_attention, "attend", "mapped_attend"),
        (ancestry, "convolution", "ancestry_kernel_call"),
    ]
    for owner, attr, name in patches:
        original = owner.__dict__[attr] if isinstance(owner, type) else getattr(owner, attr)
        if isinstance(original, staticmethod):
            setattr(owner, attr, staticmethod(timed(name, original.__func__)))
        else:
            setattr(owner, attr, timed(name, original))
    for klass in {type(c) for c in model.make_cache()}:
        if hasattr(klass, "update_and_fetch"):
            setattr(klass, "update_and_fetch", timed(f"update_and_fetch:{klass.__name__}",
                                                     klass.update_and_fetch))
    for layer in model.layers:
        attn = layer.self_attn
        attn.project_qkvr = timed("project_qkvr", attn.project_qkvr)
        if hasattr(layer.mlp, "switch_mlp"):
            layer.mlp.tree = timed("mlp_tree", layer.mlp.tree)
    model.tap_outputs = timed("tap_outputs", model.tap_outputs)

    sub = argparse.Namespace(model=args.model, contexts=[args.context], context_tokens=0,
                             generation_tokens=256)
    cases = {k: v for k, v in context_cases(sub).items() if k.startswith("long_generation")}
    name, value = next(iter(cases.items()))
    prompt = adapter.prompt_tokens({"messages": [{"role": "user", "content": value[0]}]})
    cache = model.make_cache()
    for begin in range(0, len(prompt), 2048):
        model(mx.array([prompt[begin:begin + 2048]], mx.uint32), cache=cache)
        mx.eval([c.state for c in cache])
    vocab = len(adapter.tokenizer)
    if args.compare_gather:
        original_init = Tree.__init__
        for w in args.widths:
            proposed = [(prompt[-1] + 7 * i + 3) % vocab for i in range(w)]
            parents = [-1] + list(range(w - 1))
            sliced, _, tx = executor.forward(model, proposed, parents, cache, [1], cached=False)
            mx.eval(sliced)
            del tx

            def gather_only(self, *a, **k):
                original_init(self, *a, **k)
                self._chain_paths = [[-1]] * len(self._chain_paths)

            Tree.__init__ = gather_only
            try:
                gathered, _, tx = executor.forward(model, proposed, parents, cache, [1], cached=False)
                mx.eval(gathered)
                del tx
            finally:
                Tree.__init__ = original_init
            check = {"width": w, "bit_equal": bool(mx.array_equal(sliced, gathered).item()),
                     "max_abs_diff": float(mx.max(mx.abs(sliced.astype(mx.float32)
                                                         - gathered.astype(mx.float32))).item())}
            report.setdefault("gather_checks", []).append(check)
            print(json.dumps(check), flush=True)
    for w in args.widths:
        builds, evals = [], []
        for repeat in range(args.repeats + 2):
            TIMES.clear()
            CALLS.clear()
            proposed = [(prompt[-1] + 7 * i + 3) % vocab for i in range(w)]
            mx.synchronize()
            tic = time.perf_counter()
            values, _, tx = executor.forward(
                model, proposed, [-1] + list(range(w - 1)), cache, [1], cached=False)
            built = time.perf_counter()
            mx.eval(values)
            done = time.perf_counter()
            del values, tx
            if repeat >= 2:
                builds.append((built - tic) * 1e3)
                evals.append((done - built) * 1e3)
                for key, seconds in TIMES.items():
                    report.setdefault("_acc", {}).setdefault(f"w{w}:{key}", []).append(seconds * 1e3)
                for key, count in CALLS.items():
                    report.setdefault("_calls", {})[f"w{w}:{key}"] = count
        cell = {"width": w, "build_ms_median": statistics.median(builds),
                "eval_ms_median": statistics.median(evals),
                "sections_ms_median": {
                    key.split(":", 1)[1]: round(statistics.median(v), 3)
                    for key, v in report["_acc"].items() if key.startswith(f"w{w}:")},
                "calls": {key.split(":", 1)[1]: v for key, v in report["_calls"].items()
                          if key.startswith(f"w{w}:")}}
        report["cells"].append(cell)
        print(json.dumps(cell, indent=1), flush=True)
    report.pop("_acc", None)
    report.pop("_calls", None)
    report["prompt_tokens"] = len(prompt)
    report["after"] = host()
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    adapter.close()


if __name__ == "__main__":
    main()

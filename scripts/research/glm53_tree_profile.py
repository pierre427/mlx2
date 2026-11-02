"""GLM-5.3 AST tree-verify cost by width (direct model, one owned GPU window).

Prefills a realistic prompt, then times the layered (and optionally serial)
target tree executor on chain and bushy trees of W rows against W ordinary
single-token decode steps on the same cache. Every tree transaction is
aborted, so each measurement starts from the same committed frontier.

The executor's per-row stages are also counted (packed kernel calls by route)
so the cost per width can be split into shared and per-row work.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

from glm53_ordinary_profile import CHECKPOINT, MODEL, prompt_ids


def tree(kind, width):
    if kind == "chain":
        return [-1] + list(range(width - 1))
    # Bushy: a root with two children per node, breadth first.
    return [-1] + [(i - 1) // 2 for i in range(1, width)]


def tree_ablation(model, adapter, executor, cache, layers, inputs, parents, mx, reps):
    """Cumulative subtractive attribution of one tree forward (timing only)."""
    from glm53_ordinary_profile import tensor_kind

    import mlx2.adapters.glm53_layer_tree as tree_module
    import mlx2.runtime.models.glm53_gguf as module

    weights = adapter.weights
    tensors = weights.tensors
    saved = {(weights, k): getattr(weights, k) for k in ("linear", "grouped", "moe")}
    saved.update({(model, k): getattr(model, k) for k in ("_kda", "_mla")})
    for mod in (module, tree_module):
        for k in ("hc_collapse", "hc_expand", "rms", "gate_sigmoid"):
            if hasattr(mod, k):
                saved[(mod, k)] = getattr(mod, k)

    def zeros_linear(*kinds):
        def stub():
            inner = weights.linear

            def linear(name):
                if tensor_kind(name).startswith(kinds):
                    n = tensors[name].numpy_shape[-2]
                    return lambda x: mx.zeros((*x.shape[:-1], n), dtype=mx.float32)
                return inner(name)
            weights.linear = linear
        return stub

    def stub_moe():
        weights.moe = lambda x, ids_, scores, prefix: mx.zeros_like(x)

    def stub_grouped():
        def grouped(x, name, *, groups=8):
            shape = tensors[name].numpy_shape
            n = shape[-2] if len(shape) == 3 else shape[0] // groups
            return mx.zeros((*x.shape[:-1], n), dtype=mx.float32)
        weights.grouped = grouped

    def stub_cores():
        def core(x, state, prefix, **_):
            n = tensors[prefix + ".attn_output.weight"].numpy_shape[-1]
            return mx.zeros((*x.shape[:-1], n), dtype=mx.float32)
        model._kda = core
        model._mla = core

    def stub_hc():
        def collapse(x, w, prefix, eps, *, fused):
            lead = x.shape[:-2]
            return x[..., 0, :], mx.ones((*lead, 4)), mx.ones((*lead, 4, 4))
        for mod in (module, tree_module):
            mod.hc_collapse = collapse
            mod.hc_expand = lambda branch, residual, post, comb, *, fused: residual

    def stub_rms():
        for mod in (module, tree_module):
            mod.rms = lambda x, w, eps: x.astype(mx.float32)

    def stub_sigmoid():
        for mod in (module, tree_module):
            mod.gate_sigmoid = lambda x: x

    variants = [("full", None), ("-moe", stub_moe), ("-router", zeros_linear("router_proj")),
                ("-kda_proj", zeros_linear("kda_proj")), ("-mla_proj", zeros_linear("mla_proj", "indexer_proj")),
                ("-attn_output", zeros_linear("attn_output")), ("-shared", zeros_linear("shared_expert", "dense_ffn")),
                ("-lm_head", zeros_linear("output")), ("-grouped", stub_grouped), ("-hc_proj", zeros_linear("hc_proj")),
                ("-attn_cores", stub_cores), ("-hc", stub_hc), ("-rms", stub_rms), ("-sigmoid", stub_sigmoid)]
    rows, prev = [], None
    try:
        for label, stub in variants:
            if stub:
                stub()
            times = []
            for rep in range(reps + 1):
                t0 = time.perf_counter()
                out, features, transaction = executor.forward(model, inputs, parents, cache, layers)
                mx.eval(out, features)
                elapsed = time.perf_counter() - t0
                transaction.abort()
                if rep:
                    times.append(elapsed)
            t = sorted(times)[len(times) // 2]
            rows.append({"variant": label, "ms": t * 1e3, "drop_ms": None if prev is None else (prev - t) * 1e3})
            print(f"tree ablation W{len(inputs)} {label:14s} {t*1e3:8.2f} ms"
                  + ("" if prev is None else f"  drop {(prev - t)*1e3:7.2f} ms"), flush=True)
            prev = t
    finally:
        for (obj, k), v in saved.items():
            setattr(obj, k, v)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--widths", type=lambda s: [int(v) for v in s.split(",")], default=[1, 2, 4, 8, 16])
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument("--implementations", default="layered")
    parser.add_argument("--row-token-share", action="store_true",
                        help="research: token-shared chunk-law dense rows for multi-row calls")
    parser.add_argument("--ablate", type=int, default=0, help="also ablate one chain tree of this width")
    parser.add_argument("--eval-every", type=int, default=1, help="research: tree layer barrier interval")
    parser.add_argument("--no-expert-union", action="store_true", help="research: disable the union MoE")
    parser.add_argument("--source-revision", default="unrecorded")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("refusing to reuse a nonempty evidence directory")
    args.out.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx

    from mlx2.adapters.glm53_ast_target import Glm53TargetTreeExecutor
    from mlx2.adapters.glm53_gguf import Glm53GGUFAdapter

    info = mx.device_info()
    if info.get("max_recommended_working_set_size"):
        mx.set_wired_limit(info["max_recommended_working_set_size"])
    policy = json.loads(args.policy.read_text()) if args.policy else {}
    adapter = Glm53GGUFAdapter(MODEL, execution_policy=policy)
    model = adapter.model
    if args.row_token_share:
        adapter.weights.row_token_share = True
    if args.eval_every != 1 or not getattr(model, "tree_eval_every", None):
        model.tree_eval_every = args.eval_every
    if args.no_expert_union:
        adapter.weights.expert_union = False
    knobs = {"row_token_share": bool(getattr(adapter.weights, "row_token_share", False)),
             "expert_union": bool(getattr(adapter.weights, "expert_union", False)),
             "tree_eval_every": model.tree_eval_every}
    print("knobs", knobs, flush=True)
    ids = prompt_ids(adapter, args.tokens)
    cache = model.make_cache()
    for start in range(0, len(ids), args.chunk):
        prompt = mx.array([ids[start:start + args.chunk]])
        with model._prompt_scope(prompt.shape[1]):
            logits = model(prompt, cache, logits_to_keep=1)
        mx.eval(logits, [c.state for c in cache])
    first = int(mx.argmax(logits[0, -1]).item())
    layers = (len(model.layers) - 1,)
    record = {"schema": "mlx2.glm53-tree-profile.v1", "checkpoint": CHECKPOINT,
              "source_revision": args.source_revision, "host": platform.node(), "mlx": mx.__version__,
              "execution_policy": policy, "tokens": len(ids), "rows": []}

    # Ordinary decode reference: W steps on a cloned cache, one token at a time.
    from mlx2.adapters.glm53_ast_target import clone

    def decode_steps(width):
        private = clone(cache)
        token, times = first, []
        for _ in range(width):
            t0 = time.perf_counter()
            out = model(mx.array([[token]]), private, logits_to_keep=1)
            mx.eval(out, [c.state for c in private])
            times.append(time.perf_counter() - t0)
            token = int(mx.argmax(out[0, -1]).item())
        return sum(times)

    decode_steps(2)
    one = sorted(decode_steps(1) for _ in range(args.reps))[args.reps // 2]
    record["decode_step_seconds"] = one
    print(f"decode step {one*1e3:.2f} ms", flush=True)
    # Draft rows are the model's own greedy continuation (realistic routing).
    private, tokens = clone(cache), [first]
    for _ in range(max(args.widths + [args.ablate]) - 1):
        out = model(mx.array([[tokens[-1]]]), private, logits_to_keep=1)
        tokens.append(int(mx.argmax(out[0, -1]).item()))
    del private
    record["draft_tokens"] = tokens
    record["row_token_share"] = bool(args.row_token_share)
    record["knobs"] = knobs
    digests = {}
    for implementation in args.implementations.split(","):
        executor = Glm53TargetTreeExecutor(model, implementation=implementation)
        for kind in ("chain", "bushy"):
            for width in args.widths:
                if kind == "bushy" and width < 3:
                    continue
                parents = tree(kind, width)
                inputs = tokens[:width]
                weights = adapter.weights
                before = {k: dict(getattr(weights, k)) for k in
                          ("scheduled_dense_calls", "scheduled_grouped_calls", "scheduled_expert_calls")
                          if hasattr(weights, k)}
                times = []
                for rep in range(args.reps + 1):
                    t0 = time.perf_counter()
                    out, features, transaction = executor.forward(model, inputs, parents, cache, layers)
                    mx.eval(out, features)
                    elapsed = time.perf_counter() - t0
                    transaction.abort()
                    if rep:
                        times.append(elapsed)
                import hashlib

                import numpy as np

                digests[f"{implementation}-{kind}-{width}"] = hashlib.sha256(
                    np.array(out.astype(mx.float32)).tobytes()).hexdigest()
                calls = {}
                for k, b in before.items():
                    after = getattr(weights, k)
                    calls[k] = {r: (after[r] - b.get(r, 0)) // (args.reps + 1) for r in after if after[r] != b.get(r, 0)}
                t = sorted(times)[len(times) // 2]
                row = {"implementation": implementation, "tree": kind, "width": width, "seconds": t,
                       "per_row_ms": t * 1e3 / width, "vs_w_decode_steps": t / (width * one),
                       "vs_one_decode_step": t / one, "calls_per_forward": calls}
                record["rows"].append(row)
                print(f"{implementation:8s} {kind:5s} W{width:<3d} {t*1e3:8.2f} ms  {t*1e3/width:7.2f} ms/row  "
                      f"= {t/one:5.2f} decode steps ({t/(width*one):.2f} of W steps)", flush=True)
    record["logits_sha256"] = digests
    if args.ablate:
        executor = Glm53TargetTreeExecutor(model, implementation="layered")
        record["ablation"] = tree_ablation(model, adapter, executor, cache, layers, tokens[:args.ablate],
                                           tree("chain", args.ablate), mx, args.reps)
    (args.out / "tree_profile.json").write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()

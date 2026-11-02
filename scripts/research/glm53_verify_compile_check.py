"""GLM-5.3 bit gate for the AST verify compile levers (one owned GPU window).

Prefills a realistic prompt once, then checks on cloned caches:

* taps: ``--steps`` greedy one-token ``forward_with_taps`` decodes with
  ``glm53_compile_regions`` v1 (taps eager) against v1_taps (taps replayed);
  every step's logits and features, and the final caches, bit for bit.
* tree: layered tree forwards (chain and bushy, each ``--widths``) with
  ``glm53_tree_compile`` off against v1 (each topology twice, so the compiled
  arm traces and then replays); logits, features and every row's frontier
  state bit for bit, plus each tree's timing per arm.

Research only; the served route is unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from glm53_ordinary_profile import CHECKPOINT, MODEL, prompt_ids


def digest(arrays):
    import mlx.core as mx
    import numpy as np

    h = hashlib.sha256()
    for a in arrays:
        if a is None:
            h.update(b"none")
            continue
        a = mx.contiguous(a)
        mx.eval(a)
        h.update(str((a.shape, a.dtype)).encode())
        h.update(np.array(a.astype(mx.float32) if a.dtype == mx.bfloat16 else a).tobytes())
    return h.hexdigest()


def state_arrays(caches):
    out = []

    def walk(value):
        if isinstance(value, (tuple, list)):
            for v in value:
                walk(v)
        else:
            out.append(value)

    for c in caches:
        walk(c.state)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--widths", type=lambda s: [int(v) for v in s.split(",")], default=[1, 2, 3, 4, 6])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--tree-arms", type=json.loads,
                        default={"off": {"tree_compile": "off"}, "v1": {"tree_compile": "v1"}},
                        help="JSON {arm: {model attribute: value}}; every arm must equal the first")
    parser.add_argument("--skip-taps", action="store_true")
    parser.add_argument("--census", action="store_true",
                        help="also count dispatching graph nodes per tree arm/width (one graph, no barriers) "
                             "and of one eager ordinary step")
    parser.add_argument("--source-revision", default="unrecorded")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("refusing to reuse a nonempty evidence directory")
    args.out.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx

    from mlx2.adapters.glm53_ast_target import Glm53TargetTreeExecutor, clone
    from mlx2.adapters.glm53_gguf import Glm53GGUFAdapter

    info = mx.device_info()
    if info.get("max_recommended_working_set_size"):
        mx.set_wired_limit(info["max_recommended_working_set_size"])
    policy = json.loads(args.policy.read_text())
    adapter = Glm53GGUFAdapter(MODEL, execution_policy=policy)
    model = adapter.model
    model.enable_mtp()  # the AST route's mtp_seed feature (post-norm final tap)
    ids = prompt_ids(adapter, args.tokens)
    cache = model.make_cache()
    for start in range(0, len(ids), args.chunk):
        prompt = mx.array([ids[start:start + args.chunk]])
        with model._prompt_scope(prompt.shape[1]):
            logits = model(prompt, cache, logits_to_keep=1)
        mx.eval(logits, [c.state for c in cache])
    first = int(mx.argmax(logits[0, -1]).item())
    layers = (len(model.layers) - 1,)
    record = {"schema": "mlx2.glm53-verify-compile-check.v1", "checkpoint": CHECKPOINT,
              "source_revision": args.source_revision, "execution_policy": policy,
              "tokens": len(ids), "taps": {}, "tree": []}

    # --- taps: v1 (eager taps) vs v1_taps (compiled taps) -------------------
    from mlx2.runtime.models import glm53_compile_regions

    arms = {}
    for arm in (() if args.skip_taps else ("v1", "v1_taps")):
        model.compile_regions = arm
        private, token, steps, times = clone(cache), first, [], []
        for _ in range(args.steps):
            t0 = time.perf_counter()
            out, feat = model.forward_with_taps(mx.array([[token]]), private, layers)
            mx.eval(out, feat, [c.state for c in private])
            times.append(time.perf_counter() - t0)
            steps.append(digest([out, feat]))
            token = int(mx.argmax(out[0, -1]).item())
        arms[arm] = {"steps": steps, "final_state": digest(state_arrays(private)),
                     "step_ms_median": sorted(times)[len(times) // 2] * 1e3}
        del private
    if args.skip_taps:
        arms = {"v1": {"steps": [], "final_state": None, "step_ms_median": None},
                "v1_taps": {"steps": [], "final_state": None, "step_ms_median": None}}
    record["taps"] = {
        "steps": args.steps,
        "logits_features_equal": arms["v1"]["steps"] == arms["v1_taps"]["steps"],
        "final_state_equal": arms["v1"]["final_state"] == arms["v1_taps"]["final_state"],
        "step_ms_median": {k: v["step_ms_median"] for k, v in arms.items()},
        "compile_counters": dict(glm53_compile_regions._COUNTERS),
    }
    print("taps", json.dumps(record["taps"]), flush=True)
    model.compile_regions = policy.get("glm53_compile_regions", "v1")

    # --- tree: tree_compile off vs v1 ---------------------------------------
    private, tokens = clone(cache), [first]
    for _ in range(max(args.widths) - 1):
        out = model(mx.array([[tokens[-1]]]), private, logits_to_keep=1)
        tokens.append(int(mx.argmax(out[0, -1]).item()))
    del private
    executor = Glm53TargetTreeExecutor(model, implementation="layered")

    def tree(kind, width):
        if kind == "chain":
            return [-1] + list(range(width - 1))
        return [-1] + [(i - 1) // 2 for i in range(1, width)]

    arms = args.tree_arms
    record["tree_arms"] = arms
    ok = True
    for kind in ("chain", "bushy"):
        for width in args.widths:
            if kind == "bushy" and width < 3:
                continue
            parents = tree(kind, width)
            row = {"tree": kind, "width": width}
            results = {}
            for arm, settings in arms.items():
                for attr, value in settings.items():
                    setattr(model, attr, value)
                times, digests = [], set()
                for rep in range(args.reps + 2):
                    t0 = time.perf_counter()
                    out, feat, transaction = executor.forward(model, tokens[:width], parents, cache, layers)
                    mx.eval(out, feat)
                    times.append(time.perf_counter() - t0)
                    frontier = [digest(state_arrays(transaction.prepared_cache(
                        [i for i in _path(parents, r)]))) for r in range(width)]
                    digests.add((digest([out, feat]), tuple(frontier)))
                    transaction.abort()
                steady = sorted(times[2:])
                results[arm] = {"digests": digests, "ms": steady[len(steady) // 2] * 1e3}
            reference = next(iter(results.values()))["digests"]
            equal = all(r["digests"] == reference and len(r["digests"]) == 1 for r in results.values())
            ok &= equal
            row.update({"bitwise_equal": equal, "ms": {arm: r["ms"] for arm, r in results.items()}})
            record["tree"].append(row)
            print(f"tree {kind:5s} W{width:<2d} equal={equal} "
                  + "  ".join(f"{arm} {r['ms']:7.2f} ms" for arm, r in results.items()), flush=True)
    if args.census:
        record["census"] = census(args, model, executor, cache, tokens, first, layers, arms, tree)
    regions = model.__dict__.get("_tree_regions")
    record["tree_counters"] = dict(regions.counters) if regions is not None else {}
    record["tree_all_equal"] = ok
    (args.out / "verify_compile_check.json").write_text(json.dumps(record, indent=1) + "\n")
    print("ALL", ok and record["taps"]["logits_features_equal"] and record["taps"]["final_state_equal"], flush=True)


VIEWS = {"Reshape", "Broadcast", "ExpandDims", "Squeeze", "Flatten", "Unflatten", "Slice",
         "AsStrided", "Transpose", "StopGradient", "Depends", "Split", "Full"}


def _count(path, arrays):
    import collections
    import re

    import mlx.core as mx

    flat = []

    def walk(value):
        if isinstance(value, (tuple, list)):
            for v in value:
                walk(v)
        elif isinstance(value, mx.array):
            flat.append(value)

    walk(arrays)
    mx.export_to_dot(str(path), *flat)
    counts = collections.Counter(re.findall(r'label ="([^"]+)"', path.read_text()))
    dispatch = {k: v for k, v in counts.items() if k not in VIEWS}
    return {"dispatching": sum(dispatch.values()),
            "by_primitive": dict(sorted(dispatch.items(), key=lambda kv: -kv[1])[:20])}


def census(args, model, executor, cache, tokens, first, layers, arms, tree):
    """Dispatching nodes of one tree forward body (barrier-free, eager arms) and
    of one eager ordinary step (compile regions off): launches, not time."""
    import mlx.core as mx

    from mlx2.adapters.glm53_ast_target import clone

    out = {"ordinary_eager_step": None, "tree": []}
    saved_regions, saved_every = model.compile_regions, getattr(model, "tree_eval_every", 1)
    model.compile_regions = "off"
    private = clone(cache)
    logits = model(mx.array([[first]]), private, logits_to_keep=1)
    out["ordinary_eager_step"] = _count(args.out / "ordinary.dot", [logits])
    mx.eval(logits)
    del private
    real_eval = mx.eval
    for arm, settings in arms.items():
        if settings.get("tree_compile", "off") != "off":
            continue  # a compiled region is one graph node: not comparable
        for attr, value in settings.items():
            setattr(model, attr, value)
        for width in args.widths:
            captured = {}

            def eval_hook(*items, _captured=captured, _arm=arm, _width=width):
                if not _captured:
                    _captured.update(_count(args.out / f"tree-{_arm}-W{_width}.dot", list(items)))
                return real_eval(*items)

            model.tree_eval_every = len(model.layers)
            mx.eval = eval_hook
            try:
                _, _, transaction = executor.forward(model, tokens[:width], tree("chain", width), cache, layers)
            finally:
                mx.eval = real_eval
                model.tree_eval_every = saved_every
            transaction.abort()
            out["tree"].append({"arm": arm, "width": width, **captured})
            print(f"census {arm:16s} W{width:<2d} dispatching {captured.get('dispatching')}", flush=True)
    model.compile_regions = saved_regions
    print(f"census ordinary eager step dispatching {out['ordinary_eager_step']['dispatching']}", flush=True)
    return out


def _path(parents, row):
    path = []
    while row >= 0:
        path.append(row)
        row = parents[row]
    return path[::-1]


if __name__ == "__main__":
    main()

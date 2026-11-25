"""Price Inkling 2 decode by component in mlx2 and in Rapid-MLX, the same way for both.

Original mlx2 script (2026-11-25), the follow-up to
``docs/research/2026-11-25-inkling2-rapid-gap.md``: after ``head_kernel``
about 4 ms/token of GPU time separates mlx2's ordinary decode from
Rapid-MLX's on the same weights. Stage skipping cannot price Rapid-MLX
(its compiled blocks bake the traced graph), so this prices components by
replay instead:

1. After a ``--context``-token prefill and a warm-up, one real greedy decode
   step runs with the stack's component classes wrapped. Each call's inputs
   are recorded, keyed by (class, parameter shapes, input shapes) so sliding
   and full layers are separate keys, with the number of calls per step.
   Calls made inside another priced call are marked nested.
2. Each key is then replayed: its recorded inputs are evaluated, ``--reps``
   calls are built and evaluated together, and the elapsed time / reps is
   that component's GPU cost per call, with the host build off the critical
   path. Calls per step x cost per call = ms per token for that key.
3. ``step_gpu_ms``: ``--batch`` whole decode steps chained lazily and
   evaluated once (GPU cost of a step, as in ``inkling2_step_split.py``).
   ``unpriced_ms`` = step - sum of top-level components (embedding, head,
   sampling, glue between modules).

Replay appends extra rows to the attention caches (the context grows by
``reps`` rows for that layer); that does not change the cost at these sizes.
Sanity numbers, one process, no thermal hold; the caller owns the GPU.

Usage: <python> scripts/research/inkling2_component_bench.py --stack mlx2|rapid
       <model dir> <out.json> [--context 1024] [--reps 16] [--batch 8]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

ROOT = Path(__file__).resolve().parents[2]
CHUNK = 512
CLASSES = {
    "mlx2": ("mlx2.runtime.models.inkling2.language",
             ("Attention", "SparseMoE", "AttentionResidual", "MessageController",
              "ShortConvolution", "DynamicShortConvolution")),
    "rapid": ("rapid_mlx.models.inkling2.layers",
              ("SlidingAttention", "FullAttention", "SparseMoeBlock", "BlockAttentionResidual",
               "MessageGate", "MessageInput", "ShortConvolution", "DynamicShortConvolution")),
}


def corpus_ids(encode, n):
    parts = []
    for pattern in ("docs/**/*.md", "src/mlx2/**/*.py"):
        for path in sorted(ROOT.glob(pattern)):
            parts.append(path.read_text(errors="replace"))
            if len(parts) % 25 == 0:
                ids = encode("\n\n".join(parts))
                if len(ids) >= n:
                    return ids[:n]
    raise SystemExit("corpus too small")


def load(stack, model_dir):
    if stack == "mlx2":
        sys.path.insert(0, str(ROOT / "src"))
        from mlx2.adapters.inkling2 import Inkling2Adapter

        adapter = Inkling2Adapter(str(model_dir))
        tok = adapter.tokenizer
        return (adapter.model, adapter.model.make_cache,
                lambda t: tok.encode(t, add_special_tokens=False),
                {"checkpoint": adapter.checkpoint,
                 "kernel_policy": {k: v for k, v in adapter.model.kernel_policy.items() if v}})
    import rapid_mlx.models.inkling2 as vendored

    sys.modules["mlx_lm.models.inkling2_loop"] = vendored
    from mlx_lm import load as lm_load

    model, tok = lm_load(str(model_dir))
    import importlib.metadata as md

    return (model, model.make_cache, lambda t: tok.encode(t, add_special_tokens=False),
            {"rapid_mlx": md.version("rapid_mlx")})


def arrays(value):
    if isinstance(value, mx.array):
        return [value]
    if isinstance(value, (list, tuple)):
        return [a for v in value for a in arrays(v)]
    if isinstance(value, dict):
        return [a for v in value.values() for a in arrays(v)]
    return []


def shapes(value):
    return tuple(tuple(a.shape) for a in arrays(value))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", choices=tuple(CLASSES), required=True)
    ap.add_argument("model", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=16)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--kernels", default="",
                    help="mlx2 only: comma list name=true|false applied with configure_kernels")
    ap.add_argument("--cpu", action="store_true", help="plumbing check on the CPU (no timing value)")
    a = ap.parse_args()
    if a.cpu:
        mx.set_default_device(mx.cpu)
    model, make_cache, encode, meta = load(a.stack, a.model)
    if a.kernels:
        if a.stack != "mlx2":
            raise SystemExit("--kernels is for --stack mlx2")
        opts = {k: v.lower() == "true" for k, v in (kv.split("=") for kv in a.kernels.split(","))}
        model.configure_kernels(**{**model.kernel_policy, **opts})
        meta["kernels_override"] = opts
        meta["kernel_policy"] = {k: v for k, v in model.kernel_policy.items() if v}
    import importlib

    module = importlib.import_module(CLASSES[a.stack][0])
    classes = {name: getattr(module, name) for name in CLASSES[a.stack][1]}

    state = {"record": False, "depth": 0}
    seen = {}  # key -> {"name", "nested", "calls", "self", "args", "kwargs"}
    originals = {}

    def wrap(name, cls):
        original = cls.__call__
        originals[name] = original

        def call(self, *args, **kwargs):
            if not state["record"]:
                return original(self, *args, **kwargs)
            params = tuple(tuple(v.shape) for _, v in tree_flatten(self.parameters())[:4])
            key = (name, params, shapes(args), shapes(kwargs))
            entry = seen.get(key)
            if entry is None:
                try:
                    mx.eval(arrays(args), arrays(kwargs))
                    replayable = True
                except Exception:  # noqa: BLE001 - called inside mx.compile tracing
                    replayable = False
                entry = seen[key] = {"name": name, "nested": state["depth"] > 0, "calls": 0,
                                     "self": self, "args": args, "kwargs": kwargs,
                                     "replayable": replayable}
            entry["calls"] += 1
            state["depth"] += 1
            try:
                return original(self, *args, **kwargs)
            finally:
                state["depth"] -= 1

        cls.__call__ = call

    for name, cls in classes.items():
        wrap(name, cls)

    ids = corpus_ids(encode, a.context)
    cache = make_cache()
    for s in range(0, len(ids), CHUNK):
        logits = model(mx.array([ids[s:s + CHUNK]]), cache=cache)
        mx.eval(logits)
    y = mx.argmax(logits[:, -1, :], axis=-1)
    for _ in range(8):
        y = mx.argmax(model(y[:, None], cache=cache)[:, -1, :], axis=-1)
        mx.eval(y)

    state["record"] = True
    y = mx.argmax(model(y[:, None], cache=cache)[:, -1, :], axis=-1)
    mx.eval(y)
    state["record"] = False

    gpu = []
    for _ in range(4):
        t0 = time.perf_counter()
        z = y
        for _ in range(a.batch):
            z = mx.argmax(model(z[:, None], cache=cache)[:, -1, :], axis=-1)
        t1 = time.perf_counter()
        mx.eval(z)
        gpu.append((time.perf_counter() - t1) / a.batch)
        y = z
    step_ms = 1000 * statistics.median(gpu)

    rows = []
    for key, e in seen.items():
        fn = originals[e["name"]]
        if not e["replayable"]:
            rows.append({"component": e["name"], "nested": e["nested"],
                         "calls_per_step": e["calls"], "replayable": False,
                         "note": "called inside a compiled block; not priced (in unpriced_ms)",
                         "gpu_ms_per_token": 0.0})
            continue
        for _ in range(2):  # warm-up replays
            mx.eval(arrays(fn(e["self"], *e["args"], **e["kwargs"])))
        times = []
        for _ in range(3):
            outs = []
            t0 = time.perf_counter()
            for _ in range(a.reps):
                outs.append(fn(e["self"], *e["args"], **e["kwargs"]))
            t1 = time.perf_counter()
            mx.eval(arrays(outs))
            times.append(((time.perf_counter() - t1) / a.reps, (t1 - t0) / a.reps))
        per_call = 1000 * statistics.median(t[0] for t in times)
        build = 1000 * statistics.median(t[1] for t in times)
        rows.append({"component": e["name"], "nested": e["nested"], "calls_per_step": e["calls"],
                     "param_shapes": [list(s) for s in key[1]],
                     "gpu_ms_per_call": round(per_call, 4), "host_build_ms_per_call": round(build, 4),
                     "gpu_ms_per_token": round(per_call * e["calls"], 3)})
    top = sum(r["gpu_ms_per_token"] for r in rows if not r["nested"])
    by_component = {}
    for r in rows:
        k = r["component"] + (" (nested)" if r["nested"] else "")
        by_component[k] = round(by_component.get(k, 0) + r["gpu_ms_per_token"], 3)
    for name, cls in classes.items():
        cls.__call__ = originals[name]
    report = {"script": "scripts/research/inkling2_component_bench.py", "stack": a.stack,
              "model_dir": str(a.model), "context": a.context, "reps": a.reps,
              "kind": "sanity (no thermal hold; not publishable)", "mlx": mx.__version__, **meta,
              "step_gpu_ms": round(step_ms, 3), "priced_top_level_ms": round(top, 3),
              "unpriced_ms": round(step_ms - top, 3), "by_component_ms_per_token": by_component,
              "rows": rows}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=1, default=str))
    tmp.replace(a.out)
    print(json.dumps({k: report[k] for k in ("stack", "context", "step_gpu_ms", "priced_top_level_ms",
                                             "unpriced_ms", "by_component_ms_per_token")}, indent=1))


if __name__ == "__main__":
    main()

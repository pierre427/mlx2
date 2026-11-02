"""GLM-5.3 ordinary B1 profile: system peaks, normal timings, synchronized components.

One process, one owned GPU window, exact checkpoint. Three views, never mixed:

* ``system``: measured device peaks on this host -- streaming read bandwidth
  (one large reduction) and dense FP16/BF16 matmul throughput -- the roofline
  denominators next to Apple's published 614 GB/s.
* ``normal``: a realistic prompt prefilled in fixed chunks (each chunk
  evaluated) and greedy decode steps (each token evaluated). Only those
  fences; these are the numbers to compare with the roofline.
* ``components``: the same prompt and decode replayed on a fresh cache with
  an evaluation fence around every instrumented call. Times are exclusive
  (children subtracted), so they sum to the forward wall time; the fences
  inflate the total, so use them for attribution, not throughput. Weight calls
  carry their logical bytes (packed tensor bytes; routed experts count the
  distinct experts the call touched) and FLOPs (2 x rows x out x in), giving
  achieved GB/s and TFLOP/s per call class.

Not a qualification or a controlled performance run.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = "~/mlx-models/GLM-5.3-Flash-REAP50-Q3_K_M"
CHECKPOINT = (
    "GLM-5.3-Flash-REAP50-Q3_K_M.gguf@"
    "5a865918aea87a362ff8ae419581a0447929bfd68534a44d1125b394c272762d"
)


def vm_counters():
    """Host-wide page-in/swap counters (vm_stat pages), for paging attribution."""
    import re
    import subprocess

    text = subprocess.run(["/usr/bin/vm_stat"], capture_output=True, text=True, check=False).stdout
    want = {"Pageins": "pageins", "Swapins": "swapins", "Swapouts": "swapouts", "Decompressions": "decompressions"}
    out = {}
    for line in text.splitlines():
        key = line.split(":")[0]
        if key in want:
            out[want[key]] = int(re.sub(r"\D", "", line.split(":")[1]))
    return out


def counter_delta(before, after):
    return {k: after[k] - before.get(k, 0) for k in after}


def system_peaks(mx):
    """Best-of-N device peaks; warm-up first, every repetition evaluated."""
    result = {}
    # 4 GiB FP32 read: one reduction, output negligible.
    data = mx.random.uniform(shape=(1 << 30,), dtype=mx.float32)
    mx.eval(data)
    best = math.inf
    for _ in range(6):
        t0 = time.perf_counter()
        mx.eval(mx.sum(data))
        best = min(best, time.perf_counter() - t0)
    result["read_bandwidth_gb_s"] = data.nbytes / best / 1e9
    del data
    mx.clear_cache()
    for name, dtype in (("fp16", mx.float16), ("bf16", mx.bfloat16)):
        n = 8192
        a = mx.random.normal((n, n)).astype(dtype)
        b = mx.random.normal((n, n)).astype(dtype)
        mx.eval(a, b, a @ b)
        best = math.inf
        for _ in range(5):
            t0 = time.perf_counter()
            mx.eval(a @ b)
            best = min(best, time.perf_counter() - t0)
        result[f"matmul_{name}_tflop_s"] = 2 * n**3 / best / 1e12
        del a, b
        mx.clear_cache()
    return result


def prompt_ids(adapter, tokens):
    sys.path.insert(0, str(ROOT / "scripts"))
    import realistic_haystack

    question = "\nIn one sentence, what is the main subject of the material above?"
    render = lambda text: adapter.prompt_tokens(
        {"messages": [{"role": "user", "content": text}], "enable_thinking": False}
    )
    ids = render(realistic_haystack.haystack(tokens * 6) + question)
    tail = render(question)
    return ids[: tokens - len(tail)] + tail


def top_window(row, mx, k=5):
    """Top-k (token, logprob) at one position: first-divergence evidence."""
    row = row.astype(mx.float32)
    logprobs = row - mx.logsumexp(row)
    order = mx.argsort(-logprobs)[:k]
    return [[int(i), float(logprobs[i].item())] for i in order.tolist()]


def run_pass(model, ids, chunk, decode, mx, on_phase=None, window=5):
    """Chunked prefill + greedy decode; ``run_pass.windows`` holds each step's
    bounded top-``window`` (token, logprob) list (first-divergence evidence),
    or nothing when ``window`` is 0 (ids-only timing pass). ``run_pass.finite``
    says whether every emitted logits row was finite (reduced lazily, no
    extra synchronization)."""
    cache = model.make_cache()
    chunks, steps = [], []
    logits = None
    for begin in range(0, len(ids), chunk):
        piece = mx.array([ids[begin : begin + chunk]])
        if on_phase:
            on_phase("prefill")
        t0 = time.perf_counter()
        with model._prompt_scope(piece.shape[1]):
            logits = model(piece, cache, logits_to_keep=1)
        mx.eval(logits, [c.state for c in cache])
        chunks.append(time.perf_counter() - t0)
    finite = mx.all(mx.isfinite(logits[0, -1]))
    token = int(mx.argmax(logits[0, -1]).item())
    out = [token]
    windows = [top_window(logits[0, -1], mx, window)] if window else []
    import hashlib

    import numpy as np

    run_pass.prefill_logits_sha256 = hashlib.sha256(
        np.array(logits[0, -1].astype(mx.float32)).tobytes()).hexdigest()
    for _ in range(decode):
        if on_phase:
            on_phase("decode")
        t0 = time.perf_counter()
        logits = model(mx.array([[token]]), cache, logits_to_keep=1)
        finite = finite & mx.all(mx.isfinite(logits[0, -1]))
        token = int(mx.argmax(logits[0, -1]).item())
        steps.append(time.perf_counter() - t0)
        out.append(token)
        if window:
            windows.append(top_window(logits[0, -1], mx, window))
    run_pass.windows = windows
    run_pass.finite = bool(finite.item())
    return chunks, steps, out


def kind_split(model, ids, chunk, steps, mx):
    """Per-layer-kind decode attribution (design 8.3): KDA, MLA and MLP blocks.

    Profiling mode only. The prompt is prefilled as served (chunks, no
    fences); then ``steps`` greedy decode steps (after one warm step) run with
    an evaluation fence around each ``_kda`` / ``_mla`` / ``_mlp`` call: its
    inputs and the pending upstream work are evaluated before the clock
    starts, its output and its cache state before it stops. ``other`` is the
    step wall time minus the three kinds (HC, norms, embedding, head and the
    fences themselves). Use the split for attribution, not throughput.
    """
    cache = model.make_cache()
    logits = None
    for begin in range(0, len(ids), chunk):
        piece = mx.array([ids[begin : begin + chunk]])
        with model._prompt_scope(piece.shape[1]):
            logits = model(piece, cache, logits_to_keep=1)
        mx.eval(logits, [c.state for c in cache])
    token = int(mx.argmax(logits[0, -1]).item())
    kinds = {"_kda": "kda", "_mla": "mla", "_mlp": "mlp"}
    seconds = {name: 0.0 for name in kinds.values()}
    calls = {name: 0 for name in kinds.values()}
    original = {name: getattr(model, name) for name in kinds}
    own = {name: model.__dict__[name] for name in kinds if name in model.__dict__}
    live = {"on": False}

    def fenced(name, fn):
        def run(*args, **kwargs):
            if not live["on"]:
                return fn(*args, **kwargs)
            mx.eval([a for a in args if isinstance(a, mx.array)])
            caches = [a for a in args if hasattr(a, "state") and not isinstance(a, mx.array)]
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            # The state the call wrote (its arrays are replaced during the call).
            mx.eval(out, [c.state for c in caches])
            seconds[kinds[name]] += time.perf_counter() - t0
            calls[kinds[name]] += 1
            return out
        return run

    walls = []
    try:
        for name, fn in original.items():
            setattr(model, name, fenced(name, fn))
        for step in range(steps + 1):
            live["on"] = step > 0
            t0 = time.perf_counter()
            logits = model(mx.array([[token]]), cache, logits_to_keep=1)
            mx.eval(logits, [c.state for c in cache])
            if step:
                walls.append(time.perf_counter() - t0)
            token = int(mx.argmax(logits[0, -1]).item())
    finally:
        for name in original:
            # Restore an earlier instance wrapper, else fall back to the class method.
            if name in own:
                setattr(model, name, own[name])
            else:
                model.__dict__.pop(name, None)
    n = max(1, len(walls))
    per = {name: 1e3 * value / n for name, value in seconds.items()}
    step_ms = 1e3 * sum(walls) / n
    per["other"] = step_ms - sum(per.values())
    return {"context": len(ids), "steps": len(walls), "step_ms": step_ms, "ms_per_token": per,
            "calls_per_token": {k: v / n for k, v in calls.items()},
            "note": "fenced attribution; fences inflate step_ms"}


class Profiler:
    """Exclusive nested timing with evaluation fences at every boundary."""

    def __init__(self, mx):
        self.mx = mx
        self.phase = "setup"
        self.stack = []
        self.rows = defaultdict(lambda: {"calls": 0, "seconds": 0.0, "bytes": 0.0, "flops": 0.0,
                                         "primitives": 0, "primitive_kinds": defaultdict(int)})
        self.census_dir = None  # set to a directory to count each call's new primitives

    def _arrays(self, value):
        mx = self.mx
        if isinstance(value, mx.array):
            return [value]
        if isinstance(value, (list, tuple)):
            return [a for v in value for a in self._arrays(v)]
        if isinstance(value, dict):
            return [a for v in value.values() for a in self._arrays(v)]
        return []

    def call(self, label, fn, args, kwargs, cost=None):
        mx = self.mx
        mx.eval(self._arrays(args) + self._arrays(kwargs))  # pending input work stays with the parent
        self.stack.append(0.0)
        t0 = time.perf_counter()
        try:
            out = fn(*args, **kwargs)
            if self.census_dir is not None and self._arrays(out):
                self._census(label, out)
            mx.eval(self._arrays(out))
        finally:
            elapsed = time.perf_counter() - t0
            children = self.stack.pop()
            if self.stack:
                self.stack[-1] += elapsed
        row = self.rows[(self.phase, label)]
        row["calls"] += 1
        row["seconds"] += elapsed - children
        if cost is not None:
            nbytes, flops = cost(args, out)
            row["bytes"] += nbytes
            row["flops"] += flops
        return out

    def _census(self, label, out):
        """Primitives this call adds (inputs were evaluated before it ran)."""
        import re

        path = self.census_dir / "call.dot"
        self.mx.export_to_dot(str(path), *self._arrays(out))
        kinds = re.findall(r'label\s*=\s*"([^"]+)"', path.read_text())
        row = self.rows[(self.phase, label)]
        for kind in kinds:
            if kind[:1].isdigit():
                continue
            kind = re.sub(r"\s.*|\(.*", "", kind)
            row["primitive_kinds"][kind] += 1
            row["primitives"] += 1

    def wrap(self, label, fn, cost=None):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            return self.call(label, fn, args, kwargs, cost)

        return wrapped


def tensor_kind(name):
    s = name.split(".", 2)[-1] if name.startswith("blk.") else name
    if s.startswith("ssm_") or s in ("attn_q.weight", "attn_k.weight", "attn_v.weight"):
        return "kda_proj:" + s.split(".")[0]
    if s.startswith("indexer"):
        return "indexer_proj"
    if "shexp" in s:
        return "shared_expert"
    if s.startswith("ffn_"):
        return "dense_ffn" if s.split(".")[0] in ("ffn_gate", "ffn_up", "ffn_down") else "router_proj"
    if s.startswith("attn_output"):
        return "attn_output"
    if s.startswith("attn"):
        return "mla_proj:" + s.split(".")[0]
    if s.startswith("hc_"):
        return "hc_proj"
    return s.split(".")[0]


def ablation(model, weights, ids, chunk, mx, steps=6, mode="decode"):
    """Cumulative in-graph subtractive attribution of a decode step, no fences.

    Each variant replaces one more category with shape-correct trivial stubs
    (values become garbage; only timing is used) and times decode-step
    evaluation from a fresh real prefill. The drop between consecutive
    variants is that category's real encode+GPU cost per token.
    """
    import mlx2.runtime.models.glm53_gguf as module

    tensors = weights.tensors
    originals = {
        "linear": weights.linear, "grouped": weights.grouped, "moe": weights.moe,
        "hc_collapse": module.hc_collapse, "hc_expand": module.hc_expand,
        "rms": module.rms, "gate_sigmoid": module.gate_sigmoid,
        "multi_linear": getattr(weights, "multi_linear", None),
        "moe_with_shared": getattr(weights, "moe_with_shared", None),
    }

    def unfused_fallbacks():
        weights.multi_linear = lambda *a, **k: None
        weights.moe_with_shared = lambda *a, **k: None

    def stub_kinds(*kinds):
        """Stub only the linear projections whose tensor kind starts with one of kinds."""
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

    def stub_weights():
        def linear(name):
            n = tensors[name].numpy_shape[-2]
            return lambda x: mx.zeros((*x.shape[:-1], n), dtype=mx.float32)

        def grouped(x, name, *, groups=8):
            n = tensors[name].numpy_shape[-2] if len(tensors[name].numpy_shape) == 3 else tensors[name].numpy_shape[0] // groups
            return mx.zeros((*x.shape[:-1], n), dtype=mx.float32)

        weights.linear = linear
        weights.grouped = grouped
        weights.moe = lambda x, ids_, scores, prefix: mx.zeros_like(x)

    def stub_hc():
        def collapse(x, w, prefix, eps, *, fused):
            lead = x.shape[:-2]
            return x[..., 0, :], mx.ones((*lead, 4)), mx.ones((*lead, 4, 4))

        module.hc_collapse = collapse
        module.hc_expand = lambda branch, residual, post, comb, *, fused: residual

    def stub_rms():
        module.rms = lambda x, w, eps: x.astype(mx.float32)

    def stub_sigmoid():
        module.gate_sigmoid = lambda x: x

    variants = [("full", []), ("unfused", [unfused_fallbacks]), ("-moe", [stub_moe]), ("-lm_head", [stub_kinds("output")]),
                ("-kda_proj", [stub_kinds("kda_proj")]), ("-attn", [stub_kinds("attn_output", "mla_proj")]),
                ("-shared", [stub_kinds("shared_expert", "dense_ffn")]),
                ("-weights", [stub_weights]), ("-hc", [stub_hc]),
                ("-rms", [stub_rms]), ("-sigmoid", [stub_sigmoid])]
    applied, rows = [], []
    try:
        for label, stubs in variants:
            for stub in stubs:
                stub()
            applied.append(label)
            cache = model.make_cache()
            prompt = mx.array([ids[:chunk]])
            if mode == "prefill":
                times = []
                for _ in range(3):
                    cache = model.make_cache()
                    with model._prompt_scope(prompt.shape[1]):
                        logits = model(prompt, cache, logits_to_keep=1)
                    t0 = time.perf_counter()
                    mx.eval(logits, [c.state for c in cache])
                    times.append(time.perf_counter() - t0)
                rows.append({"variant": "+".join(applied), "eval_ms": 1e3 * sorted(times)[1]})
                print(f"prefill ablation {rows[-1]['variant']:40s} {rows[-1]['eval_ms']:8.1f} ms / {chunk} rows", flush=True)
                continue
            # Real prefill (weights restored for it), then the stubbed decode.
            names = ("linear", "grouped", "moe", "multi_linear", "moe_with_shared")
            saved = {k: getattr(weights, k) for k in names}
            for k in names:
                setattr(weights, k, originals[k])
            mod_saved = {k: getattr(module, k) for k in ("hc_collapse", "hc_expand", "rms", "gate_sigmoid")}
            for k in mod_saved:
                setattr(module, k, originals[k])
            with model._prompt_scope(prompt.shape[1]):
                mx.eval(model(prompt, cache, logits_to_keep=1), [c.state for c in cache])
            for k, v in saved.items():
                setattr(weights, k, v)
            for k, v in mod_saved.items():
                setattr(module, k, v)
            token, times = ids[:chunk][-1], []
            for step in range(steps + 1):
                logits = model(mx.array([[token]]), cache, logits_to_keep=1)
                t0 = time.perf_counter()
                mx.eval(logits, [c.state for c in cache])
                if step:
                    times.append(time.perf_counter() - t0)
            rows.append({"variant": "+".join(applied), "eval_ms": 1e3 * sorted(times)[len(times) // 2]})
            print(f"ablation {rows[-1]['variant']:40s} {rows[-1]['eval_ms']:7.2f} ms", flush=True)
    finally:
        for k in ("linear", "grouped", "moe", "multi_linear", "moe_with_shared"):
            setattr(weights, k, originals[k])
        for k in ("hc_collapse", "hc_expand", "rms", "gate_sigmoid"):
            setattr(module, k, originals[k])
    return rows


def layer_census(model, ids, chunk, out_dir, mx):
    """Primitives one decode step adds per layer type (un-instrumented, real cache)."""
    import collections
    import re

    import mlx2.runtime.models.glm53_gguf as module

    cache = model.make_cache()
    with model._prompt_scope(len(ids[:chunk])):
        mx.eval(model(mx.array([ids[:chunk]]), cache, logits_to_keep=1), [c.state for c in cache])
    h = mx.random.normal((1, 1, 4, 4096)).astype(mx.float32)
    x = mx.random.normal((1, 1, 4096)).astype(mx.float32)
    mx.eval(h, x)
    kda = next(i for i, layer in enumerate(model.layers) if layer.is_linear)
    mla = next(i for i, layer in enumerate(model.layers) if not layer.is_linear)
    eps = model.args.rms_norm_eps
    calls = {
        "kda_layer(_kda)": lambda: model._kda(x, cache[kda], f"blk.{kda}"),
        "mla_layer(_mla)": lambda: model._mla(x, cache[mla], f"blk.{mla}"),
        "moe_mlp(_mlp, layer>=3)": lambda: model._mlp(x, "blk.5", 5),
        "dense_mlp(_mlp, layer<3)": lambda: model._mlp(x, "blk.0", 0),
        "hc_collapse": lambda: module.hc_collapse(h, model.weights, "blk.5.hc_attn", eps, fused=model.hc_fused)[0],
        "hc_expand": lambda: module.hc_expand(x, h, mx.ones((1, 1, 4)), mx.ones((1, 1, 4, 4)), fused=model.hc_fused),
        "rms": lambda: module.rms(x, model.weights.array("blk.5.attn_norm.weight"), eps),
    }
    result = {}
    for label, fn in calls.items():
        outs = fn()
        outs = [o for o in (outs if isinstance(outs, (list, tuple)) else [outs]) if isinstance(o, mx.array)]
        path = out_dir / "layer.dot"
        states = [c.state for c in cache] if "layer" in label else []
        mx.export_to_dot(str(path), *outs, *states)
        kinds = [re.sub(r"\s.*|\(.*", "", k) for k in re.findall(r'label\s*=\s*"([^"]+)"', path.read_text())
                 if not k[:1].isdigit()]
        census = collections.Counter(kinds)
        views = sum(census[k] for k in ("Reshape", "Broadcast", "Flatten", "Unflatten", "ExpandDims", "Slice"))
        result[label] = {"primitives": sum(census.values()), "non_view": sum(census.values()) - views,
                         "kinds": dict(census.most_common())}
        mx.eval(outs, states)
        print(f"layer census {label:26s} {result[label]['primitives']:5d} primitives, {result[label]['non_view']:5d} non-view  "
              f"{census.most_common(10)}", flush=True)
    return result


def instrument(model, weights, profiler, mx):
    import mlx2.runtime.models.glm53_gguf as module

    tensors = weights.tensors
    rows_of = lambda x: math.prod(x.shape[:-1])
    original_linear = weights.linear

    def linear(name):
        fn = original_linear(name)
        t = tensors[name]
        n, k = t.numpy_shape[-2], t.numpy_shape[-1]

        def cost(args, out):
            return t.nbytes, 2.0 * rows_of(args[0]) * n * k

        return profiler.wrap("linear:" + tensor_kind(name) + f":t{t.ggml_type}", fn, cost)

    weights.linear = linear
    original_grouped = weights.grouped

    def grouped(x, name, *, groups=8):
        t = tensors[name]
        params = math.prod(t.shape)

        def cost(args, out):
            return t.nbytes, 2.0 * math.prod(x.shape[:-2]) * params

        return profiler.call("grouped:" + tensor_kind(name), original_grouped,
                             (x, name), {"groups": groups}, cost)

    weights.grouped = grouped
    original_moe = weights.moe

    def moe(x, ids, scores, prefix):
        per_expert = sum(tensors[prefix + f".ffn_{p}_exps.weight"].nbytes for p in ("gate", "up", "down")) / 144
        params = sum(math.prod(tensors[prefix + f".ffn_{p}_exps.weight"].shape) for p in ("gate", "up", "down")) / 144

        def cost(args, out):
            import numpy as np

            distinct = len(np.unique(np.array(ids)))
            return distinct * per_expert, 2.0 * ids.size * params

        return profiler.call("moe:routed", original_moe, (x, ids, scores, prefix), {}, cost)

    weights.moe = moe
    weights.embedding = profiler.wrap("embedding", weights.embedding)
    for name in ("hc_collapse", "hc_expand", "rms", "gated_delta_kernel", "gated_delta_ops",
                 "pooled_selection", "gate_sigmoid"):
        if hasattr(module, name):
            setattr(module, name, profiler.wrap("fn:" + name, getattr(module, name)))
    for name in ("_kda", "_mla", "_mlp"):
        setattr(model, name, profiler.wrap("block:" + name, getattr(model, name)))


def peak_memory(model, adapter, ids, chunk, mx):
    """Transient peak bytes above the live set, per block and per chunk.

    Pass "chunk": each prefill chunk unfenced, as served. Pass "blocks": each
    _kda/_mla/_mlp call evaluated on its own (inputs first), so a block's peak
    includes whatever its lazy graph keeps alive at once. Pass "inner": the MoE
    call and the pooled indexer evaluated on their own.
    """
    from mlx2.runtime.models import glm53_gguf as module

    gib = 1 << 30
    out = {}

    def fenced(store, kind, fn):
        def run(*a, **k):
            mx.eval([v for v in a if isinstance(v, mx.array)])
            base = mx.get_active_memory()
            mx.reset_peak_memory()
            result = fn(*a, **k)
            mx.eval(result)
            peak = (mx.get_peak_memory() - base) / gib
            store[kind] = max(store.get(kind, 0.0), peak)
            return result
        return run

    for name in ("chunk", "blocks", "inner"):
        store, saved = {}, {}
        if name == "blocks":
            for kind in ("_kda", "_mla", "_mlp"):
                saved[(model, kind)] = getattr(model, kind)
                setattr(model, kind, fenced(store, kind, saved[(model, kind)]))
        if name == "inner":
            saved[(adapter.weights, "moe")] = adapter.weights.moe
            adapter.weights.moe = fenced(store, "moe", adapter.weights.moe)
            saved[(module, "pooled_selection")] = module.pooled_selection
            module.pooled_selection = fenced(store, "pooled_selection", module.pooled_selection)
        cache = model.make_cache()
        mx.clear_cache()
        for start in range(0, len(ids), chunk):
            prompt = mx.array([ids[start:start + chunk]])
            base = mx.get_active_memory()
            mx.reset_peak_memory()
            with model._prompt_scope(prompt.shape[1]):
                mx.eval(model(prompt, cache, logits_to_keep=1), [c.state for c in cache])
            store[f"chunk@{start}"] = (mx.get_peak_memory() - base) / gib
            store[f"active@{start}"] = mx.get_active_memory() / gib
            store[f"cache@{start}"] = mx.get_cache_memory() / gib
        for (obj, attr), fn in saved.items():
            setattr(obj, attr, fn)
        out[name] = {k: round(v, 3) for k, v in store.items()}
        print(f"peak memory {name}: {out[name]}", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--decode", type=int, default=32)
    parser.add_argument("--component-decode", type=int, default=8)
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument("--prefill-ablation", action="store_true")
    parser.add_argument("--capture-routing", action="store_true",
                        help="save one prefill chunk's routing ids/scores per MoE layer (routing.npz)")
    parser.add_argument("--kind-split", action="store_true",
                        help="only the per-layer-kind (KDA/MLA/MLP) fenced decode split over --decode steps, then exit")
    parser.add_argument("--peak-memory", action="store_true",
                        help="only attribute transient peak memory per block over a chunked prefill, then exit")
    parser.add_argument("--cache-limit-gb", type=float, default=None,
                        help="research: mx.set_cache_limit for this run (MLX buffer cache)")
    parser.add_argument("--moe-window", type=int, default=None,
                        help="research: run segmented MoE prefill in row windows of this size")
    parser.add_argument("--mla-kv-q8", action="store_true",
                        help="research: MLA latent cache as q8 (group 64), quantized-matmul attention")
    parser.add_argument("--mla-serial", action="store_true",
                        help="research: serialize MLA prefill attention pieces (mx.depends)")
    parser.add_argument("--seg-scratch-mb", type=int, default=None,
                        help="research: raise the segmented MoE v3 scratch cap (default 512 MiB)")
    parser.add_argument("--source-revision", default="unrecorded")
    args = parser.parse_args()
    if args.seg_scratch_mb:
        from functools import partial

        from mlx2.runtime.gguf import kblock_prefill_metal

        # weights.moe imports the v3 function from the module at call time.
        kblock_prefill_metal.kblock_segmented_moe_prefill_v3 = partial(
            kblock_prefill_metal.kblock_segmented_moe_prefill_v3, max_scratch_bytes=args.seg_scratch_mb << 20)
        kblock_prefill_metal.kblock_segmented_moe_prefill_v3.__name__ = "kblock_segmented_moe_prefill_v3"
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("refusing to reuse a nonempty evidence directory")
    args.out.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx

    from mlx2.adapters.glm53_gguf import Glm53GGUFAdapter

    record = {
        "schema": "mlx2.glm53-ordinary-profile.v1",
        "checkpoint": CHECKPOINT,
        "source_revision": args.source_revision,
        "host": platform.node(),
        "device": mx.device_info(),
        "mlx": mx.__version__,
        "tokens": args.tokens,
        "chunk": args.chunk,
    }
    # Wire like the server's generator does (generate.py), so the weights stay
    # resident; without it the direct-model run can page or compress them.
    info = mx.device_info()
    record["wired_limit_bytes"] = info.get("max_recommended_working_set_size")
    if record["wired_limit_bytes"]:
        mx.set_wired_limit(record["wired_limit_bytes"])
    if args.cache_limit_gb is not None:
        mx.set_cache_limit(int(args.cache_limit_gb * (1 << 30)))
    record["cache_limit_gb"] = args.cache_limit_gb
    record["system"] = system_peaks(mx)
    print(json.dumps({"system": record["system"]}), flush=True)
    policy = json.loads(args.policy.read_text()) if args.policy else {}
    record["execution_policy"] = policy
    t0 = time.perf_counter()
    adapter = Glm53GGUFAdapter(MODEL, execution_policy=policy)
    record["load_seconds"] = time.perf_counter() - t0
    model = adapter.model
    if args.moe_window:
        adapter.weights.expert_prefill_window = args.moe_window
    model.mla_serial_pieces = bool(args.mla_serial)
    model.mla_kv_q8 = bool(args.mla_kv_q8)
    record["research_knobs"] = {"moe_window": args.moe_window, "mla_serial": bool(args.mla_serial), "mla_kv_q8": bool(args.mla_kv_q8),
                                "seg_scratch_mb": args.seg_scratch_mb, "cache_limit_gb": args.cache_limit_gb}
    ids = prompt_ids(adapter, args.tokens)
    record["prompt_tokens"] = len(ids)

    if args.peak_memory:
        record["peak_memory"] = peak_memory(model, adapter, ids, args.chunk, mx)
        (args.out / "profile.json").write_text(json.dumps(record, indent=1) + "\n")
        return

    if args.kind_split:
        record["kind_split"] = kind_split(model, ids, args.chunk, args.decode, mx)
        record["scheduled_calls"] = dict(model.calls)
        record["status"] = "completed" if record["kind_split"]["steps"] == args.decode else "incomplete"
        print(json.dumps({"kind_split": record["kind_split"]}), flush=True)
        (args.out / "profile.json").write_text(json.dumps(record, indent=1) + "\n")
        adapter.close()
        return

    if args.capture_routing:
        # Save the real routing of one prefill chunk for MoE tile tuning.
        import numpy as np

        original_moe = adapter.weights.moe
        captured = {}

        def capture(x, ids_, scores, prefix):
            if x.shape[0] == args.chunk and prefix not in captured:
                captured[prefix] = (np.array(ids_), np.array(scores))
            return original_moe(x, ids_, scores, prefix)

        adapter.weights.moe = capture
        cache = model.make_cache()
        with model._prompt_scope(args.chunk):
            mx.eval(model(mx.array([ids[: args.chunk]]), cache, logits_to_keep=1))
        adapter.weights.moe = original_moe
        np.savez(args.out / "routing.npz", **{f"{k}.ids": v[0] for k, v in captured.items()},
                 **{f"{k}.scores": v[1] for k, v in captured.items()})
        print(f"captured routing for {len(captured)} MoE layers", flush=True)
    run_pass(model, ids[: args.chunk], args.chunk, 2, mx)  # warm kernels/compiles
    before = vm_counters()
    chunks, steps, out = run_pass(model, ids, args.chunk, args.decode, mx)
    normal_vm = counter_delta(before, vm_counters())
    record["normal"] = {
        "chunk_seconds": chunks,
        "prefill_seconds": sum(chunks),
        "prefill_tokens_per_second": len(ids) / sum(chunks),
        "decode_step_seconds": steps,
        "decode_median_seconds": sorted(steps)[len(steps) // 2],
        "decode_tokens_per_second": 1.0 / sorted(steps)[len(steps) // 2],
        "output_ids": out,
        "top_windows": run_pass.windows,
        "prefill_last_logits_sha256": run_pass.prefill_logits_sha256,
        "vm_delta_pages": normal_vm,
        "mlx_active_gb": mx.get_active_memory() / (1 << 30),
        "mlx_cache_gb": mx.get_cache_memory() / (1 << 30),
        "mlx_peak_gb": mx.get_peak_memory() / (1 << 30),
        # Engagement receipt: the model's route counters after the normal pass.
        "scheduled_calls": dict(model.calls),
    }
    # Host/GPU split of a decode step: graph construction (lazy, no eval) versus
    # evaluation of the built graph, on the continuing cache.
    split = []
    cache = model.make_cache()
    with model._prompt_scope(len(ids[: args.chunk])):
        mx.eval(model(mx.array([ids[: args.chunk]]), cache, logits_to_keep=1), [c.state for c in cache])
    token = out[0]
    for _ in range(8):
        t0 = time.perf_counter()
        logits = model(mx.array([[token]]), cache, logits_to_keep=1)
        t1 = time.perf_counter()
        mx.eval(logits)
        t2 = time.perf_counter()
        token = int(mx.argmax(logits[0, -1]).item())
        split.append({"build_s": t1 - t0, "eval_s": t2 - t1})
    record["normal"]["decode_host_gpu_split"] = split
    # Pipelined decode like the server generator (generate.py: mx.async_eval):
    # step n+1's graph is built on step n's lazy argmax while step n executes.
    cache = model.make_cache()
    with model._prompt_scope(len(ids[: args.chunk])):
        logits = model(mx.array([ids[: args.chunk]]), cache, logits_to_keep=1)
    y = mx.argmax(logits[0, -1])
    mx.eval(y, [c.state for c in cache])
    pipelined, count = [], 32
    t0 = time.perf_counter()
    for _ in range(count):
        logits = model(y.reshape(1, 1), cache, logits_to_keep=1)
        nxt = mx.argmax(logits[0, -1])
        mx.async_eval(nxt)
        pipelined.append(int(y.item()))
        y = nxt
    pipelined.append(int(y.item()))
    elapsed = time.perf_counter() - t0
    record["normal"]["decode_pipelined_tokens_per_second"] = count / elapsed
    # Primitive census of one decode step's lazy graph (kernel-count proxy).
    import collections
    import re

    logits = model(y.reshape(1, 1), cache, logits_to_keep=1)
    dot = args.out / "decode-step-graph.dot"
    mx.export_to_dot(str(dot), logits, *[c.state for c in cache])
    labels = re.findall(r'label\s*=\s*"([^"]+)"', dot.read_text())
    prims = collections.Counter(re.sub(r"\s.*|\(.*", "", lbl) for lbl in labels if not lbl[:1].isdigit())
    record["normal"]["decode_graph_census"] = dict(prims.most_common())
    mx.eval(logits)
    record["normal"]["decode_pipelined_ids"] = pipelined
    print(json.dumps({k: record["normal"][k] for k in
                      ("prefill_tokens_per_second", "decode_tokens_per_second", "vm_delta_pages")}), flush=True)
    print("decode graph census (top):", list(record["normal"]["decode_graph_census"].items())[:25], flush=True)
    print("decode pipelined tok/s:", round(record["normal"]["decode_pipelined_tokens_per_second"], 2), flush=True)
    print("decode build/eval ms:", [(round(x["build_s"] * 1e3, 1), round(x["eval_s"] * 1e3, 1)) for x in split], flush=True)

    record["ablation"] = ablation(model, adapter.weights, ids, args.chunk, mx)
    if args.prefill_ablation:
        record["prefill_ablation"] = ablation(model, adapter.weights, ids, args.chunk, mx, mode="prefill")
    record["layer_census"] = layer_census(model, ids, args.chunk, args.out, mx)
    profiler = Profiler(mx)
    instrument(model, adapter.weights, profiler, mx)
    phases = {"prefill": 0.0, "decode": 0.0}
    before = vm_counters()
    t_chunks, t_steps, t_out = run_pass(
        model, ids, args.chunk, args.component_decode, mx,
        on_phase=lambda p: setattr(profiler, "phase", p),
    )
    phases["prefill"], phases["decode"] = sum(t_chunks), sum(t_steps)
    # Census of one more decode step (counts only; the fences make its time meaningless).
    profiler.census_dir = args.out
    census_before = {k: (v["primitives"], dict(v["primitive_kinds"])) for k, v in profiler.rows.items()}
    profiler.phase = "census"
    cache = model.make_cache()
    with model._prompt_scope(len(ids[: args.chunk])):
        mx.eval(model(mx.array([ids[: args.chunk]]), cache, logits_to_keep=1), [c.state for c in cache])
    profiler.phase = "census-decode"
    mx.eval(model(mx.array([[t_out[0]]]), cache, logits_to_keep=1))
    profiler.census_dir = None
    del census_before
    rows = []
    for (phase, label), row in sorted(profiler.rows.items(), key=lambda kv: -kv[1]["seconds"]):
        row = {**row, "primitive_kinds": dict(row["primitive_kinds"])}
        entry = {"phase": phase, "label": label, **row,
                 "share": row["seconds"] / phases[phase] if phases.get(phase) else None}
        if row["seconds"] > 0 and row["bytes"]:
            entry["gb_s"] = row["bytes"] / row["seconds"] / 1e9
            entry["tflop_s"] = row["flops"] / row["seconds"] / 1e12
        rows.append(entry)
    record["components"] = {
        "fenced_prefill_seconds": phases["prefill"],
        "fenced_decode_seconds": phases["decode"],
        "decode_steps": args.component_decode,
        "output_matches_normal_prefix": t_out == out[: len(t_out)],
        "vm_delta_pages": counter_delta(before, vm_counters()),
        "rows": rows,
    }
    record["scheduled_packed_calls"] = adapter.diagnostics().get("scheduled_packed_calls")
    (args.out / "profile.json").write_text(json.dumps(record, indent=1) + "\n")
    census = sorted((r for r in rows if r["phase"] == "census-decode"), key=lambda r: -r["primitives"])
    print("== census of one decode step: primitives added per component class (calls)")
    for r in census[:22]:
        top = sorted(r["primitive_kinds"].items(), key=lambda kv: -kv[1])[:6]
        print(f"  {r['label']:40s} {r['primitives']:6d} ({r['calls']:3d} calls)  {top}")
    for phase in ("prefill", "decode"):
        print(f"== {phase} (fenced total {phases[phase]:.3f}s)")
        for r in [r for r in rows if r["phase"] == phase][:18]:
            print(f"  {r['label']:42s} {r['calls']:5d} {r['seconds']*1e3:9.1f} ms {100*(r['share'] or 0):5.1f}%"
                  + (f"  {r['gb_s']:7.1f} GB/s {r['tflop_s']:6.2f} TF/s" if "gb_s" in r else ""))
    adapter.close()


if __name__ == "__main__":
    main()

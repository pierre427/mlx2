#!/usr/bin/env python3
"""Where the time goes on Hy4-flash's serving paths (one process, one load).

Original mlx2 script.  A profiler to rank tuning opportunities, not
qualification and not publishable statistics: no thermal gating, medians of a
few repetitions.  The model is loaded once through ``HYV4FlashAdapter`` (with
the embedded DFly2 drafter, default policy ``{"draft_model": "embedded_dfly",
"vision": false}``) and four sections run in order, each followed by
``gc`` + ``mx.clear_cache()`` with swap and MLX memory recorded around it:

``decode``      ordinary B=1 decode after a ``--decode-contexts`` prompt:
                * pipelined steps as the engine runs them (the next forward's
                  host n-gram hashing waits on the previous token): tok/s and
                  per-step wall time;
                * split steps: ``mx.async_eval`` is a no-op while the forward
                  is built, so building the lazy graph (Python + the Engram
                  host hash and row reads, timed separately by wrapping the
                  hashers and readers) ends before the GPU starts; then
                  ``mx.eval`` of the sampled token is the GPU wait (Metal
                  encode + execution + one sync).  build + wait is the
                  serialized step; minus the pipelined step it is the overlap
                  the Engram-boundary ``async_eval`` buys;
                * host syncs of one pipelined step (the counting harness of
                  tests/test_hy_v4_flash_sync_cpu.py, mirrored here because
                  that module sets the CPU device at import);
                * per-component GPU cost of that context's next decode token
                  (see "components").
``ast``         DFly2 rounds on the external-draft engine at B=1 (greedy by
                default): one plain pass for tok/s and committed tokens per
                round, one instrumented pass whose class-level wrappers time
                (and ``mx.eval``) every drafter call, every chain-packed
                verify forward (``Hy4ChainTreeExecutor._run``) and any other
                target forward, so a round splits into draft / verify /
                engine host time; then the components of one verify-shaped
                forward (the most common verify row count) on the round's
                context, with unique experts per layer and bytes moved.
``prefill``     one chunk of each ``--prefill-chunks`` length from an empty
                cache: chunk time (normal path), the build/GPU split, the
                Engram host time during the chunk, the components (sampled
                layers, see below) and the effective TFLOP/s of the MoE and
                attention parts.  The per-chunk times of the long decode
                context's prefill (section decode) give the offset curve.

components      Inputs are captured from one real forward: ``DecoderLayer``
                is mirrored for that one forward (same sub-calls, same
                order; for forwards of at most 64 rows the hidden is checked
                bit-equal to an unmirrored forward, ``mirror_exact``), keeping
                every layer's sub-block inputs.  Each component then runs in
                isolation on its captured inputs: one graph holding the
                component for every layer of a group (distinct weights, so
                no false cache reuse; one sync amortised over the group),
                ``mx.eval``, median of ``--component-reps`` after warm-up.
                Stateful parts (attention, indexer) get a private copy of
                the pre-forward cache per repetition, made and evaluated
                before the clock.  ``build_ms`` (graph construction) and
                ``ms`` (eval) are kept apart.  Decode and verify capture all
                layers; prefill captures one layer per layer signature
                (attention kind x MLP format x Engram) and scales by the
                signature's layer count (``scaled``).  Leaf rows sum to
                ``leaf_sum_ms``; ``residual_ms`` = step - leaf sum is launch,
                sync and host overhead the isolated rows do not see (it can
                be negative: isolated rows pay their own sync).  Sub-rows
                (``parent``) split a leaf; ``attn.core`` is derived
                (attention - project - indexer - finish).

bytes / GB/s    Parameter bytes each component reads (weights + scales +
                biases; routed experts as unique experts touched x bytes per
                expert), plus latent KV rows read by attention and pooled
                keys read by the indexer; activations are not counted.
                Achieved GB/s = bytes / ms, against ``--peak-gbs`` (default
                546, an assumption passed on the command line, not measured;
                ``mx.device_info()`` reports no bandwidth).

kernel launches MLX exposes no launch counter.  The lazy graph of one split
                step is exported with ``mx.export_to_dot`` before it is
                evaluated and its primitive nodes are counted by type;
                ``estimated_kernels`` drops view-only primitives (reshape,
                slice, broadcast, ...).  An estimate: a ``Compiled`` node is
                one fused kernel, some primitives launch several (SDPA,
                sorts, sorted gathers, copies behind non-contiguous views),
                and the graph of a step split by Engram reads is complete
                here only because the split pass defers all evaluation.

GPU (the caller owns the GPU; one model per process):

    PYTHONPATH="$PWD/src:$PWD" .venv/bin/python scripts/profile_hy4_paths.py --i-own-the-gpu \\
        --model ~/mlx-models/Hy4-flash-MLX-4.6bit-mlx2 --out <empty dir>

``--cpu`` runs every section on the CPU device (smoke test on tiny fixtures).
Writes ``profile.json`` (schema ``mlx2.profile_hy4_paths.v1``) and
``profile.md`` after every section; refuses a nonempty output directory.
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import os
import re
import statistics
import subprocess
import sys
import time
from collections import Counter
from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SCHEMA = "mlx2.profile_hy4_paths.v1"
SECTIONS = ("decode", "ast", "prefill")
DEFAULT_POLICY = {"draft_model": "embedded_dfly", "vision": False}
# Primitives that normally emit no kernel (views and graph bookkeeping).
VIEW_PRIMITIVES = frozenset({
    "Broadcast", "BroadcastAxes", "Reshape", "Squeeze", "ExpandDims", "Transpose", "Slice",
    "StopGradient", "Depends", "AsStrided", "Split", "Flatten", "Unflatten", "NumberOfElements",
})
SYNC_MODULES = ("mlx2.runtime.models.hy_v4_flash", "mlx2.runtime.models.hy_v4_flash_engram",
                "mlx2.runtime.generate", "mlx2.runtime.external_speculative",
                "mlx2.runtime.drafters.hy4_dfly", "mlx2.adapters.hy_v4_flash_ast")
MIRROR_CHECK_MAX_ROWS = 64

pc = time.perf_counter
mx = None  # mlx.core, imported in main (after the device choice)
H = None  # mlx2.runtime.models.hy_v4_flash


def log(msg):
    print(f"[profile_hy4] {msg}", flush=True)


def now():
    return datetime.now(UTC).isoformat(timespec="seconds")


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def ms(seconds):
    return None if seconds is None else round(seconds * 1e3, 4)


# ------------------------------------------------------------- receipts --

def git_state():
    def run(*a):
        return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True,
                              check=False).stdout.strip()
    return {"commit": run("rev-parse", "HEAD") or None,
            "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def swap_state():
    out = {"swap_used_mb": None, "swapouts": None}
    try:
        s = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True, check=False).stdout
        m = re.search(r"used = ([\d.]+)M", s)
        out["swap_used_mb"] = float(m.group(1)) if m else None
    except Exception:  # noqa: BLE001
        pass
    try:
        s = subprocess.run(["vm_stat"], capture_output=True, text=True, check=False).stdout
        line = next(l for l in s.splitlines() if l.startswith("Swapouts"))
        out["swapouts"] = int(line.split(":")[1].strip(" ."))
    except Exception:  # noqa: BLE001
        pass
    return out


def memory_state():
    def get(name):
        fn = getattr(mx, name, None) or getattr(mx.metal, name, None)
        try:
            return int(fn()) if fn else None
        except Exception:  # noqa: BLE001
            return None
    return {"active_bytes": get("get_active_memory"), "peak_bytes": get("get_peak_memory"),
            "cache_bytes": get("get_cache_memory")}


def free():
    gc.collect()
    mx.clear_cache()


# ------------------------------------------------------------ mlx helpers --

def arrays_in(obj, out=None):
    out = [] if out is None else out
    if isinstance(obj, mx.array):
        out.append(obj)
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            arrays_in(x, out)
    elif isinstance(obj, dict):
        for x in obj.values():
            arrays_in(x, out)
    return out


def eval_tree(obj):
    arrays = arrays_in(obj)
    if arrays:
        mx.eval(arrays)


def pbytes(module):
    from mlx.utils import tree_flatten

    return int(sum(v.nbytes for _, v in tree_flatten(module.parameters())))


def linear_flops_per_row(module):
    """2 * in * out (* heads) per input row for a (quantized) linear or MultiLinear."""
    if "weight" not in module:
        return 0
    w = module["weight"]
    if w.ndim < 2:
        return 0
    inner = w.shape[-1]
    if "scales" in module:
        inner = inner * 32 // int(getattr(module, "bits", 4))
    heads = int(np.prod(w.shape[:-2])) if w.ndim > 2 else 1
    return 2 * heads * inner * int(w.shape[-2])


def module_linear_flops(module, skip=()):
    total = 0
    for name, sub in module.named_modules():
        if any(name == s or name.startswith(s + ".") for s in skip):
            continue
        if "weight" in sub and getattr(sub["weight"], "ndim", 0) >= 2:
            total += linear_flops_per_row(sub)
    return total


def graph_ops(outputs):
    """Primitive nodes of the lazy graph behind ``outputs`` (``mx.export_to_dot``)."""
    buf = io.StringIO()
    mx.export_to_dot(buf, *arrays_in(outputs))
    labels = re.findall(r'\[label ="([^"]+)", shape=rectangle\]', buf.getvalue())
    counts = Counter(label.split(" ")[0] for label in labels)
    views = sum(n for k, n in counts.items() if k in VIEW_PRIMITIVES)
    return {"method": "mx.export_to_dot primitive nodes of the unevaluated step graph",
            "primitives": len(labels), "view_primitives": views,
            "estimated_kernels": len(labels) - views,
            "by_type": dict(counts.most_common(30))}


def private_clone(cache):
    """A cache copy that owns its buffers (real copies, evaluated by the caller)."""
    obj = cache.clone()
    for p in obj._buffers():
        data = getattr(obj, p + "_data")
        if data is not None:
            setattr(obj, p + "_data", data + 0)
    return obj


def clone_all(caches):
    out = [private_clone(c) for c in caches]
    eval_tree([[getattr(c, p + "_data") for p in c._buffers()] for c in out])
    return out


@contextmanager
def no_async_eval():
    real = mx.async_eval
    mx.async_eval = lambda *a, **k: None
    try:
        yield
    finally:
        mx.async_eval = real


@contextmanager
def patched(owner, name, value):
    had = name in owner.__dict__
    saved = owner.__dict__.get(name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        if had:
            setattr(owner, name, saved)
        else:
            delattr(owner, name)


# --------------------------------------------------- host-sync counting --

class _NumpyProxy:
    def __init__(self, counter):
        self._counter = counter

    def __getattr__(self, name):
        return getattr(np, name)

    def array(self, value, *a, **k):
        if isinstance(value, mx.array):
            self._counter.record("wait", "np.array")
        return np.array(value, *a, **k)

    def asarray(self, value, *a, **k):
        if isinstance(value, mx.array):
            self._counter.record("wait", "np.asarray")
        return np.asarray(value, *a, **k)


class SyncCounter:
    def __init__(self):
        self.events = []

    def record(self, kind, what):
        frame = sys._getframe(2)
        while frame is not None and frame.f_code.co_filename == __file__:
            frame = frame.f_back
        site = "?" if frame is None else f"{frame.f_code.co_filename.rsplit('/', 1)[-1]}:{frame.f_lineno}"
        self.events.append((kind, what, site))

    def summary(self):
        waits = [e for e in self.events if e[0] == "wait"]
        dispatches = [e for e in self.events if e[0] == "dispatch"]
        return {"waits": len(waits), "dispatches": len(dispatches),
                "wait_sites": dict(Counter(f"{w}@{s}" for _, w, s in waits)),
                "dispatch_sites": dict(Counter(s for _, _, s in dispatches)),
                "method": "mirror of tests/test_hy_v4_flash_sync_cpu.py count_syncs: mx.eval, "
                          ".item, .tolist, np.array/np.asarray of an mx.array in the Hy4 modules "
                          "= wait; mx.async_eval = dispatch (call sites, not device time)"}


@contextmanager
def count_syncs():
    counter = SyncCounter()
    real_eval, real_async = mx.eval, mx.async_eval
    real_tolist, real_item = mx.array.tolist, mx.array.item

    def eval_(*a, **k):
        counter.record("wait", "mx.eval")
        return real_eval(*a, **k)

    def async_(*a, **k):
        counter.record("dispatch", "mx.async_eval")
        return real_async(*a, **k)

    def tolist(self):
        counter.record("wait", ".tolist")
        return real_tolist(self)

    def item(self):
        counter.record("wait", ".item")
        return real_item(self)

    saved = {}
    for name in SYNC_MODULES:
        module = sys.modules.get(name)
        if module is not None and hasattr(module, "np"):
            saved[name] = module.np
            module.np = _NumpyProxy(counter)
    mx.eval, mx.async_eval = eval_, async_
    mx.array.tolist, mx.array.item = tolist, item
    try:
        yield counter
    finally:
        mx.eval, mx.async_eval = real_eval, real_async
        mx.array.tolist, mx.array.item = real_tolist, real_item
        for name, value in saved.items():
            sys.modules[name].np = value


# ------------------------------------------------------- Engram host time --

class EngramProbe:
    """Wall time inside the host hashers and the row readers (instance wrappers)."""

    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.hash_s = self.read_s = self.prefetch_s = 0.0
        self.rows = self.reads = 0
        self.read_bytes = 0

    def snapshot(self):
        return {"hash_ms": ms(self.hash_s), "read_ms": ms(self.read_s),
                "prefetch_ms": ms(self.prefetch_s), "host_ms": ms(self.hash_s + self.read_s + self.prefetch_s),
                "rows_read": self.rows, "reads": self.reads, "decoded_bytes": self.read_bytes}

    @contextmanager
    def active(self):
        saved = []
        for hasher in self.model.engram_hashers.values():
            orig = hasher.table_rows

            def table_rows(ngrams, orig=orig):
                t = pc()
                try:
                    return orig(ngrams)
                finally:
                    self.hash_s += pc() - t
            hasher.table_rows = table_rows
            saved.append((hasher, "table_rows"))
        for reader in self.model.engram_rows.values():
            read = reader.read

            def read_(rows, orig=read):
                t = pc()
                try:
                    out = orig(rows)
                    self.rows += int(np.asarray(rows).size)
                    self.reads += 1
                    self.read_bytes += int(getattr(out, "nbytes", 0))
                    return out
                finally:
                    self.read_s += pc() - t
            reader.read = read_
            saved.append((reader, "read"))
            prefetch = getattr(reader, "prefetch", None)
            if callable(prefetch):
                def prefetch_(rows, orig=prefetch):
                    t = pc()
                    try:
                        return orig(rows)
                    finally:
                        self.prefetch_s += pc() - t
                reader.prefetch = prefetch_
                saved.append((reader, "prefetch"))
        try:
            yield self
        finally:
            for obj, name in saved:
                try:
                    delattr(obj, name)
                except AttributeError:
                    pass


# ------------------------------------------------------------- prompts --

def prompt_ids(adapter, n, offset=0):
    """``n`` token ids of varied repository text (never a repeated string)."""
    vocab = adapter.model.args.vocab_size
    try:
        from realistic_haystack import haystack

        chars = max(4096, 6 * n)
        while True:
            text = haystack(chars, offset=offset)
            ids = [int(i) for i in adapter.tokenizer.encode(text, add_special_tokens=False) if 3 < i < vocab]
            if len(ids) >= n or chars > 64 * n:
                break
            chars *= 2
        if len(ids) >= n and len(set(ids)) > min(16, n // 4):
            return ids[:n], "realistic_haystack"
    except Exception:  # noqa: BLE001 - a tiny fixture tokenizer maps text to specials
        pass
    rng = np.random.default_rng(1000 + n + offset)
    return [int(x) for x in rng.integers(4, vocab, size=n)], "synthetic_random_ids"


def ast_prompt(adapter):
    try:
        row = json.loads((ROOT / "benchmarks/gsm8k/test.jsonl").read_text().splitlines()[0])
        request = {"messages": [{"role": "user", "content": row["question"]
                                 + "\nThink step by step, then give the final answer after '####'."}]}
        ids = [int(i) for i in adapter.prompt_tokens(request)]
        vocab = adapter.model.args.vocab_size
        if len(ids) > 8 and all(0 <= i < vocab for i in ids) and len(set(ids)) > 8:
            return ids, "gsm8k test[0], chat template, adapter default reasoning effort"
    except Exception:  # noqa: BLE001
        pass
    ids, _ = prompt_ids(adapter, 48)
    return ids, "synthetic_random_ids"


# ------------------------------------------------------- forward timing --

def prefill_cache(model, ids, step):
    """Chunked prefill of ``ids``: (cache, last hidden row, per-chunk receipts)."""
    cache = model.make_cache()
    chunks, h = [], None
    for s in range(0, len(ids), step):
        part = ids[s:s + step]
        mx.synchronize()
        t = pc()
        h = model.hidden(mx.array([part]), cache)
        mx.eval(h)
        chunks.append({"offset": s, "tokens": len(part), "ms": ms(pc() - t),
                       "tok_s": len(part) / max(pc() - t, 1e-9)})
        h = h[:, -1:]
    return cache, h, chunks


def forward_cell(model, snapshot, ids, *, reps, split_reps, warm, head, probe):
    """Normal-path and split (build | GPU) timing of one forward from ``snapshot``."""
    inputs = mx.array([ids])

    def run(clones):
        h = model.hidden(inputs, clones)
        return model.logits(h if head == "all" else h[:, -1:])

    normal, engram_normal = [], []
    for r in range(warm + reps):
        clones = clone_all(snapshot)
        probe.reset()
        mx.synchronize()
        t = pc()
        with probe.active():
            out = run(clones)
            mx.eval(out)
        if r >= warm:
            normal.append(pc() - t)
            engram_normal.append(probe.snapshot())
        del out, clones
    split, ops = [], None
    for r in range(split_reps):
        clones = clone_all(snapshot)
        probe.reset()
        mx.synchronize()
        t0 = pc()
        with no_async_eval(), probe.active():
            out = run(clones)
        t1 = pc()
        if ops is None:
            ops = graph_ops(out)
        t2 = pc()
        mx.eval(out)
        t3 = pc()
        split.append({"build_ms": ms(t1 - t0), "gpu_wait_ms": ms(t3 - t2),
                      "total_ms": ms(t1 - t0 + t3 - t2), "engram": probe.snapshot()})
        del out, clones
    free()
    return {
        "rows": len(ids), "offset": snapshot[0].offset, "head": head,
        "normal_ms": ms(med(normal)), "normal_ms_all": [ms(x) for x in normal],
        "tok_s": len(ids) / med(normal) if normal else None,
        "engram_host_normal": engram_normal[len(engram_normal) // 2] if engram_normal else None,
        "split": split, "build_ms": med([s["build_ms"] for s in split]),
        "gpu_wait_ms": med([s["gpu_wait_ms"] for s in split]),
        "graph_ops": ops,
    }


# ----------------------------------------------------------- components --

def mlp_label(layer):
    mlp = layer.mlp
    if not isinstance(mlp, H.MoE):
        return "dense_mlp"
    bank = mlp.experts.get("gate_proj") or mlp.experts.get("gate_up_proj")
    if getattr(bank, "source_mode", None):
        return f"moe_{bank.source_mode}"
    bits = getattr(bank, "bits", None)
    return f"moe_{getattr(bank, 'mode', 'affine')}{bits}" if bits else "moe_float"


def attn_label(layer, offset, L):
    attn = layer.self_attn
    if attn.kind == H.SLIDING:
        return "sliding"
    if offset + L <= attn.candidate_topk:
        return "indexed_dense"
    return "indexed_sparse_gather" if L <= H.GATHER_MAX_QUERY else "indexed_sparse_mask"


def signature(layer, offset, L):
    return (attn_label(layer, offset, L), mlp_label(layer), "engram_layer" in layer)


class Capture:
    """Mirrors ``DecoderLayer.__call__`` for one forward, keeping sub-block inputs."""

    def __init__(self, model, layers):
        self.model = model
        self.index = {id(model.model.layers[i]): i for i in layers}
        self.rec = {}
        self.head = None

    @contextmanager
    def active(self):
        cap = self
        original = H.DecoderLayer.__call__
        head_original = H.HyperHead.__call__

        def mirrored(layer, streams, cache, engram=None):
            i = cap.index.get(id(layer))
            if i is None:
                return original(layer, streams, cache, engram)
            s0 = streams
            used = engram if (engram is not None and "engram_layer" in layer) else None
            if used is not None:
                streams = layer.engram_layer(streams, *used)
            offset = cache.offset
            post_a, x = layer.hc_attn_layer(streams)
            xn = layer.input_layernorm(x)
            a = layer.self_attn(xn, cache)
            s2 = H.hc_merge(post_a, a, streams)
            post_m, xm_raw = layer.hc_mlp_layer(s2)
            xm = layer.post_attention_layernorm(xm_raw)
            m = layer.mlp(xm)
            out = H.hc_merge(post_m, m, s2)
            rec = dict(s0=s0 if used is not None else None, engram=used, s1=streams, offset=offset,
                       post_a=post_a, xn=xn, attn_out=a, s2=s2, post_m=post_m, xm=xm, mlp_out=m)
            eval_tree([rec, out])
            cap.rec[i] = rec
            return out

        def head(module, streams):
            mx.eval(streams)
            cap.head = streams
            return head_original(module, streams)

        H.DecoderLayer.__call__ = mirrored
        H.HyperHead.__call__ = head
        try:
            yield self
        finally:
            H.DecoderLayer.__call__ = original
            H.HyperHead.__call__ = head_original


def _switch_glu_stages(glu, routed_in, indices):
    """``{name: (prep, fn(prep))}`` for the sub-stages of a stock ``SwitchGLU``, or None.

    ``prep`` builds a stage's inputs (the earlier stages' outputs); the timer
    evaluates it before the clock and frees it after the group, so a long
    prefill chunk never holds every stage's activations of every layer.
    """
    from mlx2.runtime.models import switch_layers as S

    if type(glu) is not S.SwitchGLU or getattr(S._grouped, "MODE", "off") != "off":
        return None
    token_length = int(indices.shape[-2]) if indices.ndim >= 2 else 1
    do_sort, _ = S._switch_sort_decision(glu._sort_policy, assignments=int(indices.size),
                                         token_length=token_length, invariant=False)

    def sorted_inputs():
        x = mx.expand_dims(routed_in, (-2, -3))
        return S._gather_sort(x, indices) if do_sort else (x, indices, None)

    def projected():
        xs, idx, inv = sorted_inputs()
        return (xs, idx, inv, glu.up_proj(xs, idx, sorted_indices=do_sort),
                glu.gate_proj(xs, idx, sorted_indices=do_sort))

    def activated():
        _, idx, inv, up, gate = projected()
        return idx, inv, glu.activation(up, gate)

    def downed():
        idx, inv, act = activated()
        return inv, glu.down_proj(act, idx, sorted_indices=do_sort)

    stages = {}
    if do_sort:
        stages["moe.experts.sort"] = (None, lambda p: S._gather_sort(mx.expand_dims(routed_in, (-2, -3)), indices))
    stages["moe.experts.up"] = (sorted_inputs, lambda p: glu.up_proj(p[0], p[1], sorted_indices=do_sort))
    stages["moe.experts.gate"] = (sorted_inputs, lambda p: glu.gate_proj(p[0], p[1], sorted_indices=do_sort))
    stages["moe.experts.act"] = (projected, lambda p: glu.activation(p[3], p[4]))
    stages["moe.experts.down"] = (activated, lambda p: glu.down_proj(p[2], p[0], sorted_indices=do_sort))
    if do_sort:
        stages["moe.experts.unsort"] = (downed, lambda p: S._scatter_unsort(p[1], p[0], indices.shape))
    return stages


def layer_components(model, i, rec, snapshot, L):
    """Component items of one captured layer: name, parent, group, ``fn(cache[, prep])``,
    whether it needs a private cache copy, an optional ``prep`` (inputs built before the
    clock), bytes and FLOPs; and the layer's unique routed-expert count."""
    a = model.args
    layer = model.model.layers[i]
    attn = layer.self_attn
    offset, xn, xm = rec["offset"], rec["xn"], rec["xm"]
    kind = attn_label(layer, offset, L)
    out = []

    def add(name, parent, group, fn, *, cache=False, nbytes=None, flops=None, prep=None):
        out.append(dict(name=name, parent=parent, group=group, fn=fn, cache=cache, bytes=nbytes,
                        flops=flops, layer=i, prep=prep))

    if rec["engram"] is not None:
        eng = layer.engram_layer
        rows, mask = rec["engram"]
        add("engram", None, "engram", lambda c: eng(rec["s0"], rows, mask),
            nbytes=pbytes(eng) - eng.lookup_table.nbytes + rows.nbytes,
            flops=L * module_linear_flops(eng))
    hc = layer.hc_attn_layer.hc_pre
    hc_bytes = hc.hc_fn.nbytes + hc.hc_base.nbytes + hc.hc_scale.nbytes
    add("hc_attn_pre", None, "all", lambda c: layer.hc_attn_layer(rec["s1"]), nbytes=hc_bytes)
    add("input_norm", None, "all", lambda c: layer.input_layernorm(xn), nbytes=pbytes(layer.input_layernorm))

    # Attention: parameter bytes without the indexer, plus latent rows read.
    T = offset + L
    buf = getattr(snapshot[i], snapshot[i]._kv + "_data")
    row_bytes = 0 if buf is None else buf.nbytes // max(1, buf.shape[0] * buf.shape[1])
    if row_bytes == 0:
        row_bytes = (attn.kv_lora_rank + attn.qk_rope) * 2
    if kind == "sliding":
        visible = [min(attn.window, offset + j + 1) for j in range(L)]
        rows_read = min(attn.window + L - 1, T)
    elif kind == "indexed_sparse_gather":
        K = attn.compressed_indexer.group_topk * attn.compressed_indexer.ratio
        visible = [K] * L
        rows_read = min(L * K, T)
    else:
        visible = [T] * L if kind == "indexed_sparse_mask" else [offset + j + 1 for j in range(L)]
        rows_read = T
    width = attn.kv_lora_rank + attn.qk_rope
    core_flops = 2 * attn.num_heads * 2 * width * int(sum(visible))
    idx_params = pbytes(attn.compressed_indexer) if "compressed_indexer" in attn else 0
    proj_flops = L * module_linear_flops(attn, skip=("compressed_indexer",))
    add("attention", None, kind, lambda c: attn(xn, c), cache=True,
        nbytes=pbytes(attn) - idx_params + rows_read * row_bytes, flops=proj_flops + core_flops)
    q_resid = attn.q_a_layernorm(attn.q_a_proj(xn))  # project()'s q_resid, for the indexer
    mx.eval(q_resid)
    proj_bytes = sum(pbytes(attn[n]) for n in ("q_a_proj", "q_a_layernorm", "q_b_proj", "kv_a_proj_with_mqa",
                                                 "kv_a_layernorm", "embed_q"))
    add("attn.project", "attention", kind, lambda c: attn.project(xn, offset), nbytes=proj_bytes)
    if "compressed_indexer" in attn:
        ix = attn.compressed_indexer
        sparse = kind.startswith("indexed_sparse")
        groups = T // ix.ratio

        def indexer(c):
            tk = ix.token_key_norm(ix.token_key_projection(xn))
            gk = c.append_index_keys(tk, ix)
            return ix.select(xn, q_resid, offset, gk) if sparse else gk
        add("attn.indexer", "attention", kind, indexer, cache=True,
            nbytes=idx_params + (groups * ix.head_dim * 4 if sparse else 0),
            flops=L * module_linear_flops(ix) + (2 * ix.num_heads * ix.head_dim * groups * L if sparse else 0))
    fin_bytes = sum(pbytes(attn[n]) for n in ("unembed_out", "linear_gate", "o_proj", "post_proj_norm"))
    add("attn.finish", "attention", kind, lambda c, latent: attn.finish(xn, latent), nbytes=fin_bytes,
        prep=lambda: mx.zeros((1, attn.num_heads, L, attn.kv_lora_rank), dtype=xn.dtype))
    add("hc_merge_attn", None, "all", lambda c: H.hc_merge(rec["post_a"], rec["attn_out"], rec["s1"]))
    add("hc_mlp_pre", None, "all", lambda c: layer.hc_mlp_layer(rec["s2"]), nbytes=hc_bytes)
    add("post_attn_norm", None, "all", lambda c: layer.post_attention_layernorm(xm),
        nbytes=pbytes(layer.post_attention_layernorm))

    mlp = layer.mlp
    group = mlp_label(layer)
    unique = None
    if isinstance(mlp, H.MoE):
        indices, weights = mlp.gate(xm)
        down = getattr(mlp, "routed_expert_down_proj", None)
        routed_in = down(xm) if down is not None else xm
        up = getattr(mlp, "routed_expert_up_proj", None)
        y = mlp.experts(routed_in, indices)
        y_sum = (y * weights[..., None]).sum(axis=-2).astype(xm.dtype)
        pre_norm = (up(y_sum) if up is not None else y_sum) + mlp.shared_experts(xm)
        eval_tree([indices, weights, routed_in, pre_norm])
        del y, y_sum
        unique = int(np.unique(np.array(indices)).size)
        experts = mlp.experts
        per_expert = pbytes(experts) / a.n_routed_experts
        inter = a.moe_intermediate_size
        latent_w = a.routed_expert_hidden_size or a.hidden_size
        expert_flops = 6 * latent_w * inter * int(indices.size)
        shared_bytes, router_bytes = pbytes(mlp.shared_experts), pbytes(mlp.gate)
        lat_bytes = (pbytes(down) if down is not None else 0) + (pbytes(up) if up is not None else 0)
        moe_flops = (expert_flops + L * (2 * a.hidden_size * a.n_routed_experts
                                         + module_linear_flops(mlp.shared_experts)
                                         + (linear_flops_per_row(down) if down is not None else 0)
                                         + (linear_flops_per_row(up) if up is not None else 0)))
        add("moe", None, group, lambda c: mlp(xm),
            nbytes=int(unique * per_expert) + shared_bytes + router_bytes + lat_bytes
            + pbytes(mlp.combine_output_norm), flops=moe_flops)
        add("moe.router", "moe", group, lambda c: mlp.gate(xm), nbytes=router_bytes,
            flops=L * 2 * a.hidden_size * a.n_routed_experts)
        if down is not None:
            add("moe.latent_down", "moe", group, lambda c: down(xm), nbytes=pbytes(down),
                flops=L * linear_flops_per_row(down))
        add("moe.experts", "moe", group, lambda c: experts(routed_in, indices),
            nbytes=int(unique * per_expert), flops=expert_flops)
        stages = _switch_glu_stages(experts, routed_in, indices)
        if stages is not None:
            share = {"moe.experts.up": "up_proj", "moe.experts.gate": "gate_proj",
                     "moe.experts.down": "down_proj"}
            for name, (prep, fn) in stages.items():
                part = share.get(name)
                nb = int(unique * pbytes(experts[part]) / a.n_routed_experts) if part else None
                fl = 2 * latent_w * inter * int(indices.size) if part else None
                add(name, "moe.experts", group, lambda c, p=None, fn=fn: fn(p), nbytes=nb, flops=fl, prep=prep)

        def combine(c, y):
            s = (y * weights[..., None]).sum(axis=-2).astype(xm.dtype)
            return up(s) if up is not None else s
        add("moe.combine", "moe", group, combine, nbytes=pbytes(up) if up is not None else 0,
            prep=lambda: experts(routed_in, indices))
        add("moe.shared", "moe", group, lambda c: mlp.shared_experts(xm), nbytes=shared_bytes,
            flops=L * module_linear_flops(mlp.shared_experts))
        add("moe.combine_norm", "moe", group, lambda c: mlp.combine_output_norm(pre_norm),
            nbytes=pbytes(mlp.combine_output_norm))
    else:
        add("mlp", None, group, lambda c: mlp(xm), nbytes=pbytes(mlp), flops=L * module_linear_flops(mlp))
    add("hc_merge_mlp", None, "all", lambda c: H.hc_merge(rec["post_m"], rec["mlp_out"], rec["s2"]))
    return out, unique


def head_components(model, streams, head):
    m = model.model
    collapsed = m.hc_head(streams)
    h = m.norm(collapsed)
    rows = h if head == "all" else h[:, -1:]
    logits = model.logits(rows)
    eval_tree([collapsed, h, rows, logits])
    L = rows.shape[1]
    vocab = model.args.vocab_size
    key = mx.random.key(0)
    out = []

    def add(name, fn, nbytes=None, flops=None, extra=False):
        out.append(dict(name=name, parent=None, group="head", fn=fn, cache=False, bytes=nbytes,
                        flops=flops, layer=-1, extra=extra))
    add("head.hc_head", lambda c: m.hc_head(streams), nbytes=pbytes(m.hc_head))
    add("head.norm", lambda c: m.norm(collapsed), nbytes=pbytes(m.norm))
    add("head.lm_head", lambda c: model.logits(rows), nbytes=pbytes(model.lm_head),
        flops=L * linear_flops_per_row(model.lm_head))
    add("head.sampler_argmax", lambda c: mx.argmax(logits[:, -1:], axis=-1), nbytes=logits[:, -1:].nbytes)
    add("head.sampler_categorical", lambda c: mx.random.categorical(logits[:, -1:] / 0.9, key=key),
        nbytes=4 * vocab, extra=True)
    return out


def time_group(items, snapshot, reps, warm):
    """One graph of ``items`` (one per layer), ``mx.eval``: (median build s, median eval s)."""
    builds, evals = [], []
    preps = [it["prep"]() if it.get("prep") else None for it in items]
    eval_tree(preps)
    for r in range(warm + reps):
        caches = None
        if any(it["cache"] for it in items):
            caches = [private_clone(snapshot[it["layer"]]) if it["cache"] else None for it in items]
            eval_tree([[getattr(c, p + "_data") for p in c._buffers()] for c in caches if c is not None])
        mx.synchronize()
        t0 = pc()
        outs = [it["fn"](caches[j] if caches else None, *(() if preps[j] is None else (preps[j],)))
                for j, it in enumerate(items)]
        t1 = pc()
        eval_tree(outs)
        t2 = pc()
        if r >= warm:
            builds.append(t1 - t0)
            evals.append(t2 - t1)
        del outs, caches
    del preps
    return med(builds), med(evals)


def profile_components(model, snapshot, ids, *, reps, warm, sample, head, step_ms, step_source,
                       peak_gbs, label):
    L = len(ids)
    layers = model.model.layers
    offset = snapshot[0].offset
    sigs = [signature(layer, offset, L) for layer in layers]
    if sample:
        chosen, seen = [], set()
        for i, s in enumerate(sigs):
            if s not in seen:
                seen.add(s)
                chosen.append(i)
    else:
        chosen = list(range(len(layers)))
    cap = Capture(model, chosen)
    cap_cache = clone_all(snapshot)
    inputs = mx.array([ids])
    with cap.active(), no_async_eval():
        hidden = model.hidden(inputs, cap_cache)
        mx.eval(hidden)
    mirror = None
    if L <= MIRROR_CHECK_MAX_ROWS:
        ref_cache = clone_all(snapshot)
        ref = model.hidden(inputs, ref_cache)
        mirror = bool(mx.array_equal(ref, hidden).item())
        del ref, ref_cache
    del cap_cache, hidden
    free()

    items, unique = [], {}
    for i in chosen:
        rows, u = layer_components(model, i, cap.rec[i], snapshot, L)
        items.extend(rows)
        if u is not None:
            unique[i] = u
    items.extend(head_components(model, cap.head, head))

    # Layers of the whole model per (component, group): the scale of a sampled row.
    population = Counter()
    for i, layer in enumerate(layers):
        kind, mlp, engram = sigs[i]
        base = ["hc_attn_pre", "input_norm", "hc_merge_attn", "hc_mlp_pre", "post_attn_norm", "hc_merge_mlp"]
        for name in base:
            population[(name, "all")] += 1
        for name in ("attention", "attn.project", "attn.finish") + (
                ("attn.indexer",) if "compressed_indexer" in layer.self_attn else ()):
            population[(name, kind)] += 1
        if engram:
            population[("engram", "engram")] += 1
        for name in ("moe", "moe.router", "moe.latent_down", "moe.experts", "moe.experts.sort",
                     "moe.experts.up", "moe.experts.gate", "moe.experts.act", "moe.experts.down",
                     "moe.experts.unsort", "moe.combine", "moe.shared", "moe.combine_norm", "mlp"):
            population[(name, mlp)] += 1

    groups = {}
    for it in items:
        groups.setdefault((it["name"], it["group"]), []).append(it)
    rows = []
    for (name, group), members in groups.items():
        build_s, eval_s = time_group(members, snapshot, reps, warm)
        n = len(members)
        total_in_model = 1 if group == "head" else population.get((name, group), n)
        scale = total_in_model / n
        nbytes = [m["bytes"] for m in members]
        flops = [m["flops"] for m in members]
        per_layer = eval_s / n
        row = {
            "component": name, "parent": members[0]["parent"], "group": group,
            "layers_measured": [m["layer"] for m in members], "layers_in_model": total_in_model,
            "scaled": scale != 1, "per_layer_ms": ms(per_layer), "ms": ms(per_layer * total_in_model),
            "build_ms": ms(build_s * scale), "derived": False, "extra": bool(members[0].get("extra")),
            "bytes": None if None in nbytes else int(sum(nbytes) * scale),
            "flops": None if None in flops else int(sum(flops) * scale),
        }
        rows.append(row)
        del members
    # attn.core: attention - project - indexer - finish (per group).
    by = {(r["component"], r["group"]): r for r in rows}
    for (name, group), r in list(by.items()):
        if name != "attention":
            continue
        parts = [by.get((p, group)) for p in ("attn.project", "attn.indexer", "attn.finish")]
        core = r["ms"] - sum(p["ms"] for p in parts if p is not None)
        rows.append({"component": "attn.core", "parent": "attention", "group": group,
                     "layers_measured": r["layers_measured"], "layers_in_model": r["layers_in_model"],
                     "scaled": r["scaled"], "per_layer_ms": round(core / r["layers_in_model"], 4),
                     "ms": round(core, 4), "build_ms": None, "derived": True, "extra": False,
                     "bytes": None, "flops": None})
    leaf = sum(r["ms"] for r in rows if r["parent"] is None and not r["derived"] and not r["extra"])
    for r in rows:
        r["pct_of_step"] = round(100 * r["ms"] / step_ms, 2) if step_ms else None
        r["gb_s"] = round(r["bytes"] / (r["ms"] * 1e6), 2) if r["bytes"] and r["ms"] and r["ms"] > 0 else None
        r["pct_of_peak_bw"] = round(100 * r["gb_s"] / peak_gbs, 2) if r["gb_s"] and peak_gbs else None
        r["tflops"] = round(r["flops"] / (r["ms"] * 1e9), 3) if r["flops"] and r["ms"] and r["ms"] > 0 else None
    rows.sort(key=lambda r: (r["parent"] is not None, -(r["ms"] or 0)))
    groups_summary = {}
    for r in rows:
        if r["parent"] is None and not r["extra"]:
            key = "attention" if r["component"] == "attention" else (
                "moe" if r["component"] in ("moe", "mlp") else r["component"])
            groups_summary[key] = round(groups_summary.get(key, 0.0) + r["ms"], 4)
    weight_bytes = sum(r["bytes"] or 0 for r in rows if r["parent"] is None and not r["extra"])
    cap.rec.clear()
    free()
    return {
        "label": label, "rows_per_forward": L, "offset": offset,
        "mode": "sample" if sample else "all", "layers_captured": chosen,
        "signatures": {str(i): list(sigs[i]) for i in chosen},
        "reps": reps, "warmup": warm, "mirror_exact": mirror,
        "step_ms": step_ms, "step_source": step_source,
        "leaf_sum_ms": round(leaf, 4),
        "residual_ms": round(step_ms - leaf, 4) if step_ms else None,
        "residual_pct": round(100 * (step_ms - leaf) / step_ms, 2) if step_ms else None,
        "leaf_bytes": weight_bytes,
        "leaf_bytes_at_step_gb_s": round(weight_bytes / (step_ms * 1e6), 2) if step_ms else None,
        "by_block_ms": groups_summary,
        "unique_experts": {str(k): v for k, v in unique.items()},
        "rows": rows,
    }


# ---------------------------------------------------------------- decode --

def run_decode(adapter, a, probe, ctx):
    model = adapter.model
    ids, source = prompt_ids(adapter, ctx + 1)
    prompt, follow = ids[:ctx], ids[ctx]
    cache, h, chunks = prefill_cache(model, prompt, a.prefill_step)
    tok = mx.argmax(model.logits(h)[:, -1:], axis=-1)
    mx.eval(tok)

    def step(token):
        logits = model(token, cache)
        nxt = mx.argmax(logits[:, -1:], axis=-1)
        mx.async_eval(nxt)
        return nxt

    for _ in range(a.decode_warmup):
        tok = step(tok)
    mx.eval(tok)
    with count_syncs() as counter:
        tok = step(tok)
    syncs = counter.summary()
    mx.eval(tok)
    times = []
    t_start = t_prev = pc()
    for _ in range(a.decode_steps):
        tok = step(tok)
        t = pc()
        times.append(t - t_prev)
        t_prev = t
    mx.eval(tok)
    total = pc() - t_start
    split, ops = [], None
    for _ in range(a.decode_split_steps):
        mx.eval(tok)
        probe.reset()
        t0 = pc()
        with no_async_eval(), probe.active():
            logits = model(tok, cache)
            nxt = mx.argmax(logits[:, -1:], axis=-1)
        t1 = pc()
        if ops is None:
            ops = graph_ops(nxt)
        t2 = pc()
        mx.eval(nxt)
        t3 = pc()
        e = probe.snapshot()
        split.append({"build_ms": ms(t1 - t0), "gpu_wait_ms": ms(t3 - t2), "total_ms": ms(t1 - t0 + t3 - t2),
                      "engram": e, "python_build_ms": round(ms(t1 - t0) - e["host_ms"], 4)})
        tok = nxt
    step_ms = ms(med(times))
    cell = {
        "context": ctx, "prompt_source": source, "offset_at_components": None,
        "prefill_chunks": chunks,
        "pipelined": {"steps": a.decode_steps, "tok_s": a.decode_steps / total,
                      "step_ms_median": step_ms, "step_ms_p10": ms(float(np.percentile(times, 10))),
                      "step_ms_p90": ms(float(np.percentile(times, 90)))},
        "split": {"steps": len(split), "build_ms": med([s["build_ms"] for s in split]),
                  "python_build_ms": med([s["python_build_ms"] for s in split]),
                  "engram_hash_ms": med([s["engram"]["hash_ms"] for s in split]),
                  "engram_read_ms": med([s["engram"]["read_ms"] for s in split]),
                  "engram_prefetch_ms": med([s["engram"]["prefetch_ms"] for s in split]),
                  "gpu_wait_ms": med([s["gpu_wait_ms"] for s in split]),
                  "total_ms": med([s["total_ms"] for s in split]), "steps_detail": split},
        "host_syncs_per_step": syncs,
        "graph_ops_per_step": ops,
    }
    s = cell["split"]
    if s["total_ms"] is not None and step_ms is not None:
        s["overlap_ms"] = round(s["total_ms"] - step_ms, 4)
    log(f"decode ctx={ctx}: {cell['pipelined']['tok_s']:.2f} tok/s, step {step_ms} ms "
        f"(split: build {s['build_ms']} ms [engram host {s['engram_hash_ms']}+{s['engram_read_ms']}], "
        f"gpu {s['gpu_wait_ms']} ms), ~{(ops or {}).get('estimated_kernels')} kernels")
    snapshot = clone_all(cache)
    cell["offset_at_components"] = snapshot[0].offset
    cache = None
    free()
    cell["components"] = profile_components(
        model, snapshot, [follow], reps=a.component_reps, warm=a.component_warmup, sample=False,
        head="all", step_ms=step_ms, step_source="decode pipelined step median",
        peak_gbs=a.peak_gbs, label=f"decode@{ctx}")
    c = cell["components"]
    log(f"components decode@{ctx}: leaf sum {c['leaf_sum_ms']} ms of {step_ms} ms "
        f"(residual {c['residual_ms']} ms), mirror_exact={c['mirror_exact']}")
    del snapshot
    free()
    return cell


def run_generator_b1(adapter, a, ctx):
    from mlx2.runtime.generate import BatchGenerator

    ids, _ = prompt_ids(adapter, ctx, offset=7)
    n = a.decode_steps
    batch = BatchGenerator(adapter.model, max_tokens=n + 1, completion_batch_size=1, prefill_batch_size=1)
    uid = batch.insert([ids], max_tokens=[n + 1])[0]
    got, first = 0, None
    for _ in range(10 * n + 100):
        result = batch.next()
        for r in (result[1] if isinstance(result, tuple) else result):
            if r.uid == uid:
                got += 1
                if first is None:
                    first = pc()
        if got >= n + 1:
            break
    dt = pc() - first if first else None
    close = getattr(batch, "close", None)
    if callable(close):
        close()
    del batch
    free()
    return {"context": ctx, "tokens": got - 1, "tok_s": (got - 1) / dt if dt else None,
            "note": "BatchGenerator, one lane (the ordinary serving loop), after the first token"}


# ------------------------------------------------------------------- AST --

class RoundTimer:
    """Class-level timers (with ``mx.eval``) around drafter calls and target forwards."""

    def __init__(self):
        self.depth = 0
        self.cur = None

    def begin(self):
        self.cur = Counter()
        self.cur_rows = []

    def end(self):
        out = {k: ms(v) for k, v in self.cur.items() if not k.startswith("n_")}
        out.update({k: v for k, v in self.cur.items() if k.startswith("n_")})
        out["verify_rows"] = list(self.cur_rows)
        return out

    def wrap(self, owner, name, category, *, rows=None, result=lambda out: out):
        orig = owner.__dict__[name]
        timer = self

        def wrapper(*args, **kwargs):
            if timer.depth or timer.cur is None:
                return orig(*args, **kwargs)
            timer.depth += 1
            t = pc()
            try:
                out = orig(*args, **kwargs)
                eval_tree(result(out))
                return out
            finally:
                timer.depth -= 1
                cat = category(args, kwargs) if callable(category) else category
                timer.cur[cat] += pc() - t
                timer.cur["n_" + cat] += 1
                if rows is not None and cat == "verify":
                    timer.cur_rows.append(rows(args, kwargs))
        return patched(owner, name, wrapper)

    @contextmanager
    def active(self):
        from mlx2.adapters.hy_v4_flash_ast import Hy4ChainTreeExecutor
        from mlx2.runtime.drafters.hy4_dfly import Hy4DFlyDraftModel

        stack = []
        try:
            for name in ("propose_tree", "draft_distributions"):
                stack.append(self.wrap(Hy4DFlyDraftModel, name, "draft"))
            stack.append(self.wrap(Hy4DFlyDraftModel, "append_context", "draft_context"))
            stack.append(self.wrap(
                Hy4ChainTreeExecutor, "_run",
                lambda args, kw: "verify_plain" if kw.get("plain") else "verify",
                rows=lambda args, kw: int(sum(len(r) for r in args[2])),
                result=lambda out: (out[0], out[1])))
            stack.append(self.wrap(H.Model, "hidden", "target_other"))
            for cm in stack:
                cm.__enter__()
            yield self
        finally:
            for cm in reversed(stack):
                cm.__exit__(None, None, None)


def drive_ast(adapter, a, prompt, instrument):
    batch = adapter.create_external_batch(completion_batch_size=1, prefill_step_size=a.prefill_step)
    stats0 = dict(batch.scheduler_stats)
    uid = batch.insert([prompt], max_tokens=[a.ast_max_tokens],
                       sampling_configs=[{"sampling_temp": a.ast_temp}])[0]
    timer = RoundTimer()
    calls, got, finished = [], [], False
    with timer.active() if instrument else nullcontext():
        for _ in range(4 * a.ast_max_tokens + 64):
            timer.begin() if instrument else None
            mx.synchronize()
            t = pc()
            result = batch.next()
            dt = pc() - t
            mine = [r for r in (result[1] if isinstance(result, tuple) else result) if r.uid == uid]
            row = {"ms": ms(dt), "tokens": len(mine)}
            if instrument:
                row.update(timer.end())
                timed = sum(row.get(k) or 0 for k in ("draft", "draft_context", "verify", "verify_plain",
                                                      "target_other"))
                row["host_other_ms"] = round(row["ms"] - timed, 4)
            calls.append(row)
            got.extend(int(r.token) for r in mine)
            if any(getattr(r, "finish_reason", None) for r in mine) or len(got) >= a.ast_max_tokens:
                finished = True
                break
    stats1 = dict(batch.scheduler_stats)
    close = getattr(batch, "close", None)
    if callable(close):
        close()
    del batch
    free()
    first = next((j for j, c in enumerate(calls) if c["tokens"]), None)
    rounds = calls[first + 1:] if first is not None else []
    decode_s = sum(c["ms"] for c in rounds) / 1e3
    decode_tokens = sum(c["tokens"] for c in rounds)
    delta = {k: stats1.get(k, 0) - stats0.get(k, 0) for k in stats1
             if isinstance(stats1.get(k), (int, float)) and stats1.get(k, 0) != stats0.get(k, 0)}
    out = {"tokens": got, "finished": finished, "calls": len(calls), "rounds": len(rounds),
           "prefill_and_first_token_ms": round(sum(c["ms"] for c in calls[: first + 1]), 3) if first is not None else None,
           "decode_tok_s": decode_tokens / decode_s if decode_s else None,
           "committed_per_round_mean": decode_tokens / len(rounds) if rounds else None,
           "committed_per_round_hist": dict(Counter(c["tokens"] for c in rounds)),
           "round_ms_median": med([c["ms"] for c in rounds]),
           "scheduler_stats_delta": delta}
    if instrument:
        verify_rounds = [c for c in rounds if c.get("n_verify")]
        out["instrumented"] = {
            "method": "class-level wrappers with mx.eval of each call's arrays (serializes the "
                      "round: tok/s here is below the plain pass)",
            "draft_ms_median": med([c.get("draft", 0) for c in verify_rounds]),
            "draft_context_ms_median": med([c.get("draft_context", 0) for c in verify_rounds]),
            "verify_ms_median": med([c.get("verify", 0) for c in verify_rounds]),
            "verify_plain_ms_median": med([c.get("verify_plain") for c in rounds if c.get("verify_plain")]),
            "target_other_ms_median": med([c.get("target_other") for c in rounds if c.get("target_other")]),
            "host_other_ms_median": med([c["host_other_ms"] for c in verify_rounds]),
            "round_ms_median": med([c["ms"] for c in verify_rounds]),
            "verify_rounds": len(verify_rounds),
            "committed_per_verify_round_mean": (sum(c["tokens"] for c in verify_rounds) / len(verify_rounds)
                                                if verify_rounds else None),
            "committed_per_verify_round_hist": dict(Counter(c["tokens"] for c in verify_rounds)),
            "note": "one row per batch.next() call after the first token; the engine may emit a "
                    "round's tokens one call late (parked/pipelined rounds) and runs plain-anchor "
                    "rounds (verify_plain) when its cost model parks speculation",
            "plain_anchor_rounds": sum(1 for c in rounds if c.get("n_verify_plain")),
            "verify_rows_hist": dict(Counter(r for c in rounds for r in c.get("verify_rows", []))),
            "per_round": rounds,
        }
    return out


def run_ast(adapter, a, probe, record_layers):
    model = adapter.model
    if getattr(adapter, "dfly", None) is None or adapter.drafter_kind != "dfly":
        return {"skipped": "the policy selects no embedded DFly2 drafter"}
    model.record_layers = record_layers
    try:
        prompt, source = ast_prompt(adapter)
        plain = drive_ast(adapter, a, prompt, instrument=False)
        log(f"ast plain: {plain['decode_tok_s']} tok/s, {plain['committed_per_round_mean']} tokens/round "
            f"over {plain['rounds']} rounds")
        inst = drive_ast(adapter, a, prompt, instrument=True)
        i = inst["instrumented"]
        log(f"ast instrumented: round {i['round_ms_median']} ms = draft {i['draft_ms_median']} + verify "
            f"{i['verify_ms_median']} + host {i['host_other_ms_median']} ms")
        hist = i["verify_rows_hist"]
        R = max(hist, key=lambda r: (hist[r], r)) if hist else adapter.num_draft + 1
        tokens = plain["tokens"] or inst["tokens"]
        context = prompt + tokens[: max(0, len(tokens) - R)]
        follow = (tokens[max(0, len(tokens) - R):] + prompt)[:R]
        cache, _, _ = prefill_cache(model, context, a.prefill_step)
        snapshot = clone_all(cache)
        del cache
        free()
        verify = forward_cell(model, snapshot, follow, reps=a.component_reps, split_reps=3,
                              warm=a.component_warmup, head="all", probe=probe)
        comps = profile_components(
            model, snapshot, follow, reps=a.component_reps, warm=a.component_warmup, sample=False,
            head="all", step_ms=verify["normal_ms"], step_source=f"one {R}-row verify-shaped forward (normal path)",
            peak_gbs=a.peak_gbs, label=f"verify@{snapshot[0].offset}x{R}")
        log(f"components verify {R} rows @ {snapshot[0].offset}: leaf {comps['leaf_sum_ms']} of "
            f"{verify['normal_ms']} ms, unique experts/layer median "
            f"{med(list(comps['unique_experts'].values()))}")
        del snapshot
        free()
        return {"prompt_source": source, "prompt_tokens": len(prompt),
                "policy": dict(adapter.external_policy), "num_draft": adapter.num_draft,
                "temperature": a.ast_temp, "plain": plain, "instrumented": inst,
                "verify_rows": R, "verify_forward": verify, "verify_components": comps,
                "executor_receipt": adapter.tree_executor.receipt() if adapter.tree_executor else None}
    finally:
        model.record_layers = frozenset()


# --------------------------------------------------------------- prefill --

def run_prefill(adapter, a, probe, chunk):
    model = adapter.model
    ids, source = prompt_ids(adapter, chunk, offset=3)
    snapshot = model.make_cache()
    cell = forward_cell(model, snapshot, ids, reps=a.prefill_reps, split_reps=1, warm=a.prefill_warmup,
                        head="last", probe=probe)
    cell["prompt_source"] = source
    comps = profile_components(
        model, snapshot, ids, reps=a.prefill_component_reps, warm=1, sample=True, head="last",
        step_ms=cell["normal_ms"], step_source="prefill chunk (normal path) median",
        peak_gbs=a.peak_gbs, label=f"prefill@{chunk}")
    cell["components"] = comps
    eff = {}
    for block in ("attention", "moe", "mlp"):
        rows = [r for r in comps["rows"] if r["component"] == block]
        f = sum(r["flops"] or 0 for r in rows)
        t = sum(r["ms"] for r in rows)
        if rows:
            eff[block] = {"ms": round(t, 3), "flops": f, "tflops": round(f / (t * 1e9), 3) if t > 0 else None}
    cell["effective_tflops"] = eff
    e = cell["engram_host_normal"] or {}
    log(f"prefill chunk={chunk}: {cell['normal_ms']} ms ({cell['tok_s']:.0f} tok/s), build {cell['build_ms']} ms, "
        f"engram host {e.get('host_ms')} ms, moe {eff.get('moe', {}).get('tflops')} TF/s, "
        f"attention {eff.get('attention', {}).get('tflops')} TF/s")
    free()
    return cell


# -------------------------------------------------------------- markdown --

def f(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def tree_order(rows):
    """Leaves by time, each followed by its sub-rows (same group), recursively."""
    children = {}
    for r in rows:
        if r["parent"] is not None:
            children.setdefault((r["parent"], r["group"]), []).append(r)
    out = []

    def visit(r, depth):
        out.append((depth, r))
        for c in sorted(children.get((r["component"], r["group"]), []), key=lambda c: -(c["ms"] or 0)):
            visit(c, depth + 1)
    for r in sorted((r for r in rows if r["parent"] is None), key=lambda r: (r["extra"], -(r["ms"] or 0))):
        visit(r, 0)
    return out


def md_components(c):
    lines = [f"**{c['label']}**: {c['rows_per_forward']} row(s) at offset {c['offset']}, mode {c['mode']}, "
             f"step {f(c['step_ms'])} ms ({c['step_source']}), leaf sum {f(c['leaf_sum_ms'])} ms, residual "
             f"{f(c['residual_ms'])} ms ({f(c['residual_pct'], 1)}%), mirror_exact {c['mirror_exact']}, "
             f"leaf bytes {c['leaf_bytes'] / 1e9:.3f} GB ({f(c['leaf_bytes_at_step_gb_s'], 1)} GB/s over the step)", "",
             "| component | group | layers | ms/layer | ms | % step | GB/s | % peak BW | TFLOP/s | build ms |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for depth, r in tree_order(c["rows"]):
        name = "&nbsp;&nbsp;" * 2 * depth + ("- " if depth else "") + r["component"] + (
            " (derived)" if r["derived"] else "") + (" (extra)" if r["extra"] else "")
        lines.append(f"| {name} | {r['group']} | {r['layers_in_model']}{'*' if r['scaled'] else ''} | "
                     f"{f(r['per_layer_ms'], 4)} | {f(r['ms'])} | {f(r['pct_of_step'], 1)} | {f(r['gb_s'], 1)} | "
                     f"{f(r['pct_of_peak_bw'], 1)} | {f(r['tflops'], 2)} | {f(r['build_ms'])} |")
    if c.get("unique_experts"):
        u = list(c["unique_experts"].values())
        lines.append(f"\nUnique routed experts per MoE layer: min {min(u)}, median {med(u)}, max {max(u)}.")
    lines.append("\n`*` sampled layers scaled to the model's count of that signature. "
                 "(extra) rows are not part of the leaf sum.\n")
    return lines


def write_outputs(out_dir, doc):
    tmp = out_dir / "profile.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1, default=str))
    tmp.replace(out_dir / "profile.json")
    L = ["# Hy4-flash path profile", "",
         f"- checkpoint: `{doc.get('checkpoint')}`", f"- mlx2 commit: `{doc['mlx2']['commit']}`"
         f"{' (dirty)' if doc['mlx2']['dirty'] else ''}", f"- device: {doc.get('device')} {doc.get('device_info')}",
         f"- mlx {doc.get('mlx')}; policy `{json.dumps(doc.get('policy'))}`; peak BW {doc['peak_gbs']} GB/s "
         f"({doc['peak_gbs_source']})", f"- started {doc['started']}, finished {doc.get('finished')}", ""]
    sec = doc["sections"]
    if sec.get("decode"):
        L += ["## Ordinary B=1 decode", "",
              "| context | tok/s | step ms | build ms | python build ms | engram hash+read ms | GPU wait ms | "
              "serialized ms | overlap ms | host waits/step | est. kernels/step | primitives |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for ctx, c in sec["decode"].items():
            if not isinstance(c, dict) or "pipelined" not in c:
                continue
            s, p, o = c["split"], c["pipelined"], c.get("graph_ops_per_step") or {}
            eh = None if s["engram_hash_ms"] is None else round(s["engram_hash_ms"] + (s["engram_read_ms"] or 0), 3)
            L.append(f"| {ctx} | {p['tok_s']:.2f} | {f(p['step_ms_median'])} | {f(s['build_ms'])} | "
                     f"{f(s['python_build_ms'])} | {f(eh)} | {f(s['gpu_wait_ms'])} | {f(s['total_ms'])} | "
                     f"{f(s.get('overlap_ms'))} | "
                     f"{c['host_syncs_per_step']['waits']} | {o.get('estimated_kernels')} | {o.get('primitives')} |")
        if sec["decode"].get("generator_b1"):
            g = sec["decode"]["generator_b1"]
            L.append(f"\nBatchGenerator one lane at {g['context']}: {g['tok_s']} tok/s.")
        L.append("")
        for ctx, c in sec["decode"].items():
            if isinstance(c, dict) and c.get("components"):
                L += md_components(c["components"])
    if sec.get("ast"):
        t = sec["ast"]
        L += ["## DFly2 AST rounds (B=1)", ""]
        if "skipped" in t:
            L.append(f"skipped: {t['skipped']}\n")
        else:
            p, i = t["plain"], t["instrumented"]["instrumented"]
            L += [f"- prompt: {t['prompt_source']} ({t['prompt_tokens']} tokens), temperature {t['temperature']}, "
                  f"num_draft {t['num_draft']}",
                  f"- plain: {p['decode_tok_s']} tok/s, {p['committed_per_round_mean']} committed tokens/round, "
                  f"round {p['round_ms_median']} ms, {p['rounds']} rounds",
                  f"- instrumented round {i['round_ms_median']} ms: draft {i['draft_ms_median']}, draft context "
                  f"{i['draft_context_ms_median']}, verify {i['verify_ms_median']}, engine host {i['host_other_ms_median']} ms; "
                  f"verify rows {i['verify_rows_hist']}",
                  f"- verify-shaped forward ({t['verify_rows']} rows): normal {t['verify_forward']['normal_ms']} ms, "
                  f"build {t['verify_forward']['build_ms']} ms, GPU wait {t['verify_forward']['gpu_wait_ms']} ms, "
                  f"~{(t['verify_forward']['graph_ops'] or {}).get('estimated_kernels')} kernels", ""]
            L += md_components(t["verify_components"])
    if sec.get("prefill"):
        L += ["## Prefill chunks (from an empty cache)", "",
              "| chunk | ms | tok/s | build ms | GPU wait ms | engram host ms | attention TF/s | MoE TF/s | est. kernels |",
              "|---|---|---|---|---|---|---|---|---|"]
        for chunk, c in sec["prefill"].items():
            e, eff = c.get("engram_host_normal") or {}, c["effective_tflops"]
            L.append(f"| {chunk} | {f(c['normal_ms'])} | {c['tok_s']:.0f} | {f(c['build_ms'])} | {f(c['gpu_wait_ms'])} | "
                     f"{f(e.get('host_ms'))} | {eff.get('attention', {}).get('tflops')} | {eff.get('moe', {}).get('tflops')} | "
                     f"{(c['graph_ops'] or {}).get('estimated_kernels')} |")
        L.append("")
        for chunk, c in sec["prefill"].items():
            L += md_components(c["components"])
    L += ["## Method notes", "", *[f"- {k}: {v}" for k, v in doc["method"].items()], ""]
    (out_dir / "profile.md").write_text("\n".join(L))


# ------------------------------------------------------------------ main --

def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="output directory (must be empty or absent)")
    ap.add_argument("--i-own-the-gpu", action="store_true")
    ap.add_argument("--no-wire", action="store_true", help="leave the weights pageable (default: wire them)")
    ap.add_argument("--cpu", action="store_true", help="run on the CPU device (smoke test)")
    ap.add_argument("--policy-json", default=json.dumps(DEFAULT_POLICY))
    ap.add_argument("--sections", default=",".join(SECTIONS))
    ap.add_argument("--decode-contexts", nargs="+", type=int, default=[1024, 16384])
    ap.add_argument("--decode-steps", type=int, default=32)
    ap.add_argument("--decode-warmup", type=int, default=4)
    ap.add_argument("--decode-split-steps", type=int, default=8)
    ap.add_argument("--generator-b1", action=argparse.BooleanOptionalAction, default=True,
                    help="also time the BatchGenerator one-lane loop at the first decode context")
    ap.add_argument("--component-reps", type=int, default=5)
    ap.add_argument("--component-warmup", type=int, default=2)
    ap.add_argument("--prefill-chunks", nargs="+", type=int, default=[2048, 8192])
    ap.add_argument("--prefill-reps", type=int, default=2)
    ap.add_argument("--prefill-warmup", type=int, default=1)
    ap.add_argument("--prefill-component-reps", type=int, default=2)
    ap.add_argument("--prefill-step", type=int, default=2048, help="chunk size used to build contexts")
    ap.add_argument("--ast-max-tokens", type=int, default=160)
    ap.add_argument("--ast-temp", type=float, default=0.0)
    ap.add_argument("--peak-gbs", type=float, default=546.0,
                    help="device peak memory bandwidth for the roofline column (an assumption, GB/s)")
    a = ap.parse_args(argv)
    if not (a.cpu or a.i_own_the_gpu):
        ap.error("refusing Metal execution without --i-own-the-gpu (or pass --cpu)")
    a.sections = [s for s in a.sections.split(",") if s]
    if set(a.sections) - set(SECTIONS):
        ap.error(f"--sections must be a subset of {SECTIONS}")
    out = Path(a.out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        ap.error(f"{out} exists and is not an empty directory; refusing to overwrite evidence")
    a.policy = json.loads(a.policy_json)
    return a


def main(argv=None):
    global mx, H
    a = parse(argv)
    import mlx.core as _mx

    mx = _mx
    if a.cpu:
        mx.set_default_device(mx.cpu)
    from mlx2.adapters.hy_v4_flash import HYV4FlashAdapter
    from mlx2.runtime.models import hy_v4_flash as _H

    H = _H
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        info = mx.device_info()
    except Exception:  # noqa: BLE001
        info = {}
    doc = {
        "schema": SCHEMA, "started": now(), "finished": None, "mlx2": git_state(), "mlx": mx.__version__,
        "device": str(mx.default_device()), "device_info": {k: v for k, v in info.items()},
        "peak_gbs": a.peak_gbs, "peak_gbs_source": "--peak-gbs (assumed, not measured)",
        "policy": a.policy, "model_path": str(Path(a.model).expanduser()),
        "args": {k: v for k, v in vars(a).items() if k not in ("policy",)},
        "env": {k: v for k, v in os.environ.items() if k.startswith(("MLX_", "MLX2_"))},
        "method": {
            "build_vs_gpu": "split pass: mx.async_eval disabled while the forward is built, so graph build "
                            "(Python + Engram host hash/read, wrapped and timed) finishes before any GPU "
                            "work; mx.eval of the output is the GPU wait (encode + execute + one sync)",
            "components": "inputs captured by mirroring DecoderLayer for one forward; each component of a "
                          "group of layers built into one graph and evaluated, median of reps; stateful "
                          "parts get private cache copies made before the clock",
            "kernel_launches": "estimated: primitive nodes of the unevaluated split-step graph "
                               "(mx.export_to_dot) minus view-only primitives; MLX has no launch counter",
            "bytes": "parameter bytes (routed experts: unique experts x bytes/expert) + latent KV rows read "
                     "+ indexer pooled keys; activations not counted",
            "peak_bandwidth": "--peak-gbs, an assumption",
            "sections": "one process, model loaded once; gc + mx.clear_cache between sections",
        },
        "swap": {}, "memory": {}, "sections": {},
    }
    write_outputs(out, doc)
    t0 = pc()
    swap0 = swap_state()
    adapter = HYV4FlashAdapter(str(Path(a.model).expanduser()), execution_policy=a.policy)
    doc["load_s"] = round(pc() - t0, 2)
    doc["swap"]["load"] = {"before": swap0, "after": swap_state()}
    doc["checkpoint"] = adapter.diagnostics()["checkpoint"]
    doc["layout"] = adapter.layout
    if not a.no_wire:
        # Wire the weights as serving does (weight_residency): pageable
        # ~86 GB weights get compressed/swapped by the 8K prefill transients.
        from mlx2.runtime.weight_residency import wire_serving_weights

        doc["weight_residency"] = wire_serving_weights()
    log(f"loaded {doc['checkpoint']} in {doc['load_s']} s on {doc['device']}")
    model = adapter.model
    probe = EngramProbe(model)
    record_layers = model.record_layers
    model.record_layers = frozenset()  # ordinary sections run without DFly2 Index Share recording
    try:
        args = model.args
        doc["model"] = {"layers": args.num_hidden_layers, "hidden": args.hidden_size,
                        "experts": args.n_routed_experts, "top_k": args.num_experts_per_tok,
                        "layer_types": [l.self_attn.kind for l in model.model.layers],
                        "mlp": [mlp_label(l) for l in model.model.layers],
                        "engram_layers": sorted(args.engram_layers),
                        "resident_parameter_bytes": pbytes(model)}
        for section in a.sections:
            doc["swap"][section] = {"before": swap_state()}
            mx.reset_peak_memory() if hasattr(mx, "reset_peak_memory") else None
            ts = pc()
            if section == "decode":
                cells = {}
                for k, ctx in enumerate(a.decode_contexts):
                    cells[str(ctx)] = run_decode(adapter, a, probe, ctx)
                    doc["sections"]["decode"] = cells
                    write_outputs(out, doc)
                    if k == 0 and a.generator_b1:
                        cells["generator_b1"] = run_generator_b1(adapter, a, ctx)
                        log(f"decode generator one lane ctx={ctx}: {cells['generator_b1']['tok_s']} tok/s")
            elif section == "ast":
                doc["sections"]["ast"] = run_ast(adapter, a, probe, record_layers)
            elif section == "prefill":
                cells = {}
                for chunk in a.prefill_chunks:
                    cells[str(chunk)] = run_prefill(adapter, a, probe, chunk)
                    doc["sections"]["prefill"] = cells
                    write_outputs(out, doc)
            free()
            doc["swap"][section]["after"] = swap_state()
            doc["memory"][section] = {**memory_state(), "seconds": round(pc() - ts, 1)}
            write_outputs(out, doc)
            log(f"section {section} done in {doc['memory'][section]['seconds']} s, swap "
                f"{doc['swap'][section]['before']['swap_used_mb']} -> {doc['swap'][section]['after']['swap_used_mb']} MB")
        d = adapter.diagnostics()
        doc["receipts"] = {k: d.get(k) for k in ("route", "drafter", "checkpoint", "ast", "dfly2", "engram_hash",
                                                 "engram_table_format", "latent_kv", "moe_fused_gate_up",
                                                 "moe_nax_gather", "engram")}
    finally:
        model.record_layers = record_layers
        adapter.close()
        doc["finished"] = now()
        doc["total_s"] = round(pc() - t0, 1)
        write_outputs(out, doc)
    log(f"wrote {out / 'profile.json'} and {out / 'profile.md'} ({doc['total_s']} s)")
    return doc


if __name__ == "__main__":
    main()

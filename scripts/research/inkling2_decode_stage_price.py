"""Price Inkling 2 one-token decode by stage on the real checkpoint (GPU).

Original mlx2 script (recon 2026-11-22; punch-list item 1 step one: "price
each decode stage" before writing fused one-token kernels).  Loads the
artifact with its shipped kernel defaults, prefills ``context`` tokens of
varied repository text, then:

1. ``plain``: median wall time of ``steps`` greedy decode steps, no extra
   synchronization (the served shape);
2. ``synced``: the same steps with every top-level module call fenced: its
   inputs are evaluated first, then its outputs are evaluated and the GPU
   synchronized, and the elapsed time is charged to the module class.  Nested
   priced classes (the NVFP4 expert gather inside SparseMoE) are timed the
   same way and reported as a share of their parent.  Fencing adds a sync per
   call, so ``synced`` totals exceed ``plain``; the per-class shares are what
   to read.  The remainder (embedding, head, sampling) is charged to
   ``other``.

Token ids of both passes are recorded; fencing must not change them.

3. ``skip`` (``--skip``, added after the first live window showed fencing
   costs ~4.5x the plain step, so fenced shares mostly price the fences):
   for each top-level class, a plain unfenced decode in which every call of
   that class returns its own output from the last warm-up step (already
   evaluated, tied to the live inputs with ``mx.depends`` so upstream work is
   not dropped by lazy evaluation; the first live run without the tie priced
   whole dependency chains) instead of computing.  The drop in median step time against
   ``plain`` is that stage's pipelined cost.  Skipped decodes produce
   meaningless tokens; only their timing is read.

Sanity numbers, one process, no thermal hold.  The caller owns the GPU.

Result (Inkling-2-Mini@440956c8, 306adac5e, M5 Max, one process under heavy
system contention; sanity only): plain decode 29.2 ms/step at 1K context and
30.7 at 16K.  Skip savings at 1K / 16K: Attention 12.3 / 13.8 ms (42 / 45%),
SparseMoE 7.0 / 7.5 (24%), AttentionResidual 2.6 / 3.0 (9 / 10%),
ShortConvolution 1.8 / 2.7 (6 / 9%), DynamicShortConvolution 0.1 / 0.5,
MessageController ~0; sum 23.7 / 27.7 ms.  Attention barely grows with
context, so its cost is per-layer projection and launch work (48 calls per
token), not KV reads; ``inkling2_fused_projections`` was already measured
rate-neutral.  Evidence: ~/mlx-models/.download-state/
recon1122-306adac5e-r1-inkprice/price-{1k,16k}.json.  The b0cf14f76 skip
arms (no ``mx.depends`` tie) and the fenced shares are not stage costs.

Usage: python scripts/research/inkling2_decode_stage_price.py <model dir> <out.json>
       [--context 1024] [--steps 32]
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOP = ("Attention", "MessageController", "AttentionResidual", "SparseMoE",
       "ShortConvolution", "DynamicShortConvolution")
NESTED = {"ScaledNVFP4SwitchLinear": "SparseMoE"}
WARMUP = 4


def corpus(tokenizer, n):
    parts = []
    for pattern in ("docs/**/*.md", "src/mlx2/**/*.py"):
        for path in sorted(ROOT.glob(pattern)):
            parts.append(path.read_text(errors="replace"))
            if len(parts) % 25 == 0:
                ids = tokenizer.encode("\n\n".join(parts), add_special_tokens=False)
                if len(ids) >= n:
                    return ids[:n]
    raise SystemExit("corpus too small")


def arrays(value):
    import mlx.core as mx

    if isinstance(value, mx.array):
        return [value]
    if isinstance(value, (list, tuple)):
        return [a for v in value for a in arrays(v)]
    if isinstance(value, dict):
        return [a for v in value.values() for a in arrays(v)]
    return []


def tied(value, inputs):
    """``value`` (replayed) made to depend on the live ``inputs``, so lazy
    evaluation still computes everything upstream of the skipped stage."""
    import mlx.core as mx

    if not inputs:
        return value
    if isinstance(value, mx.array):
        return mx.depends([value], inputs)[0]
    if isinstance(value, (list, tuple)):
        return type(value)(tied(v, inputs) for v in value)
    return value


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--skip", action="store_true", help="also run the subtractive skip arms")
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    sys.path.insert(0, str(ROOT / "src"))
    import mlx.core as mx

    from mlx2.adapters.inkling2 import Inkling2Adapter
    from mlx2.runtime import nvfp4_linear
    from mlx2.runtime.models.inkling2 import language

    adapter = Inkling2Adapter(str(a.model))
    model = adapter.model
    ids = corpus(adapter.tokenizer, a.context)
    charge = collections.defaultdict(float)
    calls = collections.Counter()
    state = {"on": False, "depth": 0, "skip": None, "record": None}
    replay = {}
    originals = []

    def fence(cls, name, nested):
        original = cls.__call__

        def call(self, *args, **kwargs):
            if state["skip"] == name and id(self) in replay:
                return tied(replay[id(self)], arrays(args) + arrays(kwargs))
            if state["record"] == name and not nested:
                out = original(self, *args, **kwargs)
                mx.eval(arrays(out))
                replay[id(self)] = out
                return out
            if not state["on"] or (state["depth"] and not nested):
                return original(self, *args, **kwargs)
            mx.eval(arrays(args), arrays(kwargs))
            mx.synchronize()
            t0 = time.perf_counter()
            if not nested:
                state["depth"] += 1
            try:
                out = original(self, *args, **kwargs)
                mx.eval(arrays(out))
                mx.synchronize()
            finally:
                if not nested:
                    state["depth"] -= 1
            charge[name] += time.perf_counter() - t0
            calls[name] += 1
            return out

        originals.append((cls, original))
        cls.__call__ = call

    for name in TOP:
        if hasattr(language, name):
            fence(getattr(language, name), name, False)
    for name in NESTED:
        fence(getattr(nvfp4_linear, name), name, True)

    def decode(cache, token, steps, synced, skip=None):
        out, times = [], []
        for i in range(steps + WARMUP):
            if i == WARMUP:
                charge.clear()
                calls.clear()
            state["on"] = synced and i >= WARMUP
            state["record"] = skip if i == WARMUP - 1 else None
            state["skip"] = skip if i >= WARMUP else None
            mx.synchronize()
            t0 = time.perf_counter()
            logits = model(mx.array([[token]]), cache=cache)
            token = int(mx.argmax(logits[0, -1]).item())
            mx.synchronize()
            if i >= WARMUP:
                times.append(time.perf_counter() - t0)
                out.append(token)
        state["on"] = False
        state["skip"] = state["record"] = None
        replay.clear()
        return out, times

    try:
        rows = {}
        for synced in (False, True):
            cache = model.make_cache()
            for s in range(0, a.context, 512):
                logits = model(mx.array([ids[s:s + 512]]), cache=cache)
                mx.eval(logits)
            first = int(mx.argmax(logits[0, -1]).item())
            tokens, times = decode(cache, first, a.steps, synced)
            row = {"tokens": tokens, "step_ms_median": 1e3 * statistics.median(times),
                   "tok_s": len(times) / sum(times)}
            if synced:
                total = sum(times)
                priced = {n: charge[n] for n in TOP if n in charge}
                row["ms_per_token"] = {n: 1e3 * v / a.steps for n, v in priced.items()}
                row["ms_per_token"]["other"] = 1e3 * (total - sum(priced.values())) / a.steps
                row["share"] = {n: v / total for n, v in priced.items()}
                row["nested_ms_per_token"] = {n: 1e3 * charge[n] / a.steps for n in NESTED if n in charge}
                row["calls_per_token"] = {n: calls[n] / a.steps for n in calls}
            rows["synced" if synced else "plain"] = row
            del cache
            mx.clear_cache()
        if a.skip:
            rows["skip"] = {}
            base = rows["plain"]["step_ms_median"]
            for name in TOP:
                if not hasattr(language, name):
                    continue
                cache = model.make_cache()
                for s in range(0, a.context, 512):
                    logits = model(mx.array([ids[s:s + 512]]), cache=cache)
                    mx.eval(logits)
                first = int(mx.argmax(logits[0, -1]).item())
                _, times = decode(cache, first, a.steps, False, skip=name)
                median = 1e3 * statistics.median(times)
                rows["skip"][name] = {"step_ms_median": median, "saved_ms": base - median,
                                      "saved_share": (base - median) / base}
                del cache
                mx.clear_cache()
        receipt = {"script": "scripts/research/inkling2_decode_stage_price.py",
                   "checkpoint": adapter.checkpoint, "context": a.context, "steps": a.steps,
                   "mlx2": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                          capture_output=True, text=True).stdout.strip(),
                   "mlx_version": mx.__version__, "kernel_policy": dict(model.kernel_policy),
                   "tokens_equal": rows["plain"]["tokens"] == rows["synced"]["tokens"],
                   "kind": "sanity, one process, no thermal hold", **rows}
        a.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = a.out.with_suffix(".tmp")
        tmp.write_text(json.dumps(receipt, indent=1, default=str))
        tmp.rename(a.out)
        print(json.dumps({k: receipt["synced"].get(k) for k in ("ms_per_token", "share",
                                                               "nested_ms_per_token")}, indent=1,
                         default=lambda v: round(v, 3)))
        if a.skip:
            print(json.dumps(rows["skip"], indent=1, default=lambda v: round(v, 3)))
        print("plain", round(rows["plain"]["tok_s"], 2), "tok/s; tokens_equal",
              receipt["tokens_equal"])
    finally:
        for cls, original in originals:
            cls.__call__ = original
        adapter.close()
    return 0 if receipt["tokens_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

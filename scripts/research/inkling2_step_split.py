"""Split one Inkling 2 decode step into host graph build and GPU time, for mlx2 or Rapid-MLX.

Original mlx2 script (2026-11-25), written after the same-host head-to-head
(mlx2 3123d0b85, Inkling-2-Mini@440956c8) measured Rapid-MLX's ordinary decode
20-25% faster than mlx2's on the same weights.  It answers one question per
stack: how much of a decode step is graph building on the host, how much is
GPU work, and how well the served loop overlaps them.

Each stack's own model is loaded in-process from its own virtualenv:

* ``--stack mlx2``: ``mlx2.adapters.inkling2.Inkling2Adapter`` with its shipped
  defaults (the ordinary route the server selects), run from an mlx2 tree;
* ``--stack rapid``: Rapid-MLX's vendored ``rapid_mlx.models.inkling2`` loaded
  through the ``mlx_lm`` loader the way Rapid-MLX registers it, run with
  Rapid-MLX's interpreter.  Nothing of mlx2 is imported.

After a ``--context``-token prefill (512-row chunks, repository text) and a
warm-up, greedy decode is timed four ways, ``--steps`` steps each:

1. ``pipelined``: the served shape, the next step's graph is built and
   ``mx.async_eval``'d before the previous token is read (the mlx-lm loop);
2. ``serial``: build one step, evaluate it, repeat (no overlap);
3. ``host_build``: the time spent inside ``model(...)`` alone in ``serial``
   (graph construction; MLX is lazy so no GPU work is waited on there);
4. ``gpu_batched``: ``--batch`` steps chained lazily (argmax feeds the next
   step) and evaluated once; elapsed eval time / batch is the GPU cost of a
   step with host build off the critical path.

If ``pipelined`` is close to ``gpu_batched`` the step is GPU-bound; if it is
close to ``serial`` the loop does not overlap; ``host_build`` above
``gpu_batched`` means host-bound.  Greedy token ids of the pipelined run are
recorded (stacks differ in numerics; they are informative, not an oracle).

Sanity numbers, one process, no thermal hold; the caller owns the GPU.
Usage: <python> scripts/research/inkling2_step_split.py --stack mlx2|rapid
       <model dir> <out.json> [--contexts 1024,16384] [--steps 64] [--batch 8]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[2]
CHUNK = 512


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
        encode = lambda text: tok.encode(text, add_special_tokens=False)  # noqa: E731
        return adapter.model, adapter.model.make_cache, encode, {"checkpoint": adapter.checkpoint}
    import rapid_mlx.models.inkling2 as vendored

    sys.modules["mlx_lm.models.inkling2_loop"] = vendored
    from mlx_lm import load as lm_load

    model, tok = lm_load(str(model_dir))
    encode = lambda text: tok.encode(text, add_special_tokens=False)  # noqa: E731
    import importlib.metadata as md

    return model, model.make_cache, encode, {"rapid_mlx": md.version("rapid_mlx"),
                                             "mlx_lm": md.version("mlx_lm")}


def last_token(model, x, cache):
    return mx.argmax(model(x, cache=cache)[:, -1, :], axis=-1)


def cell(model, make_cache, ids, steps, batch):
    cache = make_cache()
    t0 = time.perf_counter()
    for a in range(0, len(ids), CHUNK):
        logits = model(mx.array([ids[a:a + CHUNK]]), cache=cache)
        mx.eval(logits)
    prefill_s = time.perf_counter() - t0
    y = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(y)
    for _ in range(8):  # warm-up (compiles, kernel caches)
        y = last_token(model, y[:, None], cache)
        mx.eval(y)

    # 1. pipelined (mlx-lm style): build the next step before reading this one.
    tokens = []
    nxt = last_token(model, y[:, None], cache)
    mx.async_eval(nxt)
    t0 = time.perf_counter()
    for _ in range(steps):
        cur = nxt
        nxt = last_token(model, cur[:, None], cache)
        mx.async_eval(nxt)
        tokens.append(cur.item())
    pipelined = (time.perf_counter() - t0) / steps
    mx.eval(nxt)
    y = nxt

    # 2./3. serial with the host build timed alone.
    build, total = [], []
    for _ in range(steps):
        t0 = time.perf_counter()
        y = last_token(model, y[:, None], cache)
        t1 = time.perf_counter()
        mx.eval(y)
        t2 = time.perf_counter()
        build.append(t1 - t0)
        total.append(t2 - t0)

    # 4. GPU cost with the build off the critical path.
    gpu = []
    for _ in range(max(1, steps // batch)):
        t0 = time.perf_counter()
        z = y
        for _ in range(batch):
            z = last_token(model, z[:, None], cache)
        t1 = time.perf_counter()
        mx.eval(z)
        t2 = time.perf_counter()
        gpu.append((t2 - t1) / batch)
        y = z
    ms = lambda v: round(1000 * v, 3)  # noqa: E731
    return {
        "context": len(ids), "prefill_s": round(prefill_s, 3),
        "pipelined_ms": ms(pipelined),
        "serial_ms": ms(statistics.median(total)),
        "host_build_ms": ms(statistics.median(build)),
        "gpu_batched_ms": ms(statistics.median(gpu)),
        "pipelined_tok_s": round(1 / pipelined, 2),
        "tokens": tokens[:32],
        "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", choices=("mlx2", "rapid"), required=True)
    ap.add_argument("model", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--contexts", default="1024,16384")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    model, make_cache, encode, meta = load(a.stack, a.model)
    contexts = [int(c) for c in a.contexts.split(",")]
    ids = corpus_ids(encode, max(contexts))
    report = {"script": "scripts/research/inkling2_step_split.py", "stack": a.stack,
              "model_dir": str(a.model), "kind": "sanity (no thermal hold; not publishable)",
              "mlx": mx.__version__, **meta, "cells": []}
    for c in contexts:
        report["cells"].append(cell(model, make_cache, ids[:c], a.steps, a.batch))
        print(json.dumps({k: v for k, v in report["cells"][-1].items() if k != "tokens"}), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=1))
    tmp.replace(a.out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Size the matmul share of a Qwen3.8-27B verify forward, stock vs sp_qmm.

GPU only (``--i-own-the-gpu``). Loads the real artifact through its adapter,
prefills a context, then for each M (verify rows, one lane, L = M) times:

- ``forward``: one full trunk + lm_head forward of M tokens at that context.
  KV caches are trimmed and GDN state restored after every call.
- ``matmuls``: every ``QuantizedLinear`` of the model applied once, in model
  order, to a random bf16 (M, K) input. The weights total ~15 GB, so they
  stream from DRAM exactly as in the forward.

Arms (``--arms stock,sp``) are interleaved per repetition in alternating
order; two warm-up passes per arm and M are discarded. The ``sp`` arm swaps
eligible modules to ``mlx2.runtime.models.sp_qmm`` for 2 <= M <= max_m and
the report records how many calls it actually routed.

``--quality`` runs a teacher-forced check instead: the same token stream is
fed in chunks of ``chunk`` rows (the verify shape) through each arm, and the
per-position argmax, top-2 margin and max |logit| difference are compared.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
M27 = Path("~/mlx-models/Qwen3.8-27B-oQ4e-mtp")


def corpus_ids(tok, n):
    text = ""
    for p in sorted((ROOT / "docs").glob("*.md")):
        text += p.read_text(errors="ignore") + "\n"
        if len(text) > n * 8:
            break
    ids = tok.encode(text)
    return ids[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-own-the-gpu", action="store_true")
    ap.add_argument("--model", type=Path, default=M27)
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--ms", default="1,2,3,4,6,8,12,16,24,32")
    ap.add_argument("--arms", default="stock")
    ap.add_argument("--max-m", type=int, default=16)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--quality", action="store_true")
    ap.add_argument("--chunks", default="3,6,12")
    ap.add_argument("--quality-tokens", type=int, default=384)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        raise SystemExit("refusing Metal execution without --i-own-the-gpu")
    from sp_qmm_guard import SwapGuard
    guard = SwapGuard()
    import mlx.core as mx
    import mlx.nn as nn
    from mlx2.adapters.registry import resolve_adapter
    from mlx2.runtime.models import sp_qmm
    from mlx2.runtime.models.cache import ArraysCache

    mx.set_cache_limit(4 << 30)
    sink = open(a.out, "a") if a.out else None

    def emit(row):
        line = json.dumps(row)
        print(line, flush=True)
        if sink:
            sink.write(line + "\n"); sink.flush()

    adapter = resolve_adapter(a.model)(str(a.model))
    model = adapter.model
    tok = adapter.tokenizer
    emit({"kind": "start", "model": str(a.model), "adapter": type(adapter).__name__,
          "mlx": mx.__version__, "context": a.context, "arms": a.arms,
          "max_m": a.max_m})
    arms = a.arms.split(",")
    handle = None

    def set_arm(arm):
        nonlocal handle
        if handle is not None:
            sp_qmm.remove(handle); handle = None
        if arm.startswith("sp"):
            # "sp" routes every eligible module; "sp_lm" / "sp_b4" / "sp_b5"
            # keep only the lm_head, the 4-bit or the 5-bit modules routed.
            handle = sp_qmm.apply(model, min_m=2, max_m=a.max_m)
            only = arm[3:]
            if only:
                keep = []
                for mod, cls in handle:
                    is_lm = mod["weight"].shape[0] >= 65536
                    ok = (is_lm if only == "lm" else
                          (not is_lm and mod.bits == int(only[1:])))
                    if ok:
                        keep.append((mod, cls))
                    else:
                        mod.__class__ = cls
                handle = keep

    ids = corpus_ids(tok, max(a.context, a.quality_tokens) + 64)

    if a.quality:
        T = a.quality_tokens
        stream = mx.array(ids[:T])[None]
        prefix = 64
        def run_stream(arm, chunk):
            set_arm(arm)
            before = sp_qmm.STATS["routed"]
            cache = model.make_cache()
            mx.eval(model(stream[:, :prefix], cache=cache))
            outs = []
            for s in range(prefix, T, chunk):
                lg = model(stream[:, s:s + chunk], cache=cache).astype(mx.float32)
                mx.eval(lg)
                outs.append(lg[0])
            out = mx.concatenate(outs, axis=0)
            mx.eval(out)
            del cache
            mx.clear_cache()
            return out, sp_qmm.STATS["routed"] - before

        # Batch-variance baseline: stock one token at a time. Each arm at each
        # chunk is compared with it, so stock's own M-dependence is visible.
        ref1, _ = run_stream("stock", 1)
        ref1_sorted = mx.sort(ref1, axis=-1)
        ref1_arg = mx.argmax(ref1, axis=-1)
        ref1_margin = (ref1_sorted[:, -1] - ref1_sorted[:, -2]).tolist()
        del ref1_sorted
        for chunk in [int(c) for c in a.chunks.split(",")]:
            for arm in arms:
                lg, routed = run_stream(arm, chunk)
                mism = [i for i, v in enumerate((mx.argmax(lg, axis=-1) != ref1_arg).tolist()) if v]
                d = mx.abs(lg - ref1)
                emit({"kind": "quality_vs_m1", "chunk": chunk, "arm": arm, "routed": routed,
                      "argmax_mismatch": len(mism),
                      "mismatch_ref_margins": [round(ref1_margin[i], 4) for i in mism],
                      "max_abs_logit_diff": mx.max(d).item(),
                      "mean_abs_logit_diff": mx.mean(d).item()})
                del lg, d
        set_arm("stock")
        emit({"kind": "swap", **guard.report()})
        return

    # Prefill the context once.
    cache = model.make_cache()
    ctx = mx.array(ids[: a.context])[None]
    for s in range(0, a.context, 512):
        mx.eval(model(ctx[:, s:s + 512], cache=cache))
    mx.clear_cache()
    snap = [list(c.cache) if isinstance(c, ArraysCache) else None for c in cache]

    def restore(L):
        for c, st in zip(cache, snap):
            if st is not None:
                c.cache = list(st)
            else:
                c.trim(L)

    qlin = [(p, m) for p, m in model.named_modules() if isinstance(m, nn.QuantizedLinear)
            and "embed_tokens" not in p and ".mtp." not in f".{p}."]
    emit({"kind": "modules", "quantized_linear": len(qlin)})
    guard.timed()
    for M in [int(m) for m in a.ms.split(",")]:
        toks = mx.array(ids[a.context: a.context + M])[None]
        xs = {}
        for _, m in qlin:
            K = m["weight"].shape[1] * 32 // m.bits
            if K not in xs:
                xs[K] = mx.random.normal((M, K)).astype(mx.bfloat16)
        mx.eval(list(xs.values()))

        def fwd():
            out = model(toks, cache=cache)
            mx.eval(out)
            restore(M)

        def mm():
            outs = []
            for _, m in qlin:
                K = m["weight"].shape[1] * 32 // m.bits
                outs.append(m(xs[K]))
            mx.eval(outs)

        for what, fn in (("forward", fwd), ("matmuls", mm)):
            times = {arm: [] for arm in arms}
            routed = {arm: 0 for arm in arms}
            for arm in arms:
                set_arm(arm)
                for _ in range(2):
                    fn()
            for r in range(a.reps):
                for arm in (arms if r % 2 == 0 else arms[::-1]):
                    set_arm(arm)
                    before = sp_qmm.STATS["routed"]
                    t0 = time.perf_counter()
                    fn()
                    times[arm].append(time.perf_counter() - t0)
                    routed[arm] += sp_qmm.STATS["routed"] - before
            for arm in arms:
                emit({"kind": what, "m": M, "arm": arm,
                      "ms_median": round(statistics.median(times[arm]) * 1e3, 3),
                      "ms_min": round(min(times[arm]) * 1e3, 3),
                      "routed_per_call": routed[arm] / a.reps})
            set_arm("stock")
        mx.clear_cache()
    emit({"kind": "swap", **guard.report(), "peak_gb": round(mx.get_peak_memory() / 1e9, 2)})


if __name__ == "__main__":
    main()

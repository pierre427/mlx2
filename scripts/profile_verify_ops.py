#!/usr/bin/env python3
"""Per-op serial profile of one Qwen3.8-27B verify forward against row count.

GPU only (``--i-own-the-gpu``). Wraps the model's building blocks so every
call is evaluated and timed on its own (exclusive time: a wrapper subtracts
the time of wrapped calls nested inside it):

- ``matmul:<proj>`` every QuantizedLinear, grouped by module name;
- ``gdn_core`` the gated-delta recurrence (``gated_delta_update``);
- ``gdn_conv`` the depthwise conv1d; ``gdn_other`` the rest of the GDN mixer
  (masking, concat, split, q/k norm, gated RMSNorm);
- ``attn_other`` the full-attention mixer minus its projections (q/k norm,
  RoPE, KV update, SDPA);
- ``mlp_other`` SwiGLU; ``norm`` the decoder RMSNorms;
- ``unwrapped`` the forward's remaining time (embedding, residual adds, host).

A per-call ``mx.eval`` adds a fixed sync cost per wrapped call (reported as
``sync_floor_us`` from an empty eval), so absolute numbers are inflated but
the per-row slope of each category is what matters. The plain forward time
(no per-op syncs) is reported alongside. Arms: ``stock`` and ``sp``
(sp_qmm routed with its measured policy).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
M27 = Path("~/mlx-models/Qwen3.8-27B-oQ4e-mtp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-own-the-gpu", action="store_true")
    ap.add_argument("--model", type=Path, default=M27)
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--ms", default="1,2,3,4,6,8,12,16")
    ap.add_argument("--arms", default="stock")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        raise SystemExit("refusing Metal execution without --i-own-the-gpu")
    from sp_qmm_guard import SwapGuard
    guard = SwapGuard()
    import mlx.core as mx
    import mlx.nn as nn
    from mlx2.adapters.registry import resolve_adapter
    from mlx2.runtime.models import sp_qmm, qwen3_5, qwen38_27b, qwen3_next
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
    names = {id(m): p.split(".")[-1] for p, m in model.named_modules()}

    acc = defaultdict(float)
    counts = defaultdict(int)
    stack = []
    enabled = [False]

    def timed(category_fn, fn):
        def wrapper(*args, **kw):
            if not enabled[0]:
                return fn(*args, **kw)
            stack.append(0.0)
            t0 = time.perf_counter()
            out = fn(*args, **kw)
            mx.eval(out)
            dt = time.perf_counter() - t0
            child = stack.pop()
            cat = category_fn(args)
            acc[cat] += dt - child
            counts[cat] += 1
            if stack:
                stack[-1] += dt
            return out
        return wrapper

    # Wrap at class level; the sp subclass is wrapped separately because its
    # __call__ overrides QuantizedLinear's (its stock fallback is then nested
    # and counted once, as the child).
    lin_cat = lambda args: "matmul:" + names.get(id(args[0]), "?")  # noqa: E731
    nn.QuantizedLinear.__call__ = timed(lin_cat, nn.QuantizedLinear.__call__)
    sp_qmm._SpQuantizedLinear.__call__ = timed(lin_cat, sp_qmm._SpQuantizedLinear.__call__)
    gdn = qwen3_5.GatedDeltaNet
    gdn.__call__ = timed(lambda args: "gdn_other", gdn.__call__)
    gdn._gated_delta_update = timed(lambda args: "gdn_core", gdn._gated_delta_update)
    attn = qwen38_27b.Qwen3NextAttention
    attn.__call__ = timed(lambda args: "attn_other", attn.__call__)
    mlp = qwen3_next.Qwen3NextMLP
    mlp.__call__ = timed(lambda args: "mlp_other", mlp.__call__)
    conv_cls = type(model.layers[0].linear_attn.conv1d)
    conv_cls.__call__ = timed(lambda args: "gdn_conv", conv_cls.__call__)
    nn.RMSNorm.__call__ = timed(lambda args: "norm", nn.RMSNorm.__call__)

    text = " ".join(p.read_text(errors="ignore") for p in sorted((ROOT / "docs").glob("*.md"))[:8])
    ids = adapter.tokenizer.encode(text)
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

    handle = None

    def set_arm(arm):
        nonlocal handle
        if handle is not None:
            sp_qmm.remove(handle); handle = None
        if arm == "sp":
            handle = sp_qmm.apply(model)

    z = mx.zeros((1,))
    mx.eval(z)
    t = []
    for _ in range(200):
        t0 = time.perf_counter(); mx.eval(z + 1); t.append(time.perf_counter() - t0)
    emit({"kind": "start", "model": str(a.model), "mlx": mx.__version__,
          "context": a.context, "sync_floor_us": round(statistics.median(t) * 1e6, 1)})
    guard.timed()
    for M in [int(m) for m in a.ms.split(",")]:
        toks = mx.array(ids[a.context: a.context + M])[None]
        for arm in a.arms.split(","):
            set_arm(arm)
            plain, per = [], []
            for r in range(a.reps + 2):
                enabled[0] = False
                t0 = time.perf_counter()
                mx.eval(model(toks, cache=cache))
                dt_plain = time.perf_counter() - t0
                restore(M)
                acc.clear(); counts.clear()
                enabled[0] = True
                t0 = time.perf_counter()
                mx.eval(model(toks, cache=cache))
                dt = time.perf_counter() - t0
                enabled[0] = False
                restore(M)
                if r >= 2:
                    plain.append(dt_plain)
                    cats = dict(acc)
                    cats["unwrapped"] = dt - sum(acc.values())
                    per.append((dt, cats, dict(counts)))
            med = {}
            for cat in per[0][1]:
                med[cat] = round(statistics.median(p[1].get(cat, 0.0) for p in per) * 1e3, 3)
            emit({"kind": "ops", "m": M, "arm": arm,
                  "plain_forward_ms": round(statistics.median(plain) * 1e3, 3),
                  "synced_forward_ms": round(statistics.median(p[0] for p in per) * 1e3, 3),
                  "ms": dict(sorted(med.items(), key=lambda kv: -kv[1])),
                  "calls": per[0][2]})
            set_arm("stock")
        mx.clear_cache()
    emit({"kind": "swap", **guard.report()})


if __name__ == "__main__":
    main()

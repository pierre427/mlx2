#!/usr/bin/env python3
"""End-to-end prefill A/B for the ml-explore/mlx#4983 intake (mask tile skip).

Runs the ordinary in-process prefill of one artifact (adapter model, the
model's own ``make_cache``, chunked at ``--chunk`` rows as the scheduler
slices prompts) followed by a short greedy decode, and records per-chunk
prefill time, the greedy token ids and a top-k log-probability window for
every decoded position.  Run once per MLX build (one venv each) into its own
empty directory; ``--compare`` then reports speedup, exact token parity and,
at the first divergent token, the near-tie inputs AGENTS.md asks for (shared
prefix, top-two sets and margins from the already-selected position).

Modes:
* ``b1``     one request; sliding layers use RotatingKVCache, so every chunk
             after the first passes a dense ``[N, W - 1 + N]`` window mask.
* ``b4pad``  four ragged requests (lengths C, .75C, .5C, .25C) left padded in
             one batch (BatchKVCache / BatchRotatingKVCache with nonzero left
             padding), so full-attention layers pass ``[4, 1, N, offset + N]``
             causal-plus-padding masks too.

Not a publishable performance run: no thermal hold.  Launch inside an owned
GPU window (scripts/amoeba_owned.py).
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _corpus_ids(tok, n_tokens: int) -> list[int]:
    """Deterministic natural prose/code (the repo's own docs), no repetition
    of one string: concatenated until the token budget is met."""
    files = sorted((ROOT / "docs").rglob("*.md"))
    ids: list[int] = []
    for f in files:
        text = f.read_text(errors="ignore")
        ids.extend(tok.encode(text))
        if len(ids) >= n_tokens:
            break
    if len(ids) < n_tokens:
        raise SystemExit("corpus too short")
    return ids[:n_tokens]


def _call_kwargs(model):
    params = inspect.signature(model.__call__).parameters
    if "logits_to_keep" in params or any(p.kind == p.VAR_KEYWORD for p in params.values()):
        # Gemma 4 forwards logits_to_keep through **kwargs.
        if type(model).__module__.endswith("gemma4"):
            return {"logits_to_keep": 1}
    if "tail_row" in params:
        return {"tail_row": True}
    return {}


def _batch_caches(model, left_padding):
    from mlx2.runtime.models.cache import (BatchKVCache, BatchRotatingKVCache, KVCache,
                                           RotatingKVCache)
    out = []
    for c in model.make_cache():
        if type(c) is KVCache:
            out.append(BatchKVCache(list(left_padding)))
        elif type(c) is RotatingKVCache:
            if c.keep:
                raise SystemExit("b4pad: rotating cache with keep > 0 is not covered")
            out.append(BatchRotatingKVCache(c.max_size, list(left_padding)))
        else:
            raise SystemExit(f"b4pad: unsupported cache {type(c).__name__}")
    return out


def _topk(mx, logits, k=8):
    lp = logits.astype(mx.float32)
    lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
    idx = mx.argsort(-lp, axis=-1)[..., :k]
    vals = mx.take_along_axis(lp, idx, axis=-1)
    return idx.tolist(), vals.tolist()


def run(a) -> int:
    import mlx.core as mx

    sys.path.insert(0, str(ROOT / "src"))
    from mlx2.adapters.registry import inspect_model

    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit("refusing nonempty evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ad = inspect_model(a.model).adapter_type(str(a.model))
    model, tok = ad.model, ad.tokenizer
    checkpoint = f"{a.name or Path(a.model).name}@{ad.identity['fingerprint']}"
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                         text=True).stdout.strip()
    kw = _call_kwargs(model)
    report = {"checkpoint": checkpoint, "mlx2_commit": rev, "mlx_version": mx.__version__,
              "mode": a.mode, "chunk": a.chunk, "decode": a.decode, "call_kwargs": sorted(kw),
              "load_s": round(time.time() - t0, 1), "cells": []}
    print(f"{checkpoint} mlx {mx.__version__} rev {rev[:9]} kwargs {kw}", flush=True)
    swap0 = _swapouts()
    for ctx in a.contexts:
        ids = _corpus_ids(tok, ctx)
        for rep in range(a.reps):
            if a.mode == "b1":
                rows = [ids]
                cache = model.make_cache()
                padded = mx.array(rows)
            else:
                lengths = [ctx, ctx * 3 // 4, ctx // 2, ctx // 4]
                rows = [ids[ctx - n:] for n in lengths]  # four different suffixes
                left = [ctx - n for n in lengths]
                cache = _batch_caches(model, left)
                padded = mx.array([[0] * p + r for p, r in zip(left, rows)])
            chunk_ms = []
            logits = None
            for s in range(0, ctx, a.chunk):
                x = padded[:, s:s + a.chunk]
                ts = time.perf_counter()
                logits = model(x, cache=cache, **kw)[:, -1, :]
                mx.eval(logits, [c.state for c in cache])
                mx.synchronize()
                chunk_ms.append((time.perf_counter() - ts) * 1e3)
            tokens, windows = [[] for _ in rows], [[] for _ in rows]
            for _ in range(a.decode):
                nxt = mx.argmax(logits, axis=-1)
                ti, tv = _topk(mx, logits)
                for b in range(len(rows)):
                    tokens[b].append(int(nxt[b].item()))
                    windows[b].append({"ids": ti[b], "logprobs": [round(v, 5) for v in tv[b]]})
                logits = model(nxt[:, None], cache=cache, **kw)[:, -1, :]
                mx.eval(logits)
            cell = {"context": ctx, "rep": rep, "prefill_ms": sum(chunk_ms), "chunk_ms": chunk_ms,
                    "tokens": tokens, "windows": windows,
                    "text": [tok.decode(t) for t in tokens]}
            report["cells"].append(cell)
            print(f"ctx {ctx} rep {rep} prefill {cell['prefill_ms']:.0f} ms "
                  f"chunks {[round(c) for c in chunk_ms]} text0 {cell['text'][0][:50]!r}", flush=True)
            del cache, logits
            mx.clear_cache()
    report["swapouts_delta"] = _swapouts() - swap0
    from mlx2.runtime.technologies import window_attention as wa
    report["native_window_env"] = os.environ.get("MLX2_SDPA_NATIVE_WINDOW", "0")
    report["native_window_stats"] = dict(wa.NATIVE_WINDOW_STATS)
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    return 0


def _swapouts() -> int:
    try:
        text = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        for line in text.splitlines():
            if line.startswith("Swapouts"):
                return int(line.split(":")[1].strip().rstrip("."))
    except Exception:  # noqa: BLE001
        pass
    return -1


def _first_divergence(ta, tb, wa, wb):
    for i, (x, y) in enumerate(zip(ta, tb)):
        if x != y:
            def top2(w):
                return set(w["ids"][:2]), w["logprobs"][0] - w["logprobs"][1]
            sa, ma = top2(wa[i])
            sb, mb = top2(wb[i])
            near = sa == sb and ma <= 0.5 and mb <= 0.5
            return {"position": i, "base": x, "cand": y, "top2_same": sa == sb,
                    "margin_base": round(ma, 4), "margin_cand": round(mb, 4),
                    "near_tie_inputs_ok": near}
    return None


def compare(paths) -> int:
    base, cand = (json.loads(Path(p, "report.json").read_text()) for p in paths)
    print(f"{base['checkpoint']}  mode {base['mode']}  mlx2 {base['mlx2_commit'][:9]}")
    print(f"base mlx {base['mlx_version']}  cand mlx {cand['mlx_version']}")
    by_ctx = {}
    for name, rep in (("base", base), ("cand", cand)):
        for c in rep["cells"]:
            by_ctx.setdefault(c["context"], {}).setdefault(name, []).append(c)
    for ctx, arms in sorted(by_ctx.items()):
        b = min(c["prefill_ms"] for c in arms["base"])
        k = min(c["prefill_ms"] for c in arms["cand"])
        tb, tk = arms["base"][0], arms["cand"][0]
        exact = tb["tokens"] == tk["tokens"]
        div = None if exact else [
            _first_divergence(x, y, wx, wy)
            for x, y, wx, wy in zip(tb["tokens"], tk["tokens"], tb["windows"], tk["windows"])]
        print(f"ctx {ctx:6d}  base {b:9.0f} ms  cand {k:9.0f} ms  x{b / k:5.2f}  "
              f"tokens_exact {exact}" + ("" if exact else f"  first_div {div}"))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--name", default="")
    ap.add_argument("--out")
    ap.add_argument("--mode", choices=("b1", "b4pad"), default="b1")
    ap.add_argument("--contexts", type=int, nargs="+", default=[4096, 8192, 16384])
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--decode", type=int, default=16)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--compare", nargs=2, metavar=("BASE_DIR", "CAND_DIR"))
    a = ap.parse_args()
    if a.compare:
        return compare(a.compare)
    if not (a.model and a.out):
        ap.error("--model and --out are required")
    return run(a)


if __name__ == "__main__":
    raise SystemExit(main())

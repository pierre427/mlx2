#!/usr/bin/env python3
"""Lane matmul spot check for one model: coverage, cost and verify quality.

Loads the model through the registry adapter exactly as the server does,
installs the lane matmul in its default crossover mode (``min_rows=4``), and
at each context length (real text from the frozen Spomin corpus):

* cost: one forward of 1/4/8/16 rows, stock versus lane (alternating);
* quality: from one snapshot, 16 one-token stock steps (what decode would
  produce) versus a 16-row stock forward and a 16-row lane forward, reporting
  top-1 agreement and max |log-prob difference| against the one-token rows.
  Lane passes when it is no worse than stock's own multi-row kernel.

Writes one JSON receipt; failures are recorded, not raised.
usage: lane_model_sweep.py <artifact> <out.json> [ctx,ctx]
"""

from __future__ import annotations

import copy
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def corpus_text() -> str:
    """Real English/markdown text: the repository's docs in a fixed order."""
    files = sorted((ROOT / "docs").rglob("*.md"))
    return "\n\n".join(f.read_text(errors="ignore") for f in files)


def encode(adapter, text: str) -> list[int]:
    tok = getattr(adapter, "tokenizer", None)
    for candidate in (tok, getattr(tok, "_tokenizer", None), getattr(tok, "tokenizer", None)):
        if candidate is None or not hasattr(candidate, "encode"):
            continue
        try:
            ids = candidate.encode(text, add_special_tokens=False)
        except TypeError:
            ids = candidate.encode(text)
        ids = list(getattr(ids, "ids", ids))
        if ids:
            return [int(i) for i in ids]
    raise RuntimeError("no usable tokenizer on the adapter")


def forward_fn(model):
    """(callable(tokens [1, L], cache) -> logits [1, L, V], make_cache)."""
    candidates = [model, getattr(model, "language_model", None)]
    # Multimodal wrappers name their LLM differently (llm, text_model, ...).
    children = getattr(model, "children", None)
    if callable(children):
        candidates += [child for child in children().values() if hasattr(child, "make_cache")]
    for target in candidates:
        if target is None or not hasattr(target, "make_cache"):
            continue

        def call(x, cache, target=target):
            out = target(x, cache=cache)
            return getattr(out, "logits", out)

        return call, target.make_cache
    # Last resort, as serving does: the generic per-layer prompt cache.
    from mlx2.runtime.models.cache import make_prompt_cache

    def call(x, cache):
        out = model(x, cache=cache)
        return getattr(out, "logits", out)

    return call, lambda: make_prompt_cache(model)


def timed_forward(call, cache, tokens):
    x = mx.array([tokens], dtype=mx.uint32)
    t0 = time.perf_counter()
    y = call(x, cache)
    mx.eval(y, [c.state for c in cache if hasattr(c, "state")])
    return (time.perf_counter() - t0) * 1000, y


def logprobs(y):
    y = y.astype(mx.float32)
    return y - mx.logsumexp(y, axis=-1, keepdims=True)


def main():
    art, out = sys.argv[1], Path(sys.argv[2])
    ctxs = [int(c) for c in (sys.argv[3] if len(sys.argv) > 3 else "8192,32768").split(",")]
    rec = {"artifact": art, "contexts": {}, "errors": []}
    t_load = time.perf_counter()
    try:
        from mlx2.adapters.registry import inspect_model, resolve_adapter
        from mlx2.runtime import lane

        resolution = inspect_model(art)
        rec["family"] = resolution.descriptor.family
        adapter = resolve_adapter(art, qualification_mode=True)(art)
        model = adapter.model
        rec["load_seconds"] = round(time.perf_counter() - t_load, 1)
        mx.set_cache_limit(8 << 30)
        rec["install"] = lane.install(model)          # the proposed default
        call, make_cache = forward_fn(model)
        text_ids = encode(adapter, corpus_text())
        max_ctx = int(getattr(adapter, "max_context", 0) or 0)
    except Exception as exc:  # noqa: BLE001 - one receipt per model, always
        rec["errors"].append({"stage": "load", "error": repr(exc)[:600],
                              "trace": traceback.format_exc()[-1500:]})
        out.write_text(json.dumps(rec, indent=1))
        print(json.dumps({"artifact": art, "load_error": repr(exc)[:300]}), flush=True)
        return
    while len(text_ids) < max(ctxs) + 64:
        text_ids = text_ids + text_ids
    for ctx in ctxs:
        if max_ctx and ctx + 64 > max_ctx:
            ctx = max_ctx - 64
        row = {"tokens": ctx}
        try:
            lane.set_enabled(False)
            cache = make_cache()
            t0 = time.perf_counter()
            for s in range(0, ctx, 2048):
                y = call(mx.array([text_ids[s:min(s + 2048, ctx)]], dtype=mx.uint32), cache)
                mx.eval(y, [c.state for c in cache if hasattr(c, "state")])
            row["prefill_seconds"] = round(time.perf_counter() - t0, 2)
            verify = text_ids[ctx:ctx + 16]
            base = copy.deepcopy(cache)
            # quality: one-token stock steps are the reference
            ref_cache = copy.deepcopy(base)
            serial = []
            for tok in verify:
                _, y = timed_forward(call, ref_cache, [tok])
                serial.append(logprobs(y[0, -1]))
            serial = mx.stack(serial)
            quality = {}
            for arm, on in (("stock_16row", False), ("lane_16row", True)):
                lane.set_enabled(on)
                arm_cache = copy.deepcopy(base)
                _, y = timed_forward(call, arm_cache, verify)
                lp = logprobs(y[0])
                top_agree = int(mx.sum(mx.argmax(lp, -1) == mx.argmax(serial, -1)).item())
                quality[arm] = {"top1_agree_of_16": top_agree,
                                "max_abs_logprob_diff": round(float(mx.max(mx.abs(lp - serial)).item()), 4)}
                del arm_cache
            lane.set_enabled(False)
            row["quality"] = quality
            del ref_cache, base
            # cost: cache grows by a few hundred tokens over the whole loop
            cost = {}
            for rows in (1, 4, 8, 16):
                samples = {"stock": [], "lane": []}
                for r in range(4):
                    for arm in (("stock", "lane") if r % 2 == 0 else ("lane", "stock")):
                        lane.set_enabled(arm == "lane")
                        ms, _ = timed_forward(call, cache, verify[:rows])
                        if r:
                            samples[arm].append(ms)
                cost[rows] = {k: round(statistics.median(v), 2) for k, v in samples.items()}
            lane.set_enabled(False)
            row["forward_ms"] = cost
            del cache
            mx.clear_cache()
        except Exception as exc:  # noqa: BLE001
            row["error"] = repr(exc)[:600]
            row["trace"] = traceback.format_exc()[-1500:]
            lane.set_enabled(False)
            mx.clear_cache()
        rec["contexts"][ctx] = row
        print(json.dumps({"artifact": Path(art).name, "ctx": ctx,
                          **{k: row.get(k) for k in ("quality", "forward_ms", "error")}}), flush=True)
    rec["peak_memory_gb"] = round(mx.get_peak_memory() / 2**30, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1) + "\n")


if __name__ == "__main__":
    main()

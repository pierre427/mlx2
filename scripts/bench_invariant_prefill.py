"""Prefill and decode cost of the opt-in invariant prefill lane (Metal).

One process, one load with ``{"invariant_prefill": true}``.  Arms:

* ``stock`` -- the lane uninstalled (stock classes, stock kernels);
* ``lane``  -- the lane installed and enabled.

Prefill: a fresh-cache prefill of one real ``--prompt-tokens`` prompt in
fixed slices (``--slices``), timed to a synchronised end.  Decode: B1
ordinary greedy decode of ``--decode`` tokens after a short prompt (the lane
never opens on a one-row forward, so this measures only its installed
overhead).  Arms alternate ABBA per rep after one discarded warm-up of each.

  MLX_ENABLE_TF32=0 PYTHONPATH=src .venv/bin/python scripts/bench_invariant_prefill.py \\
      --i-own-the-gpu --model ~/mlx-models/Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP \\
      --out bench.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=8192)
    ap.add_argument("--slices", nargs="+", type=int, default=[512, 2048])
    ap.add_argument("--decode", type=int, default=128)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--policy", default="{}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")

    from mlx2.adapters.registry import resolve_adapter

    policy = {**json.loads(a.policy), "invariant_prefill": True}
    adapter = resolve_adapter(a.model, mtp=True, qualification_mode=True)(
        a.model, execution_policy=policy
    )
    import mlx.core as mx
    from mlx2.runtime.models import invariant_prefill as inv

    model = adapter.model
    trunk = model.language_model.model
    state = {"handle": adapter.invariant_prefill}
    mx.eval(model.parameters())
    mx.set_cache_limit(4 << 30)
    tok = adapter.tokenizer
    corpus = (ROOT / "docs" / "SERVING.md").read_text()
    ids = tok.encode(corpus, add_special_tokens=False)[: a.prompt_tokens - 64]
    prompt = list(adapter.prompt_tokens({"messages": [{"role": "user", "content":
                  "Summarise the following excerpt in three sentences.\n\n" + tok.decode(ids)}]}))
    short = list(adapter.prompt_tokens({"messages": [{"role": "user", "content":
                 "Write a long story about a lighthouse keeper."}]}))

    def set_arm(arm):
        handle = state["handle"]
        if arm == "stock" and handle.installed:
            handle.uninstall()
        elif arm == "lane" and not handle.installed:
            state["handle"] = inv.install(trunk)
            assert state["handle"].installed, state["handle"].refusal
        expect = inv.InvariantQuantizedLinear if arm == "lane" else __import__("mlx.nn").nn.QuantizedLinear
        assert type(trunk.layers[0].mlp.gate) is expect

    def prefill(step):
        cache = model.make_cache()
        mx.synchronize()
        t0 = time.perf_counter()
        pos = 0
        while pos < len(prompt):
            end = min(len(prompt), pos + step)
            hidden = trunk(mx.array([prompt[pos:end]], mx.uint32), cache)
            mx.eval(hidden, [c.state for c in cache])
            pos = end
        mx.synchronize()
        dt = time.perf_counter() - t0
        logits = model.logits(hidden[:, -1:, :])
        first = int(mx.argmax(logits[0, -1]).item())
        del cache
        mx.clear_cache()
        return dt, first

    def decode():
        cache = model.make_cache()
        logits = model(mx.array([short], mx.uint32), cache=cache)
        nxt = mx.argmax(logits[:, -1, :], -1)
        mx.eval(nxt)
        out = []
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(a.decode):
            logits = model(nxt.reshape(1, 1), cache=cache)
            nxt = mx.argmax(logits[:, -1, :], -1)
            mx.async_eval(nxt)
            out.append(nxt)
        mx.eval(out)
        dt = time.perf_counter() - t0
        tokens = [int(t.item()) for t in out]
        del cache
        mx.clear_cache()
        return a.decode / dt, tokens

    results = {"model": a.model, "prompt_tokens": len(prompt), "mlx": mx.__version__,
               "runs": []}
    for arm in ("stock", "lane"):  # warm-up, discarded
        set_arm(arm)
        for step in a.slices:
            prefill(step)
        decode()
    for rep in range(a.reps):
        order = ("stock", "lane") if rep % 2 == 0 else ("lane", "stock")
        for arm in order:
            set_arm(arm)
            row = {"rep": rep, "arm": arm}
            for step in a.slices:
                dt, first = prefill(step)
                row[f"prefill_{step}_s"] = round(dt, 4)
                row[f"first_{step}"] = first
            rate, tokens = decode()
            row["decode_tok_s"] = round(rate, 3)
            row["decode_tokens"] = tokens
            results["runs"].append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "decode_tokens"}), flush=True)
    summary = {}
    for arm in ("stock", "lane"):
        rows = [r for r in results["runs"] if r["arm"] == arm]
        summary[arm] = {k: statistics.median(r[k] for r in rows)
                        for k in rows[0] if k.startswith("prefill_") or k == "decode_tok_s"}
    for key in summary["stock"]:
        summary.setdefault("lane_vs_stock_pct", {})[key] = round(
            100 * (summary["lane"][key] / summary["stock"][key] - 1), 2)
    stock_tokens = {tuple(r["decode_tokens"]) for r in results["runs"] if r["arm"] == "stock"}
    lane_tokens = {tuple(r["decode_tokens"]) for r in results["runs"] if r["arm"] == "lane"}
    summary["decode_tokens_identical"] = stock_tokens == lane_tokens and len(stock_tokens) == 1
    results["summary"] = summary
    results["lane_counters"] = inv.status()
    print("SUMMARY", json.dumps(summary), flush=True)
    json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

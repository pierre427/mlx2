"""In-process prefill A/B for Flash-Next: one model load, arms rotated per rep.

Arms are ``name:step`` with name in stock | gdn (fused GDN prefill) |
gdnw (fused GDN + MoE weighted sum); step is the prefill chunk size. Each rep
prefills the same prompt from an empty cache, timing to the last chunk's
evaluated logits. The final-position logits of every arm are compared with
the first rep of ``stock:2048`` (bit-identical / max |diff| / top-1 agree).

  PYTHONPATH=src .venv/bin/python scripts/bench_prefill_inproc.py --i-own-the-gpu \
      --model ~/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP --prompt-file p.txt \
      --context 16000 --arms stock:2048 gdn:2048 gdnw:2048 --reps 4 --out r.json
"""

import argparse
import json
import time

import mlx.core as mx
from mlx.utils import tree_flatten


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--context", type=int, default=16000)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")

    from mlx2.adapters.registry import resolve_adapter
    from mlx2.runtime.models.cache import make_prompt_cache

    adapter = resolve_adapter(a.model, mtp=True)(a.model)
    model = adapter.model
    mx.eval(model.parameters())
    mx.set_cache_limit(8 << 30)
    ids = list(adapter.tokenizer.encode(open(a.prompt_file).read()))[: a.context]
    tokens = mx.array(ids, mx.uint32)[None]
    gdn = [m for _, m in model.named_modules() if hasattr(m, "set_fused_gdn_prefill_mode")]
    moe = [m for _, m in model.named_modules() if hasattr(m, "set_moe_weighted_sum")]
    print("LOADED", len(gdn), "gdn", len(moe), "moe", flush=True)

    def configure(name):
        for m in gdn:
            m.set_fused_gdn_prefill_mode("fused" if name in ("gdn", "gdnw") else "stock")
        for m in moe:
            m.set_moe_weighted_sum(name == "gdnw")

    def prefill(step):
        cache = make_prompt_cache(model)
        n = tokens.shape[1]
        start = time.perf_counter()
        pos = 0
        while pos < n:
            chunk = tokens[:, pos : pos + step]
            logits = model(chunk, cache=cache)
            mx.eval(logits, [v for _, v in tree_flatten([getattr(c, 'state', None) for c in cache]) if isinstance(v, mx.array)])
            pos += chunk.shape[1]
        elapsed = time.perf_counter() - start
        last = logits[0, -1].astype(mx.float32)
        del cache
        mx.clear_cache()
        return elapsed, last

    def counters():
        out = {}
        for m in gdn + moe:
            for k, v in vars(m).items():
                if (k.startswith("fused_gdn_prefill") or k.startswith("moe_weighted_sum")) and isinstance(v, int):
                    out[k] = out.get(k, 0) + v
        return out

    arms = [tuple(x.split(":")) for x in a.arms]
    results = {x: [] for x in a.arms}
    compare = {}
    reference = None
    configure("stock")
    prefill(2048)  # warm-up: kernels JIT, PLE pages, allocator
    for rep in range(a.reps):
        order = arms[rep % len(arms):] + arms[: rep % len(arms)]
        for name, step in order:
            configure(name)
            before = counters()
            elapsed, last = prefill(int(step))
            after = counters()
            key = f"{name}:{step}"
            results[key].append(elapsed)
            if reference is None and key == "stock:2048":
                reference = last
            if reference is not None and key not in compare:
                diff = mx.abs(last - reference).max().item()
                compare[key] = {
                    "bit_identical": bool(mx.array_equal(last, reference).item()),
                    "max_abs_logit_diff": diff,
                    "top1_agree": bool((mx.argmax(last) == mx.argmax(reference)).item()),
                    "counter_delta": {k: after.get(k, 0) - before.get(k, 0) for k in after if after.get(k, 0) != before.get(k, 0)},
                }
            print(f"rep{rep} {key} {elapsed:.2f}s {tokens.shape[1] / elapsed:.0f} tok/s", flush=True)
    rec = {"context": tokens.shape[1], "reps": a.reps, "seconds": results,
           "tok_s_median": {k: tokens.shape[1] / sorted(v)[len(v) // 2] for k, v in results.items()},
           "compare_vs_stock_2048": compare, "peak_gib": mx.get_peak_memory() / 2**30,
           "mlx": mx.__version__}
    json.dump(rec, open(a.out, "w"), indent=1)
    print(json.dumps({k: rec[k] for k in ("tok_s_median", "compare_vs_stock_2048")}, indent=1))


if __name__ == "__main__":
    main()

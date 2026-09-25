"""Teacher-forced quality gate for prefill arms on Flash-Next (one process).

Each arm ``name:step`` (name: stock | gdn | gdnw) prefills --context tokens
from an empty cache, then scores the next --score-tokens tokens in one
forward. Logprobs are compared with ``stock:<context>`` (one chunk, the
unchunked ground truth) and with ``stock:2048`` (the current default):
mean/max KL, top-1 agreement, max |delta logit|.

  PYTHONPATH=src .venv/bin/python scripts/score_prefill_arms.py --i-own-the-gpu \
      --model ~/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP --prompt-file p.txt \
      --context 16384 --arms stock:2048 stock:8192 gdnw:2048 gdn:8192 --out q.json
"""

import argparse
import json

import mlx.core as mx
from mlx.utils import tree_flatten


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--context", type=int, default=16384)
    ap.add_argument("--score-tokens", type=int, default=256)
    ap.add_argument("--offsets", type=int, nargs="+", default=[0, 20000, 40000])
    ap.add_argument("--arms", nargs="+", required=True)
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
    all_ids = list(adapter.tokenizer.encode(open(a.prompt_file).read()))
    gdn = [m for _, m in model.named_modules() if hasattr(m, "set_fused_gdn_prefill_mode")]
    moe = [m for _, m in model.named_modules() if hasattr(m, "set_moe_weighted_sum")]

    def configure(name):
        for m in gdn:
            m.set_fused_gdn_prefill_mode("fused" if name in ("gdn", "gdnw") else "stock")
        for m in moe:
            m.set_moe_weighted_sum(name == "gdnw")

    def run(ids, step):
        prompt = mx.array(ids[: a.context], mx.uint32)[None]
        tail = mx.array(ids[a.context : a.context + a.score_tokens], mx.uint32)[None]
        cache = make_prompt_cache(model)
        pos = 0
        while pos < prompt.shape[1]:
            chunk = prompt[:, pos : pos + step]
            out = model(chunk, cache=cache)
            mx.eval(out, [v for _, v in tree_flatten([getattr(c, "state", None) for c in cache]) if isinstance(v, mx.array)])
            pos += chunk.shape[1]
        logits = model(tail, cache=cache)[0].astype(mx.float32)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        mx.eval(logprobs)
        del cache
        mx.clear_cache()
        return logprobs

    def compare(p, q):
        kl = (mx.exp(q) * (q - p)).sum(axis=-1)  # KL(ref || arm)
        return {
            "kl_mean": kl.mean().item(), "kl_max": kl.max().item(),
            "top1_agree": (mx.argmax(p, -1) == mx.argmax(q, -1)).astype(mx.float32).mean().item(),
            "max_abs_logprob_diff": mx.abs(p - q).max().item(),
            "bit_identical": bool(mx.array_equal(p, q).item()),
        }

    arms = [("stock", str(a.context)), ("stock", "2048")] + [
        tuple(x.split(":")) for x in a.arms if x not in (f"stock:{a.context}", "stock:2048")]
    report = {}
    for offset in a.offsets:
        ids = all_ids[offset:]
        refs = {}
        for name, step in arms:
            configure(name)
            lp = run(ids, int(step))
            key = f"{name}:{step}"
            if key == f"stock:{a.context}":
                refs["full"] = lp
            elif key == "stock:2048":
                refs["default"] = lp
            report.setdefault(key, []).append({
                "offset": offset,
                "vs_full": compare(lp, refs["full"]),
                "vs_default": compare(lp, refs["default"]) if "default" in refs else None,
            })
            print(offset, key, json.dumps(report[key][-1]), flush=True)
    configure("stock")
    json.dump({"context": a.context, "score_tokens": a.score_tokens, "arms": report,
               "peak_gib": mx.get_peak_memory() / 2**30}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

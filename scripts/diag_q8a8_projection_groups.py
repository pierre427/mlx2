"""Localize int8 (Q8 in-place group64) prefill error by projection group.

One model load.  Teacher-forced next-token distributions on the bench texts
(``bench_q8a8_prefill_e2e.build_texts``) with int8 installed on one projection
group at a time (path-suffix match inside the auto scope), against stock
prefill: mean KL(off||arm), p99 KL, fraction of positions with KL > 1, top-1
agreement.  An ``off`` repeat checks the reference is reproducible.

Optionally ``--layers`` splits one group by layer block to find where the
error enters.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

POLICY = {"enabled": True, "scope": "all", "q8_inplace": True,
          "act_scale": "group64", "q8_only": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--groups", required=True,
                    help="JSON name -> list of path suffixes ('*' = the whole auto scope)")
    ap.add_argument("--texts", nargs="+", default=["tool_calls", "prose", "reasoning", "code"])
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--layer-blocks", type=int, default=0,
                    help="also split each group into this many layer blocks")
    ap.add_argument("--control-chunks", type=int, nargs="*", default=[],
                    help="stock-only controls: the same text prefilled in chunks of "
                    "this many rows (different kernel shapes, exact math otherwise)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")

    import mlx.core as mx

    from mlx2.adapters.registry import resolve_adapter

    adapter = resolve_adapter(a.model, mtp=False)(a.model)
    from bench_q8a8_prefill_e2e import build_texts

    from mlx2.runtime import int8_prefill
    from mlx2.runtime.models.cache import make_prompt_cache

    model = adapter.model
    mx.eval(model.parameters())
    mx.set_cache_limit(6 << 30)
    texts, tool_source = build_texts(adapter.tokenizer, Path(__file__).resolve().parents[1],
                                     a.tokens)
    texts = {k: texts[k] for k in a.texts}
    policy = int8_prefill.Int8PrefillPolicy.from_value(POLICY)
    selector = getattr(adapter, "int8_prefill_select", None)
    base = (selector("all") if callable(selector) else None) or int8_prefill.default_select("all")

    def logprobs(ids, chunk=None):
        cache = make_prompt_cache(model)
        x = mx.array(ids, mx.uint32)[None]
        chunk = chunk or x.shape[1]
        parts = []
        for pos in range(0, x.shape[1], chunk):
            parts.append(model(x[:, pos: pos + chunk], cache=cache)[0])
            mx.eval(parts[-1])
        logits = mx.concatenate(parts) if len(parts) > 1 else parts[0]
        lp = logits.astype(mx.float32)
        lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
        lp = lp.astype(mx.float16)
        mx.eval(lp)
        del cache, logits
        mx.clear_cache()
        return lp

    def compare(ref, lp):
        r32, l32 = ref.astype(mx.float32), lp.astype(mx.float32)
        kl = (mx.exp(r32) * (r32 - l32)).sum(-1)
        top1 = (mx.argmax(ref, -1) == mx.argmax(lp, -1)).astype(mx.float32)
        mx.eval(kl, top1)
        return {"mean_kl": round(kl.mean().item(), 5),
                "p99_kl": round(float(mx.sort(kl)[int(0.99 * kl.size)].item()), 4),
                "frac_kl_gt_1": round((kl > 1).astype(mx.float32).mean().item(), 4),
                "top1": round(top1.mean().item(), 4)}

    def layer_of(path):
        for seg in path.split("."):
            if seg.isdigit():
                return int(seg)
        return -1

    n_layers = 1 + max(layer_of(p) for p, _ in model.named_modules())
    refs = {k: logprobs(v) for k, v in texts.items()}
    results = {"off_repeat": {k: compare(refs[k], logprobs(v)) for k, v in texts.items()}}
    print("off_repeat", results["off_repeat"], flush=True)
    for chunk in a.control_chunks:
        name = f"off_chunk{chunk}"
        results[name] = {k: compare(refs[k], logprobs(v, chunk)) for k, v in texts.items()}
        print(name, results[name], flush=True)

    arms = []
    for name, suffixes in json.loads(a.groups).items():
        arms.append((name, suffixes, None))
        if a.layer_blocks:
            step = -(-n_layers // a.layer_blocks)
            for lo in range(0, n_layers, step):
                arms.append((f"{name}@L{lo}-{min(lo + step, n_layers) - 1}", suffixes,
                             (lo, lo + step)))
    for name, suffixes, block in arms:
        def select(path, module, suffixes=suffixes, block=block):
            if not base(path, module):
                return False
            if block is not None and not block[0] <= layer_of(path) < block[1]:
                return False
            return "*" in suffixes or any(path.endswith(s) for s in suffixes)

        try:
            handle = int8_prefill.apply(model, policy, select=select)
        except int8_prefill.Int8PrefillError as error:
            results[name] = {"error": str(error)}
            print(name, "error", error, flush=True)
            continue
        modules = handle.q8_module_count()
        results[name] = {"modules": modules}
        for k, v in texts.items():
            results[name][k] = compare(refs[k], logprobs(v))
        int8_prefill.remove(handle)
        print(name, results[name], flush=True)
    Path(a.out).write_text(json.dumps(
        {"model": a.model, "adapter": type(adapter).__name__, "policy": POLICY,
         "tokens": a.tokens, "tool_text_source": tool_source, "results": results,
         "mlx": mx.__version__}, indent=1))


if __name__ == "__main__":
    sys.exit(main())

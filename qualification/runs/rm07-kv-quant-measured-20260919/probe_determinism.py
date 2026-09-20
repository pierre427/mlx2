#!/usr/bin/env python3
"""rm07 control: what makes the EXACT path disagree with itself at 16K?

Four exact caches over the same stream, differing only in prefill chunking:
  a  step 2048 (harness reference)
  a2 step 2048 (bit-for-bit repeat of a: isolates run-to-run nondeterminism)
  b  step 1536
  c  step 4096
Teacher-forces the same window and reports, against arm a: top-1 agreement,
mean KL, and max |logit| difference.  If a2 == a exactly while b and c differ,
the disagreement is chunk-boundary accumulation order, not nondeterminism.
Metal: run under the GPU queue only.
"""
import argparse, json, sys, time
from pathlib import Path

W = Path("/private/tmp/mlx2-rm07-kv-quant-measured")
sys.path.insert(0, str(W / "src"))
sys.path.insert(0, str(W / "scripts"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--contexts", default="16384")
    p.add_argument("--score-tokens", type=int, default=256)
    p.add_argument("--out", required=True)
    p.add_argument("--i-own-the-gpu", action="store_true")
    a = p.parse_args()
    if not a.i_own_the_gpu:
        raise SystemExit("refusing Metal without --i-own-the-gpu")
    import mlx.core as mx
    import measure_kv_quant_fidelity as h
    from mlx2.runtime.models.cache import make_prompt_cache

    adapter = h.load_gpu_adapter(a.model)
    model, tok = adapter.model, adapter.tokenizer

    def enc(t):
        try:
            return tok.encode(t, add_special_tokens=False)
        except TypeError:
            return tok.encode(t)

    contexts = [int(x) for x in a.contexts.split(",")]
    ids = h.token_stream(enc, h.corpus_text(None), max(contexts) + a.score_tokens + 1)
    stream = mx.array([ids], dtype=mx.uint32)
    steps = {"a": 2048, "a2": 2048, "b": 1536, "c": 4096}
    out = {"model": a.model, "contexts": []}
    for ctx in contexts:
        t0 = time.perf_counter()
        caches = {k: list(make_prompt_cache(model)) for k in steps}
        for arm, cache in caches.items():
            s = steps[arm]
            for start in range(0, ctx, s):
                mx.eval(model(stream[:, start:min(start + s, ctx)], cache=cache))
        acc = {k: {"agree": 0, "kl": 0.0, "max_abs_logit_delta": 0.0} for k in steps if k != "a"}
        for pos in range(ctx, ctx + a.score_tokens):
            step = stream[:, pos:pos + 1]
            lg = {k: model(step, cache=c)[0, -1].astype(mx.float32) for k, c in caches.items()}
            mx.eval(lg)
            ref = lg["a"]
            lref = ref - mx.logsumexp(ref)
            pref = mx.exp(lref)
            top = int(mx.argmax(ref).item())
            for k in acc:
                o = lg[k]
                lo = o - mx.logsumexp(o)
                acc[k]["agree"] += int(mx.argmax(o).item() == top)
                acc[k]["kl"] += float(mx.maximum(mx.sum(pref * (lref - lo)), 0).item())
                acc[k]["max_abs_logit_delta"] = max(
                    acc[k]["max_abs_logit_delta"], float(mx.max(mx.abs(o - ref)).item()))
        n = a.score_tokens
        entry = {"context": ctx, "prefill_steps": steps, "scored_tokens": n,
                 "arms": {k: {"top1_agreement": v["agree"] / n, "kl_mean": v["kl"] / n,
                              "max_abs_logit_delta": v["max_abs_logit_delta"]}
                          for k, v in acc.items()},
                 "wall_s": time.perf_counter() - t0}
        out["contexts"].append(entry)
        print(json.dumps(entry), flush=True)
        del caches
        mx.clear_cache()
    Path(a.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""rm07 diagnostic: where does kv_q8 flip top-1, and what is the exact arm's own floor?

Per context: three caches over the same token stream as the fidelity harness
(repo docs/*.md, same tokenizer path):
  exact   prefill_step 2048   (the harness reference)
  exact_b prefill_step 1536   (an exact control: only chunk boundaries move)
  quant   kv_q8, prefill_step 2048 (the harness quantized arm)
Teacher-forced over the same 256 positions; per position records the exact
top-1/top-2 log-prob margin, agreement and KL for both comparisons, and the
tokens at disagreements.  Metal: run under the GPU queue only.
"""
import argparse, json, sys, time
from pathlib import Path

W = Path("/private/tmp/mlx2-rm07-kv-quant-measured")
sys.path.insert(0, str(W / "src"))
sys.path.insert(0, str(W / "scripts"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--contexts", default="4096,16384")
    p.add_argument("--score-tokens", type=int, default=256)
    p.add_argument("--operation", default="kv_q8")
    p.add_argument("--out", required=True)
    p.add_argument("--i-own-the-gpu", action="store_true")
    a = p.parse_args()
    if not a.i_own_the_gpu:
        raise SystemExit("refusing Metal without --i-own-the-gpu")
    import mlx.core as mx
    import measure_kv_quant_fidelity as h
    from mlx2.runtime.approximate_kv import LaneKVState, declared_operations, quantized_plane_count
    from mlx2.runtime.models.cache import make_prompt_cache

    adapter = h.load_gpu_adapter(a.model)
    model, tok = adapter.model, adapter.tokenizer
    op = declared_operations(adapter, adapter_fingerprint=adapter.identity["fingerprint"])[a.operation]

    def enc(t):
        try:
            return tok.encode(t, add_special_tokens=False)
        except TypeError:
            return tok.encode(t)

    contexts = [int(x) for x in a.contexts.split(",")]
    stream_ids = h.token_stream(enc, h.corpus_text(None), max(contexts) + a.score_tokens + 1)
    stream = mx.array([stream_ids], dtype=mx.uint32)
    out = {"model": a.model, "operation": a.operation, "contexts": []}
    for ctx in contexts:
        t0 = time.perf_counter()
        caches = {"exact": list(make_prompt_cache(model)),
                  "exact_b": list(make_prompt_cache(model)),
                  "quant": list(op.apply(LaneKVState(op.revision, tuple(make_prompt_cache(model)))).planes)}
        assert quantized_plane_count(caches["quant"]) > 0
        assert quantized_plane_count(caches["exact_b"]) == 0
        steps = {"exact": 2048, "exact_b": 1536, "quant": 2048}
        for arm, cache in caches.items():
            s = steps[arm]
            for start in range(0, ctx, s):
                mx.eval(model(stream[:, start:min(start + s, ctx)], cache=cache))
        rows = []
        for pos in range(ctx, ctx + a.score_tokens):
            step = stream[:, pos:pos + 1]
            lp = {}
            for arm, cache in caches.items():
                lg = model(step, cache=cache)[0, -1].astype(mx.float32)
                lp[arm] = lg - mx.logsumexp(lg)
            mx.eval(lp)
            e = lp["exact"]
            top2 = mx.argsort(-e)[:2]
            p_e = mx.exp(e)
            row = {"pos": pos,
                   "exact_top1": int(top2[0].item()),
                   "margin": float((e[top2[0]] - e[top2[1]]).item()),
                   "p_top1": float(p_e[top2[0]].item())}
            for arm in ("exact_b", "quant"):
                o = lp[arm]
                row[f"{arm}_top1"] = int(mx.argmax(o).item())
                row[f"{arm}_agree"] = row[f"{arm}_top1"] == row["exact_top1"]
                row[f"{arm}_kl"] = float(mx.maximum(mx.sum(p_e * (e - o)), 0).item())
            rows.append(row)
        def summ(key, sel):
            s = [r for r in rows if sel(r)]
            return {"n": len(s), "agree": (sum(r[key] for r in s) / len(s)) if s else None}
        dis = [r for r in rows if not r["quant_agree"] or not r["exact_b_agree"]]
        entry = {
            "context": ctx,
            "exact_b_top1_agreement": summ("exact_b_agree", lambda r: True),
            "quant_top1_agreement": summ("quant_agree", lambda r: True),
            "exact_b_kl_mean": sum(r["exact_b_kl"] for r in rows) / len(rows),
            "quant_kl_mean": sum(r["quant_kl"] for r in rows) / len(rows),
            "by_margin": {
                f">{m}": {"quant": summ("quant_agree", lambda r, m=m: r["margin"] > m),
                          "exact_b": summ("exact_b_agree", lambda r, m=m: r["margin"] > m)}
                for m in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)
            },
            "disagreements": [dict(r, exact_tok=tok.decode([r["exact_top1"]]),
                                   quant_tok=tok.decode([r["quant_top1"]]),
                                   exact_b_tok=tok.decode([r["exact_b_top1"]])) for r in dis],
            "window_text": tok.decode(stream_ids[ctx:ctx + a.score_tokens])[:1500],
            "wall_s": time.perf_counter() - t0,
        }
        out["contexts"].append(entry)
        print(json.dumps({k: v for k, v in entry.items() if k not in ("disagreements", "window_text")}), flush=True)
        del caches
        mx.clear_cache()
    Path(a.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

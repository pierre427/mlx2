"""rm15 step 2d: does SPARSE attention turn chunk-size rounding into a
DISCRETE block-selection flip?

qwen38_27b (rm07's model) is dense: chunk-size variance there is rounding.
qwen4_exp / Flash-Next add a QSA indexer that argpartitions block scores and
attends to the top-k blocks only.  A near-tie between two block scores plus a
rounding-scale perturbation selects a DIFFERENT block, and the output then
changes by O(1), not by an ulp.  That is a qualitatively larger exposure and
it is worth knowing whether it happens.

Tiny random-weight qwen4_exp, CPU, fresh cache per chunk size, everything
compared against the unchunked arm.
"""
import argparse, json
import mlx.core as mx

mx.set_default_device(mx.cpu)

TEXT = dict(
    model_type="qwen4_exp_text", hidden_size=64, intermediate_size=0,
    num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
    head_dim=32, vocab_size=128,
    linear_num_value_heads=4, linear_num_key_heads=2,
    linear_key_head_dim=32, linear_value_head_dim=32,
    linear_conv_kernel_dim=4,
    layer_types=["linear_attention", "full_attention",
                 "linear_attention", "full_attention"],
    num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
    hc_count=2, hc_lowrank=8,
    ple_layer_ids=[1], ple_embed_dim=64, ple_conv_kernel_size=4,
    ngram_size=3, heads_per_ngram=2, ngram_vocab_size_base=128,
    make_ngram_vocab_size_divisible_by=128, split_ngram_parts=1,
    indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=32,
    indexer_budget=64, indexer_compress_ratio=8,
    mtp_num_hidden_layers=0,
    rope_parameters={"type": "default", "rope_theta": 10000.0,
                     "partial_rotary_factor": 0.25},
)


def build(seed=0):
    mx.random.seed(seed)
    from mlx2.runtime.models.qwen4_exp import Model, ModelArgs
    m = Model(ModelArgs(model_type="qwen4_exp", text_config=dict(TEXT)))
    mx.eval(m.parameters())
    return m


def run(model, ids, ctx, step, score):
    from mlx2.runtime.models.cache import make_prompt_cache
    cache = list(make_prompt_cache(model))
    for start in range(0, ctx, step):
        mx.eval(model(ids[:, start:min(start + step, ctx)], cache=cache))
    outs = []
    for j in range(score):
        lg = model(ids[:, ctx + j:ctx + j + 1], cache=cache)[0, -1].astype(mx.float32)
        mx.eval(lg)
        outs.append(lg)
    return mx.stack(outs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--context", type=int, default=1024)
    p.add_argument("--score-tokens", type=int, default=32)
    p.add_argument("--steps", default="")
    p.add_argument("--out", default="")
    a = p.parse_args()
    ctx = a.context
    steps = ([int(x) for x in a.steps.split(",")] if a.steps
             else [ctx, ctx, ctx // 2, ctx // 4, 128, 100])
    model = build()
    mx.random.seed(1234)
    ids = mx.random.randint(0, 128, (1, ctx + a.score_tokens + 1)).astype(mx.uint32)

    rep = {"model": "qwen4_exp (QSA indexer, tiny random weights)",
           "device": "cpu", "context": ctx, "indexer_budget": TEXT["indexer_budget"],
           "compress_ratio": TEXT["indexer_compress_ratio"],
           "reference": "unchunked", "arms": {}}
    ref = None
    seen = set()
    for s in steps:
        out = run(model, ids, ctx, s, a.score_tokens)
        if ref is None:
            ref = out
            rep["arms"]["full"] = {"prefill_step": s, "reference": True}
            continue
        label = f"{s}r" if s in seen else str(s)
        seen.add(s)
        d = mx.abs(out - ref)
        agree = (mx.argmax(out, -1) == mx.argmax(ref, -1)).astype(mx.float32)
        mx.eval(d, agree)
        rep["arms"][label] = {
            "prefill_step": s,
            "max_abs_logit_delta": float(mx.max(d).item()),
            "mean_abs_logit_delta": float(mx.mean(d).item()),
            "top1_agreement": float(mx.mean(agree).item()),
            "logit_scale": float(mx.max(mx.abs(ref)).item()),
        }
        print(label, json.dumps(rep["arms"][label]), flush=True)
    if a.out:
        open(a.out, "w").write(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()

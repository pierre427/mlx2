"""rm15 step 3: accumulation-order noise (i) or numerical defect (ii)?

The discriminator is EPSILON SCALING.  Run the identical chunked-vs-unchunked
comparison at several working precisions.  If the divergence is pure
accumulation order, its magnitude tracks the working precision's machine
epsilon (fp64 ~1e-16, fp32 ~1e-7, bf16 ~8e-3).  If one chunk size is
computing something *wrong* (bad mask, dropped token, stale state, wrong
position), the divergence is a property of the arithmetic it performs, not of
the rounding, and it does NOT shrink when precision is raised.

Also reports, at fp32, whether the chunked and unchunked arms are
EQUIDISTANT from an fp64 ground truth.  A defect makes one arm right.
"""
import argparse, json
import mlx.core as mx

mx.set_default_device(mx.cpu)

TEXT = dict(
    model_type="qwen3_5_text", hidden_size=64, intermediate_size=128,
    num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
    head_dim=32, vocab_size=128,
    linear_num_value_heads=4, linear_num_key_heads=2,
    linear_key_head_dim=32, linear_value_head_dim=32,
    linear_conv_kernel_dim=4, full_attention_interval=4,
    mtp_num_hidden_layers=0, num_experts=0, rms_norm_eps=1e-6,
    max_position_embeddings=65536,
    rope_parameters={"type": "default", "rope_theta": 10000.0,
                     "partial_rotary_factor": 0.25},
)


def build(dtype, seed=0):
    mx.random.seed(seed)
    from mlx2.runtime.models.qwen38_27b import Model, ModelArgs
    m = Model(ModelArgs(model_type="qwen3_5", text_config=dict(TEXT)))
    m.set_dtype(dtype)
    mx.eval(m.parameters())
    return m


def prefill_logits(model, ids, ctx, step, score):
    """Prefill with the given chunk size, then teacher-force `score` tokens
    from a cache that belongs to this arm alone. Returns stacked logits."""
    from mlx2.runtime.models.cache import make_prompt_cache
    cache = list(make_prompt_cache(model))
    for start in range(0, ctx, step):
        mx.eval(model(ids[:, start:min(start + step, ctx)], cache=cache))
    outs = []
    for j in range(score):
        lg = model(ids[:, ctx + j:ctx + j + 1], cache=cache)[0, -1]
        mx.eval(lg)
        outs.append(lg.astype(mx.float64))
    return mx.stack(outs)


def compare(a, b):
    d = mx.abs(a - b)
    top_a = mx.argmax(a, axis=-1)
    top_b = mx.argmax(b, axis=-1)
    return {"max_abs_logit_delta": float(mx.max(d).item()),
            "mean_abs_logit_delta": float(mx.mean(d).item()),
            "top1_agreement": float(mx.mean((top_a == top_b).astype(mx.float32)).item())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--score-tokens", type=int, default=32)
    p.add_argument("--steps", default="")
    p.add_argument("--dtypes", default="float64,float32,bfloat16")
    p.add_argument("--out", default="")
    a = p.parse_args()
    ctx = a.context
    steps = ([int(x) for x in a.steps.split(",")] if a.steps
             else [ctx, ctx // 2, ctx // 4, 100, 64])
    mx.random.seed(1234)
    ids = mx.random.randint(0, 128, (1, ctx + a.score_tokens + 1)).astype(mx.uint32)

    rep = {"context": ctx, "score_tokens": a.score_tokens, "device": "cpu",
           "machine_eps": {}, "by_dtype": {}}
    logits = {}
    for name in a.dtypes.split(","):
        dt = getattr(mx, name)
        try:
            rep["machine_eps"][name] = float(mx.finfo(dt).eps)
        except Exception:
            rep["machine_eps"][name] = None
        model = build(dt)
        arms = {}
        logits[name] = {}
        for s in dict.fromkeys(steps):
            try:
                logits[name][s] = prefill_logits(model, ids, ctx, s, a.score_tokens)
            except Exception as e:
                arms[str(s)] = {"error": repr(e)[:200]}
        ref = logits[name].get(ctx)
        for s in steps:
            if s == ctx or s not in logits[name] or ref is None:
                continue
            arms[str(s)] = compare(logits[name][s], ref)
        rep["by_dtype"][name] = {"vs_unchunked": arms}
        del model

    # equidistance test: fp32 arms vs the fp64 ground truth
    if "float64" in logits and "float32" in logits:
        gt = logits["float64"].get(ctx)
        if gt is not None:
            eq = {}
            for s in steps:
                if s in logits["float32"]:
                    eq[str(s)] = compare(logits["float32"][s], gt)
            rep["fp32_arms_vs_fp64_ground_truth"] = eq
    if "bfloat16" in logits and "float64" in logits:
        gt = logits["float64"].get(ctx)
        if gt is not None:
            eq = {}
            for s in steps:
                if s in logits["bfloat16"]:
                    eq[str(s)] = compare(logits["bfloat16"][s], gt)
            rep["bf16_arms_vs_fp64_ground_truth"] = eq

    txt = json.dumps(rep, indent=2)
    if a.out:
        open(a.out, "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()

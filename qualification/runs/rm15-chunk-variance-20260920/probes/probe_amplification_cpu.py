"""rm15 step 3b: how big a logit swing does ONE rounding-sized perturbation
buy, and how does that grow with context?

The objection to "it is just accumulation order" is that a 6.4 max |delta
logit| looks too large for rounding.  This measures the transfer function
directly: perturb a SINGLE element of a SINGLE token's input embedding by a
known multiple of the working precision's epsilon, change nothing else, and
report the resulting max |delta logit| and top-1 agreement over the scored
window, at several context lengths.

The amplification factor (output delta / input delta) and its growth with
context are what decide whether rounding can explain the observed swing.
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
    max_position_embeddings=131072,
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


def run(model, ids, ctx, score, step, perturb=None):
    from mlx2.runtime.models.cache import make_prompt_cache
    emb = model.language_model.model.embed_tokens(ids[:, :ctx])
    if perturb is not None:
        pos, ch, delta = perturb
        add = mx.zeros(emb.shape, dtype=emb.dtype)
        add[:, pos, ch] = delta
        emb = emb + add
    cache = list(make_prompt_cache(model))
    for start in range(0, ctx, step):
        e = min(start + step, ctx)
        mx.eval(model(ids[:, start:e], cache=cache,
                      input_embeddings=emb[:, start:e]))
    outs = []
    for j in range(score):
        lg = model(ids[:, ctx + j:ctx + j + 1], cache=cache)[0, -1]
        lg = lg.astype(mx.float32)
        mx.eval(lg)
        outs.append(lg)
    return mx.stack(outs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--contexts", default="256,512,1024,2048,4096")
    p.add_argument("--score-tokens", type=int, default=64)
    p.add_argument("--dtype", default="float32")
    p.add_argument("--ulps", default="1,8,64")
    p.add_argument("--out", default="")
    a = p.parse_args()
    dt = getattr(mx, a.dtype)
    eps = float(mx.finfo(dt).eps)
    model = build(dt)
    contexts = [int(x) for x in a.contexts.split(",")]
    ulps = [float(x) for x in a.ulps.split(",")]
    mx.random.seed(1234)
    ids = mx.random.randint(0, 128, (1, max(contexts) + a.score_tokens + 1)).astype(mx.uint32)

    rep = {"device": "cpu", "dtype": a.dtype, "eps": eps,
           "score_tokens": a.score_tokens, "rows": []}
    for ctx in contexts:
        ref = run(model, ids, ctx, a.score_tokens, ctx)
        scale = float(mx.max(mx.abs(ref)).item())
        for u in ulps:
            # perturb one channel of the FIRST token's embedding
            delta = u * eps
            got = run(model, ids, ctx, a.score_tokens, ctx,
                      perturb=(0, 0, delta))
            d = mx.abs(got - ref)
            agree = (mx.argmax(got, -1) == mx.argmax(ref, -1)).astype(mx.float32)
            mx.eval(d, agree)
            rep["rows"].append({
                "context": ctx, "ulps": u, "input_delta": delta,
                "max_abs_logit_delta": float(mx.max(d).item()),
                "mean_abs_logit_delta": float(mx.mean(d).item()),
                "top1_agreement": float(mx.mean(agree).item()),
                "logit_scale": scale,
                "amplification": float(mx.max(d).item()) / delta,
            })
            print(json.dumps(rep["rows"][-1]), flush=True)
    if a.out:
        open(a.out, "w").write(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()

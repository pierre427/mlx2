"""rm15 step 2: which piece of CACHE STATE differs across prefill chunk sizes?

Prefill the same stream into fresh caches with different chunk sizes, then
diff every cache entry layer by layer. CPU only.
"""
import argparse, json, sys
import mlx.core as mx

mx.set_default_device(mx.cpu)


def build(dtype, seed=0):
    mx.random.seed(seed)
    from mlx2.runtime.models.qwen38_27b import Model, ModelArgs
    text = dict(
        model_type="qwen3_5_text", hidden_size=64, intermediate_size=128,
        num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, vocab_size=128,
        linear_num_value_heads=4, linear_num_key_heads=2,
        linear_key_head_dim=32, linear_value_head_dim=32,
        linear_conv_kernel_dim=4, full_attention_interval=4,
        mtp_num_hidden_layers=0, num_experts=0,
        rms_norm_eps=1e-6, max_position_embeddings=65536,
        rope_parameters={"type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.25},
    )
    m = Model(ModelArgs(model_type="qwen3_5", text_config=text))
    if dtype != "float32":
        m.set_dtype(getattr(mx, dtype))
    mx.eval(m.parameters())
    return m


def prefill(model, stream, ctx, step):
    from mlx2.runtime.models.cache import make_prompt_cache
    cache = list(make_prompt_cache(model))
    for start in range(0, ctx, step):
        mx.eval(model(stream[:, start:min(start + step, ctx)], cache=cache))
    return cache


def describe(cache_entry):
    """Return {name: array} of this cache entry's tensors, trimmed to offset."""
    out = {}
    if hasattr(cache_entry, "keys") and cache_entry.keys is not None:
        k, v = cache_entry.keys_and_values()
        out["keys"] = k
        out["values"] = v
        out["offset"] = mx.array([cache_entry.offset])
    else:
        # ArraysCache (GDN): [conv_state, recurrent_state]
        for i, a in enumerate(getattr(cache_entry, "cache", []) or []):
            if isinstance(a, mx.array):
                out[f"arr{i}"] = a
        off = getattr(cache_entry, "offset", None)
        if off is not None:
            out["offset"] = mx.array([off])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", default="float32")
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--steps", default="")
    p.add_argument("--out", default="")
    a = p.parse_args()
    ctx = a.context
    steps = ([int(x) for x in a.steps.split(",")] if a.steps
             else [ctx, ctx // 2, ctx // 4, 128, 192, 100])
    model = build(a.dtype)
    mx.random.seed(1234)
    ids = mx.random.randint(0, 128, (1, ctx + 64)).astype(mx.uint32)

    ref_step = ctx
    caches = {s: prefill(model, ids, ctx, s) for s in dict.fromkeys(steps)}
    is_lin = [l.is_linear for l in model.language_model.model.layers]

    rep = {"context": ctx, "dtype": a.dtype, "device": "cpu",
           "reference": f"step={ref_step} (unchunked)", "arms": {}}
    ref = [describe(c) for c in caches[ref_step]]
    for s in steps:
        if s == ref_step:
            continue
        rows = []
        for i, c in enumerate(caches[s]):
            d = describe(c)
            kind = "gdn" if is_lin[i] else "attn"
            for name in d:
                x, y = d[name], ref[i][name]
                if x.shape != y.shape:
                    rows.append({"layer": i, "kind": kind, "field": name,
                                 "shape_mismatch": [list(x.shape), list(y.shape)]})
                    continue
                dd = float(mx.max(mx.abs(x.astype(mx.float32) -
                                         y.astype(mx.float32))).item())
                if dd != 0.0:
                    rows.append({"layer": i, "kind": kind, "field": name,
                                 "max_abs_delta": dd})
        rep["arms"][str(s)] = {"prefill_step": s, "differing": rows,
                               "n_differing": len(rows)}
    txt = json.dumps(rep, indent=2)
    if a.out:
        open(a.out, "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()

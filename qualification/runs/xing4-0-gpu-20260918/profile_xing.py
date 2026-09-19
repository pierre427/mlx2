"""Per-component GPU time for Xing4.0 decode and prefill.

usage: profile_xing.py ARTIFACT OUT.json

Times the full model step and each component of one MoE decoder layer on the
real weights: mHC coefficients (pre), MLA attention, MoE MLP, mHC stream
update, RMS norms; plus the LM head.  Each timing is the median of repeated
``mx.eval`` calls after warm-up.  Layer components are scaled by the layer
count to estimate their share of a model step.
"""
import json
import statistics
import sys
import time

import mlx.core as mx

from mlx2.adapters.xing import XingAdapter
from mlx2.runtime.models import xing4_0

ARTIFACT, OUT = sys.argv[1], sys.argv[2]


def timed(fn, *, repeat=30, warm=5):
    for _ in range(warm):
        mx.eval(fn())
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        mx.eval(fn())
        samples.append((time.perf_counter() - start) * 1000)
    return round(statistics.median(samples), 3)


adapter = XingAdapter(ARTIFACT)
model = adapter.model
args = model.args
H, HC = args.hidden_size, args.hc_mult
layer = model.model.layers[10]  # MoE layer
dense = model.model.layers[0]
report = {"artifact": ARTIFACT, "layers": args.num_hidden_layers, "results": {}}


def prefilled_cache(batch, context):
    cache = model.make_cache()
    tokens = mx.random.randint(100, 20000, (batch, context))
    for start in range(0, context, 2048):
        model(tokens[:, start : start + 2048], cache=cache)
        mx.eval([c.state for c in cache])
    return cache


for mode, batch, length, context in (
    ("decode_b1", 1, 1, 2048),
    ("decode_b4", 4, 1, 2048),
    ("prefill_2k", 1, 2048, 0),
    ("prefill_8k_tail", 1, 2048, 6144),
):
    row = {}
    tokens = mx.random.randint(100, 20000, (batch, length))
    if context:
        cache = prefilled_cache(batch, context)
        base = cache[0].offset

        def step():
            for c in cache:
                c.trim(c.offset - base)
            return model(tokens, cache=cache)
    else:
        def step():
            return model(tokens, cache=model.make_cache())
    row["model_step_ms"] = timed(step, repeat=10 if length > 1 else 30)

    streams = mx.random.normal((batch, length, HC, H)).astype(mx.bfloat16)
    x = mx.random.normal((batch, length, H)).astype(mx.bfloat16)
    layer_cache = prefilled_cache(batch, context)[10] if context else None
    base = layer_cache.offset if layer_cache is not None else 0
    mask = None if length == 1 else "causal"

    def attn():
        if layer_cache is not None:
            layer_cache.trim(layer_cache.offset - base)
            m = xing4_0.create_attention_mask(x, layer_cache, return_array=True)
        else:
            m = xing4_0.create_attention_mask(x, None, return_array=True)
        return layer.self_attn(x, m, layer_cache)

    post, comb, _ = layer.attn_hc(streams)
    mx.eval(post, comb)
    parts = {
        "mhc_pre": lambda: layer.attn_hc(streams),
        "mhc_pre_eager": lambda: layer.attn_hc.reference(streams),
        "mhc_update": lambda: xing4_0.HyperConnection.update(streams, x, post, comb),
        "rms_norm": lambda: layer.input_layernorm(x),
        "attention": attn,
        "moe_mlp": lambda: layer.mlp(x),
        "dense_mlp": lambda: dense.mlp(x),
    }
    per_layer = {name: timed(fn) for name, fn in parts.items()}
    row["per_layer_ms"] = per_layer
    moe_layers = args.num_hidden_layers - args.first_k_dense_replace
    estimate = {
        "mhc": 2 * args.num_hidden_layers * (per_layer["mhc_pre"] + per_layer["mhc_update"]),
        "norms": 2 * args.num_hidden_layers * per_layer["rms_norm"],
        "attention": args.num_hidden_layers * per_layer["attention"],
        "moe": moe_layers * per_layer["moe_mlp"],
        "dense_mlp": args.first_k_dense_replace * per_layer["dense_mlp"],
    }
    h = mx.random.normal((batch, length, H)).astype(mx.bfloat16)
    estimate["lm_head"] = timed(lambda: model.lm_head(h))
    total = sum(estimate.values())
    row["estimated_ms"] = {k: round(v, 2) for k, v in estimate.items()}
    row["estimated_share"] = {k: round(v / total, 3) for k, v in estimate.items()}
    row["estimated_sum_ms"] = round(total, 2)
    report["results"][mode] = row
    print(mode, json.dumps(row), flush=True)

json.dump(report, open(OUT, "w"), indent=2)

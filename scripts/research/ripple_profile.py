#!/usr/bin/env python3
"""In-process Ripple decode/prefill attribution on one loaded artifact.

Loads the adapter once (no server) from the source tree on PYTHONPATH and
records, on Metal:

* decode ms/token over a warm cache: synchronous (eval every step) and
  pipelined (async_eval one step ahead, as the serving loop does);
* the weight-read floor: every quantized/linear module of the model called on
  one row in model order, with no other work between them;
* a per-section decode split (attention block vs MLP block vs final norm and
  head), each section synchronized separately (perturbing, but attributes);
* one prefill chunk of ``--chunk`` rows at offsets 0 and ``--chunk``.

Greedy tokens are recorded so trees can be compared for exactness. Requires
the caller's GPU ownership. Writes one JSON receipt; refuses to overwrite it.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten


def input_width(module):
    """Activation width of a linear-like module (packed affine rows unpacked)."""
    width = module.weight.shape[-1]
    if isinstance(module, nn.QuantizedLinear):
        return width * 32 // module.bits
    if type(module).__name__ == "ModelOptNVFP4Linear":
        return width * 8  # uint32 words of eight FP4 codes
    return width


def linear_modules(model):
    return [(name, m) for name, m in model.named_modules()
            if isinstance(m, (nn.QuantizedLinear, nn.Linear)) or type(m).__name__ == "ModelOptNVFP4Linear"]


def attention_probe(rows, window=1024, heads=80, kv_heads=8, dim=128):
    """One layer's prefill SDPA at real shapes on random BF16 tensors.

    ``sliding_masked``: as served, a chunk of ``rows`` at offset ``rows``
    against ``rows + window - 1`` keys with the explicit window mask.
    ``sliding_blocked``: the same chunk in ``window``-row blocks, each against
    only its own ``2 * window - 1`` keys (window-aware tiling candidate; the
    same key sets per row, not a bit-exactness claim).
    ``global_causal``: a chunk at offset ``rows`` against ``2 * rows`` keys.
    """
    from mlx2.runtime.models.base import create_causal_mask

    sdpa = mx.fast.scaled_dot_product_attention
    q = mx.random.normal((1, heads, rows, dim)).astype(mx.bfloat16)
    keys = rows + window - 1
    k = mx.random.normal((1, kv_heads, keys, dim)).astype(mx.bfloat16)
    v = mx.random.normal((1, kv_heads, keys, dim)).astype(mx.bfloat16)
    mask = create_causal_mask(rows, offset=window - 1, window_size=window)
    kg = mx.random.normal((1, kv_heads, 2 * rows, dim)).astype(mx.bfloat16)
    vg = mx.random.normal((1, kv_heads, 2 * rows, dim)).astype(mx.bfloat16)
    mx.eval(q, k, v, mask, kg, vg)
    scale = dim ** -0.5

    def masked():
        mx.eval(sdpa(q, k, v, scale=scale, mask=mask))

    def blocked():
        outs = []
        for start in range(0, rows, window):
            stop = min(rows, start + window)
            ks, ke = start, stop + window - 1
            outs.append(sdpa(q[:, :, start:stop], k[:, :, ks:ke], v[:, :, ks:ke], scale=scale,
                             mask=mask[start:stop, ks:ke]))
        mx.eval(outs)

    def global_causal():
        mx.eval(sdpa(q, kg, vg, scale=scale, mask="causal"))

    for fn in (masked, blocked, global_causal):
        fn()
    return {"rows": rows, "sliding_masked": timed(masked, 7), "sliding_blocked": timed(blocked, 7),
            "global_causal": timed(global_causal, 7),
            "note": "per layer; Ripple has 64 sliding and 16 global layers"}


def timed(fn, repeats):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    samples.sort()
    return {"median_ms": samples[len(samples) // 2] * 1e3, "min_ms": samples[0] * 1e3,
            "max_ms": samples[-1] * 1e3, "n": len(samples)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", default="{}")
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--decode-steps", type=int, default=48)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from mlx2.adapters.ripple import RippleAdapter

    load_start = time.perf_counter()
    adapter = RippleAdapter(str(args.model), execution_policy=json.loads(args.policy) or None)
    model = adapter.model
    receipt = {"schema": "mlx2.ripple-profile.v1", "mlx": mx.__version__,
               "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "checkpoint": args.model.name + "@sha256:" + adapter.identity["fingerprint"],
               "policy": json.loads(args.policy), "fused_elementwise": getattr(model, "fused_elementwise", False),
               "load_seconds": time.perf_counter() - load_start,
               "metal_active_gb": mx.get_active_memory() / 1e9}
    tokenizer = adapter.tokenizer
    text = (Path(__file__).resolve().parents[2] / "docs" / "ripple-65b.md").read_text()
    ids = tokenizer.encode(text, add_special_tokens=False)
    while len(ids) < 2 * args.chunk + 16:
        ids = ids + ids[: 2 * args.chunk + 16 - len(ids)]
    prompt = mx.array([ids[: 2 * args.chunk]])

    # Prefill: two chunks, the second crossing the window with a full cache.
    cache = model.make_cache()
    rows = {}
    for index in range(2):
        chunk = prompt[:, index * args.chunk:(index + 1) * args.chunk]
        start = time.perf_counter()
        logits = model(chunk, cache=cache)
        mx.eval(logits, [c.state for c in cache])
        elapsed = time.perf_counter() - start
        rows[f"offset_{index * args.chunk}"] = {"ms": elapsed * 1e3, "rows_per_s": args.chunk / elapsed}
    receipt["prefill_chunks"] = rows

    # Decode: synchronous and pipelined, greedy tokens recorded.
    token = mx.argmax(logits[:, -1:], axis=-1)
    tokens = []

    def step(tok):
        out = model(tok, cache=cache)
        return mx.argmax(out[:, -1:], axis=-1)

    for _ in range(4):
        token = step(token)
        mx.eval(token)
        tokens.append(int(token.item()))
    sync = []
    for _ in range(args.decode_steps):
        start = time.perf_counter()
        token = step(token)
        mx.eval(token)
        sync.append(time.perf_counter() - start)
        tokens.append(int(token.item()))
    sync.sort()
    pending = step(token)
    mx.async_eval(pending)
    start = time.perf_counter()
    for _ in range(args.decode_steps):
        nxt = step(pending)
        mx.async_eval(nxt)
        tokens.append(int(pending.item()))
        pending = nxt
    mx.eval(pending)
    pipelined = (time.perf_counter() - start) / args.decode_steps
    receipt["decode"] = {"sync_median_ms": sync[len(sync) // 2] * 1e3, "sync_min_ms": sync[0] * 1e3,
                         "pipelined_ms": pipelined * 1e3, "tokens": tokens,
                         "kv_offset_at_end": cache[4].offset}

    # Host graph construction alone (no evaluation).
    def build_only():
        step(token)

    receipt["decode"]["host_graph_build"] = timed(build_only, 10)

    # Weight-read floor: every linear-like module in model order on one row.
    modules = linear_modules(model)
    inputs = {}
    for _, module in modules:
        k = input_width(module)
        inputs.setdefault(k, mx.zeros((1, 1, k), dtype=mx.bfloat16) + 0.01)
    mx.eval(list(inputs.values()))
    nbytes = sum(v.nbytes for _, m in modules for _, v in tree_flatten(m.parameters()))

    def floor():
        mx.eval([module(inputs[input_width(module)]) for _, module in modules])

    floor()
    receipt["weight_floor"] = {**timed(floor, 7), "modules": len(modules), "bytes": nbytes}
    receipt["weight_floor"]["gb_per_s"] = nbytes / (receipt["weight_floor"]["median_ms"] / 1e3) / 1e9

    # Per-section split for one decode step (synchronized per section).
    h = model.model.embed_tokens(token)
    from mlx2.runtime.models.ricursive import scale_embeddings

    h = scale_embeddings(h, model.args.embed_scale)
    seed = h.astype(mx.float32)
    hoisted = hasattr(model.model, "shared_layer_inputs")  # absent before the copy cleanup
    shared = model.model.shared_layer_inputs(h, cache) if hoisted else None
    mx.eval(h, seed)
    attn_t = mlp_t = 0.0
    for layer, entry, kind in zip(model.layers, cache, model.args.layer_types, strict=True):
        start = time.perf_counter()
        if hoisted:
            mask, rope = shared[kind]
            a = layer.self_attn(layer.input_layernorm(h), seed, entry, mask=mask, rope=rope)
        else:
            a = layer.self_attn(layer.input_layernorm(h), seed, entry)
        h2 = h + layer.post_attention_layernorm(a)
        mx.eval(h2)
        mid = time.perf_counter()
        h = h2 + layer.post_feedforward_layernorm(layer.mlp(layer.pre_feedforward_layernorm(h2)))
        mx.eval(h)
        attn_t += mid - start
        mlp_t += time.perf_counter() - mid
    start = time.perf_counter()
    mx.eval(model.lm_head(model.model.norm(h)))
    head_t = time.perf_counter() - start
    receipt["decode_split_ms"] = {"attention_blocks": attn_t * 1e3, "mlp_blocks": mlp_t * 1e3,
                                  "norm_head": head_t * 1e3, "sections_synchronized": True}
    receipt["attention_prefill_ms"] = attention_probe(args.chunk)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in ("prefill_chunks", "weight_floor", "decode_split_ms")}
                     | {"decode": {k: v for k, v in receipt["decode"].items() if k != "tokens"}}))


if __name__ == "__main__":
    main()

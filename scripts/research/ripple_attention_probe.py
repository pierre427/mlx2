#!/usr/bin/env python3
"""Ripple decode/verify attention attribution on one loaded artifact (Metal).

1. Model level: pipelined decode ms/token over a ~``--context`` token cache,
   as served and with ``scaled_dot_product_attention`` stubbed (returns the
   queries' shape without reading K/V). The difference bounds what any
   attention kernel change can win; stubbed tokens are meaningless.
2. Component level, random BF16 at Ripple's geometry (80Q/8KV x 128):
   * decode (one row): stock ``mx.fast.scaled_dot_product_attention`` vs a
     GQA-folded call (each KV head's 10 query heads as 10 rows of one head,
     no mask: one KV read per head), at sliding (1,024) and global
     (``--context``) key counts, with exactness against stock;
   * verify (8 rows): stock SDPA vs mlx2 ``fp_verify_tensor`` with the
     served masks (sliding window bool mask, global "causal").
   Timings issue 80 calls per evaluation (one per layer) to amortize sync.
Requires the caller's GPU ownership. Writes one JSON receipt.
"""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src
import argparse
import json
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

HEADS, KV_HEADS, DIM, WINDOW = 80, 8, 128, 1024


def timed(fn, repeats=9):
    fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    samples.sort()
    return {"median_ms": samples[len(samples) // 2] * 1e3, "min_ms": samples[0] * 1e3}


def compare(a, b):
    a, b = np.asarray(a.astype(mx.float32)), np.asarray(b.astype(mx.float32))
    return {"bit_equal": bool(np.array_equal(a, b)), "max_abs": float(np.max(np.abs(a - b))),
            "mismatch_fraction": float(np.mean(a != b))}


def folded_decode(q, k, v, scale):
    """[B, 80, 1, D] -> per KV head, its 10 query heads as 10 rows."""
    b, hq, _, d = q.shape
    g = hq // k.shape[1]
    rows = q.reshape(b, k.shape[1], g, d)
    out = mx.fast.scaled_dot_product_attention(rows, k, v, scale=scale)
    return out.reshape(b, hq, 1, d)


def components(context, layers=80):
    scale = DIM ** -0.5
    rng = np.random.default_rng(7)

    def a(shape):
        return mx.array(rng.normal(size=shape).astype(np.float32)).astype(mx.bfloat16)

    out = {}
    q1 = a((1, HEADS, 1, DIM))
    for name, n in (("sliding", WINDOW), ("global", context)):
        k, v = a((1, KV_HEADS, n, DIM)), a((1, KV_HEADS, n, DIM))
        mx.eval(q1, k, v)
        stock = lambda: mx.eval([mx.fast.scaled_dot_product_attention(q1, k, v, scale=scale) for _ in range(layers)])
        fold = lambda: mx.eval([folded_decode(q1, k, v, scale) for _ in range(layers)])
        out[f"decode_{name}_{n}"] = {
            "stock_x80": timed(stock), "gqa_folded_x80": timed(fold),
            "folded_vs_stock": compare(folded_decode(q1, k, v, scale),
                                       mx.fast.scaled_dot_product_attention(q1, k, v, scale=scale))}
    from mlx2.runtime.models import fp_verify_tensor
    from mlx2.runtime.models.base import create_causal_mask

    rows = 8
    q8 = a((1, HEADS, rows, DIM))
    for name, n in (("sliding", WINDOW - 1 + rows), ("global", context + rows)):
        k, v = a((1, KV_HEADS, n, DIM)), a((1, KV_HEADS, n, DIM))
        mask = create_causal_mask(rows, offset=n - rows, window_size=WINDOW) if name == "sliding" else "causal"
        mx.eval(q8, k, v)
        supported = fp_verify_tensor.supported(q8, k, v, mask) and mx.default_device() == mx.gpu
        stock = lambda: mx.eval([mx.fast.scaled_dot_product_attention(q8, k, v, scale=scale, mask=mask)
                                 for _ in range(layers)])
        row = {"stock_x80": timed(stock), "fp_verify_supported": supported}
        if supported:
            custom = lambda: mx.eval([fp_verify_tensor.attention(q8, k, v, scale=scale, mask=mask)
                                      for _ in range(layers)])
            row["fp_verify_x80"] = timed(custom)
            row["fp_verify_vs_stock"] = compare(fp_verify_tensor.attention(q8, k, v, scale=scale, mask=mask),
                                                mx.fast.scaled_dot_product_attention(q8, k, v, scale=scale, mask=mask))
        out[f"verify8_{name}_{n}"] = row
    return out


def model_level(adapter, context, steps):
    from mlx2.runtime.models import ricursive

    model = adapter.model
    text = (Path(__file__).resolve().parents[2] / "docs" / "ripple-65b.md").read_text()
    ids = adapter.tokenizer.encode(text, add_special_tokens=False)
    while len(ids) < context:
        ids = ids + ids
    cache = model.make_cache()
    for start in range(0, context, 512):
        mx.eval(model(mx.array([ids[start:min(context, start + 512)]]), cache=cache))
    snapshot = [c.state for c in cache]
    mx.eval(snapshot)

    def decode(n):
        token = mx.array([[ids[0]]])
        pending = mx.argmax(model(token, cache=cache)[:, -1:], axis=-1)
        mx.async_eval(pending)
        start = time.perf_counter()
        for _ in range(n):
            nxt = mx.argmax(model(pending, cache=cache)[:, -1:], axis=-1)
            mx.async_eval(nxt)
            pending.item()
            pending = nxt
        mx.eval(pending)
        return (time.perf_counter() - start) / n * 1e3

    served = decode(steps)
    original = ricursive.scaled_dot_product_attention
    try:
        ricursive.scaled_dot_product_attention = lambda q, k, v, **_kw: mx.zeros_like(q)
        stubbed = decode(steps)
    finally:
        ricursive.scaled_dot_product_attention = original
    served_again = decode(steps)
    return {"context": context, "kv_offset_at_end": cache[4].offset, "served_ms": served,
            "attention_stubbed_ms": stubbed, "served_again_ms": served_again,
            "attention_upper_bound_ms": (served + served_again) / 2 - stubbed}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from mlx2.adapters.ripple import RippleAdapter

    adapter = RippleAdapter(str(args.model))
    receipt = {"schema": "mlx2.ripple-attention-probe.v1", "mlx": mx.__version__,
               "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "checkpoint": args.model.name + "@sha256:" + adapter.identity["fingerprint"],
               "model_level": model_level(adapter, args.context, args.steps)}
    adapter.close()
    mx.clear_cache()
    receipt["components"] = components(args.context)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt["model_level"]))
    print(json.dumps({k: {kk: (vv.get("median_ms") if isinstance(vv, dict) and "median_ms" in vv else vv)
                          for kk, vv in v.items()} for k, v in receipt["components"].items()}))


if __name__ == "__main__":
    main()

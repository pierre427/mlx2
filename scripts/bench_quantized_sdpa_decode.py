#!/usr/bin/env python3
"""Decode-attention microbenchmark: why is mlx2's quantized KV path slow?

Times one attention call (``L`` query rows) against caches built the way
serving builds them: a fp16 ``KVCache`` and its ``to_quantized`` conversion,
grown past a 256-row step so the views are strided like a live cache.

Backends:

* ``fp16``: ``models.base.scaled_dot_product_attention`` on the fp16 cache
  (``mx.fast`` SDPA), the baseline;
* ``current``: ``models.base.quantized_scaled_dot_product_attention``, the
  serving path. It uses two ``quantized_matmul`` calls, with GQA as a
  broadcast batch axis;
* ``gqa_rows``: the same two matmuls with GQA folded into the row
  dimension, so each KV head's rows are read once;
* ``fused``: ``mx.fast.quantized_scaled_dot_product_attention``, affine
  mode (equal K/V bits only);
* ``dequant``: dequantize K/V, then ``mx.fast`` SDPA.

It prints median ms and effective GB/s of cache bytes, and checks each
output against fp16. GPU only: it refuses without ``--i-own-the-gpu``; run
it under the lab waiters wrapper.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

GEOMETRIES = {  # full-attention layer shapes
    "qwen36-35b-a3b": (16, 2, 256),
    "qwen35-9b": (16, 4, 256),
    "qwen38-27b": (24, 4, 256),
}


def gqa_rows_attention(queries, q_keys, q_values, scale, group_size, key_bits, value_bits):
    """Composed quantized attention with GQA folded into rows (decode, no mask)."""
    import mlx.core as mx

    B, Hq, L, D = queries.shape
    Hkv = q_keys[0].shape[1]
    rows = (queries * scale).reshape(B, Hkv, (Hq // Hkv) * L, D)
    scores = mx.quantized_matmul(rows, *q_keys, transpose=True,
                                 group_size=group_size, bits=key_bits)
    scores = mx.softmax(scores, axis=-1, precise=True)
    out = mx.quantized_matmul(scores, *q_values, transpose=False,
                              group_size=group_size, bits=value_bits)
    return out.reshape(B, Hq, L, D)


def bench(fn, iters, warmup):
    import mlx.core as mx

    for _ in range(warmup):
        mx.eval(fn())
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(fn())
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def run_case(geometry, context, L, bits, iters, warmup, dtype_name):
    import mlx.core as mx

    from mlx2.runtime.models.base import (
        quantized_scaled_dot_product_attention,
        scaled_dot_product_attention,
    )
    from mlx2.runtime.models.cache import KVCache

    Hq, Hkv, D = GEOMETRIES[geometry]
    dtype = getattr(mx, dtype_name)
    key_bits, value_bits = bits
    scale = D ** -0.5
    mx.random.seed(0)
    cache = KVCache()
    grown = 0
    while grown < context:  # grow like prefill, then leave a strided tail
        n = min(2048, context - grown)
        cache.update_and_fetch(mx.random.normal((1, Hkv, n, D)).astype(dtype),
                               mx.random.normal((1, Hkv, n, D)).astype(dtype))
        grown += n
    extra = 37
    cache.update_and_fetch(mx.random.normal((1, Hkv, extra, D)).astype(dtype),
                           mx.random.normal((1, Hkv, extra, D)).astype(dtype))
    qcache = cache.to_quantized(group_size=64, key_bits=key_bits, value_bits=value_bits)
    qcache.update_and_fetch(mx.random.normal((1, Hkv, 1, D)).astype(dtype),
                            mx.random.normal((1, Hkv, 1, D)).astype(dtype))
    cache.update_and_fetch(mx.random.normal((1, Hkv, 1, D)).astype(dtype),
                           mx.random.normal((1, Hkv, 1, D)).astype(dtype))
    k, v = cache.keys_and_values()
    qk, qv = qcache.keys_and_values()
    mx.eval(k, v, qk, qv)
    q = mx.random.normal((1, Hq, L, D)).astype(dtype)
    T = k.shape[2]

    fp16_bytes = 2 * Hkv * T * D * mx.array(0, dtype).itemsize
    q_bytes = sum(x.nbytes for x in qk) * T / qk[0].shape[2] + sum(x.nbytes for x in qv) * T / qv[0].shape[2]

    backends = {
        "fp16": lambda: scaled_dot_product_attention(q, k, v, cache=cache, scale=scale, mask=None),
        # Copy q: the serving path scales queries in place (`queries *= scale`).
        "current": lambda: quantized_scaled_dot_product_attention(
            q * 1, qk, qv, scale=scale, mask=None, group_size=64,
            key_bits=key_bits, value_bits=value_bits),
        "gqa_rows": lambda: gqa_rows_attention(q, qk, qv, scale, 64, key_bits, value_bits),
        "dequant": lambda: mx.fast.scaled_dot_product_attention(
            q, mx.dequantize(*qk, group_size=64, bits=key_bits),
            mx.dequantize(*qv, group_size=64, bits=value_bits), scale=scale),
    }
    if key_bits == value_bits:
        backends["fused"] = lambda: mx.fast.quantized_scaled_dot_product_attention(
            q, qk[0], qk[1], qk[2], qv[0], qv[1], qv[2], scale=scale,
            group_size=64, bits=key_bits, mode="affine")

    ref = backends["fp16"]().astype(mx.float32)
    row = {"geometry": geometry, "context": T, "L": L, "bits": f"k{key_bits}v{value_bits}",
           "dtype": dtype_name}
    for name, fn in backends.items():
        out = fn().astype(mx.float32)
        err = float((mx.abs(out - ref).max() / (mx.abs(ref).max() + 1e-9)).item())
        ms = bench(fn, iters, warmup)
        nbytes = fp16_bytes if name == "fp16" else q_bytes
        row[name] = {"ms": round(ms, 4), "gbps": round(nbytes / ms / 1e6, 1),
                     "rel_err_vs_fp16": round(err, 5)}
    return row


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--geometries", default=",".join(GEOMETRIES))
    p.add_argument("--contexts", default="16384,32768,65536,131072")
    p.add_argument("--rows", default="1", help="query rows L (decode 1; MTP verify 2-4)")
    p.add_argument("--bits", default="8/8,8/4", help="key/value bits pairs")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--i-own-the-gpu", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    cases = [(g, int(c), int(L), tuple(int(b) for b in bits.split("/")))
             for g in args.geometries.split(",") for c in args.contexts.split(",")
             for L in args.rows.split(",") for bits in args.bits.split(",")]
    if args.dry_run:
        print(json.dumps({"cases": len(cases), "first": cases[:3]}, default=str))
        return 0
    if not args.i_own_the_gpu:
        raise SystemExit("refusing Metal execution without --i-own-the-gpu")

    import mlx.core as mx

    mx.set_cache_limit(2 * 1024**3)
    rows = []
    for g, c, L, bits in cases:
        row = run_case(g, c, L, bits, args.iters, args.warmup, args.dtype)
        rows.append(row)
        brief = {k: row[k] for k in ("geometry", "context", "L", "bits")}
        brief |= {n: row[n]["ms"] for n in ("fp16", "current", "gqa_rows", "fused", "dequant") if n in row}
        print(json.dumps(brief), flush=True)
        mx.clear_cache()
    report = {"mlx_version": mx.__version__, "device": mx.device_info() if hasattr(mx, "device_info") else None,
              "rows": rows}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

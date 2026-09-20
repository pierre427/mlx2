"""rm15 step 0: does chunking change PURE op results on CPU?

Three independent sub-probes, all CPU, all exact-arithmetic paths:
  (1) matmul: y = x @ W computed with x split into row-chunks vs whole.
      Row-chunking must NOT change results (each output row is an
      independent dot product) unless the kernel changes strategy with M.
  (2) SDPA: attention over the same KV, queries issued in chunks vs whole.
      Each query row is independent -> must be identical.
  (3) SDPA with GROWING KV (the prefill shape): chunk i attends to keys
      [0, end_i).  The *last* chunk's queries see exactly the same keys as
      the unchunked call, so rows must match if the kernel is row-independent.
"""
import itertools, json, sys
import mlx.core as mx

mx.set_default_device(mx.cpu)
mx.random.seed(0)

out = {"device": "cpu", "mlx": None, "probes": []}
try:
    import mlx.core as _m
    out["mlx"] = _m.__version__
except Exception:
    pass


def maxdiff(a, b):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def probe_matmul(M, K, N, dtype, chunks):
    x = mx.random.normal((M, K)).astype(dtype)
    W = mx.random.normal((K, N)).astype(dtype)
    full = x @ W
    mx.eval(full)
    res = {}
    for c in chunks:
        parts = [x[i:i + c] @ W for i in range(0, M, c)]
        y = mx.concatenate(parts, axis=0)
        mx.eval(y)
        res[str(c)] = maxdiff(full, y)
    return {"probe": "matmul", "M": M, "K": K, "N": N, "dtype": str(dtype),
            "max_abs_delta_vs_unchunked": res}


def probe_sdpa_static(Lq, Lk, H, D, dtype, chunks):
    q = mx.random.normal((1, H, Lq, D)).astype(dtype)
    k = mx.random.normal((1, H, Lk, D)).astype(dtype)
    v = mx.random.normal((1, H, Lk, D)).astype(dtype)
    s = D ** -0.5
    full = mx.fast.scaled_dot_product_attention(q, k, v, scale=s)
    mx.eval(full)
    res = {}
    for c in chunks:
        parts = [mx.fast.scaled_dot_product_attention(
            q[:, :, i:i + c], k, v, scale=s) for i in range(0, Lq, c)]
        y = mx.concatenate(parts, axis=2)
        mx.eval(y)
        res[str(c)] = maxdiff(full, y)
    return {"probe": "sdpa_static", "Lq": Lq, "Lk": Lk, "H": H, "D": D,
            "dtype": str(dtype), "max_abs_delta_vs_unchunked": res}


def probe_sdpa_causal_prefill(L, H, D, dtype, chunks):
    """Full causal prefill in one call vs chunked with growing KV.
    Compares the FINAL row (position L-1), which sees identical keys both ways.
    """
    q = mx.random.normal((1, H, L, D)).astype(dtype)
    k = mx.random.normal((1, H, L, D)).astype(dtype)
    v = mx.random.normal((1, H, L, D)).astype(dtype)
    s = D ** -0.5
    full = mx.fast.scaled_dot_product_attention(q, k, v, scale=s, mask="causal")
    mx.eval(full)
    res = {}
    for c in chunks:
        parts = []
        for i in range(0, L, c):
            e = min(i + c, L)
            qi = q[:, :, i:e]
            ki, vi = k[:, :, :e], v[:, :, :e]
            m = mx.arange(i, e)[:, None] >= mx.arange(0, e)[None, :]
            parts.append(mx.fast.scaled_dot_product_attention(qi, ki, vi, scale=s, mask=m))
        y = mx.concatenate(parts, axis=2)
        mx.eval(y)
        res[str(c)] = {"all_rows": maxdiff(full, y),
                       "last_row": maxdiff(full[:, :, -1], y[:, :, -1])}
    return {"probe": "sdpa_causal_prefill", "L": L, "H": H, "D": D,
            "dtype": str(dtype), "max_abs_delta_vs_unchunked": res}


for dt in (mx.float32, mx.bfloat16, mx.float16):
    out["probes"].append(probe_matmul(512, 256, 256, dt, [64, 128, 256]))
    out["probes"].append(probe_sdpa_static(512, 1024, 4, 64, dt, [64, 128, 256]))
    out["probes"].append(probe_sdpa_causal_prefill(512, 4, 64, dt, [64, 128, 256]))

print(json.dumps(out, indent=2))

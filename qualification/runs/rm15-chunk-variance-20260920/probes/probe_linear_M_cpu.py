"""rm15 step 2c: MINIMAL repro -- a dense matmul's result for a FIXED row
depends on how many rows are in the call (M).  CPU, float32.

y = x @ W.T  for x[1, S, K], W[N, K].  Compare row S-1 of the S-row call with
the same row computed in a call of M rows.  Any non-zero delta means the
kernel's reduction strategy is keyed on M -- which is exactly what changing
the prefill chunk size does to every linear layer in the model.
"""
import json
import mlx.core as mx
import mlx.nn as nn

mx.set_default_device(mx.cpu)
mx.random.seed(0)

res = []
for N in (4, 8, 16, 32, 64, 256, 1024):
    for K in (64, 256, 1024):
        for dt in (mx.float32,):
            W = mx.random.normal((N, K)).astype(dt)
            S = 512
            x = mx.random.normal((1, S, K)).astype(dt)
            full = (x @ W.T)[0, -1]
            mx.eval(full)
            row = {}
            for M in (1, 2, 8, 16, 32, 64, 128, 256):
                y = (x[:, S - M:S] @ W.T)[0, -1]
                mx.eval(y)
                row[M] = float(mx.max(mx.abs(y - full)).item())
            res.append({"N": N, "K": K, "dtype": str(dt),
                        "S": S, "delta_by_M": row,
                        "any": any(v > 0 for v in row.values())})

print(json.dumps(res, indent=2))

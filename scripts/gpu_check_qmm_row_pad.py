"""GPU check for qmm_row_pad on MLX 0.32.4 / M5: speed, error, row invariance.

For each flagship projection shape and row count, time stock
``QuantizedLinear`` against the row-padded instance (pad, matmul and slice
included), and compare both with an fp32 reference of the dequantized
weights.  Row invariance: a row's padded result must not depend on which
other rows share the call.

Usage: python scripts/gpu_check_qmm_row_pad.py OUT.json
"""

import json
import sys
import time

import mlx.core as mx
import mlx.nn as nn

from mlx2.runtime.models import qmm_row_pad as rp

SHAPES = {
    "FN o": (6144, 2560), "FN gdn_out": (6144, 2560), "FN shared_down": (640, 2560),
    "35B q_gate": (2048, 8192), "35B o": (4096, 2048),
    "27B o": (6144, 5120), "27B kv": (5120, 1024), "27B down": (17408, 5120),
    "27B gdn_out": (6144, 5120),
}
MS = [13, 16, 24, 32, 40, 48, 64, 96, 128]
NW, REPS, ROUNDS = 6, 10, 7


def layers(K, N):
    out = []
    for i in range(NW):
        layer = nn.Linear(K, N, bias=False)
        layer.weight = (0.02 * mx.random.normal((N, K), key=mx.random.key(i))).astype(mx.bfloat16)
        out.append(nn.QuantizedLinear.from_linear(layer, group_size=64, bits=4))
    mx.eval([l.parameters() for l in out])
    return out


def timed(ls, x):
    def run():
        mx.eval([l(x) for _ in range(REPS) for l in ls])
    run(); run()
    best = float("inf")
    for _ in range(ROUNDS):
        t = time.perf_counter(); run(); best = min(best, time.perf_counter() - t)
    return best / (REPS * NW) * 1e6


def main(out_path):
    print(mx.device_info()["architecture"], mx.__version__, rp.refusal(), flush=True)
    rows = []
    for name, (K, N) in SHAPES.items():
        ls = layers(K, N)
        ref_w = mx.dequantize(ls[0].weight, ls[0].scales, ls[0].biases, group_size=64, bits=4)
        for M in MS:
            target = rp.pad_target(M, K, N, 64)
            if target is None:
                continue
            x = mx.random.normal((M, K), key=mx.random.key(1000 + M)).astype(mx.bfloat16)
            ref = x.astype(mx.float32) @ ref_w.astype(mx.float32).T
            stock_y = ls[0](x)
            stock_us = timed(ls, x)
            for l in ls:
                l.__class__ = rp._subclass(nn.QuantizedLinear)
                l._row_pad_max_split = rp.MAX_SPLIT
            try:
                pad_y = ls[0](x)
                pad_us = timed(ls, x)
                # Row invariance: the first rows alone (padded to the same or
                # a different target) and inside a larger call.
                sub = ls[0](x[: max(rp.vector_limit(K, N), M // 2)])
                inv = bool(mx.array_equal(sub, pad_y[: sub.shape[0]]).item()) \
                    if rp.pad_target(sub.shape[0], K, N, 64) is not None else None
            finally:
                for l in ls:
                    l.__class__ = nn.QuantizedLinear
            err = lambda y: float(mx.abs(y.astype(mx.float32) - ref).mean().item())
            row = {
                "shape": name, "K": K, "N": N, "M": M, "target": target,
                "split": rp.split_k(M, N, K, 64),
                "stock_us": round(stock_us, 2), "row_pad_us": round(pad_us, 2),
                "speedup": round(stock_us / pad_us, 3),
                "stock_err": err(stock_y), "row_pad_err": err(pad_y),
                "bit_identical_to_stock": bool(mx.array_equal(stock_y, pad_y).item()),
                "row_invariant_subset": inv,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
        del ls
        mx.clear_cache()
    json.dump(rows, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])

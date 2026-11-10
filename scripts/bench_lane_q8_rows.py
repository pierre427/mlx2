"""Stock MLX vs mlx2 lane matmul, affine 8-bit gs64 bf16, Qwen3.8-27B dense MLP+attn shapes, M=1..8 rows.

Usage: PYTHONPATH=src .venv/bin/python scripts/bench_lane_q8_rows.py   (under the GPU lock)
"""
import time, json
import mlx.core as mx
from mlx import nn
from mlx2.runtime.lane import matmul as lane
mx.set_default_device(mx.gpu)
print("backend", lane.backend(), "available", lane.available(), "mlx", mx.__version__)
SHAPES = [(5120, 17408), (5120, 17408), (17408, 5120), (5120, 12288), (5120, 1024), (5120, 1024), (6144, 5120)]
REPS = 6  # layers' worth of distinct weights, ~1.9 GB at q8 -> no cache residency
layers = []
for r in range(REPS):
    for k, n in SHAPES:
        l = nn.Linear(k, n, bias=False); l.weight = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
        q = nn.QuantizedLinear.from_linear(l, group_size=64, bits=8)
        mx.eval(q.parameters()); layers.append((q, lane.prepare(q)))
wbytes = sum(q.weight.nbytes + q.scales.nbytes + q.biases.nbytes for q, _ in layers)
print("weight GB", round(wbytes / 1e9, 2))
def bench(m, use_lane):
    xs = [mx.random.normal((m, q.weight.shape[1] * 32 // 8)).astype(mx.bfloat16) for q, _ in layers]
    mx.eval(xs)
    def once():
        outs = [lane.lane_matmul(x, lw) if use_lane else q(x) for x, (q, lw) in zip(xs, layers)]
        mx.eval(outs); return outs
    once(); once()
    ts = []
    for _ in range(15):
        t = time.perf_counter(); once(); ts.append(time.perf_counter() - t)
    ts.sort(); return ts[len(ts) // 2], once()
for m in range(1, 9):
    ts, so = bench(m, False); tl, lo = bench(m, True)
    diff = max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item() for a, b in zip(so, lo))
    passes = lane.weight_passes(layers[0][1], m)
    print(json.dumps(dict(rows=m, stock_ms=round(ts * 1e3, 3), lane_ms=round(tl * 1e3, 3),
          stock_GBps=round(wbytes / ts / 1e9, 1), lane_GBps=round(wbytes / tl / 1e9, 1),
          lane_over_stock=round(ts / tl, 3), lane_weight_passes=passes, max_abs_diff=diff)), flush=True)

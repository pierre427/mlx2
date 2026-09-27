"""GPU gate for the omlx #3912 port: bit-exactness and a launch-chain microbench.

Builds one Flash-Next-shaped routed switch (512 experts, hidden 2560,
intermediate 640, affine q4/g64, bf16) with random weights and compares, for
random one-token routings:

  stock         gather_qmm gate+up, swiglu, gather_qmm down, (x*scores).sum
  tile4         gather_qmm gate+up, swiglu, mlx2 tile4 fused down (FN profile)
  gate_up       #3912 gate+up/SwiGLU kernel, then the tile4 fused down
  gate_up_stock #3912 gate+up/SwiGLU kernel, then the stock down tail
  two_launch    #3912 gate+up/SwiGLU kernel + #3912 down/weighted-sum kernel

Expected: gate_up == tile4 and gate_up_stock == two_launch == stock, bit for bit.

The microbench chains ``--layers`` routed calls (fresh routing per layer, one
eval per chain, the decode dependency shape) and times each arm, rotating the
arm order per rep after a discarded warm-up.

  PYTHONPATH=src .venv/bin/python scripts/check_fn_routed_decode.py --i-own-the-gpu \
      --out qualification/runs/fn-omlx-ab-20260925/routed-micro.json
"""

import argparse
import json
import statistics
import time

import mlx.core as mx
import mlx.nn as nn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-own-the-gpu", action="store_true")
    ap.add_argument("--cases", type=int, default=64)
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")
    mx.set_cache_limit(4 << 30)

    from mlx2.runtime.models import qwen3_next as QN
    from mlx2.runtime.models import qwen4_routed_decode as RD

    mx.random.seed(0)
    E, H, I = 512, 2560, 640
    sw = QN.FusedGateUpSwitchGLU(H, I, E)
    sw.gate_up_proj.weight = (mx.random.normal((E, 2 * I, H)) * 0.02).astype(mx.bfloat16)
    sw.down_proj.weight = (mx.random.normal((E, H, I)) * 0.02).astype(mx.bfloat16)
    nn.quantize(sw, group_size=64, bits=4)
    sw.set_dtype(mx.bfloat16)
    QN._enable_routed_decode(sw, "off")
    sw.eval()
    mx.eval(sw.parameters())
    print("types", type(sw.gate_up_proj).__name__, sw.gate_up_proj.scales.dtype, flush=True)

    def route(key):
        gates = mx.random.normal((1, 1, E), key=key).astype(mx.bfloat16)
        gates = mx.softmax(gates, axis=-1, precise=True)
        inds = mx.argpartition(gates, kth=-10, axis=-1)[..., -10:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = scores / scores.sum(axis=-1, keepdims=True)
        return inds, scores

    def run(arm, x, inds, scores):
        mode = {"stock": "off", "tile4": "off", "gate_up": "gate_up",
                "gate_up_stock": "gate_up", "two_launch": "two_launch"}[arm]
        variant = "stock" if arm in ("stock", "gate_up_stock") else "tile4"
        sw.routed_decode_mode = mode
        return sw(x, inds, scores=scores, variant=variant)

    arms = ["stock", "tile4", "gate_up", "gate_up_stock", "two_launch"]
    exact = {f"{b}=={r}": 0 for b, r in (("gate_up", "tile4"), ("gate_up_stock", "stock"),
                                         ("two_launch", "stock"), ("tile4", "stock"))}
    maxdiff = {k: 0.0 for k in exact}
    hidden_exact = 0
    for c in range(a.cases):
        key = mx.random.key(1000 + c)
        x = (mx.random.normal((1, 1, H), key=mx.random.key(c)) * 1.5).astype(mx.bfloat16)
        inds, scores = route(key)
        outs = {arm: run(arm, x, inds, scores) for arm in arms}
        mx.eval(outs)
        for k in exact:
            b, r = k.split("==")
            exact[k] += bool(mx.array_equal(outs[b], outs[r]).item())
            maxdiff[k] = max(maxdiff[k], mx.abs(outs[b].astype(mx.float32) - outs[r].astype(mx.float32)).max().item())
        xe = mx.expand_dims(x, (-2, -3))
        gu = sw.gate_up_proj(xe, inds)
        ref_h = sw.activation(gu[..., I:], gu[..., :I]).reshape(10, I)
        ker_h = RD.gate_up_swiglu(xe, inds, sw.gate_up_proj)
        hidden_exact += bool(mx.array_equal(ref_h, ker_h).item())
    print("exact", exact, "hidden_exact", hidden_exact, "/", a.cases, flush=True)

    # Launch-chain microbench.
    xs = [(mx.random.normal((1, 1, H), key=mx.random.key(5000 + i)) * 1.5).astype(mx.bfloat16)
          for i in range(a.layers)]
    routes = [route(mx.random.key(9000 + i)) for i in range(a.layers)]
    mx.eval(xs, routes)

    def chain(arm):
        y = xs[0]
        for i in range(a.layers):
            inds, scores = routes[i]
            y = run(arm, (y + xs[i]).astype(mx.bfloat16), inds, scores).reshape(1, 1, H)
        t0 = time.perf_counter()
        mx.eval(y)
        return time.perf_counter() - t0

    bench_arms = ["tile4", "gate_up", "two_launch", "stock"]
    for arm in bench_arms:  # warm-up, discarded
        chain(arm)
    times = {arm: [] for arm in bench_arms}
    for rep in range(a.reps):
        order = bench_arms[rep % 4:] + bench_arms[: rep % 4]
        if rep % 2:
            order = order[::-1]
        for arm in order:
            times[arm].append(chain(arm) * 1e3)
    summary = {arm: {"median_ms": statistics.median(v), "min_ms": min(v), "max_ms": max(v),
                     "us_per_layer": 1e3 * statistics.median(v) / a.layers}
               for arm, v in times.items()}
    base = summary["tile4"]["median_ms"]
    for arm in summary:
        summary[arm]["delta_vs_tile4_pct"] = 100 * (summary[arm]["median_ms"] / base - 1)
    rec = {"cases": a.cases, "bit_exact_counts": exact, "max_abs_diff": maxdiff,
           "gate_up_hidden_exact": hidden_exact, "layers": a.layers, "reps": a.reps,
           "chain_ms": times, "summary": summary, "mlx": mx.__version__,
           "device": mx.device_info().get("device_name")}
    json.dump(rec, open(a.out, "w"), indent=1)
    print(json.dumps({k: rec[k] for k in ("bit_exact_counts", "max_abs_diff", "gate_up_hidden_exact", "summary")}, indent=1))


if __name__ == "__main__":
    main()

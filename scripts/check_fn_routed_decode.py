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

``--candidate`` checks the default-off omlx #4113 candidate instead: split
gate/up tables, top-k 8 or 10, and the down traversal MLX picks for the
shape (``qmv_fast`` at Qwen3.6's intermediate 512). Defaults are the Qwen3.6
geometry (256 experts, hidden 2048, intermediate 512, top-8, q4/g64, bf16);
``--experts/--hidden/--inter/--topk`` override them. It is an exact route, so
the verdict is ``pass`` only if every case engaged the candidate (one call,
no fallback) and the gate+up hidden, the isolated down/combine and the whole
output are bit-identical to the composed reference; anything else is
``refused`` (exit 1), never tolerated drift. The served SiLU form is probed
first and a mismatch refuses.

  PYTHONPATH=src .venv/bin/python scripts/check_fn_routed_decode.py --candidate \
      --i-own-the-gpu --out qualification/runs/<run>/routed-candidate-qwen36.json

``--cpu-dry-run`` (candidate only) runs a tiny geometry on the CPU with the
kernels replaced by composed references: it exercises admission, counters and
the verdict logic, never Metal numerics, and needs no GPU ownership.
"""

import argparse
import json
import statistics
import time

import mlx.core as mx
import mlx.nn as nn


def _bits(a):
    return a.view(mx.uint16) if a.dtype.size == 2 else a.view(mx.uint32)


def _same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(_bits(a), _bits(b)).item())


def _install_reference_kernels(RD, QN):
    """CPU dry run only: composed stand-ins for the Metal kernels."""

    def gate_up(x, indices, gate, up):
        return QN.SwiGLU()(up(x, indices), gate(x, indices)).reshape(indices.size, -1)

    def down(h, indices, scores, down_proj):
        y = down_proj(h.reshape(indices.shape + (1, h.shape[-1])), indices).squeeze(-2)
        return (y * scores[..., None]).sum(axis=-2).reshape(-1)

    RD.candidate_gate_up_swiglu = gate_up
    RD.candidate_down_combine = down
    RD.candidate_runtime_refusal = lambda: None


def run_candidate(a):
    from mlx2.runtime.models import qwen3_next as QN
    from mlx2.runtime.models import qwen4_routed_decode as RD

    defaults = (16, 512, 512, 8) if a.cpu_dry_run else (256, 2048, 512, 8)
    E, H, I, K = (
        default if value is None else value
        for value, default in zip((a.experts, a.hidden, a.inter, a.topk), defaults)
    )
    if min(E, H, I) <= 0 or K not in RD.CANDIDATE_TOP_K or E < K:
        raise ValueError(f"invalid geometry E={E} H={H} I={I} topk={K}")
    if a.cases <= 0 or a.layers < 0 or a.reps < 0:
        raise ValueError("--cases must be positive; --layers/--reps non-negative")
    if a.cpu_dry_run:
        mx.set_default_device(mx.cpu)
        _install_reference_kernels(RD, QN)
    refusals = []
    silu = None
    if not a.cpu_dry_run:
        from mlx2.runtime.models.qwen4_fused_gdn import served_silu_exp

        silu = served_silu_exp()
        reason = RD.candidate_runtime_refusal()
        if reason is not None:
            refusals.append(f"runtime: {reason}")

    mx.random.seed(0)
    sw = QN.FusedDownSwitchGLU(H, I, E)
    for name, shape in (("gate_proj", (E, I, H)), ("up_proj", (E, I, H)), ("down_proj", (E, H, I))):
        getattr(sw, name).weight = mx.random.normal(shape, dtype=mx.bfloat16) * 0.02
    nn.quantize(sw, group_size=64, bits=4)
    sw.set_dtype(mx.bfloat16)
    QN._enable_routed_decode(sw, "off")
    QN._enable_routed_candidate(sw, "off")
    sw.eval()
    mx.eval(sw.parameters())

    def route(key):
        gates = mx.softmax(mx.random.normal((1, 1, E), key=key).astype(mx.bfloat16), axis=-1, precise=True)
        inds = mx.argpartition(gates, kth=-K, axis=-1)[..., -K:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        return inds, scores / scores.sum(axis=-1, keepdims=True)

    probe_x = mx.zeros((1, 1, 1, 1, H), dtype=mx.bfloat16)
    probe_i, probe_s = route(mx.random.key(0))
    admission = RD.admit_routed_candidate(probe_x, probe_i, probe_s, sw.gate_proj, sw.up_proj, sw.down_proj)
    if not admission.accepted:
        refusals.append(f"admission: {admission.reason}")
    counts = {"cases": 0, "engaged": 0, "hidden_exact": 0, "down_exact": 0, "output_exact": 0}
    maxdiff = 0.0
    for c in range(a.cases if not refusals else 0):
        x = (mx.random.normal((1, 1, H), key=mx.random.key(c)) * 1.5).astype(mx.bfloat16)
        inds, scores = route(mx.random.key(1000 + c))
        sw.routed_candidate_mode = "off"
        want = sw(x, inds, scores=scores, variant="stock")
        calls, fallbacks = sw.routed_candidate_calls, sw.routed_candidate_fallbacks
        sw.routed_candidate_mode = "two_launch"
        got = sw(x, inds, scores=scores, variant="stock")
        mx.eval(want, got)
        engaged = sw.routed_candidate_calls == calls + 1 and sw.routed_candidate_fallbacks == fallbacks
        if not engaged:
            refusals.append(f"case {c}: not engaged ({sw.routed_candidate_last_fallback})")
            break
        counts["cases"] += 1
        counts["engaged"] += 1
        xe = mx.expand_dims(x, (-2, -3))
        ref_h = QN.SwiGLU()(sw.up_proj(xe, inds), sw.gate_proj(xe, inds)).reshape(K, I)
        ker_h = RD.candidate_gate_up_swiglu(xe, inds, sw.gate_proj, sw.up_proj)
        y = sw.down_proj(ref_h.reshape(inds.shape + (1, I)), inds).squeeze(-2)
        ref_d = (y * scores[..., None]).sum(axis=-2).reshape(-1)
        ker_d = RD.candidate_down_combine(ref_h, inds, scores, sw.down_proj)
        counts["hidden_exact"] += _same(ref_h, ker_h)
        counts["down_exact"] += _same(ref_d, ker_d)
        counts["output_exact"] += _same(want, got)
        maxdiff = max(maxdiff, mx.abs(got.astype(mx.float32) - want.astype(mx.float32)).max().item())
    for key in ("hidden_exact", "down_exact", "output_exact"):
        if counts[key] != counts["cases"]:
            refusals.append(f"{key} {counts[key]}/{counts['cases']}")
    if not refusals and (a.cases <= 0 or counts["cases"] != a.cases):
        refusals.append("incomplete: no or too few cases checked")

    timing = None
    if not refusals and not a.cpu_dry_run and a.layers > 0:
        xs = [(mx.random.normal((1, 1, H), key=mx.random.key(5000 + i)) * 1.5).astype(mx.bfloat16)
              for i in range(a.layers)]
        routes = [route(mx.random.key(9000 + i)) for i in range(a.layers)]
        mx.eval(xs, routes)

        def chain(arm):
            sw.routed_candidate_mode = "two_launch" if arm == "candidate" else "off"
            y = xs[0]
            for i in range(a.layers):
                inds, scores = routes[i]
                y = sw((y + xs[i]).astype(mx.bfloat16), inds, scores=scores, variant="stock").reshape(1, 1, H)
            t0 = time.perf_counter()
            mx.eval(y)
            return time.perf_counter() - t0

        arms = ["stock", "candidate"]
        for arm in arms:  # warm-up, discarded
            chain(arm)
        times = {arm: [] for arm in arms}
        for rep in range(a.reps):
            for arm in (arms if rep % 2 == 0 else arms[::-1]):
                times[arm].append(chain(arm) * 1e3)
        timing = {arm: {"median_ms": statistics.median(v), "min_ms": min(v), "max_ms": max(v)}
                  for arm, v in times.items()}
        timing["scope"] = "synthetic routed-switch chain microbench; not a model A/B"

    rec = {
        "mechanism": "omlx#4113 routed-decode candidate (split gate/up, top-k, MLX down traversal)",
        "verdict": "refused" if refusals else ("dry-run-pass" if a.cpu_dry_run else "pass"),
        "refusals": refusals,
        "scope": ("CPU dry run with composed reference kernels; no Metal numerics"
                  if a.cpu_dry_run else "Metal kernels vs composed MLX reference at this geometry and build"),
        "geometry": {"experts": E, "hidden": H, "inter": I, "topk": K, "bits": 4, "group_size": 64,
                     "dtype": "bfloat16", "down_traversal": "qmv_fast" if RD.qmv_fast_layout(I, H) else "qmv"},
        "counts": counts, "max_abs_diff": maxdiff, "served_silu_exp": silu,
        "candidate_calls": sw.routed_candidate_calls, "candidate_fallbacks": sw.routed_candidate_fallbacks,
        "timing": timing, "mlx": mx.__version__,
        "device": "cpu" if a.cpu_dry_run else mx.device_info().get("device_name"),
    }
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps({k: rec[k] for k in ("verdict", "refusals", "geometry", "counts")}, indent=1))
    return 1 if refusals else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-own-the-gpu", action="store_true")
    ap.add_argument("--cases", type=int, default=64)
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--out", required=True)
    ap.add_argument("--candidate", action="store_true", help="check the omlx #4113 candidate")
    ap.add_argument("--experts", type=int)
    ap.add_argument("--hidden", type=int)
    ap.add_argument("--inter", type=int)
    ap.add_argument("--topk", type=int, choices=(8, 10))
    ap.add_argument("--cpu-dry-run", action="store_true")
    a = ap.parse_args()
    if not a.candidate and (a.cpu_dry_run or any(v is not None for v in (a.experts, a.hidden, a.inter, a.topk))):
        ap.error("--experts/--hidden/--inter/--topk/--cpu-dry-run need --candidate")
    if a.candidate and (a.cases <= 0 or a.layers < 0 or a.reps < 0
                        or any(v is not None and v <= 0 for v in (a.experts, a.hidden, a.inter))):
        ap.error("--cases and geometry overrides must be positive; --layers/--reps non-negative")
    if a.candidate and a.cpu_dry_run:
        if a.i_own_the_gpu:
            ap.error("--cpu-dry-run never touches the GPU; drop --i-own-the-gpu")
        raise SystemExit(run_candidate(a))
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")
    if a.candidate:
        mx.set_cache_limit(4 << 30)
        raise SystemExit(run_candidate(a))
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

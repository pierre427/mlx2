#!/usr/bin/env python3
"""Which GDN prefill implementation is fastest at Flash-Next geometry?

Production prefill runs ``gated_delta_kernel``, a Metal kernel whose body is
``for (int t = 0; t < T; ++t)`` -- strictly sequential in time, parallel only
over batch x value-head x value-row.  MLX also ships a *chunked* primitive,
``mx.fast.gated_delta_update``, whose docstring says ``chunk_size=-1`` selects
"sequential for T <= threshold and chunk-16 otherwise", with explicit choices
``0``/``1`` (sequential), ``8`` (SIMD-group) and ``16`` (NAX).

mlx2 wires that primitive at ``gated_delta.py:385`` but never reaches it during
a real prefill: ``_ENABLE_GDN_CORE`` defaults to "0" and is hard-set "0" by all
four adapters, and ``_CORE_GDN_MAX_T = 256`` is far below any production
``prefill_step``.  So the chunk-parallel form is present in the dependency and
structurally unreachable.  This harness measures whether that matters.

Arms, all at the Flash-Next GDN geometry (Hk=16, Hv=48, Dk=Dv=128, B=1) and the
production dtypes (bf16 activations, fp32 gate/state):

  ops        gated_delta_ops, the pure-MLX sequential reference (oracle)
  kernel     gated_delta_kernel, what production actually runs
  core_seq   mx.fast.gated_delta_update(chunk_size=1)
  core_8     mx.fast.gated_delta_update(chunk_size=8)   -- SIMD-group
  core_16    mx.fast.gated_delta_update(chunk_size=16)  -- NAX
  core_auto  mx.fast.gated_delta_update(chunk_size=-1, threshold=16)

Two questions are answered per T, and they are kept separate:

  1. CORRECTNESS, against two different references, because they ask different
     things.  ``vs_ops`` compares against ``gated_delta_ops``, the pure-MLX
     sequential recurrence -- "is this faithful to the mathematics?"  ``vs_kernel``
     compares against ``gated_delta_kernel``, what production runs today --
     "would swapping the path change output?"  Neither implies the other: the
     Metal kernel is itself not bit-exact against the ops loop.  ``ops`` is only
     used up to --oracle-max-t because it is a Python loop over T.  ``bit_exact``
     means max_abs_delta == 0.0, which is the bar this repo holds its GDN
     kernels to -- not a tolerance.
  2. SPEED.  Warmup, then --rounds interleaved rounds; the per-arm median is
     reported with ``speedup_vs_kernel``.  Interleaving matters: a sequential
     sweep lets thermal drift land on whichever arm runs last.

The oracle is excluded from timing unless --time-ops is passed.

Metal: run under the GPU queue only.  --dry-run prints the plan and exits.
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

SCHEMA = "mlx2.gdn-prefill-arms.v1"

# Flash-Next GDN geometry: qwen4_exp.ModelArgs linear_num_key_heads=16,
# linear_num_value_heads=48, linear_key_head_dim=linear_value_head_dim=128.
GEOMETRY = {"B": 1, "Hk": 16, "Hv": 48, "Dk": 128, "Dv": 128}

ARMS = ("ops", "kernel", "core_seq", "core_8", "core_16", "core_auto")


def build_inputs(mx, T, seed):
    """Synthetic activations shaped and scaled the way the model feeds them.

    q/k pass through normalize_gdn_qk in production, so L2-normalize them here
    with the delta-rule query scale folded in; otherwise the delta rule sees an
    unrealistically large ||k|| and the state magnitude -- hence the rounding
    behaviour -- is not representative.
    """
    from mlx2.runtime.models.gated_delta import normalize_gdn_qk

    B, Hk, Hv, Dk, Dv = (GEOMETRY[k] for k in ("B", "Hk", "Hv", "Dk", "Dv"))
    mx.random.seed(seed)
    key = mx.random.normal((B, T, Hk, Dk)).astype(mx.bfloat16)
    query = mx.random.normal((B, T, Hk, Dk)).astype(mx.bfloat16)
    q, k = normalize_gdn_qk(query, key)
    q = q.astype(mx.bfloat16)
    k = k.astype(mx.bfloat16)
    v = mx.random.normal((B, T, Hv, Dv)).astype(mx.bfloat16)
    # Linear-space decay in (0, 1], concentrated near 1 as it is at these
    # scales; beta is a sigmoid rate.  Both fp32, as _can_use_core_gated_delta
    # requires for g and as the state arithmetic uses.
    g = mx.exp(-mx.abs(mx.random.normal((B, T, Hv))) * 0.02).astype(mx.float32)
    beta = mx.sigmoid(mx.random.normal((B, T, Hv))).astype(mx.float32)
    state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    mx.eval(q, k, v, g, beta, state)
    return q, k, v, g, beta, state


def make_arm(mx, name, inputs):
    """Return a zero-arg callable that runs one arm once."""
    from mlx2.runtime.models.gated_delta import gated_delta_kernel, gated_delta_ops

    q, k, v, g, beta, state0 = inputs
    if name == "ops":
        return lambda: gated_delta_ops(q, k, v, g, beta, state0, None)
    if name == "kernel":
        return lambda: gated_delta_kernel(q, k, v, g, beta, state0, None)

    core = getattr(mx.fast, "gated_delta_update", None)
    if core is None:
        raise SystemExit("mx.fast.gated_delta_update is unavailable in this MLX build")
    chunk = {"core_seq": 1, "core_8": 8, "core_16": 16, "core_auto": -1}[name]
    return lambda: core(q, k, v, g, beta, initial_state=state0,
                        chunk_size=chunk, threshold=16)


def compare(mx, out, ref):
    """max_abs_delta on y and on the final state, plus bit-exactness."""
    (ya, sa) = out
    (yr, sr) = ref
    ya = ya.astype(mx.float32)
    yr = yr.astype(mx.float32)
    sa = sa.astype(mx.float32)
    sr = sr.astype(mx.float32)
    mx.eval(ya, sa)
    if ya.shape != yr.shape or sa.shape != sr.shape:
        return {"shape_mismatch": True, "arm_y": list(ya.shape),
                "ref_y": list(yr.shape), "arm_state": list(sa.shape),
                "ref_state": list(sr.shape)}
    dy = float(mx.max(mx.abs(ya - yr)).item())
    ds = float(mx.max(mx.abs(sa - sr)).item())
    return {"max_abs_delta_y": dy, "max_abs_delta_state": ds,
            "bit_exact": (dy == 0.0 and ds == 0.0), "shape_mismatch": False}


def time_arm(mx, fn, warmup, reps):
    for _ in range(warmup):
        y, s = fn()
        mx.eval(y, s)
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        y, s = fn()
        mx.eval(y, s)
        samples.append(time.perf_counter() - t0)
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", default="1024,4096,16384",
                   help="prefill widths T to measure")
    p.add_argument("--arms", default=",".join(a for a in ARMS if a != "ops"),
                   help="comma-separated subset of " + ",".join(ARMS))
    p.add_argument("--rounds", type=int, default=3,
                   help="interleaved timing rounds; per-arm median of round medians")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--oracle-max-t", type=int, default=4096,
                   help="use gated_delta_ops as the correctness reference only "
                        "up to this T; above it, 'kernel' is the reference")
    p.add_argument("--time-ops", action="store_true",
                   help="include the (very slow) pure-MLX oracle in timing")
    p.add_argument("--out", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--i-own-the-gpu", action="store_true")
    a = p.parse_args()

    tokens = [int(x) for x in a.tokens.split(",") if x.strip()]
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    unknown = [x for x in arms if x not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms: {unknown}; choose from {ARMS}")

    plan = {
        "schema": SCHEMA,
        "purpose": "GDN prefill implementation A/B at Flash-Next geometry",
        "geometry": GEOMETRY,
        "dtypes": {"qkv": "bfloat16", "g": "float32", "beta": "float32",
                   "state": "float32"},
        "tokens": tokens,
        "arms": arms,
        "rounds": a.rounds, "warmup": a.warmup, "reps": a.reps,
        "seed": a.seed, "oracle_max_t": a.oracle_max_t,
        "time_ops": a.time_ops,
        "exactness_bar": "bit_exact (max_abs_delta == 0.0), not a tolerance",
        "references": {
            "vs_ops": "gated_delta_ops, the pure-MLX recurrence; faithfulness",
            "vs_kernel": "gated_delta_kernel, production today; swap impact",
        },
        "oracle_max_t_note": "ops is a Python loop over T, so it is skipped above oracle_max_t",
    }
    if a.dry_run:
        print(json.dumps({"dry_run": True, "plan": plan}, indent=2))
        return
    if not a.i_own_the_gpu:
        raise SystemExit("refusing Metal without --i-own-the-gpu")

    import mlx.core as mx

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise SystemExit("this harness measures Metal; no GPU available")

    report = dict(plan)
    report["mlx_version"] = mx.__version__
    report["device"] = str(mx.default_device())
    report["host_started"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    report["results"] = []

    for T in tokens:
        inputs = build_inputs(mx, T, a.seed)
        q, k, v, g, beta, state0 = inputs
        entry = {"T": T, "arms": {}}

        # ---- correctness: compare every arm against BOTH references.
        # `ops` answers "is this faithful to the mathematical recurrence?";
        # `kernel` answers "would swapping the production path change output?".
        # They are different questions and neither implies the other: the Metal
        # kernel is itself not bit-exact against the pure-MLX ops loop.
        want_refs = (["ops"] if T <= a.oracle_max_t else []) + ["kernel"]
        run_names = list(dict.fromkeys(want_refs + arms))
        outs = {}
        for name in run_names:
            fn = make_arm(mx, name, inputs)
            y, s = fn()
            mx.eval(y, s)
            outs[name] = (y, s)
        entry["references"] = [r for r in want_refs if r in outs]
        for name in arms:
            row = entry["arms"].setdefault(name, {})
            for rname in entry["references"]:
                if name == rname:
                    row[f"vs_{rname}"] = {"reference": True, "bit_exact": True,
                                          "max_abs_delta_y": 0.0,
                                          "max_abs_delta_state": 0.0,
                                          "shape_mismatch": False}
                else:
                    row[f"vs_{rname}"] = compare(mx, outs[name], outs[rname])
        del outs
        mx.clear_cache()

        # ---- speed, interleaved rounds
        timed = [n for n in arms if n != "ops" or a.time_ops]
        fns = {n: make_arm(mx, n, inputs) for n in timed}
        per_round = {n: [] for n in timed}
        for _ in range(a.rounds):
            for n in timed:
                per_round[n].extend(time_arm(mx, fns[n], a.warmup, a.reps))
        for n in timed:
            samples = per_round[n]
            med = statistics.median(samples)
            entry["arms"].setdefault(n, {}).update({
                "median_ms": med * 1e3,
                "min_ms": min(samples) * 1e3,
                "max_ms": max(samples) * 1e3,
                "n_samples": len(samples),
                "tokens_per_s": (T / med) if med > 0 else None,
            })
        base = entry["arms"].get("kernel", {}).get("median_ms")
        if base:
            for n in timed:
                m = entry["arms"][n].get("median_ms")
                if m:
                    entry["arms"][n]["speedup_vs_kernel"] = base / m
        entry["peak_mem_gb"] = mx.get_peak_memory() / 1e9
        mx.reset_peak_memory()
        mx.clear_cache()

        report["results"].append(entry)
        print(json.dumps({"T": T, "references": entry["references"],
                          "arms": {n: {k2: v2 for k2, v2 in d.items()
                                       if k2 != "all_samples"}
                                   for n, d in entry["arms"].items()}},
                         indent=2), flush=True)
        del inputs, q, k, v, g, beta, state0, fns

    report["host_finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {a.out}", flush=True)


if __name__ == "__main__":
    main()

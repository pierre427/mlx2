#!/usr/bin/env python3
"""A/B the fused GroupRMSNorm against production, on speed AND correctness.

Three arms, all at the locked Flash-Next hyper-connection geometry (stream width
10240, group_size 2560, eps 1e-6, bf16):

  eager       GroupRMSNorm.__call__ as production runs it at any prefill width
              (the fast-path gate declines above width 8), i.e. six separate ops
              each materialising an fp32 copy of a 335 MB bf16 tensor.
  fast_rms    the same module with _declared_width(1) forcing mx.fast.rms_norm --
              the shipped fast path, which fuses only the reduce.
  fused       src/mlx2/runtime/models/qwen4_fused_group_norm.py, one Metal
              dispatch: bf16 in, fp32 accumulate in registers, bf16 out.

`eager` is the correctness reference. Every arm reports max_abs_delta, the delta
in bf16 ULPs at unit magnitude, the fraction of elements differing, and
bit_exactness. Speed is reported as median ms over interleaved rounds plus the
implied HBM traffic from the roofline in the experiment doc, so a speedup can be
read against the bytes it moved rather than in isolation.

The point of pairing them is that a fusion that is 10x faster and 1 ULP off is a
different decision from one that is 10x faster and exact: Fidelity.EXACT is the
routing default, and rm15 measured 1 bf16 ULP at GDN layer 0 amplifying to
max|delta logit| 7.55 at 16K. So the correctness column is not a footnote.

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

SCHEMA = "mlx2.fused-group-norm-ab.v1"
STREAM_WIDTH = 10240
GROUP_SIZE = 2560
EPS = 1e-06
ARMS = ("eager", "fast_rms", "fused")


def traffic_mb(arm, T):
    """Modelled HBM traffic per call at width T, in MB."""
    bf16 = T * STREAM_WIDTH * 2 / 1e6
    fp32 = T * STREAM_WIDTH * 4 / 1e6
    if arm == "fused":
        return bf16 + bf16                 # read bf16, write bf16
    if arm == "fast_rms":
        # astype(fp32) + fused reduce (read/write fp32) + *weight(fp32) + cast
        return (bf16 + fp32) + (fp32 + fp32) + (fp32 + fp32) + (fp32 + bf16)
    # eager: astype, x*x, mean, x*rsqrt, *weight, cast
    return ((bf16 + fp32) + (fp32 + fp32) + fp32 + (fp32 + fp32)
            + (fp32 + fp32) + (fp32 + bf16))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", default="1024,4096,16384")
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--rows", type=int, default=1,
                   help="batch/row count ahead of the token axis")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--scale", type=float, default=1.7)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--out", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--i-own-the-gpu", action="store_true")
    a = p.parse_args()

    tokens = [int(x) for x in a.tokens.split(",") if x.strip()]
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    unknown = [x for x in arms if x not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; choose from {ARMS}")

    plan = {
        "schema": SCHEMA,
        "geometry": {"stream_width": STREAM_WIDTH, "group_size": GROUP_SIZE,
                     "eps": EPS, "rows": a.rows, "dtype": "bfloat16"},
        "tokens": tokens, "arms": arms,
        "rounds": a.rounds, "warmup": a.warmup, "reps": a.reps,
        "scale": a.scale, "seed": a.seed,
        "reference_arm": "eager",
        "exactness_bar": "bit_exact (max_abs_delta == 0.0); Fidelity.EXACT is the routing default",
    }
    if a.dry_run:
        print(json.dumps({"dry_run": True, "plan": plan}, indent=2))
        return
    if not a.i_own_the_gpu:
        raise SystemExit("refusing Metal without --i-own-the-gpu")

    import mlx.core as mx

    from mlx2.runtime.models import qwen4_fused_group_norm as fgn
    from mlx2.runtime.models.qwen4_exp import GroupRMSNorm, _declared_width

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise SystemExit("this harness measures Metal; no GPU available")

    # The kernel is default-off and admission fails closed without the lever,
    # so the harness has to raise it explicitly rather than assume it.
    fgn.set_fused_group_norm_enabled(True)
    report_lever = fgn.fused_group_norm_enabled()

    # The real module, so `fast_rms` exercises the shipped gate rather than a
    # hand-written stand-in.
    grn = GroupRMSNorm(STREAM_WIDTH, GROUP_SIZE, EPS)

    report = dict(plan)
    report["mlx_version"] = mx.__version__
    report["device"] = str(mx.default_device())
    report["fused_lever_enabled"] = report_lever
    report["host_started"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    report["probe_bit_exact"] = None
    report["results"] = []

    for T in tokens:
        mx.random.seed(a.seed)
        x = (mx.random.normal((a.rows, T, STREAM_WIDTH)) * a.scale).astype(mx.bfloat16)
        w = (1.0 + 0.05 * mx.random.normal((STREAM_WIDTH,))).astype(mx.bfloat16)
        grn.weight = w
        mx.eval(x, w, grn.parameters())
        cand = (a.rows * T,)

        def run_eager(_x=x, _w=w):
            return fgn.eager_group_norm(_x, _w)

        def run_fast(_grn=grn, _x=x):
            with _declared_width(1):
                return _grn(_x)

        def run_fused(_x=x, _w=w, _cand=cand):
            return fgn.fused_group_norm(_x, _w, eps=EPS, group_size=GROUP_SIZE,
                                        candidate_rows=_cand)

        fns = {"eager": run_eager, "fast_rms": run_fast, "fused": run_fused}

        entry = {"T": T, "arms": {}}
        outs = {}
        for name in arms:
            y = fns[name]()
            mx.eval(y)
            outs[name] = y
        ref = outs["eager"].astype(mx.float32)

        per_round = {n: [] for n in arms}
        for _ in range(a.rounds):
            for n in arms:
                fn = fns[n]
                for _ in range(a.warmup):
                    mx.eval(fn())
                for _ in range(a.reps):
                    t0 = time.perf_counter()
                    mx.eval(fn())
                    per_round[n].append(time.perf_counter() - t0)

        for name in arms:
            samples = per_round[name]
            med = statistics.median(samples)
            v = outs[name].astype(mx.float32)
            mx.eval(v)
            d = mx.abs(v - ref)
            mx.eval(d)
            maxd = float(mx.max(d).item())
            entry["arms"][name] = {
                "median_ms": med * 1e3,
                "min_ms": min(samples) * 1e3,
                "max_ms": max(samples) * 1e3,
                "n_samples": len(samples),
                "modelled_traffic_mb": round(traffic_mb(name, T), 1),
                "implied_gbps": round(traffic_mb(name, T) / 1e3 / (med), 1) if med else None,
                "max_abs_delta": maxd,
                "bf16_ulps": maxd / 2**-8,
                "pct_elements_differing": 100.0 * float(
                    mx.mean((d > 0).astype(mx.float32)).item()),
                "bit_exact": maxd == 0.0,
            }
        base = entry["arms"]["eager"]["median_ms"]
        for name in arms:
            entry["arms"][name]["speedup_vs_eager"] = (
                base / entry["arms"][name]["median_ms"])

        # Cross-check: fused and fast_rms showed the SAME delta from eager.  If
        # they are bit-identical to each other, the difference is shared and
        # inherent (rsqrt/reduce implementation), not a property of this
        # kernel's reduction order -- which decides whether an exactly-matching
        # fused kernel is even reachable.
        if "fused" in outs and "fast_rms" in outs:
            fv = outs["fused"].astype(mx.float32)
            rv = outs["fast_rms"].astype(mx.float32)
            mx.eval(fv, rv)
            pd = mx.abs(fv - rv)
            mx.eval(pd)
            pmax = float(mx.max(pd).item())
            entry["fused_vs_fast_rms"] = {
                "max_abs_delta": pmax,
                "bf16_ulps": pmax / 2**-8,
                "bit_exact": pmax == 0.0,
                "pct_elements_differing": 100.0 * float(
                    mx.mean((pd > 0).astype(mx.float32)).item()),
            }
        entry["peak_mem_gb"] = mx.get_peak_memory() / 1e9
        mx.reset_peak_memory()
        report["results"].append(entry)
        print(json.dumps(entry, indent=2), flush=True)
        if report["probe_bit_exact"] is None:
            report["probe_bit_exact"] = entry["arms"]["fused"]["bit_exact"]
        mx.clear_cache()

    report["host_finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {a.out}", flush=True)


if __name__ == "__main__":
    main()

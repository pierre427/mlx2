"""GLM-5.3 D3 microbench: the KDA low-rank f_b/g_b launch vs the decode_v2 prologue.

    bench.sh <commit> <label> glm53_kda_lowrank_bench.py [--reps N] [--layer-set all|one]

Every arm is a dependent chain over the 34 KDA layers of one decode token
(``--layer-set all``: each layer's own checkpoint ``ssm_f_b`` / ``ssm_g_b``
and its own 4 MiB recurrent state, so weights and state stream as in a real
step; ``one``: blk.0's matrices for every layer, SLC-warm). Layer i's f_a /
g_a are slices of layer i-1's output (views, no copy launch), so each launch
waits on the one before, as in the model's per-layer chain. Arms, interleaved
per repetition (A B C D A B ...):

* ``ref``   second ``kblock_multi`` (f_b, g_b) + decode_v1 body: today's step
* ``v2``    decode_v2 (prologue inside the body): one launch per layer
* ``multi`` the second ``kblock_multi`` alone, chained (the premise: what that
            launch costs in a dependent chain)
* ``v1``    the decode_v1 body alone with precomputed a / gate, chained
            (``v2 - v1`` approximates the prologue's added cost; a synthetic
            chain delta, since v1's operands and dependencies differ)

Per repetition the chain is built, evaluated and timed wall-clock; the
per-token sync overhead is common to all arms and cancels in the deltas.
Graph construction (host) and evaluation of the prebuilt graph (submit + GPU)
are also reported apart: the model step replays compiled regions, so only
the eval column speaks to the GPU-side launch premise.
Every layer's output and both states of the ref and v2 chains are compared
bitwise (outside the timing).
Research only; GPU work runs only under the bench.sh lease.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glm53_ordinary_profile import CHECKPOINT, MODEL

KDA_LAYERS = [i for i in range(45) if i % 4 != 3]
GGUF = MODEL + "/GLM-5.3-Flash-REAP50-Q3_K_M.gguf"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-revision", default="unrecorded")
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--layer-set", choices=("all", "one"), default="all")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("refusing to reuse a nonempty evidence directory")
    args.out.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx
    import numpy as np

    from mlx2.runtime.gguf import GGUFFile
    from mlx2.runtime.gguf.kblock_metal import kblock_multi
    from mlx2.runtime.gguf.packed_metal import PackedMatrix
    from mlx2.runtime.models import glm53_kda_fused as kf

    mx.set_default_device(mx.gpu)
    file = GGUFFile(GGUF)
    layers = [0] * len(KDA_LAYERS) if args.layer_set == "one" else KDA_LAYERS
    cache = {}
    mats = []
    for layer in layers:
        if layer not in cache:
            cache[layer] = [PackedMatrix.upload(file, f"blk.{layer}.ssm_{p}_b.weight") for p in "fg"]
        mats.append(cache[layer])
    n = len(mats)
    rng = np.random.default_rng(570)
    f = lambda shape, s=1.0, loc=0.0: mx.array((rng.normal(loc, 1, shape) * s).astype(np.float32))
    inner, channels = kf.INNER, kf.CHANNELS
    per = [{"mixed": f((1, 1, channels)), "conv": f((1, 3, channels)),
            "filters": f((channels, 4, 1), 0.4), "dt_bias": f((inner,), 0.5, -2.0),
            "ssm_a": mx.array(rng.uniform(0.5, 8, kf.HEADS).astype(np.float32)),
            "beta": f((1, 1, kf.HEADS), 2.0), "state": f((1, kf.HEADS, kf.DIM, kf.DIM), 0.3),
            "norm": f((kf.DIM,), 0.2, 1.0), "a": f((1, 1, inner), 2.0), "gate": f((1, 1, inner), 2.0)}
           for _ in range(n)]
    f_a0, g_a0 = f((1, 1, 128)), f((1, 1, 128))
    mx.eval(per, f_a0, g_a0)

    def chain(arm, keep=None):
        f_a, g_a, outs = f_a0, g_a0, []
        for i in range(n):
            p, (f_b, g_b) = per[i], mats[i]
            if arm == "multi":
                y = kblock_multi([f_a.reshape(1, 128), g_a.reshape(1, 128)], [f_b, g_b])
                f_a, g_a = y[:, :128], y[:, 8192:8192 + 128]
                outs = [y]
                continue
            if arm == "v1":
                y, conv, state = kf.fused_kda_decode(
                    p["mixed"], p["conv"], p["filters"], p["a"] if i == 0 else outs[0], p["dt_bias"],
                    p["ssm_a"], p["beta"], p["gate"], p["state"], p["norm"], rms_eps=1e-5)
            elif arm == "ref":
                lr = kblock_multi([f_a.reshape(1, 128), g_a.reshape(1, 128)], [f_b, g_b])
                y, conv, state = kf.fused_kda_decode(
                    p["mixed"], p["conv"], p["filters"], lr[:, :inner].reshape(1, 1, inner), p["dt_bias"],
                    p["ssm_a"], p["beta"], lr[:, inner:].reshape(1, 1, inner), p["state"], p["norm"],
                    rms_eps=1e-5)
            else:
                y, conv, state = kf.fused_kda_decode_lowrank(
                    p["mixed"], p["conv"], p["filters"], f_a, g_a, f_b, g_b, p["dt_bias"], p["ssm_a"],
                    p["beta"], p["state"], p["norm"], rms_eps=1e-5)
            outs = [y, conv, state]
            if keep is not None:
                keep.extend(outs)
            f_a, g_a = y[..., :128], y[..., 128:256]
        return outs

    arms = ("ref", "v2", "multi", "v1")
    times = {a: [] for a in arms}
    builds = {a: [] for a in arms}
    evals = {a: [] for a in arms}
    for rep in range(args.warmup + args.reps):
        for k in range(len(arms)):
            arm = arms[(rep + k) % len(arms)]  # rotate the order every repetition
            t0 = time.perf_counter()
            out = chain(arm)
            t1 = time.perf_counter()
            mx.eval(out)
            t2 = time.perf_counter()
            if rep >= args.warmup:
                times[arm].append(t2 - t0)
                builds[arm].append(t1 - t0)  # host graph construction only
                evals[arm].append(t2 - t1)   # submit + GPU execution of a prebuilt graph
    ref, v2 = [], []
    chain("ref", ref)
    chain("v2", v2)  # every layer's output and both states, outside the timing
    mx.eval(ref, v2)
    bitwise = all(np.array_equal(np.array(a).view(np.uint32), np.array(b).view(np.uint32))
                  for a, b in zip(ref, v2))
    med = {a: statistics.median(v) * 1e3 for a, v in times.items()}
    q = {a: [round(x * 1e3, 4) for x in statistics.quantiles(v, n=10)] for a, v in times.items()}
    result = {
        "schema": "mlx2.glm53-kda-lowrank-bench.v1", "checkpoint": CHECKPOINT,
        "source_revision": args.source_revision, "host": platform.node(),
        "mlx": mx.__version__, "layer_set": args.layer_set, "layers": n, "reps": args.reps,
        "median_ms_per_token": med, "deciles_ms_per_token": q,
        "median_build_ms_per_token": {a: statistics.median(v) * 1e3 for a, v in builds.items()},
        "median_eval_ms_per_token": {a: statistics.median(v) * 1e3 for a, v in evals.items()},
        "eval_gain_ref_minus_v2_us_per_layer": (statistics.median(evals["ref"])
                                                - statistics.median(evals["v2"])) / n * 1e6,
        "eval_ref_minus_v1_us_per_layer": (statistics.median(evals["ref"])
                                           - statistics.median(evals["v1"])) / n * 1e6,
        "per_layer_us": {a: med[a] / n * 1e3 for a in arms},
        "gain_ref_minus_v2_ms_per_token": med["ref"] - med["v2"],
        "gain_ref_minus_v2_us_per_layer": (med["ref"] - med["v2"]) / n * 1e3,
        # Synthetic chain delta, not an isolated prologue time: the v1 arm's
        # gate operands and dependencies differ from v2's.
        "synthetic_v2_minus_v1_us_per_layer": (med["v2"] - med["v1"]) / n * 1e3,
        "multi_chain_us_per_launch": med["multi"] / n * 1e3,
        "ref_minus_v1_us_per_layer": (med["ref"] - med["v1"]) / n * 1e3,
        "ref_v2_every_layer_bitwise": bitwise,
        "note": "wall-clock per evaluated chain incl. one host sync; deltas cancel the sync",
    }
    (args.out / "bench.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "deciles_ms_per_token"}, indent=1), flush=True)
    if not bitwise:
        raise SystemExit("ref and v2 end states differ")


if __name__ == "__main__":
    main()

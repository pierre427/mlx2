#!/usr/bin/env python3
"""Where does the ~27% "residual / hyper-connection glue" in prefill go?

The corrected component profile (see
docs/experiments/PREFILL-GDN-COMPONENT-PROFILE-2026-09-22.md) puts 23-28% of a
Flash-Next prefill in the bucket that is neither the GDN/QSA branch nor the MoE,
and that share is roughly FLAT in T -- unlike every other component.  For what an
architecture diagram calls a residual connection, that is a lot, and it had never
been measured.

Flash-Next runs ``hc_count=4`` residual streams, so the stream a DecoderLayer
passes around is ``4 * 2560 = 10240`` wide, and every layer runs TWO
``GatedResidual`` calls over it plus TWO ``_apply_inject`` calls.  Each
``GatedResidual`` does a GroupRMSNorm over the full 10240 width, a
10240 -> 320 -> 10240 low-rank mix, a sigmoid, a reshape, a
``mean(weights * streams, axis=-2)`` and a 10240 -> 4 inject gate.  At T=16384
the 10240-wide intermediates are ~335 MB each, so the suspicion is that this
bucket is memory-traffic bound on elementwise ops rather than FLOP bound.

There is an optimization for exactly that already in the tree and switched off:
``MLX_QWEN4_COMPILE_GLUE`` (``qwen3_next._COMPILE_GLUE``, default False) enables
three ``mx.compile``d spans -- ``hyper_gate`` (``silu(down/hc_count)``),
``hyper_mix`` (``mean(weights*streams, axis=-2)``) and ``inject_apply``
(``residual + branch[...,None,:]*inject[...,None]``).  Each fusion removes one
materialised 10240-wide intermediate.  The parity matrix lists compiled glue as
"Source experiment excluded / Off", with the note that unified exact brackets
were neutral-or-slower -- but that evidence is about *decode*, and this bucket
is a *prefill* cost.  So it is worth measuring at prefill widths directly.

This harness A/Bs glue off against glue on in ONE process.  That is safe because
``compile_glue_enabled()`` reads the module global at call time, so setting
``qwen3_next._COMPILE_GLUE`` takes effect immediately with no reload; the process
still starts from the adapter's production environment profile so every other
gate matches serving.

Reported per arm and per T:
  * uninstrumented whole-``GatedResidual`` time (the honest per-call cost);
  * a seam-by-seam breakdown -- norm, down, gate law, up, sigmoid, mix, inject
    gate -- and ``_apply_inject`` timed separately;
  * bit-exactness of ``mixed``, ``inject`` and the ``_apply_inject`` output
    between the two arms, because a fusion that changes reduction order is not a
    free win in a repo whose routing default is ``Fidelity.EXACT``;
  * ``_GLUE_STATS`` deltas (builds/calls/fallbacks/skips), so engagement is
    OBSERVED rather than inferred from the flag.  A span silently skips whenever
    any argument is not bfloat16, and silently demotes itself for the life of the
    process if it raises; both would otherwise look like "glue is on".

Scaled to a whole prefill as 48 layers x 2 GatedResidual + 48 x 2 _apply_inject.

Metal: run under the GPU queue only.  --dry-run prints the plan and exits.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# Reuse the sibling harness's environment pinning and config loading so the
# production gate profile lives in exactly one place.
from profile_prefill_components import (
    dtype_audit,
    load_args,
    pin_production_environment,
    timeit,
)

SCHEMA = "mlx2.hyper-connection-profile.v2"
N_LAYERS = 48
CALLS_PER_LAYER = 2

# (name, compiled-glue lever, forced GroupRMSNorm fast-path width or None).
# `glue_off` is production today: MLX_QWEN4_COMPILE_GLUE defaults False and
# MLX_QWEN4_RMSNORM_FAST_MAX_WIDTH=8, so at any prefill width the eager RMSNorm
# branch runs and no glue span is built.
ARMS = (
    ("glue_off", False, None),
    ("glue_on", True, None),
    ("rmsnorm_fast", False, 1),
    ("glue_on+rmsnorm_fast", True, 1),
)
REFERENCE_ARM = "glue_off"


def build_gated_residual(args, quant, dtype):
    """One real GatedResidual, in the checkpoint dtype, quantized as production.

    The dtype cast is load-bearing for the same reason it is in the component
    profiler: MLX initialises parameters as float32 and this model never casts
    them, so an uncast synthetic module measures roughly 2.6x slow.
    """
    import mlx.core as mx
    from mlx.utils import tree_map

    from mlx2.runtime.models import qwen4_exp

    gr = qwen4_exp.GatedResidual(args)
    gr.update(tree_map(lambda p: p.astype(dtype), gr.parameters()))
    qwen4_exp.nn.quantize(
        gr,
        group_size=quant.get("group_size", 64),
        bits=quant.get("bits", 4),
        mode=quant.get("mode", "affine"),
    )
    gr.eval()
    mx.eval(gr.parameters())
    return gr


def set_glue(qwen3_next, enabled):
    """Flip the compiled-glue lever live, and clear the span cache.

    The cache must be cleared: a span that raised is stored as None ("demoted
    for the life of the process"), and a compiled span is keyed by shape-free
    identity, so stale entries from the other arm would contaminate this one.
    """
    qwen3_next._COMPILE_GLUE = bool(enabled)
    qwen3_next._GLUE_COMPILE_CACHE.clear()
    for k in qwen3_next._GLUE_STATS:
        qwen3_next._GLUE_STATS[k] = 0
    return qwen3_next.compile_glue_enabled()


def seams_of(gr, mx, x, hc_count):
    """Time each seam of GatedResidual.__call__ and return the outputs.

    Mirrors the real __call__ body op for op, with mx.eval at each boundary.
    The eager arithmetic is reproduced rather than delegated so that the
    glue-off arm and the glue-on arm can be compared seam by seam.
    """
    from mlx import nn

    from mlx2.runtime.models import qwen3_next

    s = {}

    def tick(name, t0):
        s[name] = s.get(name, 0.0) + (time.perf_counter() - t0)

    t = time.perf_counter()
    normed = gr.hc_norm(x)
    mx.eval(normed)
    tick("hc_norm", t)

    t = time.perf_counter()
    gate_input = gr.input_mix_weight_down(normed)
    mx.eval(gate_input)
    tick("mix_down_10240_to_320", t)

    t = time.perf_counter()
    weights = None
    if qwen3_next.compile_glue_enabled():
        weights = qwen3_next._run_glue(
            ("hyper_gate", hc_count),
            qwen3_next._build_hyper_gate(hc_count),
            gate_input,
        )
    if weights is None:
        weights = nn.silu(gate_input / hc_count)
    mx.eval(weights)
    tick("gate_law_silu", t)

    t = time.perf_counter()
    up = gr.input_mix_weight_up(weights)
    mx.eval(up)
    tick("mix_up_320_to_10240", t)

    t = time.perf_counter()
    weights = mx.sigmoid(up)
    mx.eval(weights)
    tick("sigmoid", t)

    t = time.perf_counter()
    w4 = weights.reshape(*weights.shape[:-1], hc_count, gr.hidden_size)
    st4 = normed.reshape(*normed.shape[:-1], hc_count, gr.hidden_size)
    mixed = None
    if qwen3_next.compile_glue_enabled():
        mixed = qwen3_next._run_glue(
            ("hyper_mix",), qwen3_next._build_hyper_mix, w4, st4
        )
    if mixed is None:
        mixed = mx.mean(w4 * st4, axis=-2)
    mx.eval(mixed)
    tick("stream_mix_mean", t)

    t = time.perf_counter()
    inject = 2 * mx.sigmoid(gr.block_inject_weight(normed) / hc_count)
    mx.eval(inject)
    tick("inject_gate_10240_to_4", t)

    return (mixed, x, inject), s


def apply_inject_timed(mx, residual, branch, inject, reps, warmup):
    from mlx2.runtime.models.qwen4_exp import _apply_inject

    samples = timeit(lambda: _apply_inject(residual, branch, inject),
                     mx, warmup, reps)
    out = _apply_inject(residual, branch, inject)
    mx.eval(out)
    return statistics.median(samples) * 1e3, out


@contextmanager
def forced_rmsnorm_width(width):
    """Pin the width GroupRMSNorm's fast-path gate sees.

    ``_use_fast`` accepts a query width <= ``_RMSNORM_FAST_MAX_WIDTH``, which
    production sets to 8, so at any prefill width the eager branch runs:
    ``xf = x.astype(float32)`` then ``xf * xf``, ``mean``, ``rsqrt``, ``xf *``,
    ``out * weight`` and a cast back -- each materialising an fp32 copy of a
    tensor that is 335 MB in bf16 at T=16384.  ``_declared_width`` is the
    module's own hook for pinning that gate, so using it here exercises the
    shipped ``mx.fast.rms_norm`` path rather than a new kernel.
    """
    from mlx2.runtime.models.qwen4_exp import _declared_width

    if width is None:
        yield
        return
    with _declared_width(width):
        yield


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",
                   default=str(Path.home() / "mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP"),
                   help="artifact to read config.json from (metadata only)")
    p.add_argument("--tokens", default="1024,4096,16384")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--out", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--i-own-the-gpu", action="store_true")
    a = p.parse_args()

    tokens = [int(x) for x in a.tokens.split(",") if x.strip()]
    env_profile = pin_production_environment(a.model)
    args, quant, config = load_args(a.model)
    text_config = config.get("text_config", config)
    hc_count = text_config["hc_count"]

    plan = {
        "schema": SCHEMA,
        "model_config_only": a.model,
        "weights_loaded": False,
        "hc_count": hc_count,
        "hidden_size": text_config["hidden_size"],
        "stream_width": hc_count * text_config["hidden_size"],
        "hc_lowrank": text_config.get("hc_lowrank"),
        "scaling": {"layers": N_LAYERS, "gated_residual_calls_per_layer": CALLS_PER_LAYER,
                    "apply_inject_calls_per_layer": CALLS_PER_LAYER},
        "arms": {name: {"compile_glue": g, "rmsnorm_fast_width": w}
                 for name, g, w in ARMS},
        "reference_arm": REFERENCE_ARM,
        "quantization": {"group_size": quant.get("group_size"), "bits": quant.get("bits"),
                         "mode": quant.get("mode", "affine")},
        "tokens": tokens, "rounds": a.rounds, "warmup": a.warmup, "reps": a.reps,
        "production_environment": env_profile,
        "exactness_bar": "bit_exact (max_abs_delta == 0.0); Fidelity.EXACT is the routing default",
    }
    if a.dry_run:
        print(json.dumps({"dry_run": True, "plan": plan}, indent=2))
        return
    if not a.i_own_the_gpu:
        raise SystemExit("refusing Metal without --i-own-the-gpu")

    import mlx.core as mx

    from mlx2.runtime.models import qwen3_next

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise SystemExit("this harness measures Metal; no GPU available")

    dtype = getattr(mx, str(text_config.get("dtype") or "bfloat16"))
    report = dict(plan)
    report["mlx_version"] = mx.__version__
    report["device"] = str(mx.default_device())
    report["param_dtype"] = str(dtype)
    report["host_started"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    gr = build_gated_residual(args, quant, dtype)
    by_dtype, total_mb = dtype_audit(mx, gr)
    report["gated_residual_param_mb"] = {"by_dtype": by_dtype, "total_mb": total_mb}
    print(f"GatedResidual params: {total_mb} MB {by_dtype}", flush=True)

    # _apply_inject is a module-level function and needs no weights; it is timed
    # against the same residual/branch/inject shapes the real layer produces.
    report["results"] = []

    for T in tokens:
        width = hc_count * args.hidden_size
        x = mx.random.normal((1, T, width)).astype(dtype)
        branch = mx.random.normal((1, T, args.hidden_size)).astype(dtype)
        mx.eval(x, branch)

        entry = {"T": T, "arms": {}}
        outputs = {}
        for arm, glue_enabled, fast_width in ARMS:
            live = set_glue(qwen3_next, glue_enabled)
            assert live == glue_enabled, f"glue lever did not take: {live}"
            best = None
            for _ in range(a.rounds):
                with forced_rmsnorm_width(fast_width):
                  # whole-call, uninstrumented.  Bind x as a default argument so
                  # the closure does not capture a loop variable (B023).
                  whole = timeit(lambda _x=x: gr(_x), mx, a.warmup, a.reps)
                  # seams
                  acc = {}
                  for _ in range(a.reps):
                      (mixed, _residual, inject), s = seams_of(gr, mx, x, hc_count)
                      for k, v in s.items():
                          acc.setdefault(k, []).append(v)
                  inj_ms, inj_out = apply_inject_timed(
                      mx, x, branch, inject, a.reps, a.warmup)
                  mx.eval(mixed, inject, inj_out)
                  row = {
                      "gr_whole_ms": statistics.median(whole) * 1e3,
                      "gr_whole_min_ms": min(whole) * 1e3,
                      "gr_whole_max_ms": max(whole) * 1e3,
                      # acc[k] holds one entry per rep, so the median is already
                      # a per-call figure.
                      "seams_ms": {k: statistics.median(v) * 1e3
                                   for k, v in acc.items()},
                      "apply_inject_ms": inj_ms,
                      "seam_pass_total_ms": (sum(statistics.median(v) * 1e3
                                                 for v in acc.values()) + inj_ms),
                      "glue_stats": dict(qwen3_next._GLUE_STATS),
                      "glue_enabled": glue_enabled,
                      "rmsnorm_fast_width": fast_width,
                      "n_samples": len(whole),
                  }
                  if best is None or row["gr_whole_ms"] < best["gr_whole_ms"]:
                      best = row
                      outputs[arm] = (mixed.astype(mx.float32),
                                      inject.astype(mx.float32),
                                      inj_out.astype(mx.float32))
                  mx.clear_cache()
            best["peak_mem_gb"] = mx.get_peak_memory() / 1e9
            mx.reset_peak_memory()
            entry["arms"][arm] = best
            print(json.dumps({"T": T, "arm": arm,
                              **{k: v for k, v in best.items() if k != "seams_ms"},
                              "seams_ms": {k: round(v, 3)
                                           for k, v in best["seams_ms"].items()}},
                             indent=2), flush=True)

        # exactness of every arm against the production reference arm
        ref_m, ref_i, ref_a = outputs[REFERENCE_ARM]
        entry["exactness_vs_" + REFERENCE_ARM] = {}
        for arm in entry["arms"]:
            if arm == REFERENCE_ARM:
                entry["exactness_vs_" + REFERENCE_ARM][arm] = {
                    "reference": True, "bit_exact": True}
                continue
            m1, i1, a1 = outputs[arm]
            d = {
                "mixed_max_abs_delta": float(mx.max(mx.abs(m1 - ref_m)).item()),
                "inject_max_abs_delta": float(mx.max(mx.abs(i1 - ref_i)).item()),
                "apply_inject_max_abs_delta": float(mx.max(mx.abs(a1 - ref_a)).item()),
            }
            d["bit_exact"] = all(v == 0.0 for k, v in d.items()
                                 if k.endswith("max_abs_delta"))
            entry["exactness_vs_" + REFERENCE_ARM][arm] = d
        ref = entry["arms"][REFERENCE_ARM]
        entry["speedup_vs_" + REFERENCE_ARM] = {
            arm: {
                "gr_whole": ref["gr_whole_ms"] / d["gr_whole_ms"],
                "apply_inject": ref["apply_inject_ms"] / d["apply_inject_ms"],
                "seam_pass_total": ref["seam_pass_total_ms"] / d["seam_pass_total_ms"],
            }
            for arm, d in entry["arms"].items()
        }
        # scaled to a whole prefill
        entry["prefill_estimate_ms"] = {
            arm: round(N_LAYERS * CALLS_PER_LAYER * (
                d["gr_whole_ms"] + d["apply_inject_ms"]), 1)
            for arm, d in entry["arms"].items()
        }
        entry["glue_engaged"] = {
            arm: entry["arms"][arm]["glue_stats"] for arm in entry["arms"]
        }
        report["results"].append(entry)
        print(json.dumps({"T": T,
                          "speedup": entry["speedup_vs_" + REFERENCE_ARM],
                          "exactness": entry["exactness_vs_" + REFERENCE_ARM],
                          "prefill_estimate_ms": entry["prefill_estimate_ms"],
                          "glue_engaged": entry["glue_engaged"]}, indent=2),
              flush=True)
        # x and branch are rebound on the next iteration; do not `del` them, or
        # the name looks undefined to static analysis on the following pass.
        outputs.clear()
        mx.clear_cache()

    report["host_finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {a.out}", flush=True)


if __name__ == "__main__":
    main()

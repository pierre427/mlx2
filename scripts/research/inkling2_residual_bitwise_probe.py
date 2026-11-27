"""GPU probe: is the MLX-order residual kernel bitwise equal to the composed
reference, stage by stage? (``fuse_kernels.RESIDUAL_MLX_ORDER``)

Original mlx2 script (2026-11-26). No model is loaded: random inputs at the
checkpoint's width (D 2304), so results name no checkpoint; they name the
mlx2 commit, the MLX version and the GPU architecture.

For every case (law ``residual``: N states in ``--states``, nq 1 and 3;
law ``message``: the decoder's loop messages, nq 4 query rows over a bank of
``--bank`` states; block dtypes BF16 and FP32, the partial always FP32;
``--seeds`` draws each) it runs, on the GPU:

- (a) ``rms``: keys from the raw states;
- (b) ``scores``: scores from the reference keys;
- (c) ``softmax``: weights from the reference scores;
- (d) ``mix``: the output from the reference weights and keys;
- ``e2e``: the whole mix from the raw states (``softmax_mix``),

each against the composed MLX reference of that stage (the ops of
``language._attention_residual`` and of the decoder's message fallback),
for these arms:

- ``current``: the shipped kernel's order (``_residual_source``), restated
  stage by stage below (e2e: ``softmax_mix`` with the switch off);
- ``mlx_source`` / ``mlx_fma`` / ``mlx_nocontract``: the candidate's stage
  text (``fuse_kernels.mo_phase``) with ``RESIDUAL_MLX_MAC`` set to that
  value (e2e: ``softmax_mix`` with ``RESIDUAL_MLX_ORDER`` on).

Each cell records bitwise yes/no, mismatching elements and max |diff|, plus
the candidate's route plan. ``MLX_ENABLE_TF32`` is pinned to 0 before MLX
loads (mlx2's serving value); a different explicit value is recorded and the
GEMM routes then decline.

Usage (one bounded GPU unit, a few seconds; the caller owns the GPU):
  MLX_ENABLE_TF32=0 PYTHONPATH=$PWD/src \\
    python scripts/research/inkling2_residual_bitwise_probe.py \\
    output/inkling2-residual-bitwise-<stamp>/probe.json
  [--states 2,4,8,12,16] [--bank 1,2,4,8,12,16] [--seeds 2] [--width 2304]
The output file must not exist; it is written atomically.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import types
from pathlib import Path

os.environ.setdefault("MLX_ENABLE_TF32", "0")

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("current", "mlx_source", "mlx_fma", "mlx_nocontract")
MACS = {"mlx_source": "source", "mlx_fma": "fma", "mlx_nocontract": "nocontract"}
STAGES = ("rms", "scores", "softmax", "mix", "e2e")

# ----- the shipped kernel's order, one stage per kernel (TG 256) ----------
# Restated from fuse_kernels._residual_source (SAFE_MATH off): strided d
# loop, simd_sum, the eight simdgroup partials in order, sequential softmax.
_CUR_RMS = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    constexpr int NS = 8;
    threadgroup float red[N * NS];
    threadgroup float inv_s[N];
    float acc[N];
    for (int i = 0; i < N; ++i) acc[i] = 0.0f;
    for (int d = tid; d < D; d += 256) {
        for (int i = 0; i < N; ++i) { float v = xs[i * D + d]; acc[i] += v * v; }
    }
    for (int i = 0; i < N; ++i) {
        float s = simd_sum(acc[i]);
        if (lane == 0) red[i * NS + sg] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < N) {
        float t = 0.0f;
        for (int g = 0; g < NS; ++g) t += red[tid * NS + g];
        inv_s[tid] = metal::precise::rsqrt(t / float(D) + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = tid; e < N * D; e += 256) out[e] = xs[e] * inv_s[e / D];
"""
_CUR_SCORES = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint qi = threadgroup_position_in_grid.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    constexpr int NS = 8;
    threadgroup float red[N * NS];
    float acc[N];
    for (int i = 0; i < N; ++i) acc[i] = 0.0f;
    for (int d = tid; d < D; d += 256) {
        const float qd = qs[qi * D + d];
        for (int i = 0; i < N; ++i) acc[i] += keys[i * D + d] * qd;
    }
    for (int i = 0; i < N; ++i) {
        float s = simd_sum(acc[i]);
        if (lane == 0) red[i * NS + sg] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        for (int i = 0; i < N; ++i) {
            float t = 0.0f;
            for (int g = 0; g < NS; ++g) t += red[i * NS + g];
            out[qi * N + i] = t * scale[0];
        }
    }
"""
_CUR_SOFTMAX = r"""
    const uint qi = thread_position_in_grid.x;
    if (qi >= NQ) return;
    float sc[N];
    float m = -INFINITY;
    for (int i = 0; i < N; ++i) { sc[i] = scores[qi * N + i]; m = max(m, sc[i]); }
    float l = 0.0f;
    for (int i = 0; i < N; ++i) { sc[i] = metal::precise::exp(sc[i] - m); l += sc[i]; }
    for (int i = 0; i < N; ++i) out[qi * N + i] = sc[i] / l;
"""
_CUR_MIX = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint qi = threadgroup_position_in_grid.x;
    for (int d = tid; d < D; d += 256) {
        float o = 0.0f;
        for (int i = 0; i < N; ++i) o += w[qi * N + i] * keys[i * D + d];
        out[qi * D + d] = o;
    }
"""
_cache = {}


def _kernel(name, inputs, source, header=""):
    key = (name, source, header)
    if key not in _cache:
        _cache[key] = mx.fast.metal_kernel(name=name, input_names=inputs, output_names=["out"],
                                           source=source, header=header)
    return _cache[key]


def _run(kernel, inputs, size, grid, tg, template=None):
    kwargs = {"template": template} if template else {}
    out = kernel(inputs=inputs, grid=grid, threadgroup=tg, output_shapes=[(size,)],
                 output_dtypes=[mx.float32], **kwargs)[0]
    mx.eval(out)
    return out


# ----- references (the composed MLX ops, as the model runs them) ----------
@mx.compile
def _ref_res_scores(keys, query):
    return (keys @ query.reshape(-1, keys.shape[-1]).T) * keys.shape[-1] ** -0.5


@mx.compile
def _ref_res_softmax(scores):
    return mx.softmax(scores, axis=-2)


@mx.compile
def _ref_res_mix(weights, keys):
    return weights.swapaxes(-1, -2) @ keys


def _ref_keys(states):
    return mx.fast.rms_norm(mx.stack(states, axis=-2).astype(mx.float32), None, 1e-6)


def _references(law, states, query):
    """Stage references; every later stage starts from the earlier stage's
    reference output, like the probe's stage kernels."""
    width = states[-1].shape[-1]
    keys = _ref_keys(states)  # [1, 1, N, D]
    if law == "residual":
        scores = _ref_res_scores(keys, query)  # [1, 1, N, nq]
        weights = _ref_res_softmax(scores)  # [1, 1, N, nq]
        mix = _ref_res_mix(weights, keys)  # [1, 1, nq, D]
        from mlx2.runtime.models.inkling2.language import _attention_residual

        e2e = _attention_residual(mx.stack(states, axis=-2), query, True)
        scores_qn = scores[0, 0].T  # [nq, N]
        weights_qn = weights[0, 0].T
    else:
        scores = (query @ keys.swapaxes(-1, -2)) * width ** -0.5  # [1, 1, nq, N]
        weights = mx.softmax(scores, axis=-1)
        mix = weights @ keys
        e2e = mx.softmax((query @ keys.swapaxes(-1, -2)) * width ** -0.5, axis=-1) @ keys
        scores_qn = scores[0, 0]
        weights_qn = weights[0, 0]
    mx.eval(keys, scores, weights, mix, e2e, scores_qn, weights_qn)
    return {"keys": keys, "scores": scores, "scores_qn": scores_qn, "weights": weights,
            "weights_qn": weights_qn, "mix": mix, "e2e": e2e}


# ----- arms ----------------------------------------------------------------
def _current_stages(law, states, query, ref):
    n = len(states)
    width = states[-1].shape[-1]
    q = query.reshape(-1, width).astype(mx.float32)
    nq = q.shape[0]
    xs = mx.stack(states, axis=-2).astype(mx.float32).reshape(-1)
    keys = ref["keys"].reshape(-1)
    t = [("N", n), ("D", width)]
    out = {}
    out["rms"] = _run(_kernel("probe_cur_rms", ["xs"], _CUR_RMS), [xs], n * width,
                      (256, 1, 1), (256, 1, 1), t).reshape(n, width)
    s = _run(_kernel("probe_cur_scores", ["keys", "qs", "scale"], _CUR_SCORES),
             [keys, q.reshape(-1), mx.array([width ** -0.5], mx.float32)], nq * n,
             (nq * 256, 1, 1), (256, 1, 1), t).reshape(nq, n)
    out["scores"] = s
    out["softmax"] = _run(_kernel("probe_cur_softmax", ["scores"], _CUR_SOFTMAX),
                          [ref["scores_qn"].reshape(-1)], nq * n, (32, 1, 1), (32, 1, 1),
                          t + [("NQ", nq)]).reshape(nq, n)
    out["mix"] = _run(_kernel("probe_cur_mix", ["w", "keys"], _CUR_MIX),
                      [ref["weights_qn"].reshape(-1), keys], nq * width, (nq * 256, 1, 1),
                      (256, 1, 1), t).reshape(nq, width)
    return out


# Candidate stage kernels: (phases, input names, raw-state expr, key expr,
# pre glue, post glue). The arithmetic is fuse_kernels.mo_phase's text.
_RAW = "xs[(i) * MO_D + (d)]"
_STORED = "keys[(i) * MO_D + (d)]"
_NORMED = "MO_X(i, d) * mo_inv[(i)]"
STAGE_GLUE = {
    "rms": (("rms",), ["xs", "eps"], _RAW, _NORMED, "",
            "for (int e = int(tid); e < MO_N * MO_D; e += MO_TG) out[e] = MO_KEY(e / MO_D, e % MO_D);"),
    "scores": (("scores",), ["keys", "qs", "scale"], "0.0f", _STORED, "",
               "for (int e = int(tid); e < MO_SA * MO_SB; e += MO_TG) out[e] = mo_s[e];"),
    "softmax": (("softmax",), ["sc"], "0.0f", _STORED,
                ("for (int e = int(tid); e < MO_SA * MO_SB; e += MO_TG) mo_s[e] = sc[e];\n"
                 "    threadgroup_barrier(mem_flags::mem_threadgroup);"),
                "for (int e = int(tid); e < MO_NQ * MO_N; e += MO_TG) out[e] = mo_w[e];"),
    "mix": (("mix",), ["w", "keys"], "0.0f", _STORED,
            ("for (int e = int(tid); e < MO_NQ * MO_N; e += MO_TG) mo_w[e] = w[e];\n"
             "    threadgroup_barrier(mem_flags::mem_threadgroup);"), ""),
}


def stage_source(plan, stage, mac):
    """``(input names, header, source)`` of one candidate stage kernel."""
    from mlx2.runtime.models.inkling2 import fuse_kernels as K

    phases, inputs, x, key, pre, post = STAGE_GLUE[stage]
    defines = K._mo_defines(plan, x, "qs[(j) * MO_D + (d)]", key)
    header, source = K.mo_source(plan, phases, defines, pre=pre, post=post, mac=mac)
    return inputs, header, source


def _candidate_stages(law, states, query, ref, mac, plan):
    n = len(states)
    width = states[-1].shape[-1]
    q = query.reshape(-1, width).astype(mx.float32)
    nq = q.shape[0]
    tg = plan["tg"]
    launch = ((tg, 1, 1), (tg, 1, 1))
    data = {
        "xs": mx.stack(states, axis=-2).astype(mx.float32).reshape(-1),
        "eps": mx.array([1e-6], mx.float32),
        "scale": mx.array([width ** -0.5], mx.float32),
        "keys": ref["keys"].reshape(-1),
        "qs": q.reshape(-1),
        "sc": ref["scores"].reshape(-1),  # mo_s layout [sa, sb]
        "w": ref["weights_qn"].reshape(-1),  # mo_w layout [nq, n]
    }
    sizes = {"rms": (n, width), "scores": (plan["sa"], plan["sb"]), "softmax": (nq, n),
             "mix": (nq, width)}
    out = {}
    for stage in ("rms", "scores", "softmax", "mix"):
        inputs, header, source = stage_source(plan, stage, mac)
        kernel = _kernel(f"probe_mo_{stage}_{law}_{n}_{nq}_{mac}", inputs, source, header)
        rows, cols = sizes[stage]
        out[stage] = _run(kernel, [data[name] for name in inputs], rows * cols, *launch).reshape(
            rows, cols)
    if law == "residual":
        out["scores"] = out["scores"].T  # [nq, n] like the reference view
    return out


def _e2e(law, states, query, arm):
    from mlx2.runtime.models.inkling2 import fuse_kernels as K

    module = types.SimpleNamespace()
    old = (K.RESIDUAL_MLX_ORDER, K.RESIDUAL_MLX_MAC)
    try:
        K.RESIDUAL_MLX_ORDER = arm != "current"
        if arm != "current":
            K.RESIDUAL_MLX_MAC = MACS[arm]
        out = K.softmax_mix(module, states, query, law=law)
    finally:
        K.RESIDUAL_MLX_ORDER, K.RESIDUAL_MLX_MAC = old
    if out is None:
        return None, module.__dict__.get("fuse_counts", {})
    mx.eval(out)
    return out, module.__dict__.get("fuse_counts", {})


def _compare(got, ref):
    if got is None:
        return {"ran": False}
    a = np.array(got.astype(mx.float32)).reshape(-1)
    b = np.array(ref.astype(mx.float32)).reshape(-1)
    if a.shape != b.shape:
        return {"ran": True, "shape_mismatch": [list(a.shape), list(b.shape)]}
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    bits = a.view(np.uint32) != b.view(np.uint32)
    return {"ran": True, "bitwise": bool(not bits.any()), "mismatches": int(bits.sum()),
            "elements": int(a.size), "max_abs": float(np.nanmax(diff)) if diff.size else 0.0,
            "nonfinite": int((~np.isfinite(a)).sum())}


def _inputs(law, n, nq, width, dtype, seed):
    key = mx.random.key(seed * 1000 + n * 10 + nq)
    keys = mx.random.split(key, n + 2)
    scales = np.random.default_rng(seed).uniform(0.25, 4.0, size=n)
    states = [(mx.random.normal((1, 1, width), key=keys[i]) * float(scales[i])).astype(dtype)
              for i in range(n - 1)]
    states.append(mx.random.normal((1, 1, width), key=keys[n - 1]) * float(scales[n - 1]))
    shape = (width,) if nq == 1 else (nq, width)
    query = mx.random.normal(shape, key=keys[n]) * 1.5
    mx.eval(states, query)
    return states, query


def _case(law, n, nq, width, dtype, seed):
    from mlx2.runtime.models.inkling2 import fuse_kernels as K

    states, query = _inputs(law, n, nq, width, dtype, seed)
    ref = _references(law, states, query)
    plan, reason = K.mlx_order_plan(law, n, nq, width)
    expect = {"rms": ref["keys"][0, 0], "scores": ref["scores_qn"], "softmax": ref["weights_qn"],
              "mix": ref["mix"][0, 0]}
    cells = {stage: {} for stage in STAGES}
    cur = _current_stages(law, states, query, ref)
    for stage in ("rms", "scores", "softmax", "mix"):
        cells[stage]["current"] = _compare(cur[stage], expect[stage])
    for arm in ARMS[1:]:
        if plan is None:
            for stage in ("rms", "scores", "softmax", "mix"):
                cells[stage][arm] = {"ran": False, "declined": reason}
            continue
        got = _candidate_stages(law, states, query, ref, MACS[arm], plan)
        for stage in ("rms", "scores", "softmax", "mix"):
            cells[stage][arm] = _compare(got[stage], expect[stage])
    for arm in ARMS:
        out, counts = _e2e(law, states, query, arm)
        cell = _compare(out, ref["e2e"])
        cell["counts"] = counts
        cells["e2e"][arm] = cell
    return {"law": law, "n": n, "nq": nq, "width": width, "block_dtype": str(dtype).split(".")[-1],
            "seed": seed, "plan": plan, "plan_decline": reason, "cells": cells}


def _summary(cases):
    out = {}
    for case in cases:
        for stage, arms in case["cells"].items():
            for arm, cell in arms.items():
                key = f"{case['law']}/nq{case['nq']}/{stage}/{arm}"
                agg = out.setdefault(key, {"cells": 0, "ran": 0, "bitwise": 0, "worst_max_abs": 0.0,
                                           "not_bitwise": []})
                agg["cells"] += 1
                if not cell.get("ran"):
                    continue
                agg["ran"] += 1
                if cell.get("bitwise"):
                    agg["bitwise"] += 1
                else:
                    agg["not_bitwise"].append(f"n{case['n']}/{case['block_dtype']}/s{case['seed']}")
                agg["worst_max_abs"] = max(agg["worst_max_abs"], cell.get("max_abs", 0.0))
    return out


def _git(*args):
    try:
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--states", default="2,4,8,12,16")
    parser.add_argument("--bank", default="1,2,4,8,12,16")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--width", type=int, default=2304)
    args = parser.parse_args(argv)
    if args.out.exists():
        sys.exit(f"refusing to overwrite {args.out}")
    if not mx.metal.is_available():
        sys.exit("Metal required")
    mx.set_default_device(mx.gpu)
    sys.path.insert(0, str(ROOT / "src"))
    from mlx2.runtime.models.inkling2 import fuse_kernels as K

    info = (getattr(mx, "device_info", None) or mx.metal.device_info)()
    meta = {
        "script": "scripts/research/inkling2_residual_bitwise_probe.py",
        "checkpoint": "none: synthetic random inputs at the checkpoint width, no model loaded",
        "mlx2_commit": _git("rev-parse", "HEAD"), "mlx2_dirty": bool(_git("status", "--porcelain", "--", "src", "scripts")),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "mlx_version": mx.__version__, "device": info.get("device_name"),
        "architecture": info.get("architecture"), "planned_architecture": K._architecture(),
        "MLX_ENABLE_TF32": os.environ.get("MLX_ENABLE_TF32"),
        "python": platform.python_version(), "argv": sys.argv[1:],
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    cases = []
    states = [int(v) for v in args.states.split(",") if v]
    bank = [int(v) for v in args.bank.split(",") if v]
    t0 = time.time()
    for seed in range(args.seeds):
        for dtype in (mx.bfloat16, mx.float32):
            for n in states:
                for nq in (1, 3):
                    cases.append(_case("residual", n, nq, args.width, dtype, seed))
            for n in bank:
                cases.append(_case("message", n, 4, args.width, dtype, seed))
    meta["seconds"] = round(time.time() - t0, 2)
    summary = _summary(cases)
    result = {"meta": meta, "summary": summary, "cases": cases}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
    tmp.replace(args.out)
    width = max(len(k) for k in summary)
    for key, agg in sorted(summary.items()):
        print(f"{key:<{width}}  bitwise {agg['bitwise']}/{agg['ran']} (of {agg['cells']})"
              f"  worst {agg['worst_max_abs']:.3g}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

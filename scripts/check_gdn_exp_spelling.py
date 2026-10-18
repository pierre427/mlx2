#!/usr/bin/env python3
"""Metal check: GDN prework kernels against the served MLX ops on a bf16 sweep.

The fused GDN kernels copy MLX's sigmoid spellings.  MLX 0.32.4 serves the
compiled ``nn.silu`` with ``metal::precise::exp``; a kernel still spelled
``metal::exp`` rounds apart from it on a few bf16 inputs (one near -6.84 that
real conv outputs reach).  Random draws almost never hit those inputs, so each
case here feeds conv outputs covering every bf16 value with magnitude in
[2**-6, 32] (an identity last conv tap makes each conv output exactly its qkv
value) and compares the kernel with the eager MLX chain it claims to match:

* ``tree_pre``            ``qwen38_tree_gdn.gdn_pre`` (Qwen3.8 tree/cohort GDN
                          prework): v vs ``nn.silu(conv1d)``, q/k vs
                          ``normalize_gdn_qk``, g vs ``compute_g``, beta vs
                          ``gate_sigmoid(b.float32)``.
* ``prefill_qwen4``       ``qwen4_gdn_prefill_prework`` (Flash-Next, default
                          on): v vs ``nn.silu``, q/k vs the Qwen4 L2 chain.
* ``prefill_qwen35``      the Qwen3.8 27B variant: q/k vs ``normalize_gdn_qk``.
* ``norm_gate_sigmoid``   Qwen4 ``RMSNormGated`` (z swept).
* ``norm_gate_swish``     Qwen3.8 ``Qwen3NextRMSNormGated`` (z swept).
* ``lane_glue_gdn_post``  vendored TensorFold ``lane_glue.gdn_post`` (z swept).
* ``lane_glue_mlp_act``   vendored TensorFold ``lane_glue.mlp_act`` vs
                          compiled ``swiglu`` (gate swept).
* ``expr_*``              expression-level counts over every bf16 encoding.

Each case reports mismatching elements; zero is bit-exact.  Run under the GPU
lock (``scripts/run_with_gpu_locks.py``) with ``--i-own-the-gpu``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

NK, NV, DK, DV, TAPS = 16, 48, 128, 128, 4
C = 2 * NK * DK + NV * DV


def _mismatch(a, b):
    import mlx.core as mx

    if a.shape != b.shape or a.dtype != b.dtype:
        return {"shape_or_dtype": [str(a.shape), str(a.dtype), str(b.shape), str(b.dtype)]}
    width = mx.uint16 if a.dtype.size == 2 else mx.uint32
    same = (a.view(width) == b.view(width)) | (mx.isnan(a) & mx.isnan(b))
    bad = int((~same).sum().item())
    out = {"mismatch": bad, "total": int(a.size)}
    if bad:
        flat_a = a.reshape(-1).astype(mx.float32)
        flat_b = b.reshape(-1).astype(mx.float32)
        idx = mx.argmax((~same).reshape(-1).astype(mx.int32)).item()
        out["first"] = {"index": int(idx), "kernel": float(flat_a[idx].item()),
                        "eager": float(flat_b[idx].item())}
    return out


def _sweep_rows(rows: int, width: int, seed: int):
    import mlx.core as mx
    from mlx2.runtime.models.qwen4_fused_gdn_verify import _decode_parity_grid

    grid = _decode_parity_grid()
    if width < int(grid.size):
        raise ValueError(f"a row of {width} cannot cover {int(grid.size)} values")
    # Cut the tile to width before shuffling, so every row keeps every value.
    base = mx.tile(grid, (-(-width // int(grid.size)),))[:width]
    return mx.stack([
        base[mx.random.permutation(width, key=mx.random.key(seed + r))] for r in range(rows)
    ])


def _identity_conv_weight():
    import mlx.core as mx

    weight = mx.zeros((C, TAPS, 1), dtype=mx.bfloat16)
    weight[:, TAPS - 1, 0] = 1
    return weight


def _eager_conv_silu(qkv, state, weight):
    import mlx.core as mx
    import mlx.nn as nn

    conv_input = mx.concatenate([state, qkv], axis=1)
    return nn.silu(mx.conv1d(conv_input, weight, groups=C))


def case_tree_pre(rows=4):
    import mlx.core as mx
    from mlx2.runtime.models import qwen38_tree_gdn as TG
    from mlx2.runtime.models.gated_delta import compute_g, normalize_gdn_qk
    from mlx2.runtime.models.precise_ops import gate_sigmoid

    qkv = _sweep_rows(rows, C, 7100)[None]
    state = mx.zeros((1, TAPS - 1, C), dtype=mx.bfloat16)
    weight = _identity_conv_weight()
    key = mx.random.key(7200)
    k1, k2, k3, k4 = mx.random.split(key, 4)
    b = (mx.random.normal((1, rows, NV), key=k1) * 3).astype(mx.bfloat16)
    a = (mx.random.normal((1, rows, NV), key=k2) * 3).astype(mx.bfloat16)
    z = mx.random.normal((1, rows, NV * DV), key=k3).astype(mx.bfloat16)
    zba = mx.concatenate([z, b, a], axis=-1)
    a_log = mx.linspace(-2.0, 1.0, NV).astype(mx.bfloat16)
    dt_bias = (mx.random.normal((NV,), key=k4) * 0.5).astype(mx.bfloat16)
    plan = TG.ConvPlan([list(range(-1, rows - 1))], TAPS - 1)
    q, k, v, g, beta = TG.gdn_pre(qkv, [state], weight, plan, zba, a_log, dt_bias,
                                   nk=NK, nv=NV, dk=DK, dv=DV)
    conv = _eager_conv_silu(qkv, state, weight)
    eq, ek, ev = mx.split(conv, [NK * DK, 2 * NK * DK], -1)
    eq, ek = normalize_gdn_qk(eq.reshape(1, rows, NK, DK), ek.reshape(1, rows, NK, DK))
    ev = ev.reshape(1, rows, NV, DV)
    eg = compute_g(a_log, a, dt_bias)
    eb = gate_sigmoid(b.astype(mx.float32))
    return {"v": _mismatch(v, ev), "q": _mismatch(q, eq), "k": _mismatch(k, ek),
            "g": _mismatch(g, eg.astype(mx.float32)), "beta": _mismatch(beta, eb)}


def case_prefill(numerics: str, rows=64):
    import mlx.core as mx
    from mlx2.runtime.models import qwen4_fused_gdn_prefill as PF
    from mlx2.runtime.models.gated_delta import normalize_gdn_qk

    qkv = _sweep_rows(rows, C, 7300)[None]
    state = mx.zeros((1, TAPS - 1, C), dtype=mx.bfloat16)
    weight = _identity_conv_weight()
    q, k, v, _ = PF.qwen4_gdn_prefill_prework(qkv, None, weight, numerics=numerics)
    conv = _eager_conv_silu(qkv, state, weight)
    eq, ek, ev = mx.split(conv, [NK * DK, 2 * NK * DK], -1)
    eq = eq.reshape(1, rows, NK, DK)
    ek = ek.reshape(1, rows, NK, DK)
    if numerics == "qwen4":
        eq = eq * mx.rsqrt(mx.sum(mx.square(eq), axis=-1, keepdims=True) + 1e-06)
        ek = ek * mx.rsqrt(mx.sum(mx.square(ek), axis=-1, keepdims=True) + 1e-06)
        eq = eq * DK ** (-0.5)
    else:
        eq, ek = normalize_gdn_qk(eq, ek)
    return {"v": _mismatch(v, ev.reshape(1, rows, NV, DV)), "q": _mismatch(q, eq),
            "k": _mismatch(k, ek)}


def _gate_inputs(rows):
    import mlx.core as mx

    y = mx.random.normal((1, rows, NV, DV), key=mx.random.key(7400)).astype(mx.bfloat16)
    z = _sweep_rows(rows, NV * DV, 7500)[None]
    w = (1 + 0.1 * mx.random.normal((DV,), key=mx.random.key(7600))).astype(mx.bfloat16)
    return y, z, w


def case_norm_gate(gate: str, rows=8):
    import mlx.core as mx
    from mlx2.runtime.models import qwen4_fused_gdn_prefill as PF
    from mlx2.runtime.models.qwen3_next import _precise_swiglu

    eps = 1e-6
    y, z, w = _gate_inputs(rows)
    out = PF.qwen4_gdn_prefill_norm_gate(y, z, w, eps, gate=gate)
    zr = z.reshape(1, rows, NV, DV)
    if gate == "sigmoid":
        x = mx.fast.rms_norm(y, w, eps).astype(mx.float32)
        ref = (x * mx.sigmoid(zr.astype(mx.float32))).astype(mx.bfloat16)
    else:
        ref = _precise_swiglu(y, zr, mx.fast.rms_norm(y, w, eps))
    return {"out": _mismatch(out, ref.reshape(1, rows, NV * DV))}


def case_lane_glue(rows=8):
    import mlx.core as mx
    from mlx2.runtime.models.activations import swiglu
    from mlx2.runtime.models.qwen3_next import _precise_swiglu
    from mlx2.runtime.tensorfold_qwen38 import lane_glue

    eps = 1e-6
    y, z, w = _gate_inputs(rows)
    post = lane_glue.gdn_post(y, z, w, eps)
    ref = _precise_swiglu(y, z.reshape(1, rows, NV, DV), mx.fast.rms_norm(y, w, eps))
    n = 17408
    gate = _sweep_rows(rows, n, 7700)[None]
    up = mx.random.normal((1, rows, n), key=mx.random.key(7800)).astype(mx.bfloat16)
    act = lane_glue.mlp_act(gate, up)
    return {"gdn_post": _mismatch(post, ref.reshape(1, rows, NV * DV)),
            "mlp_act": _mismatch(act, swiglu(gate, up))}


def case_expressions():
    """Counts over every bf16 encoding for the copied SiLU forms."""
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from mlx2.runtime.models import qwen4_fused_gdn as FG

    x = FG.all_16bit_encodings(mx.bfloat16)
    kernel = mx.fast.metal_kernel(
        name="check_gdn_exp_spelling_silu_forms",
        input_names=["inp"],
        output_names=["fast_t", "precise_t", "fast_f", "precise_f"],
        header=FG._HEADER,
        source=FG._SILU_PROBE_SOURCE,
        ensure_row_contiguous=True,
    )
    fast_t, precise_t, fast_f, precise_f = kernel(
        inputs=[x], template=[("T", mx.bfloat16)], grid=(x.size, 1, 1),
        threadgroup=(256, 1, 1), output_shapes=[x.shape] * 4,
        output_dtypes=[mx.bfloat16, mx.bfloat16, mx.float32, mx.float32],
    )
    served_t = nn.silu(x)
    served_f = nn.silu(x.astype(mx.float32))
    finite = mx.isfinite(x)
    out = {}
    for name, got, ref in (("bf16_fast", fast_t, served_t), ("bf16_precise", precise_t, served_t),
                           ("f32_fast", fast_f, served_f), ("f32_precise", precise_f, served_f)):
        diff = _mismatch(got, ref)
        width = mx.uint16 if got.dtype.size == 2 else mx.uint32
        bad = np.array((got.view(width) != ref.view(width)) & finite)
        if int(bad.sum()):
            values = sorted(float(v) for v in np.array(x.astype(mx.float32))[bad])
            diff["inputs_sample"] = values[:8]
            diff["inputs_range"] = [values[0], values[-1]]
        out[name] = diff
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--i-own-the-gpu", action="store_true")
    parser.add_argument("--cases", default="expr,tree_pre,prefill_qwen4,prefill_qwen35,"
                        "norm_gate_sigmoid,norm_gate_swish,lane_glue")
    args = parser.parse_args(argv)
    if not args.i_own_the_gpu:
        parser.error("run under scripts/run_with_gpu_locks.py and pass --i-own-the-gpu")
    import mlx.core as mx
    from mlx2.runtime.models import qwen4_fused_gdn as FG

    mx.set_default_device(mx.gpu)
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "src"],
                                capture_output=True, text=True).stdout.strip())
    report = {"git_head": head, "src_dirty": dirty, "mlx": mx.__version__,
              "device": mx.device_info().get("device_name"),
              "served_silu_exp": FG.served_silu_exp(), "started": time.time(), "cases": {}}
    runners = {
        "expr": case_expressions,
        "tree_pre": case_tree_pre,
        "prefill_qwen4": lambda: case_prefill("qwen4"),
        "prefill_qwen35": lambda: case_prefill("qwen35"),
        "norm_gate_sigmoid": lambda: case_norm_gate("sigmoid"),
        "norm_gate_swish": lambda: case_norm_gate("swish"),
        "lane_glue": case_lane_glue,
    }
    for name in args.cases.split(","):
        try:
            report["cases"][name] = runners[name]()
        except Exception as exc:  # noqa: BLE001 - record and continue
            report["cases"][name] = {"error": f"{type(exc).__name__}: {exc}"}
        print(name, json.dumps(report["cases"][name]), flush=True)
    report["finished"] = time.time()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

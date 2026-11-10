#!/usr/bin/env python3
"""Prototype: Q2 variants vs the original MXFP4 expert on captured inputs (CPU only).

Variants: affine g64 (current bank), affine g32, affine g32 + block Hadamard rotation of the
input dimension (256-wide blocks: W' = W H, x' = H x, so W x = W' x'), and the same with an
MSE-searched clip of each group's range. Reports weight error and whole-expert output error.
"""
import json, sys
from pathlib import Path
import numpy as np
import mlx.core as mx
mx.set_default_device(mx.cpu)
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).parent))
from convert import decode_mxfp4  # noqa: E402
from mlx2.adapters.deepseek_v41_source_artifact import inspect_artifact, read_tensors  # noqa: E402
from mlx2.adapters.deepseek_v41_source_weights import expert_names, PROJECTIONS  # noqa: E402
from mlx2.runtime.models.deepseek_v4 import limited_swiglu  # noqa: E402
from mlx2.runtime.models.deepseek_v41_quant import quantize as qat  # noqa: E402
BLK = 256


def rot(a):                       # block Hadamard over the last dim (orthonormal)
    s = a.shape
    return mx.hadamard_transform(a.reshape(*s[:-1], s[-1] // BLK, BLK)).reshape(s)


def q_affine(w, gs):
    q, s, b = mx.quantize(w, group_size=gs, bits=2)
    return mx.dequantize(q, s.astype(mx.float16).astype(mx.float32), b.astype(mx.float16).astype(mx.float32), group_size=gs, bits=2)


def q_affine_clip(w, gs, grid=(1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7)):
    """Per-group range shrink chosen by MSE (numpy, 2-bit affine levels 0..3)."""
    W = np.array(w, np.float32); n, k = W.shape; G = W.reshape(n, k // gs, gs)
    lo, hi = G.min(-1, keepdims=True), G.max(-1, keepdims=True)
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    best, besterr = None, None
    for c in grid:
        l, h = mid - half * c, mid + half * c
        sc = np.maximum((h - l) / 3, 1e-12).astype(np.float16).astype(np.float32); bi = l.astype(np.float16).astype(np.float32)
        d = np.clip(np.round((G - bi) / sc), 0, 3) * sc + bi
        e = ((d - G) ** 2).sum(-1, keepdims=True)
        if best is None:
            best, besterr = d, e
        else:
            m = e < besterr; best = np.where(m, d, best); besterr = np.minimum(e, besterr)
    return mx.array(best.reshape(n, k))


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def main():
    art = inspect_artifact(str(Path.home() / "mlx-models/DeepSeek-V4.1-Flash-1017-Source"))
    cap = Path.home() / "mlx-models/.ds41-routing-capture-20261108"
    z = np.load(ROOT / "output/ds41-source-20261108/cache-design/sched/trace.npz"); ids = z["ids"]
    out = []
    for layer in (5, 20, 35):
        counts = np.bincount(ids[:, layer].reshape(-1), minlength=384)
        for expert in (int(np.argmax(counts)), int(np.argsort(counts)[192])):
            rows = np.where((ids[:, layer] == expert).any(-1))[0][:64]
            x = mx.array(np.load(cap / "x_by_layer" / f"L{layer:02d}.npy", mmap_mode="r")[rows].astype(np.float32))
            xq = qat(x, "fp8").astype(mx.float32)
            raw = read_tensors(art, expert_names(layer, expert))
            W = {k: mx.array(decode_mxfp4(raw[f"layers.{layer}.ffn.experts.{expert}.{p}.weight"],
                                          raw[f"layers.{layer}.ffn.experts.{expert}.{p}.scale"])) for k, p in PROJECTIONS.items()}

            def run(proj):
                g, u = proj("gate", xq), proj("up", xq)
                return proj("down", qat(limited_swiglu(g, u), "fp8").astype(mx.float32))
            ref = run(lambda k, v: v @ W[k].T)
            row = {"layer": layer, "expert": expert}
            variants = {
                "g64": {k: q_affine(W[k], 64) for k in W},
                "g32": {k: q_affine(W[k], 32) for k in W},
                "g32_had": {k: q_affine(rot(W[k]), 32) for k in W},
                "g32_had_clip": {k: q_affine_clip(rot(W[k]), 32) for k in W},
                "g64_had_clip": {k: q_affine_clip(rot(W[k]), 64) for k in W},
            }
            for name, D in variants.items():
                had = "had" in name
                y = run(lambda k, v: (rot(v) if had else v) @ D[k].T)
                werr = float(np.mean([rel(D[k], rot(W[k]) if had else W[k]) for k in W]))
                row[name] = {"weight_err": round(werr, 3), "expert_err": round(rel(y, ref), 3),
                             "cos": round(float(np.sum(np.array(y) * np.array(ref)) / (np.linalg.norm(np.array(y)) * np.linalg.norm(np.array(ref)))), 3)}
            out.append(row); print(json.dumps(row), flush=True)
    (ROOT / "output/ds41-source-20261108/q2-variants-proto.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

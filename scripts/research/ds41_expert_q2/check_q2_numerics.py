#!/usr/bin/env python3
"""CPU check of the Q2 expert path against the original MXFP4 expert (DS4.1 Source).

For a few (layer, expert) pairs and captured router inputs where that expert was selected:
  * per projection: original (decoded MXFP4) vs runtime Q2 quantized_matmul vs dequantized Q2;
  * whole expert (gate/up -> limited SwiGLU -> fp8 QAT -> down) original vs Q2.
A runtime-vs-dequant mismatch means a Q2 path bug; runtime ~= dequant with large error vs the
original means Q2 is simply that lossy. CPU only.
usage: check_q2_numerics.py [--layers 5,20,35] [--rows 64]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import mlx.core as mx

mx.set_default_device(mx.cpu)
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).parent))
from convert import decode_mxfp4  # noqa: E402
from mlx2.adapters.deepseek_v41_source_artifact import inspect_artifact, read_tensors  # noqa: E402
from mlx2.adapters.deepseek_v41_source_weights import expert_names, PROJECTIONS  # noqa: E402
from safetensors.numpy import load_file  # noqa: E402
from mlx2.runtime.models.deepseek_v4 import limited_swiglu  # noqa: E402
from mlx2.runtime.models.deepseek_v41_quant import quantize  # noqa: E402

MODEL = "~/mlx-models/DeepSeek-V4.1-Flash-1017-Source"
Q2 = Path("~/mlx-models/DeepSeek-V4.1-Flash-1017-Source-Expert-Q2-g64")
CAP = Path("~/mlx-models/.ds41-routing-capture-20261108")
TRACE = ROOT / "output/ds41-source-20261108/cache-design/sched/trace.npz"


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--layers", default="5,20,35"); ap.add_argument("--rows", type=int, default=64)
    a = ap.parse_args()
    art = inspect_artifact(MODEL)
    manifest = json.loads((Q2 / "manifest.json").read_text())
    records = {(r["layer"], r["expert"]): r for r in manifest["experts"]}
    z = np.load(TRACE); ids = z["ids"]
    out = []
    for layer in map(int, a.layers.split(",")):
        counts = np.bincount(ids[:, layer].reshape(-1), minlength=384)
        for expert in (int(np.argmax(counts)), int(np.argsort(counts)[len(counts) // 2])):
            rows = np.where((ids[:, layer] == expert).any(-1))[0][: a.rows]
            x = np.load(CAP / "x_by_layer" / f"L{layer:02d}.npy", mmap_mode="r")[rows].astype(np.float32)
            xq = quantize(mx.array(x), "fp8").astype(mx.float32)
            raw = read_tensors(art, expert_names(layer, expert))
            f = load_file(str(Q2 / records[(layer, expert)]["file"]))     # same tensors q2_read verifies
            q2 = {k.split(".")[0] + ".q2" + {"qweight": "weight", "scales": "scales", "biases": "biases"}[k.split(".")[1]]: v
                  for k, v in f.items()}
            W, R, D = {}, {}, {}
            for kind, part in PROJECTIONS.items():
                p = f"layers.{layer}.ffn.experts.{expert}.{part}."
                W[kind] = mx.array(decode_mxfp4(raw[p + "weight"], raw[p + "scale"]))
                qw, qs, qb = (mx.array(q2[kind + f]) for f in (".q2weight", ".q2scales", ".q2biases"))
                R[kind] = lambda v, qw=qw, qs=qs, qb=qb: mx.quantized_matmul(v, qw, qs, qb, transpose=True, group_size=64, bits=2).astype(mx.float32)
                deq = mx.dequantize(qw, qs.astype(mx.float32), qb.astype(mx.float32), group_size=64, bits=2)
                D[kind] = deq
            row = {"layer": layer, "expert": expert, "rows": int(len(rows)), "uses": int(counts[expert])}
            for kind in ("gate", "up"):
                yo = xq @ W[kind].T; yr = R[kind](xq); yd = xq @ D[kind].T
                row[kind] = {"weight_rel_err": rel(D[kind], W[kind]), "runtime_vs_orig": rel(yr, yo), "runtime_vs_dequant": rel(yr, yd)}

            def expert_out(proj):
                g, u = proj("gate", xq), proj("up", xq)
                mid = quantize(limited_swiglu(g, u), "fp8").astype(mx.float32)
                return proj("down", mid)
            yo = expert_out(lambda k, v: v @ W[k].T); yr = expert_out(lambda k, v: R[k](v)); yd = expert_out(lambda k, v: v @ D[k].T)
            row["down_weight_rel_err"] = rel(D["down"], W["down"])
            row["expert_runtime_vs_orig"] = rel(yr, yo); row["expert_runtime_vs_dequant"] = rel(yr, yd)
            row["expert_cosine"] = float(np.sum(np.array(yr) * np.array(yo)) / (np.linalg.norm(np.array(yr)) * np.linalg.norm(np.array(yo))))
            out.append(row); print(json.dumps(row), flush=True)
    dst = ROOT / "output/ds41-source-20261108/q2-numerics-check.json"
    dst.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
